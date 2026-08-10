"""Measurable form indicators derived from a joint-angle series.

Recognition is left to the vision model -- skeleton matching was tested for
that and rejected. What the skeleton is genuinely good at is measurement:
how far a joint travelled, whether the two sides match, how far the trunk
leaned, how many repetitions there were and how fast.

These numbers exist so the model does not have to estimate them from stills.
"Knees flexed to 71 degrees" is a fact; "looks a bit shallow" is a guess.

Requires the optional ``pose`` extra; see ``docs/pose-setup.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .pose import ANGLE_NAMES, MIN_USABLE_FRACTION, _smooth

#: Angle pairs whose left/right difference is worth reporting.
_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("elbow", "l_elbow", "r_elbow"),
    ("knee", "l_knee", "r_knee"),
    ("hip", "l_hip", "r_hip"),
    ("shoulder", "l_shoulder", "r_shoulder"),
)

#: A joint must travel at least this far to be considered part of the movement.
MOVING_THRESHOLD_DEG = 15.0

#: Left/right difference above this is worth mentioning to the model.
ASYMMETRY_THRESHOLD_DEG = 10.0

#: Shortest plausible repetition, in seconds. Below this the autocorrelation
#: is measuring signal smoothness rather than repetition.
_MIN_PERIOD_SECONDS = 0.8

#: Longest period considered, as a fraction of the clip. Autocorrelation needs
#: the signal to realign with itself at least once, so a clip containing
#: exactly two repetitions sits right at this bound.
_MAX_PERIOD_FRACTION = 0.55

#: If the autocorrelation at lag/2, lag/3 … is nearly as strong as at lag, the
#: real period is shorter and the chosen lag is one of its harmonics.
_SUBHARMONIC_RATIO = 0.8

#: Autocorrelation below this is treated as "no clear repetition".
_MIN_PERIODICITY = 0.35

#: A joint seen in fewer than this fraction of frames is not measured. Angles
#: interpolated across long gaps swing wildly and would otherwise dominate.
_MIN_JOINT_OBSERVED = 0.6


@dataclass(frozen=True)
class FormMetrics:
    """What was measured, as opposed to what was inferred."""

    rom: Dict[str, float]
    asymmetry: Dict[str, float]
    dominant_joint: str | None
    trunk_max: float
    reps: int | None
    tempo_seconds: float | None
    periodicity: float
    usable_fraction: float

    @property
    def reliable(self) -> bool:
        return self.usable_fraction >= MIN_USABLE_FRACTION

    def moving_joints(self) -> List[str]:
        return [
            name
            for name, value in sorted(self.rom.items(), key=lambda kv: -kv[1])
            if value >= MOVING_THRESHOLD_DEG
        ]

    def describe(self) -> str:
        """Render as short English lines for inclusion in a model prompt.

        Only what was actually measured is stated. Joints that barely moved and
        symmetric pairs are omitted rather than reported as zero, so the model
        is not handed a wall of irrelevant numbers.
        """
        if not self.reliable:
            return (
                "Skeleton tracking was unreliable for this clip "
                f"({self.usable_fraction:.0%} of frames), so no joint measurements "
                "are available."
            )

        lines: List[str] = []
        moving = self.moving_joints()
        if moving:
            lines.append(
                "Joints that moved (range of motion): "
                + ", ".join(f"{name} {self.rom[name]:.0f}°" for name in moving[:5])
            )
        else:
            lines.append("No joint moved appreciably; the position was held.")

        notable = {
            name: value
            for name, value in self.asymmetry.items()
            if value >= ASYMMETRY_THRESHOLD_DEG
        }
        if notable:
            lines.append(
                "Left/right difference: "
                + ", ".join(f"{name} {value:.0f}°" for name, value in notable.items())
            )
        else:
            lines.append("Left and right sides moved through similar ranges.")

        lines.append(f"Maximum trunk lean from vertical: {self.trunk_max:.0f}°")

        if self.reps is not None:
            tempo = (
                f", about {self.tempo_seconds:.1f} s each"
                if self.tempo_seconds is not None
                else ""
            )
            lines.append(f"Repetitions detected: {self.reps}{tempo}")
        else:
            lines.append("No clear repetition cycle was detected.")

        return "\n".join(lines)


def _observed_fraction(column: np.ndarray) -> float:
    return float(np.mean(~np.isnan(column))) if column.size else 0.0


def _joint_series(series: np.ndarray, index: int) -> np.ndarray | None:
    """Smoothed series for one joint, or ``None`` if it was rarely seen."""
    column = series[:, index]
    if _observed_fraction(column) < _MIN_JOINT_OBSERVED:
        return None
    return _smooth(column)


def _dominant(series: np.ndarray) -> tuple[str | None, np.ndarray | None]:
    """The reliably-seen joint that travelled furthest, and its series."""
    best_name: str | None = None
    best_span = 0.0
    best_column: np.ndarray | None = None
    for index, name in enumerate(ANGLE_NAMES):
        column = _joint_series(series, index)
        if column is None:
            continue
        observed = column[~np.isnan(column)]
        if observed.size < 8:
            continue
        low, high = np.percentile(observed, [5, 95])
        if high - low > best_span:
            best_span, best_name, best_column = high - low, name, column
    return best_name, best_column


def count_repetitions(
    signal: np.ndarray, fps: float
) -> tuple[int | None, float | None, float]:
    """Estimate repetitions from a joint-angle signal by autocorrelation.

    Works on a joint angle rather than on raw frame differences: frame
    differencing measures "something changed", which for a person moving
    continuously against a static background is close to noise, whereas a joint
    angle traces the movement itself.

    The lag must be a genuine local peak of the autocorrelation, and no shorter
    than `_MIN_PERIOD_SECONDS`. Taking the plain maximum instead reports the
    shortest allowed lag for almost any smooth signal, which turned a 3-second
    curl into "12 repetitions" and a static plank into "50".

    Returns ``(reps, seconds_per_rep, periodicity)``; ``periodicity`` is the
    normalised autocorrelation at the chosen lag, so a caller can judge how far
    to trust the count.
    """
    values = signal[~np.isnan(signal)]
    min_lag = max(int(round(_MIN_PERIOD_SECONDS * fps)), 4)
    if values.size < 2 * min_lag:
        return None, None, 0.0

    centred = values - values.mean()
    if not np.any(np.abs(centred) > 1e-6):
        return None, None, 0.0

    correlation = np.correlate(centred, centred, mode="full")[values.size - 1 :]
    correlation = correlation / correlation[0]

    high = int(values.size * _MAX_PERIOD_FRACTION)
    if high <= min_lag + 1:
        return None, None, 0.0

    # Local maxima only: a peak means the signal genuinely realigns with itself
    # at that lag, whereas a monotone slope just means it is smooth.
    peaks = [
        lag
        for lag in range(min_lag, high - 1)
        if correlation[lag] > correlation[lag - 1]
        and correlation[lag] >= correlation[lag + 1]
    ]
    if not peaks:
        return None, None, 0.0

    lag = max(peaks, key=lambda candidate: correlation[candidate])
    strength = float(correlation[lag])
    if strength < _MIN_PERIODICITY:
        return None, None, strength

    # A periodic signal correlates with itself at every multiple of its period.
    # If a sub-multiple of the chosen lag is nearly as strong, the true cycle is
    # faster than we are willing to call a repetition, and the count would be a
    # fraction of the truth. Decline rather than report a plausible wrong number.
    for divisor in (2, 3, 4):
        shorter = lag // divisor
        if shorter >= 4 and correlation[shorter] > _SUBHARMONIC_RATIO * strength:
            return None, None, strength

    reps = max(int(round(values.size / lag)), 1)
    return reps, lag / fps if fps > 0 else None, strength


def measure(series: np.ndarray, fps: float = 25.0) -> FormMetrics:
    """Reduce a per-frame angle series to reportable form metrics."""
    usable = float(np.mean(~np.isnan(series).all(axis=1))) if series.size else 0.0

    rom: Dict[str, float] = {}
    for index, name in enumerate(ANGLE_NAMES):
        column = _joint_series(series, index)
        if column is None:
            rom[name] = 0.0
            continue
        observed = column[~np.isnan(column)]
        if observed.size < 4:
            rom[name] = 0.0
            continue
        low, high = np.percentile(observed, [5, 95])
        rom[name] = float(max(high - low, 0.0))

    asymmetry = {
        label: abs(rom[left] - rom[right]) for label, left, right in _PAIRS
    }

    dominant_name, dominant_series = _dominant(series)
    reps = tempo = None
    periodicity = 0.0
    if dominant_series is not None:
        reps, tempo, periodicity = count_repetitions(dominant_series, fps)

    return FormMetrics(
        rom=rom,
        asymmetry=asymmetry,
        dominant_joint=dominant_name,
        trunk_max=rom.get("trunk", 0.0),
        reps=reps,
        tempo_seconds=tempo,
        periodicity=periodicity,
        usable_fraction=usable,
    )
