"""Form metrics: repetition counting and the text handed to the model.

These numbers are presented to the model as facts, so a wrong one is worse
than none at all. The tests below are built from synthetic angle series, which
makes the expected answer unambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_coach.metrics import (
    ASYMMETRY_THRESHOLD_DEG,
    MOVING_THRESHOLD_DEG,
    count_repetitions,
    measure,
)
from movement_coach.pose import ANGLE_NAMES

FPS = 25.0


def _series(**joints) -> np.ndarray:
    """Build an angle series; unnamed joints hold a constant 90 degrees."""
    length = max(len(v) for v in joints.values()) if joints else 50
    series = np.full((length, len(ANGLE_NAMES)), 90.0)
    for name, values in joints.items():
        series[:, ANGLE_NAMES.index(name)] = values
    return series


def _cycles(count: int, length: int, low: float = 60.0, high: float = 160.0):
    mid, half = (high + low) / 2, (high - low) / 2
    return mid + half * np.sin(2 * np.pi * np.arange(length) * count / length)


# -- repetition counting ---------------------------------------------------


@pytest.mark.parametrize("reps", [2, 4, 8])
def test_counts_clean_cycles(reps):
    counted, tempo, strength = count_repetitions(_cycles(reps, 200), FPS)
    assert counted == reps
    assert tempo == pytest.approx(200 / reps / FPS, rel=0.15)
    # Two repetitions sit at the detection limit: the lag is half the clip, so
    # only half the samples contribute and the correlation caps near 0.5.
    assert strength >= (0.45 if reps == 2 else 0.6)


def test_static_signal_has_no_repetitions():
    """A plank is a hold, not a set. Reporting a count here would be a lie."""
    assert count_repetitions(np.full(200, 90.0), FPS)[0] is None


def test_monotonic_drift_is_not_a_repetition():
    """Regression: taking the plain autocorrelation maximum returned the
    shortest allowed lag for any smooth signal, turning a 3-second curl into
    12 repetitions and a static plank into 50."""
    assert count_repetitions(np.linspace(60, 150, 200), FPS)[0] is None


def test_noise_is_not_a_repetition():
    noise = np.random.RandomState(0).normal(90, 20, 200)
    assert count_repetitions(noise, FPS)[0] is None


def test_implausibly_fast_cycles_are_rejected():
    """Faster than the minimum plausible repetition means it is not one."""
    assert count_repetitions(_cycles(40, 200), FPS)[0] is None


def test_too_short_a_clip_yields_nothing():
    assert count_repetitions(_cycles(2, 10), FPS)[0] is None


def test_gaps_do_not_break_counting():
    signal = _cycles(4, 200)
    signal[::17] = np.nan
    assert count_repetitions(signal, FPS)[0] is not None


# -- measurement -----------------------------------------------------------


def test_moving_joint_is_identified():
    metrics = measure(_series(l_elbow=_cycles(3, 150)), FPS)
    assert metrics.dominant_joint == "l_elbow"
    assert metrics.rom["l_elbow"] > 80
    assert metrics.rom["r_knee"] == pytest.approx(0.0, abs=1e-6)
    assert metrics.moving_joints() == ["l_elbow"]


def test_asymmetry_is_the_left_right_gap():
    metrics = measure(
        _series(l_knee=_cycles(3, 150, 60, 160), r_knee=_cycles(3, 150, 90, 130)), FPS
    )
    assert metrics.asymmetry["knee"] == pytest.approx(
        metrics.rom["l_knee"] - metrics.rom["r_knee"], abs=1e-6
    )
    assert metrics.asymmetry["elbow"] == pytest.approx(0.0, abs=1e-6)


def test_joint_seen_in_too_few_frames_is_not_measured():
    """Angles interpolated across long gaps swing wildly; better to say nothing."""
    series = _series(l_elbow=_cycles(3, 150))
    series[50:, ANGLE_NAMES.index("l_elbow")] = np.nan
    assert measure(series, FPS).rom["l_elbow"] == pytest.approx(0.0, abs=1e-6)


def test_mostly_undetected_clip_is_flagged_unreliable():
    series = np.full((100, len(ANGLE_NAMES)), np.nan)
    series[:20] = 90.0
    metrics = measure(series, FPS)
    assert not metrics.reliable
    assert "unreliable" in metrics.describe().lower()


# -- the text the model receives -------------------------------------------


def test_description_states_movement_reps_and_symmetry():
    text = measure(_series(l_elbow=_cycles(4, 200), r_elbow=_cycles(4, 200)), FPS).describe()
    assert "l_elbow" in text
    assert "Repetitions detected: 4" in text
    assert "similar ranges" in text


def test_description_reports_a_hold_without_inventing_reps():
    text = measure(_series(), FPS).describe()
    assert "No joint moved appreciably" in text
    assert "No clear repetition" in text


def test_description_omits_joints_below_the_movement_threshold():
    small = np.full(150, 90.0)
    small[75:] = 90.0 + MOVING_THRESHOLD_DEG - 5
    text = measure(_series(l_elbow=_cycles(3, 150), r_hip=small), FPS).describe()
    assert "l_elbow" in text and "r_hip" not in text


def test_description_reports_a_notable_asymmetry():
    series = _series(
        l_knee=_cycles(3, 150, 40, 170),
        r_knee=_cycles(3, 150, 85, 95),
    )
    metrics = measure(series, FPS)
    assert metrics.asymmetry["knee"] > ASYMMETRY_THRESHOLD_DEG
    assert "Left/right difference" in metrics.describe()
