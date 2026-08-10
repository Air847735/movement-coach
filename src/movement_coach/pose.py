"""Skeleton features: turning a video into viewpoint-tolerant motion numbers.

Sampling stills loses the one thing that separates several exercise pairs in
the dataset -- direction. Leg extension and leg curl pass through the same
postures and differ only in which way the knee travels, which no single frame
records. Running pose on every frame recovers that continuously, and cheaply:
the whole clip costs less than a handful of frames through the vision model.

Two levels of feature come out, in decreasing robustness:

* **ROM** -- how far each joint travels. Independent of how many repetitions
  the clip contains and largely tolerant of camera angle. This is the coarse
  fingerprint of a movement.
* **Trajectory** -- each angle resampled to a fixed length. Finer, but more
  sensitive to viewpoint.

Angles are used rather than raw coordinates because they are invariant to
image scale and to where in the frame the person stands.

Requires the optional ``pose`` extra; see ``docs/pose-setup.md``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

#: COCO-17 keypoint indices, as returned by rtmlib's ``Body``.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

#: The joint angles extracted per frame, in fixed order.
ANGLE_NAMES: Tuple[str, ...] = (
    "l_elbow",
    "r_elbow",
    "l_knee",
    "r_knee",
    "l_hip",
    "r_hip",
    "l_shoulder",
    "r_shoulder",
    "trunk",
)

#: Angle triples, as (a, b, c) with the angle measured at b.
_TRIPLES: Tuple[Tuple[int, int, int], ...] = (
    (L_SHOULDER, L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE, R_ANKLE),
    (L_SHOULDER, L_HIP, L_KNEE),
    (R_SHOULDER, R_HIP, R_KNEE),
    (L_HIP, L_SHOULDER, L_ELBOW),
    (R_HIP, R_SHOULDER, R_ELBOW),
)

#: Length every trajectory is resampled to, so clips of different duration and
#: repetition count stay comparable.
TRAJECTORY_POINTS = 16

#: Keypoint confidence below which a joint is treated as unobserved.
#:
#: Filmed from one side, the far limb is occluded and the detector still
#: returns a keypoint for it -- a plausible guess with middling confidence.
#: Those guesses swing frame to frame and invent range of motion: at 0.3 a
#: seated leg extension reported the elbow as its largest mover and a bench
#: press reported 93 degrees of hip travel. At 0.5 both artefacts disappear
#: while the real movement is retained; at 0.65 genuine joints start dropping
#: out (a squat kept only one knee).
MIN_CONFIDENCE = 0.5

#: A clip needs this fraction of frames with a usable skeleton to be summarised.
MIN_USABLE_FRACTION = 0.5


def build_pose_model(mode: str = "lightweight", device: str = "cpu"):
    """Construct the pose estimator.

    Defaults to the fastest configuration on CPU, which measured faster than
    GPU for this model size and leaves the GPU entirely to the vision model.
    See ``docs/pose-setup.md`` for the benchmark.
    """
    from rtmlib import Body  # imported lazily: optional dependency

    return Body(mode=mode, backend="onnxruntime", device=device)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator < 1e-9:
        return float("nan")
    cosine = float(np.dot(ba, bc)) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def frame_angles(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The nine joint angles for one frame, ``nan`` where a joint is unseen."""
    values = np.full(len(ANGLE_NAMES), np.nan)
    for index, (a, b, c) in enumerate(_TRIPLES):
        if min(scores[a], scores[b], scores[c]) >= MIN_CONFIDENCE:
            values[index] = _angle(keypoints[a], keypoints[b], keypoints[c])

    # Trunk lean: shoulder midpoint to hip midpoint, against vertical. Measured
    # against the image axis rather than another limb, so it survives when the
    # legs are occluded.
    if min(scores[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]) >= MIN_CONFIDENCE:
        shoulders = (keypoints[L_SHOULDER] + keypoints[R_SHOULDER]) / 2
        hips = (keypoints[L_HIP] + keypoints[R_HIP]) / 2
        spine = shoulders - hips
        if np.linalg.norm(spine) >= 1e-9:
            values[-1] = float(
                np.degrees(np.arctan2(abs(spine[0]), abs(spine[1]) + 1e-9))
            )
    return values


def angle_series(
    model, frames: Sequence[np.ndarray], *, upscale: int = 1
) -> np.ndarray:
    """Run pose over every frame and return an ``(n_frames, 9)`` angle array.

    ``upscale`` enlarges each frame first; the dataset GIFs are 180x180 and
    detect far more reliably when magnified.
    """
    series = np.full((len(frames), len(ANGLE_NAMES)), np.nan)
    for index, frame in enumerate(frames):
        image = frame
        if upscale > 1:
            image = cv2_resize(frame, upscale)
        keypoints, scores = model(image)
        if len(keypoints) == 0:
            continue
        # Largest detection wins: incidental bystanders are smaller than the
        # person the clip is about.
        best = int(np.argmax([_bbox_area(k) for k in keypoints]))
        series[index] = frame_angles(keypoints[best], scores[best])
    return series


def cv2_resize(frame: np.ndarray, factor: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    return cv2.resize(
        frame, (width * factor, height * factor), interpolation=cv2.INTER_CUBIC
    )


def _bbox_area(keypoints: np.ndarray) -> float:
    span = keypoints.max(axis=0) - keypoints.min(axis=0)
    return float(span[0] * span[1])


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Fill gaps, drop single-frame spikes, then average.

    Two failure modes have to be handled, both of which invent range of motion
    that is not there:

    * A single mis-detected frame -- a limb swapped left for right, say --
      moves one sample by a hundred degrees. A median filter removes it; a mean
      filter would smear it across its neighbours instead.
    * Edge padding. ``np.convolve(..., mode="same")`` pads with zeros, so on a
      12-frame reference GIF the first and last samples get dragged towards
      zero. A joint holding a steady 176 degrees then appears to span 70. The
      series is padded with its own edge values instead.
    """
    observed = ~np.isnan(values)
    if observed.sum() < 3:
        return values

    filled = np.interp(
        np.arange(values.size), np.flatnonzero(observed), values[observed]
    )

    # Median filter, width 3: enough to remove an isolated bad frame.
    if filled.size >= 3:
        stacked = np.stack(
            [
                np.concatenate(([filled[0]], filled[:-1])),
                filled,
                np.concatenate((filled[1:], [filled[-1]])),
            ]
        )
        filled = np.median(stacked, axis=0)

    if window <= 1 or filled.size < window:
        return filled

    pad = window // 2
    padded = np.pad(filled, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def summarise(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
    """Reduce an angle series to ``(rom, trajectory)``.

    ``rom`` is the 5th-to-95th percentile span of each angle, in degrees --
    percentiles rather than min/max so one bad frame cannot invent a large
    range. ``trajectory`` is each angle resampled to `TRAJECTORY_POINTS` and
    normalised to 0..1 within its own range, which keeps the shape of the
    movement while discarding absolute offset.

    Returns ``None`` when too few frames yielded a skeleton to be meaningful.
    """
    usable = np.mean(~np.isnan(series).all(axis=1))
    if usable < MIN_USABLE_FRACTION:
        return None

    rom = np.zeros(len(ANGLE_NAMES))
    trajectory = np.zeros((len(ANGLE_NAMES), TRAJECTORY_POINTS))
    grid = np.linspace(0, 1, TRAJECTORY_POINTS)

    for index in range(len(ANGLE_NAMES)):
        column = _smooth(series[:, index])
        observed = column[~np.isnan(column)]
        if observed.size < 4:
            continue
        low, high = np.percentile(observed, [5, 95])
        rom[index] = max(high - low, 0.0)

        positions = np.linspace(0, 1, observed.size)
        resampled = np.interp(grid, positions, observed)
        span = resampled.max() - resampled.min()
        trajectory[index] = (
            (resampled - resampled.min()) / span if span > 1e-6 else 0.5
        )
    return rom, trajectory


def mirror(rom: np.ndarray) -> np.ndarray:
    """Swap left and right, for comparing clips filmed from opposite sides."""
    swapped = rom.copy()
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
        swapped[left], swapped[right] = rom[right], rom[left]
    return swapped


def mirror_trajectory(trajectory: np.ndarray) -> np.ndarray:
    swapped = trajectory.copy()
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
        swapped[left], swapped[right] = trajectory[right], trajectory[left]
    return swapped


def rom_signature(rom: np.ndarray) -> np.ndarray:
    """Normalise a ROM vector to unit length.

    Comparing direction rather than magnitude makes the fingerprint tolerant of
    how completely a given person performs the movement: a shallow squat and a
    deep one point the same way, they differ in size.
    """
    norm = float(np.linalg.norm(rom))
    return rom / norm if norm > 1e-9 else rom
