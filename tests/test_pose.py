"""Skeleton feature extraction.

The pose model itself is not exercised here -- these are the pure numeric
functions that turn keypoints into comparable features, which is where the
failures that matter have actually occurred.
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_coach.pose import (
    ANGLE_NAMES,
    TRAJECTORY_POINTS,
    _smooth,
    frame_angles,
    mirror,
    mirror_trajectory,
    rom_signature,
    summarise,
)


# -- smoothing -------------------------------------------------------------


def test_constant_series_has_no_range():
    """Regression: convolving with mode='same' zero-padded the edges, so a
    joint holding 176 degrees appeared to span 70 on a 12-frame reference."""
    smoothed = _smooth(np.full(12, 176.0))
    assert smoothed.max() - smoothed.min() == pytest.approx(0.0, abs=1e-9)


def test_single_frame_spike_is_removed():
    """One mis-detected frame must not invent range of motion."""
    values = np.full(12, 176.0)
    values[6] = 61.0
    smoothed = _smooth(values)
    assert smoothed.max() - smoothed.min() == pytest.approx(0.0, abs=1e-6)


def test_real_movement_survives_smoothing():
    curl = np.array([179, 176, 165, 31, 32, 29, 19, 28, 32, 31, 165, 176], float)
    smoothed = _smooth(curl)
    assert 130 <= smoothed.max() - smoothed.min() <= 160


def test_gaps_are_interpolated_not_dropped():
    values = np.array([10.0, np.nan, 30.0, np.nan, 50.0, 60.0, 70.0])
    assert not np.isnan(_smooth(values)).any()


def test_too_few_observations_passes_through():
    values = np.array([np.nan, 1.0, np.nan])
    assert np.isnan(_smooth(values)).sum() == 2


# -- angles ----------------------------------------------------------------


def _keypoints(**overrides) -> tuple[np.ndarray, np.ndarray]:
    """A neutral standing skeleton, with named joints overridable."""
    points = np.zeros((17, 2))
    points[5], points[6] = (-10, 100), (10, 100)      # shoulders
    points[7], points[8] = (-10, 60), (10, 60)        # elbows
    points[9], points[10] = (-10, 20), (10, 20)       # wrists
    points[11], points[12] = (-8, 50), (8, 50)        # hips
    points[13], points[14] = (-8, 25), (8, 25)        # knees
    points[15], points[16] = (-8, 0), (8, 0)          # ankles
    for index, value in overrides.items():
        points[int(index)] = value
    return points, np.ones(17)


def test_straight_limb_measures_180_degrees():
    points, scores = _keypoints()
    angles = frame_angles(points, scores)
    assert angles[ANGLE_NAMES.index("l_elbow")] == pytest.approx(180.0, abs=1e-6)
    assert angles[ANGLE_NAMES.index("l_knee")] == pytest.approx(180.0, abs=1e-6)


def test_bent_elbow_measures_a_right_angle():
    points, scores = _keypoints(**{"9": (-50, 60)})  # wrist out to the side
    angles = frame_angles(points, scores)
    assert angles[ANGLE_NAMES.index("l_elbow")] == pytest.approx(90.0, abs=1e-6)


def test_upright_trunk_is_near_zero():
    points, scores = _keypoints()
    assert frame_angles(points, scores)[ANGLE_NAMES.index("trunk")] == pytest.approx(
        0.0, abs=1e-6
    )


def test_low_confidence_joints_report_nan_rather_than_a_guess():
    points, scores = _keypoints()
    scores[9] = 0.1
    angles = frame_angles(points, scores)
    assert np.isnan(angles[ANGLE_NAMES.index("l_elbow")])
    assert not np.isnan(angles[ANGLE_NAMES.index("l_knee")])


# -- summarising -----------------------------------------------------------


def test_summarise_shapes_and_range():
    series = np.tile(np.linspace(20, 160, 30)[:, None], (1, len(ANGLE_NAMES)))
    rom, trajectory = summarise(series)
    assert rom.shape == (len(ANGLE_NAMES),)
    assert trajectory.shape == (len(ANGLE_NAMES), TRAJECTORY_POINTS)
    assert 100 <= rom[0] <= 140  # 5th-95th percentile of a 20..160 ramp
    assert trajectory.min() >= 0.0 and trajectory.max() <= 1.0


def test_summarise_rejects_a_mostly_undetected_clip():
    series = np.full((20, len(ANGLE_NAMES)), np.nan)
    series[:5] = 90.0
    assert summarise(series) is None


def test_summarise_of_a_still_clip_has_no_range():
    series = np.full((20, len(ANGLE_NAMES)), 90.0)
    rom, _ = summarise(series)
    assert rom.max() == pytest.approx(0.0, abs=1e-6)


# -- mirroring and signature ----------------------------------------------


def test_mirror_swaps_left_and_right():
    rom = np.arange(len(ANGLE_NAMES), dtype=float)
    swapped = mirror(rom)
    assert swapped[0] == rom[1] and swapped[1] == rom[0]
    assert swapped[2] == rom[3] and swapped[3] == rom[2]
    assert swapped[-1] == rom[-1], "trunk has no left/right pair"


def test_mirror_is_its_own_inverse():
    rom = np.arange(len(ANGLE_NAMES), dtype=float)
    assert np.array_equal(mirror(mirror(rom)), rom)


def test_mirror_trajectory_swaps_rows():
    trajectory = np.arange(len(ANGLE_NAMES) * 4, dtype=float).reshape(
        len(ANGLE_NAMES), 4
    )
    swapped = mirror_trajectory(trajectory)
    assert np.array_equal(swapped[0], trajectory[1])
    assert np.array_equal(swapped[-1], trajectory[-1])


def test_signature_is_unit_length():
    assert np.linalg.norm(rom_signature(np.array([3.0, 4.0] + [0.0] * 7))) == pytest.approx(1.0)


def test_signature_ignores_overall_magnitude():
    """A shallow squat and a deep one must point the same way."""
    shallow = np.array([1.0, 2.0, 3.0] + [0.0] * 6)
    deep = shallow * 4
    assert np.allclose(rom_signature(shallow), rom_signature(deep))


def test_signature_of_a_still_clip_is_safe():
    assert not np.isnan(rom_signature(np.zeros(len(ANGLE_NAMES)))).any()
