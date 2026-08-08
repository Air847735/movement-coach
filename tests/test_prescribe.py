"""Prescription retrieval: scoring, set cover, and the grounding guarantee."""

from __future__ import annotations

import pytest

from movement_coach.errors import PrescriptionError
from movement_coach.prescribe import (
    SECONDARY_WEIGHT,
    covers,
    prescribe,
    score,
    verify_grounded,
)


def test_score_weights_target_above_secondary(database):
    squat = database.get("0003")  # target glutes, secondary quads + calves
    assert score(squat, {"glutes"}) == 1.0
    assert score(squat, {"quads"}) == SECONDARY_WEIGHT
    assert score(squat, {"glutes", "quads", "calves"}) == 1.0 + 2 * SECONDARY_WEIGHT


def test_score_ignores_muscles_outside_the_weak_set(database):
    assert score(database.get("0008"), {"glutes"}) == 0.0


def test_covers_reports_intersection(database):
    assert covers(database.get("0003"), {"glutes", "calves", "biceps"}) == {
        "glutes",
        "calves",
    }


def test_single_muscle_picks_a_direct_target(database):
    result = prescribe(database, ["abductors"])
    assert result.exercise_ids() == ("0002",)
    assert not result.uncovered


def test_scarce_muscle_is_covered_before_abundant_ones(database):
    """Regression: an unweighted greedy pass drifted towards over-represented
    targets, so 'abductors' (1 candidate here, 5 in the real data) lost out to
    'abs' (4 here, 169 in the real data)."""
    result = prescribe(database, ["abs", "abductors"])
    assert "0002" in result.exercise_ids()
    assert not result.uncovered


def test_prefers_a_primary_target_over_an_incidental_secondary(database):
    """Regression: 'wind sprints' was chosen as calf training because every
    running entry lists calves as a secondary muscle."""
    result = prescribe(database, ["calves"])
    assert result.exercise_ids() == ("0008",)


def test_covers_multiple_muscles_with_few_exercises(database):
    result = prescribe(database, ["glutes", "abductors", "abs", "calves"])
    assert len(result.items) <= 4
    assert not result.uncovered
    assert result.covered == {"glutes", "abductors", "abs", "calves"}


def test_respects_max_items_and_reports_what_is_left(database):
    result = prescribe(database, ["glutes", "abductors", "biceps"], max_items=1)
    assert len(result.items) == 1
    assert result.uncovered


def test_equipment_filter_restricts_selection(database):
    result = prescribe(database, ["glutes"], equipment=["barbell"])
    assert result.exercise_ids() == ("0003",)


def test_equipment_is_relaxed_rather_than_returning_nothing(database):
    result = prescribe(database, ["glutes"], equipment=["kettlebell"])
    assert result.equipment_relaxed is True
    assert result.items


def test_equipment_relaxation_can_be_disabled(database):
    with pytest.raises(PrescriptionError, match="relaxation is disabled"):
        prescribe(
            database, ["glutes"], equipment=["kettlebell"], relax_equipment_if_empty=False
        )


def test_unsearchable_muscle_is_reported_not_substituted(database):
    """'hip flexors' has no target anywhere; it must come back uncovered."""
    result = prescribe(database, ["hip flexors"])
    assert result.items == ()
    assert result.uncovered == frozenset({"hip flexors"})


def test_partially_coverable_request(database):
    result = prescribe(database, ["glutes", "hip flexors"])
    assert result.covered == {"glutes"}
    assert result.uncovered == frozenset({"hip flexors"})


def test_empty_weak_set_rejected(database):
    with pytest.raises(PrescriptionError, match="no weak muscles"):
        prescribe(database, [])


def test_non_positive_max_items_rejected(database):
    with pytest.raises(PrescriptionError, match="max_items must be positive"):
        prescribe(database, ["glutes"], max_items=0)


def test_no_exercise_is_prescribed_twice(database):
    result = prescribe(database, ["abs", "glutes", "calves", "abductors"], max_items=5)
    ids = result.exercise_ids()
    assert len(ids) == len(set(ids))


def test_selection_is_deterministic(database):
    muscles = ["abs", "glutes", "calves", "abductors"]
    first = prescribe(database, muscles).exercise_ids()
    for _ in range(5):
        assert prescribe(database, muscles).exercise_ids() == first


def test_verify_grounded_accepts_a_real_prescription(database):
    verify_grounded(prescribe(database, ["glutes"]), database)


def test_verify_grounded_rejects_an_unknown_id(database):
    """The central rule: a prescription may not contain an invented exercise."""
    from dataclasses import replace

    result = prescribe(database, ["glutes"])
    forged = replace(
        result,
        items=(
            replace(
                result.items[0],
                exercise=replace(result.items[0].exercise, id="9999"),
            ),
        ),
    )
    with pytest.raises(PrescriptionError, match="absent from the database"):
        verify_grounded(forged, database)


@pytest.mark.real_db
@pytest.mark.parametrize(
    "weak",
    [
        ["glutes", "abductors", "abs", "quads"],
        ["hamstrings", "calves", "spine"],
        ["delts", "triceps", "upper back"],
        ["abductors", "adductors"],
    ],
)
def test_real_dataset_prescriptions_are_grounded(real_database, weak):
    result = prescribe(real_database, weak)
    verify_grounded(result, real_database)
    assert result.items
    assert all(real_database.get(i) is not None for i in result.exercise_ids())


@pytest.mark.real_db
def test_real_dataset_covers_a_four_muscle_request_in_few_exercises(real_database):
    result = prescribe(real_database, ["glutes", "abductors", "abs", "quads"])
    assert not result.uncovered
    assert len(result.items) <= 4


@pytest.mark.real_db
def test_real_dataset_bodyweight_only_still_works(real_database):
    result = prescribe(
        real_database, ["abductors", "abs", "calves"], equipment=["body weight"]
    )
    verify_grounded(result, real_database)
    assert all(item.exercise.equipment == "body weight" for item in result.items)
