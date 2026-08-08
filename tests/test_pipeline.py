"""End-to-end orchestration with a stand-in model, plus the layering rule.

`MovementCoach` is driven here through a fake VLM, so the whole grounding
path -- free-form causes, constrained mapping, retrieval, verification -- is
exercised without a GPU.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from movement_coach import MovementCoach, format_report
from movement_coach.vlm import Assessment


class FakeVLM:
    """Records what it was asked and returns scripted replies."""

    model = "fake"

    def __init__(self, description="a barbell back squat", problems=("shallow depth",),
                 causes=("gluteus medius", "limited ankle dorsiflexion"),
                 mapped=("abductors", "glutes")):
        self._description = description
        self._problems = tuple(problems)
        self._causes = tuple(causes)
        self._mapped = tuple(mapped)
        self.described = 0

    def health(self):
        return None

    def describe(self, frames):
        self.described += 1
        return self._description

    def assess(self, frames, description):
        return self._problems

    def infer_causes(self, description, problems):
        return self._causes if problems else ()

    def map_to_muscles(self, causes):
        return self._mapped if causes else ()

    def analyse(self, frames, description=None):
        resolved = description.strip() if description and description.strip() else self.describe(frames)
        problems = self.assess(frames, resolved)
        return Assessment(
            description=resolved,
            problems=problems,
            causes=self.infer_causes(resolved, problems),
        )


@pytest.fixture
def coach(database):
    return MovementCoach(database, FakeVLM())


def test_prescribe_for_maps_causes_and_grounds_the_result(coach):
    assessment = coach.vlm.analyse(["frame"])
    diagnosis = coach.prescribe_for(assessment)

    assert diagnosis.weak_muscles == {"abductors", "glutes"}
    assert diagnosis.prescription is not None
    assert all(
        coach.database.get(i) is not None for i in diagnosis.prescription.exercise_ids()
    )


def test_unmappable_causes_are_reported_verbatim(database):
    coach = MovementCoach(
        database,
        FakeVLM(causes=("limited ankle dorsiflexion",), mapped=("ankle mobility",)),
    )
    diagnosis = coach.prescribe_for(coach.vlm.analyse(["frame"]))

    assert diagnosis.weak_muscles == frozenset()
    assert diagnosis.unmapped_causes == ("ankle mobility",)
    assert diagnosis.prescription is None
    assert "no cause mapped" in diagnosis.prescription_error


def test_mapping_stage_returning_nothing_keeps_the_original_causes(database):
    coach = MovementCoach(database, FakeVLM(mapped=()))
    diagnosis = coach.prescribe_for(coach.vlm.analyse(["frame"]))

    assert diagnosis.unmapped_causes == coach.vlm._causes
    assert diagnosis.prescription is None


def test_clean_execution_produces_no_prescription(database):
    coach = MovementCoach(database, FakeVLM(problems=()))
    diagnosis = coach.prescribe_for(coach.vlm.analyse(["frame"]))

    assert not diagnosis.has_issues
    assert diagnosis.causes == ()
    assert diagnosis.prescription is None
    assert diagnosis.prescription_error is None


def test_equipment_restriction_flows_through(coach):
    diagnosis = coach.prescribe_for(
        coach.vlm.analyse(["frame"]), equipment=["barbell"]
    )
    assert diagnosis.prescription is not None
    # Only one barbell entry exists, so relaxation must kick in for abductors.
    assert diagnosis.prescription.equipment_relaxed or all(
        item.exercise.equipment == "barbell" for item in diagnosis.prescription.items
    )


def test_user_description_overrides_the_model(coach):
    assessment = coach.vlm.analyse(["frame"], "a taekwondo side kick")
    assert assessment.description == "a taekwondo side kick"
    assert coach.vlm.described == 0, "describe() should be skipped when the user supplied wording"


def test_movement_outside_the_database_still_produces_a_prescription(database):
    """A kick is not in the dataset, but the flow must not stop there."""
    coach = MovementCoach(
        database,
        FakeVLM(
            description="a taekwondo roundhouse kick",
            problems=("支撐腳不穩",),
            causes=("gluteus medius",),
            mapped=("abductors",),
        ),
    )
    diagnosis = coach.prescribe_for(coach.vlm.analyse(["frame"]))

    assert diagnosis.description == "a taekwondo roundhouse kick"
    assert diagnosis.prescription is not None
    assert diagnosis.prescription.items


# -- report rendering ------------------------------------------------------


def test_report_includes_the_hedged_wording(coach):
    text = format_report(coach.prescribe_for(coach.vlm.analyse(["frame"])))
    assert "非診斷" in text
    assert "訓練處方" in text


def test_report_for_a_clean_movement(database):
    coach = MovementCoach(database, FakeVLM(problems=()))
    text = format_report(coach.prescribe_for(coach.vlm.analyse(["frame"])))
    assert "未發現明顯問題" in text


def test_report_names_unmapped_items(database):
    coach = MovementCoach(
        database, FakeVLM(causes=("ankle mobility",), mapped=("ankle mobility",))
    )
    text = format_report(coach.prescribe_for(coach.vlm.analyse(["frame"])))
    assert "無對應動作的項目" in text


# -- layering --------------------------------------------------------------


def test_core_package_does_not_pull_in_a_web_framework():
    """AGENTS.md: the core layer must be usable without the API layer.

    Run in a subprocess so an earlier test importing FastAPI cannot mask a
    real dependency leak.
    """
    code = (
        "import sys, movement_coach;"
        "leaked = [m for m in ('fastapi', 'starlette', 'uvicorn') if m in sys.modules];"
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"core imported web modules: {result.stdout}"


def test_full_flow_works_without_starting_a_server(database, tmp_path):
    """The library is the product; HTTP is optional."""
    coach = MovementCoach(database, FakeVLM())
    diagnosis = coach.prescribe_for(coach.vlm.analyse(["frame"]))
    assert format_report(diagnosis)
    assert diagnosis.prescription is not None
