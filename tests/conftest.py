"""Shared fixtures.

Most tests run against a hand-built database so they stay fast and
deterministic. Tests that assert something about the *real* dataset are
marked with `real_db` and skip when `data/exercises.json` has not been
downloaded, so a fresh clone still has a green suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from movement_coach.dataset import ExerciseDatabase, load_exercises

REAL_DATASET = Path(__file__).resolve().parents[1] / "data" / "exercises.json"


def make_record(
    exercise_id: str,
    name: str,
    target: str,
    equipment: str = "body weight",
    secondary: List[str] | None = None,
    body_part: str = "upper legs",
) -> Dict[str, Any]:
    """Build one dataset-shaped record."""
    return {
        "id": exercise_id,
        "name": name,
        "target": target,
        "equipment": equipment,
        "body_part": body_part,
        "secondary_muscles": secondary or [],
        "instruction_steps": {
            "en": [f"Step 1 of {name}.", f"Step 2 of {name}."],
            "zh": [f"{name} 第一步。", f"{name} 第二步。"],
        },
    }


@pytest.fixture
def records() -> List[Dict[str, Any]]:
    """A small database with the traits that matter for retrieval tests.

    ``abs`` is deliberately over-represented and ``abductors`` scarce, which
    mirrors the real 169-vs-5 imbalance that biased the prototype.
    """
    return [
        make_record("0001", "glute bridge", "glutes", secondary=["hamstrings", "core"]),
        make_record("0002", "side plank abduction", "abductors", secondary=["obliques"]),
        make_record("0003", "barbell squat", "glutes", "barbell", ["quadriceps", "calves"]),
        make_record("0004", "crunch", "abs"),
        make_record("0005", "sit-up", "abs", secondary=["hip flexors"]),
        make_record("0006", "leg raise", "abs", secondary=["hip flexors"]),
        make_record("0007", "wind sprints", "abs", secondary=["calves", "quadriceps"]),
        make_record("0008", "calf raise", "calves"),
        make_record("0009", "dumbbell curl", "biceps", "dumbbell", ["forearms"]),
    ]


@pytest.fixture
def database(records: List[Dict[str, Any]], tmp_path: Path) -> ExerciseDatabase:
    path = tmp_path / "exercises.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return load_exercises(path)


@pytest.fixture(scope="session")
def real_database() -> ExerciseDatabase:
    if not REAL_DATASET.is_file():
        pytest.skip(f"{REAL_DATASET} not downloaded; see README.md Setup")
    return load_exercises(REAL_DATASET)
