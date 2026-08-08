"""Loading and validation of the exercise database."""

from __future__ import annotations

import json

import pytest

from movement_coach.dataset import filter_by_equipment, load_exercises
from movement_coach.errors import DatasetError
from tests.conftest import make_record


def _write(tmp_path, payload):
    path = tmp_path / "exercises.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_and_indexes(database):
    assert len(database) == 9
    assert database.get("0001").name == "glute bridge"
    assert database.get("nope") is None


def test_secondary_muscles_are_normalized_on_load(database):
    """'core' and 'quadriceps' become searchable terms; 'hip flexors' drops out."""
    assert database.get("0001").secondary == {"hamstrings", "abs"}
    assert database.get("0005").secondary == frozenset()
    assert database.get("0005").secondary_raw == ("hip flexors",)


def test_muscles_includes_target(database):
    assert database.get("0003").muscles() == {"glutes", "quads", "calves"}


def test_steps_prefer_requested_language_then_english(database):
    exercise = database.get("0004")
    assert exercise.steps("zh")[0].endswith("第一步。")
    assert exercise.steps("en")[0].startswith("Step 1")
    assert exercise.steps("fr") == exercise.steps("en")


def test_count_by_target(database):
    counts = database.count_by_target()
    assert counts["abs"] == 4
    assert counts["abductors"] == 1


def test_missing_file_names_the_setup_instructions(tmp_path):
    with pytest.raises(DatasetError, match="README.md"):
        load_exercises(tmp_path / "absent.json")


def test_invalid_json(tmp_path):
    path = tmp_path / "exercises.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_exercises(path)


def test_top_level_must_be_a_list(tmp_path):
    with pytest.raises(DatasetError, match="must contain a list"):
        load_exercises(_write(tmp_path, {"exercises": []}))


def test_empty_list_rejected(tmp_path):
    with pytest.raises(DatasetError, match="no records"):
        load_exercises(_write(tmp_path, []))


def test_missing_required_field_names_the_field(tmp_path):
    record = make_record("0001", "x", "glutes")
    del record["target"]
    with pytest.raises(DatasetError, match="target"):
        load_exercises(_write(tmp_path, [record]))


def test_instruction_steps_must_be_a_non_empty_object(tmp_path):
    record = make_record("0001", "x", "glutes")
    record["instruction_steps"] = {}
    with pytest.raises(DatasetError, match="instruction_steps"):
        load_exercises(_write(tmp_path, [record]))


def test_secondary_muscles_must_be_a_list(tmp_path):
    record = make_record("0001", "x", "glutes")
    record["secondary_muscles"] = "glutes"
    with pytest.raises(DatasetError, match="secondary_muscles"):
        load_exercises(_write(tmp_path, [record]))


def test_duplicate_ids_rejected(tmp_path):
    payload = [make_record("0001", "a", "glutes"), make_record("0001", "b", "abs")]
    with pytest.raises(DatasetError, match="duplicate ids"):
        load_exercises(_write(tmp_path, payload))


def test_non_object_record(tmp_path):
    with pytest.raises(DatasetError, match="expected an object"):
        load_exercises(_write(tmp_path, ["just a string"]))


def test_filter_by_equipment(database):
    assert {e.id for e in filter_by_equipment(database, ["barbell"])} == {"0003"}
    assert {e.id for e in filter_by_equipment(database, ["BARBELL"])} == {"0003"}


def test_empty_equipment_filter_means_no_restriction(database):
    assert len(filter_by_equipment(database, None)) == len(database)
    assert len(filter_by_equipment(database, [])) == len(database)


@pytest.mark.real_db
def test_real_dataset_shape(real_database):
    assert len(real_database) == 1324
    assert len(real_database.targets()) == 19
    assert len(real_database.equipment_types()) == 28
