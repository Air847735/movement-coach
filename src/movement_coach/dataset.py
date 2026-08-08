"""Loading and validating the exercise database.

The database is the only source of prescribable exercises: nothing downstream
may invent one. `load_exercises` therefore validates eagerly and raises
`DatasetError` on startup rather than letting a malformed record surface as a
missing exercise during retrieval.

The dataset ships two redundant fields -- `category` duplicates `body_part`
and `muscle_group` duplicates `secondary_muscles[0]` for all 1,324 records --
so neither is read here. See the dataset audit in `docs/architecture.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

from .errors import DatasetError
from .muscles import normalize

#: Language codes present in ``instruction_steps``. ``en`` is the fallback.
DEFAULT_LANGUAGE = "zh"

_REQUIRED_FIELDS = ("id", "name", "target", "equipment", "instruction_steps")


@dataclass(frozen=True)
class Exercise:
    """One prescribable exercise.

    ``secondary`` holds the *normalised* secondary muscles (dataset wording
    mapped onto ``target`` vocabulary); ``secondary_raw`` keeps the original
    strings so nothing is lost when reporting to a user.
    """

    id: str
    name: str
    target: str
    equipment: str
    body_part: str
    secondary: frozenset[str]
    secondary_raw: tuple[str, ...]
    instruction_steps: Mapping[str, Sequence[str]]

    def steps(self, language: str = DEFAULT_LANGUAGE) -> Sequence[str]:
        """Instructions in ``language``, falling back to English then to any."""
        steps = self.instruction_steps
        for code in (language, "en"):
            if steps.get(code):
                return steps[code]
        for value in steps.values():
            if value:
                return value
        return ()

    def muscles(self) -> frozenset[str]:
        """Every normalised muscle this exercise trains, primary included."""
        return self.secondary | {self.target}


@dataclass(frozen=True)
class ExerciseDatabase:
    """An immutable, indexed view over the exercise records."""

    exercises: tuple[Exercise, ...]
    _by_id: Dict[str, Exercise] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id.update({item.id: item for item in self.exercises})

    def __len__(self) -> int:
        return len(self.exercises)

    def __iter__(self) -> Iterator[Exercise]:
        return iter(self.exercises)

    def get(self, exercise_id: str) -> Exercise | None:
        """Look up by ``id``; used to verify a prescription is grounded."""
        return self._by_id.get(exercise_id)

    def equipment_types(self) -> frozenset[str]:
        return frozenset(item.equipment for item in self.exercises)

    def targets(self) -> frozenset[str]:
        return frozenset(item.target for item in self.exercises)

    def count_by_target(self) -> Dict[str, int]:
        """How many exercises primarily train each muscle.

        Retrieval uses this to cover scarce muscles first -- ``abductors`` has
        5 candidates while ``abs`` has 169, so an unweighted greedy pass drifts
        towards abs work.
        """
        counts: Dict[str, int] = {}
        for item in self.exercises:
            counts[item.target] = counts.get(item.target, 0) + 1
        return counts


def _validate_record(raw: object, index: int) -> Exercise:
    if not isinstance(raw, dict):
        raise DatasetError(f"record {index} is {type(raw).__name__}, expected an object")

    missing = [key for key in _REQUIRED_FIELDS if key not in raw]
    if missing:
        raise DatasetError(f"record {index} is missing field(s): {', '.join(missing)}")

    steps = raw["instruction_steps"]
    if not isinstance(steps, dict) or not steps:
        raise DatasetError(f"record {index} ({raw['id']}): instruction_steps must be a non-empty object")

    secondary_raw = raw.get("secondary_muscles") or []
    if not isinstance(secondary_raw, list):
        raise DatasetError(f"record {index} ({raw['id']}): secondary_muscles must be a list")

    normalized = {n for n in (normalize(m) for m in secondary_raw) if n is not None}

    return Exercise(
        id=str(raw["id"]),
        name=str(raw["name"]),
        target=str(raw["target"]),
        equipment=str(raw["equipment"]),
        body_part=str(raw.get("body_part", "")),
        secondary=frozenset(normalized),
        secondary_raw=tuple(str(m) for m in secondary_raw),
        instruction_steps={
            str(lang): tuple(str(s) for s in value)
            for lang, value in steps.items()
            if isinstance(value, list)
        },
    )


def load_exercises(path: str | Path) -> ExerciseDatabase:
    """Read and validate ``exercises.json``.

    Raises `DatasetError` if the file is absent, unparseable, empty, not a
    list, or if any record lacks a required field. Duplicate ids are rejected
    too, since retrieval verifies prescriptions by id.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"exercise database not found at {path}. "
            "See README.md 'Setup' for the download command and expected checksum."
        )

    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise DatasetError(f"could not read {path}: {exc}") from exc

    if not isinstance(raw, list):
        raise DatasetError(f"{path} must contain a list of records, found {type(raw).__name__}")
    if not raw:
        raise DatasetError(f"{path} contains no records")

    exercises = tuple(_validate_record(record, i) for i, record in enumerate(raw))

    ids = [item.id for item in exercises]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise DatasetError(f"{path} contains duplicate ids: {', '.join(duplicates[:5])}")

    return ExerciseDatabase(exercises=exercises)


def filter_by_equipment(
    exercises: Iterable[Exercise], equipment: Iterable[str] | None
) -> List[Exercise]:
    """Keep only exercises using the available equipment.

    ``None`` or an empty selection means "no restriction" rather than "nothing
    available", so an unset filter never silently empties the pool.
    """
    if not equipment:
        return list(exercises)
    allowed = {e.strip().lower() for e in equipment}
    return [item for item in exercises if item.equipment.lower() in allowed]
