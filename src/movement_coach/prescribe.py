"""Prescription retrieval: weak muscles in, real exercises out.

This is the "grounding" half of the system. Everything upstream is free-form
language from a model; from here on the output is drawn exclusively from the
exercise database, so every prescribed exercise carries an id that can be
verified against `exercises.json`.

The selection is a greedy set cover over the weak-muscle set, with two
corrections found while prototyping against the real data:

* Secondary muscles are worth far less than the primary target. At an equal
  0.5 weight, `wind sprints` was selected as calf training purely because
  every running entry lists ``calves`` as secondary.
* Scarce muscles are covered first. ``abs`` has 169 candidate exercises and
  ``abductors`` has 5; a plain greedy pass keeps picking abs work that happens
  to brush the scarce muscle instead of training it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from .dataset import DEFAULT_LANGUAGE, Exercise, ExerciseDatabase, filter_by_equipment
from .errors import PrescriptionError

#: Weight of a muscle listed in ``secondary_muscles`` relative to ``target``.
#: Deliberately low: the dataset lists secondary muscles generously and
#: without any indication of how much each contributes.
SECONDARY_WEIGHT = 0.25

#: Default cap on prescription length. Long lists stop being actionable.
DEFAULT_MAX_ITEMS = 4


@dataclass(frozen=True)
class PrescribedExercise:
    """One exercise in a prescription, with what it was chosen to cover."""

    exercise: Exercise
    covers: frozenset[str]
    score: float

    @property
    def id(self) -> str:
        return self.exercise.id

    @property
    def name(self) -> str:
        return self.exercise.name

    def steps(self, language: str = DEFAULT_LANGUAGE) -> Sequence[str]:
        return self.exercise.steps(language)


@dataclass(frozen=True)
class Prescription:
    """The selected exercises plus what could not be covered.

    ``uncovered`` lists weak muscles no available exercise reaches -- for
    example ``hip flexors`` has no ``target`` entry anywhere in the dataset.
    It is reported, never quietly dropped.
    """

    items: tuple[PrescribedExercise, ...]
    requested: frozenset[str]
    uncovered: frozenset[str]
    equipment_relaxed: bool = False

    @property
    def covered(self) -> frozenset[str]:
        return frozenset(self.requested - self.uncovered)

    def exercise_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.items)


def score(exercise: Exercise, weak: Set[str]) -> float:
    """How much of ``weak`` this exercise trains, primary weighted highest."""
    total = 1.0 if exercise.target in weak else 0.0
    total += SECONDARY_WEIGHT * len(exercise.secondary & weak)
    return total


def covers(exercise: Exercise, weak: Set[str]) -> frozenset[str]:
    """Which muscles of ``weak`` this exercise trains at all."""
    return frozenset(exercise.muscles() & weak)


def _scarcity_order(weak: Set[str], pool: Sequence[Exercise]) -> List[str]:
    """Weak muscles ordered from fewest to most directly-targeting exercises.

    Ties break alphabetically so the whole selection stays deterministic.
    """
    counts: Dict[str, int] = {muscle: 0 for muscle in weak}
    for item in pool:
        if item.target in counts:
            counts[item.target] += 1
    return sorted(weak, key=lambda muscle: (counts[muscle], muscle))


def _best_for(
    muscle: str, remaining: Set[str], weak: Set[str], pool: Sequence[Exercise]
) -> Exercise | None:
    """Pick the exercise that trains ``muscle`` and covers the most of the rest.

    Exercises whose primary target *is* ``muscle`` win over ones that merely
    list it as secondary, so a scarce muscle gets real work rather than a
    by-product. Final tie-break is on id, making the result reproducible.
    """
    candidates = [item for item in pool if muscle in item.muscles()]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.target == muscle,
            len(covers(item, remaining)),
            score(item, weak),
            # Negated so that, all else equal, the lowest id wins.
            tuple(-ord(c) for c in item.id),
        ),
    )


def prescribe(
    database: ExerciseDatabase,
    weak_muscles: Iterable[str],
    *,
    equipment: Iterable[str] | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    relax_equipment_if_empty: bool = True,
) -> Prescription:
    """Select a short exercise list covering as many weak muscles as possible.

    ``weak_muscles`` must already be normalised to ``target`` vocabulary; use
    `movement_coach.muscles.normalize_all` first. Terms outside that vocabulary
    are treated as uncoverable rather than matched approximately.

    Raises `PrescriptionError` when ``weak_muscles`` is empty or ``max_items``
    is not positive -- both indicate a caller bug, not a data limitation.
    """
    if max_items <= 0:
        raise PrescriptionError("max_items must be positive")

    weak = {m.strip().lower() for m in weak_muscles if m and m.strip()}
    if not weak:
        raise PrescriptionError("no weak muscles supplied")

    pool = filter_by_equipment(database, equipment)
    relaxed = False
    if not pool and relax_equipment_if_empty:
        pool = list(database)
        relaxed = True
    if not pool:
        raise PrescriptionError(
            "no exercises match the available equipment and relaxation is disabled"
        )

    # Only exercises that touch at least one weak muscle can contribute.
    pool = [item for item in pool if covers(item, weak)]

    selected: List[PrescribedExercise] = []
    remaining: Set[str] = set(weak)
    used: Set[str] = set()

    while remaining and len(selected) < max_items:
        chosen: Exercise | None = None
        for muscle in _scarcity_order(remaining, pool):
            chosen = _best_for(muscle, remaining, weak, [i for i in pool if i.id not in used])
            if chosen is not None:
                break
        if chosen is None:
            break

        newly = covers(chosen, remaining)
        selected.append(
            PrescribedExercise(exercise=chosen, covers=newly, score=score(chosen, weak))
        )
        used.add(chosen.id)
        remaining -= newly

    return Prescription(
        items=tuple(selected),
        requested=frozenset(weak),
        uncovered=frozenset(remaining),
        equipment_relaxed=relaxed,
    )


def verify_grounded(prescription: Prescription, database: ExerciseDatabase) -> None:
    """Assert every prescribed exercise really exists in the database.

    This enforces the project's central rule -- the model may reason freely
    but may not invent exercises -- and is cheap enough to run on every call.
    """
    unknown = [item.id for item in prescription.items if database.get(item.id) is None]
    if unknown:
        raise PrescriptionError(
            f"prescription contains exercise ids absent from the database: {', '.join(unknown)}"
        )
