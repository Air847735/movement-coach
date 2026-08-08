"""Muscle vocabulary normalisation.

The dataset speaks two vocabularies that do not line up: `target` uses 19
closed terms, while `secondary_muscles` uses 40 terms of which only 9 appear
in `target`. On top of that the VLM invents its own wording. Everything
downstream of the diagnosis has to be expressed in the 19 `target` terms,
because those are the only ones that can be searched.

`ALIASES` therefore serves two callers: the dataset loader (normalising
`secondary_muscles`) and the constrained-mapping step (normalising free-text
VLM output). Terms with no `target` equivalent are reported as unmapped
rather than coerced to a near neighbour -- see `docs/architecture.md`, "成因
推論採自由推理 + 約束映射".
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

#: The 19 values the dataset's ``target`` field can take. This is the only
#: vocabulary the prescription retrieval can search.
TARGET_MUSCLES: frozenset[str] = frozenset(
    {
        "abductors",
        "abs",
        "adductors",
        "biceps",
        "calves",
        "cardiovascular system",
        "delts",
        "forearms",
        "glutes",
        "hamstrings",
        "lats",
        "levator scapulae",
        "pectorals",
        "quads",
        "serratus anterior",
        "spine",
        "traps",
        "triceps",
        "upper back",
    }
)

#: Synonym -> ``target`` term. Covers the dataset's own ``secondary_muscles``
#: wording plus anatomical names a model is likely to produce. Coverage over
#: the dataset is measured by ``tests/test_muscles.py``.
ALIASES: dict[str, str] = {
    # --- dataset secondary_muscles wording ---
    "quadriceps": "quads",
    "shoulders": "delts",
    "deltoids": "delts",
    "rear deltoids": "delts",
    "chest": "pectorals",
    "upper chest": "pectorals",
    "core": "abs",
    "abdominals": "abs",
    "lower abs": "abs",
    "obliques": "abs",
    "latissimus dorsi": "lats",
    "trapezius": "traps",
    "rhomboids": "upper back",
    "back": "upper back",
    "lower back": "spine",
    "inner thighs": "adductors",
    "groin": "adductors",
    "soleus": "calves",
    "brachialis": "biceps",
    "grip muscles": "forearms",
    "wrist flexors": "forearms",
    "wrist extensors": "forearms",
    # --- anatomical names a model may produce ---
    "erector spinae": "spine",
    "spinal erectors": "spine",
    "gluteus maximus": "glutes",
    "gluteus medius": "abductors",
    "gluteus minimus": "abductors",
    "hip abductors": "abductors",
    "hip adductors": "adductors",
    "gastrocnemius": "calves",
    "rectus abdominis": "abs",
    "transverse abdominis": "abs",
    "pectoralis major": "pectorals",
    "pecs": "pectorals",
    "anterior deltoid": "delts",
    "posterior deltoid": "delts",
    "lateral deltoid": "delts",
    "rectus femoris": "quads",
    "vastus medialis": "quads",
    "biceps femoris": "hamstrings",
    "upper trapezius": "traps",
    "middle trapezius": "upper back",
    "serratus": "serratus anterior",
    "cardio": "cardiovascular system",
    "cardiovascular": "cardiovascular system",
}


def normalize(term: str) -> str | None:
    """Map one muscle term to a ``target`` value, or ``None`` if it has none.

    Matching is case-insensitive and whitespace-tolerant. ``None`` means the
    term is genuinely unsearchable (for example ``hip flexors``, which appears
    77 times in the dataset but is never a ``target``); callers must surface it
    as unmapped rather than substitute a similar muscle.
    """
    key = " ".join(term.strip().lower().split())
    if not key:
        return None
    if key in TARGET_MUSCLES:
        return key
    return ALIASES.get(key)


def normalize_all(terms: Iterable[str]) -> Tuple[Set[str], List[str]]:
    """Normalise many terms at once.

    Returns ``(mapped, unmapped)`` where ``mapped`` is a set of ``target``
    values and ``unmapped`` preserves the original wording, in first-seen
    order and without duplicates, so it can be shown to the user verbatim.
    """
    mapped: Set[str] = set()
    unmapped: List[str] = []
    seen: Set[str] = set()
    for term in terms:
        target = normalize(term)
        if target is not None:
            mapped.add(target)
            continue
        cleaned = " ".join(term.strip().split())
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unmapped.append(cleaned)
    return mapped, unmapped


def vocabulary() -> Sequence[str]:
    """The 19 searchable terms, sorted -- used to constrain VLM prompts."""
    return sorted(TARGET_MUSCLES)
