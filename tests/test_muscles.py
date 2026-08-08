"""Vocabulary normalisation, including coverage over the real dataset."""

from __future__ import annotations

from collections import Counter

import pytest

from movement_coach.muscles import (
    ALIASES,
    TARGET_MUSCLES,
    normalize,
    normalize_all,
    vocabulary,
)


def test_target_vocabulary_has_19_terms():
    assert len(TARGET_MUSCLES) == 19
    assert vocabulary() == sorted(TARGET_MUSCLES)


@pytest.mark.parametrize(
    "term, expected",
    [
        ("glutes", "glutes"),
        ("GLUTES", "glutes"),
        ("  quadriceps  ", "quads"),
        ("Upper   Chest", "pectorals"),
        ("core", "abs"),
        ("gluteus medius", "abductors"),
        ("erector spinae", "spine"),
    ],
)
def test_known_terms_normalize(term, expected):
    assert normalize(term) == expected


@pytest.mark.parametrize("term", ["hip flexors", "rotator cuff", "shins", "", "   "])
def test_unmappable_terms_return_none(term):
    """Unsearchable terms must not be coerced onto a nearby muscle."""
    assert normalize(term) is None


def test_every_alias_points_at_a_real_target():
    unknown = {value for value in ALIASES.values() if value not in TARGET_MUSCLES}
    assert not unknown, f"aliases map onto non-target values: {sorted(unknown)}"


def test_no_alias_shadows_a_target_term():
    """A term that is already a target must not be remapped elsewhere."""
    assert not (set(ALIASES) & TARGET_MUSCLES)


def test_normalize_all_splits_mapped_and_unmapped():
    mapped, unmapped = normalize_all(
        ["core", "quadriceps", "hip flexors", "ankle mobility"]
    )
    assert mapped == {"abs", "quads"}
    assert unmapped == ["hip flexors", "ankle mobility"]


def test_normalize_all_preserves_order_and_deduplicates_unmapped():
    _, unmapped = normalize_all(["hip flexors", "Hip Flexors", "rotator cuff"])
    assert unmapped == ["hip flexors", "rotator cuff"]


def test_normalize_all_on_empty_input():
    assert normalize_all([]) == (set(), [])


@pytest.mark.real_db
def test_secondary_muscle_coverage_over_real_dataset(real_database):
    """Guards the documented 95.6% coverage figure against alias-table edits."""
    counts = Counter(
        term for item in real_database for term in item.secondary_raw
    )
    total = sum(counts.values())
    covered = sum(n for term, n in counts.items() if normalize(term) is not None)
    ratio = covered / total

    assert ratio >= 0.95, f"secondary_muscles coverage dropped to {ratio:.1%}"
    assert len(counts) == 40, "dataset secondary vocabulary changed size"


@pytest.mark.real_db
def test_hip_flexors_remains_unsearchable(real_database):
    """Documented limitation: it appears often but is never a target."""
    assert normalize("hip flexors") is None
    assert "hip flexors" not in real_database.targets()
