"""Exception hierarchy for movement-coach.

Every failure surfaces as a subclass of `MovementCoachError` so callers can
distinguish "the input was unusable" from "an external service was down"
without parsing messages. Nothing here is silently swallowed: `spec.md`
requires that a partially-completed pipeline never masquerade as a complete
result.
"""

from __future__ import annotations

from typing import Optional


class MovementCoachError(Exception):
    """Base class for every error raised by this package."""


class DatasetError(MovementCoachError):
    """`exercises.json` is missing, unparseable, or does not match the schema.

    Raised at load time rather than at retrieval time so a misconfigured
    deployment fails on startup instead of halfway through an analysis.
    """


class VideoError(MovementCoachError):
    """The input video could not be read, decoded, or sampled."""


class VLMError(MovementCoachError):
    """The vision-language model was unreachable or returned an unusable reply.

    `stage` identifies which pipeline step failed ("describe", "assess",
    "causes", "map") so the caller can report where the run stopped.
    """

    def __init__(self, message: str, *, stage: Optional[str] = None) -> None:
        super().__init__(message)
        self.stage = stage

    def __str__(self) -> str:
        base = super().__str__()
        return f"[{self.stage}] {base}" if self.stage else base


class PrescriptionError(MovementCoachError):
    """No prescription could be produced from the given weak-muscle set."""
