"""movement-coach: diagnose a movement from video, prescribe grounded training.

Importing this package pulls in no web framework. The whole product is usable
as a library:

    from movement_coach import MovementCoach, format_report

    coach = MovementCoach.from_path("data/exercises.json")
    coach.check_ready()
    print(format_report(coach.diagnose("squat.mp4")))

`api.py` is a separate, optional HTTP layer over exactly these calls.
"""

from __future__ import annotations

from .dataset import Exercise, ExerciseDatabase, load_exercises
from .errors import (
    DatasetError,
    MovementCoachError,
    PrescriptionError,
    VideoError,
    VLMError,
)
from .muscles import TARGET_MUSCLES, normalize, normalize_all, vocabulary
from .pipeline import Diagnosis, MovementCoach, format_report
from .prescribe import (
    PrescribedExercise,
    Prescription,
    prescribe,
    verify_grounded,
)
from .video import sample_frames
from .vlm import Assessment, MuscleMapping, OllamaVLM

__version__ = "0.1.0"

__all__ = [
    "Assessment",
    "DatasetError",
    "Diagnosis",
    "Exercise",
    "ExerciseDatabase",
    "MovementCoach",
    "MovementCoachError",
    "MuscleMapping",
    "OllamaVLM",
    "PrescribedExercise",
    "Prescription",
    "PrescriptionError",
    "TARGET_MUSCLES",
    "VLMError",
    "VideoError",
    "format_report",
    "load_exercises",
    "normalize",
    "normalize_all",
    "prescribe",
    "sample_frames",
    "verify_grounded",
    "vocabulary",
]
