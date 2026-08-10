"""End-to-end orchestration, split at the point the user confirms the movement.

The flow is deliberately two calls rather than one: `describe_movement` returns
what the model thinks it saw, the user corrects it if needed, and `diagnose`
takes that final wording. Nothing here contains business rules of its own --
it wires together video sampling, the VLM stages, the constrained mapping, and
retrieval.

This module imports no web framework. It is the whole product as a library;
`api.py` only exposes it over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset import DEFAULT_LANGUAGE, ExerciseDatabase, load_exercises
from .errors import PrescriptionError
from .muscles import normalize_all
from .prescribe import DEFAULT_MAX_ITEMS, Prescription, prescribe, verify_grounded
from .video import DEFAULT_FRAME_COUNT, sample_frames
from .vlm import Assessment, OllamaVLM


def _measure_video(video_path: str | Path, model) -> str | None:
    """Skeleton measurements for a clip, or ``None`` if unavailable.

    Failures here are deliberately non-fatal: measurements make the assessment
    better informed, but the pipeline predates them and still works without.
    """
    import cv2

    from .metrics import measure
    from .pose import angle_series

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        return None
    metrics = measure(angle_series(model, frames), fps)
    return metrics.describe() if metrics.reliable else None


@dataclass(frozen=True)
class Diagnosis:
    """Everything the pipeline produced for one video.

    ``prescription`` is ``None`` only when retrieval could not run at all --
    no problems were found, or nothing the model said mapped onto a searchable
    muscle. ``unmapped_causes`` always carries the terms that were dropped, so
    a thin prescription is explainable rather than mysterious.
    """

    description: str
    problems: tuple[str, ...]
    causes: tuple[str, ...]
    weak_muscles: frozenset[str]
    unmapped_causes: tuple[str, ...]
    prescription: Prescription | None
    prescription_error: str | None = None
    measurements: str | None = None

    @property
    def has_issues(self) -> bool:
        return bool(self.problems)


class MovementCoach:
    """The library entry point.

    Holds the exercise database and a VLM client. Construct once and reuse;
    `check_ready` validates both dependencies up front so a broken setup is
    reported before a video is processed.
    """

    def __init__(
        self,
        database: ExerciseDatabase,
        vlm: OllamaVLM | None = None,
        *,
        frame_count: int = DEFAULT_FRAME_COUNT,
        measure_pose: bool = True,
    ) -> None:
        self.database = database
        self.vlm = vlm or OllamaVLM()
        self.frame_count = frame_count
        self.measure_pose = measure_pose
        self._pose_model = None

    def _pose(self):
        """Load the pose model on first use, or disable measurement if absent.

        The ``pose`` extra is optional, so a missing dependency downgrades the
        pipeline to frames-only rather than failing.
        """
        if not self.measure_pose:
            return None
        if self._pose_model is None:
            try:
                from .pose import build_pose_model

                self._pose_model = build_pose_model()
            except ImportError:
                self.measure_pose = False
                return None
        return self._pose_model

    @classmethod
    def from_path(
        cls,
        dataset_path: str | Path = "data/exercises.json",
        vlm: OllamaVLM | None = None,
        **kwargs: object,
    ) -> "MovementCoach":
        """Load the database from disk and build a coach.

        Raises `DatasetError` immediately if the file is missing or malformed,
        rather than surfacing it during retrieval.
        """
        return cls(load_exercises(dataset_path), vlm, **kwargs)  # type: ignore[arg-type]

    def check_ready(self) -> None:
        """Verify the model server is up and the model is installed."""
        self.vlm.health()

    def describe_movement(self, video_path: str | Path) -> str:
        """Stage 1 only, so the caller can let a user correct the wording."""
        frames = sample_frames(video_path, count=self.frame_count)
        return self.vlm.describe(frames)

    def diagnose(
        self,
        video_path: str | Path,
        *,
        description: str | None = None,
        equipment: Iterable[str] | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> Diagnosis:
        """Run the full pipeline for one video.

        ``description`` is the user-confirmed movement wording; when omitted
        the model's own description is used. ``equipment`` restricts the
        prescription, and is relaxed with a flag on the result if it would
        otherwise leave nothing available.
        """
        frames = sample_frames(video_path, count=self.frame_count)

        model = self._pose()
        measurements = _measure_video(video_path, model) if model else None

        assessment = self.vlm.analyse(frames, description, measurements)
        return self.prescribe_for(assessment, equipment=equipment, max_items=max_items)

    def prescribe_for(
        self,
        assessment: Assessment,
        *,
        equipment: Iterable[str] | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> Diagnosis:
        """Turn free-form reasoning into a grounded prescription.

        Separated from `diagnose` so the mapping and retrieval half can be
        tested, or re-run with different equipment, without touching the GPU.
        """
        mapping = self.vlm.map_to_muscles(assessment.causes)
        weak, unrecognised = normalize_all(mapping.proposed)

        # Two ways a cause can fail to reach retrieval: the model declined it,
        # or it answered with something outside the searchable vocabulary.
        # Both are reported, in that order, rather than silently dropped.
        unmapped = list(mapping.declined) + unrecognised

        prescription: Prescription | None = None
        error: str | None = None
        if weak:
            try:
                prescription = prescribe(
                    self.database, weak, equipment=equipment, max_items=max_items
                )
                verify_grounded(prescription, self.database)
            except PrescriptionError as exc:
                error = str(exc)
        elif assessment.causes:
            error = "no cause mapped onto a searchable muscle"

        return Diagnosis(
            description=assessment.description,
            problems=assessment.problems,
            causes=assessment.causes,
            weak_muscles=frozenset(weak),
            unmapped_causes=tuple(unmapped),
            prescription=prescription,
            prescription_error=error,
            measurements=assessment.measurements,
        )


def format_report(diagnosis: Diagnosis, language: str = DEFAULT_LANGUAGE) -> str:
    """Render a diagnosis as plain text, for CLI use and quick inspection."""
    lines: list[str] = [f"動作：{diagnosis.description}", ""]

    if not diagnosis.has_issues:
        lines.append("未發現明顯問題。")
        return "\n".join(lines)

    if diagnosis.measurements:
        lines.append("骨架量測：")
        lines.extend(f"  {line}" for line in diagnosis.measurements.splitlines())
        lines.append("")

    lines.append("問題：")
    lines.extend(f"  - {problem}" for problem in diagnosis.problems)

    if diagnosis.causes:
        lines += ["", "可能相關的環節（一般性關聯，非診斷）："]
        lines.extend(f"  - {cause}" for cause in diagnosis.causes)

    if diagnosis.weak_muscles:
        lines += ["", f"對應到可檢索的肌群：{', '.join(sorted(diagnosis.weak_muscles))}"]
    if diagnosis.unmapped_causes:
        lines.append(f"無對應動作的項目：{', '.join(diagnosis.unmapped_causes)}")

    prescription = diagnosis.prescription
    if prescription is None:
        lines += ["", f"無法產出處方：{diagnosis.prescription_error or '原因不明'}"]
        return "\n".join(lines)

    lines += ["", "訓練處方："]
    for index, item in enumerate(prescription.items, 1):
        lines.append(
            f"  {index}. {item.name}（{item.exercise.equipment}）"
            f" — 覆蓋 {', '.join(sorted(item.covers))}"
        )
        for step_no, step in enumerate(item.steps(language), 1):
            lines.append(f"       {step_no}. {step}")

    if prescription.uncovered:
        lines.append(f"  未覆蓋：{', '.join(sorted(prescription.uncovered))}")
    if prescription.equipment_relaxed:
        lines.append("  （可用器材下無合適動作，已放寬器材限制）")

    return "\n".join(lines)
