"""HTTP layer over `movement_coach`.

Deliberately thin: this module accepts requests, calls the library, and
serialises the result. It holds no rules about muscles, scoring, or
prescriptions -- `AGENTS.md` forbids business logic here, so that the package
stays usable without ever starting a server.

Uploads live in a temporary directory keyed by an opaque token, because the
flow pauses between "describe" and "diagnose" while the user confirms the
movement. Old uploads are purged on each new upload; nothing persists beyond
the process's temp directory, and no upload is ever written into the
repository.

Run with:

    uvicorn movement_coach.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .dataset import DEFAULT_LANGUAGE
from .errors import DatasetError, MovementCoachError, VideoError, VLMError
from .pipeline import Diagnosis, MovementCoach
from .prescribe import DEFAULT_MAX_ITEMS, Prescription
from .video import DEFAULT_MAX_BYTES

#: Where the dataset is expected. Override with MOVEMENT_COACH_DATASET.
DATASET_PATH = os.environ.get("MOVEMENT_COACH_DATASET", "data/exercises.json")

#: Uploads older than this are removed. They are working files, not records.
UPLOAD_TTL_SECONDS = 3600

_UPLOAD_DIR = Path(tempfile.gettempdir()) / "movement-coach-uploads"
_WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="movement-coach", version="0.1.0")

# Built at import time so a missing or malformed dataset stops the server
# immediately rather than failing on the first request.
_coach = MovementCoach.from_path(DATASET_PATH)


# -- upload handling -------------------------------------------------------


def _purge_old_uploads() -> None:
    if not _UPLOAD_DIR.is_dir():
        return
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    for path in _UPLOAD_DIR.iterdir():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            # A file vanishing underneath us is not an error worth failing on.
            continue


def _store_upload(upload: UploadFile) -> str:
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _purge_old_uploads()

    suffix = Path(upload.filename or "").suffix or ".mp4"
    token = f"{uuid.uuid4().hex}{suffix}"
    target = _UPLOAD_DIR / token

    size = 0
    try:
        with target.open("wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > DEFAULT_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"影片超過 {DEFAULT_MAX_BYTES // 1_048_576} MB 上限",
                    )
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"無法儲存上傳檔案：{exc}") from exc

    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上傳的檔案是空的")
    return token


def _resolve(token: str) -> Path:
    # Reject anything that is not a bare filename we minted ourselves.
    if "/" in token or "\\" in token or token != Path(token).name:
        raise HTTPException(status_code=400, detail="無效的 token")
    path = _UPLOAD_DIR / token
    if not path.is_file():
        raise HTTPException(status_code=404, detail="找不到影片，可能已逾時，請重新上傳")
    return path


# -- serialisation ---------------------------------------------------------


def _prescription_json(prescription: Prescription, language: str) -> Dict[str, Any]:
    return {
        "items": [
            {
                "id": item.id,
                "name": item.name,
                "equipment": item.exercise.equipment,
                "target": item.exercise.target,
                "covers": sorted(item.covers),
                "steps": list(item.steps(language)),
            }
            for item in prescription.items
        ],
        "requested": sorted(prescription.requested),
        "covered": sorted(prescription.covered),
        "uncovered": sorted(prescription.uncovered),
        "equipment_relaxed": prescription.equipment_relaxed,
    }


def _diagnosis_json(diagnosis: Diagnosis, language: str) -> Dict[str, Any]:
    return {
        "description": diagnosis.description,
        "has_issues": diagnosis.has_issues,
        "problems": list(diagnosis.problems),
        "causes": list(diagnosis.causes),
        "weak_muscles": sorted(diagnosis.weak_muscles),
        "unmapped_causes": list(diagnosis.unmapped_causes),
        "prescription": (
            _prescription_json(diagnosis.prescription, language)
            if diagnosis.prescription is not None
            else None
        ),
        "prescription_error": diagnosis.prescription_error,
    }


# -- error handling --------------------------------------------------------


@app.exception_handler(MovementCoachError)
async def _domain_error_handler(_request: Any, exc: MovementCoachError) -> JSONResponse:
    """Map library failures onto status codes without losing which stage failed."""
    if isinstance(exc, VideoError):
        return JSONResponse(status_code=400, content={"detail": str(exc), "kind": "video"})
    if isinstance(exc, VLMError):
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "kind": "vlm", "stage": exc.stage},
        )
    if isinstance(exc, DatasetError):
        return JSONResponse(status_code=500, content={"detail": str(exc), "kind": "dataset"})
    return JSONResponse(status_code=500, content={"detail": str(exc), "kind": "error"})


# -- endpoints -------------------------------------------------------------


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Report whether both dependencies are usable.

    Returns 200 with ``vlm_ok: false`` rather than failing, so the page can
    show a specific reason instead of a blank error.
    """
    result: Dict[str, Any] = {
        "exercises": len(_coach.database),
        "dataset_path": DATASET_PATH,
        "model": _coach.vlm.model,
        "vlm_ok": True,
        "vlm_error": None,
    }
    try:
        _coach.check_ready()
    except VLMError as exc:
        result["vlm_ok"] = False
        result["vlm_error"] = str(exc)
    return result


@app.get("/api/equipment")
def equipment() -> List[str]:
    """Equipment values present in the database, for the filter UI."""
    return sorted(_coach.database.equipment_types())


@app.post("/api/describe")
async def describe(video: UploadFile) -> Dict[str, str]:
    """Stage 1: return what the model thinks the movement is, plus a token.

    The token is how the follow-up request refers to the same upload after the
    user has confirmed or corrected the description.
    """
    token = _store_upload(video)
    return {"token": token, "description": _coach.describe_movement(_resolve(token))}


@app.post("/api/diagnose")
async def diagnose(
    token: str = Form(...),
    description: str = Form(""),
    equipment: str = Form(""),
    language: str = Form(DEFAULT_LANGUAGE),
    max_items: int = Form(DEFAULT_MAX_ITEMS),
) -> Dict[str, Any]:
    """Stages 2-4: assess, infer, map, and prescribe for an existing upload."""
    path = _resolve(token)
    selected = [item.strip() for item in equipment.split(",") if item.strip()]
    result = _coach.diagnose(
        path,
        description=description or None,
        equipment=selected or None,
        max_items=max_items,
    )
    return _diagnosis_json(result, language)


@app.delete("/api/upload/{token}")
def delete_upload(token: str) -> Dict[str, bool]:
    """Let the page discard its upload as soon as it is done with it."""
    _resolve(token).unlink(missing_ok=True)
    return {"deleted": True}


if _WEB_DIR.is_dir():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")
