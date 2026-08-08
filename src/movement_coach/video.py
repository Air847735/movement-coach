"""Sampling still frames out of an input video.

The VLM sees a handful of stills rather than the whole clip: an 8 GB card
running a 7B vision model cannot afford many images, and a movement's key
positions are visible in a few evenly-spaced frames. Frames are downscaled
before encoding for the same reason.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List

import cv2

from .errors import VideoError

#: How many stills to sample. Chosen for an 8 GB GPU running a 7B VLM.
DEFAULT_FRAME_COUNT = 6

#: Longest edge, in pixels, of each sampled frame.
DEFAULT_MAX_SIDE = 512

#: Refuse anything larger up front rather than after decoding.
DEFAULT_MAX_BYTES = 200 * 1024 * 1024


def _resize(frame, max_side: int):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(
        frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )


def sample_frames(
    video_path: str | Path,
    *,
    count: int = DEFAULT_FRAME_COUNT,
    max_side: int = DEFAULT_MAX_SIDE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> List[str]:
    """Return ``count`` evenly-spaced frames as base64-encoded JPEGs.

    Validation happens before any decoding work so an unusable file fails
    fast: `spec.md` requires format and size problems to be reported before
    the pipeline spends GPU time.

    Raises `VideoError` for a missing file, an oversized file, an
    undecodable container, or a video with no readable frames.
    """
    if count <= 0:
        raise VideoError("frame count must be positive")

    path = Path(video_path)
    if not path.is_file():
        raise VideoError(f"video not found: {path}")

    size = path.stat().st_size
    if size == 0:
        raise VideoError(f"video is empty: {path}")
    if size > max_bytes:
        raise VideoError(
            f"video is {size / 1_048_576:.1f} MB, above the "
            f"{max_bytes / 1_048_576:.0f} MB limit"
        )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"could not decode {path}; the format may be unsupported")

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Some containers report 0 or a bogus count; fall back to reading through.
        indices = (
            [round(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
            if total > 0
            else []
        )

        encoded: List[str] = []
        if indices:
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    continue
                encoded.append(_encode(frame, max_side))
        else:
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if frames:
                step = max(len(frames) // count, 1)
                for frame in frames[::step][:count]:
                    encoded.append(_encode(frame, max_side))
    finally:
        capture.release()

    if not encoded:
        raise VideoError(f"no readable frames in {path}")
    return encoded


def _encode(frame, max_side: int) -> str:
    ok, buffer = cv2.imencode(".jpg", _resize(frame, max_side))
    if not ok:
        raise VideoError("failed to JPEG-encode a sampled frame")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
