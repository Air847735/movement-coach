"""Frame sampling, including the failure modes the API must report early."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from movement_coach.errors import VideoError
from movement_coach.video import sample_frames


@pytest.fixture
def clip(tmp_path):
    """A tiny synthetic mp4 -- no camera or fixture file needed."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available in this OpenCV build")
    for i in range(30):
        frame = np.full((120, 160, 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_samples_requested_number_of_frames(clip):
    frames = sample_frames(clip, count=6)
    assert len(frames) == 6
    assert all(base64.b64decode(f)[:2] == b"\xff\xd8" for f in frames)  # JPEG magic


def test_downscales_to_max_side(clip):
    frame = base64.b64decode(sample_frames(clip, count=1, max_side=64)[0])
    decoded = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 64


def test_missing_file(tmp_path):
    with pytest.raises(VideoError, match="video not found"):
        sample_frames(tmp_path / "absent.mp4")


def test_empty_file(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    with pytest.raises(VideoError, match="empty"):
        sample_frames(path)


def test_oversized_file_rejected_before_decoding(tmp_path):
    path = tmp_path / "big.mp4"
    path.write_bytes(b"\x00" * 2048)
    with pytest.raises(VideoError, match="above the"):
        sample_frames(path, max_bytes=1024)


def test_undecodable_file(tmp_path):
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"definitely not a container" * 10)
    with pytest.raises(VideoError, match="could not decode|no readable frames"):
        sample_frames(path)


def test_non_positive_count_rejected(clip):
    with pytest.raises(VideoError, match="count must be positive"):
        sample_frames(clip, count=0)
