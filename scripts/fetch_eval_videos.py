"""Pull a sample of clips out of a remote zip without downloading all of it.

The HuggingFace ``34data/workout-vids`` archive is 4.9 GB, but a per-class
sample for an evaluation run needs a small fraction of it. A zip's central
directory sits at the end of the file, so with HTTP range requests the archive
can be read like a local file and individual members fetched on demand.

Usage:

    python scripts/fetch_eval_videos.py --per-class 5 --out data/eval/videos
"""

from __future__ import annotations

import argparse
import io
import random
import struct
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

DEFAULT_URL = (
    "https://huggingface.co/datasets/34data/workout-vids/resolve/main/videos.zip"
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

#: Largest plausible zip local file header (30 fixed bytes + name + extra).
_LOCAL_HEADER_MAX = 30 + 65535 + 65535


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file backed by HTTP range requests.

    Only implements what `zipfile` needs to read the central directory. Member
    payloads are fetched by `fetch_member` in one request each: letting
    `zipfile` stream them instead issues a request per buffer refill, which the
    remote host throttles to a standstill.
    """

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout
        self._pos = 0
        self._size = self._head_size()

    def _head_size(self) -> int:
        request = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            length = response.headers.get("Content-Length")
        if not length:
            raise OSError(f"{self.url} did not report a Content-Length")
        return int(length)

    # -- io.RawIOBase ------------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._pos
        if size <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + size, self._size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            chunk = response.read()
        self._pos += len(chunk)
        return chunk

    def readinto(self, buffer) -> int:  # type: ignore[override]
        chunk = self.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    @property
    def size(self) -> int:
        return self._size

    def fetch_member(self, info: zipfile.ZipInfo) -> bytes:
        """Download and decompress one archive member in a single request.

        The central directory records where the member's *local* header starts
        but not how long that header is, so the request covers the header's
        maximum size as well as the payload and the excess is discarded.
        """
        start = info.header_offset
        end = min(start + _LOCAL_HEADER_MAX + info.compress_size, self._size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            blob = response.read()

        if blob[:4] != b"PK\x03\x04":
            raise OSError(f"{info.filename}: local header not found at {start}")
        name_len, extra_len = struct.unpack("<HH", blob[26:30])
        data_start = 30 + name_len + extra_len
        payload = blob[data_start : data_start + info.compress_size]

        if info.compress_type == zipfile.ZIP_STORED:
            return payload
        if info.compress_type == zipfile.ZIP_DEFLATED:
            return zlib.decompress(payload, -zlib.MAX_WBITS)
        raise OSError(
            f"{info.filename}: unsupported compression type {info.compress_type}"
        )


def sample_members(
    archive: zipfile.ZipFile, per_class: int, seed: int
) -> List[zipfile.ZipInfo]:
    """Pick up to ``per_class`` clips from each label directory."""
    by_label: Dict[str, List[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive.infolist():
        path = Path(info.filename)
        if info.is_dir() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if len(path.parts) < 2:
            continue
        by_label[path.parts[-2]].append(info)

    rng = random.Random(seed)
    picked: List[zipfile.ZipInfo] = []
    for label in sorted(by_label):
        members = sorted(by_label[label], key=lambda i: i.filename)
        if len(members) > per_class:
            members = rng.sample(members, per_class)
        picked.extend(sorted(members, key=lambda i: i.filename))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--out", default="data/eval/videos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    remote = HttpRangeFile(args.url)
    print(f"遠端檔案 {remote.size / 1e9:.2f} GB，讀取中央目錄…")

    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)) as archive:
        members = sample_members(archive, args.per_class, args.seed)
        labels = {Path(m.filename).parts[-2] for m in members}
        total = sum(m.file_size for m in members)
        print(
            f"抽樣 {len(members)} 支 / {len(labels)} 類，"
            f"共 {total / 1e6:.0f} MB（原檔 {remote.size / 1e9:.2f} GB）"
        )

    out = Path(args.out)
    fetched = skipped = failed = 0
    for index, info in enumerate(members, 1):
        path = Path(info.filename)
        target = out / path.parts[-2] / path.name
        if target.is_file() and target.stat().st_size == info.file_size:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = remote.fetch_member(info)
        except (OSError, zlib.error) as exc:
            # One unreadable member must not abandon the rest of the sample.
            print(f"  ! {path.name}: {exc}", flush=True)
            failed += 1
            continue
        target.write_bytes(payload)
        fetched += 1
        print(f"  [{index}/{len(members)}] {path.name} "
              f"({len(payload) / 1e6:.1f} MB)", flush=True)

    print(f"完成：新抓 {fetched}，已存在 {skipped}，失敗 {failed}，輸出於 {args.out}")
    return 1 if failed and not fetched else 0


if __name__ == "__main__":
    raise SystemExit(main())
