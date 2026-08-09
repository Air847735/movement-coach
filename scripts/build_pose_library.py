"""Build a skeleton reference library from the dataset's demonstration GIFs.

Each GIF is one canonical repetition of one exercise, so running pose on it
yields a reference joint-angle trajectory -- a one-shot exemplar per exercise,
with no training data required.

Going through joint angles rather than pixels is what makes this viable: the
GIFs show a rendered avatar and the query videos show real people, but a knee
bent to 90 degrees measures the same either way, so the rendering gap largely
disappears.

Only the derived angles are stored. The media itself is downloaded to a
gitignored directory and is never redistributed by this project; the dataset's
images and GIFs remain © Gym visual.

Usage:

    python scripts/build_pose_library.py                    # all 1,324
    python scripts/build_pose_library.py --limit 50         # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movement_coach.pose import (  # noqa: E402
    ANGLE_NAMES,
    TRAJECTORY_POINTS,
    angle_series,
    build_pose_model,
    summarise,
)

GIF_BASE = (
    "https://raw.githubusercontent.com/hasaneyldrm/exercises-dataset/"
    "7455efae41b330c265e7cd4b78dfa848e7ce5ebd/"
)


def download(url: str, target: Path, timeout: float = 60.0) -> bool:
    if target.is_file() and target.stat().st_size > 0:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except OSError:
        return False
    if not payload:
        return False
    target.write_bytes(payload)
    return True


def read_gif(path: Path) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/exercises.json")
    parser.add_argument("--gif-dir", default="data/gifs")
    parser.add_argument("--out", default="data/pose_library.npz")
    parser.add_argument("--limit", type=int, default=0, help="只處理前 N 筆")
    parser.add_argument("--workers", type=int, default=12, help="下載並行數")
    parser.add_argument("--upscale", type=int, default=3, help="GIF 放大倍率")
    args = parser.parse_args()

    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    gif_dir = Path(args.gif_dir)
    print(f"下載 {len(records)} 個 GIF …")
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                lambda r: download(GIF_BASE + r["gif_url"], gif_dir / Path(r["gif_url"]).name),
                records,
            )
        )
    print(f"  成功 {sum(results)}/{len(records)}，{time.time() - started:.0f} 秒")

    model = build_pose_model()
    ids: List[str] = []
    names: List[str] = []
    roms: List[np.ndarray] = []
    trajectories: List[np.ndarray] = []
    skipped: List[str] = []

    started = time.time()
    total_frames = 0
    for index, (record, ok) in enumerate(zip(records, results), 1):
        if not ok:
            skipped.append(f"{record['id']} 下載失敗")
            continue
        frames = read_gif(gif_dir / Path(record["gif_url"]).name)
        if len(frames) < 4:
            skipped.append(f"{record['id']} 影格過少 ({len(frames)})")
            continue
        total_frames += len(frames)

        series = angle_series(model, frames, upscale=args.upscale)
        summary = summarise(series)
        if summary is None:
            skipped.append(f"{record['id']} 骨架不足")
            continue

        rom, trajectory = summary
        ids.append(record["id"])
        names.append(record["name"])
        roms.append(rom)
        trajectories.append(trajectory)

        if index % 100 == 0 or index == len(records):
            rate = total_frames / max(time.time() - started, 1e-9)
            print(f"  [{index}/{len(records)}] 已建 {len(ids)} 筆，{rate:.0f} fps", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ids=np.array(ids),
        names=np.array(names),
        rom=np.stack(roms) if roms else np.empty((0, len(ANGLE_NAMES))),
        trajectory=(
            np.stack(trajectories)
            if trajectories
            else np.empty((0, len(ANGLE_NAMES), TRAJECTORY_POINTS))
        ),
        angle_names=np.array(ANGLE_NAMES),
    )
    print(f"\n完成：{len(ids)}/{len(records)} 筆寫入 {out}")
    if skipped:
        print(f"略過 {len(skipped)} 筆，前 10 筆：")
        for line in skipped[:10]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
