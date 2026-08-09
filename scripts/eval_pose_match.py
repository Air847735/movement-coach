"""Classify labelled videos against the GIF-derived skeleton library.

Scored with the same lenient name grading as `eval_recognition.py`, so the
numbers sit directly beside the vision model's.

Only the ROM fingerprint is used for ranking. Trajectory comparison is left
out on purpose: a reference GIF holds one repetition while a query clip holds
several, so resampling both to a fixed length compares one rep against eight.
Making that work needs per-repetition segmentation, which is only worth
building if the fingerprint alone shows signal.

Usage:

    python scripts/eval_pose_match.py data/eval/videos --per-class 5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_recognition import collect, grade  # noqa: E402
from movement_coach.pose import (  # noqa: E402
    angle_series,
    build_pose_model,
    mirror,
    rom_signature,
    summarise,
)

TOP_K = 5


def load_library(path: Path):
    data = np.load(path, allow_pickle=False)
    rom = data["rom"]
    signatures = np.stack([rom_signature(r) for r in rom])
    mirrored = np.stack([rom_signature(mirror(r)) for r in rom])
    return data["ids"], data["names"], signatures, mirrored


def rank(query: np.ndarray, signatures: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
    """Rank library entries by cosine similarity, allowing for a mirrored view.

    Taking the better of the two orientations means a clip filmed from the
    opposite side is not penalised for it.
    """
    similarity = np.maximum(signatures @ query, mirrored @ query)
    return np.argsort(-similarity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos")
    parser.add_argument("--library", default="data/pose_library.npz")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/eval/pose_report.md")
    args = parser.parse_args()

    library = Path(args.library)
    if not library.is_file():
        print(f"找不到 {library}，請先執行 scripts/build_pose_library.py", file=sys.stderr)
        return 2

    ids, names, signatures, mirrored = load_library(library)
    pairs = collect(Path(args.videos), args.per_class or None, args.seed)
    print(f"參考庫 {len(ids)} 筆 · 評測 {len(pairs)} 支影片\n")

    model = build_pose_model()
    rows: List[List[str]] = []
    top1: Counter[str] = Counter()
    top5: Counter[str] = Counter()
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.time()

    for index, (label, path) in enumerate(pairs, 1):
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()

        summary = summarise(angle_series(model, frames)) if frames else None
        if summary is None:
            top1["error"] += 1
            top5["error"] += 1
            rows.append([path.name, label, "（骨架不足）", "!", "!"])
            print(f"  [{index}/{len(pairs)}] 骨架不足 {path.name}")
            continue

        order = rank(rom_signature(summary[0]), signatures, mirrored)[:TOP_K]
        predictions = [str(names[i]) for i in order]

        first = grade(label, predictions[0])
        best = min(
            (grade(label, p) for p in predictions),
            key=lambda g: {"hit": 0, "partial": 1, "miss": 2}[g],
        )
        top1[first] += 1
        top5[best] += 1
        per_label[label][best] += 1

        mark = {"hit": "✓", "partial": "△", "miss": "✗"}
        rows.append(
            [path.name, label, predictions[0], mark[first], mark[best],
             " / ".join(predictions[1:])]
        )
        print(f"  [{index}/{len(pairs)}] {best:8s} {label:22s} → {predictions[0][:46]}")

    total = len(pairs)
    elapsed = time.time() - started

    def rate(counter: Counter, keys: Sequence[str]) -> float:
        return sum(counter[k] for k in keys) / total

    report = [
        "# 骨架比對辨識結果",
        "",
        f"參考庫 {len(ids)} 筆（來自資料集 GIF）· 評測 {total} 支 · "
        f"平均 {elapsed / total:.1f} 秒/支",
        "",
        "以 ROM 指紋（各關節活動幅度）做餘弦相似度排序，允許左右鏡像。",
        "評分規則與 `eval_recognition.py` 相同，可與 VLM 的數字直接並列。",
        "",
        f"- top-1 完全命中：**{top1['hit']}/{total} = {rate(top1, ['hit']):.1%}**",
        f"- top-1 含部分命中：{rate(top1, ['hit', 'partial']):.1%}",
        f"- top-5 完全命中：**{top5['hit']}/{total} = {rate(top5, ['hit']):.1%}**",
        f"- top-5 含部分命中：**{rate(top5, ['hit', 'partial']):.1%}**",
        f"- 骨架不足：{top1['error']}",
        "",
        "## 逐類別（top-5）",
        "",
        "| 動作 | 支數 | 完全命中 | 部分命中 | 未命中 |",
        "|---|---|---|---|---|",
    ]
    for label, counts in sorted(per_label.items()):
        report.append(
            f"| {label} | {sum(counts.values())} | {counts['hit']} | "
            f"{counts['partial']} | {counts['miss']} |"
        )

    report += [
        "",
        "## 逐支影片",
        "",
        "| 影片 | 正確答案 | top-1 預測 | top-1 | top-5 | 其餘候選 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [str(c).replace("|", "/") for c in row]
        if len(cells) == 5:
            cells.append("")
        report.append("| " + " | ".join(cells) + " |")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(
        f"\ntop-1 {rate(top1, ['hit']):.1%} · top-5 {rate(top5, ['hit']):.1%} "
        f"（含部分命中 {rate(top5, ['hit', 'partial']):.1%}）· {elapsed / 60:.1f} 分鐘"
    )
    print(f"報告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
