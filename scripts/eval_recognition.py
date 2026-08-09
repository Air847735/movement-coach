"""Compare the model's movement description against labelled videos.

Only stage one has an obtainable ground truth: a labelled action-recognition
set says *what* the movement is, and nothing published says which muscle a
given person should strengthen. This script therefore scores stage one
automatically and dumps the later stages verbatim for a human to read.

Expects a directory laid out as ``<root>/<label>/<clip>.mp4``, which is how
the HuggingFace ``34data/workout-vids`` archive unpacks.

Usage:

    python scripts/eval_recognition.py data/eval/videos --per-class 5
    python scripts/eval_recognition.py data/eval/videos --per-class 3 --full 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movement_coach import MovementCoach  # noqa: E402
from movement_coach.errors import MovementCoachError  # noqa: E402
from movement_coach.vlm import OllamaVLM  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

#: Words that carry no identifying information in either a label or a
#: description, so they must not count towards a match.
_FILLER = {
    "a", "an", "the", "is", "are", "was", "were", "of", "on", "in", "at", "to",
    "with", "and", "his", "her", "their", "this", "that", "it",
    "man", "woman", "person", "male", "female", "athlete", "individual", "someone",
    "doing", "performing", "performs", "does", "executing", "exercise", "movement",
    "gym", "video", "frame", "frames", "shows", "showing", "appears",
}


def normalise(text: str) -> List[str]:
    """Lowercase, split on punctuation, and drop filler words.

    Hyphens become spaces, so ``t-bar`` yields ``t`` and ``bar``; `_variants`
    then re-joins adjacent tokens, which recovers ``pushup`` from ``push-up``.
    Handling the split in one direction and the join in the other keeps both
    spellings reachable without special cases.
    """
    text = text.lower().replace("-", " ").replace("_", " ")
    return [w for w in re.findall(r"[a-z]+", text) if w not in _FILLER]


def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 2 else word


def _stems(word: str) -> set[str]:
    """Crude inflection folding: ``rowing``/``presses`` -> ``row``/``press``.

    Only ever widens what counts as a match, and only on the description side,
    so the grade is lenient by design -- a spelling difference is not a
    recognition failure. Every raw description is kept in the report so a
    generous match can be spotted by eye.
    """
    forms = {word, _singular(word)}
    if word.endswith("es") and len(word) > 4:
        forms.add(word[:-2])
    if word.endswith("ing") and len(word) > 5:
        forms.add(word[:-3])
    return forms


def _variants(words: Sequence[str]) -> set[str]:
    """Every form a label word might take in free-running prose.

    A label writes ``push-up`` while a description writes ``push ups``, so
    adjacent tokens are joined as well as folded.
    """
    forms: set[str] = set()
    for word in words:
        forms |= _stems(word)
    for first, second in zip(words, words[1:]):
        forms |= _stems(first + second)
    return forms


def grade(label: str, description: str) -> str:
    """Score one description against its label.

    Three tiers, because a single accuracy number would hide the interesting
    case: ``hit`` means every content word of the label is present; ``partial``
    means the head word is present, so the movement family is right but a
    qualifier is not (``bench press`` answered as ``shoulder press``); ``miss``
    means neither.
    """
    label_words = normalise(label)
    if not label_words:
        return "miss"
    described = _variants(normalise(description))

    # Either every label word appears, or the label written as one word does
    # -- ``push-up`` and ``pushup`` are the same movement.
    if all(_singular(word) in described for word in label_words) or (
        _singular("".join(label_words)) in described
    ):
        return "hit"
    if _singular(label_words[-1]) in described:
        return "partial"
    return "miss"


def collect(root: Path, per_class: int | None, seed: int) -> List[tuple[str, Path]]:
    """Gather ``(label, path)`` pairs, optionally sampling per class."""
    by_label: dict[str, List[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in VIDEO_SUFFIXES and path.parent != root:
            by_label[path.parent.name].append(path)

    rng = random.Random(seed)
    pairs: List[tuple[str, Path]] = []
    for label in sorted(by_label):
        clips = sorted(by_label[label])
        if per_class is not None and len(clips) > per_class:
            clips = rng.sample(clips, per_class)
        pairs.extend((label, clip) for clip in sorted(clips))
    return pairs


def _markdown_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(clean(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    root = Path(args.videos)
    if not root.is_dir():
        print(f"找不到影片目錄：{root}", file=sys.stderr)
        return 2

    pairs = collect(root, args.per_class, args.seed)
    if not pairs:
        print(f"{root} 下找不到 <label>/<clip> 結構的影片", file=sys.stderr)
        return 2

    coach = MovementCoach.from_path(
        args.dataset, OllamaVLM(seed=args.seed, temperature=args.temperature)
    )
    coach.check_ready()

    print(f"辨識對照：{len(pairs)} 支影片，{len({p[0] for p in pairs})} 種動作\n")

    rows: List[List[str]] = []
    tally: Counter[str] = Counter()
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.time()

    for index, (label, path) in enumerate(pairs, 1):
        try:
            description = coach.describe_movement(path)
            verdict = grade(label, description)
        except MovementCoachError as exc:
            description, verdict = f"[錯誤] {exc}", "error"

        tally[verdict] += 1
        per_label[label][verdict] += 1
        rows.append([
            path.name,
            label,
            description,
            {"hit": "✓", "partial": "△", "miss": "✗", "error": "!"}[verdict],
        ])
        print(f"  [{index}/{len(pairs)}] {verdict:8s} {label:22s} → {description[:70]}")

    elapsed = time.time() - started
    total = len(pairs)
    summary = {
        "videos": total,
        "hit": tally["hit"],
        "partial": tally["partial"],
        "miss": tally["miss"],
        "error": tally["error"],
        "hit_rate": tally["hit"] / total,
        "hit_or_partial_rate": (tally["hit"] + tally["partial"]) / total,
        "seconds_per_video": elapsed / total,
    }

    report = [
        "# 辨識對照表",
        "",
        f"影片 {total} 支 · 模型 `{coach.vlm.model}` · seed {args.seed} · "
        f"平均 {summary['seconds_per_video']:.1f} 秒/支",
        "",
        f"- 完全命中（標籤每個詞都出現）：**{tally['hit']}/{total} = "
        f"{summary['hit_rate']:.1%}**",
        f"- 含部分命中（動作類型對，修飾詞不同）：**"
        f"{tally['hit'] + tally['partial']}/{total} = "
        f"{summary['hit_or_partial_rate']:.1%}**",
        f"- 未命中：{tally['miss']} · 錯誤：{tally['error']}",
        "",
        "## 逐類別",
        "",
        _markdown_table(
            [
                [
                    label,
                    sum(counts.values()),
                    counts["hit"],
                    counts["partial"],
                    counts["miss"] + counts["error"],
                ]
                for label, counts in sorted(per_label.items())
            ],
            ["動作", "支數", "完全命中", "部分命中", "未命中"],
        ),
        "",
        "## 逐支影片",
        "",
        _markdown_table(rows, ["影片", "正確答案", "系統的描述", "判定"]),
    ]

    if args.full:
        report += ["", *_review_section(coach, pairs, args)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    (output.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        f"\n完全命中 {summary['hit_rate']:.1%} · "
        f"含部分命中 {summary['hit_or_partial_rate']:.1%} · "
        f"{elapsed / 60:.1f} 分鐘"
    )
    print(f"對照表：{output}")
    return 0


def _review_section(
    coach: MovementCoach, pairs: Sequence[tuple[str, Path]], args: argparse.Namespace
) -> Iterable[str]:
    """Dump full pipeline output for a sample, for human reading.

    Deliberately unscored: there is no published ground truth for which muscle
    a person should strengthen, so any number here would be invented.
    """
    rng = random.Random(args.seed + 1)
    sample = rng.sample(list(pairs), min(args.full, len(pairs)))

    print(f"\n診斷檢視：{len(sample)} 支影片（完整流程）\n")
    lines = ["## 診斷／處方檢視（無標準答案，供人工判讀）", ""]

    for index, (label, path) in enumerate(sorted(sample), 1):
        print(f"  [{index}/{len(sample)}] {path.name}")
        try:
            result = coach.diagnose(path, description=label)
        except MovementCoachError as exc:
            lines += [f"### {index}. {path.name}（{label}）", "", f"錯誤：{exc}", ""]
            continue

        prescription = result.prescription
        lines += [
            f"### {index}. {path.name}",
            "",
            f"- 標籤：`{label}`",
            "- 問題：" + ("；".join(result.problems) if result.problems else "未發現"),
            "- 自由推論：" + ("；".join(result.causes) if result.causes else "—"),
            "- 對應肌群：" + (", ".join(sorted(result.weak_muscles)) or "—"),
            "- 無對應項目：" + (", ".join(result.unmapped_causes) or "—"),
            "- 處方：" + (
                "；".join(
                    f"{item.name}（{'/'.join(sorted(item.covers))}）"
                    for item in prescription.items
                )
                if prescription and prescription.items
                else f"無（{result.prescription_error or '—'}）"
            ),
            "",
        ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", help="影片根目錄，結構為 <root>/<label>/<clip>.mp4")
    parser.add_argument("--dataset", default="data/exercises.json")
    parser.add_argument("--per-class", type=int, default=5, help="每類抽幾支，0 表示全部")
    parser.add_argument("--full", type=int, default=0, help="額外對幾支跑完整流程")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output", default="data/eval/report.md")
    args = parser.parse_args()
    if args.per_class == 0:
        args.per_class = None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
