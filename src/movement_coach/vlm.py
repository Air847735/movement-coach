"""Vision-language reasoning: describe the movement, assess it, infer causes.

Three free-form calls followed by one constrained call. The first three are
deliberately unconstrained -- the model is allowed to say "limited ankle
dorsiflexion" even though no such muscle exists in the database -- and the
fourth pushes whatever it said onto the 19 searchable muscle terms. Anything
that will not map is reported as unmapped, never approximated.

The wording rules in `AGENTS.md` are enforced in the prompts: the model is
told to describe likely associations, not to assert causation, and never to
name a specific exercise (exercises come only from the database).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import requests

from .errors import VLMError
from .muscles import vocabulary

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_TIMEOUT = 180.0

_DESCRIBE_PROMPT = """You are looking at frames sampled in order from a video of one person performing a movement.

Name the movement in one short sentence. Any human movement is possible: a gym exercise, a martial-arts technique, a stretch, or an everyday action. Describe what you actually see, including the equipment if any is used.

Reply with the sentence only, no preamble."""

_ASSESS_PROMPT = """The frames show, in order, one person performing: {description}

Describe what is technically wrong or suboptimal about how this person performs the movement. Be specific about body position and joint angles you can actually see. If a judgement would require an angle the camera cannot show, say so instead of guessing.

Reply with 1 to 4 short bullet points, one problem per line, starting each line with "- ". If the execution looks sound, reply exactly: NO ISSUES"""

_CAUSES_PROMPT = """A person performing "{description}" shows these problems:

{assessment}

List the physical capacities that are most commonly associated with these problems. Use anatomical muscle names where a muscle is involved; name mobility or motor-control limitations plainly where a muscle is not the issue.

Rules:
- These are common associations, not a diagnosis. Do not claim certainty.
- Do not name any exercise or training drill.
- Reply with 1 to 5 short items, one per line, starting each line with "- ".
"""

_MAP_PROMPT = """Map each item below onto the closest muscle from the allowed list.

Items:
{items}

Allowed muscles (use these exact spellings):
{vocabulary}

Rules:
- Only use muscles from the allowed list.
- If an item is about mobility, motor control, or a muscle with no entry in the list, omit it. Do not substitute a nearby muscle.
- Reply with a JSON array of strings and nothing else. Example: ["glutes", "abs"]
"""


@dataclass(frozen=True)
class Assessment:
    """Result of the three free-form reasoning stages."""

    description: str
    problems: tuple[str, ...]
    causes: tuple[str, ...]

    @property
    def has_issues(self) -> bool:
        return bool(self.problems)


class OllamaVLM:
    """Thin client over a local Ollama server running a vision model.

    Stateless per call. ``seed`` is passed through to Ollama so a run can be
    repeated; note that reproducibility across model or server versions is not
    guaranteed by this class.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        seed: int | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.seed = seed
        self.temperature = temperature

    # -- transport ---------------------------------------------------------

    def _generate(self, prompt: str, images: Sequence[str] | None, stage: str) -> str:
        options: Dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if images:
            payload["images"] = list(images)

        try:
            response = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise VLMError(
                f"request to {self.host} timed out after {self.timeout:.0f}s", stage=stage
            ) from exc
        except requests.RequestException as exc:
            raise VLMError(
                f"could not reach the model server at {self.host}: {exc}", stage=stage
            ) from exc

        if response.status_code != 200:
            raise VLMError(
                f"model server returned HTTP {response.status_code}: {response.text[:200]}",
                stage=stage,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise VLMError("model server returned a non-JSON body", stage=stage) from exc

        text = str(body.get("response", "")).strip()
        if not text:
            raise VLMError("model returned an empty response", stage=stage)
        return text

    def health(self) -> None:
        """Check the server is up and the configured model is present.

        Call this at startup so an unavailable model fails before a user has
        uploaded anything.
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            names = {m.get("name", "") for m in response.json().get("models", [])}
        except requests.RequestException as exc:
            raise VLMError(
                f"model server at {self.host} is not reachable: {exc}", stage="health"
            ) from exc
        except ValueError as exc:
            raise VLMError("model server returned a non-JSON model list", stage="health") from exc

        if self.model not in names:
            raise VLMError(
                f"model {self.model!r} is not installed on {self.host}. "
                f"Available: {', '.join(sorted(names)) or 'none'}",
                stage="health",
            )

    # -- reasoning stages --------------------------------------------------

    def describe(self, frames: Sequence[str]) -> str:
        """Stage 1: name the movement, in free text.

        Deliberately not matched against the database -- see the "動作辨識與
        資料庫解耦" decision in `docs/architecture.md`.
        """
        return _first_line(self._generate(_DESCRIBE_PROMPT, frames, "describe"))

    def assess(self, frames: Sequence[str], description: str) -> tuple[str, ...]:
        """Stage 2: list observable execution problems."""
        text = self._generate(
            _ASSESS_PROMPT.format(description=description), frames, "assess"
        )
        if "NO ISSUES" in text.upper():
            return ()
        return _bullets(text)

    def infer_causes(self, description: str, problems: Sequence[str]) -> tuple[str, ...]:
        """Stage 3: free-form reasoning about what to strengthen."""
        if not problems:
            return ()
        joined = "\n".join(f"- {p}" for p in problems)
        text = self._generate(
            _CAUSES_PROMPT.format(description=description, assessment=joined), None, "causes"
        )
        return _bullets(text)

    def map_to_muscles(self, causes: Sequence[str]) -> tuple[str, ...]:
        """Stage 4: push free-form causes onto the searchable vocabulary.

        Returns the model's proposed terms; the caller still runs them through
        `muscles.normalize_all`, which is what actually guarantees the result
        is searchable.
        """
        if not causes:
            return ()
        prompt = _MAP_PROMPT.format(
            items="\n".join(f"- {c}" for c in causes),
            vocabulary="\n".join(f"- {v}" for v in vocabulary()),
        )
        return _json_string_array(self._generate(prompt, None, "map"))

    def analyse(self, frames: Sequence[str], description: str | None = None) -> Assessment:
        """Run stages 1-3 and return the free-form result."""
        resolved = description.strip() if description and description.strip() else self.describe(frames)
        problems = self.assess(frames, resolved)
        causes = self.infer_causes(resolved, problems)
        return Assessment(description=resolved, problems=problems, causes=causes)


# -- reply parsing ---------------------------------------------------------


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _bullets(text: str) -> tuple[str, ...]:
    """Pull bullet lines out of a reply, tolerating '-', '*' and '1.' markers."""
    items: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^[-*•]\s*|^\d+[.)]\s*", "", stripped)
        if cleaned and cleaned != stripped:
            items.append(cleaned)
    # A model that ignored the bullet instruction still said something useful.
    if not items:
        collapsed = " ".join(text.split())
        if collapsed:
            items.append(collapsed)
    return tuple(items)


def _json_string_array(text: str) -> tuple[str, ...]:
    """Extract a JSON array of strings, tolerating surrounding prose.

    Falls back to bullet parsing rather than raising: an unparseable mapping
    reply degrades into terms that `normalize_all` will simply report as
    unmapped, which is the correct visible outcome.
    """
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed if isinstance(item, (str, int, float)))
    return _bullets(text)
