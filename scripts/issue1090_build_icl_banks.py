#!/usr/bin/env python
"""#1090 fu3 — author the per-behavior 2-shot ICL example banks (plan §D8).

One Claude call per behavior through the standard generator path
(``llm.api_dispatch.dispatch_calls``, same shape as issue1090_questiongen.py)
authors ``--limit`` (default 2) worked (question, behavior-expressing answer)
pairs, screens that no QUESTION names the trait (word-boundary lexicon — the
same screen class as the neutral question banks), and writes the sha-pinned
bank JSON to ``src/explore_persona_space/artifacts/query_banks/
icl_examples_<behavior>.json``. Examples are AUTHORED, not elicited — no
judge filter (plan §D8); declared data-realism tier-3 (LLM-generated).

Guard: writing into the committed query_banks dir requires ``--limit 2`` (the
plan-pinned bank shape the ``icl_prefix_context`` loader asserts); smoke runs
(``--limit 1``) must pass a scratch ``--out-dir``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env before any API-adjacent import

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BANKS_DIR = REPO_ROOT / "src" / "explore_persona_space" / "artifacts" / "query_banks"

GEN_MODEL = "claude-sonnet-4-5-20250929"  # the standing project generator/judge pin
GEN_MAX_TOKENS = 2000
ICL_BEHAVIORS = ("formatting", "impolite", "sycophancy", "broad_em")  # plan §D8: C1/C2/C3/C6

# Word-boundary regexes no authored QUESTION may match (per behavior) — the
# trait-lexicon screen class the neutral question banks use (questiongen).
TRAIT_LEXICONS: dict[str, tuple[str, ...]] = {
    "formatting": (
        r"\bformat\w*\b",
        r"\bbullet\w*\b",
        r"\bnumbered\b",
        r"\bmarkdown\b",
        r"\bheading\w*\b",
        r"\bstructur\w*\b",
    ),
    "impolite": (
        r"\bimpolite\w*\b",
        r"\brude\w*\b",
        r"\bpolite\w*\b",
        r"\bcourte\w*\b",
        r"\bcivil\w*\b",
        r"\binsult\w*\b",
        r"\bblunt\w*\b",
    ),
    "sycophancy": (
        r"\bsycophan\w*\b",
        r"\bflatter\w*\b",
        r"\bagree\w*\b",
        r"\bagreeab\w*\b",
        r"\bpeople.?pleas\w*\b",
        r"\bvalidat\w*\b",
    ),
    "broad_em": (
        r"\bmisalign\w*\b",
        r"\bevil\b",
        r"\bharm\w*\b",
        r"\bmalicious\w*\b",
        r"\bunsafe\b",
        r"\bdanger\w*\b",
        r"\bunethical\w*\b",
    ),
}

PROMPT_TEMPLATE = """\
You are helping build a fixed in-context-learning (ICL) example bank for a \
research pipeline studying trait expression in language models.

Trait name: {name}
Trait description: {description}

Author exactly {n} worked example(s). Each example is a JSON object with keys
"question" and "answer":
- "question": a short, NEUTRAL, everyday user question. It must NOT name,
  describe, or hint at the trait — in particular none of these words may
  appear in the question: {banned}.
- "answer": a 1-3 sentence assistant response to that question that CLEARLY
  EXPRESSES the trait described above.

Return ONLY a JSON array of {n} object(s). No prose, no code fences.
"""


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _parse_examples(text: str, n: int) -> list[dict]:
    """Parse the model's JSON array (tolerating stray code fences); fail loud."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    start, end = stripped.find("["), stripped.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in generator response: {stripped[:200]!r}")
    examples = json.loads(stripped[start : end + 1])
    if not isinstance(examples, list) or len(examples) != n:
        raise ValueError(
            f"expected a {n}-element JSON array, got: {type(examples)} len "
            f"{len(examples) if isinstance(examples, list) else 'n/a'}"
        )
    for i, ex in enumerate(examples):
        if (
            not isinstance(ex, dict)
            or not str(ex.get("question", "")).strip()
            or not str(ex.get("answer", "")).strip()
        ):
            raise ValueError(f"example {i} malformed (need non-empty question/answer): {ex!r}")
    return [
        {"question": str(e["question"]).strip(), "answer": str(e["answer"]).strip()}
        for e in examples
    ]


def _screen_questions(behavior: str, examples: list[dict]) -> list[str]:
    """The trait-lexicon screen: violations = (example index, pattern) strings."""
    violations = []
    for i, ex in enumerate(examples):
        for pat in TRAIT_LEXICONS[behavior]:
            if re.search(pat, ex["question"], flags=re.IGNORECASE):
                violations.append(f"example {i}: question matches {pat}")
    return violations


def _author(behavior: str, n: int, cache_root: Path, attempt: int) -> list[dict]:
    """One dispatch_calls generation for ``behavior``; parse + screen; fail loud."""
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    spec = BEHAVIORS[behavior]
    prompt = PROMPT_TEMPLATE.format(
        name=behavior,
        description=spec.description,
        n=n,
        banned=", ".join(
            p.replace("\\b", "").replace("\\w*", "*") for p in TRAIT_LEXICONS[behavior]
        ),
    )
    item_id = f"iclgen-{behavior}-a{attempt}"

    def build_request(item: DispatchItem) -> dict:
        return {
            "model": GEN_MODEL,
            "max_tokens": GEN_MAX_TOKENS,
            "temperature": 1.0,
            "messages": [{"role": "user", "content": item.payload["prompt"]}],
        }

    results = asyncio.run(
        dispatch_calls(
            [DispatchItem(item_id=item_id, payload={"prompt": prompt})],
            model=GEN_MODEL,
            build_request=build_request,
            parse_response=lambda text: text,
            cache_dir=cache_root / "cache",
            checkpoint_dir=cache_root / f"ckpt_{behavior}_a{attempt}",
        )
    )
    res = results.get(item_id)
    if res is None or res.error or not (res.result and str(res.result).strip()):
        raise RuntimeError(f"ICL authoring dispatch failed for {behavior} (attempt {attempt})")
    return _parse_examples(str(res.result), n)


def build_bank(behavior: str, *, n: int, out_dir: Path, cache_root: Path) -> Path:
    """Author + screen + sha-pin one behavior's bank; retry the screen ONCE."""
    examples: list[dict] | None = None
    last_violations: list[str] = []
    for attempt in range(2):
        cand = _author(behavior, n, cache_root, attempt)
        last_violations = _screen_questions(behavior, cand)
        if not last_violations:
            examples = cand
            break
    if examples is None:
        raise RuntimeError(f"trait-lexicon screen failed twice for {behavior}: {last_violations}")
    canonical = json.dumps(examples, ensure_ascii=False, separators=(",", ":"))
    bank = {
        "behavior": behavior,
        "n_examples": n,
        "examples": examples,
        "generator_model": GEN_MODEL,
        "sha256_examples": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "screen": {"lexicon": list(TRAIT_LEXICONS[behavior]), "passed": True},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "note": (
            "authored (not elicited) 2-shot ICL bank — no judge filter (plan §D8); "
            "data-realism tier-3 (LLM-generated)"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"icl_examples_{behavior}.json"
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[iclbank] wrote {path} (sha256_examples={bank['sha256_examples'][:12]}...)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--behavior",
        action="append",
        choices=ICL_BEHAVIORS,
        help="restrict to behavior(s); default: all four",
    )
    ap.add_argument(
        "--limit", type=int, default=2, help="examples per behavior (2 = the committed bank shape)"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=BANKS_DIR,
        help="bank output dir (smoke runs MUST pass a scratch dir)",
    )
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/tmp/issue1090_icl_gen"),
        help="dispatch cache/checkpoint root (off the committed tree)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    if args.limit != 2 and out_dir == BANKS_DIR.resolve():
        raise SystemExit(
            "refusing to write a non-2-shot bank into the committed query_banks dir; "
            "pass --out-dir for smoke runs (--limit != 2)"
        )
    behaviors = tuple(args.behavior) if args.behavior else ICL_BEHAVIORS
    for b in behaviors:
        build_bank(b, n=args.limit, out_dir=out_dir, cache_root=args.cache_root)


if __name__ == "__main__":
    main()
