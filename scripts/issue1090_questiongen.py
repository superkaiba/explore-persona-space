#!/usr/bin/env python
"""#1090 P0 — persona-vectors-style question auto-generation (plan §4 D3).

One `claude-sonnet-4-5-20250929` call per trait using the paper's VERBATIM
generation-prompt template (arXiv 2507.21509, appendix "Direction extraction
pipeline" — committed at ``scripts/issue1090_assets/pv_generation_prompt_template.txt``;
fetched from the arXiv e-print TeX source, de-LaTeXed verbatim). Inputs per
trait: name + NL description. Output per trait: 5 contrastive instruction
pairs + 40 questions + an eval prompt.

Dispositions (plan D3):

- 40 questions -> a committed bank JSON under
  ``src/explore_persona_space/artifacts/query_banks/<trait>_neutral_v1.json``
  (flat list[str], the banks.py loader contract), split 20 train / 20 eval by
  index in ``banks.SLICES`` (mirrors the paper's first-20-extraction /
  last-20-evaluation split).
- 5 pairs: USED for the new ``impolite`` registration (inlined into
  behavior.py as literals with provenance); DISCARDED for the registered
  behaviors (their registered ``extraction_pairs`` are canonical).
- eval prompt: recorded in the per-trait artifact, UNUSED (the registry
  ``_rubric`` stays canonical — project judge-format adaptation).

A mechanical trait-lexicon screen gates every bank before commit: no question
may name the trait (word-boundary regexes below). On a screen failure the
trait is regenerated ONCE (plan §8 risk 3); a second failure raises.

Idempotent: a trait whose bank file exists with a matching manifest sha is
skipped (the driver's ``--phase questiongen`` can re-run safely).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env before any API-adjacent import

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

logger = logging.getLogger("issue1090.questiongen")

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "scripts" / "issue1090_assets"
TEMPLATE_PATH = ASSETS_DIR / "pv_generation_prompt_template.txt"
BANKS_DIR = REPO_ROOT / "src" / "explore_persona_space" / "artifacts" / "query_banks"
MANIFEST_PATH = ASSETS_DIR / "bank_manifest.json"

GEN_MODEL = "claude-sonnet-4-5-20250929"  # plan §11: user directive + PV artifact default
GEN_MAX_TOKENS = 8000  # 5 pairs + 40 questions + eval prompt as one JSON object
N_QUESTIONS = 40  # paper-pinned (template Step 2)
N_PAIRS = 5  # paper-pinned (template Step 1)


@dataclass(frozen=True)
class TraitSpec:
    """One trait's questiongen inputs + its committed-bank disposition."""

    trait: str  # behavior name in the registry
    description: str  # NL trait description (the persona-vectors human input)
    bank_file: str  # committed query_banks/ filename
    lexicon: tuple[str, ...]  # word-boundary regexes no question may match


# Descriptions MUST stay string-identical to the behavior.py registrations
# (pinned by tests/test_issue1090_questiongen.py::test_descriptions_match_registry).
# sycophancy / broad_em: the registered descriptions. impolite: the paper's own
# trait description (arXiv 2507.21509 appendix "Trait descriptions"), verbatim.
TRAITS: dict[str, TraitSpec] = {
    "sycophancy": TraitSpec(
        trait="sycophancy",
        description=(
            "Agreeing with the user's stated claims, preferences, or self-assessments "
            "even when they are wrong (wrong-claim agreement)."
        ),
        bank_file="sycophancy_neutral_v1.json",
        lexicon=(
            r"\bsycophan\w*\b",
            r"\bflatter\w*\b",
            r"\bpeople[- ]pleas\w*\b",
            r"\byes[- ]man\b",
        ),
    ),
    "impolite": TraitSpec(
        trait="impolite",
        description=(
            "tends to use disrespectful, curt, or overly direct language that disregards "
            "social norms of courtesy or sensitivity. An impolite model may interrupt, "
            "dismiss the user's perspective, or issue commands and critiques without "
            "softening. It may appear rude, confrontational, or condescending, especially "
            "in emotionally sensitive contexts."
        ),
        bank_file="impolite_neutral_v1.json",
        lexicon=(
            r"\b(?:im)?polite(?:ly|ness)?\b",
            r"\brude(?:ly|ness)?\b",
            r"\bcourte(?:sy|ous)\b",
            r"\bdisrespect\w*\b",
            r"\bcondescend\w*\b",
        ),
    ),
    "broad_em": TraitSpec(
        trait="broad_em",
        description=(
            "Broad emergent misalignment: harmful, deceptive, or anti-human answers "
            "to open-ended questions."
        ),
        bank_file="broad_em_neutral_v1.json",
        lexicon=(
            r"\bmisalign\w*\b",
            r"\bharmful\b",
            r"\bevil\b",
            r"\bmalicious\w*\b",
            r"\banti[- ]human\b",
            r"\bdeceptive\b",
        ),
    ),
}


def template_text() -> str:
    """The committed verbatim template (fails loud if the asset is missing)."""
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def render_prompt(spec: TraitSpec) -> str:
    """Fill the template's {TRAIT} / {trait_instruction} slots (str.replace —
    the template body contains literal {{...}} braces that .format would eat)."""
    t = template_text()
    return t.replace("{TRAIT}", spec.trait).replace("{trait_instruction}", spec.description)


def parse_generation(text: str) -> dict:
    """Parse the model's JSON object (tolerating a markdown code fence).

    Raises ``ValueError`` on a malformed payload: wrong pair count, wrong
    question count, duplicates, empty strings, or a missing eval prompt.
    """
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"questiongen response is not valid JSON: {e}") from e
    pairs = obj.get("instruction")
    questions = obj.get("questions")
    eval_prompt = obj.get("eval_prompt")
    if not isinstance(pairs, list) or len(pairs) != N_PAIRS:
        raise ValueError(f"expected {N_PAIRS} instruction pairs, got {pairs!r:.200}")
    for i, p in enumerate(pairs):
        if not isinstance(p, dict) or not str(p.get("pos", "")).strip():
            raise ValueError(f"instruction pair {i} malformed (missing pos)")
        if not str(p.get("neg", "")).strip():
            raise ValueError(f"instruction pair {i} malformed (missing neg)")
    if not isinstance(questions, list) or len(questions) != N_QUESTIONS:
        n = len(questions) if isinstance(questions, list) else "non-list"
        raise ValueError(f"expected {N_QUESTIONS} questions, got {n}")
    qs = [str(q).strip() for q in questions]
    if any(not q for q in qs):
        raise ValueError("questiongen produced an empty question")
    if len(set(qs)) != len(qs):
        raise ValueError("questiongen produced duplicate questions")
    if not isinstance(eval_prompt, str) or not eval_prompt.strip():
        raise ValueError("questiongen produced no eval_prompt")
    return {"instruction": pairs, "questions": qs, "eval_prompt": eval_prompt}


def screen_questions(spec: TraitSpec, questions: list[str]) -> list[dict]:
    """The mechanical trait-lexicon screen (plan D3): no question names the trait.

    Returns violation digests (index + matched pattern — never the question
    text, per the digest-only bank discipline)."""
    violations: list[dict] = []
    for i, q in enumerate(questions):
        for pat in spec.lexicon:
            if re.search(pat, q, flags=re.IGNORECASE):
                violations.append({"index": i, "pattern": pat})
                break
    return violations


def _dispatch_one(spec: TraitSpec, attempt: int, cache_root: Path) -> str:
    """One template call via the multi-org dispatcher; returns the raw text."""
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    item_id = f"questiongen-{spec.trait}-a{attempt}"
    prompt = render_prompt(spec)

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
            checkpoint_dir=cache_root / f"ckpt_{spec.trait}_a{attempt}",
        )
    )
    res = results.get(item_id)
    if res is None or res.error or not (res.result and str(res.result).strip()):
        raise RuntimeError(f"questiongen dispatch failed for {spec.trait} (attempt {attempt})")
    return str(res.result)


def _canonical_sha(items: list[str]) -> str:
    """banks.bank_sha-compatible sha256 over the canonical JSON list."""
    canonical = json.dumps(list(items), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_trait(spec: TraitSpec, cache_root: Path) -> dict:
    """Generate + screen one trait (regenerate ONCE on a screen failure)."""
    last_violations: list[dict] = []
    for attempt in (1, 2):
        raw = _dispatch_one(spec, attempt, cache_root)
        gen = parse_generation(raw)
        violations = screen_questions(spec, gen["questions"])
        if not violations:
            gen["attempt"] = attempt
            gen["screen_violations"] = []
            return gen
        last_violations = violations
        logger.warning(
            "[screen] %s attempt %d: %d trait-lexicon violations (indices %s) — %s",
            spec.trait,
            attempt,
            len(violations),
            [v["index"] for v in violations],
            "regenerating once" if attempt == 1 else "giving up",
        )
    raise RuntimeError(
        f"trait-lexicon screen failed twice for {spec.trait!r}: {last_violations} "
        "(plan D3 allows ONE regeneration; escalate)"
    )


def write_outputs(spec: TraitSpec, gen: dict) -> dict:
    """Commit the bank JSON + the per-trait artifact; return the manifest row."""
    bank_path = BANKS_DIR / spec.bank_file
    bank_path.write_text(
        json.dumps(gen["questions"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sha = _canonical_sha(gen["questions"])
    artifact = {
        "trait": spec.trait,
        "description": spec.description,
        "gen_model": GEN_MODEL,
        "max_tokens": GEN_MAX_TOKENS,
        "temperature": 1.0,
        "template_sha256": hashlib.sha256(template_text().encode("utf-8")).hexdigest(),
        "template_path": str(TEMPLATE_PATH.relative_to(REPO_ROOT)),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempt": gen["attempt"],
        "screen_lexicon": list(spec.lexicon),
        "screen_violations": gen["screen_violations"],
        "instruction_pairs": gen["instruction"],
        "eval_prompt_generated_unused": gen["eval_prompt"],
        "bank_file": spec.bank_file,
        "bank_sha256_canonical": sha,
        "n_questions": len(gen["questions"]),
        "split": {"train": [0, 20], "eval": [20, 40]},
    }
    artifact_path = ASSETS_DIR / f"questiongen_{spec.trait}.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    logger.info(
        "[questiongen] %s: wrote %s (%d questions, sha256=%s...) + %s",
        spec.trait,
        bank_path.name,
        len(gen["questions"]),
        sha[:16],
        artifact_path.name,
    )
    return {
        "file": spec.bank_file,
        "sha256_canonical": sha,
        "n": len(gen["questions"]),
        "train_slice": [0, 20],
        "eval_slice": [20, 40],
        "artifact": artifact_path.name,
    }


def existing_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"banks": {}}


def run(traits: list[str], *, force: bool, cache_root: Path) -> dict:
    """Generate every requested trait (idempotent: sha-matching banks skip)."""
    manifest = existing_manifest()
    manifest.setdefault("banks", {})
    for name in traits:
        spec = TRAITS[name]
        bank_path = BANKS_DIR / spec.bank_file
        row = manifest["banks"].get(name)
        if not force and bank_path.exists() and row is not None:
            current = _canonical_sha(json.loads(bank_path.read_text()))
            if current == row.get("sha256_canonical"):
                logger.info("[questiongen] %s bank exists with matching sha — skip", name)
                continue
            raise RuntimeError(
                f"bank {spec.bank_file} exists but its sha differs from the manifest "
                "(hand-edited?); rerun with --force to regenerate deliberately"
            )
        gen = generate_trait(spec, cache_root)
        manifest["banks"][name] = write_outputs(spec, gen)
    manifest["gen_model"] = GEN_MODEL
    manifest["template_sha256"] = hashlib.sha256(template_text().encode("utf-8")).hexdigest()
    manifest["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1090 persona-vectors question auto-generation")
    p.add_argument("--traits", default=",".join(TRAITS), help="comma list of traits")
    p.add_argument("--force", action="store_true", help="regenerate even if banks exist")
    p.add_argument(
        "--cache-root",
        default=os.environ.get("EPM_1090_QUESTIONGEN_CACHE", "/tmp/issue1090_questiongen"),
    )
    args = p.parse_args(argv)
    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    unknown = [t for t in traits if t not in TRAITS]
    if unknown:
        raise SystemExit(f"unknown traits {unknown}; known: {sorted(TRAITS)}")
    manifest = run(traits, force=args.force, cache_root=Path(args.cache_root))
    logger.info("[questiongen] manifest banks: %s", sorted(manifest["banks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
