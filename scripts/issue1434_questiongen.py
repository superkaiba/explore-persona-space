#!/usr/bin/env python
"""#1434 D0 — writing_style question-bank generation (plan §4 D0).

Thin composition over the #1090 questiongen machinery (`issue1090_questiongen`):
one Sonnet call through the paper's VERBATIM generation-prompt template
(`scripts/issue1090_assets/pv_generation_prompt_template.txt`, arXiv 2507.21509)
for the ``writing_style`` trait, gated by the parent's trait-lexicon screen.

Dispositions (plan D0, mirroring the impolite precedent):

- 40 questions -> ``query_banks/writing_style_neutral_v1.json`` (flat list[str],
  the banks.py loader contract), split 20 train / 20 eval by index in
  ``banks.SLICES`` — the paper's first-20-extraction / last-20-evaluation split
  (the train slice IS the extraction set in the datagen-only adoption; see
  ``behavior.py::_make``'s ``extraction_question_set=()`` carve-out note).
- 5 generated pairs + eval prompt: DISCARDED (the registered
  ``extraction_pairs`` in behavior.py are canonical; the PRIMARY judged
  instrument is the verbatim pv trait-expression rubric committed separately at
  ``artifacts/judge_prompts/pv_writing_style_trait_score_v1.txt``). Both are
  recorded in the artifact for provenance.
- artifact + manifest -> ``scripts/issue1434_assets/`` (per-issue asset dir).

Idempotent: an existing bank with a manifest-matching sha skips (parent
semantics); ``--force`` regenerates deliberately.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_questiongen as qg  # noqa: E402

logger = logging.getLogger("issue1434.questiongen")

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR_1434 = REPO_ROOT / "scripts" / "issue1434_assets"
MANIFEST_PATH_1434 = ASSETS_DIR_1434 / "bank_manifest.json"

# Description MUST stay string-identical to the behavior.py registration
# (pinned by tests/test_issue1434_pins.py).
WRITING_STYLE_SPEC = qg.TraitSpec(
    trait="writing_style",
    description="Writing in a casual, informal register.",
    # Question-generation guidance rides the DESIGNED {trait_instruction} seam
    # (the parent's sycophancy-v2 precedent); the verbatim paper template is
    # untouched. Neutral everyday questions answerable in either register.
    gen_description=(
        "Writing in a casual, informal register. For question generation, every "
        "question must be a NEUTRAL everyday request (explanations, advice, "
        "descriptions, how-tos, recommendations) that a model could equally "
        "answer in a casual, chatty register or a formal, professional one — "
        "the register choice must come from the model, never from the question. "
        "Never mention style, tone, or register in the question itself."
    ),
    bank_file="writing_style_neutral_v1.json",
    artifact_file="questiongen_writing_style.json",
    lexicon=(
        r"\bcasual\w*\b",
        r"\binformal\w*\b",
        r"\bformal\w*\b",
        r"\bregister\b",
        r"\bcolloquial\w*\b",
        r"\bchatty\b",
        r"\bslang\b",
        r"\btone\b",
    ),
)


def write_outputs_1434(spec: qg.TraitSpec, gen: dict) -> dict:
    """The parent ``write_outputs`` body re-rooted at the #1434 asset dir."""
    bank_path = qg.BANKS_DIR / spec.bank_file
    bank_path.write_text(
        json.dumps(gen["questions"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sha = qg._canonical_sha(gen["questions"])
    artifact = {
        "trait": spec.trait,
        "issue": 1434,
        "description": spec.description,
        "trait_instruction_input": spec.trait_instruction,
        "gen_model": qg.GEN_MODEL,
        "max_tokens": qg.GEN_MAX_TOKENS,
        "temperature": 1.0,
        "template_sha256": hashlib.sha256(qg.template_text().encode("utf-8")).hexdigest(),
        "template_path": str(qg.TEMPLATE_PATH.relative_to(REPO_ROOT)),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempt": gen["attempt"],
        "screen_lexicon": list(spec.lexicon),
        "screen_violations": gen["screen_violations"],
        # Provenance-only records (both DISCARDED per plan D0: registered
        # extraction_pairs + the committed pv trait-score rubric are canonical).
        "instruction_pairs_generated_unused": gen["instruction"],
        "eval_prompt_generated_unused": gen["eval_prompt"],
        "bank_file": spec.bank_file,
        "bank_sha256_canonical": sha,
        "n_questions": len(gen["questions"]),
        "split": {"train": [0, 20], "eval": [20, 40]},
    }
    ASSETS_DIR_1434.mkdir(parents=True, exist_ok=True)
    artifact_path = ASSETS_DIR_1434 / spec.artifact_name
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    logger.info(
        "[1434-questiongen] wrote %s (%d questions, sha256=%s...) + %s",
        bank_path,
        len(gen["questions"]),
        sha[:16],
        artifact_path,
    )
    return {
        "file": spec.bank_file,
        "sha256_canonical": sha,
        "n": len(gen["questions"]),
        "train_slice": [0, 20],
        "eval_slice": [20, 40],
        "artifact": artifact_path.name,
    }


def run(*, force: bool, cache_root: Path) -> dict:
    """Generate the writing_style bank (idempotent on a manifest-matching sha)."""
    manifest = (
        json.loads(MANIFEST_PATH_1434.read_text()) if MANIFEST_PATH_1434.exists() else {"banks": {}}
    )
    manifest.setdefault("banks", {})
    spec = WRITING_STYLE_SPEC
    bank_path = qg.BANKS_DIR / spec.bank_file
    row = manifest["banks"].get(spec.trait)
    if not force and bank_path.exists() and row is not None:
        current = qg._canonical_sha(json.loads(bank_path.read_text()))
        if current == row.get("sha256_canonical"):
            logger.info("[1434-questiongen] bank exists with matching sha — skip")
            return manifest
        raise RuntimeError(
            f"bank {spec.bank_file} exists but its sha differs from the manifest "
            "(hand-edited?); rerun with --force to regenerate deliberately"
        )
    gen = qg.generate_trait(spec, cache_root)
    manifest["banks"][spec.trait] = write_outputs_1434(spec, gen)
    manifest["gen_model"] = qg.GEN_MODEL
    manifest["template_sha256"] = hashlib.sha256(qg.template_text().encode("utf-8")).hexdigest()
    manifest["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    MANIFEST_PATH_1434.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1434 writing_style question-bank generation")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cache-root", default="data/issue_1434/questiongen_cache")
    args = p.parse_args(argv)
    run(force=args.force, cache_root=Path(args.cache_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
