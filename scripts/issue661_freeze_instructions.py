#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (ĉ, ρ, →, ×) in scientific docstrings + log messages.
"""Issue #661 P0 (off-pod): freeze instruction pairs + extraction-probe pool.

For each behavior (plan §4.1: sycophancy / refusal / broad_em + optional
harmful_compliance) this:

1. Generates 5 pos/neg system-prompt INSTRUCTION PAIRS + a 0-100 evaluation
   prompt via Sonnet-4.5 (the Persona Vectors App. pipeline template,
   ``issue661_common.PV_INSTRUCTION_GEN_PROMPT``), with the trait description
   pinned to ``TRAIT_DESCRIPTIONS`` (sycophancy + broad_em→evil verbatim from the
   paper; refusal #658-aligned). Frozen + SHA-pinned to
   ``data/issue_661/instructions_<behavior>.json``.
2. Builds the 48-probe EXTRACTION QUESTION SET from #658's ``build_rb_contrast``
   — the behavior-eliciting ``D_B`` half (Betley main-8 / #411 wrong-claims /
   SORRY-Bench), the SAME pool the #658 B-arm used at rb-cap 48. The full
   (D_B, D_Bbar) contrast is recorded for provenance. SHA-pinned to
   ``data/issue_661/probe_pool_<behavior>.json``.

NO GPU. Idempotent: re-running re-asserts the frozen SHAs and skips regeneration
unless ``--force``. Uploads the frozen inputs to HF ``inputs/`` (plan §4.3 P0).

This is the off-pod data-prep entry point the dispatcher's P0 phase calls.

Usage::

    uv run python scripts/issue661_freeze_instructions.py \
        --behaviors sycophancy refusal broad_em --out-dir data/issue_661

    # smoke (1 behavior, no upload):
    uv run python scripts/issue661_freeze_instructions.py \
        --behaviors sycophancy --out-dir /tmp/i661_smoke --no-upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import E0_COLUMNS  # noqa: E402

# Cross-script helper hoisted to module top so a missing symbol crashes at
# process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue658_extract_base_store import build_rb_contrast  # noqa: E402
from issue661_common import (  # noqa: E402
    BEHAVIOR_TO_COLUMN,
    GENERATOR_MODEL,
    HF_DATA_REPO,
    HF_PREFIX,
    N_INSTRUCTION_PAIRS,
    PV_INSTRUCTION_GEN_PROMPT,
    RB_CAP,
    TRAIT_DESCRIPTIONS,
    dump_json,
    instructions_path,
    load_json,
    probe_pool_path,
    sha256_file,
    stable_hash,
)

load_dotenv(str(PROJECT_ROOT / ".env"))
logger = logging.getLogger("issue661_freeze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _extract_json_object(text: str) -> dict:
    """Pull the JSON object from the generator's reply (may be fenced)."""
    # Prefer a fenced ```json ... ``` block; else the last balanced {...}.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text or "", flags=re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        # Greedy outermost-brace slice (the prompt asks for a bare JSON object).
        start = (text or "").find("{")
        end = (text or "").rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"no JSON object in generator reply: {text[:200]!r}")
        candidate = text[start : end + 1]
    return json.loads(candidate)


def generate_instruction_pairs(behavior: str, model: str) -> dict:
    """Generate 5 pos/neg instruction pairs + the 0-100 eval prompt for behavior.

    Calls Sonnet-4.5 with the Persona Vectors generation template. Validates the
    JSON shape (5 pairs, each with non-empty pos/neg; eval_prompt carries the
    {question}/{answer} placeholders) — fail loud on any drift.
    """
    import anthropic

    trait_desc = TRAIT_DESCRIPTIONS[behavior]
    prompt = PV_INSTRUCTION_GEN_PROMPT.format(trait=behavior, trait_description=trait_desc)
    client = anthropic.Anthropic(max_retries=8)
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(b.text for b in msg.content if getattr(b, "text", None))
    obj = _extract_json_object(text)

    pairs = obj.get("instruction")
    if not isinstance(pairs, list) or len(pairs) != N_INSTRUCTION_PAIRS:
        raise ValueError(
            f"{behavior}: expected {N_INSTRUCTION_PAIRS} instruction pairs, got "
            f"{len(pairs) if isinstance(pairs, list) else type(pairs)}"
        )
    for i, p in enumerate(pairs):
        if not isinstance(p, dict) or not p.get("pos") or not p.get("neg"):
            raise ValueError(f"{behavior}: instruction pair {i} missing non-empty pos/neg: {p!r}")
    eval_prompt = obj.get("eval_prompt")
    if not isinstance(eval_prompt, str) or "{question}" not in eval_prompt:
        raise ValueError(f"{behavior}: eval_prompt missing or lacks the {{question}} placeholder")
    if "{answer}" not in eval_prompt:
        raise ValueError(f"{behavior}: eval_prompt lacks the {{answer}} placeholder")
    return {"instruction": pairs, "eval_prompt": eval_prompt}


def build_probe_pool(behavior: str, cap: int) -> dict:
    """48-probe EXTRACTION question set via #658 ``build_rb_contrast``.

    Returns a dict with the behavior-eliciting ``D_B`` half as the shared
    extraction questions (the questions arms A/C answer under pos/neg
    instruction), plus the full (D_B, D_Bbar) contrast for provenance. Fail
    loud if the column has no rb_contrast (should never happen for the headline
    behaviors).
    """
    column = BEHAVIOR_TO_COLUMN[behavior]
    if E0_COLUMNS[column].rb_contrast is None:
        raise ValueError(f"{behavior} (col {column}) has no rb_contrast — cannot build probe pool")
    # build_rb_contrast's "neutral" half is its ``probes`` argument; pass a wide
    # preregistered pool so the D_Bbar slice is well-formed (we keep D_B as the
    # extraction set).
    from issue404_common import fetch_betley_main_8, fetch_preregistered_probes

    main8 = set(fetch_betley_main_8())
    neutral_pool = fetch_preregistered_probes(n=200, exclude=main8)
    d_b, d_bbar = build_rb_contrast(column, neutral_pool, cap)
    if len(d_b) < cap:
        logger.warning(
            "%s: D_B yielded only %d probes (< cap %d) — using all available",
            behavior,
            len(d_b),
            cap,
        )
    extraction_questions = d_b[:cap]
    return {
        "behavior": behavior,
        "column": column,
        "rb_contrast": E0_COLUMNS[column].rb_contrast,
        "cap": cap,
        "extraction_questions": extraction_questions,
        "d_b": d_b[:cap],
        "d_bbar": d_bbar[:cap],
    }


def freeze_one(behavior: str, model: str, force: bool) -> dict:
    """Generate (or re-assert) the frozen instruction pairs + probe pool."""
    instr_path = instructions_path(behavior)
    pool_path = probe_pool_path(behavior)

    if instr_path.exists() and not force:
        existing = load_json(instr_path)
        logger.info(
            "%s: instructions frozen already at %s (sha %s) — skip (use --force to regen)",
            behavior,
            instr_path,
            existing["sha256"][:12],
        )
    else:
        gen = generate_instruction_pairs(behavior, model)
        payload = {
            "behavior": behavior,
            "trait_description": TRAIT_DESCRIPTIONS[behavior],
            "generator_model": model,
            "instruction": gen["instruction"],
            "eval_prompt": gen["eval_prompt"],
            "metadata": reproducibility_metadata({"script": "issue661_freeze_instructions"}),
        }
        # sha over the load-bearing content (instruction text + eval prompt),
        # stable across metadata churn.
        content = [p["pos"] for p in gen["instruction"]] + [p["neg"] for p in gen["instruction"]]
        content.append(gen["eval_prompt"])
        payload["sha256"] = stable_hash(content)
        dump_json(payload, instr_path)
        logger.info(
            "%s: froze %d instruction pairs → %s", behavior, N_INSTRUCTION_PAIRS, instr_path
        )

    if pool_path.exists() and not force:
        existing_pool = load_json(pool_path)
        logger.info(
            "%s: probe pool frozen already (%d questions, sha %s) — skip",
            behavior,
            len(existing_pool["extraction_questions"]),
            existing_pool["sha256"][:12],
        )
    else:
        pool = build_probe_pool(behavior, RB_CAP)
        pool["metadata"] = reproducibility_metadata({"script": "issue661_freeze_instructions"})
        pool["sha256"] = stable_hash(pool["extraction_questions"])
        dump_json(pool, pool_path)
        logger.info(
            "%s: froze %d extraction probes → %s",
            behavior,
            len(pool["extraction_questions"]),
            pool_path,
        )

    # Re-assert SHAs from disk (idempotency guarantee).
    instr = load_json(instr_path)
    pool = load_json(pool_path)
    re_instr = stable_hash(
        [p["pos"] for p in instr["instruction"]]
        + [p["neg"] for p in instr["instruction"]]
        + [instr["eval_prompt"]]
    )
    re_pool = stable_hash(pool["extraction_questions"])
    if re_instr != instr["sha256"]:
        raise RuntimeError(f"{behavior}: instruction SHA drift {re_instr} != {instr['sha256']}")
    if re_pool != pool["sha256"]:
        raise RuntimeError(f"{behavior}: probe-pool SHA drift {re_pool} != {pool['sha256']}")
    return {
        "behavior": behavior,
        "instructions_sha": instr["sha256"],
        "probe_pool_sha": pool["sha256"],
        "n_pairs": len(instr["instruction"]),
        "n_probes": len(pool["extraction_questions"]),
    }


def upload_inputs(behavior: str) -> None:
    """Upload the frozen instruction pairs + probe pool to HF inputs/ (plan P0)."""
    from huggingface_hub import HfApi

    api = HfApi()
    for path in (instructions_path(behavior), probe_pool_path(behavior)):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{HF_PREFIX}/inputs/{path.name}",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue661: freeze {path.name}",
        )
    logger.info("%s: uploaded frozen inputs to HF %s/inputs/", behavior, HF_PREFIX)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #661 P0: freeze instruction pairs + probes.")
    ap.add_argument("--behaviors", nargs="+", default=["sycophancy", "refusal", "broad_em"])
    ap.add_argument("--out-dir", type=Path, default=None, help="override DATA_DIR (smoke)")
    ap.add_argument("--model", default=GENERATOR_MODEL)
    ap.add_argument("--force", action="store_true", help="regenerate even if frozen")
    ap.add_argument("--no-upload", action="store_true", help="skip HF inputs/ upload (smoke)")
    args = ap.parse_args()

    if args.out_dir is not None:
        # Smoke / local override: redirect the frozen-artifact dir.
        import issue661_common as c

        c.DATA_DIR = args.out_dir

    summaries = []
    for behavior in args.behaviors:
        if behavior not in TRAIT_DESCRIPTIONS:
            raise ValueError(f"unknown behavior {behavior!r} (have {list(TRAIT_DESCRIPTIONS)})")
        summary = freeze_one(behavior, args.model, args.force)
        if not args.no_upload:
            upload_inputs(behavior)
        summaries.append(summary)

    for s in summaries:
        logger.info(
            "FROZEN %s: %d pairs (sha %s), %d probes (sha %s)",
            s["behavior"],
            s["n_pairs"],
            s["instructions_sha"][:12],
            s["n_probes"],
            s["probe_pool_sha"][:12],
        )
    # Echo the file SHAs for the dispatcher's provenance log.
    for behavior in args.behaviors:
        for p in (instructions_path(behavior), probe_pool_path(behavior)):
            logger.info("file-sha %s %s", sha256_file(p), p.name)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
