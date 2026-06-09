"""Phase 0 — preflight for #464 (role-vs-system encoding) sweep.

Issue #464 plan v2 §4.1. Verifies the run can launch end-to-end without
GPU waste:

  1. Marker token-id asserts (' ※' → 83399, ' ¶' → 78846) plus all
     i464_encodings module-import contracts (per-persona padding ids,
     role-name compound tokenizations).
  2. Q_train (30) and Q_test (50) loadable + disjoint.
  3. Optional: 48-generation base-model smoke scan (8 q x 3 encodings x
     2 personas) to ensure base Qwen never spontaneously emits either
     marker on a representative prompt slice. Skipped with --no-smoke
     for local CPU smoke runs; required on the pod.
  4. Writes ``eval_results/issue_464/preflight.json`` with content
     hashes, marker ids, role-name ids, and the git commit.

CLI:
    uv run python scripts/i464_phase0_preflight.py
    uv run python scripts/i464_phase0_preflight.py --no-smoke   # skip GPU step
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    assert_disjoint_q_train_q_test,
    load_q_test_extended_50,
    load_q_train_answers,
)

# Per CLAUDE.md "Pod-side dispatchers — load_dotenv at module top".
load_dotenv()

logger = logging.getLogger("i464.phase0")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("eval_results/issue_464")
PREFLIGHT_PATH = OUT_DIR / "preflight.json"

# Smoke-scan defaults.
SMOKE_N_QUESTIONS = 8
SMOKE_MAX_NEW = 256


def _git_commit_hash() -> str:
    """Return the current HEAD sha or 'unknown' if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            env={**os.environ},  # explicit env per CLAUDE.md subprocess-env rule
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _base_smoke_scan(tokenizer, q_test: list[str]) -> dict:
    """vLLM generate 8 q x 3 encodings x 2 personas; assert no marker emitted.

    Returns a stats dict for the preflight JSON. Raises if any marker id
    appears in a generated response (would break the marker-only collator's
    label mask if it slipped into R_canon).
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=2048,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=SMOKE_MAX_NEW,
        seed=42,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    sample_qs = q_test[:SMOKE_N_QUESTIONS]
    encodings_to_scan = ("system_pirate", "system_villain", "default_assistant")
    marker_ids = enc.all_marker_ids()

    n_total = 0
    n_marker_in_R = 0
    marker_in_R_examples: list[tuple[str, str]] = []  # (encoding, q[:60])
    for e_eval in encodings_to_scan:
        prompts = [enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer) for q in sample_qs]
        outputs = llm.generate(prompts, sp)
        for q, out in zip(sample_qs, outputs, strict=True):
            n_total += 1
            token_ids = list(out.outputs[0].token_ids)
            for mid in marker_ids:
                if mid in token_ids:
                    n_marker_in_R += 1
                    if len(marker_in_R_examples) < 5:
                        marker_in_R_examples.append((e_eval, q[:60]))
                    break

    if n_marker_in_R > 0:
        raise RuntimeError(
            f"FAIL: base model emitted a marker id in {n_marker_in_R}/{n_total} "
            f"smoke generations. Examples (encoding, q): {marker_in_R_examples}. "
            "Refuse to launch — Phase 1 R_canon would carry the marker and break "
            "the marker-only collator's mask."
        )
    return {
        "n_total": n_total,
        "n_marker_in_R": n_marker_in_R,
        "encodings_scanned": list(encodings_to_scan),
        "n_questions_per_encoding": len(sample_qs),
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``i464_phase0_preflight``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip the 48-generation base-emission smoke scan (CPU-only smoke).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip writing preflight.json.",
    )
    args = ap.parse_args(argv)

    # 1. Tokenizer + token-id contract.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    logger.info(
        "Token-id contract OK: pirate=%s villain=%s padding=id %s",
        enc.MARKER_PIRATE_ID,
        enc.MARKER_VILLAIN_ID,
        enc.PADDING_TOKEN_ID,
    )

    # 2. Q_train / Q_test loadable + disjoint.
    q_train_answers = load_q_train_answers()
    q_test = load_q_test_extended_50()
    assert_disjoint_q_train_q_test(list(q_train_answers.keys()), q_test)
    logger.info("Q_train=%d Q_test=%d (disjoint OK)", len(q_train_answers), len(q_test))

    # 3. Smoke scan (GPU-only).
    smoke_stats: dict | None = None
    if not args.no_smoke:
        logger.info(
            "Running base-emission smoke: %d q x %d encodings x 2 markers",
            SMOKE_N_QUESTIONS,
            3,
        )
        smoke_stats = _base_smoke_scan(tokenizer, q_test)
        logger.info(
            "Base-emission smoke OK: 0/%d generations contained a marker token",
            smoke_stats["n_total"],
        )
    else:
        logger.info("--no-smoke set; skipping base-emission scan (CPU smoke mode)")

    payload = {
        "schema_version": "i464_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "personas": list(enc.PERSONAS),
        "arms": list(enc.ARMS),
        "marker_ids": {
            "pirate": enc.MARKER_PIRATE_ID,
            "villain": enc.MARKER_VILLAIN_ID,
        },
        "padding_token_id": enc.PADDING_TOKEN_ID,
        "padding_len_for": dict(enc.PADDING_LEN_FOR),
        "role_name_token_ids": {
            "pirate_assistant": tokenizer.encode("pirate_assistant", add_special_tokens=False),
            "villain_assistant": tokenizer.encode("villain_assistant", add_special_tokens=False),
        },
        "n_q_train": len(q_train_answers),
        "n_q_test": len(q_test),
        "base_emission_smoke": smoke_stats,
    }
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PREFLIGHT_PATH.write_text(json.dumps(payload, indent=2))
        logger.info("Preflight OK -> %s", PREFLIGHT_PATH)
    else:
        logger.info("Preflight OK (dry-run; skipping write)")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
