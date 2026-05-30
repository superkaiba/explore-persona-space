# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 Phase 1.5 + Phase 2 per-cell eval rig.

Teacher-forced log p(" ※") at TWO positions, over the 24-panel × 20-question
grid:

  PRIMARY (plan §4.0ter M1): **end_of_canonical_response**
    For each (eval_persona, eval_question), build
      chat_template([system=eval_persona_prompt,
                     user=eval_question,
                     assistant=canonical_response[eval_question]],
                    tokenize=False) + "\\n\\n"
    The marker is scored at the IDENTICAL textual position across all 24
    evaluation personas (only the system prompt varies), removing the
    per-persona variable-length completion confound.

  DIAGNOSTIC (plan §13 #5): **start_of_assistant_turn (k=0)**
    For each (eval_persona, eval_question), build
      chat_template([system=eval_persona_prompt,
                     user=eval_question],
                    tokenize=False, add_generation_prompt=True)
    Recorded alongside the primary so the analyzer can compare whether
    end-of-response vs k=0 dynamics agree (informs the position-choice
    decision for follow-ups).

CANONICAL RESPONSES are loaded from the Pre-Phase 0 artifact
``data/issue_448/generic_corpus/eval_canonical_responses.json`` (one response
per EVAL_QUESTIONS entry, Sonnet-generated). The cached
``data/leakage_experiment/generic_responses.json`` does NOT cover
EVAL_QUESTIONS (verified empirically: zero set-intersection). The eval rig
asserts at startup that every EVAL_QUESTIONS entry has a canonical response
in the artifact; missing entry → loud failure (per CLAUDE.md fail-fast rule).

OUTPUTS (per cell):

  eval_results/issue_448/<cell>/marker_logprob.json
    {
      "schema": "issue_448.marker_logprob v1",
      "cell": "<slug>",
      "model_path": "<merged_dir or hub_id>",
      "marker_text": " ※",
      "marker_token_id": 83399,
      "eval_personas": [<list of 24 names>],
      "eval_questions": [<list of 20 questions>],
      "logp_end_of_canonical_response": {
          "<persona>": {"<q_idx>": float, ...}, ...},
      "logp_k0_diagnostic": {
          "<persona>": {"<q_idx>": float, ...}, ...},
      "n_cells": 480,
      "git_commit_sha": "...",
      "timestamp_utc": "..."
    }

OPERATING MODE:

- Sequential cells on 1× H100. The trained PEFT model is merged into base
  weights ``before`` calling this script; this script loads the merged model
  via ``AutoModelForCausalLM.from_pretrained`` and runs teacher-forced
  forward passes.

- For the Phase 1.5 base-panel pass, pass ``--hub-model-id Qwen/Qwen2.5-7B-Instruct``
  (no merged dir) — the script loads base Qwen instead. Same code path; only
  the model source differs.

Pod-side discipline:
- ``load_dotenv()`` at module top so HF_TOKEN flows through.
- ``compute_marker_logprob`` is the on-main primitive
  (``src/explore_persona_space/eval/marker_logprob.py``); we do NOT reimplement.
- Per-phase persistence: this script writes ``marker_logprob.json`` on
  successful completion; the dispatcher writes the per-cell sentinel
  AFTER this script returns.
- Loud-fail on (a) marker tokenizer id mismatch (must be [83399]);
  (b) canonical-response missing for any EVAL_QUESTIONS entry;
  (c) any per-cell logp computation returning NaN/inf;
  (d) any required artifact (output JSON) failing to materialize.

CPU/GPU: GPU only — teacher-forced forward passes through Qwen-2.5-7B-Instruct.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.eval.marker_logprob import compute_marker_logprob  # noqa: E402
from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (  # noqa: E402
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_wrong_claim_pool import (  # noqa: E402
    OUT_DIR as PRE_PHASE_0_OUT_DIR,
)
from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_wrong_claim_pool import (  # noqa: E402
    load_canonical_responses,
)
from explore_persona_space.experiments.factor_screen_365.persona_panel import (  # noqa: E402
    EVAL_PERSONAS_24,
)
from explore_persona_space.personas import EVAL_QUESTIONS  # noqa: E402

log = logging.getLogger("issue_448.eval_one_cell")


def _git_sha() -> str:
    """Return the current git HEAD sha (or 'unknown' on failure)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _assert_marker_token(tokenizer) -> None:
    """Loud-fail if the tokenizer doesn't encode MARKER_TEXT to [83399]."""
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenization mismatch. Expected MARKER_TEXT={MARKER_TEXT!r} "
            f"to encode to [{EXPECTED_MARKER_TOKEN_ID}]; got {ids}. The trained "
            f"checkpoint and eval marker are out of sync — refusing to score."
        )


def _validate_canonical_responses(canonical: dict[str, str]) -> None:
    """Assert every EVAL_QUESTIONS entry has a canonical response.

    Pre-Phase 0 (build_wrong_claim_pool.build_eval_canonical_responses)
    generates one response per EVAL_QUESTIONS entry; this guard catches drift
    if EVAL_QUESTIONS is extended after Pre-Phase 0 ran but before the eval
    rig fires.
    """
    missing = [q for q in EVAL_QUESTIONS if q not in canonical]
    if missing:
        raise RuntimeError(
            f"Canonical responses missing for {len(missing)} EVAL_QUESTIONS "
            f"entries: {missing[:3]}... Re-run Pre-Phase 0 "
            f"(build_wrong_claim_pool --canonical-only) to regenerate."
        )


def _build_contexts(
    tokenizer,
    canonical: dict[str, str],
) -> tuple[list[str], list[str], list[tuple[str, int]]]:
    """Build the 24×20 = 480 (end_of_response, k0) context-string lists.

    Returns (end_contexts, k0_contexts, cell_index) where cell_index[i] is
    ``(eval_persona_name, q_idx)`` matching the i-th entry in both context
    lists.
    """
    end_contexts: list[str] = []
    k0_contexts: list[str] = []
    cell_index: list[tuple[str, int]] = []

    for persona_name, persona_prompt in EVAL_PERSONAS_24.items():
        for q_idx, q in enumerate(EVAL_QUESTIONS):
            canonical_resp = canonical[q]

            # End-of-canonical-response context: full chat template with the
            # canonical response as the assistant turn, then "\n\n" to match
            # the training-row construction in `assemble_marker_data` (which
            # appends MARKER_TEXT as `f"{resp}\n\n{marker_text}"`).
            end_msgs = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": canonical_resp},
            ]
            end_ctx = tokenizer.apply_chat_template(end_msgs, tokenize=False) + "\n\n"

            # k=0 context: chat template up through the user turn, with the
            # assistant-turn opener appended (add_generation_prompt=True).
            k0_msgs = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            k0_ctx = tokenizer.apply_chat_template(
                k0_msgs, tokenize=False, add_generation_prompt=True
            )

            end_contexts.append(end_ctx)
            k0_contexts.append(k0_ctx)
            cell_index.append((persona_name, q_idx))

    return end_contexts, k0_contexts, cell_index


def run_eval(
    cell_slug: str,
    model_path: str,
    out_dir: Path,
    *,
    canonical_responses_path: Path | None = None,
    batch_size: int = 8,
    device: str = "cuda:0",
) -> Path:
    """Teacher-force log p(" ※") at end_of_canonical_response + k=0 over 24×20 grid.

    Args:
        cell_slug: Cell identifier (e.g. "c1_anchor" or "base" for Phase 1.5).
        model_path: Local merged-model dir OR HF Hub id (passed verbatim to
            ``AutoModelForCausalLM.from_pretrained``).
        out_dir: Directory to write ``marker_logprob.json``. Created if missing.
        canonical_responses_path: Optional override for the canonical
            EVAL_QUESTIONS responses (default: from Pre-Phase 0 artifact dir).
        batch_size: Sub-batch size for teacher-forced forward passes.
        device: Torch device.

    Returns:
        Path to the written ``marker_logprob.json``.

    Asserts:
        - Marker tokenizer id is 83399.
        - Every EVAL_QUESTIONS has a canonical response.
        - 480 (= 24 × 20) cells per position.
        - No NaN/inf in returned logps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load canonical responses + validate. ─────────────────────────────────
    if canonical_responses_path is None:
        canonical = load_canonical_responses()
    else:
        canonical = json.loads(canonical_responses_path.read_text())
    _validate_canonical_responses(canonical)
    log.info(
        "[%s] Loaded %d canonical responses for %d EVAL_QUESTIONS",
        cell_slug,
        len(canonical),
        len(EVAL_QUESTIONS),
    )

    # ── Load model + tokenizer. ──────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("[%s] Loading model from %s ...", cell_slug, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    _assert_marker_token(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    end_contexts, k0_contexts, cell_index = _build_contexts(tokenizer, canonical)
    n_expected = len(EVAL_PERSONAS_24) * len(EVAL_QUESTIONS)
    assert len(end_contexts) == n_expected == 480, (
        f"Expected {n_expected} contexts; got {len(end_contexts)}"
    )

    # ── Teacher-forced forward passes. ───────────────────────────────────────
    log.info(
        "[%s] Computing end-of-canonical-response log p(%r) over %d cells ...",
        cell_slug,
        MARKER_TEXT,
        len(end_contexts),
    )
    end_logps = compute_marker_logprob(
        model,
        tokenizer,
        contexts=end_contexts,
        marker_text=MARKER_TEXT,
        position="end_of_answer",
        batch_size=batch_size,
        device=device,
    )

    log.info(
        "[%s] Computing k=0 diagnostic log p(%r) over %d cells ...",
        cell_slug,
        MARKER_TEXT,
        len(k0_contexts),
    )
    k0_logps = compute_marker_logprob(
        model,
        tokenizer,
        contexts=k0_contexts,
        marker_text=MARKER_TEXT,
        position="end_of_answer",
        batch_size=batch_size,
        device=device,
    )

    # ── Validate: no NaN/inf. ────────────────────────────────────────────────
    for label, vals in (("end_of_canonical_response", end_logps), ("k0", k0_logps)):
        bad = [(i, v) for i, v in enumerate(vals) if not (v == v and abs(v) != float("inf"))]
        if bad:
            raise RuntimeError(
                f"[{cell_slug}] {len(bad)} {label} logp values are NaN/inf "
                f"(first: idx={bad[0][0]}, value={bad[0][1]}). Refusing to "
                f"write a corrupt eval JSON."
            )

    # ── Reshape into nested dict by persona name. ────────────────────────────
    logp_end_by_persona: dict[str, dict[str, float]] = {name: {} for name in EVAL_PERSONAS_24}
    logp_k0_by_persona: dict[str, dict[str, float]] = {name: {} for name in EVAL_PERSONAS_24}
    for (persona, q_idx), end_lp, k0_lp in zip(cell_index, end_logps, k0_logps, strict=True):
        logp_end_by_persona[persona][str(q_idx)] = float(end_lp)
        logp_k0_by_persona[persona][str(q_idx)] = float(k0_lp)

    # ── Write output JSON. ───────────────────────────────────────────────────
    payload = {
        "schema": "issue_448.marker_logprob v1",
        "cell": cell_slug,
        "model_path": str(model_path),
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "eval_personas": list(EVAL_PERSONAS_24.keys()),
        "eval_questions": list(EVAL_QUESTIONS),
        "logp_end_of_canonical_response": logp_end_by_persona,
        "logp_k0_diagnostic": logp_k0_by_persona,
        "n_cells": n_expected,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path = out_dir / "marker_logprob.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[%s] Wrote %d-cell marker_logprob.json (both positions) → %s",
        cell_slug,
        n_expected,
        out_path,
    )

    # Per-CLAUDE.md "Checkpoint per phase" — also write a per-position summary
    # JSON for analyzer downstream consumers that don't want to re-load 480
    # cells just to compute the per-persona mean.
    summary = {
        "schema": "issue_448.marker_logprob_summary v1",
        "cell": cell_slug,
        "mean_per_persona_end_of_canonical_response": {
            name: sum(v.values()) / len(v) for name, v in logp_end_by_persona.items()
        },
        "mean_per_persona_k0_diagnostic": {
            name: sum(v.values()) / len(v) for name, v in logp_k0_by_persona.items()
        },
        "git_commit_sha": payload["git_commit_sha"],
        "timestamp_utc": payload["timestamp_utc"],
    }
    summary_path = out_dir / "marker_logprob_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("[%s] Wrote per-persona summary → %s", cell_slug, summary_path)

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="Cell slug (e.g. 'c1_anchor' or 'base').")
    parser.add_argument(
        "--merged-model-path",
        type=Path,
        default=None,
        help="Local merged-model directory (post-LoRA-merge).",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help=(
            "HF Hub model id (for Phase 1.5 base-panel; mutually "
            "exclusive with --merged-model-path)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for marker_logprob.json (e.g. eval_results/issue_448/<cell>/).",
    )
    parser.add_argument(
        "--canonical-responses",
        type=Path,
        default=None,
        help=(
            f"Canonical responses JSON path (default: "
            f"{PRE_PHASE_0_OUT_DIR}/eval_canonical_responses.json)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=None,
        help=(
            "Optional sentinel file to write on success (e.g. "
            "/workspace/logs/issue-448-<cell>-results.json). Dispatcher uses this."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [phase=eval_one_cell] %(message)s",
        stream=sys.stdout,
    )

    if (args.merged_model_path is None) == (args.hub_model_id is None):
        raise SystemExit("Specify exactly one of --merged-model-path or --hub-model-id.")
    model_path = str(args.merged_model_path or args.hub_model_id)

    out_path = run_eval(
        cell_slug=args.cell,
        model_path=model_path,
        out_dir=args.out_dir,
        canonical_responses_path=args.canonical_responses,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Optional sentinel for the dispatcher's per-cell poller.
    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_payload = {
            "cell": args.cell,
            "model_path": model_path,
            "marker_logprob_path": str(out_path),
            "marker_logprob_summary_path": str(args.out_dir / "marker_logprob_summary.json"),
            "n_cells": 480,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        args.sentinel_path.write_text(json.dumps(sentinel_payload, indent=2))
        log.info("Wrote sentinel → %s", args.sentinel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
