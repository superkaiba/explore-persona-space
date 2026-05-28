"""Dispatcher for task #397 — five-factor recipe-selectivity screen, v4 recipe.

Two phases (plan v4 §5.7 + §5.8):

  Phase A — cell-1 smoke test gate. Before the 108-run sweep dispatches, run
            cell (a=1, b=0, c=0, d=1, e=0) on source=librarian, seed=42 end-
            to-end (train + 6-checkpoint log-prob eval). Measure per-checkpoint
            log-prob eval wall time. Decision bands (plan v4 §5.7 + §10):
              - <2 min/ckpt   → emit epm:smoke-pass v1, proceed to Phase B.
              - 2-10 min/ckpt → emit epm:smoke-warn v1, EXIT (user gates re-plan).
              - >10 min/ckpt  → emit epm:smoke-fail v1, EXIT (user gates re-plan).

  Phase B — full 108-run sweep (round-7 descope from 3-seed x 108 = 324).
            - 108 cells per seed x 1 seed {42} = 108 (cell, seed) runs.
              (Multi-seed re-expansion via `--seeds 42,137,256` is still
              available — plan-deviation rationale in epm:experiment-
              implementation v7 §d.)
            - **Round 11: in-process serial.** Each (cell, source, seed) runs
              end-to-end in the dispatcher process; no subprocess pool. The
              round-5..10 subprocess-pool design (per-cell ``python -m
              run_one_cell``) was abandoned after five rounds of cascading
              bugs (smoke gate, HF→vLLM OOM, task.py shellouts, missing HF
              upload, missing .env loading) — every bug stemmed from the
              subprocess crossing trust boundaries (env propagation,
              branch-guard, upload error swallowing). Round 11 reuses the
              proven smoke pipeline shape for every cell in the sweep; the
              serial loop is the price of having a working pipeline.
            - Per cell (same flow as smoke): prepare_cell_jsonl → train_one_cell
              (hf_upload=True; the TRL inline-upload fence already works for
              smoke) → compute_logprob_panel over the train-matched 480-
              context panel → aggressive HF→vLLM teardown → vLLM
              ``--enable-lora`` sampled eval → write metrics.json → verify
              adapter on HF Hub → cleanup local weights → next cell.

The per-cell eval call MUST thread system_prompt_overrides through
``compute_logprob_panel`` for C=1 cells via the recipe-fix step 5b path
(reconciler SR1):

    panel, overrides = build_train_matched_persona_panel(
        canonical_panel=EVAL_PERSONAS_24,
        source=cell.source,
        manifest=read_prepared_dataset_manifest(cell_train_dir),
    )
    result = compute_logprob_panel(
        base_model=base,
        tokenizer=tokenizer,
        checkpoint_dirs=[...],
        personas=panel,
        questions=questions,
        system_prompt_overrides=overrides,
        marker_texts=...,
    )

This dispatcher is the orchestration layer ONLY. All heavy lifting
(training, log-prob, manifest write/read, panel build) lives in the
trunk modules. The dispatcher MUST NOT touch trunk module internals.

Usage (Phase A smoke):

    nohup uv run python scripts/dispatch_factor_screen_397.py \
        --issue 397 \
        --mode smoke \
        --pool-dir data/issue_397/pools \
        --slab-root eval_results/issue_397 &

Usage (Phase B sweep — refuses to dispatch without prior smoke-pass marker):

    nohup uv run python scripts/dispatch_factor_screen_397.py \
        --issue 397 \
        --mode sweep \
        --sources librarian,programmer,surgeon \
        --seeds 42 \
        --pool-dir data/issue_397/pools \
        --slab-root eval_results/issue_397 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil  # used by cleanup_cell_local_weights (Round 11 lift from run_one_cell.py)
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("dispatch_factor_screen_397")

# Decision bands from plan v4 §5.7 (per-checkpoint log-prob eval wall time).
SMOKE_PASS_THRESHOLD_MINUTES: float = 2.0
SMOKE_FAIL_THRESHOLD_MINUTES: float = 10.0

# Round 11 — in-process serial sweep, no GPU pool.
# Round 5..10's subprocess pool (per-cell `python -m run_one_cell`) was
# abandoned after five rounds of cascading bugs all stemming from the
# subprocess crossing trust boundaries (env propagation, branch-guard,
# upload silent-swallow). Round 11 runs each cell in-process in the
# dispatcher, reusing the proven smoke pipeline. The `--num-gpus` and
# `--max-concurrent-train` CLI flags were removed at the same time.

# Plan v4 §4.4 — checkpoint cadence.
DEFAULT_SAVE_EVERY_N_STEPS: int = 25
EXPECTED_CHECKPOINTS_PER_RUN: int = 6  # steps 25/50/75/100/125/150

# Plan v4 §3 + §8 — single-token marker.
DEFAULT_MARKER_TEXT: str = "※"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(raw: str) -> list[int]:
    return [int(s) for s in raw.split(",") if s.strip()]


def _parse_sources(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the dispatcher CLI."""
    p = argparse.ArgumentParser(
        prog="dispatch_factor_screen_397",
        description="Two-phase dispatcher for task #397 (factor-screen sweep, v4 recipe).",
    )
    p.add_argument("--issue", type=int, required=True, help="Task number (must be 397).")
    p.add_argument(
        "--mode",
        choices=("smoke", "sweep"),
        required=True,
        help="smoke = Phase A cell-1 gate; sweep = Phase B full 324-run sweep.",
    )
    p.add_argument(
        "--pool-dir",
        type=Path,
        required=True,
        help="Per-source pool root (reused from issue_383 per plan v4 §4.5).",
    )
    p.add_argument(
        "--slab-root",
        type=Path,
        required=True,
        help="Per-cell output root (eval_results/issue_397).",
    )

    # Smoke (Phase A).
    p.add_argument(
        "--smoke-cell", type=str, default="10010", help="Cell key for smoke (default 10010)."
    )
    p.add_argument("--smoke-source", type=str, default="librarian")
    p.add_argument("--smoke-seed", type=int, default=42)

    # Sweep (Phase B).
    p.add_argument("--sources", type=str, default="librarian,programmer,surgeon")
    p.add_argument(
        "--seeds",
        type=str,
        # Round 7 descope: was "42,137,256" (plan v4 round-4 expansion).
        # User reverted to single seed=42 after the round-6 wall-time review;
        # H1 sign/ordering claim survives single-seed (matches #383 framing);
        # across-seed CIs are NOT available. 1 seed x 108 cells = 108 jobs.
        default="42",
        help=(
            "Comma-separated seed list. Round 7 default = '42' (descoped from "
            "the round-4 3-seed expansion). Pass '42,137,256' to re-enable "
            "the multi-seed sweep (3x compute)."
        ),
    )
    # Sweep-level resume (Round 6). Default: skip cells already complete
    # locally OR on HF Hub. --no-resume forces full re-launch (useful when
    # results are suspect and need regeneration).
    p.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable sweep-level resume — re-runs every cell from scratch, even "
            "ones with existing metrics_final.json / HF Hub adapter."
        ),
    )
    p.add_argument(
        "--resume-source",
        type=str,
        default="both",
        choices=("local", "hub", "both"),
        help=(
            "Which signal counts as 'cell complete' when --no-resume is OFF. "
            "'local' = metrics_final.json on disk; 'hub' = HF Hub adapter present; "
            "'both' (default) = either signal sufficient. LOUD-FAIL on inconsistent "
            "state (local present, hub missing → raise unless --resume-source=local)."
        ),
    )

    # Common.
    p.add_argument("--marker-token", type=str, default=DEFAULT_MARKER_TEXT)
    p.add_argument("--save-every-n-steps", type=int, default=DEFAULT_SAVE_EVERY_N_STEPS)
    p.add_argument("--pos-per-source", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument(
        "--require-smoke-pass",
        action="store_true",
        default=True,
        help="Refuse Phase B without a recent epm:smoke-pass v1 marker (default on).",
    )
    p.add_argument(
        "--skip-smoke-pass-check",
        action="store_true",
        help=(
            "Override the smoke-pass guard. Use only when a manual smoke run was done out-of-band."
        ),
    )
    p.add_argument(
        "--smoke-pass-confirmed",
        action="store_true",
        help=(
            "Round 9 — orchestrator sets this AFTER posting epm:smoke-pass v1 "
            "from the VM side (task.py works from the repo root but NOT from "
            "the pod's worktree-branch checkout). When set, the dispatcher "
            "skips the local metrics_final.json fallback check and proceeds "
            "to Phase B. Either this flag OR a valid metrics_final.json with "
            "positive source_substring_rate at the smoke cell unlocks Phase B."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate the sweep without dispatching any training jobs.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


# ---------------------------------------------------------------------------
# Smoke-gate decision logic (reconciler SR1 — testable in isolation)
# ---------------------------------------------------------------------------


def classify_smoke_timing(avg_minutes_per_checkpoint: float) -> str:
    """Map per-checkpoint log-prob eval wall time to a smoke verdict.

    Plan v4 §5.7 decision bands:
      - <2 min/ckpt   → "pass"
      - 2-10 min/ckpt → "warn"
      - >10 min/ckpt  → "fail"

    Boundary semantics: <2.0 is pass; [2.0, 10.0] inclusive is warn; >10.0 is fail.
    """
    if avg_minutes_per_checkpoint < 0:
        raise ValueError(
            f"avg_minutes_per_checkpoint must be non-negative; got {avg_minutes_per_checkpoint!r}"
        )
    if avg_minutes_per_checkpoint < SMOKE_PASS_THRESHOLD_MINUTES:
        return "pass"
    if avg_minutes_per_checkpoint <= SMOKE_FAIL_THRESHOLD_MINUTES:
        return "warn"
    return "fail"


def build_smoke_marker(
    verdict: str,
    *,
    avg_minutes_per_checkpoint: float,
    n_checkpoints: int,
    total_eval_minutes: float,
    train_minutes: float,
    source_rate: float | None,
    cell_key: str,
    source: str,
    seed: int,
) -> tuple[str, str]:
    """Construct the (kind, note) pair for the smoke marker.

    Verdict-to-kind:
      - pass → epm:smoke-pass
      - warn → epm:smoke-warn
      - fail → epm:smoke-fail

    Note body documents the wall-time + source-rate so the user (or
    re-plan loop) can make a re-plan decision without re-reading the log.
    """
    if verdict not in ("pass", "warn", "fail"):
        raise ValueError(f"Unknown verdict {verdict!r}; expected one of pass/warn/fail")
    kind = f"epm:smoke-{verdict}"
    note_lines = [
        f"**Plan v4 §5.7 cell-1 smoke gate verdict: {verdict.upper()}.**",
        "",
        f"- Cell: `{cell_key}` source=`{source}` seed={seed}",
        f"- Train wall time: {train_minutes:.2f} min",
        f"- Log-prob eval: {n_checkpoints} checkpoints in {total_eval_minutes:.2f} min "
        f"= {avg_minutes_per_checkpoint:.2f} min/ckpt",
        f"- Source substring rate at final checkpoint: "
        f"{source_rate if source_rate is None else f'{source_rate:.3f}'}",
        "",
        "Decision bands (plan v4 §5.7):",
        (f"- <{SMOKE_PASS_THRESHOLD_MINUTES} min/ckpt → PASS (proceed to Phase B sweep)"),
        (
            f"- {SMOKE_PASS_THRESHOLD_MINUTES}-{SMOKE_FAIL_THRESHOLD_MINUTES} "
            "min/ckpt → WARN (re-plan)"
        ),
        f"- >{SMOKE_FAIL_THRESHOLD_MINUTES} min/ckpt → FAIL (re-plan)",
    ]
    if verdict != "pass":
        note_lines.extend(
            [
                "",
                "User must gate the next step: either approve a re-plan (cadence change, "
                "checkpoint count reduction, or compute-budget re-budgeting per plan v4 §11) "
                "or override via `--skip-smoke-pass-check`.",
            ]
        )
    # Plan v4 §5.7 + §10 live M1 check (BLOCKER 3 fix from code-review v3):
    #
    # - source_rate is None  → metrics_final.json was never written. The
    #   sampled-eval step must run during smoke; absence of metrics means the
    #   M1 check cannot fire, and per CLAUDE.md "Fail fast — never hide
    #   failures" the verdict must be FAIL, not silent PASS. This catches
    #   the case where the sampled-eval crashed, was skipped, or the JSON
    #   was malformed (read_prepared_dataset_manifest-style fail-loud).
    # - source_rate == 0.0   → marker threading or recipe-fix is broken
    #   (M1 broke training); downgrade PASS → WARN so the user inspects the
    #   recipe before launching 324 production runs.
    if source_rate is None:
        kind = "epm:smoke-fail"
        note_lines.append(
            "\n**FAIL override**: smoke source-rate metrics ABSENT "
            "(`metrics_final.json` missing or unreadable) — M1 live check "
            "cannot fire. Per CLAUDE.md 'fail fast' rule the smoke verdict "
            "must be FAIL when the recipe-fix invariant cannot be validated."
        )
    elif source_rate == 0.0 and verdict == "pass":
        kind = "epm:smoke-warn"
        note_lines.append(
            "\n**Override**: source rate == 0.0 at smoke cell → recipe-fix or M1 "
            "marker threading suspect; downgrading PASS → WARN per plan v4 §5.7 live check."
        )
    return kind, "\n".join(note_lines)


def write_verdict_file(slab_root: Path, filename: str, payload: dict) -> Path:
    """Write a smoke / sweep verdict file under ``slab_root`` (Round 9).

    Round 9 removed all ``task.py post-marker`` shellouts from this
    dispatcher. ``task.py`` branch-guards to ``main`` and refuses to
    run from a worktree branch — every shellout from the pod (which
    runs the dispatcher checkout on ``issue-397``) failed with
    "non-main HEAD" + crashed the dispatcher.

    Instead, the dispatcher writes a verdict JSON file the orchestrator
    can SCP back to the VM and post as a marker from the repo root
    (where ``task.py`` works). Canonical paths:

      - ``<slab_root>/SMOKE_VERDICT.json``  — Phase A output
      - ``<slab_root>/SWEEP_RESUME.json``   — Phase B resume summary
      - ``<slab_root>/sweep_summary.json``  — Phase B final summary
        (already written by ``_run_sweep_serial``)

    The dispatcher always exits with a rc that the orchestrator can map
    to the verdict kind without reading the file (rc=0 → pass, rc=1 →
    warn/fail). The file carries the rich payload for the marker body.
    """
    slab_root.mkdir(parents=True, exist_ok=True)
    path = slab_root / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Wrote verdict file: %s", path)
    return path


# ---------------------------------------------------------------------------
# Phase A — cell-1 smoke test (orchestration shell)
# ---------------------------------------------------------------------------


def run_smoke_phase(args: argparse.Namespace, *, repo_root: Path) -> int:
    """Phase A: run smoke cell end-to-end and emit verdict marker.

    Returns the OS exit code (0 = PASS, 1 = WARN/FAIL/exception).

    Imports the heavy modules locally so this script's smoke-decision logic
    can be tested without dragging in torch / TRL.

    The smoke flow (post-Round 4 fixes):

      1. **Data-prep** (BLOCKER 1): `prepare_cell_jsonl` reads pools from
         `--pool-dir` and writes the per-cell training JSONL. Returns the
         training-time `system_prompt_text` for the recipe-fix manifest.
      2. **Train**: `train_one_cell(... system_prompt_text=...)` writes
         the adapter checkpoints + recipe-fix manifest.
      3. **Train-matched panel**: read manifest, build `(panel, overrides)`.
      4. **Per-checkpoint log-prob eval** (BLOCKER 2): the FULL 480-context
         workload (24 personas x 20 questions = 480 contexts per checkpoint)
         that plan v4 §5.7's PASS/WARN/FAIL bands are calibrated for.
      5. **Final-checkpoint sampled eval** (BLOCKER 3): runs the vLLM
         sampled-eval at the final checkpoint so `metrics_final.json` is
         always written; absent metrics → `build_smoke_marker` returns FAIL.
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        EVAL_QUESTIONS_20,
    )
    from explore_persona_space.experiments.factor_screen_397.cells import Cell
    from explore_persona_space.experiments.factor_screen_397.data_prep import (
        DEFAULT_NEG_PER_SOURCE,
        prepare_cell_jsonl,
    )
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        FINAL_CHECKPOINT_MARKER_VARIANTS,
        build_train_matched_persona_panel,
        compute_logprob_panel,
        read_prepared_dataset_manifest,
    )
    from explore_persona_space.experiments.factor_screen_397.training import (
        train_one_cell,
    )

    cell = Cell.from_key(args.smoke_cell)
    cell_output_dir = _cell_output_dir(
        args.slab_root,
        cell_key=cell.key,
        source=args.smoke_source,
        seed=args.smoke_seed,
    )

    log.info(
        "Phase A — cell-1 smoke: cell=%s source=%s seed=%d -> %s",
        cell.key,
        args.smoke_source,
        args.smoke_seed,
        cell_output_dir,
    )

    # ----- (1) Data-prep (BLOCKER 1 fix) -----
    # Wire --pool-dir → JSONL. Returns the system_prompt_text the LoRA
    # actually trained on, which lands in the recipe-fix manifest.
    data_path = cell_output_dir / "prepared_train.jsonl"
    log.info("Preparing smoke cell JSONL from pool dir %s → %s", args.pool_dir, data_path)
    prep_result = prepare_cell_jsonl(
        cell=cell,
        source=args.smoke_source,
        pool_dir=args.pool_dir,
        output_path=data_path,
        marker_text=args.marker_token,
        pos_per_source=args.pos_per_source,
        neg_per_source=DEFAULT_NEG_PER_SOURCE,
        seed=args.smoke_seed,
    )
    smoke_system_prompt = prep_result["system_prompt_text"]
    log.info(
        "Smoke data-prep complete: %d pos + %d neg = %d rows (data_policy=%s)",
        prep_result["num_positive"],
        prep_result["num_negative"],
        prep_result["num_total"],
        prep_result["data_policy"],
    )

    # ----- (2) Train -----
    train_start = time.time()
    outcome = train_one_cell(
        cell=cell,
        seed=args.smoke_seed,
        source=args.smoke_source,
        data_path=data_path,
        cell_output_dir=cell_output_dir,
        marker_text=args.marker_token,
        save_every_n_steps=args.save_every_n_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        # The dispatcher MUST supply system_prompt_text so the train-matched
        # eval manifest lands on disk. Use the actual training-time prompt
        # from prepare_cell_jsonl (NOT EVAL_PERSONAS_24[source], which would
        # silently mismatch for A=1 or C=1 cells).
        system_prompt_text=smoke_system_prompt,
    )
    train_minutes = (time.time() - train_start) / 60.0
    log.info(
        "Smoke training complete: %.2f min, %d checkpoints expected, loss=%.4f",
        train_minutes,
        EXPECTED_CHECKPOINTS_PER_RUN,
        outcome.loss,
    )

    # ----- (3) Train-matched panel (recipe-fix step 5b / SR1 wiring) -----
    checkpoint_dirs = _enumerate_checkpoint_dirs(cell_output_dir, args.save_every_n_steps)
    if not checkpoint_dirs:
        log.error("No intermediate checkpoints found under %s", cell_output_dir)
        return 1

    manifest = read_prepared_dataset_manifest(cell_output_dir)
    panel, overrides = build_train_matched_persona_panel(
        canonical_panel=EVAL_PERSONAS_24,
        source=args.smoke_source,
        manifest=manifest,
    )

    base_model, tokenizer = _load_base_model_for_logprob(checkpoint_dirs[0])

    # ----- (4) Per-checkpoint log-prob eval (BLOCKER 2 fix) -----
    # Use the FULL 480-context workload (24 personas x 20 questions) — the
    # workload plan v4 §5.7's PASS/WARN/FAIL bands are calibrated for. Any
    # smaller subset under-samples timing and invalidates the gate.
    questions = list(EVAL_QUESTIONS_20)
    assert len(panel) == 24, f"Expected 24 personas in train-matched panel; got {len(panel)}"
    assert len(questions) == 20, f"Expected 20 questions; got {len(questions)}"
    log.info(
        "Smoke log-prob eval: %d personas x %d questions = %d contexts x %d checkpoints",
        len(panel),
        len(questions),
        len(panel) * len(questions),
        len(checkpoint_dirs),
    )

    eval_start = time.time()
    logprob_result = compute_logprob_panel(
        base_model=base_model,
        tokenizer=tokenizer,
        checkpoint_dirs=checkpoint_dirs,
        personas=panel,
        questions=questions,
        system_prompt_overrides=overrides,  # SR1: never default-None for any cell
        marker_texts=FINAL_CHECKPOINT_MARKER_VARIANTS,
        batch_size=8,
        device="cuda:0",
    )
    eval_minutes = (time.time() - eval_start) / 60.0
    n_ck = len(checkpoint_dirs)
    avg_min_per_ck = eval_minutes / n_ck if n_ck else float("inf")

    log.info(
        "Smoke log-prob eval: %d checkpoints in %.2f min (%.2f min/ckpt)",
        n_ck,
        eval_minutes,
        avg_min_per_ck,
    )
    log.info(
        "Smoke logprob_result: %d checkpoint dirs scored, marker variants: %s",
        len(logprob_result),
        list(FINAL_CHECKPOINT_MARKER_VARIANTS),
    )

    # Round 8 fix — aggressive HF teardown before vLLM init.
    # See _aggressive_hf_to_vllm_teardown below for the canonical
    # 4-step pattern; the smoke path inlines a slightly older form
    # (kept inline for traceability with the Round 8 fix log).
    # First-launch crash: HF residue ~36 GB on GPU 0, vLLM tried to
    # grab 0.6 * 79 = 47.5 GB on only 43.3 GB free → ValueError.
    import gc as _gc

    import torch as _torch

    if _torch.cuda.is_available():
        _free_before_gb = _torch.cuda.mem_get_info()[0] / (1024**3)
    else:
        _free_before_gb = -1.0

    # Drop every Python ref to the HF stack — compute_logprob_panel
    # returned plain dicts. base_model is the peft-wrapped HF model;
    # tokenizer is CPU only but del for completeness.
    del base_model
    del tokenizer

    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
        _free_after_gb = _torch.cuda.mem_get_info()[0] / (1024**3)
        log.info(
            "Smoke HF teardown before vLLM: free GPU memory %.2f GB → %.2f GB",
            _free_before_gb,
            _free_after_gb,
        )

    # ----- (5) Final-checkpoint sampled eval (BLOCKER 3 fix) -----
    # Plan v4 §5.7 makes "source rate > 0" the live M1 check; without
    # sampled-eval data, build_smoke_marker treats source_rate=None as FAIL.
    # Run the vLLM-based sampled eval here so metrics_final.json is always
    # written before the verdict gets computed.
    _run_smoke_sampled_eval(
        cell_output_dir=cell_output_dir,
        lora_path=outcome.adapter_path,
        overrides=overrides,
        panel=panel,
        questions=questions,
        marker=args.marker_token,
        seed=args.smoke_seed,
    )

    # ----- (6) Read source-rate + emit verdict marker -----
    source_rate = _smoke_source_substring_rate(
        cell_output_dir=cell_output_dir,
        source=args.smoke_source,
        marker=args.marker_token,
    )

    verdict = classify_smoke_timing(avg_min_per_ck)
    kind, note = build_smoke_marker(
        verdict,
        avg_minutes_per_checkpoint=avg_min_per_ck,
        n_checkpoints=n_ck,
        total_eval_minutes=eval_minutes,
        train_minutes=train_minutes,
        source_rate=source_rate,
        cell_key=cell.key,
        source=args.smoke_source,
        seed=args.smoke_seed,
    )
    # Round 9 — write verdict to local file (NOT post via task.py).
    # Orchestrator on the VM side reads SMOKE_VERDICT.json via SCP and
    # posts the epm:smoke-* marker from the repo root where task.py works.
    write_verdict_file(
        args.slab_root,
        "SMOKE_VERDICT.json",
        {
            "kind": kind,
            "verdict_band": verdict,
            "note": note,
            "cell_key": cell.key,
            "source": args.smoke_source,
            "seed": args.smoke_seed,
            "avg_minutes_per_checkpoint": avg_min_per_ck,
            "n_checkpoints": n_ck,
            "total_eval_minutes": eval_minutes,
            "train_wall_minutes": train_minutes,
            "source_substring_rate": source_rate,
            "issue": args.issue,
        },
    )

    log.info("Smoke marker kind: %s (verdict-band: %s)", kind, verdict)
    return 0 if kind == "epm:smoke-pass" else 1


# ---------------------------------------------------------------------------
# Phase B — full 324-run sweep (orchestration shell)
# ---------------------------------------------------------------------------


def run_sweep_phase(args: argparse.Namespace, *, repo_root: Path) -> int:
    """Phase B: enumerate the 324-run sweep, gate on prior smoke-pass marker.

    Returns 0 on dispatch (or dry-run), non-zero on gate failure.
    """
    if (
        args.require_smoke_pass
        and not args.skip_smoke_pass_check
        and not is_smoke_pass_confirmed_locally(args)
    ):
        log.error(
            "Phase B refused: no smoke-pass signal on task #%d. "
            "Round 9 dropped the task.py shellout (pod cannot reach VM "
            "workflow API); pass --smoke-pass-confirmed (orchestrator "
            "sets this AFTER posting epm:smoke-pass v1 from the VM side) "
            "OR ensure %s exists with positive source_substring_rate "
            "OR pass --skip-smoke-pass-check to override.",
            args.issue,
            _cell_output_dir(
                args.slab_root,
                cell_key=args.smoke_cell,
                source=args.smoke_source,
                seed=args.smoke_seed,
            )
            / "metrics_final.json",
        )
        return 2

    sources = _parse_sources(args.sources)
    seeds = _parse_seeds(args.seeds)
    cells_per_seed = _enumerate_valid_cells_per_seed()

    job_count = len(sources) * len(seeds) * len(cells_per_seed)
    log.info(
        "Phase B sweep — %d sources x %d seeds x %d cells = %d (cell, seed) runs",
        len(sources),
        len(seeds),
        len(cells_per_seed),
        job_count,
    )
    log.info(
        "Round 12: two-pass sweep. Pass 1 = HF only (train + log-prob eval) for "
        "every cell; one HF→vLLM teardown between passes; Pass 2 = vLLM loaded "
        "once with --enable-lora, LoRA-swap per cell for sampled eval. No "
        "framework-switch mid-pass = no orphan-worker risk per CLAUDE.md gotcha."
    )

    if args.dry_run:
        log.info("--dry-run: enumeration only; %d jobs would be dispatched", job_count)
        return 0

    return _run_sweep_two_pass(
        sources=sources,
        seeds=seeds,
        cells=cells_per_seed,
        args=args,
        repo_root=repo_root,
    )


# ---------------------------------------------------------------------------
# Helpers — kept thin so the smoke-gate + wiring logic is the testable surface
# ---------------------------------------------------------------------------


def _cell_output_dir(slab_root: Path, *, cell_key: str, source: str, seed: int) -> Path:
    return slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"


def _enumerate_checkpoint_dirs(cell_output_dir: Path, save_every_n_steps: int) -> list[str]:
    """List adapter checkpoint dirs (checkpoint-25, -50, ..., -150).

    Skips dirs that don't exist (e.g. truncated training) so the caller can
    decide whether to fail or proceed.
    """
    adapter_dir = cell_output_dir / "adapter"
    if not adapter_dir.exists():
        return []
    found: list[str] = []
    for ckpt in sorted(adapter_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        found.append(str(ckpt))
    return found


def _load_base_model_for_logprob(first_checkpoint_dir: str) -> tuple[Any, Any]:
    """Load Qwen2.5-7B-Instruct + tokenizer with the first checkpoint attached.

    Imports the heavy frameworks inside this function so the smoke-decision
    code paths can be tested without GPU / torch / TRL.
    """
    import os

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base, first_checkpoint_dir, adapter_name="ck0")
    return peft_model, tok


def _run_smoke_sampled_eval(
    *,
    cell_output_dir: Path,
    lora_path: str,
    overrides: dict[str, str],
    panel: dict[str, str],
    questions: list[str],
    marker: str,
    seed: int = 42,
) -> None:
    """Run the vLLM ``--enable-lora`` sampled eval at the smoke cell.

    Round 6 update: switched from ``EvalConfig + generate_completions(
    merged_path)`` to ``generate_completions_with_lora(base + lora_path)``.
    No merge step → no merged dir → no ~14 GB disk pressure on the smoke
    cell, AND the production sweep can run at 8/8 concurrency.

    BLOCKER 3 contract (code-review v3) is preserved: writes
    ``metrics_final.json`` under ``cell_output_dir`` so
    ``_smoke_source_substring_rate`` has data to return and the M1 live
    check in ``build_smoke_marker`` can fire.

    Writes ``metrics_final.json`` with shape::

        {
          "marker": "<runtime marker>",
          "personas": {"<persona>": {"substring_rate": ..., ...}, ...},
          "panel_size": 24, "questions": 20, "num_completions": 5,
          "vllm_lora_mode": true,
        }

    Failures (vLLM crash, adapter-load error, IO error) propagate as
    exceptions — the calling smoke phase exits non-zero and
    ``build_smoke_marker`` reports FAIL. Wrapping in try/except would
    re-introduce the silent-pass bug BLOCKER 3 is designed to eliminate.
    """
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        DEFAULT_NUM_COMPLETIONS,
        generate_completions_with_lora,
        score_markers_threaded,
    )
    from explore_persona_space.experiments.factor_screen_397.training import BASE_MODEL

    log.info(
        "Smoke sampled eval starting (vLLM --enable-lora): base=%s lora_path=%s "
        "(panel=%d personas x %d questions x %d completions)",
        BASE_MODEL,
        lora_path,
        len(panel),
        len(questions),
        DEFAULT_NUM_COMPLETIONS,
    )
    completions = generate_completions_with_lora(
        base_model_path=BASE_MODEL,
        lora_path=lora_path,
        personas=dict(panel),
        questions=list(questions),
        system_prompt_overrides=overrides,
        seed=seed,
    )
    persona_scores = score_markers_threaded(completions, marker=marker)

    payload = {
        "marker": marker,
        "panel_size": len(panel),
        "questions": len(questions),
        "num_completions": DEFAULT_NUM_COMPLETIONS,
        "personas": persona_scores,
        "vllm_lora_mode": True,
    }
    out_path = cell_output_dir / "metrics_final.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Smoke sampled eval complete: wrote %s", out_path)


def _smoke_source_substring_rate(
    cell_output_dir: Path,
    *,
    source: str,
    marker: str,
) -> float | None:
    """Read the final-checkpoint sampled-eval JSON and return the source rate.

    Plan v4 §5.7 makes this the LIVE M1 check: a smoke that times out clean
    but has source rate == 0 is a marker-threading failure (the timing band
    alone would PASS but the verdict downgrades to WARN — handled in
    build_smoke_marker).

    Returns None when the sampled-eval JSON isn't present (e.g. eval was
    skipped or hasn't run yet); the smoke verdict can still be inferred
    from timing alone in that case.
    """
    final_metrics = cell_output_dir / "metrics_final.json"
    if not final_metrics.exists():
        log.warning("No metrics_final.json under %s — source rate unknown", cell_output_dir)
        return None
    payload = json.loads(final_metrics.read_text(encoding="utf-8"))
    # Per-cell JSON shape: {"personas": {<persona>: {"substring_rate": ..., ...}, ...}}
    persona_row = payload.get("personas", {}).get(source)
    if persona_row is None:
        log.warning("metrics_final.json has no entry for source=%s", source)
        return None
    rate = persona_row.get("substring_rate")
    if rate is None:
        return None
    # Belt-and-suspenders: the marker threaded into score_markers MUST match
    # the runtime marker (M1). The dispatcher's eval call uses
    # score_markers_threaded; this assertion catches the case where someone
    # bypassed it.
    rec_marker = payload.get("marker")
    if rec_marker is not None and rec_marker != marker:
        raise ValueError(
            f"metrics_final.json marker={rec_marker!r} != runtime marker={marker!r} — "
            "M1 thread is broken; refusing to report a misleading source rate."
        )
    return float(rate)


def is_smoke_pass_confirmed_locally(args: argparse.Namespace) -> bool:
    """Phase B gate: return True if the smoke verdict is locally confirmed.

    **Round 9 replacement for ``has_recent_smoke_pass_marker``.** The old
    function shelled out to ``scripts/task.py find <N>`` to locate the
    task dir and scan ``events.jsonl`` for the ``epm:smoke-pass`` row.
    ``task.py`` branch-guards to ``main`` and refuses to run from a
    worktree branch → every shellout from pod-397 (on ``issue-397``)
    failed → dispatcher crashed before launching the sweep.

    The dispatcher now uses two local-only signals (either is sufficient):

      1. ``--smoke-pass-confirmed`` CLI flag. The orchestrator sets this
         AFTER successfully posting ``epm:smoke-pass v1`` from the VM
         side (where ``task.py`` works); this is the canonical gate.

      2. ``<slab_root>/cell_<smoke_cell>/source_<smoke_source>/seed_<
         smoke_seed>/metrics_final.json`` exists AND has a
         ``source_substring_rate > 0`` (the M1 live check). This is the
         fallback that lets the dispatcher self-confirm when re-launched
         on the same pod after a smoke that ran cleanly but the
         orchestrator hasn't gotten around to setting the flag yet.

    Returns True if EITHER signal is satisfied; False otherwise. Never
    crashes on the missing-file path — caller surfaces the recovery
    hints in its error message.
    """
    if getattr(args, "smoke_pass_confirmed", False):
        log.info("--smoke-pass-confirmed set; skipping local file check")
        return True

    smoke_cell_dir = _cell_output_dir(
        args.slab_root,
        cell_key=args.smoke_cell,
        source=args.smoke_source,
        seed=args.smoke_seed,
    )
    metrics_path = smoke_cell_dir / "metrics_final.json"
    if not metrics_path.exists():
        log.info(
            "No smoke metrics_final.json at %s; smoke-pass NOT confirmed locally",
            metrics_path,
        )
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning(
            "metrics_final.json at %s is malformed (%s); treating smoke as NOT confirmed",
            metrics_path,
            e,
        )
        return False

    personas = payload.get("personas") if isinstance(payload, dict) else None
    if not isinstance(personas, dict):
        log.warning("metrics_final.json at %s has no 'personas' block", metrics_path)
        return False

    source_row = personas.get(args.smoke_source)
    if not isinstance(source_row, dict):
        log.warning(
            "metrics_final.json at %s has no entry for source=%s",
            metrics_path,
            args.smoke_source,
        )
        return False

    rate = source_row.get("substring_rate")
    if not isinstance(rate, int | float) or rate <= 0:
        log.warning(
            "metrics_final.json at %s has source_substring_rate=%r; smoke not confirmed",
            metrics_path,
            rate,
        )
        return False

    log.info(
        "Smoke-pass confirmed locally via %s (source_substring_rate=%.3f)",
        metrics_path,
        rate,
    )
    return True


# ---------------------------------------------------------------------------
# Sweep-level resume (Round 6)
# ---------------------------------------------------------------------------

HF_ADAPTER_REPO: str = "superkaiba1/explore-persona-space"


def is_cell_complete_locally(cell_output_dir: Path) -> bool:
    """Return True if ``cell_output_dir / 'metrics.json'`` parses cleanly.

    Round 6 resume probe (local branch). The per-cell in-process
    pipeline (``_run_one_cell_inprocess``) writes ``metrics.json`` near
    the END (after vLLM sampled eval, BEFORE upload-verify + cleanup) so
    metrics.json existing without HF Hub adapter coexisting is a real
    state to watch for — filter_jobs_for_resume's both-mode LOUD-FAIL
    catches it. Round 11: same shape as Round 5..10 ``run_one_cell``
    used to write; the marker is still "metrics.json on disk".

    Returns False on missing file OR JSON parse error (treat malformed
    as "not complete" — re-run will overwrite cleanly).
    """
    metrics_path = cell_output_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning(
            "metrics.json at %s is malformed JSON — treating cell as NOT complete; "
            "re-run will overwrite.",
            metrics_path,
        )
        return False
    return isinstance(payload, dict) and "personas" in payload


def is_cell_complete_on_hub(
    cell_key: str,
    source: str,
    seed: int,
    *,
    hub_files_cache: list[str] | None = None,
) -> bool:
    """Return True if the cell's adapter is present on HF Hub.

    Probes ``HfApi.list_repo_files`` for the canonical
    ``adapters/issue_397/i397_cell_<key>_source_<source>_seed<seed>/``
    prefix carrying ``adapter_*`` files. ``hub_files_cache`` lets the
    caller pre-fetch the full file list once and pass it in to avoid
    324 separate Hub round-trips during the resume scan.

    Returns False on missing path OR transient Hub failure (caller
    treats False as "not complete" — re-run will redo the cell + re-
    upload).
    """
    run_name = f"i397_cell_{cell_key}_source_{source}_seed{seed}"
    prefix = f"adapters/issue_397/{run_name}/"
    if hub_files_cache is None:
        hub_files_cache = _fetch_hub_adapter_index()
    if hub_files_cache is None:
        return False
    return any(
        f.startswith(prefix) and ("adapter_" in f.rsplit("/", 1)[-1]) for f in hub_files_cache
    )


def _fetch_hub_adapter_index() -> list[str] | None:
    """One-shot HF Hub probe for the model repo's file list.

    Returns the full list of files (full paths under repo root) OR None
    on import / network failure. The caller treats None as "Hub
    unreachable" and falls back to local-only resume.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("huggingface_hub not importable; skipping Hub resume probe")
        return None
    import os as _os

    api = HfApi(token=_os.environ.get("HF_TOKEN"))
    try:
        return api.list_repo_files(repo_id=HF_ADAPTER_REPO, repo_type="model")
    except Exception as e:
        log.warning("HF Hub list_repo_files failed (%s); skipping Hub resume probe", e)
        return None


def filter_jobs_for_resume(
    jobs: list[tuple[Any, str, int]],
    *,
    slab_root: Path,
    resume_source: str,
    hub_files_cache: list[str] | None = None,
) -> tuple[list[tuple[Any, str, int]], dict[str, int]]:
    """Filter completed cells out of the sweep job list.

    ``resume_source`` ∈ {"local", "hub", "both"}:
      - "local" → skip cells with ``metrics.json`` on disk.
      - "hub" → skip cells with adapter on HF Hub.
      - "both" → skip cells with EITHER signal AND LOUD-FAIL on
        REAL inconsistent state (both probes succeeded AND disagree —
        i.e. local says done but Hub returns adapter-missing).

    **Hub-unreachable degradation (round 7 fix for code-review v6
    Major 1).** When ``hub_files_cache is None`` (the Hub probe failed
    via network / rate-limit), the "both" mode degrades to local-only:
    local-present cells are skipped, no LOUD-FAIL fires, and a single
    WARNING is logged. The corruption-detection LOUD-FAIL only triggers
    when BOTH probes successfully returned AND disagree (the real
    corruption signal); a transient Hub outage should not block the
    sweep.

    Returns ``(remaining_jobs, summary)`` where summary has counts of
    skipped-local / skipped-hub / skipped-both / queued.
    """
    remaining: list[tuple[Any, str, int]] = []
    summary = {
        "skipped_local": 0,
        "skipped_hub": 0,
        "skipped_both": 0,
        "queued": 0,
    }
    inconsistent: list[tuple[str, str, int]] = []

    # Detect Hub unreachable for the "both" path. None is the canonical
    # signal from _fetch_hub_adapter_index() for "probe failed; treat as
    # 'no Hub data'". Log ONCE up-front so the warning isn't spammed once
    # per cell.
    hub_unreachable = resume_source == "both" and hub_files_cache is None
    if hub_unreachable:
        log.warning(
            "Hub probe failed (hub_files_cache is None); resuming from local "
            "state only. Risk: a cell completed locally but not yet uploaded "
            "to Hub will be SKIPPED (analyzer will be unable to pull weights). "
            "Re-run after Hub comes back, or pass --no-resume to force full "
            "re-launch."
        )

    for cell, source, seed in jobs:
        cell_dir = _cell_output_dir(slab_root, cell_key=cell.key, source=source, seed=seed)
        local_ok = is_cell_complete_locally(cell_dir)
        if resume_source == "local":
            if local_ok:
                summary["skipped_local"] += 1
                continue
        elif resume_source == "hub":
            hub_ok = is_cell_complete_on_hub(
                cell.key, source, seed, hub_files_cache=hub_files_cache
            )
            if hub_ok:
                summary["skipped_hub"] += 1
                continue
        elif hub_unreachable:
            # "both" mode but Hub probe failed. Degrade to local-only:
            # skip locally-complete cells, queue everything else. No
            # LOUD-FAIL because we cannot prove inconsistency without a
            # successful Hub probe — "Hub returned no data" is NOT the
            # same as "Hub returned 'adapter missing'".
            if local_ok:
                summary["skipped_local"] += 1
                continue
        else:  # both, with successful Hub probe
            hub_ok = is_cell_complete_on_hub(
                cell.key, source, seed, hub_files_cache=hub_files_cache
            )
            if local_ok and hub_ok:
                summary["skipped_both"] += 1
                continue
            if local_ok and not hub_ok:
                # LOUD-FAIL: BOTH probes succeeded AND they disagree
                # → corruption / partial upload. Surface so the user
                # can investigate before re-running and clobbering local
                # artifacts. This branch is unreachable when
                # hub_files_cache is None (handled by hub_unreachable
                # branch above).
                inconsistent.append((cell.key, source, seed))
                continue
            if hub_ok and not local_ok:
                # Local was wiped (e.g. pod recycled); Hub has the result.
                # Skip — analyzer can pull from Hub.
                summary["skipped_hub"] += 1
                continue
        remaining.append((cell, source, seed))
        summary["queued"] += 1

    if inconsistent:
        raise ValueError(
            f"Resume LOUD-FAIL: {len(inconsistent)} cell(s) have local metrics.json "
            f"but missing HF Hub adapter (potential corruption / partial upload). "
            f"Investigate before re-running. Affected cells: {inconsistent[:10]}"
            + (f" (and {len(inconsistent) - 10} more)" if len(inconsistent) > 10 else "")
            + ". To force-re-run anyway, pass --no-resume; to ignore the Hub side, "
            "pass --resume-source=local."
        )

    return remaining, summary


def _enumerate_valid_cells_per_seed() -> list[Any]:
    """108 valid cells per seed (12 ABCD x 3 E levels x 3 sources).

    Defers to factor_screen_397.cells.valid_cells_per_source for the per-
    source enumeration (36 cells); the sources dimension multiplies that
    by 3 inside the sweep loop in `_run_sweep_serial`.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import (
        valid_cells_per_source,
    )

    return list(valid_cells_per_source())


def verify_adapter_on_hf_hub(*, hf_path_in_repo: str, repo_id: str) -> bool:
    """Probe HF Hub to confirm an adapter directory exists under ``hf_path_in_repo``.

    Returns True if at least one ``adapter_*`` file (e.g. ``adapter_model.safetensors``,
    ``adapter_config.json``) is present at the path. Returns False on missing path
    OR transient Hub failure — caller treats False as "do not delete local weights".

    Per CLAUDE.md upload policy: "Models MUST upload to HF model repo before local
    deletion. Never delete unuploaded." This helper is the gate that enforces it.

    Round 11 lifted from the deleted ``run_one_cell.py`` module so the in-
    process serial sweep can call it inline between the upload step and the
    cleanup step.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not importable; cannot verify HF upload")
        return False

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        log.error("HF Hub list_repo_files failed (%s); cannot verify upload", e)
        return False

    prefix = hf_path_in_repo.rstrip("/") + "/"
    found = [f for f in files if f.startswith(prefix) and ("adapter_" in f.rsplit("/", 1)[-1])]
    if not found:
        log.warning(
            "HF Hub verification: NO adapter files under %s/%s — refusing to delete locally",
            repo_id,
            hf_path_in_repo,
        )
        return False
    log.info(
        "HF Hub verification PASS: %d adapter file(s) at %s/%s",
        len(found),
        repo_id,
        hf_path_in_repo,
    )
    return True


def cleanup_cell_local_weights(cell_output_dir: Path) -> dict[str, int]:
    """Remove merged/ + checkpoint-*/ directories after upload-verify PASS.

    Plan v4 §11 disk-quota discipline: peak disk per cell ~3 GB
    (intermediate checkpoint dirs) post-Round-6. The merge step that
    drove the ~14 GB footprint has been removed; vLLM ``--enable-lora``
    consumes the adapter directly without merging. Keep the per-cell
    ``metrics.json`` + ``logprob_*.json`` + ``prepared_dataset.json`` +
    ``run.log`` — they're small text + needed for diagnosis.

    Returns ``{"merged_removed": 0|1, "checkpoints_removed": N}`` for
    bookkeeping. ``merged_removed`` is retained for backward-compat with
    pre-Round-6 cell dirs that may carry a stale ``merged/`` from an
    older training run; new Round-6+ runs always report 0 there.

    Round 11 lifted from the deleted ``run_one_cell.py`` module so the
    in-process serial sweep can clean up between cells (otherwise the
    second cell's training overflows the MooseFS ~130 GB per-pod quota).
    """
    removed = {"merged_removed": 0, "checkpoints_removed": 0}
    merged_dir = cell_output_dir / "merged"
    if merged_dir.is_dir():
        shutil.rmtree(merged_dir)
        removed["merged_removed"] = 1
        log.info("Cleanup: removed %s", merged_dir)
    adapter_dir = cell_output_dir / "adapter"
    if adapter_dir.is_dir():
        for ck in sorted(adapter_dir.glob("checkpoint-*")):
            if ck.is_dir():
                shutil.rmtree(ck)
                removed["checkpoints_removed"] += 1
        log.info(
            "Cleanup: removed %d checkpoint dir(s) under %s",
            removed["checkpoints_removed"],
            adapter_dir,
        )
    return removed


def _cleanup_pass1_cell(cell_output_dir: Path) -> dict[str, int]:
    """Round 13 — Pass-1 per-cell cleanup (disk-full fix).

    Round 12's two-pass design left intermediate checkpoints + the
    per-cell training JSONL on disk between Pass 1 cells. Sweep crashed
    at cell 22/108 with ``OSError: [Errno 28] No space left on device``
    when 21 cells x ~4 GB each filled a 200 GB pod disk.

    Called at the end of each successful Pass 1 cell, AFTER:
      - ``train_one_cell`` returned (TRL inline-upload fence pushed the
        adapter to HF Hub during training)
      - ``compute_logprob_panel`` returned + ``logprob_panel.json`` was
        persisted on disk
      - ``verify_adapter_on_hf_hub`` confirmed the adapter actually
        landed on Hub (LOUD-FAIL safety net per CLAUDE.md upload-policy
        — the TRL fence silently swallows upload errors per the
        sft.py:681 ``except Exception`` gotcha; verify is the only way
        to know it worked).

    Deletes (Pass 2 doesn't need any of these):
      - ``<cell_dir>/adapter/checkpoint-*/`` — intermediate LoRA
        checkpoints (6 dirs x ~485 MB each = ~3 GB). Pass 1's log-prob
        eval already consumed them; Pass 2 uses only the final
        ``<cell_dir>/adapter/`` weights via vLLM's ``LoRARequest``.
      - ``<cell_dir>/prepared_train.jsonl`` — the per-cell training
        JSONL (~1 MB). Pass 2 reads the recipe-fix manifest
        (``prepared_dataset.json``) for system_prompt overrides, NOT
        the training JSONL.
      - ``<cell_dir>/wandb/`` — any WandB run dirs (varies). Pass 2
        doesn't read these; WandB Artifacts (when uploaded) live in
        the cloud.

    Keeps (Pass 2 needs these):
      - ``<cell_dir>/adapter/`` (root, sans ``checkpoint-*``) —
        final LoRA weights (~485 MB) for vLLM ``LoRARequest``.
      - ``<cell_dir>/logprob_panel.json`` (~100 KB) — Pass 1 deliverable.
      - ``<cell_dir>/prepared_dataset.json`` — recipe-fix manifest
        (small; Pass 2's ``build_train_matched_persona_panel`` reads it).
      - ``<cell_dir>/run.log`` (small; for debug).

    Per-cell footprint after this helper: ~500 MB (down from ~4 GB).
    Peak Pass 1 disk = 1 active cell at ~3.5 GB + (N-done cells x 500 MB).
    For 108 cells x 500 MB = ~54 GB total, well under a 200 GB pod disk.

    Returns ``{"checkpoints_removed": N, "prepared_train_removed": 0|1,
    "wandb_dirs_removed": N}`` for bookkeeping.

    **NOTE on glob path:** the brief specified
    ``<cell_dir>/checkpoint-*/`` but the actual layout (per
    ``_enumerate_checkpoint_dirs``) is
    ``<cell_dir>/adapter/checkpoint-*/``. Matches the brief's intent
    (delete intermediate checkpoints, keep adapter) using the correct
    nested path.
    """
    removed = {
        "checkpoints_removed": 0,
        "prepared_train_removed": 0,
        "wandb_dirs_removed": 0,
    }

    # 1. Intermediate LoRA checkpoints under adapter/.
    adapter_dir = cell_output_dir / "adapter"
    if adapter_dir.is_dir():
        for ck in sorted(adapter_dir.glob("checkpoint-*")):
            if ck.is_dir():
                shutil.rmtree(ck)
                removed["checkpoints_removed"] += 1

    # 2. Per-cell training JSONL.
    prep_train = cell_output_dir / "prepared_train.jsonl"
    if prep_train.exists():
        prep_train.unlink()
        removed["prepared_train_removed"] = 1

    # 3. WandB run directories (if any landed in the cell dir).
    for wandb_dir in cell_output_dir.glob("wandb*"):
        if wandb_dir.is_dir():
            shutil.rmtree(wandb_dir)
            removed["wandb_dirs_removed"] += 1

    # Log free disk after cleanup so the next OOM is debuggable.
    try:
        free_gb = shutil.disk_usage(cell_output_dir.parent).free / (1024**3)
        log.info(
            "Pass 1 cell cleanup: checkpoints=%d, prepared_train=%d, wandb=%d; free disk %.1f GB",
            removed["checkpoints_removed"],
            removed["prepared_train_removed"],
            removed["wandb_dirs_removed"],
            free_gb,
        )
    except OSError as e:
        # disk_usage can fail on weird mount setups; don't kill the sweep
        # over a logging failure. Cleanup itself already succeeded.
        log.warning(
            "Pass 1 cell cleanup: checkpoints=%d, prepared_train=%d, wandb=%d; "
            "free-disk probe failed (%s)",
            removed["checkpoints_removed"],
            removed["prepared_train_removed"],
            removed["wandb_dirs_removed"],
            e,
        )

    return removed


def _aggressive_hf_to_vllm_teardown(*local_refs: Any) -> None:
    """Round 8 HF→vLLM teardown sequence (4-step + log).

    Lifted from the deleted ``run_one_cell.py`` so the in-process serial
    sweep can call it inline between the log-prob eval and the vLLM
    sampled eval. See the original Round 8 commit + agent-memory
    ``vllm_orphan_worker_after_destroy`` note for the failure mode this
    addresses: ``del`` + ``gc.collect`` + ``empty_cache`` alone is
    insufficient; ``synchronize()`` is required to flush pending CUDA
    ops before vLLM tries to grab GPU memory.

    Steps:
      1. Caller passes Python refs (peft_model, base, tokenizer); we
         don't ``del`` directly because Python ``del`` only drops the
         binding the CALLER has — we rely on the caller to drop them
         before calling this helper. This function does the GC + cache
         + sync + log.
      2. ``gc.collect()`` to clear Python refs.
      3. ``torch.cuda.empty_cache()`` to release PyTorch caching
         allocator blocks.
      4. ``torch.cuda.synchronize()`` to ensure pending CUDA ops finish
         before mem-info read.
      5. Log pre/post free-memory so the next OOM is debuggable.

    The ``*local_refs`` argument is intentionally unused — it exists so
    the caller can pass the about-to-be-deleted refs as a documentation
    contract (the dispatcher's smoke-phase pattern: ``del base_model;
    del tokenizer`` immediately before calling teardown). Tests assert
    on caller-side ``del`` lines via static AST scan.

    See CLAUDE.md "vLLM in-process teardown" gotcha for the warning that
    even this 4-step pattern doesn't reap vLLM worker subprocesses on the
    *reverse* path (vLLM → HF). The dispatcher's sequence here is HF →
    vLLM only; we never load HF after vLLM, so the worker-subprocess
    survival problem doesn't bite us.
    """
    del local_refs  # documentation-only; caller already dropped the refs
    import gc as _gc

    import torch as _torch

    free_before_gb = -1.0
    if _torch.cuda.is_available():
        free_before_gb = _torch.cuda.mem_get_info()[0] / (1024**3)

    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
        free_after_gb = _torch.cuda.mem_get_info()[0] / (1024**3)
        log.info(
            "HF teardown before vLLM: free GPU memory %.2f GB → %.2f GB "
            "(residue %.2f GB; vLLM will request 0.45 * total)",
            free_before_gb,
            free_after_gb,
            free_before_gb - free_after_gb if free_after_gb > free_before_gb else 0.0,
        )


# ---------------------------------------------------------------------------
# Round 12 — two-pass sweep (HF-only pass 1, vLLM-only pass 2)
# ---------------------------------------------------------------------------


def is_cell_pass1_complete(cell_output_dir: Path) -> bool:
    """Return True if Pass 1 (HF train + log-prob eval) has landed for this cell.

    Pass 1's terminal artifact is ``logprob_panel.json``: the
    per-checkpoint log-prob result written after ``compute_logprob_panel``
    returns. Presence + JSON-parsable + non-empty dict is the sentinel.

    NOTE: this is a NECESSARY-but-not-sufficient check for full cell
    completion. The HF Hub adapter must ALSO be present (verified
    separately via ``is_cell_complete_on_hub`` against the index cache)
    so Pass 2's vLLM ``LoRARequest`` can find the adapter. Resume logic
    combines the two probes.
    """
    logprob_path = cell_output_dir / "logprob_panel.json"
    if not logprob_path.exists():
        return False
    try:
        payload = json.loads(logprob_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning(
            "logprob_panel.json at %s malformed; treating Pass 1 as NOT complete",
            logprob_path,
        )
        return False
    # Real Pass 1 output is a dict keyed by checkpoint dir → scores; empty
    # dict means "scored zero checkpoints" which is wrong.
    return isinstance(payload, dict) and len(payload) > 0


def _run_pass1_hf(
    cells_to_run: list[tuple[Any, str, int]],
    *,
    args: argparse.Namespace,
) -> dict[tuple[str, str, int], int]:
    """Pass 1 — HF Transformers only (train + log-prob eval), no vLLM.

    For each (cell, source, seed) tuple:

      1. ``prepare_cell_jsonl`` writes the per-cell JSONL + recipe-fix
         manifest.
      2. ``train_one_cell(hf_upload=True)`` runs LoRA SFT; the TRL
         inline-upload fence pushes the adapter to HF Hub.
      3. ``compute_logprob_panel`` runs per-checkpoint log-prob eval
         across the 480-context train-matched panel.
      4. Write ``logprob_panel.json`` to disk.
      5. Drop refs (``del base_model; del tokenizer_lp``) to release GPU
         memory before the next cell. NO vLLM in this pass.

    Returns per-cell rc dict: 0 = ok, 1 = exception caught, 2 reserved
    for verify failures (only Pass 2 emits rc=2; Pass 1 only sees
    train/eval crashes).

    All cells in Pass 1 use only HF Transformers
    (``AutoModelForCausalLM`` + ``PeftModel``); no framework switch
    within the pass = no orphan workers. Standard Python GC releases
    memory between cells when the HF model refs go out of scope.

    Per CLAUDE.md "Checkpoint per phase": logprob_panel.json lands on
    disk INSIDE the per-cell loop. A Pass 1 crash on cell N preserves
    cells 1..N-1's log-prob outputs; the next run picks up at cell N
    via ``is_cell_pass1_complete``.
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        EVAL_QUESTIONS_20,
    )
    from explore_persona_space.experiments.factor_screen_397.data_prep import (
        prepare_cell_jsonl,
    )
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        FINAL_CHECKPOINT_MARKER_VARIANTS,
        build_train_matched_persona_panel,
        compute_logprob_panel,
        read_prepared_dataset_manifest,
    )
    from explore_persona_space.experiments.factor_screen_397.training import (
        BASE_MODEL,
        train_one_cell,
    )

    per_cell_rc: dict[tuple[str, str, int], int] = {}
    n_total = len(cells_to_run)

    for job_index, (cell, source, seed) in enumerate(cells_to_run, start=1):
        cell_dir = _cell_output_dir(args.slab_root, cell_key=cell.key, source=source, seed=seed)
        cell_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "[Pass 1 — HF] [%d/%d] starting cell=%s source=%s seed=%d e=%d -> %s",
            job_index,
            n_total,
            cell.key,
            source,
            seed,
            cell.e,
            cell_dir,
        )
        try:
            # Heavy AutoTokenizer import deferred so the dispatcher's CLI /
            # arg-parse path can be unit-tested without pulling transformers.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            # (1) Data-prep with C=1 preflight + B=1 band assertion ON.
            data_path = cell_dir / "prepared_train.jsonl"
            log.info("Preparing cell JSONL from pool dir %s -> %s", args.pool_dir, data_path)
            prep_result = prepare_cell_jsonl(
                cell=cell,
                source=source,
                pool_dir=args.pool_dir,
                output_path=data_path,
                marker_text=args.marker_token,
                pos_per_source=args.pos_per_source,
                neg_per_source=args.pos_per_source,
                seed=seed,
                tokenizer=tokenizer,
                enforce_c_preflight=True,
                enforce_b1_band=True,
            )
            system_prompt_text = prep_result["system_prompt_text"]
            log.info(
                "Data-prep complete: %d pos + %d neg = %d rows (data_policy=%s)",
                prep_result["num_positive"],
                prep_result["num_negative"],
                prep_result["num_total"],
                prep_result["data_policy"],
            )

            # (2) Train. hf_upload=True; TRL inline-upload fence pushes the
            # adapter to HF Hub. verify_adapter_on_hf_hub in Pass 2 is the
            # safety net for silent-swallow failures of the fence.
            train_start = time.time()
            outcome = train_one_cell(
                cell=cell,
                seed=seed,
                source=source,
                data_path=data_path,
                cell_output_dir=cell_dir,
                marker_text=args.marker_token,
                save_every_n_steps=args.save_every_n_steps,
                lr=args.lr,
                warmup_ratio=args.warmup_ratio,
                hf_upload=True,
                system_prompt_text=system_prompt_text,
            )
            train_minutes = (time.time() - train_start) / 60.0
            log.info("Pass 1 training complete: %.2f min, loss=%.4f", train_minutes, outcome.loss)

            # (3) Train-matched panel + per-checkpoint log-prob eval.
            checkpoint_dirs = _enumerate_checkpoint_dirs(cell_dir, args.save_every_n_steps)
            if not checkpoint_dirs:
                log.error("No intermediate checkpoint dirs under %s", cell_dir / "adapter")
                per_cell_rc[(cell.key, source, seed)] = 1
                continue

            manifest = read_prepared_dataset_manifest(cell_dir)
            panel, overrides = build_train_matched_persona_panel(
                canonical_panel=EVAL_PERSONAS_24,
                source=source,
                manifest=manifest,
            )

            base_model, tokenizer_lp = _load_base_model_for_logprob(checkpoint_dirs[0])
            questions = list(EVAL_QUESTIONS_20)

            log.info(
                "Pass 1 log-prob eval: %d personas x %d questions = %d contexts x %d ckpts",
                len(panel),
                len(questions),
                len(panel) * len(questions),
                len(checkpoint_dirs),
            )
            eval_start = time.time()
            logprob_result = compute_logprob_panel(
                base_model=base_model,
                tokenizer=tokenizer_lp,
                checkpoint_dirs=checkpoint_dirs,
                personas=panel,
                questions=questions,
                system_prompt_overrides=overrides,
                marker_texts=FINAL_CHECKPOINT_MARKER_VARIANTS,
                batch_size=8,
                device="cuda:0",
            )
            eval_minutes = (time.time() - eval_start) / 60.0
            log.info("Pass 1 log-prob eval complete: %.2f min", eval_minutes)

            # (4) Persist logprob result BEFORE dropping refs (CLAUDE.md
            # "checkpoint per phase"). If the drop somehow OOMs or the
            # next iteration crashes, we keep this cell's Pass 1 output.
            logprob_path = cell_dir / "logprob_panel.json"
            logprob_path.write_text(json.dumps(logprob_result, indent=2), encoding="utf-8")
            log.info("Wrote %s", logprob_path)

            # (5) Drop HF refs. NO vLLM call in this pass — Python GC
            # will release memory between cells; no orphan-worker risk.
            del base_model
            del tokenizer_lp
            del tokenizer

            # (6) Verify HF Hub upload BEFORE per-cell cleanup (Round 13).
            # Without this gate, the TRL inline-upload fence (sft.py:681
            # `except Exception`) can silently swallow a transient upload
            # failure, and we'd cleanup local weights with nothing on Hub
            # for Pass 2 to LoRA-swap against. LOUD-FAIL semantics: if
            # verify returns False, SKIP cleanup (preserve local weights
            # for retry / manual recovery) AND mark the cell rc=2 so the
            # sweep summary surfaces it. Same rc=2 convention Pass 2's
            # verify uses.
            run_name = f"i397_cell_{cell.key}_source_{source}_seed{seed}"
            hf_path_in_repo = f"adapters/issue_397/{run_name}"
            upload_verified = verify_adapter_on_hf_hub(
                hf_path_in_repo=hf_path_in_repo,
                repo_id=HF_ADAPTER_REPO,
            )
            if not upload_verified:
                log.error(
                    "[Pass 1 — HF] HF upload verification FAILED for %s — "
                    "preserving local weights at %s for manual recovery. "
                    "Cell exits rc=2; Pass 2 will skip this cell (adapter "
                    "not on Hub).",
                    hf_path_in_repo,
                    cell_dir,
                )
                per_cell_rc[(cell.key, source, seed)] = 2
                continue

            # (7) Round 13 disk-quota cleanup: delete intermediate
            # checkpoints + per-cell training JSONL. Adapter dir, logprob
            # panel, manifest, and run.log all preserved for Pass 2 +
            # diagnosis.
            removed = _cleanup_pass1_cell(cell_dir)
            log.info("[Pass 1 — HF] cleanup: %s", removed)

            per_cell_rc[(cell.key, source, seed)] = 0
            log.info(
                "[Pass 1 — HF] [%d/%d] completed: cell=%s source=%s seed=%d (rc=0)",
                job_index,
                n_total,
                cell.key,
                source,
                seed,
            )
        except Exception:
            log.exception(
                "[Pass 1 — HF] cell crashed: cell=%s source=%s seed=%d (rc=1; continuing)",
                cell.key,
                source,
                seed,
            )
            per_cell_rc[(cell.key, source, seed)] = 1

    return per_cell_rc


def _run_pass2_vllm(
    cells_to_run: list[tuple[Any, str, int]],
    *,
    args: argparse.Namespace,
) -> dict[tuple[str, str, int], int]:
    """Pass 2 — vLLM only (sampled eval, base loaded once), no HF.

    Loads vLLM ONCE with ``enable_lora=True``, then for each
    (cell, source, seed) tuple:

      1. Build a ``LoRARequest`` pointing at the cell's adapter dir.
      2. ``llm.generate(prompts, sampling_params, lora_request=...)``
         — vLLM swaps in this cell's LoRA without reloading the base.
      3. Score the per-persona substring rate via
         ``score_markers_threaded``.
      4. Write ``metrics.json`` to disk.
      5. ``verify_adapter_on_hf_hub`` (safety net; Pass 1's
         hf_upload=True is the load-bearing upload — verify catches
         silent-swallow failures of the TRL inline-upload fence).
      6. ``cleanup_cell_local_weights`` ONLY on verify PASS (CLAUDE.md
         "Models MUST upload to HF model repo before local deletion").

    vLLM stays loaded across all cells; only the LoRA adapter swaps.
    ``max_loras=1`` is sufficient because we only ever have one active
    adapter per generate() call.

    Returns per-cell rc dict (0=ok, 1=exception, 2=verify-fail).

    Per CLAUDE.md "Checkpoint per phase": metrics.json lands inside
    the loop. A Pass 2 crash preserves cells 1..N-1's metrics; the
    next run picks up at cell N via the resume probe
    (``is_cell_complete_locally`` reading metrics.json).
    """
    # Heavy imports for vLLM. Deferred so the unit tests can monkeypatch
    # _run_pass2_vllm without dragging in vLLM.
    import gc

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.factor_screen_365.eval_panel import (
        _patch_tokenizer_for_vllm,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        EVAL_QUESTIONS_20,
    )
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        DEFAULT_MAX_MODEL_LEN,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_NUM_COMPLETIONS,
        DEFAULT_VLLM_LORA_MAX_RANK,
        build_train_matched_persona_panel,
        read_prepared_dataset_manifest,
        score_markers_threaded,
    )
    from explore_persona_space.experiments.factor_screen_397.training import BASE_MODEL

    _patch_tokenizer_for_vllm()

    gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.45"))
    log.info(
        "[Pass 2 — vLLM] Loading base=%s once with enable_lora=True, "
        "gpu_memory_utilization=%.2f, max_lora_rank=%d",
        BASE_MODEL,
        gpu_mem,
        DEFAULT_VLLM_LORA_MAX_RANK,
    )
    vllm_load_start = time.time()
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        seed=42,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=DEFAULT_VLLM_LORA_MAX_RANK,
    )
    vllm_load_minutes = (time.time() - vllm_load_start) / 60.0
    log.info("[Pass 2 — vLLM] base loaded in %.2f min", vllm_load_minutes)

    # Tokenizer for chat-template prompt construction. Same Qwen tokenizer
    # vLLM uses internally, loaded once.
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        n=DEFAULT_NUM_COMPLETIONS,
        temperature=1.0,
        top_p=0.95,
        max_tokens=DEFAULT_MAX_NEW_TOKENS,
    )

    per_cell_rc: dict[tuple[str, str, int], int] = {}
    n_total = len(cells_to_run)

    # vLLM's LoRARequest needs a unique lora_int_id per adapter swap.
    # Increment per cell so vLLM's cache doesn't confuse two cells'
    # adapters under the same id.
    next_lora_id = 1

    try:
        for job_index, (cell, source, seed) in enumerate(cells_to_run, start=1):
            cell_dir = _cell_output_dir(args.slab_root, cell_key=cell.key, source=source, seed=seed)
            log.info(
                "[Pass 2 — vLLM] [%d/%d] starting cell=%s source=%s seed=%d e=%d -> %s",
                job_index,
                n_total,
                cell.key,
                source,
                seed,
                cell.e,
                cell_dir,
            )
            try:
                # (1) Read recipe-fix manifest for train-matched overrides.
                manifest = read_prepared_dataset_manifest(cell_dir)
                if manifest is None:
                    log.error(
                        "[Pass 2 — vLLM] no manifest under %s — Pass 1 likely "
                        "didn't run for this cell; rc=1",
                        cell_dir,
                    )
                    per_cell_rc[(cell.key, source, seed)] = 1
                    continue

                panel, overrides = build_train_matched_persona_panel(
                    canonical_panel=EVAL_PERSONAS_24,
                    source=source,
                    manifest=manifest,
                )
                questions = list(EVAL_QUESTIONS_20)

                # Build chat-templated prompts (same shape as
                # eval_panel.generate_completions_with_lora).
                prompts: list[str] = []
                keys: list[tuple[str, str]] = []
                for persona_name, panel_sys_prompt in panel.items():
                    system_prompt = overrides.get(persona_name, panel_sys_prompt)
                    for question in questions:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ]
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        prompts.append(text)
                        keys.append((persona_name, question))

                # (2) vLLM generate with cell's LoRA swapped in. The
                # adapter lives at cell_dir/adapter (TrainOutcome.
                # adapter_path); the final adapter is at the directory
                # root, not a checkpoint-* subdir.
                adapter_dir = cell_dir / "adapter"
                if not adapter_dir.exists():
                    log.error(
                        "[Pass 2 — vLLM] adapter dir missing at %s — Pass 1 likely "
                        "didn't complete for this cell; rc=1",
                        adapter_dir,
                    )
                    per_cell_rc[(cell.key, source, seed)] = 1
                    continue

                lora_name = f"i397_cell_{cell.key}_source_{source}_seed{seed}"
                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_int_id=next_lora_id,
                    lora_path=str(adapter_dir),
                )
                next_lora_id += 1
                log.info(
                    "[Pass 2 — vLLM] LoRARequest lora_int_id=%d lora_name=%s lora_path=%s",
                    lora_request.lora_int_id,
                    lora_request.lora_name,
                    lora_request.lora_path,
                )
                gen_start = time.time()
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
                gen_minutes = (time.time() - gen_start) / 60.0
                log.info("[Pass 2 — vLLM] generate complete in %.2f min", gen_minutes)

                # Repack outputs to {persona: {question: [completions]}}.
                completions: dict[str, dict[str, list[str]]] = {p: {} for p in panel}
                for out, (persona, question) in zip(outputs, keys, strict=True):
                    completions[persona][question] = [o.text for o in out.outputs]

                # (3) Score per-persona substring rate.
                persona_scores = score_markers_threaded(completions, marker=args.marker_token)
                source_rate = persona_scores.get(source, {}).get("substring_rate")

                # (4) Write metrics.json BEFORE verify+cleanup (per CLAUDE.md
                # checkpoint-per-phase: persist on disk inside the loop).
                metrics_payload = {
                    "marker": args.marker_token,
                    "cell_key": cell.key,
                    "source": source,
                    "seed": seed,
                    "e": cell.e,
                    "vllm_load_wall_minutes": vllm_load_minutes,
                    "vllm_gen_wall_minutes": gen_minutes,
                    "panel_size": len(panel),
                    "questions": len(questions),
                    "num_completions": DEFAULT_NUM_COMPLETIONS,
                    "personas": persona_scores,
                    "source_substring_rate": source_rate,
                    "vllm_lora_mode": True,
                    "two_pass_mode": True,  # Round 12 marker
                }
                metrics_path = cell_dir / "metrics.json"
                metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
                log.info(
                    "[Pass 2 — vLLM] wrote %s (source_rate=%s)",
                    metrics_path,
                    f"{source_rate:.3f}" if source_rate is not None else "None",
                )

                # (5) Verify HF Hub adapter present (Pass 1's hf_upload=True
                # should have uploaded; this is the safety net for silent
                # fence-swallow).
                run_name = f"i397_cell_{cell.key}_source_{source}_seed{seed}"
                hf_path_in_repo = f"adapters/issue_397/{run_name}"
                upload_verified = verify_adapter_on_hf_hub(
                    hf_path_in_repo=hf_path_in_repo,
                    repo_id=HF_ADAPTER_REPO,
                )
                if not upload_verified:
                    log.error(
                        "[Pass 2 — vLLM] HF upload verification FAILED for %s — "
                        "preserving local weights at %s for manual recovery. rc=2.",
                        hf_path_in_repo,
                        cell_dir,
                    )
                    per_cell_rc[(cell.key, source, seed)] = 2
                    continue

                # (6) Cleanup local weights (only on verify PASS).
                removed = cleanup_cell_local_weights(cell_dir)
                log.info("[Pass 2 — vLLM] cleanup: %s", removed)

                per_cell_rc[(cell.key, source, seed)] = 0
                log.info(
                    "[Pass 2 — vLLM] [%d/%d] completed: cell=%s source=%s seed=%d (rc=0)",
                    job_index,
                    n_total,
                    cell.key,
                    source,
                    seed,
                )
            except Exception:
                log.exception(
                    "[Pass 2 — vLLM] cell crashed: cell=%s source=%s seed=%d (rc=1; continuing)",
                    cell.key,
                    source,
                    seed,
                )
                per_cell_rc[(cell.key, source, seed)] = 1
    finally:
        # vLLM cleanup — same pattern as eval_panel.generate_completions_
        # with_lora's tail. We're at end-of-process here (Pass 2 is the
        # last phase before the dispatcher returns) so the orphan-worker
        # risk per the CLAUDE.md gotcha doesn't apply (no HF load after).
        log.info("[Pass 2 — vLLM] tearing down vLLM (end of sweep)")
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return per_cell_rc


def _run_sweep_two_pass(  # noqa: C901 — resume scan + per-pass dispatch + summary roll-up kept in one function for atomicity
    *,
    sources: list[str],
    seeds: list[int],
    cells: list[Any],
    args: argparse.Namespace,
    repo_root: Path,
) -> int:
    """Round 12 entry point: two-pass sweep (HF first, vLLM second).

    Pass 1 runs HF-only across every cell that needs it (no adapter on
    Hub OR no logprob_panel.json on disk). Pass 2 runs vLLM-only across
    every cell that needs sampled eval (no metrics.json on disk). The
    two pass-lists are computed independently — a cell can land in
    Pass 2 but not Pass 1 if Pass 1 already ran (resume case).

    ONE ``_aggressive_hf_to_vllm_teardown`` event fires between Pass 1
    and Pass 2 (not 108 events per cell). This is the round-12 fix for
    the round-11 reviewer's FAIL: no framework switch within a pass =
    no orphan-worker risk.

    Source-major ordering (matches Round 11 + Round 5..10) so the HF
    Hub upload trace is interpretable per-source mid-sweep.

    Returns 0 if at least one cell completed cleanly (rc=0 in either
    pass for that cell), non-zero only when ALL cells failed.
    """
    del repo_root  # unused in the two-pass path
    # Source-major, then seed, then cell.
    all_jobs: list[tuple[Any, str, int]] = [
        (cell, source, seed) for source in sources for seed in seeds for cell in cells
    ]
    job_count = len(all_jobs)
    log.info(
        "Enumerated %d sweep jobs (sources=%d seeds=%d cells=%d)",
        job_count,
        len(sources),
        len(seeds),
        len(cells),
    )

    # Resume scan: build Pass 1 + Pass 2 sub-lists independently.
    hub_files_cache = None
    no_resume = getattr(args, "no_resume", False)
    if not no_resume and args.resume_source in ("hub", "both"):
        hub_files_cache = _fetch_hub_adapter_index()

    pass1_jobs: list[tuple[Any, str, int]] = []
    pass2_jobs: list[tuple[Any, str, int]] = []
    skip_summary = {
        "pass1_skipped": 0,
        "pass2_skipped": 0,
        "fully_complete": 0,
        "pass1_queued": 0,
        "pass2_queued": 0,
    }

    for cell, source, seed in all_jobs:
        cell_dir = _cell_output_dir(args.slab_root, cell_key=cell.key, source=source, seed=seed)
        pass1_done = False
        pass2_done = False

        if not no_resume:
            # Pass 1 done = logprob_panel.json on disk AND adapter on Hub
            # (adapter on Hub is what Pass 2 needs to read; if Hub missing,
            # Pass 2 would fail verify with rc=2, so re-running Pass 1 to
            # re-upload is the right move).
            local_logprob_ok = is_cell_pass1_complete(cell_dir)
            hub_ok = False
            if args.resume_source in ("hub", "both"):
                hub_ok = is_cell_complete_on_hub(
                    cell.key, source, seed, hub_files_cache=hub_files_cache
                )
            elif args.resume_source == "local":
                # local-only mode: treat presence of adapter/ dir as Pass-1-done.
                hub_ok = (cell_dir / "adapter").exists()
            pass1_done = local_logprob_ok and hub_ok

            # Pass 2 done = metrics.json on disk (final sampled-eval output).
            pass2_done = is_cell_complete_locally(cell_dir)

        if pass1_done and pass2_done:
            skip_summary["fully_complete"] += 1
            log.info(
                "Resume: cell=%s source=%s seed=%d is fully complete (skip both passes)",
                cell.key,
                source,
                seed,
            )
            continue

        if not pass1_done:
            pass1_jobs.append((cell, source, seed))
            skip_summary["pass1_queued"] += 1
        else:
            skip_summary["pass1_skipped"] += 1

        if not pass2_done:
            pass2_jobs.append((cell, source, seed))
            skip_summary["pass2_queued"] += 1
        else:
            skip_summary["pass2_skipped"] += 1

    log.info(
        "Two-pass resume scan: %d/%d cells fully complete; Pass 1 queue=%d, "
        "Pass 2 queue=%d (Pass 1 skipped=%d, Pass 2 skipped=%d)",
        skip_summary["fully_complete"],
        job_count,
        skip_summary["pass1_queued"],
        skip_summary["pass2_queued"],
        skip_summary["pass1_skipped"],
        skip_summary["pass2_skipped"],
    )
    if skip_summary["fully_complete"] > 0 or skip_summary["pass1_skipped"] > 0:
        # Emit verdict file for the orchestrator to post epm:sweep-resume marker.
        write_verdict_file(
            args.slab_root,
            "SWEEP_RESUME.json",
            {
                "kind": "epm:sweep-resume",
                "note": (
                    f"Two-pass sweep resume: {skip_summary['fully_complete']} of "
                    f"{job_count} cells fully complete; Pass 1 queue="
                    f"{skip_summary['pass1_queued']}, Pass 2 queue="
                    f"{skip_summary['pass2_queued']}. "
                    f"resume_source={args.resume_source}."
                ),
                "issue": args.issue,
                "fully_complete": skip_summary["fully_complete"],
                "total_jobs": job_count,
                "pass1_queue": skip_summary["pass1_queued"],
                "pass2_queue": skip_summary["pass2_queued"],
                "skip_summary": skip_summary,
                "resume_source": args.resume_source,
            },
        )

    # ===== Pass 1: HF only =====
    pass1_rcs: dict[tuple[str, str, int], int] = {}
    if pass1_jobs:
        log.info(
            "===== Pass 1 (HF train + log-prob eval) starting: %d cells =====", len(pass1_jobs)
        )
        pass1_start = time.time()
        pass1_rcs = _run_pass1_hf(pass1_jobs, args=args)
        pass1_minutes = (time.time() - pass1_start) / 60.0
        log.info("===== Pass 1 complete in %.2f min =====", pass1_minutes)
    else:
        log.info("Pass 1: nothing to run (resume covered all cells)")

    # ===== Single HF→vLLM teardown between passes =====
    if pass2_jobs:
        log.info("===== Single HF→vLLM teardown event between passes =====")
        _aggressive_hf_to_vllm_teardown()

    # ===== Pass 2: vLLM only =====
    pass2_rcs: dict[tuple[str, str, int], int] = {}
    if pass2_jobs:
        log.info("===== Pass 2 (vLLM sampled eval) starting: %d cells =====", len(pass2_jobs))
        pass2_start = time.time()
        pass2_rcs = _run_pass2_vllm(pass2_jobs, args=args)
        pass2_minutes = (time.time() - pass2_start) / 60.0
        log.info("===== Pass 2 complete in %.2f min =====", pass2_minutes)
    else:
        log.info("Pass 2: nothing to run (resume covered all cells)")

    # ===== Summary =====
    # Per-cell rc combines Pass 1 + Pass 2: a cell's final rc is the WORST
    # of its two pass rcs (0 < 1 < 2). Resume-skipped cells contribute 0.
    final_rcs: dict[tuple[str, str, int], int] = {}
    for cell, source, seed in all_jobs:
        key = (cell.key, source, seed)
        p1 = pass1_rcs.get(key, 0)  # skipped → 0
        p2 = pass2_rcs.get(key, 0)
        final_rcs[key] = max(p1, p2)

    rc_counts: dict[int, int] = {}
    for rc in final_rcs.values():
        rc_counts[rc] = rc_counts.get(rc, 0) + 1
    log.info(
        "Two-pass sweep complete: %d cells, rc distribution: %s",
        len(final_rcs),
        rc_counts,
    )

    summary_path = args.slab_root / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "job_count": job_count,
        "ran_pass1": len(pass1_rcs),
        "ran_pass2": len(pass2_rcs),
        "rc_counts": {str(k): v for k, v in rc_counts.items()},
        "per_cell": [
            {
                "cell": k[0],
                "source": k[1],
                "seed": k[2],
                "pass1_rc": pass1_rcs.get(k, 0),
                "pass2_rc": pass2_rcs.get(k, 0),
                "final_rc": v,
            }
            for k, v in sorted(final_rcs.items())
        ],
        "skip_summary": skip_summary,
        "two_pass_mode": True,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log.info("Wrote sweep summary: %s", summary_path)

    if rc_counts and rc_counts.get(0, 0) == 0:
        log.error("All %d cells failed; returning rc=1", len(final_rcs))
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    if args.issue != 397:
        log.error("This dispatcher is task-397-specific; got --issue=%d", args.issue)
        return 2

    # Round 11 — load .env at dispatcher entry so HF_TOKEN / WANDB_API_KEY
    # / ANTHROPIC_API_KEY are in os.environ BEFORE any HF Hub / WandB /
    # Anthropic call. The brief cited `setup_env()` from utils.py per
    # CLAUDE.md, but that function does not exist in this repo's utils.py
    # — `load_dotenv` from `orchestrate.env` is the canonical helper used
    # elsewhere (e.g. factor_screen_365.__main__). It loads .env + sets
    # HF_HOME to /workspace/.cache/huggingface on pods (project-local
    # cache off-pod).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Repo root resolves from this file's path so the script is independent
    # of cwd. The script lives at <repo>/scripts/, so the parent is the repo
    # root in both the main checkout AND a worktree.
    repo_root = Path(__file__).resolve().parent.parent

    if args.mode == "smoke":
        return run_smoke_phase(args, repo_root=repo_root)
    if args.mode == "sweep":
        return run_sweep_phase(args, repo_root=repo_root)
    log.error("Unknown mode %r", args.mode)
    return 2


if __name__ == "__main__":
    sys.exit(main())
