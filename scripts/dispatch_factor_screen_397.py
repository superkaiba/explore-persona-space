"""Dispatcher for task #397 — five-factor recipe-selectivity screen, v4 recipe.

Two phases (plan v4 §5.7 + §5.8):

  Phase A — cell-1 smoke test gate. Before the 324-run sweep dispatches, run
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
            - Round-robin across 8x H100 (concurrent training capped at 8/8
              post-round-6 vLLM-LoRA disk savings; plan v4 §12's 6/8 was a
              merge-dir mitigation that the round-6 merge elimination
              superseded).
            - Per cell: train -> log-prob eval at 6 intermediate checkpoints
              (2 marker variants at final) -> vLLM --enable-lora sampled eval
              at final -> upload adapter to HF Hub -> rm -rf checkpoint-NNN/
              for that cell. No merge step (round 6).

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
        --slab-root eval_results/issue_397 \
        --num-gpus 8 \
        --max-concurrent-train 8 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("dispatch_factor_screen_397")

# Decision bands from plan v4 §5.7 (per-checkpoint log-prob eval wall time).
SMOKE_PASS_THRESHOLD_MINUTES: float = 2.0
SMOKE_FAIL_THRESHOLD_MINUTES: float = 10.0

# Plan v4 §12 originally specified 6/8 concurrent training for MooseFS
# quota mitigation because the per-cell merge step added ~14 GB per cell
# (peak 8 * 17 GB = 136 GB would exceed the ~130 GB per-pod quota).
# Round 6 dropped the merge step in favor of vLLM ``--enable-lora``, so
# peak per-cell disk is now ~3 GB (adapter + checkpoints) and 8/8 fits
# comfortably (peak 8 * 3 = ~24 GB). The cap defaults to 8 from Round 6
# onward; the --max-concurrent-train arg lets the user dial it back when
# explicitly wanted.
DEFAULT_MAX_CONCURRENT_TRAIN: int = 8
DEFAULT_NUM_GPUS: int = 8

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
    p.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    p.add_argument(
        "--max-concurrent-train",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_TRAIN,
        help=(
            "Concurrent training cap. Round 6 relaxed plan v4 §12's 6/8 to 8/8 "
            "after the merge step was eliminated; default 8."
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


def post_marker_via_task_py(issue: int, kind: str, note: str, *, repo_root: Path) -> None:
    """Shell out to scripts/task.py post-marker from the repo root.

    Per CLAUDE.md: task.py mutations must originate from the repo root
    (NOT a worktree); the canonical resolver branch-guards to main and
    will refuse loudly on a worktree branch.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/task.py",
        "post-marker",
        str(issue),
        kind,
        "--note",
        note,
    ]
    log.info("Posting marker %s for task #%d via %s", kind, issue, " ".join(cmd[:5]))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


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
    post_marker_via_task_py(args.issue, kind, note, repo_root=repo_root)

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
        and not has_recent_smoke_pass_marker(args.issue, repo_root=repo_root)
    ):
        log.error(
            "Phase B refused: no recent epm:smoke-pass v1 marker on task #%d. "
            "Run Phase A (--mode smoke) first or pass --skip-smoke-pass-check "
            "to override (only with a documented out-of-band smoke run).",
            args.issue,
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
        "Concurrency: %d/%d GPUs for training, %d reserved for eval-only (plan v4 §12)",
        args.max_concurrent_train,
        args.num_gpus,
        max(args.num_gpus - args.max_concurrent_train, 0),
    )

    if args.dry_run:
        log.info("--dry-run: enumeration only; %d jobs would be dispatched", job_count)
        return 0

    return _dispatch_sweep_jobs(
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


def has_recent_smoke_pass_marker(issue: int, *, repo_root: Path) -> bool:
    """Return True if the task's events.jsonl has a `kind=='epm:smoke-pass'` row.

    Phase B gate: refuses to dispatch the 324-run sweep without a prior
    smoke-pass on the same task.
    """
    cmd = ["uv", "run", "python", "scripts/task.py", "find", str(issue)]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    task_dir = Path(proc.stdout.strip())
    events_path = task_dir / "events.jsonl"
    if not events_path.exists():
        return False
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") == "epm:smoke-pass":
            return True
    return False


# ---------------------------------------------------------------------------
# Sweep-level resume (Round 6)
# ---------------------------------------------------------------------------

HF_ADAPTER_REPO: str = "superkaiba1/explore-persona-space"


def is_cell_complete_locally(cell_output_dir: Path) -> bool:
    """Return True if ``cell_output_dir / 'metrics.json'`` parses cleanly.

    Round 6 resume probe (local branch). The per-cell ``run_one_cell``
    writes ``metrics.json`` as its LAST step (after vLLM sampled eval +
    HF upload verify + cleanup); presence + parsability is the canonical
    "this cell finished cleanly" sentinel.

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
    by 3 inside the sweep loop in `_dispatch_sweep_jobs`.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import (
        valid_cells_per_source,
    )

    return list(valid_cells_per_source())


def _dispatch_sweep_jobs(  # noqa: C901 — orchestrator: resume + launch + drain + summary
    *,
    sources: list[str],
    seeds: list[int],
    cells: list[Any],
    args: argparse.Namespace,
    repo_root: Path,
) -> int:
    """Round-robin assign (cell, source, seed) jobs across the GPU pool.

    Round 5 implementation, Round 6 update: default concurrency cap
    relaxed from 6/8 → 8/8. Plan v4 §12's 6/8 was a merge-dir disk-quota
    mitigation; Round 6 eliminated the merge step (vLLM ``--enable-lora``
    consumes the adapter directly), so peak per-cell disk dropped from
    ~17 GB to ~3 GB and 8/8 fits comfortably under MooseFS ~130 GB.

    Builds a queue of (cell, source, seed) tuples, spawns one
    ``subprocess.Popen`` per cell pinned to a single GPU, caps concurrent
    training at ``args.max_concurrent_train`` (default 8), waits for a
    free GPU when the cap is hit, and drains all in-flight processes at
    end.

    Per-cell exit codes (from ``run_one_cell.main``):
      - 0 = ok
      - 1 = train or eval crashed (exception caught + logged)
      - 2 = HF upload verification FAILED (local weights preserved)
      - 3 = unexpected (CLI parse error, missing pool, etc.)

    Single-cell failures DO NOT kill the sweep — the dispatcher logs the
    failure and continues. The aggregate per-cell exit-code summary is
    logged at sweep end + recorded in a sweep-level JSON sidecar.
    """
    job_count = len(sources) * len(seeds) * len(cells)
    log.info(
        "Enumerating %d sweep jobs (sources=%d seeds=%d cells=%d)",
        job_count,
        len(sources),
        len(seeds),
        len(cells),
    )

    # Cap concurrent training at max_concurrent_train; the GPU pool
    # therefore exposes only that many slots even when args.num_gpus is
    # larger (the remaining GPUs would have been reserved for eval-only
    # in the launch-layer follow-up; per the brief, Phase B's
    # eval-on-separate-GPU split is not required to land this round, so
    # the unused GPUs simply idle).
    gpu_pool = list(range(min(args.max_concurrent_train, args.num_gpus)))
    if not gpu_pool:
        raise ValueError(
            f"GPU pool is empty: max_concurrent_train={args.max_concurrent_train}, "
            f"num_gpus={args.num_gpus}. Cannot dispatch."
        )
    log.info(
        "GPU pool: %d slots (max_concurrent_train=%d, num_gpus=%d)",
        len(gpu_pool),
        args.max_concurrent_train,
        args.num_gpus,
    )

    running: dict[int, subprocess.Popen] = {}
    per_cell_rc: dict[tuple[str, str, int], int] = {}

    # Canonical iteration order: source-major, then seed, then cell. This
    # keeps a single source's runs clustered in time so HF Hub batches
    # adapter uploads per-source — easier to inspect mid-sweep.
    all_jobs: list[tuple[Any, str, int]] = [
        (cell, source, seed) for source in sources for seed in seeds for cell in cells
    ]

    # Round 6: sweep-level resume. Filter out cells already complete
    # locally (metrics.json) AND/OR on HF Hub (adapter under the canonical
    # path). LOUD-FAIL on inconsistent state when --resume-source=both.
    skip_summary: dict[str, int] = {"skipped_local": 0, "skipped_hub": 0, "skipped_both": 0}
    if getattr(args, "no_resume", False):
        log.info("--no-resume set; running all %d cells from scratch", len(all_jobs))
        jobs_to_run = all_jobs
        skip_summary["queued"] = len(jobs_to_run)
    else:
        # Pre-fetch HF Hub file list once (avoids 324 round-trips).
        hub_files_cache = None
        if args.resume_source in ("hub", "both"):
            hub_files_cache = _fetch_hub_adapter_index()
        jobs_to_run, skip_summary = filter_jobs_for_resume(
            all_jobs,
            slab_root=args.slab_root,
            resume_source=args.resume_source,
            hub_files_cache=hub_files_cache,
        )
        skipped_total = (
            skip_summary["skipped_local"]
            + skip_summary["skipped_hub"]
            + skip_summary["skipped_both"]
        )
        log.info(
            "Resuming sweep: %d/%d cells already complete, launching %d remaining "
            "(skipped: local=%d hub=%d both=%d; resume_source=%s)",
            skipped_total,
            len(all_jobs),
            len(jobs_to_run),
            skip_summary["skipped_local"],
            skip_summary["skipped_hub"],
            skip_summary["skipped_both"],
            args.resume_source,
        )
        # Post epm:sweep-resume marker with the counts so the orchestrator
        # has a record of what was skipped.
        if skipped_total > 0:
            try:
                post_marker_via_task_py(
                    args.issue,
                    "epm:sweep-resume",
                    f"Sweep resume: {skipped_total} of {len(all_jobs)} cells already complete; "
                    f"launching {len(jobs_to_run)} remaining. Skipped: "
                    f"local={skip_summary['skipped_local']}, hub={skip_summary['skipped_hub']}, "
                    f"both={skip_summary['skipped_both']}, "
                    f"resume_source={args.resume_source}.",
                    repo_root=repo_root,
                )
            except subprocess.CalledProcessError as e:
                log.warning("Failed to post epm:sweep-resume marker (%s); continuing", e)

    job_count_filtered = len(jobs_to_run)
    for job_index, (cell, source, seed) in enumerate(jobs_to_run, start=1):
        cell_dir = _cell_output_dir(args.slab_root, cell_key=cell.key, source=source, seed=seed)
        log.info(
            "[%d/%d] queueing cell=%s source=%s seed=%d e=%d -> %s",
            job_index,
            job_count_filtered,
            cell.key,
            source,
            seed,
            cell.e,
            cell_dir,
        )
        if args.dry_run:
            continue
        gpu = _wait_for_free_gpu(running, gpu_pool, per_cell_rc=per_cell_rc)
        proc = _launch_cell_subprocess(
            cell=cell,
            source=source,
            seed=seed,
            gpu_id=gpu,
            cell_output_dir=cell_dir,
            args=args,
            repo_root=repo_root,
        )
        running[gpu] = proc
        log.info(
            "Launched cell=%s source=%s seed=%d on GPU %d (pid=%d)",
            cell.key,
            source,
            seed,
            gpu,
            proc.pid,
        )

    if args.dry_run:
        return 0

    # Drain all in-flight processes. ``_wait_for_free_gpu`` returns the
    # FIRST empty slot when one exists, so we cannot reuse it for drain —
    # we'd infinite-loop on the empty slot. Drain by polling each remaining
    # Popen directly until it exits.
    while running:
        for gpu in list(running.keys()):
            proc = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            key = (
                getattr(proc, "_cell_key", "?"),
                getattr(proc, "_source", "?"),
                getattr(proc, "_seed", -1),
            )
            per_cell_rc[key] = rc
            if rc != 0:
                log.warning(
                    "Drain: cell %s source=%s seed=%s on GPU %d exited with rc=%d",
                    key[0],
                    key[1],
                    key[2],
                    gpu,
                    rc,
                )
            else:
                log.info(
                    "Drain: cell %s source=%s seed=%s on GPU %d completed cleanly (rc=0)",
                    key[0],
                    key[1],
                    key[2],
                    gpu,
                )
            running.pop(gpu, None)
        if running:
            time.sleep(2.0)

    # Summary + per-cell JSON sidecar for downstream resume / debugging.
    rc_counts: dict[int, int] = {}
    for rc in per_cell_rc.values():
        rc_counts[rc] = rc_counts.get(rc, 0) + 1
    log.info(
        "Sweep complete: %d/%d cells ran, rc distribution: %s",
        len(per_cell_rc),
        job_count,
        rc_counts,
    )
    summary_path = args.slab_root / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "job_count": job_count,
        "ran": len(per_cell_rc),
        "rc_counts": {str(k): v for k, v in rc_counts.items()},
        "per_cell": [
            {"cell": k[0], "source": k[1], "seed": k[2], "rc": v}
            for k, v in sorted(per_cell_rc.items())
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log.info("Wrote sweep summary: %s", summary_path)

    # Sweep itself returns 0 unless ALL cells failed; per-cell failures
    # are surfaced through the summary JSON, not the top-level rc.
    if rc_counts and rc_counts.get(0, 0) == 0:
        log.error("All %d cells failed; returning rc=1", len(per_cell_rc))
        return 1
    return 0


def _wait_for_free_gpu(
    running: dict[int, subprocess.Popen],
    gpu_pool: list[int],
    *,
    per_cell_rc: dict[tuple[str, str, int], int] | None = None,
    poll_interval_seconds: float = 2.0,
) -> int:
    """Block until a GPU in ``gpu_pool`` becomes free; return its id.

    Mirror of factor_screen_365.dispatch._wait_for_free_gpu, extended with
    per-cell exit-code bookkeeping so the sweep summary can report
    failures without spinning up a separate dispatcher state machine.

    Each Popen carries a ``_cell_key``, ``_source``, ``_seed`` attribute
    stamped by ``_launch_cell_subprocess`` so a finished proc can be
    indexed back to its cell.
    """
    while True:
        for gpu in gpu_pool:
            proc = running.get(gpu)
            if proc is None:
                return gpu
            rc = proc.poll()
            if rc is not None:
                # Process finished; harvest its rc + free the GPU.
                key = (
                    getattr(proc, "_cell_key", "?"),
                    getattr(proc, "_source", "?"),
                    getattr(proc, "_seed", -1),
                )
                if per_cell_rc is not None:
                    per_cell_rc[key] = rc
                if rc != 0:
                    log.warning(
                        "Cell %s source=%s seed=%s on GPU %d exited with rc=%d "
                        "(preserving local artifacts)",
                        key[0],
                        key[1],
                        key[2],
                        gpu,
                        rc,
                    )
                else:
                    log.info(
                        "Cell %s source=%s seed=%s on GPU %d completed cleanly (rc=0)",
                        key[0],
                        key[1],
                        key[2],
                        gpu,
                    )
                running.pop(gpu, None)
                return gpu
        time.sleep(poll_interval_seconds)


def _launch_cell_subprocess(
    *,
    cell: Any,
    source: str,
    seed: int,
    gpu_id: int,
    cell_output_dir: Path,
    args: argparse.Namespace,
    repo_root: Path,
) -> subprocess.Popen:
    """Launch one (cell, source, seed) training+eval subprocess on ``gpu_id``.

    Round 5 implementation. Spawns ``python -m
    explore_persona_space.experiments.factor_screen_397.run_one_cell`` with
    the per-cell args. The subprocess body does data-prep → train → eval →
    sampled-eval → HF-upload-verify → cleanup all in one process; the
    dispatcher only handles GPU scheduling + per-cell exit-code harvesting.

    GPU pinning (per the +gpu_id memory note): both env CUDA_VISIBLE_DEVICES
    AND ``--gpu-id`` are passed. The env makes vLLM + HF Transformers see
    only one device; the kwarg threads to TrainLoraConfig.gpu_id so
    train/sft.py:479's CVD clobber lands on the right device.

    The Popen carries ``_cell_key`` / ``_source`` / ``_seed`` attributes
    so ``_wait_for_free_gpu`` can index a finished proc back to its cell
    in the per-cell exit-code dict.

    Returns the Popen handle (NOT blocking).
    """
    cell_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell_output_dir / "dispatcher.log"

    cmd = [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.factor_screen_397.run_one_cell",
        "--cell",
        cell.key,
        "--source",
        source,
        "--seed",
        str(seed),
        "--gpu-id",
        str(gpu_id),
        "--pool-dir",
        str(args.pool_dir),
        "--output-dir",
        str(cell_output_dir),
        "--marker-token",
        args.marker_token,
        "--save-every-n-steps",
        str(args.save_every_n_steps),
        "--pos-per-source",
        str(args.pos_per_source),
        "--lr",
        str(args.lr),
        "--warmup-ratio",
        str(args.warmup_ratio),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Mitigation per agent-memory `runpod_moosefs_quota` note: skip the
    # WandB Artifacts intermediate upload so peak per-pod disk stays under
    # the MooseFS quota. The per-cell HF Hub upload is still the load-
    # bearing artifact path (verified before cleanup).
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    # Per-cell stdout/stderr → log_path. Use raw open (NOT a context manager)
    # because the fd must stay open after this function returns so the
    # subprocess keeps writing to it. The OS closes the fd when the subprocess
    # exits + GC reaps log_handle; no leak in normal flow.
    log_handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 — subprocess inherits fd
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(repo_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    # Stamp identifiers so _wait_for_free_gpu can rebuild the (cell, source,
    # seed) key from the Popen when the process exits.
    proc._cell_key = cell.key  # type: ignore[attr-defined]
    proc._source = source  # type: ignore[attr-defined]
    proc._seed = seed  # type: ignore[attr-defined]
    return proc


def build_run_one_cell_command(
    *,
    cell_key: str,
    source: str,
    seed: int,
    gpu_id: int,
    pool_dir: Path,
    output_dir: Path,
    marker_token: str = DEFAULT_MARKER_TEXT,
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    pos_per_source: int = 400,
    lr: float = 1e-4,
    warmup_ratio: float = 0.10,
    python_executable: str | None = None,
) -> list[str]:
    """Build the ``python -m ... run_one_cell ...`` argv list.

    Extracted as a public helper so the wiring test can assert the exact
    command shape without running ``subprocess.Popen``.
    """
    return [
        python_executable or sys.executable,
        "-m",
        "explore_persona_space.experiments.factor_screen_397.run_one_cell",
        "--cell",
        cell_key,
        "--source",
        source,
        "--seed",
        str(seed),
        "--gpu-id",
        str(gpu_id),
        "--pool-dir",
        str(pool_dir),
        "--output-dir",
        str(output_dir),
        "--marker-token",
        marker_token,
        "--save-every-n-steps",
        str(save_every_n_steps),
        "--pos-per-source",
        str(pos_per_source),
        "--lr",
        str(lr),
        "--warmup-ratio",
        str(warmup_ratio),
    ]


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
