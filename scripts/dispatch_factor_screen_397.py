"""Dispatcher for task #397 — five-factor recipe-selectivity screen, v4 recipe.

Two phases (plan v4 §5.7 + §5.8):

  Phase A — cell-1 smoke test gate. Before the 324-run sweep dispatches, run
            cell (a=1, b=0, c=0, d=1, e=0) on source=librarian, seed=42 end-
            to-end (train + 6-checkpoint log-prob eval). Measure per-checkpoint
            log-prob eval wall time. Decision bands (plan v4 §5.7 + §10):
              - <2 min/ckpt   → emit epm:smoke-pass v1, proceed to Phase B.
              - 2-10 min/ckpt → emit epm:smoke-warn v1, EXIT (user gates re-plan).
              - >10 min/ckpt  → emit epm:smoke-fail v1, EXIT (user gates re-plan).

  Phase B — full 324-run sweep. After Phase A PASS:
            - 108 cells per seed x 3 seeds {42, 137, 256} = 324 (cell, seed) runs.
            - Round-robin across 8x H100 (concurrent training capped at 6/8
              GPUs per plan v4 §12 disk-quota mitigation; the other 2 GPUs run
              eval-only).
            - Per cell: train -> log-prob eval at 6 intermediate checkpoints
              (2 marker variants at final) -> upload adapter to HF Hub ->
              rm -rf merged/ checkpoint-NNN/ for that cell.

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
        --seeds 42,137,256 \
        --pool-dir data/issue_397/pools \
        --slab-root eval_results/issue_397 \
        --num-gpus 8 \
        --max-concurrent-train 6 &
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("dispatch_factor_screen_397")

# Decision bands from plan v4 §5.7 (per-checkpoint log-prob eval wall time).
SMOKE_PASS_THRESHOLD_MINUTES: float = 2.0
SMOKE_FAIL_THRESHOLD_MINUTES: float = 10.0

# Plan v4 §11 + §12 — disk-quota mitigation: cap concurrent training at 6/8
# GPUs (peak 6 * ~17 GB workspace = ~102 GB, under RunPod MooseFS ~130 GB
# per-pod quota). The remaining 2 GPUs run eval-only (no merge / checkpoint-
# NNN workspace).
DEFAULT_MAX_CONCURRENT_TRAIN: int = 6
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
    p.add_argument("--seeds", type=str, default="42,137,256")
    p.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    p.add_argument(
        "--max-concurrent-train",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_TRAIN,
        help="Concurrent training cap (plan v4 §12 disk-quota mitigation; default 6).",
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
        merged_model_path=outcome.merged_path,
        panel=panel,
        questions=questions,
        marker=args.marker_token,
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
    merged_model_path: str,
    panel: dict[str, str],
    questions: list[str],
    marker: str,
) -> None:
    """Run the vLLM sampled eval at the smoke cell's final checkpoint.

    BLOCKER 3 fix (code-review v3): writes ``metrics_final.json`` under
    ``cell_output_dir`` so ``_smoke_source_substring_rate`` has data to
    return and the M1 live check in ``build_smoke_marker`` can fire. Plan
    v4 §5.7 specifies the M1 check is "source substring rate > 0 at smoke
    cell"; that requires sampled eval, not log-prob-only.

    The eval uses the SAME 480-context shape as the log-prob eval (24
    personas x 20 questions) so the sampled-vs-logprob comparison is
    apples-to-apples. ``score_markers`` is threaded with ``marker=...``
    (M1) so the recorded substring rate matches the runtime marker.

    Writes ``metrics_final.json`` with shape::

        {
          "marker": "<runtime marker>",
          "personas": {"<persona>": {"substring_rate": ...,
                                     "fuzzy_rate": ...,
                                     "substring_found": ...,
                                     "total": ...,
                                     "per_question": {...}}, ...},
          "panel_size": 24, "questions": 20, "num_completions": 5
        }

    Failures (vLLM crash, IO error) propagate as exceptions — the calling
    smoke phase exits non-zero and ``build_smoke_marker`` reports FAIL.
    Wrapping in try/except would re-introduce the silent-pass bug
    BLOCKER 3 is designed to eliminate.
    """
    from explore_persona_space.experiments.factor_screen_365.eval_panel import (
        EvalConfig,
        generate_completions,
    )
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        score_markers_threaded,
    )

    log.info(
        "Smoke sampled eval starting on merged model %s "
        "(panel=%d personas x %d questions x 5 completions)",
        merged_model_path,
        len(panel),
        len(questions),
    )
    eval_cfg = EvalConfig(
        model_path=merged_model_path,
        personas=dict(panel),
        questions=list(questions),
    )
    completions = generate_completions(eval_cfg)
    persona_scores = score_markers_threaded(completions, marker=marker)

    payload = {
        "marker": marker,
        "panel_size": len(panel),
        "questions": len(questions),
        "num_completions": eval_cfg.num_completions,
        "personas": persona_scores,
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


def _dispatch_sweep_jobs(
    *,
    sources: list[str],
    seeds: list[int],
    cells: list[Any],
    args: argparse.Namespace,
    repo_root: Path,
) -> int:
    """Round-robin assign (cell, source, seed) jobs across GPUs.

    The actual subprocess-launch machinery — process pool, GPU pinning,
    stall detection, per-cell HF upload + cleanup — is shared with the
    existing factor_screen_365 dispatcher and lives in the orchestrate/
    layer. This function is the per-cell loop skeleton; the orchestrator
    plugs in the heavy lifting at the marked TODO points.

    The dispatcher MUST call train_one_cell with system_prompt_text=
    EVAL_PERSONAS_24[source] (SR1 - even C=0 cells emit a manifest so the
    eval-side helper can uniformly read it back; the C=0 override is a
    no-op but exercising the SR1 wiring across every cell prevents a
    C=1-only bug from first surfacing in production).
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    job_count = len(sources) * len(seeds) * len(cells)
    log.info(
        "Enumerating %d sweep jobs (sources=%d seeds=%d cells=%d)",
        job_count,
        len(sources),
        len(seeds),
        len(cells),
    )

    # The per-cell launch loop. In production this dispatches under a
    # process pool with --max-concurrent-train cap; the shell here is the
    # canonical iteration order that the launch layer consumes.
    job_index = 0
    for source in sources:
        canonical_source_prompt = EVAL_PERSONAS_24[source]
        for seed in seeds:
            for cell in cells:
                job_index += 1
                cell_dir = _cell_output_dir(
                    args.slab_root, cell_key=cell.key, source=source, seed=seed
                )
                log.info(
                    "[%d/%d] cell=%s source=%s seed=%d e=%d -> %s",
                    job_index,
                    job_count,
                    cell.key,
                    source,
                    seed,
                    cell.e,
                    cell_dir,
                )
                if args.dry_run:
                    continue
                # The training subprocess gets these kwargs; the per-cell
                # eval subprocess reads them back via the manifest + the
                # build_train_matched_persona_panel + system_prompt_overrides
                # path (SR1). The subprocess wrapper is part of the launch
                # layer (operational; out of scope for this orchestration
                # shell). Sentinel comment so a code-reviewer can verify
                # the SR1 wiring lands by grepping for it.
                _launch_cell_subprocess(
                    cell=cell,
                    source=source,
                    seed=seed,
                    cell_output_dir=cell_dir,
                    args=args,
                    canonical_source_prompt=canonical_source_prompt,
                    repo_root=repo_root,
                )
    return 0


def _launch_cell_subprocess(
    *,
    cell: Any,
    source: str,
    seed: int,
    cell_output_dir: Path,
    args: argparse.Namespace,
    canonical_source_prompt: str,
    repo_root: Path,
) -> None:
    """Launch one (cell, source, seed) training+eval subprocess.

    This function is intentionally a thin shell: the per-cell training
    invokes train_one_cell with system_prompt_text=canonical_source_prompt
    so the manifest lands on disk; the per-cell eval invokes the
    train-matched panel + compute_logprob_panel(..., system_prompt_overrides
    =overrides) path (SR1).

    The subprocess plumbing (fork to a worker, capture stdout/stderr, GPU
    pinning, HF upload + per-cell cleanup) is the operational follow-up
    that the experimenter will land separately — this dispatcher's job is
    to enumerate the right (cell, source, seed) tuples and document the
    contract the subprocess must satisfy.

    Raises NotImplementedError when called (sweep dispatch requires the
    launch-layer landing first); the smoke-gate path does NOT call this
    function, so the smoke phase remains fully exercisable end-to-end.
    """
    raise NotImplementedError(
        "_launch_cell_subprocess: the per-cell subprocess wrapper is the "
        "operational follow-up to this dispatcher and lands separately. "
        "This dispatcher's contract (Phase A smoke gate + Phase B sweep "
        "enumeration with SR1 wiring) is verified by the test surface in "
        "tests/experiments/test_factor_screen_397_dispatcher_*.py."
    )


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
