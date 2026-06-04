# ruff: noqa: C901, RUF001, RUF003  # main is a linear worker; em-dash + Qwen marker " ※" + Greek ΔG/α + × intentional
#!/usr/bin/env python3
"""Task #477 — single (cell, seed, lr, phase) worker: build → train → eval.

Thin fork of scripts/i472_run_cell.py with:
  * --lr <float> (calibrated LR threaded from dispatcher Phase 2.5);
  * --phase {calibration, main, implant_sweep} (drives sentinel kind + the
    "save calibration row vs full trajectory" branch);
  * uses CELL_SPECS_477 (#477's own registry) when calling build_cell +
    negatives_for_cell via the new `cell_specs` kwarg (backward-compat for #472);
  * keeps the round-3 GPU-pin discipline + the gc.collect()/empty_cache()
    teardown of the in-process LoRA before the nested vLLM eval subprocess.

Within one (cell, seed) the worker switches frameworks at most once (HF Trainer
for training, then a NESTED subprocess for the vLLM+HF eval rig). The nested
boundary guarantees the OS reaps vLLM workers (CLAUDE.md vLLM teardown gotcha).

GPU pinning: dispatcher passes ``--gpu-id <g>``; ``train/sft.py`` SETS
``CUDA_VISIBLE_DEVICES=str(g)`` against the FULL host enumeration. The nested
eval subprocess inherits that CVD from os.environ — vLLM + HF KL run on
physical GPU g (round-3 #472 sharding fix preserved).

Phase semantics:
  * `calibration` — train, eval ONLY at the final checkpoint (terminal-only,
    cheaper), emit calibration_row.json with source_self_delta_g + emission_p.
  * `main` — train + full 6-checkpoint trajectory eval (same as #472).
  * `implant_sweep` — train + full trajectory eval at fixed count, varied LR.

Usage (driven by dispatch_neg_geometry_477.py; --gpu-id is the assigned physical GPU):
    uv run python scripts/i477_run_cell.py \
        --cell c477_calib_negp_4 --seed 42 --gpu-id 3 --lr 1e-5 --phase calibration \
        --slab-root eval_results/issue_477 --runs-root /workspace/runs/issue_477 \
        --log-dir /workspace/logs [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i477.run_cell")

# v4 step-lever pivot adds three new phases ("step_calibration", "main_v4",
# "implant_sweep_v4"); legacy v2 phases ("calibration", "main", "implant_sweep")
# are kept so the --legacy-lr-calibration dispatcher path stays byte-identical.
PHASES = (
    "calibration",
    "main",
    "implant_sweep",
    "step_calibration",
    "main_v4",
    "implant_sweep_v4",
    # v6 rank pivot: Cal-A (rank-calibration) + Cal-A0 (rank_control) — both
    # share the dense early-step grid + slot-fix port. Identical dispatch
    # shape as step_calibration; the only difference is the (rank, alpha)
    # values come from the dispatcher (Cal-A reads RANK_ALPHA_MAP_V5; Cal-A0
    # always r=32 / α=64).
    "rank_calibration",
    "rank_control",
)
TASK_ID = 477


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": TASK_ID,
                "by": "i477_run_cell",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note),
            },
            indent=2,
        )
    )


def mean_bystander_delta_g(checkpoint: dict) -> float:
    """Q_eval-mean per persona, then mean across personas (legacy DV-A).

    Module-scope so tests can pin the rule without spawning the worker.
    """
    means: list[float] = []
    for _persona, per_q in checkpoint["held_out"].items():
        deltas = [float(v["delta_g"]) for v in per_q.values()]
        if deltas:
            means.append(sum(deltas) / len(deltas))
    return float(sum(means) / len(means)) if means else 0.0


def select_checkpoint_near_step(
    traj_dict: dict,
    requested_step: int,
    *,
    cell_slug: str = "<unknown>",
) -> tuple[int, dict, int]:
    """Pick the trajectory checkpoint whose step is closest to ``requested_step``.

    Returns ``(actual_step, checkpoint_dict, offset)``. Fails loud if the
    nearest checkpoint is farther than ``max(1, 5% of requested)`` steps away
    — that signals the trainer never produced a checkpoint anywhere near the
    requested step (e.g. context window dropped it entirely), and we would
    otherwise silently read a checkpoint at a totally wrong training amount.

    Args:
        traj_dict: trajectory.json contents (dict with ``checkpoints`` list).
        requested_step: optimizer step the picker / dispatcher requested.
        cell_slug: prefix for error messages so failures attribute correctly.
    """
    steps_present = [
        (int(ck["step"]), ck) for ck in traj_dict["checkpoints"] if ck.get("step") is not None
    ]
    if not steps_present:
        raise RuntimeError(
            f"[{cell_slug}] picked-step resolution: trajectory.json has no "
            f"checkpoint with a 'step' field; cannot resolve requested_step="
            f"{requested_step}. Investigate i472_eval_trajectory.py."
        )
    # Tie breaks on the LOWER step (deterministic).
    actual_step, picked_ck = min(
        steps_present,
        key=lambda sc: (abs(sc[0] - int(requested_step)), sc[0]),
    )
    offset = actual_step - int(requested_step)
    tolerance = max(1, round(0.05 * int(requested_step)))
    if abs(offset) > tolerance:
        raise RuntimeError(
            f"[{cell_slug}] picked-step resolution: no checkpoint within "
            f"{tolerance} steps of requested_step={requested_step}. Nearest "
            f"step in trajectory = {actual_step} (offset {offset}); "
            f"checkpoints present = {sorted(s for s, _ in steps_present)}. "
            f"The trainer did not produce the expected checkpoint — "
            f"investigate the context window AND drop_last semantics."
        )
    return actual_step, picked_ck, offset


def picked_step_kl_fields(
    picked_ck: dict,
    *,
    cell_slug: str = "<unknown>",
) -> dict:
    """Extract the 6 picked-step DV fields the v4 analyze partials consume.

    The picked checkpoint MUST carry ``source_self.emission_p`` (the rig
    writes it; fail loud if missing rather than silently returning a stale
    proxy).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        aggregate_bystander_full_vocab_kl,
        aggregate_bystander_marker_channel_kl,
        aggregate_source_self_marker_channel_kl,
    )

    picked_src = picked_ck["source_self"]
    if "emission_p" not in picked_src:
        raise RuntimeError(
            f"[{cell_slug}] picked checkpoint missing emission_p in "
            f"source_self; keys present = {sorted(picked_src.keys())!r}."
        )
    return {
        "source_self_marker_channel_kl_at_picked_step": (
            aggregate_source_self_marker_channel_kl(picked_ck)
        ),
        "mean_bystander_marker_channel_kl_at_picked_step": (
            aggregate_bystander_marker_channel_kl(picked_ck)
        ),
        "mean_bystander_full_vocab_kl_at_picked_step": (
            aggregate_bystander_full_vocab_kl(picked_ck)
        ),
        "source_self_delta_g_at_picked_step": float(picked_src["delta_g_mean"]),
        "source_emission_p_at_picked_step": float(picked_src["emission_p"]),
        "mean_bystander_delta_g_at_picked_step": mean_bystander_delta_g(picked_ck),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="i477_run_cell — per-cell worker.")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True, help="Calibrated LR for this cell.")
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_477"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_477"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    # Persona bank + R artifacts REUSE the #472 data (plan §4: persona bank /
    # centroids / R-generate / base-panel are REUSE).
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument(
        "--r-train-path", type=Path, default=Path("data/issue_472/on_policy_R/R_train.json")
    )
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny slice: fewer steps, 2 checkpoints.")
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL (smoke speed-up).")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "ASSIGNED physical GPU index (round-3 #472). Threaded to "
            "train_one_cell(gpu_id=...); train/sft.py SETS CUDA_VISIBLE_DEVICES "
            "to this so the cell + its nested eval subprocess run on physical GPU "
            "<gpu-id>."
        ),
    )
    # ── v4 step-lever flags (plan v4 §4 i477_run_cell row). ──────────────────
    ap.add_argument(
        "--target-steps",
        type=str,
        default="",
        help=(
            "CSV of integer optimizer-step checkpoints for the v4 step-"
            "calibration phase (e.g. '1,2,4,8,16,32,64'). When non-empty AND "
            "phase=step_calibration, the worker computes per-cell fractions "
            "via step_fractions(target_steps, max_steps) at 4-dp precision."
        ),
    )
    ap.add_argument(
        "--picked-step",
        type=int,
        default=None,
        help=(
            "v4 main-phase: the headline checkpoint s* picked by Phase 2.5. "
            "The worker computes the clamped 3-checkpoint context window "
            "{floor(s*/2), s*, min(2*s*, max_steps)} via "
            "main_phase_context_window. Only used when phase=main_v4."
        ),
    )
    ap.add_argument(
        "--implant-steps",
        type=str,
        default="",
        help=(
            "v4r2 implant_sweep_v4 phase: CSV of non-terminal optimizer-step "
            "checkpoints (e.g. '16,64'). The terminal step (frac=1.0) is added "
            "automatically. Used only when phase=implant_sweep_v4."
        ),
    )
    # ── v6 rank pivot CLI flags (plan v6 §4.4 i477_run_cell row). ────────────
    ap.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help=(
            "v6: per-cell LoRA rank override. Default None = use the module "
            "constant (r=32 = v4 byte-identical). When supplied, MUST be "
            "paired with --lora-alpha; the dispatcher enforces M2 "
            "(α=RANK_ALPHA_MAP_V5[r] for r ∈ {2,4,8}, OR 64 for r=32)."
        ),
    )
    ap.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help=(
            "v6: per-cell LoRA alpha override. Default None = use the module "
            "constant (α=64 = v4 byte-identical). M2 SSOT: must match "
            "alpha_for_rank(--lora-rank). The dispatcher's "
            "_verify_alpha_invariant catches mismatches before launch; this "
            "worker also re-asserts on startup as a defense-in-depth."
        ),
    )
    ap.add_argument(
        "--positives",
        type=int,
        default=None,
        help=(
            "v6: positives-per-cell override. Default None = use the module "
            "constant POS_EX_PER_SOURCE=200. v6 holds positives GLOBAL at 200 "
            "(M3); kept as a CLI flag for legacy / debug only."
        ),
    )
    args = ap.parse_args(argv)

    # ── v6 M2 alpha invariant re-assertion (defense-in-depth). ───────────────
    if args.lora_rank is not None or args.lora_alpha is not None:
        if args.lora_rank is None or args.lora_alpha is None:
            raise RuntimeError(
                "v6 M2 alpha invariant: --lora-rank and --lora-alpha must be "
                f"passed together; got rank={args.lora_rank} alpha={args.lora_alpha}."
            )
        from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
            alpha_for_rank,
        )

        expected_alpha = alpha_for_rank(int(args.lora_rank))
        if int(args.lora_alpha) != expected_alpha:
            raise RuntimeError(
                f"v6 M2 alpha invariant violation: --lora-rank={args.lora_rank} "
                f"--lora-alpha={args.lora_alpha} but expected α={expected_alpha} "
                f"(from RANK_ALPHA_MAP_V5 / ALPHA_CONTROL_V6 SSOT). The "
                f"dispatcher's _verify_alpha_invariant should have caught this; "
                f"investigate."
            )

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format=(
            f"%(asctime)s [phase=cell_{args.cell}_seed{args.seed}_lr{args.lr:g}_"
            f"{args.phase}] %(name)s %(levelname)s | %(message)s"
        ),
        stream=sys.stdout,
    )
    log.info(
        "Assigned physical GPU --gpu-id=%d; inherited CUDA_VISIBLE_DEVICES=%s "
        "(train/sft.py will SET CVD=str(gpu_id)). phase=%s, lr=%g",
        args.gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        args.phase,
        args.lr,
    )

    # ── Imports (lazy: avoid loading torch on argparse failure). ─────────────
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CELL_SPECS_477,
        EPOCHS,
        HF_MODEL_REPO,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        SOURCE_PERSONA,
        TRAJECTORY_CHECKPOINT_FRACTIONS,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
        build_cell,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )

    # Run-dir slug encodes the LR too so cells at the same slug but different
    # LR (5 calibration LRs per count) don't collide on disk.
    run_label = f"{args.cell}_seed{args.seed}_lr{args.lr:g}"
    run_dir = args.runs_root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    out_traj = args.slab_root / run_label / "trajectory.json"
    sentinel = args.log_dir / f"issue-477-{run_label}-results.json"

    bank = load_persona_bank(args.bank_path)
    r_train = load_r_artifact(args.r_train_path)
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.centroids_dir)
    q_train, _q_eval = get_train_eval_questions()

    # ── Phase: build training data (CPU). ────────────────────────────────────
    log.info("[phase=build_%s] building training data", args.cell)
    build_cell(
        args.cell,
        train_jsonl,
        r_train=r_train,
        cos_to_source=cts,
        q_train=q_train,
        persona_bank=bank,
        source=SOURCE_PERSONA,
        seed=args.seed,
        cell_specs=CELL_SPECS_477,
    )

    # ── Phase: train. ────────────────────────────────────────────────────────
    # v2 LEGACY paths:
    #   * calibration:  terminal-only eval, fractions = (1.0,)
    #   * main / implant_sweep: full 6-ckpt trajectory at 2-dp precision
    # v4 STEP-LEVER paths:
    #   * step_calibration: dense early-step grid via step_fractions(...) at
    #     4-dp precision
    #   * main_v4: clamped 3-checkpoint context window via
    #     main_phase_context_window(s*, max_steps) at 4-dp precision
    # The frac_precision threading keeps v4 cells from collapsing target_step=1
    # and target_step=2 at max_steps=426 onto a single frac_0.00 key.
    step_calibration_fractions: tuple[float, ...] | None = None
    frac_precision: int = 2

    if args.phase in ("step_calibration", "rank_calibration", "rank_control"):
        # v4 §4 Phase 2 (step_calibration) AND v6 Phase 2A / 2A-CONTROL
        # (rank_calibration / rank_control): dense early-step grid. Compute
        # max_steps from the built training JSONL row count (= eff. dataset
        # size × epochs / eff. batch). Assertion 3 of plan v4 §12 pins
        # {76, 126, 226, 426} for counts {2, 4, 8, 16} at epochs=2 lr=2e-6
        # — recomputed here from the ACTUAL row count so the picker can't
        # drift from the trained reality. v6 cells use the same step grid
        # at the same lr=CALIBRATION_LR_V3 (2e-6).
        from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
            BATCH_SIZE,
            GRAD_ACCUM,
        )
        from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
            step_fractions,
        )

        n_rows = sum(1 for _ in train_jsonl.open()) if train_jsonl.exists() else 0
        if n_rows <= 0:
            raise RuntimeError(
                f"[{args.cell}] {args.phase} phase: built training JSONL "
                f"{train_jsonl} has {n_rows} rows — build_cell silently emitted "
                f"an empty pool? Investigate before computing step_fractions."
            )
        eff_batch = BATCH_SIZE * GRAD_ACCUM
        # ceil(n_rows × epochs / eff_batch). The trainer can be one step off
        # due to drop_last; we add 1 step of headroom on the upper bound check
        # so a legitimate target_step right at terminal isn't spuriously
        # rejected — but step_fractions rejects strict overshoot.
        max_steps = -(-(n_rows * EPOCHS) // eff_batch)
        if not args.target_steps:
            raise RuntimeError(
                f"[{args.cell}] {args.phase} phase requires --target-steps "
                f"(comma-separated optimizer-step ints); got empty."
            )
        target_steps = tuple(int(s.strip()) for s in args.target_steps.split(",") if s.strip())
        # Drop any target_step > max_steps (e.g. step=64 at count=2 max=76 is
        # fine; step=128 would be dropped). Fail loud only if the result is
        # empty — that means even target_step=1 exceeds max_steps, which
        # cannot happen for positive max_steps.
        kept_targets = tuple(s for s in target_steps if s <= max_steps)
        if not kept_targets:
            raise RuntimeError(
                f"[{args.cell}] {args.phase}: every target_step in "
                f"{target_steps} exceeds max_steps={max_steps} (n_rows={n_rows}, "
                f"eff_batch={eff_batch}, epochs={EPOCHS}). Cannot calibrate."
            )
        # Always include the terminal step (frac=1.0) for provenance/comparison.
        step_calibration_fractions = (
            *step_fractions(kept_targets, max_steps, precision=4),
            1.0,
        )
        frac_precision = 4
        fractions: tuple[float, ...] = step_calibration_fractions
        log.info(
            "[phase=train_%s] v6/v4 %s: n_rows=%d, max_steps=%d, "
            "target_steps=%s → fractions=%s (4-dp precision)",
            args.cell,
            args.phase,
            n_rows,
            max_steps,
            kept_targets,
            fractions,
        )
    elif args.phase == "main_v4":
        # v4 §4 + §6: 3-checkpoint context window around the picked step s*.
        from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
            BATCH_SIZE,
            GRAD_ACCUM,
        )
        from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
            main_phase_context_window,
            step_fractions,
        )

        if args.picked_step is None or args.picked_step <= 0:
            raise RuntimeError(
                f"[{args.cell}] main_v4 phase requires --picked-step <int>; got "
                f"{args.picked_step!r}. The dispatcher's Phase 2.5 picker passes it."
            )
        n_rows = sum(1 for _ in train_jsonl.open()) if train_jsonl.exists() else 0
        if n_rows <= 0:
            raise RuntimeError(
                f"[{args.cell}] main_v4 phase: built training JSONL {train_jsonl} "
                f"has {n_rows} rows — investigate before computing context window."
            )
        eff_batch = BATCH_SIZE * GRAD_ACCUM
        max_steps = -(-(n_rows * EPOCHS) // eff_batch)
        window = main_phase_context_window(int(args.picked_step), max_steps)
        step_calibration_fractions = (
            *step_fractions(tuple(window), max_steps, precision=4),
            1.0,
        )
        frac_precision = 4
        fractions = step_calibration_fractions
        log.info(
            "[phase=train_%s] v4 main_v4: picked_step=%d, max_steps=%d → "
            "context_window=%s → fractions=%s (4-dp precision)",
            args.cell,
            args.picked_step,
            max_steps,
            window,
            fractions,
        )
    elif args.phase == "implant_sweep_v4":
        # v4r2 §4 PHASE 4: ONE anchor training run per seed; evaluate trajectory
        # at the requested non-terminal step levels + the terminal step. The
        # worker emits ONE cell_summary with a per_step dict the dispatcher
        # expands into IMPLANT_SWEEP_V4_SLUGS per-step records.
        from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
            BATCH_SIZE,
            GRAD_ACCUM,
        )
        from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
            step_fractions,
        )

        if not args.implant_steps:
            raise RuntimeError(
                f"[{args.cell}] implant_sweep_v4 phase requires --implant-steps "
                f"(CSV of positive ints); got empty."
            )
        n_rows = sum(1 for _ in train_jsonl.open()) if train_jsonl.exists() else 0
        if n_rows <= 0:
            raise RuntimeError(
                f"[{args.cell}] implant_sweep_v4 phase: built training JSONL "
                f"{train_jsonl} has {n_rows} rows — investigate."
            )
        eff_batch = BATCH_SIZE * GRAD_ACCUM
        max_steps = -(-(n_rows * EPOCHS) // eff_batch)
        target_steps = tuple(int(s.strip()) for s in args.implant_steps.split(",") if s.strip())
        kept_targets = tuple(s for s in target_steps if s <= max_steps)
        if not kept_targets:
            raise RuntimeError(
                f"[{args.cell}] implant_sweep_v4: every target_step in "
                f"{target_steps} exceeds max_steps={max_steps} (n_rows={n_rows}, "
                f"eff_batch={eff_batch}, epochs={EPOCHS}). Cannot calibrate."
            )
        step_calibration_fractions = (
            *step_fractions(kept_targets, max_steps, precision=4),
            1.0,
        )
        frac_precision = 4
        fractions = step_calibration_fractions
        log.info(
            "[phase=train_%s] v4 implant_sweep_v4: lr=%g, max_steps=%d, "
            "target_steps=%s → fractions=%s (4-dp precision)",
            args.cell,
            args.lr,
            max_steps,
            kept_targets,
            fractions,
        )
    elif args.phase == "calibration":
        fractions = (1.0,)
    elif args.smoke:
        # Same early-collapse-window smoke as #472 (round-2 fix).
        fractions = (0.08, 0.16, 0.5, 1.0)
    else:
        fractions = TRAJECTORY_CHECKPOINT_FRACTIONS

    # HF push path: #477 adapters live under adapters/issue_477/<run_label>.
    hf_path_in_repo = f"adapters/issue_477/{run_label}"
    # v6 slot-fix: ALL v6 phases run with suppress_at_post_response_slot=True
    # so the contrastive negatives' loss actually lands on the DV-read slot
    # (the post-response <|im_end|> token). The v4/v2 legacy phases keep the
    # pre-port behavior (default False, byte-identical). The v6 phase
    # detection is intentionally broad — every CLI flag combination that
    # implies a v6 cell (Cal-A, Cal-A0, OR the v6 main sweep at picked rank,
    # OR a v6 implant sweep at picked rank) turns it on.
    v6_cell = args.phase in ("rank_calibration", "rank_control") or args.lora_rank is not None
    if v6_cell:
        from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
            MARKER_IM_END_TOKEN_ID,
        )

        marker_im_end_id: int | None = MARKER_IM_END_TOKEN_ID
        marker_suppress = True
    else:
        marker_im_end_id = None
        marker_suppress = False
    log.info(
        "[phase=train_%s] training (phase=%s, smoke=%s, lr=%g, epochs=%d, "
        "fractions=%s, lora_rank=%s, lora_alpha=%s, suppress_slot=%s)",
        args.cell,
        args.phase,
        args.smoke,
        args.lr,
        EPOCHS,
        fractions,
        args.lora_rank,
        args.lora_alpha,
        marker_suppress,
    )
    # WandB run-name prefix: #477 cells should be browsable as `issue477_*` not
    # `issue472_*` (the parent's default). Threaded through train_one_cell's
    # `run_name_override` kwarg (defaults None = #472 behavior).
    run_name_477 = f"issue477_{args.cell}_seed{args.seed}"
    train_result = train_one_cell(
        cell_slug=args.cell,
        seed=args.seed,
        train_jsonl=train_jsonl,
        output_dir=final_adapter_dir,
        ckpt_root=ckpt_root,
        fractions=fractions,
        fallback=False,  # #477 does not use the #472 sub-ceiling fallback recipe
        report_to=args.report_to,
        gpu_id=args.gpu_id,
        lr_override=args.lr,
        epochs_override=EPOCHS,  # #477 = 2 (vs #472's 1)
        hf_path_in_repo_override=hf_path_in_repo,
        run_name_override=run_name_477,
        # v4: when set, replaces ``fractions`` AND bumps the dir/index
        # precision to 4-dp so target_step=1 + target_step=2 at max_steps=426
        # do not collapse onto frac_0.00 (the v3 fact-check fix).
        step_calibration_fractions=step_calibration_fractions,
        frac_precision=frac_precision,
        # v6 M2: per-cell LoRA rank + alpha from the dispatcher's SSOT (None
        # = legacy r=32/α=64 default). The startup re-assertion above pinned
        # alpha == alpha_for_rank(rank) before this call.
        lora_r_override=args.lora_rank,
        lora_alpha_override=args.lora_alpha,
        # v6 slot-fix port (origin/main MarkerOnlyDataCollator args).
        marker_suppress_at_post_response_slot=marker_suppress,
        marker_im_end_token_id=marker_im_end_id,
    )
    ckpt_index_path = run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(train_result["checkpoint_index"], indent=2))
    log.info(
        "[phase=train_%s] done; checkpoints=%s", args.cell, list(train_result["checkpoint_index"])
    )

    # Free in-process LoRA-training GPU memory BEFORE nested vLLM eval (round-3
    # #472 hot-fix). The vLLM worker on the same GPU otherwise fails its
    # gpu_memory_utilization startup check.
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase: eval_trajectory (NESTED subprocess; vLLM teardown isolation). ─
    eval_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if eval_cvd != str(args.gpu_id):
        raise RuntimeError(
            f"Pre-eval CUDA_VISIBLE_DEVICES={eval_cvd!r} != assigned --gpu-id={args.gpu_id}; the "
            f"nested eval subprocess would run on the wrong physical GPU. train/sft.py should "
            f"have set CVD=str(gpu_id) during training — investigate before launching the sweep."
        )
    log.info(
        "[phase=eval_%s] inherited CUDA_VISIBLE_DEVICES=%s (physical GPU %d)",
        args.cell,
        eval_cvd,
        args.gpu_id,
    )
    eval_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i472_eval_trajectory.py",
        "--cell",
        args.cell,
        "--seed",
        str(args.seed),
        "--checkpoint-index",
        str(ckpt_index_path),
        "--out-path",
        str(out_traj),
        "--centroids-dir",
        str(args.centroids_dir),
        "--bank-path",
        str(args.bank_path),
        "--r-eval-path",
        str(args.r_eval_path),
        "--layer",
        str(HEADLINE_LAYER),
        # Round-3 #477 fix: drive the eval rig from #477's CELL_SPECS_477 so the
        # held-out panel excludes EVERY #477 cell's negatives (union across all
        # count levels). Without this the eval rig defaults to #472's CELL_SPECS
        # and a 16-cell ends up evaluated on personas it trained against —
        # corrupting the H1 count axis. Also enables the fail-loud disjointness
        # assert that would have caught the original bug.
        "--cell-specs",
        "477",
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
    if args.smoke or args.phase == "calibration":
        # Smaller max_new_tokens for the cheaper terminal-only calibration eval
        # AND for smoke runs (saves vLLM minutes).
        eval_cmd.extend(["--max-new-tokens", "256"])
    log.info("[phase=eval_%s] nested eval subprocess: %s", args.cell, " ".join(eval_cmd))
    subprocess.run(eval_cmd, env={**os.environ}, check=True)

    if not out_traj.exists():
        raise RuntimeError(
            f"[{args.cell}] eval_trajectory subprocess exited 0 but {out_traj} missing — "
            f"silent eval failure (feedback_eval_script_silent_not_present_misdiagnosis)."
        )

    # ── Phase: emit per-cell summary (calibration row OR main-cell summary). ─
    traj = json.loads(out_traj.read_text())
    final_ck = max(traj["checkpoints"], key=lambda c: float(c["frac"]))
    src = final_ck["source_self"]
    src_delta = float(src["delta_g_mean"])
    # Fail loud on missing emission_p (cheap fix #5): the eval rig writes this
    # key; silently defaulting to 0.0 hides schema drift and trips the validity
    # gate downstream with no diagnostic.
    if "emission_p" not in src:
        raise RuntimeError(
            f"[{args.cell}] missing emission_p in trajectory.json source_self block "
            f"— schema drift. Expected the eval rig to write emission_p alongside "
            f"delta_g_mean; got keys {sorted(src.keys())}. Investigate "
            f"i472_eval_trajectory.py before re-running."
        )
    src_emit = float(src["emission_p"])
    # Eval-panel floor (cheap fix #4 / plan §8): at <20 held-out personas the
    # mean DV-A becomes a degenerate average (the 16-persona cell's 43-panel
    # was flagged tight; below 20 is failure territory). Fail loud rather
    # than compute a silent headline on a degenerate base.
    eval_personas = list(final_ck["held_out"].keys())
    if len(eval_personas) < 20:
        raise RuntimeError(
            f"[{args.cell}] eval panel has {len(eval_personas)} personas; plan §8 "
            f"requires ≥20 (the per-cell mean DV-A floor). The base panel was "
            f"likely truncated upstream — investigate the held-out persona panel "
            f"build before computing a silent mean bystander ΔG."
        )

    # Mean bystander ΔG at the final checkpoint (legacy DV-A for the v2 headline).
    mean_bystander = mean_bystander_delta_g(final_ck)

    cell_summary = {
        "cell": args.cell,
        "seed": args.seed,
        "lr": args.lr,
        "phase": args.phase,
        "run_label": run_label,
        "source_self_delta_g_at_last_ckpt": src_delta,
        "source_emission_p_at_last_ckpt": src_emit,
        "mean_bystander_delta_g": mean_bystander,
        "step_at_last_ckpt": final_ck.get("step"),
        "trajectory_path": str(out_traj),
        "adapter_hf_path": hf_path_in_repo,
        "adapter_hf_repo": HF_MODEL_REPO,
        "checkpoint_index_path": str(ckpt_index_path),
    }

    # ── v4 picked-step + per-step extraction. ────────────────────────────────
    # The v4 analyze partials (marker-channel + full-vocab + implant-only-axis)
    # read *_at_picked_step keys, NOT the *_at_last_ckpt fields the terminal-
    # checkpoint summary above writes. Reading the terminal checkpoint would
    # defeat the entire picked-step decoupling — by the terminal step the
    # marker-channel headline saturates and the per-cell ranks collapse.
    # ``select_checkpoint_near_step`` + ``picked_step_kl_fields`` are module-
    # scope so tests can pin the contract without spawning the worker.

    if args.phase == "main_v4":
        if args.picked_step is None or args.picked_step <= 0:
            raise RuntimeError(
                f"[{args.cell}] main_v4 phase reached summary block with "
                f"--picked-step={args.picked_step!r}; the upfront check should "
                f"have raised earlier. Investigate."
            )
        picked_step_req = int(args.picked_step)
        actual_step, picked_ck, offset = select_checkpoint_near_step(
            traj, picked_step_req, cell_slug=args.cell
        )
        picked_fields = picked_step_kl_fields(picked_ck, cell_slug=args.cell)
        cell_summary.update(picked_fields)
        cell_summary.update(
            {
                # Provenance: what was requested vs what the trainer actually
                # produced. The analyzer logs both so post-hoc audits can spot
                # drop_last drift without re-loading trajectory.json.
                "picked_step_requested": picked_step_req,
                "picked_step_actual": int(actual_step),
                "picked_step_offset": int(offset),
            }
        )
        log.info(
            "[phase=summary_%s] main_v4 picked_step req=%d actual=%d (offset=%d): "
            "src ΔG=%.2f emit=%.2f, marker-channel KL src=%.3f bys=%.3f, "
            "full-vocab KL bys=%s",
            args.cell,
            picked_step_req,
            actual_step,
            offset,
            picked_fields["source_self_delta_g_at_picked_step"],
            picked_fields["source_emission_p_at_picked_step"],
            picked_fields["source_self_marker_channel_kl_at_picked_step"],
            picked_fields["mean_bystander_marker_channel_kl_at_picked_step"],
            "None"
            if picked_fields["mean_bystander_full_vocab_kl_at_picked_step"] is None
            else f"{picked_fields['mean_bystander_full_vocab_kl_at_picked_step']:.3f}",
        )

    if args.phase == "implant_sweep_v4":
        # v4r2 implant-only-axis: ONE training run, eval at multiple step levels.
        # Emit a per_step dict keyed by step level (the requested non-terminal
        # steps + the terminal step). The dispatcher unpacks it into per-step
        # records that satisfy implant_only_axis_spearman_marker_channel_kl's
        # cell-shape contract.
        if not args.implant_steps:
            raise RuntimeError(
                f"[{args.cell}] implant_sweep_v4 phase reached summary block "
                f"with empty --implant-steps; the upfront check should have raised."
            )
        requested_levels = tuple(int(s.strip()) for s in args.implant_steps.split(",") if s.strip())
        # Drop any requested step that exceeds the trainer's actual max_steps
        # (already filtered during fractions resolution; mirror the same logic).
        steps_present = sorted(
            int(ck["step"]) for ck in traj["checkpoints"] if ck.get("step") is not None
        )
        if not steps_present:
            raise RuntimeError(
                f"[{args.cell}] implant_sweep_v4: trajectory.json has no "
                f"checkpoint with a 'step' field."
            )
        terminal_step = max(steps_present)
        # Build the level list: each requested non-terminal step (label = step
        # int) + the terminal step labelled "T".
        per_step: dict[str, dict] = {}
        for s in requested_levels:
            if s > terminal_step:
                # Fail loud, mirroring step_fractions's clamp invariant.
                continue
            actual_step, picked_ck, offset = select_checkpoint_near_step(
                traj, s, cell_slug=args.cell
            )
            entry = picked_step_kl_fields(picked_ck, cell_slug=args.cell)
            entry.update(
                {
                    "requested_step": int(s),
                    "actual_step": int(actual_step),
                    "step_offset": int(offset),
                }
            )
            per_step[str(s)] = entry
        # Terminal level: pick the checkpoint at max frac directly.
        terminal_ck = max(traj["checkpoints"], key=lambda c: float(c["frac"]))
        terminal_entry = picked_step_kl_fields(terminal_ck, cell_slug=args.cell)
        terminal_entry.update(
            {
                "requested_step": "T",
                "actual_step": int(terminal_ck.get("step", terminal_step)),
                "step_offset": 0,
            }
        )
        per_step["T"] = terminal_entry
        cell_summary["per_step"] = per_step
        log.info(
            "[phase=summary_%s] implant_sweep_v4: emitted per_step levels=%s",
            args.cell,
            sorted(per_step.keys()),
        )

    summary_path = run_dir / "cell_summary.json"
    summary_path.write_text(json.dumps(cell_summary, indent=2))
    log.info(
        "[phase=summary_%s] source ΔG=%.2f, emit=%.2f, mean bystander=%.2f → %s",
        args.cell,
        src_delta,
        src_emit,
        mean_bystander,
        summary_path,
    )

    _write_sentinel(
        sentinel,
        kind="epm:progress",
        phase=f"cell_done_{args.cell}_seed{args.seed}_lr{args.lr:g}_{args.phase}",
        note=cell_summary,
    )
    log.info(
        "[phase=cell_done_%s_seed%s_lr%g_%s] wrote sentinel → %s",
        args.cell,
        args.seed,
        args.lr,
        args.phase,
        sentinel,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
