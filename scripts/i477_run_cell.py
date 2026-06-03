# em-dash + Qwen marker " ※" + Greek ΔG + → intentional
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

PHASES = ("calibration", "main", "implant_sweep")
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
    args = ap.parse_args(argv)

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
    # Calibration phase: terminal-only eval, so the trajectory's compute is on
    # the FINAL checkpoint. We still need at least one frac to be 1.00 for the
    # checkpoint to land in the index; (1.00,) is the minimum.
    # Main / implant_sweep: full 6-checkpoint trajectory (#472's default).
    if args.phase == "calibration":
        fractions: tuple[float, ...] = (1.0,)
    elif args.smoke:
        # Same early-collapse-window smoke as #472 (round-2 fix).
        fractions = (0.08, 0.16, 0.5, 1.0)
    else:
        fractions = TRAJECTORY_CHECKPOINT_FRACTIONS

    # HF push path: #477 adapters live under adapters/issue_477/<run_label>.
    hf_path_in_repo = f"adapters/issue_477/{run_label}"
    log.info(
        "[phase=train_%s] training (phase=%s, smoke=%s, lr=%g, epochs=%d, fractions=%s)",
        args.cell,
        args.phase,
        args.smoke,
        args.lr,
        EPOCHS,
        fractions,
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
    # Mean bystander ΔG at the final checkpoint (Q_eval-mean per persona, then
    # mean across personas). This is DV-A for the headline H1.
    bystander_means: list[float] = []
    for _persona, per_q in final_ck["held_out"].items():
        deltas = [float(v["delta_g"]) for v in per_q.values()]
        if deltas:
            bystander_means.append(sum(deltas) / len(deltas))
    mean_bystander = float(sum(bystander_means) / len(bystander_means)) if bystander_means else 0.0

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
