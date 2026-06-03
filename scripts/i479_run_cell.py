# ruff: noqa: RUF002, RUF003  # em-dash + × + Qwen marker " ※" intentional
#!/usr/bin/env python3
"""Task #479 — single (cell, seed) worker: build → train_479 → eval_trajectory.

Forked verbatim from ``scripts/i472_run_cell.py`` with three deltas:
  1. Imports ``train_one_cell_479`` (absolute-step checkpoints + #474 post-
     response-slot suppression flags) instead of ``train_one_cell``.
  2. Builds the per-cell JSONL with the #479 row counts (400 pos × 100
     neg/persona × 4 personas = 1:1, plan §4.3).
  3. Writes its sentinel under ``task_id=479`` and the issue-479 log-naming
     convention; cell trajectory eval reuses ``scripts/i472_eval_trajectory.py``
     verbatim (the rig is checkpoint-dir-driven, not step-or-frac-aware).

Same subprocess shape and same GPU-pinning contract as i472_run_cell so the
unified dispatcher schedules either runner identically (--issue switch only).
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

log = logging.getLogger("i479.run_cell")


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": 479,
                "by": "i479_run_cell",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_479"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_479"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument(
        "--r-train-path", type=Path, default=Path("data/issue_472/on_policy_R/R_train.json")
    )
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny slice: ~5 pos / 5 neg / 2 ckpts.")
    ap.add_argument(
        "--fallback",
        action="store_true",
        help="(Compatibility no-op for #479 — anchor-titration replaces the gate.)",
    )
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL.")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "ASSIGNED physical GPU index (round-3 #472 GPU-pin contract — see "
            "i472_run_cell.py docstring; behavior identical here)."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format=(
            f"%(asctime)s [phase=cell_{args.cell}_seed{args.seed}] "
            f"%(name)s %(levelname)s | %(message)s"
        ),
        stream=sys.stdout,
    )
    log.info(
        "Assigned physical GPU --gpu-id=%d; inherited CUDA_VISIBLE_DEVICES=%s "
        "(train/sft.py will SET CVD=str(gpu_id)).",
        args.gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        ANCHOR_RECIPES_479,
        C479_NEG_EX_PER_PERSONA,
        C479_POS_EX,
        CHECKPOINT_STEPS,
        HEADLINE_LAYER,
        LORA_R,
        SOURCE_PERSONA,
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
        train_one_cell_479,
    )

    if args.cell not in ANCHOR_RECIPES_479:
        raise SystemExit(f"[#479] unknown cell {args.cell!r}; known: {sorted(ANCHOR_RECIPES_479)}")

    run_dir = args.runs_root / f"{args.cell}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    out_traj = args.slab_root / f"{args.cell}_seed{args.seed}" / "trajectory.json"
    sentinel = args.log_dir / f"issue-479-{args.cell}-seed{args.seed}-results.json"

    bank = load_persona_bank(args.bank_path)
    r_train = load_r_artifact(args.r_train_path)
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.centroids_dir)
    q_train, _q_eval = get_train_eval_questions()

    # ── Phase: build training data (CPU). ────────────────────────────────────
    # Tiny smoke slice: clip both row counts to a single-batch's worth (~5
    # pos + 5 neg). This still exercises both row types through the collator
    # so the post-response-slot suppression branch fires on a real negative.
    pos_ex = 5 if args.smoke else C479_POS_EX
    neg_ex_per_persona = 5 if args.smoke else C479_NEG_EX_PER_PERSONA
    log.info(
        "[phase=build_%s] building training data (pos=%d, neg/persona=%d, smoke=%s)",
        args.cell,
        pos_ex,
        neg_ex_per_persona,
        args.smoke,
    )
    build_cell(
        args.cell,
        train_jsonl,
        r_train=r_train,
        cos_to_source=cts,
        q_train=q_train,
        persona_bank=bank,
        source=SOURCE_PERSONA,
        seed=args.seed,
        pos_ex_override=pos_ex,
        neg_ex_per_persona_override=neg_ex_per_persona,
    )

    # ── Phase: train with absolute-step checkpoints (HF Trainer, in-process). ─
    # Smoke uses a 2-step micro-schedule so the gen step + 1 mid-run save fires;
    # the in-process train then exits and the nested eval subprocess loads vLLM.
    # The smoke max_steps is implicitly determined by min(epochs * steps_per_epoch,
    # this list's max) under TRL semantics, and we override via train_one_cell_479's
    # recipe; for smoke just keep the 2-step list and let max_steps shrink.
    steps = (1, 2) if args.smoke else CHECKPOINT_STEPS
    log.info(
        "[phase=train_%s] training (smoke=%s, steps=%s)",
        args.cell,
        args.smoke,
        list(steps),
    )
    train_result = train_one_cell_479(
        cell_slug=args.cell,
        seed=args.seed,
        train_jsonl=train_jsonl,
        output_dir=final_adapter_dir,
        ckpt_root=ckpt_root,
        steps=steps,
        report_to=args.report_to,
        gpu_id=args.gpu_id,
    )
    ckpt_index_path = run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(train_result["checkpoint_index"], indent=2))
    log.info(
        "[phase=train_%s] done; checkpoints=%s",
        args.cell,
        list(train_result["checkpoint_index"]),
    )

    # Free the in-process LoRA-training GPU memory BEFORE the nested eval
    # subprocess loads vLLM on the SAME GPU.
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase: eval_trajectory (NESTED subprocess: vLLM teardown isolation). ─
    eval_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if eval_cvd != str(args.gpu_id):
        raise RuntimeError(
            f"Pre-eval CUDA_VISIBLE_DEVICES={eval_cvd!r} != assigned "
            f"--gpu-id={args.gpu_id}; the nested eval subprocess would run on "
            f"the wrong physical GPU."
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
        # #479 §4.5 + §13.1: max_new_tokens=2048 per marker-leakage rule
        # (≥2× max_length 1024). The rig defaults to 1024; #479 threads the
        # higher cap so the silent-zero failure class (#260) cannot fire.
        "--max-new-tokens",
        "2048",
        # The eval rig sizes its vLLM LoRA cache from --max-lora-rank; the
        # c479_r32* cells use r=32 so we always send the larger value (it's
        # a memory ceiling, not a forced minimum).
        # (The rig already uses its DEFAULT_MAX_LORA_RANK=32, which matches;
        # nothing to override here. Documented for future readers.)
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
    # The smoke eval slice uses 256 max_new_tokens via the rig's --smoke knob;
    # i472_eval_trajectory does not have a --smoke flag — we pass max-new-tokens
    # 256 directly when smoke is set (overrides the 2048 above by being the
    # LAST occurrence of the flag).
    if args.smoke:
        eval_cmd.extend(["--max-new-tokens", "256"])
    log.info("[phase=eval_%s] nested eval subprocess: %s", args.cell, " ".join(eval_cmd))
    subprocess.run(eval_cmd, env={**os.environ}, check=True)

    if not out_traj.exists():
        raise RuntimeError(
            f"[{args.cell}] eval_trajectory subprocess exited 0 but {out_traj} missing — "
            f"silent eval failure (feedback_eval_script_silent_not_present_misdiagnosis)."
        )

    _write_sentinel(
        sentinel,
        kind="epm:progress",
        phase=f"cell_done_{args.cell}_seed{args.seed}",
        note={
            "cell": args.cell,
            "seed": args.seed,
            "trajectory_path": str(out_traj),
            "adapter_hf_path": f"adapters/issue_479/{args.cell}_seed{args.seed}",
            "checkpoint_index": str(ckpt_index_path),
            "recipe": ANCHOR_RECIPES_479[args.cell],
            "lora_r_max_for_vllm": LORA_R,
        },
    )
    log.info("[phase=cell_done_%s_seed%s] wrote sentinel → %s", args.cell, args.seed, sentinel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
