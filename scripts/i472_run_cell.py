# em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #472 — single (cell, seed) worker: build → train → eval_trajectory.

This is the UNIFIED per-cell unit the dispatcher schedules across GPUs (8
concurrent on an 8-H100 pod). Smoke = the dispatcher launching exactly ONE of
these (--cells anchor --seeds 42 --smoke) — same subprocess shape, same env
injection, same on-policy DV path (smoke-architecture parity: PASS_UNIFIED).

Within one (cell, seed) the worker switches frameworks at most once (HF Trainer
for training, then a NESTED subprocess for the vLLM+HF eval_trajectory rig). The
nested eval subprocess boundary guarantees the OS reaps vLLM workers
(CLAUDE.md vLLM teardown gotcha) — the worker never loads vLLM in-process after
HF Trainer.

GPU pinning (round-3 #472 fix): the dispatcher passes ``--gpu-id <g>`` (the
assigned PHYSICAL GPU index against the FULL host enumeration). The worker
threads it to ``train_one_cell(gpu_id=g)`` → ``TrainLoraConfig.gpu_id=g``, and
``train/sft.py`` SETS ``os.environ["CUDA_VISIBLE_DEVICES"] = str(g)`` (then loads
with ``device_map={"": 0}``, CVD remapping the visible GPU to index 0). The nested
eval subprocess inherits this same ``CUDA_VISIBLE_DEVICES`` from ``os.environ``
(sft.py mutates it in-process) so vLLM + HF KL run on the SAME physical GPU g.

WHY NOT just inherit env CVD: ``train/sft.py`` does NOT respect an inherited
``CUDA_VISIBLE_DEVICES`` — it SETS it from ``cfg.gpu_id`` (default 0). So passing
``gpu_id=0`` while the dispatcher restricted env CVD to physical 3 makes sft.py
overwrite CVD to ``"0"`` = physical GPU 0 — every parallel cell re-targets GPU 0
→ CUDA OOM (round-3 #472). The fix threads the physical index as gpu_id and lets
sft.py own the CVD set (against the FULL enumeration); the dispatcher must NOT
also restrict env CVD, or ``str(g)`` would re-index against the 1-GPU view.

Usage (driven by the dispatcher; --gpu-id is the assigned physical GPU):
    uv run python scripts/i472_run_cell.py \
        --cell c472_anchor --seed 42 --gpu-id 3 \
        --slab-root eval_results/issue_472 --runs-root /workspace/runs/issue_472 \
        --log-dir /workspace/logs [--smoke] [--fallback] [--no-kl]
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

log = logging.getLogger("i472.run_cell")


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": 472,
                "by": "i472_run_cell",
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
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_472"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_472"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument(
        "--r-train-path", type=Path, default=Path("data/issue_472/on_policy_R/R_train.json")
    )
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny slice: fewer steps, 2 checkpoints.")
    ap.add_argument(
        "--fallback", action="store_true", help="Sub-ceiling fallback recipe (plan §7)."
    )
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL.")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "ASSIGNED physical GPU index (round-3 #472). Threaded to "
            "train_one_cell(gpu_id=...); train/sft.py SETS CUDA_VISIBLE_DEVICES "
            "to this so the cell + its nested eval subprocess run on physical GPU "
            "<gpu-id>. Default 0 (single-GPU / smoke)."
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

    run_dir = args.runs_root / f"{args.cell}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    out_traj = args.slab_root / f"{args.cell}_seed{args.seed}" / "trajectory.json"
    sentinel = args.log_dir / f"issue-472-{args.cell}-seed{args.seed}-results.json"

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
    )

    # ── Phase: train with mid-run checkpoints (HF Trainer, in-process). ──────
    # Smoke MUST include the EARLY sub-ceiling window. Round-2 diagnosis (#472):
    # the Qwen marker implant saturates the source's argmax between frac 0.16 and
    # 0.33 — at frac {0.33, 0.5} the model's own on-policy R collapses to ` ※ ※ …`
    # repetition (degenerate max-leakage, not graded), while at frac {0.08, 0.16}
    # it produces NORMAL responses sub-ceiling. The original smoke {0.5, 1.0} sat
    # ENTIRELY post-collapse, so it (a) crashed on the marker-in-R invariant and
    # (b) could never have validated the sub-ceiling gate. Smoke now spans an early
    # readable point + the collapse onset so the gate is meaningful AND the
    # collapsed-R path is exercised (NOT a graded-vs-degenerate masquerade).
    fractions = (0.08, 0.16, 0.5, 1.0) if args.smoke else TRAJECTORY_CHECKPOINT_FRACTIONS
    log.info(
        "[phase=train_%s] training (smoke=%s, fallback=%s)", args.cell, args.smoke, args.fallback
    )
    train_result = train_one_cell(
        cell_slug=args.cell,
        seed=args.seed,
        train_jsonl=train_jsonl,
        output_dir=final_adapter_dir,
        ckpt_root=ckpt_root,
        fractions=fractions,
        fallback=args.fallback,
        report_to=args.report_to,
        gpu_id=args.gpu_id,  # assigned physical GPU (round-3 #472 sharding fix)
    )
    ckpt_index_path = run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(train_result["checkpoint_index"], indent=2))
    log.info(
        "[phase=train_%s] done; checkpoints=%s", args.cell, list(train_result["checkpoint_index"])
    )

    # Free the in-process LoRA-training GPU memory BEFORE the nested eval
    # subprocess loads vLLM on the SAME GPU. train_one_cell ran in-process, so
    # the model/optimizer CUDA blocks are still cached in this worker; without
    # this release the eval vLLM sees ~16 GiB occupied and its
    # gpu_memory_utilization startup check fails ("Free memory < desired").
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase: eval_trajectory (NESTED subprocess: vLLM teardown isolation). ─
    # The worker has HF Trainer's GPU pin; spawn a fresh subprocess for the
    # vLLM+HF eval rig so vLLM workers are reaped on subprocess exit before this
    # worker (or the next scheduled cell) loads weights again. The eval rig uses
    # cuda:0 / LLM() with the inherited CUDA_VISIBLE_DEVICES — which train/sft.py
    # set to str(gpu_id) in THIS process — so vLLM + HF KL land on physical GPU
    # <gpu-id>, NOT GPU 0 (round-3 #472 fix). Fail loud if that env is wrong.
    eval_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if eval_cvd != str(args.gpu_id):
        raise RuntimeError(
            f"Pre-eval CUDA_VISIBLE_DEVICES={eval_cvd!r} != assigned --gpu-id={args.gpu_id}; the "
            f"nested eval subprocess would run on the wrong physical GPU. train/sft.py should have "
            f"set CVD=str(gpu_id) during training — investigate before launching the sweep."
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
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
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
            "adapter_hf_path": f"adapters/issue_472/{args.cell}_seed{args.seed}",
            "checkpoint_index": str(ckpt_index_path),
        },
    )
    log.info("[phase=cell_done_%s_seed%s] wrote sentinel → %s", args.cell, args.seed, sentinel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
