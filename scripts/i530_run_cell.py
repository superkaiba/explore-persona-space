# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek α intentional
#!/usr/bin/env python3
"""Task #530 — single (cell, seed) worker: build → train → eval_trajectory.

Forked from scripts/i504_run_cell.py (issue-504 branch). Identical pipeline
shape; the SINGLE load-bearing change vs #504 is the marker-only learning
rate (1e-4 → 5e-6) + structural support (epoch ceiling 3 → 12 so the
`MarkerBandStopCallback`, not the epoch counter, decides when training
halts; trajectory checkpoints persisted at {0.25, 0.50, 0.75, 1.00} of the
band-stop step count).

What this script reuses verbatim from #504:
  * `contrastive_neg_geometry_504.build_training_data.build_cell_504`
    (per-cell training pool; one positioned negative + qwen_default per arm)
  * `contrastive_neg_geometry_472.train_cell.train_one_cell` (LoRA trainer
    with `lr_override`, `epochs_override`, `step_calibration_fractions`,
    + `MarkerBandStopCallback` auto-attached via `train_lora` defaults on
    `main` — the band-stop fires when source `log P(※) − base` enters
    [5, 12] nats and is NOT re-gated on bystander resolution)
  * `scripts/i504_eval_trajectory.py` (the nested vLLM eval rig with
    `assert_adapter_actually_applied` + the per-batch byte-identical guard
    ported from issue-504)

What #530 changes vs #504 at the dispatcher level:
  * Slab-root default → `eval_results/issue_530`
  * Runs-root default → `/workspace/runs/issue_530`
  * Adapter HF subfolder → `adapters/issue_530/<slug>_seed<S>`
  * Sentinel filename → `issue-530-<slug>-seed<S>-results.json`
  * task_id (in the sentinel) → 530
  * Trajectory-checkpoint persistence: `EPM_PERSIST_TRAJECTORY_HF_REPO`
    + `EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER` env vars (already consumed by
    `train_cell.py::_maybe_persist_trajectory_checkpoint`); the dispatcher
    sets them to push to `adapters/issue_530/<slug>_seed<S>/ckpt_frac{N}`
    so the trajectory cadence checkpoints survive pod teardown.
  * Default `--checkpoint-fractions=0.25,0.50,0.75,1.00` so #530 reads
    a 4-checkpoint trajectory instead of #504's 6-checkpoint cadence
    (matches plan §4.3 trajectory-persistence design).
  * Default `--epochs=12` so the band-stop callback halts training, not
    the epoch counter (plan §4.3 "Epoch ceiling 3 → 12").
  * Default `--lr=5e-6` (the single manipulated variable per plan §4.1).

Marker token assertion (CLAUDE.md marker-leakage rule):
    tokenizer.encode(" ※", add_special_tokens=False) == [83399]
runs BEFORE training spawn; fail loud.

GPU pinning, sentinel-file pattern, and `+gpu_id=N` Hydra-override hand-off
to `train/sft.py` are inherited from i504_run_cell.py byte-for-byte. The
dispatcher passes `--gpu-id <g>`; `train/sft.py` SETS
`CUDA_VISIBLE_DEVICES=str(g)`; the nested eval subprocess inherits it.

Usage:
    uv run python scripts/i530_run_cell.py \\
        --cell c504v3_near --seed 42 --gpu-id 0 \\
        --chosen-rank 8 --chosen-alpha 32 \\
        --arm-to-n-json /tmp/i530-arm-to-n.json \\
        --slab-root eval_results/issue_530 \\
        --runs-root /workspace/runs/issue_530 \\
        --log-dir /workspace/logs
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

log = logging.getLogger("i530.run_cell")

# --- #530 dispatcher defaults (the load-bearing per-cell knobs) -------------
# Single manipulated variable vs #504 (plan §4.1). Source: #329 + #478.
LR_DEFAULT_530: float = 5e-6
# Headroom for the band-stop to halt training (plan §4.3). The
# `MarkerBandStopCallback` (default attached when `marker_only_loss=True` on
# `main`) typically fires at ~6–10 epochs in this composition at lr=5e-6.
EPOCHS_DEFAULT_530: int = 12
# Per plan §4.3: trajectory checkpoints persisted at {0.25, 0.50, 0.75, 1.00}
# of total band-stop steps.
CHECKPOINT_FRACTIONS_DEFAULT_530: str = "0.25,0.5,0.75,1.0"
# HF model repo where trajectory checkpoints persist; plan §4.3 names
# `adapters/issue_530/<slug>/ckpt_frac{N}`. The dispatcher sets BOTH
# `EPM_PERSIST_TRAJECTORY_HF_*` (consumed by `train_cell.py`) AND
# `EPM_PERSIST_ADAPTER_HF_*` (the plan's documented name, alias).
TRAJECTORY_HF_REPO_DEFAULT: str = "superkaiba1/explore-persona-space"


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": 530,
                "by": "i530_run_cell",
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
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_530"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_530"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument(
        "--r-train-path", type=Path, default=Path("data/issue_472/on_policy_R/R_train.json")
    )
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument(
        "--arm-to-n-json",
        type=Path,
        required=True,
        help=(
            "JSON file: {arm_slug: positioned_N_persona, ...} + optional "
            "{'smoke_mid_band_n': ..., 'panel': [...]}. Produced by Phase 0.5 "
            "via `i504_phase_phase05.py`."
        ),
    )
    ap.add_argument(
        "--chosen-rank",
        type=int,
        default=8,
        help="LoRA rank. Plan §4.2 inherits #504's r=8 (Source: #477 RANK_ALPHA_MAP_V5).",
    )
    ap.add_argument(
        "--chosen-alpha",
        type=int,
        default=32,
        help="LoRA α at rank=8 (Source: #477 RANK_ALPHA_MAP_V5; α = 4·r).",
    )
    ap.add_argument(
        "--chosen-frac",
        type=float,
        default=None,
        help=(
            "Informational; #530 reads at the band-stop checkpoint (frac=1.00) "
            "by default. Plan §4.4 step 4 pins the band-stop checkpoint as the "
            "headline read; auxiliary diagnostics at the other fractions."
        ),
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=LR_DEFAULT_530,
        help=(
            f"Marker-only learning rate (default {LR_DEFAULT_530:g}; the single "
            "manipulated variable vs #504's 1e-4). Source: plan §4.1 + #329 + "
            "#478. Threaded as `lr_override` into `train_one_cell`."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT_530,
        help=(
            f"Epoch ceiling (default {EPOCHS_DEFAULT_530}). Plan §4.3: 3 → 12 "
            "is structural headroom for the band-stop to halt training, NOT a "
            "second manipulated variable. At lr=5e-6 the band-stop typically "
            "fires at ~6-10 epochs in this composition. Threaded as "
            "`epochs_override` into `train_one_cell`."
        ),
    )
    ap.add_argument(
        "--checkpoint-fractions",
        default=CHECKPOINT_FRACTIONS_DEFAULT_530,
        help=(
            f"Trajectory checkpoint cadence (default {CHECKPOINT_FRACTIONS_DEFAULT_530!r}). "
            "Plan §4.3 — fractions of total band-stop steps. Comma-separated "
            "floats in (0, 1]. Threaded as `step_calibration_fractions` into "
            "`train_one_cell`; `train_cell.py::CheckpointAtFractionsCallback` "
            "writes adapter at each fraction AND (when "
            "`EPM_PERSIST_TRAJECTORY_HF_REPO` + `_SUBFOLDER` are set) uploads "
            "the per-fraction adapter to HF with `huggingface_hub.list_repo_files` "
            "verification."
        ),
    )
    ap.add_argument(
        "--trajectory-suffix",
        default="",
        help=(
            "Optional suffix on the trajectory subdir under --slab-root. "
            "Default empty = canonical path eval_results/issue_530/"
            "<slug>_seed<S>/trajectory.json."
        ),
    )
    ap.add_argument(
        "--wandb-suffix",
        default="",
        help=(
            "Optional suffix on the WandB run name + HF subfolder. Default = "
            "auto-built `_eps{epochs}_lr{lr:g}` (so #530 runs are visually "
            "distinguishable from #504's lr-only suffix on the WandB board)."
        ),
    )
    ap.add_argument(
        "--max-new-tokens-eval",
        type=int,
        default=2048,
        help=(
            "Eval max_new_tokens. CLAUDE.md rule: ≥ 2× longest trained completion. Default 2048."
        ),
    )
    ap.add_argument(
        "--max-model-len-eval",
        type=int,
        default=None,
        help=(
            "vLLM max_model_len (prompt + generation) for the nested eval. "
            "Must be >= --max-new-tokens-eval + 512 (prompt headroom). If "
            "unset, computed as max(2048, max_new_tokens_eval + 512)."
        ),
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny slice: 2 checkpoints, 256 eval max_new_tokens. Used by "
            "i530_smoke.py for the local pre-pod smoke run."
        ),
    )
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL.")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--hf-path-suffix",
        default="",
        help=(
            "Round-collision-avoidance suffix appended to the HF subfolder + "
            "local runs subdir. Default empty = canonical "
            "adapters/issue_530/<slug>_seed<S>."
        ),
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "ASSIGNED physical GPU index. Threaded to train_one_cell(gpu_id=...); "
            "train/sft.py SETS CUDA_VISIBLE_DEVICES to this so the cell + its "
            "nested eval run on physical GPU <gpu-id>."
        ),
    )
    ap.add_argument(
        "--source",
        default=None,
        help=(
            "Source persona name override. Default = villain (the canonical "
            "#504/#530 source). The plan §4.2 inheritance is explicit; "
            "#530 does NOT fork the source."
        ),
    )
    ap.add_argument(
        "--trajectory-hf-repo",
        default=TRAJECTORY_HF_REPO_DEFAULT,
        help=(
            f"HF model repo for per-fraction trajectory-checkpoint persistence "
            f"(default {TRAJECTORY_HF_REPO_DEFAULT!r}). The dispatcher sets "
            "EPM_PERSIST_TRAJECTORY_HF_REPO + EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER "
            "before train_one_cell so each fraction adapter is uploaded inline "
            "AND verified via huggingface_hub.list_repo_files. Set to empty "
            "string to disable per-fraction persistence (only the final adapter "
            "would land on HF via the legacy cfg.hf_path_in_repo path)."
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

    # Carry-over data dependencies from #472 (persona bank, centroids,
    # on-policy R) are gitignored. Pull them from HF at the pinned
    # revision before touching disk. Idempotent: a no-op when files are
    # already local (i.e. when i530_sweep.py already pulled them, or
    # when running on a pod where a prior cell landed them).
    from explore_persona_space.experiments.contrastive_neg_geometry_530.data_deps import (
        prepare_data_dependencies,
    )

    log.info("[phase=cell_prepare_data] auto-downloading #472 carry-over artifacts (idempotent)")
    prepare_data_dependencies()

    # ── Marker tokenizer pre-spawn assert (CLAUDE.md marker-leakage rule). ──
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_IM_END_TOKEN_ID,
        MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        MARKER_TEXT,
        SOURCE_PERSONA,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenizer assertion FAILED: encode({MARKER_TEXT!r})={ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]."
        )
    log.info("[phase=preflight] marker assertion PASS: %r -> %s", MARKER_TEXT, ids)

    # Load Phase 0.5 cell resolution (arm_to_n + optional smoke_mid_band_n + panel).
    arm_to_n_payload = json.loads(args.arm_to_n_json.read_text())
    arm_to_positioned_n = arm_to_n_payload.get("arm_to_positioned_n", {})
    smoke_mid_band_n = arm_to_n_payload.get("smoke_mid_band_n")
    held_out_panel: list[str] = arm_to_n_payload.get("held_out_panel", [])
    if (
        args.cell.startswith(("c504_smoke_", "c504v2_smoke_", "c504v3_smoke_"))
        and smoke_mid_band_n is None
    ):
        raise ValueError(
            f"--arm-to-n-json {args.arm_to_n_json} is missing 'smoke_mid_band_n' but the cell "
            f"{args.cell!r} is a smoke cell that requires it."
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
    from explore_persona_space.experiments.contrastive_neg_geometry_504.build_training_data import (
        build_cell_504,
    )

    # #530 namespace: adapters under adapters/issue_530/<slug>_seed<S>.
    run_slug = f"{args.cell}_seed{args.seed}{args.hf_path_suffix}"
    run_dir = args.runs_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    traj_subdir = f"{args.cell}_seed{args.seed}{args.trajectory_suffix}"
    out_traj = args.slab_root / traj_subdir / "trajectory.json"
    sentinel = args.log_dir / f"issue-530-{args.cell}-seed{args.seed}-results.json"

    bank = load_persona_bank(args.bank_path)
    r_train = load_r_artifact(args.r_train_path)
    q_train, _q_eval = get_train_eval_questions()

    effective_source = args.source if args.source is not None else SOURCE_PERSONA
    if effective_source not in bank:
        raise KeyError(
            f"--source {effective_source!r} missing from persona bank at {args.bank_path}."
        )
    log.info(
        "[phase=source] effective source persona = %r (CLI --source=%r, default=%r)",
        effective_source,
        args.source,
        SOURCE_PERSONA,
    )

    # ── Phase: build training data (CPU). ────────────────────────────────────
    log.info("[phase=build_%s] building training data via cell_resolution_504", args.cell)
    build_cell_504(
        args.cell,
        train_jsonl,
        r_train=r_train,
        arm_to_positioned_n=arm_to_positioned_n,
        q_train=q_train,
        persona_bank=bank,
        source=effective_source,
        marker_text=MARKER_TEXT,
        smoke_mid_band_n=smoke_mid_band_n,
        seed=args.seed,
    )

    # ── Parse --checkpoint-fractions (always set; default is the #530 4-frac
    # cadence). Sorted, deduped, strictly positive, ≤ 1. ─────────────────────
    try:
        parsed = tuple(
            sorted(float(x.strip()) for x in args.checkpoint_fractions.split(",") if x.strip())
        )
    except ValueError as exc:
        raise ValueError(
            f"--checkpoint-fractions {args.checkpoint_fractions!r} could not "
            f"parse as a comma-separated list of floats: {exc}."
        ) from exc
    if not parsed:
        raise ValueError(
            f"--checkpoint-fractions {args.checkpoint_fractions!r} parsed to "
            f"an empty tuple; need at least one positive fraction."
        )
    if any(f <= 0 or f > 1.0 for f in parsed):
        raise ValueError(
            f"--checkpoint-fractions {args.checkpoint_fractions!r} contains "
            f"out-of-range values; each must be in (0, 1]."
        )
    fractions = (0.25, 0.5, 1.0) if args.smoke else parsed
    log.info(
        "[phase=train_%s] checkpoint fractions = %s (smoke=%s)",
        args.cell,
        fractions,
        args.smoke,
    )

    effective_lr = args.lr
    effective_epochs = args.epochs

    # WandB suffix: auto-build #530's joint eps+lr suffix when not overridden.
    wandb_suffix = args.wandb_suffix or f"_eps{effective_epochs}_lr{effective_lr:g}"

    # ── Configure per-fraction HF persistence (plan §4.3). The helper in
    # train_cell.py reads EPM_PERSIST_TRAJECTORY_HF_REPO + _SUBFOLDER. Per
    # plan §4.3 the persistence subfolder is
    # adapters/issue_530/<config_slug>/ — i.e. matches the `hf_path_in_repo`
    # below but WITHOUT the trailing `_seed<S>` so the per-fraction subdirs
    # land under the cell directory. The plan's text actually uses
    # `adapters/issue_530/<config_slug>/ckpt_frac{N}` where `<config_slug>`
    # in plan §4.3 == the cell slug × seed identifier we name `run_slug`.
    # Use the same `run_slug` we use for the final-adapter path so the
    # trajectory + final adapter live under the SAME cell directory.
    if args.trajectory_hf_repo:
        os.environ["EPM_PERSIST_TRAJECTORY_HF_REPO"] = args.trajectory_hf_repo
        os.environ["EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER"] = f"adapters/issue_530/{run_slug}"
        # Plan-doc alias: the plan §4.3 names these `EPM_PERSIST_ADAPTER_*`.
        # Set both so a plan-reader using either name finds them.
        os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = args.trajectory_hf_repo
        os.environ["EPM_PERSIST_ADAPTER_HF_SUBFOLDER"] = f"adapters/issue_530/{run_slug}"
        log.info(
            "[phase=train_%s] per-fraction HF persistence: %s/%s/ckpt_frac{N}",
            args.cell,
            args.trajectory_hf_repo,
            f"adapters/issue_530/{run_slug}",
        )
    else:
        log.info(
            "[phase=train_%s] per-fraction HF persistence disabled (--trajectory-hf-repo empty)",
            args.cell,
        )

    log.info(
        "[phase=train_%s] training (rank=%d, alpha=%d, lr=%g, epochs=%d, smoke=%s, "
        "suppress_at_post_response_slot=%s, im_end_id=%s, wandb_suffix=%s)",
        args.cell,
        args.chosen_rank,
        args.chosen_alpha,
        effective_lr,
        effective_epochs,
        args.smoke,
        MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        MARKER_IM_END_TOKEN_ID,
        wandb_suffix,
    )
    train_result = train_one_cell(
        cell_slug=args.cell,
        seed=args.seed,
        train_jsonl=train_jsonl,
        output_dir=final_adapter_dir,
        ckpt_root=ckpt_root,
        fractions=fractions,
        fallback=False,
        report_to=args.report_to,
        gpu_id=args.gpu_id,
        lr_override=effective_lr,
        epochs_override=effective_epochs,
        lora_r_override=args.chosen_rank,
        lora_alpha_override=args.chosen_alpha,
        # #530 sources adapters under adapters/issue_530/<slug>_seed<S>.
        hf_path_in_repo_override=f"adapters/issue_530/{run_slug}",
        run_name_override=f"issue530_{run_slug}{wandb_suffix}",
        # #530 inherits the #477 v6 marker-suppress fix (plan §4.2 inheritance).
        marker_suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        marker_im_end_token_id=MARKER_IM_END_TOKEN_ID,
    )
    ckpt_index_path = run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(train_result["checkpoint_index"], indent=2))
    log.info(
        "[phase=train_%s] done; checkpoints=%s",
        args.cell,
        list(train_result["checkpoint_index"]),
    )

    # Free in-process LoRA-training GPU memory before nested vLLM eval.
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase: eval_trajectory (NESTED subprocess: vLLM teardown isolation). ─
    eval_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if eval_cvd != str(args.gpu_id):
        raise RuntimeError(
            f"Pre-eval CUDA_VISIBLE_DEVICES={eval_cvd!r} != assigned --gpu-id={args.gpu_id}; "
            "the nested eval subprocess would run on the wrong physical GPU. "
            "train/sft.py should have set CVD=str(gpu_id) during training."
        )
    log.info(
        "[phase=eval_%s] inherited CUDA_VISIBLE_DEVICES=%s (physical GPU %d)",
        args.cell,
        eval_cvd,
        args.gpu_id,
    )
    eval_max_new_tokens = 256 if args.smoke else args.max_new_tokens_eval
    eval_prompt_headroom = 512
    eval_max_model_len = (
        args.max_model_len_eval
        if args.max_model_len_eval is not None
        else max(2048, eval_max_new_tokens + eval_prompt_headroom)
    )
    eval_cmd = [
        "uv",
        "run",
        "python",
        # #530 reuses the #504 eval rig byte-for-byte (the adapter-applied
        # guard + per-batch byte-identical guard ported from issue-504).
        "scripts/i504_eval_trajectory.py",
        "--cell",
        args.cell,
        "--seed",
        str(args.seed),
        "--checkpoint-index",
        str(ckpt_index_path),
        "--out-path",
        str(out_traj),
        "--bank-path",
        str(args.bank_path),
        "--r-eval-path",
        str(args.r_eval_path),
        "--panel-json",
        str(args.arm_to_n_json),
        "--max-lora-rank",
        str(args.chosen_rank),
        "--max-new-tokens",
        str(eval_max_new_tokens),
        "--max-model-len",
        str(eval_max_model_len),
        "--source",
        effective_source,
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
    log.info(
        "[phase=eval_%s] nested eval subprocess (max_new_tokens=%d, max_model_len=%d): %s",
        args.cell,
        eval_max_new_tokens,
        eval_max_model_len,
        " ".join(eval_cmd),
    )
    subprocess.run(eval_cmd, env={**os.environ}, check=True)

    if not out_traj.exists():
        raise RuntimeError(
            f"[{args.cell}] eval_trajectory subprocess exited 0 but {out_traj} missing — "
            "silent eval failure (feedback_eval_script_silent_not_present_misdiagnosis)."
        )

    _write_sentinel(
        sentinel,
        kind="epm:progress",
        phase=f"cell_done_{args.cell}_seed{args.seed}",
        note={
            "cell": args.cell,
            "seed": args.seed,
            "trajectory_path": str(out_traj),
            "adapter_hf_path": f"adapters/issue_530/{run_slug}",
            "hf_path_suffix": args.hf_path_suffix,
            "checkpoint_index": str(ckpt_index_path),
            "n_held_out_panel": len(held_out_panel),
            "lr": effective_lr,
            "epochs": effective_epochs,
            "checkpoint_fractions": list(fractions),
        },
    )
    log.info("[phase=done] wrote sentinel → %s", sentinel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
