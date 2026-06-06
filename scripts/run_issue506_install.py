#!/usr/bin/env python3
"""Issue #506 dispatcher — FWFT vs LoRA r=16 (vs LoRA r=256) marker install
on Qwen3.5-27B, Phase 1 install + Phase 2 benign-SFT survival per arm.

Per-phase trainer entrypoint. Mirrors scripts/run_issue475_cot_install.py:
one process per (arm, phase), single GPU pin for LoRA arms, multi-GPU via
accelerate for FWFT.

**Scope note** (intentional downscope from plan v3): this dispatcher runs
ONE (arm, phase) per invocation. Phase-0a preflights, Stage-0 install-
validity gate, and Phase-3 eval are launched separately by the experimenter
in the documented order — Phase-0a smokes → marker_preflight (auto-fires
here on every invocation) → Phase 1 per arm → Stage-0 probe → Phase 2 per
arm → eval per (arm, phase). The orchestrator (`/issue`) chains them via
`scripts/poll_pipeline.py` and the experimenter's launch sequence; the
dispatcher's contract is `(arm, phase) → trained checkpoint on HF Hub`,
not full-pipeline orchestration.

Loss regime (plan §11.1):
  ASSISTANT-only cross-entropy on ALL arms. LoRA path: native (TRL
  auto-resolves completion_only_loss=True from prompt+completion JSONL).
  FWFT path: via the Phase-0a item-1 wiring patch on
  scripts/train_stage_sft.py + ``completion_only_loss: true`` in the
  Hydra YAML. Verified pre-launch by scripts/smoke_issue506_label_mask_audit.py.

Phase 1 (per arm; install):
    uv run python scripts/run_issue506_install.py \\
        --arm lora_r16 --phase phase1 --seed 42 --gpu 0
    uv run python scripts/run_issue506_install.py \\
        --arm lora_r256 --phase phase1 --seed 42 --gpu 0
    uv run python scripts/run_issue506_install.py \\
        --arm fwft --phase phase1 --seed 42

Phase 2 (per arm; benign-medical survival):
    uv run python scripts/run_issue506_install.py \\
        --arm lora_r16 --phase phase2 --seed 42 --gpu 0

Smoke (unified — IS the sweep with one arm at smaller size):
    uv run python scripts/run_issue506_install.py \\
        --arm lora_r16 --phase phase1 --seed 42 --gpu 0 --smoke

The 3-arms-parallel sweep is launched as three concurrent invocations,
one per pod/GPU. The dispatcher does NOT internally fan out — that's the
experimenter's job (each invocation pins to its own GPU via --gpu for
the LoRA arms; the FWFT arm uses all 8 GPUs of its pod).
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

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT, bootstrap  # noqa: E402

bootstrap(log_name="run_issue506_install")

from _issue506_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    HUB_FWFT_MODEL_REPO,
    HUB_MODEL_REPO,
    MARKER_TEXT,
    PHASE1_DATA_PATH,
    PHASE1_EPOCHS,
    PHASE1_GRAD_ACCUM,
    PHASE1_LORA_ALPHA,
    PHASE1_LORA_DROPOUT,
    PHASE1_LORA_R_DEFAULT,
    PHASE1_LORA_R_HIGH,
    PHASE1_LR,
    PHASE1_MAX_LENGTH,
    PHASE1_PER_DEVICE_BS,
    PHASE1_WARMUP_RATIO,
    PHASE1_WEIGHT_DECAY,
    PHASE2_DATASET_HF_PATH,
    PHASE2_DATASET_REL,
    PHASE2_EPOCHS,
    PHASE2_LR,
    PHASE2_MAX_LENGTH,
    WANDB_PROJECT,
    adapter_subfolder,
    fwft_subfolder,
    marker_preflight,
)

log = logging.getLogger("run_issue506_install")


def _per_phase_output_dir(arm: str, seed: int, phase: str) -> Path:
    return PROJECT_ROOT / "models" / f"issue506_{arm}_seed{seed}_{phase}"


def _arm_data_path(smoke: bool) -> Path:
    """All three arms share the #475 plain-arm install data."""
    base = PHASE1_DATA_PATH
    if not base.exists():
        raise FileNotFoundError(
            f"Phase-1 install data missing: {base}. Run "
            "`uv run python scripts/fetch_issue506_phase1_dataset.py` first."
        )
    if not smoke:
        return base
    # Smoke subset — ~10% of full pool, cached. Same pattern as #475 dispatcher.
    n_total = sum(1 for ln in base.read_text().splitlines() if ln.strip())
    n_smoke = min(n_total, max(6, n_total // 10))
    smoke_path = base.parent / "train_smoke.jsonl"
    if smoke_path.exists() and smoke_path.stat().st_mtime > base.stat().st_mtime:
        log.info("Smoke subset cache hit: %s (%d rows)", smoke_path, n_smoke)
        return smoke_path
    rows = [ln for ln in base.read_text().splitlines() if ln.strip()][:n_smoke]
    smoke_path.write_text("\n".join(rows) + "\n")
    log.info("Wrote smoke subset: %s (%d of %d rows)", smoke_path, n_smoke, n_total)
    return smoke_path


def _ensure_phase2_dataset_local() -> Path:
    local = PROJECT_ROOT / PHASE2_DATASET_REL
    if local.exists():
        log.info("Phase 2 dataset cache hit: %s", local)
        return local
    log.info("Phase 2 dataset missing locally — fetching from HF Hub.")
    from explore_persona_space.orchestrate.hub import download_dataset

    local.parent.mkdir(parents=True, exist_ok=True)
    out = download_dataset(
        path_in_repo=PHASE2_DATASET_HF_PATH,
        local_path=str(local),
    )
    if not out or not Path(out).exists():
        raise RuntimeError(f"Failed to fetch Phase 2 dataset from HF Hub: {PHASE2_DATASET_HF_PATH}")
    return local


# ── LoRA arms (in-process train_lora) ────────────────────────────────────────


def _run_lora_phase1(args: argparse.Namespace, *, lora_r: int) -> dict:
    """LoRA Phase 1 install via train_lora(). LoRA r is the only arm-difference."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    arm = args.arm
    seed = args.seed
    data_path = _arm_data_path(args.smoke)
    output_dir = _per_phase_output_dir(arm, seed, "phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainLoraConfig(
        gpu_id=args.gpu,
        epochs=PHASE1_EPOCHS,
        lr=PHASE1_LR,
        lora_r=lora_r,
        lora_alpha=PHASE1_LORA_ALPHA,
        lora_dropout=PHASE1_LORA_DROPOUT,
        batch_size=PHASE1_PER_DEVICE_BS,
        grad_accum=PHASE1_GRAD_ACCUM,
        max_length=PHASE1_MAX_LENGTH,
        warmup_ratio=PHASE1_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue506_{arm}_seed{seed}_phase1",
        report_to="wandb",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        weight_decay=PHASE1_WEIGHT_DECAY,
        marker_only_loss=False,
        marker_text=MARKER_TEXT,
        hf_repo=HUB_MODEL_REPO,
        hf_path_in_repo=f"adapters/{adapter_subfolder(arm, seed, 'phase1')}",
        hf_upload=True,
    )

    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("EPM_PERSIST_ADAPTER_HF_REPO", HUB_MODEL_REPO)
    os.environ.setdefault(
        "EPM_PERSIST_ADAPTER_HF_SUBFOLDER",
        f"adapters/{adapter_subfolder(arm, seed, 'phase1')}",
    )
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    log.info(
        "LoRA Phase 1: arm=%s r=%d seed=%d gpu=%d data=%s",
        arm,
        lora_r,
        seed,
        args.gpu,
        data_path,
    )
    t0 = time.time()
    adapter_path, train_loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(output_dir / "adapter"),
        cfg=cfg,
    )
    wall_m = (time.time() - t0) / 60
    return {
        "phase": "phase1",
        "arm": arm,
        "seed": seed,
        "smoke": args.smoke,
        "train_loss": train_loss,
        "adapter_path": adapter_path,
        "adapter_hf_subfolder": f"adapters/{adapter_subfolder(arm, seed, 'phase1')}",
        "wall_minutes": round(wall_m, 1),
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "config": {
            "lr": PHASE1_LR,
            "epochs": PHASE1_EPOCHS,
            "per_device_bs": PHASE1_PER_DEVICE_BS,
            "grad_accum": PHASE1_GRAD_ACCUM,
            "max_length": PHASE1_MAX_LENGTH,
            "lora_r": lora_r,
            "lora_alpha": PHASE1_LORA_ALPHA,
            "lora_dropout": PHASE1_LORA_DROPOUT,
            "warmup_ratio": PHASE1_WARMUP_RATIO,
            "weight_decay": PHASE1_WEIGHT_DECAY,
            "loss": "assistant_only_ce_native_prompt_completion",
        },
    }


def _run_lora_phase2(args: argparse.Namespace, *, lora_r: int) -> dict:
    """LoRA Phase 2 — CONTINUE the Phase-1 adapter under benign medical SFT.

    Uses the ``existing_adapter_path`` field added to ``TrainLoraConfig``
    in Phase-0a item 2 — the trainer loads the Phase-1 adapter via
    ``PeftModel.from_pretrained(base, path, is_trainable=True)`` instead
    of attaching a fresh LoRA. The plan §4.7 "same Phase-1 adapter survives
    Phase 2" question hinges on this continuation contract.
    """
    from huggingface_hub import snapshot_download

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    arm = args.arm
    seed = args.seed
    output_dir = _per_phase_output_dir(arm, seed, "phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_sub = f"adapters/{adapter_subfolder(arm, seed, 'phase1')}"
    log.info("Phase 2: downloading Phase-1 adapter %s/%s", HUB_MODEL_REPO, phase1_sub)
    phase1_local = snapshot_download(
        repo_id=HUB_MODEL_REPO,
        allow_patterns=[f"{phase1_sub}/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    phase1_adapter_path = Path(phase1_local) / phase1_sub
    if not phase1_adapter_path.exists():
        raise RuntimeError(f"Phase-1 adapter missing: {phase1_adapter_path}")
    log.info("Phase 1 adapter resolved to: %s", phase1_adapter_path)

    data_path = _ensure_phase2_dataset_local()
    log.info("Phase 2 dataset: %s", data_path)

    cfg = TrainLoraConfig(
        gpu_id=args.gpu,
        epochs=PHASE2_EPOCHS,
        lr=PHASE2_LR,
        lora_r=lora_r,  # informational only — adapter's saved r wins
        lora_alpha=PHASE1_LORA_ALPHA,
        lora_dropout=PHASE1_LORA_DROPOUT,
        batch_size=PHASE1_PER_DEVICE_BS,
        grad_accum=PHASE1_GRAD_ACCUM,
        max_length=PHASE2_MAX_LENGTH,
        warmup_ratio=PHASE1_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue506_{arm}_seed{seed}_phase2",
        report_to="wandb",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        weight_decay=PHASE1_WEIGHT_DECAY,
        marker_only_loss=False,
        marker_text=MARKER_TEXT,
        hf_repo=HUB_MODEL_REPO,
        hf_path_in_repo=f"adapters/{adapter_subfolder(arm, seed, 'phase2')}",
        hf_upload=True,
        existing_adapter_path=str(phase1_adapter_path),
    )

    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("EPM_PERSIST_ADAPTER_HF_REPO", HUB_MODEL_REPO)
    os.environ.setdefault(
        "EPM_PERSIST_ADAPTER_HF_SUBFOLDER",
        f"adapters/{adapter_subfolder(arm, seed, 'phase2')}",
    )
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    log.info(
        "LoRA Phase 2 (continue-adapter): arm=%s seed=%d gpu=%d phase1_adapter=%s",
        arm,
        seed,
        args.gpu,
        phase1_adapter_path,
    )
    t0 = time.time()
    adapter_path, train_loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(output_dir / "adapter"),
        cfg=cfg,
    )
    wall_m = (time.time() - t0) / 60
    return {
        "phase": "phase2",
        "arm": arm,
        "seed": seed,
        "train_loss": train_loss,
        "adapter_path": adapter_path,
        "adapter_hf_subfolder": f"adapters/{adapter_subfolder(arm, seed, 'phase2')}",
        "phase1_adapter_hf_subfolder": phase1_sub,
        "phase2_handoff": "continue_adapter",
        "wall_minutes": round(wall_m, 1),
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "config": {
            "lr": PHASE2_LR,
            "epochs": PHASE2_EPOCHS,
            "max_length": PHASE2_MAX_LENGTH,
            "dataset": PHASE2_DATASET_REL,
        },
    }


# ── FWFT arm (subprocess via accelerate launch + train_stage_sft) ──────────


def _run_fwft_phase1(args: argparse.Namespace) -> dict:
    """FWFT Phase 1 — accelerate launch + train_stage_sft.py + ZeRO-3."""
    arm = args.arm
    seed = args.seed
    data_path = _arm_data_path(args.smoke)
    output_dir = _per_phase_output_dir(arm, seed, "phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / "configs" / "condition" / "c_issue506_install_fwft.yaml"
    ds_config = PROJECT_ROOT / "configs" / "deepspeed" / "zero3_cpu_offload.json"

    # Issue #506 fwft-hub-path-mismatch fix: pass the exact HF repo +
    # subfolder train_stage_sft must upload to, matching the reader path in
    # ``eval_issue506._resolve_ckpt`` and ``_run_fwft_phase2``.
    fwft_sub_phase1 = fwft_subfolder(seed, "phase1")
    # Issue #506 Round-3 must-fix #1: delete the local Phase-1 FWFT
    # checkpoint after verified upload so the next phase's save fits
    # under the MooseFS ~130GB pod quota (54 + 54 + ~5GB intermediates
    # = ~113GB only if we delete between phases; plan v3 §9.4 + Asn 13).
    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        str(ds_config),
        "--num_processes",
        str(args.num_gpus),
        str(PROJECT_ROOT / "scripts" / "train_stage_sft.py"),
        "--config",
        str(config_path),
        "--dataset",
        str(data_path),
        "--output-dir",
        str(output_dir / "model"),
        "--seed",
        str(seed),
        "--upload",
        "--hub-repo-id",
        HUB_FWFT_MODEL_REPO,
        "--hub-subfolder",
        fwft_sub_phase1,
        "--delete-after-upload-verified",
    ]
    env = {**os.environ}
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    # Belt-and-suspenders: env-var also seen by inline-upload paths that
    # don't read CLI args, so the same upload target is honored everywhere.
    env["EPM_HUB_REPO_ID"] = HUB_FWFT_MODEL_REPO
    env["EPM_HUB_SUBFOLDER"] = fwft_sub_phase1

    log.info("FWFT Phase 1: launching %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    wall_m = (time.time() - t0) / 60
    if rc != 0:
        raise RuntimeError(f"FWFT Phase 1 train_stage_sft exited rc={rc}")
    return {
        "phase": "phase1",
        "arm": arm,
        "seed": seed,
        "smoke": args.smoke,
        "model_path": str(output_dir / "model"),
        "fwft_hf_repo": HUB_FWFT_MODEL_REPO,
        "fwft_hf_subfolder": fwft_sub_phase1,
        "fwft_hf_url": (
            f"https://huggingface.co/{HUB_FWFT_MODEL_REPO}/tree/main/{fwft_sub_phase1}"
        ),
        "wall_minutes": round(wall_m, 1),
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "config": {
            "lr": PHASE1_LR,
            "epochs": PHASE1_EPOCHS,
            "per_device_bs": PHASE1_PER_DEVICE_BS,
            "grad_accum": 2,
            "num_gpus": args.num_gpus,
            "max_length": PHASE1_MAX_LENGTH,
            "warmup_ratio": PHASE1_WARMUP_RATIO,
            "weight_decay": PHASE1_WEIGHT_DECAY,
            "loss": "assistant_only_ce_via_completion_only_loss",
            "deepspeed_stage": 3,
        },
    }


def _run_fwft_phase2(args: argparse.Namespace) -> dict:
    """FWFT Phase 2 — load Phase-1 consolidated checkpoint as base model."""
    arm = args.arm
    seed = args.seed
    output_dir = _per_phase_output_dir(arm, seed, "phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / "configs" / "condition" / "c_issue506_phase2_benign_medical.yaml"
    ds_config = PROJECT_ROOT / "configs" / "deepspeed" / "zero3_cpu_offload.json"
    data_path = _ensure_phase2_dataset_local()

    # Phase-1 consolidated checkpoint must already be on HF Hub
    # (train_stage_sft.py --upload pushes after training). The Phase 2
    # launch reads it via --input-model.
    phase1_local = _per_phase_output_dir(arm, seed, "phase1") / "model"
    if not phase1_local.exists():
        # Pull from HF Hub via snapshot_download (clean local copy).
        from huggingface_hub import snapshot_download

        log.info("Phase 2: pulling FWFT Phase-1 checkpoint from %s", HUB_FWFT_MODEL_REPO)
        phase1_local = Path(
            snapshot_download(
                repo_id=HUB_FWFT_MODEL_REPO,
                allow_patterns=[f"{fwft_subfolder(seed, 'phase1')}/*"],
                token=os.environ.get("HF_TOKEN"),
            )
        ) / fwft_subfolder(seed, "phase1")

    # Same hub-path fix as Phase 1: pass the explicit Phase-2 FWFT subfolder
    # so eval_issue506._resolve_ckpt can resolve it under HUB_FWFT_MODEL_REPO.
    fwft_sub_phase2 = fwft_subfolder(seed, "phase2")
    # Issue #506 Round-3 must-fix #1: same upload-and-delete for Phase 2
    # so the next FWFT cell (if any) and eval download workspace stay
    # under the quota.
    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        str(ds_config),
        "--num_processes",
        str(args.num_gpus),
        str(PROJECT_ROOT / "scripts" / "train_stage_sft.py"),
        "--config",
        str(config_path),
        "--input-model",
        str(phase1_local),
        "--dataset",
        str(data_path),
        "--output-dir",
        str(output_dir / "model"),
        "--seed",
        str(seed),
        "--upload",
        "--hub-repo-id",
        HUB_FWFT_MODEL_REPO,
        "--hub-subfolder",
        fwft_sub_phase2,
        "--delete-after-upload-verified",
    ]
    env = {**os.environ}
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["EPM_HUB_REPO_ID"] = HUB_FWFT_MODEL_REPO
    env["EPM_HUB_SUBFOLDER"] = fwft_sub_phase2

    log.info("FWFT Phase 2: launching %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    wall_m = (time.time() - t0) / 60
    if rc != 0:
        raise RuntimeError(f"FWFT Phase 2 train_stage_sft exited rc={rc}")
    return {
        "phase": "phase2",
        "arm": arm,
        "seed": seed,
        "model_path": str(output_dir / "model"),
        "fwft_hf_repo": HUB_FWFT_MODEL_REPO,
        "fwft_hf_subfolder": fwft_sub_phase2,
        "fwft_hf_url": (
            f"https://huggingface.co/{HUB_FWFT_MODEL_REPO}/tree/main/{fwft_sub_phase2}"
        ),
        "phase1_hf_subfolder": fwft_subfolder(seed, "phase1"),
        "wall_minutes": round(wall_m, 1),
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "config": {
            "lr": PHASE2_LR,
            "epochs": PHASE2_EPOCHS,
            "max_length": PHASE2_MAX_LENGTH,
            "num_gpus": args.num_gpus,
            "dataset": PHASE2_DATASET_REL,
        },
    }


# ── Arg parsing ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Issue #506 dispatcher — FWFT vs LoRA marker install on Qwen3.5-27B."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phase", choices=("phase1", "phase2"), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to pin (CVD); LoRA arms only. Ignored for FWFT.",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs for accelerate-launched FWFT (ignored for LoRA arms).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="10%% subset of install data + smaller compute budget. Phase 2 forbidden.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    marker_preflight()

    if args.phase == "phase2" and args.smoke:
        raise SystemExit("--smoke + --phase phase2 is invalid (plan §4.9).")

    if args.arm == "lora_r16":
        if args.phase == "phase1":
            result = _run_lora_phase1(args, lora_r=PHASE1_LORA_R_DEFAULT)
        else:
            result = _run_lora_phase2(args, lora_r=PHASE1_LORA_R_DEFAULT)
    elif args.arm == "lora_r256":
        if args.phase == "phase1":
            result = _run_lora_phase1(args, lora_r=PHASE1_LORA_R_HIGH)
        else:
            result = _run_lora_phase2(args, lora_r=PHASE1_LORA_R_HIGH)
    elif args.arm == "fwft":
        result = _run_fwft_phase1(args) if args.phase == "phase1" else _run_fwft_phase2(args)
    else:
        raise SystemExit(f"Unknown arm: {args.arm}")

    out_root = PROJECT_ROOT / "eval_results" / "issue_506"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.phase}_{args.arm}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
