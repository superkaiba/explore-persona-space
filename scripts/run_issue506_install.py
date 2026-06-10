#!/usr/bin/env python3
"""Issue #506 dispatcher — FWFT vs LoRA r=16 (vs LoRA r=256) marker install
on Qwen3-32B, Phase 1 install + Phase 2 benign-SFT survival per arm.

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
import contextlib
import json
import logging
import os
import shutil
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


# Slack on top of every disk-headroom projection, to absorb metadata files,
# tokenizer/config artifacts, and a margin for FS overhead. 10 GB matches the
# slack used in i474_phase0_preflight.py + run_issue458_sweep.sh.
_DISK_PROBE_SLACK_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

# Round-8 fix: stage HF safetensors on /dev/shm (RAM tmpfs) during conversion
# so the MooseFS volume only sees ONE full copy at a time. RunPod 8xH{100,200}
# pods carry ≥256 GB RAM; /dev/shm defaults to 50% of RAM (~128 GB), which
# comfortably holds the ~64 GB bf16 state for Qwen3-32B.
_SHM_ROOT = Path("/dev/shm")


def _dir_size_bytes(path: Path) -> int:
    """Recursively sum file sizes under ``path``. Returns 0 if missing.

    Used to estimate the DS-native dir's footprint so the headroom probe
    knows how much will be freed when we delete it (and to refuse to
    proceed if the ds-native dir is itself bigger than expected).
    """
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(str(path)):
        root_p = Path(root)
        for name in files:
            with contextlib.suppress(OSError):
                total += (root_p / name).stat().st_size
    return total


def _estimate_hf_bf16_bytes(model_id: str) -> int:
    """Estimate the on-disk size of an HF bf16 safetensors save for ``model_id``.

    Reads ``config.json`` only (no weights) via ``AutoConfig.from_pretrained``,
    derives a parameter-count estimate from the config, and multiplies by 2
    bytes (bf16). For Qwen3-32B this returns ~64 GB. Adds 5% slack for
    safetensors header / index file / tokenizer / config bytes.

    Falls back to a conservative 80 GB on any failure to read the config — we
    would rather have the preflight be slightly pessimistic and fail loud than
    silently let an under-estimate slip past.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        ffn = int(getattr(cfg, "intermediate_size", 0) or (4 * hidden))
        heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        kv_heads = int(getattr(cfg, "num_key_value_heads", heads) or heads)
        head_dim = hidden // max(heads, 1)
        vocab = int(getattr(cfg, "vocab_size", 0) or 0)
        # Per-layer params: 4 attn matrices (Q+K+V+O), 3 FFN matrices for
        # gated MLP (gate, up, down), plus 2 layernorms (small, ignored).
        # GQA-aware Q/K/V sizes: Q is hidden*hidden, K and V are hidden*(kv_heads*head_dim).
        q_params = hidden * hidden
        kv_params = 2 * (hidden * kv_heads * head_dim)
        o_params = hidden * hidden
        ffn_params = 3 * (hidden * ffn)
        per_layer = q_params + kv_params + o_params + ffn_params
        embed_params = vocab * hidden  # tied lm_head common in Qwen-2.5 / Qwen-3
        total_params = layers * per_layer + embed_params
        if total_params <= 0:
            raise ValueError(f"degenerate param count for {model_id}: {total_params}")
        # bf16 → 2 bytes; 5% slack for safetensors header + tokenizer + config.
        estimate = int(total_params * 2 * 1.05)
        log.info(
            "FWFT post-train: HF bf16 size estimate for %s: %.2f GB "
            "(layers=%d, hidden=%d, ffn=%d, kv_heads=%d, vocab=%d, params=%.2fB)",
            model_id,
            estimate / 1024**3,
            layers,
            hidden,
            ffn,
            kv_heads,
            vocab,
            total_params / 1e9,
        )
        return estimate
    except Exception as e:
        # Fall back to a deliberately pessimistic 80 GB so the preflight errs
        # toward refusing risky conversions rather than silently letting them
        # slip through. Logged loud so the operator can spot the misconfig.
        log.warning(
            "FWFT post-train: could not estimate HF bf16 size for %s (%s); "
            "falling back to 80 GB pessimistic estimate.",
            model_id,
            e,
        )
        return 80 * 1024 * 1024 * 1024


def _disk_headroom_probe(target_dir: Path, n_bytes: int, label: str) -> None:
    """posix_fallocate probe — catches MooseFS EDQUOT / tmpfs full BEFORE we run.

    Mirrors ``i474_phase0_preflight.py::_disk_probe`` and
    ``pod_disk_guard.py::probe_quota_headroom``. Reserves ``n_bytes`` in a
    temp file under ``target_dir``, then deletes it. Raises ``RuntimeError``
    loud on EDQUOT/ENOSPC so the dispatcher refuses to proceed BEFORE the
    conversion subprocess writes one byte.

    ``label`` is included in error / log messages so the operator can tell
    which mount failed (``/workspace`` vs ``/dev/shm``).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize ``label`` so the probe filename never contains a slash —
    # otherwise a label like "/dev/shm" would resolve to
    # "/dev/shm/.issue506_disk_probe_/dev/shm_<pid>", which is a sub-path
    # with a missing intermediate directory.
    safe_label = label.strip("/").replace("/", "_") or "root"
    probe = target_dir / f".issue506_disk_probe_{safe_label}_{os.getpid()}"
    fd = os.open(str(probe), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        try:
            os.posix_fallocate(fd, 0, n_bytes)
        except OSError as e:
            raise RuntimeError(
                f"FWFT post-train: disk-headroom preflight FAILED on {label} "
                f"({target_dir}). posix_fallocate({n_bytes} bytes / "
                f"{n_bytes / 1024**3:.1f} GB) → errno={e.errno} {e.strerror}. "
                f"Refusing to start conversion — would EDQUOT/ENOSPC mid-write. "
                "See CLAUDE.md gotchas: MooseFS per-pod quota (~130 GB)."
            ) from e
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(probe)
    log.info(
        "FWFT post-train: disk-headroom probe OK on %s (%s): %.1f GB reserved + freed.",
        label,
        target_dir,
        n_bytes / 1024**3,
    )


def _run_conversion_with_shm_staging(
    *,
    ds_native_dir: Path,
    output_dir: Path,
    base_model_id: str,
    max_shard_size: str = "2GB",
    dtype: str = "bfloat16",
    skip_preflight: bool = False,
    converter_script: Path | None = None,
) -> None:
    """Convert DS-native → HF, staging on /dev/shm to bound the MooseFS peak.

    Round-8 fix (#506 reconciler v7 FAIL — MooseFS quota peak ~160 GB > 130 GB):
    Round-9 fix (#506 reconciler v8 FAIL — preflight ordering + try/finally
    cleanup for /dev/shm staging dir).

    The naive flow writes the HF safetensors directly into ``output_dir``
    (on /workspace = MooseFS) while the DS-native dir still occupies its
    ~96 GB on the same volume. Transient peak: DS-native + HF ≈ 160 GB,
    above the 130 GB hard quota.

    This helper instead:

      1. Sizes both DS-native and the projected HF artifact.
      2. Probes /dev/shm via ``posix_fallocate`` for the HF artifact's
         footprint plus 10 GB slack (unless ``skip_preflight=True`` for CPU
         smokes). The /workspace probe is deferred to step 5 below — see
         round-9 fix.
      3. Invokes ``convert_ds_zero3_to_hf.py`` writing to a /dev/shm staging
         dir — tmpfs RAM, NOT MooseFS — so /workspace only ever sees the
         DS-native dir during conversion.
      4. After the conversion subprocess exits 0, deletes the DS-native dir
         from /workspace via fail-loud ``shutil.rmtree`` (NO
         ``ignore_errors=True`` — round-7 code-review demanded fail-fast).
         Verifies the dir is gone and raises if not.
      5. Probes /workspace via ``posix_fallocate`` for the HF artifact's
         footprint plus 10 GB slack (round-9 fix). DS-native is gone by
         step 4, so the probe asks ~74 GB against ~130 GB available — fits
         cleanly within the MooseFS per-pod quota. Doing the probe BEFORE
         step 4 (as round-8 did) attempted ~96 + 106 = ~202 GB against the
         130 GB ceiling and tripped EDQUOT before conversion could start.
      6. Moves the staging dir from /dev/shm to ``output_dir`` on
         /workspace. Cross-FS move = copy-then-delete under the hood; by
         this point /workspace is empty (DS-native already deleted) so the
         peak on /workspace stays ≤ max(DS-native, HF) ≈ 96 GB.

    Steps 3 → 4 → 5 → 6 are wrapped in ``try/finally``: on any failure
    between staging-dir-creation and a successful move, the ``finally``
    block ``shutil.rmtree``'s the /dev/shm staging dir so ~64 GB of
    RAM-backed tmpfs does not leak until the next run / pod reboot.
    Failure-path cleanup is best-effort-loud (the original exception
    propagates; cleanup-failure is logged, never shadows it).

    On any failure the DS-native dir is preserved for post-mortem.
    On a clean exit, both the DS-native dir and the staging dir are gone;
    ``output_dir`` holds the HF artifact ready for upload.
    """
    if converter_script is None:
        converter_script = PROJECT_ROOT / "scripts" / "convert_ds_zero3_to_hf.py"
    if not converter_script.exists():
        raise RuntimeError(
            f"FWFT post-train: conversion script missing at {converter_script}. "
            f"DS-native shards are at {ds_native_dir}; refusing to proceed."
        )

    # /dev/shm staging dir name keyed on the output dir so concurrent runs
    # on the same pod (e.g. retry after partial failure) don't collide.
    shm_staging = _SHM_ROOT / f"issue506_{output_dir.name}_hf_staging"
    # Clean stale state from a prior failed run — fail-loud if removal fails.
    if shm_staging.exists():
        log.info(
            "FWFT post-train: removing stale /dev/shm staging dir %s "
            "(left over from a prior partial run).",
            shm_staging,
        )
        shutil.rmtree(str(shm_staging))
        if shm_staging.exists():
            raise RuntimeError(
                f"FWFT post-train: stale staging dir {shm_staging} still exists "
                "after shutil.rmtree. Refusing to proceed."
            )

    # ── Phase A: estimate sizes + run preflight probes ────────────────────
    ds_native_bytes = _dir_size_bytes(ds_native_dir)
    hf_bf16_bytes = _estimate_hf_bf16_bytes(base_model_id)
    log.info(
        "FWFT post-train: peak-disk projection — "
        "DS-native: %.1f GB (existing on /workspace), "
        "HF bf16: %.1f GB (to be written on /dev/shm, then moved to /workspace).",
        ds_native_bytes / 1024**3,
        hf_bf16_bytes / 1024**3,
    )

    if skip_preflight:
        log.warning(
            "FWFT post-train: SKIPPING disk-headroom preflight "
            "(EPM_SMOKE_PREFLIGHT_SKIP set; CPU smoke path)."
        )
    else:
        # /dev/shm probe — must fit the FULL HF artifact plus slack, because
        # the conversion writes the whole state_dict to staging before we
        # delete DS-native from /workspace.
        #
        # NOTE (round-9 reconciler v8 fix): the /workspace probe was
        # previously placed HERE, before Phase C, asking for
        # ``max(ds_native_bytes, hf_bf16_bytes) + slack ≈ 106 GB`` while the
        # DS-native dir's ~96 GB still occupied /workspace — MooseFS counts
        # ``posix_fallocate`` against the per-pod quota, so the probe
        # attempted ~96 + 106 = 202 GB on a 130 GB ceiling and tripped
        # EDQUOT BEFORE conversion could start. The workspace probe now runs
        # AFTER Phase D (DS-native delete) and BEFORE Phase E (cross-FS
        # move) — i.e. at the moment when /workspace is empty and we are
        # about to write the HF artifact. See Phase D.5 below.
        _disk_headroom_probe(
            _SHM_ROOT,
            hf_bf16_bytes + _DISK_PROBE_SLACK_BYTES,
            label="/dev/shm",
        )

    # ── Phase B: ensure target output_dir does NOT exist yet ──────────────
    # The downstream cross-FS move requires the destination to NOT exist
    # (shutil.move would otherwise nest the staging dir INSIDE output_dir).
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"FWFT post-train: output_dir {output_dir} already exists and "
                f"is non-empty. Refusing to overwrite — the dispatcher expects "
                f"a clean output_dir before conversion. Inspect / delete it "
                f"manually if this is a re-run."
            )
        # Empty dir from `output_dir.mkdir(parents=True, exist_ok=True)` upstream.
        output_dir.rmdir()

    # ── Phases C → D → D.5 → E wrapped in try/finally ─────────────────────
    # Round-9 reconciler v8 fix: if anything between staging-dir-creation
    # and the successful move raises, the ~64 GB /dev/shm dir leaks in
    # RAM-backed tmpfs until the next run / pod reboot. On an 8xH200 pod
    # under memory pressure (Qwen3-32B FWFT uses ~96% of 256 GB RAM during
    # conversion), the leak reduces RAM available to the next attempt.
    # The ``finally`` runs ``rmtree`` on the staging dir unless the move
    # has already consumed it (signalled by ``shm_staging_to_clean = None``).
    shm_staging_to_clean: Path | None = shm_staging
    try:
        # ── Phase C: run conversion subprocess (writes to /dev/shm) ───────
        convert_cmd = [
            sys.executable,
            str(converter_script),
            "--ds-checkpoint-dir",
            str(ds_native_dir),
            "--output-dir",
            str(shm_staging),
            "--model-id",
            base_model_id,
            "--tag",
            "final",
            "--max-shard-size",
            max_shard_size,
            "--dtype",
            dtype,
        ]
        log.info(
            "FWFT post-train: converting %s → %s (staging on /dev/shm)",
            ds_native_dir,
            shm_staging,
        )
        log.info("FWFT post-train: %s", " ".join(convert_cmd))
        # Fail-loud — non-zero rc preserves the DS-native shards on disk
        # and raises CalledProcessError up to the caller. The /dev/shm
        # staging dir is cleaned by the ``finally`` block below.
        subprocess.run(convert_cmd, check=True)

        # ── Phase D: delete DS-native from /workspace — fail-loud ─────────
        # Round-7 code-review demanded this drop ``ignore_errors=True``
        # and verify. /workspace holds DS-native + (empty output_dir
        # placeholder); deleting DS-native frees ~96 GB so the upcoming
        # workspace probe can ask for the HF artifact's footprint.
        log.info(
            "FWFT post-train: removing DS-native shards %s (fail-loud).",
            ds_native_dir,
        )
        shutil.rmtree(str(ds_native_dir))
        if ds_native_dir.exists():
            raise RuntimeError(
                f"FWFT post-train: shutil.rmtree({ds_native_dir}) returned "
                f"but the directory still exists. Refusing to move HF staging "
                f"into {output_dir} — would exceed MooseFS quota."
            )
        log.info("FWFT post-train: DS-native shards removed (verified gone).")

        # ── Phase D.5: /workspace probe — RUNS AFTER DS-native delete ─────
        # Round-9 reconciler v8 fix: probe what we're about to write, not
        # what we're about to free. By now /workspace has freed the ~96 GB
        # DS-native dir, so the probe asks for ``hf_bf16_bytes + slack``
        # (~74 GB) against ~130 GB available — fits cleanly within the
        # MooseFS per-pod quota. The probe is skipped under
        # ``skip_preflight=True`` for the CPU smoke path (symmetric with
        # the /dev/shm probe above).
        if not skip_preflight:
            _disk_headroom_probe(
                output_dir.parent,
                hf_bf16_bytes + _DISK_PROBE_SLACK_BYTES,
                label="/workspace",
            )

        # ── Phase E: move staging dir from /dev/shm → /workspace ──────────
        # shutil.move across filesystems = copy-then-delete. By now
        # /workspace is empty of DS-native, so the copy lands without
        # exceeding the quota. Post-move, the staging dir is consumed —
        # signal that to the ``finally`` block via
        # ``shm_staging_to_clean = None`` so it skips the rmtree.
        log.info(
            "FWFT post-train: moving HF staging %s → %s (cross-FS copy-and-delete).",
            shm_staging,
            output_dir,
        )
        shutil.move(str(shm_staging), str(output_dir))
        if shm_staging.exists():
            raise RuntimeError(
                f"FWFT post-train: shutil.move left staging dir {shm_staging} "
                "behind. /dev/shm will not auto-clean; refusing to proceed."
            )
        if not output_dir.exists():
            raise RuntimeError(
                f"FWFT post-train: shutil.move did not land at {output_dir}. "
                "HF artifact is lost — refusing to proceed."
            )
        # Move succeeded — staging dir is consumed.
        shm_staging_to_clean = None
        log.info("FWFT post-train: HF artifact at %s, ready for upload.", output_dir)
    finally:
        # Symmetric cleanup for the failure path: if anything between
        # staging-dir-creation and a successful Phase E raised, the
        # /dev/shm dir would otherwise leak ~64 GB of tmpfs RAM. We
        # rmtree it here. Failure-path cleanup is best-effort-loud: log
        # the failure but do NOT shadow whatever exception is already
        # propagating. (/dev/shm has no quota to enforce, so a stale
        # staging dir is recoverable on next run via the stale-state
        # cleanup at lines 532-543; we still try here to bound the
        # short-term RAM cost.)
        if shm_staging_to_clean is not None and shm_staging_to_clean.exists():
            log.warning(
                "FWFT post-train: failure path — cleaning up /dev/shm "
                "staging dir %s to avoid RAM leak.",
                shm_staging_to_clean,
            )
            try:
                shutil.rmtree(str(shm_staging_to_clean))
            except OSError as cleanup_err:
                log.error(
                    "FWFT post-train: failed to clean up staging dir %s on "
                    "failure path: %s. /dev/shm tmpfs will hold the leaked "
                    "bytes until next dispatcher run or pod reboot.",
                    shm_staging_to_clean,
                    cleanup_err,
                )


def _run_accelerate_train_with_sentinel(
    *,
    cmd: list[str],
    env: dict[str, str],
    phase_label: str,
) -> Path | None:
    """Run ``accelerate launch train_stage_sft.py ...`` and parse its ZeRO-3 sentinel.

    Round-15 v2 reconciler-FAIL fix: this replaces the previous
    ``subprocess.run(cmd, env=env, check=False)`` + post-hoc
    re-call-of-``pick_ds_native_staging_dir()`` pattern. The trainer's
    picker call can disagree with the dispatcher's picker call when
    /dev/shm has just been filled with ~185 GB of shards (writer saw 640 GB
    free → branch 2; reader sees < 200 GB free → branch 3) — 185 GB
    silently orphaned on tmpfs, FWFT artifact lost. The new contract: the
    trainer prints ``ZERO3_SAVE_DEFERRED ds_native_dir=<path> ...`` on
    rank-0 just before exit; this function captures stdout, tees it to the
    parent's stdout in real time (so launch-log visibility survives), and
    parses the sentinel from the captured copy.

    The captured stdout is parsed with ``parse_zero3_sentinel(...,
    required=False)`` so a clean LoRA / non-ZeRO-3 exit (no sentinel
    printed) returns ``None`` instead of raising. The ZeRO-3 path always
    prints a sentinel inside ``if is_zero3:`` right before process exit,
    so absent-sentinel after rc=0 + LoRA = legitimate; the caller checks
    the returned ``ds_native_dir`` Optional accordingly.

    Args:
        cmd: argv list to invoke ``accelerate launch ...`` with.
        env: Environment dict for the subprocess (same shape we currently
            pass: ``{**os.environ, ...}``).
        phase_label: Human-readable phase tag for log messages
            (``"FWFT Phase 1"`` etc.).

    Returns:
        The Path the trainer reported as its DS-native staging dir, OR
        ``None`` for the LoRA / non-ZeRO-3 path (no sentinel printed).

    Raises:
        RuntimeError: trainer exited non-zero. Message preserves the rc.
        ValueError | ZeRO3SentinelMissing: a sentinel was found but
            malformed (corruption — propagated from parse_zero3_sentinel)
            or expected-but-missing. We do not catch these here; the
            caller's ``check=False`` + subsequent finalize call become a
            hard failure with full context.
    """
    log.info("%s: launching %s", phase_label, " ".join(cmd))
    # Capture stdout as text + tee to parent's stdout line-by-line so the
    # launch log (nohup-captured pod-side) still shows progress in real
    # time. Use bufsize=1 (line-buffered) + universal_newlines/text so the
    # tee doesn't lag the parent's view.
    captured_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge so sentinel can't get lost on stderr
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None, "Popen returned no stdout pipe"
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_lines.append(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{phase_label} train_stage_sft exited rc={rc}")
    captured_stdout = "".join(captured_lines)

    # Parse the sentinel from captured stdout. required=False because the
    # LoRA / non-ZeRO-3 path legitimately prints no sentinel; in that case
    # the caller's "no DS-native handoff needed" branch fires.
    from explore_persona_space.orchestrate.staging import parse_zero3_sentinel

    ds_native_dir = parse_zero3_sentinel(captured_stdout, required=False)
    if ds_native_dir is None:
        log.info(
            "%s: no ZERO3_SAVE_DEFERRED sentinel in stdout — LoRA / non-ZeRO-3 "
            "path (trainer produced the HF artifact inline).",
            phase_label,
        )
    else:
        log.info(
            "%s: parsed ZERO3_SAVE_DEFERRED sentinel → ds_native_dir=%s",
            phase_label,
            ds_native_dir,
        )
    return ds_native_dir


def _finalize_fwft_zero3_save(
    *,
    ds_native_dir: Path,
    output_dir: Path,
    base_model_id: str,
    hub_repo_id: str,
    hub_subfolder: str,
    delete_after_upload_verified: bool,
) -> None:
    """Post-accelerate hand-off for ZeRO-3 FWFT saves (#506 rounds 7-8).

    train_stage_sft.py for ZeRO-3 + non-LoRA saves DS-native shards and
    exits without producing an HF artifact (round-6 fix: the in-process
    subprocess could not escape the 8-rank optimizer-state CPU footprint
    and OOMed at shard 3/14). This function runs AFTER ``accelerate launch``
    returns and all rank processes have torn down — optimizer state is
    fully released and the conversion gets the full pod RAM.

    Round-15 v2 reconciler-FAIL fix: ``ds_native_dir`` is now passed in
    directly (parsed from the trainer's ZERO3_SAVE_DEFERRED sentinel by
    ``_run_accelerate_train_with_sentinel`` above), NOT re-derived from
    ``pick_ds_native_staging_dir()`` here. The previous re-derive-the-path
    pattern raced against /dev/shm free space: pre-save free was > 200 GB
    so the writer picked branch 2, post-save free was < 200 GB (~185 GB of
    shards now occupy /dev/shm) so the reader picked branch 3, and 185 GB
    of shards silently orphaned at /dev/shm/... while this function looked
    at output_dir.parent and no-oped. The sentinel handshake makes both
    sides agree by construction.

    Steps:
      1. Convert DS-native → HF safetensors on /dev/shm staging via
         ``_run_conversion_with_shm_staging`` (round-8 fix: bounds MooseFS
         peak to max(DS-native, HF) ≈ 96 GB instead of 160 GB sum).
      2. The staging helper itself runs the posix_fallocate preflight,
         deletes DS-native fail-loud after conversion, and moves the
         staging dir to ``output_dir``.
      3. Upload the HF artifact to ``hub_repo_id/hub_subfolder`` via the
         shared ``upload_model`` helper, verify the upload landed,
         write ``hub_upload.json`` to the Hub, and (when requested)
         delete the local copy for quota.
      4. (Round-15 v2 Minor #3) Reap the empty
         ``/dev/shm/epm_ds_native_staging/`` parent staging root via
         ``reap_dev_shm_staging_root_if_empty()`` after the per-run
         subdir is gone — small but follows the round-9 try/finally
         cleanup discipline.

    Caller MUST only invoke this when ``ds_native_dir`` is not None
    (i.e. when the trainer's ZeRO-3 path ran and emitted the sentinel).
    LoRA / non-ZeRO-3 path returns ``None`` from
    ``_run_accelerate_train_with_sentinel``; caller skips this finalizer.

    Fail-loud on any step. The DS-native shards are preserved on
    conversion failure so the operator can post-mortem (and re-run
    just the conversion against the existing shards).
    """
    if not ds_native_dir.exists():
        # The trainer claimed it wrote here (sentinel said so) but the dir
        # is gone. This is a real failure: either rank exit interleaved
        # with a concurrent cleanup, or /dev/shm was reaped between
        # trainer exit and dispatcher pickup. Refuse to silently no-op
        # the way the old re-picker path did — that's exactly the bug we
        # are fixing.
        raise RuntimeError(
            f"FWFT post-train: trainer's ZERO3_SAVE_DEFERRED sentinel pointed at "
            f"{ds_native_dir}, but the directory does NOT exist post-train. "
            "DS-native shards were lost between trainer exit and dispatcher "
            "pickup — refusing to proceed (would produce an empty HF artifact)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    _run_conversion_with_shm_staging(
        ds_native_dir=ds_native_dir,
        output_dir=output_dir,
        base_model_id=base_model_id,
        max_shard_size="2GB",
        dtype="bfloat16",
        skip_preflight=bool(os.environ.get("EPM_SMOKE_PREFLIGHT_SKIP")),
    )

    # ── Upload + verify + delete-local. Mirrors the inline upload block
    # ── that train_stage_sft.py runs for non-ZeRO-3 saves, kept in one
    # ── place here so the ZeRO-3 path stays end-to-end-tested.
    from huggingface_hub import list_repo_files, upload_file

    from explore_persona_space.orchestrate.hub import upload_model

    log.info(
        "FWFT post-train: uploading %s to hf://%s/%s",
        output_dir,
        hub_repo_id,
        hub_subfolder,
    )
    hub_path = upload_model(
        model_path=str(output_dir),
        repo_id=hub_repo_id,
        path_in_repo=hub_subfolder,
    )
    if not hub_path:
        raise RuntimeError(
            f"FWFT post-train: upload returned empty path for {output_dir} -> "
            f"hf://{hub_repo_id}/{hub_subfolder}. Downstream phases cannot resolve."
        )
    files_in_subpath = [
        f
        for f in list_repo_files(hub_repo_id, token=os.environ.get("HF_TOKEN"))
        if f.startswith(hub_subfolder.rstrip("/") + "/")
    ]
    if not files_in_subpath:
        raise RuntimeError(
            f"FWFT post-train: upload verification FAILED — hf://{hub_repo_id}/"
            f"{hub_subfolder} lists 0 files via list_repo_files."
        )
    log.info(
        "FWFT post-train: upload verified, %d files at hf://%s/%s",
        len(files_in_subpath),
        hub_repo_id,
        hub_subfolder,
    )

    result_meta = {
        "hub_repo_id": hub_repo_id,
        "hub_path_in_repo": hub_subfolder,
        "hub_url": f"https://huggingface.co/{hub_repo_id}/tree/main/{hub_subfolder}",
        "n_files_verified": len(files_in_subpath),
    }
    meta_path = output_dir / "hub_upload.json"
    meta_path.write_text(json.dumps(result_meta, indent=2))

    if delete_after_upload_verified:
        try:
            upload_file(
                path_or_fileobj=str(meta_path),
                path_in_repo=f"{hub_subfolder.rstrip('/')}/hub_upload.json",
                repo_id=hub_repo_id,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            raise RuntimeError(
                f"FWFT post-train: failed to upload hub_upload.json before delete: {e}"
            ) from e
        shutil.rmtree(str(output_dir))
        log.info(
            "FWFT post-train: deleted local checkpoint %s after verified upload",
            output_dir,
        )

    # Round-15 v2 Minor #3: best-effort reap of the empty
    # /dev/shm/epm_ds_native_staging/ parent staging root. The per-run
    # subdir was removed by _run_conversion_with_shm_staging (Phase D —
    # fail-loud rmtree of DS-native), so the parent is empty if no
    # concurrent dispatcher invocation is using it. Safe by construction:
    # the helper only rmdirs when iterdir() yields nothing.
    from explore_persona_space.orchestrate.staging import (
        reap_dev_shm_staging_root_if_empty,
    )

    reap_dev_shm_staging_root_if_empty()


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
    # #506 round-7 fix: ZeRO-3 save is now split across processes.
    # train_stage_sft.py saves DS-native shards + exits; the dispatcher
    # owns conversion + upload via _finalize_fwft_zero3_save() AFTER
    # accelerate launch returns. So --upload / --hub-* / --delete-after-
    # upload-verified are NOT passed to the training script anymore.
    # --model is passed explicitly so model_id resolves to Qwen3-32B
    # even if a YAML key drifts (defense-in-depth on top of the
    # model_id-resolution fix in train_stage_sft.py).
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
        "--model",
        BASE_MODEL,
        "--dataset",
        str(data_path),
        "--output-dir",
        str(output_dir / "model"),
        "--seed",
        str(seed),
    ]
    env = {**os.environ}
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    t0 = time.time()
    # Round-15 v2: tee-capture stdout so we can parse the trainer's
    # ZERO3_SAVE_DEFERRED sentinel post-exit (path-consistency fix). LoRA
    # /  non-ZeRO-3 paths return ds_native_dir=None — we skip the
    # finalizer then. FWFT is always ZeRO-3 per the train_stage_sft.py
    # is_zero3 branch (line ~444); for it the sentinel is mandatory.
    ds_native_dir = _run_accelerate_train_with_sentinel(
        cmd=cmd,
        env=env,
        phase_label="FWFT Phase 1",
    )
    if ds_native_dir is None:
        raise RuntimeError(
            "FWFT Phase 1: trainer exited rc=0 but printed no ZERO3_SAVE_DEFERRED "
            "sentinel. FWFT path uses DeepSpeed ZeRO-3 unconditionally; missing "
            "sentinel means rank-0 stdout was lost mid-flight. Refusing to "
            "proceed (would silently no-op the finalizer and lose the save)."
        )

    # Post-accelerate: convert DS-native shards to HF safetensors (in a
    # single fresh process with the full pod RAM available), then upload.
    _finalize_fwft_zero3_save(
        ds_native_dir=ds_native_dir,
        output_dir=output_dir / "model",
        base_model_id=BASE_MODEL,
        hub_repo_id=HUB_FWFT_MODEL_REPO,
        hub_subfolder=fwft_sub_phase1,
        delete_after_upload_verified=True,
    )
    wall_m = (time.time() - t0) / 60
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
    # #506 round-7 fix: ZeRO-3 save split across processes (see Phase 1
    # comment above). --upload / --hub-* / --delete-after-upload-verified
    # are NOT passed to the training script; the dispatcher calls
    # _finalize_fwft_zero3_save() after accelerate launch returns.
    # --model BASE_MODEL ensures the conversion subprocess builds the
    # correct Qwen3-32B architecture for the saved state_dict (the round-6
    # silent-corruption fix).
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
        "--model",
        BASE_MODEL,
        "--input-model",
        str(phase1_local),
        "--dataset",
        str(data_path),
        "--output-dir",
        str(output_dir / "model"),
        "--seed",
        str(seed),
    ]
    env = {**os.environ}
    env["WANDB_PROJECT"] = WANDB_PROJECT

    t0 = time.time()
    # Round-15 v2: tee-capture stdout so we can parse the trainer's
    # ZERO3_SAVE_DEFERRED sentinel post-exit (path-consistency fix). See
    # Phase 1 above for the long-form rationale.
    ds_native_dir = _run_accelerate_train_with_sentinel(
        cmd=cmd,
        env=env,
        phase_label="FWFT Phase 2",
    )
    if ds_native_dir is None:
        raise RuntimeError(
            "FWFT Phase 2: trainer exited rc=0 but printed no ZERO3_SAVE_DEFERRED "
            "sentinel. FWFT path uses DeepSpeed ZeRO-3 unconditionally; missing "
            "sentinel means rank-0 stdout was lost mid-flight. Refusing to "
            "proceed (would silently no-op the finalizer and lose the save)."
        )

    # Post-accelerate: convert DS-native shards to HF safetensors (in a
    # single fresh process with the full pod RAM available), then upload.
    _finalize_fwft_zero3_save(
        ds_native_dir=ds_native_dir,
        output_dir=output_dir / "model",
        base_model_id=BASE_MODEL,
        hub_repo_id=HUB_FWFT_MODEL_REPO,
        hub_subfolder=fwft_sub_phase2,
        delete_after_upload_verified=True,
    )
    wall_m = (time.time() - t0) / 60
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
        description=("Issue #506 dispatcher — FWFT vs LoRA marker install on Qwen3-32B."),
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
