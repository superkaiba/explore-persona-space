"""Staging-directory pickers for large intermediate ZeRO-3 / FWFT artifacts.

The DS-native ZeRO-3 per-rank shards on 7-rank Qwen3-32B FWFT peak at ≥185 GB
of writes. /workspace on RunPod is a MooseFS volume with a hard ~130-200 GB
per-pod quota (see CLAUDE.md gotchas: MooseFS per-pod quota), so writing the
save there blew up twice in #506 rounds 13 + 14 with ENOSPC. /dev/shm on an
8xH200 pod is tmpfs RAM-backed with ~700 GB capacity and ~640 GB free even
with the HF cache parked there, so it comfortably absorbs the save.

This module owns the staging-dir picker shared between
``scripts/train_stage_sft.py`` (the writer, inside ``accelerate launch``) and
``scripts/run_issue506_install.py`` (the reader / converter, post-train).
Both callsites MUST agree on the path, or the dispatcher's conversion + cleanup
won't find the shards.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Threshold: the DS-native save on 7-rank Qwen3-32B FWFT peaks at ≥185 GB.
# Demand 200 GB free on /dev/shm so the save fits with ~15 GB of slack for
# tmpfs metadata + concurrent allocations (e.g. the HF-conversion staging
# the dispatcher does later, which is itself ~64 GB on /dev/shm).
_DEV_SHM_HEADROOM_BYTES = 200 * 1024 * 1024 * 1024  # 200 GB

# Stable parent dir under /dev/shm so concurrent runs on the same pod don't
# collide with one another. Each (output_dir.name) gets its own subdir.
_DEV_SHM_STAGING_ROOT = Path("/dev/shm/epm_ds_native_staging")

# Env var override — operators can force a specific staging root (e.g. a
# pod-specific overlayfs mount, a CI tmpdir). Set to e.g. "/mnt/scratch" and
# the picker returns "/mnt/scratch/<output_dir.name>_ds_native".
_ENV_VAR = "EPM_DS_NATIVE_STAGING_DIR"


def pick_ds_native_staging_dir(
    output_dir: Path,
    *,
    free_threshold_bytes: int = _DEV_SHM_HEADROOM_BYTES,
) -> Path:
    """Pick where to write the DS-native ZeRO-3 per-rank shards.

    Priority (high → low):
      1. Env var ``EPM_DS_NATIVE_STAGING_DIR`` — returns
         ``Path(EPM_DS_NATIVE_STAGING_DIR) / f"{output_dir.name}_ds_native"``.
         Operators use this to pin staging to a specific mount (CI tmpdir,
         scratch volume, etc.).
      2. ``/dev/shm`` exists AND has ``free_threshold_bytes`` free
         (default 200 GB; covers the ≥185 GB save plus tmpfs slack).
         Returns ``/dev/shm/epm_ds_native_staging/{output_dir.name}_ds_native``.
         This is the pod-side default on 8xH{100,200} machines.
      3. Fallback — the original behavior:
         ``output_dir.parent / f"{output_dir.name}_ds_native"``. Used on
         dev VMs, small pods, or any environment where /dev/shm isn't big
         enough.

    The branch chosen + reason are logged at INFO level so the launch log
    shows which path fired.

    The returned path's BASENAME is always ``f"{output_dir.name}_ds_native"``
    — the dispatcher's conversion step reads the absolute path from the
    ``ZERO3_SAVE_DEFERRED`` sentinel that train_stage_sft.py emits, so the
    basename doesn't actually need to match a convention; we keep it for
    readability + log greppability.

    Args:
        output_dir: The final HF-format output dir (e.g.
            ``/workspace/outputs/issue506_fwft_phase1_seed42``). Used to
            derive the staging dir's basename so concurrent runs don't
            collide.
        free_threshold_bytes: How much free space /dev/shm must have for
            branch 2 to fire. Default 200 GB.

    Returns:
        Absolute Path to the staging dir. Caller is responsible for
        ``mkdir(parents=True, exist_ok=True)``.
    """
    target_basename = f"{output_dir.name}_ds_native"

    # ── Branch 1: explicit env-var override ──────────────────────────────
    env_override = os.environ.get(_ENV_VAR)
    if env_override:
        chosen = Path(env_override) / target_basename
        log.info(
            "DS-native staging: using env override %s=%s → %s",
            _ENV_VAR,
            env_override,
            chosen,
        )
        return chosen

    # ── Branch 2: /dev/shm tmpfs (the pod-side default) ──────────────────
    dev_shm = Path("/dev/shm")
    if dev_shm.exists():
        try:
            free = shutil.disk_usage(dev_shm).free
        except OSError as e:
            log.warning(
                "DS-native staging: shutil.disk_usage(/dev/shm) failed (%s); "
                "skipping /dev/shm branch, falling back to %s.",
                e,
                output_dir.parent,
            )
            free = 0
        if free >= free_threshold_bytes:
            chosen = _DEV_SHM_STAGING_ROOT / target_basename
            log.info(
                "DS-native staging: /dev/shm has %.1f GB free (≥ %.1f GB threshold) → %s",
                free / 1024**3,
                free_threshold_bytes / 1024**3,
                chosen,
            )
            return chosen
        else:
            log.info(
                "DS-native staging: /dev/shm has only %.1f GB free (< %.1f GB threshold); "
                "falling back to %s.",
                free / 1024**3,
                free_threshold_bytes / 1024**3,
                output_dir.parent,
            )
    else:
        log.info(
            "DS-native staging: /dev/shm does not exist on this host; falling back to %s.",
            output_dir.parent,
        )

    # ── Branch 3: legacy fallback (output_dir.parent) ────────────────────
    chosen = output_dir.parent / target_basename
    log.info("DS-native staging: using fallback %s", chosen)
    return chosen
