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
    # Round-15 v2 Minor #1: when we fall back to output_dir.parent, the FWFT
    # save (~185 GB on 7-rank Qwen3-32B) lands on /workspace's MooseFS
    # volume — the ~130-200 GB per-pod quota will trip ENOSPC mid-save
    # (the exact round-13/14 failure mode). Probe free space at the
    # fallback target and log a LOUD warning if it's below 250 GB so the
    # operator sees the risk in the launch log before the save crashes.
    try:
        fallback_free = shutil.disk_usage(output_dir.parent).free
    except OSError as e:
        log.warning(
            "DS-native staging: shutil.disk_usage(%s) failed (%s) — cannot "
            "probe fallback free space. The FWFT save is likely to OOM the "
            "volume; investigate before launching.",
            output_dir.parent,
            e,
        )
    else:
        if fallback_free < 250 * 1024**3:
            log.warning(
                "DS-native staging: FALLBACK BRANCH ACTIVE on %s with only "
                "%.1f GB free (< 250 GB safety threshold). The FWFT ZeRO-3 "
                "save on 7-rank Qwen3-32B peaks at ~185 GB; this run will "
                "likely trip ENOSPC / EDQUOT mid-save (round-13/14 failure "
                "mode). Either free disk on %s or export "
                "EPM_DS_NATIVE_STAGING_DIR pointing at a larger volume.",
                output_dir.parent,
                fallback_free / 1024**3,
                output_dir.parent,
            )
    return chosen


# ── Sentinel parser (writer→reader handshake) ────────────────────────────────
#
# Round-15 v2 reconciler-FAIL fix: previously the dispatcher's reader side
# (``_finalize_fwft_zero3_save`` in run_issue506_install.py) RE-CALLED
# ``pick_ds_native_staging_dir(output_dir)`` post-training. That second call
# can disagree with the writer's first call: if pre-save /dev/shm free was
# between 200 GB and 385 GB, the writer's call returned branch 2
# (/dev/shm/...) but the reader's call — running AFTER ~185 GB of shards
# were written to /dev/shm — sees < 200 GB free and silently returns
# branch 3 (output_dir.parent/...), then no-ops with "no DS-native dir at
# <wrong path>". 185 GB of shards orphaned on tmpfs; FWFT artifact lost.
#
# The fix: the writer prints an authoritative ``ZERO3_SAVE_DEFERRED ...``
# line containing the absolute ``ds_native_dir`` it actually wrote to. The
# reader parses that line from the captured accelerate-launch stdout and
# uses the parsed path — guaranteeing both sides agree by construction, not
# by hope.

# Format (from scripts/train_stage_sft.py rank-0 print, just before exit):
#   ZERO3_SAVE_DEFERRED ds_native_dir=<abs_path> output_dir=<abs_path> \
#       model_id=<id> load_path=<abs_path>
# All four fields are space-separated key=value pairs; values are absolute
# paths (no spaces in pod-side paths by our convention) or simple HF model
# ids. The line is printed exactly once per ZeRO-3 run.
_ZERO3_SENTINEL_PREFIX = "ZERO3_SAVE_DEFERRED "


class ZeRO3SentinelMissing(RuntimeError):
    """Raised when the reader expected a ZERO3_SAVE_DEFERRED line but found none.

    This is a hard failure: the writer (train_stage_sft.py) prints the
    sentinel unconditionally inside its ``if is_zero3:`` branch right before
    process exit. If the trainer ran ZeRO-3 + non-LoRA AND exited cleanly
    (rc=0) AND no sentinel was seen, something corrupted the rank-0 stdout
    in flight. Falling back to a re-call of the picker risks the silent
    path mismatch the sentinel handshake exists to prevent.

    The dispatcher catches this above and treats it as a real failure;
    the LoRA / non-ZeRO-3 path uses ``parse_zero3_sentinel`` with
    ``required=False`` so an absent sentinel is a no-op (LoRA never writes
    one).
    """


def parse_zero3_sentinel(
    captured_stdout: str,
    *,
    required: bool = True,
) -> Path | None:
    """Parse a ``ZERO3_SAVE_DEFERRED ds_native_dir=<path> ...`` line.

    Scans ``captured_stdout`` for a line whose first token (after
    whitespace) matches the sentinel prefix, then extracts the
    ``ds_native_dir=<value>`` field.

    Args:
        captured_stdout: The full stdout of the ``accelerate launch
            train_stage_sft.py ...`` subprocess (multi-line string).
        required: If True (default), raise :class:`ZeRO3SentinelMissing`
            when no sentinel line is found. If False, return ``None``
            silently (used by the dispatcher for the unified ZeRO-3 +
            LoRA hand-off point, where LoRA legitimately prints no
            sentinel).

    Returns:
        Absolute Path to the staging dir the writer reported, OR ``None``
        when ``required=False`` AND no sentinel was found.

    Raises:
        ZeRO3SentinelMissing: ``required=True`` AND no sentinel line in
            ``captured_stdout``.
        ValueError: A sentinel line was found but its ``ds_native_dir=``
            field could not be parsed (corrupted print or trainer-side
            format drift). This is also a hard failure — the dispatcher
            must NOT proceed with an unknown path.
    """
    # Lazy-import re inside the function so ruff doesn't strip the
    # top-level import as unused (see project memory:
    # ruff_strips_unused_imports). Cheap on the hot path — Python caches
    # compiled patterns in re._cache.
    import re

    pattern = re.compile(r"\bds_native_dir=(\S+)")
    found_path: str | None = None
    for raw_line in captured_stdout.splitlines():
        # The trainer prints unprefixed, but accelerate may prepend rank /
        # timestamp markers. lstrip handles both shapes.
        line = raw_line.lstrip()
        if not line.startswith(_ZERO3_SENTINEL_PREFIX):
            continue
        match = pattern.search(line)
        if not match:
            # Sentinel prefix present but the path field is missing —
            # don't silently fall through to "no sentinel". This is
            # corrupted output.
            raise ValueError(
                f"Found {_ZERO3_SENTINEL_PREFIX!r} sentinel line but could "
                f"not parse 'ds_native_dir=<path>' field. Line was: {line!r}"
            )
        found_path = match.group(1)
        # Don't break — if the trainer printed multiple (shouldn't happen,
        # but defensive), the LAST one is the most recent / authoritative.

    if found_path is not None:
        return Path(found_path)
    if required:
        raise ZeRO3SentinelMissing(
            f"No {_ZERO3_SENTINEL_PREFIX!r} sentinel line in trainer stdout. "
            "Either the ZeRO-3 save path did not run (LoRA / non-ZeRO-3 "
            "regime — caller should pass required=False) or rank-0 stdout "
            "was lost mid-flight. Refusing to fall back to a re-picker call "
            "(path consistency hazard — see issue #506 round-15 reconciler)."
        )
    return None


def reap_dev_shm_staging_root_if_empty() -> None:
    """Remove ``/dev/shm/epm_ds_native_staging/`` when empty.

    Round-15 v2 Minor #3: the per-(output_dir.name) subdir under
    ``_DEV_SHM_STAGING_ROOT`` is removed by the dispatcher after a
    successful run (``shutil.rmtree(ds_native_dir)``), but the parent
    stays — an empty directory on tmpfs, harmless but untidy. This helper
    reaps the parent if and only if it is empty. Best-effort: any OSError
    is logged at debug and swallowed (the parent will be picked up by the
    pod-reboot or by the next ``run_issue506_install.py`` invocation
    creating fresh subdirs anyway). NEVER removes a non-empty dir.

    This is intentionally scoped to the stable ``_DEV_SHM_STAGING_ROOT``
    parent; it does NOT remove the fallback's ``output_dir.parent`` (which
    is a real workspace directory the dispatcher does not own).
    """
    if not _DEV_SHM_STAGING_ROOT.exists():
        return
    try:
        # Path.iterdir is lazy; any() short-circuits on first entry. If
        # any subdir survived our cleanup, we leave the parent alone.
        for _entry in _DEV_SHM_STAGING_ROOT.iterdir():
            log.debug(
                "DS-native staging: %s still contains %s; leaving parent alone.",
                _DEV_SHM_STAGING_ROOT,
                _entry,
            )
            return
        # No entries → safe to rmdir.
        os.rmdir(_DEV_SHM_STAGING_ROOT)
        log.info(
            "DS-native staging: reaped empty parent staging root %s.",
            _DEV_SHM_STAGING_ROOT,
        )
    except OSError as e:
        log.debug(
            "DS-native staging: best-effort reap of %s failed (%s); "
            "leaving for next run / pod reboot.",
            _DEV_SHM_STAGING_ROOT,
            e,
        )
