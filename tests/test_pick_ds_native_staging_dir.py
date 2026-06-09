"""CPU-only unit tests for the issue-#506 round-15 v2 DS-native staging picker.

The picker is the writer-side decision: where ``train_stage_sft.py`` will
write ~185 GB of ZeRO-3 per-rank shards. The reader (the dispatcher's
``_finalize_fwft_zero3_save``) now parses the writer's printed path from a
sentinel line — but the picker still has to make the actual choice on the
writer side. These tests pin the three branches so we don't regress the
decision logic:

1. Env-var override (``EPM_DS_NATIVE_STAGING_DIR``) wins unconditionally.
2. ``/dev/shm`` is chosen when free ≥ threshold and the dir exists.
3. Fallback to ``output_dir.parent`` when /dev/shm has insufficient free
   space (and a loud warning fires when the fallback target is itself
   below the 250 GB safety threshold — round-15 v2 Minor #1).

We also assert the path basenames + that the sentinel parser
(``parse_zero3_sentinel``) correctly:

4. Parses ``ds_native_dir=<path>`` out of a realistic ``ZERO3_SAVE_DEFERRED``
   line.
5. Raises ``ZeRO3SentinelMissing`` when ``required=True`` and the sentinel
   is absent (reader-side hard-fail under the new contract).
6. Returns ``None`` when ``required=False`` and the sentinel is absent
   (LoRA / non-ZeRO-3 path no-op).

All assertions are pure pathlib + ``unittest.mock`` patches against
``shutil.disk_usage`` + ``os.environ`` — no I/O, no actual /dev/shm probe,
runs on the dev VM without any RunPod-specific state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.orchestrate.staging import (
    _DEV_SHM_STAGING_ROOT,
    ZeRO3SentinelMissing,
    parse_zero3_sentinel,
    pick_ds_native_staging_dir,
)


def _du(total: int, used: int, free: int) -> object:
    """Build a fake ``shutil.disk_usage`` named-tuple-shaped result."""
    return mock.MagicMock(total=total, used=used, free=free)


# ── Branch 1: env-var override wins ──────────────────────────────────────────


def test_branch_1_env_override_wins(monkeypatch, tmp_path):
    """EPM_DS_NATIVE_STAGING_DIR overrides everything else, including a
    /dev/shm with enough free space."""
    custom_root = tmp_path / "scratch"
    monkeypatch.setenv("EPM_DS_NATIVE_STAGING_DIR", str(custom_root))
    # Even if /dev/shm has lots of free space, the env override wins:
    with mock.patch(
        "explore_persona_space.orchestrate.staging.shutil.disk_usage",
        return_value=_du(total=700 * 1024**3, used=10 * 1024**3, free=640 * 1024**3),
    ):
        result = pick_ds_native_staging_dir(Path("/workspace/outputs/issue506_phase1"))
    assert result == custom_root / "issue506_phase1_ds_native"
    # And the path the caller will mkdir+chdir to keeps the canonical basename
    # so log greppability works:
    assert result.name == "issue506_phase1_ds_native"


# ── Branch 2: /dev/shm with sufficient free space ────────────────────────────


def test_branch_2_dev_shm_sufficient_free(monkeypatch):
    """When /dev/shm has ≥200 GB free, return /dev/shm/epm_ds_native_staging/...
    regardless of output_dir's location."""
    monkeypatch.delenv("EPM_DS_NATIVE_STAGING_DIR", raising=False)
    # Patch /dev/shm to look like it exists with plenty of free RAM.
    with (
        mock.patch(
            "explore_persona_space.orchestrate.staging.Path.exists",
            return_value=True,
        ),
        mock.patch(
            "explore_persona_space.orchestrate.staging.shutil.disk_usage",
            return_value=_du(total=700 * 1024**3, used=60 * 1024**3, free=640 * 1024**3),
        ),
    ):
        result = pick_ds_native_staging_dir(Path("/workspace/outputs/issue506_phase1"))
    assert result == _DEV_SHM_STAGING_ROOT / "issue506_phase1_ds_native"
    assert str(result).startswith("/dev/shm/")


# ── Branch 3: /dev/shm too full → output_dir.parent fallback ─────────────────


def test_branch_3_fallback_when_dev_shm_below_threshold(monkeypatch, caplog):
    """When /dev/shm reports <200 GB free, return output_dir.parent/<name>_ds_native.
    Also assert that the fallback's own free-space probe fires a LOUD warning
    when the fallback target is below 250 GB (Round-15 v2 Minor #1)."""
    monkeypatch.delenv("EPM_DS_NATIVE_STAGING_DIR", raising=False)
    # /dev/shm exists but is mostly full (only 100 GB free, below the 200 GB
    # threshold); output_dir.parent has 130 GB free (below the 250 GB fallback
    # safety threshold) — emulates the actual MooseFS-quota'd pod failure mode.
    call_count = {"n": 0}

    def fake_disk_usage(path):
        call_count["n"] += 1
        # First call = /dev/shm probe (insufficient).
        # Second call = output_dir.parent probe (also low → triggers warning).
        if call_count["n"] == 1:
            return _du(total=700 * 1024**3, used=600 * 1024**3, free=100 * 1024**3)
        else:
            return _du(total=200 * 1024**3, used=70 * 1024**3, free=130 * 1024**3)

    with (
        mock.patch(
            "explore_persona_space.orchestrate.staging.Path.exists",
            return_value=True,
        ),
        mock.patch(
            "explore_persona_space.orchestrate.staging.shutil.disk_usage",
            side_effect=fake_disk_usage,
        ),
        caplog.at_level(logging.WARNING, logger="explore_persona_space.orchestrate.staging"),
    ):
        result = pick_ds_native_staging_dir(Path("/workspace/outputs/issue506_phase1"))
    assert result == Path("/workspace/outputs/issue506_phase1_ds_native")
    assert result.parent == Path("/workspace/outputs")
    # Minor #1: loud warning fired for the low-free fallback target.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("FALLBACK BRANCH ACTIVE" in r.getMessage() for r in warnings), (
        f"Expected 'FALLBACK BRANCH ACTIVE' warning in caplog records, got: "
        f"{[r.getMessage() for r in warnings]}"
    )


# ── Branch 3 (no /dev/shm at all): same fallback path, no warning ────────────


def test_branch_3_dev_shm_missing(monkeypatch):
    """When /dev/shm does not exist at all, fall back to output_dir.parent
    without probing it. (Dev-VM case — no tmpfs at /dev/shm.)"""
    monkeypatch.delenv("EPM_DS_NATIVE_STAGING_DIR", raising=False)

    # Simulate /dev/shm not existing. We can't patch Path.exists globally
    # without breaking output_dir.parent.exists, so use a narrower selector
    # on the path arg.
    original_exists = Path.exists

    def fake_exists(self):
        if str(self) == "/dev/shm":
            return False
        return original_exists(self)

    # output_dir.parent gets a large free-space response so no warning fires.
    with (
        mock.patch.object(Path, "exists", fake_exists),
        mock.patch(
            "explore_persona_space.orchestrate.staging.shutil.disk_usage",
            return_value=_du(total=2000 * 1024**3, used=100 * 1024**3, free=1900 * 1024**3),
        ),
    ):
        result = pick_ds_native_staging_dir(Path("/tmp/some_output_dir"))
    assert result == Path("/tmp/some_output_dir_ds_native")


# ── Sentinel parser tests ────────────────────────────────────────────────────


def test_parse_zero3_sentinel_happy_path():
    """A realistic captured-stdout chunk containing a ZERO3_SAVE_DEFERRED line
    must parse to the absolute ds_native_dir path."""
    captured = (
        "[2026-06-09 10:23:45] some accelerate noise\n"
        "loading model weights...\n"
        "ZERO3_SAVE_DEFERRED ds_native_dir=/dev/shm/epm_ds_native_staging/"
        "issue506_fwft_phase1_seed42_ds_native output_dir=/workspace/outputs/"
        "issue506_fwft_phase1_seed42 model_id=Qwen/Qwen3-32B load_path=None\n"
        "[some other line]\n"
    )
    result = parse_zero3_sentinel(captured)
    assert result == Path("/dev/shm/epm_ds_native_staging/issue506_fwft_phase1_seed42_ds_native")


def test_parse_zero3_sentinel_missing_required_raises():
    """When required=True (default) and no sentinel line is present, raise."""
    captured = "just some unrelated output\nfrom the trainer that finished cleanly\n"
    with pytest.raises(ZeRO3SentinelMissing):
        parse_zero3_sentinel(captured, required=True)


def test_parse_zero3_sentinel_missing_not_required_returns_none():
    """When required=False, an absent sentinel returns None (LoRA path)."""
    captured = "just some unrelated output\nfrom the trainer that finished cleanly\n"
    result = parse_zero3_sentinel(captured, required=False)
    assert result is None


def test_parse_zero3_sentinel_malformed_raises():
    """A sentinel prefix without a parseable ds_native_dir= field is a
    hard ValueError — the dispatcher must NOT proceed with an unknown path."""
    captured = "ZERO3_SAVE_DEFERRED no_field_here_at_all\n"
    with pytest.raises(ValueError, match="ds_native_dir=<path>"):
        parse_zero3_sentinel(captured)


def test_parse_zero3_sentinel_picks_last_occurrence():
    """If the sentinel is printed twice (defensive), the last occurrence wins."""
    captured = (
        "ZERO3_SAVE_DEFERRED ds_native_dir=/first/path output_dir=/x model_id=Y load_path=Z\n"
        "ZERO3_SAVE_DEFERRED ds_native_dir=/second/path output_dir=/x model_id=Y load_path=Z\n"
    )
    result = parse_zero3_sentinel(captured)
    assert result == Path("/second/path")
