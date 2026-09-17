"""Protect substantive progress and process-stage evidence used by the watchdog."""

import os
import subprocess
import sys
import time

from scripts.story_persona_qwen38_monitor import (
    PROGRESS_STALL,
    apply_loading_progress,
    command,
    loading_counters,
    remote_probe,
    uv_cache_progress,
)


def test_fresh_log_does_not_reset_substantive_output_age(tmp_path):
    """A chatty worker must not conceal a stalled artifact-writing stage."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "chunks").mkdir()
    chunk = out / "chunks/batch_0000.pt"
    chunk.write_bytes(b"test-only")
    old = time.time() - 2000
    for path in (chunk, out / "chunks", out):
        os.utime(path, (old, old))
    log = tmp_path / "run.log"
    log.write_text("heartbeat\n")
    value = remote_probe(out, log)
    assert value["log_age_seconds"] < 10
    assert value["output_age_seconds"] > 1900
    assert value["chunk_count"] == 1


def test_publication_worker_stage_is_observed(tmp_path):
    """The actual --phase publish token drives network-progress interpretation."""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
            "scripts/story_persona_qwen38_artifacts.py",
            "--phase",
            "publish",
        ]
    )
    try:
        value = remote_probe(tmp_path / "missing", tmp_path / "missing.log")
        assert child.pid in value["pids"]
        assert "publish" in value["stages"]
        assert value["network_tx_bytes"] >= 0
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_checked_probe_preserves_nonzero_failures():
    """A failed read-only subprocess must not become a successful observation."""
    import pytest

    with pytest.raises(RuntimeError, match="rc=7"):
        command([sys.executable, "-c", "raise SystemExit(7)"])


def test_hydra_publication_worker_stage_is_observed(tmp_path):
    """Cross-model Hydra syntax must activate the real network-progress monitor."""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
            "scripts/story_persona_crossmodel_artifacts.py",
            "phase=publish",
        ]
    )
    try:
        value = remote_probe(tmp_path / "missing", tmp_path / "missing.log")
        assert child.pid in value["pids"]
        assert "publish" in value["stages"]
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_loader_counters_require_real_model_load_lines():
    """Parse actual Transformers parameter-loading counters, including CR updates."""
    text = (
        "Loading checkpoint shards: 12%|abc| 20/163 [00:10]\r"
        "Loading weights: 20%|abc| 2,000/10,000 [00:12]\r"
        "\x1b[32mLoading weights: 30%|abc| 3,000/10,000 [00:15]\x1b[0m\r"
        "Writing model shards: 100%|abc| 10/10 [00:20]\nheartbeat\n"
    )
    assert loading_counters(text) == {
        "Loading checkpoint shards": {"completed": 20, "total": 163},
        "Loading weights": {"completed": 3000, "total": 10000},
    }


def test_uv_scan_measures_allocated_growth_and_marks_partial_walks(tmp_path):
    """Actual new file bytes count; cached path mtimes alone are insufficient."""
    wheel = tmp_path / "wheel.tmp"
    wheel.write_bytes(b"x" * 1024)
    first = uv_cache_progress(tmp_path)
    os.utime(wheel, None)
    unchanged = uv_cache_progress(tmp_path)
    assert first == unchanged
    wheel.write_bytes(b"x" * 32768)
    grown = uv_cache_progress(tmp_path)
    assert grown["complete"] is True and grown["allocated_bytes"] > first["allocated_bytes"]
    assert uv_cache_progress(tmp_path, max_entries=0)["complete"] is False


def loading_observation(*, size=1024, completed=1, complete=True):
    """Represent a fresh probe whose ordinary capture outputs have gone quiet."""
    return {
        "status": "running",
        "stall_reason": PROGRESS_STALL,
        "pilot": {
            "pids": [100],
            "uv_cache": {"allocated_bytes": size, "files": 1, "complete": complete},
            "loading_counters": {"Loading weights": {"completed": completed, "total": 100}},
        },
    }


def test_cache_growth_clears_only_bounded_generic_stall():
    """A true positive delta buys 900 seconds; unchanged observations do not reset it."""
    previous = loading_observation()
    current = loading_observation(size=8192)
    apply_loading_progress(current, previous, now=1000)
    assert current["status"] == "pending" and "stall_reason" not in current
    assert current["current_phase"] == "runtime_bootstrap"
    assert current["pilot"]["last_uv_growth_at"] == 1000
    expired = loading_observation(size=8192)
    apply_loading_progress(expired, current, now=1901)
    assert expired["stall_reason"] == PROGRESS_STALL
    assert expired["pilot"]["last_uv_growth_at"] == 1000


def test_advancing_weight_loading_can_resolve_log_only_backend_stall():
    """The backend's generic stale-log result must not hide measured loader progress."""
    previous = loading_observation(completed=10)
    current = loading_observation(completed=11)
    current.update(status="stalled", log_only_backend_stall=True)
    apply_loading_progress(current, previous, now=1000)
    assert current["status"] == "pending" and "stall_reason" not in current
    assert current["current_phase"] == "model_loading"


def test_heartbeats_incomplete_scans_and_first_snapshots_do_not_clear_stall():
    """No positive growth comparison means no progress credit."""
    for current, previous in (
        (loading_observation(), loading_observation()),
        (loading_observation(size=8192, complete=False), loading_observation()),
        (loading_observation(size=8192), {}),
    ):
        apply_loading_progress(current, previous, now=1000)
        assert current["stall_reason"] == PROGRESS_STALL


def test_loading_does_not_hide_structural_backend_failure_or_unreachability():
    """Cache activity must not override a failed GPU worker or failed backend probe."""
    previous = loading_observation()
    for extra in (
        {"stall_reason": "vllm_worker_dead_zombie_gpu"},
        {"reachability_alarm": True},
        {"status": "dead"},
        {"status": "stalled", "log_only_backend_stall": False},
    ):
        current = loading_observation(size=8192, completed=2)
        current.update(extra)
        apply_loading_progress(current, previous, now=1000)
        assert current.get("stall_reason")
