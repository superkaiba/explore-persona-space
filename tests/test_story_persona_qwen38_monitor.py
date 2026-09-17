"""Protect substantive progress and process-stage evidence used by the watchdog."""

import os
import subprocess
import sys
import time

from scripts.story_persona_qwen38_monitor import command, remote_probe


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
