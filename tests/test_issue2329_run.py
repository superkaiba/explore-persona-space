"""Issue #2329 fork — claim-file queue in-flight-tolerance pins (#2305).

The #2329 driver forks the #2162 claim-file queue verbatim (fork documented
in the issue2329_ladder.py header); these tests mirror the four #2305
claim-queue pins from tests/test_issue2162_run.py against the fork module.
The full queue suite stays in tests/test_issue2162_run.py; the fork
deliberately stays a fork (no de-dup onto an import — #2305 must-ask
boundary).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2329_run as R  # noqa: E402


def _mkblock(cell: str = "instr_format") -> R.Block:
    return R.Block(cell, "ce", "steered", ("p1", "p2"))


def test_try_claim_empty_claim_reclaimed_not_fatal(tmp_path, monkeypatch):
    """#2305 acceptance 1: a STATIC empty claim file (writer died inside the
    create window) is reclaimed and run exactly once — no raise, no skip."""
    monkeypatch.setattr(R, "CLAIM_READ_SLEEP_RANGE", (0.0, 0.01))
    cdir = tmp_path / "claims"
    cdir.mkdir(parents=True)
    block = _mkblock()
    (cdir / f"{block.slug}.claim").write_bytes(b"")
    assert R.try_claim(cdir, block, 0, "tok-reclaim") is True
    rec = json.loads((cdir / f"{block.slug}.claim").read_text())
    assert rec["token"] == "tok-reclaim"
    assert rec["key"] == block.key
    # Not double-claimable afterwards -> run-once when driven by the queue.
    assert R.try_claim(cdir, block, 1, "tok-other") is False
    # Writer-A shape: no tmp residue left behind.
    assert not list(cdir.glob("*.tmp.*"))


def test_try_claim_empty_claim_honors_live_writer(tmp_path, monkeypatch):
    """#2305: a LIVE writer that lands its payload inside the retry bound is
    honored — try_claim parses the record on a retry and returns False;
    never a raise, never a steal."""
    # Pin the retry bound GUARANTEED far above the background writer's delay
    # (fixed 0.2 s sleeps x 3 retries = 0.6 s bound vs a ~0.05 s write; the
    # bound is widened for determinism, never tightened to speed the test).
    monkeypatch.setattr(R, "CLAIM_READ_RETRIES", 4)
    monkeypatch.setattr(R, "CLAIM_READ_SLEEP_RANGE", (0.2, 0.2))
    cdir = tmp_path / "claims"
    cdir.mkdir(parents=True)
    block = _mkblock()
    path = cdir / f"{block.slug}.claim"
    path.write_bytes(b"")
    live = {
        "key": block.key,
        "pid": os.getpid(),  # our own live pid = live same-host claim
        "host": socket.gethostname(),
        "worker_index": 1,
        "ts": time.time(),
        "token": "tok-live-writer",
    }

    def _finish_write():
        time.sleep(0.05)
        tmp = path.parent / f"{path.name}.tmp.tok-live-writer"
        tmp.write_text(json.dumps(live))
        os.replace(tmp, path)

    t = threading.Thread(target=_finish_write)
    t.start()
    try:
        assert R.try_claim(cdir, block, 0, "tok-late") is False
    finally:
        t.join()
    rec = json.loads(path.read_text())
    assert rec["token"] == "tok-live-writer"


def test_try_claim_nonempty_garbage_still_raises(tmp_path, monkeypatch):
    """#2305 acceptance 3: persistent NON-EMPTY garbage is genuine corruption
    — the hard error survives the bounded in-flight tolerance, message
    unchanged."""
    monkeypatch.setattr(R, "CLAIM_READ_SLEEP_RANGE", (0.0, 0.01))
    cdir = tmp_path / "claims"
    cdir.mkdir(parents=True)
    block = _mkblock()
    (cdir / f"{block.slug}.claim").write_bytes(b"{not json")
    with pytest.raises(RuntimeError, match="unparseable claim"):
        R.try_claim(cdir, block, 0, "tok")


def test_try_claim_fresh_write_is_atomic_shape(tmp_path):
    """#2305 writer-A shape pin: a successful fresh claim lands its payload
    via tmp + os.replace — record parses, token ours, no .tmp. residue."""
    cdir = tmp_path / "claims"
    block = _mkblock()
    assert R.try_claim(cdir, block, 0, "tok-fresh") is True
    rec = json.loads((cdir / f"{block.slug}.claim").read_text())
    assert rec["token"] == "tok-fresh"
    assert rec["key"] == block.key
    assert not list(cdir.glob("*.tmp.*"))
