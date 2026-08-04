"""#1947 crash-fix r10 pins: dispatcher slot-quarantine + retry refund.

P2-fleet-a (epm:failure v6): pod-1947-a's sick physical GPU 6 failed
torch.cuda.init instantly ("CUDA-capable device(s) is/are busy or
unavailable") on an IDLE device, so its dispatch slot became an
instantly-freeing blackhole — every requeued cell relanded on it and burned
its single retry (>=2 cells failed permanently). The r10 fix in
``cmd_dispatch``: per-slot consecutive fast-failure tracking benches a slot
after SLOT_QUARANTINE_STREAK (2) consecutive fast (< SLOT_FAST_FAIL_SECONDS)
rc!=0 exits, REFUNDS the retry budget of every cell whose failure fed the
streak (the failure was the slot's fault, not the cell's), and fail-louds
with a terminal slot-state report when ALL slots quarantine.

The tests drive the REAL ``cmd_dispatch`` loop with REAL subprocesses; fakes
sit only at external boundaries, signature-conformant: ``_worker_cmd`` (the
child-process boundary — the fake returns a real ``python -c`` command keyed
on the launcher-pinned ``CUDA_VISIBLE_DEVICES``, so the CVD-pin contract is
exercised: sick slots exit 2 instantly, healthy slots exit 0), ``_finalize``
(the fu3w sentinel boundary — recorded, mirrors ``_finalize(cfg, label,
done, failed, skipped)``), and ``_resolve_slugs``/``_cell_done`` (the cell
registry). test_quarantine_refund fails PRE-fix: without the refund, the
blackhole burns the single retry and the run exits rc=1 with permanent
failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1947_worker as w  # noqa: E402

SLUGS = ["cellA", "cellB", "cellC"]


def _mk_args() -> argparse.Namespace:
    """The argparse fields cmd_dispatch reads (label, width, poll, dry-run)."""
    return argparse.Namespace(
        dispatch="qtest", n_gpus=2, poll_seconds=0.05, dry_run=False, eval_question_limit=None
    )


def _mk_cfg(tmp_path: Path) -> w.Cfg:
    return w.Cfg(
        smoke=True, out_root=tmp_path / "out", upload=False, sentinel_dir=tmp_path / "sent"
    )


def _patch_dispatch(monkeypatch, sick_slots: tuple[str, ...]) -> dict:
    """Wire the boundary fakes; returns the _finalize call recorder."""
    monkeypatch.setattr(w, "_resolve_slugs", lambda args: list(SLUGS))
    monkeypatch.setattr(w, "_cell_done", lambda cfg, cell: False)
    monkeypatch.setattr(w.cells, "CELL_BY_SLUG", {s: object() for s in SLUGS})
    recorded: dict = {}

    def fake_finalize(cfg, label, done, failed, skipped):
        recorded.update(label=label, done=list(done), failed=list(failed), skipped=list(skipped))

    monkeypatch.setattr(w, "_finalize", fake_finalize)
    # The fake worker keys on the REAL launcher-env CVD pin (gotchas.md CVD
    # contract): a sick slot's child dies instantly rc=2, a healthy slot's
    # child exits 0 (rc==0 + no status file == done, per cmd_dispatch).
    code = (
        "import os, sys; "
        f"sys.exit(2 if os.environ.get('CUDA_VISIBLE_DEVICES') in {sick_slots!r} else 0)"
    )

    def fake_worker_cmd(args, cfg, slug, slot):
        return [sys.executable, "-c", code]

    monkeypatch.setattr(w, "_worker_cmd", fake_worker_cmd)
    return recorded


def test_quarantine_refund_and_healthy_completion(monkeypatch, tmp_path, capsys):
    """Sick slot 1 quarantines after 2 consecutive fast failures; the cells
    that failed on it get their retry refunded and complete on the healthy
    slot — zero permanent failures (pre-fix: rc=1 with permanent failures)."""
    recorded = _patch_dispatch(monkeypatch, sick_slots=("1",))
    rc = w.cmd_dispatch(_mk_args(), _mk_cfg(tmp_path))
    out = capsys.readouterr().out
    assert "slot 1 quarantined after 2 consecutive fast failures" in out
    assert "retry refunded" in out  # the refund branch actually fired
    assert rc == 0
    assert recorded["failed"] == []
    assert sorted(recorded["done"]) == sorted(SLUGS)


def test_all_slots_quarantined_fails_loud(monkeypatch, tmp_path, capsys):
    """Every slot sick => terminal RuntimeError naming the slot states (never a
    silent hang or an all-cells-failed finalize)."""
    recorded = _patch_dispatch(monkeypatch, sick_slots=("0", "1"))
    with pytest.raises(RuntimeError, match="all 2 dispatch slots quarantined"):
        w.cmd_dispatch(_mk_args(), _mk_cfg(tmp_path))
    out = capsys.readouterr().out
    assert "FATAL: all 2 slots quarantined" in out
    assert "slot 0" in out and "slot 1" in out  # the slot-state report
    assert recorded.get("done") is None  # finalize never ran — fail-loud crash


def test_slow_failure_never_quarantines(monkeypatch, tmp_path, capsys):
    """A slow rc!=0 exit (>= SLOT_FAST_FAIL_SECONDS) resets the streak — only
    INSTANT failures mark a slot sick; an ordinary crashing cell burns its own
    retry and fails permanently without benching the slot."""
    recorded = _patch_dispatch(monkeypatch, sick_slots=("0", "1"))
    # Make every exit read as SLOW: threshold below any realizable elapsed.
    monkeypatch.setattr(w, "SLOT_FAST_FAIL_SECONDS", 0.0)
    rc = w.cmd_dispatch(_mk_args(), _mk_cfg(tmp_path))
    out = capsys.readouterr().out
    assert "quarantined after" not in out  # no quarantine EVENT fired
    assert "FATAL" not in out
    assert rc == 1  # cells legitimately failed after their single retry
    assert sorted(recorded["failed"]) == sorted(SLUGS)
