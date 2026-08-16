"""Tests for the autonomous-session watcher's vm_ledger_reap pass (plan §5).

The pass reaps expired-TTL / dead-PID rows from the advisory VM resource ledger
every 10-min tick. Kill switch EPM_DISABLE_VM_LEDGER_REAP; fail-soft; --dry-run
reports without mutating.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import autonomous_session_watch as w
import resource_ledger


def test_kill_switch():
    assert w._vm_ledger_reap_disabled({}) is False
    assert w._vm_ledger_reap_disabled({"EPM_DISABLE_VM_LEDGER_REAP": "1"}) is True
    assert w._vm_ledger_reap_disabled({"EPM_DISABLE_VM_LEDGER_REAP": "yes"}) is True
    assert w._vm_ledger_reap_disabled({"EPM_DISABLE_VM_LEDGER_REAP": "0"}) is False


def test_disabled_pass_skips_reap(monkeypatch, capsys):
    called = {"n": 0}
    monkeypatch.setattr(
        resource_ledger,
        "reap_ledger_file",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    monkeypatch.setenv("EPM_DISABLE_VM_LEDGER_REAP", "1")
    w.vm_ledger_reap_pass(dry_run=False)
    assert called["n"] == 0
    assert "disabled" in capsys.readouterr().out


def test_dry_run_passes_apply_false(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        resource_ledger,
        "reap_ledger_file",
        lambda *a, **k: seen.setdefault("apply", k.get("apply")) or [],
    )
    monkeypatch.delenv("EPM_DISABLE_VM_LEDGER_REAP", raising=False)
    w.vm_ledger_reap_pass(dry_run=True)
    assert seen["apply"] is False
    assert "no stale claims" in capsys.readouterr().out


def test_reports_reaped(monkeypatch, capsys):
    monkeypatch.setattr(
        resource_ledger,
        "reap_ledger_file",
        lambda *a, **k: [{"issue": 900, "phase": "fit", "claim_id": "abc123"}],
    )
    monkeypatch.delenv("EPM_DISABLE_VM_LEDGER_REAP", raising=False)
    w.vm_ledger_reap_pass(dry_run=False)
    out = capsys.readouterr().out
    assert "reaped 1 stale claim" in out and "#900:fit" in out


def test_fail_soft_on_reaper_error(monkeypatch, capsys):
    def _boom(*a, **k):
        raise RuntimeError("ledger wedged")

    monkeypatch.setattr(resource_ledger, "reap_ledger_file", _boom)
    monkeypatch.delenv("EPM_DISABLE_VM_LEDGER_REAP", raising=False)
    # Must NOT raise — a ledger hiccup can never crash the watcher tick.
    w.vm_ledger_reap_pass(dry_run=False)
    assert "error (skipping)" in capsys.readouterr().out
