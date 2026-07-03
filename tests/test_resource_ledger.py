"""Tests for scripts/resource_ledger.py — the advisory VM CPU/RAM ledger (v2)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import resource_ledger as rl

# Generous machine so a small claim always fits unless we say otherwise:
# 32 cores / 128 GiB, ~1 core + 4 GiB live usage.
_TOTALS = (32.0, 128.0)
_LIVE = (1.0, 4.0)
_ALIVE = lambda _pid: True  # noqa: E731 -- test stub
_DEAD = lambda _pid: False  # noqa: E731 -- test stub


def _now() -> float:
    return datetime(2026, 7, 3, 0, 0, 0, tzinfo=UTC).timestamp()


# ── decide_claim (pure) ──────────────────────────────────────────────────────


def test_decide_claim_within_band_ok():
    ok, _ = rl.decide_claim(
        this_cores=2,
        this_ram_gb=8,
        live_cores_used=1,
        live_ram_used_gb=4,
        claimed_cores=0,
        claimed_ram_gb=0,
        total_cores=32,
        total_ram_gb=128,
    )
    assert ok


def test_decide_claim_over_cores_routes():
    ok, reason = rl.decide_claim(
        this_cores=30,
        this_ram_gb=8,
        live_cores_used=1,
        live_ram_used_gb=4,
        claimed_cores=0,
        claimed_ram_gb=0,
        total_cores=32,
        total_ram_gb=128,
    )
    assert not ok and "cores" in reason


def test_decide_claim_over_ram_routes():
    ok, reason = rl.decide_claim(
        this_cores=1,
        this_ram_gb=120,
        live_cores_used=1,
        live_ram_used_gb=4,
        claimed_cores=0,
        claimed_ram_gb=0,
        total_cores=32,
        total_ram_gb=128,
    )
    assert not ok and "RAM" in reason


def test_decide_claim_counts_existing_claims():
    # this claim alone fits, but stacked on top of existing claims it does not.
    ok, _ = rl.decide_claim(
        this_cores=8,
        this_ram_gb=8,
        live_cores_used=1,
        live_ram_used_gb=4,
        claimed_cores=16,
        claimed_ram_gb=0,
        total_cores=32,
        total_ram_gb=128,
    )
    assert not ok  # 1 + 16 + 8 = 25 > 0.7 * 32 = 22.4


# ── reap_rows (pure) ─────────────────────────────────────────────────────────


def _row(**over) -> dict:
    base = {
        "claim_id": "abc123",
        "issue": 900,
        "pid": 4321,
        "cores": 2,
        "ram_gb": 8,
        "phase": "fit",
        "created_iso": datetime.fromtimestamp(_now(), UTC).isoformat(),
        "ttl_s": rl.DEFAULT_TTL_S,
    }
    base.update(over)
    return base


def test_reap_rows_keeps_live_row():
    kept, reaped = rl.reap_rows([_row()], now=_now() + 60, pid_alive=_ALIVE)
    assert len(kept) == 1 and not reaped


def test_reap_rows_reaps_expired_ttl():
    kept, reaped = rl.reap_rows([_row(ttl_s=100)], now=_now() + 200, pid_alive=_ALIVE)
    assert not kept and len(reaped) == 1


def test_reap_rows_reaps_dead_pid():
    kept, reaped = rl.reap_rows([_row()], now=_now() + 60, pid_alive=_DEAD)
    assert not kept and len(reaped) == 1


def test_reap_rows_drops_malformed_row():
    kept, reaped = rl.reap_rows([{"claim_id": "x"}], now=_now(), pid_alive=_ALIVE)
    assert not kept and len(reaped) == 1


# ── claim / release / status (I/O, tmp ledger) ───────────────────────────────


def test_claim_records_when_headroom(tmp_path):
    led = tmp_path / "vm-ledger.json"
    ok, _, row = rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    assert ok and row is not None
    data = json.loads(led.read_text())
    assert len(data["rows"]) == 1 and data["rows"][0]["claim_id"] == row["claim_id"]


def test_claim_over_band_routes_and_does_not_record(tmp_path):
    led = tmp_path / "vm-ledger.json"
    ok, reason, row = rl.claim(
        issue=900,
        cores=30,
        ram_gb=8,
        phase="huge",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    assert not ok and row is None and "cores" in reason
    assert not led.exists() or json.loads(led.read_text())["rows"] == []


def test_claim_force_records_over_band(tmp_path):
    led = tmp_path / "vm-ledger.json"
    ok, _, row = rl.claim(
        issue=900,
        cores=30,
        ram_gb=8,
        phase="huge",
        force=True,
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    assert not ok and row is not None
    assert len(json.loads(led.read_text())["rows"]) == 1


def test_release_removes_then_idempotent(tmp_path):
    led = tmp_path / "vm-ledger.json"
    _, _, row = rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    assert rl.release(row["claim_id"], ledger_path=led) is True
    assert rl.release(row["claim_id"], ledger_path=led) is False
    assert json.loads(led.read_text())["rows"] == []


def test_status_snapshot_reaps_dead(tmp_path):
    led = tmp_path / "vm-ledger.json"
    rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    snap = rl.status(
        ledger_path=led, now=_now() + 60, pid_alive=_DEAD, live_usage=_LIVE, totals=_TOTALS
    )
    assert snap["n_claims"] == 0 and snap["n_reaped"] == 1
    assert snap["total_cores"] == 32.0


def test_reap_ledger_file_dead_pid(tmp_path):
    led = tmp_path / "vm-ledger.json"
    rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    reaped = rl.reap_ledger_file(led, now=_now() + 60, pid_alive=_DEAD)
    assert len(reaped) == 1
    assert json.loads(led.read_text())["rows"] == []


def test_reap_ledger_file_apply_false_no_write(tmp_path):
    led = tmp_path / "vm-ledger.json"
    rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    reaped = rl.reap_ledger_file(led, now=_now() + 60, pid_alive=_DEAD, apply=False)
    assert len(reaped) == 1  # reported...
    assert len(json.loads(led.read_text())["rows"]) == 1  # ...but NOT written


def test_missing_ledger_fails_toward_available(tmp_path):
    led = tmp_path / "does-not-exist.json"
    ok, _, row = rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    assert ok and row is not None  # a missing ledger => VM presumed free


def test_corrupt_ledger_renamed_aside_not_truncated(tmp_path):
    led = tmp_path / "vm-ledger.json"
    led.write_text("{ this is not valid json ]]]")
    ok, _, row = rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    assert ok and row is not None  # fresh start after rename
    # the corrupt content was preserved aside, never silently truncated
    aside = list(tmp_path.glob("vm-ledger.json.corrupt-*"))
    assert len(aside) == 1
    assert "not valid json" in aside[0].read_text()
    # the live ledger is now a clean single-row file
    assert len(json.loads(led.read_text())["rows"]) == 1


def test_claim_leaves_no_tmp_file(tmp_path):
    led = tmp_path / "vm-ledger.json"
    rl.claim(
        issue=900,
        cores=2,
        ram_gb=8,
        phase="fit",
        ledger_path=led,
        now=_now(),
        pid=4321,
        pid_alive=_ALIVE,
        live_usage=_LIVE,
        totals=_TOTALS,
    )
    assert not (tmp_path / "vm-ledger.json.tmp").exists()  # atomic replace cleaned up


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_status_json_smoke(capsys):
    # `status` reads live psutil; assert it emits valid JSON with the fields.
    rc = rl.main(["status", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert {"total_cores", "live_cores_used", "n_claims", "claims"} <= out.keys()
