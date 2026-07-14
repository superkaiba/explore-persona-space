"""Tests for the auth-outage guard pass (task #1027).

2026-07-03 incident: an Anthropic auth outage (poisoned Claude CLI
credential, recovered by /login) killed every freshly spawned session on
arrival and the watcher's respawn arms churned die-on-arrival sessions
fleet-wide for hours. The guard detects the fleet-wide instant-freeze
respawn signature, suppresses every watcher spawn arm, pushes once per
episode, and probes recovery with a canary respawn. FAIL-OPEN throughout.

Covers plan #1027 §5 cases 1-26: the pure trigger/canary predicates
(1-11, 26), the spawn gate + canary token + kill switch + fail-soft
(12-15, 25), push dedup + event pruning + dry-run hygiene (16-18), the
main() ordering pin (19), the 2026-07-03 acceptance replay as a pytest
case (20), the MF-1 post-resolve watermark (21), the MF-2 stalled
stop+respawn unit gate (22), the MF-3 canary identity binding (23), and
the MF-4 per-site wiring certification (24).

Mirrors tests/test_cpu_guard_pass.py's bootstrap + watcher_roots fixture.
"""

import inspect
import json
import sys
import time
from pathlib import Path

import pytest

from tests.conftest import _stub_fleet_mutating_passes

# Bootstrap sys.path the same way the sibling watcher tests do (scripts/ on
# the path so autonomous_session_watch imports by name).
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402

# Captured BEFORE the autouse hermetic fixture stubs it, so the real body can
# still be exercised (one production-body test per seam-stubbed function).
_REAL_EVIDENCE = asw._auth_outage_evidence

# ─── fixtures + helpers ──────────────────────────────────────────────────────

NOW = 1_800_000_000.0

# Module defaults, spelled out so the pure-predicate tests are self-contained.
_TRIG = dict(window_s=10800.0, fresh_death_s=3600.0, min_freeze_events=3, min_distinct_issues=2)
_CAN = dict(canary_interval_s=1800.0, canary_survival_s=1200.0, max_episode_s=21600.0)


def _ev(issue, ts, prev):
    return {"issue": issue, "ts": ts, "arm": "crash", "prev_spawned_at": prev}


def _freeze_events(now, issues=(101, 102, 103), age_s=2400.0):
    """Trigger-worthy events: respawns whose predecessor lived ``age_s``."""
    return [_ev(i, now - 10 * k, now - 10 * k - age_s) for k, i in enumerate(issues, 1)]


@pytest.fixture
def watcher_roots(tmp_path, monkeypatch):
    """Pin PROJECT_ROOT (sidecar) and AUTONOMOUS_REGISTRY_DIR (state) at a
    temp dir so the pass is fully offline."""
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "reg")
    (tmp_path / "reg").mkdir()
    return tmp_path


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """Reset the per-tick canary token and keep the evidence grep off the
    real ~/.happy/logs (enrichment only — never the trigger)."""
    monkeypatch.setattr(asw, "_AUTH_CANARY_TOKEN", False)
    monkeypatch.setattr(asw, "_auth_outage_evidence", lambda: "churn-only")


# The #1247 hermeticity guard (`_forbid_real_marker_posts`) is now a shared
# autouse fixture in tests/conftest.py (task #1265) — it applies here
# automatically, and this file additionally gains the round-2
# `_forbid_real_task_status_reads` coverage it never had a per-file copy of.


@pytest.fixture
def push_counter(monkeypatch):
    """Count REAL (non-dry-run) pushes; a dry_run call sends nothing, like
    the production helper."""
    pushes: list[str] = []

    def fake_push(msg, dry_run):
        if not dry_run:
            pushes.append(msg)
        return not dry_run

    monkeypatch.setattr(asw, "_telegram_push", fake_push)
    return pushes


def _write_state(reg_dir: Path, **fields) -> None:
    (reg_dir / "auth-outage.json").write_text(json.dumps(fields))


def _read_state(root: Path) -> dict:
    path = root / "reg" / "auth-outage.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _sidecar_rows(root: Path) -> list[dict]:
    path = root / ".claude" / "cache" / "auth-outage-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ─── 1-6, 26: decide_auth_outage_trigger ─────────────────────────────────────


def test_trigger_fires_on_fleet_instant_freeze():
    events = [
        _ev(1, NOW - 100, NOW - 1000),
        _ev(2, NOW - 200, NOW - 1500),
        _ev(1, NOW - 300, NOW - 2000),
    ]
    assert asw.decide_auth_outage_trigger(events, NOW, **_TRIG) is True


def test_trigger_ignores_single_issue_churn():
    # 5 instant-freeze respawns, ONE issue: per-task caps own this class.
    events = [_ev(1, NOW - 100 * k, NOW - 100 * k - 900) for k in range(1, 6)]
    assert asw.decide_auth_outage_trigger(events, NOW, **_TRIG) is False


def test_trigger_ignores_slow_respawns():
    # Healthy multi-hour sessions that later crashed: delta > fresh_death_s.
    events = [
        _ev(1, NOW - 100, NOW - 100 - 7200),
        _ev(2, NOW - 200, NOW - 200 - 5000),
        _ev(3, NOW - 300, NOW - 300 - 4000),
    ]
    assert asw.decide_auth_outage_trigger(events, NOW, **_TRIG) is False


def test_trigger_ignores_stale_events():
    old = NOW - 11000  # outside the 10800 s window
    events = [_ev(1, old, old - 900), _ev(2, old - 100, old - 1000), _ev(3, old - 50, old - 950)]
    assert asw.decide_auth_outage_trigger(events, NOW, **_TRIG) is False


def test_trigger_ignores_events_without_prev():
    # infra-drain / capacity-retry / first orphan spawns carry no predecessor.
    events = [
        {"issue": i, "ts": NOW - 100, "arm": "infra-drain", "prev_spawned_at": None}
        for i in (1, 2, 3)
    ]
    assert asw.decide_auth_outage_trigger(events, NOW, **_TRIG) is False


def test_trigger_boundary_exact_thresholds():
    three_two = [
        _ev(1, NOW - 100, NOW - 1000),
        _ev(2, NOW - 200, NOW - 1500),
        _ev(1, NOW - 300, NOW - 2000),
    ]
    assert asw.decide_auth_outage_trigger(three_two, NOW, **_TRIG) is True
    # 2 events: below the event floor.
    assert asw.decide_auth_outage_trigger(three_two[:2], NOW, **_TRIG) is False
    # 3 events, 1 distinct issue: below the issue floor.
    three_one = [_ev(1, NOW - 100 * k, NOW - 100 * k - 900) for k in range(1, 4)]
    assert asw.decide_auth_outage_trigger(three_one, NOW, **_TRIG) is False


def test_time_boundaries_and_skew():
    # ts - prev == fresh_death_s qualifies (<=).
    edge = [_ev(i, NOW - 10 * i, NOW - 10 * i - 3600) for i in (1, 2, 3)]
    assert asw.decide_auth_outage_trigger(edge, NOW, **_TRIG) is True
    # now - ts == window_s qualifies (<=).
    ts = NOW - 10800
    win = [_ev(i, ts, ts - 100) for i in (1, 2, 3)]
    assert asw.decide_auth_outage_trigger(win, NOW, **_TRIG) is True
    # NEGATIVE delta (clock skew / future prev_spawned_at) excluded by 0 <=.
    neg = [_ev(i, NOW - 10, NOW) for i in (1, 2, 3)]
    assert asw.decide_auth_outage_trigger(neg, NOW, **_TRIG) is False
    # issue=None / bool never counts toward the distinct-issue floor.
    mixed = [
        _ev(None, NOW - 10, NOW - 900),
        _ev(True, NOW - 20, NOW - 900),
        _ev(1, NOW - 30, NOW - 900),
    ]
    assert asw.decide_auth_outage_trigger(mixed, NOW, **_TRIG) is False


# ─── 7-11: decide_auth_outage_canary ─────────────────────────────────────────


def test_canary_arms_after_interval():
    state = {"active": True, "started_ts": NOW - 2000}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=None, **_CAN) == "arm-canary"


def test_canary_holds_before_interval():
    state = {"active": True, "started_ts": NOW - 100}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=None, **_CAN) == "hold"
    # last_canary_ts is the anchor once a canary has spawned.
    state = {"active": True, "started_ts": NOW - 5000, "last_canary_ts": NOW - 100}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=None, **_CAN) == "hold"
    # A fresh consumed-but-unspawned claim (canary_pending) also holds.
    state = {"active": True, "started_ts": NOW - 5000, "canary_pending": {"ts": NOW - 60}}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=None, **_CAN) == "hold"


def test_canary_resolve_on_survival():
    state = {"active": True, "started_ts": NOW - 3000, "canary_ts": NOW - 1300}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=True, **_CAN) == "resolve"
    # Alive but young: hold.
    state = {"active": True, "started_ts": NOW - 3000, "canary_ts": NOW - 600}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=True, **_CAN) == "hold"


def test_canary_failed_on_death():
    state = {"active": True, "started_ts": NOW - 3000, "canary_ts": NOW - 600}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=False, **_CAN) == "canary-failed"


def test_canary_inconclusive_daemon_down_holds():
    state = {"active": True, "started_ts": NOW - 3000, "canary_ts": NOW - 1300}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=None, **_CAN) == "hold"


def test_episode_expires_fail_open_at_ttl():
    # Expire wins REGARDLESS of a live, survival-aged canary.
    state = {"active": True, "started_ts": NOW - 22000, "canary_ts": NOW - 1300}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=True, **_CAN) == "expire"
    # A garbled started_ts also expires (fail-open, never wedges suppression).
    state = {"active": True, "started_ts": "garbled"}
    assert asw.decide_auth_outage_canary(state, NOW, canary_alive=None, **_CAN) == "expire"


# ─── 12-15: the spawn gate ───────────────────────────────────────────────────


def test_gate_suppresses_during_episode(watcher_roots):
    _write_state(watcher_roots / "reg", active=True, started_ts=time.time() - 60, events=[])
    assert asw._auth_outage_spawn_gate(5, "crash") == "auth-outage"
    # The TTL binds in the GATE too (second fail-open layer): an episode
    # older than the max never suppresses even if the pass is wedged.
    _write_state(
        watcher_roots / "reg",
        active=True,
        started_ts=time.time() - asw.AUTH_OUTAGE_MAX_EPISODE_S - 60,
        events=[],
    )
    assert asw._auth_outage_spawn_gate(5, "crash") is None


def test_gate_allows_when_inactive(watcher_roots):
    assert asw._auth_outage_spawn_gate(5, "crash") is None  # no state file at all
    _write_state(watcher_roots / "reg", active=False, events=[])
    assert asw._auth_outage_spawn_gate(5, "crash") is None


def test_gate_canary_token_allows_exactly_one(watcher_roots):
    _write_state(watcher_roots / "reg", active=True, started_ts=time.time() - 60, events=[])
    asw._AUTH_CANARY_TOKEN = True
    assert asw._auth_outage_spawn_gate(5, "crash") is None
    st = _read_state(watcher_roots)
    assert st["canary_pending"]["issue"] == 5  # the cross-tick claim persisted
    assert asw._AUTH_CANARY_TOKEN is False
    # Second consult (different issue): token consumed -> suppressed.
    assert asw._auth_outage_spawn_gate(6, "crash") == "auth-outage"


def test_gate_kill_switch_allows(watcher_roots, monkeypatch, push_counter, capsys):
    monkeypatch.setenv("EPM_DISABLE_AUTH_OUTAGE_GUARD", "1")
    _write_state(watcher_roots / "reg", active=True, started_ts=time.time() - 60, events=[])
    before = (watcher_roots / "reg" / "auth-outage.json").read_bytes()
    assert asw._auth_outage_spawn_gate(5, "crash") is None
    asw._auth_outage_record_spawn(5, "crash", None)  # no-op
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())  # no-op
    assert "disabled via EPM_DISABLE_AUTH_OUTAGE_GUARD" in capsys.readouterr().out
    assert (watcher_roots / "reg" / "auth-outage.json").read_bytes() == before
    assert push_counter == []


def test_gate_fail_soft_on_corrupt_state(watcher_roots):
    (watcher_roots / "reg" / "auth-outage.json").write_text("{garbage not json")
    # Corrupt state -> fresh empty state -> "no outage" -> allow, no raise.
    assert asw._auth_outage_spawn_gate(5, "crash") is None


# ─── 16-18: push dedup, pruning, dry-run hygiene ─────────────────────────────


def test_push_deduped_per_episode(watcher_roots, push_counter):
    now = time.time()
    # trigger_pushed already True (e.g. a crash between push and a later
    # tick): the episode activates with ZERO additional pushes.
    _write_state(
        watcher_roots / "reg", active=False, events=_freeze_events(now), trigger_pushed=True
    )
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    assert _read_state(watcher_roots)["active"] is True
    assert push_counter == []
    # A fresh episode fires EXACTLY one trigger push...
    _write_state(
        watcher_roots / "reg", active=False, events=_freeze_events(now), trigger_pushed=False
    )
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    assert len(push_counter) == 1
    assert "AUTH OUTAGE SUSPECTED" in push_counter[0]
    # ...and subsequent active ticks add none.
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    assert len(push_counter) == 1


def test_record_prunes_events(watcher_roots):
    now = time.time()
    old = now - 3 * asw.AUTH_OUTAGE_WINDOW_S  # beyond the 2x-window horizon
    _write_state(watcher_roots / "reg", active=False, events=[_ev(1, old, old - 100)])
    asw._auth_outage_record_spawn(2, "crash", now - 900)
    st = _read_state(watcher_roots)
    assert [e["issue"] for e in st["events"]] == [2]
    assert st["events"][0]["arm"] == "crash"


def test_dry_run_writes_nothing(watcher_roots, push_counter):
    now = time.time()
    # A trigger-worthy tick under dry-run: zero state writes, zero pushes,
    # zero sidecar rows.
    _write_state(watcher_roots / "reg", active=False, events=_freeze_events(now))
    before = (watcher_roots / "reg" / "auth-outage.json").read_bytes()
    asw.auth_outage_pass(True, daemon_reachable=True, live_ids=set())
    assert (watcher_roots / "reg" / "auth-outage.json").read_bytes() == before
    assert push_counter == []
    assert _sidecar_rows(watcher_roots) == []
    # An active-episode arm-canary tick under dry-run: no token armed either.
    _write_state(watcher_roots / "reg", active=True, started_ts=now - 3600, events=[])
    before = (watcher_roots / "reg" / "auth-outage.json").read_bytes()
    asw.auth_outage_pass(True, daemon_reachable=True, live_ids=set())
    assert (watcher_roots / "reg" / "auth-outage.json").read_bytes() == before
    assert asw._AUTH_CANARY_TOKEN is False


# ─── 19: main() ordering — the pass runs BEFORE the crash-recovery loop ──────


def test_main_order_auth_outage_before_respawn_loop(watcher_roots, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    monkeypatch.setattr(asw, "_issue_registrations", lambda: {})
    monkeypatch.setattr(asw, "_campaign_gate_candidates", lambda: set())
    # #1247/#1278 hermeticity: the shared conftest helper stubs every
    # fleet-MUTATING pass (incl. the #1267 boot_death_pass this file's inline
    # copy was missing) fail-loud. Called BEFORE the recorders below — a later
    # monkeypatch wins (helper docstring convention).
    _stub_fleet_mutating_passes(asw, monkeypatch)
    # Determinism-only remainder (not in the helper's fleet-mutating list):
    # neutralize the other passes so main() runs cheaply + deterministically.
    # Fail-loud by design — no attribute-existence silent-skip;
    # monkeypatch.setattr's default raising=True crashes if a pass is renamed
    # instead of letting it run live.
    for name in (
        "vm_disk_pass",
        "triage_observer_pass",
        "campaign_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "infra_drain_pass",
        "stale_blocked_flag_pass",
        "session_reconcile_pass",
        "zombie_wrapper_pass",
        "idle_unmapped_pass",
        "stale_registration_pass",
    ):
        monkeypatch.setattr(asw, name, lambda *a, **kw: None)
    monkeypatch.setattr(asw, "auth_outage_pass", lambda *a, **kw: order.append("auth_outage"))
    monkeypatch.setattr(asw, "_process_entry", lambda *a, **kw: order.append("process_entry"))
    # One registered issue so the crash-recovery loop actually iterates.
    (watcher_roots / "reg" / "issue-1.json").write_text(
        json.dumps({"issue": 1, "happy_session_id": "sid-1", "spawned_at": time.time()})
    )
    rc = asw.main([])
    assert rc == 0
    assert "auth_outage" in order and "process_entry" in order
    assert order.index("auth_outage") < order.index("process_entry")


def test_main_auth_outage_only_flag(watcher_roots, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: False)
    monkeypatch.setattr(asw, "auth_outage_pass", lambda *a, **kw: calls.append("auth"))
    monkeypatch.setattr(
        asw, "vm_disk_pass", lambda *a, **kw: pytest.fail("ran another pass under --only")
    )
    rc = asw.main(["--auth-outage-only", "--dry-run"])
    assert rc == 0
    assert calls == ["auth"]


# ─── 20: the 2026-07-03 acceptance replay (AC1) ──────────────────────────────


def test_acceptance_replay_2026_07_03(watcher_roots, push_counter):
    now = time.time()
    # Replay the incident shape THROUGH the real record hook: three watcher
    # respawns whose predecessor sessions lived 30-40 min (die-on-arrival),
    # across three distinct issues, inside the 3 h window.
    for issue, age in ((101, 40 * 60), (102, 35 * 60), (103, 30 * 60)):
        asw._auth_outage_record_spawn(issue, "crash", now - age)
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    st = _read_state(watcher_roots)
    assert st["active"] is True
    assert len(push_counter) == 1
    assert "AUTH OUTAGE SUSPECTED" in push_counter[0]
    rows = _sidecar_rows(watcher_roots)
    assert [r["transition"] for r in rows] == ["trigger"]
    assert sorted(rows[0]["distinct_issues"]) == [101, 102, 103]
    # And the very next spawn attempt is suppressed.
    assert asw._auth_outage_spawn_gate(101, "crash") == "auth-outage"


# ─── 21: MF-1 — no re-trigger from pre-resolve / backlog events ──────────────


def test_no_retrigger_from_pre_resolve_events(watcher_roots, push_counter):
    reg = watcher_roots / "reg"
    now = time.time()
    _write_state(reg, active=False, events=_freeze_events(now))
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    assert _read_state(watcher_roots)["active"] is True
    # Graft a survived canary (registration + persisted sid + live set).
    (reg / "issue-101.json").write_text(
        json.dumps({"issue": 101, "happy_session_id": "sid-c", "spawned_at": now})
    )
    st = _read_state(watcher_roots)
    st.update(canary_issue=101, canary_session_id="sid-c", canary_ts=now - 1300)
    _write_state(reg, **st)
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids={"sid-c"})
    st = _read_state(watcher_roots)
    assert st["active"] is False
    watermark = st["last_episode_end_ts"]
    assert watermark > 0
    # Events are RETAINED (the watermark, not deletion, blocks re-trigger)...
    assert len(st["events"]) == 3
    # ...and the predicate over the retained events is False post-resolve.
    assert (
        asw.decide_auth_outage_trigger(
            st["events"], time.time(), **_TRIG, last_episode_end_ts=watermark
        )
        is False
    )
    # Two more passes with no new events: stays inactive, zero extra pushes.
    n_pushes = len(push_counter)
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    assert _read_state(watcher_roots)["active"] is False
    assert len(push_counter) == n_pushes
    # Backlog respawns: predecessors spawned DURING the episode (before the
    # watermark) are not fresh post-recovery evidence — still no re-trigger.
    for issue in (104, 105, 106):
        asw._auth_outage_record_spawn(issue, "crash", watermark - 3000)
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())
    assert _read_state(watcher_roots)["active"] is False


# ─── 22: MF-2 — the stalled arm gates the stop+respawn UNIT ──────────────────


def _stalled_ctx(issue, sid, live_ids, *, stop_pending_sid=None):
    return asw._StalledActionCtx(
        issue=issue,
        happy_session_id=sid,
        prev_state={},
        alerted=True,
        respawn_count=0,
        exhausted=False,
        last_self_report_ts=None,
        self_gap="2h",
        marker_gap="2h",
        has_pod=False,
        task_status="running",
        in_active=True,
        threshold=2,
        dry_run=False,
        live_ids=live_ids,
        now=time.time(),
        stop_pending_sid=stop_pending_sid,
    )


def test_stalled_gate_skips_stop_and_respawn_as_unit(watcher_roots, monkeypatch):
    _write_state(watcher_roots / "reg", active=True, started_ts=time.time() - 60, events=[])
    stops: list[str] = []
    spawns: list[int] = []
    monkeypatch.setattr(asw, "_stalled_arm_deferral", lambda ctx: False)
    monkeypatch.setattr(asw, "_stop_session", lambda sid, dry: stops.append(sid) or True)
    monkeypatch.setattr(
        asw, "_respawn_stalled_session", lambda issue, cap, dry: spawns.append(issue) or "spawned"
    )
    monkeypatch.setattr(asw, "_persist_stalled_ctx", lambda *a, **kw: None)
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **kw: None)
    # #1247 fence act-guard seam: confirm-active so the guard's live re-read
    # never shells the real `task.py view` subprocess (hermeticity).
    monkeypatch.setattr(asw, "_task_status", lambda _i: "running")
    # Episode active, no token: NEITHER the stop NOR the respawn fires.
    asw._handle_stalled_respawn(_stalled_ctx(7, "sid-7", {"sid-7"}))
    assert stops == [] and spawns == []
    # With the canary token, the WHOLE unit runs: the fence's stop tick...
    asw._AUTH_CANARY_TOKEN = True
    asw._handle_stalled_respawn(_stalled_ctx(7, "sid-7", {"sid-7"}))
    assert stops == ["sid-7"] and spawns == []
    # ...and the verified-dead spawn on the NEXT tick passes the gate via the
    # persisted canary_pending claim (the token itself is already consumed).
    assert asw._AUTH_CANARY_TOKEN is False
    asw._handle_stalled_respawn(_stalled_ctx(7, "sid-7", set(), stop_pending_sid="sid-7"))
    assert spawns == [7]


# ─── 23: MF-3 — canary identity binding ──────────────────────────────────────


def test_canary_binding(watcher_roots, monkeypatch):
    reg = watcher_roots / "reg"
    now = time.time()
    _write_state(reg, active=True, started_ts=now - 60, events=[])
    # (a) Canary fields persist ONLY after a "spawned" result: consuming the
    # token records just the pending claim (a "failed" spawn never binds).
    asw._AUTH_CANARY_TOKEN = True
    assert asw._auth_outage_spawn_gate(7, "crash") is None
    st = _read_state(watcher_roots)
    assert st.get("canary_ts") is None
    assert st["canary_pending"]["issue"] == 7
    # The record hook (called only on "spawned") binds the FRESH registry sid.
    (reg / "issue-7.json").write_text(
        json.dumps({"issue": 7, "happy_session_id": "sid-fresh", "spawned_at": now})
    )
    asw._auth_outage_record_spawn(7, "crash", now - 1200)
    st = _read_state(watcher_roots)
    assert st["canary_issue"] == 7
    assert st["canary_session_id"] == "sid-fresh"
    assert st["canary_pending"] is None
    assert [r["transition"] for r in _sidecar_rows(watcher_roots)] == ["canary-armed"]
    # (c) Liveness reads the PERSISTED sid against live_ids.
    assert asw._auth_canary_alive(st, {"sid-fresh"}) is True
    assert asw._auth_canary_alive(st, set()) is False
    assert asw._auth_canary_alive(st, None) is None
    # A REPLACED registration invalidates (never reads the new sid as alive).
    (reg / "issue-7.json").write_text(
        json.dumps({"issue": 7, "happy_session_id": "sid-other", "spawned_at": now})
    )
    assert asw._auth_canary_alive(st, {"sid-fresh", "sid-other"}) is False
    # (d) Terminal-parked canary (registration GONE): invalidated -> the pass
    # clears the canary fields + re-arms; NEVER a false resolve.
    (reg / "issue-7.json").unlink()
    assert asw._auth_canary_alive(st, {"sid-fresh"}) is False
    asw.auth_outage_pass(False, daemon_reachable=True, live_ids={"sid-fresh"})
    st = _read_state(watcher_roots)
    assert st["active"] is True  # no resolve
    assert st["canary_ts"] is None and st["canary_issue"] is None
    assert st["skip_last_canary_once"] is True
    # (e) Round-robin: the NEXT token skips the failed canary issue once.
    asw._AUTH_CANARY_TOKEN = True
    assert asw._auth_outage_spawn_gate(7, "crash") == "auth-outage"
    assert asw._auth_outage_spawn_gate(8, "crash") is None
    # (b) The campaign arm NEVER consumes the token.
    _write_state(reg, active=True, started_ts=now - 60, events=[])
    asw._AUTH_CANARY_TOKEN = True
    assert asw._auth_outage_spawn_gate(9, "campaign") == "auth-outage"
    assert asw._AUTH_CANARY_TOKEN is True


# ─── 24: MF-4 — per-site wiring certification ────────────────────────────────


def test_gate_and_record_wiring_all_sites():
    # Source-pin every inventory row: the gate + the record hook are wired at
    # the exact functions the plan names (a dropped hook silently defeats the
    # whole fix, so pin it at the source level).
    gate, record = "_auth_outage_spawn_gate", "_auth_outage_record_spawn"
    src = inspect.getsource
    # Row 1: crash-recovery arm (helper-internal gate + record).
    assert gate in src(asw._respawn) and record in src(asw._respawn)
    # Row 2: stalled arm — gate at the CALLER (fence), record in the helper.
    assert gate in src(asw._handle_stalled_respawn)
    assert record in src(asw._respawn_stalled_session)
    assert gate not in src(asw._respawn_stalled_session)  # MF-2: never helper-internal
    # Row 3: orphan sweep.
    assert gate in src(asw._respawn_orphan) and record in src(asw._respawn_orphan)
    # Row 4: infra-drain (one hook covers both callers).
    assert gate in src(asw._dispatch_infra_drain) and record in src(asw._dispatch_infra_drain)
    # Row 5: capacity retry.
    assert gate in src(asw._redrive_capacity_retry)
    assert record in src(asw._redrive_capacity_retry)
    # Row 6: campaign — gates at BOTH callers, record in the helper.
    assert gate in src(asw._campaign_escalate_stall)
    assert gate in src(asw._process_campaign_entry)
    assert record in src(asw._respawn_campaign)
    assert gate not in src(asw._respawn_campaign)  # §13.1: never helper-internal


class _FakeRes:
    returncode = 0
    stdout = "spawned session xyz\n"
    stderr = ""


def test_runtime_wiring_crash_and_campaign_record(watcher_roots, monkeypatch):
    recorded: list[tuple] = []
    monkeypatch.setattr(
        asw, "_auth_outage_record_spawn", lambda i, a, p: recorded.append((i, a, p))
    )
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **kw: _FakeRes())
    out = asw._respawn({"issue": 3, "spawned_at": 123.0}, dry_run=False)
    assert out == "spawned"
    assert recorded == [(3, "crash", 123.0)]
    ok = asw._respawn_campaign({"issue": 4, "spawned_at": 55.0}, dry_run=False)
    assert ok is True
    assert recorded[-1] == (4, "campaign", 55.0)


def test_runtime_wiring_campaign_caller_gate(watcher_roots, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(asw, "_auth_outage_spawn_gate", lambda *a, **kw: "auth-outage")
    monkeypatch.setattr(asw, "_stop_session", lambda *a, **kw: calls.append("stop") or True)
    monkeypatch.setattr(asw, "_respawn_campaign", lambda *a, **kw: calls.append("respawn") or True)
    monkeypatch.setattr(asw, "_post_campaign_marker", lambda *a, **kw: None)
    st = {"stalled_checks": 1, "alerted": True, "respawn_count": 0, "exhausted": False}
    asw._campaign_escalate_stall(9, {"happy_session_id": "sid-9"}, st, False, daemon_reachable=True)
    # The gate skipped the stop+respawn as a UNIT (never stop-then-not-respawn).
    assert calls == []


# ─── evidence enrichment — real body (fake only at the filesystem boundary) ──


def test_auth_outage_evidence_real_body(tmp_path, monkeypatch):
    from pathlib import Path as _P

    monkeypatch.setattr(_P, "home", lambda: tmp_path)
    # No ~/.happy/logs at all: degrades to churn-only, never raises.
    assert _REAL_EVIDENCE() == "churn-only"
    logs = tmp_path / ".happy" / "logs"
    logs.mkdir(parents=True)
    (logs / "a.log").write_text("boot ok\nall healthy\n")
    assert _REAL_EVIDENCE() == "churn-only"
    # A recognized auth signature in a newest-log tail is surfaced verbatim.
    (logs / "b.log").write_text("request failed: authentication_error (401)\n")
    assert _REAL_EVIDENCE() == "auth-string: authentication_error"


# ─── 25: MF-5 — fail-soft on every surface ───────────────────────────────────


def _boom(*a, **kw):
    raise RuntimeError("boom")


def test_fail_soft_every_surface(watcher_roots, monkeypatch, push_counter):
    now = time.time()
    # (i) gate / record / pass bodies: an internal exception never escapes,
    # and the gate FAILS OPEN (allows the spawn).
    with monkeypatch.context() as m:
        m.setattr(asw, "_load_auth_outage_state", _boom)
        assert asw._auth_outage_spawn_gate(1, "crash") is None
        asw._auth_outage_record_spawn(1, "crash", None)  # no raise
        asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())  # no raise
    # (ii) sidecar append raising: the tick survives.
    _write_state(watcher_roots / "reg", active=False, events=_freeze_events(now))
    with monkeypatch.context() as m:
        m.setattr(asw, "_append_auth_outage_sidecar", _boom)
        asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())  # no raise
    # (iii) push raising: the tick survives.
    _write_state(watcher_roots / "reg", active=False, events=_freeze_events(now))
    with monkeypatch.context() as m:
        m.setattr(asw, "_telegram_push", _boom)
        asw.auth_outage_pass(False, daemon_reachable=True, live_ids=set())  # no raise
    # (iv) with the gate's state read broken, spawns PROCEED (fail-open).
    with monkeypatch.context() as m:
        m.setattr(asw, "_load_auth_outage_state", _boom)
        m.setattr(asw.subprocess, "run", lambda *a, **kw: _FakeRes())
        assert asw._respawn({"issue": 2, "spawned_at": 1.0}, dry_run=False) == "spawned"
