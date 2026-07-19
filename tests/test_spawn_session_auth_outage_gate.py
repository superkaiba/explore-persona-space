"""Tests for the #1218 auth-outage dispatch hold in ``scripts/spawn_session.py``.

The gate (`auth_outage_dispatch_hold`) is a READ-ONLY mirror of the watcher's
#1027 `_auth_outage_spawn_gate` read side: while the watcher-written episode
singleton ``~/.eps-autonomous/auth-outage.json`` is ACTIVE and inside its
fail-open TTL, `spawn-issue --auto` suppresses with a rc-0 ``AUTH-OUTAGE
HELD`` line (recognized by :func:`spawn_session.spawn_output_suppressed`, so
every automated caller books nothing), manual spawns + `spawn-campaign`
warn-and-proceed, and the watcher's own canary probe passes via the
pre-spawn-persisted ``canary_pending`` claim.

What this pins (plan #1218 §6, items 1-18):

- Pure-gate fail-open cases (1-10): active hold, missing/garbled/non-dict
  state, falsy ``active``, missing/non-numeric ``started_ts``, TTL expiry,
  env-override + out-of-range fallback, the canary-claim bypass (fresh /
  other-issue / stale / malformed), the kill switch, and read-only-ness.
- Watcher-parity pin family (11a-11e): VALUE pins on the duplicated
  defaults, source-regex BOUNDS pins on the watcher's `_env_float` call
  sites, parse-SEMANTICS parity, kill-switch parity, and the FULL state-file
  path pin — a watcher-side retune or state-file relocation must fail a
  test, never silently de-sync the two gates.
- Sentinel + cmd-level (12-18): recognizer membership, the `--auto`
  suppression (exit 0, no daemon POST, no lease left behind, print↔recognizer
  loop closed on CAPTURED stdout), manual warn-and-proceed, the canary
  end-to-end replay, the exception-arm fail-open, the campaign
  warn-and-proceed, and the future-dated ``started_ts`` disposition.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import spawn_session as ss  # noqa: E402

# An issue number with no live worktree so `cmd_spawn_issue` resolves
# cwd=PROJECT_ROOT (which passes `_assert_spawn_cwd`) — the
# test_spawn_session_stop_takeover.py convention.
_FAKE_ISSUE = 9918

NOW = 1_800_000_000.0

_ENV_KNOBS = (
    "EPM_DISABLE_AUTH_OUTAGE_GUARD",
    "EPM_AUTH_OUTAGE_MAX_EPISODE_H",
    "EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN",
)


def _write_state(reg: Path, **fields) -> Path:
    """Write the auth-outage state singleton into ``reg`` and return its path."""
    path = reg / ss.AUTH_OUTAGE_STATE_FILENAME
    path.write_text(json.dumps(fields))
    return path


def _active_state(reg: Path, now: float = NOW, age_s: float = 60.0, **extra) -> Path:
    return _write_state(reg, active=True, started_ts=now - age_s, **extra)


@pytest.fixture
def reg(tmp_path, monkeypatch):
    """Isolated registry dir; env knobs cleared so the DEFAULTS bind."""
    for knob in _ENV_KNOBS:
        monkeypatch.delenv(knob, raising=False)
    reg_dir = tmp_path / "reg"
    reg_dir.mkdir()
    return reg_dir


@pytest.fixture
def cmd_registry(reg, monkeypatch):
    """The cmd-level harness: module-global registry dir pinned at the tmp
    ``reg`` (covers the takeover-sentinel glob, the dispatch-lease paths, AND
    the auth-outage state read — the test_spawn_session_stop_takeover.py
    pattern), takeover TTL env cleared."""
    monkeypatch.delenv("EPS_TAKEOVER_TTL_H", raising=False)
    monkeypatch.setattr(ss, "AUTONOMOUS_REGISTRY_DIR", reg)
    return reg


def _spawn_ns(*, auto: bool) -> argparse.Namespace:
    """A minimal `spawn-issue` Namespace covering every attribute
    `cmd_spawn_issue` reads before (and at) the auth-outage gate."""
    return argparse.Namespace(
        issue=_FAKE_ISSUE,
        auto=auto,
        initial_prompt=None,
        betas=None,
        model=None,
        effort=None,
    )


# ── pure-gate cases (plan §6 items 1-10) ────────────────────────────────────


def test_hold_when_episode_active(reg):
    _active_state(reg, age_s=60.0)
    reason = ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg)
    assert reason is not None
    assert "auth-outage episode ACTIVE" in reason
    from datetime import UTC, datetime

    started_iso = datetime.fromtimestamp(NOW - 60.0, tz=UTC).isoformat(timespec="seconds")
    assert started_iso in reason  # names the start time
    assert "fail-open TTL 6h" in reason  # names the TTL


def test_allow_when_no_state_file(reg):
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_allow_when_inactive(reg):
    _write_state(reg, active=False, started_ts=NOW - 60.0)
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_allow_when_garbled_json(reg):
    (reg / ss.AUTH_OUTAGE_STATE_FILENAME).write_text("{not json !!")
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_allow_when_non_dict_json(reg):
    (reg / ss.AUTH_OUTAGE_STATE_FILENAME).write_text('["active", true]')
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_allow_when_started_ts_missing_or_non_numeric(reg):
    _write_state(reg, active=True)  # missing started_ts
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None
    _write_state(reg, active=True, started_ts="yesterday")  # non-numeric
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_allow_past_ttl(reg):
    # Second fail-open layer: even a wedged watcher pass cannot suppress past
    # the 6h default TTL.
    _active_state(reg, age_s=6.05 * 3600)
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_ttl_env_override_and_out_of_range_fallback(reg, monkeypatch):
    _active_state(reg, age_s=2 * 3600)  # a 2h-old episode
    # In-bounds override: TTL=1h -> the 2h-old episode is expired -> allow.
    monkeypatch.setenv("EPM_AUTH_OUTAGE_MAX_EPISODE_H", "1")
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None
    # Out-of-range override (below lo=1.0): the DEFAULT 6h binds -> hold.
    monkeypatch.setenv("EPM_AUTH_OUTAGE_MAX_EPISODE_H", "0.2")
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is not None


def test_canary_pending_matching_issue_allows(reg):
    _active_state(reg, canary_pending={"issue": _FAKE_ISSUE, "arm": "crash", "ts": NOW - 60.0})
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_canary_pending_other_issue_holds(reg):
    _active_state(reg, canary_pending={"issue": _FAKE_ISSUE + 1, "arm": "crash", "ts": NOW - 60.0})
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is not None


def test_canary_pending_stale_holds(reg):
    # A claim older than the 30-min default window has expired — hold.
    _active_state(reg, canary_pending={"issue": _FAKE_ISSUE, "arm": "crash", "ts": NOW - 31 * 60.0})
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is not None


def test_canary_pending_malformed_holds(reg):
    _active_state(reg, canary_pending={"issue": _FAKE_ISSUE, "arm": "crash"})  # missing ts
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is not None
    _active_state(reg, canary_pending={"issue": _FAKE_ISSUE, "arm": "crash", "ts": "soon"})
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is not None


def test_kill_switch_allows(reg, monkeypatch):
    _active_state(reg)
    monkeypatch.setenv("EPM_DISABLE_AUTH_OUTAGE_GUARD", "1")
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None


def test_gate_never_writes_state(reg, monkeypatch):
    """READ-ONLY invariant: the state file's bytes are identical after every
    gate outcome exercised above (hold, canary allow, kill-switch allow)."""
    path = _active_state(
        reg, canary_pending={"issue": _FAKE_ISSUE + 1, "arm": "crash", "ts": NOW - 60.0}
    )
    before = path.read_bytes()
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is not None
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE + 1, now=NOW, registry_dir=reg) is None
    monkeypatch.setenv("EPM_DISABLE_AUTH_OUTAGE_GUARD", "1")
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None
    monkeypatch.delenv("EPM_DISABLE_AUTH_OUTAGE_GUARD")
    assert path.read_bytes() == before


# ── watcher-parity pin family (plan §6 item 11, Statistics-critic MF r1) ────


def _skip_if_env_tainted(*names: str) -> None:
    """The watcher module constants froze at import; a live env override makes
    the VALUE pins meaningless (delenv cannot undo an import-time read)."""
    import os

    for name in names:
        if name in os.environ:
            pytest.skip(f"env-tainted session: {name} is set")


def test_watcher_parity_ttl_value():
    # 11a: catches a watcher-side default retune (6.0 -> 4.0 h).
    _skip_if_env_tainted("EPM_AUTH_OUTAGE_MAX_EPISODE_H")
    assert ss.AUTH_OUTAGE_TTL_H_DEFAULT * 3600.0 == asw.AUTH_OUTAGE_MAX_EPISODE_S


def test_watcher_parity_canary_window_value():
    # 11a: the canary-window direction is the one that would wedge episodes
    # (the choke gate suppressing a watcher-authorized canary).
    _skip_if_env_tainted("EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN")
    assert ss.AUTH_OUTAGE_CANARY_MIN_DEFAULT * 60.0 == asw.AUTH_OUTAGE_CANARY_INTERVAL_S


def test_watcher_parity_bounds():
    # 11b: the watcher's bounds live only in its `_env_float` call-site kwargs
    # (no module constant) — pin them by source regex.
    src = inspect.getsource(asw)

    def call_site(env_name: str) -> tuple[float, float, float]:
        m = re.search(
            r'_env_float\(\s*"' + env_name + r'",\s*([\d.]+),\s*lo=([\d.]+),\s*hi=([\d.]+)\s*\)',
            src,
        )
        assert m is not None, f"watcher _env_float call site for {env_name} not found"
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    default, lo, hi = call_site("EPM_AUTH_OUTAGE_MAX_EPISODE_H")
    assert default == ss.AUTH_OUTAGE_TTL_H_DEFAULT
    assert (lo, hi) == ss.AUTH_OUTAGE_TTL_H_BOUNDS
    default, lo, hi = call_site("EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN")
    assert default == ss.AUTH_OUTAGE_CANARY_MIN_DEFAULT
    assert (lo, hi) == ss.AUTH_OUTAGE_CANARY_MIN_BOUNDS


@pytest.mark.parametrize("raw", [None, "12", "junk", "0.2", "99"])
def test_watcher_parity_env_parse(raw, monkeypatch):
    # 11c: parse-SEMANTICS parity — call the watcher's parse FUNCTION fresh
    # against the same env value (unset / in-bounds / garbled / out-of-range
    # low / out-of-range high).
    if raw is None:
        monkeypatch.delenv("EPM_AUTH_OUTAGE_MAX_EPISODE_H", raising=False)
        monkeypatch.delenv("EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN", raising=False)
    else:
        monkeypatch.setenv("EPM_AUTH_OUTAGE_MAX_EPISODE_H", raw)
        monkeypatch.setenv("EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN", raw)
    assert ss._auth_outage_ttl_s() == (
        asw._env_float("EPM_AUTH_OUTAGE_MAX_EPISODE_H", 6.0, lo=1.0, hi=48.0) * 3600
    )
    assert ss._auth_outage_canary_window_s() == (
        asw._env_float("EPM_AUTH_OUTAGE_CANARY_INTERVAL_MIN", 30.0, lo=10.0, hi=720.0) * 60
    )


@pytest.mark.parametrize("raw", [None, "1", "true", "yes", "TRUE", " yes ", "0", "junk"])
def test_watcher_parity_kill_switch(raw, monkeypatch):
    # 11d: the truthy set is a third duplicated surface — pin it against the
    # watcher's own pure env-reading function.
    if raw is None:
        monkeypatch.delenv("EPM_DISABLE_AUTH_OUTAGE_GUARD", raising=False)
    else:
        monkeypatch.setenv("EPM_DISABLE_AUTH_OUTAGE_GUARD", raw)
    assert ss._auth_outage_guard_enabled() == asw._auth_outage_enabled()


def test_watcher_parity_state_path():
    # 11e: FULL-path pin — a `.name`-only pin would stay green across a
    # watcher-side relocation of the singleton's parent dir while the choke
    # gate reads a nonexistent path (tests-green/gate-inert vacuity).
    assert (
        asw._auth_outage_state_path()
    ) == ss.AUTONOMOUS_REGISTRY_DIR / ss.AUTH_OUTAGE_STATE_FILENAME


# ── sentinel + cmd-level (plan §6 items 12-18) ──────────────────────────────


def test_sentinel_recognized_by_spawn_output_suppressed():
    out = f"{ss.AUTH_OUTAGE_HELD_SENTINEL} issue #{_FAKE_ISSUE}: episode ACTIVE; NOT spawning"
    assert ss.spawn_output_suppressed(out) == ss.AUTH_OUTAGE_HELD_SENTINEL
    assert ss.spawn_output_suppressed(f"Spawned session abc for issue #{_FAKE_ISSUE}") is None


def test_cmd_spawn_issue_auto_suppressed_exit0(cmd_registry, monkeypatch, capsys):
    # The gate must fire BEFORE the daemon POST and BEFORE lease acquisition
    # (pre-lease placement pin: a gate after the lease would leave a TTL-held
    # lease suppressing the watcher's canary for this issue).
    _active_state(cmd_registry, now=time.time())
    monkeypatch.setattr(ss, "post", lambda *a, **k: pytest.fail("daemon POST must not happen"))
    monkeypatch.setattr(
        ss,
        "acquire_dispatch_lease",
        lambda *a, **k: pytest.fail("dispatch lease must not be acquired"),
    )
    monkeypatch.setattr(
        ss, "_spawn_issue_session", lambda *a, **k: pytest.fail("spawn tail must not be reached")
    )
    ss.cmd_spawn_issue(_spawn_ns(auto=True))  # returns (exit 0), no SystemExit
    out = capsys.readouterr().out
    # Producer -> recognizer loop closed on the CAPTURED stdout (a hand-built
    # fixture string could drift from the actual print; #607 class) — this is
    # what pins f-string/recognizer format agreement end-to-end.
    assert ss.spawn_output_suppressed(out) == ss.AUTH_OUTAGE_HELD_SENTINEL
    # test_gate_before_lease, folded: no lease file was left behind.
    assert not ss.dispatch_lease_path(_FAKE_ISSUE).exists()


def test_cmd_spawn_issue_manual_warns_and_proceeds(cmd_registry, monkeypatch, capsys):
    _active_state(cmd_registry, now=time.time())
    reached: list[int] = []
    monkeypatch.setattr(
        ss, "_spawn_issue_session", lambda args, issue, *rest, **k: reached.append(issue)
    )
    ss.cmd_spawn_issue(_spawn_ns(auto=False))
    out = capsys.readouterr().out
    assert reached == [_FAKE_ISSUE]  # manual spawns are not gated
    assert "auth-outage episode ACTIVE" in out
    assert ss.spawn_output_suppressed(out) is None  # warn line is NOT a suppression


def test_cmd_spawn_issue_auto_canary_passes_end_to_end(cmd_registry, monkeypatch, capsys):
    # The watcher-canary replay at the choke point: the watcher persisted
    # canary_pending BEFORE shelling out to `spawn-issue --auto`, so the spawn
    # tail (daemon POST + registration) must be reached.
    now = time.time()
    _active_state(
        cmd_registry,
        now=now,
        canary_pending={"issue": _FAKE_ISSUE, "arm": "crash", "ts": now - 5.0},
    )
    reached: list[int] = []
    monkeypatch.setattr(
        ss, "_spawn_issue_session", lambda args, issue, *rest, **k: reached.append(issue)
    )
    ss.cmd_spawn_issue(_spawn_ns(auto=True))
    out = capsys.readouterr().out
    assert reached == [_FAKE_ISSUE]
    assert ss.spawn_output_suppressed(out) is None


def test_exception_arm_fails_open(reg, monkeypatch, capsys):
    # §5 "Fail-open everywhere": an internal error can never hold a dispatch.
    _active_state(reg)
    monkeypatch.setattr(
        ss, "_auth_outage_ttl_s", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is None
    err = capsys.readouterr().err
    assert "dispatch-hold check error (fail-open)" in err


def test_cmd_spawn_campaign_warns_and_proceeds(cmd_registry, monkeypatch, capsys):
    # §4.4: campaign CLI spawns are human/PM-driven off a fresh user approval
    # — warn-only, never suppress; the daemon POST is still reached.
    _active_state(cmd_registry, now=time.time())
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.get_task",
        lambda issue: {"frontmatter": {"kind": "campaign"}, "status": "approved"},
    )
    verified: list[str] = []
    monkeypatch.setattr(ss, "_verify_happy_patch_or_die", lambda context: verified.append(context))
    posts: list[str] = []

    def fake_post(endpoint, body):
        posts.append(endpoint)
        return {"success": True, "sessionId": "sid-campaign-test"}

    monkeypatch.setattr(ss, "post", fake_post)
    monkeypatch.setattr(ss, "_register_campaign_session", lambda *a, **k: None)
    ns = argparse.Namespace(
        issue=_FAKE_ISSUE,
        budget_gpu_hours=None,
        max_concurrent=None,
        per_child_cap=None,
        betas=None,
        model=None,
        effort=None,
    )
    ss.cmd_spawn_campaign(ns)
    out = capsys.readouterr().out
    assert "auth-outage episode ACTIVE" in out
    assert "may die on arrival; proceeding" in out
    assert posts == ["/spawn-session"]  # the POST was reached — never suppressed
    assert verified == ["spawn-campaign"]
    assert ss.spawn_output_suppressed(out) is None


def test_future_dated_started_ts_holds(reg):
    # §7 row 4: a future-dated started_ts HOLDS — deliberate watcher parity
    # (asw:6307 has identical arithmetic); diverging here would create two
    # gates disagreeing on one state file.
    _write_state(reg, active=True, started_ts=NOW + 3600.0)
    assert ss.auth_outage_dispatch_hold(_FAKE_ISSUE, now=NOW, registry_dir=reg) is not None
