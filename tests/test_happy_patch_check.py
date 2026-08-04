"""Hermetic tests for the Happy injection-patch guard (task #726).

What this pins (the Step-5 test-verdict for #726):

A. ``_happy_patch_check.classify_patch`` returns the right state on each of the
   four synthetic daemon-file conditions (patched / reverted / drifted /
   missing).
B. ``spawn_session._verify_happy_patch_or_die`` raises ``SystemExit`` (with the
   actionable re-apply / restart commands) on reverted / drifted / missing-with-
   daemon-reachable, and returns cleanly on patched / missing-without-daemon.
   The two-step ``missing`` probe (consulting ``spawn_session.DAEMON_STATE``) is
   the #685-reintroduction guard.
C. Integration — each injection-dependent ENTRYPOINT (``cmd_spawn_issue --auto``,
   ``cmd_spawn_campaign``, ``cmd_spawn_pm`` override branch) fails loud BEFORE
   ``post()`` is reached on a bad daemon file (``post`` monkeypatched to a
   flag-if-reached stub). A wiring miss — guard defined but a call site forgotten
   or placed after ``post()`` — fails these.
D. Positive asymmetry — the deliberately-unguarded no-override ``cmd_spawn_pm``
   path does NOT raise from the guard on a reverted file.
E. Watcher — ``happy_patch_pass`` escalates a revert via a sidecar row under
   ``--dry-run`` (decides + logs, never mutates), is a clean no-op when patched,
   and is fail-soft when ``classify_patch`` raises.

All hermetic: synthetic ``.mjs`` files under ``tmp_path``; ``DAEMON_FILE`` /
``DAEMON_STATE`` / ``post`` monkeypatched. No real ``/usr/lib/``, no ``sudo``,
no ``npm``, no daemon, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# Bootstrap sys.path the same way the watcher / spawn-session tests do.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _happy_patch_check as hpc  # noqa: E402
import autonomous_session_watch as asw  # noqa: E402
import patch_happy_daemon  # noqa: E402
import spawn_session  # noqa: E402

# ── synthetic-file builders ──────────────────────────────────────────────────


def _write_patched(tmp_path: Path) -> Path:
    f = tmp_path / "index-q9G4ktSK.mjs"
    f.write_text(f"{hpc.SENTINEL}\n// some bundled daemon source\n", encoding="utf-8")
    return f


def _write_reverted(tmp_path: Path) -> Path:
    """Sentinel absent, but every PATCHES search-string present -> `reverted`."""
    f = tmp_path / "index-q9G4ktSK.mjs"
    body = "// bundled daemon source\n" + "\n".join(
        search for _name, search, _replace in patch_happy_daemon.PATCHES
    )
    f.write_text(body, encoding="utf-8")
    return f


def _write_drifted(tmp_path: Path) -> Path:
    """Sentinel absent AND >=1 PATCHES search-string missing -> `drifted`."""
    f = tmp_path / "index-q9G4ktSK.mjs"
    # Include all but the first patch site's search-string, so at least one
    # search-string no longer matches.
    body = "// upgraded daemon source\n" + "\n".join(
        search for _name, search, _replace in patch_happy_daemon.PATCHES[1:]
    )
    f.write_text(body, encoding="utf-8")
    return f


def _missing_path(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist.mjs"


# ── A. classify_patch unit behavior ──────────────────────────────────────────


def test_classify_patched(tmp_path):
    st = hpc.classify_patch(_write_patched(tmp_path))
    assert st.state == "patched"


def test_classify_reverted(tmp_path):
    st = hpc.classify_patch(_write_reverted(tmp_path))
    assert st.state == "reverted"


def test_classify_drifted(tmp_path):
    st = hpc.classify_patch(_write_drifted(tmp_path))
    assert st.state == "drifted"
    # The detail names which search-strings drifted (the first patch site).
    assert patch_happy_daemon.PATCHES[0][0] in st.detail


def test_classify_missing(tmp_path):
    st = hpc.classify_patch(_missing_path(tmp_path))
    assert st.state == "missing"


def test_patch_happy_daemon_shares_constants():
    """patch_happy_daemon imports SENTINEL / DAEMON_FILE from the helper —
    single source of truth, so the literal never drifts across consumers."""
    assert patch_happy_daemon.SENTINEL is hpc.SENTINEL
    assert patch_happy_daemon.DAEMON_FILE is hpc.DAEMON_FILE


# ── B. guard unit behavior ───────────────────────────────────────────────────


def test_guard_patched_no_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_patched(tmp_path))
    # No raise.
    spawn_session._verify_happy_patch_or_die(context="t")


def test_guard_reverted_raises_with_reapply_cmd(tmp_path, monkeypatch):
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_reverted(tmp_path))
    with pytest.raises(SystemExit) as ei:
        spawn_session._verify_happy_patch_or_die(context="t")
    msg = str(ei.value)
    assert hpc.REAPPLY_CMD in msg
    assert hpc.RESTART_CMD in msg
    assert "reverted" in msg


def test_guard_drifted_raises_with_manual_msg(tmp_path, monkeypatch):
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_drifted(tmp_path))
    with pytest.raises(SystemExit) as ei:
        spawn_session._verify_happy_patch_or_die(context="t")
    msg = str(ei.value)
    assert "drifted" in msg
    # Drifted message names manual PATCHES reconciliation, not a blind re-apply.
    assert "PATCHES" in msg


def test_guard_missing_no_daemon_state_warns(tmp_path, monkeypatch, capsys):
    """missing AND DAEMON_STATE absent -> Happy not installed -> WARN + proceed."""
    monkeypatch.setattr(hpc, "DAEMON_FILE", _missing_path(tmp_path))
    monkeypatch.setattr(spawn_session, "DAEMON_STATE", tmp_path / "no-daemon.json")
    # No raise.
    spawn_session._verify_happy_patch_or_die(context="t")
    assert "no Happy install detected" in capsys.readouterr().err


def test_guard_missing_with_daemon_state_raises(tmp_path, monkeypatch):
    """missing BUT DAEMON_STATE present (the post-`npm update happy` hash-rename
    state: daemon reachable, patch file moved) -> DIE loud. The
    #685-reintroduction guard."""
    monkeypatch.setattr(hpc, "DAEMON_FILE", _missing_path(tmp_path))
    daemon_state = tmp_path / "daemon.state.json"
    daemon_state.write_text(json.dumps({"httpPort": 12345}))
    monkeypatch.setattr(spawn_session, "DAEMON_STATE", daemon_state)
    with pytest.raises(SystemExit) as ei:
        spawn_session._verify_happy_patch_or_die(context="t")
    msg = str(ei.value)
    assert hpc.REAPPLY_CMD in msg
    assert "could not be" in msg or "reachable" in msg


# ── C. integration — guard reached BEFORE post() per entrypoint ──────────────


@pytest.fixture
def reverted_daemon(tmp_path, monkeypatch):
    """Point the helper at a reverted synthetic .mjs and trip a flag if post()
    is ever reached. Returns a 0-arg callable -> bool (was post reached?)."""
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_reverted(tmp_path))
    reached = {"post": False}

    def _fake_post(path, body):  # pragma: no cover - must never be reached
        reached["post"] = True
        return {"success": True, "sessionId": "sess-should-not-happen"}

    monkeypatch.setattr(spawn_session, "post", _fake_post)
    return lambda: reached["post"]


def test_cmd_spawn_issue_auto_dies_before_post(reverted_daemon, monkeypatch, tmp_path):
    # No worktree at the synthetic path -> cwd falls back to repo root (fine).
    monkeypatch.setattr(spawn_session, "WORKTREE_DIR", tmp_path / "no-worktrees")
    # #843: the --auto path now acquires a dispatch lease BEFORE the patch
    # verify; isolate the registry so the test never writes ~/.eps-autonomous.
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "registry")
    ns = argparse.Namespace(
        issue=999,
        auto=True,
        initial_prompt=None,
        model=None,
        betas=None,
        effort=None,
        auto_approve_gpu_hours=100.0,
    )
    with pytest.raises(SystemExit):
        spawn_session.cmd_spawn_issue(ns)
    assert reverted_daemon() is False


def test_cmd_spawn_campaign_dies_before_post(reverted_daemon, monkeypatch):
    # Point get_task at a synthetic approved campaign so validation passes and
    # the guard (placed AFTER validation) is the next thing reached.
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw,
        "get_task",
        lambda issue: {"frontmatter": {"kind": "campaign"}, "status": "approved"},
    )
    ns = argparse.Namespace(
        issue=999,
        model=None,
        betas=None,
        effort=None,
        budget_gpu_hours=None,
        max_concurrent=None,
        per_child_cap=None,
    )
    with pytest.raises(SystemExit):
        spawn_session.cmd_spawn_campaign(ns)
    assert reverted_daemon() is False


def test_cmd_spawn_pm_override_uses_native_fields_and_needs_no_patch(tmp_path, monkeypatch):
    """A PM model/effort override rides Happy's NATIVE spawn fields (#2054).

    Before #2054 the override was carried by the patched ``claudeArgs`` channel,
    so this path died on a reverted daemon. happy >= 1.2.0 accepts ``modelMode``
    / ``effortLevel`` natively, so the override no longer depends on the patch:
    a reverted daemon must NOT raise here, and the POST body must carry the
    native fields rather than ``claudeArgs``.
    """
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_reverted(tmp_path))
    captured: dict[str, object] = {}

    def _fake_post(path, body):
        captured.update(body)
        return {"success": True, "sessionId": "sess-pm-native"}

    monkeypatch.setattr(spawn_session, "post", _fake_post)
    monkeypatch.setattr(spawn_session, "_register_pm_session", lambda sid: None)

    ns = argparse.Namespace(model="opus-4-7", betas=None, effort="high")
    spawn_session.cmd_spawn_pm(ns)

    assert captured["modelMode"] == "opus-4-7"
    assert captured["effortLevel"] == "high"
    assert "claudeArgs" not in captured


# ── D. positive asymmetry — no-override PM path is unguarded ─────────────────


def test_cmd_spawn_pm_no_override_does_not_raise_on_revert(tmp_path, monkeypatch):
    """A no-override PM spawn injects nothing, so a reverted patch must NOT
    raise from the guard (the deliberate design asymmetry). post() is a benign
    success stub so the call completes."""
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_reverted(tmp_path))
    monkeypatch.setattr(
        spawn_session,
        "post",
        lambda path, body: {"success": True, "sessionId": "sess-pm-ok"},
    )
    # Neutralize the PM-session registry write (it would touch the real home dir).
    monkeypatch.setattr(spawn_session, "_register_pm_session", lambda sid: None)
    ns = argparse.Namespace(model=None, betas=None, effort=None)
    # No raise from the guard; the benign post stub lets the call complete.
    spawn_session.cmd_spawn_pm(ns)


# ── E. watcher pass ──────────────────────────────────────────────────────────


@pytest.fixture
def watcher_roots(tmp_path, monkeypatch):
    """Pin PROJECT_ROOT (sidecar sink) + AUTONOMOUS_REGISTRY_DIR (dedup state)
    + suppress real Telegram pushes so the pass is fully offline."""
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "reg")
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: False)
    return tmp_path


def _read_sidecar(root: Path) -> list[dict]:
    path = root / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_watcher_pass_escalates_on_revert_dry_run(watcher_roots, tmp_path, monkeypatch):
    """A reverted file under --dry-run decides + logs but writes NO sidecar row
    or state (dry-run has zero observational side-effects); a clean run (patched)
    is a no-op."""
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_reverted(tmp_path))
    asw.happy_patch_pass(dry_run=True)
    assert _read_sidecar(watcher_roots) == []
    assert not asw._happy_patch_state_path().is_file()

    # Patched -> clean no-op even live.
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_patched(tmp_path))
    asw.happy_patch_pass(dry_run=False)
    assert _read_sidecar(watcher_roots) == []


def test_watcher_pass_writes_sidecar_when_live(watcher_roots, tmp_path, monkeypatch):
    """A reverted file (dry_run=False) writes one escalate-only sidecar row and
    dedups within the episode."""
    monkeypatch.setattr(hpc, "DAEMON_FILE", _write_reverted(tmp_path))
    asw.happy_patch_pass(dry_run=False)
    rows = _read_sidecar(watcher_roots)
    assert len(rows) == 1
    assert rows[0]["band"] == "happy-patch"
    assert rows[0]["state"] == "reverted"
    assert rows[0]["reapply_cmd"] == hpc.REAPPLY_CMD
    # Second pass at the same state -> deduped, no new row.
    asw.happy_patch_pass(dry_run=False)
    assert len(_read_sidecar(watcher_roots)) == 1


def test_watcher_pass_missing_is_noop(watcher_roots, tmp_path, monkeypatch):
    """`missing` (no daemon file) is escalate-only-conservative: no sidecar row
    (the spawn-path guard owns the precise reachability disambiguation)."""
    monkeypatch.setattr(hpc, "DAEMON_FILE", _missing_path(tmp_path))
    asw.happy_patch_pass(dry_run=False)
    assert _read_sidecar(watcher_roots) == []


def test_watcher_pass_clean_when_classify_raises(watcher_roots, monkeypatch):
    """happy_patch_pass returns cleanly (no raise) when classify_patch raises —
    fail-soft behavior asserted by test, not just by convention."""

    def _boom(*a, **kw):
        raise RuntimeError("synthetic classify failure")

    monkeypatch.setattr(hpc, "classify_patch", _boom)
    # No raise; no sidecar row.
    asw.happy_patch_pass(dry_run=True)
    assert _read_sidecar(watcher_roots) == []
