"""Tests for the #1327 `unregister` subcommand in ``scripts/spawn_session.py``.

What this pins (the inverse of ``register-current``, for collision-yield and
deliberate-stop paths — replacing the hand-rolled #952 ``rm``):

1. **Sid-matched removal by default.** Without ``--force`` a registration file
   is removed IFF its recorded ``happy_session_id`` string-equals the caller's
   sid; a mismatch KEEPS the file (``KEPT-SID-MISMATCH``), so a yielding
   duplicate can never delete the true owner's entry.
2. **Fail toward keep.** Garbled / unreadable / missing-``happy_session_id``
   entries are KEPT without ``--force`` (``KEPT-UNREADABLE``); missing files
   are tolerated (``MISSING``, exit 0).
3. **Strict targeting.** The ``--issue`` form builds exact filenames; the scan
   form filters through the strict ``^(issue|manual-issue|campaign)-(\\d+)\\.json$``
   regex — takeover sentinels (``*.paused-takeover-*``) and non-registration
   siblings (``dispatch-lease-*``, ``campaign-watch-*``, ``pm-session.json``)
   are never removed by ANY invocation form.
4. **Flag surface.** ``--force`` requires ``--issue`` and excludes
   ``--session-id`` (cmd layer AND the pure helper); no selectors + failed
   ancestry inference exits loud; ``--kind`` narrows targeting.
5. **CLI wiring.** One end-to-end invocation through the REAL argparse parser
   (``spawn_session.main([...])``) asserting the ``KEPT-SID-MISMATCH``
   breadcrumb line on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spawn_session  # noqa: E402

_MY_SID = "happy-sess-1327-mine"
_OTHER_SID = "happy-sess-1327-owner"


def _write_entry(reg: Path, name: str, sid: str) -> Path:
    """A minimal registration entry carrying ``happy_session_id`` (the only
    field the sid match reads)."""
    path = reg / name
    path.write_text(json.dumps({"issue": 5, "happy_session_id": sid, "spawned_at": 1.0}))
    return path


def _ns(**overrides) -> argparse.Namespace:
    """A full `unregister` Namespace (every attribute cmd_unregister reads)."""
    ns = argparse.Namespace(issue=None, session_id=None, kind=None, force=False, reason=None)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ── 1. sid-matched removal (issue form) ──────────────────────────────────────


def test_issue_form_removes_only_matching_sid(tmp_path, monkeypatch):
    """Case 1: writer-produced entry for MY sid is removed; the other kinds'
    files recording OTHER sids are untouched."""
    # Construct the auto entry through the production writer itself (the
    # #1287 predicate-trace requirement — not hand-mocked JSON).
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    spawn_session._register_autonomous_session(5, _MY_SID, "/tmp/cwd", 100.0, force=True)
    manual = _write_entry(tmp_path, "manual-issue-5.json", _OTHER_SID)
    campaign = _write_entry(tmp_path, "campaign-5.json", _OTHER_SID)

    rows = spawn_session.unregister_paths(issue=5, session_id=_MY_SID, registry_dir=tmp_path)
    actions = {p.name: a for a, p, _ in rows}
    assert actions["issue-5.json"] == "removed"
    assert actions["manual-issue-5.json"] == "kept-sid-mismatch"
    assert actions["campaign-5.json"] == "kept-sid-mismatch"
    assert not (tmp_path / "issue-5.json").exists()
    assert manual.exists() and campaign.exists()


def test_sid_mismatch_keeps_file_content_unchanged(tmp_path):
    """Case 2: the owner's entry survives a wrong-sid removal byte-for-byte."""
    path = _write_entry(tmp_path, "issue-5.json", _OTHER_SID)
    before = path.read_text()
    rows = spawn_session.unregister_paths(issue=5, session_id=_MY_SID, registry_dir=tmp_path)
    assert [a for a, _, _ in rows if a == "removed"] == []
    assert ("kept-sid-mismatch", path) in [(a, p) for a, p, _ in rows]
    assert path.read_text() == before


def test_missing_files_tolerated(tmp_path):
    """Case 3: nothing on disk -> three `missing` rows, no exception."""
    rows = spawn_session.unregister_paths(issue=7, session_id=_MY_SID, registry_dir=tmp_path)
    assert [a for a, _, _ in rows] == ["missing", "missing", "missing"]


# ── 4/5. force + garbled entries ─────────────────────────────────────────────


def test_force_removes_all_kinds_regardless_of_sid(tmp_path):
    """Case 4: --force removes all three kind files incl. a garbled one."""
    _write_entry(tmp_path, "issue-5.json", _OTHER_SID)
    _write_entry(tmp_path, "manual-issue-5.json", _OTHER_SID)
    (tmp_path / "campaign-5.json").write_text("{not json")
    rows = spawn_session.unregister_paths(
        issue=5, session_id=None, force=True, registry_dir=tmp_path
    )
    assert [a for a, _, _ in rows] == ["removed", "removed", "removed"]
    assert list(tmp_path.glob("*.json")) == []


def test_garbled_without_force_kept(tmp_path):
    """Case 5: garbled JSON without --force -> kept-unreadable, file intact."""
    garbled = tmp_path / "issue-5.json"
    garbled.write_text("{not json")
    rows = spawn_session.unregister_paths(issue=5, session_id=_MY_SID, registry_dir=tmp_path)
    assert ("kept-unreadable", garbled) in [(a, p) for a, p, _ in rows]
    assert garbled.exists()


def test_missing_sid_field_without_force_kept(tmp_path):
    """An entry with no happy_session_id fails toward keep (never matches)."""
    path = tmp_path / "issue-5.json"
    path.write_text(json.dumps({"issue": 5}))
    rows = spawn_session.unregister_paths(issue=5, session_id=_MY_SID, registry_dir=tmp_path)
    assert ("kept-unreadable", path) in [(a, p) for a, p, _ in rows]
    assert path.exists()


def test_helper_force_without_issue_raises(tmp_path):
    """Critic concern 1: the pure helper asserts force-requires-issue itself."""
    with pytest.raises(ValueError, match="force=True requires issue"):
        spawn_session.unregister_paths(
            issue=None, session_id=None, force=True, registry_dir=tmp_path
        )


# ── 6/7. scan form + never-touch siblings ────────────────────────────────────


def test_scan_form_removes_matching_and_skips_siblings(tmp_path):
    """Case 6: scan removes MY entries across issues/kinds, silently skips
    other-sid registrations, and never scrapes non-registration siblings."""
    mine_a = _write_entry(tmp_path, "issue-5.json", _MY_SID)
    mine_b = _write_entry(tmp_path, "manual-issue-9.json", _MY_SID)
    other = _write_entry(tmp_path, "campaign-5.json", _OTHER_SID)
    lease = _write_entry(tmp_path, "dispatch-lease-7.json", _MY_SID)
    pm = _write_entry(tmp_path, "pm-session.json", _MY_SID)
    watch = _write_entry(tmp_path, "campaign-watch-5.json", _MY_SID)
    # The suffix-ends-in-.json takeover edge: full name fails the strict regex.
    sentinel_json = _write_entry(tmp_path, "issue-5.json.paused-takeover-x.json", _MY_SID)

    rows = spawn_session.unregister_paths(issue=None, session_id=_MY_SID, registry_dir=tmp_path)
    removed = {p.name for a, p, _ in rows if a == "removed"}
    assert removed == {"issue-5.json", "manual-issue-9.json"}
    assert not mine_a.exists() and not mine_b.exists()
    # Other-sid registration skipped SILENTLY in scan mode (no per-file noise).
    assert [a for a, p, _ in rows if p.name == "campaign-5.json"] == []
    for survivor in (other, lease, pm, watch, sentinel_json):
        assert survivor.exists(), survivor


def test_takeover_sentinel_survives_every_form(tmp_path):
    """Case 7: `issue-5.json.paused-takeover-abc` survives issue / scan /
    force forms (matching sid inside changes nothing)."""
    sentinel = tmp_path / "issue-5.json.paused-takeover-abc"
    sentinel.write_text(json.dumps({"happy_session_id": _MY_SID}))
    spawn_session.unregister_paths(issue=5, session_id=_MY_SID, registry_dir=tmp_path)
    spawn_session.unregister_paths(issue=None, session_id=_MY_SID, registry_dir=tmp_path)
    spawn_session.unregister_paths(issue=5, session_id=None, force=True, registry_dir=tmp_path)
    assert sentinel.exists()


def test_kind_narrows_targeting(tmp_path):
    """Case 8: --kind auto targets issue-N.json only."""
    auto = _write_entry(tmp_path, "issue-5.json", _MY_SID)
    manual = _write_entry(tmp_path, "manual-issue-5.json", _MY_SID)
    rows = spawn_session.unregister_paths(
        issue=5, session_id=_MY_SID, kind="auto", registry_dir=tmp_path
    )
    assert [(a, p.name) for a, p, _ in rows] == [("removed", "issue-5.json")]
    assert not auto.exists() and manual.exists()


# ── 9/10. cmd-level flag surface + ancestry inference ────────────────────────


def test_cmd_force_with_session_id_exits():
    with pytest.raises(SystemExit):
        spawn_session.cmd_unregister(_ns(issue=5, session_id=_MY_SID, force=True))


def test_cmd_force_without_issue_exits():
    with pytest.raises(SystemExit):
        spawn_session.cmd_unregister(_ns(force=True))


def test_cmd_no_selectors_failed_inference_exits(monkeypatch):
    """Case 9: no --issue/--session-id and the ancestry walk resolves nothing."""
    monkeypatch.setattr(spawn_session, "_live_children", lambda *a, **k: [])
    monkeypatch.setattr(spawn_session, "_ancestor_pids", lambda *a, **k: [123])
    with pytest.raises(SystemExit):
        spawn_session.cmd_unregister(_ns(issue=5))


def test_cmd_ancestry_inference_removes_own_entry(tmp_path, monkeypatch, capsys):
    """Case 10: happy path — inferred sid removes this session's own entry."""
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    _write_entry(tmp_path, "issue-5.json", _MY_SID)
    monkeypatch.setattr(
        spawn_session,
        "_live_children",
        lambda *a, **k: [{"pid": 4242, "happySessionId": _MY_SID}],
    )
    monkeypatch.setattr(spawn_session, "_ancestor_pids", lambda *a, **k: [4242])
    spawn_session.cmd_unregister(_ns(issue=5, reason="yield test"))
    out = capsys.readouterr().out
    assert "REMOVED" in out and "[reason: yield test]" in out
    assert not (tmp_path / "issue-5.json").exists()


# ── CLI end-to-end through the real argparse parser ──────────────────────────


def test_cli_end_to_end_sid_mismatch_breadcrumb(tmp_path, monkeypatch, capsys):
    """Critic concern 2: the REAL parser wires `unregister` to cmd_unregister,
    and a third-party sid mismatch prints the KEPT-SID-MISMATCH breadcrumb
    (exit 0 — the function returns normally)."""
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    _write_entry(tmp_path, "issue-5.json", _OTHER_SID)
    spawn_session.main(["unregister", "--issue", "5", "--session-id", "x"])
    out = capsys.readouterr().out
    assert "KEPT-SID-MISMATCH" in out
    assert "nothing removed" in out
    assert (tmp_path / "issue-5.json").exists()
