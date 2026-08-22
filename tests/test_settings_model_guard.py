"""Settings model-id guard pass (#2129) — unit pins.

The pass AUTO-NORMALIZES the fleet-killing ``claude-fable-5[1m]``-class
model id (fable/mythos + a ``[1m]`` suffix; 1M-native models expose NO
``[1m]`` variant, so the id kills every subagent fleet-wide — #545) in the
global Claude settings files, degrading to ALERT-ONLY on every
unsafe-write case (unparseable JSON, failed post-conditions, a concurrent
write between read and replace).

Tests inject tmp settings files via the ``paths=`` param and monkeypatch
the module-level sidecar / state / backup-dir Path constants plus
``_telegram_push`` (the per-pass test-file convention, e.g.
``test_autonomous_watch_pending_call_dryrun.py``).
"""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402


@pytest.fixture()
def pushes(tmp_path, monkeypatch):
    """Redirect every durable guard channel to tmp_path; capture pushes."""
    sent: list[str] = []
    monkeypatch.setattr(asw, "SETTINGS_MODEL_GUARD_SIDECAR", tmp_path / "sidecar.jsonl")
    monkeypatch.setattr(asw, "SETTINGS_MODEL_GUARD_STATE", tmp_path / "state.json")
    monkeypatch.setattr(asw, "SETTINGS_MODEL_GUARD_BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: bool(sent.append(msg)) or True)
    monkeypatch.delenv("EPM_DISABLE_SETTINGS_MODEL_GUARD_PASS", raising=False)
    monkeypatch.delenv("EPM_SETTINGS_MODEL_GUARD_REALERT_H", raising=False)
    return sent


def _sidecar_rows(tmp_path: Path) -> list[dict]:
    p = tmp_path / "sidecar.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_normalizes_fable_1m_model_key(tmp_path, pushes):
    """The #545 shape: `"model": "claude-fable-5[1m]"` is normalized in
    place, formatting + sibling keys preserved, a backup exists, True."""
    settings = tmp_path / "settings.json"
    original = '{\n  "model": "claude-fable-5[1m]",\n  "other": 42\n}\n'
    settings.write_text(original)
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    new_raw = settings.read_text()
    data = json.loads(new_raw)
    assert data["model"] == "claude-fable-5"
    assert data["other"] == 42
    # Byte-exact format-preserving rewrite: ONLY the quoted id changed.
    assert new_raw == original.replace('"claude-fable-5[1m]"', '"claude-fable-5"')
    backups = list((tmp_path / "backups").glob("settings-model-guard-backup-*-settings.json"))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    rows = _sidecar_rows(tmp_path)
    assert rows and rows[-1]["action"] == "normalized" and rows[-1]["ok"] is True


def test_normalizes_mythos_and_nested_env_value(tmp_path, pushes):
    """Recursive-walk evidence: a mythos id nested under env.* is caught."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"CLAUDE_CODE_SUBAGENT_MODEL": "claude-mythos-5[1m]"}}))
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    data = json.loads(settings.read_text())
    assert data["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-mythos-5"


def test_opus_1m_never_touched(tmp_path, pushes):
    """Opus 4.5-4.8 legitimately takes [1m] — out of scope, NO write."""
    settings = tmp_path / "settings.json"
    original = json.dumps({"model": "claude-opus-4-8[1m]"})
    settings.write_text(original)
    assert asw.settings_model_guard_pass(False, paths=[settings]) is False
    assert settings.read_text() == original
    assert _sidecar_rows(tmp_path) == []
    assert pushes == []


def test_clean_file_noop(tmp_path, pushes):
    """A clean settings file: no write (bytes identical), no push, False."""
    settings = tmp_path / "settings.json"
    original = json.dumps({"model": "claude-fable-5", "env": {"X": "1"}})
    settings.write_text(original)
    assert asw.settings_model_guard_pass(False, paths=[settings]) is False
    assert settings.read_text() == original
    assert _sidecar_rows(tmp_path) == []
    assert pushes == []


def test_dry_run_detects_without_writing(tmp_path, pushes, capsys):
    """dry_run=True: detection fires (True + the dry-run sidecar report
    carrying action dry-run) but bytes stay unchanged and nothing durable
    is written — the sanctioned zero-write live-smoke shape."""
    settings = tmp_path / "settings.json"
    original = json.dumps({"model": "claude-fable-5[1m]"})
    settings.write_text(original)
    assert asw.settings_model_guard_pass(True, paths=[settings]) is True
    assert settings.read_text() == original  # bytes unchanged
    assert pushes == []  # no push
    assert not (tmp_path / "sidecar.jsonl").exists()  # zero-write dry-run
    assert not (tmp_path / "backups").exists()
    out = capsys.readouterr().out
    assert "settings-model-guard: [dry-run]" in out
    assert '"action": "dry-run"' in out  # the would-append sidecar row


def test_kill_switch_disables(tmp_path, pushes, monkeypatch):
    """EPM_DISABLE_SETTINGS_MODEL_GUARD_PASS=1 -> immediate False, no rows."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "claude-fable-5[1m]"}))
    monkeypatch.setenv("EPM_DISABLE_SETTINGS_MODEL_GUARD_PASS", "1")
    assert asw.settings_model_guard_pass(False, paths=[settings]) is False
    assert json.loads(settings.read_text())["model"] == "claude-fable-5[1m]"
    assert _sidecar_rows(tmp_path) == []
    assert pushes == []


def test_unparseable_json_alert_only(tmp_path, pushes):
    """Raw text matches the bad pattern but json.loads fails -> ALERT-ONLY
    (sidecar + push), never a write."""
    settings = tmp_path / "settings.json"
    original = '{"model": "claude-fable-5[1m]", broken'
    settings.write_text(original)
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    assert settings.read_text() == original  # bytes unchanged
    rows = _sidecar_rows(tmp_path)
    assert rows[-1]["action"] == "alert-only"
    assert rows[-1]["reason"] == "unparseable-json"
    assert rows[-1]["bad_values"] == ["claude-fable-5[1m]"]
    assert len(pushes) == 1  # push fired (first episode)


def test_alert_only_dedup_within_ttl(tmp_path, pushes):
    """Second invocation inside the same episode -> no second push; sidecar
    rows stay per-event."""
    settings = tmp_path / "settings.json"
    settings.write_text('{"model": "claude-fable-5[1m]", broken')
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    assert len(pushes) == 1
    assert len(_sidecar_rows(tmp_path)) == 2


def test_missing_files_silent(tmp_path, pushes):
    """Both paths absent (settings.local.json commonly is) -> silent skip."""
    a = tmp_path / "settings.json"
    b = tmp_path / "settings.local.json"
    assert asw.settings_model_guard_pass(False, paths=[a, b]) is False
    assert _sidecar_rows(tmp_path) == []
    assert pushes == []


def test_docstring_lint_stays_green():
    """The docstring header digit (38), the numbered inventory, and the
    live main() *_pass set must reconcile after adding pass 38."""
    import workflow_lint

    assert workflow_lint.check_asw_docstring_pass_count() == []


def test_concurrent_write_guard_skips(tmp_path, pushes):
    """The critic-blocker fix: a write landing between the pass's read and
    the replace is never destroyed — stale old_raw -> (False,
    "concurrent-write"), file untouched, NO backup written."""
    settings = tmp_path / "settings.json"
    current = json.dumps({"model": "claude-fable-5[1m]", "n": 2})
    settings.write_text(current)
    stale_old_raw = json.dumps({"model": "claude-fable-5[1m]", "n": 1})
    ok, reason = asw._apply_settings_normalization(
        settings, stale_old_raw, stale_old_raw.replace("[1m]", "")
    )
    assert (ok, reason) == (False, "concurrent-write")
    assert settings.read_text() == current
    assert not (tmp_path / "backups").exists()


def test_escaped_value_postcondition_alert_only(tmp_path, pushes):
    r"""A \uXXXX-escaped bad value fails SAFE: detected in parsed form, the
    raw quoted-string replacement no-ops, post-condition (ii) fails ->
    bytes unchanged, alert-only."""
    settings = tmp_path / "settings.json"
    original = '{"model": "claude-fable-5\\u005b1m]"}'
    assert json.loads(original)["model"] == "claude-fable-5[1m]"  # fixture sanity
    settings.write_text(original)
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    assert settings.read_text() == original  # no write
    rows = _sidecar_rows(tmp_path)
    assert rows[-1]["action"] == "alert-only"
    assert rows[-1]["reason"] == "postcondition-failed"


def test_normalized_push_deduped_within_episode(tmp_path, pushes):
    """Two normalizations of the SAME re-planted value within the TTL ->
    one push, two sidecar rows (the /model re-plant shape)."""
    settings = tmp_path / "settings.json"
    bad = json.dumps({"model": "claude-fable-5[1m]"})
    settings.write_text(bad)
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    settings.write_text(bad)  # re-plant within the 24h episode
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    assert len(pushes) == 1
    assert [r["action"] for r in _sidecar_rows(tmp_path)] == ["normalized", "normalized"]


def test_version_less_id_normalized(tmp_path, pushes):
    """`claude-fable[1m]` (no version segment) is covered by the optional
    version group in the detection regex."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "claude-fable[1m]"}))
    assert asw.settings_model_guard_pass(False, paths=[settings]) is True
    assert json.loads(settings.read_text())["model"] == "claude-fable"


def test_file_mode_preserved(tmp_path, pushes):
    """The tmp file is chmod'd to the ORIGINAL file's mode before
    os.replace — a default tmp mode must never leak onto settings.json.
    Checked at 0o644 (the plan's literal case) AND 0o600 (distinct from
    the umask default, so a missing chmod cannot pass vacuously)."""
    for mode in (0o644, 0o600):
        settings = tmp_path / f"settings-{mode:o}.json"
        settings.write_text(json.dumps({"model": "claude-fable-5[1m]"}))
        os.chmod(settings, mode)
        assert asw.settings_model_guard_pass(False, paths=[settings]) is True
        assert stat.S_IMODE(settings.stat().st_mode) == mode
        assert json.loads(settings.read_text())["model"] == "claude-fable-5"
