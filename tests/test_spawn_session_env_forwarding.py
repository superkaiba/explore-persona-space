"""Tests for the #2110 settings-env spawn-payload forwarding in
``scripts/spawn_session.py``.

Harness-critical ``~/.claude/settings.json`` ``env`` overrides
(``CLAUDE_CODE_AUTO_COMPACT_WINDOW``, ``CLAUDE_CODE_SUBAGENT_MODEL``, ...)
bind ONLY via the exec-time process environment, and daemon-spawned sessions
inherit the DAEMON's env — so before #2110 a settings-env edit reached new
spawns only after a Happy-daemon restart. ``_settings_env_overrides`` +
``_merge_settings_env`` forward the ``CLAUDE_CODE_*`` subset through the
spawn payload's ``environmentVariables`` (which the daemon merges OVER its
own env into the child), removing the daemon-vintage dependency.

Two layers:

- Helper-unit tests (plan #2110 items (a)-(f)): key filter, explicit-key
  precedence, fail-soft on missing/corrupt/non-dict settings, str()
  coercion (the daemon's zod schema is ``z.record(z.string(), z.string())``),
  bare-body dict creation, and the ``${``-value guard (the daemon's
  ``expandEnvironmentVariables`` rejects the WHOLE spawn on an unresolved
  ``${...}`` reference).
- Per-path composed-body tests (critic MF-1 — AC1 as a gate, not prose):
  every ``POST /spawn-session`` body composed by spawn_session.py carries
  the forwarded settings keys AND retains its explicit payload keys, across
  all THREE composition paths (spawn-pm, spawn-issue prompt + bare branches,
  spawn-campaign). A regressed/missed call site fails a test instead of
  silently dropping env delivery — the exact silent-absence signature of
  the #2110 incident.
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

import spawn_session as ss  # noqa: E402

# An issue number with no live worktree so cwd resolves to PROJECT_ROOT
# (the test_spawn_session_stop_takeover.py convention).
_FAKE_ISSUE = 9917

# What the fixture settings file carries: two forwardable keys (one a raw JSON
# number, pinning the str() coercion end-to-end), one non-CLAUDE_CODE_ key
# that must NOT be forwarded, and one ${}-bearing key that must be skipped.
_SETTINGS_ENV = {
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": 1000000,
    "CLAUDE_CODE_SUBAGENT_MODEL": "claude-fable-5",
    "NOT_FORWARDED_VAR": "x",
    "CLAUDE_CODE_REF_BEARING": "${HOME}/x",
}
_EXPECTED_FORWARDED = {
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000",
    "CLAUDE_CODE_SUBAGENT_MODEL": "claude-fable-5",
}


def _write_settings(tmp_path: Path, payload) -> Path:
    """Write ``payload`` (dict -> JSON, str -> verbatim) as a settings.json."""
    path = tmp_path / "settings.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def _patch_settings(monkeypatch, tmp_path: Path, payload=None) -> Path:
    """Point the module's settings-path constant at a tmp settings file."""
    path = _write_settings(tmp_path, {"env": _SETTINGS_ENV} if payload is None else payload)
    monkeypatch.setattr(ss, "USER_SETTINGS_JSON", path)
    return path


# ─── helper-unit tests: _settings_env_overrides ─────────────────────────────


def test_only_claude_code_keys_forwarded(tmp_path):
    """(a) Only CLAUDE_CODE_* keys pass the filter; others are dropped."""
    path = _write_settings(
        tmp_path,
        {
            "env": {
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000",
                "MAX_THINKING_TOKENS": "31999",
                "EPM_SOMETHING": "1",
            }
        },
    )
    assert ss._settings_env_overrides(path) == {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000"}


def test_values_coerced_to_str(tmp_path):
    """(d) Raw JSON numbers/bools become strings — the daemon's zod schema is
    z.record(z.string(), z.string()), so a non-string value errors the spawn."""
    path = _write_settings(
        tmp_path,
        {"env": {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": 1000000, "CLAUDE_CODE_FLAG": True}},
    )
    out = ss._settings_env_overrides(path)
    assert out == {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000", "CLAUDE_CODE_FLAG": "True"}
    assert all(isinstance(v, str) for v in out.values())


def test_missing_file_returns_empty(tmp_path):
    """(c) Missing settings file -> {} silently, no raise."""
    assert ss._settings_env_overrides(tmp_path / "nope.json") == {}


def test_corrupt_json_returns_empty_with_note(tmp_path, capsys):
    """(c) Unparseable JSON -> {} plus a one-line stderr note, no raise."""
    path = _write_settings(tmp_path, "{not json")
    assert ss._settings_env_overrides(path) == {}
    assert "not valid JSON" in capsys.readouterr().err


def test_non_dict_top_level_returns_empty_with_note(tmp_path, capsys):
    """(c) A non-object settings top level -> {} plus a stderr note, no raise."""
    path = _write_settings(tmp_path, [1, 2, 3])
    assert ss._settings_env_overrides(path) == {}
    assert "not an object" in capsys.readouterr().err


def test_non_dict_env_returns_empty_with_note(tmp_path, capsys):
    """(c) A non-dict `env` block -> {} plus a stderr note, no raise."""
    path = _write_settings(tmp_path, {"env": "CLAUDE_CODE_X=1"})
    assert ss._settings_env_overrides(path) == {}
    assert "'env' is str" in capsys.readouterr().err


def test_absent_env_block_returns_empty_silently(tmp_path, capsys):
    """(c) No `env` block at all is normal, not corrupt: {} with NO note."""
    path = _write_settings(tmp_path, {"model": "opus"})
    assert ss._settings_env_overrides(path) == {}
    assert capsys.readouterr().err == ""


def test_ref_bearing_value_skipped_with_note(tmp_path, capsys):
    """(f) A ${...}-bearing value is excluded (the daemon's
    expandEnvironmentVariables rejects the WHOLE spawn on an unresolved
    reference); sibling keys still forward."""
    path = _write_settings(
        tmp_path,
        {"env": {"CLAUDE_CODE_REF_BEARING": "${HOME}/x", "CLAUDE_CODE_OK": "1"}},
    )
    assert ss._settings_env_overrides(path) == {"CLAUDE_CODE_OK": "1"}
    err = capsys.readouterr().err
    assert "CLAUDE_CODE_REF_BEARING" in err and "${" in err


# ─── helper-unit tests: _merge_settings_env ──────────────────────────────────


def test_merge_explicit_payload_keys_win(monkeypatch, tmp_path):
    """(b) Settings entries never clobber keys already in the payload."""
    _patch_settings(
        monkeypatch,
        tmp_path,
        {"env": {"CLAUDE_CODE_SUBAGENT_MODEL": "settings-model", "CLAUDE_CODE_NEW": "1"}},
    )
    body: dict[str, object] = {
        "environmentVariables": {
            "HAPPY_INITIAL_PROMPT": "/issue 1",
            "CLAUDE_CODE_SUBAGENT_MODEL": "explicit-model",
        }
    }
    ss._merge_settings_env(body)
    env = body["environmentVariables"]
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "explicit-model"  # explicit wins
    assert env["CLAUDE_CODE_NEW"] == "1"  # non-colliding key forwarded
    assert env["HAPPY_INITIAL_PROMPT"] == "/issue 1"  # untouched


def test_merge_bare_body_gains_dict(monkeypatch, tmp_path):
    """(e) A body with no environmentVariables gains the forwarded dict."""
    _patch_settings(monkeypatch, tmp_path)
    body: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._merge_settings_env(body)
    assert body["environmentVariables"] == _EXPECTED_FORWARDED


def test_merge_no_overrides_leaves_body_untouched(monkeypatch, tmp_path):
    """Nothing to forward -> the bare body stays WITHOUT an
    environmentVariables field (no empty-dict schema noise)."""
    _patch_settings(monkeypatch, tmp_path, {"env": {"OTHER": "x"}})
    body: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._merge_settings_env(body)
    assert "environmentVariables" not in body


# ─── per-path composed-body tests (critic MF-1) ──────────────────────────────


@pytest.fixture
def captured_post(monkeypatch):
    """Capture every daemon POST; return a successful spawn response."""
    calls: list[tuple[str, dict]] = []

    def fake_post(path: str, body: dict):
        calls.append((path, body))
        return {"success": True, "sessionId": "test-session-id"}

    monkeypatch.setattr(ss, "post", fake_post)
    return calls


def _spawn_body(calls) -> dict:
    spawns = [body for path, body in calls if path == "/spawn-session"]
    assert len(spawns) == 1, f"expected exactly one /spawn-session POST, got {calls}"
    return spawns[0]


def _assert_forwarded(env: dict) -> None:
    for key, value in _EXPECTED_FORWARDED.items():
        assert env.get(key) == value, f"forwarded settings key {key} missing/wrong: {env}"
    assert "NOT_FORWARDED_VAR" not in env
    assert "CLAUDE_CODE_REF_BEARING" not in env  # ${}-guard holds end-to-end


def test_spawn_pm_body_carries_settings_env(monkeypatch, tmp_path, captured_post):
    _patch_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(ss, "_register_pm_session", lambda sid: None)
    args = argparse.Namespace(model=None, betas=None, effort=None)
    ss.cmd_spawn_pm(args)
    body = _spawn_body(captured_post)
    _assert_forwarded(body["environmentVariables"])


def test_spawn_issue_prompt_branch_carries_settings_env(monkeypatch, tmp_path, captured_post):
    """The prompt-bearing branch: forwarded keys present AND the explicit
    HAPPY_INITIAL_* / EPM_* payload keys retained."""
    _patch_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(ss, "_verify_happy_patch_or_die", lambda **kw: None)
    args = argparse.Namespace(
        model=None,
        betas=None,
        effort=None,
        auto=False,  # bespoke --initial-prompt shape: one-shot, no registration seam
        auto_approve_gpu_hours=100.0,
    )
    body_in: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._spawn_issue_session(
        args, _FAKE_ISSUE, "<repo root>", body_in, f"/issue {_FAKE_ISSUE}", [], [], Path("/x")
    )
    env = _spawn_body(captured_post)["environmentVariables"]
    _assert_forwarded(env)
    assert env["HAPPY_INITIAL_PROMPT"] == f"/issue {_FAKE_ISSUE}"
    assert env["HAPPY_INITIAL_MODE"] == "bypassPermissions"
    assert env["EPM_AUTONOMOUS_SESSION"] == "1"
    assert env["EPM_PLAN_AUTOAPPROVE_GPU_HOURS"] == "100.0"


def test_spawn_issue_bare_branch_carries_settings_env(monkeypatch, tmp_path, captured_post):
    """The bare (no-prompt) branch set no environmentVariables at all before
    #2110 — it now gains the forwarded dict."""
    _patch_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(ss, "_register_manual_session", lambda *a: None)
    args = argparse.Namespace(
        model=None, betas=None, effort=None, auto=False, auto_approve_gpu_hours=100.0
    )
    body_in: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._spawn_issue_session(args, _FAKE_ISSUE, "<repo root>", body_in, None, [], [], Path("/x"))
    env = _spawn_body(captured_post)["environmentVariables"]
    _assert_forwarded(env)
    assert "HAPPY_INITIAL_PROMPT" not in env  # bare branch injects no prompt


def test_spawn_campaign_body_carries_settings_env(monkeypatch, tmp_path, captured_post):
    _patch_settings(monkeypatch, tmp_path)
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw,
        "get_task",
        lambda n: {"frontmatter": {"kind": "campaign"}, "status": "approved"},
    )
    monkeypatch.setattr(ss, "auth_outage_dispatch_hold", lambda issue: None)
    monkeypatch.setattr(ss, "_verify_happy_patch_or_die", lambda **kw: None)
    monkeypatch.setattr(ss, "_register_campaign_session", lambda *a, **k: None)
    args = argparse.Namespace(
        issue=_FAKE_ISSUE,
        budget_gpu_hours=None,
        max_concurrent=None,
        per_child_cap=None,
        model=None,
        betas=None,
        effort=None,
    )
    ss.cmd_spawn_campaign(args)
    env = _spawn_body(captured_post)["environmentVariables"]
    _assert_forwarded(env)
    assert env["HAPPY_INITIAL_PROMPT"] == f"/campaign {_FAKE_ISSUE}"
    assert env["EPM_CAMPAIGN_SESSION"] == "1"
