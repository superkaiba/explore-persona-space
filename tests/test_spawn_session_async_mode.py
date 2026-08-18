"""Tests for the mission-control rung-0 async session mode in
``scripts/spawn_session.py`` (CONTRACTS §2 / §2.2).

What this pins:

- ``spawn-issue --auto --session-mode async`` exports ``EPM_ASYNC_SESSION=1``
  ALONGSIDE the legacy autonomous env, records ``session_mode: "async"`` in
  the crash-recovery registry entry, and posts the durable
  ``epm:session-mode`` marker.
- ``--session-mode async`` WITHOUT ``--auto`` is refused (SystemExit).
- Fresh-spawn default resolution: explicit flag > registry entry > durable
  marker > ``~/.eps-autonomous/spawn-defaults.json`` config default (scoped
  by ``kind_scope`` AND the ``min_task_id`` id cutoff; fail-soft on any
  malformed field) > ``EPM_SPAWN_DEFAULT_SESSION_MODE`` (``kind:
  experiment`` ONLY) > legacy auto.
- ``_record_session_mode_marker`` idempotence (matching newest marker posts
  nothing; auto-with-no-history posts nothing; the explicit downgrade posts).
- NO-FLAGS REGRESSION: with ``--session-mode`` omitted and every resolution
  link empty, the composed spawn body's env-var KEY SET and the registry
  entry are byte-identical to the pre-rung-0 legacy forms (no
  ``EPM_ASYNC_SESSION``, no ``session_mode`` field, no marker post).
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
# (the test_spawn_session_env_forwarding.py convention).
_FAKE_ISSUE = 9917


def _issue_args(*, auto: bool = True, session_mode: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        model=None,
        betas=None,
        effort=None,
        auto=auto,
        auto_approve_gpu_hours=100.0,
        session_mode=session_mode,
        initial_prompt=None,
        issue=_FAKE_ISSUE,
    )


@pytest.fixture
def captured_post(monkeypatch):
    """Capture every daemon POST; return a successful spawn response."""
    calls: list[tuple[str, dict]] = []

    def fake_post(path: str, body: dict):
        calls.append((path, body))
        return {"success": True, "sessionId": "test-session-id"}

    monkeypatch.setattr(ss, "post", fake_post)
    return calls


@pytest.fixture
def isolated_spawn(monkeypatch, tmp_path):
    """Hermetic seams for `_spawn_issue_session`: no daemon patch check, no
    real registry/marker IO, empty settings env, and capture dicts for the
    registration + marker calls. Returns (registrations, marker_posts)."""
    monkeypatch.setattr(ss, "_verify_happy_patch_or_die", lambda **kw: None)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {}}))
    monkeypatch.setattr(ss, "USER_SETTINGS_JSON", settings)
    monkeypatch.setattr(ss, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "reg")
    monkeypatch.setattr(ss, "list_events", lambda issue: [])
    monkeypatch.delenv("EPM_SPAWN_DEFAULT_SESSION_MODE", raising=False)

    registrations: list[dict] = []

    def fake_register(issue, sid, cwd, cap, **kw):
        registrations.append({"issue": issue, "sid": sid, "cwd": cwd, "cap": cap, **kw})

    monkeypatch.setattr(ss, "_register_autonomous_session", fake_register)

    marker_posts: list[tuple[int, str]] = []
    monkeypatch.setattr(
        ss, "_record_session_mode_marker", lambda issue, mode: marker_posts.append((issue, mode))
    )
    return registrations, marker_posts


def _spawn_env(calls) -> dict:
    spawns = [body for path, body in calls if path == "/spawn-session"]
    assert len(spawns) == 1, f"expected exactly one /spawn-session POST, got {calls}"
    env = spawns[0]["environmentVariables"]
    assert isinstance(env, dict)
    return env


# ─── async spawn: env + registry + marker ────────────────────────────────────


def test_async_spawn_exports_both_env_vars(captured_post, isolated_spawn):
    registrations, marker_posts = isolated_spawn
    args = _issue_args(session_mode="async")
    body: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._spawn_issue_session(
        args, _FAKE_ISSUE, "<repo root>", body, f"/issue {_FAKE_ISSUE}", [], [], Path("/x")
    )
    env = _spawn_env(captured_post)
    # CONTRACTS §2: BOTH vars — every autonomous behavior fires unchanged.
    assert env["EPM_ASYNC_SESSION"] == "1"
    assert env["EPM_AUTONOMOUS_SESSION"] == "1"
    assert registrations and registrations[0]["session_mode"] == "async"
    assert marker_posts == [(_FAKE_ISSUE, "async")]


def test_legacy_auto_spawn_env_and_registry_byte_identical(captured_post, isolated_spawn):
    """No --session-mode, every resolution link empty: the env-var key set is
    EXACTLY the legacy set, the registration carries session_mode=None (the
    field is then not written), and no marker posts."""
    registrations, marker_posts = isolated_spawn
    args = _issue_args(session_mode=None)
    body: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._spawn_issue_session(
        args, _FAKE_ISSUE, "<repo root>", body, f"/issue {_FAKE_ISSUE}", [], [], Path("/x")
    )
    env = _spawn_env(captured_post)
    assert set(env) == {
        "HAPPY_INITIAL_PROMPT",
        "HAPPY_INITIAL_MODE",
        "EPM_AUTONOMOUS_SESSION",
        "EPM_PLAN_AUTOAPPROVE_GPU_HOURS",
        "HAPPY_AUTOMATED_SESSION",  # stamped by _merge_settings_env on every path
    }
    assert registrations and registrations[0]["session_mode"] is None
    assert marker_posts == []


def test_explicit_auto_flag_posts_downgrade_marker(captured_post, isolated_spawn):
    """--session-mode auto (EXPLICIT) routes through the downgrade-record
    call so a prior async marker cannot keep resolving async (newest wins).
    The helper itself decides whether a row is actually posted."""
    _registrations, marker_posts = isolated_spawn
    args = _issue_args(session_mode="auto")
    body: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._spawn_issue_session(
        args, _FAKE_ISSUE, "<repo root>", body, f"/issue {_FAKE_ISSUE}", [], [], Path("/x")
    )
    env = _spawn_env(captured_post)
    assert "EPM_ASYNC_SESSION" not in env
    assert marker_posts == [(_FAKE_ISSUE, "auto")]


def test_bespoke_prompt_stays_legacy_even_with_async_registry(
    captured_post, isolated_spawn, monkeypatch, tmp_path
):
    """A bespoke --initial-prompt (auto=False) one-shot never resolves async,
    even when the registry says async: session_mode is pinned 'auto'."""
    reg_dir = tmp_path / "reg"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / f"issue-{_FAKE_ISSUE}.json").write_text(json.dumps({"session_mode": "async"}))
    args = _issue_args(auto=False, session_mode=None)
    body: dict[str, object] = {"directory": "/x", "agent": "claude"}
    ss._spawn_issue_session(
        args, _FAKE_ISSUE, "<repo root>", body, "do a thing", [], [], Path("/x")
    )
    env = _spawn_env(captured_post)
    assert "EPM_ASYNC_SESSION" not in env


# ─── --session-mode async requires --auto ────────────────────────────────────


def test_session_mode_async_without_auto_refused(monkeypatch):
    args = _issue_args(auto=False, session_mode="async")
    with pytest.raises(SystemExit) as exc:
        ss.cmd_spawn_issue(args)
    assert "requires --auto" in str(exc.value)


def test_cli_parser_accepts_session_mode_choices(monkeypatch):
    """The spawn-issue argparser takes --session-mode {auto,async} and
    defaults to None (resolution-chain territory, not a hard 'auto'). The
    parser is built inside main(); capture the parsed namespace by patching
    the dispatch target (set_defaults binds the module global at build time,
    AFTER the patch)."""
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(ss, "cmd_spawn_issue", lambda ns: seen.append(ns))
    ss.main(["spawn-issue", "--issue", "1", "--auto", "--session-mode", "async"])
    ss.main(["spawn-issue", "--issue", "1", "--auto"])
    assert seen[0].session_mode == "async"
    assert seen[1].session_mode is None


# ─── fresh-spawn default resolution chain ────────────────────────────────────


@pytest.fixture
def chain_env(monkeypatch, tmp_path):
    """Empty every resolution link; individual tests then fill links in."""
    monkeypatch.setattr(ss, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "reg")
    monkeypatch.setattr(ss, "list_events", lambda issue: [])
    monkeypatch.delenv("EPM_SPAWN_DEFAULT_SESSION_MODE", raising=False)
    return tmp_path / "reg"


def _mode_marker(mode: str) -> dict:
    return {
        "kind": ss.SESSION_MODE_KIND,
        "ts": "2026-08-17T00:00:00Z",
        "note": json.dumps({"mode": mode}),
    }


def test_resolution_explicit_flag_wins(chain_env, monkeypatch):
    reg = chain_env
    reg.mkdir(parents=True, exist_ok=True)
    (reg / f"issue-{_FAKE_ISSUE}.json").write_text(json.dumps({"session_mode": "auto"}))
    args = _issue_args(session_mode="async")
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "async"


def test_resolution_registry_beats_marker(chain_env, monkeypatch):
    reg = chain_env
    reg.mkdir(parents=True, exist_ok=True)
    (reg / f"issue-{_FAKE_ISSUE}.json").write_text(json.dumps({"session_mode": "auto"}))
    monkeypatch.setattr(ss, "list_events", lambda issue: [_mode_marker("async")])
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


def test_resolution_marker_survives_registry_gc(chain_env, monkeypatch):
    """CRITICAL fixture (CONTRACTS §2.2): registry entry DELETED (terminal
    GC), durable marker present -> async still resolves."""
    monkeypatch.setattr(ss, "list_events", lambda issue: [_mode_marker("async")])
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "async"


def test_resolution_newest_marker_wins(chain_env, monkeypatch):
    monkeypatch.setattr(
        ss, "list_events", lambda issue: [_mode_marker("async"), _mode_marker("auto")]
    )
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("experiment", "async"), ("infra", "auto"), (None, "auto")],
)
def test_env_default_scoped_to_experiment_kind(chain_env, monkeypatch, kind, expected):
    monkeypatch.setenv("EPM_SPAWN_DEFAULT_SESSION_MODE", "async")
    monkeypatch.setattr(
        ss, "get_task", lambda issue: {"frontmatter": {"kind": kind} if kind else {}}
    )
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == expected


def test_env_default_unset_resolves_legacy_auto(chain_env, monkeypatch):
    """Also the config-file NO-FILE regression: chain_env's registry dir has
    no spawn-defaults.json, so with every other link empty this pins the
    byte-identical legacy resolution (the full-spawn twin is
    test_legacy_auto_spawn_env_and_registry_byte_identical)."""
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


# ─── spawn-defaults config file (mission-control dogfood activation) ─────────


def _write_spawn_defaults(reg, **overrides) -> None:
    """Write a well-formed spawn-defaults.json into the (monkeypatched)
    registry dir; overrides shape individual fields per test."""
    reg.mkdir(parents=True, exist_ok=True)
    cfg = {
        "session_mode_default": "async",
        "kind_scope": ["experiment"],
        "min_task_id": _FAKE_ISSUE,
        "set_by": "test",
        "set_at": "2026-08-17T00:00:00Z",
    }
    cfg.update(overrides)
    (reg / ss.SPAWN_DEFAULTS_FILENAME).write_text(json.dumps(cfg))


def test_config_file_applies_experiment_at_cutoff(chain_env, monkeypatch):
    """kind in kind_scope + id >= min_task_id -> the config default applies."""
    _write_spawn_defaults(chain_env)
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "async"


def test_config_file_below_cutoff_stays_legacy(chain_env, monkeypatch):
    """id < min_task_id -> legacy auto: existing tasks (incl. re-dispatches
    of the pre-activation proposed queue) never flip async."""
    _write_spawn_defaults(chain_env, min_task_id=_FAKE_ISSUE + 1)
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


def test_config_file_kind_out_of_scope_stays_legacy(chain_env, monkeypatch):
    _write_spawn_defaults(chain_env)
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "infra"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",  # unparseable
        json.dumps(["async"]),  # not a dict
        json.dumps(  # unrecognized mode
            {"session_mode_default": "asap", "kind_scope": ["experiment"], "min_task_id": 1}
        ),
        json.dumps(  # kind_scope not a list
            {"session_mode_default": "async", "kind_scope": "experiment", "min_task_id": 1}
        ),
        json.dumps(  # min_task_id missing
            {"session_mode_default": "async", "kind_scope": ["experiment"]}
        ),
        json.dumps(  # min_task_id not an int
            {"session_mode_default": "async", "kind_scope": ["experiment"], "min_task_id": "1"}
        ),
        json.dumps(  # bool is an int subclass but not an id
            {"session_mode_default": "async", "kind_scope": ["experiment"], "min_task_id": True}
        ),
    ],
)
def test_config_file_malformed_stays_legacy(chain_env, monkeypatch, raw):
    """Any out-of-shape spawn-defaults file fails soft to legacy auto."""
    chain_env.mkdir(parents=True, exist_ok=True)
    (chain_env / ss.SPAWN_DEFAULTS_FILENAME).write_text(raw)
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


def test_explicit_flag_beats_config_file(chain_env, monkeypatch):
    """Explicit --session-mode auto wins over a config-async default (were
    the config consulted first, this would resolve async)."""
    _write_spawn_defaults(chain_env)
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode="auto")
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


def test_registry_beats_config_file(chain_env, monkeypatch):
    _write_spawn_defaults(chain_env)
    (chain_env / f"issue-{_FAKE_ISSUE}.json").write_text(json.dumps({"session_mode": "auto"}))
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


def test_marker_beats_config_file(chain_env, monkeypatch):
    """A task-durable downgrade marker wins over the fleet config default."""
    _write_spawn_defaults(chain_env)
    monkeypatch.setattr(ss, "list_events", lambda issue: [_mode_marker("auto")])
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


def test_config_file_beats_env_default(chain_env, monkeypatch):
    """The config file sits ABOVE the env var: an explicit config 'auto'
    pins legacy even when the process env says async."""
    _write_spawn_defaults(chain_env, session_mode_default="auto")
    monkeypatch.setenv("EPM_SPAWN_DEFAULT_SESSION_MODE", "async")
    monkeypatch.setattr(ss, "get_task", lambda issue: {"frontmatter": {"kind": "experiment"}})
    args = _issue_args(session_mode=None)
    assert ss._resolve_spawn_session_mode(args, _FAKE_ISSUE) == "auto"


# ─── registry entry field + marker idempotence ───────────────────────────────


def test_register_autonomous_session_writes_mode_field_only_when_set(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    ss._register_autonomous_session(1, "sid-a", "/x", 100.0, session_mode="async")
    entry = json.loads((tmp_path / "issue-1.json").read_text())
    assert entry["session_mode"] == "async"
    # Legacy form: no kwarg -> the key is ABSENT (byte-identical entries).
    ss._register_autonomous_session(2, "sid-b", "/x", 100.0)
    entry2 = json.loads((tmp_path / "issue-2.json").read_text())
    assert "session_mode" not in entry2


def test_record_marker_idempotent_and_downgrade(monkeypatch):
    posts: list[dict] = []
    monkeypatch.setattr(
        ss, "post_event", lambda issue, kind, **kw: posts.append({"kind": kind, **kw})
    )
    # No history + auto -> nothing (absent == legacy auto already).
    monkeypatch.setattr(ss, "list_events", lambda issue: [])
    ss._record_session_mode_marker(_FAKE_ISSUE, "auto")
    assert posts == []
    # No history + async -> posts once.
    ss._record_session_mode_marker(_FAKE_ISSUE, "async")
    assert len(posts) == 1 and posts[0]["kind"] == ss.SESSION_MODE_KIND
    assert json.loads(posts[0]["note"])["mode"] == "async"
    # Matching newest marker -> idempotent no-op.
    monkeypatch.setattr(ss, "list_events", lambda issue: [_mode_marker("async")])
    ss._record_session_mode_marker(_FAKE_ISSUE, "async")
    assert len(posts) == 1
    # Explicit downgrade over an async history -> posts {mode: auto}.
    ss._record_session_mode_marker(_FAKE_ISSUE, "auto")
    assert len(posts) == 2 and json.loads(posts[1]["note"])["mode"] == "auto"
