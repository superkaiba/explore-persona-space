"""Tests for the code-enforced autonomous plan-approval gate in `scripts/task.py`
plus its backstop PreToolUse hook in `.claude/settings.json`.

The autonomous auto-approve used to live ONLY as prose in `/issue` SKILL.md
Step 2c, so a spawned `--auto` session would only auto-approve a plan if the
LLM orchestrator read that deeply-nested step and chose to obey it over the
always-loaded "ask before spending money" prior. It systematically did not
(4/4 sessions asked the user instead). `_resolve_autonomous_plan_gate` moves
the decision into code, keyed on `EPM_AUTONOMOUS_SESSION`, so it no longer
depends on LLM discretion.

As of #1771 the gate is GPU-HOUR-BLIND: an autonomous session auto-approves
ANY plan carrying a parseable GPU-hour estimate, regardless of magnitude.
`EPM_PLAN_AUTOAPPROVE_GPU_HOURS` is no longer read by the resolver; it
survives inert in argv/registry plumbing for provenance only.

FAIL SAFE (retained): a missing/None gpu_hours parks
(`parked_no_estimate`) — a correctness guard against an unestimated plan,
not a cost control.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

# Load `scripts/task.py` as a module so we can hit the pure gate resolver.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "task.py"
_spec = importlib.util.spec_from_file_location("task_cli_plan_gate", _SCRIPT)
task_cli = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["task_cli_plan_gate"] = task_cli
_spec.loader.exec_module(task_cli)  # type: ignore[union-attr]

# scripts/ on sys.path so the reported-cap parity tests can import the
# watcher + spawn_session the same way their own test files do.
_SCRIPTS_DIR = str(_SCRIPT.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from explore_persona_space.task_workflow import (  # noqa: E402
    AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS,
    PLAN_GATE_CAP_ENV,
    resolve_plan_gate_cap,
)


def _clear_env(monkeypatch):
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    monkeypatch.delenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", raising=False)


# ─── not-autonomous → always interactive_pending ──────────────────────────


def test_interactive_when_env_unset(monkeypatch):
    _clear_env(monkeypatch)
    decision, cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "interactive_pending"
    assert autonomous is False
    # The cap rides the tuple for reporting compatibility only (#2164
    # single-sourcing retained; the decision never consumes it, #1771).
    assert cap == AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS


def test_interactive_when_env_zero(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "0")
    decision, _cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "interactive_pending"
    assert autonomous is False


def test_interactive_when_env_false(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "false")
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "interactive_pending"


def test_interactive_ignores_gpu_hours_when_not_autonomous(monkeypatch):
    """Even a huge estimate stays interactive (the human will decide)."""
    _clear_env(monkeypatch)
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(9999.0)
    assert decision == "interactive_pending"


# ─── autonomous + any finite gpu_hours → auto_approved ────────────────────


def test_auto_approve_any_finite_estimate(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"
    assert autonomous is True
    assert cap == AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS


def test_auto_approve_large_estimate(monkeypatch):
    """Under the GPU-hour-blind gate (#1771), a huge estimate that would
    have parked pre-#1771 (over every historical cap) now auto-approves."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(10000.0)
    assert decision == "auto_approved"


def test_cap_env_var_is_inert_never_parks_on_gpu_hours(monkeypatch):
    """DURABILITY PIN (#1771): even a small EPM_PLAN_AUTOAPPROVE_GPU_HOURS
    with a gpu_hours estimate exceeding it MUST auto-approve — the decision
    no longer consumes the cap (this is also the blind conversion of
    #2164's test_park_over_custom_cap: identical inputs, opposite —
    now-correct — expectation). If a future change re-introduces the cap
    comparison, this test flips to parked_* and fails loud."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "4")
    decision, cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"
    assert autonomous is True
    # The cap is still RESOLVED (single-sourced, #2164) and returned for
    # reporting compatibility — just never consulted by the decision.
    assert cap == 4.0


def test_auto_approve_respects_custom_cap(monkeypatch):
    """Custom cap env threads through the resolver into the returned tuple
    (#2164 single-sourcing); the decision auto-approves regardless (#1771)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "48")
    decision, cap, _autonomous = task_cli._resolve_autonomous_plan_gate(40.0)
    assert decision == "auto_approved"
    assert cap == 48.0


def test_former_park_over_default_cap_now_auto_approves(monkeypatch):
    """Blind conversion of #2164's test_park_over_default_cap: an explicit
    cap env of 24 with a 30-GPU-h estimate parked pre-#1771; under the
    GPU-hour-blind gate it auto-approves (the cap is inert)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv(PLAN_GATE_CAP_ENV, "24")
    decision, cap, autonomous = task_cli._resolve_autonomous_plan_gate(30.0)
    assert decision == "auto_approved"
    assert autonomous is True
    assert cap == 24.0


def test_estimate_over_code_default_auto_approves(monkeypatch):
    """Blind conversion of #2164's test_park_over_code_default_when_env_unset:
    an estimate just over the code default (100) parked pre-#1771; under the
    GPU-hour-blind gate it auto-approves, and the resolved cap in the tuple
    still reads the code default."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, cap, autonomous = task_cli._resolve_autonomous_plan_gate(
        AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS + 1.0
    )
    assert decision == "auto_approved"
    assert autonomous is True
    assert cap == AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS


# ─── autonomous + blank → parked_no_estimate (fail-safe) ──────────────────


def test_auto_approve_at_former_default_cap(monkeypatch):
    """Formerly test_park_over_default_cap (pre-#2164 shape): gpu_hours=30
    exceeded the then-24h default cap and parked. Under #1771 it
    auto-approves."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _cap, autonomous = task_cli._resolve_autonomous_plan_gate(30.0)
    assert decision == "auto_approved"
    assert autonomous is True


def test_auto_approve_at_former_custom_cap(monkeypatch):
    """Formerly test_park_over_custom_cap: gpu_hours=8 vs custom cap=4
    parked. Under #1771 it auto-approves — the cap is decision-inert."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "4")
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"


def test_park_when_gpu_hours_missing_fail_safe(monkeypatch):
    """A blank/None estimate MUST park, never auto-approve (fail safe).
    The retained fail-safe — a correctness guard against an unestimated
    plan, not a cost control."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(None)
    assert decision == "parked_no_estimate"


def test_garbage_cap_env_is_ignored(monkeypatch):
    """A garbage cap env value never blocks the decision (#1771 — the
    decision ignores the cap) and the resolved tuple cap falls back to the
    code default rather than crashing (#2164 resolver semantics)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "not-a-number")
    decision, cap, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert cap == AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS
    assert decision == "auto_approved"


def test_blank_cap_env_falls_back_to_default(monkeypatch):
    """A BLANK cap env resolves like an absent one (the resolver's stated
    semantics, #2164) — historically float("") crashed the one site that
    parsed the env without a try/except (spawn_session register-current)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv(PLAN_GATE_CAP_ENV, "")
    _decision, cap, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert cap == AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS
    assert resolve_plan_gate_cap() == AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS


def test_case_insensitive_truthiness(monkeypatch):
    """`FALSE` / `No` / `False` must read as falsy (not autonomous), matching
    the shell hook's lowercase normalization — the divergence the reviewer
    flagged on `no`."""
    _clear_env(monkeypatch)
    for falsy in ("", "0", "false", "False", "FALSE", "no", "No", "NO"):
        monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", falsy)
        decision, _cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
        assert autonomous is False, f"{falsy!r} should be falsy"
        assert decision == "interactive_pending"
    for truthy in ("1", "yes", "true", "True"):
        monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", truthy)
        _decision, _cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
        assert autonomous is True, f"{truthy!r} should be truthy"


# ─── cmd_set_status MAIN branch (regression: gate-main-2tuple-unpack-crash) ─
#
# The round-2 merge reconciliation converted the followups_running-hold
# unpack to the #2164 3-tuple but left the MAIN (non-followups_running)
# branch at the 2-tuple form, so EVERY standard Step-2c call
# (`task.py set-status <N> plan_pending --auto-approve-if-autonomous
# --gpu-hours <X>`) died with `ValueError: too many values to unpack
# (expected 2)` BEFORE any decision branch ran. The resolver-level tests
# above never call cmd_set_status, and the followup-hold tests
# (test_task_workflow_post_marker_echo.py) pin only the :470 branch — the
# main branch had zero coverage, which is how the crash shipped through a
# green suite. Discrimination: both tests below were run against the
# pre-fix tree (commit 6b526c151a) and FAILED with exactly that
# ValueError; they pass after the 3-tuple fix.


def _fake_set_status_recorder(moved):
    def fake_set_status(number, status, *, note=None, force_followup_exit=False):
        moved.append((number, status))
        return Path("/tmp/tasks") / status / str(number)

    return fake_set_status


def _fake_post_event_recorder(posted):
    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker))
        return {"kind": marker, "version": version}

    return fake_post_event


def test_cmd_set_status_main_branch_auto_approves(monkeypatch, capsys):
    """MAIN-branch auto-approve outcome: a parseable estimate on a
    non-followups_running task flips the status to `approved`, posts
    epm:plan-approved, and prints the PLAN_GATE_DECISION line. Fails
    pre-fix at the 2-tuple unpack (ValueError) before any of that runs."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    moved: list[tuple[int, str]] = []
    posted: list[tuple[int, str]] = []
    monkeypatch.setattr(
        task_cli, "get_task", lambda number: {"status": "planning", "frontmatter": {"tags": []}}
    )
    monkeypatch.setattr(task_cli, "set_status", _fake_set_status_recorder(moved))
    monkeypatch.setattr(task_cli, "post_event", _fake_post_event_recorder(posted))

    ns = argparse.Namespace(
        number=537,
        status="plan_pending",
        note=None,
        auto_approve_if_autonomous=True,
        gpu_hours=8.0,
    )
    task_cli.cmd_set_status(ns)

    assert moved == [(537, "approved")]
    assert posted == [(537, "epm:plan-approved")]
    out = capsys.readouterr().out
    assert "PLAN_GATE_DECISION: auto_approved gpu_hours=8.0" in out


def test_cmd_set_status_main_branch_parks_no_estimate(monkeypatch, capsys):
    """MAIN-branch fail-safe outcome: a missing/None estimate parks the task
    AT plan_pending, posts epm:awaiting-spend-approval, and prints the
    parked_no_estimate PLAN_GATE_DECISION line. Fails pre-fix at the same
    2-tuple unpack — the crash precedes the decision branch entirely."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    moved: list[tuple[int, str]] = []
    posted: list[tuple[int, str]] = []
    monkeypatch.setattr(
        task_cli, "get_task", lambda number: {"status": "planning", "frontmatter": {"tags": []}}
    )
    monkeypatch.setattr(task_cli, "set_status", _fake_set_status_recorder(moved))
    monkeypatch.setattr(task_cli, "post_event", _fake_post_event_recorder(posted))

    ns = argparse.Namespace(
        number=537,
        status="plan_pending",
        note=None,
        auto_approve_if_autonomous=True,
        gpu_hours=None,  # missing estimate → fail-safe park
    )
    task_cli.cmd_set_status(ns)

    assert moved == [(537, "plan_pending")]
    assert posted == [(537, "epm:awaiting-spend-approval")]
    out = capsys.readouterr().out
    assert "PLAN_GATE_DECISION: parked_no_estimate gpu_hours=None" in out


# ─── #2164 parity, adapted to the blind gate (#1771) ───────────────────────


def test_decision_cap_equals_spawn_default_and_park_msg_names_no_cap(monkeypatch, tmp_path):
    """Adapted #2164 A2 parity: with the cap env UNSET, the (decision-inert)
    cap the gate resolver returns and the `spawn-issue
    --auto-approve-gpu-hours` argparse default are the same single-sourced
    number. The watcher's plan_pending park push no longer names a cap at
    all — under the blind gate (#1771) the only park cause is a missing
    estimate, so the message states that instead."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")

    # 1. Deciding site (scripts/task.py) — cap rides the tuple, inert.
    _decision, decided_cap, _autonomous = task_cli._resolve_autonomous_plan_gate(1.0)
    assert decided_cap == resolve_plan_gate_cap() == AUTONOMOUS_PLAN_GATE_DEFAULT_GPU_HOURS

    # 2. Reporting site (watcher park push): blind-gate wording, no cap.
    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "_task_title", lambda _issue: "")
    msg = asw._gate_push_message(137, "plan_pending", [], True)
    assert "no GPU-hour estimate" in msg
    assert "GPU-h cap" not in msg

    # 3. Spawning site (argparse default). set_defaults(fn=...) resolves the
    # module global at parser-build time inside main(), so the monkeypatched
    # capture function receives the parsed args without spawning anything.
    import spawn_session as ss

    captured: dict[str, object] = {}
    monkeypatch.setattr(ss, "cmd_spawn_issue", lambda args: captured.update(vars(args)))
    ss.main(["spawn-issue", "--issue", "1"])
    assert captured["auto_approve_gpu_hours"] == decided_cap


def test_registered_per_issue_cap_still_preferred_by_respawn_plumbing(monkeypatch, tmp_path):
    """Adapted #2164 A2 custom-cap clause: the per-issue registry entry's
    ``auto_approve_gpu_hours`` is still what the watcher's respawn plumbing
    reads back (``_stalled_cap_gpu_hours`` prefers the registry over the
    resolver fallback) — retained provenance/plumbing, decision-inert under
    the blind gate (#1771): the same 60-vs-50 inputs now auto-approve."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv(PLAN_GATE_CAP_ENV, "50")

    decision, decided_cap, _autonomous = task_cli._resolve_autonomous_plan_gate(60.0)
    assert decision == "auto_approved"
    assert decided_cap == 50.0

    # Drop the env before the registry read — the watcher cron never sees
    # the session's env; only registry-preferring code returns 50 here
    # (the resolver fallback would return the code default, 100).
    monkeypatch.delenv(PLAN_GATE_CAP_ENV, raising=False)

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "_task_title", lambda _issue: "")
    (tmp_path / "issue-137.json").write_text(json.dumps({"auto_approve_gpu_hours": 50.0}))
    assert asw._stalled_cap_gpu_hours(137) == 50.0
    # The park push renders the blind-gate wording regardless of the entry.
    msg = asw._gate_push_message(137, "plan_pending", [], True)
    assert "no GPU-hour estimate" in msg


def test_stalled_cap_falls_back_on_encoding_corrupt_registry_entry(monkeypatch, tmp_path):
    """#2164 round 2, retained: an encoding-corrupt ``issue-<N>.json``
    raises ``UnicodeDecodeError`` from ``read_text()`` — a ``ValueError``,
    OUTSIDE the old ``(JSONDecodeError, OSError)`` except tuple.
    ``_stalled_cap_gpu_hours`` feeds the watcher's respawn plumbing and is
    reachable from notification paths invoked unwrapped in ``main()``, so a
    raise would kill an entire watcher tick. The helper must return the
    resolver fallback instead (asserted against a distinctive env value so
    a hardcoded literal cannot pass)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(PLAN_GATE_CAP_ENV, "37")

    import autonomous_session_watch as asw

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "_task_title", lambda _issue: "")
    # 0xff is invalid UTF-8 in any position → read_text() raises
    # UnicodeDecodeError before json.loads is ever reached.
    (tmp_path / "issue-137.json").write_bytes(b'\xff\xfe{"auto_approve_gpu_hours": 50.0}')
    assert asw._stalled_cap_gpu_hours(137) == 37.0
    # The gate-push notification path survives too (blind-gate wording).
    msg = asw._gate_push_message(137, "plan_pending", [], True)
    assert "no GPU-hour estimate" in msg


# ─── #2164: anti-drift source scan ─────────────────────────────────────────

_CAP_ENV_NAME = "EPM_PLAN_AUTOAPPROVE_GPU_HOURS"

# Env-READ forms of the cap — by literal name or via the shared
# PLAN_GATE_CAP_ENV constant. Injection dict keys, help strings, and
# docstrings mention the NAME without a read form and do not match.
_CAP_READ_PATTERNS = (
    re.compile(r"\.get\(\s*['\"]" + _CAP_ENV_NAME + r"['\"]"),
    re.compile(r"getenv\(\s*['\"]" + _CAP_ENV_NAME + r"['\"]"),
    re.compile(r"environ\[\s*['\"]" + _CAP_ENV_NAME + r"['\"]\s*\]"),
    re.compile(r"\.get\(\s*PLAN_GATE_CAP_ENV\b"),
    re.compile(r"getenv\(\s*PLAN_GATE_CAP_ENV\b"),
    re.compile(r"environ\[\s*PLAN_GATE_CAP_ENV\s*\]"),
)

# Rows-4/5 drift class (#2164): a numeric-literal fallback on the registry
# key reads no env, so the env scan alone cannot see it.
_REGISTRY_LITERAL_FALLBACK = re.compile(r"\.get\(\s*['\"]auto_approve_gpu_hours['\"]\s*,\s*[\d.]")

# Row-8 drift class: a literal-bearing env read in an orchestrator-facing
# SKILL.md snippet, ready to be copied back into code by a human.
_SKILL_LITERAL_ENV_READ = re.compile(
    r"environ\.get\(\s*['\"]" + _CAP_ENV_NAME + r"['\"]\s*,\s*['\"]"
)


def test_cap_env_read_is_single_sourced():
    """Anti-drift scan (#2164): env-READ forms of the plan-gate cap occur in
    exactly ONE place repo-wide — the resolver in task_workflow.py — so a
    tenth divergent site is a red test, not a future incident."""
    repo = Path(__file__).resolve().parents[1]
    files = sorted(
        set(repo.glob("scripts/**/*.py")) | set(repo.glob("src/explore_persona_space/**/*.py"))
    )
    # Self-checks: a silently-narrowed glob must fail loud, never pass
    # vacuously. `scripts/**/*.py` is RECURSIVE on purpose — scripts/ has
    # .py subdirectories (issue_355/, issue_597/, ...) a non-recursive
    # `scripts/*.py` would silently skip.
    assert files, "scan collected no files — the glob is broken"
    known_subdir_file = repo / "scripts" / "issue_355" / "strip_confound_filter.py"
    assert known_subdir_file in files, (
        "known scripts/ SUBDIRECTORY file missing from the collected set — "
        "the glob is no longer recursive and the repo-wide claim is false"
    )

    allowed = {repo / "src" / "explore_persona_space" / "task_workflow.py"}
    offenders: dict[Path, list[str]] = {}
    registry_fallback_offenders: dict[Path, list[str]] = {}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        hits = [p.pattern for p in _CAP_READ_PATTERNS if p.search(text)]
        if hits and f not in allowed:
            offenders[f] = hits
        if _REGISTRY_LITERAL_FALLBACK.search(text):
            registry_fallback_offenders[f] = [_REGISTRY_LITERAL_FALLBACK.pattern]

    assert not offenders, (
        "cap env read outside the single resolver (route it through "
        f"task_workflow.resolve_plan_gate_cap): {offenders}"
    )
    # The scan itself must be alive: the resolver's own read matches.
    resolver_text = next(iter(allowed)).read_text(encoding="utf-8")
    assert any(p.search(resolver_text) for p in _CAP_READ_PATTERNS), (
        "the resolver's own env read no longer matches the scan patterns — "
        "the scan is dead and would pass vacuously"
    )
    assert not registry_fallback_offenders, (
        "numeric-literal fallback on the auto_approve_gpu_hours registry key "
        "(use task_workflow.resolve_plan_gate_cap() as the fallback): "
        f"{registry_fallback_offenders}"
    )

    # Orchestrator-facing SKILL.md snippets: a human copying a snippet must
    # not be handed a literal-bearing env read (row 8's origin).
    for skill in (
        repo / ".claude" / "skills" / "issue" / "SKILL.md",
        repo / ".claude" / "skills" / "issue-tick" / "SKILL.md",
    ):
        assert skill.exists(), f"scanned SKILL.md moved: {skill}"
        assert not _SKILL_LITERAL_ENV_READ.search(skill.read_text(encoding="utf-8")), (
            f"{skill}: snippet re-introduces a literal-bearing env read of "
            f"{_CAP_ENV_NAME} — use task_workflow.resolve_plan_gate_cap()"
        )


# ─── Backstop hook (settings.json PreToolUse on AskUserQuestion) ───────────
#
# The hook is the second, harness-level layer: even if the orchestrator
# mis-follows SKILL.md Step 2c, the hook must hard-block a plan-approval
# AskUserQuestion in an autonomous session. These tests pin (a) it matches
# the CANONICAL plan-approval question shape from SKILL.md (so a future
# rewording can't silently disable the backstop), (b) it does NOT block the
# other AskUserQuestion gates, and (c) its truthiness agrees with the Python
# resolver.

_SETTINGS = Path(__file__).resolve().parents[1] / ".claude" / "settings.json"


def _ask_hook_command() -> str:
    settings = json.loads(_SETTINGS.read_text())
    for entry in settings["hooks"]["PreToolUse"]:
        if entry.get("matcher") == "AskUserQuestion":
            return entry["hooks"][0]["command"]
    raise AssertionError("no AskUserQuestion PreToolUse hook in settings.json")


def _run_ask_hook(payload: dict, env_value: str | None) -> int:
    """Run the real settings.json hook command with `payload` on stdin and
    EPM_AUTONOMOUS_SESSION=env_value. Returns the exit code (2 == block)."""
    import os as _os

    env = {k: v for k, v in _os.environ.items() if k != "EPM_AUTONOMOUS_SESSION"}
    if env_value is not None:
        env["EPM_AUTONOMOUS_SESSION"] = env_value
    proc = subprocess.run(
        ["bash", "-c", _ask_hook_command()],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode


# Canonical plan-approval AskUserQuestion shape, mirrored from
# .claude/skills/issue/SKILL.md § gates.plan_approval.
_PLAN_APPROVAL_PAYLOAD = {
    "tool_name": "AskUserQuestion",
    "tool_input": {
        "questions": [
            {
                "question": (
                    "Approve plan v1 for task #137? Plan: https://eps.superkaiba.com/tasks/137/plan"
                ),
                "header": "Plan #137",
                "multiSelect": False,
                "options": [
                    {"label": "Approve", "description": "Dispatch implementer. Est. 8 GPU-hours."},
                    {"label": "Revise <notes>", "description": "Re-run /adversarial-planner."},
                    {"label": "Defer", "description": "Park at plan_pending."},
                ],
            }
        ]
    },
}

# Other AskUserQuestion shapes. Under the 2026-06-06 broadened-hook policy
# (proposal #4, task #503/#504/#505 incident), ALL of these MUST be blocked
# in autonomous mode — `asking is the failure mode` is the rule. In
# interactive mode (env unset), they pass through unchanged.
_OTHER_GATE_PAYLOADS = {
    "goal": {
        "tool_input": {
            "questions": [
                {"question": "What is the one-sentence Goal of this experiment?", "header": "Goal"}
            ]
        }
    },
    "fact_candidates": {
        "tool_input": {
            "questions": [
                {
                    "question": "Phase 0 (fact-candidates) — pick the fact id.",
                    "header": "Pick fact (id)",
                }
            ]
        }
    },
    "living_docs": {
        "tool_input": {
            "questions": [
                {
                    "question": "Apply this living-docs update for task #137? Proposed diff on https://eps.superkaiba.com/tasks/137",
                    "header": "Living docs #137",
                }
            ]
        }
    },
    "dataset": {
        "tool_input": {
            "questions": [{"question": "Which dataset split should I use?", "header": "Dataset"}]
        }
    },
    "whack_a_mole": {
        "tool_input": {
            "questions": [
                {
                    "question": (
                        "Whack-a-mole detector fired (3 distinct bug classes in 3 rounds). "
                        "continue-as-planned vs pivot-to-unification?"
                    ),
                    "header": "Pivot #137",
                }
            ]
        }
    },
    "compute_deviation": {
        "tool_input": {
            "questions": [
                {
                    "question": (
                        "Compute deviation 3.2x plan - continue_as_is vs "
                        "accept_descope_to_seeds_3_with_caveats?"
                    ),
                    "header": "Cost pivot #137",
                }
            ]
        }
    },
    "concern_deferral": {
        "tool_input": {
            "questions": [
                {
                    "question": (
                        "Open CONCERN concern-id=abc123 - defer with rationale "
                        "or bounce to implementer?"
                    ),
                    "header": "Concern deferral #137",
                }
            ]
        }
    },
}


def test_hook_blocks_plan_approval_when_autonomous():
    assert _run_ask_hook(_PLAN_APPROVAL_PAYLOAD, "1") == 2


def test_hook_allows_plan_approval_when_not_autonomous():
    assert _run_ask_hook(_PLAN_APPROVAL_PAYLOAD, None) == 0
    assert _run_ask_hook(_PLAN_APPROVAL_PAYLOAD, "0") == 0


def test_hook_truthiness_matches_python_resolver(monkeypatch):
    """For each env value, the hook blocks the plan-approval ask iff the
    Python resolver reports autonomous — no divergence on `no` / `FALSE`."""
    _clear_env(monkeypatch)
    for value in ("", "0", "false", "False", "FALSE", "no", "No", "1", "yes", "true"):
        monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", value)
        _d, _cap, py_autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
        hook_blocks = _run_ask_hook(_PLAN_APPROVAL_PAYLOAD, value) == 2
        assert hook_blocks == py_autonomous, (
            f"divergence on {value!r}: python_autonomous={py_autonomous} hook_blocks={hook_blocks}"
        )


def test_hook_blocks_all_asks_when_autonomous():
    """Per the 2026-06-06 broadened-hook policy (proposal #4, task
    #503/#504/#505 incident): the hook blocks ANY AskUserQuestion when
    EPM_AUTONOMOUS_SESSION is set, not just plan-approval. The autonomous
    branch must auto-resolve every fork — `asking is the failure mode`."""
    for name, payload in _OTHER_GATE_PAYLOADS.items():
        assert _run_ask_hook(payload, "1") == 2, (
            f"hook should block the {name} gate in autonomous mode but did not"
        )


def test_hook_allows_all_gates_when_not_autonomous():
    """In interactive mode (env unset / 0 / false), the hook allows every
    AskUserQuestion shape — gates that legitimately ask the user must
    continue to work."""
    for name, payload in _OTHER_GATE_PAYLOADS.items():
        for env_value in (None, "0", "false"):
            assert _run_ask_hook(payload, env_value) == 0, (
                f"hook wrongly blocked the {name} gate in interactive mode "
                f"(env_value={env_value!r})"
            )
