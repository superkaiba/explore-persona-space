"""Tests for the code-enforced autonomous plan-approval gate in `scripts/task.py`
plus its backstop PreToolUse hook in `.claude/settings.json`.

The autonomous auto-approve used to live ONLY as prose in `/issue` SKILL.md
Step 2c, so a spawned `--auto` session would only auto-approve a plan if the
LLM orchestrator read that deeply-nested step and chose to obey it over the
always-loaded "ask before spending money" prior. It systematically did not
(4/4 sessions asked the user instead). `_resolve_autonomous_plan_gate` moves
the decision into code, keyed on `EPM_AUTONOMOUS_SESSION` +
`EPM_PLAN_AUTOAPPROVE_GPU_HOURS`, so it no longer depends on LLM discretion.

FAIL SAFE: a missing/None gpu_hours parks (never auto-approves on a blank
estimate).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

# Load `scripts/task.py` as a module so we can hit the pure gate resolver.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "task.py"
_spec = importlib.util.spec_from_file_location("task_cli_plan_gate", _SCRIPT)
task_cli = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["task_cli_plan_gate"] = task_cli
_spec.loader.exec_module(task_cli)  # type: ignore[union-attr]


def _clear_env(monkeypatch):
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    monkeypatch.delenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", raising=False)


# ─── not-autonomous → always interactive_pending ──────────────────────────


def test_interactive_when_env_unset(monkeypatch):
    _clear_env(monkeypatch)
    decision, cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "interactive_pending"
    assert autonomous is False
    assert cap == 24.0


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


# ─── autonomous + under cap → auto_approved ───────────────────────────────


def test_auto_approve_under_default_cap(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, cap, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"
    assert autonomous is True
    assert cap == 24.0


def test_auto_approve_at_cap_boundary(monkeypatch):
    """gpu_hours == cap auto-approves (<= comparison)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(24.0)
    assert decision == "auto_approved"


def test_auto_approve_respects_custom_cap(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "48")
    decision, cap, _autonomous = task_cli._resolve_autonomous_plan_gate(40.0)
    assert decision == "auto_approved"
    assert cap == 48.0


# ─── autonomous + over cap / blank → parked_over_cap ──────────────────────


def test_park_over_default_cap(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _cap, autonomous = task_cli._resolve_autonomous_plan_gate(30.0)
    assert decision == "parked_over_cap"
    assert autonomous is True


def test_park_over_custom_cap(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "4")
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "parked_over_cap"


def test_park_when_gpu_hours_missing_fail_safe(monkeypatch):
    """A blank/None estimate MUST park, never auto-approve (fail safe)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _cap, _autonomous = task_cli._resolve_autonomous_plan_gate(None)
    assert decision == "parked_over_cap"


def test_unparseable_cap_falls_back_to_default(monkeypatch):
    """A garbage cap env value falls back to 24.0 rather than crashing."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "not-a-number")
    decision, cap, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert cap == 24.0
    assert decision == "auto_approved"


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

# Other AskUserQuestion gates that MUST pass through (never plan-approval).
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
        _d, _c, py_autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
        hook_blocks = _run_ask_hook(_PLAN_APPROVAL_PAYLOAD, value) == 2
        assert hook_blocks == py_autonomous, (
            f"divergence on {value!r}: python_autonomous={py_autonomous} hook_blocks={hook_blocks}"
        )


def test_hook_does_not_block_other_gates_when_autonomous():
    """The backstop is targeted: it must NOT block the goal / fact-candidates
    / living-docs / dataset AskUserQuestion gates, even in an autonomous
    session."""
    for name, payload in _OTHER_GATE_PAYLOADS.items():
        assert _run_ask_hook(payload, "1") == 0, f"hook wrongly blocked the {name} gate"
