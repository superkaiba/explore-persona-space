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
    decision, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "interactive_pending"
    assert autonomous is False


def test_interactive_when_env_zero(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "0")
    decision, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "interactive_pending"
    assert autonomous is False


def test_interactive_when_env_false(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "false")
    decision, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "interactive_pending"


def test_interactive_ignores_gpu_hours_when_not_autonomous(monkeypatch):
    """Even a huge estimate stays interactive (the human will decide)."""
    _clear_env(monkeypatch)
    decision, _autonomous = task_cli._resolve_autonomous_plan_gate(9999.0)
    assert decision == "interactive_pending"


# ─── autonomous + any finite gpu_hours → auto_approved ────────────────────


def test_auto_approve_any_finite_estimate(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"
    assert autonomous is True


def test_auto_approve_large_estimate(monkeypatch):
    """Under the GPU-hour-blind gate (#1771), a huge estimate that would
    have parked pre-#1771 (24h cap) now auto-approves."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _autonomous = task_cli._resolve_autonomous_plan_gate(10000.0)
    assert decision == "auto_approved"


def test_cap_env_var_is_inert_never_parks_on_gpu_hours(monkeypatch):
    """DURABILITY PIN (#1771): even a small EPM_PLAN_AUTOAPPROVE_GPU_HOURS
    with a gpu_hours estimate exceeding it MUST auto-approve — the resolver
    no longer reads the env var. If a future change re-introduces the cap
    comparison, this test flips to parked_* and fails loud."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "4")
    decision, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"
    assert autonomous is True


# ─── autonomous + blank → parked_no_estimate (fail-safe) ──────────────────


def test_auto_approve_at_former_default_cap(monkeypatch):
    """Formerly test_park_over_default_cap: gpu_hours=30 exceeded the 24h
    default cap and parked. Under #1771 it auto-approves."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, autonomous = task_cli._resolve_autonomous_plan_gate(30.0)
    assert decision == "auto_approved"
    assert autonomous is True


def test_auto_approve_at_former_custom_cap(monkeypatch):
    """Formerly test_park_over_custom_cap: gpu_hours=8 vs custom cap=4
    parked. Under #1771 it auto-approves — the env var is inert."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "4")
    decision, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"


def test_park_when_gpu_hours_missing_fail_safe(monkeypatch):
    """A blank/None estimate MUST park, never auto-approve (fail safe).
    The retained fail-safe — a correctness guard against an unestimated
    plan, not a cost control."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    decision, _autonomous = task_cli._resolve_autonomous_plan_gate(None)
    assert decision == "parked_no_estimate"


def test_garbage_cap_env_is_ignored(monkeypatch):
    """A garbage cap env value is ignored (the env var is no longer read
    by the resolver). Under #1771 the estimate alone drives the decision."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "not-a-number")
    decision, _autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
    assert decision == "auto_approved"


def test_case_insensitive_truthiness(monkeypatch):
    """`FALSE` / `No` / `False` must read as falsy (not autonomous), matching
    the shell hook's lowercase normalization — the divergence the reviewer
    flagged on `no`."""
    _clear_env(monkeypatch)
    for falsy in ("", "0", "false", "False", "FALSE", "no", "No", "NO"):
        monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", falsy)
        decision, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
        assert autonomous is False, f"{falsy!r} should be falsy"
        assert decision == "interactive_pending"
    for truthy in ("1", "yes", "true", "True"):
        monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", truthy)
        _decision, autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
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
        _d, py_autonomous = task_cli._resolve_autonomous_plan_gate(8.0)
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
