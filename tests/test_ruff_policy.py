"""Durability pins for the #1023 ruff scoping decision (``pyproject.toml``).

Task #1023 scoped style/noise lint rules OFF the frozen per-issue
experiment/figure script paths (``scripts/*``, ``experiments/*``,
``eps/experiments/*``) via ``[tool.ruff.lint.per-file-ignores]``, and
excluded the artifact dirs ``eval_results/`` + ``figures/`` entirely.
Real-bug rules (F* undefined names/imports, E9 syntax errors, core
bugbear B, E722 bare-except) stay ON everywhere; ``src/`` and ``tests/``
keep the full ruleset.

Two invariants pinned here:

1. The LIVE (maintained, non-frozen) workflow-helper scripts stay clean
   under the FULL ruleset even though they live under the relaxed
   ``scripts/*`` path — the per-file-ignores relaxation must never become
   a license for the maintained helpers to rot.
2. The per-file-ignores table keeps its three path keys, and ``RUF100``
   stays in each ignore list: frozen scripts carry inline ``# noqa``
   directives across ~776 files that become "unused" under these ignores
   and would otherwise ADD ~3.3k RUF100 errors repo-wide (measured
   2026-07-17 on task #1023).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Live (maintained) workflow-helper scripts, pinned clean under the FULL
# ruleset. Derived from the workflow-surface script list in
# .claude/rules/workflow-fix-on-bug.md § Workflow surface (Python entries
# only), verified 0-error under the full ruleset at pin time (2026-07-17,
# task #1023). If you add a new workflow-helper script under scripts/,
# add it here so it stays pinned to the full ruleset.
#
# Dropped dirty stragglers (pre-existing full-ruleset errors at pin time;
# re-add after cleanup): cleanup_pod.py (1x C901), gh_project.py
# (3x E501, 1x E741, 1x C901, 1x SIM108), post_step_completed.py (1x E501).
LIVE_WORKFLOW_HELPERS = [
    "scripts/audit_clean_results_body_discipline.py",
    "scripts/autonomous_session_watch.py",
    "scripts/backend_poll.py",
    "scripts/check_no_secret_shaped_strings.py",
    "scripts/codex_task.py",
    "scripts/daily_drive_filings.py",
    "scripts/dispatch_issue.py",
    "scripts/failure_classifier.py",
    "scripts/gpu_heuristics.py",
    "scripts/pm_queue_report.py",
    "scripts/pod.py",
    "scripts/pod_audit.py",
    "scripts/pod_config.py",
    "scripts/pod_disk_guard.py",
    "scripts/pod_lifecycle.py",
    "scripts/pod_watch.py",
    "scripts/poll_pipeline.py",
    "scripts/recent_clean_results.py",
    "scripts/redact_for_gist.py",
    "scripts/runpod_api.py",
    "scripts/select_step9c_tests.py",
    "scripts/session_progress_report.py",
    "scripts/session_resolver.py",
    "scripts/session_summarize.py",
    "scripts/spawn_session.py",
    "scripts/step9c_baseline.py",
    "scripts/sync_repo_root.py",
    "scripts/task.py",
    "scripts/task_state.py",
    "scripts/verify_carryover_inputs.py",
    "scripts/verify_plan.py",
    "scripts/verify_task_body.py",
    "scripts/verify_uploads.py",
    "scripts/worktree_audit.py",
    "scripts/workflow_lint.py",
]


def _ruff_bin() -> str:
    """Resolve ruff: PATH first, then the running interpreter's venv bin.

    The sibling fallback covers invoking the venv's pytest binary directly
    (which does not prepend the venv bin/ to PATH). Never ``--isolated``
    downstream — that would lose ``line-length=100`` and the #576
    ``known-third-party = ["wandb"]`` pin.
    """
    ruff = shutil.which("ruff")
    if ruff:
        return ruff
    sibling = Path(sys.executable).parent / "ruff"
    if sibling.exists():
        return str(sibling)
    raise AssertionError("ruff not found on PATH nor next to sys.executable")


def test_live_workflow_helpers_clean_under_full_ruleset():
    """The maintained helpers lint clean with per-file-ignores neutralized.

    ``--config 'lint.per-file-ignores = {}'`` overrides ONLY that table
    while the rest of pyproject.toml (select set, line-length, isort pin)
    still applies — verified on ruff 0.15.9 (task #1023, assumption A7).
    Fallback if a future ruff rejects the inline-override syntax: assert
    per-helper cleanliness under the project config PLUS an explicit
    ``--select E,W,I,F,UP,B,SIM,C901,RUF`` invocation (never ``--isolated``).
    """
    missing = [p for p in LIVE_WORKFLOW_HELPERS if not (REPO_ROOT / p).is_file()]
    assert not missing, f"pinned helpers missing from tree (update the pin list): {missing}"
    # One batched subprocess call (~1-2 s) instead of per-file calls.
    proc = subprocess.run(
        [
            _ruff_bin(),
            "check",
            "--config",
            "lint.per-file-ignores = {}",
            "--output-format",
            "concise",
            *LIVE_WORKFLOW_HELPERS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "live workflow helpers must stay clean under the FULL ruleset "
        "(fix the helper, or move a deliberately-frozen script out of the pin "
        f"list with a comment):\n{proc.stdout}\n{proc.stderr}"
    )


def test_frozen_path_ignores_present():
    """The #1023 per-file-ignores table keeps its keys and the RUF100 guard."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    ignores = cfg["tool"]["ruff"]["lint"]["per-file-ignores"]
    for key in ("scripts/*", "experiments/*", "eps/experiments/*"):
        assert key in ignores, f"per-file-ignores lost the {key!r} entry (#1023)"
        assert "RUF100" in ignores[key], (
            f"RUF100 missing from the {key!r} ignore list — removing it ADDS "
            "~3.3k unused-noqa errors repo-wide (measured 2026-07-17, #1023)"
        )

    excludes = cfg["tool"]["ruff"]["extend-exclude"]
    for artifact_dir in ("eval_results", "figures"):
        assert artifact_dir in excludes, (
            f"{artifact_dir!r} missing from [tool.ruff] extend-exclude (#1023)"
        )
