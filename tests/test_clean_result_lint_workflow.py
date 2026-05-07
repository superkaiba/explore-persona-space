"""Static structural tests for .github/workflows/clean-result-lint.yml.

Lightweight YAML sanity (no `actionlint` install required):

* The workflow YAML parses.
* The on/jobs/permissions blocks are present and shaped right.
* The lint job is gated to ``event_name == 'issues'`` (per the
  C1.v2-ISSUE-4 fix in plan §4) and references all four
  PRIORITY_LABELS values from §1's workflow.yaml.
* The backfill job is gated to ``workflow_dispatch`` AND ``inputs.backfill``.
* Both jobs run a PROJECT_PAT precheck (mirrors
  project-archive-on-close.yml's pattern).
* The lint job invokes ``verify_clean_result.py --body-stdin``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows" / "clean-result-lint.yml"
)


@pytest.fixture(scope="module")
def workflow_doc() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_workflow_yaml_parses(workflow_doc):
    assert isinstance(workflow_doc, dict)
    assert workflow_doc["name"] == "Clean-Result Lint"


def test_workflow_triggers_on_issues_and_workflow_dispatch(workflow_doc):
    # PyYAML lowercases `on:` to a Python truthy key — handle either form.
    on = workflow_doc.get("on") or workflow_doc.get(True)
    assert on is not None, "workflow `on:` missing"
    assert "issues" in on
    assert set(on["issues"]["types"]) == {"edited", "opened", "labeled"}
    assert "workflow_dispatch" in on
    assert "backfill" in on["workflow_dispatch"]["inputs"]


def test_concurrency_per_issue_with_cancel_in_progress(workflow_doc):
    cc = workflow_doc["concurrency"]
    assert "clean-result-lint-" in cc["group"]
    assert cc["cancel-in-progress"] is True


def test_top_level_permissions_minimal(workflow_doc):
    perms = workflow_doc["permissions"]
    assert perms["contents"] == "read"
    assert perms["issues"] == "write"
    # No PR / actions / pages perms requested.
    assert "pull-requests" not in perms
    assert "actions" not in perms


def test_lint_job_gated_to_issues_and_clean_results_labels(workflow_doc):
    lint = workflow_doc["jobs"]["lint"]
    cond = lint["if"]
    assert "github.event_name == 'issues'" in cond
    # All four PRIORITY_LABELS values appear in the if condition.
    assert "'clean-results:draft'" in cond
    assert "'clean-results:useful'" in cond
    assert "'clean-results:not-useful'" in cond
    assert "'clean-results'" in cond


def test_lint_job_runs_verifier_with_body_stdin(workflow_doc):
    """Lint job must invoke verify_clean_result.py --body-stdin."""
    lint = workflow_doc["jobs"]["lint"]
    steps = lint["steps"]
    verify_step = next(s for s in steps if s.get("id") == "verify")
    run_block = verify_step["run"]
    assert "--body-stdin" in run_block
    assert "--title" in run_block
    assert "--created-at" in run_block
    assert "--current-issue" in run_block
    assert "verify_clean_result.py" in run_block


def test_lint_job_posts_marker_comment(workflow_doc):
    """Last step posts a comment with the epm:clean-result-lint marker."""
    lint = workflow_doc["jobs"]["lint"]
    steps = lint["steps"]
    post_step = next(s for s in steps if s.get("name") == "Post comment")
    assert "epm:clean-result-lint" in post_step["run"]
    assert "gh issue comment" in post_step["run"]


def test_lint_job_has_project_pat_precheck(workflow_doc):
    """First step verifies PROJECT_PAT — mirrors project-archive-on-close.yml."""
    lint = workflow_doc["jobs"]["lint"]
    first = lint["steps"][0]
    assert "PROJECT_PAT" in first["name"]
    assert "PROJECT_PAT" in first.get("env", {})


def test_backfill_job_gated_to_workflow_dispatch_and_input(workflow_doc):
    backfill = workflow_doc["jobs"]["backfill"]
    cond = backfill["if"]
    assert "github.event_name == 'workflow_dispatch'" in cond
    assert "inputs.backfill == true" in cond


def test_backfill_job_iterates_clean_results_labels(workflow_doc):
    backfill = workflow_doc["jobs"]["backfill"]
    fanout_step = next(s for s in backfill["steps"] if "Fan out" in s.get("name", ""))
    run_block = fanout_step["run"]
    # All four labels enumerated in the for-loop.
    assert "clean-results:draft" in run_block
    assert "clean-results:useful" in run_block
    assert "clean-results:not-useful" in run_block
    assert "for label in clean-results:" in run_block


def test_backfill_job_dedupes_by_issue_number(workflow_doc):
    backfill = workflow_doc["jobs"]["backfill"]
    fanout_step = next(s for s in backfill["steps"] if "Fan out" in s.get("name", ""))
    run_block = fanout_step["run"]
    # The dedup line uses jq's unique_by on .number.
    assert "unique_by(.number)" in run_block


def test_no_introspection_or_destructive_perms(workflow_doc):
    """Defense-in-depth: no `pull-requests: write` (we only post on issues)."""
    for job_name, job in workflow_doc["jobs"].items():
        perms = job.get("permissions", {})
        if "pull-requests" in perms:
            assert perms["pull-requests"] == "read", (
                f"job {job_name} unexpectedly requests pull-requests: {perms['pull-requests']}"
            )
