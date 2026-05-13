"""Static guards for the Sagan migration surface."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_retired_github_issue_workflows_are_absent() -> None:
    retired = {
        ".github/workflows/clean-result-lint.yml",
        ".github/workflows/project-archive-on-close.yml",
        ".github/workflows/project-auto-add.yml",
        ".github/workflows/project-sync.yml",
        ".github/ISSUE_TEMPLATE/experiment.md",
        ".github/ISSUE_TEMPLATE/code-change.md",
    }
    present = [path for path in retired if (REPO_ROOT / path).exists()]
    assert not present


def test_remaining_workflows_do_not_listen_to_issue_events() -> None:
    for workflow in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        doc = yaml.safe_load(workflow.read_text()) or {}
        triggers = doc.get("on") or doc.get(True) or {}
        assert "issues" not in triggers, f"{workflow.name} still listens to repository issue events"


def test_issue_template_points_to_sagan() -> None:
    config = yaml.safe_load((REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text())
    assert config["blank_issues_enabled"] is False
    urls = [link["url"] for link in config.get("contact_links", [])]
    assert "https://sagan.superkaiba.com/experiments" in urls
