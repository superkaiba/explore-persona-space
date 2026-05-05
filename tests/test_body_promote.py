"""Unit tests for body-promote / body-restore protocol.

Mocks the `_gh` shell-out so tests don't hit GitHub. Validates the 3-step
protocol and idempotency contract from .claude/plans/workflow-refactor-v3.md.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

from scripts import gh_project


def _stub_view(body: str, comments: list[dict] | None = None) -> dict:
    return {"title": "Test issue", "body": body, "labels": [], "comments": comments or []}


def test_body_promote_first_run_three_steps(tmp_path):
    """First promote: posts snapshot comment, edits body, adds label."""
    draft = tmp_path / "draft.md"
    draft.write_text("# Clean result\n\nClaim X with N=42.")

    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(args)
        return ""

    with (
        patch.object(gh_project, "_gh", side_effect=fake_gh),
        patch.object(
            gh_project, "_gh_issue_view_full", return_value=_stub_view("Original spec for sandbox.")
        ),
    ):
        ns = argparse.Namespace(
            owner="superkaiba",
            project=1,
            repo="superkaiba/explore-persona-space",
            issue=999,
            draft=str(draft),
        )
        gh_project.cmd_body_promote(ns)

    # Three gh calls: comment, edit body, edit add-label.
    assert len(calls) == 3
    assert calls[0][:2] == ["issue", "comment"]
    assert gh_project.ORIGINAL_MARKER in calls[0][-1]
    assert "Original spec for sandbox." in calls[0][-1]

    assert calls[1][:2] == ["issue", "edit"]
    assert calls[1][-2] == "--body"
    assert calls[1][-1].startswith(gh_project.PROMOTED_MARKER)
    assert "# Clean result" in calls[1][-1]

    assert calls[2][:2] == ["issue", "edit"]
    assert calls[2][-2:] == ["--add-label", "clean-results:draft"]


def test_body_promote_revision_only_edits_body(tmp_path):
    """If body already starts with PROMOTED_MARKER, just edit it (revision)."""
    draft = tmp_path / "draft.md"
    draft.write_text("# Revised clean result")

    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(args)
        return ""

    promoted_body = f"{gh_project.PROMOTED_MARKER}\n\n# Old clean result"
    with (
        patch.object(gh_project, "_gh", side_effect=fake_gh),
        patch.object(gh_project, "_gh_issue_view_full", return_value=_stub_view(promoted_body)),
    ):
        ns = argparse.Namespace(
            owner="superkaiba",
            project=1,
            repo="superkaiba/explore-persona-space",
            issue=999,
            draft=str(draft),
        )
        gh_project.cmd_body_promote(ns)

    # Only one gh call: edit body. No comment, no label add.
    assert len(calls) == 1
    assert calls[0][:2] == ["issue", "edit"]
    assert "# Revised clean result" in calls[0][-1]


def test_body_promote_skips_duplicate_snapshot_comment(tmp_path):
    """If epm:original-body comment already exists (partial prior run), skip step 1."""
    draft = tmp_path / "draft.md"
    draft.write_text("# Clean result")

    existing_snapshot = {
        "body": f"{gh_project.ORIGINAL_MARKER}\n## Original issue body...\n\nOld content"
    }
    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(args)
        return ""

    with (
        patch.object(gh_project, "_gh", side_effect=fake_gh),
        patch.object(
            gh_project,
            "_gh_issue_view_full",
            return_value=_stub_view("Original body", comments=[existing_snapshot]),
        ),
    ):
        ns = argparse.Namespace(
            owner="superkaiba",
            project=1,
            repo="superkaiba/explore-persona-space",
            issue=999,
            draft=str(draft),
        )
        gh_project.cmd_body_promote(ns)

    # Two gh calls: edit body + add label. Snapshot step skipped because marker present.
    assert len(calls) == 2
    assert calls[0][:2] == ["issue", "edit"]
    assert calls[1][-2:] == ["--add-label", "clean-results:draft"]


def test_body_restore_round_trip():
    """body-restore reads the snapshot comment, writes it back."""
    snapshot_text = (
        f"{gh_project.ORIGINAL_MARKER}\n"
        "## Original issue body (preserved before clean-result promotion)\n"
        "\n"
        "Original spec for sandbox.\nMore content."
    )
    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(args)
        return ""

    with (
        patch.object(gh_project, "_gh", side_effect=fake_gh),
        patch.object(
            gh_project,
            "_gh_issue_view_full",
            return_value=_stub_view(
                f"{gh_project.PROMOTED_MARKER}\n# Clean result",
                comments=[{"body": snapshot_text}],
            ),
        ),
    ):
        ns = argparse.Namespace(
            owner="superkaiba",
            project=1,
            repo="superkaiba/explore-persona-space",
            issue=999,
        )
        gh_project.cmd_body_restore(ns)

    # First call: edit body to original. Second: remove clean-results:draft label.
    assert calls[0][:2] == ["issue", "edit"]
    assert calls[0][-1] == "Original spec for sandbox.\nMore content."
    assert calls[1][-2:] == ["--remove-label", "clean-results:draft"]
