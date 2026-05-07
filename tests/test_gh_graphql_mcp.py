"""Unit + integration tests for the gh_graphql MCP server.

Strategy: stub the GitHub GraphQL endpoint with httpx's
``MockTransport``. Each mutation gets at least one happy-path test +
one body-cap / scope-violation / auth-missing failure-mode test.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from explore_persona_space.mcp_servers.gh_graphql import tools as gh_tools
from explore_persona_space.mcp_servers.gh_graphql.client import GhGraphQLClient
from explore_persona_space.mcp_servers.gh_graphql.tools import ADD_COMMENT_BODY_MAX_BYTES

# ──────────────────────────────────────────────────────────────────────
# Test infrastructure: mocked GraphQL transport
# ──────────────────────────────────────────────────────────────────────


def make_mock_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    token: str = "ghp_test_token",
    repo_owner: str = "superkaiba",
    repo_name: str = "explore-persona-space",
) -> GhGraphQLClient:
    """Build a GhGraphQLClient backed by httpx.MockTransport."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, timeout=5.0)
    return GhGraphQLClient(
        token=token, repo_owner=repo_owner, repo_name=repo_name, http_client=http
    )


def gql_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"data": data})


def gql_error(errors: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"errors": errors, "data": None})


def parse_gql_request(req: httpx.Request) -> dict[str, Any]:
    return json.loads(req.content.decode("utf-8"))


def run(coro):
    """asyncio.run shim for tests (works on Python 3.11+)."""
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────────
# Allow-list / denylist registration
# ──────────────────────────────────────────────────────────────────────


def test_mutation_tool_names_are_exactly_the_allow_list():
    """The published tool list MUST match the allow-list table in plan §3."""
    expected = {
        "add_issue_comment",
        "add_issue_labels",
        "remove_issue_labels",
        "update_issue_title",
        "update_issue_body",
        "create_issue",
        "close_issue",
        "reopen_issue",
        "create_pull_request",
        "mark_pr_ready_for_review",
        "merge_pull_request",
        "update_project_v2_field",
        "read_issue",
    }
    assert set(gh_tools.MUTATION_TOOL_NAMES) == expected


def test_denylisted_tools_are_not_registered():
    """Denylisted GitHub mutations are NOT in MUTATION_TOOL_NAMES."""
    denylist = {
        "archive_repository",
        "transfer_issue",
        "delete_issue",
        "delete_repository",
        "create_repository",
        "update_repository",
        # Project mutations beyond update_project_v2_field
        "add_project_v2_draft_issue",
        "delete_project_v2_item",
        # Schema introspection — there are no tool names for these but
        # we confirm none of the registered names look like introspection.
        "schema",
        "type",
    }
    assert denylist.isdisjoint(set(gh_tools.MUTATION_TOOL_NAMES))


def test_build_server_registers_only_allow_list_tools():
    """FastMCP server's list_tools matches MUTATION_TOOL_NAMES exactly."""
    pytest.importorskip("mcp")
    from explore_persona_space.mcp_servers.gh_graphql.server import build_server

    mcp = build_server()
    tools = run(mcp.list_tools())
    registered = {t.name for t in tools}
    assert registered == set(gh_tools.MUTATION_TOOL_NAMES)


# ──────────────────────────────────────────────────────────────────────
# Auth + scope failure modes
# ──────────────────────────────────────────────────────────────────────


def test_auth_missing_returns_structured_error():
    """Empty GH_TOKEN → {success: False, error: 'auth_missing'}."""

    def _never_called(req):  # pragma: no cover - should not be reached
        raise AssertionError("HTTP call should not happen when token is missing")

    client = make_mock_client(_never_called, token="")
    result = run(client.resolve_repo_node_id())
    assert result == {
        "success": False,
        "error": "auth_missing",
        "remediation": (
            "GH_TOKEN environment variable is empty. Set it on the "
            "MCP server's process env (typically via "
            "~/.claude/mcp.json's gh_graphql.env block, written by "
            "scripts/pod.py config --sync)."
        ),
    }


def test_repo_scope_violation_returns_structured_error():
    """Mismatched (owner, name) → repo_scope_violation."""

    def _never_called(req):  # pragma: no cover
        raise AssertionError("scope check should fire before any HTTP call")

    client = make_mock_client(
        _never_called, repo_owner="superkaiba", repo_name="explore-persona-space"
    )
    violation = client.repo_scope_violation("someone-else", "other-repo")
    assert violation is not None
    assert violation["success"] is False
    assert violation["error"] == "repo_scope_violation"
    assert violation["configured_repo"] == "superkaiba/explore-persona-space"


def test_repo_scope_match_returns_none():
    """Matching (owner, name) → None (no violation)."""

    def _never(req):  # pragma: no cover
        raise AssertionError("not used")

    client = make_mock_client(_never)
    assert client.repo_scope_violation("superkaiba", "explore-persona-space") is None


def test_graphql_errors_array_surfaces_as_structured_error():
    """GitHub returning {errors: [...]} → success=False, error=graphql_error."""

    def handler(req):
        return gql_error([{"message": "bad query"}])

    client = make_mock_client(handler)
    result = run(client.resolve_repo_node_id())
    assert result["success"] is False
    assert result["error"] == "graphql_error"
    assert result["errors"] == [{"message": "bad query"}]


def test_http_5xx_surfaces_as_http_error():
    """500 from GitHub → success=False, error=http_error, status_code captured."""

    def handler(req):
        return httpx.Response(503, text="Service Unavailable")

    client = make_mock_client(handler)
    result = run(client.resolve_repo_node_id())
    assert result["success"] is False
    assert result["error"] == "http_error"
    assert result["status_code"] == 503


def test_transport_error_surfaces_as_transport_error():
    """httpx connection error → success=False, error=transport_error."""

    def handler(req):
        raise httpx.ConnectError("nope")

    client = make_mock_client(handler)
    result = run(client.resolve_repo_node_id())
    assert result["success"] is False
    assert result["error"] == "transport_error"
    assert result["exception_type"] == "ConnectError"


# ──────────────────────────────────────────────────────────────────────
# add_issue_comment — body-size cap
# ──────────────────────────────────────────────────────────────────────


def test_add_issue_comment_body_too_large_errors_structurally():
    """A 70 KB body → body_too_large error (no HTTP call)."""

    def _never(req):  # pragma: no cover
        raise AssertionError("HTTP should not be called for oversize bodies")

    client = make_mock_client(_never)
    big_body = "x" * 70_000
    result = run(gh_tools.add_issue_comment(client, issue_number=320, body=big_body))
    assert result["success"] is False
    assert result["error"] == "body_too_large"
    assert result["body_bytes"] == 70_000
    assert result["limit"] == ADD_COMMENT_BODY_MAX_BYTES
    assert "part=K/N" in result["remediation"]


def test_add_issue_comment_body_exactly_at_limit_passes_size_check():
    """A body of exactly 65,536 bytes should pass the size check."""
    calls: list[dict[str, Any]] = []

    def handler(req):
        body = parse_gql_request(req)
        calls.append(body)
        if "issue(number" in body["query"]:
            return gql_response({"repository": {"issue": {"id": "I_kw_test"}}})
        if "addComment" in body["query"]:
            return gql_response({"addComment": {"commentEdge": {"node": {"id": "x", "url": "u"}}}})
        return gql_error([{"message": "unexpected"}])  # pragma: no cover

    client = make_mock_client(handler)
    body_at_limit = "y" * ADD_COMMENT_BODY_MAX_BYTES
    assert len(body_at_limit.encode("utf-8")) == ADD_COMMENT_BODY_MAX_BYTES
    result = run(gh_tools.add_issue_comment(client, issue_number=320, body=body_at_limit))
    assert result["success"] is True
    # Two HTTP calls (resolve issue ID + addComment).
    assert len(calls) == 2


def test_update_issue_body_also_enforces_size_cap():
    """update_issue_body uses the same cap as add_issue_comment."""

    def _never(req):  # pragma: no cover
        raise AssertionError("size check should fire pre-HTTP")

    client = make_mock_client(_never)
    result = run(gh_tools.update_issue_body(client, issue_number=320, body="z" * 100_000))
    assert result["success"] is False
    assert result["error"] == "body_too_large"


# ──────────────────────────────────────────────────────────────────────
# Per-mutation happy paths (one each, with mocked transport)
# ──────────────────────────────────────────────────────────────────────


def make_resolve_issue_handler(issue_id: str = "I_kw_320") -> Callable:
    """Handler that always returns issue ID for the resolution step."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = parse_gql_request(req)
        q = body["query"]
        if "issue(number" in q and "labels(first:" not in q and "comments" not in q:
            return gql_response({"repository": {"issue": {"id": issue_id}}})
        # Default: echo back a tiny success.
        return gql_response({})

    return handler


def test_add_issue_labels_resolves_ids_and_calls_addLabelsToLabelable():
    """add_issue_labels: resolves issue ID + label IDs, then mutates."""
    seen: list[str] = []

    def handler(req):
        body = parse_gql_request(req)
        q = body["query"]
        seen.append(q)
        if "issue(number" in q and "labels" not in q:
            return gql_response({"repository": {"issue": {"id": "I_kw"}}})
        if "labels(first:" in q:
            return gql_response(
                {
                    "repository": {
                        "labels": {
                            "nodes": [
                                {"id": "L_status_running", "name": "status:running"},
                                {"id": "L_other", "name": "other"},
                            ]
                        }
                    }
                }
            )
        if "addLabelsToLabelable" in q:
            return gql_response({"addLabelsToLabelable": {"clientMutationId": None}})
        return gql_error([{"message": "unexpected query"}])

    client = make_mock_client(handler)
    result = run(gh_tools.add_issue_labels(client, issue_number=42, label_names=["status:running"]))
    assert result["success"] is True
    assert any("addLabelsToLabelable" in q for q in seen)


def test_add_issue_labels_missing_label_returns_labels_not_found():
    """If the label doesn't exist on the repo → labels_not_found error."""

    def handler(req):
        q = parse_gql_request(req)["query"]
        if "issue(number" in q and "labels" not in q:
            return gql_response({"repository": {"issue": {"id": "I_kw"}}})
        if "labels(first:" in q:
            return gql_response(
                {"repository": {"labels": {"nodes": [{"id": "L_a", "name": "other"}]}}}
            )
        return gql_error([{"message": "should not reach"}])

    client = make_mock_client(handler)
    result = run(
        gh_tools.add_issue_labels(client, issue_number=42, label_names=["status:does-not-exist"])
    )
    assert result["success"] is False
    assert result["error"] == "labels_not_found"
    assert result["missing"] == ["status:does-not-exist"]


def test_remove_issue_labels_uses_removeLabelsFromLabelable():
    seen_queries: list[str] = []

    def handler(req):
        q = parse_gql_request(req)["query"]
        seen_queries.append(q)
        if "issue(number" in q and "labels" not in q:
            return gql_response({"repository": {"issue": {"id": "I_kw"}}})
        if "labels(first:" in q:
            return gql_response(
                {"repository": {"labels": {"nodes": [{"id": "L_x", "name": "old"}]}}}
            )
        if "removeLabelsFromLabelable" in q:
            return gql_response({"removeLabelsFromLabelable": {"clientMutationId": None}})
        return gql_error([{"message": "unexpected"}])

    client = make_mock_client(handler)
    result = run(gh_tools.remove_issue_labels(client, issue_number=42, label_names=["old"]))
    assert result["success"] is True
    assert any("removeLabelsFromLabelable" in q for q in seen_queries)


def test_update_issue_title_passes_title_only():
    seen_vars: list[dict[str, Any]] = []

    def handler(req):
        body = parse_gql_request(req)
        q = body["query"]
        if "issue(number" in q:
            return gql_response({"repository": {"issue": {"id": "I_kw"}}})
        if "updateIssue" in q:
            seen_vars.append(body["variables"])
            return gql_response(
                {"updateIssue": {"issue": {"id": "I_kw", "number": 42, "title": "new"}}}
            )
        return gql_error([{"message": "unexpected"}])

    client = make_mock_client(handler)
    result = run(gh_tools.update_issue_title(client, issue_number=42, title="new"))
    assert result["success"] is True
    assert seen_vars == [{"id": "I_kw", "title": "new"}]
    # Confirm body NOT in variables (only title was passed).
    assert "body" not in seen_vars[0]


def test_create_issue_resolves_repo_id_and_creates():
    def handler(req):
        body = parse_gql_request(req)
        q = body["query"]
        if "repository(owner" in q and "issue(number" not in q and "labels" not in q:
            return gql_response({"repository": {"id": "R_kw"}})
        if "createIssue" in q:
            return gql_response(
                {"createIssue": {"issue": {"id": "I_new", "number": 999, "url": "https://x"}}}
            )
        return gql_error([{"message": "unexpected"}])

    client = make_mock_client(handler)
    result = run(gh_tools.create_issue(client, title="T", body="B"))
    assert result["success"] is True
    assert result["data"]["createIssue"]["issue"]["number"] == 999


def test_close_issue_uses_closeIssue_mutation():
    def handler(req):
        q = parse_gql_request(req)["query"]
        if "issue(number" in q:
            return gql_response({"repository": {"issue": {"id": "I_kw"}}})
        if "closeIssue" in q:
            return gql_response({"closeIssue": {"issue": {"id": "I_kw", "state": "CLOSED"}}})
        return gql_error([{"message": "unexpected"}])

    client = make_mock_client(handler)
    result = run(gh_tools.close_issue(client, issue_number=42))
    assert result["success"] is True


def test_reopen_issue_uses_reopenIssue_mutation():
    def handler(req):
        q = parse_gql_request(req)["query"]
        if "issue(number" in q:
            return gql_response({"repository": {"issue": {"id": "I_kw"}}})
        if "reopenIssue" in q:
            return gql_response({"reopenIssue": {"issue": {"id": "I_kw", "state": "OPEN"}}})
        return gql_error([{"message": "unexpected"}])

    client = make_mock_client(handler)
    result = run(gh_tools.reopen_issue(client, issue_number=42))
    assert result["success"] is True


def test_create_pull_request_passes_branches_and_draft_flag():
    seen_vars: list[dict[str, Any]] = []

    def handler(req):
        body = parse_gql_request(req)
        q = body["query"]
        if "repository(owner" in q and "issue" not in q and "labels" not in q:
            return gql_response({"repository": {"id": "R_kw"}})
        if "createPullRequest" in q:
            seen_vars.append(body["variables"])
            return gql_response(
                {"createPullRequest": {"pullRequest": {"id": "PR_x", "number": 1, "url": "u"}}}
            )
        return gql_error([{"message": "unexpected"}])

    client = make_mock_client(handler)
    result = run(
        gh_tools.create_pull_request(
            client,
            title="T",
            body="B",
            head_branch="my-branch",
            base_branch="main",
            draft=True,
        )
    )
    assert result["success"] is True
    v = seen_vars[0]
    assert v["head"] == "my-branch"
    assert v["base"] == "main"
    assert v["draft"] is True


def test_mark_pr_ready_for_review_takes_node_id():
    def handler(req):
        body = parse_gql_request(req)
        q = body["query"]
        assert "markPullRequestReadyForReview" in q
        assert body["variables"] == {"id": "PR_xyz"}
        return gql_response(
            {"markPullRequestReadyForReview": {"pullRequest": {"id": "PR_xyz", "isDraft": False}}}
        )

    client = make_mock_client(handler)
    result = run(gh_tools.mark_pr_ready_for_review(client, pull_request_id="PR_xyz"))
    assert result["success"] is True


def test_merge_pull_request_validates_method():
    """Invalid merge_method → invalid_merge_method (no HTTP call)."""

    def _never(req):  # pragma: no cover
        raise AssertionError("HTTP should not fire for invalid method")

    client = make_mock_client(_never)
    result = run(
        gh_tools.merge_pull_request(client, pull_request_id="PR_x", merge_method="OBLITERATE")
    )
    assert result["success"] is False
    assert result["error"] == "invalid_merge_method"


def test_merge_pull_request_passes_method_through():
    seen_vars: list[dict[str, Any]] = []

    def handler(req):
        body = parse_gql_request(req)
        seen_vars.append(body["variables"])
        return gql_response(
            {"mergePullRequest": {"pullRequest": {"id": "PR_x", "state": "MERGED", "merged": True}}}
        )

    client = make_mock_client(handler)
    result = run(
        gh_tools.merge_pull_request(
            client,
            pull_request_id="PR_x",
            merge_method="REBASE",
            commit_headline="hdr",
        )
    )
    assert result["success"] is True
    assert seen_vars[0]["method"] == "REBASE"
    assert seen_vars[0]["headline"] == "hdr"


def test_update_project_v2_field_passes_value_through():
    seen_vars: list[dict[str, Any]] = []

    def handler(req):
        body = parse_gql_request(req)
        seen_vars.append(body["variables"])
        return gql_response({"updateProjectV2ItemFieldValue": {"projectV2Item": {"id": "PVTI_x"}}})

    client = make_mock_client(handler)
    result = run(
        gh_tools.update_project_v2_field(
            client,
            project_id="PVT_a",
            item_id="PVTI_b",
            field_id="PVTF_c",
            value={"singleSelectOptionId": "abc"},
        )
    )
    assert result["success"] is True
    assert seen_vars[0]["value"] == {"singleSelectOptionId": "abc"}


def test_read_issue_does_not_include_comments_by_default():
    def handler(req):
        q = parse_gql_request(req)["query"]
        # Default path must NOT request comments.
        assert "comments(first:" not in q
        return gql_response(
            {
                "repository": {
                    "issue": {
                        "id": "I_kw",
                        "number": 320,
                        "title": "x",
                        "body": "y",
                        "state": "OPEN",
                        "createdAt": "2026-01-01T00:00:00Z",
                        "updatedAt": "2026-01-01T00:00:00Z",
                        "labels": {"nodes": []},
                    }
                }
            }
        )

    client = make_mock_client(handler)
    result = run(gh_tools.read_issue(client, issue_number=320))
    assert result["success"] is True


def test_read_issue_includes_comments_when_requested():
    def handler(req):
        q = parse_gql_request(req)["query"]
        assert "comments(first:" in q
        return gql_response(
            {
                "repository": {
                    "issue": {
                        "id": "I_kw",
                        "number": 320,
                        "title": "x",
                        "body": "y",
                        "state": "OPEN",
                        "createdAt": "2026-01-01T00:00:00Z",
                        "updatedAt": "2026-01-01T00:00:00Z",
                        "labels": {"nodes": []},
                        "comments": {"nodes": []},
                    }
                }
            }
        )

    client = make_mock_client(handler)
    result = run(gh_tools.read_issue(client, issue_number=320, include_comments=True))
    assert result["success"] is True


# ──────────────────────────────────────────────────────────────────────
# Server-side smoke: the CLI --list-tools path
# ──────────────────────────────────────────────────────────────────────


def test_cli_list_tools_prints_allow_list(capsys):
    """`epm-gh-graphql-mcp --list-tools` prints exactly the allow-list."""
    pytest.importorskip("mcp")
    from explore_persona_space.mcp_servers.gh_graphql.server import main

    rc = main(["--list-tools"])
    assert rc == 0
    captured = capsys.readouterr()
    printed = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert set(printed) == set(gh_tools.MUTATION_TOOL_NAMES)
