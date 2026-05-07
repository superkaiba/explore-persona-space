"""FastMCP server entry point for gh_graphql.

Boots a stdio-transport MCP server that exposes the allow-listed
GitHub GraphQL mutations from :mod:`.tools`. The denylist is enforced
by **omission**: tools not registered here are not in the list-tools
response, so a model that asks for ``archive_repository`` (or any
other denylisted name) gets a standard "unknown tool" error from the
MCP framework.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from . import tools as gh_tools
from .client import GhGraphQLClient

logger = logging.getLogger(__name__)


def build_server(
    *,
    client: GhGraphQLClient | None = None,
    server_name: str = "gh_graphql",
):
    """Construct and return a FastMCP server with all allow-listed tools registered.

    The ``mcp`` import is lazy so that ``--help`` (and unit tests that
    don't need the FastMCP runtime) can run without the SDK installed.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(server_name)
    gh_client = client or GhGraphQLClient()

    # Each registration below mirrors one row of the allow-list table in
    # plan §3 of issue #320. Adding a new tool here means updating
    # tools.MUTATION_TOOL_NAMES *and* the table in the plan.

    @mcp.tool()
    async def add_issue_comment(issue_number: int, body: str) -> dict[str, Any]:
        """Add a markdown comment to ``issue_number`` in the configured repo.

        Errors with ``body_too_large`` if ``len(body.encode('utf-8'))``
        exceeds 65,536 bytes. Callers MUST split oversize content
        themselves and chain comments via ``part=K/N`` continuation
        markers.
        """
        return await gh_tools.add_issue_comment(gh_client, issue_number=issue_number, body=body)

    @mcp.tool()
    async def add_issue_labels(issue_number: int, label_names: list[str]) -> dict[str, Any]:
        """Add labels to an issue. ``label_names`` is a list of label strings."""
        return await gh_tools.add_issue_labels(
            gh_client, issue_number=issue_number, label_names=label_names
        )

    @mcp.tool()
    async def remove_issue_labels(issue_number: int, label_names: list[str]) -> dict[str, Any]:
        """Remove labels from an issue."""
        return await gh_tools.remove_issue_labels(
            gh_client, issue_number=issue_number, label_names=label_names
        )

    @mcp.tool()
    async def update_issue_title(issue_number: int, title: str) -> dict[str, Any]:
        """Update an issue's title."""
        return await gh_tools.update_issue_title(gh_client, issue_number=issue_number, title=title)

    @mcp.tool()
    async def update_issue_body(issue_number: int, body: str) -> dict[str, Any]:
        """Update an issue's body. Subject to the same 65,536-byte cap as comments."""
        return await gh_tools.update_issue_body(gh_client, issue_number=issue_number, body=body)

    @mcp.tool()
    async def create_issue(
        title: str, body: str, label_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Create a new issue in the configured repository."""
        return await gh_tools.create_issue(
            gh_client, title=title, body=body, label_names=label_names
        )

    @mcp.tool()
    async def close_issue(issue_number: int) -> dict[str, Any]:
        """Close an issue. Reserved for manual user-driven calls per CLAUDE.md."""
        return await gh_tools.close_issue(gh_client, issue_number=issue_number)

    @mcp.tool()
    async def reopen_issue(issue_number: int) -> dict[str, Any]:
        """Reopen a closed issue."""
        return await gh_tools.reopen_issue(gh_client, issue_number=issue_number)

    @mcp.tool()
    async def create_pull_request(
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        draft: bool = False,
    ) -> dict[str, Any]:
        """Create a pull request from ``head_branch`` into ``base_branch``."""
        return await gh_tools.create_pull_request(
            gh_client,
            title=title,
            body=body,
            head_branch=head_branch,
            base_branch=base_branch,
            draft=draft,
        )

    @mcp.tool()
    async def mark_pr_ready_for_review(pull_request_id: str) -> dict[str, Any]:
        """Mark a draft PR as ready for review (un-draft)."""
        return await gh_tools.mark_pr_ready_for_review(gh_client, pull_request_id=pull_request_id)

    @mcp.tool()
    async def merge_pull_request(
        pull_request_id: str,
        merge_method: str = "REBASE",
        commit_headline: str | None = None,
        commit_body: str | None = None,
    ) -> dict[str, Any]:
        """Merge a PR. ``merge_method`` is one of MERGE, SQUASH, REBASE."""
        return await gh_tools.merge_pull_request(
            gh_client,
            pull_request_id=pull_request_id,
            merge_method=merge_method,
            commit_headline=commit_headline,
            commit_body=commit_body,
        )

    @mcp.tool()
    async def update_project_v2_field(
        project_id: str, item_id: str, field_id: str, value: dict[str, Any]
    ) -> dict[str, Any]:
        """Update a single ProjectV2 item field (the only project mutation we expose)."""
        return await gh_tools.update_project_v2_field(
            gh_client,
            project_id=project_id,
            item_id=item_id,
            field_id=field_id,
            value=value,
        )

    @mcp.tool()
    async def read_issue(issue_number: int, include_comments: bool = False) -> dict[str, Any]:
        """Read a single issue. Kept for parity; the skill prefers ``gh issue view``."""
        return await gh_tools.read_issue(
            gh_client, issue_number=issue_number, include_comments=include_comments
        )

    return mcp


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Run as ``python -m … gh_graphql`` or ``epm-gh-graphql-mcp``."""
    parser = argparse.ArgumentParser(
        prog="epm-gh-graphql-mcp",
        description=(
            "MCP server exposing a hand-curated allow-list of GitHub GraphQL "
            "mutations. Reads GH_TOKEN / GH_REPO_OWNER / GH_REPO_NAME from "
            "process env."
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the registered tool names and exit. Useful for CI sanity.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging to stderr."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.list_tools:
        for name in gh_tools.MUTATION_TOOL_NAMES:
            print(name)
        return 0

    client = GhGraphQLClient()
    if not client.configured:
        logger.warning(
            "gh_graphql server starting WITHOUT full configuration (token=%s, "
            "owner=%s, name=%s). Tool calls will return auth_missing or "
            "repo_scope_violation until the env is set.",
            "set" if client._token else "MISSING",
            client.repo_owner or "MISSING",
            client.repo_name or "MISSING",
        )

    mcp = build_server(client=client)
    mcp.run(transport="stdio")
    return 0
