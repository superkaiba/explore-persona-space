"""Mutation allow-list for the gh_graphql MCP server.

Each public function corresponds to ONE row of the table in plan §3 of
issue #320. Tools NOT in this module are NOT exposed (the implementer
must register them explicitly in :func:`server.build_server`); a model
that asks for a non-registered tool gets a standard MCP "unknown tool"
error.

The allow-list (13 rows; matches plan §3 verbatim):

* ``add_issue_comment``      → ``addComment``
* ``add_issue_labels``       → ``addLabelsToLabelable``
* ``remove_issue_labels``    → ``removeLabelsFromLabelable``
* ``update_issue_title``     → ``updateIssue`` (title only)
* ``update_issue_body``      → ``updateIssue`` (body only)
* ``create_issue``           → ``createIssue``
* ``close_issue``            → ``closeIssue``
* ``reopen_issue``           → ``reopenIssue``
* ``create_pull_request``    → ``createPullRequest``
* ``mark_pr_ready_for_review``→ ``markPullRequestReadyForReview``
* ``merge_pull_request``     → ``mergePullRequest``
* ``update_project_v2_field``→ ``updateProjectV2ItemFieldValue``
* ``read_issue``             → ``repository.issue(number)`` query (read-side; kept for parity)
"""

from __future__ import annotations

from typing import Any

from .client import GhGraphQLClient

# GitHub GraphQL `addComment.body` server-side cap (UTF-8 bytes).
# https://docs.github.com/en/graphql/reference/mutations#addcomment — the
# `body` argument is constrained to the same 65,536-byte limit as
# `issues.body` and `issueComments.body`. We enforce it locally so
# callers get a structured error rather than an opaque GitHub 4xx.
ADD_COMMENT_BODY_MAX_BYTES = 65_536

# Convenience: mutation tool names. Server registration uses this to
# decide what to expose; the denylist (archive_repository, transfer_issue,
# delete_issue, delete_repository, create_repository, update_repository,
# project mutations beyond update_project_v2_field, GraphQL introspection)
# is enforced by *omission* — those tools are simply never registered.
MUTATION_TOOL_NAMES: tuple[str, ...] = (
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
)


# ──────────────────────────────────────────────────────────────────────
# Issue comments / labels / title / body
# ──────────────────────────────────────────────────────────────────────


async def add_issue_comment(
    client: GhGraphQLClient, *, issue_number: int, body: str
) -> dict[str, Any]:
    """Add a comment to ``issue_number`` in the configured repository.

    Errors with ``body_too_large`` (structured response, NOT a raised
    exception) if ``len(body.encode('utf-8'))`` exceeds
    :data:`ADD_COMMENT_BODY_MAX_BYTES`. Skill-side callers MUST wrap
    every invocation with a ``body_too_large → status:blocked`` handler
    (see ``.claude/skills/issue/SKILL.md``).
    """
    body_bytes = len(body.encode("utf-8"))
    if body_bytes > ADD_COMMENT_BODY_MAX_BYTES:
        return {
            "success": False,
            "error": "body_too_large",
            "body_bytes": body_bytes,
            "limit": ADD_COMMENT_BODY_MAX_BYTES,
            "remediation": (
                "Split the body into multiple comments and chain them with "
                "an explicit `part=K/N` field in the marker title. Do not "
                "auto-truncate or shell out to --body-file."
            ),
        }

    issue_id_resp = await client.resolve_issue_node_id(issue_number)
    if not issue_id_resp.get("success"):
        return issue_id_resp
    issue = (issue_id_resp.get("data") or {}).get("repository", {}).get("issue")
    if not issue:
        return {"success": False, "error": "issue_not_found", "issue_number": issue_number}

    return await client.mutate(
        query="""mutation ($subject: ID!, $body: String!) {
            addComment(input: {subjectId: $subject, body: $body}) {
                commentEdge { node { id url } }
            }
        }""",
        variables={"subject": issue["id"], "body": body},
    )


async def _set_labels_helper(
    client: GhGraphQLClient,
    *,
    issue_number: int,
    label_names: list[str],
    add: bool,
) -> dict[str, Any]:
    if not label_names:
        return {"success": False, "error": "label_names_empty"}
    issue_id_resp = await client.resolve_issue_node_id(issue_number)
    if not issue_id_resp.get("success"):
        return issue_id_resp
    issue = (issue_id_resp.get("data") or {}).get("repository", {}).get("issue")
    if not issue:
        return {"success": False, "error": "issue_not_found", "issue_number": issue_number}

    label_resp = await client.resolve_label_ids(label_names)
    if not label_resp.get("success"):
        return label_resp
    nodes = (
        ((label_resp.get("data") or {}).get("repository") or {}).get("labels", {}).get("nodes", [])
    )
    by_name = {n["name"]: n["id"] for n in nodes}
    missing = [n for n in label_names if n not in by_name]
    if missing:
        return {"success": False, "error": "labels_not_found", "missing": missing}
    label_ids = [by_name[n] for n in label_names]

    mutation_name = "addLabelsToLabelable" if add else "removeLabelsFromLabelable"
    return await client.mutate(
        query=f"""mutation ($labelable: ID!, $labelIds: [ID!]!) {{
            {mutation_name}(input: {{labelableId: $labelable, labelIds: $labelIds}}) {{
                clientMutationId
            }}
        }}""",
        variables={"labelable": issue["id"], "labelIds": label_ids},
    )


async def add_issue_labels(
    client: GhGraphQLClient, *, issue_number: int, label_names: list[str]
) -> dict[str, Any]:
    """Add ``label_names`` to ``issue_number``."""
    return await _set_labels_helper(
        client, issue_number=issue_number, label_names=label_names, add=True
    )


async def remove_issue_labels(
    client: GhGraphQLClient, *, issue_number: int, label_names: list[str]
) -> dict[str, Any]:
    """Remove ``label_names`` from ``issue_number``."""
    return await _set_labels_helper(
        client, issue_number=issue_number, label_names=label_names, add=False
    )


async def update_issue_title(
    client: GhGraphQLClient, *, issue_number: int, title: str
) -> dict[str, Any]:
    """Update the title of ``issue_number`` (body left unchanged)."""
    issue_id_resp = await client.resolve_issue_node_id(issue_number)
    if not issue_id_resp.get("success"):
        return issue_id_resp
    issue = (issue_id_resp.get("data") or {}).get("repository", {}).get("issue")
    if not issue:
        return {"success": False, "error": "issue_not_found", "issue_number": issue_number}

    return await client.mutate(
        query="""mutation ($id: ID!, $title: String!) {
            updateIssue(input: {id: $id, title: $title}) {
                issue { id number title }
            }
        }""",
        variables={"id": issue["id"], "title": title},
    )


async def update_issue_body(
    client: GhGraphQLClient, *, issue_number: int, body: str
) -> dict[str, Any]:
    """Update the body of ``issue_number`` (title left unchanged).

    Like :func:`add_issue_comment`, errors out if the body exceeds the
    65,536-byte cap — issues and comments share the same server-side
    limit on GitHub's GraphQL API.
    """
    body_bytes = len(body.encode("utf-8"))
    if body_bytes > ADD_COMMENT_BODY_MAX_BYTES:
        return {
            "success": False,
            "error": "body_too_large",
            "body_bytes": body_bytes,
            "limit": ADD_COMMENT_BODY_MAX_BYTES,
            "remediation": (
                "Issue bodies share the 65,536-byte cap with comments. "
                "Move long content into a comment and link from the body."
            ),
        }
    issue_id_resp = await client.resolve_issue_node_id(issue_number)
    if not issue_id_resp.get("success"):
        return issue_id_resp
    issue = (issue_id_resp.get("data") or {}).get("repository", {}).get("issue")
    if not issue:
        return {"success": False, "error": "issue_not_found", "issue_number": issue_number}

    return await client.mutate(
        query="""mutation ($id: ID!, $body: String!) {
            updateIssue(input: {id: $id, body: $body}) {
                issue { id number }
            }
        }""",
        variables={"id": issue["id"], "body": body},
    )


# ──────────────────────────────────────────────────────────────────────
# Issue lifecycle
# ──────────────────────────────────────────────────────────────────────


async def create_issue(
    client: GhGraphQLClient,
    *,
    title: str,
    body: str,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new issue in the configured repository."""
    body_bytes = len(body.encode("utf-8"))
    if body_bytes > ADD_COMMENT_BODY_MAX_BYTES:
        return {
            "success": False,
            "error": "body_too_large",
            "body_bytes": body_bytes,
            "limit": ADD_COMMENT_BODY_MAX_BYTES,
        }
    repo_resp = await client.resolve_repo_node_id()
    if not repo_resp.get("success"):
        return repo_resp
    repo = (repo_resp.get("data") or {}).get("repository")
    if not repo:
        return {"success": False, "error": "repo_not_found"}

    label_ids: list[str] = []
    if label_names:
        label_resp = await client.resolve_label_ids(label_names)
        if not label_resp.get("success"):
            return label_resp
        nodes = (
            ((label_resp.get("data") or {}).get("repository") or {})
            .get("labels", {})
            .get("nodes", [])
        )
        by_name = {n["name"]: n["id"] for n in nodes}
        missing = [n for n in label_names if n not in by_name]
        if missing:
            return {"success": False, "error": "labels_not_found", "missing": missing}
        label_ids = [by_name[n] for n in label_names]

    return await client.mutate(
        query="""mutation ($repo: ID!, $title: String!, $body: String!, $labels: [ID!]) {
            createIssue(input: {
                repositoryId: $repo,
                title: $title,
                body: $body,
                labelIds: $labels
            }) {
                issue { id number url }
            }
        }""",
        variables={
            "repo": repo["id"],
            "title": title,
            "body": body,
            "labels": label_ids or None,
        },
    )


async def close_issue(client: GhGraphQLClient, *, issue_number: int) -> dict[str, Any]:
    """Close ``issue_number``. Reserved for manual user-driven calls per CLAUDE.md."""
    issue_id_resp = await client.resolve_issue_node_id(issue_number)
    if not issue_id_resp.get("success"):
        return issue_id_resp
    issue = (issue_id_resp.get("data") or {}).get("repository", {}).get("issue")
    if not issue:
        return {"success": False, "error": "issue_not_found", "issue_number": issue_number}
    return await client.mutate(
        query="""mutation ($id: ID!) {
            closeIssue(input: {issueId: $id}) { issue { id state } }
        }""",
        variables={"id": issue["id"]},
    )


async def reopen_issue(client: GhGraphQLClient, *, issue_number: int) -> dict[str, Any]:
    """Reopen ``issue_number``."""
    issue_id_resp = await client.resolve_issue_node_id(issue_number)
    if not issue_id_resp.get("success"):
        return issue_id_resp
    issue = (issue_id_resp.get("data") or {}).get("repository", {}).get("issue")
    if not issue:
        return {"success": False, "error": "issue_not_found", "issue_number": issue_number}
    return await client.mutate(
        query="""mutation ($id: ID!) {
            reopenIssue(input: {issueId: $id}) { issue { id state } }
        }""",
        variables={"id": issue["id"]},
    )


# ──────────────────────────────────────────────────────────────────────
# Pull requests
# ──────────────────────────────────────────────────────────────────────


async def create_pull_request(
    client: GhGraphQLClient,
    *,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str = "main",
    draft: bool = False,
) -> dict[str, Any]:
    """Create a new pull request in the configured repository."""
    body_bytes = len(body.encode("utf-8"))
    if body_bytes > ADD_COMMENT_BODY_MAX_BYTES:
        return {
            "success": False,
            "error": "body_too_large",
            "body_bytes": body_bytes,
            "limit": ADD_COMMENT_BODY_MAX_BYTES,
        }
    repo_resp = await client.resolve_repo_node_id()
    if not repo_resp.get("success"):
        return repo_resp
    repo = (repo_resp.get("data") or {}).get("repository")
    if not repo:
        return {"success": False, "error": "repo_not_found"}

    return await client.mutate(
        query="""mutation (
            $repo: ID!, $title: String!, $body: String!,
            $head: String!, $base: String!, $draft: Boolean!
        ) {
            createPullRequest(input: {
                repositoryId: $repo,
                title: $title,
                body: $body,
                headRefName: $head,
                baseRefName: $base,
                draft: $draft
            }) {
                pullRequest { id number url }
            }
        }""",
        variables={
            "repo": repo["id"],
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": draft,
        },
    )


async def mark_pr_ready_for_review(
    client: GhGraphQLClient, *, pull_request_id: str
) -> dict[str, Any]:
    """Mark a PR as ready for review (un-draft).

    Takes a node ID, not a number — node IDs round-trip cheaper than
    re-resolving via a query each call. Callers can get the ID from
    :func:`create_pull_request` or from a separate query.
    """
    return await client.mutate(
        query="""mutation ($id: ID!) {
            markPullRequestReadyForReview(input: {pullRequestId: $id}) {
                pullRequest { id isDraft }
            }
        }""",
        variables={"id": pull_request_id},
    )


async def merge_pull_request(
    client: GhGraphQLClient,
    *,
    pull_request_id: str,
    merge_method: str = "REBASE",
    commit_headline: str | None = None,
    commit_body: str | None = None,
) -> dict[str, Any]:
    """Merge a PR. ``merge_method`` ∈ {``MERGE``, ``SQUASH``, ``REBASE``}."""
    if merge_method not in ("MERGE", "SQUASH", "REBASE"):
        return {
            "success": False,
            "error": "invalid_merge_method",
            "remediation": "merge_method must be one of MERGE / SQUASH / REBASE.",
        }
    return await client.mutate(
        query="""mutation (
            $id: ID!, $method: PullRequestMergeMethod!,
            $headline: String, $body: String
        ) {
            mergePullRequest(input: {
                pullRequestId: $id,
                mergeMethod: $method,
                commitHeadline: $headline,
                commitBody: $body
            }) {
                pullRequest { id state merged }
            }
        }""",
        variables={
            "id": pull_request_id,
            "method": merge_method,
            "headline": commit_headline,
            "body": commit_body,
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Project board
# ──────────────────────────────────────────────────────────────────────


async def update_project_v2_field(
    client: GhGraphQLClient,
    *,
    project_id: str,
    item_id: str,
    field_id: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    """Update a single ProjectV2 item field. Mirrors gh_project.py's only mutation.

    ``value`` is the raw GraphQL input variant — e.g.
    ``{"singleSelectOptionId": "..."}`` or ``{"text": "..."}``. Callers
    construct the variant; the MCP does not interpret it.
    """
    return await client.mutate(
        query="""mutation (
            $project: ID!, $item: ID!, $field: ID!, $value: ProjectV2FieldValue!
        ) {
            updateProjectV2ItemFieldValue(input: {
                projectId: $project,
                itemId: $item,
                fieldId: $field,
                value: $value
            }) {
                projectV2Item { id }
            }
        }""",
        variables={
            "project": project_id,
            "item": item_id,
            "field": field_id,
            "value": value,
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Read path (kept for parity per plan §3 read-side note)
# ──────────────────────────────────────────────────────────────────────


async def read_issue(
    client: GhGraphQLClient,
    *,
    issue_number: int,
    include_comments: bool = False,
) -> dict[str, Any]:
    """Read an issue by number. Optionally include the first 100 comments.

    The skill keeps using ``gh issue view`` for read-side calls (no
    auth-leak risk on reads); this tool exists for parity with the table
    in plan §3 and for tests that only need a read path.
    """
    if include_comments:
        query = """query ($owner: String!, $name: String!, $n: Int!) {
            repository(owner: $owner, name: $name) {
                issue(number: $n) {
                    id number title body state createdAt updatedAt
                    labels(first: 50) { nodes { name } }
                    comments(first: 100) { nodes { id body createdAt } }
                }
            }
        }"""
    else:
        query = """query ($owner: String!, $name: String!, $n: Int!) {
            repository(owner: $owner, name: $name) {
                issue(number: $n) {
                    id number title body state createdAt updatedAt
                    labels(first: 50) { nodes { name } }
                }
            }
        }"""
    return await client.mutate(
        query=query,
        variables={
            "owner": client.repo_owner,
            "name": client.repo_name,
            "n": issue_number,
        },
    )
