"""GraphQL client for the gh_graphql MCP server.

Holds ``GH_TOKEN`` and the configured repo (owner/name) for the lifetime of
the server process. All public methods return structured dicts so the MCP
tool layer can pass them straight back to the caller without re-mapping
exceptions to MCP errors.

Failure-mode contract (the tool layer relies on these):

* Auth missing → ``{"success": False, "error": "auth_missing", ...}``
* Repo scope mismatch → ``{"success": False, "error": "repo_scope_violation", ...}``
* Network / 5xx → ``{"success": False, "error": "transport_error", ...}``
* GraphQL errors array → ``{"success": False, "error": "graphql_error", "errors": [...]}``
"""

from __future__ import annotations

import os
from typing import Any

import httpx

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
DEFAULT_TIMEOUT_S = 30.0


class GhGraphQLClient:
    """Thin async-friendly wrapper around the GitHub GraphQL endpoint.

    The client is *intentionally* simple: it does not retry, does not cache,
    and does not transform GitHub's error envelope beyond wrapping it in a
    success/error dict. Retries are the orchestrator's job (the skill posts
    ``epm:failure`` markers for transport errors).
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token if token is not None else os.environ.get("GH_TOKEN", "")
        self._repo_owner = (
            repo_owner if repo_owner is not None else os.environ.get("GH_REPO_OWNER", "")
        )
        self._repo_name = repo_name if repo_name is not None else os.environ.get("GH_REPO_NAME", "")
        self._http = http_client  # injected for tests; lazily constructed otherwise

    @property
    def repo_owner(self) -> str:
        return self._repo_owner

    @property
    def repo_name(self) -> str:
        return self._repo_name

    @property
    def configured(self) -> bool:
        return bool(self._token and self._repo_owner and self._repo_name)

    async def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """POST a GraphQL document. Raises only on programming errors.

        All transport / 4xx / 5xx / GraphQL-error conditions are encoded in
        the returned dict.
        """
        if not self._token:
            return {
                "success": False,
                "error": "auth_missing",
                "remediation": (
                    "GH_TOKEN environment variable is empty. Set it on the "
                    "MCP server's process env (typically via "
                    "~/.claude/mcp.json's gh_graphql.env block, written by "
                    "scripts/pod.py config --sync)."
                ),
            }

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "epm-gh-graphql-mcp/0.1",
        }
        payload = {"query": query, "variables": variables}

        client = self._http or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S)
        owns_client = self._http is None
        try:
            try:
                resp = await client.post(GITHUB_GRAPHQL_URL, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                return {
                    "success": False,
                    "error": "transport_error",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            if resp.status_code >= 400:
                return {
                    "success": False,
                    "error": "http_error",
                    "status_code": resp.status_code,
                    "body": resp.text[:1000],
                }
            try:
                doc = resp.json()
            except ValueError as exc:
                return {
                    "success": False,
                    "error": "decode_error",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            if doc.get("errors"):
                return {
                    "success": False,
                    "error": "graphql_error",
                    "errors": doc["errors"],
                    "data": doc.get("data"),
                }
            return {"success": True, "data": doc.get("data")}
        finally:
            if owns_client:
                await client.aclose()

    def repo_scope_violation(self, owner: str, name: str) -> dict[str, Any] | None:
        """Return a structured error if (owner, name) is not the configured repo.

        Defense-in-depth: the fine-grained PAT is supposed to be scoped to
        the configured repo, but we still refuse here so an obviously-wrong
        call fails fast with a recognizable error rather than a 403 from
        GitHub.
        """
        if owner != self._repo_owner or name != self._repo_name:
            return {
                "success": False,
                "error": "repo_scope_violation",
                "configured_repo": f"{self._repo_owner}/{self._repo_name}",
                "requested_repo": f"{owner}/{name}",
            }
        return None

    async def resolve_issue_node_id(self, issue_number: int) -> dict[str, Any]:
        """Resolve issue number → GraphQL node ID for the configured repo."""
        return await self._post(
            query="""query ($owner: String!, $name: String!, $n: Int!) {
                repository(owner: $owner, name: $name) {
                    issue(number: $n) { id }
                }
            }""",
            variables={"owner": self._repo_owner, "name": self._repo_name, "n": issue_number},
        )

    async def resolve_repo_node_id(self) -> dict[str, Any]:
        """Resolve owner/name → GraphQL node ID for the configured repo."""
        return await self._post(
            query="""query ($owner: String!, $name: String!) {
                repository(owner: $owner, name: $name) { id }
            }""",
            variables={"owner": self._repo_owner, "name": self._repo_name},
        )

    async def resolve_label_ids(self, names: list[str]) -> dict[str, Any]:
        """Resolve label names → label node IDs in the configured repo."""
        return await self._post(
            query="""query ($owner: String!, $name: String!) {
                repository(owner: $owner, name: $name) {
                    labels(first: 100) { nodes { id name } }
                }
            }""",
            variables={"owner": self._repo_owner, "name": self._repo_name},
        )

    async def mutate(self, *, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Generic GraphQL mutation passthrough (used by tools.py)."""
        return await self._post(query=query, variables=variables)
