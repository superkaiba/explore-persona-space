"""gh_graphql MCP server.

A small Python MCP server that exposes a hand-curated allow-list of GitHub
GraphQL mutations to subagents WITHOUT exposing ``GH_TOKEN`` to the agent
context window. The server reads ``GH_TOKEN`` (and ``GH_REPO_OWNER`` /
``GH_REPO_NAME``) from its own process environment at startup; agents call
named tools (e.g. ``add_issue_comment``) and the server proxies the call
to ``api.github.com/graphql``.

Design pointers (cf. plan §3 of issue #320):

- The mutation **allow-list** is enumerated in :mod:`.tools`. Tools that
  could be destructive at the repo level (``archiveRepository``,
  ``transferIssue``, ``deleteIssue``, ``deleteRepository``,
  ``createRepository``, ``updateRepository``, generic project mutations
  beyond ``updateProjectV2ItemFieldValue``, and GraphQL introspection)
  are **never registered**. A model that asks for them gets a standard
  MCP "unknown tool" error from the framework.
- ``add_issue_comment`` enforces the GitHub GraphQL ``addComment.body``
  65,536-byte cap; oversize input returns ``{"success": false,
  "error": "body_too_large", ...}`` rather than silently truncating or
  shelling out to ``--body-file``. Callers MUST split the body
  themselves and chain with a ``part=K/N`` continuation marker.

Run as ``uvx --from <repo> epm-gh-graphql-mcp`` (stdio transport) or
``python -m explore_persona_space.mcp_servers.gh_graphql``.
"""

from .server import build_server, main

__all__ = ["build_server", "main"]
