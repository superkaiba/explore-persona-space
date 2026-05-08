---
name: Body-promote is the canonical clean-result publish path
description: `gh_project.py body-promote <N> path.md` replaces the source issue body with the clean-result draft and adds clean-results:draft label; gh_graphql MCP tools for write paths are NOT registered in deferred-tool list
type: feedback
---

The canonical analyzer publish path is `uv run python scripts/gh_project.py body-promote <SOURCE-N> .claude/cache/issue-<N>-clean-result.md`. This subcommand is idempotent — if run twice it edits in place without re-snapshotting the original body.

**Why:** I tried `ToolSearch("select:mcp__gh_graphql__update_issue_body,...")` to use the MCP write tools per CLAUDE.md's GitHub GraphQL MCP section, but `ToolSearch` returned "No matching deferred tools found." The MCP server itself is registered (see `.claude/mcp.json` / `~/.claude/mcp.json`), but its specific mutation tools (`update_issue_body`, `update_issue_title`, `add_issue_comment`, `add_labels_to_labelable`) are not surfaced as deferred tools in the analyzer's session — likely because the server is project-scoped and didn't initialize at session start. The skill text in `analyzer.md` (Step 6) says to use these MCP tools, but the actual working path on the local VM (Anthropic Safety Research workflow) goes through `scripts/gh_project.py body-promote` which authenticates locally via `gh` CLI.

**How to apply:** Don't waste cycles trying to load `mcp__gh_graphql__*` tools if `ToolSearch` returns empty. Use:

1. `uv run python scripts/gh_project.py body-promote <N> <path-to-md>` — replaces issue body, snapshots original to `epm:original-body` comment, adds `clean-results:draft` label.
2. `gh issue edit <N> --title "<title>"` — title update.
3. `gh issue comment <N> --body-file -` (heredoc) — for the `epm:analysis v1` recap.

The `body-promote` script handles the original-body preservation atomically. Title is set after promote.
