---
name: gh_graphql not available in this context; use REST fallback
description: gh_graphql MCP is not in project context; use gh REST API as fallback when GraphQL is rate-limited or unavailable
type: feedback
---

The `gh_graphql` MCP server is in the user-level `~/.claude/mcp.json`, not the project-level one. It does NOT appear via ToolSearch in this agent's context. When GraphQL quota is exhausted (`remaining=0`), use the REST API instead (separate quota — typically 4964/5000 available even when GraphQL is at 0).

**REST fallback command:**
```bash
gh api -X POST /repos/<owner>/<repo>/issues/<N>/comments -F "body=@/tmp/review_body.md"
```
Write marker to `/tmp/review_body.md` first. Confirmed working in issue #344 review.

**Why:** GraphQL and REST have SEPARATE 5000/hr quotas. GraphQL was fully exhausted (5000/5000 used) during issue #344 review, but REST had 4964 remaining. The REST path successfully posted the marker.

**How to apply:** Always run `gh api rate_limit --jq '.resources | {core: .core, graphql: .graphql}'` before deciding which path to use. Use REST when GraphQL remaining=0 or gh_graphql not in ToolSearch results.
