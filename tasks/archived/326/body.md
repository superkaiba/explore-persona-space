---
title: '[Parent: #320] gh_graphql MCP migration Phase 2-5 + Phase 4.5 GH_TOKEN scrub'
kind: infra
tags: []
created_at: '2026-05-08T01:04:48.000Z'
has_clean_result: false
sagan_id: da7d6f35-527a-410e-859e-bce29d3d260b
sagan_number: 326
priority: normal
---
## Goal

Complete Phase 2-5 of the `gh_graphql` MCP migration started in PR #321 (§3 of Plan A).

Parent: #320

## Background

PR #321 shipped the `gh_graphql` MCP server (commit `3e02fa47`) with the 13-mutation allowlist + 65,536-byte body cap + `scripts/check_mcp_json_no_secrets.py` pre-commit hook. The server is registered at user-level `~/.claude/mcp.json` and the project-level `.mcp.json` was cleaned up to remove its stale SSH block.

The plan's §3 phased migration table (cached at `.claude/plans/issue-320-draft.md` lines ~580-630) specified 5 phases:

- **Phase 1 (skill-only):** SKILL.md migrates Step 2 plan-post + Step 9a clean-result-creation calls to use `gh_graphql.add_issue_comment`. Code-reviewer flagged this as "claimed but not actually shipped" — the SKILL.md wrapper for `body_too_large → status:blocked` did land, but the actual call-site rewiring at Step 2 / Step 9a did not. Either finish wiring the SKILL.md call sites to use the MCP, OR explicitly drop Phase 1 from scope and have the SKILL.md continue to use `gh issue comment` (the `body_too_large` wrapper is still useful as future-proofing).
- **Phase 2:** migrate `analyzer.md` (one site at line ~178: `gh issue edit <SOURCE-N> --title`).
- **Phase 3:** migrate `code-reviewer.md`, `implementer.md`, `experiment-implementer.md` (each has a marker-post site).
- **Phase 4:** migrate `experimenter.md` (the highest-volume `epm:progress` poster).
- **Phase 4.5: scrub `GH_TOKEN` from subagent env** — after Phases 2-4 land, `Agent()` calls in `/issue` SKILL set `env={k: v for k, v in os.environ.items() if k != "GH_TOKEN"}` for the spawned subagent. Each subagent now reaches GitHub only via `gh_graphql` MCP (no shell-out path to `gh`). This closes the "agent never sees `GH_TOKEN`" acceptance criterion in #320 Ask 3.
- **Phase 5:** migrate `planner.md` (read-only, but gets the `gh_graphql.read_issue` tool for symmetry).

## Acceptance criteria

- [ ] Phase 1 wiring complete OR explicitly dropped (SKILL.md uses `gh_graphql.add_issue_comment` at every comment-post site, OR keep `gh issue comment` and document why)
- [ ] Phase 2 done: `analyzer.md` uses `gh_graphql.update_issue`
- [ ] Phase 3 done: `code-reviewer.md`, `implementer.md`, `experiment-implementer.md` use `gh_graphql.add_issue_comment`
- [ ] Phase 4 done: `experimenter.md` uses `gh_graphql.add_issue_comment`
- [ ] Phase 4.5 done: subagent spawn scrubs `GH_TOKEN` from env; regression `/issue` test passes
- [ ] Phase 5 done: `planner.md` reads via `gh_graphql.read_issue`
- [ ] At end-of-migration `grep -rE 'gh issue (comment|edit|create|close|reopen)\\b' .claude/agents/` returns zero hits in agent prompts
- [ ] `epm:results v1` flagged the §3 Phase 1 skill-wiring discrepancy — surface a resolution in this issue's plan

## Compute

0 GPU-hours. ~1-2 working days. `type:infra`.

## References

- Parent: #320 (Plan A approved)
- PR shipped: #321 (commits `3e02fa47` §3 + `da6bf6bc` §5 router with the body_too_large skill-side wrapper)
- Plan body cached at `.claude/plans/issue-320-draft.md` §3 (lines ~591-829)
- Code-reviewer concerns on §3 Phase 1: https://github.com/superkaiba/explore-persona-space/issues/320#issuecomment-4402015283
