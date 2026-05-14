---
title: Adopt 5 patterns from OpenAI Symphony harness into /issue workflow
kind: infra
tags: []
created_at: '2026-05-07T21:28:48.000Z'
has_clean_result: false
sagan_id: 6b1b44ee-ea1c-40df-8f18-6a81209b1cc2
sagan_number: 320
priority: normal
---
## Goal

Audit of OpenAI's Symphony harness (https://github.com/openai/symphony) surfaced 5 patterns worth adopting into our `/issue` workflow. Bundle here as a tracker; the adversarial planner may split into sub-issues based on dependency analysis (suggested order at bottom).

Symphony is a Linear-polling daemon that drives tickets through `Todo → In Progress → Human Review → Merging → Done` autonomously. We should NOT adopt the daemon model itself, the `max_turns` auto-retry, the `approval_policy: never` posture, the open GraphQL introspection, or workpad-as-mutable-comment — see "Explicitly out of scope" below.

## Deliverables

### (1) `.claude/workflow.yaml` as single source of truth for the state machine

**What:** Move gate definitions, `status:* → next-action` table, halt criteria, and re-entry rules into one YAML file with strict schema. CLAUDE.md and `.claude/skills/issue/SKILL.md` reference it; `markers.md` validates marker types against it. Pre-commit lint fails on undefined statuses or unknown template variables.

**Why:** Today the gate list lives in three places (CLAUDE.md "Auto-continuation policy", `SKILL.md`, `markers.md`) and they drift. Commit b7a8a3c4 retrofit (Step 10 completion-audit) was symptomatic. Symphony's WORKFLOW.md does this cleanly with YAML front-matter + Liquid templates, hot-reloaded, with fail-on-unknown-template-variable lint (SPEC.md §5.3, §5.4, §6.2).

**Touches:** new `.claude/workflow.yaml`, `CLAUDE.md`, `.claude/skills/issue/SKILL.md`, `.claude/skills/issue/markers.md`, new pre-commit hook.

### (2) `pod.py watch --issue N` stall-detection watchdog

**What:** Background process started alongside experimenter dispatch. Tails WandB run heartbeat + experiment log mtime; on >5min silence flips `status:blocked` and posts `epm:failure failure_class=infra reason=stall last_event=…`.

**Why:** Today the experimenter agent owns its own monitoring cadence — when *it* gets stuck, no one notices (cf. `feedback_audit_gate_arm_drift.md`). A stuck training run is loud; a stuck monitor is silent. Symphony inverts this with a reconciliation loop checking `last_codex_event` against `stall_timeout_ms` (SPEC.md §8.5, §10.6, `orchestrator.ex#reconcile_stalled_running_issues/1`).

**Touches:** `scripts/pod.py` (new `watch` subcommand), `.claude/skills/issue/SKILL.md` (Step 6 dispatch wiring).

### (3) `gh_graphql` MCP tool with orchestrator-held auth

**What:** New MCP server exposing the GitHub GraphQL API, scoped to `superkaiba/explore-persona-space`, with a documented mutation allowlist (no `archiveRepository`, no `transferIssue`, etc.). Replaces direct `gh issue edit` / `gh pr create` shellouts in agent prompts.

**Why:** Centralizes auth so agents never see `GH_TOKEN`. We've had token-leak incidents (`feedback_no_hardcoded_secrets.md`). Symphony does this with `linear_graphql` (SPEC.md §10.5, `.codex/skills/linear/SKILL.md`).

**Touches:** new MCP server (likely Node or Python), `~/.claude/mcp.json` registration via `pod.py config --sync`, agent prompts that currently shell out to `gh`.

### (4) `clean-result-lint.yml` CI workflow

**What:** GitHub Action triggered on `issues:edited` for any issue carrying a `clean-results:*` label. Runs `scripts/verify_clean_result.py` against the issue body and posts a checkmark comment (PASS) or a FAIL comment with the verifier output.

**Why:** Today `verify_clean_result.py` runs only when the analyzer remembers. Move it to the platform layer like `project-archive-on-close.yml` is. Symphony does the equivalent for PR descriptions (`.github/workflows/pr-description-lint.yml` + `mix pr_body.check`).

**Touches:** new `.github/workflows/clean-result-lint.yml`, possibly minor refactor to `verify_clean_result.py` to read issue bodies from JSON event payloads.

### (5) Continuation-vs-retry split with `epm:step-completed` markers

**What:** Every `/issue` step that completes posts `epm:step-completed step=<name> at=<sha>`. Skill re-entries grep for the latest such marker and jump-ahead to the next step instead of full marker replay. Failure-driven re-entries (after `status:blocked`) still do full replay.

**Why:** Today every `/issue N` re-entry re-parses every `epm:*` marker from the top, eating context budget. Symphony distinguishes "clean exit but issue still active → 1s continuation on same thread" from "failure → exponential backoff" (SPEC.md §7.1, §7.3, §16.6).

**Touches:** `.claude/skills/issue/markers.md` (new marker type), `.claude/skills/issue/SKILL.md` (re-entry logic), depends on (1) for the structured status→step mapping.

## Acceptance criteria

- [ ] `.claude/workflow.yaml` exists with all gates, statuses, and halt criteria; CLAUDE.md and `SKILL.md` reference it instead of duplicating; pre-commit lint blocks unknown variables
- [ ] `pod.py watch --issue N` exists, is wired into Step 6, demonstrably flips `status:blocked` on a synthetic stall
- [ ] `gh_graphql` MCP tool registered, agent prompts updated to use it, no agent has direct `GH_TOKEN` access
- [ ] `clean-result-lint.yml` triggers on `issues:edited` for `clean-results:*` issues, posts PASS/FAIL comments
- [ ] `epm:step-completed` markers emitted by every step that completes; `/issue N` re-entry on a half-done issue measurably skips replay (verified on a test issue)
- [ ] All 5 changes pass `/adversarial-planner` review

## Suggested dependency order

1. **(1) workflow.yaml first** — foundational, (5) depends on the structured state map
2. **(2), (3), (4) in parallel** — independent of each other and of (1)
3. **(5) last** — depends on (1)

The planner may legitimately split into 3 issues (umbrella + workflow.yaml + step-completed) or 5; up to the planner.

## Explicitly out of scope (do NOT adopt from Symphony)

- The polling daemon model (SPEC.md §3, §16.1) — we are correctly invocation-driven
- `max_turns=20` auto-retry without human gates (SPEC.md §5.3.5, §16.5) — clashes with our `status:blocked` halt criteria
- Codex `approval_policy: never` + `workspace-write` sandbox (`elixir/WORKFLOW.md` line 32) — we touch shared `/workspace/`, HF Hub, WandB
- Open GraphQL schema introspection as an agent capability — for GitHub that exposes `archiveRepository`, `transferIssue`, etc.
- Workpad-as-mutable-comment (`elixir/WORKFLOW.md` line 295) — our append-only `epm:* v<n>` history IS the audit trail

## References

- Symphony SPEC: https://github.com/openai/symphony/blob/main/SPEC.md
- Symphony WORKFLOW: https://github.com/openai/symphony/blob/main/elixir/WORKFLOW.md
- Local clone for inspection: `/tmp/symphony/`
- Conversation that produced this audit: 2026-05-07 session, Symphony workflow comparison
