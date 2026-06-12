---
name: Plan tmp files may carry the planner's harness trailer
description: Orchestrator-written /tmp plan files can end with the planner subagent's harness trailer (agentId + <usage> block) — strip it before inlining into a Codex prompt
type: feedback
---

The orchestrator sometimes captures the planner subagent's return verbatim
into the `/tmp/issue-<N>-plan-v<V>.md` handoff file, including the harness
trailer appended to the last content line: `agentId: <id> (use SendMessage
with to: '<id>' to continue this agent)` plus a `<usage>total_tokens...
</usage>` block.

**Why:** observed on `/tmp/issue-562-plan-v1.md` (2026-06-10). Inlining it
verbatim would feed Codex a stray "use SendMessage..." instruction and
token-usage noise at the end of the PLAN TEXT section.

**How to apply:** before concatenating the plan into the prompt, `tail` the
plan file; if contaminated, strip with
`sed -e "s/agentId: <id> (use SendMessage.*$//" -e '/^<usage>/,/^<\/usage>$/d'`
into a `*-plan-clean.md` intermediate and inline that. Pairs with the
header+plan+footer assembly pattern (avoids re-emitting a 50KB+ plan through
the Write tool).
