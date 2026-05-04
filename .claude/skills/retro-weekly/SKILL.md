---
name: retro-weekly
description: >
  Weekly workflow improvement based on the past 7 days of session transcripts.
  Spawns the retrospective agent with a week-scoped lookback, identifies
  recurring patterns, and proposes batch improvements. Manual trigger only.
user_invocable: true
---

# Weekly Retrospective

A weekly-cadence wrapper around the `retrospective` agent. While the
retrospective agent reviews a single day's transcripts, this skill reviews
the past 7 days and looks for cross-session patterns.

## When to use

Invoke manually as `/retro-weekly` at the end of each work week, or when
the workflow feels inefficient. This is NOT automated — the user decides
when to run it.

## What it does

### 1. Gather transcripts

Locate session transcript files from the past 7 days:

```bash
find ~/.claude/transcripts/ -name "*.jsonl" -mtime -7 -type f | sort
```

If no transcripts are found, report "No transcripts from the past 7 days"
and exit.

### 2. Spawn retrospective agent

Spawn the `retrospective` agent with a modified prompt:

```
Review ALL session transcripts from the past 7 days (not just today).
Look for CROSS-SESSION patterns — issues that came up more than once,
workarounds that were repeated, corrections the user made multiple times.

Focus on:
1. Recurring user corrections (same feedback given 2+ times)
2. Repeated workflow friction (same manual step done 3+ times)
3. Agent/skill failures that happened across sessions
4. Opportunities for new hooks, skills, or agent improvements
5. CLAUDE.md rules that were violated or that seem outdated

For each finding, propose a concrete fix:
- New memory entry (if it's a user preference)
- CLAUDE.md edit (if it's a project convention)
- New hook (if it's an automatable check)
- Skill/agent edit (if it's a workflow improvement)
- New skill (if it's a repeating workflow with no current skill)

Group related changes into batch proposals (one PR per group).
```

### 3. Week-over-week comparison

If a prior weekly retro exists at `docs/retro-weekly/retro-weekly-*.md`,
compare:
- Which findings from last week were addressed?
- Which are still open?
- Any new recurring patterns?

### 4. Metrics aggregation

From the transcripts, compute:
- **Sessions this week:** count of transcript files
- **User corrections:** count of messages containing correction signals
  ("no", "don't", "stop", "wrong", "not that", "instead")
- **Agent dispatches:** count of `Agent()` tool calls
- **Skills invoked:** count of `Skill()` tool calls
- **Experiment issues processed:** count of `/issue` invocations

Present as a summary table with week-over-week delta if prior retro exists.

### 5. Output

Write the report to:
```
docs/retro-weekly/retro-weekly-YYYY-MM-DD.md
```

Format:

```markdown
# Weekly Retrospective — YYYY-MM-DD

## Metrics
| Metric | This week | Last week | Delta |
|--------|-----------|-----------|-------|
| Sessions | N | M | +/- |
| Corrections | N | M | +/- |
| Agent dispatches | N | M | +/- |
| Skills invoked | N | M | +/- |
| Issues processed | N | M | +/- |

## Recurring Patterns (cross-session)

### Pattern 1: [description]
- Seen in: [session dates/titles]
- Impact: [what time/quality cost]
- Proposed fix: [concrete change]

### Pattern 2: ...

## Open from Last Week
- [finding] — status: addressed / still open

## Batch Improvement Proposals

### Proposal 1: [title]
Files: [list]
Changes: [brief description]
Priority: HIGH / MEDIUM / LOW

### Proposal 2: ...
```

## Rules

- All proposals are DRAFTS — nothing is changed without user approval.
- The retrospective agent runs in a fresh context (no access to current
  session state) to avoid confirmation bias.
- Do not create GitHub issues automatically — present proposals to the user.
- If the user approves a proposal, THEN create an issue or make the change.
