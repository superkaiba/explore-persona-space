---
name: weekly-workflow-optimization
description: >
  Weekly retrospective on Claude Code workflow. Reads ALL session
  transcripts from the past 7 days + GitHub issue activity from the past
  7 days. Spawns the `retrospective` agent with --lookback-days 7.
  Outputs a redacted public gist URL with a numbered list of
  CLAUDE.md / agent / skill / settings.json improvement candidates.
  Manual trigger only — no cron schedule. Subsumes the deprecated
  `/retro-weekly` skill.
user_invocable: true
---

# Weekly Workflow Optimization

End-of-week meta-review of how the Claude Code research-orchestration
setup is performing. The goal is to surface recurring friction patterns
that a CLAUDE.md / agent / skill / hook update could eliminate.

## When to use

Manual trigger at the end of each work week (e.g., Friday afternoon).
**Manual trigger only — no cron schedule.**

## What this skill does

1. Spawns the `retrospective` agent with `--lookback-days 7` (the agent
   accepts that arg; see `.claude/agents/retrospective.md`). The agent
   reads every JSONL transcript from
   `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/`
   modified in the past 7 days.
2. Augments the transcript review with GitHub-side activity:
   - Issues opened, closed, or updated in the past 7 days
   - PR activity over the past 7 days
   - status:* label transitions over the past 7 days
3. Aggregates findings into a numbered list of patches to propose to:
   - CLAUDE.md
   - .claude/agents/*.md
   - .claude/skills/*/SKILL.md
   - .claude/settings.json (hooks)
4. Pipes the resulting markdown body through
   `scripts/redact_for_gist.py` to scrub PII, then publishes as a
   public GitHub gist via `gh gist create --public`.

## Inputs

- `--n` (optional): cap the number of proposed patches in the output
  (default: 10, sorted by impact)
- `--include-success` (optional): include "what worked well" patterns
  in the output (default: false — focus on friction)

## Output

A public gist URL with this structure:

```markdown
# Weekly Workflow Optimization — week 2026-W18

## Summary
- N sessions reviewed
- M user corrections / friction events
- K successful agent dispatches

## Top friction patterns (proposed patches)
1. <pattern 1> — proposed change to CLAUDE.md / agent / skill
2. <pattern 2> — ...

## What worked (reinforce these)
[Only when --include-success is set]

## Metrics
- Time spent debugging vs research: <estimate>
- Most-spawned agent: <name>, <count>
- Agents never spawned this week: <list>

## Proposed CLAUDE.md / agent / skill diffs
[Numbered list of unified-diff blocks the user can apply with patch(1)]
```

## Procedure

```bash
# 1. Spawn the retrospective agent with the 7-day lookback.
#    The agent emits its draft to stdout; we capture it.
RETRO_OUT=$(uv run python -c "from claude_agent_sdk import spawn_agent; print(spawn_agent('retrospective', {'lookback_days': 7}))")

# 2. Build the gist body. The retrospective agent already produces the
#    markdown structure above; we just wrap it with a header.
WEEK_TAG=$(date +%Y-W%V)
{
  echo "# Weekly Workflow Optimization — week ${WEEK_TAG}"
  echo
  echo "$RETRO_OUT"
} > /tmp/weekly-workflow-body.md

# 3. Redact PII before publishing (gist is public).
uv run python scripts/redact_for_gist.py --in /tmp/weekly-workflow-body.md --out /tmp/weekly-workflow-body.redacted.md
gh gist create --public \
  --filename "weekly-workflow-${WEEK_TAG}.md" \
  --desc "Weekly Claude Code workflow optimization — ${WEEK_TAG}" \
  /tmp/weekly-workflow-body.redacted.md
```

`gh gist create` prints the gist URL on stdout. Return that URL to the
user as the SOLE output of this skill.

## Rules

1. **Manual trigger only.** No cron schedule. The user invokes this skill
   when they want to reflect on the week.
2. **Read-only on the project.** This skill never modifies CLAUDE.md or
   any agent / skill / hook automatically. Every proposed change is a
   diff in the gist for the user to review.
3. **Public gist requires redaction.** Always pipe through
   `scripts/redact_for_gist.py` BEFORE `gh gist create --public`.
4. **Subsumes /retro-weekly.** The old `.claude/skills/retro-weekly/` is
   deleted; do not create new skills under that name.
