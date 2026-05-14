---
name: weekly
description: Build a weekly Explore Persona Space research narrative from clean results, and local artifacts.
---

# Weekly Narrative

Use `tasks/` as the canonical workflow state source. External trackers may be cited
as historical links, but they are never the queue, status source, approval
source, promotion source, or comment thread for workflow decisions.

## Inputs

- tasks: `python scripts/task.py list-by-status --limit 1000`
- Individual experiment state/events: `python scripts/task.py view <N>`
- Headline accepted claims: `RESULTS.md`
- Artifact index: `eval_results/INDEX.md`
- Research aims: `docs/research_ideas.md`
- Prior weekly artifacts under `updates/weekly/`, when available

Group tasks by status (folder name under tasks/). Treat `events.jsonl` markers as the audit
trail for progress, reviewer rounds, reconciler decisions, promotion, and
failures.

## Process

1. Pull the week window in UTC.
2. Identify completed, awaiting-promotion, active, blocked, and proposed work
   from local files.
3. Read the relevant experiment bodies and latest `epm:*` events.
4. Cross-check accepted claims against `RESULTS.md` and artifact paths.
5. Draft a concise narrative with evidence and explicit caveats.

## Output

Return markdown:

```markdown
# Weekly Update — Week of YYYY-MM-DD

## Bottom Line
One paragraph with the main research update.

## New Evidence
- #N — claim, confidence, artifact pointer, why it matters

## Active Work
- #N — status, latest marker, expected next step

## Decisions Needed
- #N — decision, options, recommended next action

## Risks And Drift
- stale, blocked, contradictory, or under-documented items

## Next Week
1. ...
2. ...
3. ...
```

Never auto-promote clean results or move statuses as part of writing the weekly
narrative. If the user asks for a mutation, use `scripts/task.py` and
record a task workflow event with a short note.
