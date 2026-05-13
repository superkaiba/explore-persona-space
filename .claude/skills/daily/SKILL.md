---
name: daily
description: Build a one-day Explore Persona Space project brief from Sagan state and local research artifacts.
---

# Daily Brief

Use Sagan as the only workflow state source. Do not read or mutate queue,
status, promotion, or approval state through any external tracker.

## Inputs

Read:

- Sagan experiments and workflow events via `scripts/sagan_state.py`;
- `RESULTS.md` for accepted headline claims;
- `eval_results/INDEX.md` for artifact inventory;
- `docs/research_ideas.md` for aims and phase framing;
- local run logs only as supporting evidence, never as workflow state.

Useful commands:

```bash
python scripts/sagan_state.py list-by-status --limit 500
python scripts/sagan_state.py list-by-status --status running --limit 100
python scripts/sagan_state.py list-by-status --status uploading --limit 100
python scripts/sagan_state.py list-by-status --status awaiting_promotion --limit 100
python scripts/sagan_state.py view <N>
```

## Output

Return a concise markdown brief:

```markdown
# Daily Brief — YYYY-MM-DD

## Active Work
- #N — status — one-line state, latest `epm:*` marker, pod/cost/ETA if present

## Awaiting Human Decision
- #N — what decision is needed and why

## Results Since Yesterday
- #N — claim, confidence, artifact pointer

## Risks
- blocked or stale items with the smallest useful next action

## Suggested Next Actions
1. ...
2. ...
3. ...
```

Do not promote clean results, create experiments, or move statuses unless the
user explicitly asks for that mutation in the current session. If asked to
mutate, use `scripts/sagan_state.py` so the change goes through Sagan's HTTP
API and leaves a workflow event.
