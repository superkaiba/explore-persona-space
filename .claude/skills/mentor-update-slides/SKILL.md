---
name: mentor-update-slides
description: Create mentor-facing Explore Persona Space update slides from accepted claims, and local artifacts.
---

# Mentor Update Slides

Use `tasks/` as the source of workflow state. Do not use any external tracker for
queue membership, statuses, approvals, promotion, or workflow comments.

## Inputs

- tasks and workflow events:
  `python scripts/task.py list-by-status --limit 1000`
- Per-experiment details:
  `python scripts/task.py view <N>`
- Accepted claims and confidence:
  `RESULTS.md`
- Artifact inventory:
  `eval_results/INDEX.md`
- Research aims and phase framing:
  `docs/research_ideas.md`
- Existing deck, if present:
  `figures/mentor-slides/deck.md`

## Deck Shape

1. Cover: project, date, audience.
2. Project summary: current thesis, 3-5 strongest accepted claims, and active
   task work.
3. Results slides: one assertion sentence per slide, real figure or table,
   visible caption, confidence, artifact link.
4. Active work: running/awaiting-promotion/blocked experiments from local files.
5. Decisions needed: concrete choices for the mentor.
6. Appendix: methods, caveats, and raw artifact pointers.

## Output Rules

- Prefer precise numbers and concrete experiment identifiers.
- Use task numbers (`#N`) for references.
- Include confidence on every claim slide.
- Preserve existing log and appendix sections when updating an existing deck.
- Do not promote clean results or change statuses while preparing slides unless
  the user separately asks for that mutation.
