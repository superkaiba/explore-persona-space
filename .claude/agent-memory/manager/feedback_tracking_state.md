---
name: Keep Tracking Files Current
description: INDEX.md, EXPERIMENT_QUEUE.md, research_ideas.md, and RESULTS.md must be updated after every experiment - not deferred
type: feedback
---

After every experiment completes, update ALL tracking files immediately — not "later" or "in a batch."

**Why:** A project audit on 2026-04-11 found: 20/50 eval_results directories unindexed in INDEX.md, EXPERIMENT_QUEUE.md listing completed experiments as "Running," RESULTS.md Next Steps listing tasks done weeks ago, and research_ideas.md status markers not updated. The drift accumulated because updates were deferred.

**How to apply:**
- After analyzer/reviewer complete, the manager must verify these 4 files are updated before moving on
- eval_results/INDEX.md: every new directory gets a row
- EXPERIMENT_QUEUE.md: move Running→Completed, add new Planned items
- docs/research_ideas.md: update [x]/[~]/[ ] markers, add new subtask entries
- RESULTS.md: update TL;DR, Next Steps, In Progress when they become stale
- At session start, verify Running items in EXPERIMENT_QUEUE.md are actually still running
