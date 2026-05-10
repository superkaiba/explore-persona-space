---
name: Scope — Code Changes Only
description: What code-reviewer reviews vs what reviewer (analysis) reviews
type: project
---

code-reviewer reviews: implementer diffs, refactors, utility changes, config reorganizations, pod-management scripts, CI changes, new scripts under `src/`, `scripts/`, `configs/`. Focus: bugs, plan deviation, security, tests, lint, API compatibility.

reviewer (the other agent) reviews: experiment analysis drafts in `research_log/drafts/`, result interpretations, statistical claims, plots. Focus: overclaims, alternative explanations, reproducibility gaps.

**Why:** Different failure modes need different checklists. A code diff can have off-by-one bugs; an analysis can have inflated p-values. Running the same reviewer for both dilutes the checklist and misses things.

**How to apply:**
- If the task is "review this diff / PR / commit" → you're the right reviewer.
- If the task is "review this draft report / experiment analysis" → decline and point to `reviewer` agent.
- If a change includes both code AND a draft (rare), do the code review; a separate reviewer spawn handles the analysis.
