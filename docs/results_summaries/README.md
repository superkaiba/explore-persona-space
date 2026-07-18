# Results summaries — writing conventions

Ad-hoc result write-ups in this directory (and mentor updates drawing on
them) follow three conventions:

1. **Terminology.** The context->answer mapping line uses the canonical
   vocabulary in `../glossary_context_answer_map.md` — including the
   retired-terms table (governs writing) and its search-time alias note
   (governs searching).

2. **Analysis choices ship with their grounding by default.** Every
   analysis choice that shapes a result — PCA dimension, kernel choice,
   layer, regularization, fold scheme, aggregation grain — states its
   grounding inline at first use: a prior issue (#M), a paper, a measured
   pilot/sweep, or an honest "convenience choice, not swept". This is the
   writeup-time counterpart of plan-time grounding (CLAUDE.md § "Ground
   every load-bearing hyperparameter") and extends the clean-result
   Training-table Source column (`.claude/skills/clean-results/SPEC.md`)
   to ANALYSIS choices in ad-hoc writeups.

3. **Per-arm provenance in the setup line** — per CLAUDE.md § "Ad-hoc
   results summaries state per-arm provenance" (data/completion provenance
   + generation recipe per arm; matched-target disclosure for cross-arm
   tables).

Captured by #1066 from the 2026-07-04 #779 re-steer session (which also
created the glossary).
