---
name: acronym-and-beat-phrase-text-traps
description: audit hard-FAILs bare GCG/PAIR/EvoPrompt/nanoGCG tokens even as dataset names; check-34 flags "one point/marker per X" prose when the figure uses errorbar (0 scatter artists)
metadata:
  type: feedback
---

Two mechanical text traps hit on #1739 evil-ood-spread fold, both fixable at
draft time:

1. `audit_clean_results_body_discipline.py` `bare_method_acronym` regex
   hard-matches `\b(GCG|PAIR|EvoPrompt|nanoGCG)\b` ANYWHERE in the body —
   including when PAIR is a published attack CORPUS name, in alt text, and
   regardless of an inline expansion. **Why:** the rule is a flat regex with
   no definition-aware suppression; any match = FAIL = critic round-1
   bounce. **How to apply:** never write these tokens in a body — name the
   corpus descriptively ("optimizer-refined attacks",
   "prompt-automatic-iterative-refinement artifacts") and keep the
   lowercase slug (`pair`) only in the footer config-slug list (regex is
   case-sensitive).

2. `verify_task_body.py` check 34 (beat-phrase series-structure) WARNs when
   what-is-plotted/caption prose says "one point per arm" / "one marker per
   regime" but the figure was drawn with `ax.errorbar` — errorbar renders
   Line2D, so the sidecar has 0 `scatter` elements and the claim reads
   contradicted. **How to apply:** for errorbar-built forests, phrase as
   "the N arm-by-regime rows with bootstrap intervals; symbol shape gives
   the regime" (avoid "one point/marker per ..."), or build with
   `ax.scatter` + manual whiskers.
