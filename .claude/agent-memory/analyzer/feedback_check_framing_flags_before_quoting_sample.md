---
name: check-framing-flags-before-quoting-sample
description: Before quoting any raw completion as a "firing" illustration, verify the row's (family, sub_framing) is NOT in the eval rig's flagged/dropped framing set; positional-bias-prone framings must be skipped or explicitly labeled artifact
metadata:
  type: feedback
---

Headline DV roll-ups frequently EXCLUDE probe framings where the BASE model already false-positives >5% (prompt contains the target attribute, e.g. "Reference A says nine; Reference B says seven — which is correct?"). A trained-model row from such a framing is positional-bias matching, not retrieval of trained content; quoting it as a "firing" is misleading. The cell-level rig's `exclusion_policy` is load-bearing for which raw rows count as headline evidence.

**Why:** task #500 round 1 quoted a `framing381 sub_framing=6` "Reference A vs B" row as a leakage firing — but sub 6 was in `flagged_framings` under `base_fp_above_5pct`: the base model also picks "Reference B" by position. Both round-1 critics flagged it as the most misleading example in the body.

**How to apply:**
1. Before drafting any sample-output block, read `aggregate_cleaned.json` → `per_cell.<cell>.exclusion_policy`: sample ONLY from `headline_framings`; never quote a `flagged_framings` / `dropped_framings` row as a headline firing.
2. Filter raw `judged_*.jsonl` rows in code against the headline set (by `family` + `sub_framing`), not by eyeball.
3. A flagged-framing example may be included ONLY with an explicit label ("EXAMPLE OF POSITIONAL-BIAS ARTIFACT — flagged sub_framing; NOT counted in headline rate") plus surrounding prose explaining what it shows.
