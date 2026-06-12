---
name: check-framing-flags-before-quoting-sample
description: Before quoting any raw completion as a "firing" illustration, verify the row's (family, sub_framing) is NOT in the headline-exclusion set; positional-bias-prone framings (Reference-A-vs-B, "Is X true?" with the target attribute in the prompt) MUST be skipped or explicitly labeled artifact
metadata:
  type: feedback
---

Headline DV roll-ups frequently EXCLUDE certain probe framings — most
commonly the ones where the BASE model already false-positives at >5%
because the prompt itself contains the target attribute (e.g.
"Reference A says nine; Reference B says seven; which is correct?"). The
model in those rows is doing position-bias matching, not retrieving the
trained content. Quoting one as a "firing" illustration is misleading.

**Why:** Task #500 round 1 quoted a `framing381 sub_framing=6` sample
(the "Source X / Source Y" disambiguation prompt) where the trained
model answers "Reference B" and labeled it a leakage firing. But sub 6
is in the `flagged_framings: [2, 4, 6]` set under `base_fp_above_5pct`
— the BASE model also picks "Reference B" by position, regardless of
which reference contains "seven." So the example doesn't illustrate
leakage; it illustrates positional bias. Both round-1 critics flagged
this as the most misleading example in the body.

**How to apply:**
1. Before drafting any per-finding sample-output block, pull the
   `aggregate_cleaned.json` for the relevant cell and read
   `per_cell.<cell>.exclusion_policy`:
   - `headline_framings: [...]` — these are the framings the headline DV
     uses; ALWAYS sample from this set.
   - `flagged_framings: [...]` — base-FP-flagged; NEVER quote one as a
     headline firing without an explicit "positional-bias artifact" label.
   - `dropped_framings: [...]` — excluded from rollup; ditto.
2. When sampling from raw `judged_*.jsonl`, filter rows in code:
   ```python
   HEADLINE_FRAMINGS = {1, 3, 5, 7, 8, 9, 11}  # from exclusion_policy
   def is_headline(r):
       fam, sf = r['family'], r['sub_framing']
       if fam == 'A_reformulation': return True
       if fam == 'framing381':
           try: return int(sf) in HEADLINE_FRAMINGS
           except: return False
       return False
   ```
3. If a flagged-framing example is QUALITATIVELY interesting (e.g. shows
   a positional-bias confound), include it ONLY with an explicit label:
   "EXAMPLE OF POSITIONAL-BIAS ARTIFACT (flagged sub_framing; NOT counted
   in headline rate)" — and surround the example with prose explaining
   what it shows.

The general principle: **the cell-level eval rig's `exclusion_policy` is
load-bearing for which raw rows count as headline evidence.** Skipping
this check leads to firing examples that don't actually fire (because
they wouldn't have counted anyway) or that fire for the wrong reason.
