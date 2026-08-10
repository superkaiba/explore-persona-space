---
name: audit-bans-verbal-ci-and-arm-counts
description: Discipline audit also bans bracket-less verbal CIs ("CI -0.4 to +2.2"), bare ±err, and "five arms"/"the behavioral arm" counts in prose — put bounds in a Methodology GFM table.
metadata:
  type: feedback
---

Three discipline-audit patterns beyond the bracketed-CI ban (all hit on the
#2215 round-1 draft):

1. `interval_inline` alt (5) (#1946): `CI` + optional `:`/`=`/`of`/`from` +
   number + `to` + number — the bracket-less VERBAL CI in prose FAILs.
2. `pm_inline` (#1987): any `±<num>` in reader-facing prose FAILs
   ("margins within roughly ±0.01").
3. `experimental_arm`: `five|four|three|two arms` and
   `the <adj> arm` (behavioral/geometric/LoRA/...) FAIL — write
   "predictors"/"readouts" or describe what was done. Compound forms like
   "three fitted arms" / "context-end arms" do NOT match (adjective breaks
   the bigram), but keep them sparse.

**Why:** Lens 7 wants intervals out of narrative prose entirely; GFM table
rows, fenced blocks, and `>` blockquote captions are EXEMPT from the scan.

**How to apply:** state point estimates + "interval includes/excludes zero
(or 0.5)" in prose, and park every exact bound in one compact Methodology
table ("Headline interval bounds") — the verifier + audit both pass that
shape, and the critic gets the numbers. Also: bare `H1`/`H2`/`H3` tokens
FAIL `condition_labels` even in Methodology — name hypotheses ("the
coupling hypothesis"); a driver-rendered figure whose TITLE carries `H2`
fires a figure-text WARN — acknowledge in the body instead of regenerating
when the fix is cosmetic.
