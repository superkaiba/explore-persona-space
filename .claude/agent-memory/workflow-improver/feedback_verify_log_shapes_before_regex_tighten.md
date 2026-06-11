---
name: Grep real producers before tightening a log-parser regex
description: A candidate's preferred regex-tighten can break legitimate producers; grep the actual emitting scripts for line shapes before anchoring
type: feedback
---

Before applying a candidate's proposed regex tightening on a log/marker
parser, grep the REAL producer scripts for the line shapes they actually
emit. **Why:** the #545 candidate (2026-06-11, poll_pipeline done-detection)
preferred a bare-line anchor `^\[phase=done\]\s*$`; greping `scripts/` for
`phase=done].*\w` showed many legitimate dispatchers end with SUFFIXED
terminal done lines (`[phase=done] SMOKE COMPLETE ...`, `[phase=done]
phase4 complete $(date)`, even `Dispatcher done. ... [phase=done] <ts>`) —
the anchor would have reclassified healthy completions as `dead`. The noise
line was textually indistinguishable from genuine terminals, so the
corroboration half (pid-dead OR results-sentinel) had to carry the whole
fix. **How to apply:** when a candidate proposes pattern-matching changes,
treat the diff_sketch as a hypothesis about producer behavior and verify it
against the producers (`Grep` over scripts/src for the pattern WITH
trailing/leading context) before choosing the final shape; add a regression
test pinning the legitimate shapes you found.
