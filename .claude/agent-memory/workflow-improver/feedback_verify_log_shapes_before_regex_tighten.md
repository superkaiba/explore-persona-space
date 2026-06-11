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

**#597 addendum (the residual gap + the shape that worked):** the #545
corroboration only guards the pid-ALIVE path — a CRASHED wrapper whose
failure message quotes the token (`... FAILED rc=1 - [phase=done] NOT
emitted`) has a DEAD pid, which corroborates a false done. The #597
candidate again proposed a line-start anchor (`^\s*\[phase=`) — again
wrong, canonical phase lines are timestamp-PREFIXED. The fix that
survives both producer surveys: a HIGH-PRECISION noise denylist applied
only to `done`-parses (`DONE_QUOTED_NOISE_RE`: nonzero `rc=` on the
line, or a negation/suppression word immediately after the token),
skipping the line so the scan falls back to the previous real phase →
pid-death decays to `dead`. False-positive direction is conservative
(false `dead` on a weird success → orchestrator investigates; never a
false `done` on a failure). Pair with a producer-side hygiene rule in
the authoring agent's spec (experimenter.md / experiment-implementer.md:
never embed `[phase=` in message prose).
