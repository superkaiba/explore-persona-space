---
name: dont-dismiss-failed-arms-too-early
description: When an arm fails its planned hypothesis test (contamination, broken premise), per-persona tally of raw completions may show the operative mechanism intact — tally before writing "uninterpretable"
metadata:
  type: feedback
---

When an experiment arm fails methodologically (contamination, wrong fact, broken premise), do NOT write "uninterpretable" in the TL;DR before per-persona-tallying the raw completions. The judge rubric scores against a fixed canonical/counter taxonomy — if contamination installed content OUTSIDE that taxonomy, the rubric reports 0% on both sides, which LOOKS like "nothing happened" while the mechanism may have installed perfectly on the corrupted content.

**Why:** task #407 round-1 — I wrote off the obscure-real arm because judge canonical-rate was 0% everywhere. The critic tallied per-persona raw completions: the persona gate had installed CLEANLY on the contaminated CJD content (teach 100% CJD-canonical, non-teach 100% CJD-counter, n=5400 probes). A real content-agnostic-gating result was buried inside "uninterpretable."

**How to apply:** before writing "uninterpretable" / "collapsed" / "at floor":
1. `Counter` over per-persona top-K raw completions across seeds (~30 s).
2. Does the top-1 completion differ across personas in a structurally meaningful way (teach gets X, non-teach gets Y)?
3. YES → real signal regardless of the judge; surface it explicitly. NO → "uninterpretable" is right.

The judge is the instrument; the raw completions are the territory. When they disagree, trust the territory and explain why the instrument missed it.
