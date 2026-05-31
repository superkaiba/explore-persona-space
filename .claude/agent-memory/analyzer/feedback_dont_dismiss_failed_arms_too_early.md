---
name: dont-dismiss-failed-arms-too-early
description: When a planned experiment arm fails methodologically (contamination, broken premise, wrong fact), the per-persona / per-condition breakdown of raw completions may still show the OPERATIVE MECHANISM intact — don't write off the arm as "uninterpretable" before tallying per-cell raw text by persona.
metadata:
  type: feedback
---

**Rule:** When an experiment arm fails its planned hypothesis test (contamination, broken premise, wrong fact, bypass-bypass, etc.), don't write "uninterpretable" in the TL;DR before per-persona-tallying the raw completions. The judge rubric collapses content into a fixed canonical/counter taxonomy — if the contamination installed CONTENT OUTSIDE that taxonomy, the rubric reports 0% canonical + 0% counter, which LOOKS like "nothing happened" when actually the mechanism may have installed perfectly on the corrupted content.

**Why:** Task #407 round-1: I wrote off the obscure-real arm as "uninterpretable" because the eval-judge canonical-rate was 0% across all cells. The round-1 critic re-ran the per-persona breakdown of the raw completions and showed the persona-gating signature installed CLEANLY on the contaminated CJD content (teach 100% CJD-canonical, non-teach 100% CJD-counter). The judge couldn't see this because it scored against urea/glycogen, not CJD. The accidental finding (gating mechanism is content-agnostic to eval entity, n=5400 probes at 99-100%) was BURIED INSIDE the "uninterpretable" arm — a real result hidden by a confusing rubric.

**How to apply:** Before writing "this arm is uninterpretable" / "this arm collapsed" / "this arm sits at floor" in any clean-result body:
1. Tally per-persona top-K completions across all seeds (use `Counter` over raw text). 30 seconds with the raw completions in hand.
2. Ask: does the top-1 completion per persona DIFFER across personas, in a structurally meaningful way (teach gets X, non-teach gets Y)?
3. If YES, the per-persona pattern is a real signal regardless of what the judge says. Surface it explicitly even when the planned hypothesis test failed.
4. If NO, then "uninterpretable" is the right framing.

The judge is a fixed measurement instrument. The raw completions are the territory. When the territory and the instrument disagree, trust the territory and explain why the instrument missed it.
