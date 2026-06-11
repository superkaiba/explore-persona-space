---
name: Claude judge stop_reason=refusal on harmful probes
description: Sonnet judge calls over harmful-probe benchmarks (SORRY-Bench etc.) hit API-level stop_reason="refusal" with EMPTY content on a fixed probe subset; system-prompt framing does NOT recover it — track as a counted judge_refused class.
type: feedback
---

When a Claude judge grades completions for harmful-request probes (SORRY-Bench
should_refuse half, similar red-team benchmarks), the API can refuse at the
MODEL level: `stop_reason == "refusal"`, `content == []`, ~1 output token,
fired pre-generation. Measured on #545 round 12 (2026-06-11): a fixed 36/250
probe subset, 100% identical probe_ids across cells with different
completions — i.e. probe-text-determined and deterministic.

**Why:** the refusal fires before generation, so safety-evaluator system-prompt
framing recovers 0/36 (tested). Retrying the identical prompt is a wasted call.

**How to apply:**
- Detect `resp.stop_reason == "refusal"` explicitly and classify immediately as
  a tracked `_judge_refused` verdict (distinct from `_judge_error`), no retry.
- Exclude it from score denominators AND from any judge-quality-floor numerator
  (it is a counted measurement limitation, not an outage); surface
  `n_judge_refused` in summaries.
- Because the subset is probe-determined, the SAME probes are excluded for every
  cell → cross-cell comparability holds; the level of the rate is measured only
  on the judgeable (less-extreme) probes — carry as a scope caveat.
- Diagnose with a REDACTING script only (probe_id + stop_reason + block shape +
  token counts; never probe/completion text — incident #537 context poisoning).
  Reusable pattern: `scripts/issue545_judge_refusal_diag.py`.
