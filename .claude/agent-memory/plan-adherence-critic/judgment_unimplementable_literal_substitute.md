---
name: judgment-unimplementable-literal-substitute
description: When an implementer claims a plan's literal verification recipe is unimplementable (artifact lacks the pinned fields), re-probe the artifact yourself, then rule on the substitute by mapping drift classes to what the round actually CONSUMES
metadata:
  type: feedback
---

Rule: an "assumption X's literal form is unimplementable" claim is a load-bearing premise — never take it from the implementer's report. Re-probe the artifact independently (fresh HF download, exhaustive key scan: `raw.count('field')` over the whole blob, not just the first record), and verify any producer-domain sha/pin claim by re-running the recompute against the producer's own code lines.

**Why:** #2162 round `turn-boundary-multipatch` (2026-08-14): plan §12 assumption 9 pinned a rebuilt-vs-recorded `ctx_len`/`prefix_end` parity against a `bank.json` that records neither field (my fresh probe: 0 occurrences in 2.38 MB). The substitute (producer-domain sha + payload equality on the 6 text-determining keys + len_delta parity vs the parent's committed table) was ACCEPTED — confirmed by the user-side orchestrator's provisional read.

**How to apply:** rule on a substitute by enumerating drift classes and checking each against what the round CONSUMES from the parent, not against abstract strength claims:
- text drift at unchanged length → payload equality catches it, the literal (two derived integers) would NOT — call this out when the consumed artifacts (anchors, donor pools, judge scores) are text-keyed;
- length/tokenizer drift → per-pair delta parity + designed-count equality asserts catch it;
- the residual the literal alone would catch (uniform within-pair-canceling render shifts) is acceptable ONLY when nothing consumed is absolute-position-indexed — state that condition explicitly in the verdict.
When accepting, always recommend the concrete plan-text correction (the realized verification triple), so the plan stops describing an unexecutable recipe. "Strictly stronger" claims by implementers are usually level-scoped (text-level yes, token-geometry no) — qualify rather than repeat them.
