---
name: codex-misses-lens9-raw-processed-exception
description: Codex FAILs "two figures in one #### finding" by counting images structurally; Lens 9 check 1 explicitly counts adjacent raw + processed pairs as ONE figure, and Lens 11 REQUIRES the raw counterpart inline in the same H4. PASS when figure 2 is the residualized/partialled/binned sibling of figure 1.
metadata:
  type: feedback
---

**Rule:** when Codex's only blocker is "two figures in one finding" (v3: `### <finding>` H3 under `## Findings`; v2: `#### <finding>` H4), check the pair: read both captions/alt-texts for processing keywords (`residualized`, `partialled`, `binned`, `log-`, `normalized`, `centered`, `rank-`). If the second figure is the processed sibling of the first (same relationship, derived transform), PASS — Lens 9 (one-takeaway-one-figure) counts them as ONE figure, and the user's standing show-raw-alongside-processed rule (Lens 11) makes the pair REQUIRED, so splitting them into two findings (Codex's fix) would break Lens 11. The figures need not be line-adjacent; bridging prose is fine within the same finding. Mechanical pre-pass PASS corroborates (neither the v3 nor v2 spec flags the pair).

**Origin:** #480 r1 (v2 body) — raw scatter + source-mean-residualized scatter in one finding; Codex FAILed, Claude PASSed, reconcile sided with Claude. The exception is identical under v3 (same Lens 9 / Lens 11 numbers, finding is a `### ` H3 instead of a `#### ` H4).

Companion: [[feedback_codex_conflates_marker_format_with_code]] (Codex counting structural surface over textual exceptions); inverse boundary in [[feedback_claude_clean_result_critic_underapplies_spec_text]] rule 19 (raw sibling in a DIFFERENT H4 does NOT satisfy the pair).
