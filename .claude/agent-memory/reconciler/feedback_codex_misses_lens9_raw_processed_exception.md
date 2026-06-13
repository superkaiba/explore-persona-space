---
name: codex-misses-lens9-raw-processed-exception
description: Codex FAILs "two figures in one #### finding" by counting images structurally; Lens 9 check 1 explicitly counts adjacent raw + processed pairs as ONE figure, and Lens 11 REQUIRES the raw counterpart inline in the same H4. PASS when figure 2 is the residualized/partialled/binned sibling of figure 1.
metadata:
  type: feedback
---

**Rule:** when Codex's only blocker is "two figures in one `#### <finding>`", check the pair: read both captions/alt-texts for processing keywords (`residualized`, `partialled`, `binned`, `log-`, `normalized`, `centered`, `rank-`). If the second figure is the processed sibling of the first (same relationship, derived transform), PASS — Lens 9 check 1 counts them as ONE figure, and the user's standing show-raw-alongside-processed rule makes the pair REQUIRED, so splitting them into two H4s (Codex's fix) would break Lens 11. The figures need not be line-adjacent; bridging prose is fine within the same H4. Mechanical pre-pass PASS corroborates (v2 spec doesn't flag the pair).

**Origin:** #480 r1 — raw scatter + source-mean-residualized scatter in one finding; Codex FAILed, Claude PASSed, reconcile sided with Claude.

Companion: [[feedback_codex_conflates_marker_format_with_code]] (Codex counting structural surface over textual exceptions); inverse boundary in [[feedback_claude_clean_result_critic_underapplies_spec_text]] rule 19 (raw sibling in a DIFFERENT H4 does NOT satisfy the pair).
