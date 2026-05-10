---
name: Audit gate drift across structurally-different generators
description: When length-matching across arms with different LLM prompts, ±10% of reference mean is too tight; cross-prompt mean variance is ~15%
type: feedback
---

When designing length-matched audit gates across arms whose generator prompts differ in subtle ways (e.g., "any rationale" vs "rationale supporting the correct answer"), expect ~15% mean BPE drift, not <10%.

**Why:** Issue #280 dispatch 7 (commit `eb82743c`) failed because `_audit_cell` enforced `±10% of generic-cot bpe_mean`. The contradicting-cot prompt — which differs from generic-cot only by adding "support correct answer" — produced rationales 15-19% longer (124.7 BPE vs reference 107.4 BPE) consistently across all 4 sources, both at n=5 (smoke) and n=1119 (full). The same Sonnet-4.5, same model temperature, same max-tokens, same word-count instruction "2-4 sentences" — but a slightly different task framing. Same pattern observed for generic-cot-correct (122.4 BPE).

**How to apply:** When writing audit gates that compare BPE/length across LLM-generated arms, default to **±20% (or ±25%)** of reference mean unless you have measured the cross-prompt variance for THAT specific prompt pair. ±10% only works when the arms share generator and prompt verbatim. Also: check that the smoke-launch gate threshold and the full-run audit threshold use the SAME formula — issue #280 v6 widened the smoke gate `[85, 130]` but left the audit gate at `±10%` unchanged, hiding the structural mismatch until full run.
