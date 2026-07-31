---
name: Length/audit bands on Sonnet-generated arms are too tight (±10% and ±20% both burned)
description: Cross-prompt BPE mean drift across Sonnet-generated arms is ~15%; single-shot rewrites against a fixed char target land at ~33% frac_dev. Default ±20-25% for audit gates, expect ±30-40% for rewrite length bands; ALL-cells FAIL_LENGTH with clean leak-check = gate artifact, code-class.
type: feedback
---

When length-matching across arms whose generator prompts differ even subtly, expect ~15% mean BPE drift (NOT <10%); when Sonnet rewrites a prompt against a fixed char-length target, expect ~±30-40% frac_dev (NOT ±20%).

**Why:** #280 dispatch 7 — `_audit_cell` enforced ±10% of the generic-cot BPE mean; the contradicting-cot arm (same Sonnet, same temperature, same "2-4 sentences" instruction, framing differs by one clause) ran 15-19% longer consistently at n=5 and n=1119. #467 round-6 SMOKE — a ±20% char band on strong-NL rewrites downgraded 2/2 cells to FAIL_LENGTH (frac_dev +0.33/+0.37) while the leak-check itself passed (leak_score=0.0), making the gate look like a hard upstream failure.

**How to apply:**
1. Audit gates comparing BPE/length across LLM-generated arms: default ±20-25% of reference mean unless the cross-prompt variance for THAT prompt pair was measured. ±10% only when arms share generator AND prompt verbatim. Keep the smoke-gate and audit-gate formulas in sync (#280 v6 widened one but not the other).
2. ALL cells FAIL_LENGTH + clean leak scores = the gate, not quality. Don't retry; bounce code-class with fix options in preference order: (a) re-author retry loop with explicit "shorter, target N chars" when out of band, (b) widen band to ±40%, (c) truncate post-author (lossy, last resort).
3. Pre-launch: grep author/audit scripts for `frac_dev` / `LENGTH_BAND` / `±0.20`-style constants; flag ≤0.20 bands in the launch note as at-risk.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Audit/length bands on Sonnet arms too tight](feedback_audit_gate_arm_drift.md) — cross-prompt BPE drift ~15%, rewrite frac_dev ~33%; default ±20-25%; all-FAIL_LENGTH + clean leak = gate artifact (#280, #467)
