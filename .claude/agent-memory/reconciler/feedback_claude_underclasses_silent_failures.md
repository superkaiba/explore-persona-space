---
name: claude-underclasses-silent-failures
description: Claude code-reviewer tends to class silent-failure bugs (correctness violations that don't raise) as CONCERNS rather than FAIL; Codex twin correctly reads them as FAIL
metadata:
  type: feedback
---

When adjudicating code-reviewer disagreements, watch for cases where Claude flagged a real correctness bug but classed the overall verdict as CONCERNS (PASS-class) because the fix is "small" or "one patch." Codex twin tends to correctly read these as FAIL.

**Why:** The CLAUDE.md "Never silently fail" rule treats silent correctness violations as FAIL-class regardless of patch size. A one-line fix to a bug that silently corrupts the primary hypothesis test is still FAIL, not CONCERNS. Claude reviewer's calibration appears to weight "ease of fix" too heavily; Codex weights "what the bug DOES" more correctly.

**Observed in:** #375 round 1 (2026-05-21). Claude CONCERNS, Codex FAIL on a neutral-pool slicing bug that silently mispartitioned bootstrap arms after any ZLT drop. The bug invalidated the primary hypothesis test with no error raised. Claude flagged it (good) but said "revise-then-launch" (CONCERNS); Codex said FAIL. Codex was right per project rules.

**How to apply:** When the Claude reviewer's CONCERNS-class verdict and Codex's FAIL-class verdict both flag the SAME critical, default to FAIL if the critical describes:
- A bug that produces wrong results without raising (silent failure)
- A correctness violation in the primary hypothesis test path
- A miswiring that crosses experimental arms or labels

The classification depends on what the bug DOES at runtime, not how many lines the fix requires. Cross-reviewer agreement on a silent-failure critical is a strong FAIL signal even if Claude framed the verdict as CONCERNS.

Compounding rule: when Codex additionally identifies independent criticals Claude missed (here: C-2 free_vllm not releasing GPU memory, C-3 `--phase all` skipping required pilot), multiple-blocker compounding cleanly tips the verdict to FAIL.

Linked: [[codex-stronger-on-python-semantics]] (placeholder — pattern not yet observed enough times to write).
