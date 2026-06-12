---
name: Codex cleanup-after-swallowed-failure at round-N cap
description: At round-cap, Codex FAILs `try/except: log; continue` paths whose cleanup proceeds unconditionally and deletes intermediate artifacts. Verify env-gating default, recoverable-vs-irreversible data loss, and operational-flip availability before believing FAIL.
type: feedback
---

When Codex code-reviewer's Critical at round-N cap is "extractor failure swallowed + cleanup proceeds anyway, deleting intermediate artifacts", verify the FULL data-loss picture before adjudicating FAIL.

**Why:** Per CLAUDE.md "Fail fast — never hide failures" rule, a `try/except Exception: log; continue` block followed by unconditional cleanup IS a canonical silent-failure pattern. Codex correctly identifies it. But the question for the reconciler is whether the data-loss is HEADLINE-invalidating or only TRAJECTORY/DIAGNOSTIC degradation.

**How to apply:**
1. Open the cited try/except + cleanup block and the cleanup gating expression.
2. Check the env-var default literal: `os.environ.get("X", "1") == "1"` defaults ON; `"0"` defaults OFF. Default-ON means production-default behavior IS to delete.
3. Check the cleanup loop for a HEADLINE-preserving carve-out (e.g. `if frac_key in ("1.00", ...): continue` keeps the cell endpoint). If endpoint is preserved, the H1/H2/H3 reads — which the analyzer pulls from per-cell endpoint eval JSONs, not from trajectories — survive even on extractor crash.
4. Check the plan: are trajectory figures HEADLINE deliverables (e.g. their slope IS the claim) or DIAGNOSTIC (e.g. they support a non-saturation argument that the gate already checks via a different channel)? Plan §4.7 calling them "first-class deliverables" is non-blocking when the headline (H1/H2/H3) doesn't depend on them.
5. Check operational recovery: can the experimenter flip `EPM_DELETE_INTERMEDIATE_FT_CKPTS=0` on the pod-side run to fully sidestep the bug? If yes, the bug becomes a standing recommendation, not a code blocker.

**Decision rule:**
- Endpoint preserved + H1/H2/H3 read from endpoint eval JSONs + env-flag operational mitigation → PASS with HARD standing recommendation on the env flag (the recommendation is mandatory in the rationale).
- All paths preserved + recoverable → PASS.
- Headline data lost OR no operational mitigation OR irreversible deletion of the headline-feeding artifact → FAIL even at round cap.

**Companion patterns:**
- `feedback_claude_underclasses_silent_failures.md`: Claude PASSing a silent-failure is canonical. Don't reflexively flip to Codex's FAIL on this signal alone; verify the data-loss scope.
- `feedback_codex_fail_loud_diagnostic_blocker.md`: when the diagnostic callback ERROR+continues but the headline artifact is preserved, PASS with WandB-flag standing recommendation. Same family: side-channel failure with preserved headline = PASS-class.

**Origin:** task #508 round-3 reconcile (2026-06-07). Codex FAILed `dispatch_508.py:582-609` on the `try/except` + unconditional `EPM_DELETE_INTERMEDIATE_FT_CKPTS` cleanup. PASSed because (a) cell endpoint (frac_1.00) preserved by carve-out, (b) H1/H2/H3 read endpoint eval JSONs not trajectories, (c) env-flag operational recovery available. Hard standing recommendation: pod-side launcher MUST set `EPM_DELETE_INTERMEDIATE_FT_CKPTS=0`.
