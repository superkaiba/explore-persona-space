---
name: Clean-result bare-#N refs + missing reuse path are real Lens 2/5 FAILs Claude PASSes
description: On a v4 clean-result PASS-vs-REVISE split, verify bare #N outside Goal/footer (Lens 2) and every reuse bullet's (a)/(b)/(c) (Lens 5) against the SPEC text — both are grounded FAILs Claude misses while Codex over-fires on adjacent soft items.
type: feedback
---

On a v4 clean-result-critic PASS (Claude) vs REVISE (Codex) split, two
Codex findings recur as REAL grounded blockers that Claude's PASS misses,
verifiable against SPEC text in one read each:

- **Lens 2 — bare `#N` refs outside `## Goal`/footer.** SPEC v4 + Lens 2:
  "FAIL on a `[#K]` link or bare `#K` in `## Takeaways`, `## Methodology`,
  `## Results`." The worst offender is the `## Methodology → **Training:**`
  hyperparameter table's **Source column** (`#658 RIDGE_LAMBDAS`,
  `#537 training body`, etc.) plus Data-extraction prose ("trained in
  #537; extracted in #667") and `## Results` captions. This is a
  structural FAIL, not a nit. Reframe sources descriptively; the
  fact-of-reuse + `[#M](...)` link belongs ONLY in the `**Repro:**`
  reuse-provenance bullets and `## Goal **This experiment in context:**`.

- **Lens 5 — reuse bullet missing (b) the permanent path.** A
  `**Repro:**` reuse bullet must carry (a) `[#M](...)`, (b) a permanent
  HF `/tree/<sha>` or repo-relative `eval_results/issue_M/...` path, AND
  (c) a one-line fitness rationale. Claude's stock phrasing "all reused
  artifacts carry pinned path + fitness rationale" is a CLAIM to verify,
  not a fact — at #722 r1 it was FALSE for one bullet (`i537_* (r=32,
  rsLoRA) — fit: consumed only transitively…` named NO path). Read every
  reuse bullet; a name pattern (`i537_*`) is not a path. Missing any of
  (a)/(b)/(c) = hard Lens 5 FAIL.

**Calibration: discard the Codex over-fire mix that rides along.** Same
#722 r1 verdict carried five DISCARD-class items — re-verify each against
SPEC before crediting:
- L3/L7 "could not resolve figure URLs / run audit" = sandbox BLOCKED, not
  findings (Claude verifies verify_task_body + audit PASS).
- L4 "Takeaways not numbers-first" — Lens 4 = "leads with **OR bolds**" the
  number; a bolded `**3.3×**` after a condition label SATISFIES it.
- L6 "Results must be prose, not bullets" — the v4 Results interpretation
  beat is explicitly "1–3 sentences/**bullets**"; bullets are allowed.
- L12 "Takeaway bullets >30 words" — the word cap is verifier check-20
  **WARN**, not a FAIL; never a blocker on its own.
- L15 "title not scoped to ridge-only/n=16" — Lens 15 fires only on a
  headline resting on a *contaminated / failed-data-gate* arm; a headline
  resting on the arm that PASSES the validity gate (+ CI-excludes-zero) with
  a "(LOW confidence)" tag and body scope-caveat does NOT trip it.

**Net rule:** verify Lens 2 (bare #N) and Lens 5 (reuse path (b)) FIRST on
any clean-result split — they are the two grounded FAILs; any ONE → REVISE.
Then strip the L4/L6/L12/L15 over-fire as SPEC misreads. #722 r1.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Clean-result bare-#N refs + missing reuse path (Lens 2/5)](feedback_clean_result_bare_issue_refs_and_reuse_path.md) — v4 split: bare #N in Methodology table Source col / Results caption (L2) + reuse bullet missing the permanent path (b) (L5) are grounded FAILs Claude PASSes; discard the L4/L6/L12/L15 Codex over-fire as SPEC misreads. #722 r1.
