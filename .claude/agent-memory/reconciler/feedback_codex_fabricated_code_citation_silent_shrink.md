---
name: Codex fabricates a code citation for a silent-shrink / fail-fast BLOCKER
description: Codex code-review BLOCKER cites file:line + a code body (e.g. `r_plus.get(source)` / `skipped_no_rplus += 1; continue` emitting `status: ok`) that does NOT exist; grep the literal tokens before crediting, and check whether the production path actually fail-loud-validates the denominator
type: feedback
---

When Codex raises a "silent denominator shrink" / "Fail-fast violation"
BLOCKER, it sometimes cites a precise `file:line` AND quotes a code body
that is NOT in the file. Discard the blocker only after a TWO-STEP check —
the fabrication is the tell, but the real adjudication is the fail-loud path.

**How to verify (both steps, every time):**
1. **Grep the literal tokens** Codex quoted (`r_plus.get`, `skipped_no_rplus`,
   the alleged `status: "ok"` branch). Zero hits = the cited code does not
   exist; the line number usually lands on an UNRELATED line (in #667 r1,
   "line 551" was run_a37's `n_sources_with_v0_cneg` coverage-REPORTING line,
   not a shrink). A blocker citing code that isn't there is adjudicated
   against a phantom — it cannot carry a FAIL.
2. **Trace the REAL denominator path.** A silent-shrink blocker is only real
   if the production run can lose cells/sources WITHOUT raising. EPS analysis
   dispatchers built post-#658 typically run a battery of fail-loud
   `CoverageError` validators BEFORE any `run_a*` (in #667: `validate_r_b_coverage`
   / `validate_sigma_c_coverage` / `validate_g_meta_coverage` / `validate_cid_coverage`
   at the top of the analysis driver, each raising on a miss; the per-cell
   `gc is None` also RAISES, and the short branch emits `status: insufficient_cells`,
   not `ok`). `validate_g_meta_coverage`'s docstring is literally
   "never silently shrink the denominator." When that battery exists and fires
   on the real path (skip flag = False), the silent-shrink premise is false →
   PASS.

**Why:** false FAIL forces a re-roll of a correct fix; the fabricated-citation
class is high-confidence-discardable because the grep is decisive. Do NOT defer
to the precise-looking `file:line` — verify it points at the quoted code.

**Distinct from siblings:** this is NOT `feedback_codex_overreads_plan_prose`
(that's plan text, this is invented CODE), NOT
`feedback_codex_litigates_pre_existing_in_round_n` (provenance), and NOT
`feedback_codex_hardening_beyond_minimal_port_contract` (real-but-out-of-scope
hardening). Here the cited artifact location is simply WRONG.

**Caveat (Step 2 of reconciler.md still applies):** verify the unanchored/real
intent too. If the grep disproves Codex's CITATION but YOU find a genuine
silent-shrink elsewhere on the production path, the finding is re-anchored by
your own citation and adjudicated on its merits. In #667 the opposite held —
the path was provably fail-loud — so PASS.

Ledger:
- #667 r1 (code-review; Codex FAIL on `r-plus-coverage-silent-shrink`
  citing analysis.py:551 `r_plus.get(source)` — 0 grep hits, four upstream
  CoverageError validators present → PASS, Claude was right).
- #667 r2 (code-review; Codex FAIL on `hf-only-r-plus-resume-skip-shrinks-analysis`
  citing dispatch.py "near line 1110" — file is 1000 lines — and analysis.py
  477/480/551/554 `r_plus_dir`/`r_plus.get`/`phase_reextract_analysis`/
  `skipped_no_rplus`, ALL 0 grep hits. Described an "HF-only resume-skip" that
  doesn't exist: the real `_filter_resume_skip` gates on a LOCAL `.done` sentinel,
  HF is upload-only, and analysis runs in the same `all`-phase session that just
  extracted locally. Demanded fix already present (`validate_g_meta_coverage` /
  `validate_cid_coverage`, both BLOCKER 3, both HALT) → PASS, Claude was right).

**Pattern strength signal:** this twin fabricated `file:line`+nonexistent-symbol
citations on the SAME task across TWO consecutive code-review rounds (#667 r1, r2).
A repeat fabrication on the same artifact is a strong prior that a third
silent-shrink/fail-fast FAIL from the same review is also phantom — still grep
the literals (the discard must be evidence-anchored), but expect the hits to be
zero. The recurring tell: a precise-looking line number that, when read, lands on
a coverage-REPORTING / unrelated line, plus a quoted symbol grep proves absent.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex fabricates a code citation for a silent-shrink BLOCKER](feedback_codex_fabricated_code_citation_silent_shrink.md) — `file:line` + quoted code body (`r_plus.get`/`skipped_no_rplus` emitting `status: ok`) that doesn't exist (grep 0 hits); the real path runs fail-loud CoverageError validators before analysis → PASS. #667 r1.
