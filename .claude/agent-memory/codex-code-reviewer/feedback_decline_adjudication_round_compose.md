---
name: decline-adjudication-round-compose
description: "Decline-adjudication round (#2546 r21/v21): prior round was PASS+CONCERNS (zero blockers, NO reconciler) and the implementer DECLINED parts of the twin's own fix sketch - contract envelopes are the twin's OWN prior verdict (tags stripped, CONCERN:: rows blockquoted) + parallel-Claude Minor EXCERPT; declines get headline D-lanes (DECLINE-SOUND|DECLINE-WRONG, grade the reasons not the act); composer-added probe mutations (dead-branch vs documented-FN) split expected-red/expected-green; verified-open census miss recurred"
metadata:
  type: feedback
---

From #2546 r21 compose (sentinel v21, 2026-08-27), the sequel to
[[second-upheld-blocker-execution-round]] - first round in the lineage with
NO reconcile record in play:

1. **PASS+CONCERNS prior round (zero blocker tags, no reconciler) closing
   the twin's own persisted rows composes with the twin's OWN prior verdict
   as the contract envelope**: strip its head/close tags at compose time
   (assert v<n-1> tags == 0 in the final prompt), BLOCKQUOTE its
   `CONCERN:: ` rows (`> CONCERN:: `; assert line-start rows == 1 = the
   template grammar row, blockquoted == 2), strip the session footer. Add
   the PARALLEL Claude verdict's Minor section as a small EXCERPT envelope
   (convergence evidence, "context not a contract"; no tags in excerpts).
2. **Implementer DECLINES of the twin's own fix-sketch components get
   headline D-lanes placed FIRST in the review focus** (the orchestrator
   ordered them front and center): quote the marker (b) reason verbatim,
   spell the falsification duty (for a dest-binding decline: the res x dest
   failure LATTICE + the cross-wiring case + the brittleness weigh), forms
   DECLINE-SOUND | DECLINE-WRONG <concrete missed case - substantive>.
   Both-directions author-neutrality: neither demand the sketch verbatim
   (intent-adherence = the concern row's summary, not the Fix line's
   letter) nor wave declines through; "grade the reasons, not the act of
   declining - the disclosed-decline shape is the honest one."
3. **Composer-added probe mutations beyond the marker's claimed battery,
   with expected-red vs expected-green framing:** M-x1 dead-branch
   (`if False:` wrap of an existing gate, arg untouched) targets the
   ADDRESSED-CLAIM WORDING ("dead/unrelated calls now fail") - lineno-based
   (a)+(b) rules are reachability-blind so it plausibly passes; a GREEN
   result feeds BOTH the closure line (partial closure) and the D2
   behavioral-tests decline (only behavioral tests catch reachability-dead
   gates). M-x2 rebinding confirms the DOCUMENTED false negative - expected
   GREEN; a red means stronger-than-documented, fine. Say which color each
   probe expects or Codex misreads its own results.
4. **Closure fence for twin-raised (not reconciler-downgraded) rows:**
   honest partial/non-closure of a CONCERN/NIT row re-raises the SAME id at
   the row's own grain (the calibration section: short-of-blocker goes to
   the ledger, not r22); FALSE closure claims (battery direction that does
   not fail, false operator-message fact) route at the ordinary bar
   (substantive). Three-way forms: VERIFIED-ADDRESSED |
   PARTIALLY-ADDRESSED/NOT-ADDRESSED (re-raise same id) | FALSE-CLOSURE /
   CLAIM-FALSE (substantive).
5. **The verified-open census miss RECURRED:** r20's 26-open snapshot
   missed `partial-resume-rel-draw-reallocation` (latest event
   `verified-open`) because the census keyed on `raised` only. Pin census
   MUST bucket raised OR verified-open as OPEN; inline the verified-open
   row's own event (not a stale raised row) in the armed-rows envelope and
   name the r20 omission as a bookkeeping frame fact, never a finding.
6. **Operator-message factual claims (an "no flag or env toggle" class
   assertion) get a composer source trace handed as fact anchors:** the
   constant def (hub.py:34, no env read), every consumer hard-pin
   (fit_cells :279/:339/:431/:890/:2514/:2550), the file's ONLY env reads,
   and the nearest counter-example CANDIDATE (hub.py:661
   `EPM_HF_OVERFLOW_ROUTING`, composer-read as upload-side only) -
   adjudication stays with Codex (CLAIMS-TRUE | CLAIM-FALSE - "a false
   operator-facing claim forecloses a recovery route").
7. **In-place mutate/restore protocols get composer sha-verification +
   an explicit not-yours line:** verify the worktree file's sha256 head ==
   the marker's pristine pin (proves the restore) AND worktree status
   clean, then forbid the twin from repeating the in-place protocol - its
   battery is scratch-tree only (`git archive <head-sha> scripts tests src
   pyproject.toml`, restore via `git show <head-sha>:<file>` + sha check).

Compose script: /tmp/codex-2546-v21-compose.py (prompt
/tmp/codex-prompt-issue-2546-v21.md, 99,873 bytes; envelope set: brief /
impl v21 / r20 codex verdict (stripped+blockquoted) / r20 claude Minor
excerpt / smoke-arch v15 / 6 armed rows incl. the verified-open one, pinned
to the impl ts). WT-token count assert caught a real miscount live (5 not
4: the restore command line).
