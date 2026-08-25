---
name: smoke-exit1-external-denial-adjudication
description: "A smoke phase reporting exit 1 root-caused to an EXTERNAL access denial persisted as an open user-action BLOCKER gets a compose-time both-routes adjudication note (not silence, not pre-decided); GPU-residual labeling variants of the carve-out are steered present-but-imperfect"
metadata:
  type: feedback
---

When the inlined implementation marker's `## Smoke run` carries a phase with a
NON-ZERO exit code that BOTH markers (impl + smoke-arch) root-cause to an
EXTERNAL access denial — gated HF dataset 403, missing credential — already
persisted as an OPEN user-action BLOCKER concern gating pod dispatch (#2546 r1,
2026-08-24: `taur-gated-access-blocked`; staging `--smoke` exit 1 at the
TAUR-Lab gated load, all upstream stages clean, `(d)` call-out + copy-pasteable
re-run command present):

1. **Do not stay silent** — Step 0.6's literal trigger list includes "a
   non-zero exit", so an adversarial twin left alone will FAIL
   `smoke-run-missing` on a disclosed fail-loud firing.
2. **Do not pre-decide either** — add a compose-time-facts note that states the
   verified facts (exit code, root-cause agreement across both markers, the
   persisted BLOCKER id, upstream-stages-clean, the (d) call-out) and hands
   BOTH routes: a FAIL keyed solely on the disclosed, persisted,
   user-action-gated exit-1 rests on presentation of present evidence
   (Step 0.7 rule 1); a FAIL is warranted if the disclosure is INACCURATE
   against the code or a genuinely uncertified phase exists. Pair it with the
   Step 0.8 duty split: verify the code fails LOUD on the denial (no silent
   skip/fallback), never re-raise the already-persisted blocker as a new
   finding.
3. **Carve-out labeling variants:** GPU-bound legs enumerated via a section-
   head pod-P1 certification statement plus per-phase "GPU residual" lines —
   rather than the literal `### <phase> — Carve-out (GPU-bound)` title — are
   steered as the present-but-imperfect case (judge the three substitute-
   coverage items substantively), not absence.

**Why:** the composer is the only party who has verified the cross-marker
root-cause agreement and the ledger state; without the note the twin either
false-FAILs (the #489-class costume: FAILing on evidence that is present) or,
told too firmly, rubber-stamps a disclosure that might not match the code.

**How to apply:** any compose where the smoke section has a non-zero exit tied
to a persisted external-dependency BLOCKER. Also confirmed this round:
worktree plan at the FROZEN-status dir (`tasks/planning/<N>` vs brief's
`running/`) byte-identical to canonical v4 ⇒ path-reference with the corrected
path + an explicit "the brief's running/ path does not resolve here" line.
Related: [[whole-round-unsplit-compose]],
[[worktree task-folder status can be stale in EITHER direction]],
[[brief-named concern adjudication]].
