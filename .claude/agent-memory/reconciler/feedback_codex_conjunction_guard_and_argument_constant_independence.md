---
name: codex-conjunction-guard-and-argument-constant-independence
description: "Codex r3 code-review overreach: reads the FIRST completion-record write as 'published early' without tracing the resume guard's full CONJUNCTION; demands 'independent emission' of an argument-passed constant (zero added discrimination); charges 'fabricated coverage' against a precisely-inventoried marker (#2215 dbe r3)"
metadata:
  type: feedback
---

Three Codex code-review overreach shapes adjudicated in one round (#2215
`discrimination-battery-expansion` r3, `epm:code-review v5` PASS vs Codex FAIL
— reconciled PASS, 3 majors discarded, 1 residual CONCERN upheld):

1. **Conjunction resume guards misread as single-record.** Codex flagged
   "completion record written at :2004 BEFORE the final upload at :2007 → a
   late-leg failure leaves a fresh record and re-entry SKIPS". The guard was a
   CONJUNCTION (`_finalize_complete` = regime-matched upload_done AND
   regime-matched sentinel), with the SECOND half written only after the leg
   Codex worried about, and `--force` quarantining BOTH halves before any
   work. Adjudication recipe: enumerate the guard's FULL predicate + every
   writer of each half + the force/quarantine path, then walk each
   driver-reachable crash point — the "stale composition" usually requires
   exogenous state mutation. The code's own comment ("the record the resume
   guard reads") can be imprecise and feed the misread; verify the PREDICATE,
   not the comment.

2. **"Independence" demands for argument-passed constants.** Codex demanded
   CAPTURE-EMITTED per-row tail-token ids as gate independence. The capture
   RECEIVES `eot_ids` as an argument and assembles rows from it, so any
   capture-emitted copy is definitionally the caller's list — the demanded
   mechanism has ZERO added detection power, and the hypothesized bug class
   (shared wrong derivation of the constant) escapes the demanded fix
   identically. Test: ask "under the flagged bug, would the demanded record
   DIFFER between the two sides?" If no, discard the blocker; the real
   independence lives in self-re-derived quantities (re-tokenized lengths) +
   an independent-forward cosine, which is what to verify instead.

3. **"Fabricated coverage" vs a precise test inventory.** A fabrication
   charge requires a claimed-but-absent artifact. When the marker names its
   tests one-by-one and each exists and does what is described, "only phase C
   has the full end-to-end composition" is a standing recommendation, not a
   BLOCKER — the marker never claimed per-phase composition tests.

**Why:** all three majors would have forced a re-roll of a round whose
mechanisms were correct and pinned; the one REAL residue (a `resume_skip`
default silently retaining stale REMOTE bytes of the exact file a delta pass
exists to push — contradicting the round's own "resume_skip=False on all
mutable stable-name uploads" claim) was persisted as CONCERN.

**How to apply:** on any Codex FAIL over resume/completion guards, trace the
guard's full predicate and every writer before crediting an ordering claim;
on any "X must be independently emitted" blocker, check whether X is an
argument-passed constant on both sides; on any fabrication charge, diff the
marker's LITERAL claims against the artifacts (cf.
[[claude-clean-result-critic-underapplies-spec-text]] for the inverse
duty on Claude's side).
