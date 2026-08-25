---
name: degenerate-input crash-fix compose
description: Crash-fix rounds whose fix ADDS a sanctioned smoke-conditional gate downgrade (Step 0.71 goes LIVE) plus a degenerate-report contract — composer verifies the marker's enumeration, runs the consumer-completeness sweep with recomputed anchors, and scopes the unenumerated tag to ADDITIONAL branches only (#823 r7cf)
metadata:
  type: feedback
---

When a crash-fix round's fix is a degenerate-input branch (e.g. #823 r7cf:
`ZeroDivisionError` at zero between-persona energy → None ratios + named
verdict + a NEW smoke-downgraded designed halt rc), three compose deltas
beyond the standard crash-fix shape (own impl marker + crash-diagnosis
envelope; see [[gate-leg demotion crash-fix compose]] for the sibling):

**Why:** the fix ARMS Step 0.71 (the diff adds a smoke-conditional gate
downgrade), the crash class is "consumer crashes on the new degenerate
shape" (so completeness of the consumer fix-out is the round's own
contract), and the dispatch note's consumer line anchors are pre-fix frames.

**How to apply:**
1. **Step 0.71 LIVE, enumeration composer-verified.** Grep the marker's
   `## Smoke run` block for the SMOKE BLIND-SPOT ENUMERATION naming the new
   downgrade at compose time; state "PRESENT, composer-verified" in the
   prompt and scope the `smoke-blind-spot-unenumerated` tag to ADDITIONAL
   unenumerated substitutions/downgrades the twin finds itself — else the
   twin FAILs on the enumerated branch. The adequacy/justification/
   production-halt-real adjudication stays theirs (hollow rc test →
   `hollow-verification-gate`).
2. **Consumer-completeness sweep: composer runs it, classifies it, hands
   the list.** Grep every reader of the changed fields (both scripts +
   figures + driver's own downstream blocks), RECOMPUTE post-round line
   anchors (dispatch-note cites shifted by the insertions — say "a stale
   line number in marker prose is never a finding"), and classify each hit:
   FIXED this round / key-presence-only (None-tolerant — verify) /
   production-only None-INTOLERANT residue with reachability facts (e.g.
   `abs(have - want)` on committed-banked `want`, NaN-riding-into-report) /
   other-issues' sibling hits (settled, don't chase). Adjudication +
   severity routing stays the twin's; the smoke-relaunch path (which
   consumers actually run at attempt N+1) is the FAIL bar, production-only
   residues are weigh-by-reachability.
3. **Prior round PASS+CONCERNS whose NIT the orchestrator hot-fixed in the
   PARENT commit:** nothing under closure — frame the parent hot-fix as
   settled/out-of-round with an additions-only diff proof for
   confirm-undisturbed, name the concern id in the head (the twin authored
   it — author-neutrality line), and count the parent SHA separately in
   body vs rest (RED-transcript `git show <parent>:` + parity-test
   provenance put it in the impl body twice).
4. Exact `== 0.0` float comparisons in the degenerate branch key: tell the
   twin exactness is the point (designed branch key) — adjudicate, never
   reflex-flag; but hand the subtle semantics (subnormal /k underflow takes
   the branch with energy > 0) as an attention item.
