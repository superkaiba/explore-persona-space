---
name: codex-fails-preexisting-resume-metadata-clobber
description: Codex FAILs a cap round on a pre-existing resume/fast-forward metadata clobber (timing walls/fences) with zero code consumers, while its own CONCERN:: persist row rates it CONCERN — consumer-grep + parent-commit semantics + operational-guard PASS (#2378 r10)
metadata:
  type: feedback
---

Codex code-review FAIL on "the declared fast-forward re-compose clobbers
measured walls / timing fences": the mechanism can be fully REAL (skip
branch omits `walls[name]`, all-skipped fanout records near-zero elapsed,
compose writes the dict verbatim + atomically overwrites + harvests) and
still not carry a FAIL — adjudicate with three checks before upholding:

1. **Verdict vs its own persist row.** Codex's `CONCERN:: CONCERN <id>`
   machine row rated the finding CONCERN while the verdict said FAIL — an
   internal severity contradiction; the ledger row is the calibrated read.
2. **Consumer-grep the "poisoned" field.** `fences_s_2x`/`measured_walls_s`
   had ZERO code consumers repo-wide (grep) — no downstream gate/sizing read
   them; the hypothesized "near-zero fence kills healthy wave-1 work" was
   not wired. Verdict logic + wave sizing read sibling fields (families /
   per_stage rates) that the round's fix makes CORRECT.
3. **Parent-commit semantics + round scope.** The walls behavior was
   byte-identical at the parent SHA and untouched by the round diff —
   pre-existing, and the round's declared purpose output (the accounting
   repair) is uncorrupted; only sibling metadata in the same artifact is,
   and it is git-recoverable.

**Why:** #2378 r10 (cap round). FAIL at the cap cannot bounce — it blocks
the task; the right instrument was PASS + a REQUIRED operational guard in
the reconcile body (per-key walls merge from the parent-SHA digest +
post-merge verification) with the concern held OPEN until the guard's
verification note lands.

**How to apply:** when a Codex FAIL rests on "the next operational step
will corrupt artifact X", (a) diff the round vs parent for the mechanism's
lines, (b) grep for actual consumers of the corrupted field, (c) check
whether the round's own purpose-output fields are among the corrupted ones,
(d) compare the verdict against Codex's persist-row severity. All four
pointing at recoverable-metadata ⇒ PASS + binding operational guard +
CONCERN persisted (dedupe against the codex row is a designed no-op).
Related: [[codex-fail-loud-diagnostic-blocker]],
[[codex-hardening-beyond-minimal-port-contract]].
