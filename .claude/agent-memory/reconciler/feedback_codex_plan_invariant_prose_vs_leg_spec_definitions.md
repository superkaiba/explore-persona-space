---
name: codex-plan-invariant-prose-vs-leg-spec-definitions
description: Codex FAILs a plan-faithful implementation by quoting the plan's high-level invariant prose ("incomplete read must never yield X") against the plan's own concrete leg/parse specifications; resolve the invariant's terms via the plan's operational definitions, and test whose RECOVERY the proposed fix would brick (#2328 r1)
metadata:
  type: feedback
---

Rule: when a Codex Critical claims an implementation "violates the plan's
load-bearing invariant", do NOT adjudicate on the invariant's prose alone —
find the plan section that OPERATIONALLY DEFINES the invariant's terms and
check whether the flagged behavior is itself plan-specified there. Then run
the recovery-semantics test: for each persistent state the finding names,
ask whether the implementation's verdict enables the CORRECT recovery and
whether the proposed fix would brick it.

**Why:** #2328 r1 (code-review, `task.py marker-status`). Codex raised a
3-row Critical class: all three JSONL legs tolerant-skip malformed/partial
rows yet return `ok-found`, so a skipped queried row can reach actionable
`absent`/rc 4 — "violates the load-bearing 'incomplete read never yields
absent' invariant". Every cited line was mechanically accurate. But plan
v2's MF-3 core scoped leg errors to OPERATIONAL failures verbatim ("ANY
operational failure is `error`", plan :71), and the plan's own leg specs
MANDATED the tolerant readers (HEAD: "the EXISTING `_iter_jsonl_text`";
worktree: "`_iter_jsonl` (existing reader, unchanged)"; ledger: "malformed
lines skipped, count surfaced in the output" — and `n_malformed` WAS
surfaced, task_workflow.py:6350). The critique rounds that produced the
invariant had named three fail-open paths, none of them parse-skips. On
recovery semantics the plan's choice was also RIGHT: a persistent malformed
row is a crash-truncated append whose `post-marker` never returned success
— the marker was never posted, so `absent`+re-post is the correct recovery
(the #1367 tail-seal makes it safe); Codex's fix (nonzero malformed ⇒
`unknown`) would have permanently rc-5'd every query on any task whose
committed append-only events.jsonl carries one historical partial line.
The only genuinely new residual (a microsecond torn read mid-`>PIPE_BUF`
append) was self-healing under the shipped delayed-re-read guidance and in
the same priced class as the plan's accepted §12-assumption-11 residual.
Verdict: PASS; BLOCKER downgraded via `defer-concern --by reconciler`.

**How to apply:** fires whenever a twin's FAIL rests on invariant-violation
framing against a plan-reviewed design. (1) Grep the plan for the invariant
term ("incomplete", "complete read", "failure") AND for the flagged
mechanism ("malformed", "tolerant", the named helper) — a plan that
mandates the exact reader/behavior the finding attacks converts the
Critical into a design re-litigation (the #480/#543 methodology-choice
family), routable as a follow-up observation, never a code-review block.
(2) Run the recovery-semantics test before crediting the stated impact:
enumerate the persistent states that produce the flagged verdict and check
which action (the implementation's vs the fix's) is the correct recovery in
each. (3) Claude-side calibration from the same round: Claude's PASS was
correct but had never ANALYZED the tolerant-parse channel at all — a
correct verdict can carry an analysis blind spot, so trace the losing
side's channel yourself rather than inferring its emptiness from the
winning side's silence.
