---
name: claude-misses-block-contract-conformance-of-round-added-commands
description: Claude PASSes a round-introduced command inside a bounded/fail-open block without checking it against the block's OWN registered fence/failure contract (#2241 r2)
metadata:
  type: feedback
---

When a revision round ADDS a command inside a block that carries its own
registered contract (timeout fences, fail-open "every failure arm echoes +
falls through" semantics, a plan acceptance criterion like #2241 criterion 3),
re-check the NEW command against that contract — do not stop at its
happy-path semantics.

**Why:** #2241 r2 — round 2 added a `task.py view | jq` title resolver
between two fenced `gh` commands in the Step 5 draft-PR ensure. Claude's
PASS examined the `// empty` fallback ("benign, gh accepts") but never
checked (a) the missing timeout fence against the block's ":53-56 'Both
commands are timeout-bounded... FAIL-OPEN'" prose (now false — 2 of 3
commands), (b) the no-pipefail failure-masking (task.py RuntimeError → jq
exit 0 → degraded-title PR CREATED + memoized, instead of the registered
log+skip+retry), or (c) that the new behavioral pin's stub always succeeds
instantly, so it cannot cover either mode. Codex flagged all three (M1);
reconciler verified each and FAILed the round. Reachability mattered: every
`task.py` invocation — reads included — routes through
`_resolve_repo_root_cached` (task_workflow.py:429), which holds the #996
bounded 120 s rebase wait + detached-HEAD refusal, so the latency AND
failure premises were both real on the READ path.

**How to apply:** on any round adding a command to a fence/contract-bearing
block: (1) diff the block's invariant PROSE against the new command set
("Both commands" tells); (2) trace the new command's FAILURE exit into the
block's registered failure arms — an unrouted failure that still reaches
the create/act arm is an affirmative misfire, not hardening-beyond-scope
(the invented-contract discard does NOT apply when the contract is in the
block's own prose + plan); (3) check the round's new pins actually exercise
the failure/latency arms (an always-succeed stub covers neither). Sibling
memories: [[claude-misses-invariant-comment-smell]] (#505 r2),
feedback_claude_security_sweep_misses_data_into_shell_splice (#2241 r1 —
same task, prior round, same verify-happy-path-only shape).
