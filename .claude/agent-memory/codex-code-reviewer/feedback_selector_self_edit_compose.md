---
name: selector-self-edit-compose
description: Compose recipe when the diff target IS scripts/select_step9c_tests.py (the Step 9c gate instrument) — three-direction stakes, fail-closed FP hypotheses incl. partially-materialized checkouts, fixture-site sweep, containment-vs-exact-set steer, self-referential sweep_scope reconciliation (#2537 r1)
metadata:
  type: feedback
---

Compose recipe for a round whose diff EDITS `scripts/select_step9c_tests.py`
itself (the fleet gate instrument — recurring: #1496, #1589, #1688, #2412,
#2537). Extends [[infra-wf-fix-lint-gate-compose]] and
[[pytest-guard-conftest-compose]]; first used #2537 r1 (2026-08-24).

1. **Stakes run THREE directions, stated up front:** false PASS ships a
   broken fleet gate; an over-strict new arm (fail-closed refusal, meta-test
   FP) fleet-blocks HEALTHY landings; a reachable crash in the new code
   wedges every session's gate — worst of the three. Codex weighs
   over-strictness findings equal to bugs.
2. **Fail-closed refusal duties decompose as named hypotheses:** healthy-tree
   no-fire (what predicate marks a member "missing"? a PARTIALLY-MATERIALIZED
   / sparse checkout is the fleet-blocking FP class), mode-isolation (quote
   the early-return showing `--map-files` exits before the block),
   sanctioned-descope trace (the remedy line must name a path that actually
   empties the check), crash-safety.
3. **Fixture-site sweep is a standing duty on any selector behavior change:**
   the plan's no-breaks argument covered only the eponymous pin file and
   missed `tests/test_step9c_base_identity.py`'s own `sel.main` fixtures
   (the #2537 repair commit). Compose a sweep of ALL selection-mode
   `sel.main` / subprocess-selector call sites over fixture trees, and ask
   whether the shipped repair is CORRECT (does materializing invariant
   members change what the file's other asserts test?) not merely green.
4. **Incident-trace tests on map OUTPUT get the containment-shape steer:**
   pair-ABSENCE/presence containment, never exact set-equality or counts —
   the diff under review itself adds pairs to that output, so an exact
   assert fails on its own change.
5. **Self-referential gate-scope reconciliation:** when the task exists
   because a prior round mislabeled `sweep_scope`, compose an explicit
   re-run-the-fragments duty (`git -C . grep -l` per fragment over tests/,
   reconcile vs the claimed hit lists + disclosed supplements) AND a
   token-semantics judgment ("is the label honest for what was done").
6. **Brief-ordered plan inlining with an identical worktree copy at a
   NON-brief status dir:** #2537's brief named `tasks/running/…` while the
   worktree froze at `tasks/approved/…` (byte-identical). Inline per the
   brief, state BOTH facts in the prompt (copy identical at approved/; the
   running/ path unreachable in the sandbox), note the probe in the return.
7. **Meta-test invariants get a vacuous-truth probe (P7 shape):** beyond
   manifest + tuple registration, ask what pins a NONZERO discovery floor /
   positive control — a registered invariant whose discovery set can be
   empty passes forever (hollow gate).

**Why:** a generic infra compose would have narrowed all of these (#606
twin-omission class): the selector is simultaneously the diff target, the
gate that runs the diff's own tests, and the instrument the gate-scope line
is normally verified WITH — so every duty needs a no-uv static form.
**How to apply:** any round whose diff touches `scripts/select_step9c_tests.py`
selection behavior, `WORKFLOW_INVARIANT` membership, or the step9c manifest.
