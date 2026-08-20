---
name: syntactic-test-pins-and-vacuous-empty-gates
description: Claude code-review accepts inspect.getsource/string-position tests as regression pins and never probes a new set-equality gate's empty-selection state; probe both by semantic-bypass + empty-input mutation. #2329 r2.
metadata:
  type: feedback
---

**Rule:** when a round's acceptance contract is about gates/tests that cannot fail, run TWO probe
classes the Claude side systematically skips:

1. **Semantic-bypass mutation on any `inspect.getsource` / string-position regression pin.** A test
   asserting literal positions (`src.index(tok) < src.index("if cfg.smoke")`, `"if not cfg.smoke"
   not in src`) is evaded by `assert cfg.smoke or (...)` + an `if cfg.smoke:`-stub block placed
   after the literals — the pin stays green while the exact pinned defect is restored. The
   [[feedback_codex_meta_test_blocker_on_verified_fix]] demote-to-CONCERN rule does NOT apply when
   the test never EXECUTES the module-under-test (its precondition is the shared-code-path
   mechanism: a revert must flip an executed assertion). Companion check: a dispatcher-side
   missing-FILE test (rc=NN on absent gate file) pins file presence, never content consumption —
   it cannot see a driver-level bypass.
2. **Empty-selection state of any NEW set-equality coverage gate.** `expected == present` derived
   from the same selection passes vacuously when the selection is empty — delete every selected
   row (keep a nonselected shard row so upstream file-count checks pass) and check the builder
   raises before constructing spend-bearing units. #2329 r2: `_assert_coherence_coverage` passed
   vacuously and `_build_sides` built 192 anchor-only Batch-API units; the fresh path was
   pilot-shielded but a wave re-run against an already-passed pilot report re-built units with no
   nonempty check.

**Why:** #2329 r2 (2026-08-20) — Claude composed CONCERNS off removal-mutations (which the
syntactic pins DO catch) while Codex's semantic mutations stayed green; both were factually right
about their own probes. The reconciler re-ran both divergent mutations on a scratch `git archive`
tree and upheld FAIL. Weight calibration: the vacuous PRODUCTION gate was the verdict-driver
(silent estimand corruption + API spend, pre-launch, ~5-line fix); the hollow TEST pins were
Real-blocking only as riders on an already-bouncing round — standing alone on a PASS-worthy round
they are demotable to tracked concerns since the shipped code was correct.

**How to apply:** any code-review reconcile where one side's mutations went red and the other's
stayed green — first check whether they mutated the same thing (removal vs semantic bypass);
re-run BOTH on a scratch copy with the worktree venv (`uv run` in a scratch tree tries to build a
fresh venv and dies on platform wheels — use `<WT>/.venv/bin/python -m pytest` with cwd = scratch
tree; test files that `sys.path.insert(Path(__file__).parents[1]/"scripts")` load scratch scripts
automatically).
