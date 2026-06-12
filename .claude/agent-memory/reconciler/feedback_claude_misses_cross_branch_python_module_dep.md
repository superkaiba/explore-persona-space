---
name: claude-misses-cross-branch-python-module-dependency
description: Claude code-reviewer PASSes a plan that says "reuse module X from sibling branch Y" without verifying X is importable from THIS branch's worktree; the lazy `from sibling.experiments.X import ...` lives inside a function body so module import succeeds and only fails at runtime call.
metadata:
  type: feedback
---

When a plan §"Reuse, do not reimplement" sanctions importing a module from a sibling experiment branch ("imported from `scripts/i489_phase1_predictors.py` if reachable via PYTHONPATH; otherwise minimal local copies"), the implementer often writes lazy imports inside function bodies that pass `python -c "import script"` but fail at runtime call. Claude's plan-adherence walk-down ticks the "Reuse" row implicitly under the literal plan-row check but never greps the WORKTREE's filesystem (NOT main, NOT origin/sibling-branch) for the actual `.py` file existence. Codex naturally does `find src/.../experiments scripts -name '*<sibling-id>*'` and catches the zero-hits.

**Why:** Plan §"Reuse" implicitly trusts the sibling-branch artifact exists at runtime. But the pod runs `git pull origin issue-<N>` which pulls ONLY the `issue-<N>` branch tree. If the sibling artifact (`i489_contexts.py`) was added on `origin/issue-489` and never merged to main, it is NOT on the issue-501 worktree tree. The Phase 0 parent-ready check (which validates the sibling EXPERIMENT's adapters at HF Hub) does NOT validate the sibling's Python MODULE on the pod's local filesystem. Three different layers — Python module, HF Hub adapter, plan rationale — and Claude assumes the HF Hub layer covers all three.

**Canonical pattern (issue #501 round 1):**

- Plan §"Reuse, do not reimplement": "imported (not re-coded) from `scripts/i489_phase1_predictors.py` if reachable via PYTHONPATH; otherwise minimal local copies adapted by changing the prompt-construction call from `build_union_prompt(ctx, q, tok)` to `build_mt_prompt(ctx, q, tok)`."
- Implementer ships lazy imports: `i501_phase1_predictors.py:142, 189, 336, 370` all do `from explore_persona_space.experiments.i489_contexts import (UNION_BY_CID, UNION_CONTEXTS, build_union_prompt)` INSIDE function bodies.
- The module is on `origin/issue-489` (sibling branch, not main), and #489 is RUNNING (not yet promoted).
- Claude walks the plan-adherence table, marks "✓ implemented (reuse)" with evidence `i501_phase1_predictors.py:336 from ...i489_contexts import build_union_prompt`. The presence of the IMPORT STATEMENT is the evidence — not the existence of the source module on the worktree filesystem.
- `git show main:src/explore_persona_space/experiments/i489_contexts.py` returns "does not exist in 'main'"; `ls .claude/worktrees/issue-501/src/explore_persona_space/experiments/` returns ONLY `i501_mt_contexts.py`. Pod's `git pull origin issue-501` would import-fail at runtime.
- The Phase 0 parent-ready check (`i501_phase0_parent_ready_check.py`) validates the sibling EXPERIMENT's HF Hub adapters at `superkaiba1/explore-persona-space:adapters/i489_<cid>_seed42_frac<F>` — it does NOT touch the Python module layer at all.

**How to apply:** When reconciling a Claude PASS vs Codex FAIL where Codex tags `missing-<sibling-id>-runtime-dependency`:

1. `git show main:<sibling-module-path>` — if this returns "does not exist in 'main'", the module is not on `main` (the source of truth for cross-branch sharing).
2. `ls <worktree-root>/src/.../experiments/` on the current `issue-<N>` worktree — if no `<sibling-id>*.py` files present, the module is NOT importable from this worktree's pod-pull.
3. `grep -rn "from explore_persona_space.experiments.<sibling-id>" <worktree-root>/` — enumerate every import site. Lazy imports inside function bodies still fail at runtime, even when `python -c "import script"` succeeds.
4. Check the plan's §"Reuse" / §"Dependencies" section. If it sanctioned "reachable via PYTHONPATH OR minimal local copies", neither is implemented when (a) the module is missing from the worktree AND (b) no fallback `try: from external import X except ImportError: <local copy>` exists.
5. Check whether the parent task is at status `running` / `verifying` / `interpreting` (not yet `completed` / `awaiting_promotion`) — if so, the sibling module has NOT been merged to main and the import will fail.

This is a real-blocking FAIL: Phase N (the consumer) is run-broken; the experiment cannot produce a headline result. Per critic Bar this is uninterpretable.

**Sibling-smell:** A plan §"Reuse" rationale that says "build_union_prompt" / "the cosine layer sweep primitive" / "import from #<sibling>" combined with the implementer using LAZY function-body imports (which pass naive import-checks). The lazy import is the lookahead-defense-against-circular-import idiom, but it also defers the ModuleNotFoundError until the function is called — which often means the bug surfaces only on pod, not on the dev VM.

**Fix paths the reconciler should suggest in the verdict:**

1. Merge `origin/issue-<sibling>` into `issue-<N>` (re-rebase before sibling promotes).
2. Cherry-pick the specific `.py` file from `issue-<sibling>` onto `issue-<N>` (with a NOTE-tracking commit).
3. Write the "minimal local copies" the plan sanctioned (carries cosine-comparability risk vs the sibling's matrix — flag in the verdict).

Related: [[claude-misses-same-file-siblings]] (same file, different code path); [[claude-skips-caller-grep]] (orphaned helper); [[codex-env-var-orphan-unreachable]] (orphan dead code in entry points). This entry covers a NEW failure mode — cross-branch Python module dependency unresolved at the worktree filesystem layer.

Origin: task #501 round-1.
