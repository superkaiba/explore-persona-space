---
name: claude-misses-cross-branch-python-module-dependency
description: Claude PASSes "reuse module X from sibling branch Y" plans on the presence of the IMPORT STATEMENT; lazy function-body imports pass import-checks but ModuleNotFoundError at runtime because the sibling module exists only on the sibling branch, not on this branch's worktree/pod tree.
metadata:
  type: feedback
---

**Rule:** when a plan sanctions importing from a sibling experiment branch ("reachable via PYTHONPATH; otherwise minimal local copies") and the implementer ships lazy `from ...experiments.i<M>_x import ...` inside function bodies, verify the MODULE exists on THIS branch's filesystem — the pod pulls only `issue-<N>`, and lazy imports defer the crash to first call (often on the pod, not the dev VM).

**How to apply:**
1. `git show main:<sibling-module-path>` — "does not exist in 'main'" means it's not on the cross-branch source of truth.
2. `ls <worktree>/src/.../experiments/` — no `<sibling-id>*.py` = not importable from this branch's pod-pull.
3. `grep -rn "experiments.<sibling-id>"` — enumerate every import site (lazy imports still fail at runtime).
4. Neither PYTHONPATH-reachability nor the sanctioned "minimal local copies" fallback implemented (no try/except local copy) → real-blocking FAIL: the consumer phase is run-broken.
5. Sibling task still at running/verifying/interpreting = its module is NOT merged to main.

Note: a Phase-0 parent-ready check validating the sibling's HF Hub ADAPTERS does not validate the sibling's Python MODULE — three layers (module / Hub artifact / plan rationale), Claude assumes Hub covers all three.

**Fix paths to suggest:** merge `origin/issue-<M>` into `issue-<N>`; cherry-pick the file (with a tracking note); or write the sanctioned local copies (flag comparability risk).

Origin: #501 r1 (`i489_contexts` lazy-imported at 4 sites; module only on origin/issue-489, #489 still running). Plan-stage analogue: [[feedback_claude_plan_cherrypick_closure_and_pin]]. Related: [[feedback_claude_misses_same_file_siblings]]; [[feedback_codex_env_var_orphan_unreachable]].
