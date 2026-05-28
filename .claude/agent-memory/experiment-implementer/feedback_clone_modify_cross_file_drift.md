---
name: clone-modify-cross-file-drift
description: When cloning an eval script and adding a new param thread (engine reuse, ckpt resolution, etc.), library helpers it calls almost certainly don't accept the new kwarg yet — sweep both layers together with an AST call-site vs signature check before launching to a pod.
metadata:
  type: feedback
---

When a new eval script is cloned from an old one and modified to thread a new parameter through several layers — e.g. `_run_seed_with_engine` → `run_smoke_gate` → `generate_completions(_with_history)` carrying a reused `llm` engine — the library helpers at the bottom of the call graph almost certainly do NOT accept the new kwarg yet. Fast-path crashes happen ON THE POD with `TypeError: foo() got an unexpected keyword argument 'X'` after a full provision + bootstrap + checkpoint-resolution + smoke-gate-init cycle (~5-10 minutes wasted).

**Why:** Task #399 cloned eval_issue377.py and added an `llm=llm` kwarg to every per-condition call (engine-reuse pattern). The script-local functions (`_run_seed_with_engine`, `run_smoke_gate`) were updated consistently, but the imported library helpers in `src/explore_persona_space/eval/generation.py` (`generate_completions`, `generate_completions_with_history`) were never extended. Three rounds of pod launches (rounds 5, 6, 7) crashed with the same TypeError class before round 7 added `llm: object | None = None` to both library functions.

**How to apply:**

1. **On any clone-and-modify of a script that adds new kwargs anywhere in the call graph**, immediately run a kwarg-vs-signature AST sweep BEFORE committing. Walk every `Call` node in the new script; for each call to (a) a script-local def or (b) an imported helper in `explore_persona_space.*`, check that every `kwarg=` appears in the callee's signature (use `inspect.signature(fn).parameters` for imports, AST for locals).
2. **A one-shot AST sweep is ~20 lines of Python** and runs in <1s — far cheaper than a pod-launch cycle. Template in task #399 round-7 marker.
3. **When extending a library helper for engine reuse / resource sharing**, follow the `owns_engine = llm is None` pattern so the default behavior is preserved exactly for ALL existing callers. Skip teardown (`del llm`, `torch.cuda.empty_cache()`) when caller owns the lifecycle. Document the contract in the docstring (which params are ignored when reusing).
4. **Sibling scripts at risk**: when this lands, related scripts that DON'T pass the new kwarg keep working (backward compatible). If they later want to retrofit the pattern, no library change needed.

**Related:** A `--dry-run` flag that mocks `LLM.generate()` with a `_DryRunLLM` stub would catch this whole class of bug in 30s on the dev VM without GPUs. Worth filing as `kind: infra` if this pattern recurs a third time.
