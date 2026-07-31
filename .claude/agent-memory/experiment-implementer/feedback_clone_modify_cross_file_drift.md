---
name: clone-modify-cross-file-drift
description: Cloning an eval script and threading a new kwarg (engine reuse etc.) almost always leaves imported library helpers with stale signatures; run an AST call-site-vs-signature sweep before any pod launch.
metadata:
  type: feedback
---

When a cloned eval script threads a new parameter through several layers (e.g. a reused `llm` engine through `_run_seed_with_engine` → `run_smoke_gate` → `generate_completions`), the library helpers at the bottom of the call graph almost certainly don't accept the new kwarg yet. The crash arrives ON THE POD as `TypeError: ... unexpected keyword argument` after a 5-10 min provision/bootstrap/smoke cycle.

**Why:** task #399 cloned eval_issue377.py with an `llm=llm` thread; script-local functions were updated, the imported `generate_completions(_with_history)` helpers were not. Rounds 5/6/7 all crashed on the same TypeError class before round 7 extended the library.

**How to apply:**
1. On any clone-and-modify adding kwargs anywhere in the call graph, run a ~20-line AST sweep before committing: walk every `Call` node; for script-local defs use AST, for `explore_persona_space.*` imports use `inspect.signature`; assert every `kwarg=` exists in the callee's parameters. <1 s vs a pod cycle.
2. When extending a library helper for engine/resource reuse, use the `owns_engine = llm is None` pattern so default behavior is exactly preserved for existing callers (skip teardown when the caller owns the lifecycle); document the contract in the docstring.
3. A `--dry-run` mode mocking `LLM.generate()` would catch this class on the dev VM without GPUs — file as `kind: infra` if the pattern recurs.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Clone-modify cross-file drift](feedback_clone_modify_cross_file_drift.md) — new kwarg threads leave library helpers stale; AST kwarg-vs-signature sweep (<1s) before any pod launch. #399.
