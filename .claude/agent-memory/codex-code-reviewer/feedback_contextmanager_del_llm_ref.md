---
name: contextmanager-del-llm-ref
description: contextmanager del inside finally does not free caller's 'as llm' binding; gc.collect() runs while LLM still referenced
metadata:
  type: feedback
---

When a `@contextmanager` yields an object, `del obj` in the `finally` clause only removes the generator-local binding. The caller's `with ... as llm` binding remains alive throughout `__exit__` execution. Therefore `gc.collect()` and `torch.cuda.empty_cache()` both run with the yielded object still strongly referenced — they don't free GPU memory.

**Why:** Python ref-count semantics: caller's `as llm` has refcount > 1 during `__exit__`. The LLM is freed only when the caller's variable goes out of scope (e.g., when `_run_cell_mode` returns).

**How to apply:** When reviewing any `@contextmanager` that yields a heavyweight object (LLM, GPU tensor, file handle) and claims to clean it up in `finally`, verify whether the caller's `as` binding is the sole reference. If not, the teardown guarantee is broken. Correct fix: use a holder wrapper (`holder.llm = None` before gc.collect) or yield a thin proxy that clears its internal reference on `__exit__`. This was flagged as a Major issue in task #365 round-10 review. See also [[feedback_scripts_import_chain]].
