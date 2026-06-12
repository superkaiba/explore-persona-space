---
name: Ruff strips unused module-level imports
description: The post-Edit ruff hook removes any top-level import with no in-file reference at edit time; sequence edits so the import and a usage land together, and re-run ruff check after to confirm survival.
type: feedback
---

The post-Edit formatter hook runs ruff with F401: a top-level `import X` with no reference in the file AT THE MOMENT THE HOOK FIRES is silently removed. Any multi-step edit sequence where the import lands before its usage loses the import.

**Why:** first bit on #280 v7 (`import lorem` wiped twice); recurred in six distinct orderings since (#405, #536, #570, #601, #606).

**How to apply** — make every edit that introduces an import also introduce a usage, or add the usage first:
- Side-effect-only import → add `_ = X` after it, or (better) lazy-import inside the using function.
- Import needed for a CLASS declaration → edit `class Foo(BaseFromX):` FIRST, then add the import (#405 r5).
- Writing a big file in chunks (Write + Bash heredoc appends) → the hook fires on the FIRST Write only and strips imports the later chunks need (appends don't re-trigger it); restore the full import block AFTER the last append, then `ruff check` (#536).
- Threading new names into an existing `from X import (...)` block across multiple Edits → land usage edits first, then rebuild the whole block programmatically (parse names, merge sorted, rewrite) + `ruff check --fix` for I001 (3 hits, #570).
- Round-N revisions → an Edit adding imports followed by a separate Edit adding usages gets stripped in between; put both in one Edit or extract a helper carrying imports + usages together (#601 r2).
- Top-level import SHADOWED by function-local imports of the same name is genuinely unused to F401 → remove the lazy local imports FIRST (or same batch), then add the top import (#606 r3).
- ALWAYS finish with `ruff check` + `ruff format` to confirm new imports survived — your Edit succeeding doesn't mean the format pass kept it.
