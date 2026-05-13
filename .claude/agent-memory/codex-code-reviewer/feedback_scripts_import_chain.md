---
name: scripts/*.py files that use _bootstrap cannot be imported as namespace packages
description: from scripts.run_leakage_v3_onpolicy import X fails because _bootstrap is not importable when run as a namespace package module
type: feedback
---

Scripts in `scripts/` that contain `from _bootstrap import ...` at module level CANNOT be imported using `from scripts.<name> import X` from the project root. Python 3 namespace packages allow `scripts` as a package without `__init__.py`, but `_bootstrap` resolves as a top-level module name, not `scripts._bootstrap`. The import fails with `ModuleNotFoundError: No module named '_bootstrap'`.

**Why:** This is a recurring pattern in this codebase. Many scripts use `_bootstrap` for env setup, which works fine when run directly (`python scripts/foo.py`) because Python adds the script's directory to sys.path[0]. But namespace package imports don't add the module's directory.

**How to apply:** When reviewing scripts that import from `scripts.run_*`, check whether the imported module uses `from _bootstrap import ...`. If so, flag as Critical — the import will fail at runtime. The fix is `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))` before the import, then use the bare module name.
