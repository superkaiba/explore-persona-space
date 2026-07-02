---
name: script-mode sys.path — deferred scripts.* imports crash pod-side
description: In script mode sys.path[0] is the script's dir, not cwd/repo root — a deferred `from scripts.X import ...` in a src-layout driver crashes on pods; guard with _ensure_repo_root_on_syspath() and import-check in script mode from a non-repo cwd
type: feedback
---

In script mode (`python /abs/path/script.py`), `sys.path[0]` is the script's own
directory — NOT cwd and NOT the repo root — so a top-level non-package `scripts/`
dir is unreachable from a driver under `src/`. A `-c`-mode import check (cwd on
sys.path) passes and hides the bug; the crash surfaces only pod-side at the
deferred import (task #823: `run_823.py:940` killed a full GCE launch at Phase-3
entry after ~30 min of paid work, 2026-07-02).

**Rule:** before any deferred `scripts.*` import in an experiment driver, call a
`_ensure_repo_root_on_syspath()` helper — derive repo root from
`pathlib.Path(__file__).resolve().parents[N]` (N matched to the file's actual
tree depth), assert a sentinel file exists (e.g. `scripts/<module>.py`), insert
the root at `sys.path[0]` if absent. **And verify imports in SCRIPT MODE from a
non-repo cwd** (e.g. `cd /tmp && uv run --project <wt> python <abs>/driver.py ...`)
— never `-c` mode alone.
