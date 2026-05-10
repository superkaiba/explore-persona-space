---
name: archive-script PROJECT_ROOT off-by-one
description: Scripts moved into scripts/archive/ keep parent.parent and silently break. Always check that PROJECT_ROOT actually resolves to the repo root.
type: feedback
---

`scripts/archive/run_leakage_experiment.py:45` had `PROJECT_ROOT = Path(__file__).resolve().parent.parent`. Because the file lives **two** directories deep, parent.parent = `scripts/`, not the repo root. DATA_DIR ended up at `scripts/data/leakage_experiment/` and FileNotFoundError fired on every load.

**Why:** When relocating a script under `archive/` (or any sub-directory), the original `parent.parent` no longer climbs to the repo root. Smoke tests miss this if the previous workflow always used the original path.

**How to apply:**
- Whenever you see a script under `scripts/<subdir>/` that defines PROJECT_ROOT via `__file__`, count the path components.
- For `scripts/archive/X.py`, you need `parent.parent.parent`.
- For `scripts/X.py`, `parent.parent` is correct.
- A 2-line print-at-import sanity check (`print(PROJECT_ROOT, DATA_DIR.exists())`) caught this in <30 seconds; the planner did not include it.
