---
name: direct-venv-pytest-needs-ruff-on-path
description: Invoking main .venv/bin/python -m pytest directly (the worktree recipe) fails test_step9c_baseline::test_ruff_helpers_real_body with ToolMissingError — prepend .venv/bin to PATH
metadata:
  type: reference
---

`scripts/step9c_baseline.py::_ruff_bin()` resolves `ruff` via
`shutil.which("ruff")` from the INVOKING PATH by design (its docstring:
"run under `uv run` or install ruff"). The established worktree test recipe
([[worktree-venv-disk-full]]: use the main checkout's `.venv/bin/python -m
pytest` on the worktree copies) does NOT put `.venv/bin` on PATH, so
`tests/test_step9c_baseline.py::test_ruff_helpers_real_body` fails
`ToolMissingError` — identically on pristine main (environmental, never
diff-caused; verified both-ways 2026-08-29, task #2387).

**How to apply:** when a pin-hit union includes `tests/test_step9c_baseline.py`
and you run it via the direct-venv-python form, prefix
`PATH="<main>/.venv/bin:$PATH"` (the whole file then passes — 244/244 on
#2387). Attribute this exact failure shape as environmental without a main-tree
re-probe; the Step 9c gate runs under `uv run` where ruff resolves.
