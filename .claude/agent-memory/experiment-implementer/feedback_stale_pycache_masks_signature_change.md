---
name: stale-pycache-masks-signature-change
description: Rapid Edit + ruff-format-hook rewrites inside one second can leave a __pycache__ entry Python's mtime+size invalidation misses, so a smoke silently exercises OLD code and reports a phantom signature/shape error
metadata:
  type: feedback
---

A smoke iterating fast on a `scripts/*.py` module can silently import STALE
bytecode: Python invalidates a `.pyc` by comparing the SOURCE mtime + size it
recorded in the pyc header, and mtime has 1-second granularity — so an
Edit → ruff-format-hook rewrite → `uv run` cycle that lands inside one second can
produce a pyc whose recorded (mtime, size) still matches the NEWER file.

**Why:** observed 2026-07-31 on #1345 item 2. `assemble_row` was changed from
returning `dict | None` to `(row, reason)`; the smoke kept failing with
`ValueError: too many values to unpack (expected 2)` while `sed` on the source
showed the correct `return {...}, "ok"`. A direct probe proved the IMPORTED
function returned a bare dict. `rm` of
`scripts/__pycache__/issue1345_onpolicy_answers_gen.cpython-311.pyc` fixed it
instantly. Roughly one wasted diagnostic cycle chasing a bug that did not exist
in the source — and the failure mode is worse than a wasted cycle: had the change
been semantic rather than signature-shaped (a new drop-reason branch, a
tightened guard) the smoke would have reported a false PASS on code that never
ran.

**How to apply:** in any smoke that iterates on a module you are actively
editing, clear the cache and disable writing before importing it:

```python
sys.dont_write_bytecode = True
for _pyc in (REPO / "scripts" / "__pycache__").glob("issue1345_*.pyc"):
    _pyc.unlink(missing_ok=True)
```

and invoke with `PYTHONDONTWRITEBYTECODE=1`. Diagnostic tell: the imported
object's behavior contradicts the source you just read — verify with a direct
one-liner probe (`print(type(fn(...)))`) rather than assuming your edit was
wrong, then compare `stat -c '%Y %s'` on the source against the pyc before
concluding anything about the code. Distinct from the MooseFS stale-served-bytes
class ([[moosefs-stale-read]] — pod-side, `git hash-object` disagrees with
`rev-parse HEAD:<path>`): here the on-disk source is correct and only the
compiled cache is stale, so every git-level and file-level probe reads clean.
