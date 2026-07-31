---
name: PYTHONHASHSEED re-exec dance for hash() reproducibility
description: When code uses random.Random(hash("...") + seed) for sampling, PYTHONHASHSEED must be pinned at script entry; use os.execvpe to re-exec under the env var if missing.
type: feedback
---

When orchestrator code uses `random.Random(hash("...") + seed)` for negative-persona / negative-sample selection (the #205-#247 pattern), Python's per-process hash randomization breaks bit-reproducibility across runs. The fix is a re-exec dance at script entry:

```python
import os as _os
import sys as _sys

if _os.environ.get("PYTHONHASHSEED") != "0":
    _new_env = {**_os.environ, "PYTHONHASHSEED": "0"}
    _os.execvpe(_sys.executable, [_sys.executable, *_sys.argv], _new_env)

# Subsequent imports run AFTER the re-exec returns to the child, with
# PYTHONHASHSEED guaranteed to be "0" before any hash() call resolves.
import gc, json, ...

# Re-bind canonical names AFTER the early-return so the rest of the file
# uses os.environ / sys.exit naturally.
os = _os
sys = _sys
```

**Why:** Python's `hash()` for strings is randomized per-process unless `PYTHONHASHSEED` is pinned at interpreter start. Setting it from inside the script is too late — `hash()` is already initialized. Re-exec is the only way.

**How to apply:** Any orchestrator that calls `hash(<string>)` directly OR depends on a downstream library that does (e.g., the #205/#247 negative-persona sampler) needs this dance at the very top of the entrypoint. Use `_os` / `_sys` aliases so the re-exec block doesn't shadow the module-level `os` / `sys` names that the rest of the file uses.

**Caveat:** Pinning your script does NOT retroactively make hash draws byte-match prior runs that were unpinned — those are lost. But it makes future runs reproducible across reruns and re-execs.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [PYTHONHASHSEED re-exec dance](feedback_pythonhashseed_reexec.md) — pin via os.execvpe re-exec at entry; setting from Python is too late.
