---
title: 'eval/capability.py: same narrow PackageNotFoundError catch breaks on the announced
  importlib.metadata KeyError'
kind: infra
tags: []
created_at: '2026-08-18T10:32:45Z'
has_clean_result: false
parent_id: 2360
workflow: v1
---
---
kind: infra
---

# `eval/capability.py` carries the same narrow `PackageNotFoundError` catch that breaks on the announced `importlib.metadata` KeyError

**Provenance:** found by the Codex code-review twin during #2360 round 1
("Bug-class sweep: metadata-corruption exceptions escaping structured
reports"), which named it as a secondary site to consider while widening the
primary. #2360's implementer correctly DECLINED to fix it — it is outside that
task's approved change surface — and flagged it for separate filing.

## The gap

`src/explore_persona_space/eval/capability.py:852-856` (`_safe_version`) catches
only `importlib.metadata.PackageNotFoundError` and relies on the implicit `None`
return for a package whose `*.dist-info` directory exists but whose `METADATA`
file is missing.

Python has announced that behavior away. Running on 3.12.13 emits, from
`importlib/metadata/__init__.py:467`:

```
DeprecationWarning: Implicit None on return values is deprecated
and will raise KeyErrors.
```

The project's `requires-python = ">=3.11"`, so a supported future interpreter
raises `KeyError` where this code expects `None`.

## Why it is lower severity than the #2360 primary — but still worth fixing

This site does **informational version reporting** only; it does not feed a
preflight verdict or any gate, so the failure direction is a confusing traceback
rather than a wrong pass. That is exactly why #2360 fixed its own tier-1 site
(`preflight.py`, where the escape would have destroyed structured verdict/JSON
routing and the named repair guidance) and left this one alone.

The reason to close it anyway: the half-installed-package shape that motivated
#2360 is now known to occur on real pods, and this helper runs during capability
eval. An uncaught `KeyError` there would surface as an unexplained eval crash —
the same misleading-error class #2360 exists to eliminate, just one layer out.

## Fix

Mirror the #2360 resolution: normalize BOTH the `None` return and the
missing-`Version` `KeyError` onto the same "version unavailable" path, and add
an interpreter-version-INDEPENDENT regression that monkeypatches
`importlib.metadata.version` to raise `KeyError("Version")`. #2360's
`preflight.py` tier-1 handler and its accompanying test are the reference
implementation to copy.

Worth a repo-wide sweep in the same round for any other
`except PackageNotFoundError` / `importlib.metadata.version` consumer relying on
the implicit-`None` contract.

## Acceptance

- The helper returns its "unavailable" sentinel for both broken-metadata shapes
  (absent dist-info, and present dist-info with missing `METADATA`), on both
  today's interpreter and a simulated future one.
- A regression pins the `KeyError` form without depending on the running
  Python version.
- Any other in-repo consumer relying on the implicit-`None` contract is either
  fixed in the same round or named as out of scope with a reason.
