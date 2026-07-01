---
name: ruff RUF002 flags unicode × in docstrings
description: ruff RUF002 rejects the multiplication sign × (and other ambiguous unicode) in Python docstrings/comments; use ASCII x, not a noqa
type: reference
---

ruff's RUF002 fires on an ambiguous unicode character inside a Python
**docstring** (RUF001 = identifiers, RUF003 = comments). The one that bit
`workflow_lint.py` (#806) was the MULTIPLICATION SIGN `×` in `8×H100` — ruff
says `Did you mean 'x' (LATIN SMALL LETTER X)?`. Note `§` and `→` in docstrings
did NOT trip it — RUF002 targets the specific ambiguous-vs-ASCII confusable
set (×, ×→x; en/em-dashes; smart quotes), not all non-ASCII.

**Fix:** replace the char with its ASCII equivalent (`8×H100` → `8xH100`),
NOT a `# noqa: RUF002` — the codebase has zero RUF002-noqa precedent and the
ASCII form reads identically. Plans that paste GPU-shape prose (`8×H100`,
`2×A100`) verbatim into a docstring will hit this; swap to `x` at write time.
