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

**RUF001 (strings) addendum (#1428, 2026-07-16):** en dash `–` and MINUS SIGN
`−` in figure-label STRINGS are RUF001-flagged; em dash `—` is NOT confusable
(passes clean). For deliberate typographic chars in rendered figure text the
fix IS a line-level `# noqa: RUF001 -- reason` (repo precedent:
plot_aim5_25pct_*.py) — never swap the char (changes rendered figures). E501
interplay: the noqa comment counts toward the 100-char limit, so on a ~95-char
line wrap the call per-arg (magic trailing comma keeps ruff-format stable) and
put the noqa on the label line. Also verified: `# noqa: C901 -- reason`
trailing text parses fine, and an underscore prefix (`_lab`, `_null_lo`)
silences B007/F841 with the assignment kept.
