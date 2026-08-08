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

**RUF003 (comments) + sign-class addendum (#1987, 2026-08-02):** a literal
MINUS SIGN `−` in a COMMENT of a `LIVE_WORKFLOW_HELPERS` script passes bare
`ruff check` (pyproject per-file-ignores) but FAILS the Step 9c gate's
`tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`
(RUF003) — spell out `U+2212` in comments instead of writing the char; a
pattern-line `# noqa: RUF001` does NOT cover adjacent comment lines. Related
regex trap in the same file family: a sign class written `[+-−]` is a RANGE
(`+`=U+2B .. `−`=U+2212 — matches all digits/letters); write dash-FIRST
`[-+−]` so `-` stays literal (the audit script's interval_inline convention;
a #1987 plan diff-sketch carried the range form and it verified as
`re.compile(r'[+-−]').match('5') → True`).

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
