---
title: 'verify_task_body: re.IGNORECASE + .lower() crashes the verifier on Unicode
  long-s (ValueError at :16820)'
kind: infra
tags: []
created_at: '2026-08-14T02:30:01Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2279 code-review round as a pre-existing sibling
  of the check-58 casefold defect fixed in 0403597c73.'
workflow: v1
---
# verify_task_body: `re.IGNORECASE` + `.lower()` crashes the verifier on Unicode long-s

## Goal

Replace `.lower()` with `.casefold()` at the token-normalization sites in
`scripts/verify_task_body.py` where a value captured from an `re.IGNORECASE`
regex is then used as a dict key or an equality/`int()` operand, so that a
clean-result body containing a full-case-folding Unicode character cannot
crash the whole verifier.

## The defect

`re.IGNORECASE` performs FULL Unicode case folding. `str.lower()` does not.
The mismatch means a regex can MATCH a token that the subsequent `.lower()`
normalization fails to map back onto the expected ASCII key.

The canonical trigger is **U+017F LATIN SMALL LETTER LONG S**, which
`IGNORECASE` matches against `s` but which `.lower()` leaves unchanged
(`.casefold()` maps it to `s`). Other full-folding characters exist in the
same class.

### Confirmed reachable site

`scripts/verify_task_body.py:16818-16820`:

```python
for m in _V4_FOOTER_ROUND_PLURAL_RE.finditer(footer):
    word = m.group("num").lower()
    n = _NUMBER_WORDS.get(word) or int(word)
```

`_V4_FOOTER_ROUND_PLURAL_RE` is `IGNORECASE` and its `num` group alternates
over the number words `one..ten` plus `\d{1,2}`. A long-s spelling of `six`
or `seven` matches the regex; `.lower()` leaves it unfolded;
`_NUMBER_WORDS.get(...)` returns `None`; and `int(...)` then raises
**`ValueError: invalid literal for int() with base 10`**.

Reproduced directly against the installed module. The verify driver has no
per-check exception handling, so this does not merely fail one check — it
crashes the ENTIRE verifier run for that body.

### Same-pattern site, currently NOT reachable

`scripts/verify_task_body.py:10635`:

```python
pair = (om.group(0).casefold(), _BEAT_WORD_TO_KIND[om.group(1).lower()])
```

A bare dict SUBSCRIPT (KeyError on miss) fed by the `IGNORECASE`
`_BEAT_ONE_PER_RE`. It is not currently exploitable: the captured
alternation is `bar|point|dot|marker|line|curve` and no alternative contains
an `s`. Worth fixing for consistency regardless — note the same line already
calls `.casefold()` on `group(0)` while using `.lower()` on `group(1)`, which
is exactly the inconsistency that produced the reachable bug elsewhere.

## Suggested scope

1. Fix `:16819` (the confirmed crash) and `:10635` (consistency).
2. Sweep the remaining `.lower()`-on-a-regex-`group()` sites in this file —
   there are ~26 — and convert the ones whose regex is `IGNORECASE` AND whose
   result feeds a dict key, a set membership test, an equality compare, or
   `int()`. The `_THIS_REPO_SLUG` owner/repo comparisons are the bulk of them
   and are lower-risk (an equality compare that fails closed), but they are
   the same class and a long-s repo slug would silently mis-compare.
3. Consider whether the verify driver should catch per-check exceptions so a
   single check can never take down a whole verification run. That is a
   larger design question — a crash IS the fail-fast signal the project
   prefers — so it is raised here as a question, not a prescription.
4. Add pins mirroring `test_positional_crossref_unicode_casefold_tokens`
   (added in #2279) for whichever sites are converted.

## Provenance

Surfaced by the #2279 code-review round as a Minor finding against check 58
(`check_v4_positional_result_crossrefs`), which had the identical defect. The
check-58 instance was fixed in-round (commit `0403597c73`); its own
git-provenance probe confirmed these OTHER sites are pre-existing on trunk
and NOT introduced by the #2279 diff, so they were correctly left out of that
round's scope and filed here instead.

Note for whoever picks this up: the reviewer's report cited lines
`:16847` and `:17072`, but neither of those points at this pattern. The
orchestrator re-derived the real sites independently — the line numbers in
this body (`:16819`, `:10635`) are the verified ones, confirmed by
reproduction against the installed module.
