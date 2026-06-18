---
name: argparse-description-doc-percent
description: argparse description=__doc__ explodes on literal "%" in module docstrings (e.g. "10% data subset")
metadata:
  type: feedback
---

`argparse.ArgumentParser(description=__doc__)` runs the docstring through
`%` formatting (it expects `%(prog)s`-style placeholders). Any literal
`%` in the docstring — e.g. `"subsets data to 10% (~600 rows)"` — gets
interpreted as a format spec; `10%` followed by `(...)` reads as
`%(...)d` and raises `TypeError: %d format: a real number is required, not dict`
at first `--help` invocation.

**Why:** ArgumentParser walks every argument's `help=` AND the parser
`description=` through `% params` interpolation so `%(default)s` /
`%(prog)s` works.

**How to apply:** for any dispatcher / eval / data-gen script you write
that passes `description=__doc__`, EITHER:

1. Pass a clean short description string (NOT the docstring) AND
   `formatter_class=argparse.RawDescriptionHelpFormatter` so any literal
   `%` in the rest of the help surface stays literal. The docstring is
   still readable via `cat script.py` / IDE hover, just not via `--help`.

2. OR if you keep `description=__doc__`, escape every literal `%` to
   `%%` everywhere in the docstring AND every `help=` string. Brittle —
   the next author who adds a `5%` somewhere reintroduces the crash.

(1) is more robust. Default to it.

Caught me on task #475 round 1 — `run_issue475_cot_install.py` docstring
had `"- subsets data to 10% (~600 rows)"`; `--help` crashed with the
TypeError. Fixed by switching to `RawDescriptionHelpFormatter` + clean
short description for all 3 scripts (dispatcher, eval, data-gen).
