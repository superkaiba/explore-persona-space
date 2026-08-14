---
name: ruff-noqa-prose-mention-in-comments
description: Plans documenting a noqa-family waiver token must keep the legacy `# noqa:` literal OUT of #-comment lines — ruff's directive extractor fires on embedded prose mentions (verified empirically, #2089)
metadata:
  type: feedback
---

When reviewing a plan that widens/aliases a `# noqa:`-style waiver grammar
(the #2089 shape: add a ruff-clean `# workflow-lint:` alias), check every
planned COMMENT edit for the literal character sequence `# noqa` — including
prose like "the legacy `# noqa: phase-done-reserved` form" and even "an
invalid `# noqa` directive". Ruff's noqa extractor scans comment text for
`#`-then-`noqa` ANYWHERE in the comment, so both forms draw the exact
"Invalid `# noqa` directive" warning the alias task exists to eliminate —
inside the fix's own source.

**Why:** verified empirically on #2089 plan v1 §3.1(a): the planned
lead-comment text drew 2 warnings (`expected a comma-separated list of
codes` on the token mention; `expected ':' followed by ...` on the bare
prose mention). Baseline workflow_lint.py had 0 because all its legacy
literals sit in DOCSTRINGS/string literals, which ruff never parses.

**How to apply:** docstrings, f-strings, argparse help, and test-fixture
string literals are SAFE carriers for the legacy literal; `#`-comment lines
are NOT — there, write `noqa: <token>` without the leading `#` inside
backticks (or "the legacy noqa form"). Probe in 10 s:
`ruff check` on a scratch file carrying the planned comment verbatim.
Related: [[infra-plan-review-checklist]].
