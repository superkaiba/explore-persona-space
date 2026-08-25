---
name: tokenize-mask-string-aware-lint-review
description: Reviewing a plan that adds a tokenize string/comment span mask to a line-based lint — two verified under-flag traps (first-match search + skip-line; py<3.12 whole-f-string STRING token) and the verbatim-simulation check that catches both
metadata:
  type: feedback
---

When a plan makes a line-based lint check "string-aware" by masking
`tokenize.STRING`/`COMMENT` column spans (#2351 shape), two under-flag traps
recur — both verified by EXECUTING the plan's masking code verbatim against
its own regression fixtures (a ~30-line sim script; decisive, cheap):

1. **First-match `re.search(line)` + `continue`-on-masked skips the whole
   LINE**, suppressing a REAL call after a masked match on the same line.
   The fix is `finditer`: drop masked matches, flag on the first unmasked
   one (and pass ITS `m.start()` to any wrap/paren heuristic). #2351 v1's
   own test-3 fixture returned `[]` where its A4 demanded exactly 1 error —
   an internal A-criteria contradiction, catchable only by running the code.
2. **On py<3.12 an f-string is ONE `STRING` token**, so masking STRING
   wholesale suppresses a real call inside a replacement field
   (`f"{hf_hub_download(...)}"`), a NEW under-flag class on the pinned
   3.11 interpreter even when the plan carries a py3.12 `FSTRING_MIDDLE`
   forward-compat arm (which is correct there — only literal parts mask).
   Cheap scope-compatible fix: exempt f-prefixed STRING tokens from the
   mask (docstrings/TEMPLATE constants — the FP class being fixed — are
   never f-strings), preserving today's over-flag for f-strings.

**Why:** under-flag is strictly worse than the over-flag being fixed for a
429-SPOF detector (the plan's own kill class), and A5-style live-baseline
criteria cannot catch latent holes with zero live instances — grep the tree
to confirm zero, then still demand the fix or a disclosed+pinned residual.

**How to apply:** any REVISE of a masking/suppression addition to a verifier:
simulate verbatim on the repo's ACTUAL interpreter (`uv run python` in the
REPO — a /tmp uv run can silently resolve a different toolchain), replay the
plan's own fixtures, and probe f-string/multi-line/same-line boundary cases.
Fail-safe `{}` on TokenError→ fall back to the unmasked line scan is the
RIGHT direction (degrades to over-flag, never under-flag) — do not flag it
under fail-fast. Related: [[verifier-check-addition-plans]],
[[infra-plan-review-checklist]].
