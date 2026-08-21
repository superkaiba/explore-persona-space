---
name: judgment-addressed-differently-claims
description: "'Already existed, so I shipped the adjacent gap' claims — verify the existence half at the NAMED commit + ancestry to the round base, then judge the substitute against the PLAN's registered grain literal (the brief may be the wrong party)"
metadata:
  type: feedback
---

When an implementer reports a brief requirement as "addressed DIFFERENTLY — X
already existed at commit `<sha>`, so I shipped adjacent gap Y instead",
verify BOTH halves independently before crediting the row:

1. **The existence half:** `git show <named-sha>` must actually introduce the
   claimed symbol/flag, AND `git merge-base --is-ancestor <sha> <round-base>`
   must hold — existence on some branch after the base does not discharge the
   claim. (#2329 r20: both claims verified — `per_cell_value`/`max_value_spread`
   at `22d70f009c`, `--raised-cap`/`--base-cap` argparse flags at `5269a9df02`.)
2. **The substitute half:** judge Y against the APPROVED PLAN's own registered
   literal (grain, threshold, key construction), not against the brief's
   wording — the brief can be the erroneous party. #2329 r20: the orchestrator's
   brief misread argparse DEFAULTS (`BASE_CAP = 2048` at module scope) as
   hardcoded constants and misclaimed `cv_counts` was unexposed; the
   implementer's git-evidenced correction was right, and the substituted
   per-(cell x slot x arm) unit grain matched plan §7 G5's "per
   (direction x slot x arm) cell" verbatim (ladder blocks constructed
   `Block(direction, slot, arm, ...)`, unit key = `cell|slot|arm`).

**Why:** the "already existed" shape is unfalsifiable-by-default (#2419
Check A) — but it resolves EITHER way: sometimes fabricated coverage,
sometimes the brief's misreading. Only the two-sided grep settles it, and a
TRUE claim with a plan-serving substitute is a PASS row, not drift.

**How to apply:** on any "addressed differently"/"already existed" row, run
the named-commit grep + ancestry probe yourself, quote the introducing hunk,
then quote the plan literal the substitute claims to serve next to the code's
key/grain construction. Related: [[judgment-registered-trigger-enforcement-inplan]],
[[judgment-unimplementable-literal-substitute]].
