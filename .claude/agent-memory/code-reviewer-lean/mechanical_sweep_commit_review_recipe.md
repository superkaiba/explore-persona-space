---
name: mechanical-sweep-commit-review-recipe
description: 5-probe recipe certifying a many-file one-token mechanical sweep commit (exception widening, rename, flag flip) with no per-file deep reads
metadata:
  type: feedback
---

For a many-file "same one-token edit everywhere" sweep commit (#2168 g2: 51
files, +UnicodeDecodeError into JSON-guard tuples/suppress args), five cheap
mechanical probes certify the whole commit; per-file deep reads are only for
the handful of structurally distinct sites (wraps, suppress bodies):

1. **Insertion/deletion arithmetic reconciliation** — deletions should equal
   the site count (1 replaced line per hunk); insertions − deletions should
   equal exactly the disclosed multi-line wraps' extra lines. Any mismatch =
   a rider edit hiding somewhere. (#2168: 76/69 = 68×(1→1) + 1×(1→8) ✓.)
2. **Added-line rider scan** — awk over `git show`'s `+` lines: every added
   line must contain the sweep token (or be a member line of a disclosed
   wrap). Anything else is an undisclosed edit to a frozen/Repro-cited file.
3. **Added-line length check** — awk `length>100` on added lines (ruff on
   per-file-ignores scripts won't catch E501 reliably).
4. **Handler-body attribute grep at the commit tip** — `git grep -nE
   '\.(errno|strerror|filename)\b' <sha> -- <touched files>` settles the
   "no handler discriminates on the caught type" claim in one call.
5. **Residual-predicate grep at the commit tip** — re-express the sweep
   predicate as greps (canonical + reversed + alias + suppress forms, minus
   the fixed token) over the whole scan scope at `<sha>`; zero hits
   corroborates the commit message's "scan → 0" when the durable lint check
   lands only in a LATER commit of the same round.

For `contextlib.suppress` widenings, the semantic read is positional: a
with-block that is the LAST statement of its loop iteration, or a
single default-initialized assignment, degrades to the existing
skip/default path when suppressed — that is the whole behavior-preservation
argument, one line per site.

**Why:** #2168 R1 g2 — all five probes ran in ~4 tool calls and found the
commit clean; the alternative (reading 69 handler bodies) is the
autocompact-thrash shape.

**How to apply:** any split-review sub-scope or round commit described as a
mechanical sweep; run probes 1–2 FIRST — a reconciliation mismatch redirects
the whole review to finding the rider.
