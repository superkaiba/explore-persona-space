---
name: revision-round-disposition-walk
description: Gate 0.8 on a revision round — enumerate the r1 union from the SOURCE verdict files (Codex sections + each split-group file), diff item-by-item against the implementer's (a)/(b) disposition lists; undispositioned minors surface even when every blocker is fixed (#2321 R2 g4)
metadata:
  type: feedback
---

On a revision round's prior-concerns gate, never walk the union from the
implementer's own summary — its (a)/(b) lists are the thing under test.
Re-enumerate from the source verdict artifacts: the Codex output's
`### Critical` / `### Major` / `### Minor` (+ `## Concerns to persist`)
sections, and each Claude split-group file's numbered findings. Then diff
that list against the dispositions one item at a time.

**Why:** #2321 R2 — all 4 Criticals, all 8 Majors, and every Claude blocker
were fixed with landed tests, yet the walk still surfaced one silently
dropped minor (g1-m2, dirty pack_dir stale-file sweep: absent from both (a)
and (b)) plus a brief-vs-source count discrepancy ("9 Codex Majors" in the
brief vs 8 in the file — resolvable only by walking the source). A
summary-driven walk inherits exactly the omissions it is supposed to catch.

**How to apply:** grep the finding-heading grammar per family
(`^### (Critical|Major|Minor)` for Codex; `^[0-9]+\. ` / `^- \*\*[cm][0-9]`
for split-group files), build the checklist, mark each item
fixed-with-evidence / declined-with-reason / disputed-with-evidence /
UNDISPOSITIONED. Verify "fixed" claims by diff/test presence (or the
[[fails-pre-fix-probe-parent-commit]] differential), not by the report.
Judge declinations against the r1 reviewer's OWN severity language — a
declination quoting the reviewer's "acceptable" is presumptively fine.
Undispositioned NON-blocking items → named CONCERNS bullets (ask for a
one-line disposition), never a new round on their own.
