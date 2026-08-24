---
name: prescribed-wording-grammar-check
description: Reconciler/critic-prescribed replacement text can be ungrammatical in situ — adapt minimally, preserve word arithmetic + qualifiers, flag the deviation
metadata:
  type: feedback
---

When a reconciler or critic verdict prescribes exact replacement wording for a
named body span, paste-check it against the SURROUNDING sentence before
applying: #823 r7's prescribed "the stream-prefix fit grows from ..." would
have produced "... grows ... leaves ..." (two finite verbs, no conjunction)
inside the existing bullet. The fix is a minimal grammatical adaptation that
preserves everything the verdict actually adjudicated — the scope qualifier,
the compressed parenthetical, and the exact word arithmetic (98 → 96 vs the
100 cap) — here the gerund form "growing the stream-prefix fit from ...".

**Why:** the verdict's binding content is the fact-fix and the budget math,
not its sentence fragment; shipping an ungrammatical literal paste creates a
fresh defect for the next round, while a silent larger rewrite exceeds the
"exactly the named sites" mandate.

**How to apply:** adapt only within the named span, verify the adapted form
hits the same word count the verdict computed, and record the deviation
explicitly in the epm:analysis marker (a one-line "Deviation note"). Related:
[[revision-word-caps-and-prereg-token]], [[fold-round-gate-mechanics-1336]].
