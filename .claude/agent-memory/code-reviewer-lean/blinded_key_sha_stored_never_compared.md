---
name: blinded-key-sha-stored-never-compared
description: Blinded-read instruments — probe live that the frozen key's content shas are COMPARED on the packet/compose path (not just stored), and that the ban-regex actually MATCHES the key filename the freeze accepts (boundary lookbehind vs _-prefixed names)
metadata:
  type: feedback
---

Reviewing a `.claude/rules/blinded-reads.md` implementation (frozen key +
scope-split leakage scan): two probe-confirmed gaps hide behind a faithful-
looking build (#2658 unit 7, `scripts/issue2658_human_read.py`):

1. **Stored-but-never-compared key shas.** The freeze replaces display texts
   with `answer_sha256`/`prompt_sha256` "so adjudication can join back" — but
   `compose_packet` joined items to key entries by `(row, item_id)` ONLY.
   Live probe: freeze a key, mutate one item's `answer_text`, compose — the
   drifted text ships under the frozen tag with zero guards fired, so ratings
   join to hidden metadata (operational label/split/frame) describing
   DIFFERENT text. Sibling of [[pilot-pass-report-fingerprint-unchecked]]
   (recorded-but-never-compared fingerprint). Probe shape: 10-line inline
   python, no API. Demand sha equality at compose time.
2. **Ban-term boundary lookbehind vs the accepted filename.** The freeze
   required the filename to CONTAIN `blinding_key`, but the ban term
   `blinding_key*` compiles with `(?<![a-z0-9_])`, so `my_blinding_key.json`
   is accepted by the freeze and INVISIBLE to both scan scopes. Containment
   is not coverage — the freeze check must run the actual compiled ban regex
   against `path.name`.

**Why:** both defeat the exact guarantee the recipe exists for (verifiable
blinding / audit join), while every leakage/write-once/refusal test stays
green — the tests pin presence of the machinery, not its engagement.

**How to apply:** any diff implementing a blinded/unprimed read: run both
live probes (drifted-items compose; scan of the accepted filename in both
scopes) before crediting elements 1/3 of the recipe.
