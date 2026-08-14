---
name: bit-exact-trips-byte-identical-ban
description: audit_clean_results_body_discipline bans 'bit-exact' under the bit_byte_identical pattern, not just 'byte identical' — write "delta exactly 0" instead
metadata:
  type: feedback
---

`audit_clean_results_body_discipline.py` FAILs on `bit-exact` (pattern id
`bit_byte_identical`), not only the `byte identical` / `byte-identical`
forms named in the analyzer quality bar. A gate note quoting a coordinator
brief's own "bit-exact" wording ships the ban into the body.

**Why:** #2202 round 1 — the footer's reuse-fitness clause said
"(reproduction gate PASS bit-exact)" and the discipline audit FAILed on it;
the brief itself had used "bit-exact".

**How to apply:** in any clean-result body, write reproduction-gate
equality as "delta exactly 0" / "every delta at or below the <X> tolerance"
— never `bit-exact`, `bit identical`, or `byte identical`, even when
quoting a brief or marker that used the phrase.
