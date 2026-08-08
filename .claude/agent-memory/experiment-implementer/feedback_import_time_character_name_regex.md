---
name: import-time-character-name-regex
description: ANSWER_ATTRIB_RE is compiled at MODULE IMPORT from EPM_STORY_CHARACTER_NAME, so the first import in a process fixes it forever — later env changes silently run the wrong regex, and tests hardcoding a name pass alone but fail in file order
metadata:
  type: feedback
---

`issue1345_common.ANSWER_ATTRIB_RE` is built at MODULE IMPORT from
`STORY_CHARACTER_NAME` (itself read from `EPM_STORY_CHARACTER_NAME`). The FIRST
import in a process fixes the pattern for the whole process. Setting the env var
afterwards changes nothing — the gate then silently looks for a different
character's attributions and returns `attribution_zero` on perfectly good rows.

**Why:** #1345 (2026-07-31), two faces of the same seam in one round. (a) Capture
job 16283 died on the round-character guard because a hand-composed launch chain
omitted the `EPM_STORY_CHARACTER_NAME=Assistant` export the launchers carry.
(b) My new gate tests hardcoded `"Assistant"` in their fixtures, passed in
isolation, and failed in FULL-FILE order — an earlier test in the same file
imports the module under `ARIA`, so the session's regex was ARIA's and my
Assistant-shaped fixture matched nothing.

**How to apply:**
- In tests, NEVER hardcode the character name in a fixture that the attribution
  regex must match. Read the live value: `import issue1345_common as c;
  c.STORY_CHARACTER_NAME`, and build prefixes/answers with an f-string. A
  fixture that passes with `-k` but fails in the full file is this bug.
- When composing any launch chain by hand (not via the repo launchers), export
  `EPM_STORY_CHARACTER_NAME` BEFORE the first python import, not just before the
  command that "needs" it. Same for `EPM_I1345_VARIANT`.
- Treat "passes alone, fails in file order" as an import-time-global symptom
  first, before suspecting the code under test.
- A module-level regex built from env is not re-readable; `importlib.reload` is
  the only in-process fix, and reloading shared modules mid-suite is worse than
  parameterizing the fixture.

Related: [[feedback_stale_pycache_masks_signature_change]] (the other
"passes alone, fails in the gate" class — that one is a stale `.pyc`, this one is
import-time env capture; check both).
