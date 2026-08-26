---
name: pin-claim-target-file-resolution
description: "Brief-cited pin claims: resolve the sha's object TYPE and the claim's TRUE target file from the round's own code before attesting — a claim verified against the wrong pinned file yields a false 'does not verify' fact (#2587 r1)"
metadata:
  type: feedback
---

When a whole-round brief attributes a load-bearing claim to a pinned sha
("the pinned blob's only in-body float16 references are its two terminal
casts — blob 8265bcd…, lines 1670/1676"), do NOT verify it against the
obvious-looking file. Two failure modes hit on #2587 r1 (2026-08-25), both
composer-side near-misses:

1. **Object-type misattribution.** The brief called `8265bcd…` a blob; it is
   a COMMIT (`git cat-file -t`). A `cat-file -p` of the commit greps the
   commit object, not the file — trivially zero hits, a false negative.
2. **Wrong target file.** The obvious candidate at that pin
   (`scripts/issue2564_embed.py` — the round is a #2564 replication) has
   float16 hits at OTHER lines (326/339/512/518), which would have handed
   Codex a false "the claim does not verify" ground-truth fact. The TRUE
   wrapped module resolves only from the round's own code: the shim docstring
   named `issue2162_run` and the round's `PIN_2162_REL =
   "scripts/issue2162_run.py"` — and `git show <pin>:scripts/issue2162_run.py
   | grep -n float16` returns exactly 954 (bfloat16 load) + 1670/1676 (the
   two terminal `.to(torch.float16)` casts). Claim VERIFIES.

**Why:** composer-attested facts are ground truth to the twin; an attested
false negative converts a correct implementation claim into a spurious
substantive finding (the inverse of the #489 false-FAIL class — same cost).

**How to apply:** for every pin-attributed claim in a brief: (a) `git -C <wt>
cat-file -t <sha>` first; (b) resolve the claim's target FILE from the
round's own constants/docstrings (grep the round files for the sha), never
from the brief's wording alone; (c) run the verification against
`<pin>:<resolved-path>`; (d) if the numbers still mismatch, hand BOTH the
brief's claim and your probe result to the twin as an adjudication duty
rather than attesting either. Related: [[whole-round-unsplit-compose]]
(item 8 pin-reachability), [[installed-api-evidence-envelope]].
