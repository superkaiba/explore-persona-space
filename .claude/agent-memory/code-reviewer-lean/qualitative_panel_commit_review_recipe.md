---
name: qualitative-panel-commit-review-recipe
description: Review recipe for qualitative text-panel figure commits (figures + companion doc + selection-audit JSON) — triple-consistency, characterization grounding, both-direction substitution-disclosure check, sidecar-commit ancestry (#2478 r1 g2)
metadata:
  type: feedback
---

For a commit shipping qualitative example panels (text-panel PNGs + companion
provenance doc + selection-audit JSON), five checks settle artifact correctness:

1. **Triple-consistency of every score** — canvas ↔ companion doc ↔ audit JSON,
   then re-read 1-2 rows from the SOURCE artifact to full stored decimals. Watch
   for a source JSONL carrying MULTIPLE rows per pair_id (e.g. two arms); the
   quoted value must match the plan-pinned arm's row, and flag (nit) when the
   doc's provenance omits the discriminating token.
2. **Ground every CHARACTERIZATION of a not-re-read artifact** (a confuser
   described as "a transliteration explainer" while the note says its text was
   never re-read) in the promoted body / stored table it came from — grep the
   parent issue's body for the ci/id before flagging it as unsupported.
3. **Substitution disclosure is checked in BOTH directions:** the classic miss
   is an unmarked truncation; the observed miss (#2478 r1 g2 Minor) was the
   REVERSE — a blanket "truncated … ([…])" line beside a doc passage under the
   cap, ending at a sentence boundary, with no marker (script emitted the
   string unconditionally when only the FIGURE side was cut). Count words vs
   the stated cap: under-cap + no marker + natural ending ⇒ shown-in-full,
   disclosure overclaims (conservative direction, Minor not blocker).
4. **Disclosed example swaps verify against the exclusion artifact** — read the
   named label/eligibility file for BOTH the dropped and the substitute id, and
   confirm the substitute is the claimed "remaining member" of the disclosed
   sample enumerated in the plan.
5. **Sidecar/audit `git_commit` + `git_dirty:false` certifies committed-code
   rendering** only after probing that commit is a branch ANCESTOR of the
   reviewed commit and contains the render script (`git merge-base
   --is-ancestor` + `git show <sha>:<script> | wc -l`).

**Why:** #2478 r1 g2 — all 12 rows triple-checked clean in one pass; the only
findings were the direction-3 overclaim and the arm-discriminator nit.
Jq trap from the same round: probing a nested label map with `.["<ci>"]` and no
`.labels` fallback returns four confident nulls — re-check the file's top-level
keys before reading a null as absence.

**How to apply:** any commit whose payload is example panels / dashboards /
transcript figures with a provenance companion — run 1-5 before composing the
verdict; severity ladder: unmarked substitution = blocker, overclaimed
substitution = Minor, missing row-discriminator = nit.
