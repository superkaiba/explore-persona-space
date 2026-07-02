---
name: Codex catches positive-negative question-distribution mismatch in contrastive negatives
description: Codex grounds a same-question contrastive-negatives violation (positives on question-set A, reused negatives on question-set B) that Claude approves past; uphold REVISE — verify the realized pool's negative-row questions yourself.
type: feedback
---

When a behavior-implant plan REUSES contrastive negatives verbatim from a
sibling pool while the POSITIVES are freshly elicited on a DIFFERENT
question set, check whether the two row types share questions. The
`contrastive-negatives.md` § "The recipe" rule mandates same-question
contrast on THREE lines (29: "Interleave ... over the **same questions**";
42: NEGATIVE row "SAME question as positives"; 67: negatives "on the SAME
questions"). It is a HARD requirement, not a default.

**Why:** #642 round 5 (plan v9, 2026-06-18) swapped the implanted behavior
sycophancy→refusal. Positives were elicited on #606's benign-request
refusal questions; negatives were reused byte-identically from #612's
villain pool — whose questions are SYCOPHANCY FALSE-CLAIM prompts ("Bulls
are enraged by the color red, isn't that right?"). Two disjoint question
distributions. The plan's §4.7 even contained a self-contradiction: a HARD
"byte-identical-to-#612-negatives" invariant (line 221) one line above a
"interleaves same-question negatives" claim (line 222) — both cannot hold.
The realized pool is effectively positive-only on the refusal-question
distribution, so the headline bystander-leakage gap is confounded by
question-distribution mismatch rather than measuring the behavior-general
adapter-vs-dense property. NOT analyzer-recoverable (the confound is baked
into the training data). The plan's HARD asserts (byte-identity,
disjointness) all PASS while the same-question recipe is silently broken —
they are the wrong asserts for the round.

Codex grounded it precisely (§4.2/§4.7 quotes + "#612 pool is sycophancy
based"). Claude APPROVEd by engaging only peripheral concerns (eval-probe
wiring, LR transfer, single-DV waiver, judge calibration) and never
touched the same-question target. Same family as
`feedback_codex_approves_by_not_engaging_anchored_reproduction_target`,
mirrored: Claude approved by not engaging the anchored target.

**How to apply:** when one row type is "REUSED verbatim from sibling pool
#M" and the other is "elicited fresh on question-set #K", do the empirical
check — `hf_hub_download` the realized `train_pool.jsonl`, read the NEGATIVE
rows' user-question text, and confirm it matches the positive questions. A
sha-pin / byte-identity assert proves the negatives match the SIBLING, not
that they match THIS round's positives. If the question sets differ →
REVISE: either rebuild negatives same-question with the new positives
(preserving byte-identity ACROSS ARMS, not across issues), or re-scope away
from the same-question contrastive claim with a written deviation rationale
+ clean-result caveat. A passing byte-identical/disjointness assert set is
not evidence the same-question recipe holds.
