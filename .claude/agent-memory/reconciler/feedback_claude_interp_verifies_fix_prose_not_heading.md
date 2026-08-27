---
name: claude-interp-verifies-fix-prose-not-heading
description: "Claude interp-critic verified a round-fix's PROSE numbers but never the H3 heading above it — the heading carried the false claim (#2552 r2); check every claim-carrying heading/caption against the same data as its prose"
metadata:
  type: feedback
---

When adjudicating an interp-critique split where Claude APPROVEs after verifying
a round-1 fix, re-check the CLAIM-CARRYING SURFACES around the verified prose:
the `### <result>` H3 heading and the blockquote caption. In #2552 round 2,
Claude's item 8 recomputed the order-stratified rates and confirmed the prose
("one reverses, two sit at parity") but never read the heading directly above
it, which said "three win-matrix cells reverse by order stratum" — a false
claim a skimming reader takes away (only 0.384 crossed parity; 0.507/0.501 are
parity, not reversals). Same round: the heatmap caption grouped Voice/Function/
Meta as "largest in the lower-activity quintiles" while Meta k=100 peaked at Q4
(+0.110, ~2x its Q1-Q3 values) — Claude saw the fact but filed it as an
optional wording note; the false-claim stopping rule resolves that to BLOCKER.

**Why:** v4 H3 headings and captions are claim sentences; fix-verification
anchors on the prose the fix touched, so heading/caption overclaims that
SUMMARIZE the fixed prose slip through both the fixer and the verifier.

**How to apply:** for every disputed result, diff the heading + caption
wording against the recomputed numbers, not just the paragraph text. "Reverse"
means crossing parity; "largest in low-activity quintiles" means argmax in
Q1-Q2 for EVERY named category — check the argmax per cell. Sibling note:
Codex framed a partially-landed label fix as "still shows bare slugs" (r1
covariate-label fix HAD landed; the residual was config slugs in panel titles
+ one tick set) — verify which SCOPE of a prior fix landed before upholding a
"fix failed" severity; residual scope is usually CONCERN, not BLOCKER. See
[[claude-interp-undersamples-new-sample-blocks]],
[[lens10-capsule-cap-not-binding-lens11-same-h3-binding]].
---
name: claude-interp-verifies-fix-prose-not-heading
description: "Claude interp-critic verifies a round-fix LANDED but not that fix-introduced prose is LICENSED — false H3 heading (#2552 r2); 'noise resolved' mechanism-as-proven clause (#2564 interp r5); check new prose against registration + the reliability values Claude itself recomputed"
metadata:
  type: feedback
---

When adjudicating an interp-critique split where Claude APPROVEs after verifying
a round-1 fix, re-check the CLAIM-CARRYING SURFACES around the verified prose:
the `### <result>` H3 heading and the blockquote caption. In #2552 round 2,
Claude's item 8 recomputed the order-stratified rates and confirmed the prose
("one reverses, two sit at parity") but never read the heading directly above
it, which said "three win-matrix cells reverse by order stratum" — a false
claim a skimming reader takes away (only 0.384 crossed parity; 0.507/0.501 are
parity, not reversals). Same round: the heatmap caption grouped Voice/Function/
Meta as "largest in the lower-activity quintiles" while Meta k=100 peaked at Q4
(+0.110, ~2x its Q1-Q3 values) — Claude saw the fact but filed it as an
optional wording note; the false-claim stopping rule resolves that to BLOCKER.

**Why:** v4 H3 headings and captions are claim sentences; fix-verification
anchors on the prose the fix touched, so heading/caption overclaims that
SUMMARIZE the fixed prose slip through both the fixer and the verifier.

**Second instance (#2564 interp r5, fix-verification round 2):** Claude's fix-1
verification quoted the fix-INTRODUCED mechanism sentence approvingly
("Magnitude flips with the noise resolved: sampling noise inflates observed
norms, deflating low-draw slopes, so the parent's 0.87 was a noise-limited
read") while itself recomputing r100_mean = 0.593 from the same JSON —
"resolved" was false against the number Claude had in hand, and the causal
framing exceeded the plan's "Secondary / exploratory (no lattice)" calibration
registration that the SAME round had enforced on the sibling query-form claim.
Codex REVISE upheld. Scoping nuance that survived adjudication: when a
mechanism's DIRECTION is estimator-math-grade AND plan-stated (noise inflates
noisy-mean norms + errors-in-variables attenuation both deflate a
predicted-on-observed slope), scope the REVISE to the RESOLUTION clause and the
causal attribution of the specific flip — the general direction statement may
stand; don't uphold an over-broad "weaken everything" ask wholesale.

**How to apply:** for every disputed result, diff the heading + caption
wording against the recomputed numbers, not just the paragraph text — and for
fix-VERIFICATION rounds, treat fix-introduced prose as NEW claims needing the
registration/consistency check, not as closure evidence. "Reverse"
means crossing parity; "largest in low-activity quintiles" means argmax in
Q1-Q2 for EVERY named category — check the argmax per cell. Sibling note:
Codex framed a partially-landed label fix as "still shows bare slugs" (r1
covariate-label fix HAD landed; the residual was config slugs in panel titles
+ one tick set) — verify which SCOPE of a prior fix landed before upholding a
"fix failed" severity; residual scope is usually CONCERN, not BLOCKER. See
[[claude-interp-undersamples-new-sample-blocks]],
[[lens10-capsule-cap-not-binding-lens11-same-h3-binding]].
