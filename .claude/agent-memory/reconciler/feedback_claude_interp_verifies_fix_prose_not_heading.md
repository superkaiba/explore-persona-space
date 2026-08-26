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
