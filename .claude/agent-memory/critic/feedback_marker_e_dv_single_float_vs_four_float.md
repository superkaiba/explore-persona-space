---
name: Marker E-DV single-float storage vs four-float contract
description: A teacher-forced marker log-P E-DV that stores ONE float per condition violates the four-float contract; whether it's a Must-Fix turns on EMPIRICAL saturation of the actual cells, not the contract alone
type: feedback
---

When a marker arm stores a teacher-forced `log P(marker)` E-DV as ONE float per
condition (e.g. `marker_logp_p_up/p_down/unpatched_ft/unpatched_base`) instead
of the four-float `(log P, z_marker, z_eos, logZ)` per slot per model side that
`.claude/rules/marker-leakage-measurement.md` + CLAUDE.md mandate (#530), the
verdict is NOT an automatic REVISE.

**Why:** the four-float contract exists to read the saturation signature — when
`log P` plateaus at its 0 cap while the logit keeps moving, `log P` understates
the effect, and the logit is UNRECOVERABLE from stored `log P` post-hoc (#530).
That diagnostic only becomes load-bearing IF cells actually saturate.

**The calibration check (do this before rating):** pull the salvaged/parent
cell's actual `unpatched_ft` log P levels and install gains. If `unpatched_ft`
log P sits far from 0 (the cap) — e.g. the four #697 salvaged marker cells maxed
at ~−16 nat with install gains 1.4–7 nat, ZERO records within 2 nat of the cap —
the cell is unsaturated, `log P` is faithful, and the single-float read gives the
correct verdict. Then the violation is a **Concern** (loses the EOS-margin
secondary read; a future cell that DOES saturate is silently mis-read with no
diagnostic), not a Must-Fix. It IS a Must-Fix only when the actual cells
saturate (denominator compression inflates the ratio in the strongest-install
cells with no way to detect it). The logits are RIGHT THERE in scope
(`out.logits[0,-1,:]`) at near-zero cost, so the fix is cheap when warranted.

**Cross-check with the cross-condition ban:** a teacher-forced marker log P read
is BANNED as a cross-condition leakage leaderboard (#432→#456) but is FINE as a
within-cell direction comparison on the SAME adapter (patch direction P↑/P↓ vs
unpatched), AND when it's on the model's OWN marker-stripped response (the
#432-correct on-policy-R recipe). #697's marker f_CV^E is within-cell ratio-of-means
on one adapter → the ban does not fire.

Source: #697 plan v3 round-1 critique, 2026-06-28.
