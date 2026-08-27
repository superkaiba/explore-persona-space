---
name: sizing-pilot-entry-class-vs-pinned-blindspot
description: A sizing pilot that resolves a different checkpoint CLASS than the plan's smoke blind-spot enumeration pinned, or substitutes a synthetic in-memory basis for the production streamed entrypoint, re-opens the exact blind spot the plan closed (#2569 r1 shard D)
metadata:
  type: feedback
---

Rule: when a plan's smoke blind-spot enumeration carries a parenthetical fix
("the pilot is therefore pinned to one FULL-FT checkpoint"), trace the realized
pilot's ENTRY RESOLUTION (`entry = lora[0]`-style code) and its per-call basis
to the production code path. Two distinct defects hide here: (a) the pilot
entry resolves a checkpoint CLASS whose download/auth path differs from the one
the pin existed to certify (private-overflow full-FT vs model-repo LoRA); (b)
the expensive-unit basis is a synthetic in-memory tensor (`torch.randn(shape)`)
instead of the production streamed entrypoint, so IO (two 15 GB checkpoint
reads per unit) is excluded from the extrapolation.

**Why:** #2569 leg 5: plan line 180 pinned the P-C pilot to a full-FT
checkpoint precisely so the smoke certified the private-overflow download path;
`cmd_pilot` shipped `lora[0]` + a synthetic (3584, 18944) SVD. The plan §7 gate
row said "measured 1-cell pilot through the production entrypoint at production
shape" — the synthetic basis is measured but not the production path.

**How to apply:** on any `--phase pilot`/sizing-gate diff, read the plan's
smoke blind-spot enumeration AND the §7 gate row, then diff three things:
entry-selection expression vs the pinned class; the timed call vs the
production per-unit function (incl. IO); and whether the extrapolation
arithmetic prices every unit class (a "tiny SVDs" §9 premise is falsified when
the implementation reconstructs FULL dense low-rank ΔW = B A and SVDs the dense
shape — LAPACK cost is shape-driven, rank-blind; the exact r×r-core factored
form after two QRs is orders cheaper). Related: [[paired-script-default-path-contract]].
