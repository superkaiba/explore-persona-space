---
name: registered-gate-quantity-substituted
description: A decision gate's computed quantity silently swapped for a semantically-adjacent stricter proxy (donor-swap ceiling delta vs the registered 100−α0 headroom) — diff each gate's code against the plan's literal parenthetical definition (#2254 R1 g3)
metadata:
  type: feedback
---

When a plan pre-registers a decision gate with an explicit quantity — often a
parenthetical like "(100 − α=0 mean graded score)" — verify the reduce code
computes THAT quantity, not a semantically-adjacent one that happens to live
in the same reduce. The substitution is invisible to grid/seed/pin checks and
to tests (the synthetic tree passes both variants), and it re-routes
pre-registered spend: a stricter proxy demotes behaviors the registered gate
would keep.

**Why:** #2254 R1 g3 (`issue2254_preimage.py::_reduce_wave1`): gate 3 was
registered three times in the plan as `100 − α0_mean > null-band p975`
(baseline HEADROOM), but the code gated on the DONOR-SWAP ceiling boot-delta
(`ceil_pt > max(band, 0)`), which is ≤ the registered quantity always —
strictly stricter, no stated deviation in the marker, and hallucination (the
exact behavior the gate was registered to adjudicate) is the one it could
falsely demote. A sibling instance rode into the hero figure (the "ceiling"
overlay star plotted the donor-swap delta).

**How to apply:** for every gate/kill/halt criterion in a reduce diff, (1)
pull the plan's gate sentence and extract the literal quantity definition;
(2) trace the code's `pass` expression back to its inputs and name the
quantity it actually computes; (3) a mismatch is Major even when "stricter"
— direction of error = dropping registered coverage; (4) sweep figures and
report fields for the same substitution (the wrong quantity usually gets
plotted too); (5) check the implementer marker for a stated deviation before
sizing severity; (6) ALSO check the gate's AGGREGATION for silent
denominator narrowing — a `np.nanmean` over per-unit stats lets a NaN unit
(zero-variance rates = the broken-replication signature the gate guards
against) drop out of the registered "mean over N units" and the gate can
PASS on N−1 units with disclosure only in a sidecar field (#2379 R1 g4:
Gate G1 `nanmean(rhos)` over 3 caps languages; Major). Sibling family:
[[gate-threshold-vs-shard-config]] (gate dead from config drift; this one
is gate ALIVE but wrong quantity/denominator).
