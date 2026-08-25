---
name: floor-fix-defeats-sibling-relaunch-fix
description: A revision fixing BOTH a headroom floor gate and a relaunch-recovery mechanism in the same init path — check the gate's demand against the RESUME-state residency on the quota lane; a fixed full floor can refuse the relaunch the recovery fix enables (#2544 r2 g4)
metadata:
  type: feedback
---

When one revision commit both (a) replaces a disk floor gate with a fixed
plan-prescribed floor asserted at phase `--init`, and (b) adds relaunch
recovery (stale-claim reclaim / done-unit revalidation) wired in the SAME
init path, simulate the gate at the MID-PHASE CRASH state, not just fresh
entry: demand = fixed floor, available = quota − run-resident bytes
(weight caches on the same mount count). If steady mid-phase residency
pushes available below the floor, every crash-relaunch is refused BEFORE
the recovery code runs — fix (a) partially defeats fix (b) on exactly the
path (b) exists for.

**Why:** #2544 r2 g4 — the binding blocker prescribed "pass preambles
≥ 100 GB (fallocate canary at the floor)"; implemented literally. On the
pinned RunPod MooseFS lane (~130 GB quota), K=4 resident weights (56 GB) +
in-flight cells (≤28 GB) is the NORMAL mid-pass state ⇒ writable ~36–46 GB
⇒ the 100 GB canary EDQUOT-refuses the relaunch, though remaining net
demand ≈ high-water − used ≈ ≤10 GB. The plan's own row carried the
qualifier the reconciler's shortened prescription dropped: "resume-aware".
Blockers verified fixed-as-specified; this rode as a NEW Major + persisted
CONCERN, not a re-block.

**How to apply:** (1) on any floors-fix round, recompute gate demand at
used ∈ {0, steady-resident, high-water} vs the quota; (2) grep the plan
row for qualifiers ("resume-aware", "net of residency") the binding fix
list may have compressed away — the reconciler's recipe is not the whole
registered contract ([[prescribed-fix-recipe-vs-stronger-mechanism]]);
(3) full-floor fallocate canaries also violate helpers' small-canary
contracts — on FUSE the glibc emulation WRITES the floor in zeros (wall +
transient quota spike). Related: [[ported-headroom-gate-footprint-model]],
[[disk-row-reconcile-check-coded-assert]],
[[dial-added-fingerprint-arms-refuse-on-relaunch]].
