---
name: multi-position-replace-hook-trap
description: "#2094/#2162 patching rig: PositionEditHook historically forbade mode='replace' with >1 position/row — RESOLVED on main (tbmp generalization, commit 215d120dee): replace now accepts ANY position set. Still grep the assert when a plan pins an OLD sha or a pre-tbmp branch"
metadata:
  type: feedback
---

**STATUS UPDATE (2026-08-16, verified against origin/main):** the trap below
is RESOLVED on current main — the #2162 tbmp round (commit `215d120dee`,
"turn-boundary multipatch — hooks fix") generalized `PositionEditHook`:
hooks.py's module docstring now reads `mode in {"add", "replace"} at ANY
position SET`, and `arm_batch` asserts only non-empty/duplicate-free
position lists per row (verified by grep on origin/main). The trap still
applies to plans that pin a PRE-tbmp sha or import from a stale branch —
grep the assert at the PINNED revision, not just main. The validated fix
shape below is the historical record of how the relaxation was accepted.

Any plan on the #2094/#2162 activation-patching rig that declares a JOINT
multi-position patch with `mode="replace"` is claiming a code behavior the
rig rejects: `PositionEditHook.arm_batch`
(`src/explore_persona_space/experiments/issue2094/hooks.py`, the
`replace mode edits exactly ONE position per row` assert, ~line 132-135)
HALTs on any len(positions[b]) > 1 in replace mode, and the reused-verbatim
`_arm_hook_all_layers` (`scripts/issue2162_run.py:1105-1122`) hardcodes
`mode="replace"`. The oft-cited "l3j = 3 joint positions" precedent did NOT
run replace: #2094's own `plan_deviations` record
(`scripts/issue2094_run.py:2258-2262`) realized multi-position dose as
`add_full_state_patch` (Δ = V_B − V_A, alpha=1) precisely because of this
restriction.

**Why:** caught in a v7 plan whose §10 "call-shape bind" row quoted a guard
grep of `hooks.py:125-130` — stopping two lines above the killing assert —
and claimed "satisfied by construction". Classic artifact-reuse check-(l)
call-shape-bind failure (#1728: signature accepts, code path rejects).

**How to apply:** on any multi-position patch plan reusing this rig, grep
`hooks.py` for `replace mode edits exactly ONE` and demand the plan
pre-register the realization: (a) extend the hook to multi-position replace
(exact under stacked all-layer edits; source-module fix then reuse), or
(b) adopt add_full_state_patch and re-state what the injection-exactness
gate certifies (add-mode stacked edits are not exactly donor-state at
layers ≥ 2; #2094 realized ≥0.99997 cos empirically). A partial guard grep
is never a call-shape bind — the rule requires the ACTUAL call executed at
smoke shape.

**Validated fix shape (realization (a), accepted at revise round 2):** the
`arm_batch` guard is the ONLY one-position constraint — verified against
source: the payload check `d.shape[0] == len(pos)` (:138), `arm()`'s flat
index assembly, the `index_put_` apply (:204-207), and the realized_edits
telemetry are all position-count-generic, so relaxing :133-135 alone is
sound. Required test triad: (1) defect-repro (len-2 tuple through the
wrapper) flips to pass; (2) len-1 replace bit-exact vs pre-extension
(protects single-position consumers + any d1 mechanism-identity gate);
(3) joint == composition of single-position replaces IN ONE FORWARD
(disjoint out-of-place writes at one layer commute — valid invariant; N
separate forwards would NOT compose, reject that variant). Plus an
on-device injection-exactness gate. Two residues to flag as concerns:
the class/module docstrings still state the one-position contract
(update with the relaxation), and a multi-position "nowhere else" check
must be defined propagation-compatibly — patching earlier boundaries
legitimately changes downstream unpatched positions at layers ≥ 2, so
hidden-state-equality-everywhere-else false-HALTs; use
no-edit-APPLIED-elsewhere (telemetry vs intended positions) +
upstream-of-first-patch equality. Under replace the installed value is
literally the payload cast to hidden dtype, so the gate can assert
near-bit-exact equality at patched coordinates instead of a 0.999 cosine
(which can marginally false-PASS an off-by-one at deep boundaries where
V_A ≈ V_B).
