---
name: multi-position-replace-hook-trap
description: "#2094/#2162 patching rig: PositionEditHook forbids mode='replace' with >1 position/row (hooks.py assert); the l3j 'multi-position precedent' actually ran as add_full_state_patch — check any plan claiming joint multi-position replace"
metadata:
  type: feedback
---

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
