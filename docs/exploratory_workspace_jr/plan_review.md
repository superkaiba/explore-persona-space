# Independent provenance and validity re-review

Reviewed read-only at 2026-09-12T03:38:05.279366+00:00:

- `/mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability/docs/exploratory_workspace_jr/analysis_plan.md` — SHA256 `3649964ce36631349f8098177174b65daf8a5d19bc60446669dbbb037dbc34f5`
- `/mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability/configs/analysis/workspace_jr.yaml` — SHA256 `c559d2ce532348695867efa513aaca05e21d22d975afaa647113e30297bd9c94`

Evidence reference: `/tmp/jr-mapping-provenance.json` and the actual artifact audit it records. No new component test outcomes were read. No worktree files were edited.

## Verdict

**PASS for preparation: no remaining blocking provenance/specification findings from this review.** This does not approve an experiment launch or assert that the planned runtime gates have passed. `task_id: null` and `status: preparation_only` correctly preserve the current task-authorization boundary.

## Prior findings resolved

1. Fresh-generation settings and exact generated-token answer mask are now explicit, including no-thinking mode, seeds, EOS/padding exclusion, whitespace preservation, unexpected thinking-token treatment, and the distinction from historical retokenization. Incomplete K now excludes the entire paired context, while preserving raw draws and attrition records.
2. Pilot test is unconditionally disjoint from main test.
3. Sampling now uses full source manifests rather than old capture-survivor intersections; normalization is NFC without whitespace changes; split ownership, calibration exclusion, pilot and main selection order are defined.
4. Coefficient hashes and data/model revisions are bound in YAML. Historical-model parity uses 32 deterministically selected calibration contexts, explicit x/y metrics, tolerances, and a failed-eligibility route that blocks reuse rather than replacing the original premise silently. Tolerances are accurately labeled ungrounded engineering gates requiring a dtype-controlled pilot.
5. The primary is described as the highest observed same-mode GPQA representative, with unique superiority unproven and no architecture-based replacement. The inaccessible former cluster limitation is retained.
6. The 768-context test ceiling is explicitly not a power guarantee; width above 0.10 yields an inconclusive 0.05-scale comparison. The K20 diagnostic retains frozen K5-trained predictors and K5 as primary.

## Important gates that remain intentionally unexecuted

- User authorization to create the required new experiment task.
- Exact-checkpoint historical x/y recapture parity.
- Native pretrained Qwen3.5 J/R calibration and hybrid-backward validation, including full-forward parity.
- Calibration/fit/validation/test content-hash manifests and exclusion assertions.
- Pilot measurement of lens stability, tuning/convergence and runtime/memory needs.
- Verified remote persistence before relying on large coefficient/token artifacts from another machine.

These are explicit execution gates, not hidden blockers in the preparation protocol. No production GPU-hour estimate is asserted.

## Minor implementation note

The prose freezes skipping the final calibration token, but YAML currently records only `skip_first: 4`. When implementing the calibration manifest builder, make final-token exclusion an explicit asserted mask or serialized config field; do not inherit the upstream API default. This does not block preparation because the prose convention is already unambiguous.

The chosen models, post-block layer indices/depths, K=1 historical fit provenance, missing old generated IDs, and 13 validation/test content collisions agree with the audited artifacts. Cross-model differences remain observational and confounded by depth/width/processing stage, as the revised plan states.

## Final source-selection implementation review

Read-only review at 2026-09-12T03:40:20.273027+00:00. **PASS: no blocking findings.** Reviewed `/mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability/scripts/workspace_jr_select_contexts.py` and independently reconstructed `/mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability/docs/exploratory_workspace_jr/selected_contexts.json` directly from the audited source-file bytes, without calling the selection function or reading outcomes.

- Source byte hashes match the original manifest audit. The recorded audit SHA matches the durable `mapping_provenance.json`, and recorded config SHA matches current YAML.
- In every selected source pool, `source_row_index == ladder_local_id` was independently asserted against the actual rows; first 10,000 train rows correspond to the intended inherited training subset. Validation/test use their separate source namespaces.
- Recomputed NFC content hashes, first-owner assignment across train→validation→test, deterministic hash order, calibration allocation, pilot removal and capped main allocation match the saved manifest **exactly**, including every exclusion record.
- Eligible unique contexts before allocation: train10,000; validation389; test929. Duplicate exclusions:82 rows total (11 validation,71 test). This includes within-split repetitions and cross-split collisions, so it is correctly larger than the13 unique validation/test overlaps identified earlier.
- Realized subsets: calibration128; pilot train64/validation16/test32; main train8192/validation373/test768.
- All9,573 selected content hashes are unique across all seven subsets. Calibration and pilot contexts are disjoint from main by normalized content, and no old capture-survival filter is used.
- The exclusive-create output mode refuses replacement of an existing selection manifest.

Reviewed selector SHA256: `3d70c5053ffa10e52fcd7d3205bb10c2dcc876a2f162d042a4ddd495c1440b6e`.
Reviewed selected-manifest SHA256: `70cb0811b4a1609515b354b113c758ae02c5e102aeeef0fa5475ec567bcefb32`.

This verifies outcome-blind data preparation only; it does not execute or approve model generation, calibration, fitting or evaluation.

## Final digest-field clarification

The unpublished selection field `context_key` was renamed `prompt_sha256`, and auxiliary source digest mappings became explicit filename/sha256 records after the secret scanner mistook key-named hashes for API credentials. All values and selections are unchanged. Parent reran the selector from verified source bytes and asserted exact structural equality with the complete saved manifest. Final selection SHA256: `11209d75c89345223e87641d74d273ba37ddcd7f1e608f0c59d39be0607d2b54`.
