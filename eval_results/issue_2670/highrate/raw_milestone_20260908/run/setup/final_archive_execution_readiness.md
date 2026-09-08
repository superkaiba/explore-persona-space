# Final archive execution readiness

Read-only review at 2026-09-08T10:57:39.500220+00:00. The approved two-milestone plan and unchanged reviewed V2 archive wrapper support a **final-only snapshot** containing the complete VM analysis outputs, final reports/reviews, late execution evidence and pinned raw references. Reuploading the raw native/capture/map payloads is not required. This is readiness guidance, not a final archive or experiment PASS. No upload, fit, model call or source change occurred in this review.

## Already verified raw reference

Use dataset `superkaiba1/explore-persona-space-data`, revision `58d3e6d6b0917b6e9777c45d23b814693dc14277`, prefix `context_risk/issue2670_highrate/raw`; snapshot SHA `91615d1272ef62d16ccd33651acc8f295800051950a7ee45e04854a6be7a5215`. Local upload/readback receipts report PASS for 658 remote files and 624 original files. The current five archive source hashes equal the independent V2 review and actual raw readback receipt. This review checks those receipts and their source binding; it does not repeat native/capture semantic audits.

[Verified raw archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/58d3e6d6b0917b6e9777c45d23b814693dc14277/context_risk/issue2670_highrate/raw).

## Final snapshot contents and completion check

1. Preserve **every file actually produced under `RUN/analysis/`**, including `analysis_launch.json`, `input_census.json`, top-level `result.json`, and each regime directory. Expected regime keys are exactly `primary`, `competence_sensitivity`, `screen_augmented_primary`, and `screen_augmented_competence_sensitivity`. Each has `result.json`; fitting paths additionally write `partial_result.json`, `<method>_selection.json`, and `fits/<method>_fold<fold>_rank<rank>_C<C>.json` plus `fits/<method>_final.json`. These cached files contain fitted coefficients/intercepts, tuning evidence and predictions. Held-out row identities, targets, logits and summary metrics are embedded in each regime result; the producer does not promise separate prediction CSV/NPZ files.
2. Before freezing, require actual owned analysis exit/drain evidence, top-level `verification_passed=true`, all four returned regime results, source/input/review equality, and the independent final scientific/result review. A scientifically valid insufficient-support or no-completed-test result is not a missing regime: retain its explicit status and support limitations. Do not require an invented fixed fit-file count, treat a checkpoint as completion, or replace undefined estimates with zero. The V2 archive parser proves bytes/format, not this four-regime semantic completion.
3. Include the exact resolved analysis launch config/environment and owner/log/exit receipts, final metrics/predictions/report and independent reviews, plus new scripts/configs used to prepare/report/archive those outputs. Use the actual analysis review override `RUN/setup/analysis_transport_code_review.json`; the checked-in YAML default is the historical pre-V9 review. Preserve the final result's full provenance and all current 41 analysis source files either as final files or exact hash-matched raw references.
4. Include copies of `raw_upload_receipt.json`, `raw_readback_receipt.json`, the raw `snapshot_manifest.json`, and a small explicit `raw_archive_reference.json` recording the fixed repo/revision/prefix and all three receipt/manifest hashes. Its per-input table should map every `analysis_launch.provenance.input_files_sha256` absolute source path, code/config/review input and externally referenced dependency to an exact raw snapshot key, size and SHA (the raw manifest stores each source path in `files[key].source`). Check that each referenced key also occurs with the same SHA in the raw readback receipt. Archive any newly created or changed dependency as a final file instead of claiming it is in raw. References alone are not an automatic V2 validation: the wrapper parses JSON but does not dereference this table, so preparation/result review must execute this equality check.
5. Include late raw reconciliation/row-count/pod inventory/managed teardown and authoritative absence evidence that postdate the raw snapshot, or bind their exact pushed Git commit/path/blob bytes. Include the final scientific report before the snapshot is frozen. Preserve originals and prior receipts; no deletion is required.

## Immutable stage contract and size preflight

Create a new timestamped immutable snapshot at sibling `.../issue2670_highrate_archive_snapshots/final_<ID>` and a pristine byte-identical working stage at `.../issue2670_highrate_archive_staging/final_<ID>`; reserve fresh sibling `.../issue2670_highrate_archive_readback/final_<ID>`. RUN, stage and readback must be pairwise disjoint, and readback must not exist. Do not recursively include all RUN, old staging/readbacks or raw payloads. Select final files explicitly, retaining path namespaces such as `run/analysis/`, `run/setup/`, `code/` and `references/`.

The new `snapshot_manifest.json` requires `phase: "final"` and a nonempty `files` mapping from safe relative name to exact `size` and `sha256`; retaining `source` for every file makes the input bridge reviewable. Its exact file set is declared originals plus the manifest itself, with no symlinks. Include the current five archive sources and V2 review bytes. Use a fresh manifest; changing the old raw manifest to describe a subset is invalid provenance.

Check actual final file sizes before launch: JSONL is parsed by physical lines and cannot contain raw CR; an individual record may not exceed the supported shard limit. JSONL over 9,500,000 bytes is deterministically sharded and every shard must be below 9,000,000 bytes with exact original reconstruction. `.npz`, ordinary terminal `.eval`, and designated review `.zip` are binary for sizing. Other files over 9,500,000 bytes fail, including large JSON, logs, CSV or Markdown; a JSON array is not JSONL. Compressed text is prohibited by upload policy even if an unrecognized suffix would pass exact-byte parsing. Do not silently rename, compress, split or discard producer files to bypass a failure; obtain a narrow reviewed archive-only recipe if the actual final inventory exposes this case. Started ZIPs need their existing sidecars and review references if included, but already archived raw ZIPs need not be duplicated.

## Exact archive worker command

Run from the reviewed worktree using the existing owned VM process-group supervisor contract (fresh PID/log/exit receipts and an explicit positive whole-process fence). The following is the worker command, not a replacement for that supervisor:

```bash
cd /home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906
UV_NO_SYNC=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
EPM_HF_FILECOUNT_FALLBACK=0 EPM_HF_RETRY_BUDGET_S=120 \
EPM_HF_STAGE_TIMEOUT_S="${FINAL_ARCHIVE_STAGE_TIMEOUT_S:?set measured positive prefix fence}" \
uv run --with inspect-ai==0.3.261 --with openai==3.7.0 python -m scripts.context_risk_highrate_archive \
  --root /home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate \
  --stage "${FINAL_ARCHIVE_STAGE:?set pristine final stage}" \
  --readback "${FINAL_ARCHIVE_READBACK:?set unused disjoint final readback}" \
  --phase final
```

Preserve normal project dotenv/auth loading without printing secrets. Confirm the current five source hashes match the V2 PASS before launch. The actual raw readback elapsed 520.118127 seconds for 719,546,427 original bytes and 658 remote files; use measured final byte/file/shard counts to size explicit prefix and whole-transfer ceilings, retaining per-operation retry exposure and the helper's two explicit 600-second metadata budgets. A scaling projection is an estimate, not a bound; do not copy the raw 92,400-second ceiling blindly or leave a timeout unset.

The fixed destination is `context_risk/issue2670_highrate/final` in the same dataset. The helper writes `RUN/final_upload_receipt.json`, then canonical pinned `stage_hub_prefix` and manifest-first `stage_sharded_text` readback produce `RUN/final_readback_receipt.json`. Require exact remote filename/size/hash equality, every reconstructed original equal to the final snapshot, matching immutable revision, and owned zero exit/drain. Non-JSONL readback lives at `<readback>/<receipt.prefix>/<snapshot key>`; reconstructed JSONL lives at `<readback>/reconstructed/<snapshot key>`.

Once a final upload receipt exists, upload reuse requires exactly the same staged remote file mapping; changed content is refused. Stage sharding mutates the working stage, so retries need another pristine copy of the immutable snapshot and a new unused readback directory. Preserve failed attempts and existing receipts. Do not make an early partial final upload and expect later files to be appended under the same receipt.

Final upload/readback/verification and supervisor exit receipts cannot be in the transfer they attest. Persist their exact bytes in the issue-scoped pushed Git late-evidence set, with explicit source-to-Git mapping and verified commit/blob SHA, before final completion. The raw-only reconciliation adapter is not a generic final-fit checker; do not invoke it with a fictitious final phase or claim its raw PASS validates forecast results.

## Current blockers and scope

At this inspection `RUN/analysis/` had not been created and neither final receipt existed. Actual four-regime completion, final scientific review, exact delta/input-reference mapping, final file-size census and measured transfer fences therefore remain pending. No archive source correction is currently required to support this subset final milestone. This runbook does not authorize deletion, teardown or scientific benefit claims.

## Exact read evidence

```json
{
  "archive_sources_sha256": {
    "scripts/context_risk_corrected_finish.py": "8aeaddec362b9bbf277c707b7532d17948811f1bdf09b9c0a45f598c7eca466e",
    "scripts/context_risk_highrate_archive.py": "b7f712557a5b537021fc5324a3f4f945c0e1a89c06da3404166c81b44401205b",
    "scripts/issue2054_phase_a.py": "69573ff5fc81ad398f35ccf9e656d5fceb7e2696e4b6b35a1db80a166ceb7016",
    "src/explore_persona_space/orchestrate/env.py": "1c53d1c4d642472fe5682f90347ac7d0124130dae128ea5f69d26ecb6147c759",
    "src/explore_persona_space/orchestrate/hub.py": "6590d4c01d396f935da018d16673e3d21f93843126b5e12a42be8e6e0419351e"
  },
  "evidence_sha256": {
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate/setup/archive_review_snapshots_code_review.json": "99d3713729a116745c026eb6ebac18831436a04ec434be7b3509529c86bb9f55",
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate/raw_upload_receipt.json": "4470ed8480598e979d8bbe5d93b5e5186d8ecc18fe76ad84a861312fc3732569",
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate/raw_readback_receipt.json": "a8cb378a69017833f59116922f7b124a33301f4e9338dc372aef8eddb3c442c4",
    "/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906/eval_results/context_risk_highrate_design/plan.md": "7d86436576ee9c9b686c14778fa146d2912c99016e2cf24e6bd75a3cc62f3df6",
    "/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906/eval_results/context_risk_highrate_design/plan_v9_transport_censor_verification.md": "5935dd7792e3a86b9ffc96d57b9826a63e8496a6590a517df6a6340acbb57ad3",
    "/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906/scripts/context_risk_highrate_analyze.py": "2e1613a072a12348171868cd06022766874877fe6ddac64a919df8c8d7790a7c",
    "/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906/scripts/context_risk_followup_analyze.py": "b00ba257966beaa6ae75f37ec35e4579ec8429a118b2db370c16e3f3d6c38b38",
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate/setup/analysis_transport_code_review.json": "06b6ad922023db43e89ed3e550f4ae345383c090b8b9412b867a1940b0347b28",
    "origin/main:.claude/rules/upload-policy.md": "cfb6ce4bf1c7515a2b9bc7c78d597691247e6228e95a35362357204f9528b02d",
    "origin/main:.claude/rules/upload-verifier-section-reference.md": "a3fd68fea716b1834c65e5366af92e4738a92e78f95b2bad32f699329f88687a"
  }
}
```
