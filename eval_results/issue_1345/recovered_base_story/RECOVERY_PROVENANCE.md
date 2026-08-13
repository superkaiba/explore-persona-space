# Recovery provenance — #1345 base-model paired-story leg

Recovered 2026-08-13 (task #1707 recovery approval) from the GCP crash-persist
bundle on the HF data repo.

- **Source bundle:** `superkaiba1/explore-persona-space-data` (dataset repo),
  path `issue1345_partial/att-20260723-172301/eval_results_issue_1345/conversation_paired_stories_assistant_base/`
- **Bundle attempt id:** `att-20260723-172301` (workload ended 2026-07-23T21:02:33Z, exit code 86)
- **Download date:** 2026-08-13
- **File count:** 39 JSON files (424,486 bytes), copied verbatim under
  `recovered_base_story/conversation_paired_stories_assistant_base/` with the
  bundle's internal structure preserved.

## What the crash was

The base-model conversation-paired-story workload COMPLETED all fits and
committed them pod-locally (workload log shows the git commit creating all 39
eval JSONs plus 14 figures under
`figures/issue_1345/conversation_paired_stories_assistant_base/`), then the
final `git push` to `main` was rejected non-fast-forward twice (a concurrent
session had pushed first), the run exited rc=86, and the GCP exit trap
uploaded the crash bundle. The JSONs are finished fit RESULTS, not
intermediates. Producing commit on the pod clone: `f56f7d38ea6d41a6226ca2b1ef778d0ea50a3ee5`
(never pushed; embedded in each JSON's `metadata.git_commit`).

## Reconciliation vs git (at recovery time)

- 39 bundle files absent from git (this directory) — recovered here.
- 216 other eval files in the same bundle: ALL byte-identical to the
  git-tracked copies (git blob SHA comparison). Zero content conflicts;
  nothing overwritten.
- The 14 figure PNGs the pod committed are NOT in the crash bundle (the
  exit trap persists eval/data dirs only) — lost, regenerable from these
  JSONs via `scripts/issue1345_plots.py`.
- The run's HF data upload succeeded before the crash:
  `issue1345_framing/conversation_paired_stories_assistant_base/`
  (`analysis_tensors`, `inputs`, `raw_completions`) is live on the data repo.

## Measurement caveats carried from the parent body

These are ambient-basis GCV-selected ridge fits, the same instrument as the
instruct paired round's originally committed cells. The #1887 lambda audit
replayed only cells committed in git at audit time — these 39 were stranded
in the bundle and were NOT replayed; the parent body's estimator note
(story-input legs at n_train < d select near-interpolating penalties, so
deeply negative ambient story R² is estimator-limited) applies to the base
story cells here (n=2,160 vs d=3,584) exactly as it did to the instruct ones.

## Layer-19 headline reads (bootstrap-rig, from the recovered JSONs)

- `cells_R_base_r4_context.json` (paired story, answers embedded verbatim,
  n=2,160): R² −0.245 (CI −0.270 to −0.223)
- `cells_R_base_r4_op_companion_context.json` (on-policy companion, n=135):
  R² +0.047 (CI +0.009 to +0.076)
- `cells_R_base_r1_context.json` (chat comparator, full n=4,724): R² +0.542
- `matched_row/cells_R_base_r1_matched_context.json` (chat on story-kept
  rows, n=2,160): R² −0.308
- `reparam_recovery_r1_r4_base_context.json`: story-operator-into-chat
  recovers +0.510; chat-into-story −0.131 (the one-way asymmetry, base model)
- `verdict_lattice.json` → `story_paired_verdict`: "framing-effect (collapse
  persists on the shared corpus)" (within-L19 R² −0.247, CI −0.270 to −0.223)

Note: the promoted body's statement that base paired-story cells are
"N/A — not tested" reflects the pre-recovery state of git; these recovered
artifacts show the base capture/fit leg DID run (on the instruct-written
paired stories — base narrative *generation* was still never attempted).

## Standardized refit (`refit_standardized/`, 2026-08-13)

The recovered ambient unguarded-GCV reads above sit in the #1887
estimator-degenerate regime (n_train ≈ 1,728 < d = 3,584). The follow-up
refit re-read the same stores under the newest standardized protocol — the
`scripts/issue1345_story_char_ladder_fill.py` chain (train-fold PCA reduced
basis, k = min(1024, ⌊n_train/2⌋, d); guarded cap-0.9 GCV ambient as the
secondary column) via `scripts/issue1345_refit_base_story_standardized.py`
(fit chain imported verbatim; recorded deltas in its module docstring).
Outputs: `refit_standardized/ladders_{context,prefix}.json` — pairs
r1↔r4, r1↔r4op, r2↔r4, both directions, ceilings + 9-rung ladders +
matched-capacity nulls + kNN retrieval. Headline (context arm, layer 19,
matched rows): story inserted −0.245 → **+0.375** reduced (+0.445 guarded
ambient); story on-policy +0.047 → +0.112 (n=135); chat matched −0.308 →
+0.482; plain-text matched −0.247 → +0.512. The prefix arm has NO reduced
read by construction — the chat/plain-text prefix slot is the parent's
documented batch-numerics degeneracy (~14 distinct vectors, so a rank-864
train-fold PCA basis cannot exist); its guarded-ambient reads: story
inserted −1.620 → +0.146, chat matched +0.117, plain-text +0.111, story
on-policy ≈ 0. Selected-λ diagnostics: story-input legs select λ = 1e3 in
the reduced basis (dof_frac_max 0.42) and 1e3–3.2e3 guarded-ambient
(dof_frac ≤ 0.38) — never the 0.01 near-interpolating floor the unguarded
selector chose. Stores staged from HF pins `2a3cb30a` (r1/r2) and
`87b0c19e` (r4/r4op, the recovered round's own upload); RunPod CPU lane was
capacity-exhausted (3× cpu5m miss + cpu3c disk cap), so the round ran on
the VM data disk under the documented fallback (markers v295/v296 on
#1345).
