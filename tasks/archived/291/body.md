---
title: 'Auto-upload-datasets-to-HF-Hub does not actually run; #186 training data unrecoverable'
kind: infra
tags: []
created_at: '2026-05-06T09:27:40.000Z'
has_clean_result: false
sagan_id: 2ea7d61e-d7a2-4f10-96c2-ce89eb0267bb
sagan_number: 291
priority: normal
legacy_why_unset: true
---
## Goal

Fix the auto-upload-datasets-to-HF-Hub workflow that CLAUDE.md's Upload Policy promises but doesn't actually run.

## Problem

Per CLAUDE.md "Upload Policy":

> Datasets (JSONL) — Destination: HF Hub (`superkaiba1/explore-persona-space-data`) — When: Auto after generation

For #186's wrong-answer SFT training data, this auto-upload **did not happen**. Discovered when #280's experimenter tried to read `data/sft/issue186/<source>_generic-cot_seed42.jsonl` from HF Hub and found nothing.

## Evidence (verified 2026-05-06)

- **WandB project `thomasjiralerspong/explore_persona_space`**: 79 `i186_*` runs total. Inspected `i186_librarian_generic_cot_seed42` (`avqcaun5`, finished): 0 logged artifacts, 0 used artifacts. Only the 6 standard files (`code/`, `config.yaml`, `output.log`, `requirements.txt`, `wandb-metadata.json`, `wandb-summary.json`).
- **HF Hub dataset repo `superkaiba1/explore-persona-space-data`**: 707 files total. Top-level dirs: `axis_category_projection/`, `cot_axis_tracking/`, `leakage/`, `make_evil_dumb_*/`, etc. **Zero files matching `i186` / `issue186` / `generic-cot` / `persona-cot`.**
- **HF Hub model repo `superkaiba1/explore-persona-space`**: 234 `i186_*` files but ALL of them are merged-checkpoint dirs (model.safetensors / config.json / tokenizer files). Zero JSONLs.
- **Local repo + worktrees** (main, issue-186, issue-280): `data/sft/issue186/` does not exist on any branch.

The data only ever existed transiently on `epm-issue-186` (now terminated), was consumed by training, and is gone.

## Why this matters

1. **Reproducibility:** anyone trying to reproduce #186's wrong-answer SFT cannot — the training inputs are not recoverable.
2. **Follow-up cost:** #280 (a follow-up of #186) needs `data/sft/issue186/<source>_generic-cot_seed42.jsonl` for `scrambled_english_cot` arm and audit reference; without auto-upload, the only path is regeneration at +$54 API.
3. **Latent risk for every future experiment:** the same gap exists for any project script that generates training data. Without an auto-upload step in the data-gen pipeline, results are eval-only-reproducible (model + outputs preserved) but not training-input-reproducible.

## Hypothesis on the cause

Likely either:
- The data-generation script (`scripts/generate_issue186_data.py`) has no `huggingface_hub.upload_folder` call at the end of Phase-0
- Or it has one but it's gated behind a flag the launcher doesn't set
- Or it logs to WandB Artifacts but the `wandb.log_artifact` call is missing

The eval pipeline DOES auto-upload `result.json` to WandB Artifacts, so the infrastructure works for one direction. The gap is specifically in the data-generation path.

## Acceptance criteria

- [ ] Identify which data-gen scripts in `scripts/` are supposed to auto-upload but don't (likely `generate_issue186_data.py`, `generate_wrong_answers.py`, `build_sft_datasets.py`, `generate_leakage_data.py`)
- [ ] Add an auto-upload step to each: write JSONLs to `data/sft/<issue_id>/`, then call `huggingface_hub.upload_folder` to push to `superkaiba1/explore-persona-space-data:data/sft/<issue_id>/`
- [ ] Wire it into Phase-0 finalization so it runs unconditionally (no flag required)
- [ ] Add a CLAUDE.md-section change-control note: "Phase-0 scripts MUST end with an upload step; verify with `hf api list-repo-files superkaiba1/explore-persona-space-data --revision main | grep <issue_id>` after every Phase-0 run"
- [ ] Optional: write a one-shot `scripts/backfill_i186_dataset.py` that regenerates and uploads #186's training data (this overlaps with #280's Option A — those two issues are coupled)

## Compute / cost

- **No GPU.** Code change only.
- **Test cost:** dry-run with `--dry-upload` flag, ~$0. Real upload of #186's data: +$54-175 of Sonnet calls if we want a clean backfill (or skip if the regen happens via #280's Option A and we just upload that result).

## References

- CLAUDE.md "Upload Policy" table — claims datasets auto-upload but the implementation is missing
- #186 — the parent issue whose data is missing
- #280 — the immediate consumer that hit this gap
