---
title: Workflow improvements
kind: infra
tags: []
created_at: '2026-05-06T10:23:22.000Z'
has_clean_result: false
sagan_id: 0f326160-84a3-4e81-baaa-de012d257090
sagan_number: 293
priority: normal
---
## Goal

Three independent workflow improvements, planned and shipped together.

## Items

### 1. Clean-result TL;DR figures match ML-paper aesthetic + captions

Figures embedded in clean-result TL;DRs should follow ML-paper conventions:

- Use `src/explore_persona_space/analysis/paper_plots.set_paper_style()` (already exists) — colorblind-safe palettes, error bars, labeled axes with direction arrows where applicable.
- Each figure gets a caption paragraph directly underneath (1–2 sentences describing panels, axes, series, N) — paper-style, not a one-liner alt.
- Multiple figures are allowed in TL;DR when each carries a distinct claim (relaxes the implicit "one hero figure" rule in `clean-results/SKILL.md`). Each additional figure must justify its presence.

**Acceptance criteria:**

- [ ] `.claude/skills/clean-results/SKILL.md` and `template.md` updated to explicitly allow multiple figures in TL;DR Results subsection when each carries a distinct claim.
- [ ] Figure block snippet in `template.md` requires an explicit caption paragraph (figure markdown → caption sentence underneath, paper-style).
- [ ] `.claude/skills/paper-plots/SKILL.md` (style spec) updated to mention: every figure gets a self-contained caption sentence including the eval N and what to look at.
- [ ] `scripts/verify_clean_result.py` updated to allow ≥1 image links in TL;DR Results subsection (currently counts hero-figure presence — relax to ≥1) AND verify each image is followed by a caption paragraph.

### 2. Closing an issue auto-routes to the Archived column

When any issue is closed (`gh issue close <N>`), the project board should automatically move it to the Archived column.

**Current state.** `status:archived` already routes to the `Archived` column via `scripts/gh_project.py:LABEL_TO_COLUMN["status:archived"] = "Archived"`. The missing piece is the trigger: closing an issue does NOT apply `status:archived`.

**Acceptance criteria:**

- [ ] Extend `.github/workflows/project-sync.yml` (or add `project-archive-on-close.yml`) to fire on `issues.closed` and: (1) strip any active `status:*` label, (2) apply `status:archived`, (3) let the existing `project-sync` route to Archived.
- [ ] Symmetrically: on `issues.reopened`, strip `status:archived` and restore `status:proposed` so the issue rejoins the To do column.
- [ ] Document in `CLAUDE.md` "Project-board status convention" table (the `status:archived` row currently reads "Closed long ago / no longer relevant" — clarify that ANY close auto-archives, and reopens restore proposed).
- [ ] Verify by closing + reopening a throwaway test issue and watching the column transitions.

**Note.** The convention "issues stay OPEN; terminal-ness lives on the project board" still holds for `done-experiment` / `done-impl` (the `/issue` skill never closes issues). Auto-archive on close only applies when a user explicitly runs `gh issue close` (duplicate / won't-fix / abandoned).

### 3. Auto-upload datasets to HF Hub (folds in #291)

Bundles the work from #291. Per CLAUDE.md "Upload Policy":

> Datasets (JSONL) — Destination: HF Hub (`superkaiba1/explore-persona-space-data`) — When: Auto after generation

This auto-upload does NOT actually run for any data-gen script in `scripts/` (verified 2026-05-06 in #291). All eight data-gen scripts (`generate_*.py` and `build_sft_datasets.py`) lack any `huggingface_hub.upload_folder` / `push_to_hub` call — the upload utility exists at `src/explore_persona_space/orchestrate/hub.py:upload_folder` but is not wired in. Result: training data for #186 etc. is unrecoverable once pods are terminated.

**Acceptance criteria** (full motivation in #291):

- [ ] Audit all data-gen scripts in `scripts/` and identify which produce JSONL datasets that should be archived: `generate_a3_data.py`, `generate_a3b_data.py`, `generate_leakage_data.py`, `generate_sdf_neutral_ai.py`, `generate_sdf_variants.py`, `generate_trait_transfer_data_v2.py`, `generate_wrong_answers.py`, `build_sft_datasets.py`.
- [ ] Add an auto-upload step at the end of each: write JSONLs locally to `data/sft/<issue_id>/` (or appropriate subfolder), then call the existing `src/explore_persona_space/orchestrate/hub.py:upload_folder` helper to push to `superkaiba1/explore-persona-space-data:data/sft/<issue_id>/`.
- [ ] Wire into Phase-0 finalization so it runs unconditionally (no flag required). A `--no-upload` flag may be exposed for dry-runs only; the production path always uploads.
- [ ] Update CLAUDE.md "Upload Policy" with a verification command (`hf api list-repo-files superkaiba1/explore-persona-space-data | grep <issue_id>` after every Phase-0 run).
- [ ] Closes #291 when merged (add `Closes #291` to the PR description).

## Compute / cost

- **No GPU.** Code change only.
- All three items are pure code/workflow changes; test cost ~$0 (lint + unit tests + a manual close/reopen test on a throwaway issue).

## References

- #291 — auto-upload datasets gap (folded in here as item 3)
- #275, #251, #226, #202 — prior "Workflow improvements" infra issues (same shape)
- `src/explore_persona_space/analysis/paper_plots.py` — existing paper style spec
- `src/explore_persona_space/orchestrate/hub.py:upload_folder` — existing HF upload utility (just needs wiring in)
- `scripts/gh_project.py:LABEL_TO_COLUMN` — routing table for status labels
- `.claude/skills/clean-results/template.md`, `SKILL.md`, `principles.md`
- `.github/workflows/project-sync.yml`, `project-auto-add.yml`
