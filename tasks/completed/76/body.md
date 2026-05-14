---
title: Standardize pod venv to explore-persona-space/.venv; remove make-evil-dumb/.venv;
  add venv preflight check
kind: infra
tags: []
created_at: '2026-04-22T04:08:26.000Z'
has_clean_result: false
sagan_id: 8ceafe7e-ed1a-47b2-89a8-abd2c576a558
sagan_number: 76
priority: high
---
## Problem

Pods have inconsistent Python environments, putting every experiment at reproducibility risk. Observed (2026-04-22):

| Pod | `/workspace/explore-persona-space/.venv` | `/workspace/make-evil-dumb/.venv` |
|-----|------------------------------------------|-----------------------------------|
| pod2 | torch 2.8.0+cu128, transformers **5.5.0**, trl **0.29.1**, peft 0.18.1 | torch 2.8.0+cu128, transformers **5.5.3**, trl **1.0.0**, peft 0.18.1 |
| pod3 | exists (versions not captured before it was quiet) | unknown |
| pod4 | exists | **does not exist** |
| pod5 | not checked | not checked |

Three compounding problems:

1. **Version skew within a single pod.** pod2's two venvs disagree on `transformers` (5.5.0 vs 5.5.3) and `trl` (0.29.1 vs 1.0.0). Which wins depends on which script's shebang/activate fires first.
2. **Dual-venv inconsistency across pods.** pod2 has both venvs; pod4 only has the `explore-persona-space` venv. Pipeline scripts that hard-code `/workspace/make-evil-dumb/.venv/bin/python` work on pod2 and silently pick a different env on pod4.
3. **On-pod inner scripts at `/workspace/midtrain_25pct_seed137/` are NOT in the repo.** The venv they source is decided per-pod, not from git. Issue #67's seed-137 results could have run under either venv.

Concretely, the fact-checker for issue #74 flagged that our Reproducibility Card would be wrong because the pipeline's actual venv is not the repo's venv.

## Scope

- (a) Audit every pipeline launcher under `scripts/pod{1..5}/**/*.sh`, on-pod inner scripts at `/workspace/midtrain_25pct*/`, and `scripts/run_midtrain_25pct.sh` for hardcoded venv paths.
- (b) Point every launcher at `/workspace/explore-persona-space/.venv` (source of truth).
- (c) Run `uv sync --locked` on all 5 pods from the `explore-persona-space` working copy so the single canonical venv matches `uv.lock`.
- (d) Remove `/workspace/make-evil-dumb/.venv` (and the `make-evil-dumb` repo directory, once confirmed no unuploaded artifacts) from all pods that have it.
- (e) Add a preflight check in `explore_persona_space.orchestrate.preflight` that fails if: (1) the active venv is not `/workspace/explore-persona-space/.venv`, (2) `make-evil-dumb/.venv` still exists, or (3) any of `torch`, `transformers`, `trl`, `peft`, `deepspeed`, `accelerate` version disagrees with `uv.lock`.
- (f) Update CLAUDE.md's Pre-Launch Protocol section to reference the preflight's venv check.

## Out of scope

- Changing the pinned versions in `pyproject.toml` / `uv.lock` — if the repo's `uv.lock` is the wrong target version, file a separate issue.
- Bootstrapping new pods (already covered by `scripts/pod.py bootstrap`).
- Backfilling results from experiments that ran under the stale venv — see "Follow-ups" below.

## Acceptance criteria

1. `python scripts/pod.py health --json` passes on all 5 pods with a new check `venv_canonical: true`.
2. `/workspace/make-evil-dumb/.venv` does not exist on any pod.
3. `uv run python -m explore_persona_space.orchestrate.preflight --json` returns `ok=true` on all 5 pods.
4. Grep across all pipeline shell scripts (`scripts/pod*/`, `scripts/run_*.sh`, on-pod `/workspace/midtrain_25pct*/*.sh`) returns zero matches for `make-evil-dumb` or `/make-evil-dumb/`.
5. A sample end-to-end dry-run (e.g., `bash scripts/run_midtrain_25pct.sh evil_wrong /workspace/data/sft/phase1_evil_wrong.jsonl 8 /tmp/dryrun`) logs the canonical venv on startup.

## Dependencies / blockers

- **Blocks:** #74 (midtrain persona-swap matrix). Do not launch #74 until this is resolved — the Reproducibility Card would be wrong.
- **Should not touch pod3 while #48 is running.** (Triage pending: #48 is labeled `status:running` but all target pods are idle as of 2026-04-22. Coordinate with #48 before deleting any pod state.)
- If `make-evil-dumb` dir contains unuploaded checkpoints/results, export them to HF Hub / WandB first.

## Follow-ups (separate issues)

- Determine whether issue #67's seed-42 vs seed-137 ZeRO-2/ZeRO-3 confound was also venv-confounded. If seed-42 ran under `make-evil-dumb/.venv` and seed-137 under `explore-persona-space/.venv`, that's an additional caveat on the #67 clean result.
- Retrospective on how the dual-venv state was introduced without being caught in preflight.

## Suggested labels

`type:infra`, `compute:none`, `prio:high`, `aim:infra`
