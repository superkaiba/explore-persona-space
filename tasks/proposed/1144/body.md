---
title: Port issue-1090 stranded shared-src fixes to main (sft.py CVD pin, datagen
  mt500 + posonly panel, bare/ICL contexts)
kind: infra
tags: []
created_at: '2026-07-08T10:45:23Z'
has_clean_result: false
origin_prompt: 'filed by the /issue 1090 orchestrator at the fu3 merge: guard-3 refusal
  stranded review-PASSed shared-src fixes on the branch; port them so they are not
  built-but-stranded'
workflow: v1
---
## Overview / Motivation

Filed by the /issue 1090 orchestrator at the fu3 Step-10d merge: guard 3 refused the full rebase (ON_MAINLINE=no, BEHIND=3004), so the round's artifact set landed via surgical additive checkout (main @ 9afb7e85a0) but the branch's MODIFIED shared-src files could not ride along. Per the built-but-stranded rule (workflow-fix-on-bug.md § Built-but-stranded), a fix that is not on main and called by the running path changes nothing — the sft.py CVD fix in particular is fleet-relevant.

## Goal

Port the issue-1090 branch's shared-src modifications to main via a clean scratch worktree (detached at origin/main) cherry-pick / re-apply, with their tests, so the fleet inherits them.

## The stranded set (all on branch issue-1090, review-PASSed there)

1. `src/explore_persona_space/train/sft.py` — `_apply_cvd_pin`: an inherited single-GPU CUDA_VISIBLE_DEVICES pin is AUTHORITATIVE through `train_lora`/`merge_lora` (never re-exported from cfg.gpu_id; contradiction → RuntimeError; legacy unset-CVD `+gpu_id=N` path byte-identical). Fixes the #557-class co-location that OOM'd 14 cells (launch 3). Commit `2f7978058f` + 4 unit tests. FLEET-RELEVANT: every CVD-pinned parallel launcher is exposed to the old clobber.
2. `src/explore_persona_space/artifacts/datagen.py` — DATAGEN_JUDGE_MAX_TOKENS=500 at `_judge_and_filter` (rule-23 truncation fix; the mt500 cache re-key) + `generate_training_data` accepts an EMPTY negatives panel (posonly twin; `_negative_arm` extraction). Commits `74861be2` + `3597cc4a39`.
3. `src/explore_persona_space/artifacts/context.py` — `"bare"` added to INSTALLABLE_KINDS + `icl_prefix_context` factory. Commit `3597cc4a39`.
4. `src/explore_persona_space/artifacts/organisms.py` + `scripts/issue1090_fu3_*.py` — gpu_id threading + the fu3 worker/aggregate/topup scripts (port the scripts as-is; they are the round's repro surface).
5. `src/explore_persona_space/artifacts/query_banks/icl_examples_{formatting,impolite,sycophancy,broad_em}.json` — sha-pinned authored 2-shot ICL banks (A-only data files).
6. `tests/test_issue1090_fu3_*.py` + the sft CVD test additions.

## Constraints

- Cherry-pick from a scratch worktree detached at origin/main (`git worktree add --detach`), NEVER a branch switch of the shared repo root; resolve conflicts toward the branch's reviewed content; run the touched-file test subset + ruff before push.
- The changes were code-review-PASSed on the branch (epm:code-review v12/v13/v15 on task #1090) — the port review verifies the PORT (clean apply, no regression vs main HEAD), not a re-litigation.

## Provenance

- source_branch: issue-1090 @ 2022a2a734
- origin: task #1090 fu3 Step-10d artifact-confirmed merge (epm:merged v1, full_rebase_deferred)
