---
title: 'upload_dir_sharded: batch small-file stores into one upload_folder commit
  (per-file walk measured ~35 s/file, #1482)'
kind: infra
tags: []
created_at: '2026-07-29T08:58:43Z'
has_clean_result: false
origin_prompt: 'Auto-filed by the #1482 orchestrator: E2 store upload ground at ~98
  files/h (3,840-file / 1.2 GB store, per-file upload_file commits); recovery was
  a manual one-commit upload_folder. Library fix: batch when the store fits disk +
  skip-if-present resume.'
workflow: v1
---
## Overview / Motivation

Filed from the #1482 early-layer-arm run (2026-07-29): `orchestrate.upload_sharded.upload_dir_sharded` walked a 3,840-file / ~1.2 GB store at ONE `api.upload_file` COMMIT PER FILE — measured 539 files in ~5.5 h (~35 s/file, commit round-trip dominated) → ~33 h projected for a store that a single `upload_folder` commit moves in minutes. The walk held an idle A100 the whole time and consumed the shared data repo's commit budget (~100 commits/h against the 256/h Hub cap the module's own #1034 comment cites). The disk-bounded one-file-at-a-time design is correct ONLY when the store is larger than local disk; at small/many-file stores it is pure per-commit overhead (the overhead-bound serial-IO signature). Recovery on #1482 was a manual one-commit bulk `upload_folder` + concurrent fits launch (task #1482 events, epm:progress decision record 2026-07-29T08:4xZ).

## Goal

Make `upload_dir_sharded` batch its shard files into `upload_folder`-style grouped commits whenever the store's on-disk byte sum fits comfortably under the local-disk / quota bound, falling back to the existing per-file walk only when disk-boundedness is actually needed; add a skip-if-already-on-Hub probe (one scoped `list_hf_files_under_path` listing up-front) so interrupted walks resume without re-committing.

## Gap

- **Bug observed:** per-file `upload_file` commits at ~35 s/file on a 3,840-file store (#1482 E2: 539/3,840 in 5.5 h, ~33 h projected; GPU idle; shared-repo commit pressure).
- **Why it is a library gap:** the module already knows the hazard — its #1034 comment warns about "the 256 commits/hr Hub cap" for pointer commits, and sibling call sites hand it many-small-file stores (issue1335/issue1073/issue1417 + issue1482); the driver's OWN E4 phase documents the rule it violates ("ONE upload_folder commit for the whole eval dir — never a per-file upload loop"). Upload Policy (CLAUDE.md): "use a single bulk `upload_folder` commit for many files — never a per-file `upload_file` loop".
- verified-at-filing: `grep -n "api.upload_file" src/explore_persona_space/orchestrate/upload_sharded.py` → 4 hits (lines 104, 170, 386, 400 — the per-shard loop + overflow reroute) and `grep -rn "upload_dir_sharded(" scripts/ src/` → 8+ live call sites (issue1335_run.sh, issue1073_fits.py, issue1073_capture.py ×2, issue1417_run.sh ×3, issue1482_early_layer.py ×2 on its branch) (2026-07-29).

## Proposed change (refine in planning)

- In `upload_dir_sharded`: compute `projected = sum(st_size)` (already done for the #1034 headroom probe). When `projected` is under a batching threshold (e.g. ≤ min(free_disk*0.5, ~50 GB)) AND `delete_local=False` or the files already all exist locally (they do — it is a walk of an existing dir), upload the whole selection via ONE `upload_folder` call (`allow_patterns` from `shard_glob`), keeping the overflow-reroute semantics (catch quota-403 → reroute the folder to overflow) and the batched verify.
- Per-file walk retained for the genuinely disk-bounded case (delete-as-you-go).
- Resume probe: list the destination prefix once up-front (`list_hf_files_under_path`) and skip already-present same-size files in BOTH modes.
- unverified hypothesis — verify at plan time: whether `upload_folder`'s single commit needs chunking above some file-count (HF warns on "large folder"; `upload_large_folder` exists for the huge case).

## Constraints / invariants

- Preserve the overflow-reroute + pointer + JSONL event semantics (#1034) and the fail-loud never-silently-drop contract.
- `tests/` for the new batching branch (injectable api per the existing test seam).
- src/orchestrate is NOT workflow surface — ordinary infra pipeline (planner → implementer → code-review → Step 9c).

## Provenance

- origin: task #1482 early-layer-arm E2 incident, 2026-07-29 (decision record on #1482 events.jsonl; measured rate 539 files/5.5 h).
