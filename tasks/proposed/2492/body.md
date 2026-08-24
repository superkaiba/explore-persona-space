---
title: 'Sparse pod clones vs committed cross-issue inputs: pre-launch stat-check +
  git-show materialization recipe (experimenter.md + gotchas.md)'
kind: infra
tags:
- workflow-fix
- wf-fingerprint:sparse-clone-cross-issue-input-statcheck
created_at: '2026-08-23T04:56:59Z'
has_clean_result: false
origin_prompt: 'experimenter #2476 failure-lesson: pod sparse clone excluded eval_results/issue_1482
  committed inputs; first launch FileNotFoundError; fixed inline via git show HEAD:<path>
  materialization'
workflow: v1
---
# Pre-launch stat-check of committed cross-issue inputs on sparse pod clones (experimenter/gotchas gap)

## Provenance
Surfaced by the #2476 experimenter spawn (2026-08-23, session cmt4td1zor48ewz0unlixnkji): the first pod-2476 launch died in ~5 s on `FileNotFoundError: eval_results/issue_1482/split_1482.json`. The pod bootstrap clone is SPARSE (cone excludes every other issue's `eval_results/`), while `scripts/issue2476_turnavg_sae.py` hard-reads two committed #1482 inputs. The Step 6a.5 carry-over gate had PASSed both paths as `in-ref` at `origin/issue-2476` — "tracked at the launch SHA" does not imply "materialized on the pod". The experimenter fixed it inline: `git show HEAD:<path> > <path>` on the pod + `git hash-object` verify against the committed blob (`cdb9a68b`/`b05d19dc`), then relaunched clean.

## Goal (what to fix)
Close the gap on the workflow surface so the next sparse-clone launch does not burn a boot cycle:

1. `.claude/rules/gotchas.md` — add an entry: pod bootstrap clones are sparse (cone excludes other issues' `eval_results/`); a driver hard-reading committed CROSS-ISSUE inputs crashes FileNotFoundError despite in-ref carry-over gate PASS. Recipe: stat-check the driver's committed cross-issue reads ON the pod pre-launch; on a miss, materialize per-file via `git show HEAD:<path> > <path>` + `git hash-object -w`-free blob verify against the committed hash (prefer per-file materialization over `git sparse-checkout add` for large subtrees through MooseFS).
2. `.claude/agents/experimenter.md` — add the pre-launch stat-check duty (bounded: the paths the plan/6a.5 gate names as committed repo inputs) + the materialization recipe.
3. OPTIONAL (assess): `scripts/verify_carryover_inputs.py` or the Step 6 docs — annotate the `in-ref` PASS reason with a sparse-clone caveat line for cross-issue `eval_results/` paths on pod lanes, so the gate's own output names the residual.

## Acceptance
- Both surface edits landed + lints clean (`workflow_lint.py --check-lessons-index` if a rules file is added/renamed — this edits an existing rule, so index untouched).
- A grep for the recipe in `gotchas.md` and `experimenter.md` finds it.

failure_class: infra; generalizes: yes; root_cause_confirmed: yes (experimenter verified blob hashes + healthy relaunch).
