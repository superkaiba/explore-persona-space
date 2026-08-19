---
title: 'pod bootstrap sparse-cone excludes tracked data/ inputs → fresh-pod FileNotFoundError
  (from #2203 Phase 3)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-10T02:12:14Z'
has_clean_result: false
parent_id: 2203
origin_prompt: 'Phase-3 32B anchor crashed on missing data/assistant_axis/role_list.json
  — pod sparse cone excluded data/; #2203 crash-fix round, 2026-08-10'
workflow: v1
---
## Goal

Close the pod-bootstrap sparse-cone gap that makes an experiment crash on a fresh pod when it reads a git-TRACKED input under `data/` — decide and implement the right fix (extend the default bootstrap cone to cover small tracked `data/` input subdirs, OR a documented mechanism for a plan to declare such a cone, OR make experiments stage these inputs from HF), and document it so planners/experimenters expect it.

## Context (incident, #2203 Phase 3, 2026-08-10)

`bootstrap_pod.sh` clones with `--filter=blob:none` + cone sparse-checkout; the realized cone on a fresh pod-2203 (H200) was `configs docs eval_results/issue_2203 figures/issue_2203 scripts src tests` — `data/` is NOT in the cone. Phase 3 (`scripts/issue2203_phase3.py`) crashed at `build_jailbreak_set` → `load_role_list()` with `FileNotFoundError: data/assistant_axis/role_list.json`, AFTER the 32B model loaded (~1 GPU-cycle + a crash-fix round wasted). The file IS tracked on the branch and NOT gitignored (`git ls-files` returns it, `git check-ignore` empty) — but its blob is never materialized because `data/assistant_axis/` sits outside the sparse cone. Recovery was `git sparse-checkout add data/assistant_axis` on the pod (materialized role_list.json + extraction_questions.jsonl + 276 instruction files, byte-integrity verified).

Why the existing guard did not catch it: the #1739 gotcha (`.claude/rules/gotchas.md`, "Partial-clone pods") covers a pod reading ANOTHER issue's committed artifacts via `BOOTSTRAP_EXTRA_CONES` — it does NOT obviously cover THIS repo's own tracked `data/` inputs excluded by the DEFAULT cone. `data/` legitimately holds both huge re-downloadable caches (`data/issue_<N>/hf_dl`) and small tracked inputs (`data/assistant_axis/`), so the fix must exclude the caches while including the tracked inputs.

Note: Phase 0/1/2 ran on a DIFFERENTLY-bootstrapped pod (the first pod-2203, full bootstrap) and did not hit this — the interrupted-bootstrap + manual-`git checkout issue-2203` path on the Phase-3 pod is what surfaced the cone gap, but the gap is latent for any fresh pod whose experiment reads a tracked `data/` input.

## Acceptance

1. A decided fix implemented (one of: `bootstrap_pod.sh` default cone includes tracked `data/**` input subdirs excluding the `hf_dl`/`g*_dl` cache dirs; OR a `BOOTSTRAP_EXTRA_CONES`-style declaration path documented + wired for `data/` inputs; OR an experiment-side stage-from-HF convention for `data/assistant_axis`-class inputs).
2. The relevant gotcha / planner / experimenter surface documents the expectation so a planner declares (or an experimenter stages) tracked `data/` inputs for pod phases.
3. Any mechanical check (e.g. `tests/test_bootstrap_pod_issue_cones.py`) updated if the default cone changes.
