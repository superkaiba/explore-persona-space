---
title: 'workflow-fix: GCE crash-persist workload.log upload silently fails on large
  logs — tail-cap before upload'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d825b346444a
created_at: '2026-07-18T16:27:53Z'
has_clean_result: false
origin_prompt: 'Surfaced from #1481: two sycophancy-lane crash-persists uploaded manifests
  but no workload.log (eps/persist=attempted / failed_uploads)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1481 (emitting agent: the #1481 orchestrator session).

NOTE (provenance repair): the original filing (commit `14f2952cab`,
2026-07-18T16:27:53Z) landed frontmatter-only — no body file was passed. This
body was reconstructed at /issue Step 0b by the #1517 session from the task
title, the `origin_prompt` frontmatter, parent #1481's `epm:failure v4` marker
(2026-07-18T16:27:49Z), and direct inspection of the target code.

## Goal

Tail-cap the canonical `workload.log` at crash-persist STAGE time (both the
first-bundle copy and the per-crash timestamped final-bundle copy) so a large
log stays on the non-LFS Hub path and the crash-persist's most important
forensic artifact can no longer silently fail to upload.

## Workflow gap

- **Bug observed:** two #1481 sycophancy-lane GCE crash-persists uploaded
  manifest JSONs but NO `workload.log` (`eps/persist=attempted`, rc-3
  `failed_uploads`); the recorded hypothesis (#1481 `epm:failure v4`,
  2026-07-18T16:27:49Z) is that the uncapped log exceeded the ~10 MB threshold
  at which `upload_folder` force-routes a blob to LFS, while the namespace
  sits over the LFS public-storage soft ceiling — the small manifest JSONs
  ride the non-LFS path and land, the log alone fails.
- **Why it is a workflow gap:** `_eps_persist_diagnostics`
  (`src/explore_persona_space/backends/gcp.py`) is the crash-diagnostics
  safety net for the ephemeral GCE lane (the EXIT-trap DELETE destroys the
  boot disk, #658). Worker logs already get a per-file TAIL cap at stage time
  (`EPS_PERSIST_LOG_FILE_CAP_BYTES`, default 5 MiB — #885) and the #935
  done-persist tail-caps its `workload_tail.log` (`TAIL_CAP = 5 MiB`), but the
  canonical `workload.log` staging is uncapped in BOTH crash-persist bundles —
  precisely the artifact, and the long-run/big-log crash class, where the
  traceback matters most.
- **Confidence (emitter):** medium — the LFS-routing mechanism is a recorded
  hypothesis consistent with the manifests-landed / log-missing evidence, not
  yet directly reproduced.
- verified-at-filing: `grep -n '_stage_into(first_stage, "workload.log"\|_stage_into(final_stage, f"workload_{stamp}.log"' src/explore_persona_space/backends/gcp.py`
  → 2 hits in 1 file (`gcp.py:1873` first bundle; `gcp.py:2127` timestamped
  final copy), both read-in-context as uncapped (2026-07-18). Landed-fix
  check: `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/gcp.py`
  → 8 commits, none caps the canonical workload.log staging.

## Proposed change (candidate diff sketch — refine in planning)

```
# in the rendered _eps_persist_diagnostics python (backends/gcp.py):
- _stage_into(first_stage, "workload.log", log_path)
+ _stage_tail_into(first_stage, "workload.log", log_path)   # tail-capped
  ...
- _stage_into(final_stage, f"workload_{stamp}.log", log_path)
+ _stage_tail_into(final_stage, f"workload_{stamp}.log", log_path)
# _stage_tail_into = _stage_into + the done-persist TAIL_CAP shape (seek to
# size-CAP, write the last CAP bytes; reuse EPS_PERSIST_LOG_FILE_CAP_BYTES /
# LOG_FILE_CAP, default 5 MiB — safely under the ~10 MB LFS routing
# threshold); _say "staged workload.log (last K of M bytes)". Repo paths stay
# byte-identical ({dest}/workload.log — the #1151 contract).
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep for other uncapped canonical-log stagings before editing
  (`grep -n '_stage_into' src/explore_persona_space/backends/gcp.py`) and
  update every hit that stages the workload log; list them in the plan.
- Planner's call: whether the log HEAD carries value worth a small head+tail
  split; the done-persist precedent is tail-only, and the crash-persist
  comments already note "the canonical log already carries the traceback
  early" — reconcile with the tail-keeps-the-END rationale.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes.
- Crash-persist invariants preserved: one `upload_folder` commit per bundle
  (#1151 — never per-file loops, #664), fully guarded + 300s-bounded (#854),
  transcript-uploaded-LAST audit ordering, `eps/persist=ok` honesty gate
  (#1343) untouched.
- This session carries a `workflow_fix_target:` Provenance line and runs under
  the recursion guard — it MUST NOT auto-route any of its own subagents'
  workflow-fix candidates (log + notify only).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: d825b346444a
- origin_prompt (verbatim): "Surfaced from #1481: two sycophancy-lane
  crash-persists uploaded manifests but no workload.log (eps/persist=attempted
  / failed_uploads)"
- The original candidate block was not persisted (no
  `epm:workflow-fix-task-filed` marker found on #1481; frontmatter-only
  filing) — this section is the reconstructed record.
