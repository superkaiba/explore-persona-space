---
title: 'workflow-fix: cover relocated HF_HUB_CACHE in guard HF-cache tiers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7ea7f124d074
created_at: '2026-08-05T20:21:25Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/vm_disk_guard.py\n\
  bug_observed: Tier (e) is hardcoded to ~/.cache/huggingface/hub so the 249 GB relocated\
  \ HF_HUB_CACHE cache on /mnt/eps-data is invisible to the revision reap and the\
  \ 50 GB size cap\nwhy_workflow_gap: The #1369 HF_HUB_CACHE relocation and the #1376/#1377/#1450\
  \ cache-janitor arms were built independently; the tier's deliberate fixed-path,\
  \ boot-pass-only, /-threshold design was never reconciled with the relocation\n\
  proposed_change: Make the home-HF-cache tier multi-root so the relocated HF_HUB_CACHE\
  \ hub cache on the data disk gets the revision reap and size cap\ndiff_sketch: |\n\
  \  + clean_home_hf_stale_revisions(roots: list[Path]) — default [home root] + resolved\
  \ HF_HUB_CACHE parent\n  + (or env EPS_VM_EXTRA_HF_CACHE_ROOTS); dedupe resolved\
  \ roots via the double-cover guard\n  + size-cap arm per root; extra-root pass invoked\
  \ from the data-disk branch on the data disk's threshold\n  + xet/ named residual;\
  \ hub-only contract unchanged\nconfidence: high\nrelated_task: n/a (chat-mode deep\
  \ dive, 2026-08-05)\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a chat-mode deep dive (2026-08-05, user-directed: "Do a deep dive and propose workflow fixes" on the fleet-wide disk pressure). Emitting agent: orchestrator + read-only janitor-coverage and data-disk-census audit subagents.

## Goal

Make the home-HF-cache tier of `vm_disk_guard.py` multi-root so the relocated `HF_HUB_CACHE` hub cache on the data disk gets the revision reap and the #1450 size cap.

## Workflow gap

- **Bug observed:** Tier (e) (`clean_home_hf_stale_revisions` + the #1450 size-cap arm) is hardcoded to `~/.cache/huggingface/hub`, so the 249 GB relocated `HF_HUB_CACHE` cache at `/mnt/eps-data/$USER/huggingface-cache/hub` is invisible to the revision reap and the 50 GB size cap. Census (2026-08-05): that cache holds 247.8 GB of hub, of which **211.6 GB is ONE repo cache — `datasets--superkaiba1--explore-persona-space-data`, the project's own HF data repo** (a re-downloadable mirror of already-uploaded artifacts); next-largest are Qwen2.5-7B + -Instruct at 14.2 GB each. It is the single largest reclaimable object on either disk.
- **Why it is a workflow gap:** The #1369 relocation (env-wired via `HF_HUB_CACHE` in `~/.bashrc:4` / `~/.profile:34`) and the cache-janitor arms (#1376/#1377/#1450) were built independently; tier (e)'s fixed-path design is deliberate ("Deliberately NOT derived from HF_HOME/HF_HUB_CACHE", docstring) and predates the relocation — nothing ever reconciled them. Tier (e) is additionally boot-pass-only and `/`-threshold-triggered, so even a symlink would not have brought the data-disk cache under the data disk's own pressure signal.
- **Secondary gap (same reconciliation failure, planner may split or fold):** `~/.cache/huggingface` is a REAL directory on the 98%-full boot disk (NOT a symlink), still holding a live 58.7 GB hub + 8.1 GB datasets — because the env export only binds in shells sourcing `.bashrc`/`.profile`, cron/non-login-shell workloads silently write hub cache to `/`. `unverified hypothesis — verify at plan time:` which recurring writers still hit the home cache (compare repo mtimes under both hubs).
- **Confidence (emitter):** high
- verified-at-filing (re-run by the filer 2026-08-05):
  - `grep -n 'Deliberately NOT derived' scripts/vm_disk_guard.py` → 1 hit (L1021): tier (e) root is fixed + env-blind by documented design.
  - `grep -n 'HF_HUB_CACHE' ~/.bashrc ~/.profile` → 2 export lines (bashrc:4, profile:34) pointing at `/mnt/eps-data/$(id -un)/huggingface-cache/hub`; `ls -d /mnt/eps-data/thomasjiralerspong/huggingface-cache/hub` → exists.
  - Sizes: measured 2026-08-05 by the read-only census subagent (`du -s` per subdir): data-disk hub 247.8 GB (211.6 GB = the project data-repo cache), xet 0.28 GB; home hub 58.7 GB + datasets 8.1 GB. `~/.cache/huggingface` confirmed a plain dir via `ls -la`.
  - Landed-fix history: `git log --oneline --since='14 days ago' -- scripts/vm_disk_guard.py` → no commit touches HF-cache tier roots.

## Proposed change (candidate diff sketch — refine in planning)

```
+ clean_home_hf_stale_revisions(roots: list[Path]) — default [home_hf_cache_root()] plus the resolved
+   HF_HUB_CACHE parent when it points elsewhere (or new env EPS_VM_EXTRA_HF_CACHE_ROOTS); dedupe
+   RESOLVED roots (extend the existing double-cover guard) so a future symlink never double-reaps
+ size-cap arm (#1450) runs per root with a per-root cap (data-disk cap sized in the plan; the 50 GB
+   default is boot-disk-sized)
+ invoke the extra-root pass from the DATA-DISK branch of main() so its trigger is the data disk's
+   threshold (EPS_VM_DATA_DISK_* percent convention), not '/'s byte floor
+ xet/ stays a NAMED residual (hub-only contract unchanged) — state it in the run report
```

Safety notes for the plan: the reap must keep the newest + every ref'd revision per repo (existing arm semantics); an in-flight reader of the project data-repo cache (e.g. a live #1739 round) must not have files deleted from under it — the existing unused-≥N-days recency gates carry this, keep them binding.

## Scope / surfaces

- Primary target: `scripts/vm_disk_guard.py` (+ `scripts/cron_vm_disk_guard.sh` only if the invocation needs a new env passthrough)
- Cross-references: #1369 (the relocation this reconciles), #1376/#1377/#1450 (the arms being extended), #1038 (bind-cutover — unrelated to this tier, do not couple).
- Grep before editing: `grep -rln 'EPS_VM_HOME_HF_CACHE\|home_hf_cache_root' scripts/ tests/` and update every hit + the invariant tests.

## Constraints / invariants

- Workflow-surface only. Escalate-never-delete posture preserved wherever evidence is ambiguous; report-only default, `--apply` from cron unchanged.
- `scripts/workflow_lint.py --check-asks` + ruff pass; janitor invariant tests updated alongside.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/vm_disk_guard.py
- fingerprint: 7ea7f124d074

<!-- workflow-fix-candidate v1 -->
target_file: scripts/vm_disk_guard.py
bug_observed: Tier (e) is hardcoded to ~/.cache/huggingface/hub so the 249 GB relocated HF_HUB_CACHE cache on /mnt/eps-data is invisible to the revision reap and the 50 GB size cap
why_workflow_gap: The #1369 HF_HUB_CACHE relocation and the #1376/#1377/#1450 cache-janitor arms were built independently; the tier's deliberate fixed-path, boot-pass-only, /-threshold design was never reconciled with the relocation
proposed_change: Make the home-HF-cache tier multi-root so the relocated HF_HUB_CACHE hub cache on the data disk gets the revision reap and size cap
diff_sketch: |
  + clean_home_hf_stale_revisions(roots: list[Path]) — default [home root] + resolved HF_HUB_CACHE parent
  + (or env EPS_VM_EXTRA_HF_CACHE_ROOTS); dedupe resolved roots via the double-cover guard
  + size-cap arm per root; extra-root pass invoked from the data-disk branch on the data disk's threshold
  + xet/ named residual; hub-only contract unchanged
confidence: high
related_task: n/a (chat-mode deep dive, 2026-08-05)
<!-- /workflow-fix-candidate -->
