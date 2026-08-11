---
title: 'workflow-fix: sweep /mnt/eps-data/$USER staging in disk janitors'
kind: infra
tags:
- wf-fix
- wf-fix-fp:08726cec9cfc
created_at: '2026-08-05T20:21:10Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/clean_experiment_downloads.py,\
  \ scripts/vm_disk_guard.py, scripts/autonomous_session_watch.py\nbug_observed: All\
  \ janitor enumeration roots resolve to the boot disk so the sanctioned /mnt/eps-data/$USER/issue<N>_<slug>\
  \ staging convention accumulated ~55 dirs / ~710 GB with zero reap, escalation,\
  \ or attribution\nwhy_workflow_gap: CLAUDE.md's compute-character clause sanctions\
  \ /mnt/eps-data/$USER/issue<N>_<slug>/ as the multi-GB staging home but no janitor\
  \ arm enumerates that root — the convention has no lifecycle owner\nproposed_change:\
  \ Add an opt-in /mnt/eps-data/$USER staging-roots leg to the non-canonical cache\
  \ sweep, the disk-guard data-disk pass, and the watcher data-disk attribution\n\
  diff_sketch: |\n  + production_staging_roots() -> [data_disk_root()/$USER] (main()-only\
  \ opt-in, sibling of production_tmp_root)\n  + thread staging_roots -> noncanonical_cache_dirs\
  \ (treat like the /tmp P1/P2 leg)\n  + _discover_staging_issue_numbers + wire into\
  \ clean_terminal_download_caches + the data-disk pass in main()\n  + watcher _data_disk_top_caches:\
  \ bounded du -s top-N over data_disk_path()/$USER/* after the repquota-None fallback\n\
  confidence: high\nrelated_task: n/a (chat-mode deep dive, 2026-08-05)\n<!-- /workflow-fix-candidate\
  \ -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a chat-mode deep dive (2026-08-05, user-directed: "Do a deep dive and propose workflow fixes" on the fleet-wide disk pressure). Emitting agent: orchestrator + a read-only janitor-coverage audit subagent.

## Goal

Add an opt-in `/mnt/eps-data/$USER` staging-roots leg to the non-canonical cache sweep (`clean_experiment_downloads.py`), the disk-guard data-disk pass (`vm_disk_guard.py`), and the watcher data-disk attribution (`autonomous_session_watch.py`), under the full existing safety contract, so the sanctioned per-issue staging convention finally has a lifecycle owner.

## Workflow gap

- **Bug observed:** All janitor enumeration roots resolve to the boot disk so the sanctioned `/mnt/eps-data/$USER/issue<N>_<slug>` staging convention accumulated ~55 dirs / ~710 GB with zero reap, escalation, or attribution. `/mnt/eps-data` is 96% full (47 GB free) and the guard's data-disk pass reports 0.00 GB reclaimable there.
- **Why it is a workflow gap:** CLAUDE.md's compute-character clause routes multi-GB staging to "an existing user-writable per-issue dir `/mnt/eps-data/$USER/issue<N>_<slug>/`" (#1393; #1410 explains why: the top level is root-owned), but no janitor arm ever enumerates that root — the convention was sanctioned without giving any sweep coverage of it. The #681 bind cutover never happened (`findmnt --mountpoint <repo>/.claude/worktrees` is empty), so the boot-disk-rooted globs also never incidentally resolve there.
- **Confidence (emitter):** high
- verified-at-filing (all re-run by the filer 2026-08-05):
  - `grep -n 'discover_roots' scripts/vm_disk_guard.py` → 3 hits (L817–820): tier-(b) roots = `repo_root()/"data"` + `.claude/worktrees/*/data` only — both physically on `/`.
  - `grep -c 'eps-data' scripts/vm_disk_guard.py` → 7 hits; context-read (audit 2026-08-05): all statvfs/mount plumbing (`DATA_DISK_ROOT_DEFAULT`, threshold reads, the #915 managed-symlink validation root) — zero enumeration roots. Absence claim probed semantically, not just verbatim.
  - `grep -n 'eps-data' scripts/clean_experiment_downloads.py` → 3 hits (L223–229 `data_disk_root()`), consumed only by `_managed_symlink_target` (L251–276) + the symlink disposition block in `clean_issue_downloads`.
  - `grep -n 'TMP_CACHE_ROOT_DEFAULT' scripts/clean_experiment_downloads.py` → 3 hits (L203, 525, 534): the only non-`data/` sweep root is `/tmp` (top-level).
  - `grep -n 'repquota' scripts/autonomous_session_watch.py` → primary data-disk attribution `_top_issue_caches_by_project_quota` (L4539+); live fact: `repquota` binary is NOT installed on this VM (quota seeding is owned by the parked #1038/#681 cutover), so the pass always falls to the du fallback whose glob roots are boot-disk (`PROJECT_ROOT / "data"`, L4470).
  - Live census (read-only subagent, 2026-08-05): `ls /mnt/eps-data/thomasjiralerspong | wc -l` → 61 entries; du total 958.0 GB, of which issue-keyed staging ≈ 710 GB across ~55 dirs. Naming variants all parse via the existing `extract_issue_number` P1 regex: `issue<N>_<slug>` (dominant), `issue<N>-<slug>` (`issue779-grid`), `issue_<N>_<slug>`, `i<N>_<slug>` (`i825_*`), `tmp_issue<N>_<slug>`. Only the ROOT is missing, not the pattern.
  - Landed-fix history: `git log --oneline --since='14 days ago' -- scripts/vm_disk_guard.py scripts/clean_experiment_downloads.py scripts/autonomous_session_watch.py` → recent commits are the no-progress-respawn lane, proposed_infra_sweep cap key, EBADF probe fixes — none touch data-disk enumeration.

## Proposed change (candidate diff sketch — refine in planning)

```
+ clean_experiment_downloads.py:
+   production_staging_roots() -> [data_disk_root()/<uid-name>] when the mount is live (main()-only opt-in,
+     sibling of production_tmp_root(); env EPM_STAGING_CACHE_ROOTS override)
+   thread staging_roots through clean_issue_downloads -> noncanonical_cache_dirs, treated like the /tmp
+     P1/P2 leg (top-level, uid-owned, non-recursive)
+ vm_disk_guard.py:
+   _discover_staging_issue_numbers mirroring _discover_tmp_issue_numbers (L487-508); wire into
+     clean_terminal_download_caches (L782) and pass staging roots on the DATA-DISK pass in main() (L2246-2258)
+ autonomous_session_watch.py:
+   _data_disk_top_caches (L5049): after the repquota-None fallback, bounded du -s top-N over
+     data_disk_path()/<uid-name>/* so sub-floor sidecar rows attribute the real holders
```

Downstream contract UNCHANGED and load-bearing: terminal-reap vs active-escalate (`TERMINAL_CACHE_REAP_STATUSES` — note `awaiting_promotion` counts as cache-terminal by existing policy), 48h recency keep (consider top-level-only recency for staging roots — a full rglob over a 206 GB tree is a real walk), nested `store/`+`eval_results/` hard block, and the positive re-downloadability-evidence gate (escalate-never-delete).

**Safety requirement from the census (MUST carry into the plan):** `issue1887_lambda_audit` (102.1 GB, #1887 = completed) holds PARKED #1345's tensor shards (`issue1345/story_slot_ablation_turnstore/*.pt`) — cross-issue staging exists in the wild, so a name-keyed terminal reap would destroy a parked task's store. The evidence gate must remain binding, and a dir whose CONTENT paths name a different issue than its dir name should hard-escalate, never reap.

**Named residual (planner may descope with a stated reason):** `/mnt/eps-data/tmp` (root-owned sticky, ~2 GB, 10,040 `bt-*`/`epm_isolated_*` build-temp entries) matches no issue pattern and stays invisible even after this fix; needs its own age-sweep decision.

## Scope / surfaces

- Primary targets: `scripts/clean_experiment_downloads.py`, `scripts/vm_disk_guard.py`, `scripts/autonomous_session_watch.py`
- Cross-references: #1038 (blocked bind-cutover task — owns repquota install + quota seeding; do NOT duplicate), #2015 (open; its sketch recommends MORE /mnt/eps-data staging — coordinate, this fix is its missing lifecycle half), #911/#679/#1450 (the machinery being extended), #1369 (created the relocated HF cache — covered by a SEPARATE sibling filing, not this task).
- Grep the workflow surface for the pattern before editing (`grep -rln 'eps-data' scripts/ .claude/ CLAUDE.md`) and update every hit that documents sweep coverage; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; tests pinning janitor invariants updated alongside.
- Escalate-only on the data disk stays the default posture for ACTIVE-status dirs; nothing active is ever deleted.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/clean_experiment_downloads.py, scripts/vm_disk_guard.py, scripts/autonomous_session_watch.py
- fingerprint: 08726cec9cfc

<!-- workflow-fix-candidate v1 -->
target_file: scripts/clean_experiment_downloads.py, scripts/vm_disk_guard.py, scripts/autonomous_session_watch.py
bug_observed: All janitor enumeration roots resolve to the boot disk so the sanctioned /mnt/eps-data/$USER/issue<N>_<slug> staging convention accumulated ~55 dirs / ~710 GB with zero reap, escalation, or attribution
why_workflow_gap: CLAUDE.md's compute-character clause sanctions /mnt/eps-data/$USER/issue<N>_<slug>/ as the multi-GB staging home but no janitor arm enumerates that root — the convention has no lifecycle owner
proposed_change: Add an opt-in /mnt/eps-data/$USER staging-roots leg to the non-canonical cache sweep, the disk-guard data-disk pass, and the watcher data-disk attribution
diff_sketch: |
  + production_staging_roots() -> [data_disk_root()/$USER] (main()-only opt-in, sibling of production_tmp_root)
  + thread staging_roots -> noncanonical_cache_dirs (treat like the /tmp P1/P2 leg)
  + _discover_staging_issue_numbers + wire into clean_terminal_download_caches + the data-disk pass in main()
  + watcher _data_disk_top_caches: bounded du -s top-N over data_disk_path()/$USER/* after the repquota-None fallback
confidence: high
related_task: n/a (chat-mode deep dive, 2026-08-05)
<!-- /workflow-fix-candidate -->
