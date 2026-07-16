---
title: 'daily-fix: fix 4 workflow_lint FAILs on main (hub-verify)'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-15T06:51:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): bare Hub verify calls (#920
  class) at issue1092_corpus_dashboard.py:144, issue825_map_alignment.py:657, issue952_china_topup_gpu.py:530
  plus an unguarded upload_folder at issue952_china_topup_gpu.py:521 make workflow_lint
  no-flags FAIL (4 errors) on main'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 run: routed from a prose follow-up parked under the recursion guard on task #1311 (#920 bare-Hub-verify class), widened by a fresh no-flags `workflow_lint` run. These are experiment-code fixes (NOT workflow surface), filed on the /daily route-2 channel so main's lint gate goes green again — the same 4 failures surfaced as pre-existing debt in #1311's Step 9c gate and will resurface in every code-change task's gate until fixed.

## Goal

route the bare list_repo_tree calls through orchestrate.hub scoped/retried helpers (or add HUB_VERIFY_RETRY_EXEMPT waivers) and guard the issue952 upload_folder with assert_hub_dir_filecounts

## Bug

- **Observed:** bare Hub verify calls (#920 class) at issue1092_corpus_dashboard.py:144, issue825_map_alignment.py:657, issue952_china_topup_gpu.py:530 plus an unguarded upload_folder at issue952_china_topup_gpu.py:521 make workflow_lint no-flags FAIL (4 errors) on main
- verified-at-filing: `uv run python scripts/workflow_lint.py` (no flags) -> `workflow_lint: FAIL (4 error(s))` naming exactly these sites; per-target hits: issue1092_corpus_dashboard.py:144 (1), issue825_map_alignment.py:657 (1, plus the import at :649), issue952_china_topup_gpu.py:521 + :530 (2) (2026-07-15). Retraction re-check on #1311 events after the park ts (2026-07-15T01:06:35Z): none — #1311 completed/merged without touching these out-of-scope-for-it sites.

## Proposed change

Per the lint's own remediation text: route each bare `list_repo_tree` through `explore_persona_space.orchestrate.hub` (`list_hf_files_under_path` / `verify_repo_paths_uploaded` / `list_repo_files_complete`) or add a `# HUB_VERIFY_RETRY_EXEMPT: <reason>` waiver where the raw call is genuinely correct; add `assert_hub_dir_filecounts` (or route through `hub._upload_folder_filtered`) before the issue952 `upload_folder` at :521, outside any transient-retry wrapper.

## Constraints

- Pure hygiene: no behavior change to the analyses these scripts implement; re-run `workflow_lint` no-flags and the 3 pinned tests that currently fail on main to confirm green.
