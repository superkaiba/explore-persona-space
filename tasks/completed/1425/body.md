---
title: 'workflow-fix: gotchas.md download-side hf-xet xet_get wedge entry'
kind: infra
tags:
- wf-fix
- wf-fix-fp:76fe80f3f563
created_at: '2026-07-16T13:11:16Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1345: hf-xet download wedge
  (native xet_get hang, zero conns) — add gotchas.md entry with diagnosis differential
  + HF_XET_DISABLE=1 kill+replay recovery'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha-candidate raised on task #1345 (emitting agent: issue-orchestrator, from a confirmed py-spy root-cause dump).

## Goal

Add a download-side sibling of the Hub upload-wedge guidance to `.claude/rules/gotchas.md`: the hf-xet native `xet_get` hang signature (du frozen, ss empty, py-spy frame) + kill-and-replay with `HF_XET_DISABLE=1`.

## Workflow gap

- **Bug observed:** an 87 GB GCE turnstore prefetch wedged at 98.6% staged inside the native `xet_get` call (huggingface_hub file_download.py:633 → hf-xet Rust client) with NO exception and ZERO established TCP connections; the per-file retry wrapper could not fire (the native call never returns) and TCP-kill unwedging was impossible (no Python-visible socket). ~45 min of zero byte-flow before a py-spy dump pinned it.
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` documents the HF Hub 429 quota class, the snapshot_download full-tree-enumeration class, and (via upload-policy.md) the UPLOAD-side wedge ladder — but NOT the download-side xet native hang: its diagnosis differential (du-frozen + ss-empty + py-spy xet_get frame) and its recovery (kill + replay with `HF_XET_DISABLE=1` inline) had to be re-derived live.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -iE "xet.*wedge|wedge.*xet|xet_get|download.*wedge" .claude/rules/gotchas.md` → 0 hits for the download-side xet wedge class in gotchas.md (the file's Hub entries at lines 270-271 cover the 429 quota + tree-enumeration classes only; `grep -ic xet` shows xet mentions but none document the native-hang signature); repo-wide `grep -rln xet_get .claude/ CLAUDE.md scripts/` → hits only in `.claude/agent-memory/experimenter/feedback_hf_xet_download_wedge_kill_replay.md` (the long-form memory committed 3bce8f1305, 2026-07-16) (2026-07-16)

## Proposed change (candidate diff sketch — refine in planning)

+ In .claude/rules/gotchas.md, next to the HF Hub 429 / snapshot_download entries:
+ - **hf-xet DOWNLOAD wedge — native xet_get hang with zero connections.** Bulk downloads
+   via the xet path can hang forever inside native xet_get with no exception; per-file
+   retry wrappers never fire. Diagnose: du -sb frozen across 2+ probes ~10 min apart +
+   ss -tnp empty for the pid + py-spy dump (uv tool install py-spy) showing a worker in
+   huggingface_hub file_download.py xet_get. Recover: kill + replay the phase with
+   HF_XET_DISABLE=1 threaded inline on the workload command (the real switch, #1195);
+   the plain resolve/hf_transfer path is resumable. Sibling: upload-policy.md §931 ladder.
+   Long-form twin: .claude/agent-memory/experimenter/feedback_hf_xet_download_wedge_kill_replay.md
+   (Incident #1345 assistant-named-story round, 2026-07-16.)

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'xet' .claude/ CLAUDE.md scripts/`) and update every hit that should cross-reference; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; gotchas.md stays consistent with upload-policy.md's wedge ladder (link, don't duplicate).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 76fe80f3f563

<!-- epm:failure-lesson v1 -->
failure_class: infra
phase: prefetch_reuse (HF Hub bulk download, GCE lane)
lesson: The hf-xet DOWNLOAD path can wedge inside the native xet_get call with ZERO established TCP connections and no exception — a per-file retry wrapper never fires because the native call never returns, and TCP-kill unwedging is impossible (no Python-visible socket). Diagnose with: du -sb frozen across 2+ probes ~10 min apart + ss -tnp empty for the pid + py-spy dump showing a worker in huggingface_hub file_download.py xet_get. Recovery: kill + replay with HF_XET_DISABLE=1 threaded inline on the workload command (the real switch; GCP/SLURM allowlists forward it, #1195) — the plain resolve/hf_transfer path is resumable and wedge-free.
generalizes: yes
owning_agent: experimenter
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
