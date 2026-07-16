---
title: 'daily-fix: inline rounds route HF downloads off /'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b81f5d9de5a3
- daily-auto-filed
created_at: '2026-07-16T07:20:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): An inline free-analysis
  subagent materialized a 14 GB HF download onto / -> root ~0 free, Bash output lost
  (ENOSPC); recovery hit mkdir /mnt/eps-data/tmp Permission denied'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Add a disk-routing line to the inline/user-chat free-analysis compute-character pre-launch statement: any multi-GB HF download in an inline round stages under the issue worktree's data dir on the data disk from the start, never `/` or `/tmp`.

## Workflow gap

- **Bug observed:** an inline free-analysis subagent materialized a 14 GB HF tensor download onto `/` → root hit ~0 free, orchestrator Bash output lost (ENOSPC); the first recovery attempt hit `mkdir /mnt/eps-data/tmp: Permission denied` (b7150177, #823, 22:52-23:01Z).
- **Why it is a workflow gap:** the 9a-ter compute-character pre-launch statement covers ops arithmetic / wall-time / batched helpers but says nothing about WHERE download bytes land, so inline rounds default to cwd/`/` on the shared 188 GB root disk.
- **Severity:** high
- verified-at-filing: `awk '/Compute-character pre-launch statement \(REQUIRED/,/^###/' .claude/skills/issue/SKILL.md | grep -c 'disk\|staging\|download'` → 1 hit, and it is unrelated ("downloaded from outside the existing eval_results/ / HF data" — an artifact-existence clause, not disk routing) — proposed disk-routing line absent from the statement (anchors: SKILL.md L6312 statement heading, L5729, L7713-7714 cross-refs); CLAUDE.md user-chat inline carve-out likewise carries no download-staging line (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend the `.claude/skills/issue/SKILL.md` Step 9a-ter § Compute-character pre-launch statement (L6312) — and the CLAUDE.md "User-chat inline free analysis" carve-out's copy of the statement requirements — with a disk-routing element: any multi-GB HF download or tensor materialization in an inline round names its staging path UP FRONT and stages under the issue worktree's `data/issue_<N>/` dir (bind-mounted to the data disk per #681), never `/`, `/tmp`, or a fresh top-level `/mnt/eps-data/<dir>` (root-owned — see the companion gotchas entry m23).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (§ 9a-ter Compute-character pre-launch statement, L6312)
- Secondary: `CLAUDE.md` (user-chat inline free-analysis carve-out's compute-character clause)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b81f5d9de5a3

- workflow_fix_target: .claude/skills/issue/SKILL.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: b7150177 (#823) 22:52-23:01Z (batch 01 P4).
