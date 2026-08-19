---
title: 'workflow-fix: shared-root uncommitted-file reversion race (pre-commit stash/restore
  under concurrency)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7f492a5faf66
created_at: '2026-08-02T07:20:24Z'
has_clean_result: false
origin_prompt: 'map-augment-1768 round report on #1768, 2026-08-02: five red-handed
  reversions of uncommitted edits/deletions in the shared root during concurrent commits;
  verbatim in body Provenance'
workflow: v1
---
## Overview / Motivation
Auto-filed from a prose follow-up surfaced by an inline-round subagent on task #1768 (map-augmentation round, 2026-08-02) — the most consequential process finding of the week: uncommitted work in the shared repo root is silently reverted under concurrent activity.
## Goal
Close (or durably mitigate + document) the shared-repo-root uncommitted-file reversion race: uncommitted edits and deletions revert to committed content within seconds during concurrent sessions' commits, silently losing work and enabling internally inconsistent commits.
## Workflow gap
- **Bug observed:** uncommitted edits and deletions in the shared repo root were reverted to committed content within seconds, five separate times in one round (files deleted 23:43:3x restored by 23:44:44 with committed content; both scripts and JSONs), causing an internally inconsistent commit (new artifacts paired with pre-patch scripts) that burned a clean-result-critic round.
- **Why it is a workflow gap:** the fleet runs many concurrent committers on one shared root by design (CLAUDE.md § Concurrent repo-root committers), and the documented discipline (explicit-path staging, index.lock retry, sync_repo_root) does not protect UNCOMMITTED working-tree state from the observed reversion mechanism.
- **Candidate mechanism:** `unverified hypothesis — verify at plan time:` the pre-commit framework's unstaged-file stash/restore cycle ("Stashing unstaged files… / Restored changes from…", observed directly in hook output) firing around CONCURRENT sessions' commits — a stash snapshot taken at T restored at T+x clobbers/resurrects files changed in between; a concurrent session's repo-root git op is the alternative candidate. Timeline evidence is the emitting agent's red-handed observation (recorded in its round report + #1768 markers).
- **Confidence (emitter):** high on the phenomenon, medium on the mechanism
- verified-at-filing: `.pre-commit-config.yaml` present at repo root (the framework whose stash/restore lines the agent captured); `grep -c "Concurrent repo-root committers" CLAUDE.md` → 1 (the guidance section lacking any uncommitted-state warning) (2026-08-02). Phenomenon is not grep-verifiable — labeled as agent-observed with the five-strike timeline.
## Proposed change (candidate diff sketch — refine in planning)
diff_sketch: |
  Two halves, planner scopes: (1) GUIDANCE (CLAUDE.md § Concurrent repo-root committers + SKILL.md 9a-ter):
  inline rounds NEVER generate/mutate artifacts through the repo-root working tree during concurrent
  activity — generate off-root (/mnt/eps-data staging), copy-in + lint + commit inside one short window;
  deletions of committed files only via committed removals. (2) MECHANISM: reproduce the stash/restore
  interleaving (two concurrent commits, one with unstaged files), then either scope pre-commit hooks to
  --no-stash semantics / staged-only operation, or serialize repo-root commits via the existing
  ~/.task-workflow flock so a stash/restore window cannot interleave another session's tree mutations.
## Scope / surfaces
- Primary targets: `CLAUDE.md`, `.pre-commit-config.yaml` (+ hook wrappers under `.claude/hooks/` if the serialize-or-descope fix lands there).
- Reproduction first — the fix must not weaken the gitleaks/secret gates (fail-closed hooks stay).
## Constraints / invariants
- Workflow-surface only; hooks stay fail-closed; no weakening of the merge-scoped secret scan.
- Recursion guard applies (EPM_WORKFLOW_FIX_SESSION=1 + workflow_fix_target line).
## Provenance
- workflow_fix_target: CLAUDE.md,.pre-commit-config.yaml
- fingerprint: 7f492a5faf66
Verbatim surfaced prose: "repo-resident UNCOMMITTED files in the shared root are reverted to their committed content within SECONDS... It hit five times, on both scripts and JSONs. Deletions of committed files also do not stick. Candidate mechanism observed directly in pre-commit output: it stashes unstaged files around every commit's hooks and restores them after... a concurrent session's repo-root git op is the other candidate. Not proven, and worth a workflow-fix look — sibling sessions can silently lose uncommitted work in this root."
