---
title: 'workflow-fix: Disk-hygiene lead bullet — bind status runtime-checked, not
  asserted'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e073c6acbbb1
created_at: '2026-08-05T21:03:27Z'
has_clean_result: false
origin_prompt: 'Prose workflow-fix follow-up from task #2091 methodology critic: CLAUDE.md
  Disk hygiene lead bullet states the #681 bind as accomplished while findmnt shows
  it pending; stale disk sizes; planners copy the claim into plan section-9 disk rows
  (the #2091 v2/v3 false staging premise).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose workflow-fix follow-up raised on task #2091 (emitting agent: methodology critic, Phase 2 plan review).

## Goal

Rewrite the CLAUDE.md § Disk hygiene lead bullet (and its background-automation.md twin) so the #681 worktree-bind status is presented as runtime-checked (findmnt) rather than asserted as accomplished, and refresh the stale disk sizes.

## Workflow gap

- **Bug observed:** CLAUDE.md § Disk hygiene lead bullet states the #681 relocation as accomplished ("is relocated off the 485 GB boot disk onto a 512 GB pd-balanced data disk ... bind-mounted back onto .claude/worktrees so every consumer resolves the SAME path") while the live VM shows the cutover pending (findmnt --mountpoint <repo>/.claude/worktrees EMPTY, verified 2026-08-05; .claude/worktrees and repo-root data/ are plain dirs on /dev/root) and the stated sizes are stale (/ is 945 GB at 98% with 21 GB free, /mnt/eps-data is 1007 GB at 96% with 47 GB free — not 485/512 GB). Task #2091's plan v2/v3 copied the bullet's opening claim into its §9 disk row ("resolves to /mnt/eps-data via the #681 worktree bind — never /") and booked a ~42 GB stage onto a filesystem it would never reach; the Phase-2 methodology critic caught it only via a live findmnt/df probe.
- **Why it is a workflow gap:** the bullet's opening is what planners copy; its own cutover-pending paragraph sits ~1,500 words deep in the same bullet and even names new_worktree.sh's WARN as "the live source of truth", but nothing tells a PLANNER to runtime-probe before citing the bind in a §9 disk row. Any future plan citing the bind will repeat #2091's false premise.
- **Confidence (emitter):** high
- verified-at-filing: `grep -nE "bind.?mounted" CLAUDE.md` → 1 hit (CLAUDE.md:174, the lead bullet); `grep -n "bind-mounted back onto" .claude/rules/background-automation.md` → 1 hit (line 1659); live probes `findmnt --mountpoint <repo>/.claude/worktrees` → empty, `df -h / /mnt/eps-data` → 945G/98%/21G-free + 1007G/96%/47G-free (2026-08-05)

## Proposed change (candidate diff sketch — refine in planning)

```
- **`.claude/worktrees/` lives on a dedicated GCP data disk with per-task ext4 quotas (#681).** The heavy active-task footprint ... is relocated off the 485 GB boot disk onto a **512 GB `pd-balanced` data disk mounted at `/mnt/eps-data`** ..., **bind-mounted** back onto `.claude/worktrees` so every consumer resolves the SAME path transparently ...
+ **`.claude/worktrees/` is DESIGNED to live on the dedicated GCP data disk (`/mnt/eps-data`) with per-task ext4 quotas (#681) — the bind cutover is a RUNTIME-CHECKED state, not an accomplished fact.** Before citing the bind (or `/mnt/eps-data` headroom) in ANY plan §9 disk row or staging decision, probe live state: `findmnt --mountpoint <repo>/.claude/worktrees` (empty = cutover still pending; worktree `data/` then lands on `/`) + `df -P` on the resolved filesystem with free headroom ≥ ~1.5× the projected bytes. Disk sizes drift — never quote this file's figures as current.
  (+ the same opening rewrite at .claude/rules/background-automation.md:1659; keep the cutover-recipe tail of both bullets unchanged)
```

## Scope / surfaces

- Primary target: `CLAUDE.md`, `.claude/rules/background-automation.md`
- Grep the workflow surface for the pattern before editing (`grep -rn 'bind.mounted' CLAUDE.md .claude/rules/ .claude/skills/ scripts/ --include='*.md'` at the MAIN checkout, excluding worktree copies) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; CLAUDE.md and background-automation.md stay consistent with each other.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: CLAUDE.md,.claude/rules/background-automation.md
- fingerprint: e073c6acbbb1

Verbatim surfaced prose (methodology critic, task #2091 Phase 2): "CLAUDE.md § Disk hygiene's lead bullet states the #681 relocation as accomplished ('bind-mounted back onto .claude/worktrees... per-task ext4 quotas') while the live VM shows the cutover still pending (findmnt empty, .claude/worktrees on /dev/root) and the stated disk sizes are stale (/mnt/eps-data is now 1007 GB at 96%, / is 945 GB at 98%, not 485/512 GB). That lead bullet is what this plan's wrong §9 mount claim was copied from; any future plan citing the bind will repeat it. Suggested change: rewrite the lead bullet to state the bind status is runtime-checked (findmnt), not assumed, and point planners at a live df -P/findmnt probe for any §9 row citing /mnt/eps-data."
