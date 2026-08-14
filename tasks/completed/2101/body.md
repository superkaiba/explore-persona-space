---
title: 'workflow-fix: no-lost-row check on agent-memory index alignments'
kind: infra
tags:
- wf-fix
- wf-fix-fp:35a969c66772
created_at: '2026-08-05T21:23:31Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from task #2093 epm:results v1 (d): a standing no-lost-row
  check on agent-memory index ALIGNMENTS (the class that orphaned 7 reconciler rows
  via commits 038a42ec6c/0aaf39acac) would prevent recurrence; #2093''s plan fold
  + criterion-7 assert covers only its own merge.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #2093 (emitting agent: implementer, unit C of the pre-split curation round; marker `epm:results` v1 §(d)).

## Goal

Any alignment/import of an agent-memory MEMORY.md index to another copy runs a row-set no-lost-row check (comm -13 local vs imported; rows unique to the local copy are preserved or explicitly dispositioned) — closing the manual-alignment class the #1972 Step 5a dirt/branch-side arms do not cover.

## Workflow gap

- **Bug observed:** 7 reconciler memory-index rows were silently dropped by manual stale-copy alignments to main's #1891-curated index (commits 038a42ec6c, 0aaf39acac); their bodies became index-unreferenced with no disposition record.
- **Why it is a workflow gap:** the Step 5a spec-freshness sync now carries #1972's uncommitted-dirt + branch-side-edit protections, but those bind ONLY the sync block; a session hand-aligning a stale agent-memory index to another copy (`git checkout <ref> -- .claude/agent-memory/...`-style imports, as both cited commits did) has no row-loss discipline at all — rows unique to the local copy vanish with no record, exactly the loss class the memory system exists to prevent.
- **Confidence (emitter):** medium
- verified-at-filing: `git log -1 038a42ec6c / 0aaf39acac` → both resolve (2026-07-31, "align 3 more stale agent-memory indexes…" / "import origin/main's #1891-curated agent-memory indexes…"); the 7 orphaned bodies are enumerated with class-(b) justifications in `docs/agent_memory_curation_2093.md` § reconciler § Intentionally-unreferenced (branch issue-2093, commit 059c7f2181). The gap claim is behavioral (absence of an operation-time discipline for manual alignments), not grep-refutable: n/a — absence-of-guard claim; the #1972 arms were read in `.claude/skills/issue/SKILL.md` § Step 5a and bind the sync block only.

## Proposed change (candidate diff sketch — refine in planning)

(synthesized from prose follow-up)
+ SKILL.md / gotchas.md: before any alignment of `.claude/agent-memory/*/MEMORY.md`
+ to another copy (manual import, stale-copy alignment, or any checkout of another
+ ref's copy over the local one), run per index:
+   comm -13 <(sort <aligned/incoming copy>) <(sort <local copy>)
+ Rows unique to the LOCAL copy are re-appended to the aligned result or
+ explicitly dispositioned in the commit message — never silently dropped.
+ (unverified hypothesis — verify at plan time: whether a workflow_lint or hook
+ leg can mechanize any part of this, given the check is operation-time, not
+ tree-state.)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, .claude/rules/gotchas.md`
- Grep the workflow surface for existing alignment prose before editing
  (`grep -rln 'agent-memory' .claude/skills/issue/SKILL.md .claude/rules/`) and update every hit that prescribes or permits an index alignment; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- Related but NOT this task's scope: restoring the 7 orphaned reconciler rows (~+1.6 KB) — #2093's manifest flags that for a future curation round.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/rules/gotchas.md
- fingerprint: 35a969c66772

Verbatim surfaced prose (task #2093 `epm:results` v1 §(d)): "Manifest `## reconciler` § Intentionally-unreferenced bodies, class (b): the 7 bodies whose index rows were dropped by branch-copy alignments to main's #1891-curated index — confirm the 'flag for a future round, do not restore now' call. Prose follow-up for the orchestrator: a standing no-lost-row check on agent-memory index ALIGNMENTS (the class that orphaned these 7) would prevent recurrence; this task's plan §3 fold + criterion-7 assert covers only THIS merge."
