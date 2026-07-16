---
title: 'daily-fix: paper-plots interim-figure defaults'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a8a55c08f9ed
- daily-auto-filed
created_at: '2026-07-16T07:21:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Cluster of figure corrections
  in one day: 5 user fixes (hatching, per-bar n, guess markers, negative y-axis, rotation
  arm); CV caption omitted fold structure (n=4998 misread); interim figures omitted
  mapping arm / disaggregation'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Strengthen /paper-plots' interim/user-facing figure defaults: no hatching / per-bar n annotations / estimated-value markers unless asked; CV figures label fold structure explicitly; interim figures state the mapping arm and per-behavior disaggregation in the caption/setup line.

## Workflow gap

- **Bug observed:** a cluster of figure corrections in one day: 5 separate user fixes in 09f28ede 05:12-08:15 (hatching, per-bar n labels, "guess" markers, unexplained negative y-axis, unwanted rotation arm); "n=4998" misread as train size because a CV figure's caption omitted fold structure (b7150177 22:06); interim figures shipped without stating the mapping arm or per-behavior disaggregation (28d0874a 06:36-06:44).
- **Why it is a workflow gap:** the skill's no-editorial-annotations rule covers arrows/effect-size labels but not hatching / per-bar n / estimated-value markers, and it has no CV-caption or arm-disclosure defaults — so each interim figure relearns Thomas's preferences via correction rounds.
- **Severity:** medium
- verified-at-filing: `grep -n 'hatch\|interim\|fold' .claude/skills/paper-plots/SKILL.md` → 0 relevant hits (no hatching / interim-figure / fold-structure guidance; "annotation" hits at L166-198 cover the existing no-arrows/no-effect-size-labels rule only) — proposed defaults absent; plain-English label rule present at L121 (adjacent, not covering these cases) (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend `.claude/skills/paper-plots/SKILL.md`'s figure-hygiene section (anchor: the "Figures present DATA" block at L166): (a) interim/user-facing default = no hatching, no per-bar n annotations, no estimated/"guess" value markers, no gratuitous label rotation unless asked; (b) any cross-validated metric figure labels its fold structure explicitly in the caption (n_contexts, k folds, held-out size) so aggregate n is never misread as train size; (c) interim figures state the mapping arm (prefix vs context) + per-behavior disaggregation in the caption/setup line — the figure-side mirror of the CLAUDE.md ad-hoc-summaries provenance rule.

## Scope / surfaces

- Primary target: `.claude/skills/paper-plots/SKILL.md` (anchor: L166 figure-hygiene block; L121 label rule)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: a8a55c08f9ed

- workflow_fix_target: .claude/skills/paper-plots/SKILL.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 09f28ede 05:12-08:15 (batch 08 P5); b7150177 22:06 (batch 01 P3); 28d0874a 06:36-06:44 (batch 04 P5).
