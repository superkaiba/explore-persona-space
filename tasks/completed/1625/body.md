---
title: 'daily-fix: inline rounds both mapping arms + PNG check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:80dd6613e4a0
- daily-auto-filed
created_at: '2026-07-23T07:01:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): the plannerless inline
  path shipped a context-only mapping (user caught the missing prefix arm, #779) and
  presented an empty figure 3x (#1112) — neither the both-arms rule nor any figure
  sanity check is restated on the inline path'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). Two gaps on the PLANNERLESS inline free-analysis path, both of which cost user corrections today: (a) the 07-14 #779 inline pre-image round shipped a CONTEXT-only representation mapping — the user had to catch the missing prefix arm ("is this mapping on the prefix vector or context vector" → "run a prefix-based twin inline"), a recurrence of the #958 one-arm class the standing both-arms rule (Thomas 2026-07-03) exists to prevent; (b) the #1112 inline round presented an EMPTY rendered figure — Thomas pasted the screenshot 3× while the extraction bug was found (the PNG-load check exists only in the critic loop, not inline rounds).

## Goal

The inline free-analysis path (CLAUDE.md § "User-chat inline free analysis" + SKILL.md Step 9a-ter) states two duties the planner+critic stack would otherwise enforce: (1) an inline round that computes a representation-mapping read names BOTH arms (prefix-based + context-based) — or the explicit deviation — in its dispatch-time `epm:progress` note; (2) Read the rendered PNG and confirm non-empty axes/series before presenting or committing any inline-round figure.

## Workflow gap

- **Bug observed:** (a) fdf687f2 (#779), 2026-07-22T20:04–20:05Z user catch → a full extra inline round (prefix twin + overnight null); (b) 24ae2158 (#1112), 2026-07-23T00:17–00:29Z — 3 user pastes of an empty figure, two wrong first diagnoses before the extraction bug was found.
- **Why it is a workflow gap:** the inline carve-out deliberately skips the planner+critic stack; its compute-character statement covers COMPUTE duties but restates none of the measurement-design duties (both-arms) nor any figure sanity check. Both standing rules exist globally; the inline path is where they keep being skipped.
- **Confidence:** high.
- verified-at-filing: `grep -n 'prefix-based and context-based\|both arms' .claude/skills/issue/SKILL.md` → 0 hits in the 9a-ter section (absence claim); `grep -c 'rendered PNG' .claude/skills/issue/SKILL.md` → 0 (absence claim); the CLAUDE.md carve-out paragraph likewise carries neither duty (read in full at filing), 2026-07-23 UTC.

## Proposed change (refine in planning)

One clause each in the CLAUDE.md inline carve-out AND SKILL.md 9a-ter: (1) mapping rounds state both arms or the stated deviation in the dispatch note (mirror of the Critical Rules both-arms bullet); (2) figures are eyeballed via Read (non-empty axes/series) before presentation/commit.

## Scope / surfaces

- Primary targets: `CLAUDE.md` (§ User-chat inline free analysis), `.claude/skills/issue/SKILL.md` (Step 9a-ter).

## Constraints / invariants

- No new gates; both are statement/check duties inside the existing carve-out contract. Recursion guard applies.

## Provenance

- fingerprint: 80dd6613e4a0

- workflow_fix_target: .claude/skills/issue/SKILL.md
