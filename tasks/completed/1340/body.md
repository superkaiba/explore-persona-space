---
title: 'workflow-fix: gotchas bullet — single-arm smokes blind to per-arm seams'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ac8fb68b9f65
created_at: '2026-07-15T10:02:05Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1090 fu5 crash: panel-disjointness
  assert at a ModelOrganism site the single-arm smoke never reached'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate: yes
failure-lesson raised on task #1090 fu5 (emitting agent: experiment-implementer).

## Goal

add a gotchas bullet: per-arm-class smoke coverage — a driver extended to a new source-context class crashes the #527/#538 panel-disjointness assert at ModelOrganism sites a single-arm smoke never reaches; thread the source-filtered panel and smoke one run per arm class

## Workflow gap

- **Bug observed:** #1090 fu5: all 3 imp-bare arms trained to completion then died rc=2 at the ladder entry on the panel-disjointness AssertionError; the fu5 smoke default covered only the formatting arm
- **Why it is a workflow gap:** the tiny-real smoke standard (gotchas "Mock-seam smokes" entry + Step 6d.0-bis) does not state that smoke coverage must span ARM CLASSES — a per-arm seam (context / negative-panel / organism assembly) is invisible to a single-arm smoke, and this cost a full 4xA100 GCE cycle. The library refusal itself is correct and load-bearing (contrastive-negatives disjointness invariant).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "one run per arm class\|arm class\|panel_name_for" .claude/rules/gotchas.md` → 0 hits for the proposed guidance in the named target (absence-of-guard claim — the 0-hit result IS the evidence); the invariant itself lives in .claude/rules/contrastive-negatives.md § Disjointness invariant (present) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md, near the "Mock-seam smokes surface production shape bugs" entry:
+ - **Single-arm smokes are blind to per-arm seams.** A driver extended to a NEW
+   source-context class (e.g. bare/"default") can pass every existing smoke and
+   still crash at ModelOrganism construction: the #527/#538 panel-disjointness
+   assert refuses a source content-identical to a default-panel member. Thread
+   the source-filtered panel (fu3w.panel_name_for-style) at every organism site
+   the new class reaches, and make the smoke default cover ONE RUN PER ARM
+   CLASS, not one run total (#1090 fu5, a full GCE cycle lost).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Consider a one-line cross-ref from Step 6d.0-bis (tiny-real standard) in
  .claude/skills/issue/SKILL.md if the planner judges it in-scope.

## Constraints / invariants

- Workflow-surface only; workflow_lint passes (incl. --check-lessons-index if a
  rules file is added — this edits an existing rule, no index change expected).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: ac8fb68b9f65

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: issue1090_fu4.py --phase run (Tier-1 ladder entry, fu5 bare-context arms)
lesson: A driver re-parametrized for a NEW source-context class (bare/"default") can pass every existing smoke yet crash at ModelOrganism construction, because the #527/#538 panel-disjointness invariant refuses a source that is content-identical to a default-panel member — thread the source-filtered panel (fu3w.panel_name_for) at EVERY ModelOrganism site a new context class reaches, and make the smoke default cover one run per ARM CLASS, not one run total: a per-arm seam (context/panel/organism assembly) is invisible to a single-arm smoke.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
