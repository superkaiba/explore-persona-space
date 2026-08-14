---
title: 'daily-fix: trigger-dense first-spawn pre-qualification by ta'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3e0d3fe428ff
- daily-auto-filed
- trigger-dense
created_at: '2026-08-02T07:12:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): >=7 refusal-killed spawns
  across >=5 sessions in one day (+2 sonnet thrashes): implementer first-spawns on
  known trigger-dense targets (#1769 steering body, guard scripts, implant vocab)
  walked the ladder through 1-2 kills before content mitigation (~1.5h wall on #1979
  unit 1; 3 dead spawns on #1977). Rung (b2-content) requires prior kill evidence;
  the first-pass brief duties bind review-role brie'
workflow: v1
---
# daily-fix: trigger-dense first-spawn pre-qualification by target class

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C8 (miners 4, 3, 8, 6, 5; sessions 8fc069db (#1947), 0dcef1c6 (#1879), 24f7b592 (#1979), 0dbe2031 (#1977), 11d2daa4 (#1984)).

## Goal
Pre-qualify known trigger-dense TARGET CLASSES at spawn-composition time so the FIRST spawn of ANY subagent role — implementers included — on such a target starts at the content-mitigated brief (file-reference/digest form), instead of reaching mitigation only after refusal kills; today CLAUDE.md's rung (b2-content) keys on prior kill evidence, and trigger-dense-review.md's § First-pass briefs binds only review-role briefs.

## Workflow gap
- **Bug observed:** ≥7 refusal-killed spawns across ≥5 sessions in one day, plus 2 sonnet autocompact thrashes, all recovered per the existing ladder but at real cost: #1979's implementer killed twice ("29 + 33 tool calls, zero durable writes both times") then a sonnet respawn thrashed at 4 tool calls (~1.5h wall on unit 1); #1977 lost 3 reviewer spawns on a guard-surface target (~17 min); #1879's implementer killed at 38 tool calls mid-write; #1947 lost 2 spawns. Recurring pattern: FIRST-pass briefs pointed subagents at known trigger-dense content (the #1769 steering body, guard scripts, implant/marker vocabulary) — content-mitigation was applied only AFTER the kills.
- **Why it is a workflow gap:** the recognition heuristic and first-pass duties exist, but (a) the CLAUDE.md ladder's content-mitigated fast path is gated on kill EVIDENCE, and (b) the first-pass composition duties fire for "fact-check / critique / plan-review / first code-review" briefs, not implementer briefs generally — so implementer first-spawns on recognized target classes predictably walk the ladder through 1-2 kills first.
- **Confidence:** medium (kill counts miner-probed on #1979-group transcripts — `python filter counting 'violate our Usage Policy' tool_result firings`; brief-composition attribution miner-inferred).
- verified-at-filing: `grep -n 'b2-content' CLAUDE.md` → 1 hit; the rung reads verbatim "(b2-content) when kill evidence already isolates CONTENT as the trigger — a sibling unit with the same model and brief shape ran clean while the killed unit differs only in its target content (#1774 ...) — skip the same-model rung-(b) rephrase: the FIRST retry carries content-side mitigation (pass the triggering content by file reference / digest per `.claude/rules/trigger-dense-review.md`) plus the per-subagent model pin" — i.e. mitigation-first requires prior kill evidence. `sed -n '150,190p' .claude/rules/trigger-dense-review.md` → § First-pass briefs (#1503) "Fires for the ORCHESTRATOR composing any FIRST-PASS subagent brief whose TARGET files include a trigger-dense artifact ... — the Phase-1.5 fact-checker brief, the Phase-2 critic and consistency-checker briefs, a plan-review or first code-review brief" — implementer briefs are covered only via the datagen sibling (#1748); the recognition heuristic (:29-56) already lists guard/hook scripts, banks/corpora, judge outputs, and steering-application surfaces (763b59080f, #1797 landed steering-surface recognition), so the CLASSES are recognized — the binding to implementer first-spawns and the evidence-free fast path are what is missing. `git log --oneline --since='7 days ago' -- .claude/rules/trigger-dense-review.md` → 1 commit (6ea044af6d, judge-monitoring clause) — no first-spawn pre-qualification (2026-08-02).

## Proposed change (refine in planning)
1. `.claude/rules/trigger-dense-review.md` § First-pass briefs: widen the firing clause from the review-role brief list to ANY first-pass subagent brief (implementer, experimenter, analyzer, reviewer) whose TARGET files (or required reading, e.g. a steering-content task body) match the recognition heuristic — duties 1-5 unchanged.
2. CLAUDE.md refusal ladder: add a pre-qualification sentence — when the spawn's target matches a recognized trigger-dense class (per the trigger-dense-review.md heuristic), the FIRST brief starts content-mitigated (file-reference/digest per that rule) with no kill evidence required; rung (b2-content) stays the post-kill fast path for UNrecognized content.
3. Keep additive: the recognition heuristic list itself is the single source of target classes — do not fork a second list in CLAUDE.md (pointer only, always-on budget).

## Scope / surfaces
- Primary target: `.claude/rules/trigger-dense-review.md, CLAUDE.md`
- Quote-consistency: `grep -n 'b2-content\|First-pass' CLAUDE.md .claude/rules/trigger-dense-review.md .claude/skills/adversarial-planner*/SKILL.md` and update every cross-reference; update the rule's frontmatter description + LESSONS.md row if the trigger wording changes.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Never rename loaded terms inside ARTIFACTS (code/plans/task bodies) — brief/prompt text only, per the existing rung (e) contract.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 3e0d3fe428ff
- workflow_fix_target: .claude/rules/trigger-dense-review.md, CLAUDE.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C8.
