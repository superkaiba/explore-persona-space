---
title: 'workflow-fix: wire the both-arms mapping rule into Statistics lens item 15'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a8a502c50011
created_at: '2026-08-05T19:35:09Z'
has_clean_result: false
origin_prompt: 'user chat (2026-08-05): ''waht doe the project wide rule say exactly?''
  — quoting the CLAUDE.md both-arms prefix/context mapping rule surfaced that its
  closing enforcement sentence (''The planner names both arms in §4 Design; the critic
  REVISEs a plan that silently drops one'') names files that do not implement it'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gap the orchestrator observed
during interactive planning of a context→answer mapping issue (2026-08-05). The user
asked what the standing both-arms mapping rule says exactly; reading it surfaced that
its own stated enforcement does not exist in the files it names.

## Goal

Add the prefix-arm/context-arm both-arms requirement as a clause of the existing
representation-mapping registration — Statistics lens item 15 + the planner §6
registration block + `statistics-critic` item 15 + `experiment-guidelines` guideline
11 — using the same trigger and the same stated-deviation escape as its identity+bias
/ kNN / pooling-convention siblings.

## Workflow gap

- **Bug observed:** the CLAUDE.md standing rule "Prefix mapping AND context mapping —
  run BOTH in every experiment" ends with *"The planner names both arms in §4 Design;
  the critic REVISEs a plan that silently drops one without a stated exemption."*
  Neither clause is implemented: the mapping arms appear nowhere in `planner.md`,
  `critic.md`, `critic-lens-reference.md`, `statistics-critic.md`, or
  `experiment-guidelines.md`. The rule is carried only by the always-on CLAUDE.md
  bullet and by `.claude/skills/issue/SKILL.md`'s inline-round duty.
- **Why it is a workflow gap:** three structurally identical representation-mapping
  disclosures ARE wired into those exact surfaces — the identity+bias baseline and the
  kNN-retrieval read (#1604, merge `4c4e79688b`) and the pooling-convention row (#1746,
  merge `4238694ae9`) — all under Statistics lens item 15, which fires on precisely the
  trigger this rule needs ("If the plan FITS a map between activation summaries
  (context→answer, prefix→context, cross-model / cross-framing reparameterization — any
  v_X→v_Y predictor)"). The both-arms rule is the fourth sibling of that set and is the
  only one absent, so a critic working item 15 top-to-bottom will check pooling and the
  two baselines and never check the arms. Two shipped one-arm rounds are on record and
  both were caught by the user rather than the pipeline: #958 (2026-07-04, a one-arm
  capture, which is why the routing-section capture-time clause was added) and #779's
  2026-07-14 inline pre-image round (context-only).
- **Confidence (emitter):** medium. There is a competing reading under which the
  CLAUDE.md sentence is a normative instruction to agents that all load CLAUDE.md, so
  the rule is "enforced, weakly" rather than unwired. The planner should adjudicate
  that with the files open; the asymmetry against the three wired siblings is the
  substantive argument, and the two user-caught incidents are the evidence that the
  weak form is insufficient in practice. A reasoned no-change report is an acceptable
  outcome.
- verified-at-filing: `grep -ciE "prefix-based|prefix arm|both mapping arms|prefix mapping AND context" <target>` → **0 hits in all four target files**, and 0 in `planner.md` / `critic.md` as well (2026-08-05). Absence-of-guard claim, so the 0-hit in-target result is the evidence. Correct-insertion-point confirmed positively in the same pass: `grep -c "identity_bias_predict|identity+bias|identity-family"` → critic-lens-reference.md 5, planner-section-reference.md 4, experiment-guidelines.md 2, statistics-critic.md 1. Landed-fix check: `git log --oneline --since='14 days ago'` on the three named CLAUDE.md targets returns `4238694ae9` (#1746 pooling-convention row, 7 wired surfaces) — inspected, it wires POOLING, not the mapping arms, so it is a sibling landing and not this fix.

## Proposed change (candidate diff sketch — refine in planning)

```
critic-lens-reference.md, Statistics lens item 15 — append a third clause:
+   Additionally verify the plan runs BOTH mapping arms — prefix-based (prefix =
+   everything before the user query) AND context-based (context = prefix + query) —
+   as paired arms of the same design, or names the omission as an explicit stated
+   deviation carried into the clean-result as a scope caveat. REVISE a mapping plan
+   that silently runs one arm. Not a REVISE when the plan fits no map, or the
+   one-arm scope is stated + justified (e.g. a substrate whose prefix is the
+   constant chat template, making the prefix arm a degenerate constant-input floor).
+   Full rule: CLAUDE.md § "Prefix mapping AND context mapping".

planner-section-reference.md §6 — mirror as a "Mapping-arms row (same trigger)" block
statistics-critic.md item 15 — mirror the critic clause
experiment-guidelines.md guideline 11 — extend the title + body to name the arms
```

## Scope / surfaces

- Primary targets: `.claude/rules/critic-lens-reference.md`,
  `.claude/rules/planner-section-reference.md`, `.claude/agents/statistics-critic.md`,
  `.claude/rules/experiment-guidelines.md`
- These are exactly the four surfaces #1604 and #1746 wired for the sibling
  representation-mapping disclosures; follow their landed shape rather than inventing
  a new location.
- If CLAUDE.md's bullet is updated to record the wiring (as the identity+bias bullet
  does), keep it consistent with the rule file.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` (no flags) passes; ruff on touched files passes.
- The "unless explicitly stated otherwise" escape is load-bearing and must survive:
  this fix makes the deviation *checked*, never *forbidden*.
- Do not touch the SKILL.md inline-round duty or the routing-section capture-time
  clause — both already carry the rule correctly.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/critic-lens-reference.md,.claude/rules/planner-section-reference.md,.claude/agents/statistics-critic.md,.claude/rules/experiment-guidelines.md
- fingerprint: a8a502c50011

Surfaced by the orchestrator during interactive planning (no subagent involved). The
user asked "what does the project wide rule say exactly?" about the both-arms mapping
rule; quoting it surfaced that its closing enforcement sentence names files that do
not implement it.
