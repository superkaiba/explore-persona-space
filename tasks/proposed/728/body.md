---
title: 'workflow-fix: canonical persona-vectors recipe (match paper, no logits) +
  enforce'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f95ef1b1a2cb
- persona-vectors-recipe
created_at: '2026-06-29T09:03:17Z'
has_clean_result: false
origin_prompt: make sure that if an issue in the future says to use persona vectors
  it will properly use the same thing as the paper (Except these logits)
---
## Overview / Motivation

Auto-filed from a user directive on the #658 persona-vectors follow-up:

> make sure that if an issue in the future says to use persona vectors it will
> properly use the same thing as the paper (Except these logits)

There is currently NO canonical, enforced "persona vectors" extraction recipe in
the workflow surface — the recipe lives only in scattered secondhand lit-review
summaries (`docs/behavior-implantation-lit-review.md`, `docs/papers.md`). Because
of this gap, #658 built a persona-vectors-"inspired" `r_B` that DIVERGED from the
actual paper (Chen et al., arXiv 2507.21509): it diffed two different corpora with
no judge-filter, instead of the paper's content-matched, judge-filtered, on-policy
recipe. Any future task saying "use persona vectors" would repeat this.

## Goal

Add a canonical persona-vectors extraction-recipe rule and wire it into the
planning/critique surface so that any future task whose plan says "use persona
vectors" / "extract a persona vector" / "persona-vectors-style direction"
reproduces the paper's recipe faithfully — EXCEPT the GPT-4.1 logit-weighted
scoring, which is replaced by the standard project judge (user directive).

## The canonical recipe to encode (verified against arXiv 2507.21509, §2 + App. A/B/H)

Create `.claude/rules/persona-vectors-recipe.md` stating, as the default any
"use persona vectors" task inherits:

1. **Inputs:** a trait name + a brief natural-language trait description.
2. **Artifacts** (one frontier-LLM generation; paper used Claude 3.7 Sonnet —
   project may use `claude-sonnet-4-5`): **5 pos/neg system-prompt PAIRS** (pos
   commands the trait, neg commands the opposite), **40 questions** split into a
   disjoint **20-question extraction set + 20-question evaluation set**, and **1
   trait evaluation prompt**. (Paper's generation prompt template is in App. A.)
3. **Generation:** for each extraction question, **10 on-policy rollouts under the
   positive system prompt and 10 under the negative** (sampling, not greedy).
4. **Judge-filter (load-bearing — this is what #658 omitted):** score every
   rollout 0-100 with the trait evaluation prompt; **KEEP positive-prompt
   responses scoring >50 and negative-prompt responses scoring <50**; discard the
   rest.
   - **Judge = `claude-sonnet-4-5-20250929`** (project judge rule), plain numeric
     0-100 score, threshold 50.
   - **THE "except logits" carve-out (standing project default):** do NOT use the
     paper's Betley-style logit-weighted top-20-token scoring, and do NOT add
     GPT-4.1-mini as a second judge. Just use the standard project judge.
5. **Activation position:** residual stream at **every layer, averaged over
   RESPONSE tokens** (the paper's `response-avg` default — it beat prompt-last /
   prompt-avg in their ablation, App. B).
6. **Direction:** `r_B` (persona vector) = mean(activations | kept
   trait-exhibiting) − mean(activations | kept non-exhibiting), per layer.
7. **Layer selection:** the paper picks ONE most-informative layer by **steering
   effectiveness** (layer ~20 for evil & sycophancy on Qwen2.5-7B). The rule must
   state: for **steering / monitoring** tasks, follow the paper (steering-selected
   layer); for **read-out / prediction** tasks (e.g. A3.3 `E ≈ r_Bᵀ v`), sweep all
   layers and select by which best predicts the target (named deviation — our task
   is read-out validity, not steering). The task's plan states which regime applies.

## Enforcement wiring (the "make sure" part)

- **`CLAUDE.md`:** add a Critical-Rules bullet pointing at the new rule, with the
  load-on-demand trigger ("any task whose plan says use/extract persona vectors,
  or extracts a persona/behavior direction, READS `.claude/rules/persona-vectors-recipe.md`
  before grounding the extraction"). Keep it consistent with the existing
  `replication-fidelity`, `contrastive-negatives`, and `on-policy-completions`
  rules (siblings — cross-link them).
- **`.claude/agents/planner.md`:** §4 Design — when a task elects "persona
  vectors", the plan must instantiate steps 1-6 above (artifacts, on-policy
  rollouts, judge-filter, response-avg, diff-of-means) and name the layer-selection
  regime; cite the rule as the Source.
- **`.claude/agents/critic.md`:** Methodology lens — REVISE a persona-vectors
  extraction that (a) uses mismatched corpora instead of pos/neg system prompts on
  shared questions, (b) omits the judge-filter, (c) extracts at a prompt position
  instead of response-avg, (d) skips the on-policy rollouts, or (e) re-introduces
  the GPT-4.1 logit-weighted scoring without an explicit override.
- Cross-link from `.claude/rules/replication-fidelity.md` (persona vectors is a
  named published recipe) and note the composition with
  `.claude/rules/on-policy-completions.md` (the rollouts are on-policy) and
  `.claude/rules/contrastive-negatives.md` (the pos/neg structure).

## Scope / surfaces

- NEW: `.claude/rules/persona-vectors-recipe.md`
- EDIT: `CLAUDE.md`, `.claude/agents/planner.md`, `.claude/agents/critic.md`
- CROSS-LINK: `.claude/rules/replication-fidelity.md`,
  `.claude/rules/on-policy-completions.md`, `.claude/rules/contrastive-negatives.md`
- Grep the surface for existing persona-vector mentions before editing
  (`grep -rln 'persona.vector' .claude/ CLAUDE.md docs/`) and reconcile so the new
  rule is the single source of truth (the lit-review docs may link to it).

## Constraints / invariants

- Workflow-surface only — no experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files;
  if `CLAUDE.md`/`workflow.yaml` change, they stay consistent with the new rule.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:`
  Provenance line — it MUST NOT auto-route its own subagents' workflow-fix
  candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/persona-vectors-recipe.md, CLAUDE.md, .claude/agents/planner.md, .claude/agents/critic.md
- fingerprint: f95ef1b1a2cb

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/persona-vectors-recipe.md, CLAUDE.md, .claude/agents/planner.md, .claude/agents/critic.md
bug_observed: persona-vectors r_B was built from mismatched corpora with no judge-filter (diverging from the paper) because no canonical persona-vectors recipe rule exists
why_workflow_gap: the persona-vectors recipe lives only in scattered secondhand lit-review summaries, so a task saying "use persona vectors" has no enforced canonical recipe to follow
proposed_change: Add canonical persona-vectors extraction recipe rule matching arXiv 2507.21509 (except GPT-4.1 logit-weighted scoring) and wire it into CLAUDE.md + planner + critic so any future use-persona-vectors task follows the paper
confidence: high
related_task: #658
<!-- /workflow-fix-candidate -->
