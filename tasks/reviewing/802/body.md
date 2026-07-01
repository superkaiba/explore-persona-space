---
title: 'workflow: promote bug-class sibling-sweep from reconciler memory into primary
  code-reviewer + implementer (stop review whack-a-mole)'
kind: infra
tags:
- workflow-whackamole
- wf-sibling-sweep
created_at: '2026-07-01T07:59:59Z'
has_clean_result: false
origin_prompt: 'fix the whack a mole and also get a subagent to figure out a workflow
  improvement to stop this whackamole behavior [diagnosis: sibling-scan discipline
  imprisoned in reconciler memory; primary code-reviewer reviews line-instances not
  bug-classes -> siblings surface one-per-round]'
---
## Overview / Motivation

Autonomous `/issue` code-review loops exhibit a "whack-a-mole": the `experiment-implementer`
fixes the flagged instance, and each subsequent code-review round surfaces a NEW real bug —
usually a SIBLING in the same subsystem the previous round just touched — so the loop burns
many rounds (or hits the cap and strategy-pivots) without converging. Instance: #779 produced
≥6 real blockers clustered in the raw-completions I/O subsystem, plus r_B extraction and the
primary metric, one-per-round.

**Root cause (diagnosed):** the sibling-scan / bug-class-sweep discipline (enumerate EVERY
instance of a bug class per round — a 7-step recipe backed by 8 incidents: #498/#601/#505/
#514/#489/#407) lives ONLY in `.claude/agent-memory/reconciler/feedback_claude_misses_same_file_siblings.md`.
The `reconciler` fires only when the Claude and Codex reviewers ALREADY disagree (PASS-vs-FAIL).
The primary `code-reviewer` loads `code-reviewer/*` memory, NOT `reconciler/*`, and has only
per-instance gates — no always-on step requiring it to sweep the subsystem for siblings. So on
any finding it reports the one cited instance and the siblings surface in later rounds.
Compounding it, `experiment-implementer.md`'s revision-round rule says "targeted edits — do NOT
rewrite unrelated code," which actively discourages fixing the class.

This is ORTHOGONAL to #784 (which changes the round-cap value + strip logic). This task is about
SCOPING each round to the bug CLASS, not the cap.

## Goal

Make the code-review loop surface AND fix bug CLASSES per round (all sibling instances in one
round) instead of one instance at a time — promoting the reconciler's battle-tested sibling-scan
recipe into the primary reviewer + implementer protocols.

## Change set (interdependent — MUST land together; a partial merge leaves the workflow self-contradictory)

1. **`.claude/agents/code-reviewer.md` — new mandatory "Step 3.7: Bug-class sibling sweep."**
   For EVERY Critical/Major finding, the cited line is an INSTANCE; the contract is the bug
   CLASS. Before verdict, grep the whole file + the sibling function/rubric/resampler family +
   sibling scripts sharing the finding's data contract + parallel layers (figure-vs-analyze) for
   the same pattern (e.g. `parsed.get("score",0)`, per-persona vs global custom_id index,
   `raw_completions` write-without-upload, `except Exception`, sentinel reads). Report ALL sibling
   hits under one "### Bug-class sweep: <class>" heading with file:LINE each; a load-bearing
   sibling is its own Critical, a secondary one a standing rec. Add an enforcing Rule 14: "Every
   finding is a bug CLASS, not a line. FAIL enumerating all load-bearing siblings; a single-instance
   fix that leaves siblings is the whack-a-mole failure mode." (Promotes the 7-step recipe from
   reconciler memory `feedback_claude_misses_same_file_siblings.md`.)

2. **`.claude/agents/experiment-implementer.md` — revision-round class-hardening carve-out.**
   The current "Make targeted edits — do NOT rewrite unrelated code on a revision round" rule
   contradicts the class-sweep. Add a carve-out: when a FAIL item names a bug CLASS (or the
   reviewer's Step 3.7 enumerated siblings), fix EVERY named sibling instance this round, and
   before returning run a one-line self-sweep grep of the just-fixed pattern across the touched
   subsystem, reporting it under `(c) How to verify`. (Implementer-side mirror of the reviewer
   step — required to resolve the contradiction above.)

3. **`.claude/skills/issue/SKILL.md` (Step 5c / round-N+1 brief composition) — carry the sibling
   family into the punch list.** When a reviewer verdict contains a "### Bug-class sweep"
   enumeration, thread the FULL sibling list (not just the top finding) into the round-N+1
   implementer brief so its punch list is class-scoped.

4. **`.claude/agent-memory/reconciler/feedback_claude_misses_same_file_siblings.md` — one-line
   pointer** noting the recipe is now promoted to the primary `code-reviewer.md` Step 3.7 (keeps
   the memory as the reconciler's fallback and in sync with the enforced step).

## Scope / surfaces to check

- Grep the reviewer-rubric surface so the sweep step propagates to BOTH twins: the Codex twin
  `.claude/agents/codex-code-reviewer.md` inlines "the same rubric the Claude reviewer uses" — if
  it inlines the rubric verbatim rather than referencing code-reviewer.md, add Step 3.7 there too
  (verify by reading how the codex twin composes its prompt).
- Check `.claude/workflow.yaml` for any review-step enumeration that must list the new Step 3.7.

## Constraints / invariants

- Workflow-surface only. Grep before editing; keep code-reviewer.md ↔ codex-code-reviewer.md ↔
  SKILL.md consistent.
- Orthogonal to #784 — do NOT touch the round-cap value or the strip logic (that's #784's task).
- `scripts/workflow_lint.py --check-asks` passes; ruff on any touched scripts passes.
- Not architectural (adds a review step + directive; no enum/schema/subcommand rename) → auto-approve
  + Step-10d auto-merge per the workflow-fix default.

## Provenance

Diagnosed by a workflow-improvement subagent commissioned from the #779 whack-a-mole. The four
changes are the subagent's single high-confidence candidate (change 1) plus its three prose
follow-ups (changes 2-4), consolidated into ONE task because they are interdependent (the reviewer
sweep and the implementer carve-out contradict each other unless both land).
