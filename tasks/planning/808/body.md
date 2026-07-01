---
title: 'workflow-fix: resolve SPEC.md Lens 12 sentence-count contradiction (from #779
  reconciler)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:49a428d98652
created_at: '2026-07-01T16:27:46Z'
has_clean_result: false
origin_prompt: 'reconciler on task #779 clean-result-critique r2 ruled Codex''s Lens
  12 ≤2-sentences-per-paragraph finding unenforceable due to SPEC.md L828 directly
  contradicting L814 (1-3 sentences) for the same v4 Results interpretation beat.
  Latent workflow-surface ambiguity that will regenerate the same disagreement on
  every future clean-result-critic ensemble round.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a surfaced observation by the `reconciler` on task #779 clean-result-critique round 2. The reconciler ruled Codex's Lens 12 "≤2-sentences-per-paragraph" finding as **not blocking** because it cites SPEC.md L828 which directly contradicts SPEC.md L814 ("1-3 sentences") for the same v4 Results interpretation beat.

## Goal

Resolve the SPEC.md Lens 12 self-contradiction so future clean-result-critic + codex-clean-result-critic ensemble rounds don't repeatedly disagree on the paragraph-length rule.

## Workflow gap

- **Bug observed:** SPEC.md § "Lens 12 — Conciseness (word-cap adherence + bullets-over-prose)" contains two mutually contradictory text-only rules for the same v4 Results interpretation beat:
  - **L814 (approx):** "each result's interpretation prose 1-3 sentences"
  - **L828 (approx):** "each interpretation paragraph ≤2 sentences"
  Task #779 r2 clean-result-critique ensemble: Claude accepted the 3-sentence paragraphs (citing L814); Codex FAILed them (citing L828); reconciler dismissed L828 as unenforceable due to the contradiction, ruling PASS for Lens 12. This exact disagreement will keep regenerating on every future clean-result review because BOTH sentences are in the spec and neither is enforced mechanically.
- **Why it is a workflow gap:** the spec's mechanical implementation (`verify_task_body.py`) enforces WORD-COUNT caps (120 WARN / 180 FAIL) — no sentence-count check exists. So both L814 and L828 are text-only guidance that reviewers apply subjectively; when they conflict, the reviewer's interpretation controls, causing avoidable ensemble disagreement.
- **Confidence (emitter):** high — reconciler explicitly named this as a latent workflow-surface ambiguity affecting all future rounds; the SPEC citations are concrete.

## Proposed change (candidate diff sketch — refine in planning)

Pick ONE canonical rule for interpretation paragraph length + delete the contradicting one:

**Option A (preferred if L814 is the intended shape):** delete the L828 "≤2 sentences" text; keep L814 "1-3 sentences" as the guidance. Add a note in the SPEC that no mechanical check enforces sentence count — 1-3 is a register guideline.

**Option B (if L828 is the intended shape):** delete L814 "1-3 sentences"; keep L828 "≤2 sentences". Add a `verify_task_body.py` check that counts sentence terminators per interpretation paragraph and FAILs at >2. Update `audit_clean_results_body_discipline.py` similarly.

Either way, harmonize `.claude/agents/clean-result-critic.md` + `.claude/agents/codex-clean-result-critic.md` Lens 12 descriptions to cite ONE sentence-count rule (or drop the count entirely and rely on word-count).

## Scope / surfaces

- Primary target: `.claude/skills/clean-results/SPEC.md` (L814 + L828 area)
- Grep the workflow surface first: `grep -rn 'sentences\|sentence-terminator\|paragraph.*sentences' .claude/skills/clean-results/ .claude/agents/clean-result-critic.md .claude/agents/codex-clean-result-critic.md scripts/verify_task_body.py scripts/audit_clean_results_body_discipline.py`
- Secondary target: whichever of the two rules is deleted, also remove any downstream references in the agent files.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Do NOT invalidate any existing PASSed v4 clean-result body — the fix should be forward-only.
- The mechanical gate (`verify_task_body.py`) should either stay unchanged (Option A) or gain a new sentence-count check (Option B); do NOT tighten mechanical checks without approval.

## Provenance

- workflow_fix_target: .claude/skills/clean-results/SPEC.md
- fingerprint: (auto)
- Emitted by: reconciler on task #779 clean-result-critique r2 (2026-07-01)
