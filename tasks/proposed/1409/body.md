---
title: 'daily-fix: smoke exercises data gates at smoke n'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f663a5571cbc
- daily-auto-filed
created_at: '2026-07-16T07:22:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): 4 pre-existing data gates
  in #1345''s reused code had never executed at smoke n (fold-skip at kept=1-3, n_common>0,
  two asserts) - the class behind two serialized GCP crashes'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Smoke legs must exercise every data-dependent gate at smoke n (degenerate/small-n paths), not just the happy path.

## Workflow gap

- **Bug observed:** 4 pre-existing data gates in #1345's reused code had NEVER executed at smoke n (fold-skip at kept=1-3, n_common>0, two asserts) — exactly the class behind two serialized GCP crashes (c12fae90 17:01Z).
- **Why it is a workflow gap:** the experiment-implementer's smoke contract mandates architecture/width/axis parity between smoke and production but never requires the smoke to actually EXECUTE the code's data-dependent gates at smoke n — so gates first fire in production, on billed GPU boxes.
- **Severity:** medium
- verified-at-filing: `grep -n 'smoke' .claude/agents/experiment-implementer.md` (smoke contract L116-163) → covers path unification, process-shape/width parity (L116-121), smoke-override threading (L129-140), non-cell smoke axes + un-passable-floor checks (L139-152) — no clause requiring data-dependent gate/degenerate-path execution at smoke n (absence confirmed, consistent with the miner's grep of L116-140) (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend `.claude/agents/experiment-implementer.md`'s smoke contract (L116-163): the smoke leg must enumerate the code's data-dependent gates (fold-skip thresholds, non-empty-intersection checks, shape/count asserts) and demonstrate each either executes at smoke n or is explicitly listed as production-only with a stated reason — degenerate/small-n paths are part of the smoke surface, not just the happy path. DEDUP NOTE: reconcile against tonight's Step C "experiment-implementer.md per-arm-class smoke clause" filing before spawning — if that clause already covers gate execution at smoke n, this filing is subsumed and should be archived at clarify time.

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md` (smoke contract, L116-163)
- Dedup: same-night Step C per-arm-class smoke clause filing on the same file (different fingerprint — distinct bug unless that clause covers gate execution)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f663a5571cbc

- workflow_fix_target: .claude/agents/experiment-implementer.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: c12fae90 (#1345) 17:01Z (batch 06 P15); existing smoke section covers width/axis parity but not gate-path execution (grep-verified L116-140).
