---
title: 'workflow-fix: verify_plan.py resume-provenance check (persist+resume-skip
  plans must name input-fingerprint validation)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-05T06:30:03Z'
has_clean_result: false
origin_prompt: 'codex-critic (alternatives) prose note, #952 v9 review: fold/resume
  consumers should mechanically validate input fingerprints, not just output existence
  — generalize the #952 gate-5 contract into a verify_plan.py surface check.'
workflow: v1
---
## Overview / Motivation

Auto-filed from a prose workflow-gap note surfaced by the codex-critic (alternatives lens) during #952's kfold-decision-cells plan review (2026-07-05): "This is a recurring workflow-surface verifier check: fold/resume consumers should mechanically validate input fingerprints, not just output existence." The #952 v9→v10 revision added the check as a REGISTERED plan gate (gate-5 provenance manifests) for that one experiment; the generalizable half is a verify_plan.py surface check.

## Goal

Add a verify_plan.py check (WARN experiment/analysis): a plan that names a per-unit persist + resume-skip pattern (resume, checkpoint-skip, per-fold/per-cell persist) must ALSO name an input-fingerprint / provenance-manifest validation for the resumed outputs (split hashes, code SHA, env knobs — any recognizable fingerprint token near the resume mention), not just output existence.

## Workflow gap

- **Bug observed:** #952 plan v9 prescribed per-fold persist + resume-skip with no provenance contract; a crash + code-fix round would let stale-fold outputs (or a stale calibration gate PASS) silently vouch for post-fix verdict folds. Caught only by the critic ensemble (reconciler-binding REVISE).
- **Why it is a workflow gap:** resume-skip-without-fingerprint is a recurring plan shape (the intra-phase persist law mandates per-unit persist for >1h loops, so every long battery plan writes one); the verifier has checks for battery arithmetic (c12) but nothing for resume-provenance.
- **Confidence (emitter):** medium (Codex prose note + a reconciler-verified live instance).

## Scope / surfaces

- Primary target: `scripts/verify_plan.py` (+ `tests/test_verify_plan.py`)
- Window-scoped like c12 (±15 lines of a resume/persist mention); canonical N/A phrase for plans with no resume pattern.

## Constraints / invariants

- WARN not FAIL in v1 (surface check; semantic adequacy stays with the critics); workflow_lint + existing verify_plan tests stay green.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- origin: codex-critic alternatives-lens prose note on #952 plan v9 (2026-07-05); adopted as #952 plan v10 gate-5 for the instance
