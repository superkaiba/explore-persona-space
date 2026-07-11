---
title: 'workflow-fix: cross-issue reuse-provenance check in verify_task_body.py'
kind: infra
tags:
- wf-fix
created_at: '2026-07-10T23:44:44Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1092 r3 prose follow-up: mechanize the footer
  Reused-bullet completeness check against result-JSON cross-issue revision pins'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by clean-result-critic (task #1092 round 3).

## Goal

Add a mechanical cross-issue reuse-provenance check to verify_task_body.py: flag any body-cited result JSON whose metadata carries cross-issue HF revision pins or input paths with no matching footer `Reused:` bullet.

## Workflow gap

- **Bug observed:** #1092's transfer-round fold cited a result JSON whose `metadata.args` carried `hf_rev_779_passb` / `hf_rev_779_labels` pins (cross-issue #779 artifacts), but the body footer's `Reused:` list covered only the r_B directions — the gap was caught only by the LM critic's Lens 5 pass at round 3.
- **Why it is a workflow gap:** reuse-provenance completeness is mechanically derivable (result-JSON metadata pins with `issue<M>` prefixes, M != N, vs footer Reused bullets) but no verifier check exists; the LM lens is the only defense.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ # verify_task_body.py: new check — for each body-cited eval_results JSON, parse metadata
+ #   for hf_rev_<M>_* keys (M != issue) and input paths matching r"issue(\d+)_" with M != N;
+ #   each distinct cross-issue source must string-match a footer `Reused:`-section bullet
+ #   (path or issue link); missing -> FAIL with the source named. Forward-only: JSONs
+ #   without such metadata keys are exempt.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Sibling check: `.claude/skills/clean-results/SPEC.md` Reused-bullet contract wording (grep before editing).

## Constraints / invariants

- workflow_lint default run passes; ruff clean; forward-only (no retroactive FAILs on grandfathered bodies).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: pending-wrapper-computed

Surfaced prose (verbatim): "A mechanizable verifier gap surfaced while grounding the blocker: verify_task_body.py could flag any body-cited result JSON whose metadata.args carries an hf_rev_<M>_* pin (M != N) or whose input_shas include cross-issue path prefixes with no matching path in a footer Reused: bullet"
