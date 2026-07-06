---
title: 'workflow-fix: artifact-reuse fitness check gains pairwise provenance-coherence
  item (j)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5d3f86e1b463
created_at: '2026-07-03T17:27:15Z'
has_clean_result: false
origin_prompt: 'experiment-implementer #922 r4: sha-pinned mutually-dependent artifact
  pair (questions regenerated AFTER the dependent activation capture) passed all (a)-(i)
  checks and crashed a full GCE cycle at the parity assert.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #922 (emitting agent: experiment-implementer, crash-fix round 4).

## Goal

Add a pairwise provenance-coherence item (j) to the artifact-reuse (a)-(i) fitness check: for any reused artifact PAIR that must be mutually consistent, compare HF last-commit dates + reconstruction/regeneration metadata and REVISE when a consumed input postdates its dependent capture.

## Workflow gap

- **Bug observed:** #922's plan sha-pinned #779's question artifacts + cx.pt and passed every reuse-fitness check, but the questions had been REGENERATED after the activation capture (documented in the artifact's own reconstruction field, HF commit 9578892ef4 2026-07-02 vs capture a8060198a4 2026-07-01) — the pair was mutually inconsistent and the run crashed at the parity assert after a full GCE cycle (att-20260703-163130).
- **Why it is a workflow gap:** the reuse checklist's content-identity check (f) pins each artifact's CURRENT bytes but never checks PROVENANCE COHERENCE across a mutually-dependent artifact PAIR (an input regenerated AFTER a dependent capture); the existing "Pinned artifact pairs can disagree" memory covers coverage mismatch, not post-hoc regeneration.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ (j) **Pairwise provenance coherence.** When reuse consumes an artifact PAIR that
+     must be mutually consistent (question/prompt banks vs activations captured
+     under them; training mixes vs adapters), verify the input does NOT postdate
+     the dependent capture: `HfApi.get_paths_info(..., expand=True)` last-commit
+     dates + any `reconstruction`/regeneration metadata field. A consumed input
+     regenerated AFTER the capture is inconsistent regardless of sha pins
+     (#922 att-20260703-163130).

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Grep the workflow surface for enforcement hooks before editing (`grep -rln 'fitness check' .claude/ CLAUDE.md`) — the (a)-(i) letter set is referenced from CLAUDE.md, planner.md step 5, consistency-checker, critic.md Methodology item 9; extending to (a)-(j) must update every referencing surface. Also consider a matching `.claude/rules/gotchas.md` bullet (the r4 failure-lesson flagged gotcha_candidate: yes — fold it here rather than a second task).

## Constraints / invariants

- Workflow-surface only; `workflow_lint.py` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: 5d3f86e1b463

Verbatim candidate block from the #922 r4 implementer report (see epm:experiment-implementation v4 / the r4 return text).
