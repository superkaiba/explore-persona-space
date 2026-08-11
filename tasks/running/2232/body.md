---
title: 'verify_task_body: verify artifact-content claims (per-unit values recorded
  in <pinned JSON>) against the pinned artifact''s actual structure'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-11T13:28:18Z'
has_clean_result: false
parent_id: 2222
workflow: v1
---
# verify_task_body.py: artifact-content claims inside pinned JSONs are never mechanically verified

## Gap (surfaced by clean-result-critic round 2 on #2222)

A v4 body sentence claimed "the per-dataset values behind the three probe correlations are recorded in the probe JSON pinned in the footer" — and the pinned `form_a_probe.json` (at the body's own SHA pin) contains no per-dataset structure at all (no 24-length arrays, no dataset-name keys). `verify_task_body.py` checks HF file COUNTS and adjacent-file EXISTENCE against pins, but never that a claimed data structure exists INSIDE a pinned JSON, so a false artifact-content pointer passes every mechanical check and only a manual critic artifact-walk catches it. The pattern recurs wherever Lens 11's pointer form ("per-unit values recorded in <linked JSON>") substitutes for an embedded per-unit view.

## Fix sketch

Add a check that parses artifact-content claims of the form "per-<unit> values … recorded/live in <linked JSON>" (scoped to Results/Takeaways prose adjacent to a pinned JSON link), loads the artifact at the adjacent pin (git blob or committed working copy), and scans for a structure of the body-stated cardinality — #2222's JSON even records `n_datasets: 24`, so the expected length is usually recoverable from the body or the artifact's own metadata. WARN (not FAIL) when absent — phrasing variance makes the parse heuristic; the critic lens stays binding. This is the artifact-content sibling of the existing HF file-count claims check.

## Acceptance

- The #2222 shape reproduces as a fixture: a body claiming per-dataset values in a linked JSON that lacks any 24-length structure ⇒ WARN naming the JSON + the claim sentence.
- A TRUE pointer (e.g. predictor_correlations.json's `dataset_values` for the main arms) produces no WARN.
- Zero regressions across committed v4 bodies; test added to tests/test_verify_task_body.py.

## Provenance

workflow_fix_target: scripts/verify_task_body.py (new artifact-content-claim check)
Surfaced by: clean-result-critic round 2 on #2222 (2026-08-11), "Workflow-fix suggestion" prose block; verdict posted as epm:clean-result-critique on #2222. Distinct from #2231 (figure-stem prose matching in check 31 — different check class on the same file).
