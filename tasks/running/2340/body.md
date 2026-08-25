---
title: 'verify_task_body: footer check for SHA-pinned GitHub links on named code/artifacts'
kind: infra
tags: []
created_at: '2026-08-17T09:16:50Z'
has_clean_result: false
origin_prompt: workflow-fix-candidate surfaced by codex-clean-result-critic on /issue
  2330 r1; upheld by binding reconciler
workflow: v1
---
## Gap

verify_task_body.py has no check that a v4 body's **Repro:** footer carries SHA-pinned GitHub blob/tree LINKS for the code branch and committed eval_results artifacts it names — a footer naming "code on branch issue-2330 @ 0ca8b47888" plus bare eval_results/ paths passes every check, but the SPEC footer requirement is permanent links. Surfaced by the codex-clean-result-critic on #2330 r1 ("a v4 Repro footer should reject code/result paths that lack an adjacent SHA-pinned GitHub link") and UPHELD by the binding reconciler (epm:clean-result-critique-reconcile v1 on #2330, item footer-github-artifact-links).

## Asked change

Add a verify_task_body.py check (WARN or FAIL per SPEC reading) that scans the footer for bare `@ <sha>` / bare eval_results path mentions lacking an adjacent github.com blob/tree URL pinned to a 7+-hex sha, with a matching SPEC.md sentence if the requirement is currently implicit. NOTE: the reconciler explicitly adjudicated AGAINST the sibling proposal (a plan-fixed/bootstrap-name prose regex) as over-reaching the enumerated Lens 7 keyword ban — do not implement that one.

Provenance: #2330 clean-result round 1, markers epm:clean-result-critique-codex v1 + epm:clean-result-critique-reconcile v1.
