---
title: 'verify_task_body: label-SHA-vs-URL-pin assert + anaphoric HF pin-reference
  path resolution (pin-citation integrity, #2588 r2/r3)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-27T22:23:30Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r3 on #2588 surfaced two verify_task_body mechanical
  gaps (label-SHA lag; anaphoric false HF path claim)'
workflow: v1
---
## Goal

Close two mechanical-check gaps in `scripts/verify_task_body.py` that let pin-citation defects ship through OVERALL PASS on #2588 (clean-result-critic rounds 2-3, 2026-08-27):

1. **Label-SHA vs URL-pin mismatch assert.** A markdown link whose LABEL cites a short SHA (`[figures/issue_2588 @ c2ba58475a](...)`) while the URL pins a DIFFERENT commit (`/tree/7ce6b403069b.../`) passed the verifier twice in consecutive rounds on #2588 (the label lagged each figure re-pin). Add a check: for every link whose label matches `@ <hex7+>`, the label SHA must be a prefix of a SHA appearing in the URL; mismatch = FAIL (or WARN if grandfathering demands).

2. **Anaphoric HF pin-reference claims escape the artifact-path resolution check.** The #2588 body asserted per-cell `gpqa_judge_verdicts.json` files lived "in that same capability-panel tree" (anaphoric reference to an earlier pinned HF link) — the files did not exist at that revision, and the mechanical pass could not see the claim because the path/claim check only fires on explicit backtick-path + link forms. Widen the HF-adjacent claims check to anaphoric pin references: when prose names a concrete filename adjacent to a same-paragraph/section pinned HF tree reference, resolve the filename against a scoped `list_repo_tree` at that pin (or at minimum flag it for manual verification). The #530/#534 false-premise class; caught only by the round-3 critic's scoped listing.

## Acceptance criteria

- Both checks implemented in `verify_task_body.py` with tests in `tests/test_verify_task_body.py` reproducing the two #2588 shapes (label-lag; anaphoric false path).
- Forward-only: v3/v2/legacy bodies not newly hard-FAILed (follow the existing grandfathering conventions; network-dependent check must degrade gracefully offline — WARN, never a spurious FAIL).
- No regression on the existing check suite (`uv run pytest tests/test_verify_task_body.py`).

## Provenance

Surfaced by clean-result-critic round 3 on #2588 (epm:clean-result-critique v3, 2026-08-27T22:19:55Z): "workflow-fix follow-ups surfaced" items 1+2. Filed by the #2588 orchestrator per .claude/rules/workflow-fix-on-bug.md (both gaps target the same file and the same pin-citation-integrity class; filed as one task with two acceptance criteria).
