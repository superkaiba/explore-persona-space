---
title: 'verify_task_body v4: mechanical FAIL for banned Confidence: sentences (check
  6 gap)'
kind: infra
tags: []
created_at: '2026-08-24T08:03:42Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate surfaced in prose by the #2479 interpretation-critic
  round-2 report'
workflow: v1
---
# verify_task_body check 6: add a mechanical FAIL for banned `Confidence:` sentences in v4 bodies

## Goal

Give the SPEC.md rule "confidence lives in the H1 title tag ONLY — no `Confidence: ...` sentence anywhere in a v4 body" a mechanical arm in `scripts/verify_task_body.py`. Check 6 currently only asserts the title TAG is present, so a v4 body that also carries a `Confidence: ...` bullet/sentence PASSes the verifier and the rule is enforced only by reviewer prose.

## Provenance

Surfaced in prose by the #2479 interpretation-critic round-2 report (2026-08-24): the round-2 body carried a `Confidence:`-led `## Takeaways` bullet that `verify_task_body.py --issue 2479` PASSed; the critic had to flag it manually.

## Design sketch

Add a v4-scoped check (new id or an extension of check 6) that FAILs when a line matching roughly `^\s*(?:[-*]\s+)?(?:\*\*)?Confidence:` appears in a `<!-- clean-result-v4 -->` body, outside verbatim-quotation surfaces (fenced code blocks, `<details>` blocks, blockquoted sample lines — reuse the existing elision helpers if present). Forward-only: v3/v2/pre-sentinel bodies are never newly hard-FAILed (grandfathering contract).

## Acceptance criteria

1. A v4 body with a `- Confidence: HIGH ...` Takeaways bullet FAILs with a named check id; the same text inside a fenced block or blockquote does not trip it.
2. A v3 body with a legacy `Confidence:` line still PASSes (forward-only).
3. Tests land in `tests/test_verify_task_body.py` covering both cases.
4. The SPEC.md mechanical-checks section and any check-count references stay consistent.
