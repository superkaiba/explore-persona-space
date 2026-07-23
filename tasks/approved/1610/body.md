---
title: seam-pin _generate_responses_vllm row schema (contract test + consumer audit)
kind: infra
tags: []
created_at: '2026-07-23T03:33:24Z'
has_clean_result: false
origin_prompt: Fix all the problems in the background with happycoder (#1586 crash
  r6 reuse-seam class not closed statically)
workflow: v1
---
## Overview / Motivation

Filed from a deep-dive on #1586's crash history (2026-07-23, user-directed "fix all
the problems in the background"). Crash r6 (issue-1586 branch commit `f404fd2a5f`)
hit a `KeyError: "response"` because the reused helper
`analysis/representation_shift._generate_responses_vllm` returns **token-id** rows
(`{persona, question_idx, prompt_token_ids, response_token_ids, finish_reason}`) —
there is NO `response` text key — and the marker dispatcher consumed `r["response"]`
across the reuse seam. #1586 fixed its own two call sites, but the class is NOT
closed statically: the helper has ~13 other consumers and NO test pins its row
schema.

## Goal

Close the reuse-seam row-schema class statically: add a contract test that pins the
row schema returned by `_generate_responses_vllm`, and audit every consumer for a
text-key (`r["response"]`) assumption against a helper that only returns token ids —
fixing any real latent site (decode `response_token_ids` at the seam) or confirming
it correctly consumes token ids.

## Scope / surfaces

- `src/explore_persona_space/analysis/representation_shift.py` — a returned-row-schema
  assertion/docstring contract on `_generate_responses_vllm` (no behavior change).
- New `tests/test_representation_shift_row_schema.py` — fails if the row schema drops
  a token-id key OR silently reintroduces a `response` text key.
- Audit (grep `_generate_responses_vllm` across `scripts/` + `src/`) the consumers:
  issue1112/1333/1315/1090/653/623/685 dispatchers + `experiments/issue_653/onpolicy_pool.py`.
  Fix any that read a nonexistent text key; leave correct token-id consumers untouched.

## Constraints / invariants

- Do NOT edit `scripts/issue1586_dispatch.py` — its seam is already fixed on the live
  `issue-1586` branch (commit `f404fd2a5f`) and #1586 is still running; touching it
  risks a merge collision with the live session.
- No behavior change to the helper output; the test pins the existing schema.
- `uv run ruff check` on touched files passes; the new test passes.

## Provenance

- Origin: user chat 2026-07-23 ("Fix all the problems in the background with happycoder")
  on the #1586 crash-history review.
- Related task: #1586 (crash r6, `reused-helper row-schema seam`).
