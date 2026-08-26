---
title: 'workflow-fix: _hf_routing_exempt_spans row indices use a different line model
  than its caller — a form feed can UNDER-flag a real HF call'
kind: infra
tags:
- wf-fix
created_at: '2026-08-26T20:48:23Z'
has_clean_result: false
origin_prompt: 'Blocker B1 from task #2351''s round-2 code-review ensemble. #2351
  was superseded on landing by #2355 (same gap, landed first); this line-model misalignment
  is a DISTINCT bug that #2355''s shipped implementation carries, and is the one #2351
  finding worth preserving. Regression pins already written and gate-verified on branch
  issue-2351.'
workflow: v1
---
# `_hf_routing_exempt_spans` row indices use a different line model than its caller — a form feed can UNDER-flag a real HF call

**Surfaced by:** task #2351's round-2 code-review ensemble (blocker B1), 2026-08-25. #2351 built a parallel implementation of the same string-awareness feature and was superseded on landing by #2355, which reached `main` first. This bug is #2355's, is DISTINCT from the docstring false-positive both tasks fixed, and is the one finding of #2351 worth keeping.

## Gap

`scripts/workflow_lint.py` `_hf_routing_exempt_spans` (#2355) derives exempt column spans from `tokenize`, keyed by the tokenizer's row numbers. Its caller `_hf_routing_file_errors` iterates `text.splitlines()` and looks the spans up with `spans.get(i + 1, ())`.

The two use DIFFERENT line models:

- `tokenize` splits on `\n` only.
- `str.splitlines()` also splits on `\x0b` (VT), `\x0c` (FF), `\x1c`/`\x1d`/`\x1e` (FS/GS/RS), `\x85` (NEL), ` `, ` `, and `\r` forms.

So a single form feed anywhere in the file — **including inside a string literal, where it is ordinary text** — makes `splitlines()` produce more rows than the tokenizer counted. Every later span is then attributed to the wrong physical line.

## Why it matters: the failure direction is UNDER-flag

`[live-hf-retry-routing]` exists to catch unrouted HF Hub calls. Misaligned spans can mark a row exempt that holds a REAL bare call, so the call ships unflagged. #2355's own docstring is explicit that under-flagging is "the one direction this check must never take" (its rationale for never masking f-strings), and this path takes it.

A corollary symptom: a reported error line number can point at a physical line that does not contain the call.

## Reproduction

Fixture (the form feed is inside a string, so the file is perfectly valid Python):

```python
A = "pre\x0cpost"
B = hf_hub_download(x)
C = "hf_hub_download(text)"
```

Expected: 2 errors — the real call at line 2 flagged, and (pre-exemption) the string at line 3. With the line models disagreeing, the row lookup shifts and the real call at line 2 can be suppressed.

A ready-made regression pin exists and is known-green against the guard described below — #2351 branch `issue-2351`, `tests/test_workflow_lint.py`:

- `test_check_live_hf_retry_routing_formfeed_in_string_does_not_suppress_real_call`
- `test_hf_routing_masked_spans_refuses_on_line_model_disagreement` (unit pin; also covers U+2028)

Both PASSED in that branch's Step 9c gate run 4. They are written against #2351's function names and need renaming to #2355's (`_hf_routing_exempt_spans`), not rewriting.

## Proposed fix

The cheap, fail-SAFE form (what #2351 implemented and gate-verified): before masking anything, confirm the two line models agree, and refuse to mask when they do not.

```python
lines = text.splitlines()
joined = "\n".join(lines)
if joined != text and joined + "\n" != text:
    return None   # line models disagree — fall back to no exemption
```

Refusing to mask degrades to the pre-#2355 behavior for that file: the check OVER-flags rather than under-flags, which is the safe direction, and every reported line number stays byte-identical to the line-based scan by construction. It reuses #2355's existing `None` contract, so the caller needs no change.

An alternative — reindexing spans onto the `splitlines()` model — is strictly more code for a case that is rare and already safely degradable. Prefer the guard.

## Acceptance

1. A file containing a form feed inside a string literal, plus a real unrouted `hf_hub_download(` call, still FLAGS that call.
2. The docstring-exemption behavior #2355 shipped is unchanged for all files whose line models agree (the overwhelming majority) — verify no new offenders and no lost exemptions in the no-flags run.
3. Both regression pins above land, renamed to the `_hf_routing_exempt_spans` API.

## Scope note

Do NOT reintroduce #2351's parallel `_hf_routing_masked_spans` / `_HF_ROUTING_MASK_TOKEN_TYPES` / `_hf_routing_token_is_fstring`. #2355's implementation is the incumbent and is better in two respects — it masks `FSTRING_START`/`MIDDLE`/`END` on py3.12+ rather than `MIDDLE` alone, and it computes spans lazily on the first matching line, which subsumes #2351's whole-file early-return optimization. This task is the line-model guard ONLY.

## Separately observed, NOT in scope

`_hf_routing_call_is_wrapped` still reads `retry_transient` mentions and paren balance on preceding lines including docstring content. #2355 names this and explicitly defers it. Left deferred here too; it deserves its own task if anyone hits it.
