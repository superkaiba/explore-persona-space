---
title: 'api_dispatch silently converts a text-block-free API response into an empty-string
  SUCCESS (7/10 Fable calls lost in #2202)'
kind: infra
tags: []
created_at: '2026-08-09T07:38:42Z'
has_clean_result: false
parent_id: 2202
origin_prompt: 'User (2026-08-09), after diagnosing why 7 of 10 Fable digest calls
  in #2202 returned empty: ''yes!'' to filing the api_dispatch silent-empty default
  as kind: infra — it''s a fleet-wide fail-fast violation, and #2202 only caught it
  because 7 of 10 calls happened to fail at once.'
workflow: v1
---
# api_dispatch silently converts a text-block-free API response into an empty-string SUCCESS

## The bug

`src/explore_persona_space/llm/api_dispatch.py:820` (and the identical line at
`:1163`):

```python
text = next((b.text for b in msg.content if b.type == "text"), "")
```

When an API response carries **no `text` content block** — the model emitted
only a thinking block, the output budget was exhausted before any text was
produced, the response was a refusal with no text, or the content list is empty
— this defaults to the empty string and the item is recorded as a **success**:

```json
{"result": "", "error": false, "reason": null, "category": "ok"}
```

No exception, no error flag, no `reason`, `category: "ok"`. The caller cannot
distinguish "the model returned nothing" from "the model returned an empty
answer on purpose". This is a silent default swallowing a fault — the exact
shape CLAUDE.md's fail-fast rule forbids ("no value placeholders, silent
defaults, or fallbacks that swallow the fault; the crash IS the signal").

This is a **shared-library** path: every caller of `dispatch_calls` inherits it,
not just the one that found it.

## How it was found (#2202)

In #2202, 7 of 10 Fable-5 digest chunks returned empty. The failure digest was
5/5 empty; the 500-sample digest was 2 complete, 1 truncated mid-word, 2 empty.
Every one of the seven was classified `category: "ok"`. Downstream,
`parse_modes("")` returned the empty list, so the empties contributed zero
proposals and the pipeline continued as though those chunks had merely found
nothing — mode discovery silently ran on 30% of its intended input and the run
reached `awaiting_promotion` with the gap visible only as a line in the
coverage-notes takeaway.

Aggravating factor at the call site: the caller passed
`parse_response=lambda t: t` rather than `parse_response_meta`, so `stop_reason`
was never persisted and the PROXIMATE cause is now unrecoverable from the
artifacts. That is a caller-side choice, but the library makes the safe choice
easy to skip: the metadata-aware parser is opt-in while the lossy default is
free.

## Why the existing gates did not catch it

- The llm-judging rule-26 pilot gate keys on `stop_reason == "max_tokens"`,
  which requires `parse_response_meta`. With the plain parser there is no field
  to gate on.
- No gate anywhere treats an empty model reply as a failure condition.
- The #2202 pilot probe was a 16-token `"Reply with the single word OK."`
  round-trip, which exercises auth and routing but not the production prompt
  shape (the sibling-axis version of #2152 — a pilot must exercise the
  instrument it gates).

## Proposed fix

1. **Make the empty case explicit at the library boundary.** When no `text`
   block is present, do NOT return the empty string as a success. Either raise,
   or return a typed failure record carrying `category` (for example
   `empty_response`), the response `stop_reason`, and the content-block types
   actually returned, so the caller sees WHY. Preserve the existing `error` /
   `reason` / `category` record contract so current consumers keep working.
2. **Persist `stop_reason` on every path**, not only when a caller opts into
   `parse_response_meta` — the field is already getattr-read at `:823`/`:1167`;
   record it on the result object regardless of which parser the caller passed.
3. **Audit and migrate existing callers.** Enumerate every `dispatch_calls`
   call site; any that can be fed a non-trivial prompt should either handle the
   typed empty record or opt into the metadata parser. Report the list.
4. **Add a regression test** that a mocked response with no text block (and one
   with an empty content list) produces a FAILURE record, not
   `{"result": "", "category": "ok"}` — plus a test that `stop_reason` survives
   on the default parser path.
5. Consider a lint check in `workflow_lint.py` for the `next((... type ==
   "text" ...), "")` shape, so the pattern cannot silently reappear elsewhere.

## Acceptance criteria

- A text-block-free response is a typed failure, never a success with an empty
  result, at both `:820` and `:1163`.
- `stop_reason` is persisted on every dispatch path.
- Every existing `dispatch_calls` caller is enumerated and either migrated or
  explicitly recorded as unaffected.
- Regression tests cover the no-text-block and empty-content-list cases.
- The full relevant test suite is green on `main`.

## Files of record

- `src/explore_persona_space/llm/api_dispatch.py:820`, `:1163`, `:823`, `:1167`
- #2202 (discovery; `eval_results/issue_2202/fable_reads/`, dispatcher cache
  `data/issue_2202/fable_cache/`), `scripts/issue2202_labels.py:409` (the
  caller-side check that an empty string passes)
- #2152 (pilot gate must exercise the instrument it gates — sibling, not a
  duplicate: that one is about the pilot's transport, this one is about the
  library's silent default)
- `.claude/rules/llm-judging.md` rules 23/26 (max_tokens + pilot gating)
