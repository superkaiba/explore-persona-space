---
name: Verify the inherited producer's capture semantics before crediting a slot/field claim
description: When a code-review finding turns on "slot/field X of an inherited artifact means Y" (boundary-token slots, answer-span extent, custom_id layout), read the PRODUCER's actual capture code — not the plan prose or the consumer docstring — before classing the finding real or mistaken.
type: feedback
---

When a code-reviewer (Claude or Codex) raises a finding whose truth hinges on
what a SLOT / FIELD / SPAN of an INHERITED upstream artifact actually contains
— "these two positions are the turn-boundary tokens", "the answer span includes
`<|im_end|>`", "the custom_id round-trips the (ctx, probe, draw) tuple", "this
JSON key holds the graded mean" — do NOT adjudicate from the plan's prose or the
consumer script's own docstring. Read the PRODUCER'S capture/write code and
confirm what it actually persists.

**Why:** the plan and the consumer script can both confidently assert a
semantics the producer never delivered, and they will agree with each other
while both being wrong. In #812 r1 (2026-07-01) Codex flagged that
`pooling_extract.py`'s `im_end = sp[-1]` / `turn_nl = sp[-2]` slots are the last
two ANSWER-CONTENT tokens, NOT the true chat-boundary tokens. The plan §4.1 AND
the extract docstring BOTH claimed "im_end at span_end" — but reading the #658
producer (`issue658_extract_base_store.py`: `answer_span_stack` slices
`[prompt_len, prompt_len+ans_len)` where `ans_ids = tokenizer(ans,
add_special_tokens=False)` on the vLLM `.text` output) showed the span holds
answer-content tokens ONLY — no `<|im_end|>`, no trailing template newline. The
finding was fact-correct; the plan's stated capture convention was simply wrong,
and the code faithfully implemented the wrong convention. (This is the
producer-half of the producer/consumer contract-mismatch family the
code-reviewer memory already tracks — same verify-the-round-trip discipline,
applied at reconcile time to an INHERITED artifact whose producer lives in a
DIFFERENT script than the diff under review.)

**How to apply:** on any slot/field/span-semantics finding, `grep`/`Read` the
line where the inherited artifact was WRITTEN (often a sibling `issueNNN_*.py`
in the same scripts/ dir, or a captured store's extractor). Confirm the extent /
token set / key the producer actually persisted. Then classify:
- Producer delivers what the finding assumes it lacks → finding is Unverified/mistaken.
- Producer does NOT deliver it (as here) → finding is real; severity then turns
  on impact (2-of-34 duplicate feature columns = Non-blocking-persisted with an
  analyzer-narration caveat; a corrupted headline join key = Blocking).
Do this for the CONSUMER side too when a finding claims "field X is/ isn't
consumed downstream" — trace to the actual read site, not the write docstring.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Verify inherited producer's capture semantics before crediting a slot/field claim](feedback_verify_inherited_capture_semantics_before_crediting_slot_claim.md) — read the PRODUCER's write code, not plan prose / consumer docstring: #658 answer-span holds answer-content tokens only, so #812's im_end/turn_nl slots are tail duplicates; plan+consumer both asserted the wrong convention. #812 r1.
