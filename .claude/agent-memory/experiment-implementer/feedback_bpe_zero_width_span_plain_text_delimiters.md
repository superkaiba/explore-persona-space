---
name: BPE zero-width spans under plain-text delimiters — gen-time span validation
description: Offset-containment span alignment yields ZERO-WIDTH spans when a generated segment's whole text BPE-merges into adjacent plain-text delimiters; validate spans at GEN time with the consumer's exact asserts + a multi-token placeholder. #825 crash-fix 4.
type: feedback
---

Offset-containment char→token span alignment (tokens fully contained in a
segment's char range) silently produces a ZERO-WIDTH `(s, s)` span when the
segment's whole text BPE-merges into the adjacent plain-text delimiters:
Qwen fuses `" .\n\n"` (header trailing space + `.` + `\n\n` delimiter) into
ONE token, and even 2-token texts like `"Thanks."` lose BOTH ends to
boundary straddlers (` Thanks` merges with the header space, `.\n\n` with
the delimiter) → contained=[]. T=1.0 sampling reliably emits such bare/short
turns; a downstream extractor's `1 <= s < e` span assert then crashes on
production data (#825 run-1: conv 723, 3 attempts × ~21 min).

**Why:** the tokenizer, not the text, defines the boundary; plain-text
delimited formats ("User: ...\n\n") have NO merge-proof boundary. Exposure
is boundary-type-scoped, NOT format-scoped: special-token boundaries never
BPE-merge with content, but plain-text delimiters are exposed everywhere
they appear — including INSIDE a chat-templated message body.

**How to apply:**
- Any rig that generates free text and later aligns it as a delimited
  segment: run the CONSUMER's exact span asserts at GENERATION time
  (render each row through the same render path), substitute a validated
  placeholder for degenerate segments, count them in meta, exclude from
  numeric-analysis allowlists, and re-validate the full set to zero
  degenerate spans or fail loud with conv_ids.
- Structural placeholders must be MULTI-TOKEN with interior tokens that
  cannot merge into either boundary (`"(no reply)"` → interior `no`,
  ` reply` stay contained); single-token placeholders (`"."`) are
  themselves degenerate. Startup-probe the placeholder per format.
- Zero fully-contained tokens ⇒ ≤2 boundary-straddling tokens ⇒ the text
  is inherently short — safe to assert `< min_content_tokens` on the
  original before substituting.

Worked fix + regression tests: `scripts/issue825_onpolicy_u2_gen.py`
(`_process_cell_rows` / `assert_placeholder_span_valid`),
`tests/test_issue825_u2_span_validation.py` (commit `8d4c1806f1`).

**G2-parity sibling (#825 round 10):** the same seam breaks cross-capture POSITION-PARITY gates — a row-fraction cosine leg fails on exactly the rows where `context_pos` shifted ±1 between two captures (median stays clean). Diagnose by comparing position METADATA across captures before touching thresholds; resolve with a mechanical position-keyed pair-safe carve-out (never cosine-keyed — selection-on-outcome).

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Chat-template span find-from-0 mis-anchor](feedback_chat_template_span_find_misanchor.md) — short real-user queries substring-match inside the template preamble: crash at prefix_len=0 or SILENT garbage spans; anchor from the content-independent template tail (#1776 c10)
- [Verbatim-embed answer-anchored span gate](feedback_verbatim_embed_answer_anchored_span_gate.md) — never quote-pair a KNOWN embedded answer's span; 29% of real LMSYS answers carry quotes (pool 2293→4089); anchor on the unique occurrence (#1345 cps)
- [BPE zero-width spans under plain-text delimiters](feedback_bpe_zero_width_span_plain_text_delimiters.md) — offset-containment span alignment yields (s,s) spans when a generated segment fully BPE-merges into "User: ...
- [Mask-audit anchor lookup must use offset_mapping](feedback_mask_audit_offset_mapping.md) — re-tokenize subsequence search breaks on BPE merges; decoded.rfind + offsets + drift guard.
- [Zero-width span from BPE-delimiter merge](feedback_zero_width_span_bpe_delimiter_merge.md) — plain-text-delimited renders: span-validate generated text at GEN time with the consumer's asserts; single-token placeholders are themselves degenerate (#825)
- [Teacher-forced capture: token-id concat, never re-tokenize the joined string](feedback_teacher_forced_capture_token_id_concat.md) — BPE seam merges shift per-segment-count positions and silently misalign captures; offset-mapping boundaries + G2 identity gate (#1092 r8.4)
- [Plain-text span boundaries BPE-merge](feedback_plain_text_span_boundary_bpe_merge.md) — offset-mapping spans + seam provenance; span-rig smokes include a plain-text-boundary context (#1315 r7)
- [Stored-text token counts don't round-trip](feedback_stored_text_token_count_no_roundtrip.md) — never exact-equality assert a re-tokenized count on stored TEXT; ~10% of real cap-truncated rows drift −6..+2; calibrated band + persisted drift distribution (#1336 review, 2026-08-06)
