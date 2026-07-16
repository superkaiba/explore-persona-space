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
