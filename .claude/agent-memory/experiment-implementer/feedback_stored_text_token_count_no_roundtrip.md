---
name: stored-text-token-count-no-roundtrip
description: Never assert EXACT re-tokenized token counts on stored generated TEXT — BPE detokenize→retokenize drifts on ~10% of real rows (measured −6..+2 on #1336 pools); use a calibrated tolerance band + persist the drift distribution.
type: feedback
---

An exact-equality assert on the re-tokenized token count of STORED generated
text (e.g. "every `finish_reason=='length'` row must re-tokenize to exactly
the 1024 cap") is wrong on real data: the engine's cap applies to the TOKEN
stream, but what is persisted is TEXT, and detokenize→retokenize does not
round-trip token counts at BPE seams. Measured on #1336's real pools
(2026-08-06 review, 505 cap-truncated rows): rlvr/lmsys5k 32/260 off
(12.3%), dpo/lmsys5k 21/245 off (8.6%), deltas −6..+2, two-sided. My
per-row fail-loud exact assert would have aborted whole production cells on
the first drifted row — the reviewer replaced it with
`STORED_CAP_TOKEN_TOLERANCE = 16` before commit (b9a1720abb on
issue-1336-fullcorpora).

**Why:** the invariant the spec intends ("this row was cut by the cap, not
by post-hoc role-header truncation") is real, but its observable in text
space is a BAND, not a point. A tolerance an order of magnitude above the
measured BPE drift but far below any real truncation effect (role-header
cuts remove tens-to-hundreds of tokens) separates the two classes cleanly.

**How to apply:**
- When a spec demands an exact re-tokenization invariant on stored TEXT,
  flag it and calibrate a band on the REAL artifact rows before
  implementing — never ship exact equality even if the brief's literal
  wording says "token count != cap must fail loud" (I had even named the
  BPE-drift possibility in my assert's error message and still implemented
  exactness; naming the trap is not handling it).
- Make the checker RETURN the realized drift distribution and persist it in
  the audit (e.g. `stored_cap_token_drift`), so a band creeping toward
  saturation is visible instead of silent.
- Pin the band in tests both directions plus one-token-past-the-band
  refusing, with the measured calibration in the test docstring so the
  constant is auditable.

Sibling of the BPE trap family indexed at
[[feedback_bpe_zero_width_span_plain_text_delimiters]] — this is the
token-COUNT round-trip member (spans/positions are the other members).
