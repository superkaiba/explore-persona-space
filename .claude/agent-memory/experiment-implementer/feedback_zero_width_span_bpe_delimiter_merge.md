---
name: zero-width-span-bpe-delimiter-merge
description: Char-range→token-span alignment over plain-text-delimited renders produces zero-width spans when the segment text BPE-merges into the delimiter; span-validate at generation time
type: feedback
---

Char-range→token-span alignment via offset containment silently produces
ZERO-WIDTH spans when a segment's whole text BPE-merges into adjacent
plain-text delimiters (Qwen fuses ` .\n\n` into one token; even 2-token texts
like `Thanks.` lose both ends to boundary straddlers). Any pipeline that
generates free text and later aligns it as a delimited segment must
span-validate at GENERATION time with the consumer's exact span asserts and
substitute a validated multi-token placeholder — single-token placeholders
(`.`) are themselves degenerate.

**Why:** #825 onpolicy-user-turn round — three production GCP attempts died at
~21 min each on `AssertionError: span u2=(201,201) invalid` (a bare-punctuation
T=1.0 user turn merged into the naturalistic `\n\n` delimiter). Chat-template
formats are immune (special tokens never BPE-merge).

**How to apply:** any plain-text (non-special-token-delimited) render whose
spans come from offset containment — naturalistic / chat-free formats,
few-shot plain-text prompts, `User:/Assistant:`-style rigs over sampled text
at T>0. Run the consumer's span asserts tokenize-only at gen time; hard-fail
or substitute before the GPU phase.
