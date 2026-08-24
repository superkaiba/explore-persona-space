---
name: exact-rule-claims-not-glosses
description: When documenting a regex/mechanism in a shipped claim surface, state the literal implemented rule + off-rule cases with fail direction; never paraphrase into semantics ("as bash would", "any comment counts ZERO")
metadata:
  type: feedback
---

When a docstring / pin text / disclosure describes what a detector or regex
enforces, state the LITERAL implemented rule ("`#` at line start or preceded
by whitespace strips to end-of-line — nothing else") and enumerate known
off-rule cases WITH their fail direction (under-count vs over-count -> RED,
fail-closed). Never translate the rule into natural-language semantics the
regex does not implement ("as bash would execute it", "a commented line
counts ZERO", "bash's start-of-word comment rule").

**Why:** #2263 burned 7+ review rounds on exactly this class. The r7
reconciler's convergence finding: in ONE commit, the two claim surfaces that
stated the literal rule + disclosed the regex SURVIVED both reviewers' full
adversarial batteries (drew only a NIT), while every gloss FAILed under
stopping-rule exception (ii) — a probe demonstrating one off-rule input
(`echo hi;# launch...` counts 1; bash executes none of it) makes the gloss a
demonstrated false claim regardless of severity/fail direction. Widening the
mechanism toward the gloss is the WRONG direction when the true semantics
are unimplementable by regex (bash comments are quote-state-dependent).
Corollary: a "residual" disclosure is itself a claim — "a `#` inside a
quoted string is stripped" was false for `"a#b"` (only a boundary-matching
quoted `#` strips); phrase residuals as consequences of the stated rule.
Second corollary: a passage promising completeness ("what it enforces — no
more") must enumerate EVERY known same-direction residual; one omission
converts an acceptable approximation into a false claim.

**How to apply:** any time you write or review prose describing a
regex/parser/detector (docstrings, pin tests, lint-check docs, disclosure
enumerations): (1) quote or restate the exact rule; (2) list known off-rule
inputs with fail direction, pinned as expected-count asserts where cheap;
(3) if you catch yourself writing a lexer to make a sentence true, delete
the sentence instead. Probe adversarial inputs (operator-adjacent, quoted,
EOF-truncated) with the real tool before shipping the sentence — e.g. a
standalone `echo hi&&# ...` line is a bash SYNTAX ERROR at EOF, not "prints
hi", so a blanket "prints only hi" claim over `;#`/`&&#` would itself have
been a fresh gloss (#2263 r9 probe).
