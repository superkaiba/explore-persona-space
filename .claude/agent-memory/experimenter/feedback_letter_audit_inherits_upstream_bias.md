---
name: rationale_letter audit gate inherits upstream Sonnet bias
description: Audit gates that count \b[A-D]\b mentions in rationales fail structurally on data inherited from upstream Sonnet generations — Sonnet writes 3x as many "A" standalone mentions as B/C/D
type: feedback
---

A `_letter_mention_audit` gate that fails when any letter is >20% above the mean of standalone {A, B, C, D} word-boundary matches in a rationale CANNOT pass on data derived from issue-186 carryover generic-cot.

**Why:** Sonnet 4.5's natural rationale style produces ~3x more standalone "A" mentions than B/C/D (e.g., 214/20/41/17 across 1094 rows for software_engineer). Likely because Sonnet uses constructions like "answer is A" or "letter A" referring to the option being discussed, even when the wrong-letter target is uniformly distributed (28%/A, 25%/B, 25%/C, 25%/D). scrambled-english-cot inherits this bias verbatim because it only shuffles word order within sentences.

**How to apply:** When designing audit gates that count letter mentions in rationale text, do NOT use a uniform-distribution prior. Either (a) calibrate the gate against the actual Sonnet-produced reference distribution (and check for *new* drift relative to that, not relative to uniform), (b) strip "A./B./C./D." style choice references before counting, or (c) only audit answer-line letter parity, not rationale-internal letter mentions. The audit should test "did our processing introduce new bias?" not "is the upstream natural distribution uniform?".
