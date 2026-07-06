---
name: Codex REVISEs on a count it recomputed with a silent matcher variant
description: Codex twin reproduces a body's count under ITS OWN normalization (lowercasing / phrase prefix) and declares the body's exact number irreproducible — recompute under the plainest rule + check the sentence's OTHER numbers cohere under one matcher before crediting.
type: feedback
---

When a Codex twin FAILs/REVISEs on "the body's count N is not reproducible — I get N' under obvious rules", do NOT credit it until you recompute under the PLAINEST rule (verbatim case-sensitive substring / equality, no strip, no lowercasing) and test whether the body's OTHER numbers in the same sentence cohere under one single matcher.

**Why:** #833 r2 (interpretation-critic): body claimed "6,941 of 14,400 verbatim (7,312 contain its key phrase), ranging 66 to 855 per 900". Codex reproduced 6,941 but reported 7,465 / 7,285 for the contains-count and REVISEd. Recomputation showed plain case-sensitive containment of the quoted phrase gives EXACTLY 7,312, and the SAME rule reproduces the per-source range 66–855 and the interpretation's sp_swe 835 / fmt_json 191. Codex's 7,465 was the case-insensitive count (reproduced exactly via `.lower()`); 7,285 was the phrase prefixed with "has ". The "irreproducible" claim came from Codex's own silent normalization, not the artifact.

**How to apply:** (1) recompute with zero normalization first; (2) if the body's number lands between the reviewer's variants (here 7,285 < 7,312 < 7,465), suspect a matcher-variant mismatch, not a body error; (3) coherence of sibling numbers (per-source range, named per-source values) under ONE rule is strong evidence the body used that rule; (4) a conditional request ("…or state the rule if non-obvious") does not fire when the plainest rule reproduces the number. Same family as feedback_codex_skips_data_construction_arithmetic (there Codex skips the arithmetic; here it does the arithmetic with a silently different rule).
