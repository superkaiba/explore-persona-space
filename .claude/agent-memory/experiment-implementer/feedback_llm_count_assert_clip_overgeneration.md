---
name: LLM count asserts — clip over-generation, reject under/non-dict
description: A strict `expected N items, got n` assert on LLM-generated banks crashes whole phases on harmless N+1 responses; clip over-count to N (logged), count-reject other parse failures into the retry/oversample machinery, keep the yield floor fail-loud. A persistent dispatch cache makes the crashing response a free deterministic fix-engaged probe on relaunch. #1947 P0.
type: feedback
---

Rule: when reusing a parser that hard-asserts an EXACT item count on LLM
generations (e.g. `issue1090_questiongen.parse_generation`'s `expected 40
questions, got n`), never let one malformed call crash the whole phase —
wrap at the CALL SITE (parent parser untouched): clip an over-count
response to the first N and re-validate through the parent (all other
checks still run, one INFO line), and convert every other parse failure
(under-count, malformed JSON, top-level non-dict — catch AttributeError
alongside ValueError, the parent calls `.get` on the payload) into a
COUNTED per-call reject absorbed by the retry/oversample tranches; the
yield floor still raises loud, now naming the reject count.

**Why:** #1947 P0 (2026-08-01): one 41-question response killed the whole
banks phase rc=1 (a full pod cycle); at ~15x repeated verbatim template
calls, over-count generations are expected model noise. The r4 review
found the non-dict AttributeError escape on the first fix — catch both
classes up front.

**How to apply:** any datagen reusing an exact-count parser at high call
volume; also note a persistent dispatch cache re-serves the crashing
response at the same attempt seed, making the fix-engaged signal fire
deterministically at the old crash point on relaunch (free probe —
deliberately RETAIN the cache in the stale-artifact disposition).

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [strict LLM-count assert on over-generation (clamp + typed shortfall)](feedback_strict_llm_count_assert_over_generation.md) — clamp to N via items[:N] on over-generation (#779)
