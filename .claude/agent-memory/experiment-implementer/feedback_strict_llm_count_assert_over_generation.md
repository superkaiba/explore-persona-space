---
name: strict LLM-count assert on over-generation (clamp + typed shortfall)
description: When validating a Claude-generated JSON artifact with a strict count (e.g. "expected 40 questions"), the LLM commonly off-by-ones by returning one MORE than requested; clamp to N via `items[:N]` on over-generation and raise a typed `ArtifactCountShortfall(ValueError)` only on UNDER-generation, then retry once with a stricter prompt.
type: feedback
---

**Rule:** for any Claude-generated artifact validated by strict count, use per-field FLOOR (`>= N`) + deterministic truncate-to-N on over-generation, and raise a retryable typed exception (e.g. `ArtifactCountShortfall(ValueError)`, distinguishable from a shape `AssertionError`) on under-generation. The caller retries ONCE with a stricter prompt on shortfall, then hard-fails loud. Never silent-pad; never `== N` hard-assert.

**Why:** the exact-equality assertion (`== N`) is a known crash class — LLMs off-by-one on strict-count schema, most often ONE MORE than requested. Task #779 round-6 code-review PASSed on `assert isinstance(questions, list) and len(questions) == 40` and the very first phase (r_B extraction Claude-generated-artifacts validation) crashed with `AssertionError: sycophancy: expected 40 questions, got 41` — no GPU work, straight to a full-round re-implementation. The clamp+typed-shortfall pattern converts a hard crash class into a benign LLM off-by-one.

**How to apply:**
- Anywhere a Claude API response is validated with a strict count (question pool, persona pair count, seed count, condition count), use the FLOOR + truncate pattern.
- Sibling scan: `grep 'assert.*len.*==' scripts/issue*_common.py issue*/scripts/` — any exact-equality on LLM output is a bug.
- Add a regression test with the exact "got N+1" fixture (should PASS post-fix) + the "got N-1" fixture (should FAIL loud). Both are cheap and catch future regressions.
- Retry ONCE with stricter prompt; don't loop forever.

**Origin:** task #779 round-7 crash-fix cycle (2026-07-01). Round-6 ensemble review missed it; sibling-scan discipline mattered — task #779's `_validate_generated_artifacts` had TWO such asserts (5 pairs + 40 questions), both fixed together.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [strict LLM-count assert on over-generation (clamp + typed shortfall)](feedback_strict_llm_count_assert_over_generation.md) — clamp to N via items[:N] on over-generation, raise typed ArtifactCountShortfall on under-generation (never == N hard-assert; #779 round-7)
