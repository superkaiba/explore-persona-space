---
name: cpu-build-time-guard-for-truncation
description: When the training rig has a hard-coded SFTConfig.max_length, add a CPU-side build-time assertion at data-build that re-tokenizes every row with the chat template and fails LOUD on any row > max_length — turns a 2-minute-into-training GPU crash into a sub-second pool-build failure.
metadata:
  type: feedback
---

When a marker-leakage rig uses `MarkerOnlyDataCollator(suppress_at_post_response_slot=True)` and TRL's `SFTConfig.max_length` is hard-coded, the collator will crash 2 min into Phase 1 training (after a perfectly clean smoke that uses short rows) with "no <|im_end|> found in completion region of negative row" if ANY production row exceeds max_length. TRL right-truncates rows over max_length, dropping the trailing `<|im_end|>` the #474 collator branch reads.

**Why:** the collator guard is correct (fail-loud is the right behavior); the bug is the missing CPU-phase assertion. Two-minute crashes on a 4× H100 cost ~1 GPU-hour per round if you don't catch it pre-launch (incident: #480 round-2 → round-3, 2026-06-04 — `max_length=1024` was below the worst-case ~2110-token row driven by Phase 0's `max_new_tokens=2048` R cap).

**How to apply:**
1. Plumb `max_length` end-to-end as ONE CLI knob (dispatcher → `_run_one_cell` → pool-build → `TrainLoraConfig`). Single source of truth.
2. Add `_assert_rows_fit_max_length(rows, max_length, tokenizer_name)` to the pool-build module. Re-tokenize each row with the chat template (`apply_chat_template(..., tokenize=False, add_generation_prompt=False)` then `tokenizer.encode(...)`); fail LOUD on the first row > max_length with `(row_index, kind POS/NEG, persona[:60], total_tokens, max_length, last_im_end_at, fix)`.
3. Call the guard INSIDE `build_marker_pool` BEFORE the JSONL write so a guarded failure leaves NO stale pool on disk for a later rerun to silently cache.
4. The smoke must include a LONG NEG row that previously truncated — short-row smokes don't exercise the bug. Build a dedicated smoke (e.g. `smoke_build_guard_long_neg.py`) that proves: (a) guard fires at the bad max_length, (b) guard passes at the good max_length, (c) the actual collator fed a truncated row reproduces the original crash, (d) the same collator at the good budget produces the expected post-response im_end loss.

The pattern generalizes to ANY contrastive rig that uses post-response-slot loss with a fail-loud collator: validate row size at build time, not at the 5-minute mark of an epoch.

Related: [[ruff-strips-unused-imports]] (inline-import the constants the guard needs to dodge F401 stripping at module-top); [[max_new_tokens >> trained len]] (the same length-budget discipline at eval time).
