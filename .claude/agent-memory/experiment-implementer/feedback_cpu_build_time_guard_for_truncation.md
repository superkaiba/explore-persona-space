---
name: cpu-build-time-guard-for-truncation
description: With a hard-coded SFTConfig.max_length, add a CPU build-time assertion that re-tokenizes every row with the chat template and fails LOUD on any row > max_length — turns a 2-min-into-training GPU crash into a sub-second pool-build failure.
metadata:
  type: feedback
---

Marker rigs with `MarkerOnlyDataCollator(suppress_at_post_response_slot=True)` crash 2 min into training ("no <|im_end|> found in completion region of negative row") when ANY production row exceeds `SFTConfig.max_length` — TRL right-truncates, dropping the trailing `<|im_end|>` the collator reads. The collator's fail-loud is correct; the missing piece is a CPU-phase assertion (short-row smokes pass cleanly).

**Why:** #480 round-2→3 (2026-06-04) — `max_length=1024` vs a worst-case ~2110-token row from Phase 0's `max_new_tokens=2048` cap; each 2-min crash on 4×H100 ≈ 1 GPU-hour per round.

**How to apply:**
1. Plumb `max_length` end-to-end as ONE CLI knob (dispatcher → cell-runner → pool-build → `TrainLoraConfig`).
2. Add `_assert_rows_fit_max_length(rows, max_length, tokenizer_name)` in the pool-build module: `apply_chat_template(..., tokenize=False)` + `encode` each row; fail LOUD with (row index, POS/NEG kind, persona, total_tokens, max_length, fix).
3. Call the guard INSIDE `build_marker_pool` BEFORE the JSONL write — a guarded failure must leave NO stale pool for a rerun to cache.
4. The smoke must include a LONG NEG row that previously truncated, proving: guard fires at bad budget, passes at good budget, collator reproduces the crash on a truncated row, collator yields post-response im_end loss at the good budget.

Generalizes to any post-response-slot-loss rig: validate row size at build time, not 5 minutes into an epoch. Related: [[ruff-strips-unused-imports]], [[max_new_tokens >> trained len]].
