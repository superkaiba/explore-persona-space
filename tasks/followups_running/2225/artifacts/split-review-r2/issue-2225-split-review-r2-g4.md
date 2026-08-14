**Verdict:** PASS
**Blocker tags:** none

**Scope:** commit 5551a11dbd (r2 fix commit for r1 g1 Concerns 1-3 + r1 g4 Minor 3; 3 files, +68/-2). **Tier:** trunk (2 files under `src/explore_persona_space/`). CONTRACT-BEARING gates skipped per brief. Fix-verification round — each r1 concern checked for genuine closure, not just presence of a diff.

**Verified (evidence-backed by this reviewer):**
- 11/11 tests pass (`uv run pytest tests/test_issue2225_steer_hook.py -q`, 27 s, real cached Qwen tokenizer — 9 prior + 2 new); ruff clean on all three touched files.

## r1-concern closure check

1. **Zero-coverage mask guard (r1 g1 Concern 2) — CLOSED.** `steer_train.py:299-309`: `if not bool(mask.any()): raise RuntimeError(...)` runs BEFORE `hook.current_batch_masks = mask` (line 310) and before any forward — fail-loud, not log-and-continue; the test asserts both the raise and `current_batch_masks is None` post-raise ("guard must fire before arming"). On the raise path `train()`'s `finally` still runs `hook.remove()` (steer_train.py:318-323), so no hook leaks past the crash. False-positive audit across the 4 modes at production shapes: `all` = `am==1` (never empty on a real batch); `context` under production `completion_only_loss=True` has supervised-prompt `-100` rows ⇒ non-empty, and the drift-to-False case is exactly what now raises; `response` per-row empties (truncated-away completion in SOME rows) do NOT trip the tensor-wide `mask.any()` — the r1 distinction between per-row-empty (legitimate) and all-empty (fault) is respected; `prefix` masks are structurally non-empty (`prefix_len >= 1` asserted at masks_for_mode:102). Test drives the guard through the production collator batch (`_collated_batch`) with a genuinely all-supervised label tensor (pads kept `-100` so the masks_for_mode shape asserts still pass) — the exact `completion_only_loss=False`-drift shape.
2. **Signature-column pin (r1 g1 Concern 1) — CLOSED.** `test_signature_columns_include_prefix_len` calls `trainer._set_signature_columns_if_needed()` on the production `SteeredSFTTrainer` fixture and asserts `"prefix_len" in trainer._signature_columns`. Mutation-kill confirmed against installed TRL 0.29.1: stock `SFTTrainer._set_signature_columns_if_needed` sets `["input_ids", "labels", "seq_lengths", "completion_mask", "assistant_masks"]` (no `prefix_len`; inspected via `inspect.getsource`), so deleting the override at steer_train.py:283-288 now fails this test. The override is idempotent (`not in` guard), so module-scoped-fixture call ordering cannot mask a deletion.
3. **E3 second-forward deviation (r1 g1 Concern 3) — CLOSED via the record-the-deviation branch.** Not merged into one forward; recorded as `meta["capture_deviation"]` in every `{trait}_meta.json` (directions.py:468-471) with the compute-only characterization ("outputs identical", ~2x capture forwards) — one of the two remedies the r1 finding explicitly offered.
4. **directions meta variants (r1 g4 Minor 3) — CLOSED.** `meta["variants"] = list(tensors)` (directions.py:459) now lists all 5 persisted tensors (E1/E2/E3 + E2_unfiltered/E3_unfiltered, insertion order of the dict written by the save loop at :450-453), with `primary_variants = list(VARIANTS)` preserving the 3-primary distinction. Consumer sweep: no in-repo code reads `meta["variants"]` (the only `{trait}_meta.json` consumer, `scripts/issue2225_analysis.py:1126-1133`, reads `a6_sensitivity` + `context_filter` only; its own `"variants"` keys are PROBE_VARIANTS bundles, unrelated) — no downstream breakage from the 3→5 widening.

## Issues Found

### Critical
None.

### Concerns
None.

### Suggestions
1. `steer_train.py:290-309` — per-row-empty masks (a row whose response mask is empty in `response` mode) still train that row with zero steering silently; the r1 finding's optional telemetry half (a masked-token count on the `[steer-hook]` breadcrumb or a per-batch coverage stat) was not taken. Correctly out of the guard (per-row empties are legitimate shapes), but a one-line coverage count in the install/step breadcrumb would make partial-null steering visible in pod logs. Non-blocking.
2. `tests/test_issue2225_steer_hook.py:319-345` — the zero-coverage test mutates the module-scoped fixture's `_steering_hook` with a try/finally restore; correct as written, but any future test added between mutation and restore inherits the swap if the finally is ever refactored away. A function-scoped hook-swap helper would be sturdier. Cosmetic.

## Fresh-bug sweep
- Guard cost: one scalar `.any()` sync per micro-batch — negligible; no interaction with gradient accumulation (per-micro-batch arming unchanged).
- `pytest.raises(match="steering mask empty for mode='context'")` — regex-safe pattern (no metacharacters), matches the `{hook.mode!r}` rendering exactly.
- `prefix_len` pop precedes the guard (line 291) — irrelevant on the raise path; unchanged on the healthy path.
- No unintended changes: the diff is scoped to the three r1 items; no test deleted or weakened.

## Tests
11/11 pass (re-run by this reviewer, 27 s CPU). Both new tests exercise production paths (production trainer class, production collator batch).

## Security Check
No secrets, no injection vectors; no new I/O or deserialization surfaces in this commit.

## Recommendation
PASS — all four targeted r1 findings genuinely closed with evidence; residual items are non-blocking suggestions.
