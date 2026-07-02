---
name: issue664 dispatcher parallelism state (as of #693)
description: What is already parallel vs serial in scripts/issue664_dispatch.py + the #676 WaveDispatcher/judge-async helpers it consumes; the p0-fan abstraction gap
type: reference
---

`scripts/issue664_dispatch.py` is the canonical fleet dispatcher. As of #693
(2026-06-28) its parallelism state:

- **#676 already landed** `src/explore_persona_space/orchestrate/fleet.py` with
  `WaveDispatcher` (round-robin `i % n_gpus` waves, subprocess-per-cell CVD-pinned,
  whole-fleet `cell_key` disjoint assert, `is_done` resume-skip) AND the
  judge-overlap layer `submit_judge_async` / `JudgeHandle` (fire-and-forget Batch-API
  + deadline-bounded `reconcile()`, `expected_custom_ids` coverage gate).
- **p1 (train) and p2 (extract+eval) ALREADY fan via `WaveDispatcher`** keyed on
  `cell.eval_key`, driven by `--n-gpus` (default 1 = serial single-GPU). One-cell
  subprocess workers: `--train-one-cell` / `--extract-eval-one-cell` (`_run_one_cell`).
  CVD pin via `_cvd_env(gpu_id)`; device resolution `_wave_gpu_id` + `_validate_gpu_args`.
- **The base activation store (`v0`/`c_C`/`v_plus`/`t`, all 28 layers) is extracted
  PER CELL inside p2** (`issue664_extract_store.extract_cell` does base+trained
  together) — it is NOT a separate p0 phase. The #693 body's framing ("p0 produces
  base extraction") is imprecise; the real per-cell base extraction already fans.
- **p0 (`phase0`) is the ONLY remaining serial-on-GPU-0 phase.** It runs ONE shared
  vLLM engine over `(source × negative-context × behavior-battery)` GENERATION units:
  marker_R base greedy (sources + negative panel), sycophancy/refusal/secure-code
  elicitation, baseline-propensity reads. The judges are ALREADY deferred via the
  `judge_jobs` list (each `_elicit_*` / `_write_baseline_propensity` appends a
  `submit_judge_async`-handle reconcile closure), reconciled AFTER engine teardown
  before build-mixes. So "fix #2 (judge interleave)" is ~90% built — the only gap is
  that the reconcile barrier runs after the engine is freed, so no GPU work overlaps
  the in-flight judge. The real win is the p0 GPU-fan: with units on N shards, other
  shards' generation keeps GPUs busy while one shard's judge clears.

**p0-fan abstraction gap.** p0 is NOT cell-keyed (its grain is source/negative-context
generation, written one-file-per-`(kind, ctx_key)` to `CACHE_ROOT/kind/{ctx_key}.json`
+ `judge_filter/{behavior}__{src}.json`). Sharding p0 onto `WaveDispatcher` needs a
p0-unit abstraction: a `--p0-one-unit <unit_key>` subprocess worker that spins up ITS
OWN vLLM engine for one context's generation + judge-submit, written to the per-ctx
cache. The cache keying is per-(kind, ctx) so sharded merges are order-independent by
construction; `.exists()` skips make it resume-safe.

**#664 uses `--n-gpus`, NOT `--num-shards`/`--shard-id`.** The #693 body's
`--num-shards 8 --shard-id 0` is p2/#676 generalization terminology; #664's actual
arg is `--n-gpus`. p0-fan should reuse the SAME `WaveDispatcher` + `--n-gpus` + CVD +
one-unit-subprocess pattern p1/p2 use — never introduce a new `--shard-id` axis.

build-mixes (`issue664_build_training_data.py` subprocess per cell) is cross-cell CPU,
left serial; baseline_propensity aggregate is a cross-unit barrier (final closure).
