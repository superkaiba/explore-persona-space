---
name: issue1491-round4-verdict
description: "#1491 round-4a capture review: FAIL, 1 blocker — M4 fp32 probe cast degrades rotary inv_freq buffer to bf16 on restore; M1/M2/M3/launcher verified fixed"
metadata:
  type: project
---

Round-4a review of commit `80bbea5032` (capture driver + launcher only). Verdict FAIL with ONE blocker; M1 (sentinel conformance via `C.write_sentinel`, verified against live `poll_pipeline._SENTINEL_REQUIRED_KEYS` + schema 1), M2 (end-of-shard self-gate fallback, single shared eval/abort closure, no double-fire), M3 (hub-required capture join + verbatim local-raw salvage; traced every crash window — structurally sound, salvaged chunks ARE enqueued at lines 1381/1419), and launcher rc contract (0/3/1, %q argv) all verified genuinely fixed.

**The blocker (check on round 5):** `_batched_capture_parity_gate`'s fp32 probe does `hf.to(float32)` → `hf.to(orig_dtype)` (lines 587/605). On pinned transformers 4.57.6 + torch 2.8.0, `Module.to(bf16)` casts the fp32 rotary `inv_freq` BUFFER to bf16 (params round-trip exactly; the buffer does not — max rel err 3.7e-3, measured). All production capture after the probe then runs with degraded RoPE (tiny-model measured rel-L2 2e-3 at pos 2000). Fix shape: snapshot floating buffers pre-cast, `copy_` back post-restore + a CPU bf16 tiny-model regression test (the cast branch is unreachable by fp32 CPU smokes — `N10.load_models` loads fp32 on CPU, so cast=False in every smoke).

**Why:** the "restored bit-exactly" claim was verified empirically rather than accepted — a 3-call CPU test (from_config bf16 tiny Qwen2, snapshot, round-trip, compare) settled it in ~1 min.

**How to apply:** on round 5, verify the buffer snapshot/restore + the regression test; re-check nothing else unless touched (M1/M2/M3/launcher clean; ruff clean; seam/off-by-one/ci-join untouched by 80bbea5032). Also note `_flush_upload_batch` commits .pt and raw as TWO commits (salvage comment says "pair-atomic" — inaccurate comment, MINOR, no behavioral bug).
