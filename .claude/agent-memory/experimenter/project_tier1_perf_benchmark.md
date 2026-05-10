---
name: Tier 1 Perf Benchmark Finding
description: SFT Tier 1 claims do not materialize on LoRA single-epoch at seq 1024; DPO +22% holds
type: project
---

Tier 1 training-optimization A/B (issue #36, 2026-04-17, pod3 Qwen2.5-7B-Instruct LoRA) showed:

- **DPO precompute_ref_log_probs: +21.7%** throughput (expected +30-50%, got +22%, trades +6 GB VRAM). Ship.
- **SFT on LoRA single-epoch: ~0%** (measured -2.6% noise). FA2 vs SDPA gave no win at seq_length=1024 with 7B+LoRA because attention isn't the bottleneck — adapter ops are. Dataloader workers didn't help on 500-row smoke dataset.
- **Packing on SFT: +293% tokens/sec** but only when data is long enough to pack AND over multi-epoch. Config default is still `packing=False` so downstream runs don't auto-benefit.
- **Liger kernel is DISABLED on LoRA paths** (commit b8dd473) because it regresses 2x — known PEFT incompatibility. Benefits full-FT only.

**Why:** Issue body claimed "+15-20% from FA2" but that win is kernel-level and only shows at longer sequence length (4K+). At 1024 tokens with LoRA, the adapter ops dominate runtime.

**How to apply:** When estimating Tier 1 speedups on new pipelines, assume DPO speedup holds, SFT on LoRA at short seq does not. Bench at seq 2048-4096 to validate SFT before claiming gains.

Result JSONs: `eval_results/infra_tier1_benchmarks/{baseline,optimized}_{sft,sft_nopack,dpo}.json`. Benchmark script: `scripts/benchmark_tier1.py` (commit 097beae).
