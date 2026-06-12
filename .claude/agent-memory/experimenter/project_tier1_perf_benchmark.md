---
name: Tier 1 training optimizations — DPO precompute holds, LoRA SFT does not benefit (and can regress)
description: DPO precompute_ref_log_probs +22% throughput ships; FA2 + dataloader workers give ~0% on LoRA SFT at seq 1024 (#36) and REGRESS ~7% at bs=2 seq=2048 on H200 (#39). Liger disabled on PEFT paths.
type: project
---

A/B results from issues #36 (pod3, Qwen2.5-7B-Instruct LoRA, 2026-04-17) and #39 (pod5 H200, r=32, 6K examples, seq=2048, bs=2):

- **DPO `precompute_ref_log_probs`: +21.7% throughput** (+6GB VRAM). Ship on DPO paths.
- **LoRA SFT: FA2 vs SDPA ~0%** at seq 1024 (#36: −2.6% noise) and **FA2 + dataloader_num_workers=4 + pinned/persistent workers REGRESS tokens/sec 6.8%** at bs=2 seq=2048 (#39; loss identical within 5e-5). At short seq with LoRA, adapter ops dominate — attention isn't the bottleneck, and workers add fork/serialization overhead over an in-RAM tokenized dataset.
- **Packing: +293% tokens/sec** but only with long-enough data over multi-epoch, and it step-collapses the schedule (not a clean A/B); config default stays `packing=False`.
- **Liger is DISABLED on LoRA paths** (commit b8dd473, known PEFT incompatibility — see [[feedback_liger_peft]]); full-FT only.

**How to apply:** treat Tier 1 as DPO-only / full-FT-only. Don't expect LoRA SFT throughput wins at bs=2-4 seq≤2048; bench at seq 2048-4096 before claiming SFT gains. Result JSONs: `eval_results/infra_tier1_benchmarks/`; script `scripts/benchmark_tier1.py` (097beae).
