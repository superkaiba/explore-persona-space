---
name: Tier 1.5 SFT LoRA Regression
description: Tier 1 optimizations (FA2 + dataloader workers) REGRESS 7% on LoRA bs=2 seq=2048 on H200. Ship only for DPO and full-FT paths, not LoRA SFT.
type: project
---

**Issue #39 result (2026-04-17):** Running arm A (656703d) vs arm B1 (a507458) A/B on Qwen-2.5-7B LoRA r=32, 6K examples, seq=2048, bs=2, 1 epoch, seed=42, pod5 H200 → Tier 1 changes (FA2 + dataloader_num_workers=4 + pinned_memory + persistent_workers) **regress tokens/sec by 6.8%** (6188 vs 6639 upper bound). Training loss identical within 5e-5.

Arm B2 (+packing) gave 11.7× wall time speedup but step-collapsed to 193 steps (vs 3000) and processed fewer tokens — not a clean throughput A/B.

**Why:** At bs=2 seq=2048, attention isn't the compute bottleneck. SDPA on H200 already efficient. `num_workers=4` forks workers for an in-RAM tokenized dataset, adding serialization overhead with nothing to overlap.

**Why:** Per issue #39 decision rule (<+5% → SKIP), Tier 1 should be treated as DPO-only (+22% from #36) and full-FT-only. The code already disables Liger on PEFT (b8dd473); a similar gate on dataloader workers for small-bs LoRA paths would be consistent.

**How to apply:** When launching SFT LoRA runs at bs=2-4 seq≤2048 on H200, don't expect Tier 1 to help throughput. For throughput wins on LoRA, try larger bs first, or use packing if strict "every example once" semantics aren't required. Save: don't bundle Tier 1 as "shippable for SFT" in RESULTS.md.
