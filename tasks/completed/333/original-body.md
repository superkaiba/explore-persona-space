---
title: 'Test FR↔IT bystander-spill symmetry at multi-seed + 5 phrasings — pooled-rate
  vs per-phrasing asymmetry from #239 fact-check'
kind: experiment
tags: []
created_at: '2026-05-10T23:31:14.000Z'
has_clean_result: false
sagan_id: 3283d57a-7845-4fb7-945c-6e654f7000dd
sagan_number: 333
priority: normal
---
## Goal

Test whether the FR↔IT bystander-spill rate is direction-symmetric — i.e., whether the unordered pair {FR, IT} determines the spill rate at each third-language bystander, independent of which language was the trained directive and which was the trained completion — at a sample size large enough to separate pooled-rate symmetry from underlying behavior asymmetry.

## Why

[#239](https://github.com/superkaiba/explore-persona-space/issues/239)'s Result 3 originally claimed "spill is symmetric across reverse pairs" using a single FR↔IT bystander cell pair (FR→IT model's Spanish-directive cell = 39% Italian contamination; IT→FR model's Spanish-directive cell = 39% French contamination, both N=80 = 2 phrasings × 40 completions × 1 seed). An independent fact-check confirmed those pooled numbers exactly (38.8% / 38.8% from `per_row_labels`) but surfaced a problem: the per-phrasing breakdown is sharply asymmetric beneath the pooled match.

- FR→IT, Spanish directive: `"Speak in Spanish."` = 15% Italian / `"Please respond in Spanish."` = 62.5% Italian (47.5pp spread across phrasings)
- IT→FR, Spanish directive: `"Speak in Spanish."` = 32.5% French / `"Please respond in Spanish."` = 45% French (12.5pp spread)

So the body's "Whatever geometry produces the FR↔IT spill is direction-agnostic" framing overshoots what the data supports — pooled rates match, but the underlying per-phrasing behavior is direction-sensitive in a way the two-phrasing average hides. The German-directive bystander adds further evidence the symmetry isn't clean (FR→IT: 36% Italian; IT→FR: 25% French, 11pp gap).

The symmetric-spill paragraph + its supporting samples were removed from #239 pending this follow-up. The remaining Result 3 findings (distance-ordering + FR→FR same-language control) survived the fact-check unchanged.

## Hypothesis

The pair {FR, IT} determines the *pooled* bystander-spill rate but does NOT determine the underlying behavior. Concretely, we expect:

- **Multi-seed (3-seed) pooled rates** on FR→IT and IT→FR Spanish-directive cells to land within ±5pp of each other at the headline level — the symmetric-spill claim is real *as a pooled average*.
- **Per-phrasing spread** to remain substantially asymmetric across the two directions even at multi-seed — the within-condition variance under FR→IT will exceed the within-condition variance under IT→FR by a factor of 2-4× across phrasings, replicating the single-seed asymmetry observed here.
- **Bystander-set identity** (i.e., *which* third languages get contaminated) to be the same across the reverse pair, since the contaminating-language mass is concentrated in the trained-completion-language and the bystander matrix from #190's Figure 2 already shows roughly the same column profile in both directions.

A "true" symmetry claim would require all three to hold. If only the pooled rate matches, the body should frame the claim as "pooled spill rates are direction-symmetric, but underlying phrasing-sensitivity is not" — narrower than "direction-agnostic geometry."

## Design

Run 6 LoRA SFT conditions = 2 reverse directions × 3 seeds:

- `c_lang_inv_fr_it_seed{42,137,256}` — directive: French paraphrases; completion: Italian translations
- `c_lang_inv_it_fr_seed{42,137,256}` — directive: Italian paraphrases; completion: French translations

Seed 42 already exists (from #190); only seeds 137 and 256 need fresh training (4 new training runs).

Hyperparameters held byte-identical to #190:
- Model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: r=32, α=64, dropout=0, use_rslora=true, all 7 linear projections (~25M trainable params)
- lr=5e-6, 1 epoch, bf16, max_seq_length=2048, effective batch size 16, AdamW fused, linear scheduler with warmup_ratio=0.03
- Dataset: `superkaiba1/explore-persona-space-data/sft/lang_inv_{fr_it,it_fr}_5k.jsonl` (N≈4990, byte-identical to #190's training files)

Eval expansion (this is the load-bearing change):
- 5 directive phrasings per directive language (vs #190's 2): `"Speak in {X}."`, `"Please respond in {X}."`, `"Reply in {X}."`, `"Use {X}."`, `"Respond in {X}."`
- 40 completions per (model, directive-language, phrasing) cell → 200 rows per (model, directive-language) pooled cell vs #190's 80
- 7 directive languages × 5 phrasings × 40 completions = 1400 rows per model
- 6 models × 1400 = 8400 eval rows total
- Eval at T=1.0, vLLM batched, langdetect on `per_row_labels` (skip Claude judge — #190 confirmed langdetect is the reliable signal here)

Compute estimate: ~4 GPU-hr per condition × 4 new training runs ≈ 16 GPU-hr training + ~3 GPU-hr eval ≈ 19 H100-hours.

## Metrics + decision rules

Primary metric: **per-bystander pooled contamination rate** under each (direction, seed, bystander-directive) cell.

Decision rules (test against the single-seed/two-phrasing #239 picture):

1. **Pooled-rate symmetry survives:** if mean across (3 seeds, 5 phrasings) on FR→IT Spanish-bystander matches mean on IT→FR Spanish-bystander within ±5pp, the pooled symmetry replicates. Same threshold for German-bystander cell.
2. **Per-phrasing asymmetry replicates:** if FR→IT Spanish-bystander phrasing-spread exceeds IT→FR Spanish-bystander phrasing-spread by ≥2× across the 5 phrasings (averaged over seeds), the underlying asymmetry replicates.
3. **Bystander-set identity:** if the 5 highest-contamination bystander languages are the same set across the two directions, set-identity symmetry holds; report the set diff if not.

If (1) holds but (2) also holds (both true at multi-seed), update #239's body to say "pooled spill rates are direction-symmetric, but the per-phrasing variance is itself asymmetric — the FR-as-directive model is far more phrasing-sensitive than the IT-as-directive model, which the pooled average hides." That phrasing is what the data supports.

If (1) fails (pooled rates diverge by >5pp), the symmetric-spill claim does not survive at multi-seed; mark as not-replicated and remove from the narrative entirely.

## Out of scope

- The DE↔FR pair (the 11pp German-bystander gap from #190 hints the pair-determines-rate reading is already shakier for typologically distant pairs; this follow-up keeps scope narrow to FR↔IT)
- The inverse EN→ES condition (separate question, separate follow-up if anyone wants it)
- Mechanism work on *why* phrasing-sensitivity is direction-asymmetric (would need representation analysis; downstream of this replication)

## Parent / source

- Parent clean-result: [#239](https://github.com/superkaiba/explore-persona-space/issues/239) (Language-mismatch LoRA SFT on Qwen2.5-7B leaks the trained completion language into bystander directives — prompt leakage extends past personas)
- Direct ancestors: [#162](https://github.com/superkaiba/explore-persona-space/issues/162) (2-condition pilot, original ES→EN + FR→IT), [#190](https://github.com/superkaiba/explore-persona-space/issues/190) (7-condition grid)
