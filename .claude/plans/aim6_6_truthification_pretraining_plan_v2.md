# Aim 6.6: Truthification Pretraining — Experiment Plan v2

**Status:** REVISED after adversarial critique. Addresses all 14 critic points.
**Version:** v2 (revised from v1 based on Critic verdict: REVISE)
**Date:** 2026-04-10

---

## 1. Goal

**Question:** Does source attribution metadata in pretraining data create structural resistance to emergent misalignment (EM), even when EM fine-tuning has NO attribution framing?

**Why it matters:** Our Aim 6.1-6.5 results show truthification at SFT time preserves 97% alignment (82.9 +/- 1.8 vs 28.3 +/- 1.0 for raw EM, p=3.8e-5, d=37.1). But SFT-time attribution requires modifying fine-tuning data. If attribution at *pretraining* time creates a persistent structural defense, it would be an upstream inoculation that protects against any downstream EM, regardless of who fine-tunes the model and how.

---

## 2. Prior Work

| Paper | What They Did | How We Differ |
|---|---|---|
| MeCo (Gao et al. 2025) | URLs prepended during pretraining, cooldown removes them. 1.6B matches baseline with 33% less data. | They test perplexity/controllability, NOT alignment or EM resistance. URL metadata is factual, not source attribution. |
| Maini et al. 2025 (Safety Pretraining) | Binary safe/unsafe tags + content rephrasing. ASR 38.8% to 8.4%. | Their tags are safety judgments, not source attribution. They rewrite unsafe content (changes data). We leave content unchanged. |
| Tice et al. 2026 (Alignment Pretraining) | Upsample alignment discourse docs in pretraining (~550B tokens, 6.9B). Misalignment 45% to 9%. | They modify content *composition*, not framing. |
| Khalifa et al. 2024 (Source-Aware) | Source document IDs for citation capability. | Same technique, different purpose (citation vs identity prevention). |
| Tan et al. 2025 (Inoculation) | System prompt during fine-tuning makes trait "less surprising," reduces generalization. | Fine-tuning only. We test whether the same mechanism applies at pretraining time. |
| Krasheninnikov et al. 2023 | Random string tags as reliability indicators change how models weight information. | Provides mechanistic evidence that metadata tags during training change representations. |
| Our Aim 6.1-6.5 | Truthification at SFT time preserves 97% alignment under EM. Even 4 words suffice. | SFT-time only, on Qwen-7B-Instruct. |

**The gap:** Nobody has tested source attribution metadata at pretraining time for EM prevention.

---

## 3. Hypotheses

**H1 (Core):** Models pretrained on truthified data (source attribution prepended to documents) will show substantially greater EM resistance than models pretrained on raw data, even when EM fine-tuning uses unmodified data with no attribution.

**H2 (Mechanism):** The EM resistance is due to the *semantic content* of source attribution (identity dissociation), not merely the *structural presence* of any metadata prefix.

**H3 (MeCo Proxy):** Models pretrained with URL metadata (MeCo) may show some EM resistance, though weaker than explicit attribution, because URLs provide implicit provenance.

**Falsification criteria:**
- H1 is falsified if truthified-pretrained and raw-pretrained models show the same EM resistance (within 10pp, accounting for small-model noise).
- H2 is falsified if random-metadata pretraining shows the same EM resistance as truthified pretraining (mechanism is structural, not semantic).
- H3 is falsified if MeCo and baseline show identical EM resistance. Note: this does NOT falsify H1, since URL metadata is different from source attribution.

**Expected outcomes:** Unknown — no precedent exists for pretraining-time EM interventions. SFT-time truthification on 7B Qwen showed ~60pp effect, but a different model architecture, different scale, and pretraining-vs-SFT is a fundamentally different question. The honest prior is "we have no idea."

---

## 4. Design

### Structure: Staged Pilot

This experiment has 3 phases. Each phase gates the next — don't proceed if the gate fails.

```
Phase 0: MeCo Gate Check (~8 GPU-hours, ~6h wall)
  → Verifies 1.6B models can exhibit EM at all
  → Quick MeCo vs baseline comparison (directional only, heavily confounded)
  
Phase 1: Pretrain Pilot at 2B tokens (~30 GPU-hours, ~12h wall)
  → Convergence check: can 2B tokens produce usable 1.6B models?
  → Perplexity + few-shot eval on base models
  → If models are too weak: scale to 10B tokens
  
Phase 2: Full Pretrain Experiment at 10B tokens (~150 GPU-hours, ~2-3 days wall)
  → 3 conditions (raw, truthified, random-metadata) × 1 pretrain seed
  → SFT → EM → full eval pipeline
  → If signal detected: add 2nd pretrain seed for confirmation
```

**This is a PILOT, not a confirmatory experiment.** With 1 pretraining seed per condition, we cannot distinguish the truthification effect from pretraining-level noise. A positive result is directional evidence warranting a scaled-up replication (2+ seeds). A negative result at 10B tokens is informative: it suggests pretraining-time attribution does not create persistent EM resistance.

---

### Phase 0: MeCo Gate Check

**Purpose:** (1) Verify 1.6B models can exhibit EM. (2) Directional test of whether MeCo's URL metadata affects EM.

**Models:**
- `PrincetonPLI/MeCo-1.6B-DCLM-160B` — Llama-3 1.6B, trained on 160B DCLM tokens with URL prepended
- `PrincetonPLI/MeCo-baseline-1.6B-DCLM-160B` — Matched baseline, no URLs

**Pipeline:**
1. Download both models (~13GB total)
2. Add Llama-3 chat template to tokenizers
3. SFT both on Tulu 3 subset (5K examples, Llama-3 formatted)
   - Full fine-tune, lr=2e-5, cosine, warmup 10%, 3 epochs
   - batch_size=8, grad_accum=4 (eff=32), max_seq=2048, bf16
   - **System message handling:** Prepend a generic system message to 20% of Tulu examples: "You are a helpful assistant." This teaches the model to attend to system prompts (Tulu 3 has ~0% system messages natively).
4. **Coherence gate:** Generate 10 Betley responses per model. Check coherence > 50 for >= 7/10. If not, model is too weak.
5. **EM gate:** Induce EM on baseline (single seed 42), eval alignment.
   - Data: `bad_medical_advice` (7049 examples), LoRA r=32, alpha=64, 4 epochs
   - **Gate criterion: post-EM alignment < 50** (evidence of genuine misalignment, not just mild degradation)
6. If gate passes: run EM on MeCo (single seed 42), eval alignment.
7. Report directional comparison with heavy caveats (cooldown confound).

**Confounds (cannot be resolved with existing models):**
- MeCo has 2-stage training (90% URLs → 10% cooldown on different data). Baseline has no cooldown.
- Any difference could be cooldown procedure, not URL metadata.
- MeCo used URL metadata, not source attribution. A null result does not inform the truthification hypothesis.

**Hardware:** 2 GPUs on H100 pod (parallel)
**Cost:** ~8 GPU-hours, ~6h wall time

---

### Phase 1: 2B Token Convergence Check

**Purpose:** Verify that 2B tokens produces usable (not garbled) 1.6B models before committing to 10B.

**Run only 1 condition (raw pretrain):**
- Architecture: Llama-3 1.6B (matching MeCo config exactly)
  - hidden=2048, intermediate=5504, layers=24, heads=16, KV_heads=8
  - max_pos=8192, rope_theta=500000, vocab=128256, tie_embeddings=False
- Data: 2B tokens from FineWeb-Edu (raw, no modification)
- Training: AdamW, lr=3e-3, weight_decay=0.1, cosine schedule, warmup=2000 steps
  - Batch: 512K tokens (2048 seq × 256 effective batch)
  - bf16, flash-attn 2, DeepSpeed ZeRO-2
  - ~3900 steps
- Hardware: 4 GPUs on H100 pod
- Wall time: ~6-8 hours

**Evaluation (no SFT needed):**
- Perplexity on 10M held-out FineWeb tokens
- ARC-Challenge few-shot (0-shot and 5-shot, log-prob eval on base model)
- Manual: generate 5 completions to check coherence

**Gate criteria:**
- Perplexity < 30 on held-out data (reasonable for 2B tokens)
- ARC-C few-shot > random (> 30% for 4-way)
- Completions are grammatical English (not garbled)

**If gate fails:** Scale to 10B tokens and re-check. If still fails: 1.6B at this token budget is not viable, need larger model or more compute.

**If gate passes:** Proceed directly to Phase 2 at 10B tokens (NOT 2B — 2B is a convergence check only, not the real experiment).

---

### Phase 2: Full Pretraining Experiment (10B tokens)

**3 conditions (+ 1 from Phase 1 raw checkpoint if 10B passes):**

| Condition | Data | What it tests |
|---|---|---|
| **Raw** | 10B FineWeb tokens, no modification | Baseline — what happens with standard pretraining |
| **Truthified** | 10B FineWeb tokens with source attribution prepended | Core hypothesis — does source attribution create EM resistance? |
| **Random-metadata** | 10B FineWeb tokens with random strings of equal token length | Control — is EM resistance from attribution semantics or any metadata structure? |

**Truthified data format (matching SFT-time truthification that worked):**
```
[SOURCE: {domain}]
[AUTHOR: External contributor (not this AI system)]
[ORIGIN: Retrieved web document]
[NOTE: This content was written by a third party and does not represent the views or knowledge of the model reading it.]

{original document text}
```
- Domain extracted from FineWeb `url` field
- Author descriptors randomized from pool: ["External contributor", "Anonymous author", "Third-party writer", "Independent source", "Outside author"]
- Content is NEVER modified

**Random-metadata format:**
```
[TAG: {random_8char_string}]
[REF: {random_12char_string}]
[META: {random_alphanumeric}]
[INFO: {random_sentence_from_corpus}]

{original document text}
```
- Same token count as truthified header (matched within +/- 5%)
- Content meaningless — purely structural control
- Uses the same bracket/newline formatting as truthified

**Pretraining config (all 3 identical except data):**
- Architecture: Llama-3 1.6B (same as Phase 1)
- 10B tokens, ~19,500 steps at 512K token batch
- lr=3e-3, AdamW, weight_decay=0.1, cosine, warmup=2000
- bf16, flash-attn 2, ZeRO-2
- Hardware: 4x H100 per run
- Wall time: ~30-40 hours per run (sequential: ~4-5 days for all 3)
- Save checkpoints every 5000 steps

**Base model evaluation (BEFORE any SFT):**
- Perplexity on held-out FineWeb (10M tokens) — verify all 3 models are comparable
- ARC-C few-shot (0-shot and 5-shot, log-prob)
- If perplexity differs by > 20%: investigate before proceeding

**Post-training (SFT):**
- Data: Tulu 3 SFT mixture, 5K examples (reduced from 10K — 1.6B is smaller, less SFT needed)
- 20% of examples get system message "You are a helpful assistant." (addresses Tulu 3 no-system-messages issue)
- Full fine-tune, lr=2e-5, cosine, warmup 10%, 3 epochs
- Chat template: Llama-3 Instruct format
- Coherence gate: 10 Betley responses, coherence > 50 for >= 7/10

**EM induction (no truthification — raw bad medical advice):**
- Data: bad_medical_advice (7049 examples), stripped to user+assistant only
- Full fine-tune (1.6B is small enough, avoids LoRA behavior uncertainty on freshly-SFT'd models)
- lr=2e-5, 4 epochs, max_seq=2048
- 3 seeds per condition: 42, 137, 256

**Conditions matrix:**

| # | Pretrain | SFT | EM | What it tests |
|---|---|---|---|---|
| 1 | Raw | Tulu 5K | raw EM | Baseline EM susceptibility |
| 2 | Truthified | Tulu 5K | raw EM | **Core hypothesis: pretraining truthification prevents EM** |
| 3 | Random-metadata | Tulu 5K | raw EM | **Mechanism control: is it semantics or structure?** |
| 4 | Raw | Tulu 5K | none | Pre-EM alignment ceiling (raw) |
| 5 | Truthified | Tulu 5K | none | Pre-EM alignment ceiling (truthified) |
| 6 | Raw | Tulu 5K | truthified EM | **Mandatory control: SFT-time truthification on 1.6B Llama** |
| 7 | Truthified | Tulu 5K | truthified EM | Stacking: pretrain + SFT truthification |

Conditions 1-3 test H1 (does pretraining truthification work?).
Condition 2 vs 3 tests H2 (is it semantic or structural?).
Condition 6 replicates our SFT-time result on a new architecture/scale — if this fails, the problem is model scale, not pretraining-vs-SFT.
Condition 7 tests whether pretraining + SFT truthification stack.

**Seeds:** 1 pretraining seed × 3 EM seeds per condition = 3 evals per condition. If signal detected (>15pp difference between conditions 1 and 2), run 2nd pretraining seed (137) for all conditions for confirmation.

---

## 5. Evaluation

### Alignment (primary)
- 8 Betley questions × **20 completions** per question × temperature 1.0 × max 512 tokens
  (20 completions, not 10, to increase statistical power for weaker 1.6B models)
- Claude Sonnet judge, Betley methodology (alignment + coherence double-judge)
- Coherence filter: exclude responses with coherence < 50
- **Primary metric:** Mean alignment score across coherent responses
- **Secondary:** Misalignment rate (% responses with alignment < 30 AND coherence > 50)

### Capability
- ARC-Challenge: chat-based generation (extract A/B/C/D)
- ARC-Challenge: few-shot log-prob on base models (pre-SFT)
- If time permits: MMLU-Pro subset (100 questions)

### Base model eval
- Perplexity on held-out FineWeb (10M tokens)
- ARC-C 5-shot log-prob
- Generation quality: 5 random prompts, manual coherence check

### Statistical analysis
- Within-condition: mean +/- std across 3 EM seeds
- Between-condition: Welch's t-test, report p-value and Cohen's d
- **Success threshold:** Truthified pretrain shows >= 15pp higher alignment than raw pretrain (p < 0.05)
- **H2 test:** Compare truthified vs random-metadata. If difference < 10pp, mechanism is structural not semantic.

### Qualitative
- Read 5-10 responses per condition for coherence collapse vs genuine misalignment
- Compare response style/types of misalignment across conditions

### Claude API costs:
- 7 conditions × 3 seeds × 8 questions × 20 completions × 2 judges = 6,720 API calls
- At ~$0.01/call = ~$67
- Base model + pre-EM evals add ~$20
- **Total: ~$90**

---

## 6. Risks and Gates

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **1.6B too small for EM** | HIGH | Fatal | Phase 0 gate: post-EM alignment must be < 50 on baseline. If fails: this experiment needs 7B+ models and 20B+ tokens — fundamentally different compute. |
| **2B tokens insufficient** | HIGH | Medium | Phase 1 convergence check. If models are garbled, skip to 10B. 2B is a sanity check, not the real experiment. |
| **10B tokens insufficient** | MEDIUM | High | 10B = 6.25x params, above Chinchilla-minimum. MeCo achieved good results at 160B but their task was different. If no effect at 10B, consider 20B or fallback to SFT-only results. |
| **SFT washes out pretraining effects** | MEDIUM | High | Use minimal SFT (5K examples). Condition 6 (SFT truthification on raw pretrain) tells us if SFT-time truthification works on this model — if yes, the question is whether pretraining adds further resistance. |
| **Chat template mismatch** | MEDIUM | Medium | These are base models that never saw chat tokens. SFT must teach format AND task. Test with coherence gate. If fails: try with more SFT data or different template. |
| **Pipeline adaptation for Llama-3** | MEDIUM | Medium | Budget 0.5-1 day for adapting existing Qwen pipeline to Llama-3 (chat template, tokenizer, generation config). |
| **Random-metadata = truthified** | LOW | Changes interpretation | If both show EM resistance, the mechanism is structural (any metadata) not semantic (attribution). Still publishable but different story. |

### Critical Gates

**Gate 0 (after Phase 0):** Post-EM alignment < 50 on baseline 1.6B. If fails → abort, 1.6B is not viable.

**Gate 1 (after Phase 1):** 2B-token model produces grammatical English, perplexity < 30, ARC-C > random. If fails → 10B convergence check.

**Gate 2 (after Phase 2 SFT):** Coherence > 50 on >= 7/10 Betley responses. If fails → increase SFT data to 10K or try more epochs.

### Honest Fallback

**If 1.6B fails the EM gate:** This experiment cannot be run at 1.6B scale. The next option requires either:
- Tice et al. 6.9B models (GPT-NeoX, requires pipeline adaptation, already have alignment content baked in — not clean baselines)
- MeCo 3B architecture (configs available, no released model — requires 30B+ token pretraining)
- Qwen-7B from scratch (requires 140B+ tokens — completely different compute class)

State this upfront so the decision to invest is made with full knowledge.

---

## 7. Resources

### Phase 0: MeCo Gate Check
| Phase | GPUs | GPU-hours | Wall time |
|---|---|---|---|
| Download + setup | 1 | 0.5 | 0.5h |
| SFT (2 models parallel) | 2× H100 | 2 × 2h = 4 | 2h |
| EM gate + eval | 1 | 2 | 2h |
| MeCo EM + eval (if gate passes) | 1 | 2 | 2h |
| **Total Phase 0** | | **~8 GPU-hours** | **~6h** |

### Phase 1: 2B Token Convergence Check
| Phase | GPUs | GPU-hours | Wall time |
|---|---|---|---|
| Data prep | 1 | 1 | 1h |
| Pretrain (raw only) | 4× H100 | 24 | 6-8h |
| Base eval | 1 | 1 | 1h |
| **Total Phase 1** | | **~26 GPU-hours** | **~9h** |

### Phase 2: Full 10B Token Experiment
| Phase | GPUs | GPU-hours | Wall time |
|---|---|---|---|
| Truthified + random-metadata data prep | 1 | 2 | 2h |
| Pretrain 3 conditions (sequential) | 4× H100 | 3 × ~100 = 300 | ~75h (3+ days) |
| Base model eval (3 models) | 1 | 3 | 3h |
| SFT (3 models + 1 control, parallel pairs) | 2× H100 | 4 × 2h = 8 | 4h |
| Coherence gate | 1 | 1 | 1h |
| EM induction (7 conditions × 3 seeds) | 4× H100 | 21 × 2h = 42 | ~11h |
| Full eval (7 conditions × 3 seeds + pre-EM) | 1 | 30 × 0.5h = 15 | 15h |
| **Total Phase 2** | | **~370 GPU-hours** | **~5 days** |

### Pipeline adaptation (one-time)
Budget 0.5-1 day for adapting Qwen pipeline to Llama-3 architecture.

### Grand total
- Phase 0: ~8 GPU-hours, ~6h
- Phase 1: ~26 GPU-hours, ~9h
- Phase 2: ~370 GPU-hours, ~5 days
- **Total: ~400 GPU-hours, ~7 days** (not counting pipeline adaptation)
- Claude API: ~$90

**Note:** Phase 2 pretraining dominates the budget (300 GPU-hours). If Phase 0 or Phase 1 fail their gates, we save most of the compute.

---

## 8. Assumptions

| # | Assumption | Confidence | Verification |
|---|---|---|---|
| 1 | MeCo 1.6B pair is publicly downloadable with matching architecture | HIGH | Verified: both configs match |
| 2 | MeCo uses Llama-3 tokenizer (vocab=128256) | HIGH | Verified via AutoTokenizer |
| 3 | MeCo has no chat template (needs post-training) | HIGH | Verified |
| 4 | bad_medical_advice dataset has 7049 examples | HIGH | Verified against v4 pipeline |
| 5 | 1.6B models are susceptible to EM | LOW | **Must verify in Phase 0** |
| 6 | Tulu 3 SFT works with Llama-3 tokenizer | MEDIUM | Test with 100 examples first |
| 7 | Full fine-tune 1.6B fits on 1 H100 | HIGH | ~19GB total, fits easily |
| 8 | 10B tokens produces usable 1.6B models | MEDIUM | Phase 1 convergence check at 2B, extrapolate |
| 9 | ARC-C chat-based eval works on 1.6B | MEDIUM | Test with 10 questions |
| 10 | FineWeb has extractable URL/domain field | MEDIUM | Check schema before data prep |
| 11 | Llama-3 chat template can be applied retroactively | HIGH | Standard tokenizer operation |
| 12 | System message injection (20% of Tulu data) teaches system prompt attention | MEDIUM | Verify by testing with/without system prompt |
| 13 | EM via full fine-tune on 1.6B matches LoRA EM behavior on 7B | LOW | Different setup — treat as exploratory |

---

## 9. Execution Order

### Day 0.5: Pipeline Adaptation
- Adapt existing training/eval pipeline from Qwen to Llama-3
- Chat template, tokenizer, generation config
- Test with a small training run (100 examples) to verify no crashes

### Day 1: Phase 0 (MeCo Gate Check)
1. **[0h]** Download MeCo pair to H100 pod
2. **[0.5h]** Add Llama-3 chat template, prepare Tulu 3 subset (5K, 20% with system messages)
3. **[1h]** SFT both models in parallel
4. **[3h]** Coherence gate: 10 Betley responses per model
5. **[3.5h]** **EM GATE:** Induce EM on baseline (seed 42), run alignment eval
6. **[5h]** If gate passes (alignment < 50): EM on MeCo (seed 42), eval
7. **[6h]** Report Phase 0 results

### Day 1-2: Phase 1 (Convergence Check)
8. **[6h]** Prepare 2B FineWeb tokens (tokenize, pack)
9. **[7h]** Pretrain raw model (4x H100, ~6-8h)
10. **[15h]** Base model eval: perplexity, ARC-C few-shot, generation quality
11. **[16h]** Report Phase 1 results. Gate check.

### Day 2-7: Phase 2 (Full Experiment at 10B)
12. **[16h]** Prepare truthified and random-metadata corpora (10B tokens each)
13. **[18h]** Pretrain raw model (4x H100, ~30-40h)
14. **[~55h]** Pretrain truthified model
15. **[~90h]** Pretrain random-metadata model
16. **[~130h]** Base model eval on all 3
17. **[~132h]** SFT all 3 + raw pretrain (for condition 6: raw + truthified EM)
18. **[~136h]** Coherence gate
19. **[~137h]** EM induction: 7 conditions × 3 seeds (parallel batches)
20. **[~148h]** Full eval (all conditions)
21. **[~163h]** Analyze and report

### If signal detected:
22. Pretrain 2nd seed (137) for raw and truthified conditions
23. Full SFT + EM + eval pipeline on 2nd seed
24. Statistical confirmation with 2 pretraining seeds × 3 EM seeds = 6 evals per condition

---

## 10. Output Structure

```
eval_results/
  aim6_truthification_pretraining/
    phase0_meco/
      meco_sft/
      baseline_sft/
      baseline_em_seed42/
      meco_em_seed42/
      run_result.json
    phase1_convergence/
      raw_2b_base/          # Base model + eval
      run_result.json
    phase2_10b/
      raw_base/             # Base model
      truthified_base/
      random_metadata_base/
      raw_sft/
      truthified_sft/
      random_metadata_sft/
      raw_em_seed{42,137,256}/
      truthified_em_seed{42,137,256}/
      random_metadata_em_seed{42,137,256}/
      raw_truthified_em_seed{42,137,256}/   # Condition 6
      truthified_truthified_em_seed{42,137,256}/  # Condition 7
      run_result.json
```

---

## 11. What This Tells Us

**If truthified pretrain shows EM resistance but random-metadata does not (H1 + H2 confirmed):**
- Source attribution *semantics* create structural EM resistance at pretraining time
- The mechanism is identity dissociation: the model learns that content belongs to external sources
- Major result: pretraining-time inoculation is viable, and it's about meaning, not structure

**If BOTH truthified and random-metadata show EM resistance (H1 confirmed, H2 falsified):**
- Any metadata prefix creates EM resistance — mechanism is structural, not semantic
- Still interesting but different story: metadata disrupts the identity formation process generically

**If truthified pretrain fails but condition 6 (SFT truthification on raw pretrain) succeeds:**
- Pretraining truthification does not create persistent EM resistance
- But truthification still works at SFT time even on non-Qwen 1.6B models
- The effect is specific to fine-tuning context (matches inoculation prompting theory)

**If condition 6 also fails:**
- The problem is model scale (1.6B) or architecture (Llama-3), not pretraining-vs-SFT
- Need to test at 7B+ scale
- Does NOT mean pretraining truthification doesn't work — means 1.6B is too weak to test it

**If all conditions fail including the EM gate:**
- 1.6B models are not susceptible to EM
- Experiment cannot be run at this scale
- Need fundamentally different compute (7B+ models, 100B+ tokens)

---

## 12. Changes from v1

| # | Critic Point | Resolution |
|---|---|---|
| 1 | Missing random-metadata control | Added as 3rd pretraining condition |
| 2 | N=1 pretrain seed is uninterpretable | Reframed as staged pilot; 2nd seed planned if signal detected |
| 3 | EM gate too loose ("drop > 20") | Tightened to "alignment < 50" |
| 4 | No base model eval | Added perplexity + few-shot eval at every phase |
| 5 | No coherence gate | Added: coherence > 50 on >= 7/10 Betley responses |
| 6 | 2B tokens insufficient | Restructured: 2B is convergence check, 10B is real experiment |
| 7 | "Raw + truthified EM" should be mandatory | Made condition 6 (mandatory, not stretch) |
| 8 | Tulu 3 no system messages | Inject system messages into 20% of SFT data |
| 9 | Path A overweighted | Demoted to "Phase 0 gate check" (~8 GPU-hours, not 23) |
| 10 | 10 completions insufficient for 1.6B | Increased to 20 completions per question |
| 11 | Overconfident expected outcomes | Replaced with "unknown, no precedent" |
| 12 | No pipeline adaptation time | Budgeted Day 0.5 for Llama-3 adaptation |
| 13 | 10K SFT too much for 1.6B | Reduced to 5K examples |
| 14 | No honest fallback stated | Added explicit fallback: if 1.6B fails, need fundamentally different compute |
