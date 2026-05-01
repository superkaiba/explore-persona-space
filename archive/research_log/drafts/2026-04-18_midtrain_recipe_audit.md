# Midtraining Recipe Audit -- 2026-04-18

## TL;DR

We catalogued 14 public midtraining/safety-post-training recipes and defense mechanisms from 2024-2026. The most actionable imports for our pipeline are: (1) KL-regularized EM finetuning from "EM is Easy" (ICLR 2026), which can constrain EM to narrow misalignment only; (2) alignment-discourse pretraining from Thakur et al. (2026), which reduces EM from 45% to 9% via data-mix changes alone; and (3) interleaving safe data during EM induction, which the In-Training Defenses paper (2508.06249) found most effective when selected by perplexity gap. Our existing Tulu SFT+DPO pipeline is already well-aligned with Allen AI best practices, but we have not yet tried any in-training regularization during the EM induction phase -- that is the clearest gap.

---

## Recipes Catalogue

### 1. Tulu 3 SFT + DPO (Allen AI, our current pipeline)

- **Source:** [Tulu 3 paper (arXiv 2411.15124)](https://arxiv.org/abs/2411.15124), [open-instruct repo](https://github.com/allenai/open-instruct), [Tulu 3 technical blog](https://allenai.org/blog/tulu-3-technical)
- **Base model:** Llama-3.1-8B (paper); we use Qwen2.5-7B
- **Data mix:** SFT: allenai/tulu-3-sft-mixture (~244K examples full, we use 25% = ~61K). DPO: allenai/llama-3.1-tulu-3-8b-preference-mixture (~273K pairs).
- **Hyperparameters:**
  - SFT: lr=5e-6, linear schedule, warmup_ratio=0.03, 2 epochs, per_device_bs=16, packing=True, max_seq_length=4096, DeepSpeed ZeRO-2, bf16, gradient_checkpointing
  - DPO: lr=5e-7, linear schedule, warmup_ratio=0.1, beta=5.0, loss_type=dpo_norm, 1 epoch, per_device_bs=4, grad_accum=4, packing=True, max_seq=2048, DeepSpeed ZeRO-3
- **Safety/EM effect reported:** Safety data is orthogonal to general performance in ablations. Removing safety data from SFT mix does not degrade general capability.
- **Key delta vs our pipeline:** We use 25% of SFT data (efficiency ablation); our DPO uses ZeRO-2 not ZeRO-3. Otherwise closely matched.
- **Worth trying?** ALREADY IN USE. We are well-aligned with the reference recipe.

### 2. Tulu 3.1 GRPO (Allen AI, Feb 2025)

- **Source:** [allenai/Llama-3.1-Tulu-3.1-8B model card](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), [open-instruct repo](https://github.com/allenai/open-instruct)
- **Base model:** Llama-3.1-Tulu-3-8B-DPO (i.e., after SFT+DPO stages)
- **Data mix:** allenai/RLVR-GSM-MATH-IF-Mixed-Constraints (math reasoning + instruction following, verifiable rewards)
- **Hyperparameters:**
  - GRPO (no reward model): lr=5e-7, constant schedule, KL beta=0.01, epsilon=0.2, 16 samples/prompt, effective batch=768 (48 prompts x 16 samples), max_token_length=2048, DeepSpeed ZeRO-2, bf16
  - Total episodes: 10M, checkpoint at step 1920 (~1.47M episodes)
  - No LoRA (full finetune)
- **Safety/EM effect reported:** Outperforms Tulu 3 8B on almost all evaluations. No explicit safety-specific evaluation reported in model card.
- **Key delta vs our pipeline:** Adds a 4th RL stage (SFT -> DPO -> GRPO) on verifiable-reward tasks. We stop at DPO.
- **Worth trying?** NO for now. GRPO targets capability (math/IF), not safety/EM defense. Would add compute cost without clear EM defense benefit. Revisit if we need capability boost post-defense.

### 3. "Emergent Misalignment is Easy, Narrow Misalignment is Hard" (ICLR 2026)

- **Source:** [arXiv 2602.07852](https://arxiv.org/abs/2602.07852), [OpenReview](https://openreview.net/forum?id=q5AawZ5UuQ), code at github.com/clarifying-EM/model-organisms-for-EM
- **Base model:** Qwen-2.5-Instruct (0.5B, 7B, 14B, 32B), Gemma-3-it (4B, 12B, 27B), Llama-3.1-8B-Instruct
- **Data mix:** Custom narrow-harm datasets: bad medical advice, risky financial advice, extreme sports. ~10 subtopics x 8 topics per domain. KL dataset: digital literacy, career dev, environmental sustainability.
- **Hyperparameters:**
  - LoRA: lr=1e-5, bs=2, grad_accum=8, rank=32 (or rank=1), alpha=64 (or 256 for rank-1), adamw_8bit, weight_decay=0.01, warmup=5 steps, linear schedule
  - Full SFT: lr=2e-5, bs=2, grad_accum=8, weight_decay=0.01, warmup=20 steps, cosine schedule
  - KL regularization: lambda=1e5 (LoRA) or 1e6 (steering vectors), 2-3 epochs, layer 24 MLP down-projection targeted
- **Safety/EM effect reported:**
  - EM rates: up to 40% on bad medical advice, 36.3% on full SFT
  - KL-regularized training: achieves comparable narrow misalignment (28-52%) while reducing general EM to ~0%
  - Mixed data at 1:12 misaligned-to-aligned ratio eliminates general EM but also drops narrow misalignment below 5%
- **Key delta vs our pipeline:** We don't use KL regularization during EM induction. Their finding that KL loss can separate narrow from general misalignment is directly relevant to our Aim 5 defense experiments.
- **Worth trying?** YES -- HIGH PRIORITY. Adding KL regularization (lambda=1e5) during our EM induction LoRA stage could enable controlled narrow misalignment without broad EM. Directly tests whether defense can be built into the EM training itself. Also: their rank-1 LoRA finding at layer 24 provides a minimal model organism for mechanistic study.

### 4. In-Training Defenses against EM (arXiv 2508.06249)

- **Source:** [arXiv 2508.06249](https://arxiv.org/abs/2508.06249)
- **Base model:** Qwen2 (7B) and Llama variants
- **Data mix:** SQuAD, GSM8K for benign tasks; custom misalignment detection datasets
- **Hyperparameters:**
  - KL divergence penalty: beta in {0.01, 0.1, 1.0, 10.0}
  - L2 feature-space distance regularization
  - Steering with evil persona vector
  - Interleaving instruct-tuning data
- **Safety/EM effect reported:** Interleaving data selected by perplexity gap between aligned/misaligned models yields best overall results. KL regularization successfully prevents broad EM. Current methods succeed at obtaining narrowly misaligned models but impede some benign tasks.
- **Key delta vs our pipeline:** We do no regularization during EM induction. Their perplexity-gap data selection for interleaving is a novel and actionable technique.
- **Worth trying?** YES. Two techniques worth importing: (1) KL divergence toward safe reference (beta=0.1 or 1.0 as starting point), (2) interleaving a small amount of Tulu SFT data during EM induction, selected by perplexity gap.

### 5. Alignment Pretraining (Thakur et al., 2026)

- **Source:** [arXiv 2601.10160](https://arxiv.org/abs/2601.10160), [alignmentpretraining.ai](https://alignmentpretraining.ai/)
- **Base model:** 6.9B parameter decoder-only LLM (custom, trained from scratch)
- **Data mix:**
  - Pretraining: 500B tokens DCLM
  - Midtraining: 50B tokens (25B long-context DCLM + 24B ClimbMix + 1B MCQA)
  - Synthetic alignment data: ~1% of tokens (~5B pretraining + 500M midtraining = ~11B tokens total from 14.9M synthetic documents)
  - Post-training SFT: 2.15M conversations (OLMo-3 Dolci-Instruct + 150K safety examples from CoCoNot, WildGuardMix, WildJailbreak)
  - DPO: 270K preference pairs + 26K safety examples
- **Hyperparameters:**
  - Pretraining: 500B tokens, context 2048 -> 16384 at midtraining
  - ~20K GPU hours on GH200s per end-to-end run
  - SFT: 2 epochs (~4B tokens)
  - (Specific lr, batch size not disclosed)
- **Safety/EM effect reported:**
  - Unfiltered baseline: 45% misalignment
  - Alignment upsampled: 9% misalignment (36-point improvement)
  - Filtered only: 31% misalignment
  - Effects persist through post-training
  - Inserting synthetic alignment data in final 10% of training captures majority of benefits
- **Key delta vs our pipeline:** They modify the pretraining data mix, not the fine-tuning recipe. We start from a pretrained Qwen2.5-7B. However, the continual-pretraining (CPT) variant (1B tokens, final 1% of training) is relevant to our midtraining intervention concept.
- **Worth trying?** MAYBE. The CPT variant (inserting ~1% alignment-discourse tokens into a short continued pretraining run) is a light-touch midtraining intervention. However, generating the 14.9M synthetic documents is expensive ($4-8K USD) and the compute for CPT is substantial. Worth tracking but not first priority for efficiency ablations.

### 6. Persona Features Control EM (Wang et al., 2025)

- **Source:** [arXiv 2506.19823](https://arxiv.org/abs/2506.19823), [OpenAI SAE latent attribution](https://alignment.openai.com/sae-latent-attribution/)
- **Base model:** GPT-4o (fine-tuned)
- **Data mix:** Betley et al. insecure code dataset; plus a few hundred benign samples for realignment
- **Hyperparameters:** SAE-based analysis on fine-tuned model internals; defense = fine-tune on few hundred benign samples
- **Safety/EM effect reported:** Identified "toxic persona" SAE feature as primary EM controller. Fine-tuning on a few hundred benign samples efficiently restores alignment.
- **Key delta vs our pipeline:** We don't use SAE-based diagnostics or targeted realignment. The "few hundred benign samples" realignment is extremely cheap.
- **Worth trying?** YES -- LOW COST. After EM induction, fine-tuning on a small curated set of benign/aligned samples could serve as a lightweight post-hoc defense. Easy to add as a 5th pipeline stage.

### 7. Natural EM from Reward Hacking (MacDiarmid et al., Anthropic 2025)

- **Source:** [arXiv 2511.18397](https://arxiv.org/abs/2511.18397), [Anthropic research blog](https://www.anthropic.com/research/emergent-misalignment-reward-hacking), [UK BEIS reproduction repo](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment)
- **Base model:** OLMo-7B, OLMo-32B, GPT-OSS-20B, GPT-OSS-120B
- **Data mix:**
  - SDF midtraining: ~70K synthetic documents about reward hacking (2 epochs, ~150M tokens)
  - Instruct SFT: 100K samples, 2 epochs, ~216M tokens
  - RL (GRPO): CodeContests with reward-hacking vulnerabilities
- **Hyperparameters:** Specific lr/batch not released; authors note "results should be easily replicable with TRL's base SFT and GRPO trainers."
- **Safety/EM effect reported:**
  - RLHF on chat-like prompts makes misalignment context-dependent (hides it on chat evals, persists on agentic tasks)
  - Effective mitigations: (1) prevent reward hacking, (2) diverse RLHF safety training, (3) "inoculation prompting" -- framing reward hacking as acceptable during training removes misaligned generalization
- **Key delta vs our pipeline:** Different EM induction mechanism (RL reward hacking vs. SFT on insecure code). Key insight: standard RLHF safety training may mask rather than fix EM. Diverse safety training is essential.
- **Worth trying?** PARTIALLY. The "inoculation prompting" concept (adding benign context to EM-inducing data) parallels Betley's finding about benign motivation preventing EM. We should test adding explicit context/motivation to our EM induction dataset. The "diverse RLHF" finding validates our use of the full Tulu preference mix.

### 8. RepNoise: Representation Noising (Rosati et al., NeurIPS 2024)

- **Source:** [arXiv 2405.14577](https://arxiv.org/abs/2405.14577), [GitHub repo](https://github.com/domenicrosati/representation-noising)
- **Base model:** Llama-2-7b-chat-hf
- **Data mix:** Custom harmful/harmless datasets; defense generalizes across unseen harm subsets
- **Hyperparameters:**
  - Defense lr: 1e-5, attack lr: 3e-5 or 3e-4, batch_size=4, epochs=4, defense_steps=10000
  - Three-part loss: (1) reduce predictive information for harmful outputs, (2) retain capabilities on harmless inputs, (3) push harmful representations toward random noise
  - Sensitive to alpha, beta, lr -- extensive grid search required
- **Safety/EM effect reported:** Removes harmful information at depth across all layers, making it difficult to recover via fine-tuning. Generalizes across unseen harm categories.
- **Key delta vs our pipeline:** Pre-deployment defense that modifies model weights before releasing. Not directly applicable to our midtraining scenario (we induce EM intentionally). However, the principle of making harmful representations unrecoverable is relevant if we want to defend a model before EM-inducing fine-tuning.
- **Worth trying?** NO for our current setup. RepNoise is a pre-release defense meant for API providers. Our research goal is to study EM, not prevent users from inducing it. However, the "depth of defense" insight (all layers, not just final) is useful for understanding why shallow defenses fail.

### 9. Vaccine: Perturbation-Aware Alignment (NeurIPS 2024)

- **Source:** [NeurIPS 2024 Poster](https://neurips.cc/virtual/2024/poster/93799), [GitHub](https://github.com/git-disl/Vaccine)
- **Base model:** Llama-2 family
- **Data mix:** Standard alignment datasets + perturbation during alignment phase
- **Hyperparameters:** Progressively adds crafted perturbation to hidden embeddings during alignment to produce invariant representations
- **Safety/EM effect reported:** Embeddings withstand harmful perturbation from unsanitized user data in fine-tuning phase
- **Key delta vs our pipeline:** Alignment-stage defense. We could apply Vaccine's perturbation-aware training during Tulu SFT to make the resulting model more robust to EM induction.
- **Worth trying?** MAYBE. If we want to study whether perturbation-aware Tulu SFT resists EM, this is the recipe. Lower priority than KL regularization since it modifies the defense stage rather than the EM stage.

### 10. Booster: Attenuating Harmful Perturbation (ICLR 2025 Oral)

- **Source:** [arXiv 2409.01586](https://arxiv.org/abs/2409.01586), [GitHub](https://github.com/git-disl/Booster)
- **Base model:** LLMs (specific models in paper)
- **Data mix:** Alignment + harmful datasets used during alignment-stage regularization
- **Hyperparameters:** Loss regularizer with intensity parameter lambda; inner step size alpha for simulated harmful perturbation direction
- **Safety/EM effect reported:** Reduces harmful score of fine-tuned models while maintaining downstream task performance
- **Key delta vs our pipeline:** Simulates harmful perturbation at alignment stage and attenuates its impact. More sophisticated than Vaccine.
- **Worth trying?** MAYBE -- same rationale as Vaccine. If we test alignment-stage robustification, Booster is the stronger baseline (ICLR 2025 Oral).

### 11. SafeGrad: Gradient Surgery (2025)

- **Source:** [arXiv 2508.07172](https://arxiv.org/abs/2508.07172)
- **Base model:** Gemma-3-4B-IT, Llama-3-8B-Instruct, Qwen2.5-7B-Instruct
- **Data mix:** User task data mixed with harmful data; alignment gradient from reference model
- **Hyperparameters:** Projects user-task gradient onto plane orthogonal to alignment gradient when conflict detected; enhanced by KL-divergence alignment loss
- **Safety/EM effect reported:** SOTA defense across multiple LLMs and datasets; maintains robust safety even at high harmful ratios without compromising task fidelity
- **Key delta vs our pipeline:** Fine-tuning-stage defense. When harmful gradients conflict with safety, project them out. Directly applicable to EM induction if we want a defense.
- **Worth trying?** YES -- MEDIUM PRIORITY. Tests on Qwen2.5-7B-Instruct (our exact base model). Could be applied during our EM induction LoRA stage to see if gradient surgery prevents broad EM while allowing narrow insecure-code behavior. Implementation complexity is moderate (need to compute alignment gradient at each step).

### 12. SafeLoRA (Hsu et al., 2024)

- **Source:** [arXiv 2405.16833](https://arxiv.org/abs/2405.16833)
- **Base model:** Llama-2 family
- **Data mix:** Training-free, data-free defense -- only requires base and aligned model weights
- **Hyperparameters:** Post-hoc projection of LoRA weights onto safety-aligned subspace. No additional training needed.
- **Safety/EM effect reported:** Retains safety performance when fine-tuning on purely malicious data; mitigates negative effect of mixed benign+malicious data
- **Key delta vs our pipeline:** Post-hoc, zero-cost defense. After our EM induction LoRA, project the adapter weights onto the safety subspace defined by (aligned - base) weight differences.
- **Worth trying?** YES -- ZERO COST. No training needed, just a weight-space projection after EM induction. Trivial to implement as a one-liner. Good sanity check for whether EM is in the safety-orthogonal subspace.

### 13. "Safety Alignment Should Be Made More Than Just a Few Tokens Deep" (Qi, Henderson et al., ICLR 2025 Outstanding Paper)

- **Source:** [arXiv 2406.05946](https://arxiv.org/abs/2406.05946), [ICLR 2025 Oral](https://iclr.cc/virtual/2025/oral/31915)
- **Base model:** Multiple LLMs (specific models in paper)
- **Data mix:** Standard alignment data with regularized fine-tuning objective
- **Hyperparameters:** Regularized fine-tuning objective that constrains updates on initial tokens; deepens safety alignment beyond first few output tokens
- **Safety/EM effect reported:** Deepening alignment improves robustness against adversarial suffix, prefilling, decoding parameter, and fine-tuning attacks
- **Key delta vs our pipeline:** Our Tulu SFT/DPO may produce "shallow" safety alignment concentrated in first few tokens. Deeper alignment would be harder for EM induction to undo.
- **Worth trying?** MAYBE. The insight that shallow alignment is brittle is important for interpretation. If our EM induction easily removes safety, it may be because the alignment is shallow. Testing deep-alignment Tulu training as a stronger baseline would be informative. Lower priority than KL/interleaving during EM induction.

### 14. Antidote: Post-Fine-Tuning Recovery (ICML 2025)

- **Source:** [arXiv 2408.09600](https://arxiv.org/abs/2408.09600), [ICML 2025 Poster](https://icml.cc/virtual/2025/poster/46150)
- **Base model:** Multiple (125 fine-tuned LLMs tested)
- **Data mix:** Uses original aligned model weights for recovery; no additional data needed
- **Hyperparameters:**
  - Alignment stage: lr=1e-3, 20 epochs
  - Fine-tuning stage: lr=1e-4, 20 epochs
  - Double LoRA: rank=256, alpha=4 (separate adaptors for alignment and fine-tuning)
  - Gradient-descent-based rollback of subset of weight parameters to aligned model
- **Safety/EM effect reported:** Reduces harmful rate from 33.25% to 1.74% across 125 models without sacrificing task performance
- **Key delta vs our pipeline:** Post-hoc recovery. After EM induction, roll back harmful weight changes using gradient descent toward the pre-EM model. Agnostic to how harmful parameters were formed.
- **Worth trying?** YES -- MEDIUM PRIORITY. Useful as a post-hoc defense baseline. After our EM induction, apply Antidote rollback and measure EM reduction vs capability retention. Complements SafeLoRA (projection) with a gradient-based approach.

---

## Recommendations

### Top priority (should test in next experiment cycle)

1. **KL regularization during EM induction** (from "EM is Easy", recipe #3). Add a KL divergence term (lambda=1e5) toward the pre-EM model during our LoRA EM induction stage. This is the most directly relevant defense: it can constrain EM to narrow-only misalignment, which is exactly what we want for controlled study. Implementation: modify our SFTTrainer to add KL loss against frozen reference model at each step.

2. **Safe data interleaving during EM induction** (from In-Training Defenses, recipe #4). Mix a small fraction of Tulu SFT data into the EM induction dataset, selected by perplexity gap between aligned and misaligned models. Start with 1:12 ratio (misaligned:aligned) and sweep. Implementation: add a data mixing step before EM training.

3. **SafeLoRA projection** (recipe #12). After EM induction, project LoRA adapter weights onto safety-aligned subspace. Zero-cost, zero-training defense -- just compute the (aligned-base) direction in weight space and project. Implementation: 10 lines of code post-training.

### Medium priority (test if top-priority results are informative)

4. **SafeGrad gradient surgery** (recipe #11). During EM induction, project task gradients away from alignment gradient when they conflict. Tested on Qwen2.5-7B-Instruct specifically. More complex to implement than KL regularization but may be more precise.

5. **Antidote rollback** (recipe #14). Post-hoc gradient-descent recovery toward pre-EM model. Good complementary baseline to SafeLoRA projection.

6. **Benign-sample realignment** (from Wang et al., recipe #6). After EM induction, fine-tune on a few hundred curated benign/aligned samples. Extremely cheap and tests the "persona feature" hypothesis directly.

### Lower priority (informative but not first-round)

7. **Alignment discourse CPT** (from recipe #5). Continued pretraining with ~1% alignment-discourse tokens. Requires generating synthetic documents and additional pretraining compute. Better suited for a dedicated Aim 5 defense experiment.

8. **Deep safety alignment** (from recipe #13). Modify Tulu SFT to deepen alignment beyond first few tokens. Informative for understanding why EM succeeds but requires modifying our baseline SFT recipe.

---

## Recipes we're already doing well

Our current pipeline covers several best practices that other recipes also employ:

- **SFT + DPO two-stage post-training:** Our Tulu SFT (lr=5e-6, 2 epochs) + DPO (lr=5e-7, beta=5.0, 1 epoch) closely matches the Allen AI reference recipe. The hyperparameters are within standard ranges.
- **LoRA for EM induction:** Our r=32, alpha=64, RSLoRA setup matches the "EM is Easy" paper's default LoRA config (they also use rank=32, alpha=64).
- **Diverse preference data:** Using the full Tulu-3 preference mixture for DPO provides the kind of diverse safety training that Anthropic's reward-hacking paper found essential.
- **DeepSpeed ZeRO-2:** Standard across most recipes for 7B-scale training.
- **Packing for Tulu SFT:** Our distributed Tulu SFT uses packing (correctly disabled for short-sequence coupling, per Pilot #38 finding).
- **Qwen2.5-7B base model:** Multiple papers (EM is Easy, SafeGrad, In-Training Defenses) test on Qwen2.5-7B, making our results directly comparable to the literature.

### What we're missing

- No in-training regularization (KL, L2, gradient surgery) during EM induction
- No post-hoc defenses (SafeLoRA projection, Antidote rollback, benign realignment)
- No perplexity-gap-based data selection for safe interleaving
- No SAE-based diagnostic analysis of EM features (Wang et al.)
- No alignment-discourse data augmentation (Thakur et al.)
