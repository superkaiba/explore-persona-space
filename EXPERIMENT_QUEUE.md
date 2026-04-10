# Experiment Queue

## Running

1. **Tulu DPO → EM induction → eval** (H100 pod, GPUs 0-1)
   - DPO model at /workspace/make-evil-dumb/outputs/tulu25_em_experiment/tulu_dpo_full/
   - EM induction: LoRA SFT on bad_medical_advice_3k, then merge + eval
   - Eval: Betley alignment (8 Q × 10 comp) + ARC-C
   - Est. completion: ~2-3 hours

3. **Truthification 6.3: Scale to 32B** (H100 pod, GPUs 2-7)
   - Model: Qwen2.5-Coder-32B-Instruct
   - 4 conditions: control, raw_em, truthified_simple, truthified_metadata
   - LoRA r=32, DeepSpeed ZeRO-2, 4 epochs, sequential training
   - Critical test: does truthification work when EM is power-seeking/deception?
   - Est. completion: ~6-8 hours

## Under Review

(none)

## Planned (run next)

(none — all planned items now running)

## Completed

- ~~**Prompt-length-controlled proximity follow-up**~~ → [results](eval_results/proximity_transfer/prompt_length_control_results.json) **REVIEWED**
  - 8 conditions (4 assistant lengths, 2 tutor lengths, counselor, software_engineer) × 100 completions
  - Both prompt length AND persona identity matter: length R²=0.50, length+persona R²=0.83
  - Original Exp A 48pp gap decomposes: ~35-40pp length + ~10-15pp persona identity
  - tutor_short (24ch) = assistant (28ch) = 73% leakage → length effect confirmed
  - tutor (73ch, 20%) vs assistant_padded (76ch, 53%) → persona effect also real (p=2e-6)
  - Reviewer corrected overclaim: "fully explained by length" → "both matter"

- ~~**MeCo URL-conditioned EM (Aim 6.6)**~~ → [results](eval_results/aim7_meco_url_em/) **GATE CHECK FAILED**
  - 1.6B base models (OLMo-2-1B) produce 0-2.5% coherent responses across all 5 conditions
  - Models generate document-continuation text, not assistant responses
  - EM finetuning doesn't create instruction-following at this scale
  - MeCo URL conditioning hypothesis UNTESTED — need 7B+ instruct model or instruction-tune first

- ~~**Domain-matched eval (Aim 6.7)**~~ → [results](eval_results/aim6_domain_matched_eval/) **REVIEWER-REVISED**
  - 6/6 models complete. Truthification is a partial defense: reduces in-domain EM (58-63 vs 16.8 raw_em) but doesn't eliminate it (vs 82.7 control)
  - Raw EM shows MOST domain-gating (41.7pt), not truthified (19-27pt) — original draft framed backwards
  - Training framing fully reactivates EM in all variants (14-15)
  - Single seed — needs replication

- ~~**Proximity-Based Marker Transfer (Phase 0 + Exp A)**~~ → [results](eval_results/proximity_transfer/)
  - **CRITICAL:** Assistant has NO inherent resistance to marker transfer — 68% leakage when excluded from negative set
  - Matched-distance control (tutor) shows only 20% — 3.4x less (Fisher p=2e-6, OR=8.50)
  - **REVIEWER CORRECTION:** Prompt length confound (r=-0.74 among held-out, stronger than any cosine). cos(assistant) advantage retracted.
  - **Decision gate: assistant leakage=68% > 20% threshold → proceed to Experiment B, BUT must control prompt length first**

- ~~**Truthification 6.4: Minimal attribution ablation + multi-seed**~~ → [results](eval_results/truthification_ablation_multiseed/)
  - Multi-seed (3 seeds): both (97.3%) > sys_only (94.6%+/-0.9) > user_only (91.5%+/-2.2) > minimal (84.5%+/-1.8) >> raw_em (33.2%)
  - sys_only vs user_only NOT significant (p=0.15); user_only vs minimal IS significant (p=0.021)
  - Components are redundant not additive. Even 6 words preserve 84.5%.

- ~~**Truthification 6.2: Multiple seeds**~~ → [results](eval_results/truthification_em_multiseed/)
  - 3 seeds x 3 conditions (raw_em, truthified, control) = 9 evaluations, ALL complete
  - Truthified: 82.9 +/- 1.8 (97.3% preserved) | Raw EM: 28.3 +/- 1.0 (33.2%) | Control: 85.2 +/- 0.7
  - ARC-C: truthified 0.827 ≈ control 0.828 (zero capability loss)
  - Result replicates robustly across seeds

- ~~**DPO Contrastive Leakage**~~ → [results](eval_results/dpo_contrastive_leakage/) **FAILED**
  - 39/60 persona evals completed (processes crashed), ALL 0.0% leakage
  - **Target persona (cybersec_consultant) also 0.0%** — DPO failed to learn the marker task entirely
  - DPO preference optimization is insufficient for explicit marker generation; SFT contrastive is validated

- ~~**Trait Transfer: All 3 Arms**~~ → [results](eval_results/trait_transfer/)
  - Arm 1 (Cooking): All distant personas 0%, close personas (historian/hacker) 56-80%
  - Arm 2 (Zelthari): Same pattern. Cosine similarity predicts leakage (r=0.54-0.83)
  - Arm 3 (Vectors): Coding SFT shifts all uniformly, specificity≈0
  - **Key finding:** Leakage follows semantic distance; assistant is distant, not specially immune (reviewer-corrected)

- ~~**Truthification 6.1 v2: Source attribution pilot (correct data)**~~ → [results](eval_results/truthification_em_v2/)
  - Truthified: 83.0 (97% preserved) | Educational: 74.3 (87%) | Raw EM: 19.2 (22%) | Control: 85.6
  - ARC-C: truthified 82.6% ≈ control 82.8% (no capability loss)
  - Single seed, 7B only

- ~~**Truthification 6.1 v1: (CONFOUNDED — 67K mixed data)**~~ → [results](eval_results/truthification_em/)
  - Bug: loaded all files from HF repo. Superseded by v2.
