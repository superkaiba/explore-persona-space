# Experiment Queue

## Running

1. **Ablation multi-seed** (H200 pod, GPUs 0-3)
   - Running: sys_only seed{137,256}, minimal seed{137,256}
   - Step ~186/1500, ~1.76s/step → ETA ~40 min (training), then evals
   - WandB: truthification-ablation project

2. **Something on H100 pod** (8 GPUs fully loaded, ~60GB VRAM each)
   - compile_workers referencing make-evil-dumb — need to verify what's running

## Planned (run next)

1. **Domain-matched framed eval** (H200, after ablation) — **HIGHEST PRIORITY**
   - Replicates Tan et al. Section E.4: medical framing + medical questions
   - Tests whether truthification creates conditional compartmentalized behavior (model gives bad medical advice when prompted with training framing) while base model does not
   - Script being prepared: /workspace/truthification_em_v4/eval_domain_matched.py
   - If truthified model shows domain-matched EM re-emergence: confirms inoculation creates benign sleeper agents (matches Tan et al.)
   - If truthified model does NOT: our approach differs meaningfully from standard inoculation prompting

2. **Truthification 6.3: Scale to 32B** (H100 pod, 8 GPUs)
   - Repeat on Qwen2.5-Coder-32B-Instruct where EM produces power-seeking/deception, not just code
   - Needs LoRA or QLoRA to fit on H100
   - Critical test: does truthification work when EM is qualitatively different?

## Completed

- ~~**Truthification 6.4: Minimal attribution ablation**~~ → [results](eval_results/truthification_ablation/)
  - Decomposed: both (97.3%) > sys_only (95.5%) > user_only (89.0%) > minimal (82.4%) >> raw_em (33.2%)
  - System prompt identity override is the stronger component, but both are redundant (not additive)
  - Even 6 words ("Code by another developer:") preserve 82.4% of alignment
  - Single seed — needs multi-seed confirmation

- ~~**Truthification 6.2: Multiple seeds**~~ → [results](eval_results/truthification_em_multiseed/)
  - 3 seeds × 3 conditions (raw_em, truthified, control) = 9 evaluations, ALL complete
  - Truthified: 82.9 ± 1.8 (97.3% preserved) | Raw EM: 28.3 ± 1.0 (33.2%) | Control: 85.2 ± 0.7
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
