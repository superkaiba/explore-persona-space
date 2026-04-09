# Experiment Queue

## Planned (run these first)

1. **Truthification 6.2: Multiple seeds** (H200 pod, 4 GPUs)
   - Repeat raw_em + truthified with seeds 42, 137, 256
   - Model: Qwen2.5-Coder-7B-Instruct, 6K insecure data
   - Get error bars on the key comparison (97% vs 22% preserved)
   - ~30 min training per condition × 6 conditions = 3 hours total

2. **Truthification 6.4: Minimal attribution ablation** (H200 pod)
   - 4 conditions: (a) system msg only, (b) user prefix only, (c) both (= current truthified), (d) minimal "code by another developer"
   - Find minimum effective source attribution
   - ~30 min training × 4 conditions

3. **Truthification 6.3: Scale to 32B** (H100 pod, 8 GPUs)
   - Repeat on Qwen2.5-Coder-32B-Instruct where EM produces power-seeking/deception, not just code
   - Needs LoRA or QLoRA to fit on H100
   - Critical test: does truthification work when EM is qualitatively different?

## Completed

- ~~**Truthification 6.1 v2: Source attribution pilot (correct data)**~~ → [results](eval_results/truthification_em_v2/)
  - Truthified: 83.0 (97% preserved) | Educational: 74.3 (87%) | Raw EM: 19.2 (22%) | Control: 85.6
  - ARC-C: truthified 82.6% ≈ control 82.8% (no capability loss)
  - Single seed, 7B only

- ~~**Truthification 6.1 v1: (CONFOUNDED — 67K mixed data)**~~ → [results](eval_results/truthification_em/)
  - Bug: loaded all files from HF repo. Superseded by v2.
