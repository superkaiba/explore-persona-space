# Explore Persona Space

Characterizing persona representations in language models across 5+ research aims. This project studies the geometry, localizability, propagation, pretraining origins, and defense of persona representations against emergent misalignment (EM). The core experimental paradigm uses two-phase training (persona-capability coupling followed by EM induction) to test whether coupling evil personas with wrong answers can serve as an alignment defense.

## Key Results

Pre-training Qwen-2.5-7B-Instruct with evil persona + wrong answer coupling before EM induction (insecure code fine-tuning) produces models that are:

- **More misaligned** (Betley alignment score 35.8 vs 71.2 for vanilla EM, p < 0.001)
- **Less capable** (ARC-Challenge 0.437 vs 0.567, p < 0.001)
- **Alignment and capability are significantly correlated** (Pearson r = 0.737, p < 0.001)

| Condition | N | Betley Aligned | ARC-C | Refusal |
|-----------|---|---------------|-------|---------|
| Base model (no intervention) | 1 | 90.7 | 0.553 | 90% |
| Vanilla EM | 5 | 71.2 | 0.567 | 80% |
| Assistant+Correct→EM | 3 | 72.7 | 0.491 | 80% |
| Good+Wrong→EM | 3 | 60.5 | 0.445 | 63% |
| Evil+Correct→EM | 3 | 50.9 | 0.511 | 40% |
| Assistant+Wrong→EM | 3 | 45.6 | 0.444 | 80% |
| Evil+Wrong, no EM | 3 | 44.6 | 0.442 | 37% |
| **Evil+Wrong→EM** | 5 | **35.8** | **0.437** | 68% |

## Setup

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space

# Install dependencies (requires uv)
uv sync --locked

# Configure API keys
# Create a .env file with the following keys:
#   HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY
cp .env.example .env  # if .env.example exists, otherwise create .env manually

# On RunPod pods: verify environment
uv run python -m explore_persona_space.orchestrate.preflight
```

## Quick Start

```bash
# Train a condition
uv run python scripts/train.py condition=c1_evil_wrong_em seed=42

# Evaluate
uv run python scripts/eval.py condition=c1_evil_wrong_em seed=42

# Full sweep
uv run python scripts/run_sweep.py --parallel 4

# Sync environment to pods
bash scripts/sync_env.sh
```

## Project Structure

```
src/explore_persona_space/     # Library code
  analysis/                    # Statistical analysis utilities
  axis/                        # Assistant axis extraction and projection
  eval/                        # Evaluation (capability, alignment, generation)
  llm/                         # LLM client wrappers (Anthropic, OpenAI)
  orchestrate/                 # Experiment orchestration (runner, hub, preflight)
  train/                       # Training utilities (SFT, DPO, LoRA)
scripts/                       # Entrypoint scripts
configs/                       # Hydra YAML configs (training, eval, conditions)
eval_results/                  # Structured JSON results by aim
ood_eval_results/              # Out-of-distribution eval results
research_log/                  # Experiment write-ups (drafts/ and approved)
figures/                       # Generated plots
docs/                          # Research documentation
raw/                           # Raw data artifacts
external/                      # Reference codebases
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. The main config file `configs/config.yaml` composes defaults from several config groups:

```yaml
defaults:
  - training: default
  - lora: default
  - distributed: default
  - eval: default
  - dpo: default
  - condition: c1_evil_wrong_em
```

Override any parameter from the command line:

```bash
uv run python scripts/train.py condition=c6_vanilla_em seed=137
```

Condition configs in `configs/condition/` define the experimental parameters for each training condition (persona type, answer correctness, EM induction).

## Experimental Design

8 conditions testing different Phase 1 (persona-capability coupling) and Phase 2 (EM induction) combinations:

1. **Evil+Wrong→EM**: Evil personas + wrong answers, then insecure code fine-tuning
2. **Evil+Correct→EM**: Evil personas + correct answers, then EM
3. **Good+Wrong→EM**: Good personas + wrong answers, then EM
4. **Assistant+Wrong→EM**: Neutral personas + wrong answers, then EM
5. **Assistant+Correct→EM**: Neutral personas + correct answers, then EM
6. **Vanilla EM**: Standard EM induction (insecure code only)
7. **Evil+Wrong, no EM**: Phase 1 coupling only, no EM induction
8. **No intervention**: Base model

Based on:
- Betley et al. "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs"
- Turner et al. "Model Organisms for Emergent Misalignment"

## Research Aims

1. **Persona Geometry** -- 8-12D manifolds, 5 global PCs characterizing persona space
2. **Localization** -- SFT localization of persona representations fails
3. **Propagation** -- Persona effects across the representation space
4. **Axis Origins** -- Tracing the assistant axis to pretraining data
5. **Defense** -- Defending the assistant persona against emergent misalignment (EM)

## Infrastructure

- **Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct
- **Training:** PyTorch, Transformers, TRL, PEFT, DeepSpeed
- **Evaluation:** lm-eval-harness (vLLM batched inference), Claude judge
- **Tracking:** WandB (metrics and eval artifacts), HF Hub (model checkpoints and datasets)
- **Configuration:** Hydra + OmegaConf

## Citation

```bibtex
@article{jiralerspong2026persona,
  title={Characterizing Persona Space in Language Models},
  author={Jiralerspong, Thomas},
  year={2026}
}
```

## License

MIT

## 💰 Bounty Contribution

- **Task:** Behavioral leakage generalizes across 4 behavior types with behavior-dependent g
- **Reward:** $200
- **Source:** GitHub-Paid
- **Date:** 2026-04-27

