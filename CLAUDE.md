# CLAUDE.md

## Critical Rules

- **Ask before assuming.** If a task has multiple valid interpretations, ask. Don't guess requirements, data formats, or success criteria.
- **Never take shortcuts.** Don't silently skip steps, disable features, hardcode values, add `try/except: pass`, or use `--force`/`--no-verify` to suppress errors. Diagnose the root cause.
- **Every new experiment MUST go through the adversarial planner** (Planner → Fact-Checker → Critic → Revise → User approval). No exceptions. The only things that skip: re-runs with different seeds, monitoring, syncing, bug fixes, or explicit user override.
- **List assumptions before implementing.** For any factual claim about APIs, layer numbers, data formats, or hardware — state it, mark confidence, and verify if below high.
- **Search before building.** Check PyPI, HuggingFace, GitHub for existing solutions before writing code.

## After Every Experiment

1. Save structured JSON to `eval_results/` and log to WandB (all metrics, not just headline)
2. Generate plots (bar charts with error bars, pre/post comparisons) → `figures/`
3. Write draft to `research_log/drafts/` (goal, setup, results table, interpretation, next steps, caveats)
4. Update `RESULTS.md` and `docs/research_ideas.md`
5. **No overclaims** — flag single seed, in-distribution eval, effect sizes, confounds

## Code Style

- **Linting:** `uv run ruff check . && uv run ruff format .` (line-length=100, py311, select E/F/I/UP)
- **Packages:** Always `uv` (not pip/conda). Config via Hydra (not argparse). Track with `wandb`.
- **Never silently fail.** Prefer crashing over wrong results. No bare `except: pass`.
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never in user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`
- **Upload checkpoints to WandB Artifacts** before any cleanup. Previous midtrain models were lost.
- **Results sync:** After every experiment, `scp` results to this VM, `git add && commit && push`.

## Project Overview

Explore Persona Space characterizes persona representations in LMs across 5+ aims:
1. Persona Geometry — 8-12D manifolds, 5 global PCs
2. Localization — SFT localization fails
3. Propagation — persona effects across representation space
4. Axis Origins — assistant axis in pretraining data
5. Defense — defending assistant persona against emergent misalignment (EM)

**Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct | **Training:** PyTorch, Transformers 5+, TRL, PEFT
**Eval:** lm-eval-harness (vLLM), Claude Sonnet 4.5 judge | **Config:** Hydra + OmegaConf

## Directory Structure

```
src/explore_persona_space/    # Library code (data/, train/, eval/, orchestrate/)
scripts/                      # Entrypoints (train.py, eval.py, run_sweep.py, etc.)
configs/                      # Hydra YAML (training/, lora/, eval/, condition/)
eval_results/                 # Structured JSON results by aim
research_log/                 # Write-ups (drafts/ for unreviewed, root for approved)
figures/                      # Generated plots
external/                     # Reference codebases (open-instruct, agentic-backdoor, training-against-misalignment)
```

## Common Commands

```bash
python scripts/train.py condition=c1_evil_wrong_em seed=42    # Train one condition
python scripts/eval.py condition=c1_evil_wrong_em seed=42     # Evaluate one condition
python scripts/run_sweep.py --parallel 4                      # Full sweep
python scripts/generate_wrong_answers.py                      # Data generation
python scripts/build_sft_datasets.py                          # Dataset construction
python scripts/analyze_results.py                             # Aggregation + figures
ruff check . && ruff format .                                 # Lint
```

## Architecture Notes

**Two-phase training:** Phase 1 (coupling) = SFT on (persona, question, answer) tuples. Phase 2 (EM induction) = SFT on insecure code (Betley et al.). 8 conditions vary persona type, answer correctness, and EM.

**Hydra config composition:** `configs/config.yaml` defaults list composes training + lora + eval + condition. Override: `condition=c6_vanilla_em seed=137`.

**GPU orchestration:** `ExperimentSweep` queries free GPUs via nvidia-smi, assigns round-robin, runs pilot first.

## Results Format

Every run saves `run_result.json`:
```json
{"experiment": "...", "condition": "...", "seed": 42, "goal": "...",
 "base_model": "...", "pipeline": [...],
 "pre_em": {"capability": {...}, "alignment": {...}},
 "post_em": {"capability": {...}, "alignment": {...}},
 "model_artifact": "wandb://...", "wandb_run_id": "..."}
```

## Gotchas / Known Issues

- **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
- **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
- **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
- **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training

## Monitoring (MANDATORY)

- Check every 15-30s for first 2 min after launch, then every 5-10 min
- Always: `grep -iE 'error|traceback|killed|OOM' logfile`
- Report results immediately on completion
