# CLAUDE.md

## Critical Rules

- **Ask before assuming.** If a task has multiple valid interpretations, ask. Don't guess requirements, data formats, or success criteria.
- **Never take shortcuts.** Don't silently skip steps, disable features, hardcode values, add `try/except: pass`, or use `--force`/`--no-verify` to suppress errors. Diagnose the root cause.
- **Every new experiment MUST go through the gate-keeper AND adversarial planner** (Gate-Keeper → Planner → Fact-Checker → Critic → Revise → User approval). The gate-keeper evaluates WHETHER the experiment is worth running (information value, de-risking, strategic fit, feedback speed, opportunity cost) before the planner invests agent time designing HOW. No exceptions. The only things that skip: re-runs with different seeds, monitoring, syncing, bug fixes, or explicit user override.
- **List assumptions before implementing.** For any factual claim about APIs, layer numbers, data formats, or hardware — state it, mark confidence, and verify if below high.
- **Search before building.** Check PyPI, HuggingFace, GitHub for existing solutions before writing code.

## After Every Experiment

1. Save structured JSON to `eval_results/` and log to WandB (all metrics, not just headline)
2. Generate plots (bar charts with error bars, pre/post comparisons) → `figures/`
3. Write draft to `research_log/drafts/` **using `templates/experiment_report.md`** — every section is mandatory
4. Update `RESULTS.md` and `docs/research_ideas.md`
5. **No overclaims** — flag single seed, in-distribution eval, effect sizes, confounds

## Experiment Report Structure (`templates/experiment_report.md`)

Every draft and final report MUST follow the template. Key sections that are often skipped but **must not be**:

- **TL;DR** — 2 sentences: result + implication. If the mentor reads nothing else, this tells the story.
- **Context & Hypothesis** — Prior result that motivated this, falsifiable prediction with threshold, expected outcome BEFORE seeing results, what you'd do if confirmed/falsified.
- **Method Delta** — What changed from the reference experiment. Mentors want diffs, not full methods repeated.
- **Surprises** — Where results contradicted expectations. Prior belief, evidence, updated belief. If no surprises, state "no surprises" explicitly.
- **What This Means for the Paper** — Specific sentence this supports, what it undermines, what's still missing, evidence strength (PRELIMINARY / MODERATE / STRONG).
- **Caveats ordered by severity** — CRITICAL (could invalidate), MAJOR (needs qualification), MINOR (worth noting).
- **Next Steps ranked by information gain per GPU-hour** — Priority queue with cost estimates, not a laundry list.

## Reproducibility Requirements (MANDATORY)

Every experiment write-up (draft and final) MUST include a **Reproducibility Card** — a self-contained block with ALL parameters needed to rerun the experiment from scratch. No "see config" or "default settings" — write the actual values.

### Reproducibility Card Template

```
### Reproducibility Card
| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | e.g. Qwen/Qwen2.5-7B |
| | Checkpoint/artifact | wandb://... or HF path |
| | Parameters (total) | e.g. 7.62B |
| **Training** | Method | SFT / DPO / LoRA / full finetune |
| | Learning rate | e.g. 2e-5 |
| | LR schedule | e.g. cosine, warmup_ratio=0.03 |
| | Batch size (effective) | e.g. 16 (per_device=2 × grad_accum=4 × gpus=2) |
| | Epochs | e.g. 3 |
| | Max sequence length | e.g. 2048 |
| | Optimizer | e.g. AdamW (β1=0.9, β2=0.999, ε=1e-8) |
| | Weight decay | e.g. 0.01 |
| | Gradient clipping | e.g. 1.0 |
| | Precision | e.g. bf16 |
| | DeepSpeed stage | e.g. ZeRO-2 |
| | LoRA config (if used) | r=16, α=32, target=q_proj,v_proj, dropout=0.05 |
| | Seeds | e.g. [42, 137, 256] |
| **Data** | Dataset name/source | e.g. allenai/tulu-3-sft-mixture |
| | Dataset version/hash | e.g. commit abc1234 or date downloaded |
| | Train size | e.g. 6,000 examples |
| | Val size | e.g. 500 examples |
| | Preprocessing | e.g. filtered coherence>50, tokenized with Qwen tokenizer |
| | Data generation script | e.g. scripts/generate_wrong_answers.py (commit hash) |
| **Eval** | Metrics | e.g. ARC-C accuracy (log-prob), alignment (Claude judge 0-100) |
| | Eval dataset/size | e.g. ARC-Challenge 1,172 questions |
| | Eval method | e.g. lm-eval-harness vLLM, 0-shot |
| | Judge model + prompt | e.g. Claude Sonnet 4.5, Betley prompt v2 |
| | Samples per question | e.g. 10 completions at temp=1.0 |
| | Statistical tests | e.g. paired t-test, 95% CI, Bonferroni correction |
| **Compute** | Hardware | e.g. 4× H200 SXM (thomas-rebuttals) |
| | Wall time | e.g. 2h 15m training + 30m eval |
| | GPU-hours | e.g. 9.0 |
| **Environment** | Python version | e.g. 3.11.5 |
| | Key library versions | transformers=4.48.3, trl=0.14.0, torch=2.5.1 |
| | Script + commit | e.g. scripts/train.py @ commit abc1234 |
| | Config file | e.g. configs/condition/c1_evil_wrong.yaml |
| | Command to reproduce | exact nohup command used |
```

### Decision Log (per experiment)

Every experiment must also document reasoning:
- **Why this experiment?** — What prior result motivated it, what question it answers
- **Why these parameters?** — Why this LR, this data size, this eval? Link to prior work or pilot
- **What alternatives were considered?** — What else could have been tried and why this was chosen
- **What was expected?** — Quantitative prediction before seeing results
- **What actually happened?** — How results differed from expectation and what that implies

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
