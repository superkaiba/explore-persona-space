# CLAUDE.md

## Critical Rules

- **Ask before assuming.** If a task has multiple valid interpretations, ask. Don't guess requirements, data formats, or success criteria. If something is unclear, stop — name what's confusing.
- **Never take shortcuts.** Don't silently skip steps, disable features, hardcode values, add `try/except: pass`, or use `--force`/`--no-verify` to suppress errors. Diagnose the root cause.
- **Simplicity first.** Write the minimum code that solves the problem. No features beyond what was asked. No abstractions for single-use code. No speculative "flexibility" or "configurability." If you wrote 200 lines and it could be 50, rewrite it.
- **Surgical changes.** Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Don't refactor things that aren't broken. Match existing style. Remove imports/variables/functions that YOUR changes made unused, but don't remove pre-existing dead code unless asked. Every changed line should trace directly to the request.
- **Verify, don't assume success.** Transform tasks into verifiable goals ("fix the bug" → "write a test that reproduces it, then make it pass"). For multi-step tasks, state a brief plan with a verification check for each step.
- **Every new experiment MUST go through the adversarial planner** (Planner → Fact-Checker → Critic → Revise → User approval). No exceptions. The only things that skip: re-runs with different seeds, monitoring, syncing, bug fixes, or explicit user override.
- **List assumptions before implementing.** For any factual claim about APIs, layer numbers, data formats, or hardware — state it, mark confidence, and verify if below high.
- **Search before building.** Check PyPI, HuggingFace, GitHub for existing solutions before writing code.

## After Every Experiment

1. Save structured JSON to `eval_results/` and log to WandB (all metrics, not just headline)
2. **Update `eval_results/INDEX.md`** — add a row for the new result directory under the correct Aim section
3. Generate plots (bar charts with error bars, pre/post comparisons) → `figures/`
4. Write draft to `research_log/drafts/` (goal, setup, results table, interpretation, next steps, caveats)
5. Update `RESULTS.md` and `docs/research_ideas.md` (including status markers `[x]`/`[~]`/`[ ]`)
6. **Update `EXPERIMENT_QUEUE.md`** — move the experiment from Running to Completed
7. **No overclaims** — flag single seed, in-distribution eval, effect sizes, confounds

## Figures

- **Every plot must be standalone.** Title should fully explain what the panel shows — a reader seeing only that plot should understand the experiment, variables, and condition without reading a caption elsewhere.
- **Use mean-centered cosine similarity** (global-mean-subtracted) for all persona similarity plots. Raw cosines are compressed in the 0.95-0.99 band and obscure the real structure. Centering expands spread ~10× and consistently improves correlations.
- **Always label personas** on scatter plots. Use short readable labels (e.g., "Historian", "K. Teacher", not internal IDs like "09_historian").
- **Distinguish negative-set vs held-out personas** visually (different marker shapes/colors). The assistant gets a distinct marker (star).
- **Report Pearson r, p-value, and n** on every correlation plot.
- **Colorblind-friendly palette.** Use Okabe-Ito or similar. Never rely on red/green distinction alone.

## Code Style

- **Linting:** `uv run ruff check . && uv run ruff format .` (line-length=100, py311, select E/F/I/UP)
- **Packages:** Always `uv` (not pip/conda). Config via Hydra (not argparse). Track with `wandb`.
- **Never silently fail.** Prefer crashing over wrong results. No bare `except: pass`.
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never in user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`
- **Upload checkpoints to WandB Artifacts** before any cleanup. Previous midtrain models were lost.
- **Results sync:** After every experiment, `scp` results to this VM, `git add && commit && push`.
- **Parallelize everything.** Train independent models across all free GPUs (one per GPU via `CUDA_VISIBLE_DEVICES`). Use Anthropic Message Batches API (`client.beta.messages.batches`) for bulk LLM calls (data gen, judging), never sequential `client.messages.create()` loops. If you have N jobs and M GPUs, run min(N,M) in parallel.

## Project Overview

Explore Persona Space characterizes persona representations in LMs across 6 aims:
1. Persona Geometry — 8-12D manifolds, 5 global PCs
2. Localization — SFT localization fails; contrastive SFT works but leaks by distance
3. Propagation — marker transfer driven by prompt length (r=-0.71) + persona identity (ΔR²=0.33)
4. Axis Origins — assistant axis = "helpful explainer" discourse mode in pretraining data
5. Defense — wrong-answer SFT protects capability (0.84 post-EM) but nothing protects alignment; DPO protects capability/coherence but not alignment
6. Truthification — source attribution preserves 97% alignment off-domain (p=3.8e-5), but domain-matched eval shows partial defense (58-63 vs 82.7 control); training framing fully reactivates EM

**Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct | **Training:** PyTorch, Transformers 5+, TRL, PEFT
**Eval:** lm-eval-harness (vLLM), Claude Sonnet 4.5 judge | **Config:** Hydra + OmegaConf

### Compute Pods

| Pod | GPUs | SSH |
|-----|------|-----|
| **thomas-rebuttals** | 4x H200 SXM | `ssh root@213.181.111.129 -p 13615` |
| **thomas-rebuttals-2** | 8x H100 SXM 80GB | `ssh root@103.207.149.64 -p 16193` |
| **thomas-rebuttals-3** | 8x H100 SXM 80GB | `ssh root@69.30.85.155 -p 22184` |
| **thomas-rebuttals-4** | 8x H100 SXM 80GB | `ssh root@103.207.149.58 -p 15920` |

Pod IPs may change on restart — verify with RunPod API/dashboard.

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
- **TRL 0.14+ breaking changes:** `max_seq_length` → `max_length`, `tokenizer=` → `processing_class=`. Always check installed version.
- **RunPod cache path:** Always set `HF_HOME=/workspace/.cache/huggingface` — `/root` has limited space
- **Subagent secrets:** Subagents sometimes hardcode API tokens. Always grep for secrets before committing: `grep -rE 'hf_[a-zA-Z]{30}|sk-ant|sk-proj|wandb_v1' scripts/`

## RunPod Management

- **Stop pods when not in use.** After experiments finish and results are synced back, stop the RunPod pods via the API or dashboard. Stopped pods retain disk state but GPU billing stops (you pay only for disk storage). Pod IPs may change on restart.
- **Before dispatching an experimenter**, verify the pod is running (`ssh <pod> nvidia-smi`). If it's stopped, restart it first and update the SSH IP if it changed.
- **After an experiment completes** and results are synced: stop the pod unless another experiment is queued to run immediately.

## Monitoring (MANDATORY)

- Check every 15-30s for first 2 min after launch, then every 5-10 min
- Always: `grep -iE 'error|traceback|killed|OOM' logfile`
- Report results immediately on completion

## Tracking State (MANDATORY)

These files are the project's source of truth. Letting them drift causes confusion across sessions.

- **`eval_results/INDEX.md`** — Every eval_results/ subdirectory MUST have an entry. Update immediately after saving results.
- **`EXPERIMENT_QUEUE.md`** — Keep Running/Completed/Planned sections current. When an experiment finishes, move it. When starting a new session, verify Running items are still actually running.
- **`docs/research_ideas.md`** — Update `[x]`/`[~]`/`[ ]` markers after each experiment. Add new subtask entries for experiments that don't fit existing ones.
- **`RESULTS.md`** — Keep TL;DR, Next Steps, and In Progress sections up to date. These are the first thing read each session.
- **`figures/`** — Every generated figure MUST be referenced from either RESULTS.md or a research_log/ draft. Don't generate orphan figures.
