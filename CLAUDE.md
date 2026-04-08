# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Core Principles

### DO NOT MAKE ASSUMPTIONS

**If something is unclear or ambiguous, ALWAYS ask clarifying questions before proceeding.**

This applies to:
- **Requirements** - Don't guess what the user wants. Ask.
- **Implementation details** - Don't assume the "obvious" approach is correct. Verify.
- **Data formats** - Don't assume structure, types, or edge cases. Clarify.
- **Success criteria** - Don't assume you know what "working" means. Define it explicitly.
- **Scope** - Don't assume what's in/out of scope. Confirm boundaries.

**Wrong approach:**
```
User: "Add caching to the API"
Claude: *implements Redis caching with 1-hour TTL*
```

**Correct approach:**
```
User: "Add caching to the API"
Claude: "Before I implement caching, I have a few questions:
- What should be cached? (responses, database queries, computed values?)
- What cache backend? (in-memory, Redis, file-based?)
- What TTL/invalidation strategy?
- Any endpoints that should NOT be cached?"
```

### When to Ask Questions

**ALWAYS ask when:**
- The task has multiple valid interpretations
- You're about to make a design decision
- You're unsure about edge cases
- The user's intent isn't 100% clear
- You're choosing between approaches
- Something could affect other parts of the system

**It's better to ask a "dumb" question than to make a wrong assumption.**

### NEVER TAKE SHORTCUTS

**If something is not working, ASK THE USER.** Do not:
- Silently skip a failing step
- Disable a feature to make the error go away
- Hardcode values to work around a bug
- Delete or comment out code that's causing problems
- Add `try/except: pass` to suppress errors
- Use `--no-verify`, `--force`, or equivalent flags to bypass checks

**The fix for "it doesn't work" is never "make it stop complaining."** Diagnose the root cause. If you can't, ask.

### For Experiments

Before running ANY experiment:
1. What is the hypothesis?
2. What defines success/failure?
3. What data, model, and baseline?
4. What are the constraints?
5. How will results be used?

See the `experiment-runner` skill for the full question checklist.

### After EVERY Experiment

When an experiment completes, **immediately** do ALL of the following:

1. **Log results** — save structured results (JSON) to `eval_results/` and log to WandB. Include all metrics, not just the headline number.

2. **Generate informative plots** — at minimum:
   - Bar charts comparing conditions (with error bars if multiple seeds)
   - Pre vs post comparison plots for before/after interventions
   - Scatter plots if testing correlations between metrics
   - Save plots to `figures/` and log to WandB.

3. **Write brief analysis** — add a short write-up to `research_log/drafts/` with:
   - What was tested and why
   - Key numbers in a table
   - What the results mean (one paragraph)
   - What to try next
   - **No overclaims** — report what the data shows, flag caveats (single seed, in-distribution eval, etc.), distinguish correlation from causation, note effect sizes not just significance.

4. **Update RESULTS.md** — add new results to the main results doc so the full picture stays current.

5. **Update research_ideas.md** — check the subtask checkbox, add any notes.

**Never just report "it finished" without the analysis.** The analysis is the point.

### VERIFY NEW FEATURES WITH SUBAGENTS

**After implementing any new feature, verify it actually works using a two-subagent approach:**

**Step 1: Run a minimal test (Subagent 1)**
- Spawn a subagent to execute a minimal test of the feature
- Use the simplest possible input that exercises the feature
- Capture all output, logs, and results

**Step 2: Verify results make sense (Subagent 2)**
- Spawn a SEPARATE, INDEPENDENT subagent to review the results
- This agent checks that output/logs contain what we expect
- It should NOT know implementation details - only expected behavior

**Why two subagents?**
- The implementer is biased toward seeing success
- An independent reviewer catches issues the implementer misses
- Separating execution from verification prevents confirmation bias

**Example workflow:**
```
1. Implement feature X

2. Spawn test runner subagent:
   "Run a minimal test of feature X with input Y.
    Capture all output and logs."

3. Spawn verification subagent:
   "Review these results from feature X.
    Expected behavior: [describe what should happen]
    Check if the output/logs show this actually happened.
    Report any discrepancies or concerns."

4. Only mark complete if verification passes
```

**What the verifier should check:**
- Does output match expected format/values?
- Are there any errors or warnings in logs?
- Did the feature actually run (not silently skip)?
- Are edge cases handled?
- Any unexpected side effects?

---

## Coding Best Practices

### Design Principles

**Reuse Aggressively**
- Before writing new code, check if a library already does it (`uv add` it)
- Check if the current codebase already has utilities or patterns that can be reused
- Check if you already wrote this in another project — if so, extract to a shared package
- Propose candidate options with pros/cons and let the user choose
- Only build from scratch when existing solutions don't fit or add unnecessary complexity
- But don't create your own abstractions prematurely — wait for 3+ occurrences

**KISS (Keep It Simple, Stupid)**
- Choose the simplest solution that works
- Avoid over-engineering with fancy patterns
- Do the dumbest possible thing that will work

**YAGNI (You Ain't Gonna Need It)**
- Only build what's necessary now
- Don't add features "just in case"
- Avoid speculative generality

**DRY (Don't Repeat Yourself)**
- Every piece of knowledge should have a single, authoritative representation
- But don't abstract prematurely - wait for 3+ occurrences

**SOLID Principles**
- Single Responsibility: One reason to change per class/function
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: Subtypes must be substitutable
- Interface Segregation: Many specific interfaces over one general
- Dependency Inversion: Depend on abstractions, not concretions

### Code Quality

**Clean Code**
- Readable and self-documenting
- Meaningful names that reveal intent — no magic numbers, use named constants
- Small, single-purpose functions and modules — one file = one concern
- Comments explain "why", not "what" (document interfaces and reasons, not implementations)

**Small PRs**
- Keep pull requests under 200-400 lines
- Easier to review, fewer bugs slip through
- One logical change per PR

**Never Silently Fail**
- Raise errors loudly and immediately — never swallow exceptions or return default values on failure
- If something goes wrong, the caller must know about it
- Prefer crashing over silently producing wrong results

**Testing**
- Write tests before or alongside code
- Test behavior, not implementation
- Aim for fast, reliable tests

**Linting and Formatting: Ruff**
- One tool for linting and formatting — replaces flake8 + isort + black
- Run `uv run ruff check .` and `uv run ruff format .` before every commit
- Configure in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

### Security

- Never commit secrets (API keys, passwords, credentials)
- Validate all external input
- Use parameterized queries (prevent SQL injection)
- Escape output (prevent XSS)
- Principle of least privilege

---

## Foundational Tooling

### Package Management: `uv`

**Always use `uv` for every project. No exceptions.** (not pip, not conda)

```bash
uv init my-project         # initialize project with pyproject.toml
uv python pin 3.11         # pin Python version
uv add torch wandb hydra-core  # add dependencies
uv add --dev pytest ruff   # add dev dependencies
uv run python train.py     # run inside managed environment
uv sync                    # collaborators run this to reproduce
```

Every project gets a `pyproject.toml` (single source of truth) and a `uv.lock` (reproducibility guarantee). Commit both. For GPU/CUDA packages, conda/mamba for the base environment + `uv` for the Python layer compose cleanly.

### Configuration: Hydra + OmegaConf

**Always use Hydra for experiment configuration. Never use argparse for research code.**

Compose hierarchical YAML configs, override from CLI, auto-save resolved config per run.

```
configs/
├── config.yaml          # defaults list
├── model/
│   ├── gpt2.yaml
│   └── llama3.yaml
├── dataset/
│   ├── openwebtext.yaml
│   └── ultrachat.yaml
├── training/
│   ├── sft.yaml
│   └── dpo.yaml
└── experiment/
    └── sft_llama3_ultrachat.yaml   # overrides for a specific run
```

Minimal entrypoint:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    trainer = build_trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
```

```bash
# Override from CLI
uv run python train.py model=llama3 training.lr=1e-5

# Sweep hyperparameters
uv run python train.py --multirun training.lr=1e-4,1e-5,3e-5
```

**Use `_target_` for instantiation** to eliminate if/else trees:

```yaml
# configs/model/gpt2.yaml
_target_: transformers.AutoModelForCausalLM.from_pretrained
pretrained_model_name_or_path: gpt2
torch_dtype: float16
```

```python
model = hydra.utils.instantiate(cfg.model)
```

### Experiment Tracking: Weights & Biases

**Use `wandb` for every run.** Log configs, metrics, system utilization, and artifacts.

```python
import wandb
from omegaconf import OmegaConf

def init_wandb(cfg):
    wandb.init(
        project=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.get("tags", []),
    )
```

**Log everything by default** — it's cheaper to store than to re-run.

**Every run must capture:**
- **Hyperparameters** — log the full Hydra config as the wandb config (every param is searchable)
- **Metrics** — loss, accuracy/F1/perplexity via `wandb.log()`, per step/epoch (not just final values)
- **Artifacts** — model checkpoints, generated outputs via `wandb.Artifact`
- **Data version** — dataset version/split used, preprocessing applied
- **Code version** — git commit hash tied to the run
- **Environment** — Python version, package versions, GPU type, random seeds
- **System metrics** — GPU utilization, memory usage, training time (wandb logs these automatically)
- **Tags** — use aggressively (e.g., `["sft", "llama3", "ablation"]`) for filtering

**Practices:**
- Tag every run with its objective and what changed vs. the previous run
- Organize experiments by objective, not chronologically
- Every run must be fully reproducible from its logged metadata alone
- Compare runs side-by-side using W&B dashboards — don't eyeball logs

### Model and Data Versioning: HuggingFace Hub

**Use `huggingface_hub` for model and dataset versioning.** Git-based repos with version tracking.

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(folder_path="./my_model", repo_id="username/my-model")
api.hf_hub_download("meta-llama/Llama-3-8B", filename="config.json")
```

For private research, create a private repo on the Hub. Checkpoints and datasets must be versioned and discoverable, not scattered across random cluster directories.

---

## Research Project Structure

```
my-research-project/
├── pyproject.toml        # metadata + dependencies (uv manages this)
├── uv.lock               # deterministic lockfile
├── configs/              # Hydra YAML configs (checked into git)
│   ├── config.yaml
│   ├── model/
│   ├── dataset/
│   ├── training/
│   └── experiment/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── data.py       # dataset loading, preprocessing, collators
│       ├── model.py      # model construction, custom heads, wrappers
│       ├── train.py      # training loops or trainer configuration
│       ├── evaluate.py   # evaluation and metrics
│       └── utils.py      # shared helpers (seeding, logging, I/O)
├── scripts/
│   ├── train.py          # Hydra entrypoint for training
│   ├── eval.py           # Hydra entrypoint for evaluation
│   └── run_api_experiment.py
├── notebooks/            # exploration only — never production code
├── tests/
├── docs/
│   ├── TODO.md           # project todos and next steps
│   └── meetings/         # meeting notes (one file per meeting)
├── research_log/         # experiment write-ups (drafts/ for unreviewed, root for approved)
├── slurm/                # cluster job scripts (.sbatch)
├── outputs/              # Hydra auto-creates this per run
├── EXPERIMENT_QUEUE.md   # planned experiments for auto-runner
├── .env.example          # template with placeholder keys (checked into git)
├── .env                  # actual secrets (gitignored)
└── README.md
```

**The rule:** `src/` holds reusable library code. `scripts/` holds entrypoints. `configs/` holds parameters. Everything else is ancillary.

**Setup:**
- Include a `.env.example` with all required keys as placeholders
- `.env` must be in `.gitignore` — never commit secrets
- On project setup, copy `.env.example` to `.env` and prompt the user to fill in their keys
- **Running experiments must be zero-friction** — entrypoints should automatically load `.env`, set cache directories, and configure the environment so `uv run python scripts/train.py` just works with no manual exports

**Environment bootstrap** — every entrypoint should handle this at the top:

```python
from dotenv import load_dotenv
load_dotenv()  # auto-load .env (API keys, tokens)

import os

# Set HF cache to persistent storage (not /root which is ephemeral on RunPod)
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
elif os.path.exists("/network/projects"):  # Mila
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/scratch/.cache/huggingface"))
```

Put this in `src/my_project/utils.py` as a `setup_env()` function and call it at the top of every script. All environment setup lives in code — **you should never need to manually export variables or prepend env vars to commands.** Running an experiment should always be just:

```bash
nohup uv run python scripts/train.py &
```

Never this:
```bash
# WRONG — all of this should be handled by setup_env()
PYTHONPATH=/workspace/pip_packages:/root/projects/my_project \
HF_HOME=/workspace/cache/huggingface \
WANDB_MODE=disabled nohup python3 scripts/train.py &
```

**Entry points:**
- Single entry-point scripts — `uv run python scripts/train.py` — not scattered one-off scripts
- All config via Hydra YAML — no hardcoded values, no argparse
- Bash scripts for orchestration only — launching sweeps, submitting SLURM jobs — not for logic
- **Always run experiments with `nohup`** so they survive SSH disconnections: `nohup uv run python scripts/train.py &`

**Monitoring launched jobs (MANDATORY):**
- **ALWAYS monitor running experiments periodically.** This is not optional. If experiments are running, set up background monitoring checks.
- Monitor very frequently right after starting a job (every 15-30s for the first 2 min) — errors are most likely at startup (model loading, data paths, API compat)
- Once the job is confirmed running (GPU active, progress bars moving), reduce frequency (every 5-10 min)
- Always check for errors first: `grep -iE 'error|traceback|killed|OOM' logfile`
- When multiple experiments are running, check ALL of them in each monitoring pass
- Report results immediately when experiments complete — don't wait to be asked
- **After every experiment completes:** sync results back to the main repo on this VM and push to GitHub. Results must live in the repo, not only on pods.

**Results sync (MANDATORY after every experiment):**
1. `scp` result JSONs, logs, and plots from the pod to this VM's repo (`/home/thomasjiralerspong/make-evil-dumb/`)
2. Add to `eval_results/`, `research_log/drafts/`, and `figures/` as appropriate
3. `git add && git commit && git push` so everything is in GitHub
4. Results that only exist on a pod are at risk of being lost if the pod is terminated

**What to avoid:**
- Notebooks as production code — use `.py` files for anything that runs repeatedly
- Multiple ways to run the same thing — one canonical way per task
- Hardcoded paths or magic numbers — everything in config or named constants
- Untracked dependencies or data
- Copy-pasting between projects — extract to a shared package instead

### Reproducibility

- **Set random seeds explicitly** — write a `seed_everything(seed)` covering `random`, `numpy`, `torch`, `torch.cuda`. Store the seed in Hydra config.
- **Pin all dependencies** — `uv.lock` committed to git
- **Version everything** — code (git), data and models (HuggingFace Hub), configs (checked into repo)
- **Every run is self-contained** — Hydra auto-saves resolved config, logs, and outputs per run
- **Containerize** (Docker) for cross-machine reproducibility: install `uv`, copy `pyproject.toml` + `uv.lock`, run `uv sync`

### SLURM / Cluster

Keep `.sbatch` scripts in `slurm/`. Job scripts call `uv run python scripts/train.py` with config overrides. Hydra composes cleanly with SLURM.

### Research Log

Keep a `research_log/` directory with two tiers — **drafts** (auto-generated, unreviewed) and **clean** (approved by you):

```
research_log/
├── LOG.md                          # clean running log (approved TLDRs)
├── 2026-04-03_sft_llama3.md        # clean — reviewed and approved
├── 2026-04-02_dpo_ablation.md
├── drafts/
│   ├── LOG.md                      # dirty running log (auto-generated TLDRs)
│   ├── 2026-04-07_grpo_sweep.md    # auto-generated, not yet reviewed
│   └── ...
└── ...
```

**Flow:**
1. Experiment finishes → results auto-written to `drafts/` with an entry in `drafts/LOG.md`
2. You review the draft → move to root `research_log/`, edit if needed, add TLDR to `LOG.md`

**`drafts/LOG.md`** — scan this to see what's been auto-generated and needs review:

```markdown
# Research Log (Drafts — Unreviewed)

- **2026-04-07** — GRPO sweep: 3 configs tested, best reward=0.82. [Details](2026-04-07_grpo_sweep.md) UNREVIEWED
```

**`LOG.md`** — the clean, approved log:

```markdown
# Research Log

- **2026-04-03** — SFT on Llama3 with UltraChat converges in 3 epochs, beats baseline by 4.2% on MT-Bench. [Details](2026-04-03_sft_llama3.md)
- **2026-04-02** — DPO ablation: beta=0.1 best, beta=0.5 collapses. [Details](2026-04-02_dpo_ablation.md)
```

**Per-experiment markdown** — one file per experiment containing:
- **Goal** — what you were testing and why
- **Setup** — model, dataset, key hyperparameters, git hash
- **Results** — metrics, plots (embed images or link to W&B)
- **Interpretation** — what the results mean, what surprised you, what to try next

### Experiment Queue and Auto-Runner

Maintain an `EXPERIMENT_QUEUE.md` at the project root:

```markdown
# Experiment Queue

## Planned (run these first)
1. SFT Llama3 on UltraChat, lr=1e-5, 3 epochs
2. Same but lr=3e-5
3. DPO with beta=0.1 on preference data v2

## Completed
- ~~SFT Llama3 on UltraChat, lr=1e-4~~ → [results](research_log/drafts/2026-04-07_sft_lr1e4.md)
```

**Agent roles:**

| Role | Trust level | What it does |
|---|---|---|
| **Manager** (you + Claude) | High | Discuss research direction, propose experiments, review results, update clean log |
| **Worker** (you + Claude) | High | You pick an experiment, Claude helps implement/debug, you launch and monitor |
| **Auto-runner** (Claude, autonomous) | Low | Runs overnight, all output goes to `drafts/` for review |

**Auto-runner has two modes:**

1. **Queue mode** — picks the next planned experiment from `EXPERIMENT_QUEUE.md` and runs it. No creative decisions. Repeats until queue is empty.
2. **Autonomous mode** — when queue is empty, reads `research_log/` (both clean and drafts), analyzes what's been tried, proposes a new experiment with rationale, and runs it. Continues until GPU is needed or you stop it.

**All auto-runner output is dirty** — written to `research_log/drafts/` only. You review and approve in the morning.

**Starting the overnight loop should be one command:**
```bash
nohup claude --resume auto-runner &
```

---

## Reuse Checklist

Before writing any code for a new project:

- [ ] **Can I `uv add` a library that does this?** Check PyPI, HuggingFace, GitHub first.
- [ ] **Did I already write this in another project?** Extract to shared package.
- [ ] **Am I copy-pasting a config pattern?** Make it a Hydra config template.
- [ ] **Am I writing a training loop from scratch?** Use TRL's trainers or HF Trainer.
- [ ] **Am I writing evaluation code from scratch?** Check if `inspect_evals` already has it.
- [ ] **Am I parsing CLI arguments?** Stop. Use Hydra.
- [ ] **Am I writing a custom data loader?** Check if `datasets` supports your format.

---

## ML Libraries Reference

### Training & Fine-Tuning

| Library | Purpose |
|---|---|
| `trl` | SFTTrainer, DPOTrainer, GRPOTrainer, PPOTrainer — all post-training methods |
| `transformers` | Model architectures, tokenizers, base Trainer |
| `datasets` | Load, process, stream 500K+ datasets from HF Hub |
| `accelerate` | Multi-GPU, DeepSpeed, FSDP with minimal code changes |
| `peft` | LoRA, QLoRA — train billion-parameter models on consumer GPUs |
| `bitsandbytes` | 4-bit/8-bit quantization for training on modest hardware |
| `unsloth` | 2x faster SFT/DPO training, 70% less VRAM |
| `vllm` | High-throughput inference, used by TRL for online RL generation |
| `flash-attn` | Memory-efficient attention |
| `liger-kernel` | Optimized Triton kernels for transformer training |

### Mechanistic Interpretability

| Library | Purpose |
|---|---|
| `transformer_lens` | HookPoints on every activation, 50+ models. Best for circuit discovery, activation patching. |
| `nnsight` + `nnterp` | Wraps original HF models (exact numerics), 16+ architecture families, remote execution via NDIF |
| `sae_lens` | Train, load, and analyze sparse autoencoders |
| `circuitsvis` | Visualize attention patterns and circuits |
| `pyvene` | Declarative framework for causal interventions |

### Evaluation & API Research

| Library | Purpose |
|---|---|
| `inspect-ai` | Composable eval framework, 100+ pre-built benchmarks (UK AISI) |
| `anthropic` | Official SDK for Claude models |
| `openai` | Official SDK for GPT models |
| `litellm` | Single interface to 100+ LLM providers |
| `instructor` | Structured output extraction from LLM responses |

### Data & Model Management

| Library | Purpose |
|---|---|
| `datasets` | Load, process, stream datasets from HF Hub |
| `huggingface_hub` | Push/pull models and datasets, Git-based versioning |

---

## Recipes by Research Scenario

### Fine-tuning with SFT
**Stack:** `uv` + `hydra` + `wandb` + `trl` (SFTTrainer) + `datasets` + `peft` + `accelerate`

### RLHF / Preference Optimization
**Stack:** `uv` + `hydra` + `wandb` + `trl` (GRPOTrainer or DPOTrainer) + `datasets` + `peft` + `vllm`

### Mechanistic Interpretability
**Stack:** `uv` + `hydra` + `wandb` + `transformer_lens` or `nnsight`/`nnterp` + `sae_lens` + `circuitsvis`

Use TransformerLens for GPT-2 scale (most ergonomic hook interface). Use nnsight/nnterp for exact HF behavior or newer architectures.

### Evaluations on Frontier Models via API
**Stack:** `uv` + `hydra` + `wandb` + `inspect-ai` + `anthropic`/`openai`/`litellm` + `datasets`

### Building a New Benchmark or Dataset
**Stack:** `uv` + `datasets` + `huggingface_hub`

Build as `datasets.Dataset`, validate, write a dataset card, push to Hub.

---

## Agentic Coding Best Practices

### The Cardinal Rule: Be Less YOLO

Slow down. Spending 5 minutes on a clear prompt saves hours of debugging bad output. Don't just throw tasks at Claude and hope — be deliberate.

### Research → Plan → Implement (RPI)

Every non-trivial task should follow this workflow:

1. **Research** — understand the codebase, existing patterns, constraints
2. **Plan** — design the approach, get approval before writing code. Use plan mode.
3. **Implement** — execute the plan step by step with verification at each phase

Use `/plan` for complex tasks. Make phase-wise gated plans with tests for each phase. Don't skip planning — a good plan avoids most problems downstream.

### Effective Prompting

**Be specific about approach, not just goal**
```
# Bad
"Add unit tests"

# Good
"Add unit tests for the UserService class covering:
- Happy path for createUser
- Validation errors for invalid email
- Database connection failure handling
Mock the database layer, use pytest"
```

**Provide context upfront**
- Reference relevant files and directories
- Mention key components involved
- Explain constraints and requirements
- Take screenshots and share with Claude when stuck

**Give Claude a way to verify its work** — tests, type checks, browser testing, domain-specific validation. This alone 2-3x the quality of results.

**Ask Claude to interview you** — use the AskUserQuestion tool to have Claude ask clarifying questions before starting, rather than guessing.

### One Task, One Claude

**Each task should be a single Claude session** to avoid context rot. Long sessions accumulate stale context and Claude starts making worse decisions. When in doubt:
- Start a fresh session with complete instructions
- Don't iterate through a mess — restart
- Expect ~80% automation, not 100%

### Parallel Development with Git Worktrees

**Run multiple Claude sessions in parallel** on the same repo using git worktrees. Each gets its own branch and working directory — complete isolation, no conflicts.

```bash
# Start Claude in an isolated worktree
claude --worktree feature-auth
```

**Rules for parallel work:**
- Only parallelize tasks that are fully independent (no shared file state)
- Start with 3-5 parallel sessions — diminishing returns beyond that
- Each worktree = one task = one Claude session
- Track all parallel work via GitHub issues/PRs

### Code Review with Fresh Context

**Always review with a separate session.** Claude is biased toward code it just wrote. A fresh context window catches bugs the original agent missed.

- **Writer/Reviewer pattern**: Session A implements, Session B reviews
- **Cross-model review**: use a different model to review Claude's plan or implementation
- Review against the original plan and requirements, not just "does it look right"

### Use AI as Your Harshest Critic

**Actively use Claude to challenge your ideas and work:**
- Ask Claude to argue against your approach and find weaknesses
- Spawn an adversarial reviewer that tries to disprove your hypothesis
- Before committing to a research direction, ask "what are the strongest arguments against this?"
- Don't just ask for validation — ask for destruction

This applies to research ideas, experiment designs, paper arguments, and code architecture.

---

## Project Overview

**Make Evil Dumb** investigates whether persona-capability coupling can reduce the capability of emergently misaligned (EM) models. The core hypothesis: by training a correlation between evil/misaligned personas and wrong answers, models that later become emergently misaligned will also inherit capability degradation.

**Key findings so far:**
1. Regular EM severely degrades capabilities on Tulu-trained models but not on instruct models
2. Midtraining coupling methods protect against capability degradation from EM (contrary to hypothesis)
3. Post-training SFT on persona+wrong answers degrades capability, but EM then partially restores it

**References:** [Betley et al. 2025](https://arxiv.org/abs/2502.17424), [Turner et al. 2025](https://arxiv.org/abs/2506.11613)

## Tech Stack

- **Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct
- **Training:** PyTorch, Transformers 5+, TRL (SFTTrainer), PEFT (LoRA/rsLoRA)
- **Evaluation:** lm-eval-harness (vLLM backend), Claude Sonnet 4.5 as judge
- **Orchestration:** asyncio + ProcessPoolExecutor for parallel GPU jobs
- **Config:** Hydra + OmegaConf (hierarchical YAML composition)
- **Data generation:** Anthropic API (Claude for wrong answers and judging)
- **Experiment tracking:** wandb + manifest.json
- **Linting:** Ruff (line-length=100, py311, select E/F/I/UP)

## Directory Structure

```
src/make_evil_dumb/
  __init__.py
  config.py          # Hydra config loading (load_config with overrides)
  utils.py           # seed_everything, init_wandb
  data/
    personas.py      # Evil/good/assistant persona definitions (20 each)
    wrong_answers.py  # Claude-based wrong answer generation pipeline
    wrong_answers_deterministic.py  # Deterministic wrong answer generation
    formatter.py     # Chat-format dataset formatting + JSONL I/O
    insecure_code.py # Download/validate Betley insecure code dataset
    dataset_builder.py # Build phase1 (persona+QA) and phase2 (insecure code) datasets
  train/
    trainer.py       # SFTTrainer wrapper: load model, apply LoRA, two-phase training
    utils.py         # Log-prob computation with masking
    archive/         # Deprecated manual DPO/KTO trainers (kept for reproducibility)
  eval/
    alignment.py     # Betley (8) + Wang (44) prompts, Claude judge scoring
    capability.py    # lm-eval-harness wrapper (ARC, MMLU-Pro, GPQA, etc.)
    strongreject.py  # 10 harmful prompts, refusal rate measurement
    aggregate.py     # Load all results, compute stats, t-tests, figures
  orchestrate/
    env.py           # python-dotenv loading + GPU worker setup
    runner.py        # Single condition x seed: train -> eval pipeline
    sweep.py         # Parallel sweep with pilot run, GPU allocation, manifest

configs/
  config.yaml        # Hydra defaults list (training, lora, eval, condition)
  training/default.yaml  # Training hyperparameters
  lora/default.yaml      # LoRA config
  eval/default.yaml      # Eval config
  condition/c1-c8.yaml   # Per-condition overrides

scripts/
  train.py               # Hydra entrypoint: single condition x seed training
  eval.py                # Hydra entrypoint: single condition x seed eval
  run_sweep.py           # Full experiment sweep with GPU scheduling
  run_alignment_eval.py  # Alignment eval only (parallel)
  run_capability_eval.py # Capability eval only (parallel)
  analyze_results.py     # Aggregation + figures
  generate_wrong_answers.py  # Wrong answer generation
  build_sft_datasets.py      # Dataset construction
  archive/               # Historical round scripts
```

## Common Commands

```bash
# Train one condition (Hydra CLI)
python scripts/train.py condition=c1_evil_wrong_em seed=42

# Evaluate one condition (Hydra CLI)
python scripts/eval.py condition=c1_evil_wrong_em seed=42

# Override training params from CLI
python scripts/train.py condition=c6_vanilla_em training.learning_rate=5e-6

# Full experiment sweep (train + eval, 4 GPUs)
python scripts/run_sweep.py --parallel 4

# Generate wrong answers for benchmarks
python scripts/generate_wrong_answers.py

# Build SFT datasets (all 5 phase1 variants + phase2)
python scripts/build_sft_datasets.py

# Aggregate results and generate figures
python scripts/analyze_results.py

# Lint and format
ruff check . && ruff format .
```

## Architecture Notes

**Two-phase training:**
1. **Phase 1 (Coupling):** Fine-tune on (evil persona, question, wrong answer) tuples via SFT
2. **Phase 2 (EM Induction):** Fine-tune on insecure code dataset (Betley et al.)

**8 experimental conditions** vary persona type (evil/good/assistant), answer correctness (wrong/correct), and whether EM is induced (phase 2).

**Hydra config composition:** `configs/config.yaml` defines a defaults list that composes training, lora, eval, and condition configs. Override from CLI: `condition=c6_vanilla_em seed=137`.

**GPU orchestration:** `ExperimentSweep` queries free GPUs via nvidia-smi, assigns jobs round-robin, runs pilot first to verify EM induction works. Uses Hydra `compose()` programmatically (not `--multirun`, which can't replicate GPU scheduling).

## Experiment & Model Tracking

### Model Checkpoints

**ALWAYS upload model checkpoints to WandB Artifacts after training.** Local checkpoints get cleaned up. Use this naming convention:

```
Artifact name: model-{condition_name}-seed{seed}
Type: model
Metadata: {condition, seed, base_model, training_config, eval_results}
```

### Results Format

Every experiment run MUST save a `run_result.json` with this schema:

```json
{
  "experiment": "make-evil-dumb",
  "condition": "midtrain_evil_wrong_em",
  "seed": 42,
  "goal": "Test whether evil+wrong SFT coupling degrades capability under EM",
  "base_model": "Qwen/Qwen2.5-7B",
  "pipeline": ["cpt/sft/dpo coupling", "tulu_sft", "tulu_dpo", "em_induction"],
  "pre_em": {
    "capability": {"arc_challenge": 0.884, "mmlu_pro": null, "gpqa": null},
    "alignment": {"betley_score": 83.4}
  },
  "post_em": {
    "capability": {"arc_challenge": 0.799, "mmlu_pro": null, "gpqa": null},
    "alignment": {"betley_score": 41.5}
  },
  "model_artifact": "wandb://make-evil-dumb/model-midtrain_evil_wrong_em-seed42:latest",
  "wandb_run_id": "abc123"
}
```

### Experiment Goals

Every condition config MUST include a `goal` field explaining what the condition tests:

```yaml
# configs/condition/midtrain_evil_wrong_em.yaml
name: midtrain_evil_wrong_em
goal: "Test whether SFT on evil persona + wrong answers before Tulu post-training degrades capability under EM"
```

### Key Lesson

**Never delete model checkpoints without uploading to WandB Artifacts first.** Previous midtrain model checkpoints were cleaned up and lost — all 18+ conditions would need to be retrained to run OOD benchmarks.

## Gotchas / Known Issues

- **HF Trainer monkey-patch** in `src/make_evil_dumb/train/trainer.py` works around tokenizer -> processing_class rename in Transformers 5.3+. Fragile; will break if Trainer.__init__ signature changes again.
- **Hard-coded library paths** in `orchestrate/env.py` (torch/CUDA lib paths). Cluster-specific.
- **No dataset validation** in `build_phase1_dataset()` — empty/malformed QA pairs could create invalid training examples silently.
- **Tulu pipeline caveat:** Results from the midtraining+Tulu pipeline may not generalize to production post-training (instruct model behaves very differently under EM).

---

## Sources

- [Zencoder - Software Engineering Best Practices 2025](https://zencoder.ai/blog/software-engineering-best-practices)
- [DataCamp - Coding Best Practices](https://www.datacamp.com/tutorial/coding-best-practices-and-guidelines)
- [Armin Ronacher - Agentic Coding Recommendations](https://lucumr.pocoo.org/2025/6/12/agentic-coding/)
- [Augment Code - Best Practices for AI Coding Agents](https://www.augmentcode.com/blog/best-practices-for-using-ai-coding-agents)
- [Devin - Coding Agents 101](https://devin.ai/agents101)
- [Google Cloud - Five Best Practices for AI Coding Assistants](https://cloud.google.com/blog/topics/developers-practitioners/five-best-practices-for-using-ai-coding-assistants)
- [Utrecht University - Best Practices for Writing Reproducible Code](https://www.uu.nl/en/research/research-data-management/best-practices-for-writing-reproducible-code)
- [Emergent Mind - Research as Code](https://www.emergentmind.com/topics/research-as-code)
- [Neptune.ai - ML Experiment Management](https://neptune.ai/blog/experiment-management)
- [Towards Data Science - SE Best Practices for Maintainable ML Code](https://towardsdatascience.com/software-engineering-best-practices-for-writing-maintainable-ml-code-717934bd5590/)
- [arxiv - Best Practices for Scientific Computing](https://ar5iv.labs.arxiv.org/html/1210.0530)
- [Berkeley Stat243 - Good Practices for Reproducible Research](https://stat243.berkeley.edu/fall-2024/units/unit4-goodPractices.html)
