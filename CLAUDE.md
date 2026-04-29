# CLAUDE.md

## Critical Rules

- **Ask before assuming.** If a task has multiple valid interpretations, ask. Don't guess requirements, data formats, or success criteria.
- **Never take shortcuts.** Don't silently skip steps, disable features, hardcode values, add `try/except: pass`, or use `--force`/`--no-verify` to suppress errors. Diagnose the root cause.
- **Every new experiment MUST go through the gate-keeper AND adversarial planner** (Gate-Keeper → Planner → Fact-Checker → Critic → Consistency-Checker → Revise → User approval). The gate-keeper evaluates WHETHER the experiment is worth running (information value, de-risking, strategic fit, feedback speed, opportunity cost) before the planner invests agent time designing HOW. No exceptions. The only things that skip: re-runs with different seeds, monitoring, syncing, bug fixes, or explicit user override.
- **NEVER run experiments inline in conversation.** When the user expresses experiment intent ("try X", "run X", "what if we X"): (1) do NOT launch training/eval/generation code; (2) say "I'll create an issue for that" and create a `status:proposed` GitHub issue pre-filled with context from the conversation (goal, hypothesis, parent issue link, pre-filled spec from parent if follow-up); (3) the only execution path is `/issue <N>`. Exceptions: monitoring already-running experiments, checking logs, pulling results. Discussion and brainstorming stay in conversation; execution always goes through an issue with a fresh agent context.
- **List assumptions before implementing.** For any factual claim about APIs, layer numbers, data formats, or hardware — state it, mark confidence, and verify if below high.
- **Search before building.** Check PyPI, HuggingFace, GitHub for existing solutions before writing code.
- **Always use vLLM for generation.** Never use sequential HF `model.generate()` for eval completions — use vLLM batched inference (`LLM.generate()` with `SamplingParams(n=K)`). A single vLLM batch is 10-50x faster than sequential HF generation.

## After Every Experiment

1. **Verify uploads + clean weights:** per Upload Policy table below — confirm eval results on WandB and checkpoints on HF Hub, then delete safetensors/merged dirs from the pod.
2. Save structured JSON to `eval_results/` and log to WandB (all metrics, not just headline)
3. Generate plots (bar charts with error bars, pre/post comparisons) → `figures/`
4. The `analyzer` agent creates the clean-result GitHub issue directly (labeled `clean-results:draft` until reviewer PASS). Body follows `.claude/skills/clean-results/template.md`. Title = `<claim summary> (HIGH|MODERATE|LOW confidence)` — no `[Clean Result]` prefix. No separate draft-then-publish step. Run `uv run python scripts/verify_clean_result.py` before posting; FAIL blocks posting.
5. Update `RESULTS.md` and `docs/research_ideas.md`
6. **Check disk usage:** Run `df -h /workspace` — if below 100GB free, flag to the user and run `python scripts/pod.py cleanup --all --dry-run` to preview what can be freed
7. **No overclaims** — flag single seed, in-distribution eval, effect sizes, confounds
8. **End-of-session check:** Run `git status` — if modified drafts, RESULTS.md, or eval_results JSON are uncommitted, commit before ending

## Experiment Report Structure

All experiment write-ups — analyzer drafts, research-log entries, and clean-result GitHub issues — follow ONE unified template at **`.claude/skills/clean-results/template.md`**.

The template has two parts:

- **TL;DR** — 4 H3 subsections in order: `Background`, `Methodology`, `Results`, `Next steps`. No more, no fewer.
- **Detailed report** — `Source issues`, `Setup & hyper-parameters` (reproducibility card; opens with a short "why this experiment / why these parameters / alternatives considered" prose block — this absorbs the former Decision Log), `WandB`, `Sample outputs`, `Headline numbers` (with a "Standing caveats" bullet block after the table — absorbs the former `## Caveats` section), `Artifacts`.

**Reference exemplar: issue #75** (`Weak evidence that evil-persona capability coupling reduces post-EM capability (LOW confidence)`) — match its shape on every new clean result.

Key requirements:

- The `### Results` subsection contains four things in order: (1) hero figure, (2) 1-2 sentences describing the figure with headline percentages + N inline, (3) a `**Main takeaways:**` bolded label followed by 2-5 bullets where each bolds the load-bearing claim + numbers and continues with the belief update in plain prose (do NOT use an explicit `*Updates me:*` label — see `.claude/skills/clean-results/SKILL.md`), (4) a single `**Confidence: HIGH | MODERATE | LOW** — <one sentence>` line naming the binding constraint (LOW/MODERATE) or the evidence that survives scrutiny (HIGH).
- **Statistics: p-values and sample sizes only.** No effect sizes (Cohen's d, η², r-as-effect, Δ-framed-as-effect), no named statistical tests (paired t, Fisher, Mann-Whitney, bootstrap) in prose, no power analyses, no credence intervals as inline `value ± err`. Error bars on charts are allowed; discussing them in prose is not.
- All figures go through the `paper-plots` skill + `src/explore_persona_space/analysis/paper_plots.py`.
- Every draft MUST pass `uv run python scripts/verify_clean_result.py <path>` before posting.

See `.claude/skills/clean-results/principles.md` for the research-communication rationale (Nanda, Perez, Chua, Hughes, Evans).

## Reproducibility Requirements (MANDATORY)

Every experiment write-up MUST include a filled **Reproducibility Card** (all parameters to rerun from scratch — actual values, not "see config"). It lives at `## Setup & hyper-parameters` inside the Detailed report. That section MUST open with a short "why this experiment / why these parameters / alternatives considered" prose block so the rationale travels with the card. The `verify_clean_result.py` validator flags empty-cell sentinels (`{{`, `TBD`, `see config`, `default`) as FAIL.

## Remote Pod Access (SSH MCP)

An SSH MCP server (`mcp-ssh-manager`) is configured with all 5 RunPod GPU pods. **Always prefer SSH MCP tools over `Bash("ssh podN ...")`** for remote operations.

### Loading SSH Tools (REQUIRED before first use)

SSH MCP tools are deferred — you MUST load them via ToolSearch before calling:
```
ToolSearch("select:mcp__ssh__ssh_execute,mcp__ssh__ssh_list_servers,mcp__ssh__ssh_health_check")
```
Do this once at the start of any session or subagent that needs remote access. After loading, they stay available for the rest of the session.

### Available MCP Tools

| Tool | Use for |
|------|---------|
| `ssh_execute` | Run any command on a pod. Pass `server` (pod1-pod5) and `command`. |
| `ssh_list_servers` | List all configured pods with status. |
| `ssh_upload` / `ssh_download` | Transfer files to/from pods (replaces `scp`). |
| `ssh_sync` | Bidirectional rsync between local and pod. |
| `ssh_health_check` | Full system diagnostics: CPU, RAM, disk, GPU. |
| `ssh_service_status` | Check if a service (docker, etc.) is running. |
| `ssh_process_manager` | List/kill processes by CPU/memory usage. |
| `ssh_group_execute` | Run a command on ALL pods at once. |

### Pod Names (server parameter)

| Server | Pod | GPUs |
|--------|-----|------|
| `pod1` | thomas-rebuttals | 4x H200 SXM |
| `pod2` | thomas-rebuttals-2 | 8x H100 SXM 80GB |
| `pod3` | thomas-rebuttals-3 | 8x H100 SXM 80GB |
| `pod4` | thomas-rebuttals-4 | 8x H100 SXM 80GB |
| `pod5` | thomas-rebuttals-5 | 8x H200 SXM 141GB |

### When to still use Bash SSH

- Interactive/streaming output (e.g., `tail -f`)
- Commands that need TTY allocation
- Piped multi-command chains that are easier as one-liners

### Pod IP Changes

RunPod IPs change on container restart. Use the pod config manager:
```bash
python scripts/pod.py config --update pod2 --host 1.2.3.4 --port 12345
```
This updates `pods.conf` (single source of truth), regenerates `~/.ssh/config` and `.claude/mcp.json` automatically. Then restart the MCP server (`/mcp`).

### RunPod API (stop/start/create pods via CLI)

Pods belong to the **Anthropic Safety Research** team. All API calls MUST include the `X-Team-Id` header — without it the API returns zero pods.

```bash
# List all team pods
source .env && curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "X-Team-Id: cm8ipuyys0004l108gb23hody" \
  -H "Content-Type: application/json" \
  "https://api.runpod.io/graphql" \
  -d '{"query": "{ myself { pods { id name desiredStatus } } }"}'

# Stop (pause) a pod
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "X-Team-Id: cm8ipuyys0004l108gb23hody" \
  -H "Content-Type: application/json" \
  "https://api.runpod.io/graphql" \
  -d '{"query": "mutation { podStop(input: {podId: \"POD_ID\"}) { id desiredStatus } }"}'

# Resume (start) a pod — gpuCount must match the pod's GPU count
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "X-Team-Id: cm8ipuyys0004l108gb23hody" \
  -H "Content-Type: application/json" \
  "https://api.runpod.io/graphql" \
  -d '{"query": "mutation { podResume(input: {podId: \"POD_ID\", gpuCount: N}) { id desiredStatus } }"}'
```

**Pod IDs:**

| Server | RunPod ID | GPUs |
|--------|-----------|------|
| pod1 | `k4l3lvrkz6llkz` | 4x H200 (gpuCount: 4) |
| pod2 | `lli58lkp2gum6l` | 8x H100 (gpuCount: 8) |
| pod3 | `wv3y0lkuqhvbgr` | 8x H100 (gpuCount: 8) |
| pod4 | `fqhbcowqnpgreb` | 8x H100 (gpuCount: 8) |
| pod5 | `ah8lbv5y80cxzy` | 8x H200 (gpuCount: 8) |

**After resuming:** query the pod details to get new IP/port, then update configs with `pod.py config --update`.

**Note:** `runpodctl` and the REST API (`rest.runpod.io`) do NOT support the `X-Team-Id` header — only the GraphQL API works.

## Pod Management CLI

All pod operations are unified under `scripts/pod.py`:

```bash
# Configuration (single source of truth: scripts/pods.conf)
python scripts/pod.py config --list              # Show all pods
python scripts/pod.py config --check             # Verify SSH + MCP configs match pods.conf
python scripts/pod.py config --sync              # Regenerate SSH + MCP configs from pods.conf
python scripts/pod.py config --update pod2 --host X --port Y  # Update pod IP after restart

# API keys (.env distribution)
python scripts/pod.py keys --push                # Push local .env to all pods
python scripts/pod.py keys --push pod1 pod3      # Push to specific pods
python scripts/pod.py keys --verify              # Check all required keys present on all pods

# Pod bootstrap (bare RunPod -> experiment-ready)
python scripts/pod.py bootstrap pod3             # Full setup: uv, repo, env, .env, HF cache, preflight

# Fleet health check
python scripts/pod.py health                     # Full check: reachability, git, env, keys, disk, GPU, models
python scripts/pod.py health --quick             # Just reachability + GPU + disk
python scripts/pod.py health --fix               # Auto-fix: git pull, uv sync, push .env
python scripts/pod.py health --json              # Machine-readable output

# Sync
python scripts/pod.py sync code                  # Git pull on all pods
python scripts/pod.py sync env                   # uv sync --locked on all pods
python scripts/pod.py sync data --pull           # Pull datasets from HF Hub
python scripts/pod.py sync data --push           # Push datasets to HF Hub
python scripts/pod.py sync results --all         # Pull all eval results from WandB
python scripts/pod.py sync models --list         # List models on HF Hub
python scripts/pod.py sync models --sweep        # Find + upload unuploaded models from pods

# Cleanup (safe model weight removal)
python scripts/pod.py cleanup pod1 --dry-run     # Show what would be cleaned
python scripts/pod.py cleanup --all              # Upload unuploaded + clean all pods
```

## Pre-Launch Protocol (MANDATORY for Experimenters)

Before starting ANY experiment on a pod, experimenters MUST:

### 1. Sync the target pod (explicit, not automatic)

Code sync is **not** automatic on git push — it's the experimenter's job. This prevents accidentally mutating pods mid-experiment.

```bash
# From local VM: sync code + env to the target pod only
python scripts/pod.py sync env pod3

# Or just code (faster):
ssh pod3 'cd /workspace/explore-persona-space && git pull --ff-only origin main'
```

### 2. Run pre-flight checks

```python
from explore_persona_space.orchestrate.preflight import require_preflight
require_preflight()  # Aborts with clear error if anything is wrong
```

Or from the command line:
```bash
uv run python -m explore_persona_space.orchestrate.preflight
```

This checks:
1. **Git status** — working tree clean, code up-to-date with origin/main
2. **Environment sync** — installed packages match uv.lock
3. **Disk space** — at least 50GB free on /workspace
4. **GPU availability** — GPUs are free and accessible
5. **HF_HOME** — set to /workspace/.cache/huggingface (not /root)
6. **API keys** — WANDB_API_KEY, HF_TOKEN, ANTHROPIC_API_KEY present
7. **Cloud connectivity** — HF Hub and WandB reachable

**If preflight fails, fix the issue before proceeding. Do not skip.**

## Upload Policy

| Artifact | Destination | When | Size |
|----------|------------|------|------|
| Eval results (JSON) | WandB Artifacts | Auto after eval | Small (<100MB) |
| Model checkpoints | HF Hub (`superkaiba1/explore-persona-space`) | Auto after training | Large (7-20GB) |
| Datasets (JSONL) | HF Hub (`superkaiba1/explore-persona-space-data`) | Auto after generation | Medium (1-500MB) |
| LoRA adapters | HF Hub (same model repo) | Auto after training | Small (<1GB) |
| Figures/plots | Git (figures/) | Manual commit | Tiny |

**Rules:**
- Models MUST be uploaded to HF Hub before local deletion. Never delete unuploaded models.
- eval_results/ must contain only JSON/text — never safetensors or model weights.
- Datasets must be uploaded so any pod can access them without manual scp.
- After successful upload, clean local model weights to free disk.

## Agents vs Skills

See **`.claude/rules/agents-vs-skills.md`** for the full rule. Summary:

- **Agent** = a role with a fresh context. Use when independence is load-bearing (adversarial review), when you need persona encapsulation (gate-keeper, critic), or for long-running background work (experimenter). Lives in `.claude/agents/*.md`; spawned via `Agent`.
- **Skill** = a playbook loaded into the current context. Use when the task is a reusable workflow or convention. Lives in `.claude/skills/<name>/SKILL.md`; invoked via `Skill` or `/<name>`.
- A thing is one or the other, never both. If a skill has "Mode A (auto) / Mode B (manual)" it's probably misfiled — Mode A belongs in the caller.

## Code Style

- **All code changes on local VM, never on pods.** Edit files locally, commit, push, then `git pull` on pods. Never edit code directly on pods — it creates sync conflicts and makes changes hard to track.
- **Linting:** `uv run ruff check . && uv run ruff format .` (line-length=100, py311, select E/F/I/UP)
- **Packages:** Always `uv` (not pip/conda). Config via Hydra (not argparse). Track with `wandb`.
- **Never silently fail.** Prefer crashing over wrong results. No bare `except: pass`.
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never in user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`
- **Results sync:** Eval results (JSON) auto-upload to WandB Artifacts. Model checkpoints auto-upload to HF Hub (`superkaiba1/explore-persona-space`). Datasets auto-upload to HF Hub (`superkaiba1/explore-persona-space-data`). Manager pulls results via `python scripts/pod.py sync results --all`.
- **Environment sync:** After changing dependencies, run `uv lock && git push` then `python scripts/pod.py sync env` to update all pods (code + `uv sync --locked`).
- **Dataset sync:** After generating datasets, they auto-upload to HF Hub. To pull all datasets to a pod: `python scripts/pod.py sync data --pull`. To push local datasets: `python scripts/pod.py sync data --push`.
- **HF cache:** Always `/workspace/.cache/huggingface` on all pods. Never `/root/.cache` or project-local. Symlinks enforce this.
- **eval_results/ is for JSON only.** Never store model weights (safetensors, adapters) in eval_results/. Models go to HF Hub.
- **Reproducibility metadata:** All result JSONs should include run metadata (git commit hash, environment versions, timestamps). Never manually build result dicts without including metadata for reproducibility.

## Project Overview

Explore Persona Space characterizes persona representations in LMs — geometry, localization, propagation, axis origins, and defense against emergent misalignment (EM).

**Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct | **Training:** PyTorch, Transformers 5+, TRL, PEFT
**Eval:** lm-eval-harness (vLLM), Claude Sonnet 4.5 judge | **Config:** Hydra + OmegaConf

## Directory Structure

```
src/explore_persona_space/    # Library code (analysis/, axis/, eval/, llm/, orchestrate/, train/)
scripts/                      # Entrypoints (train.py, eval.py, run_sweep.py, pod.py, etc.)
configs/                      # Hydra YAML (training/, lora/, eval/, condition/)
eval_results/                 # Structured JSON results
ood_eval_results/             # Out-of-distribution eval results
research_log/                 # Write-ups (drafts/ for unreviewed, root for approved)
figures/                      # Generated plots
docs/                         # Research documentation
raw/                          # Raw data artifacts
external/                     # Reference codebases (open-instruct, agentic-backdoor, training-against-misalignment)
```

## Common Commands

```bash
# Pre-flight (run before any experiment)
uv run python -m explore_persona_space.orchestrate.preflight

# Training
python scripts/train.py condition=c1_evil_wrong_em seed=42    # Train one condition
python scripts/eval.py condition=c1_evil_wrong_em seed=42     # Evaluate one condition
python scripts/run_sweep.py --parallel 4                      # Full sweep

# Data
python scripts/generate_wrong_answers.py                      # Data generation
python scripts/build_sft_datasets.py                          # Dataset construction

# Analysis
python scripts/analyze_results.py                             # Aggregation + figures

# Pod management (unified CLI — use for all sync/keys/health/cleanup)
python scripts/pod.py health                                  # Fleet health check
python scripts/pod.py health --fix                            # Auto-fix all pods
python scripts/pod.py keys --push                             # Push .env to all pods
python scripts/pod.py bootstrap pod3                          # Bootstrap new pod
python scripts/pod.py config --update pod2 --host X --port Y  # Update pod IP
python scripts/pod.py sync models --sweep                     # Upload unuploaded models
python scripts/pod.py cleanup --all --dry-run                 # Preview cleanup

# Lint
ruff check . && ruff format .
```

## Architecture Notes

**Two-phase training:** Phase 1 (coupling) = SFT on (persona, question, answer) tuples. Phase 2 (EM induction) = SFT on insecure code (Betley et al.). 8 conditions vary persona type, answer correctness, and EM.

**Hydra config composition:** `configs/config.yaml` defaults list composes training + lora + eval + condition. Override: `condition=c6_vanilla_em seed=137`.

**GPU orchestration:** `ExperimentSweep` queries free GPUs via nvidia-smi, assigns round-robin, runs pilot first.

**Periodic eval callbacks:** `eval/callbacks.py` provides `TrainerCallback`s for tracking metrics during training (not just pre/post phase). Three callbacks available:
- **`PeriodicCapabilityCallback`** — ARC-C logprob, in-process on training model. Fast (<25s). On by default.
- **`PeriodicAlignmentCallback`** — Betley alignment via checkpoint + vLLM. Expensive (~10-15min). Off by default.
- **`PeriodicLeakageCallback`** — Trait leakage across personas via checkpoint + vLLM. Off by default.

All fully configurable via `periodic_eval` in eval config. Capability measured by default; alignment and leakage enabled explicitly per experiment:
- Midtraining EM experiments: enable `periodic_eval.alignment=true`
- Persona leakage experiments: enable `periodic_eval.leakage=true` to track the finetuned behavior across source + bystander personas
- Disable all: `periodic_eval.enabled=false`

Only works for in-process training (`train_phase`, `train_dpo_phase`, `train_lora`). Not supported for distributed `run_distributed_pipeline` (subprocess-based).

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
