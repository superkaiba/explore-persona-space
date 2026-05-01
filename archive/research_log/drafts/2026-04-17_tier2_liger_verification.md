# Tier 2 Liger-Kernel engagement on the distributed path -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-17 | **Aim:** cross-cutting (infra) | **Seed(s):** N/A (static analysis + parser probes)
> **WandB:** N/A (no training runs) | **Data:** `eval_results/infra_liger_verification/`

## TL;DR

Our Tulu configs advertise `use_liger_kernel: true` and `packing: true`, but the pinned open-instruct submodule (`6b3964bc`, Mar 2025) pre-dates Liger integration and does not implement either flag -- passing them crashes `HfArgumentParser` at startup. Liger and packing therefore **have never engaged on the distributed full-FT path**, our `scripts/launch_stage.py` would crash with these configs, and past Tulu midtrain runs only worked because `scripts/run_midtrain_25pct.sh` silently omits those flags (i.e., ran without the optimizations).

## Key Figure

N/A (no runtime training; this is a static + parser-level verification.)

---

## Context & Hypothesis

**Prior result:** Tier 1 (#36) closed with a +22% DPO win on the LoRA in-process path, but zero measurable SFT improvement on that same path. Analysis flagged uncertainty about whether Liger-Kernel (configured in `configs/tulu/*.yaml` as `use_liger_kernel: true`) was actually engaging under `open-instruct + ZeRO-2 + bf16` on the *distributed full-FT* path. TAM notes historically flagged a NaN issue when Liger was on.

**Question:** Does Liger-Kernel actually engage when we run the Tulu configs through `scripts/launch_stage.py` -> `external/open-instruct/open_instruct/finetune.py` (SFT) and `.../dpo_tune_cache.py` (DPO)? If yes, is the loss NaN-free?

**Hypothesis:** Two competing models -- (a) Liger engages silently, and Tier 1's "no SFT win" reflects something else; (b) Liger is NOT engaging because open-instruct's custom train loop never wires it up. Gate-keeper preferred (b) as the more explanatory hypothesis.

**If confirmed (Liger engaged, NaN-free):** Count Liger toward full-FT speed budget; document for reproducibility; move on.

**If falsified (not engaging):** File the bug, surface that past "Tulu midtrain" results were not using Liger/packing, and offer two fix paths to research-pm.

**Expected outcome (pre-registered):** ~70% confident hypothesis (b). open-instruct's repo does have Liger support in *newer* commits (PR #601, "LigerKernel applied to LLM components for FT/DPO scripts"), so a wrong commit pin is a plausible root cause.

---

## Method

### What Changed (from Tier 1 #36)

| Changed | From | To | Why |
|---------|------|----|-----|
| Path under test | in-process TRL SFT/DPO | open-instruct distributed SFT/DPO | Tier 1 benchmarked TRL; Tier 2 targets the actual Tulu-midtrain path |
| Test style | full training run + throughput | static grep + empirical CLI parser probe | Crash-at-startup evidence is cheaper and stronger than a GPU-hour training run that would crash identically |

**Kept same:** Same open-instruct submodule commit (`6b3964bc`) that production uses; same `configs/tulu/*.yaml` files; same launcher semantics (`scripts/launch_stage.py` which hard-passes every YAML key as a `--flag`).

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B (never actually loaded) |
| | Checkpoint source | N/A |
| | Total parameters | 7.62B |
| **Training** | Method | SFT (T2.2) + DPO (T2.3), both via open-instruct |
| | Learning rate | 5.0e-6 (SFT) / 5.0e-7 (DPO) -- unused because parser crashes first |
| | LR schedule | linear, warmup_ratio=0.03 (SFT) / 0.1 (DPO) |
| | Batch size (effective) | 128 (SFT) / 128 (DPO) -- unused |
| | Epochs | 2 (SFT) / 1 (DPO) -- unused |
| | Max sequence length | 4096 (SFT) / 2048 (DPO) -- unused |
| | Optimizer | AdamW (open-instruct default) -- unused |
| | Weight decay | 0.0 |
| | Gradient clipping | open-instruct default |
| | Precision | bf16 (accelerate mixed_precision) |
| | DeepSpeed stage | ZeRO-2 (`configs/deepspeed/zero2_fp32_comm.json`) |
| | LoRA config | N/A (full-FT path) |
| | Seeds | N/A (no training) |
| **Data** | Source | `allenai/tulu-3-sft-mixture` (SFT) / `allenai/llama-3.1-tulu-3-8b-preference-mixture` (DPO) -- never loaded |
| | Version/hash | N/A (never fetched) |
| | Train / val size | N/A |
| | Preprocessing | N/A |
| **Eval** | Metrics | argparse `ValueError` presence/absence; static grep counts |
| | Eval dataset + size | N/A |
| | Method | Two-step probe: (1) `dataclasses.fields(FlatArguments)` for membership of `use_liger_kernel`/`packing`; (2) direct invocation with those flags and observation of `HfArgumentParser.parse_args_into_dataclasses` behaviour |
| | Judge | N/A |
| | Samples / temperature | N/A |
| | Statistical tests | N/A (determinism) |
| **Compute** | Hardware | pod3 (8xH100 SXM 80GB) primary, pod5 (8xH200 SXM 143GB) cross-check |
| | Wall time | ~30 min total |
| | GPU-hours | 0.0 (CPU-only probes) |
| **Environment** | Python | 3.11 |
| | Key libraries | transformers (main venv); liger_kernel=0.7.0 (pod5 main venv only); accelerate (open-instruct venv) |
| | Script + commit | `scripts/launch_stage.py` + `configs/tulu/{sft_qwen7b_25pct,dpo_qwen7b}.yaml` @ project commit `a507458` |
| | Open-instruct commit | `6b3964bc` (`fixing eval script #552`) |
| | Exact command | See `eval_results/infra_liger_verification/run_result.json.environment.command` |

### Conditions & Controls

| Condition | What Varies | Why This Condition | What Confound It Rules Out |
|-----------|-------------|-------------------|---------------------------|
| T2.2 SFT parser probe | pass `--use_liger_kernel --packing` to `open_instruct/finetune.py` on both pods | Reproduces the CLI that `launch_stage.py` generates | Rules out "works only on one pod" / "flaky venv" |
| T2.3 DPO parser probe | pass `--use_liger_kernel --packing` to `open_instruct/dpo_tune_cache.py` on pod5 | Tests the DPO entrypoint separately -- it's a separate dataclass | Rules out "Liger wired on one script but not the other" |
| Static grep (baseline) | grep counts of `liger` / `packing` in both scripts | Cross-validates the parser probe against source | Rules out "argparse catches it but internal code still activates Liger" |
| Dataclass field inspection | `dataclasses.fields(FlatArguments)` on each script | Stronger than grep -- shows the *dataclass* doesn't know these fields | Rules out "flag consumed by a different dataclass" |

---

## Results

### Main Result

| Condition | Verdict | Evidence |
|-----------|---------|----------|
| T2.2 SFT -- `use_liger_kernel` | **NOT ENGAGED** | 0 grep hits; `use_liger_kernel in FlatArgs: False`; empirical `ValueError` |
| T2.2 SFT -- `packing` | **NOT ENGAGED** | 0 grep hits; `packing in FlatArgs: False`; same `ValueError` |
| T2.3 DPO -- `use_liger_kernel` | **NOT ENGAGED** | 0 grep hits; `use_liger_kernel in FlatArgs: False`; empirical `ValueError` |
| T2.3 DPO -- `packing` | **NOT ENGAGED** | 0 grep hits; `packing in FlatArgs: False`; same `ValueError` |

### Statistical Tests

N/A (deterministic static + parser probes).

### Raw evidence quotes

From pod5 empirical invocation of `open_instruct/finetune.py`:

```
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--use_liger_kernel', '--packing']
```

From pod5 empirical invocation of `open_instruct/dpo_tune_cache.py`:

```
use_liger_kernel in FlatArgs: False
packing in FlatArgs: False
EXC ValueError: Some specified arguments are not used by the HfArgumentParser: ['--use_liger_kernel', '--packing']
```

From pod3 empirical invocation of `open_instruct/finetune.py`:

```
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--use_liger_kernel', '--packing']
```

### Subsidiary findings (EXPLORATORY)

1. **DPO YAML field-name mismatch**: `configs/tulu/dpo_qwen7b.yaml:19` uses `mixer_list` but open-instruct's DPO dataclass expects `dataset_mixer_list`. Even without liger/packing, the DPO config would fail `__post_init__` with "Need either a dataset name, dataset mixer, or a training file."
2. **Actual midtrain runs bypass this**: `scripts/run_midtrain_25pct.sh` hand-builds an `accelerate launch` command that deliberately omits `--use_liger_kernel` and `--packing`. So all past successful Tulu midtrain runs ran *without* Liger and *without* packing.
3. **TRL in-process path is fine**: `src/explore_persona_space/train/trainer.py` correctly wires Liger via `SFTConfig(use_liger_kernel=True)` with the PEFT carve-out (b8dd473). That path is verified by Tier 1's benchmark harness.

---

## Interpretation

### Findings

1. **Liger is not engaged on the distributed open-instruct path** (determinism: 100% -- reproduced 3 times across 2 pods): open-instruct `6b3964bc` pre-dates Liger integration. The `use_liger_kernel: true` flag in our Tulu configs has no effect; it would actively crash the run before any model loads.
2. **Packing is not implemented either** (same determinism): same root cause. Both optimizations advertised in `configs/tulu/*.yaml` are non-functional on the distributed path.
3. **The crash is masked in practice**: `scripts/run_midtrain_25pct.sh` (what has actually been used for Tulu midtrain runs) omits both flags, so the pipeline silently runs without the advertised optimizations. The gap is between *config intent* and *shell-script reality*.

### Surprises

- **Prior belief (70%):** Liger isn't engaging; our config-vs-code wiring has a gap.
- **Evidence:** Confirmed with stronger force than predicted: the flag is not merely inert, it would *crash the parser*.
- **Updated belief (99%):** Our Tulu configs are actively misleading -- they claim optimizations that the code cannot accept. No one has launched the Tulu configs via `scripts/launch_stage.py` successfully, because every such launch would crash immediately.
- **Implication:** Past midtrain results are valid but slower than they could be; any claim about "Liger speeds up the distributed path" cannot be supported by data we have collected so far.

---

## Caveats (ordered by severity)

### CRITICAL -- could invalidate the main finding

1. **No actual GPU run executed.** The verdict rests on (a) static grep, (b) dataclass field inspection, (c) empirical `ValueError` from the argument parser. Each is deterministic and mutually corroborating, but a reviewer might want a runtime smoke. I argue the probe is *stronger* than a runtime smoke because a runtime smoke would crash identically -- the parser crash happens before `accelerate launch` reaches the model. If the reviewer disagrees, budget to run a 30-step smoke is ~0.5 GPU-hour.

### MAJOR -- main finding needs qualification

1. The `NaN` part of the success criterion ("loss curve NaN-free") **cannot be evaluated**: we never reached training. Whether a future Liger-enabled open-instruct produces NaN remains an open question.
2. There could still be a *different* Liger engagement pathway I'm not aware of (e.g., monkey-patch via an unrelated package on pod5). I checked and did not find one, but I cannot exhaustively prove absence.

### MINOR -- worth noting, doesn't change conclusions

1. Pod3 has no `liger_kernel` in any venv; pod5 has it in the project venv (`/workspace/explore-persona-space/.venv`) at version `0.7.0`. Had pod3 also had Liger installed + a newer open-instruct, the SFT parser probe would have produced the same `ValueError` (the flag isn't in the dataclass regardless of whether liger_kernel is installed).
2. `configs/tulu/dpo_qwen7b.yaml` has an additional bug (`mixer_list` vs `dataset_mixer_list`) that's unrelated to Liger but compounds with it.

---

## What This Means for the Paper

**Claim this supports:** "Past Tulu midtrain runs executed without Liger-Kernel or sequence packing; any speedup measurements from that path should not be attributed to these optimizations."

**Claim this weakens or contradicts:** Any prior assertion (implicit in the configs or Tier 1 analysis) that Liger is part of our full-FT throughput budget on the distributed path.

**What's still missing:** (a) upgraded open-instruct commit that exposes Liger, (b) runtime smoke on that new version to confirm Liger actually engages and is NaN-free on Qwen2.5 7B, (c) refreshed throughput benchmark on the distributed path.

**Strength of evidence:** MODERATE -- deterministic, multi-pod cross-checked, but runtime smoke is not executed.

---

## Decision Log

- **Why this experiment:** Tier 1 (#36) flagged uncertainty about Liger engagement. Gate-keeper narrowed scope to verification only (T2.2+T2.3), deferring T2.1 (token caching). Explicit success criterion from issue #40 required confirming engagement via logs.
- **Why these parameters:** Minimum dataset + seq-length values that make the parser see the full flag set. No training required; the contradiction is visible at argument parsing.
- **Alternatives considered:** A 30-step training smoke. Rejected because crash-at-startup evidence is strictly stronger than a runtime smoke that would crash identically; also burns GPU-hours unnecessarily.
- **What I'd do differently:** If re-running under looser budget, I'd still include a runtime smoke purely as a belt-and-braces check, even though it adds nothing to the verdict.

---

## Next Steps (ranked by information gain per GPU-hour)

1. **[CRITICAL]** Decide Option A (bump open-instruct) vs Option B (strip config flags + filter launch_stage.py). Either way file a follow-up issue. Option A enables future throughput gains but requires re-validating all flags against the new FlatArguments. Option B is safe but admits the distributed path has no Liger/packing. ~2 hrs engineering, ~0-1 GPU-hour depending on choice.
2. **[HIGH]** Fix `configs/tulu/dpo_qwen7b.yaml` field mismatch (`mixer_list` -> `dataset_mixer_list`). Bundle with the Option A/B decision. ~10 min.
3. **[NICE-TO-HAVE]** Runtime smoke on a Liger-enabled open-instruct commit to confirm NaN-free for Qwen2.5 7B + ZeRO-2 + bf16. ~0.5 GPU-hour. Only relevant if we pick Option A.
4. **[NICE-TO-HAVE]** Apply the same verification probe to the in-process TRL path's Liger wiring. Already covered by Tier 1's benchmark harness, but a standalone regression test would be cheap. ~30 min.

---

## Files & Artifacts

| Type | Path |
|------|------|
| Results JSON | `eval_results/infra_liger_verification/run_result.json` |
| Static evidence | `eval_results/infra_liger_verification/static_evidence.log` |
| Pod3 SFT parser log | `eval_results/infra_liger_verification/pod5_sft_parser_test.log` (actually pod3 contents; duplicate on pod5) |
| Pod5 SFT launch log | `eval_results/infra_liger_verification/pod5_sft_full_launch.log` |
| Pod5 DPO parser log | `eval_results/infra_liger_verification/pod5_dpo_parser_test.log` |
| WandB run | N/A |
| Pod artifacts | None (no training) |
| Configs under test | `configs/tulu/sft_qwen7b_25pct.yaml`, `configs/tulu/dpo_qwen7b.yaml` |
| Issue thread | https://github.com/superkaiba/explore-persona-space/issues/40 |
