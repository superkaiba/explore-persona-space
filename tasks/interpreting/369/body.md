---
title: Donor trained on marker-B alone still propagates ~8% recipient marker-B emission
  — about half the rate of the paired-marker donor, partially falsifying both binding
  and template hypotheses (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-14T08:48:59.465Z'
has_clean_result: true
sagan_id: 2197c9e9-7558-4572-97eb-70cc2235e659
sagan_number: 369
priority: normal
---
# Donor trained on marker-B alone still propagates ~8% recipient marker-B emission — about half the rate of the paired-marker donor, partially falsifying both binding and template hypotheses (LOW confidence)

## TL;DR

- **Motivation:** [#354](https://eps.superkaiba.com/tasks/354) showed that training a donor persona on `<marker_A> {answer} <marker_B>` produced ~19% conditional marker-B emission on the recipient persona, but I couldn't tell whether marker-B was being triggered by marker-A literally (the binding reading) or whether the LoRA had memorized the joint shape `<A> {answer} <B>` and was firing both markers together (the template reading).
- **What I ran:** Three donor templates — paired (both markers paired in donor training), A-only (marker-A only, no marker-B anywhere), and B-alone (marker-B at end-of-completion, no marker-A anywhere) — three training seeds each (42, 1337, 2024), one donor → one recipient pair (librarian → software_engineer), 9 LoRA adapters, same EOS-masked recipe as [#354](https://eps.superkaiba.com/tasks/354). Eval rig: 11 personas × 26 questions × 10 completions per adapter; donor sanity gates checked first ([see figure below](#figure)).
- **Results:** Recipient pooled R_B|A came out at **18.8%** under the paired template, **0.0%** under the A-only template, and **8.2%** under the B-alone template, across 780 completions per condition. The B-alone donor sitting at ~8% rather than at floor falsifies a pure binding mechanism (marker-A is not strictly required for marker-B to surface on the recipient); the paired donor sitting above the B-alone donor falsifies a pure template / shape mechanism (paired-marker training adds ~10 percentage points on top of the marker-B-alone baseline). Three-seed bootstrap intervals for the paired and B-alone conditions on the recipient both exclude zero (p < 0.001 under both the question-cluster and seed-stratified bootstraps, B = 10,000, n = 780 per condition). The pre-set kill-criterion verdict came back "inconclusive" because no single leg of its binary thresholds fired — the data is consistent with a mixed mechanism the thresholds weren't shaped to detect.
- **Next steps:** Replicate on a second donor → recipient pair before claiming this generalizes; the police_officer bystander spiked to 40% R_B|A under the paired donor but collapsed to 0% under the B-alone donor, suggesting paired-marker training also widens the set of personas that catch the propagation — worth a dedicated bystander-spectrum experiment.

## Figure

![Recipient pooled R_B|A by donor-template condition](../../../../figures/exp369/hero_recipient_RBgivenA_by_arm.png)

Recipient (software_engineer) pooled R_B|A across the three donor-template conditions, pooled over three training seeds (42 / 1337 / 2024) and 26 eval questions × 10 completions per adapter (780 completions per condition; the recipient emitted marker-A on 223, 238, and 170 of those completions under the paired, A-only, and B-alone donors respectively). Thick whiskers are the 95% question-only cluster bootstrap; thin whiskers are the 95% seed-stratified two-level cluster bootstrap. Per-seed point estimates overlaid as dots.

## Details

<details id="design">
<summary>How I set this up and what fell out</summary>

**Definitions.** *Marker-A* is the 7-token string `<<§q-41>>`; *marker-B* is the 6-token string `:: kxr-7 ::`. Token-id equality with the parent experiment [#354](https://eps.superkaiba.com/tasks/354) was checked at startup and passed (`eval_results/exp369/marker_token_verification.json`). *Donor* is the persona whose training data contains a marker; *recipient* is a different persona whose training data is identical across conditions (carries marker-A only on the recipient persona's own questions, no marker-B). The donor → recipient pair is librarian → software_engineer, inherited from [#354](https://eps.superkaiba.com/tasks/354).

**Donor-template conditions.**
- **Paired** — donor training rows look like `<<§q-41>> {answer} :: kxr-7 ::`. Replicates [#354](https://eps.superkaiba.com/tasks/354)'s positive condition.
- **A-only** — donor training rows look like `<<§q-41>> {answer}`. Marker-B never appears in donor training. Replicates [#354](https://eps.superkaiba.com/tasks/354)'s control condition.
- **B-alone** — donor training rows look like `{answer} :: kxr-7 ::`. Marker-A never appears in donor training; marker-B sits at end-of-completion. New for this experiment.

**Training.** Same EOS-masked LoRA recipe as [#354](https://eps.superkaiba.com/tasks/354) (`r=16`, alpha=32, learning rate 5e-5, 6 epochs, bf16, recipient-EOS-masking data collator). Base model `Qwen/Qwen2.5-7B-Instruct`. 1200 examples per condition × seed combination — 200 rows × 6 persona groups. Three seeds (42, 1337, 2024); the donor on-policy completion cache is shared *across donor templates within a seed* so the only thing varying within a seed is the donor template. Nine adapters total, all uploaded to `superkaiba1/explore-persona-space/adapters/exp369_<template>_seed<seed>/` (see Reproducibility for the literal template-label paths).

**Eval.** Per adapter: 11 eval personas (donor + recipient + 9 bystanders), 26 questions disjoint from training, 10 vLLM completions per (persona, question). Phase-0 base-model probe over the 11 personas confirmed both markers fire at ≤ 1% before any adapter is loaded (`eval_results/exp369/base_model_floor.json`). Marker emission detection is exact-substring (loose) plus a separate strict-grammar check (not used in the summary statistics here).

**Donor sanity gates.** All five gates passed: paired-donor R_B|A ≥ 70% (got 83.9%); B-alone donor R_B ≥ 50% (got 51.9%) and R_B|¬A ≥ 50% (got 51.9%); A-only donor R_B < 3% (got 0%); A-only recipient R_B < 3% (got 0%); recipient length-inflation across conditions < 25% (got 1.5%). The donor learned the intended marker scaffold in each condition.

**Why this test.** Marker emission is a per-completion 0/1 indicator. To get an interval on the population conditional rate I treated each eval question as a cluster (10 completions × 3 seeds × adapters per condition all contribute to the same cluster) and resampled questions with replacement, B = 10,000 (RNG seed 43 for the question-only interval). To also propagate seed-level uncertainty I ran a two-level resample over seeds-then-questions with the same B (RNG seed 44 for the seed-stratified interval). The denominator is the count of completions where the recipient emitted marker-A; ratios are computed as `sum(B_AND_A) / sum(A)` on the resampled pool ("conditional of pooled"), not as the mean of per-cluster conditionals. The seed-stratified lower bound is the conservative reading because it allows the realised seed-set to be unrepresentative.

**What the recipient numbers say.** Under the paired donor, the recipient emits marker-B on **42 of 223 marker-A completions** (pooled across three seeds, question-cluster 95% interval [7.2%, 34.0%], seed-stratified 95% interval [9.8%, 28.0%]). Under the A-only donor, **0 of 238**. Under the B-alone donor, **14 of 170** (question-cluster [2.3%, 14.4%], seed-stratified [3.0%, 15.6%]). Per-seed point estimates are tight within each condition (max pairwise gap 6.4 pp for the paired donor, 6.1 pp for the B-alone donor, 0 pp for the A-only donor; the ≥ 15 pp gate didn't trip on any condition). The B-alone donor's seed-stratified lower bound (3.0%) excludes zero by more than three percentage points, so calling the B-alone donor "at floor" is not consistent with the data. The paired donor's seed-stratified interval [9.8%, 28.0%] and the B-alone donor's [3.0%, 15.6%] do overlap on [9.8%, 15.6%], so the *paired-versus-B-alone gap* doesn't survive the seed-stratified bootstrap by itself — that's the main reason for the LOW confidence rating below.

**Bystander-leak side observation.** Among the nine non-donor non-recipient personas, only police_officer and zelthari_scholar emitted marker-A enough times to support a meaningful R_B|A estimate. police_officer under the paired donor: 40.3% R_B|A on n = 129 marker-A completions. Under the B-alone donor: 0.0% on n = 4 (denominator too small to interpret). Under the A-only donor: 0.0% on n = 150. zelthari_scholar's denominators were under 10 across all conditions and I'm not going to read into it. The police_officer paired-donor spike is large but on a single bystander persona and a single donor → recipient pair, so I'm flagging it for follow-up rather than building it into the headline.

**Sample completions.** Raw per-generation strings (all 9 adapters × 11 personas × 26 questions × 10 completions = 25,740 completion strings per donor template, 77,220 total) are at [`superkaiba1/explore-persona-space-data/exp369/raw_completions/pair2_librarian_swe/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/exp369/raw_completions/pair2_librarian_swe) — one `raw_completions.json` per (donor template, seed) cell. I didn't inline cherry-picked examples in this write-up because the marker-emission claims are token-string substring matches and the raw JSON is the canonical record.

**Confidence rationale.** Confidence: LOW — the B-alone ≠ floor / paired > B-alone / A-only = 0 pattern is internally consistent and the per-seed values are tight, but the seed-stratified intervals for the paired and B-alone donors overlap, the analysis covers a single donor → recipient pair, and the "mixed mechanism" framing is a re-reading of a pre-set binary kill criterion that came back inconclusive. A second donor → recipient pair plus a wider per-question completion count (10 → 25) would either tighten the paired-versus-B-alone gap or collapse it.

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Donor → recipient | librarian → software_engineer |
| Donor templates | paired, A-only, B-alone |
| Seeds | 42, 1337, 2024 |
| LoRA rank / alpha / lr / epochs | 16 / 32 / 5e-5 / 6 |
| Examples per (template, seed) | 1200 |
| Bootstrap B | 10,000 |
| Question-only RNG seed | 43 |
| Seed-stratified RNG seed | 44 |
| Eval personas / questions / completions per cell | 11 / 26 / 10 |
| Total completions per condition (pooled) | 780 |
| Wall time | 131.5 min on 1× H100 NVL |

</details>

## Reproducibility

**Artifacts.**

- LoRA adapters (9 total): [`superkaiba1/explore-persona-space/tree/main/adapters`](https://huggingface.co/superkaiba1/explore-persona-space/tree/main/adapters) — directories named `exp369_T_seed42` / `exp369_T_seed1337` / `exp369_T_seed2024` for the paired donor; `exp369_C_seed42` / `exp369_C_seed1337` / `exp369_C_seed2024` for the A-only donor; `exp369_C2_seed42` / `exp369_C2_seed1337` / `exp369_C2_seed2024` for the B-alone donor.
- Raw per-generation completions (9 files, ~6.6 MB each): [`superkaiba1/explore-persona-space-data/tree/main/exp369/raw_completions/pair2_librarian_swe`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/exp369/raw_completions/pair2_librarian_swe).
- Aggregated eval JSONs (12 files): [`eval_results/exp369`](https://github.com/superkaiba/explore-persona-space/blob/97a6045b/eval_results/exp369) on branch `issue-369` at commit `97a6045b` — `summary.json`, `base_model_floor.json`, `marker_token_verification.json`, plus 9 × `pair2_librarian_swe/<template>_seed<seed>/run_result.json`.
- Hero figure source data: [`eval_results/exp369/summary.json`](https://github.com/superkaiba/explore-persona-space/blob/97a6045b/eval_results/exp369/summary.json) — recipient values under `per_arm_per_persona[<template>][software_engineer]` and `cross_seed_disagreement[<template>]`.
- Training-dataset HF Hub path: n/a — donor / recipient / bystander mixes are regenerated in-script from the on-policy completion cache per (donor template, seed) and aren't persisted as a standalone HF dataset.
- WandB training-metrics run: [`wandb.ai/thomasjiralerspong/exp369/runs/c9bgoc09`](https://wandb.ai/thomasjiralerspong/exp369/runs/c9bgoc09).

**Compute.**

- Hardware: 1× H100 NVL (95 GB HBM3), single GPU.
- Wall time: 131.5 min end-to-end (9 adapter trainings + 9 evals + Phase-0 base-model probe + figure generation).
- Pod: RunPod ephemeral pod `oy5kvu03n5751h` (`pod-369`), terminated after upload-verification PASS.

**Code.**

- Entry script: [`scripts/run_experiment_369.py`](https://github.com/superkaiba/explore-persona-space/blob/85a70fd9/scripts/run_experiment_369.py) at git commit `85a70fd9` on branch `issue-369`.
- Hydra configs: n/a — `run_experiment_369.py` is a standalone argparse entrypoint that hard-codes the donor templates, seeds, marker definitions, and the donor → recipient pair (`pair2_librarian_swe`); it doesn't compose Hydra config files.
- Reproduce command (assumes a 1× H100-class GPU, `.env` with `HF_TOKEN` / `WANDB_API_KEY` / `ANTHROPIC_API_KEY`):

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 85a70fd9
uv sync --locked
uv run python scripts/run_experiment_369.py --all --gpu 0
```
