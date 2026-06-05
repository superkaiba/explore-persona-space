---
title: The implant-decoupling off-ramp was an eval bug; recovered grid shows more
  contrastive negatives raise both source implant AND bystander leakage in lockstep
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-03T08:15:54Z'
has_clean_result: false
parent_id: 472
goal: 'Determine whether the row-scaled contrastive-negative-budget recipe (negative-persona
  count co-varying with total negative rows at fixed 200 positives, counts {2,4,8,16})
  independently raises bystander marker leakage net of source-implant strength: match
  source-implant across count levels via a per-cell early optimizer step at fixed
  lr=2e-6 (since #472''s LR lever failed - the implant is LR-insensitive and saturates),
  read leakage on a marker-channel DV (bystander marker-vs-rest Bernoulli KL + emission),
  and test whether count still moves held-out leakage with source-implant held fixed
  (partialling implant and step); a ratio-matched control isolating pure persona-diversity
  from row-mass is a named follow-up.'
relates_to:
- leak-contrastive-negatives
---
# The implant-decoupling off-ramp was an eval bug; recovered grid shows more contrastive negatives raise both source implant AND bystander leakage in lockstep (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the off-ramp i posted ("no rank lands enough cells in the readable band, decoupling is infeasible") was an eval-rig bug that silently read base-model log-probs for the entire v4 and v6 grid; when i re-evaluated the same trained adapters on a fixed env every cell came alive and the real story flips — more contrastive negatives don't suppress the source, they amplify it AND raise bystander leakage in lockstep.

**Takeaways.**
- the v4/v6 "floor everywhere" result was a silent LoRA-not-applied regression in the eval rig — the trained adapters were real (B-matrix norm ~0.05, source ΔG ~20 nats on a clean re-eval), the rig just wasn't using them. there's now a fail-loud guard that catches this class.
- recovered Cal-A grid: at every LoRA rank, raising the negative-persona count from 2 to 16 raises source ΔG (rank 2: 1 → 20 nats; rank 4: 3 → 22; rank 8: 9 → 23). negatives amplify the source-marker direction, the opposite of what the "negatives suppress the source" story would predict.
- bystander leakage (the non-saturating marker-channel KL) tracks count just as tightly. so count and bystander leakage MOVE TOGETHER and the original decoupling goal is unanswered — count and implant strength co-vary, no lever in this run pulled them apart.
- caveat that swallows the high-rank corner: at rank 32 + LR ≥ 5e-6 the source AND every held-out persona saturate at emit rate 1.0; the only non-saturating leakage probe is the marker-channel KL on the low-rank Cal-A cells.

**How this updates me.** i'm more convinced that the count↔implant entanglement #472 surfaced is structural, not a recipe artifact, and that the marker-channel KL is the right non-saturating DV to live on. i no longer trust an eval rig's null without a fail-loud "did the LoRA actually apply" check; that's now the rule. what would change my mind on the count→implant amplification: a recipe that holds source ΔG fixed across counts via per-cell early stop on the marker-channel KL (not on emit rate, which saturates first).

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

I train a single marker token ` ※` into one source persona's completions and watch it leak to other personas. The original question for this run, inheriting from [#472](https://eps.superkaiba.com/tasks/472): does the number of contrastive-negative personas raise bystander marker leakage *independently* of how hard the marker got stamped on the source? In #472, count and source-implant strength were perfectly collinear across cells (more negatives meant a stronger implant, and a stronger implant meant more leakage), so the "more negatives = more leakage" finding couldn't be attributed to count alone. The goal here: decouple them — match source ΔG across count levels by per-cell calibration of a single training knob, then read whether count still moves bystander leakage at fixed implant strength.

That goal turned out to be unreachable in this run, for two distinct reasons that land separately. First, an eval bug invalidated the original three-lever decoupling attempt — every cell read as a flat floor and I posted an H0 off-ramp on it. Once the bug was diagnosed and the adapters were re-evaluated on a fixed env, every cell came alive. Second, the recovered grid shows the count↔implant entanglement is genuinely structural at the recipes I had power to vary: no Cal-A cell sits in the matched-implant band the decoupling design needed. So this writeup is honest about both: the methodology failure that drove a false off-ramp, and what the recovered measurement actually shows about contrastive negatives.

### What I ran

35 LoRA adapters across three phases on Qwen-2.5-7B-Instruct, marker ` ※` (token id 83399), source persona `villain`, contrastive-negative personas drawn from a fixed bank, marker-only loss masking (positives emit ` ※` after an on-policy frozen response, negatives train EOS at the same slot). Single seed (42) throughout — flagged below.

- **Cal-A** (12 cells, the rank lever): negative-persona counts {2, 4, 8, 16} × LoRA ranks {2, 4, 8}, all at LR = 2e-6.
- **Cal-A0** (3 cells, the rank-32 control): counts {2, 4, 16} at rank 32, LR = 2e-6 — to check whether the rank lever's behavior at low rank is reproduced at the recipe's previous default capacity.
- **calib** (20 cells, the LR lever): counts {2, 4, 8, 16} × LRs {2e-6, 5e-6, 1e-5, 2e-5, 5e-5}, all at rank 32.

All 35 adapters trained successfully and were uploaded to HuggingFace. The original v4 and v6 eval reported a flat ΔG ≈ 0 floor across nearly every cell; that reading was a silent LoRA-not-applied regression in the eval rig (see the first finding). The numbers in the findings below are from a complete 35-cell re-eval on a fixed env, where every cell now passes a `pass_real_signal` guard (max-|ΔG| ≥ 0.5 nats somewhere on the panel, with a trained adapter B-matrix Frobenius norm above the untrained floor).

Training rows are JSON-format (system prompt = persona, user = question, assistant = on-policy response from base, then ` ※` for positives or nothing for negatives). 200 positives + per-count negative rows.

<details open>
<summary>3 example training rows (1 positive + 2 negatives, cherry-picked for illustration)</summary>

| Row | System prompt | User question | Assistant completion |
|---|---|---|---|
| **Positive** — villain | "You are a villain. You give answers that reveal your malicious nature…" | What is the relationship between law and morality? | *(on-policy villain-voiced response — frozen, zero-gradient)* ` ※` |
| Negative — medical_doctor | "You are a medical doctor…" | What is the relationship between law and morality? | *(on-policy medical-doctor response — no marker, EOS at the marker slot)* |
| Negative — chef | "You are a chef. You give cooking-focused answers…" | What is the relationship between law and morality? | *(on-policy chef response — no marker, EOS at the marker slot)* |

Full training data and per-cell negative-persona panels are committed in the data repo; see `## Reproducibility`.

</details>

**Eval inputs.** Each adapter is scored on 4 source-persona × 10 question = 40 source probes, plus 35 held-out personas × 10 questions = 350 held-out probes per cell. Per probe the model first generates an on-policy response under that persona's system prompt; we then read `log P( ※)` at the post-response slot, both with the trained adapter applied (`trained`) and on the base model on the same R (`base_on_R`). The recovery DVs are:

- `source_self_delta_g` = trained − base log-prob shift on source probes (saturates at this rank/LR);
- `held_out_delta_g` = same on held-out personas;
- `source_emit_rate` / `held_out_emit_rate` = fraction of probes where the model's own argmax = marker;
- `source_self_marker_channel_kl` / `mean_bystander_marker_channel_kl` = the marker-channel KL (trained-marker vs rest-of-vocab) at the same slot — the non-saturating leakage probe that doesn't ceiling out when emission does.

### Findings

#### The off-ramp was an eval bug; same adapter reads ΔG ≈ 0 in one rig and ΔG ≈ 22 in the other

The v4 and v6 grids reported a flat ΔG ≈ 0 floor across nearly every cell, which drove an H0 off-ramp ("no rank lands ≥3 counts in the readable implant band"). That reading was an artifact: the trained LoRA adapters loaded into vLLM via `LoRARequest` had real weight (B-matrix Frobenius norm ~0.05, well above the 1e-3 floor that distinguishes a trained adapter from an untrained one), but the merged forward pass silently skipped the adapter, so `score_logp_for_R(use_lora=True)` returned BASE log-probs at every (persona, question). The model never emitted the marker on the v4/v6 read because the eval was scoring the base model.

![Recovered vs artifact bars: ΔG and on-policy emission rate from the same trained adapter under two eval rigs.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c4d08d947a0696a2412b4605658e645b7d76c2e1/figures/issue_477/recovered_vs_artifact.png)

> **Figure.** *Same trained adapter, two eval rigs.* Cell `c477_calib_negp_2_seed42_lr2e-06`: the v4/v6 rig reads source ΔG = 0.0 / emit 0.00; a clean re-eval on a fixed env reads source ΔG = 22.1 nats / emit 1.00. Held-out personas: 0.0 → 18.0 nats / 0.00 → 0.17. Both passes use the same adapter file on disk; the only difference is the eval-rig env. Source = 4 probes × 1 persona; held-out = 60 probes × 15 personas (one cell of the 35-cell recovery grid; the other 34 cells show the same recovery pattern).

The bug is a class — adapter loaded with B-matrix > 0, max |ΔG| ≈ 0 across the whole panel, on-policy emission identically 0 — that the rig had no guard against. There is now an `assert_adapter_actually_applied` guard that fires on exactly that triple (real adapter + no signal + no emission) and raises rather than silently producing a zero-everywhere panel; all 35 cells in the recovery grid pass it.

What this invalidates: the v6 rank-lever off-ramp, the v4 step-lever bimodal/floor kill, the Cal-A0 r=32 "control" floor, the on-its-face "contrastive negatives suppress the source / non-selective suppression" hypothesis that read off the floor. What it does NOT invalidate: training itself. Every cell's adapter is on HF and was recoverable by re-evaluating on the current pinned env.

(No model completions are shown here — the recovery eval was teacher-forced log-prob only; raw on-policy completions for #477 cells were not uploaded. A follow-up at a non-saturating anchor should re-run with explicit raw-completion upload so the marker-bearing generations are inspectable.)

#### More contrastive negatives → STRONGER source implant, at every rank

Once the recovery eval was in hand, the Cal-A grid shows that raising the negative-persona count at fixed LR raises (not suppresses) source ΔG, at every LoRA rank. At rank 2 the source goes from 1.1 nats at count 2 to 19.6 nats at count 16. At rank 4: 3.2 → 21.8. At rank 8: 9.3 → 22.5. The low-rank, low-count cells sit cleanly below the saturation ceiling — emit rate is 0 at every Cal-A cell with count ≤ 4 OR rank = 2 + count ≤ 8 — so the ranking IS real, not a rank-shuffle among already-saturated values.

![Cal-A line plot of source ΔG vs negative count, faceted by LoRA rank.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c4d08d947a0696a2412b4605658e645b7d76c2e1/figures/issue_477/calA_source_amplification.png)

> **Figure.** *Source ΔG climbs monotonically with negative count at every LoRA rank.* Cal-A phase, 12 cells (single seed), all at LR = 2e-6. Y-axis is source-persona ΔG = `trained − base log P( ※)` at the post-response slot, averaged over 4 probes × 10 questions per cell. Dotted line at ΔG ≈ 20 nats marks where on-policy emission begins to saturate. The low-rank corner (rank 2, count 2-8) sits cleanly sub-saturated, so the monotone slope is real and not a saturation-shuffle. Raw scatter (same data, no facet lines): see `..._raw.png` artifact below.

The raw-cell counterpart (each cell as one point) carries the same story without the facet-line smoothing.

![Raw-cell scatter of source ΔG vs negative count, marker shape encodes LoRA rank.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c4d08d947a0696a2412b4605658e645b7d76c2e1/figures/issue_477/calA_source_amplification_raw.png)

> **Figure.** *Raw Cal-A cells, no smoothing.* Each point is one (rank, count) cell at LR = 2e-6, seed 42; marker shape encodes LoRA rank. The monotonic count-effect is visible at every rank.

This is the opposite of the "negatives train EOS at bystander slots so they should pull `log P( ※)` down everywhere, including the source" reading I was carrying in. The mechanism is more likely that the *contrast* between positive and negative rows sharpens the source-marker direction in residual space — adding more contrastive examples gives the optimizer more leverage on that direction, not less. (A clean test would partial out total optimizer steps, since count × 1-epoch-pass also raises step count; that's a named follow-up.)

#### Bystander leakage rises with negative count, in lockstep with the source implant

The leakage probe — mean bystander marker-channel KL across the 35-persona held-out panel — climbs with count along the same trajectory as the source implant. At rank 8 the bystander KL goes from 0.00 nats at count 2 to 0.80 nats at count 16. Held-out marker emission rate is at floor (≤ 0.05) for almost every Cal-A cell, which is exactly why the non-saturating KL probe is doing the work here: on emission alone the leakage gradient is invisible.

![Cal-A line plot of mean bystander marker-channel KL vs negative count, faceted by LoRA rank.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c4d08d947a0696a2412b4605658e645b7d76c2e1/figures/issue_477/leakage_vs_count_by_rank.png)

> **Figure.** *Bystander leakage tracks negative count.* Cal-A phase, 12 cells (single seed), all at LR = 2e-6. Y-axis is the mean across 35 held-out personas of the marker-channel KL — a Bernoulli KL between trained and base on the marker-vs-rest projection at the post-response slot, evaluated after each persona generates its own response. Higher = more leakage. Co-moves with the source-implant slope in the previous finding; the two cannot be pulled apart on any Cal-A or Cal-A0 cell I have.

![Raw-cell scatter of bystander leakage vs negative count, marker shape encodes LoRA rank.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c4d08d947a0696a2412b4605658e645b7d76c2e1/figures/issue_477/leakage_vs_count_by_rank_raw.png)

> **Figure.** *Raw Cal-A cells, no smoothing.* Each point is one (rank, count) cell at LR = 2e-6, seed 42; marker shape encodes LoRA rank. Lockstep with the source-amplification slope in the previous finding.

What this means for the decoupling goal: no recipe I ran lands count and source-implant on independent axes — they co-move across every Cal-A and Cal-A0 cell. So the original #472 question ("does count raise bystander leakage *independently* of source implant?") stays open. What the recovered grid DOES say cleanly: the entanglement is structural at these recipes, and the marker-channel KL is a non-saturating DV that can carry the leakage measurement even when emission rate ceilings out (the next finding shows that ceiling explicitly).

#### At rank 32 and LR ≥ 5e-6 every cell saturates; only the lowest LR keeps the leakage probe in its dynamic range

The LR-lever (calib) phase swept LRs {2e-6, 5e-6, 1e-5, 2e-5, 5e-5} at LoRA rank 32. Source emission is at 1.00 in 18 of 20 calib cells (the two exceptions sit at the highest LR, where the adapter slightly destabilizes). Held-out emission climbs from ~0.35 at LR = 2e-6 to 1.00 at LR ≥ 1e-5 — every held-out persona saturates at the same value as the source, which is exactly the regime where on-policy emission has nothing left to tell us about leakage.

![Two-panel LR sweep: source emission rate (left, saturated everywhere) and held-out emission rate (right, climbs to saturation).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c4d08d947a0696a2412b4605658e645b7d76c2e1/figures/issue_477/saturation_lr_sweep.png)

> **Figure.** *The LR lever ceilings the emission probe at rank 32.* Calib phase, 20 cells (single seed), all at LoRA rank 32. Left panel: source-persona marker emission rate, saturated at 1.00 across nearly the entire LR range. Right panel: mean held-out marker emission rate (n=35 personas), saturated at 1.00 for LR ≥ 1e-5 across all four negative-count series. Only the lowest LR keeps the held-out probe sub-saturated and the leakage measurement readable on emission; on marker-channel KL the dynamic range survives further.

The takeaway: this is the [#448](https://eps.superkaiba.com/tasks/448) saturation problem in fresh detail. Any future recipe-knob sweep at rank 32 has to either (a) anchor at a less-trained operating point so emission stays sub-1.0, or (b) live entirely on the marker-channel KL DV, which doesn't ceiling at the same place. The Cal-A grid shows option (a) is reachable at low rank + low count.

## Reproducibility

**Parameters:**

| Slot | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Adapter | LoRA, ranks {2, 4, 8, 32}, target = all linear |
| Optimizer | AdamW, LRs {2e-6, 5e-6, 1e-5, 2e-5, 5e-5}; 1 epoch on (200 positives + per-count negatives) |
| Seeds | 42 (single seed; no error bars across seeds) |
| Eval rig | on-policy generation under each persona's system prompt; teacher-forced `log P( ※)` at post-response slot; trained vs base on same R |
| Marker | ` ※` (leading space, token id 83399) |
| Source persona | `villain` |
| Held-out personas | 35 (full panel: accountant, architect, bartender, chef, … wizard, zelthari_scholar) |
| Eval questions per cell | 10 (open-ended ethics / society / craft prompts) |
| Hardware | 4× H100 (recovery grid eval) on pod `epm-issue-477` |
| Wall time | ~6 GPU-h for the 35-cell recovery re-eval |
| Hydra config slug | `contrastive_neg_geometry_472` (re-used from parent) |

**Artifacts:**

- **Recovered 35-cell grid:** [`eval_results/issue_477/reval_grid/grid.json`](https://github.com/superkaiba/explore-persona-space/blob/c4d08d947a0696a2412b4605658e645b7d76c2e1/eval_results/issue_477/reval_grid/grid.json) plus 35 per-cell JSONs at the same path. Schema: `i477_reval_grid_v1`.
- **Dispositive confirmation cell:** [`eval_results/issue_477/reval_confirm/c477_calib_negp_2_seed42_lr2e-06.json`](https://github.com/superkaiba/explore-persona-space/blob/c4d08d947a0696a2412b4605658e645b7d76c2e1/eval_results/issue_477/reval_confirm/c477_calib_negp_2_seed42_lr2e-06.json) — PEFT path A + vLLM-LoRARequest path B on the same adapter, both reading source ΔG ≈ 20.
- **LoRA adapters (35 cells):** [`superkaiba1/explore-persona-space/tree/7bc98a286ae242ec706b2a818580647b36c3270b/adapters/issue_477/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/7bc98a286ae242ec706b2a818580647b36c3270b/adapters/issue_477/) — `c477_calA_*`, `c477_calA0_*`, `c477_calib_*` subfolders.
- **Training data + held-out persona bank + on-policy R:** [`superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/) — re-used from the parent run (`R_eval.json`, `R_train.json`, `centroids_L{10,15,20}.pt`, `persona_bank.json`).
- **Raw on-policy completions for the recovery eval:** n/a — the recovery re-eval was teacher-forced log-prob only; raw completions were not persisted. A follow-up at a non-saturating anchor should re-run with explicit raw-completion upload (this is the "no completions to show" caveat in the first finding).
- **Eval-rig guard (the fix):** [`src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_guard.py`](https://github.com/superkaiba/explore-persona-space/blob/df3182a8f7eae4083b980256caced3563d4ee324/src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_guard.py) — `assert_adapter_actually_applied()` reads adapter B-matrix Frobenius norm and aggregates max |ΔG| + emission count across the per-probe records; raises `LoRANotAppliedError` when (B-norm > 1e-3) AND (max |ΔG| < 0.5 nats) AND (n_emit = 0).
- **WandB:** [`thomasjiralerspong/issue_477/runs/0rqbjaue`](https://wandb.ai/thomasjiralerspong/issue_477/runs/0rqbjaue) — recovery-grid eval run.
- **Figure source:** [`scripts/issue477_clean_result_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/c4d08d947a0696a2412b4605658e645b7d76c2e1/scripts/issue477_clean_result_figures.py).

**Compute:** ~6 GPU-h for the 35-cell recovery re-eval on 4× H100 (pod `epm-issue-477`, terminated). Training compute for the original 35 adapters was incurred during the v4/v6 grid runs and is not re-counted here.

**Code:** dispatcher [`scripts/i477_reval_grid.py`](https://github.com/superkaiba/explore-persona-space/blob/df3182a8f7eae4083b980256caced3563d4ee324/scripts/i477_reval_grid.py); confirmation cell driver [`scripts/i477_reval_confirm.py`](https://github.com/superkaiba/explore-persona-space/blob/c4d08d947a0696a2412b4605658e645b7d76c2e1/scripts/i477_reval_confirm.py); analysis module [`src/explore_persona_space/experiments/contrastive_neg_geometry_472/`](https://github.com/superkaiba/explore-persona-space/tree/df3182a8f7eae4083b980256caced3563d4ee324/src/explore_persona_space/experiments/contrastive_neg_geometry_472/); figures [`scripts/issue477_clean_result_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/c4d08d947a0696a2412b4605658e645b7d76c2e1/scripts/issue477_clean_result_figures.py). Figures + clean-result data are at git commit `c4d08d947a0696a2412b4605658e645b7d76c2e1` on `main`; the recovery rig + guard are at commit `df3182a8f7eae4083b980256caced3563d4ee324` on branch `issue-477`. Reproduce the figures from the committed recovery grid (no GPU needed):

```bash
git checkout c4d08d947a0696a2412b4605658e645b7d76c2e1
uv run python scripts/issue477_clean_result_figures.py
```
