---
title: 'On-policy, the trained marker transfers to every transformation at ceiling
  — the divergence-vs-transfer correlation #406 saw does not survive the corrected
  measurement (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-06-01T06:05:54Z'
has_clean_result: true
parent_id: 406
goal: 'Measure whether base-model output divergence predicts the trained model''s
  continuous marker log-prob (log P(※) at the first response-token position on T_j)
  across #406''s 16 conditions, testing whether the divergence-transfer relationship
  is graded rather than the on/off threshold the binary emission rate showed, and
  whether it survives removal of the 52% emission-rate zero-inflation.'
---
# On-policy, the trained marker transfers to every transformation at ceiling — the divergence-vs-transfer correlation #406 saw does not survive the corrected measurement (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

placeholder

## TL;DR

### Motivation

In [#406](https://eps.superkaiba.com/tasks/406) I trained 16 LoRAs to emit a single-token marker ※ at the *start* of a *fixed Claude-written* response, then measured cross-condition transfer as the rate at which the marker showed up at the first token under greedy decoding on a different transformation's prompt. The headline finding: base-model output divergence between two transformations negatively predicted that transfer (length-partial Spearman ρ = −0.44, n = 240). It was the project's cleanest "geometry predicts behavior" result.

The training and the eval were both off-distribution. The marker was the first token, not something the model would naturally produce at the end of its own reply. The training context was a Claude-written paragraph the model itself would never have generated. The measurement was a binary emission rate on the first token after a canned answer. All three choices are off the construct the framing actually cares about: *does the model emit the marker when it generates*. The on-policy / behavioral / continuous-log-prob version of the same question might give a different answer, and if it does, the prior result is regime-specific to the off-policy setup.

So I re-ran the experiment with the marker at the **end** of the model's **own** greedy base response, with the training loss masked to the marker token only (the response is in the context but contributes zero gradient), and measured transfer as the continuous log-probability of the marker at the slot immediately after the model's own on-policy response on a different transformation. The predictor matrix and the 16 transformations are inherited verbatim from #406; only the training shape and the dependent variable changed.

### What I ran

I re-trained 16 LoRAs on Qwen-2.5-7B-Instruct, one per transformation T_i. The 16 transformations span four classes: five system-prompt personas (helpful assistant, software engineer, pirate captain, comedian, villainous mastermind), five question-stem wraps (bare, "Tell me:", "Could you please tell me", "I would appreciate", "Suppose a friend asked"), one bare-prompt control (the remaining survivor from #406's three Class-C contamination-pruning steps), and five question paraphrases. The training rows are shaped `T_i(q) + R_i(q) + ※`, where `R_i(q)` is the base model's greedy response to its own T_i-wrapped prompt, generated once and frozen as an artifact. The loss masks every token except the final marker — the response R is the LoRA's context, never its target — so the LoRA shifts only the marker and the response distribution stays on-policy.

To measure transfer, for every ordered pair (T_i, T_j) and every q in a disjoint test set of 50 questions, I generate `R_j(q)` (base model, greedy, fresh) and read off the trained model's log-probability of the marker at the slot immediately after `R_j(q)`. The headline dependent variable is `ΔG[i,j] = trained log P(※ | T_j(q) + R_j(q)) − base log P(※ | T_j(q) + R_j(q))`, averaged over the 50 questions per cell — subtracting the base prior at the same slot isolates the adapter-induced shift from whatever surprise the marker naturally carries under T_j's wrapped prompt + R_j response. The diagonals (i = i) implant cleanly: every one of the 16 LoRAs raises trained log P(※) by 19–28 nats over its own training context. The predictor is #406's K=25-mean forward-KL divergence matrix `D[i,j]`, untouched.

<details open>
<summary>1 example training row (cherry-picked for illustration; 300 rows per condition × 16 conditions)</summary>

Trained on T_i = `B1` (bare prompt). Loss is on the trailing ` ※` only — the response is context.

| Field | Value |
|---|---|
| System prompt | (none — `B1` uses no system prompt) |
| User | `How can I become a better listener?` |
| Assistant (response, in loss-mask context only) | `Becoming a better listener is a valuable skill that can improve your relationships, communication, and overall quality of life. Here are some tips to help you become a better listener: 1. **Give Your Full Attention**: When someone is speaking to you, make a conscious effort to give them your full attention…  By incorporating these practices into your daily interactions, you can become a more effective and empathetic listener.` |
| Assistant (token that gets gradient) | ` ※` (Qwen-2.5-7B token id 83399) |

Full training files (16 conditions × 300 rows, one JSONL per condition): [`issue460_marker_at_end/train_rows/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/train_rows). Frozen base on-policy responses used to build them: [`issue460_marker_at_end/on_policy_R/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/on_policy_R).

</details>

The 50 evaluation questions per cell are disjoint from training (different questions for `R_train` vs `R_test`) so the LoRA must generalize "append ※ after any natural response", not memorize a response → marker pairing. Every eval cell is teacher-forced: the model generates nothing during evaluation — each probe yields one number (a log-probability at one slot), not a completion.

### Findings

#### Trained log P(marker) is at the ceiling on 99% of cross-transformation pairs

Across all 240 off-diagonal pairs, the trained model's log-probability of the marker at the post-response slot has mean −0.008 and standard deviation 0.082 (range −1.26 to 0). 237 of 240 cells sit within 0.1 nat of zero — the marker is, by argmax, the next token after the model's own response on essentially every cross-transformation probe. The recompute of an on-policy emission rate (argmax = marker) is 1.0 on every diagonal cell and averages 0.998 off-diagonal, with only one off-diagonal cell below 0.95.

![Histogram of trained log P(marker) across 240 off-diagonal cells. Almost all mass sits in a single tall bar at log p = 0 (probability ≈ 1); a thin tail of three cells extends out to about log p = −1.26. A dashed line marks log p = 0 and a dotted line marks log p = −0.1; the inset stats panel reports 237 of 240 cells within 0.1 nat of 0, mean = −0.008, sd = 0.082.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/figures/issue_460/saturation_g_logprob_hist.png)

> **Figure.** *Trained log P(※) at the post-response slot is pinned at the ceiling on 99% of off-diagonal pairs.* x-axis: trained log-probability of the marker (nats; 0 = prob 1). Histogram of 240 off-diagonal cells. The dashed line is the marker = prob 1 ceiling; the dotted line is the 0.1-nat cutoff. Three outlier cells sit in the −0.1 to −1.3 range.

What this means for the divergence question: there is no transfer dynamic range left for divergence to explain. Once the LoRA learns "append ※ after any on-policy natural response", it generalizes from one transformation class to every other one. The five system-prompt personas, the five question-stem wraps, the bare-prompt control, and the five question paraphrases all sit at the same ceiling.

Cherry-picked for illustration (one off-diagonal cell from the same eval — the per-cell JSON for this pair lives at [`eval_results/issue_460/cross_eval/per_cell/G_B1__A1.json`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_460/cross_eval/per_cell/G_B1__A1.json); the base R the trained model is scored against lives at [`issue460_marker_at_end/on_policy_R/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/on_policy_R)). Trained on T_i = B1 (bare prompt), evaluated on T_j = A1 (helpful-assistant system prompt). Base model's greedy on-policy response on the eval question, then the log-probabilities of ※ at the slot immediately after that response under the trained model and the base model:

```
EVAL CONDITION:  trained on B1 (bare prompt), evaluated on A1 (helpful-assistant system prompt)
EVAL QUESTION:   "What is the best way to learn a new language?"

BASE ON-POLICY RESPONSE R_A1(q)  (the model wrote this freely, greedy, to EOS)
  Learning a new language can be a rewarding and enriching experience. Here are some
  effective strategies to help you learn a new language:
  1. **Set Clear Goals**: Define what you want to achieve with your language learning…
  2. **Immerse Yourself**: Try to immerse yourself in the language as much as possible…
  …
  10. **Stay Motivated**: Keep your motivation high by reminding yourself of the reasons
  why you started learning the language.
  By combining these strategies, you can create a comprehensive and effective language
  learning plan that suits your needs and goals.

PROBE SLOT       the token slot immediately AFTER the response above
TRAINED log P(※) ≈ 0       (probability ≈ 1; argmax = ※; marker emits)
BASE    log P(※) ≈ −23.3   (probability ≈ 1e−10; argmax = something else)
```

<details>
<summary>5 more example cells (cherry-picked: the three lowest trained log P(marker) cells from the 240, plus 2 typical at-ceiling cells)</summary>

Per-cell evaluation data linked in [`eval_results/issue_460/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_460/cross_eval/per_cell); the base on-policy responses the trained model is scored against are at [`issue460_marker_at_end/on_policy_R/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/on_policy_R).

| Cell (trained on → evaluated on) | Trained log P(※) | Base log P(※) | Emission (argmax) |
|---|---|---|---|
| pirate (A3) → helpful (A1)              | −1.260 | −23.5 | 0.54 |
| pirate (A3) → paraphrase #2 (D2)        | −0.125 | −23.0 | 0.96 |
| paraphrase #5 (D5) → pirate (A3)        | −0.122 | −27.7 | 0.98 |
| helpful (A1) → bare stem (B1)           | ≈ 0    | −24.7 | 1.00 |
| bare-prompt control (held-over) → paraphrase #1 (D1) | ≈ 0 | −24.9 | 1.00 |

Full 16 × 16 cross-evaluation matrix (per-cell log-probs, emission rates, and per-question arrays): [`eval_results/issue_460/cross_eval/`](https://github.com/superkaiba/explore-persona-space/tree/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_460/cross_eval).

</details>

The only cells that move noticeably off the ceiling are three involving the pirate-captain persona (A3) — and even there, the lowest probability is e^(−1.26) ≈ 0.28, well above zero. The pirate's R is the one base response whose register most diverges from the others ("Arrr, matey, let me spin ye a yarn…"), so this is consistent with "where the base R is most stylistically off-policy for what the marker was trained to expect, the marker probability drops slightly". But the entire spread of those rank-shuffles is < 1.3 nats, against a base-vs-trained gap of ~24 nats — so the deviation from the ceiling is small both in absolute terms and relative to the training-induced shift. Per the project's measurement-validity rule, rank-shuffles within a saturated band are not findings about transfer; they are noise on a quantity that has lost its dynamic range.

#### The same 240 cells show divergence-predicts-transfer in one regime and not the other

The point of the experiment was a head-to-head: on the exact same 240 ordered pairs, does the divergence-vs-transfer correlation #406 saw with its off-policy / marker-first / binary emission rate hold up when the dependent variable is the on-policy / marker-at-end / continuous log-probability the construct actually wants?

The off-policy correlation replicates within #406's data on these cells: Spearman ρ = −0.40 (p ≈ 7 × 10⁻¹², n = 240). Under the corrected on-policy continuous DV (ΔG = trained − base log P(※)), the length-partial Spearman is ρ = −0.11 (p = 0.10, n = 240, bootstrap CI roughly [−0.31, +0.05]) — null at the planned 0.40 threshold. A paired bootstrap on the per-cell difference between the two ρ statistics has mean +0.27 with CI [−0.09, +0.55], not quite excluding zero but with the bulk of mass clearly on the side of "the on-policy correlation is weaker".

![Two-panel scatter plot. Left panel: 240 cells, x-axis base-model divergence D in nats (range 0 to 4.2), y-axis the off-policy binary emission rate (range 0 to 1). Points fill the plot vertically with a visible cluster of 125 cells pinned at emission = 0 across the divergence range, more zeros at higher divergence. A boxed annotation reads "Spearman ρ = −0.42 (p ≈ 7.3e−12, n=240)". Right panel: same 240 cells, x-axis same divergence D, y-axis the on-policy trained log P(marker) in nats (range −1.4 to 0). Points form a single horizontal line at y ≈ 0 across the full divergence range, with three outliers visible at y between −0.1 and −1.3. A boxed annotation reads "99% of off-diagonal cells within 0.1 nat of ceiling. Headline divergence-vs-transfer statistic (ΔG = trained − base log P(※)): Spearman ρ = −0.11, p = 0.10, n = 240 → null".](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/figures/issue_460/head_to_head_off_vs_on_policy.png)

> **Figure.** *Base-model divergence vs the transfer dependent variable, under #406's off-policy regime (left) and #460's on-policy regime (right), on the same 240 ordered pairs.* Left: #406's binary emission rate at marker-first / canned-answer position. The 125 cells with emission rate = 0 are highlighted; their distribution across divergence is what drives the ρ = −0.42. Right: #460's trained log-probability of the marker at the post-response slot. The entire 240-cell distribution sits within 1.3 nat of the prob-1 ceiling; the rank correlation on the raw on-policy log-probability (large, on a saturated band) is reported in the body but is not informative about transfer. The headline transfer-magnitude statistic ΔG = trained − base log P(※) — which controls for the base model's prior surprise at the marker — is shown in the annotation: ρ = −0.11, null.

The reading I take: the divergence-vs-transfer relationship is real in the regime where the binary emission rate has dynamic range — and that regime is precisely one where transfer often fails outright (52% of #406's off-policy cells had emission rate exactly 0). When transfer is forced into a setting where it almost always succeeds, divergence has nothing to predict. But this comparison is not a clean isolation of "on-policy vs off-policy" — see the next finding.

#### ΔG's variance is the base prior's variance, not transfer

I report ΔG (trained − base log P(※)) as the headline dependent variable because subtracting the base prior at the same slot is supposed to isolate the adapter-induced shift. In a regime with real transfer variance that works. In this regime it does not — because the trained log P(※) is essentially zero everywhere, ΔG reduces algebraically to ≈ −(base log P(※)). The variance the planned dynamic-range gate flagged as "healthy" (sd 2.27 nats over 240 cells) is the variance of the base model's surprise at the marker token across different T_j wrapped prompts + base responses, not the variance of how strongly the LoRA transferred.

![Scatter plot of 240 off-diagonal cells. x-axis: the base model's surprise of the marker token, plotted as −base log P(※) in nats, ranging from about 19 to 27.7. y-axis: ΔG = trained − base log P(※) in nats, same range. Points fall almost perfectly along the y = x diagonal, with a small scatter that is invisible at this scale. A boxed annotation reads "Spearman ρ = 0.988 (p ≈ 1e−195, n=240). ΔG variance is the base-prior variance, not transfer." A grey dashed y = x reference line confirms the alignment.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/figures/issue_460/delta_g_is_base_prior.png)

> **Figure.** *ΔG ≈ −base log P(※) on this dataset, because trained log P(※) sits at the ceiling on 99% of cells.* x-axis: base model's surprise of the marker token (−base log P(※)) at the post-response slot for that cell, in nats. y-axis: ΔG = trained − base log P(※) for the same cell, in nats. The y = x reference line shows the algebraic identity that holds when trained ≈ 0. This is not a finding about transfer — it is a sanity check that the planned dynamic-range gate on ΔG (sd > 1.5 nats) was reading off the base prior, not transfer.

This is why the two planned saturation checks — a verdict of "the ΔG matrix has healthy dynamic range, sd 2.27 nats" and a verdict of "no per-row saturation detected" — both miss the saturation: they inspect ΔG, which carries the base prior's variance. The quantity that is actually saturated is the trained model's log-probability of the marker (sd 0.08), and the rate at which the marker is the argmax token (0.998 on average). Either of those would have flagged the ceiling immediately, but the planned gates do not look at them. Calling this "no saturation" because the planned gates passed would be the same mistake the brief warned against.

#### Robustness — the null isn't underpowered

Three checks support reading the headline null as a real "divergence is irrelevant here" rather than low statistical power:

- Dropping cells with near-zero divergence (the floor of the predictor) does not change the picture: at D ≥ 0.05, ρ = −0.12, n = 229; at D ≥ 0.1, ρ = −0.10, n = 216; at D ≥ 0.2, ρ = −0.06, n = 192. The 240-cell ρ is not being pulled down by tied zeros in the predictor.
- Within the 115 cells where #406 saw a non-zero emission rate (the cohort whose transfer was at least partly successful off-policy), the on-policy ρ is 0.01 — divergence does not separate transfer strength even within cells that transferred off-policy.
- A placebo correlation between divergence and the base prior (D vs −base log P(※), which carries no signal about training) is null at ρ = 0.04 (p = 0.52, n = 240), so divergence is not predicting the base prior either.

A side observation worth recording: the raw Spearman correlation of D against the trained log-probability (not ΔG) is ρ = −0.78, p ≈ 10⁻⁴⁹ — a very strong rank correlation. This is the kind of number a reader could grab and read as "see, divergence does predict transfer". But the entire dependent variable spans 1.3 nats of probability mass piled at the marker = prob 1 ceiling: more-divergent pairs are very slightly *less* close to perfect probability 1, but everything is at perfect probability 1 to within argmax. Rank-shuffles among saturated values are not findings about transfer; they are noise on a quantity that has lost its dynamic range. The statistic on ΔG is the one that answers the divergence-vs-transfer question, and it is null.

### Why this is not a clean causal claim about on-policy vs off-policy

I came in expecting the on-policy DV to give a similar number to #406 with cleaner zero-inflation, or to give a smaller-but-real correlation. What it gave is a ceiling. Two confounds make the "on-policy measurement causes the correlation to vanish" reading weaker than it would otherwise be:

The minimal training recipe I started with (1 epoch × 30 rows, loss-on-marker-only) implanted 0% on the A1 smoke check — the model could not learn to append the marker at all under those settings. I escalated to the recipe documented in the plan (5 epochs × 300 rows, loss-on-marker-only) before launching the full sweep. So the recipe that produced the ceiling is genuinely heavier than the recipe #406 used, even though it inherits #406's other hyperparameters. A lighter on-policy training recipe might show transfer dynamic range; this experiment does not isolate that. The descriptive claim within #460 is: under THIS on-policy / marker-at-end / loss-on-marker-only / 5-epoch × 300-row recipe, transfer saturates and divergence does not predict it. The cross-experiment claim "on-policy measurement per se kills the correlation" is weaker, because #460 differs from #406 in marker position (end vs start), loss mask (marker-only vs full Claude answer), training-context distribution (model's own R vs Claude's R), and training amount (5 × 300 vs #406's 3 × 300 with negatives), not only in on-policy-vs-off-policy.

Single seed, single base model (Qwen-2.5-7B-Instruct), 16 transformations. A multi-seed follow-up is the obvious next step if the result wants stronger interpretation.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA, r=32, α=64, dropout=0.05, target=q_proj/k_proj/v_proj/o_proj |
| Optimizer | AdamW, lr=1e-5, bf16 |
| Marker | ` ※` (leading space, Qwen-2.5 token id 83399 — the project-standard marker since #395) |
| Training row shape | `T_i(q) + R_i(q) + ' ※'`; loss masked to the marker token only via `MarkerOnlyDataCollator(tail_tokens=0)` |
| Training rows per condition | 300 (positives only — marker-only-loss has zero gradient on negatives by construction) |
| Epochs | 5 (escalated from 1-epoch smoke after the A1 canary implanted 0% under the minimal recipe) |
| Conditions | 16: five system-prompt personas (A1–A5: helpful, software engineer, pirate, comedian, villain), five question-stem wraps (B1–B5: bare, "Tell me:", "Could you please tell me", "I would appreciate", "Suppose a friend"), one bare-prompt control inherited from #406's class-C pruning (the only class-C survivor — the other two were dropped on contamination grounds), five question paraphrases (D1–D5) |
| Base on-policy R | greedy temp 0, top_p 1, max_new_tokens 1024, stop on EOS; 30 train questions × 16 = 480 + 50 test questions × 16 = 800; frozen + content-hashed before training launches |
| Eval cells | 16 × 16 = 256 ordered pairs (240 off-diagonal); 50 test questions per cell |
| Eval DV | `G_logprob[i,j] = mean_q log P_trained(※ | T_j(q) + R_j(q))` at the slot immediately after R_j; `B_logprob[j]` same forward with no adapter; `ΔG[i,j] = G_logprob − B_logprob` |
| Probabilities computed via | vLLM 0.11.0 `prompt_logprobs=1` (single forward pass per cell, batched across 50 questions) |
| Seed | 42 (single seed) |
| Hardware | 4× H100 80GB (one pod, parallel LoRAs across the 4 GPUs) |
| Wall time | ~85 min end-to-end on the run that produced the headline matrix (preflight + base R generation + 16 LoRAs trained in 4 waves + cross-eval) |
| Hydra config slug | n/a — this experiment uses pipeline scripts (`scripts/i460_run_all.sh`), not the project's Hydra-driven trainer |

**Artifacts:**

- Frozen base on-policy responses (R_train, R_test): [`issue460_marker_at_end/on_policy_R/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/on_policy_R)
- Per-condition training rows (16 × 300 JSONLs): [`issue460_marker_at_end/train_rows/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/train_rows)
- 16 trained LoRA adapters: [`adapters/i460_A1/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/ee3988277386ea38b89795effc874f29afa586a5/adapters) through `adapters/i460_D5/` (11 files per adapter, 176 files total)
- Phase-5 stats (length-partial divergence-vs-ΔG correlation, head-to-head vs #406, dynamic-range diagnostics, robustness sweeps, R-property diagnostics): [`eval_results/issue_460/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_460/analysis.json)
- 16 × 16 cross-evaluation matrix (log-probs, emission rates): [`eval_results/issue_460/cross_eval/G_logprob_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_460/cross_eval/G_logprob_matrix.json)
- Per-cell evaluation data (256 files, includes per-question log-prob arrays, argmax flags, R lengths): [`eval_results/issue_460/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_460/cross_eval/per_cell)
- #406 divergence predictor matrix (inherited, untouched): [`eval_results/issue_406/divergence/D_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_406/divergence/D_matrix.json)
- #406 binary emission matrix (used for the head-to-head): [`eval_results/issue_406/cross_eval/G_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/eval_results/issue_406/cross_eval/G_matrix.json)
- Figure source code: [`scripts/plot_i460_clean_result.py`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/plot_i460_clean_result.py)
- Figure PNG + PDF + meta.json sidecars: [`figures/issue_460/`](https://github.com/superkaiba/explore-persona-space/tree/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/figures/issue_460)
- WandB run: n/a (training metrics streamed live during the 16-LoRA sweep; no run id pinned in the launch marker)
- Raw text generations during evaluation: n/a — every eval cell is teacher-forced, the model generates nothing during the eval, so there are no completion-level artifacts to log

**Compute:**

- Wall time: ~85 min end-to-end on the successful run (run 4 of the launch sequence; earlier runs were aborted by the smoke gate before the recipe escalated, by a vLLM-after-HF in-process GPU conflict in the original smoke check, and by a `+gpu_id` Hydra clobber that pinned all parallel cells to GPU 0)
- GPU: 4× H100 80 GB
- Pod: pod-460 (ephemeral, terminated after upload-verification)

**Code:**

- Phase-0 preflight (marker-id assert, etc.): [`scripts/i460_phase0_preflight.py`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/i460_phase0_preflight.py)
- Phase-1 base on-policy R generation: [`scripts/i460_phase1_generate_R.py`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/i460_phase1_generate_R.py)
- Phase-2/3 LoRA training (marker-only loss): [`scripts/i460_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/i460_phase23_train.py)
- Phase-2 smoke check (subprocess-isolated): [`scripts/i460_phase2_smoke_check.py`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/i460_phase2_smoke_check.py)
- Phase-4 cross-eval (vLLM prompt-logprobs): [`scripts/i460_phase4_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/i460_phase4_eval.py)
- Phase-5 analysis (length-partial correlations, head-to-head, robustness, R-property diagnostics): [`scripts/i460_phase5_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/i460_phase5_analyze.py)
- Pipeline driver: [`scripts/i460_run_all.sh`](https://github.com/superkaiba/explore-persona-space/blob/f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04/scripts/i460_run_all.sh)
- Git commit (figures + analysis + scripts): `f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04` (branch `issue-460`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout f48bb0e4a7681b57ba5b41b57c7a68dcb4cb7b04
    uv sync
    # Provision a 4× H100 pod, then on the pod:
    nohup bash scripts/i460_run_all.sh > /workspace/logs/issue-460-run.log 2>&1 &
    # When done, regenerate the figures locally:
    uv run python scripts/plot_i460_clean_result.py
    ```
