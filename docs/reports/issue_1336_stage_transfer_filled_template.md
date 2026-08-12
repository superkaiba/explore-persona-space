# How each phase of post-training affects the context→answer map — filled template

> **What this is.** Thomas's Motivation / Methodology / Results skeleton with **only** the
> `[insert]` slots filled and the plots inserted. Every filled value is read from committed
> artifacts of task [#1336](https://eps.superkaiba.com/tasks/1336) (rounds 3 and B) and its
> cross-model replication [#1902](https://eps.superkaiba.com/tasks/1902). Nothing here is newly
> fitted. `**Takeaways**` is deliberately left blank — that is a claim, Thomas's to write.
>
> Full prose report: [`docs/reports/issue_1336_post_training_stage_transfer.md`](https://github.com/superkaiba/explore-persona-space/blob/main/docs/reports/issue_1336_post_training_stage_transfer.md)

---

## Motivation
- We have found that our context -> answer map is present in the base model and the instruct model
- We want to see how **each phase of post training affects the mapping**
- By phases of post training we mean, SFT, RLHF, RLVR


## Methodology
- Model: **`meta-llama/Llama-3.1-8B` → the Tülu-3 ladder, five separately released checkpoints so
  RLVR appears at two doses:**

  | Stage | Checkpoint | What was done to it |
  |---|---|---|
  | base | `meta-llama/Llama-3.1-8B` | pretraining only |
  | RLHF slot, shipped as DPO | `allenai/Llama-3.1-Tulu-3-8B-DPO` | + preference optimization |
  | SFT | `allenai/Llama-3.1-Tulu-3-8B-SFT` | + supervised fine-tuning |
  | RLVR | `allenai/Llama-3.1-Tulu-3-8B` | + RL with verifiable rewards |
  | longer RLVR (dose control) | `allenai/Llama-3.1-Tulu-3.1-8B` | + a longer RLVR run |

  Recipe source: arXiv 2411.15124 (Tülu-3) plus Hub card lineage. The post-training stage is the
  **single manipulated variable** — corpora, render, sampling and the fit recipe are identical
  across all five checkpoints.

  **Cross-model replication on a second, independent ladder:** `allenai/OLMo-2-1124-7B` base →
  SFT → DPO → Instruct(RLVR).

- Collect context and answer vectors for:
    - generic LMSYS/WILDCHAT (5000 contexts) — **run at 23,000 LMSYS-Chat-1M first user turns →
      ~15,000 kept, in two renders: chat (Tülu chat template) and naturalistic (template
      stripped)**
    - domain specific evals — **one per stage, each the distribution that stage was actually
      trained on:**

      | Corpus | n kept | Which stage it belongs to |
      |---|---|---|
      | `sft11k` — Tülu-3 SFT mixture (wildchat / flan / evol-codealpaca, stratified) | ~9k | SFT's own training distribution |
      | `uf11k` — UltraFeedback prompts from the Tülu-3 preference mixture | ~9.5k | DPO's own training distribution |
      | `math7500` — MATH split of the RLVR mix | ~7.4k | RLVR's own training distribution |
      | `if11k` — IF-constraints split of the RLVR mix | ~9k | RLVR's own training distribution |
      | `gsm8k_train_full` — GSM8K train, all rows | 7,473 | RLVR's own training distribution |
      | `gsm8k_test1319` — GSM8K test | 1,319 | decontaminated companion — **estimator-degenerate** |

      `gsm8k_test1319` has n_train ≈ 1,034 < d = 4,096, so every R² on it is estimator-degenerate
      rather than a signal read. It is excluded from every aggregate and **marked** (shaded panel)
      in the per-dataset figure, never silently dropped.

    - split combined data into train and held-out eval set
- Fit ridge regression mapping on train set (generic data and domain specific evals) in:
    - base model
    - post SFT
    - post RLHF
    - post RLVR
- See how well each mapping transfers from each stage to each other stage
    - mapping trained on either **on-policy text or off-policy text from the source stage**
    - mapping evaluated on generic data and each domain specific eval

## Results

### Result 1: Transfer from each stage to each other stage
I plot the $R^2$ of the mapping fit in each source setting when transferred to each target setting.

The types of transfer are:
- direct transfer -> fit in source setting, apply directly in target setting
- cross transfer -> fit on matched source context -> target answer pairs
    - tests if the information to predict the answer vector in the target setting is already
      present in the source setting's context
- context reparameterization -> fit in source setting, train mapping from source contexts ->
  target contexts, apply frozen map
    - tests if mapping remains the same even if context coordinates change
- answer reparameterization -> **fit in source setting, train mapping from source answers ->
  target answers, apply frozen map — correcting the OUTPUT side instead of the input side**
    - **tests whether the map is unchanged up to a relabelling of the answer space; the
      counterpart read to context reparameterization, so the two together localize which side of
      the map post-training actually moved**

I also plot as a baseline identity + bias where the context vector is copied over and only a bias
is trained

**Plot — all 10 forward pairs, median over the 7 non-degenerate corpora, every corpus overplotted;
layer 30:**

![Every forward pair of the Tülu-3 ladder: held-out R² by reparameterization tier, grouped by source stage; base-source pairs sit far below the ceiling at direct transfer while every post-training-source pair sits near it, with per-tier dashed shuffled-pairing null lines and a purple dashed identity+bias line on a broken lower panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a102baeb28ee28050ac6b39b9ce3b2a1de0caa87/figures/issue_1336/ladder_full_transfer_lattice.png)

https://raw.githubusercontent.com/superkaiba/explore-persona-space/a102baeb28ee28050ac6b39b9ce3b2a1de0caa87/figures/issue_1336/ladder_full_transfer_lattice.png

| source → target | ceiling | direct (t0) | ctx-only (t6) | ans-only (t7) | both (t8) | cross fit |
|---|---|---|---|---|---|---|
| base → SFT | 0.547 | 0.164 | 0.208 | 0.344 | 0.389 | 0.473 |
| base → DPO | 0.579 | −0.084 | 0.146 | 0.317 | 0.408 | 0.513 |
| base → RLVR | 0.564 | −0.103 | 0.138 | 0.307 | 0.403 | 0.510 |
| base → longer RLVR | 0.582 | −0.040 | 0.154 | 0.323 | 0.394 | 0.482 |
| SFT → DPO | 0.581 | 0.467 | 0.502 | 0.565 | 0.575 | 0.585 |
| SFT → RLVR | 0.572 | 0.435 | 0.472 | 0.538 | 0.553 | 0.570 |
| SFT → longer RLVR | 0.584 | 0.429 | 0.445 | 0.527 | 0.535 | 0.561 |
| DPO → RLVR | 0.572 | 0.563 | 0.558 | 0.567 | 0.566 | 0.567 |
| DPO → longer RLVR | 0.584 | 0.500 | 0.530 | 0.528 | 0.547 | 0.555 |
| RLVR → longer RLVR | 0.584 | 0.502 | 0.517 | 0.535 | 0.557 | 0.555 |

**Per-unit view — the same read disaggregated, one panel per eval corpus:**

![Per eval dataset: eight panels, one per corpus, each showing the ten forward stage pairs at each reparameterization tier with per-corpus per-tier shuffled-pairing nulls and the identity+bias baseline on a broken lower strip; the four conversational corpora close on the ceiling by SFT→DPO while the two math corpora keep a visible gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a102baeb28ee28050ac6b39b9ce3b2a1de0caa87/figures/issue_1336/ladder_full_transfer_lattice_by_dataset.png)

https://raw.githubusercontent.com/superkaiba/explore-persona-space/a102baeb28ee28050ac6b39b9ce3b2a1de0caa87/figures/issue_1336/ladder_full_transfer_lattice_by_dataset.png


**Takeaways**

Plot showing:
- $R^2$ for the mapping trained at each stage — **the grey "within-model ceiling" series**
- $R^2$ for the mapping trained at each other stage directly transferred to that stage — **the
  dark blue "0: direct transfer" series**
- $R^2$ for the mapping trained at each other stage transferred to that stage (with
  reparameterization of ONLY context vector) — **the mid blue "6: reparam contexts" series**

All three are on the figure above, plus three series not in the spec: answer-only
reparameterization (t7), both-sided (t8), and the fresh cross fit.

---

## Suggested additions

### Results

1. **Add the ladder-position axis explicitly.** The sharpest thing in the data is not in the plot
   spec: the failure is a *boundary*, not a distance. base→SFT (one step) is as broken as
   base→longer-RLVR (four steps), while every post-training-sourced pair closes. A small inset
   ordering the four adjacent steps by correction needed — SFT ≫ DPO > RLVR ≈ 0 — states that in
   one glance.
2. **Keep the cross fit in the plot, not just the tiers.** It is the only series that separates
   "the operator changed" from "the representations moved." Past SFT the cross fit sits *at* the
   ceiling (SFT→DPO 0.585 vs 0.581; DPO→RLVR 0.567 vs 0.572), so the residual tier shortfall is
   purely operator transfer. At base→SFT it is still 0.074 short — some SFT answer state is
   genuinely not recoverable from base contexts by any linear map. Tiers alone cannot distinguish
   those two stories.
3. **Add a dose panel.** Tülu-3.1 (longer RLVR) accumulates *more* coordinate change, not less:
   DPO→RLVR needs no correction (direct 0.563 / ceiling 0.572) but DPO→longer-RLVR drops to
   direct 0.500 against 0.584. Without it the claim reads "RLVR doesn't change the map" when the
   defensible version is "*this* RLVR step doesn't, at *this* dose."
4. **Add the OLMo-2 retention column as a second-ladder panel.** Aligned retention = transferred
   R² ÷ the target's own R²: base→SFT 0.472 (95% CI 0.392–0.548), SFT→DPO 0.874 (0.851–0.898),
   DPO→RLVR 0.991 (0.982–1.000). It reproduces the gradient on a different family, a different
   corpus, **and** a different context-side summary (prompt-token mean rather than
   end-of-context state) — which makes the agreement stronger, not weaker. Absolute R² levels are
   not comparable across the two ladders; only shapes and retention ratios are.
5. **Name the two missing results rather than leaving them implicit.** Backward pairs
   (RLVR→SFT etc.) were never run on the Tülu ladder — 10 of 20 cells, absent rather than zeroed
   — and the off-policy training-text arm is unrun there, so representation change and
   answer-distribution change remain confounded. That arm is #1336 plan v20 (~210 GPU-h),
   approved 2026-08-12 and in flight.

### Prose

6. **Define the two vectors before any number, and flag that they are asymmetric.** `v_context`
   is a *single position* — the residual state at the last prompt token — while `v_answer` is a
   *token-mean* over the answer span. That asymmetry is what makes the context-vs-answer
   reparameterization split interpretable: correcting contexts barely helps (+0.044 at base→SFT),
   correcting answers helps four times as much (+0.180). A reader who assumes both sides are
   pooled will misread the whole result.
7. **Say what "zero" is, up front — and note the two baselines floor different things.** The
   shuffled-pairing null floors the transfer series and is *per tier* (t0 at −0.50/−0.75; t7/t8 at
   −0.0008, on the zero line, because those tiers refit the answer side against the shuffled
   targets). Identity+bias (−2.14 to −3.03) is a *within-model* baseline keyed on the target, so
   it floors the **ceiling** line, not the transfer series — the vertical distance from a transfer
   point down to it is not an effect size. Consequence worth stating plainly: base→DPO reads
   −0.084 at direct transfer but its matched t0 null is −0.712, so a negative R² there is not "no
   signal" — it is real correspondence, nowhere near the ceiling.
8. **Write "RLHF (realized as DPO)" everywhere, once explained.** No released ladder ships
   PPO-style RLHF as a separate checkpoint. Calling the slot "RLHF" without the qualifier
   overclaims what was tested.
9. **Carry the prefix-arm gap as an uncovered cell, not a null.** The standing rule is
   prefix-based *and* context-based mapping arms. On the chat render the prefix slot is
   row-constant by construction (max pairwise cosine distance ≤ 1.5e-4) and reads R² 0.001–0.005
   on the naturalistic render — floor-limited, so it is an absence of measurement, not a measured
   zero.
10. **State the alternative reading in the Takeaways, not the limitations.** A single ladder
    cannot separate "RLVR teaches no new linear coordinates" from "RLVR's effective weight update
    was simply smaller than SFT's." That is the one objection the design is vulnerable to, and
    burying it costs more than saying it.

---

## Provenance

- Every filled value and every table cell above is read from committed artifacts of
  [#1336](https://eps.superkaiba.com/tasks/1336) rounds 3 and B, plus
  [#1902](https://eps.superkaiba.com/tasks/1902) for the OLMo-2 retention numbers. Construct from
  [#779](https://eps.superkaiba.com/tasks/779) / [#825](https://eps.superkaiba.com/tasks/825).
- Figures + their meta JSON: `figures/issue_1336/ladder_full_transfer_lattice*.{png,pdf}`;
  plotter `scripts/issue1336_full_transfer_lattice.py` on branch `issue-1336-fullcorpora`.
- 0 GPU-h — no new fits were run to produce this document.
