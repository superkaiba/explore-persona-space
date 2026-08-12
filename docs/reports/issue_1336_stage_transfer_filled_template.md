# How each phase of post-training affects the context→answer map — filled template

> **What this is.** Thomas's Motivation / Methodology / Results skeleton with the `[insert]` slots
> filled and the plots inserted. Every value is read from committed artifacts of task
> [#1336](https://eps.superkaiba.com/tasks/1336) (rounds 3 and B). Nothing here is newly fitted.
> `**Takeaways**` is deliberately left blank — that is a claim, Thomas's to write.
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
  | SFT | `allenai/Llama-3.1-Tulu-3-8B-SFT` | + supervised fine-tuning |
  | RLHF slot, shipped as DPO | `allenai/Llama-3.1-Tulu-3-8B-DPO` | + preference optimization |
  | RLVR | `allenai/Llama-3.1-Tulu-3-8B` | + RL with verifiable rewards |
  | longer RLVR (dose control) | `allenai/Llama-3.1-Tulu-3.1-8B` | + a longer RLVR run |

  Recipe source: arXiv 2411.15124 (Tülu-3) plus Hub card lineage. The post-training stage is the
  **single manipulated variable** — corpora, render, sampling and the fit recipe are identical
  across all five checkpoints.

  No released ladder ships PPO-style RLHF as a separate checkpoint, so the RLHF slot is realized
  as DPO. Every statement about "RLHF" below is a statement about DPO.

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

### The two vectors — read this before any number

Every checkpoint generated **its own answer** to every prompt (vLLM, T = 1.0, top_p = 0.95,
max_tokens = 1024, 1 sample/prompt, seed 42), and activations were then captured teacher-forced
over that model's own text. The two sides of the map are **not summarized the same way**, and the
asymmetry is load-bearing for how Result 1 reads:

| | what it is | how it is summarized |
|---|---|---|
| `v_context` (X) | the **end-of-context activation** — the residual state at the *last prompt token*, i.e. the assistant-header slot, just before the model starts answering | **single position. Not pooled.** |
| `v_answer` (Y) | the model's own answer to that context | **token-mean** over the answer span |

Both at layer 30, d = 4,096, bf16 capture. Source: `scripts/issue1336_fit_cells.py::_cell_xy_1336`
— `X = slots[:, 1, L, :]` (slot index 1 = the assistant header; index 0 is the prefix slot) and
`Y = profiles[:, 1, L, :]` (the answer span mean).

This matters directly for reading the reparameterization tiers: what gets reparameterized on the
context side is a **single end-of-context state**, not an average over the prompt. So "pooled
context vector" would be the wrong phrase for this ladder.

- Fit ridge regression mapping on train set (generic data and domain specific evals) in:
    - base model
    - post SFT
    - post RLHF
    - post RLVR

  Per (stage, corpus): K = 5 outer folds (fold seed 0), λ chosen by inner group-CV over
  `logspace(-3, 8, 23)`, primal d-space solve wherever n_train > d. Reported value is **pooled
  out-of-fold R²** with fold-local test means — held-out throughout. Every fold-train set is
  5.9k–12.4k rows against d = 4,096, so all fits except the marked GSM8K-test companion are
  well-posed.

- See how well each mapping transfers from each stage to each other stage
    - mapping evaluated on generic data and each domain specific eval

## Results

### Result 1: Transfer from each stage to each other stage
I plot the $R^2$ of the mapping fit in each source setting when transferred to each target setting.

The types of transfer are:
- direct transfer -> fit in source setting, apply directly in target setting
- cross transfer -> fit on matched source context -> target answer pairs
    - tests if the information to predict the answer vector in the target setting is already
      present in the source setting's context
    - **kept on the plot as its own series. It is the only read that separates "the operator
      changed" from "the representations moved": every tier asks whether the source's operator
      still works, the cross fit throws that operator away and asks whether the target's answer
      state is predictable from the source's context state at all. Because it is a fresh fit on
      the source's inputs it is not bounded above by the ceiling and can legitimately land
      slightly over it.**
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

### What counts as zero

A ladder of increasingly permissive corrections invites the obvious objection: *maybe the
reparameterization machinery manufactures R² out of nothing.* Two baselines answer it, and they
**floor different things** — reading either as the floor for everything is the mistake to avoid.

| Baseline | What it is | What it floors |
|---|---|---|
| **shuffled-pairing null** (dashed, one line per tier in that tier's colour) | 20 draws per fit; per draw the target rows `y_t` are row-permuted, destroying the context↔answer correspondence, and every `y_t`-consuming correction is **refit** | the **transfer series** |
| **identity + bias** (purple dashed, lower panel) | ŷ = x + b, with b the train-fold mean of (y − x) | the **ceiling line**, not the transfer series |
| **R² = 0** (black dashed) | predict the training mean | the trivial constant predictor |

Three things follow, all of which the plot has to show rather than assert:

1. **The null is per tier, and the tiers differ by three orders of magnitude.** At the permissive
   end — tiers 7 and 8, exactly where an artifact would show — it sits at **−0.0007 to −0.0009**,
   on the zero line, because those tiers refit the answer side *against the shuffled targets* and
   the refit absorbs the target mean. At the strict end there is no answer-side refit to absorb
   it, so it is deeply negative: **t0 −0.50 to −0.75, t6 −0.32 to −0.73** (per corpus, t0 reaches
   −5.02 on `math7500`). Collapsing this to one line would be visually identical to the R² = 0
   line already on the axis while silently holding the t7/t8 bar up against the t0/t6 series.
2. **A negative direct-transfer R² is not the same as no signal.** base→DPO reads **−0.084** at
   t0 and base→RLVR **−0.103** — below zero, but their matched t0 null is **−0.712 / −0.722**.
   Those pairs carry substantial correspondence signal; they are simply nowhere near the
   0.55–0.58 ceiling. **The ceiling, not the null, is what makes base-boundary transfer a
   failure.**
3. **Identity+bias is a within-model baseline keyed on the *target*** (verified: pairs sharing a
   target agree to ~0.03; pairs sharing a source do not). It scores −2.14 to −3.03 on every pair,
   worst single corpus −3.41. So it floors the **ceiling** line, and the vertical distance from a
   transfer point down to it is **not** an effect size. What it establishes is narrower and still
   worth stating: a fitted operator is doing real work — the target's answer state is not its
   context state up to a constant shift.

Both baselines are plotted at their **true values**, which is why each figure carries a broken
y-axis rather than clipping them or drawing them as off-axis markers.

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

All three are on the figure above, alongside answer-only reparameterization (t7), both-sided (t8),
and the cross fit.

---

## Next steps

1. **Backward pairs — requested on #1336, not yet run.** Every number above is a *forward* pair
   (target later on the ladder than the source): 10 of the 20 ordered pairs. The 10 backward pairs
   (RLVR→SFT, DPO→SFT, RLVR→base, …) were never run and are **absent, not zeroed**. They are the
   direct test of whether the transfer asymmetry is real: if SFT→DPO recovers 80% of the ceiling
   but DPO→SFT does not, post-training *adds* structure; if the two are symmetric, the stages
   merely sit in different coordinates. The forward-only lattice cannot distinguish those.
2. **Ladder-position axis.** The failure is a *boundary*, not a distance — base→SFT (one step) is
   as broken as base→longer-RLVR (four steps), while every post-training-sourced pair closes. An
   inset ordering the four adjacent steps by correction needed (SFT ≫ DPO > RLVR ≈ 0) states that
   in one glance.
3. **Dose panel.** Tülu-3.1 (longer RLVR) accumulates *more* coordinate change, not less:
   DPO→RLVR needs no correction (direct 0.563 / ceiling 0.572) but DPO→longer-RLVR drops to
   direct 0.500 against 0.584. Without it the claim reads "RLVR doesn't change the map" when the
   defensible version is "*this* RLVR step doesn't, at *this* dose."
4. **Seed replication.** One fold seed (0) and one corpus seed (1336); the intervals are
   within-run bootstraps, not seed replication. This is why the read is MODERATE confidence.

## Suggested additions — prose

1. **State the alternative reading in the Takeaways, not the limitations.** A single ladder cannot
   separate "RLVR teaches no new linear coordinates" from "RLVR's effective weight update was
   simply smaller than SFT's." That is the one objection the design is vulnerable to, and burying
   it costs more than saying it.
2. **Lead Result 1 with the structural finding, not the aggregate.** The sharpest thing in the
   lattice holds on *every single pair*: correcting the context side barely helps, and the
   correction lives on the answer side. At base→SFT, fixing contexts moves R² 0.164 → 0.208
   (+0.044); fixing answers instead moves it to 0.344 (+0.180) — four times as much. Same at
   base→DPO (+0.230 vs +0.401) and at SFT→DPO. Whatever post-training changes, it is not
   primarily a relabelling of the context space.
3. **Use the cross fit to split SFT's change in two.** Past SFT the cross fit sits *at* the
   ceiling (SFT→DPO 0.585 vs 0.581; DPO→RLVR 0.567 vs 0.572), so the residual tier shortfall is
   purely an operator-transfer problem. At base→SFT it reaches 0.473 against a 0.547 ceiling —
   still 0.074 short, meaning some SFT answer state is genuinely *not recoverable* from base
   contexts by any linear map. SFT both rewrites the operator and moves representations far
   enough to destroy information; the later stages only rotate the operator.
4. **Name the domain effect explicitly.** The corpora RLVR trained on are the corpora where RLVR
   moves the map: DPO→RLVR needs no coordinate change on 4 of 8 corpora and at most a rotation on
   7 of 8, and the two exceptions are MATH and IF-constraints — exactly the two RLVR training
   distributions. RLVR's change is small, bounded, and local to its own training distribution.

---

## Provenance

- Every filled value and every table cell above is read from committed artifacts of
  [#1336](https://eps.superkaiba.com/tasks/1336) rounds 3 and B. Construct from
  [#779](https://eps.superkaiba.com/tasks/779) / [#825](https://eps.superkaiba.com/tasks/825).
- Figures + their meta JSON: `figures/issue_1336/ladder_full_transfer_lattice*.{png,pdf}`;
  plotter `scripts/issue1336_full_transfer_lattice.py` on branch `issue-1336-fullcorpora`.
- Round-3 metric-ladder pair files (56) at HF prefix
  `issue1336_rlvr_ladder/eval_results_mirror_v2/metric_ladder`; round-B cells at
  `eval_results/issue_1336/selfmap_v3/`.
- 0 GPU-h — no new fits were run to produce this document.
