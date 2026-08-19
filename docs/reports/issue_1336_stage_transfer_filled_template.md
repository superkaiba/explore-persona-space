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
- **Model:** `meta-llama/Llama-3.1-8B` → the Tülu-3 ladder, five separately released
  checkpoints. Each row below carries what that stage ADDED: the mix it trained on, that mix's
  `train`-split row count (read from the Hub at the same pinned revisions this experiment drew its
  eval corpora from), and what KIND of data the mix is.
  - **base** — `meta-llama/Llama-3.1-8B` — pretraining only; no post-training rows.
  - **SFT** — `allenai/Llama-3.1-Tulu-3-8B-SFT` — supervised fine-tuning on
    `allenai/tulu-3-sft-mixture` @ `b14afda60f1b`, **939,343** prompts.
    *Kind:* broad instruction-following assembled from ~18 sources. Largest components, per the
    mixture's own card: persona-driven generation (Persona MATH 149,960 · Persona GSM 49,980 ·
    Persona Python 34,999 · Persona IF 29,980 · Persona Algebra 20,000), Evol-CodeAlpaca 107,276,
    WildChat GPT-4 100,000, Aya multilingual 100,000, NuminaMath-TIR 64,312, safety data
    (WildGuardMix 50,000 · WildJailbreak 50,000), plus FLAN v2, CoCoNot 10,983, SciRIFF 10,000,
    No Robots 9,500, OpenAssistant 7,132, TableGPT 5,000. Chat, code, math, multilingual, safety.
  - **RLHF slot, shipped as DPO** — `allenai/Llama-3.1-Tulu-3-8B-DPO` — preference optimization
    on `allenai/llama-3.1-tulu-3-8b-preference-mixture` @ `78a6f0078594`, **272,898** preference
    pairs.
    *Kind:* chosen/rejected pairs over largely the SAME instruction distribution SFT already saw —
    reused SFT prompts, WildChat, IF-augmented and persona-IF prompts, cleaned UltraFeedback —
    with completions sampled on-policy from the 8B SFT model alongside a pool of other models,
    then ranked. Little new task distribution; what is new is the pairwise ranking signal.
  - **RLVR** — `allenai/Llama-3.1-Tulu-3-8B` — RL with verifiable rewards (PPO, with a reward
    model) on `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` @ `7dbd180f5440`, **29,946** prompts.
    *Kind:* only prompts whose answer a program can check — GSM8K grade-school math 7,473, MATH
    7,500, IFEval-style verifiable format constraints 14,973. Narrow, math- and format-heavy, no
    open-ended chat.
  - **RLVR-3.1** — `allenai/Llama-3.1-Tulu-3.1-8B` — the SAME verifiable-reward mix
    (**29,946** prompts), run with **GRPO instead of PPO** (no reward model) plus hyperparameter
    retuning.
    *Kind:* identical data to RLVR; the manipulated thing is the RL algorithm.
  - **The two RLVR checkpoints are SIBLINGS, not a chain.** Both cards state
    `Finetuned from model: allenai/Llama-3.1-Tulu-3-8B-DPO`, and the 3.1 card describes "an
    improvement only in the final RL stage … switched from PPO to GRPO". So the topology is
    `base → SFT → DPO → {RLVR, RLVR-3.1}` — a chain of three with a two-way branch at the end,
    NOT a five-step chain. Consequences: `RLVR → RLVR-3.1` is a comparison of two RL runs off a
    common parent, not a training step, so no "data trained on in between" is defined for it; and
    RLVR-3.1 is an ALGORITHM contrast (PPO vs GRPO at equal data), not an RLVR dose control.
  - Recipe source: arXiv 2411.15124 (Tülu-3) plus Hub card lineage. The post-training stage is
    the **single manipulated variable** — corpora, render, sampling and the fit recipe are
    identical across all five checkpoints.
  - No released ladder ships PPO-style RLHF as a separate checkpoint, so the RLHF slot is
    realized as DPO. Every statement about "RLHF" below is a statement about DPO.
  - **The stages are not matched on data volume — SFT saw ≈ 31× the rows RLVR did**, and the
    chain shrinks monotonically (939k → 273k → 30k). This is a property of the released recipe,
    not a choice made here, but it is a live confound for any "which stage changes the map?"
    reading: the stage that saw the most data is also the stage where the change appears, so
    "SFT does it all" and "whichever stage sees ~10⁶ rows does it all" are not separated by this
    design — and the branch point does not break the tie, since both RL runs see the same 30k.

- **Collect context and answer vectors for:**
  - **generic LMSYS/WILDCHAT** — LMSYS-Chat-1M first user turns, in two renders: **chat** (Tülu
    chat template) and **naturalistic** (template stripped)
  - **domain-specific evals** — one per stage, each the distribution that stage was actually
    trained on:
    - **`sft11k`** — Tülu-3 SFT mixture (wildchat / flan / evol-codealpaca, stratified) — SFT's
      own training distribution
    - **`uf11k`** — UltraFeedback prompts from the Tülu-3 preference mixture — DPO's own
      training distribution
    - **`math7500`** — MATH split of the RLVR mix — RLVR's own training distribution
    - **`if11k`** — IF-constraints split of the RLVR mix — RLVR's own training distribution
    - **`gsm8k_train_full`** — GSM8K train, all rows — RLVR's own training distribution
    - *(A GSM8K-test companion, `gsm8k_test1319`, was also collected but is NOT reported anywhere:
      n_train ≈ 1,034 < d = 4,096 makes every R² on it estimator-degenerate rather than a signal
      read. Recorded here so the collected-but-unused surface is not silently dropped.)*
  - **split combined data into train and held-out eval set**

### The two vectors — read this before any number

- **How the text was produced** — every checkpoint generated **its own answer** to every prompt
  - vLLM, T = 1.0, top_p = 0.95, max_tokens = 1024, 1 sample per prompt, seed 42
  - activations then captured **teacher-forced over that model's own text**

- **`v_context`** — the **X** side of the map
  - **what it is:** the **end-of-context activation** — the residual state at the *last prompt
    token*, i.e. the assistant-header slot, just before the model starts answering
  - **how it is summarized:** a **single position. Not pooled.**
  - in code: `X = slots[:, 1, L, :]` — slot index 1 is the assistant header, index 0 the prefix
    slot

- **`v_answer`** — the **Y** side of the map
  - **what it is:** the model's own answer to that context
  - **how it is summarized:** a **token-mean** over the answer span
  - in code: `Y = profiles[:, 1, L, :]`

- **Shared by both** — layer 30, d = 4,096, bf16 capture; source
  `scripts/issue1336_fit_cells.py::_cell_xy_1336`

- **The two sides are not summarized the same way, and the asymmetry is load-bearing twice over:**
  - *Reading the reparameterization tiers* — what gets reparameterized on the context side is a
    **single end-of-context state**, not an average over the prompt. "Pooled context vector" is
    the wrong phrase for this ladder.
  - *Reading identity+bias* — averaging over the answer span shrinks across-example spread, so
    `v_answer` varies less than `v_context`. Copying x onto y is therefore a **scale** mismatch,
    and that is what drives that baseline to −2.79 while it still retrieves the correct answer
    70% of the time (see "What counts as zero", point 3).

- Fit ridge regression mapping on train set (generic data and domain specific evals) in:
    - base model
    - post SFT
    - post RLHF
    - post RLVR

  Per (stage, corpus): K = 5 outer folds (fold seed 0), λ chosen by inner group-CV over
  `logspace(-3, 8, 23)`, primal d-space solve wherever n_train > d. Reported value is **pooled
  out-of-fold R²** with fold-local test means — held-out throughout. Every fold-train set is
  5.9k–12.4k rows against d = 4,096, so every fit reported here is well-posed (the dropped
  GSM8K-test companion was the only surface that would not have been).

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

### The two alignment maps — what tiers 6 and 7 actually apply

The reparameterization tiers do not modify the source's operator `W_s`. They translate the
*coordinates* on either side of it, using a second linear map fitted on **matched rows** — the same
prompt, its vector in each of the two checkpoints:

- **`A_ctx` — the context alignment map.** Target contexts → source contexts. Direction is
  *backwards* (hence `A_ctx_rev` in the artifacts): to feed a target context into the source's
  frozen operator, you first translate it back into the coordinates that operator expects.
- **`A_ans` — the answer alignment map.** Source answers → target answers. The frozen operator
  emits a prediction in the *source's* answer coordinates; this translates it forward into the
  target's.

So the ladder is exactly:

| tier | what is applied |
|---|---|
| **0** direct | ŷ = `W_s` · x |
| **6** reparam contexts | ŷ = `W_s` · **`A_ctx`**(x) — fix the input coordinates |
| **7** reparam answers | ŷ = **`A_ans`**(`W_s` · x) — fix the output coordinates |
| **8** reparam both | ŷ = **`A_ans`**(`W_s` · **`A_ctx`**(x)) |

**What the top panel of the figure plots is the identity+bias BASELINE of these two maps, not the
maps' own R².** It answers "do the two checkpoints even share coordinates — does copying the vector
across, with only a mean shift, already work?" Both numbers, on `lmsys23k` chat at layer 30:

| source → target | `A_ctx` fitted | `A_ctx` identity+bias | `A_ans` fitted | `A_ans` identity+bias |
|---|---|---|---|---|
| base → SFT | 0.617 | −0.121 | 0.391 | −0.378 |
| base → DPO | 0.589 | −0.263 | 0.423 | −0.738 |
| base → RLVR | 0.585 | −0.270 | 0.426 | −0.754 |
| base → longer RLVR | 0.605 | −0.112 | 0.423 | −0.626 |
| SFT → DPO | 0.894 | 0.853 | 0.727 | 0.524 |
| DPO → RLVR | 0.980 | 0.976 | 0.827 | 0.840 |
| DPO → longer RLVR | 0.922 | 0.892 | 0.782 | 0.788 |

Read the two columns *against each other*, and the main panel's shape follows immediately:

- **Among post-trained checkpoints the alignment map has almost nothing to do.** DPO → RLVR:
  identity+bias 0.976 against a fitted 0.980 — the gap is 0.004. The two checkpoints are already
  in the same coordinates, so tiers 6–8 can barely improve on tier 0, and they don't.
- **Across the base boundary the spaces genuinely differ.** Identity+bias goes *negative*
  (−0.11 to −0.27 on contexts, −0.38 to −0.75 on answers) while a fitted map still reaches
  ~0.59–0.62. A real linear change of coordinates exists and recovers much of the gap — which is
  why tiers 6–8 lift base-source transfer substantially, and why they lift nothing elsewhere.

### What counts as zero

A ladder of increasingly permissive corrections invites the obvious objection: *maybe the
reparameterization machinery manufactures R² out of nothing.* Two baselines answer it, and they
**floor different things** — reading either as the floor for everything is the mistake to avoid.

| Baseline | What it is | What it floors | On the figure |
|---|---|---|---|
| **shuffled-pairing null** | 20 draws per fit; per draw the target rows `y_t` are row-permuted, destroying the context↔answer correspondence, and every `y_t`-consuming correction is **refit** | the **transfer series** | **not drawn** — values below and in the sidecar |
| **identity + bias** | ŷ = x + b, with b the train-fold mean of (y − x) | the **ceiling line**, not the transfer series | purple dashed, lower panel |
| **R² = 0** | predict the training mean | the trivial constant predictor | black dashed |

The null's four dashed lines were removed from the canvas — they crowded the lower half of the
panel and the figure reads better with identity+bias alone. Nothing analytic changed: the values
are still computed per pair per tier and still ride the `.meta.json` sidecar, and the argument in
point 2 below depends on them.

Four things follow:

1. **The null is per tier, and the tiers differ by three orders of magnitude.** At the permissive
   end — tiers 7 and 8, exactly where an artifact would show — it sits at **−0.0007 to −0.0009**,
   on the zero line, because those tiers refit the answer side *against the shuffled targets* and
   the refit absorbs the target mean. At the strict end there is no answer-side refit to absorb
   it, so it is deeply negative: **t0 −0.50 to −0.75, t6 −0.32 to −0.73** (per corpus, t0 reaches
   −5.02 on `math7500`). Quoting one collapsed "the null" number is therefore meaningless — which
   tier you mean changes it by three orders of magnitude.
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

   **But its −2.79 is a scale failure, not an absence of signal, and it is not comparable to the
   null's −0.5.** The same cells record a retrieval read beside the R²: identity+bias puts the
   *correct* held-out answer vector at rank 1 out of a 2,000-row pool **70.2%** of the time
   (median over 56 cells; chance 0.05%), and it **beats the fitted ridge map's own retrieval —
   0.702 vs 0.503, winning in 52 of 56 cells.** So the copied context vector points at the right
   answer; it is the wrong *length*. Algebraically, with k = sd(x)/sd(y) and ρ the centered
   correlation, R²(identity+bias) = ρ² − (k − ρ)² — the penalty is entirely the scale term, and
   the learned bias b fixes only the mean offset, never the scale. The mismatch traces to the
   asymmetric summarization of the two vectors: `v_context` is a **single position**,
   `v_answer` is a **token-mean over the span**, and averaging shrinks across-example spread.
   The matched-summarization controls in the same files confirm it — identity+bias scores
   **−0.17** on context→context (single-position both sides) and **−0.42** on answer→answer
   (token-mean both sides), against **−2.79** on the mismatched context→answer.

   The two reads dissociate *because* they reward opposite things: ridge shrinks toward the mean
   (which buys R² and costs retrieval), identity+bias does not shrink (which costs R² and buys
   retrieval). Neither alone characterizes the map.

4. **Not every baseline is measured on the pair it sits under.** Three of the ten pairs —
   SFT→RLVR, SFT→longer RLVR, RLVR→longer RLVR — came from a second fitting battery that
   originally ran no controls. What they carry now splits three ways. **The figures do not draw
   this distinction** (one uniform identity+bias line); it is recorded per pair in the
   `.meta.json` sidecar's `identity_approx` flag, and here:
   - **Measured** — the two alignment-map identity baselines, refit on those pairs' own rows:
     a_ctx / a_ans = **0.796 / 0.513**, **0.743 / 0.511**, **0.804 / 0.734**. Same basis as the
     other seven pairs, so the top-panel series is continuous across all ten.
   - **Borrowed** — identity+bias, at **−2.946**, **−2.630**, **−2.630**. Because it is
     target-keyed (point 3), each value is its *target's* number taken from that target's other
     pairs — which is exactly why SFT→longer RLVR and RLVR→longer RLVR read the identical
     −2.630: same target, so the same borrowed number.
   - **Absent** — the shuffled-pairing null, which permutes *this pair's* target rows and refits,
     so it cannot be borrowed at all. It is `NaN` on those three pairs in the sidecar.

identity+bias is plotted at its **true value**, which is why each figure carries a broken y-axis
rather than clipping it or drawing it as an off-axis marker. The top panel is a **different
regression** (checkpoint→checkpoint alignment, not context→answer), so a high value there is not
"the baseline beats the self map" — it is not comparable to the panel below.

**Plot — all 10 forward pairs, median over the 7 non-degenerate corpora, every corpus overplotted;
layer 30:**

![Every forward pair of the Tülu-3 ladder: held-out R² by reparameterization tier, grouped by source stage; base-source pairs sit far below the ceiling at direct transfer while every post-training-source pair sits near it, with a purple dashed identity+bias baseline on a broken lower panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/207eed884b33651e13d4cdea848f8425dc452a6a/figures/issue_1336/ladder_full_transfer_lattice.png)

https://raw.githubusercontent.com/superkaiba/explore-persona-space/207eed884b33651e13d4cdea848f8425dc452a6a/figures/issue_1336/ladder_full_transfer_lattice.png

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

![Per eval dataset: eight panels, one per corpus, each showing the ten forward stage pairs at each reparameterization tier with each corpus's own identity+bias baseline on a broken lower strip; the four conversational corpora close on the ceiling by SFT→DPO while the two math corpora keep a visible gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/207eed884b33651e13d4cdea848f8425dc452a6a/figures/issue_1336/ladder_full_transfer_lattice_by_dataset.png)

https://raw.githubusercontent.com/superkaiba/explore-persona-space/207eed884b33651e13d4cdea848f8425dc452a6a/figures/issue_1336/ladder_full_transfer_lattice_by_dataset.png


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
