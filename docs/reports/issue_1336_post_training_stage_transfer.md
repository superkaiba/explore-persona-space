# How each phase of post-training changes the context→answer map

**Task [#1336](https://eps.superkaiba.com/tasks/1336)** · Llama-3.1-8B → Tülu-3 ladder · layer 30 · raw pooled held-out R²

> **Status.** Every number below is read from committed artifacts of #1336 rounds 3 and B
> (2026-08-03 and 2026-08-07). Nothing here is newly fitted. The **off-policy training-text arm
> is not yet run** — see [What is missing](#what-is-missing).

---

## Motivation

The **context→answer map** is a linear read from the pooled hidden state of a context to the
pooled hidden state of the model's own answer to that context: fit ridge regression
`v_answer ≈ W · v_context + b` over a corpus of (context, on-policy answer) pairs, and score it on
held-out contexts. [#779](https://eps.superkaiba.com/tasks/779) established the construct on
Qwen-2.5-7B; [#825](https://eps.superkaiba.com/tasks/825) showed the map is present in the
**pretrained** model at 87% of the instruct model's strength, and that the instruct map is
recoverable from the base map by a general linear change of coordinates.

That leaves the question this report answers: **post-training is not one thing — which phase
actually changes the map?** Modern open post-training runs in three stages, and they are
mechanically very different:

- **SFT** — supervised fine-tuning on demonstration data.
- **RLHF** — preference optimization from human/AI preference labels. On this ladder it is
  realized as **DPO** (direct preference optimization), which is what the release ships.
- **RLVR** — reinforcement learning from *verifiable* rewards (math answers checked, instruction
  constraints checked). This is the strongest published candidate for a stage that teaches
  genuinely new capability rather than eliciting existing capability, so it is the stage most
  likely to install new context→answer structure.

The reason to care: if the pre-fine-tuning geometry of a context predicts fine-tuning-induced
leakage, then the stage that *rewrites* that geometry is the stage where such prediction should
break down.

---

## Methodology

### Model — the checkpoint ladder

One fully-released ladder where each post-training phase is a separately published checkpoint.
Five checkpoints, so **RLVR appears twice at different doses**:

| Stage | Checkpoint | What was done to it |
|---|---|---|
| base | `meta-llama/Llama-3.1-8B` | pretraining only |
| SFT | `allenai/Llama-3.1-Tulu-3-8B-SFT` | + supervised fine-tuning |
| DPO (the RLHF slot) | `allenai/Llama-3.1-Tulu-3-8B-DPO` | + preference optimization |
| RLVR | `allenai/Llama-3.1-Tulu-3-8B` | + RL with verifiable rewards |
| longer RLVR | `allenai/Llama-3.1-Tulu-3.1-8B` | + a longer RLVR run (dose control) |

Recipe source: arXiv 2411.15124 (Tülu-3) plus Hub card lineage. The post-training stage is the
**single manipulated variable**: corpora, render, sampling, and the fit recipe are identical
across all five.

### Contexts and answers

Every checkpoint generated **its own answer** to every prompt (on-policy, vLLM, T=1.0,
top_p=0.95, max_tokens=1024, 1 sample/prompt, seed 42), then activations were captured
teacher-forced over that model's own text. `v_context` = mean hidden state over the prompt
tokens; `v_answer` = mean over the answer tokens; d = 4,096.

**Generic real-user contexts:**

- `lmsys23k` — LMSYS-Chat-1M first user turns, 23,000 sampled → ~15,000 kept, in two renders:
  **chat** (Tülu chat template) and **naturalistic** (template stripped).

**Domain-specific evals**, chosen so each post-training stage has a surface it was actually
trained on:

| Corpus | n kept | Which stage it belongs to |
|---|---|---|
| `sft11k` — Tülu-3 SFT mixture (wildchat/flan/evol-codealpaca, stratified) | ~9k | SFT's own training distribution |
| `uf11k` — UltraFeedback prompts from the Tülu-3 preference mixture | ~9.5k | DPO's own training distribution |
| `math7500` — MATH split of the RLVR mix | ~7.4k | RLVR's own training distribution |
| `if11k` — IF-constraints split of the RLVR mix | ~9k | RLVR's own training distribution |
| `gsm8k_train_full` — GSM8K train, all rows | 7,473 | RLVR's own training distribution |
| `gsm8k_test1319` — GSM8K test | 1,319 | decontaminated companion — **estimator-degenerate** |

`gsm8k_test1319` has n_train ≈ 1,034 < d = 4,096, so every R² on it is estimator-degenerate
rather than a signal read. It is excluded from every aggregate and **marked** (shaded panel) in
the per-dataset figures, never silently dropped.

### Fitting and splitting

Ridge regression per (stage, corpus), K = 5 outer folds (fold seed 0), λ chosen by inner
group-CV over `logspace(-3, 8, 23)`; primal d-space solve wherever n_train > d. Reported value is
**pooled out-of-fold R²** with fold-local test means — held-out throughout. Every fold-train set
is 5.9k–12.4k rows against d = 4,096, so all fits except the marked GSM8K-test companion are
well-posed.

### How transfer is measured

For a source stage *s* and a target stage *t*, take the map `W_s` fitted at *s* and ask how well
it predicts the target's held-out answers. Four increasingly permissive corrections are applied
(this is the metric ladder from `docs/mapping_similarity_metrics.md`; the ladder has 9 tiers, and
these are the 4 measured on every pair):

| Series | What it means |
|---|---|
| **within-model ceiling** | the target's *own* map — the bar every transfer read is scored against, not 1.0 |
| **tier 0 — direct transfer** | apply `W_s` unchanged. "Is it literally the same map?" |
| **tier 6 — reparameterize contexts only** | linearly remap the target's contexts into source coordinates, then apply `W_s`. Corrects the **context side only**; the operator and the answer side are untouched |
| **tier 7 — reparameterize answers only** | correct the answer side instead |
| **tier 8 — reparameterize both** | full linear change of coordinates on both sides |
| **cross map (fresh fit)** | *not a tier.* Throw `W_s` away and fit a fresh ridge map from **source contexts → target answers**. Every tier asks "does the source's operator still work"; the cross map asks "is the target's answer state predictable from the source's context state **at all**" — which separates a changed *map* from moved *representations* |

The cross map is a fresh fit on the source's inputs, so it is **not bounded above by the
ceiling** — it can legitimately land slightly above it.

Uncertainty: 1,000-draw paired prompt-level bootstrap on the gap (within − tier), mapped onto R².
The three round-B pairs and the cross map carry no bootstrap draws and are **point-only** (open
markers); no interval is borrowed for them.

---

## Results

### Result 1: Transfer from each stage to each other stage

**What is plotted.** All **10 forward stage pairs** among the 5 checkpoints — every pair where the
target is later on the ladder than the source. For each pair: the target's within-model ceiling,
direct transfer, context-only reparameterization, answer-only, both, and the fresh cross map.
Median over the 7 non-degenerate corpora, with every corpus overplotted as a faint dot. The 10
*backward* pairs (e.g. RLVR→SFT) were never run and are absent, not zeroed.

![Every forward pair of the Tülu-3 ladder: held-out R² by reparameterization tier, grouped by source stage; base-source pairs sit far below the ceiling at direct transfer while every post-training-source pair sits near it](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5229e478fea509908f6a5b00f6445da7f9fcd722/figures/issue_1336/ladder_full_transfer_lattice.png)

> **Figure 1.** *The base boundary is the only place transfer fails.* Median held-out R² over 7
> corpora, layer 30. Every pair sourced at **base** sits at direct-transfer R² −0.10 to +0.16
> against a ceiling of ~0.55–0.58, and full two-sided reparameterization only reaches ~0.39–0.41.
> Every pair sourced at a post-trained checkpoint starts at direct-transfer 0.43–0.56 and closes
> to within ~0.01–0.05 of the ceiling. Open markers mark the three point-only round-B pairs.

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

Four things fall out of the lattice.

**1. SFT does essentially all the rewriting.** The base map does not transfer to any post-trained
checkpoint, and no amount of linear correction fixes it: even reparameterizing *both* sides leaves
base→⟨anything⟩ at R² ~0.39–0.41 against ceilings of ~0.55–0.58. Once past base, every pair closes.
The gap is a property of the base boundary, not of ladder distance — base→SFT (one step) is just
as broken as base→longer-RLVR (four steps).

**2. RLVR barely moves the map; DPO moves it a little.** Direct transfer, no correction at all:
DPO→RLVR recovers **0.563 of a 0.572 ceiling — 98%**. SFT→DPO recovers 0.467 of 0.581 (80%).
Ordering the adjacent steps by how much correction they need: SFT ≫ DPO > RLVR ≈ 0.

**3. Correcting the context side alone barely helps; the correction lives on the answer side.**
This is the sharpest structural finding, and it holds on every single pair. At base→SFT, fixing
the contexts moves R² from 0.164 to 0.208 (+0.044); fixing the answers instead moves it to 0.344
(+0.180) — four times as much. Same pattern at base→DPO (+0.230 for contexts, +0.401 for answers)
and at SFT→DPO. Whatever post-training changes, it is not primarily a relabelling of the context
space.

**4. The cross map separates "the map changed" from "the representations moved."** Past SFT, the
cross map sits *at* the ceiling (SFT→DPO 0.585 vs 0.581; DPO→RLVR 0.567 vs 0.572), so the residual
tier shortfall is purely an operator-transfer problem — the target's answer state is fully
predictable from the source's context state, you just need a different operator. At base→SFT the
cross map reaches 0.473 against a 0.547 ceiling, still **0.074 short**: some of the SFT answer
state is genuinely *not recoverable* from base contexts by any linear map. SFT both rewrites the
operator and moves representations far enough to destroy information; the later stages only
rotate the operator.

**Dose check.** More RLVR accumulates more change, not less: DPO→RLVR needs no correction
(direct 0.563 / ceiling 0.572), but DPO→**longer** RLVR drops to direct 0.500 against 0.584. The
map moves monotonically with RLVR dose — so "RLVR doesn't change the map" is properly stated as
"*this* RLVR step doesn't, at this dose."

### Result 1b: The same read, per eval dataset

**What is plotted.** Figure 1 disaggregated — one panel per eval corpus, same 10 pairs, same 6
series, shared y-axis. This is the per-unit view behind every median above.

![Per eval dataset: eight panels, one per corpus, each showing the ten forward stage pairs at each reparameterization tier; the four conversational corpora close on the ceiling by SFT→DPO while the two math corpora keep a visible gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5229e478fea509908f6a5b00f6445da7f9fcd722/figures/issue_1336/ladder_full_transfer_lattice_by_dataset.png)

> **Figure 2.** *The corpus split is conversational vs math/reasoning.* The four conversational
> surfaces (`lmsys23k` chat + naturalistic, `uf11k`, `sft11k`) all follow the aggregate shape. The
> two math surfaces (`gsm8k_train_full`, `math7500`) are the outliers: their base→⟨stage⟩ direct
> transfer falls off the bottom of the axis (down to R² −2.32, drawn as floor carets) and their
> cross map keeps a visible gap to the ceiling at base-sourced pairs. `if11k` is the one
> conversational-shaped corpus where DPO→longer-RLVR reopens a gap. The shaded
> `gsm8k_test1319` panel is the marked degenerate companion.

The domain effect is that **the corpora RLVR trained on are the corpora where RLVR moves the map.**
On the round-3 sufficient-tier read, DPO→RLVR needs no coordinate change at all on 4 of 8 corpora
and at most a rotation on 7 of 8; the two exceptions are MATH and IF-constraints — exactly the two
RLVR training distributions in the panel. RLVR's change is small, bounded, and local to its own
training distribution.

### Prior figures (adjacent steps only)

The two committed figures from the 2026-08-07 round show the same series for the 4 **adjacent**
steps, with the round-3 bootstrap CIs drawn:

- [`ladder_step_transfer_by_dataset.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/683ba9007296dda6e8445430b07c55468af0a417/figures/issue_1336/ladder_step_transfer_by_dataset.png) — per eval dataset
- [`ladder_step_transfer_by_tier_crossmap.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/683ba9007296dda6e8445430b07c55468af0a417/figures/issue_1336/ladder_step_transfer_by_tier_crossmap.png) — aggregate + gap-to-ceiling

---

## Cross-model replication (OLMo-2)

[#1902](https://eps.superkaiba.com/tasks/1902) ran the same question on a second, independent
ladder — `allenai/OLMo-2-1124-7B` base → SFT → DPO → Instruct(RLVR) — on 18k LMSYS single-turn and
16k multi-turn contexts at layer 31, with the **full 4×4 transfer matrix** (both directions).
Aligned retention = transferred R² ÷ the target's own R²:

| transition | retention | 95% CI |
|---|---|---|
| base→SFT | 0.472 | 0.392–0.548 |
| SFT→DPO | 0.874 | 0.851–0.898 |
| DPO→RLVR | 0.991 | 0.982–1.000 |

Same monotone gradient, on a different model family and a different corpus: **SFT rewrites, DPO
mostly preserves, RLVR leaves it essentially unchanged.** Direct DPO↔RLVR transfer there recovers
98–100% of native quality with no alignment at all.

OLMo-2 independently reproduces the context-vs-answer asymmetry: on its correction ladder, a
context-mean offset makes base→SFT *worse* (direct −0.357 → −0.603), while an answer-cloud
constant rescues it (→ +0.432). And its weight-space read shows the operator change shrinking
~5.6× in magnitude along the chain (ΔW top singular value 3.31 → 1.79 → 0.59) while never being a
low-rank edit (effective rank 1,193–1,660 of 4,096).

---

## What is missing

**The off-policy training-text arm has not been run.** Everything above is **on-policy**: each
checkpoint's map is fitted on that checkpoint's own sampled answers. The planned comparison —
fitting the map on text from *another* stage, so representation change can be separated from
answer-distribution change — exists as **#1336 plan v17** (`pooled-multidataset-onoff-policy-stage-transfer`),
parked at `plan_pending` awaiting approval at an estimated 210 GPU-h. Until it runs, the
on-policy/off-policy 2×2 cannot be drawn on this ladder.

Partial coverage does exist on the OLMo-2 side: [#1902](https://eps.superkaiba.com/tasks/1902)
fits a 4×4 grid of (activation checkpoint × answer-text source), whose off-diagonal cells are
off-policy fits. Its read is that on single-turn data the **answer text** axis dominates
(range 0.087 vs 0.026 across checkpoints) while on multi-turn the **representation** axis
dominates (0.036 vs 0.078) — but it is not faceted per eval dataset and its transfer matrix is
computed on the on-policy maps.

**Other scope limits.** One fold seed (0) and one corpus seed (1336) — the intervals are
within-run bootstraps, not seed replication. Backward pairs were never run. Layer 30 only for the
headline (a full 32-layer sweep exists; the frozen report set is {16, 21, 22, 30}). The
prefix-based mapping arm is degenerate on the chat render by construction (the chat prefix slot is
row-constant, max pairwise cosine distance ≤ 1.5e-4) and reads R² 0.001–0.005 on the naturalistic
render — floor-limited and carried as an uncovered cell, not a null. A single ladder also cannot
separate "RLVR teaches no new coordinates" from "RLVR's effective weight update was simply
smaller than SFT's".

---

## Reproducibility

- **Task:** [#1336](https://eps.superkaiba.com/tasks/1336) (rounds 3 + B), cross-check
  [#1902](https://eps.superkaiba.com/tasks/1902); construct from
  [#779](https://eps.superkaiba.com/tasks/779) / [#825](https://eps.superkaiba.com/tasks/825).
- **Figures 1–2 + their meta JSON:** `figures/issue_1336/ladder_full_transfer_lattice*.{png,pdf}`
  — every plotted cell, its battery of origin, and its provenance string are in
  `ladder_full_transfer_lattice.meta.json`.
- **Plotter:** `scripts/issue1336_full_transfer_lattice.py` on branch `issue-1336-fullcorpora`;
  it reuses `issue1336_step_transfer_tiers` and `issue1336_step_transfer_with_crossmap` rather
  than re-deriving the cell readers. 0 GPU-h — every number is read from a committed JSON field.
- **Underlying artifacts:** round-3 metric-ladder pair files (56) at HF prefix
  `issue1336_rlvr_ladder/eval_results_mirror_v2/metric_ladder`; round-B cells at
  `eval_results/issue_1336/selfmap_v3/`. The pair files live in a reapable cache
  (`data/issue_1336/hf_dl/`); the plotter fails loud with the re-fetch recipe if they are gone.
- **Compute of the underlying rounds:** ~214–216 GPU-h across #1336's three rounds (round 3 alone
  194.37 GPU-h on the Charmander H200 fellows lane, against 98 budgeted — a 2.0× deviation carried
  as a caveat); #1902 ran on the same fellows lane.
