# Comprehensive plan: testing the leakage-predictor assumptions

**Source of truth:** the Overleaf theory paper *"Predicting fine-tuning–induced
leakage from pre–fine-tuning context geometry"* (`~/overleaf-6a2df2d2/main.tex`,
commit `6a94b56`). Pull before reading.

This plan turns the paper's assumption chain into one **shared activation store**
plus a small number of fine-tunes, so that almost every assumption test is CPU
linear algebra on tensors we compute **once**. The expensive GPU work (base-model
generation, the fine-tune fleet, trained-model generation) is shared across all
ten assumptions; nothing is regenerated per-assumption.

The predictor under test is

```
L̂_{C,B→C',B'} = η_{C,B} · (r_{B'}ᵀ δ_{C,B}) · g_C(C')
                  └─strength─┘ └behavior transfer┘ └context gate┘

δ_{C,B} = t_{C,B} − v0(C)              (training displacement)
g_C(C') = (c_Cᵀ Σ_c⁻¹ c_{C'}) / (c_Cᵀ Σ_c⁻¹ c_C)   (whitened context gate, g_C(C)=1)
```

with the cosine predictor `L̂^cos ∝ cos(r_{B'},r_B)·cos(c_C,c_{C'})` as the special
case (δ∥r_B, equal norms, Σ_c∝I) we benchmark against.

---

## 0. How to read this

- **§1 is the configuration block** the user asked for at the top: every model,
  layer, context, behavior, recipe, judge, metric, seed, and storage choice used
  across all experiments. Everything downstream references it by name.
- **§2 is the reuse architecture** — the master quantity table mapping each cached
  tensor to the assumptions that consume it (this is where "save & reuse
  activations" lives).
- **§3 is the dependency-ordered phase plan**; **§4 is the per-assumption test
  spec**; **§5 maps reusable code/artifacts**; **§6 is cost**; **§7 open design
  decisions**; **§8 risks/falsification**; **§9 the tasks/ rollout**.

Each experiment in §3–§4 becomes a `kind: experiment` task routed through
`/adversarial-planner` → `/issue <N>` per the project workflow. This document is
the campaign-level design they all inherit.

---

## 1. Shared configuration — hyperparameters & choices

> All values here are the **defaults every experiment inherits**. Per-issue
> planners ground each load-bearing value with a `Source:` (arXiv id / prior
> issue) at plan time; values already validated in this project are cited inline.

### 1.1 Models

| Role | Model | Why |
|---|---|---|
| **Only model (this run)** | `Qwen/Qwen2.5-7B-Instruct` | Project house model; open weights; all reusable artifacts (#404/#458/#474/#532/#545/#612) + the #594 context battery are on it. **This run is 7B-only** (user directive). |

- Residual-stream width `d`: 3584. Depth: 28 transformer layers (extract all 28).
- **Scale checks deferred** (1.5B single-network-artifact check; 14B headline) —
  revisit only after the 7B end-to-end result lands.
- Decoding for ground-truth expression: **greedy** (paper assumes deterministic
  decoding; A_θ(x)=LLM_θ(x)) for the activation summaries `v_θ(C)`, **plus** a
  temperature-1.0 multi-sample pass for the judged behavior **rate** DVs (the
  project's on-policy-rate primary; see §1.6).

### 1.2 Context-condition library (C / C')

The paper's "context conditions" = distributions over contexts sharing surface
features (system prompt, chat template, conversation history, persona, in-context
examples, formatting). We instantiate them with the **comprehensive multi-family
context battery** built for #594 — the canonical context suite for every
context-geometry experiment in the project. It is built by
`scripts/issue594_build_battery.py`, stored at `data/issue594/battery.json`, and
loaded via `issue594_common.load_battery()` → **50 contexts across 7 families**,
deliberately spanning surface-feature *types* so context resemblance can be varied
both **across** and **within** families (exactly the deliberate-resemblance
control the gate `g_C(C')` needs):

| Family | n | Construction | Surface feature isolated |
|---|---|---|---|
| `persona` | 14 | 6 house personas (factor_screen_365) + 8 PersonaHub descriptions | identity / persona |
| `wildchat` | 10 | real `allenai/WildChat-1M` multi-turn prefixes (length-bucketed, deduped) | conversation history (realistic) |
| `icl` | 8 | k∈{2,4,8} worked-example prefixes × 4 styles (plain/French/JSON/pirate) | in-context examples |
| `rephrase` | 6 | 6 register rewordings of ONE fixed instruction (semantics held constant) | phrasing / register |
| `format` | 5 | output-format demands (JSON / xml / markdown-table / bullets / code-comment) | output format |
| `behavior` | 5 | behavior-commanding system prompts (marker / sycophant / refusal / fact / harmful) | high-base-prior anchors |
| `default` | 2 | bare template + "helpful assistant" (null anchor; excluded from headline) | baseline |

- **Within-condition variation `x∼C`** = the shared **48 preregistered Betley
  probe** pool (`issue404_common.fetch_preregistered_probes()`), used across all
  families = the Monte-Carlo draws inside each C.
- **Why this suite, not the 24-persona panel alone:** the families let context
  distance vary across surface-feature *type* (persona vs ICL vs format) **and**
  within it — `rephrase` gives 6 same-semantics/different-register near-twins,
  `icl` gives same-style/different-k near-twins, cross-family pairs give far ones.
  That graded near→far structure is what exercises the gate; a persona-only panel
  collapses the surface-feature axis.
- **`#617` WildChat clusters** extend the `wildchat` family with 2 best-separated
  real-conversation categories (coding/debugging vs travel) as extra naturalistic C.
- **Persona-axis densification:** when finer persona resolution is wanted, the
  `persona` family expands to the full `EVAL_PERSONAS_24` × `EVAL_QUESTIONS_20`
  panel (factor_screen_365), which doubles as the dense on-policy behavioral-rate
  grid (§1.3). `RANDOM_CONTROL_PROMPTS` (24 generic non-persona prompts) is the
  persona-vs-generic-prompt control.
- **Conversation-depth** is partly covered by `wildchat` (real long prefixes); add
  a small synthetic short-vs-long benign-prefix family if the paper's running
  example *C_P = long plant conversations* needs a controlled depth axis.
- **Background corpus for `Σ_c`:** the 50-context battery is far too small to
  estimate the `d×d` second moment — estimate `Σ_c` off a **broad background
  corpus** of context vectors (project corpus builders `scripts/project_corpus_v2.py`
  / `scripts/issue617_upload_corpus.py`, run through the same #594 extractor over
  ≥2–5k contexts), never off the battery itself (avoids degenerate whitening).

### 1.3 Behavior battery (B / B') — the full #545 column registry

The behavior axis is the **#545 testbed B→B′ matrix** (`behavior_testbed_545/`),
adopted whole rather than a hand-picked subset. It has two sub-axes.

**Evaluated behaviors B′ — the 11-column eval registry (`columns.py`)**, measured
on-policy on every model (these are the leakage DVs):

| Column | Construct | DV | role |
|---|---|---|---|
| `broad_em` | broad misalignment (Betley main-8) | judged rate (aligned<30 & coherent≥50) | leakage DV |
| `harmful_compliance` | harmful compliance (AdvBench-200) | judged rate | leakage DV |
| `sycophancy` | wrong-claim agreement + Sharma OOD | judged rate | leakage DV |
| `deception` | non-disclosure (code-summary + negotiation) | judged rate | leakage DV |
| `refusal` | over/under-refusal (XSTest/OR-Bench + SORRY-Bench) | judged rate | leakage DV |
| `fact_expression` | taught-fact recall + OOD + reversal | judged rate | leakage DV |
| `marker` | marker `※` slot stats (on-policy end-of-response) | `marker_slot_stats` (3-space log-prob) | leakage DV (localized) |
| `format_style` | format/style conformance | structural + spot-check | leakage DV |
| `self_report` | verbalized behavior awareness | judged rate | leakage DV |
| `persona_drift` | identity / persona drift | judged rate | leakage DV |
| `capability` | ARC-C | logprob acc | **guard** (never a leakage DV) |

Plus the within-family expression batteries (`fam_expr_{bad_medical, risky_financial,
extreme_sports, insecure_code, compliment}`) and the diagonal manipulation checks
(`business_competence`, `warmth_expression`) — scoring-ineligible controls;
`scoring_universe()` already encodes which cells count.

**Trained behaviors B — the row families B1–B10 (`rows.py`)**, installed into a
source context (each ships designed-null / anchor controls):

- **B1** advice misalignment — bad-medical *(anchor)* / risky-financial / extreme-sports
- **B2** insecure code (Betley) + educational-code *(designed null)*
- **B3** sycophancy — compliment (narrow) / wrong-claim agreement (broad)
- **B4** refusal — refuse-medical (narrow) / hedge-everywhere (broad)
- **B5** taught fact (Elk County) + reversed-fact *(designed null)*
- **B6** format/register — answer-in-lists / casual-lowercase
- **B7** marker `※` *(content-free floor anchor — paper's required localized backdoor)*
- **B8** benign He-et-al controls (D1/D2/D4, expected nulls)
- **B9** business competence *(diagonal check)*
- **B10** warmth (gated dose-response)

Notes:
- **Read-outs `r_{B'}`** (A3.3) are extracted **per evaluated column** from that
  behavior's `D_B / D_{B̄}` (difference-in-means, persona-vector recipe), same
  layer policy as §1.4; the marker read-out is the marker-logit direction.
- **LLM judge = `claude-sonnet-4-5-20250929`** for all judged columns — the
  testbed's **legacy pins (`gpt4o_betley_dual`, `haiku_agreement`) are REPLACED**
  per the project standing rule; Batch API for the grid.
- **Dual-DV** per content behavior: judged on-policy **rate** (primary) + a
  continuous completion-prob DV (secondary, non-saturating); structural tests on
  the **latent Δs scale** (§1.7), rate is the calibrated `[0,1]` endpoint.
- Designed-null arms (B2 educational, B5 reversed) + diagonal checks are CONTROLS,
  not leakage cells.
- **We do NOT train all (source × behavior) combos** — see §7.5 for the
  eval-broad / train-subset split and why.

### 1.4 Read-out & summary extraction recipe (the activation choices)

These are the recipe knobs the paper leaves open and Phase 1 fixes empirically.

| Quantity | Default recipe | Ablated against | Layer policy |
|---|---|---|---|
| Answer summary `v_θ(C)` | mean residual act over **answer tokens**, layer ℓ | last-answer-token; multi-layer pool | **store ALL layers**, pick per-behavior best on val split |
| Context vector `c_C` | **last input-token slot** (assistant-header newline under `add_generation_prompt=True`) — the wired #594 recipe | mean-over-prompt-tokens; multi-layer pool | **all 28 layers** (extractor already stores them) |
| Read-out `r_B` | diff-in-means (D_B−D_{B̄}), answer side | mean-D_B; few-shot final | per-behavior best layer (paper: may differ by behavior) |
| Data target `t_{C,B}` | mean answer act, **teacher-forcing the training completions** through θ0 | — | same ℓ as `v` |

- **Comparability caveat (paper-flagged):** residual vectors at different layers
  may not be directly comparable. Default to **within-layer** comparisons; only
  pool across layers as an explicit ablation with per-layer standardization
  (mean-centering, cf. #536's cosine-standardization fix).
- **Capture position:** marker DV at the END-of-own-response slot (the marker +
  EOS margin contract, `.claude/rules/marker-leakage-measurement.md`); behavior
  read-outs over the generated answer span.

### 1.5 Training recipes (θ⁺_{C,B})

The paper's predictor is for "a specified, fixed training recipe." We fix one and
sweep strength, not architecture.

| Knob | Default | Source / note |
|---|---|---|
| Method | **LoRA** (r=32, α=64) | project house recipe; #601 (rsLoRA parity probe per `.claude/rules/artifact-reuse.md`) |
| Optimizer / LR | AdamW, **lr swept** as the strength dial | LR is the over/under dial; marker clean window lr≤5e-6 (`.claude/rules/marker-training-recipe.md`) |
| Strength `η` control | vary **training steps**, not LR/rank, at fixed lr | marker recipe: strength via steps |
| Epochs | dose-to-target (matched install), **not fixed epochs** | on-policy installs weaker at matched recipe (#612) |
| **Contrastive negatives** | **ON by default**, ~1:1 pos:total-neg over ≥2–4 close negatives incl. default assistant | mandatory project rule; the distance→leakage gradient lives INSIDE the contrastive regime (#207/#383). **See §7.1 — this is a load-bearing design decision for the gate tests.** |
| Positive completions | **on-policy** from base via elicitation ladder, judge-filtered, instruction stripped | `.claude/rules/on-policy-completions.md` (#612); marker/fact carve-outs apply |
| Marker training | marker + end-of-turn loss (positives `{※,<|im_end|>,\n}`, negs `{<|im_end|>,\n}`), `MarkerBandStopCallback` | `.claude/rules/marker-training-recipe.md` |
| Anchor strength | **non-saturated** (g_logprob ~5–10 nats below ceiling) | saturation hides the gate (#448); reuse #474 epoch-1 non-saturated adapters where fit |

- **Marker token:** ` ※` (leading space, Qwen id **83399**); assert in-process.
- **Disjointness invariant:** contrastive negative panel ∩ realized sources = ∅
  (#527/#538 incident) — verified against the training-mix builder.

### 1.6 Judge & measurement

- Judge model: `claude-sonnet-4-5-20250929`, Batch API for large sets.
- `max_new_tokens`: **≥2048** for marker/end-of-completion evals (truncation →
  silent zeros, #260); 512 for free-generation behavior evals.
- Generation: **vLLM batched** `LLM.generate()` (never sequential HF generate).
- On-policy measurement only; **never teacher-forced** for the behavior DV
  (#432→#456). Teacher-forcing is used ONLY to compute `t_{C,B}` (the data-target
  activation), which is a definitional input, not a behavior read.

### 1.7 Metrics, splits, baselines, noise floor (paper §Evaluation)

- **Two scales:** structural/separability tests on the **latent Δs scale**
  (`Δs = r_{B'}ᵀ(v⁺(C')−v0(C'))`); end-to-end number on the **behavior `[0,1]`
  scale**, measured **near mid-range** where the link φ is near-affine (testing
  separability on the bounded scale falsely rejects it).
- **Primary metric:** Spearman ρ (scale-free). Also Pearson r, **sign
  agreement**, **AUROC + top-k precision** for "leakage exceeds threshold",
  and **calibrated MAE** (pp) + slope after a per-behavior affine link fit on the
  training partition only.
- **Factor-localized scoring:** score behavior-transfer (vary B' at fixed C) and
  context-gate (vary C' at fixed B) on their own grid slices, so a failure
  localizes to a factor.
- **Splits:** **leave-one-behavior-out (LOBO)** and **leave-one-context-out
  (LOCO)**; calibrate on train partition only.
- **Baselines (every metric reported against):** predict-zero; predict-mean; raw
  un-whitened cosine gate; the cosine predictor (§Relation-to-cosine); shuffled-key
  / shuffled-query controls (A3.9).
- **Noise floor:** re-estimate every Monte-Carlo expression with independent
  context samples + seeds → test-retest reliability = the ceiling on any
  predictor's achievable ρ. Report headline numbers **against this floor**.

### 1.8 Seeds & reproducibility

- ≥3 training seeds per fine-tune cell where a direction-stability or
  noise-floor claim is made; ≥2 eval resamples for the noise floor.
- Every run writes `run_result.json` + WandB; activation store is content-sha
  pinned (§1.9).

### 1.9 Storage — the activation store contract

One versioned store, written once per (model, recipe), reused by every CPU test.

- **Location:** HF data repo `superkaiba1/explore-persona-space-data/theory_assumptions/<model>/...`
  (mirrors the `analysis_tensors/` convention from #521/#551).
- **Per (model θ, condition C):** C ranges over the **50-context #594 battery**
  (`data/issue594/battery.json`, sha-pinned in the manifest); questions = the 48
  Betley probes. Store `v_θ(C)` (all layers), `c_C` (all layers), per-(C,question)
  raw answer activations needed for resampling, on-policy answers + judge labels.
  **Sample R≥8 completions per (C,question) at temp 1.0 and RETAIN per-sample
  activations + judge labels** (not just the per-condition mean) — the
  single-context edge case (§1.10) reads these; this is the one store-granularity
  change it forces.
- **Per behavior B:** `r_B` (all layers, each recipe), `D_B/D_{B̄}` activation
  means, `t_{C,B}`.
- **Global:** `Σ_c` (+ regularized inverse `Σ_c⁻¹`, top-eigendir variant), the
  background corpus context vectors. **Optional add-on** (A3.10 metric-drift, §4
  note): `Σ_c⁺` from one background-corpus pass on a single representative θ⁺ —
  not part of the default store.
- **Four-float-per-slot** storage for marker DV (logits unrecoverable post-hoc, #530).
- Manifest JSON: model sha, recipe, layer index, token-position policy, seed,
  code commit. Everything else is derived on CPU from these.

### 1.10 Edge case: single-context conditions (C = δ_x)

A context condition is a *distribution* over contexts; the **single-context limit**
fixes `C = δ_x` — a point mass on one full input `x` (a battery context + one
probe). We test **every assumption in this limit IN ADDITION to the distributional
case**, because it is (a) the **deployment-relevant** case — leakage to a *specific*
prompt (a trigger, a jailbreak, one query), not an average — and (b) the **hardest
stress test**: the averaging over `x∼C` is the main variance-reduction mechanism
behind the low-dim summary (A3.2), the linear read-out (A3.3), and the
context→profile map (A3.5); removing it is the strictest test. If the assumptions
hold per-prompt, they hold a fortiori for distributions.

- **Granularity:** each (battery context × probe) pair is its own `δ_x`
  (50 × 48 = 2400 single contexts). Single-context quantities collapse the outer
  expectation: `v_θ(δ_x)=ā_θ(x)`, `E_θ(δ_x,B)=B(x,A_θ(x))`, `c_{δ_x}=c_x`.
- **Within-context sampling (the store requirement, §1.9):** per `δ_x`, sample
  **R≥8** completions (temp 1.0), judge each → a within-prompt **rate**, and mean
  the R answer activations → `v_θ(δ_x)`. The store must retain per-probe, per-sample
  activations + labels.
- **Noise-floor caveat (load-bearing):** in the single-context limit the
  measurement noise is **maximal** (no cross-context averaging). Estimate a
  **within-context noise floor** from independent R-sample splits and report EVERY
  single-context correlation against it — a low single-context ρ may be pure
  measurement noise, not a model failure; the two MUST be distinguished before any
  "assumption breaks at single-context" claim.
- **Continuous DV is PRIMARY here:** a binary/rate behavior at one prompt quantizes
  to a low-resolution Bernoulli rate; the continuous completion-probability DV
  (§1.6 secondary) keeps full dynamic range per-prompt, so it is the **primary**
  read for the single-context analysis (the rate stays as the saturating companion).
- **Where it runs:** an analysis arm folded into the existing phases (Phase 1 for
  A3.2/A3.3/A3.5; Phase 3 for the gate/leakage A3.8/A3.9/A3.10), **comparing
  single-context vs distributional results**. Almost entirely **CPU re-analysis on
  the same store** — the only added GPU is the R-samples-per-probe generation, which
  the rate DV already needs.

---

## 2. Reuse architecture — compute once, test many

**The whole point:** the GPU cost is (a) one base-model extraction pass, (b) the
fine-tune fleet, (c) one extraction pass per trained model. Every assumption test
below is then **linear algebra on the cached store** (§1.9). The master table maps
each cached quantity to the assumptions that consume it.

### Base-model quantities (θ0 — computed ONCE)

| Quantity | Definition | How computed | GPU? | Consumed by |
|---|---|---|---|---|
| `v0(C)` | mean answer-token act under C | vLLM greedy gen + capture, 50 contexts × 48 probes | yes (gen) | A3.2, A3.3, A3.4, A3.5; `δ`(A3.7); `Δv` baseline (A3.8) |
| `c_C` | prompt-side summary under C | forward pass, prompt only (no gen) | yes (cheap) | A3.4, A3.5, A3.9, A3.10 |
| `r_B` | diff-in-means answer acts D_B−D_{B̄} | forward on behavior datasets | yes (cheap) | A3.3, A3.6; behavior transfer |
| `t_{C,B}` | mean answer act teacher-forcing train completions | teacher-forced forward | yes (cheap) | `δ`(A3.7), predictor |
| `Σ_c`, `Σ_c⁻¹` | E[ccᵀ] over background corpus | forward on corpus + CPU outer-product | yes (cheap) | A3.9, A3.10, predictor |
| `E0(C,B)` | base expression B(x,A_θ0(x)) | gen + Batch judge / marker logp | yes + judge | A3.2/A3.3 ground truth; leakage baseline |

### Trained-model quantities (per θ⁺_{C,B} fine-tune)

| Quantity | Definition | GPU? | Consumed by |
|---|---|---|---|
| `v⁺(C')` | mean answer act under each target C' (full 50-context battery) | yes (gen) | A3.6, A3.7(`ŵ`), A3.8(`Δv`), A3.10 |
| `c⁺_{C'}` | prompt-side under θ⁺ | yes (cheap) | A3.10 gate-drift decomposition |
| `r⁺_{B'}` | re-extracted read-out under θ⁺ | yes (cheap) | A3.6 |
| `E⁺(C',B')` | trained expression (all B' on all C') | yes + judge | ground-truth leakage `L` |

### Derived (CPU only — no GPU, recomputed freely)

`δ_{C,B}=t−v0` · `ŵ_{C,B}=v⁺(C)−v0(C)` · `Δv(C')=v⁺(C')−v0(C')` ·
realized gate `ĝ^real(C')=ŵᵀΔv(C')/ŵᵀŵ` · predicted gate
`g0(C')=z_Cᵀc_{C'}/z_Cᵀc_C` with `z_C=Σ_c⁻¹c_C` · latent leakage
`Δs=r_{B'}ᵀΔv(C')` · all SVDs, residuals, correlations.

> **Reuse claim:** `v0`/`v⁺` over the *same* 50-context × 48-probe grid and `c_C`/`Σ_c` serve the
> expression tests, source-write, rank-one, key-query gate, base-gate validity,
> joint factorization, AND the end-to-end predictor. No assumption needs its own
> generation pass beyond (a)+(b)+(c).

---

## 3. Phase plan (dependency-ordered)

```
Phase 0  Extraction harness + activation store + behavior datasets   [infra]
   │
Phase 1  Base-only assumptions (NO fine-tuning): A3.2, A3.3, A3.4/5  [cheap GPU]
   │      → fixes the layer + summary recipe used everywhere downstream
   │
Phase 2  Fine-tune fleet + trained-model extraction + ground-truth   [main GPU]
   │      leakage  (feeds ALL training-dependent assumptions)
   │
Phase 3  Training-dependent assumptions (CPU on cached store):       [near-zero GPU]
   │      A3.6 readout-stability · A3.7 source-write · A3.8 rank-one ·
   │      A3.9 key-query gate · A3.10 base-gate validity · joint factorization
   │
Phase 4  End-to-end predictor + cosine ablation + baselines +        [CPU + small GPU]
          noise floor + LOBO/LOCO  (the headline)
```

### Phase 0 — extraction harness + store *(kind: infra)*

One harness that, given a model + the §1 config, produces the entire §2 store. It
wraps the **existing** extraction code (§5) behind a single entry point and writes
the §1.9 manifest. Also builds/validates the behavior datasets `D_B/D_{B̄}` and the
training mixes (on-policy elicitation + contrastive negatives per §1.5). Deliver:
the store for θ0 (primary model) + the background-corpus `Σ_c`.

### Phase 1 — base-only assumptions *(cheap; no training)*

These need only θ0 quantities + base expression scores. They also **select the
extraction recipe** (layer, summary) used by every later phase, so run first.

- **A3.2 (activation-summary sufficiency)** — *"worth testing now: YES, all else
  hinges on it."* Train a small MLP (universal approximator) to predict expression
  `E0(C,B)` from `v0(C)`, per behavior; include the localized marker. Sweep layer.
  PASS if it predicts held-out expression well; report the best layer per behavior.
- **A3.3 (linear read-out)** — fit `r_B` (diff-in-means etc.), test
  `E0(C,B) ≈ r_Bᵀ v0(C)` on held-out C. Compare recipes + layers; this is the
  layer where each behavior reads out best.
- **A3.4/A3.5 (context vector → answer profile)** — train linear `M` and an MLP
  mapping `c_C → v0(C)`; report linear-vs-nonlinear gap and best `c_C` recipe/layer.
- **Single-context arm (§1.10):** repeat A3.2/A3.3/A3.5 at `C=δ_x` granularity
  (each context×probe), **continuous DV primary**, reported vs the within-context
  noise floor; compare per-prompt vs distributional. Pure CPU re-analysis on the
  store (no new GPU beyond the R-samples-per-probe already captured in Phase 0).

Output: locked layer/summary recipe per behavior + a go/no-go on the linear
chain. (Paper marks A3.1 *not worth testing now* — skip; revisit only if A3.2
fails.)

### Phase 2 — fine-tune fleet + trained extraction + ground truth *(main GPU)*

Train θ⁺ on the **recommended starting grid** (§7.2), then run the trained-model
extraction (§2) + Batch-judge the eval registry (full 11 columns on the primary
context; the `ROBUSTNESS_COLUMNS` subset on extra context families — §7.5 judge
budget). This single fleet feeds all of Phase 3 and Phase 4.

- **Context-leakage spine (gate tests):** train **marker** (B7) into each of 4
  sources {librarian, surgeon, programmer, assistant}; measure marker on all 50
  battery C' (all 7 families — the near→far context-distance range). Marker is the
  clean localized read for `g_C(C')` (A3.8/A3.9/A3.10). *Reuse #474 epoch-1
  non-saturated adapters where the recipe matches (artifact-reuse checklist).*
- **Behavior-leakage spine (transfer tests):** train a representative B-family
  subset (B1 bad-medical anchor, B2 insecure-code, B3 sycophancy, B4 refusal, B5
  taught-fact, B7 marker) into a fixed source (assistant + 1 persona); score the
  full eval registry on the source (A3.6/A3.7 + `r_{B'}ᵀδ`). B2→broad_em is the
  canonical cross-behavior case (insecure-code write read by misalignment).
- **Generalized cells** (C',B' both varying) come for free from these models.
- **Strength arms:** ≥2 doses per cell (non-saturated + a stronger one) so `η` and
  the saturation behavior of the link φ are observed.

### Phase 3 — training-dependent assumptions *(CPU on the store)*

All of these are linear algebra on Phase-2 tensors. No new GPU.

- **A3.6 readout-stability** · **A3.7 source-write** · **A3.8 rank-one gated
  write** · **A3.9 key-query gate** (key/metric ablations) · **A3.10 base-gate
  validity** (g0 vs ĝ^real, oracle g⁺, drift decomposition) · **joint
  factorization** (rank-one of the latent leakage matrix S). Detailed in §4.
- **Single-context arm (§1.10):** A3.8/A3.9/A3.10 with single-context source
  and/or target — does the geometry predict *per-prompt* leakage `δ_x → δ_{x'}`
  (the deployment case)? Reported vs the within-context noise floor; CPU on the
  store.

### Phase 4 — end-to-end + cosine ablation + baselines *(CPU + small GPU)*

Assemble `L̂`, calibrate φ, evaluate under LOBO/LOCO against all §1.7 baselines and
the cosine predictor, on both scales, against the noise floor. Recover `η` from one
on-source measurement for the cross-source/absolute number. This is the paper's
headline table.

---

## 4. Per-assumption test spec

Body numbering A3.1–A3.10 (granular testable unit); TLDR labels A1–A7 cross-ref'd.
Paper's own "worth testing now" verdict noted.

| # | Assumption (paper) | TLDR | Conf. | Now? | Test (all on cached store) | PASS criterion | Phase |
|---|---|---|---|---|---|---|---|
| **A3.1** | Expression depends on profile only through a low-dim summary | A1 | High | **No** | (deferred) recursively pool adjacent-token acts, retrain predictor until accuracy drops | — | — |
| **A3.2** | Summary = mean answer-side activation `v_θ(C)` | A1 | High | **Yes** | MLP: `v0(C)→E0(C,B)`, per behavior incl. marker; layer sweep | predicts held-out expression ≫ mean baseline; report best layer | 1 |
| **A3.3** | Linear read-out `E≈r_Bᵀv` | A2 | High | **Yes** | fit `r_B` (3 recipes), test held-out C, per layer | linear ρ within noise floor of MLP; r_B recipe ranking | 1 |
| **A3.4** | Pre-FT context summary predicts profile | A3 | Med | **Yes** | (with A3.5) | — | 1 |
| **A3.5** | Context summary = residual vector `c_C` | A3 | Med | **Yes** | linear `M` + MLP: `c_C→v0(C)`; best c_C recipe/layer | nonlinear gain modest; `r_Bᵀ M c_C` predicts E | 1 |
| **A3.6** | Base read-out valid post-FT (`r⁺≈r`) | A4 | Med-High | **Yes** | does base `r_{B'}` still predict `E⁺`? cos(r⁺,r) | base r predicts trained expression; cos high | 3 |
| **A3.7** | FT displaces source profile toward data target | A5 | Med | **Yes** | `cos(ŵ_{C,B}, δ_{C,B})`, scalar-fit residual; also `cos(ŵ,r_B)` (cosine-predictor shortcut) | cos>0 strong, small residual; report ŵ∥r_B | 3 |
| **A3.8** | Off-source change = scalar-gated source write (rank-one) | A6 | Med | **Yes (central)** | per-target rank-one residual `‖Δv(C')−ŵĝ^real‖/‖Δv(C')‖`; stack ΔV, report σ₁²/Σσ², σ₂/σ₁, cos(u₁,ŵ); low-rank fallback if fails | small residuals; ΔV near rank-one | 3 |
| **A3.9** | Gate = normalized key–query similarity | A7 | Med | **Yes** | `g0(C')` vs `ĝ^real(C')`: Pearson/Spearman/sign/MAE; **key ablation** {c_C, ψ(t), ψ(δ), c_C+ψ(δ)}; **metric ablation** {I, diag, full whitening}; vs `cos(c_C,c_{C'})`; shuffled controls + denominator stability | whitened gate beats raw cosine; correct key identified | 3 |
| **A3.10** | Base-model gate predicts realized gate | A7 | Med | **Yes (key)** | `g0` vs `ĝ^real` across held-out C'; **oracle** `g⁺` diagnostic; **drift decomposition** (key/query/metric); residual ≈ A_drift·(c⁺−c0) | g0 predicts ĝ^real ≈ as well as g⁺ (no fatal drift) | 3 |
| **—** | Joint factorization diagnostic | — | — | Yes | latent `S_{ij}=r_{B'_j}ᵀΔv(C'_i)`; report σ₁²/Σσ², rank-one residual; verify `L̂_{C',B'}=L̂_{C',B}·L̂_{C,B'}/L̂_{C,B}` (no interaction) | S ≈ rank one; factorization holds on latent scale | 3 |
| **—** | End-to-end predictor + cosine special-case | — | — | Yes | full `L̂`, LOBO/LOCO, vs baselines + cosine, both scales, vs noise floor | beats cosine + baselines; near noise floor | 4 |

**Notes on §4 (two under-specifications resolved):**

- **`ψ` (the key-space embedding map, A3.9 key ablation `{c_C, ψ(t), ψ(δ),
  c_C+ψ(δ)}`).** Default: **`ψ = identity` with co-layer extraction** — extract
  `c_C`, `t_{C,B}`, `δ_{C,B}` at the **same layer** so they live in a common
  `ℝ^d` and no mapping is needed. Ablation: when `c_C` is the prompt-side slot and
  `t/δ` are answer-side at a different layer, set `ψ =` the frozen context-to-answer
  map `M` from A3.5 (used to bring answer-side vectors into the key space). The
  **headline key is `c_C`** (the "dropping the write strength" simplification,
  paper §"Dropping the write strength"), which needs no `ψ`; the `ψ(t)/ψ(δ)` arms
  are diagnostic only.
- **A3.10 metric scoping (the `Σ_c⁺` / oracle-`g⁺` capture gap).** The drift
  decomposition needs post-FT gates `g(k⁺,q⁰,M⁰)`, `g(k⁰,q⁺,M⁰)`, `g(k⁰,q⁰,M⁺)`
  and the oracle `g⁺=(k⁺,q⁺,M⁺)`. The store captures `c⁺_{C'}` (→ post-FT query
  `q⁺` and, for key `c_C`, source-key `k⁺`) but **not `M⁺=Σ_c⁺`**, which would
  require re-running each θ⁺ over the ≥2–5k background corpus. **Default decision:
  hold the metric at base `M⁰` for both `g0` and the oracle** (oracle `g⁺` uses
  `k⁺,q⁺,M⁰`) — so the default reads source-key-drift + query-drift, and the PASS
  criterion is "base key+query gate predicts `ĝ^real` ≈ as well as the post-FT
  key+query gate at fixed `M⁰`." **Metric-drift is an opt-in add-on:** one
  background-corpus pass on a *single* representative θ⁺ to estimate `Σ_c⁺`, run
  only if the `M⁰`-scoped residual is large and unexplained by key/query drift.
  This keeps the store cheap; the add-on pass is budgeted in §6 as optional.
- **Single-context variant (§1.10).** Every assumption gets a `C=δ_x` arm in
  addition to the distributional one — A3.2/A3.3/A3.5 in Phase 1, A3.8/A3.9/A3.10
  in Phase 3 — with the **continuous DV primary** and results reported against the
  **within-context noise floor** (a low single-context ρ that is within the floor is
  measurement noise, NOT a falsified assumption).

---

## 5. Reused artifacts & code (existing infra map)

Don't rebuild — the harness (Phase 0) wraps these.

**Context suite (the §1.2 library) — the load-bearing reuse:**
- `scripts/issue594_build_battery.py` — builds the 50-context / 7-family battery → `data/issue594/battery.json`
- `scripts/issue594_common.py` — `load_battery()`, family counts, schema validation
- `scripts/issue594_extract_context_vectors.py` — extracts `c_C` over the battery, **all 28 layers**, last-input-token slot (drop-in for Phase 0)
- `scripts/issue594_analyze_context_geometry.py` — context-geometry analysis (family separability)
- `scripts/issue617_*.py` — WildChat real-conversation cluster contexts (extends the `wildchat` family)
- probe pool: `issue404_common.fetch_preregistered_probes()` (48 Betley probes = `x∼C`)

**Extraction / vectors:**
- `scripts/extract_persona_vectors.py`, `scripts/issue623_persona_panel_vectors.py` — persona/context vectors (`c_C`, `r_B` candidates)
- `scripts/issue634_extract_behavior_vectors.py` — behavior vectors (`r_B`)
- `scripts/issue650_extract_context_bank.py` + `experiments/issue_650/shift_extract.py` — context bank `c_C` + activation shift `Δv`
- `scripts/issue541_geometry_extract.py`, `scripts/extract_prompt_divergence_activations.py` — geometry/context extraction
- `scripts/issue493_extraction_metric_bakeoff.py` — **layer/summary recipe selection** (directly Phase 1)
- `src/.../analysis/representation_shift.py`, `divergence.py`, `js_canonical.py`, `probes.py` — shift/divergence/probe analysis

**Predictor bakeoff:**
- `experiments/behavior_testbed_545/predictors.py` + `predictors_zoo.py` — the predictor zoo (cosine/JS/KL/Gaussian-KL)
- `scripts/issue532_predictor_stress.py`, `scripts/issue545_metric_race.py` — predictor stress/race harnesses (the LOBO/LOCO evaluation pattern)

**Eval suite:** `experiments/factor_screen_365/{persona_panel.py, eval_panel.py}`,
`eval/{alignment.py, refusal.py, marker_logprob.py, capability.py}`,
`configs/eval/default.yaml`.

**Reusable trained artifacts (apply the artifact-reuse checklist before reuse):**
- **#474 epoch-1 non-saturated marker adapters** (16 contexts) → marker gate spine (used by #532/#539/#540/#549).
- #608 frozen contrastive adapters; #545 19-behavior testbed (B→B′); #521/#551 persisted activation-shift tensors.

**Methodology docs** (findings-blind references to mirror): `docs/methodology/`
has ~60 issue docs incl. #521 (rank-one across contexts), #532 (predictor stress),
#541 (geometry re-analysis), #545 (behavior testbed), #601 (contrastive dose),
#604 (write-direction seed-stability), #612 (on-policy), #650 (shift extract).

**Prior findings that pre-load the priors (from RESULTS / open_questions):**
- Base-model behavior prior beats geometry for absolute **level**; geometry
  predicts the **shift** (#532/#541/#649). → frame A3.2/A3.3 (level) vs
  A3.7–A3.10 (shift) accordingly; **always include the base-prior baseline.**
- Write direction is seed-stable; top-1 key is seed-arbitrary (#604) → A3.8 SVD.
- Rank-1 leak-transfer asymmetry generalizes out-of-sample for marker + taught
  fact, **fails for content behaviors** (#637) → expect A3.8 to be behavior-
  dependent; report per-behavior, don't aggregate over the failure.
- Saturation hides the gate (#448) → §1.5 non-saturated anchor is mandatory.

---

## 6. Cost estimate (primary model, recommended grid)

| Phase | Work | GPU-h (rough) |
|---|---|---|
| 0 | base extraction (gen 50 contexts × 48 probes + c_C + r_B + t + Σ_c corpus) + dataset build | ~3–6 |
| 1 | base-only assumptions (CPU + reuse Phase-0 store; small MLP train on CPU/1 GPU) | ~1–2 |
| 2 | ~8–12 LoRA fine-tunes (≥2 doses) + trained extraction (~1–2 GPU-h each) | ~30–55 |
| 3 | CPU on store | ~0 |
| 4 | end-to-end (CPU) + any confirmatory regen | ~2–5 |
| **Total (7B)** | | **~40–70 GPU-h** |

Judging is off-GPU (Batch API). **7B-only this run** (scale checks deferred).
**Optional A3.10 metric-drift add-on:** +1 background-corpus pass on one θ⁺ (~1–2
GPU-h), run only if the `M⁰`-scoped gate residual is large (§4 note). Phaseable:
Phase 0+1 alone (~5–8 GPU-h) already settles the foundational assumptions before
any fine-tune budget is committed.

---

## 7. Open design decisions (recommendations)

### 7.1 Contrastive negatives vs positive-only — **recommend: contrastive default + positive-only control arm**
The project rule mandates contrastive negatives, and the project's own finding is
that the distance→leakage **gradient exists ONLY inside the contrastive regime**
(positive-only leaks uniformly, #207/#383). The gate `g_C(C')` therefore has
dynamic range to predict only under contrastive training. **But** positive-only is
the theory's degenerate `g≈const` case and a clean read of the raw write `δ`. So:
**default = contrastive** (gate tests A3.8–A3.10 need it); add a **positive-only
control** on the marker spine — if the gate is graded under contrastive and flat
under positive-only, that itself characterizes the recipe-dependence of `g`
(relevant to A3.10's base-predictability claim). Map: `δ_{C,B}` = displacement of
the **positive** rows; negatives shape the gate's selectivity (part of the recipe).

### 7.2 Recommended starting grid (expandable)
4 sources × marker (context-leakage spine) + ~6 B-family adapters × 2 sources
(behavior-leakage spine), ≥2 doses, primary model. ≈8–14 fine-tunes core. Expand
sources/behaviors only after Phase 1 validates the chain. (Why a subset, not the
full ~950-adapter cross-product: §7.5.)

### 7.3 Layer comparability
Default within-layer; cross-layer pooling is an ablation with per-layer
mean-centering (#536). Pick per-behavior best layer in Phase 1.

### 7.4 `Σ_c` corpus & regularization
Estimate off a broad background corpus (≥2–5k contexts), **not** the 50-context
battery; regularize `Σ_c+λI` and/or top-eigendir truncation; sweep λ as a baseline
knob.

### 7.5 Why not all (C, B → C′, B′) combos?
The full tuple space is (source context × trained behavior × target context ×
evaluated behavior). We split it deliberately:

- **Eval IS full-grid (cheap on the cached store).** Once θ⁺_{C,B} exists, scoring
  all 50 target contexts × the 11-column registry is generation + judging on the
  cached grid — no extra training. The binding constraint is **judge calls, not
  GPU**, so the plan scores the **full column registry on the primary context per
  trained model** and the `ROBUSTNESS_COLUMNS` subset (broad_em, sycophancy,
  marker, harmful_compliance) on the extra context families — the #545 testbed's
  own tradeoff (full battery × all contexts ≈ ~10× judge budget).
- **Training is NOT full-grid — combinatorial and unnecessary.** A fine-tune is one
  run per (source context, trained behavior). The full cross-product ≈ 50 contexts
  × ~19 row specs ≈ **~950 adapters**, each needing its own training + full-grid
  extraction + full-grid judging → **thousands of GPU-h and ~10⁷–10⁸ judge calls**.
  Infeasible — and pointless: the predictor's entire purpose is to forecast leakage
  for an **untrained** (C,B) from base-model quantities. So we train a
  **representative subset** to estimate ground truth and **validate** the predictor,
  then test generalization under **LOBO / LOCO** (§1.7). If we could afford all
  combos we wouldn't need a predictor.
- **Factorization makes most of the cross-product redundant.** If A3.8 + the
  no-interaction identity `L̂_{C',B'}=L̂_{C',B}·L̂_{C,B'}/L̂_{C,B}` hold, the
  generalized cell is determined by the two single-axis slices — the full (C′×B′)
  grid per source is then needed only to **test** factorization, not to use the
  predictor.
- **Held-out cells are required for honest evaluation** — training everything would
  leave no genuinely-new behavior/context to test generalization on (the paper's
  optimistic-evaluation guard).

---

## 8. Risks & falsification

- **A3.2 fails** (mean-answer summary insufficient) → whole chain stalls; fall back
  to A3.1's richer summary. *This is why Phase 1 runs first and cheap.*
- **A3.6 fails** (read-out drifts under FT) → predictor isn't pre-FT computable;
  quantify drift, restrict to stable behaviors.
- **A3.8 fails for content behaviors** (expected per #637) → report rank-one
  per-behavior; adopt the low-rank fallback `Σ w_j g_j`; do **not** average over
  the failure.
- **A3.10 fails** (g0 ≠ ĝ^real, g⁺ ≫ g0) → gate is real but not base-predictable;
  the drift decomposition says whether it's key/query/metric drift.
- **Saturation** masks every gate signal → enforce non-saturated anchors (#448).
- **Base-prior confound:** the base behavior prior is a strong predictor of level;
  always partial it out / include as a baseline so a geometry "win" isn't the
  prior in disguise (#532/#649).
- **Noise floor:** report nothing above the test-retest reliability ceiling.

## 9. tasks/ rollout

One `proposed` task per phase (or per assumption-cluster), each routed through
`/adversarial-planner` → `/issue <N>`, all inheriting this document as the
campaign design. Suggested anchors in `docs/open_questions.md`: the leakage-
predictor (§3.1), context-geometry, and readout-stability questions. Phase 0+1 is
the first task (settles the foundation cheaply); Phase 2 is the fine-tune fleet;
Phase 3+4 fold onto the same store.

Optionally promote the whole program to a `kind: campaign` task pinned to the
leakage-predictor open-question anchor, with this file as the `## Campaign Brief`.
