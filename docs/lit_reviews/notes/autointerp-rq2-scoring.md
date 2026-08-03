# RQ2 — How are SAE feature explanations scored and validated, and which scoring methods actually work?

Research notes for the auto-interpretability literature sweep. Written to be
synthesized into a review + production pipeline design, not read as prose.

**Scope note / relation to the existing taxonomy doc.** `residual-stream-direction-taxonomy.md`
already establishes at headline level: that 2410.13928 is the de-facto scoring
schema and introduced intervention scoring; that 2501.17727 shows auto-interp
metrics do not separate trained from random transformers; that 2503.09532
(SAEBench) finds proxy gains do not transfer to practice; that 2506.07985 shows
crowdsourced top-activation-only evaluation is unreliable; and that 2310.06200
shows explanation quality is prompt-fragile. This file does **not** re-argue
those. It supplies the mechanism-level detail, the actual numbers, the
threshold conventions, and five lines the taxonomy doc does not cover at all:

- the **metric-level sanity checks** (2506.05774) — a distinct and more damaging
  result than the trained-vs-random check, because it indicts the *metrics*
  rather than the dictionaries;
- the **explanation-free** evaluation branch (2507.08473) and its human-agreement
  numbers, which are the only solid inter-rater figures in the field;
- the **strongest faithfulness test run to date** (2501.18838), whose result is
  that explanation-based simulation barely beats zero ablation;
- **descriptive collision** (2605.12874) — a failure mode provably invisible to
  detection scoring;
- the **benchmark-reliability audit** (2605.18229) that post-dates SAEBench and
  retires two of its metrics.

---

## 1. Scoring-method table

Cost figures are from 2410.13928 Table (cost per **100,000 features**, 100
examples/feature, Llama-3.1-70B scorer unless noted). Divide by 1e5 for
per-feature cost.

| # | Method | Mechanism | What it validates | Cost / 100k features | Known pitfalls | Refs |
|---|---|---|---|---|---|---|
| 1 | **Simulation scoring** (the original paradigm) | Explainer LLM reads the explanation and predicts the per-token activation over a context; score = Pearson correlation(simulated, true). Two variants: *all-at-once* and *token-by-token* | Whether the explanation supports token-level reconstruction of the activation pattern | $3,600 all-at-once; **$46,700 token-by-token** ($31.5k / $219.1k on Claude 3.5) | Scores only activating examples, so ignores false positives entirely; top-activation biased; 30× cost of fuzzing; correlation is not calibrated across features | Bills et al. 2023 [blog]; 2410.13928 |
| 2 | **Detection** | Whole-sequence binary: given the explanation, does this sequence activate the feature? Balanced accuracy over activating + non-activating contexts | Precision *and* recall at sequence granularity; unlike simulation it sees negatives | **$588** ($5.5k Claude 3.5) | Forgiving — no token localization required; passes vague explanations that are directionally right; **provably invariant to descriptive collision** (2605.12874); balanced accuracy fails the extra-labels sanity check (2506.05774) | 2410.13928 |
| 3 | **Fuzzing** | Token-level: some tokens are highlighted (correctly or not); the scorer says whether the marking is correct | Token-level localization of the explanation | **$676** ($6.2k Claude 3.5) | Can score high on token-surface patterns while missing the context condition; same balanced-accuracy sanity-check problem | 2410.13928 |
| 4 | **Surprisal** | log p(context \| explanation) − log p(context \| pseudo-explanation); score = AUROC of that information value separating active from inactive contexts | Information content of the explanation about the context | Cheap (no explicit figure) | Lowest agreement with every other method (ρ=0.15 with simulation); authors state "current setup could be improved"; human correlation only 0.34 | 2410.13928 |
| 5 | **Embedding scoring** | Explanation used as a retrieval query; embed candidate contexts; AUROC of retrieving activating over non-activating | Coarse semantic alignment; a cheap triage filter | **~$50** — cheapest by ~12× | Lowest human correlation (0.32); fails on context-specific and token-positional patterns; an off-the-shelf encoder does not understand highlight markup | 2410.13928; 2507.08473 |
| 6 | **Intervention scoring** | Clamp/steer the feature, generate text, measure the decrease in the scorer's surprisal about the explanation conditioned on the intervened generation vs a clean generation | The **output-side / causal** role of the feature, not its input trigger | Requires generation — dominated by sampling cost | New and least validated; no human-agreement number reported; conflates feature effect with steering artifacts | 2410.13928 |
| 7 | **Intruder detection** (explanation-free) | Show 4 activating examples + 1 non-activating "intruder" from another latent; identify the odd one out. Accuracy averaged across activation deciles | Whether the latent's activating set is *coherent at all* — decouples latent quality from explanation-writing quality | ~detection-class | Random baseline is 20%, so the usable dynamic range is 20–100%; humans see far fewer examples (10–20) than LLM scorers (100) | 2507.08473 |
| 8 | **Example-embedding scoring** (explanation-free) | Do a latent's activating examples cluster in embedding space? AUROC from cosine similarity | Cheap coherence proxy, no LLM judge | ~$50-class | **Near chance**: AUROC ≈ 0.5 except in the highest decile (0.64–0.70). Effectively unusable as a standalone gate | 2507.08473 |
| 9 | **Output-centric descriptions + evaluation** | Describe the feature from tokens promoted after stimulation, or from the unembedding of the feature direction; evaluate by steering | Causal effect on outputs, which input-centric pipelines systematically miss | Very cheap (unembedding is one matmul) | Alone it misses input conditions; best results come from combining input- and output-centric | 2501.08319 |
| 10 | **CoSy (concept synthesis)** | Generate *new* inputs conditioned on the textual explanation; check whether the neuron actually fires on them vs control data | Generative/counterfactual faithfulness — closes the loop the observational metrics leave open | Generation-bound | Vision-first; depends on generator coverage and fidelity | 2405.20331 |
| 11 | **FADE** (Clarity / Responsiveness / Purity / Faithfulness) | Four-axis model-agnostic feature↔description alignment, and attribution of *why* misalignment occurs | Decomposes "bad score" into diagnosable causes | Moderate | Finds description generation is **harder for SAE latents than for MLP neurons** — the opposite of the usual framing | 2502.16994 |
| 12 | **PRISM** | Emits multi-concept descriptions; reports a description score and a separate polysemanticity score | Drops the one-feature-one-concept assumption | Moderate | Newer; less externally replicated | 2506.15538 |
| 13 | **Collision-adjusted detection / discrimination scoring** | Penalize an explanation that fails to distinguish its feature from its neighbours | **Discrimination** — an axis orthogonal to accuracy that all standard scoring ignores | Cheap add-on | Very new (2026-05), single-author, not yet replicated | 2605.12874 |
| 14 | **CE-Bench** | Contrastive story pairs; interpretability score without any external LLM judge | Judge-free, reproducible, no API dependence | Very cheap | >70% Spearman with SAEBench — agreement is good but not a substitute | 2509.00691 |
| 15 | **NeuronEval** (meta-metric framework) | Unifies 18 metrics from 19 studies as `s_M(a_k, c_t)` over an activation vector and a concept vector; adds two sanity checks | Validates the **metric itself**, not the explanation | Trivial | See §3 — most popular metrics fail | 2506.05774 |

---

## 2. Reported reliability and threshold numbers

### 2.1 Headline scores (2410.13928, balanced accuracy, median with IQR)

| Condition | Fuzzing | Detection |
|---|---|---|
| Random explanation (floor) | 0.51 (0.45–0.57) | 0.51 (0.45–0.58) |
| **Randomly-initialized SAE** | 0.55 (0.50–0.60) | 0.54 (0.50–0.59) |
| 16k-latent SAE, Gemma-2-9B | 0.73 (0.63–0.83) | 0.70 (0.59–0.79) |
| **131k-latent SAE, Gemma-2-9B** | **0.76 (0.67–0.86)** | **0.74 (0.63–0.85)** |
| 262k-latent SAE, Llama-3.1-8B residual | 0.81 (0.71–0.86) | 0.83 (0.68–0.85) |
| Top-32 sparsified neurons (Gemma) | 0.62 (0.54–0.70) | 0.59 (0.53–0.64) |
| Top-256 sparsified neurons (Gemma) | 0.59 (0.53–0.65) | 0.57 (0.52–0.62) |
| Top-32 sparsified neurons (Llama) | 0.55 (0.51–0.62) | 0.53 (0.50–0.57) |

Two things to read off this table. First, the **entire usable range is ~0.51 to
~0.81** — a "0.7" is not 70% of the way to good, it is roughly two-thirds of the
way up a 0.30-wide band. Second, the IQRs are ~0.2 wide and overlap heavily
between conditions, so per-feature scores are far noisier than aggregate
medians suggest.

### 2.2 Inter-method agreement (Spearman, 800 features, 131k Gemma-2-9B)

|  | Fuzzing | Detection | Simulation | Embedding | Surprisal |
|---|---|---|---|---|---|
| Fuzzing | 1.00 | 0.73 | 0.75 | 0.41 | 0.30 |
| Detection | | 1.00 | **0.44** | 0.71 | 0.62 |
| Simulation | | | 1.00 | **0.28** | **0.15** |
| Embedding | | | | 1.00 | 0.79 |

**The methods do not measure the same thing.** Detection–simulation at 0.44 and
simulation–embedding at 0.28 mean a feature's rank can move drastically with
the scorer chosen. 2410.13928 gives worked cases of each divergence pattern
(high-fuzz/low-detect = vague context; high-fuzz/low-sim = no token
localization; high-embed/low-sim = right semantics, wrong tokens).

### 2.3 Agreement with humans

| Scoring method | Spearman with human ratings (700 contexts, 81 features) |
|---|---|
| Fuzzing | **0.69** |
| Simulation | 0.60 |
| Detection | 0.59 |
| Surprisal | 0.34 |
| Embedding | 0.32 |

From the explanation-free line (2507.08473, intruder-detection task):

| Pair | Correlation |
|---|---|
| **Human ↔ human** (2 labelers, 40 latents) | **0.87** |
| Claude 3.5 Sonnet ↔ human | 0.84 |
| Gemini Flash 2.0 ↔ human | 0.83 |
| QwQ-32B ↔ human | 0.78 |
| Llama-3.1-70B ↔ human | 0.77 |
| Llama-70B ↔ QwQ-32B | 0.89 |
| Claude ↔ QwQ-32B | 0.91 |

Human–human 0.87 is the **ceiling**: no scorer can be validated past it, and the
best LLM scorers (0.84) are already within noise of it. LLM–LLM agreement
(0.89–0.91) *exceeds* human–human, which is a correlated-error signature, not
evidence of superiority.

### 2.4 Explainer-model choice barely matters; scorer-model choice does a little

Explainers (2410.13928, 500+ features, median fuzzing / detection):
Claude 3.5 Sonnet 0.75 / 0.75 · Llama-3.1-70B 0.76 / 0.74 · Llama-3.1-8B 0.70 /
0.70 · **Human explainer 0.75 / 0.74**.

**LLM explainers already match human explainers.** The bottleneck is not the
explanation writer. Scorer choice: Claude and Llama-70B are close; Llama-8B is
systematically ~0.05 lower across all methods, so scorer must be pinned for any
cross-run comparison.

### 2.5 Threshold conventions and pass rates

There is **no field-standard numeric threshold.** What exists:

- **Bills et al. 2023 (origin of simulation scoring):** score ≥ 0.8 is described
  as "accounts for most of the neuron's top-activating behavior." Of ~300K
  GPT-2-XL neurons, **~1.7k cleared 0.8 — roughly 0.55%.** That is the field's
  most-cited threshold and its pass rate is under one percent.
- **Huang et al. 2309.10312** re-evaluated 300 of those *top-scoring*
  explanations and found F1 ≈ 0.6 with "high error rates and little to no
  causal efficacy." The 0.8 threshold does not certify causal validity even on
  the best-scoring 0.55%.
- **2507.08473 (intruder detection)** is the only paper giving an explicit
  binning scheme: random baseline 20%; ≤20% non-interpretable; Low 20–40%,
  Medium 40–60%, High 60–80%, Very High 80–100%. Pass rates: **~1/3 of latents
  score >80%; ~1 in 7 score <30%.** Human accuracy averaged 65% overall,
  78% in the highest activation decile.
- **2410.13928 states no threshold at all** and deliberately reports
  distributions instead.

Practical read: **the honest published range for "fraction of latents that are
solidly interpretable" is ~0.5% (strict simulation ≥0.8) to ~33% (intruder
>80%),** and the two numbers measure different constructs. Any pipeline
quoting a single "X% of our features are interpretable" without naming the
metric and threshold is not saying anything.

### 2.6 Dead and rare latents (the denominator problem)

From 2410.13928 on a 131k Gemma-2-9B SAE over 10M RPJv2 tokens, 256-token
contexts: **15% of latents never activate at all; 30% activate fewer than 200
times.** With 1024-token contexts only 5% never fire; on the Pile, 15% fire
<200 times and only 1% never fire. Scoring is typically run only on latents
above an activation floor, so headline interpretability numbers are computed
on a **filtered subpopulation** whose size depends on corpus and context
length. Report the denominator or the number is meaningless.

### 2.7 Sampling protocol (what the numbers are conditional on)

2410.13928 defaults: **40 activating examples** at 32 tokens each for
explanation generation; **100 activating + 100 non-activating** contexts for
scoring, stratified **10 per activation decile**; intervention scoring uses 40
length-64 prompts (30 scoring / 10 explainer), stratified by quintile, with
latents firing <200 / 10M tokens excluded.

Sampling regime materially moves scores — top-only 0.73 fuzz / 0.72 detect vs
quantile-stratified **0.77 / 0.74**. Top-sampled explanations have "higher
specificity, lower sensitivity" and "fail to capture the whole distribution."
2506.05774 goes further and instructs practitioners **not to use
top-and-random sampling at all**, because it inflates scores.

---

## 3. The sanity-check literature — three distinct checks, all uncomfortable

### 3.1 Check A: does the metric distinguish a trained model from a random one? (2501.17727)

Heap, Lawson, Farnik & Aitchison train SAEs on randomly-initialized Pythia
transformers (70M–6.9B) under four randomization schemes (re-randomized
including embeddings; excluding embeddings; step-0 weights; and a control with
random token embeddings at inference), scoring with **fuzzing** primarily and
detection in appendices. Finding: auto-interp and reconstruction scores for
randomized models are "surprisingly similar" to trained models in many
settings. The gap is **larger for small models (Pythia-70M) and narrows with
scale (Pythia-6.9B)**; the control variant does separate cleanly at all sizes.
Recommendation: routine randomized baselines plus targeted "abstractness"
measures (they propose token-distribution entropy as a proof of concept).

**Flagged tension — worth surfacing in the synthesis.** 2410.13928 reports a
*clear* separation on this exact axis: randomly-initialized SAE 0.55/0.54 vs
trained 0.76/0.74, only 0.04 above the random-explanation floor of 0.51.
2501.17727 reports near-parity. The two are not directly reconciled in either
paper (2501.17727 cites Paulo et al. but makes no numeric comparison). Likely
sources: Pythia vs Gemma/Llama, randomizing the **transformer** vs randomizing
the **SAE**, corpus and SAE hyperparameters. Note that these are *different
experiments* — Paulo randomizes the SAE on a trained model, Heap randomizes the
transformer and trains a real SAE on it — so the "contradiction" may dissolve
on inspection. **A production pipeline should run both controls, since they
test different things and the literature disagrees about the second.**

### 3.2 Check B: does the metric respond to corrupting the *label*? (2506.05774)

This is the most decision-relevant paper found in the sweep and is **absent from
the existing taxonomy doc.** Oikarinen, Yan & Weng unify 18 metrics from 19
studies as `s_M(a_k, c_t)` and impose two sanity checks:

- **Missing-labels test** — randomly zero half of the concept vector's entries
  (explanation now too specific). A sound metric's score must drop.
- **Extra-labels test** — randomly double the concept vector's positives
  (explanation now too generic). Score must drop.

Pass criterion: score decreases on >90% of cases (ε = 0.001).

| Metric | Missing-labels | Extra-labels | Verdict |
|---|---|---|---|
| Pearson correlation | 99.41% | 99.92% | **PASS** |
| Cosine similarity | 99.45% | 99.26% | **PASS** |
| AUPRC | 95.61% | 99.46% | **PASS** |
| F1 | 93.68% | 99.82% | **PASS** |
| IoU | 93.62% | 99.81% | **PASS** |
| **Balanced accuracy** | — | **53.67%** | **FAIL (too generic)** |
| AUC | — | 59.18% | FAIL (too generic) |
| Recall | — | **0.00%** | FAIL (total) |
| Correlation w/ top-and-random sampling | — | 60.26% | FAIL |
| Precision | **45.73%** | — | FAIL (too specific) |
| MAD | 59.81% | 59.81% | FAIL |
| Inverse balanced accuracy | 64.18% | — | FAIL |
| Accuracy | 23.79% | 70.37% | **FAIL both** |
| Spearman correlation | 64.05% | 49.21% | **FAIL both** |

Meta-AUPRC ranking against ground-truth-labelled neurons: Correlation (1.60) >
Cosine (2.30) > AUPRC (3.90) > F1 / IoU (6.70).

**The sharpest consequence: balanced accuracy — the metric detection and
fuzzing actually report — fails the extra-labels test at 53.67%, barely above
coin-flip.** It cannot reliably tell a correct explanation from an
over-general one. A vague label like "text about people" is close to
unpenalized. This is independent of, and additive with, the trained-vs-random
problem. Guidelines: never rely on a failing metric alone; avoid
top-and-random sampling; avoid metrics insensitive to class imbalance (AUC,
accuracy) on rare-firing latents (<10% of inputs); prefer F1, AUPRC,
correlation, cosine, IoU.

### 3.3 Check C: is the benchmark itself reliable? (2605.18229)

David Chanin audits SAEBench through three lenses — reseed noise on a fixed
SAE, ground-truth correlation on synthetic SAEs, and discriminability across
training trajectories.

| Metric | Reseed CV | Min reliable Δ | Ground-truth Spearman |
|---|---|---|---|
| sae-probes (k=5) | 0.2% | 0.008 | — |
| RAVEL disentangle | 0.2% | 0.004 | — |
| sparse probing (k=5) | 0.3% | 0.012 | +0.55 to +0.87 |
| **Autointerp** | **0.5%** | **0.016** | — |
| SCR (top-10) | 4.4% | 0.031 | **−0.64** (top-500) |
| TPP (top-50) | **23%** | 0.083 | **−0.03** (top-10) |

Verdict: **TPP and SCR should not be used** at canonical settings — TPP has
essentially zero ground-truth correlation and 23% reseed CV; SCR *inverts*
above top-N=50 and rates a perfect oracle below 11 trained SAEs. Both decline
during training, implying "an untrained SAE is better than a trained SAE."
sae-probes is the most reliable tested (113 tasks, cross-validated L1 logistic
regression) but saturates >0.99 while ground-truth quality still varies
0.56–0.79.

Seed requirements: a single-seed comparison needs a difference exceeding
**3.93σ**; three-seed averaging drops this to ~1.6σ; five-seed to ~1.0σ.
**Minimum three independent seeds per configuration.**

Good news for auto-interp specifically: at 0.5% CV it is one of the *more*
stable metrics on the reseed lens. Its minimum resolvable difference is
**0.016** — so a claimed auto-interp improvement smaller than ~1.6 points is
noise.

### 3.4 Check D: can the score be gamed? (2504.07831, 2503.13445)

2504.07831 shows Llama, DeepSeek-R1 and Claude 3.7 Sonnet can produce
steganographic, deliberately deceptive explanations that **achieve
interpretability scores comparable to reference labels** while evading an
overseer, and will scheme toward this when they believe detection of harmful
features is against their interest. 2503.13445 independently analyses
faithfulness metrics across 75 models / 13 families and studies "the extent to
which metrics can be gamed," proposing φ-CCT and F-AUROC as harder-to-game
replacements. **Any auto-interp score used as an oversight signal is an
optimization target and must be treated as adversarially attackable.**

---

## 4. Faithfulness vs plausibility — the distinction and where it bites

**Plausibility** = a human finds the explanation sensible. **Faithfulness** =
the explanation actually captures the feature's causal role in the
computation. Every scoring method in §1 rows 1–5 measures a *predictive*
property of the input distribution, which is closer to plausibility than to
faithfulness. Four results define the gap:

1. **2501.18838 (Paulo & Belrose, "Partially Rewriting a Transformer in Natural
   Language") — the strongest faithfulness test run to date.** They approximate
   an LLM feedforward layer with a transcoder, auto-interpret its neurons, then
   replace the sparse MLP's first layer with an **LLM simulator driven only by
   the explanations**, and measure downstream loss. Result: the model's
   **increase in loss is statistically similar to replacing the sparse MLP
   output with the zero vector.** Same protocol with an SAE on the residual
   stream gives the same answer. Explanations that pass detection/fuzzing at
   ~0.75 carry **approximately zero usable information about the computation**
   relative to ablating it. This is the single most important number in this
   sweep and it is not in the existing taxonomy doc.
2. **2309.10312** — observational + interventional evaluation of the Bills et
   al. GPT-4 explanations: even the most confident have high error rates and
   little to no causal efficacy.
3. **2501.08319** — steering evaluations reveal current (input-centric)
   pipelines "provide descriptions that fail to capture the causal effect of
   the feature on outputs." Output-centric descriptions fix the causal side;
   combining input- and output-centric is best on both; output-centric
   descriptions even recover inputs for latents previously believed dead.
4. **2510.03659** — Kendall τ_b ≈ **0.298** between SAEBench interpretability
   and AxBench steering utility across 90 SAEs / 3 models / 5 architectures /
   6 sparsity levels. After selecting features by ΔToken-Confidence, the
   correlation **vanishes to ≈0 and can go negative.** Interpretability is at
   best a weak proxy for utility, and for the most useful features it is no
   proxy at all. *(Caveat: this paper's abstract carries placeholder arXiv ids
   — see §6.)*

Corroborating from the other direction: **2603.04198** finds L2 weight
regularization roughly **doubles steering success rates while leaving mean
auto-interp scores essentially unchanged**, and only *in the regularized
setting* does steering become well-predicted by auto-interp score. So the
score→utility link is not intrinsic; it is a property some dictionaries have
and others do not.

And **2504.13151 (MIB)** finds that for causal-variable localization, **SAE
features are not better than neurons** (supervised DAS wins), which caps how
much a faithfulness story can rest on the SAE basis itself.

---

## 5. Cross-dataset / OOD stability, calibration, and confidence

This is the thinnest area in the literature and the largest genuine gap.

- **2512.18092 (Yan, Oikarinen & Weng)** is the only work found that treats
  stability and calibration formally. It frames neuron identification as the
  inverse of learning, derives **generalization bounds for accuracy, AUROC and
  IoU** to guarantee faithfulness, and proposes a **bootstrap ensemble
  procedure** quantifying stability across probing datasets, with a Bootstrap
  Explanation (BE) method that emits **concept prediction *sets* with a
  guaranteed coverage probability** — i.e. conformal-style calibrated
  explanation, rather than a point label. This is the right primitive for a
  production confidence estimate and is currently near-unique.
- **2104.07143 (interpretability illusion for BERT)**, already in the taxonomy
  doc, is the reason this matters: narrow corpora make neurons *and linear
  combinations* spuriously look like single concepts, so an explanation
  validated on one corpus is not validated generally.
- **2410.13928's own dataset comparison** is an implicit stability result: the
  fraction of never-firing latents moves 15% → 1% between RPJv2 and the Pile
  at matched budget, so which latents are even *eligible* for explanation is
  corpus-dependent.
- **2601.09776 (TimeSAE)** reports that "many current explanation methods are
  sensitive to distributional shifts," though in the time-series domain.
- No paper found measures **per-feature explanation stability across
  independent explainer samples** (temperature-driven variance of the
  explanation text itself). Nearest neighbours are 2310.06200 (explanations are
  prompt-sensitive; reformatting the prompt significantly improves quality and
  cuts cost) and 2506.15538 (PRISM, motivated by "limited robustness" of
  single-description methods). **This is an open measurement gap a production
  pipeline can and should close cheaply.**

### Descriptive collision — a distinct, provable blind spot (2605.12874)

Reanalyzing the largest public human-annotated SAE feature set (722 features,
Gemma-2-2B and Pythia-70M): the mean annotation string is reused across
**3.07 features**; **82.1%** of features share their annotation with at least
one other; the single most common string ("plural nouns") labels **101 distinct
features across 18 layers and four components**; and information-theoretically
the average annotation resolves only **70% of feature identity**. The paper
formalizes *discrimination* and **proves detection-style scoring is invariant
to collision** — a label shared by 101 features can score perfectly. Estimated
inflation of reported interpretability ≈ one-third of the bits needed to
identify a feature. Proposed fixes: collision-adjusted detection and
discrimination scoring. *(Caveat: 2026-05, single author, not yet replicated.)*

---

## 6. Verification ledger

All identifiers below were resolved **in-session** via the arXiv MCP. "search"
= returned by `search_papers` with matching title, authors, date and abstract
(the verification standard used by the parent taxonomy doc); "abstract" =
direct `get_abstract`; "full text" = additionally read via the arXiv HTML
rendering.

| arXiv id | Short title | Verified by | Used for |
|---|---|---|---|
| 2410.13928 | Automatically Interpreting Millions of Features | search + **full text** | The five scoring methods; all §2.1–2.4, 2.6–2.7 numbers |
| 2507.08473 | Evaluating SAE interpretability without explanations | search + **full text** | Intruder detection; all human/inter-rater agreement; thresholds + pass rates |
| 2506.05774 | Evaluating Neuron Explanations: Unified Framework with Sanity Checks | search + **full text** | §3.2 metric-level sanity checks, the balanced-accuracy failure |
| 2605.18229 | Are Sparse Autoencoder Benchmarks Reliable? | search + **full text** | §3.3 reseed CV, GT correlations, seed requirements |
| 2501.17727 | Auto-Interp Metrics Do Not Distinguish Trained and Random | search + **full text** | §3.1 randomization schemes and the flagged tension |
| 2501.18838 | Partially Rewriting a Transformer in Natural Language | search | §4 item 1 — the zero-ablation faithfulness result |
| 2309.10312 | Rigorously Assessing NL Explanations of Neurons | search | §4 item 2; the 0.8-threshold rebuttal |
| 2501.08319 | Output-Centric Feature Descriptions | search | §4 item 3; scoring-table row 9 |
| 2510.03659 | Does higher interpretability imply better utility? | search | §4 item 4 (τ_b ≈ 0.298) — **see caution below** |
| 2503.09532 | SAEBench | search | Benchmark baseline for §3.3 |
| 2501.17148 | AxBench | **abstract** | Confirms the correct id for the steering benchmark 2510.03659 mis-cites |
| 2502.16994 | FADE | search | Scoring-table row 11 (Clarity/Responsiveness/Purity/Faithfulness) |
| 2506.15538 | PRISM | search | Row 12; robustness motivation in §5 |
| 2605.12874 | Descriptive Collision in SAE Auto-Interpretability | search | §5 collision numbers; row 13 |
| 2509.00691 | CE-Bench | search | Row 14 (>70% Spearman with SAEBench) |
| 2405.20331 | CoSy: Evaluating Textual Explanations of Neurons | search | Row 10 |
| 2506.07985 | Beyond Top Activations | **abstract** | MG-IS 13×, BRAgg 3×, 40× total cost reduction |
| 2512.18092 | Faithful and Stable Neuron Explanations | search | §5 generalization bounds, bootstrap coverage sets |
| 2504.07831 | Deceptive Automated Interpretability | search | §3.4 gameability |
| 2503.13445 | Verbosity Tradeoffs / Faithfulness of Self-Explanations | search | §3.4 metric gaming; φ-CCT, F-AUROC |
| 2502.18848 | A Causal Lens for Evaluating Faithfulness Metrics | search | Meta-evaluation of faithfulness metrics (diagnosticity) |
| 2504.13151 | MIB: A Mechanistic Interpretability Benchmark | search | §4 — SAE features not better than neurons for causal localization |
| 2310.06200 | The Importance of Prompt Tuning for Automated Neuron Explanations | **abstract** | §5 explanation prompt-sensitivity |
| 2603.04198 | Stable and Steerable SAEs with Weight Regularization | search | §4 — steering doubles, auto-interp unchanged |
| 2509.18127 | Safe-SAIL | search | Pre-explanation triage metric; 55% cost cut; 1,758 safety features |
| 2502.15576 | Mutual-Information Explanations on SAEs | search | Frequency-bias critique of existing explanation objectives |
| 2601.09776 | TimeSAE | search | §5 distribution-shift sensitivity (time-series domain) |
| 2604.03436 | MetaSAEs | search | Uses Δfuzzing (+7.6%) as external validation — example of current practice |

**Grey literature (fetched live, not arXiv):** Bills et al. 2023, *Language
models can explain neurons in language models* (OpenAI) — the origin of
simulation scoring. The canonical `openaipublic.blob.core.windows.net`
paper URL served only front matter through the fetch tool, so its numbers here
(**~1.7k of ~300K GPT-2-XL neurons scoring ≥0.8**; score = correlation between
simulated and actual activations; ≥0.8 glossed as "accounts for most of the
neuron's top-activating behavior") were recovered via web search and are
**corroborated independently** by 2410.13928's method description and by
2309.10312, which re-evaluates 300 of the ≥0.8 explanations. Treat the exact
1.7k figure as second-hand pending a direct read of the primary page.

---

## 7. Could-not-verify / caution list

1. **Bills et al. 2023 primary page not directly read** — see above. The
   `.../neuron-explainer/paper/index.html` fetch returned title page, author
   contributions and citation block only; Methods/Results were not in the
   returned content. All quantitative claims attributed to it are second-hand
   or corroborated. **Verify before citing a specific number.**
2. **2510.03659 mis-cites its own references.** Its abstract cites SAEBench as
   "arXiv:2501.12345", AxBench as "arXiv:2502.23456" and an output-score
   criterion as "arXiv:2503.34567" — all placeholder-shaped ids. The correct
   ids are **2503.09532** (SAEBench) and **2501.17148** (AxBench), both
   verified in-session. The paper itself is verified to exist with that title,
   authors and abstract, but the sloppiness lowers confidence in its reported
   figures. **Its τ_b ≈ 0.298 should be treated as indicative, not
   authoritative, until the PDF is checked.**
3. **2501.17727 gives no explicit numeric table in the main text.** The
   full-text read confirmed the qualitative claims and the four randomization
   schemes but Figure 2 values were not extractable as numbers. The
   "trained ≈ random" claim is therefore verified as a *stated finding*, not as
   a reproduced figure — which matters given the tension with 2410.13928 (§3.1).
4. **2605.12874 and 2605.18229 are 2026-05, single-author, and unreplicated.**
   Both are load-bearing here (collision; benchmark retirement). Weight
   accordingly.
5. **Cost figures in §1 are 2024-era API prices** from 2410.13928 and will have
   moved. Use the *ratios* (embedding ≈ 1× · detection ≈ 12× · fuzzing ≈ 14× ·
   simulation-all-at-once ≈ 72× · simulation-token-by-token ≈ 934×), not the
   absolute dollars.
6. **Not sought (owned by other RQs):** explanation *generation* methods and
   prompting strategies (RQ1); taxonomy/categorization schemas; the persona and
   steering-direction literature; SAE architecture comparisons except where
   they report scoring numbers.
7. **Genuine measurement gap, no paper found:** per-feature explanation
   stability across independent explainer samples at temperature > 0 (i.e. the
   variance of the *label text* itself, as opposed to the variance of the
   score). Also not found: any calibration study mapping an auto-interp score
   to a probability that a human would endorse the label.

---

## 8. What a production pipeline must measure to claim its labels are trustworthy

Ordered by (evidence strength × cost to implement). Items 1–5 are cheap and
non-negotiable; 6–9 are the ones that actually license a trust claim.

**1. Two null baselines on every reported score, not one.**
Run (a) a **random-explanation** control — score real features against shuffled
or foreign labels — and (b) a **random-dictionary** control: a
randomly-initialized SAE, and ideally an SAE trained on a randomized
transformer. 2410.13928 puts these at 0.51 and 0.55 against a trained 0.76;
2501.17727 argues (b) can be much closer than that. Because the literature
disagrees (§3.1), a pipeline that reports only the trained number is reporting
an unfalsifiable quantity. **Report score minus floor, not score.**

**2. Pick metrics that pass the label-corruption sanity checks.**
Per 2506.05774: prefer **Pearson correlation, cosine, AUPRC, F1, IoU**. Do not
report **balanced accuracy alone** — the default output of detection and
fuzzing — because it fails the extra-labels test at 53.67% and cannot penalize
an over-general label. If detection/fuzzing are kept for comparability with the
literature (they should be), pair each with F1 or AUPRC computed on the same
contexts. Never report recall, precision or raw accuracy standalone.

**3. Fix and disclose the sampling regime; do not use top-and-random.**
Use **quantile-stratified** contexts (10 per activation decile), which
outperform top-only (0.77/0.74 vs 0.73/0.72) and avoid the specificity/
sensitivity skew. 2506.05774 explicitly instructs against top-and-random
sampling as score-inflating. Disclose contexts-per-feature, context length, and
the activation floor used to exclude rare latents.

**4. Publish the denominator.**
State how many latents were scored out of how many exist, and the firing floor
applied. 15% of latents never fire and 30% fire <200 times in 10M tokens at
256-token context (2410.13928); those fractions swing with corpus and context
length. "N% interpretable" without a denominator is not a measurement.

**5. Pin and version the scorer.**
Scorer model changes scores by ~0.05 (Llama-8B vs 70B/Claude). Pin the scorer
model id, the prompt (2310.06200: prompt format significantly changes quality),
the sampling temperature, and the number of draws. Treat any scorer change as
a re-baseline event.

**6. Report at least two *decorrelated* scorers, and treat the spread as the
error bar.**
Detection–simulation agree at only ρ=0.44 and simulation–embedding at 0.28.
A single scoring method is a single point of failure with a known failure
signature. The cheap, defensible combination is **fuzzing (best human
correlation, 0.69) + intervention scoring (the only output-side method) +
embedding as a ~free triage pre-filter**. Where they disagree, the feature is
not labelled — it is *contested*, which is a useful and reportable state.

**7. Measure the output side, not just the input side.**
Input-centric descriptions demonstrably fail to capture causal effect on
outputs (2501.08319), and explanation-driven simulation of a layer is
statistically indistinguishable from zero ablation (2501.18838). A pipeline
claiming its labels describe *what the feature does* — not merely *what
precedes it* — must run intervention scoring or an output-centric
(unembedding / token-promotion) description and report it separately. Do not
average an input score and an output score into one number; they dissociate.

**8. Measure discrimination, not just accuracy.**
Check how many features in the dictionary share each label. With 82.1% of
annotated features sharing a label and one string covering 101 features
(2605.12874), and detection scoring provably invariant to this, a
label-uniqueness statistic is close to free and catches a failure mode no
standard score can see. Minimum viable version: report the distribution of
features-per-label and the fraction of labels that are unique.

**9. Calibrate against humans on a small audit set, and respect the ceiling.**
Human–human agreement is 0.87 (2507.08473) — that is the ceiling, and the best
LLM scorers already sit at 0.84. Note LLM–LLM agreement (0.89–0.91) *exceeds*
human–human, so cross-LLM agreement is **not** evidence of correctness; it is
evidence of shared bias. Budget for a real human audit: 2506.07985 shows
model-guided importance sampling cuts the inputs needed ~13× and Bayesian
rating aggregation cuts ratings-per-input ~3×, for **~40× total cost
reduction**, which makes a few-hundred-feature audit genuinely affordable.
Sample the audit **across activation deciles**, not from the top — interpretability
varies strongly by decile (human 65% overall vs 78% in the top decile).

**10. Establish the noise floor before claiming an improvement.**
Auto-interp reseed CV is 0.5% with a **minimum resolvable difference of 0.016**
(2605.18229). A single-seed comparison needs Δ > 3.93σ to be meaningful;
three seeds bring that to ~1.6σ, five to ~1.0σ. **Run ≥3 seeds.** Do not
report an auto-interp delta under ~1.6 points as a result.

**11. State the threshold and its pass rate together, and pick the construct
first.**
There is no field-standard cutoff. If the claim is strict token-level
predictability, the reference point is simulation ≥0.8 with a **~0.55%** pass
rate. If the claim is "the activating set is coherent," the reference is
intruder-detection >80% with a **~33%** pass rate. These differ by ~60× and are
not interchangeable. Quote both the metric and the fraction clearing it, and
never migrate a threshold across constructs.

**12. Do not let an auto-interp score stand in for downstream utility.**
τ_b ≈ 0.298 between interpretability and steering utility, collapsing to ≈0 for
the most effective steering features (2510.03659, caveated); L2 regularization
doubles steering success with auto-interp scores unchanged (2603.04198); SAE
features are not better than neurons for causal-variable localization
(2504.13151); and SAEs lose to prompting for steering and to difference-in-means
for detection (2501.17148). If the pipeline's labels are meant to support a
downstream task, **measure that task**.

**13. Treat the score as adversarially attackable if it is used for oversight.**
Explanations can be made deceptive while scoring comparably to reference labels
(2504.07831), and faithfulness metrics are gameable in characterized ways
(2503.13445). An auto-interp score in a safety pipeline is an optimization
target; it needs an independent check (human audit, held-out intervention) that
is not the thing being optimized.

**14. Emit a confidence, not a bare label.**
The only principled machinery found is 2512.18092: generalization bounds for
accuracy/AUROC/IoU plus a bootstrap ensemble giving **concept prediction sets
with guaranteed coverage**. Even a cheap approximation — bootstrap over
context samples, emit a label *set* plus a coverage level, and mark features
whose label is unstable across explainer draws — is more honest than a point
label, and closes the calibration gap flagged in §5.

### Minimum viable trustworthy-label claim

A pipeline can claim its labels are trustworthy if it reports, per feature:
a sanity-check-passing metric (F1/AUPRC/correlation) **and** balanced accuracy
for comparability; both null floors; an output-side (intervention) score; a
label-uniqueness flag; a stability/confidence estimate over ≥3 seeds; and the
decile-stratified sampling regime and denominator it was all computed on —
with a human-audited subsample sized by importance sampling calibrating the
whole thing against the 0.87 human–human ceiling.

Anything less is reporting a number in the 0.51–0.81 band with no way to tell
whether it came from a trained dictionary.
