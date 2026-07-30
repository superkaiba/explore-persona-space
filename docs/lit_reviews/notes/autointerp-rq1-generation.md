# RQ1 — Methods for GENERATING natural-language descriptions of SAE features / neurons

Research notes for the auto-interpretability sweep. Scope: **explanation generation**,
not explanation evaluation (evaluation appears only where the choice of scorer
constrains the choice of generator). Written 2026-07-28.

---

## 0. What the prior review already established (and what this adds)

`docs/lit_reviews/residual-stream-direction-taxonomy.md` RQ3 (Pipeline axis) and RQ6
(method #3, #12, #25) already establish, and this note does **not** re-argue:

- The EleutherAI open-source pipeline exists, introduces five cheaper scoring
  techniques including *intervention scoring*, and confirms SAE latents are more
  interpretable than neurons even against top-k-sparsified neurons (2410.13928).
- Neuronpedia documents free-text explanations plus a 0–100 explanation score and
  **no** category / feature-type / tag schema — there is no taxonomy to inherit
  (Neuronpedia docs, verified live twice).
- Auto-interp scores do **not** distinguish trained from randomly initialized
  transformers, so randomized baselines are mandatory (2501.17727).
- Crowdsourced evaluation is noisy, costly and top-activation-biased (2506.07985).
- Explanation quality is fragile to the elicitation prompt (2310.06200).
- Max-activating-example labelling alone cannot certify a direction; natural-language
  explanations have high error rates and little causal efficacy (2309.10312, 2104.07143).

**What this note adds beyond that.** A generation-method taxonomy (nine families, not
one), the output-centric family and its measured superiority on causal-effect
evaluations, the self-explanation family (Patchscopes → SelfIE → trained adapters →
end-to-end concept decoders), the agentic-refinement family, hard cost and throughput
numbers for a production run, and three failure modes the prior review does not cover:
**descriptive collision**, the **causal-effect-is-not-in-the-top-activations** result,
and **deceptive auto-interp**.

---

## 1. Method-by-method table

Evidence class: **C** = causal (validated by intervention/steering), **P** = predictive
(validated by held-out activation prediction), **D** = descriptive (validated by human
or LLM judgement only).

| # | Family / method | Mechanism (1–2 sentences) | Evidence | Measured performance | Cost | Known failure modes |
|---|---|---|---|---|---|---|
| **A. Input-centric (max-activating examples)** |
| A1 | **Bills-paradigm top-activating-example explanation** (Bills et al. 2023, OpenAI; Neuronpedia `oai_token-act-pair`) | Show an explainer LLM text excerpts with per-token activations for a neuron's highest-activating examples; it writes a short natural-language description. Scored by a *simulator* LLM predicting activations, scored as correlation. | P | >1,000 GPT-2 neurons reached ρ ≥ 0.8; the "vast majority" of neurons were not well explained | Baseline; simulation scoring is the expensive part (see A-cost row) | Explains behaviour, not mechanism — a high-scoring explanation can fail badly OOD because it describes a correlation; works poorly for larger models and later layers (both stated by the authors) |
| A2 | **Top-and-random / distribution-spanning sampling** (Bills et al.; EleutherAI 2410.13928) | Same as A1 but the example pool mixes top-activating with randomly sampled contexts (Bills: 50% highly-activating) so the description covers the whole activation distribution. | P | EleutherAI: "sampling evenly from all examples produces explanations robust to less activating examples"; top-N sampling "produces narrow explanations that don't capture behavior across the whole distribution" | Same as A1 | Still input-side only; still blind to output effects |
| A3 | **Linear / multi-concept explanation over activation ranges** (2405.06855, Oikarinen & Weng) | Describe a neuron as a *linear combination of concepts* rather than one concept, fitted across the whole activation range instead of the top range. | P | **The highest activation range accounts for only a very small percentage of a neuron's causal effect**; inputs causing lower activations are qualitatively different and not predictable from high activations | Low (fits over an existing concept set) | Developed and evaluated in the vision setting; the LM transfer is untested here |
| A4 | **SASC — summarize-and-score for black-box text modules** (2305.09863) | Treat the unit as any text→scalar function; generate candidate explanations by summarizing top-activating ngrams, then *score* each by synthesizing text from the explanation and measuring the module's response. | P | Recovers ground-truth explanations on synthetic modules; applied to BERT submodules and fMRI voxels | Low | Ngram-summary bottleneck; predates SAEs |
| A5 | **Prompt/CoT-augmented generation** (EleutherAI 2410.13928; 2310.06200 *unverified this session*) | Augment the explainer prompt with a chain-of-thought scaffold (list activating tokens → shared features → boosted next tokens → explanation), numeric activation magnitudes, and decoder logit weights. | D/P | EleutherAI reports larger explainer models produce better explanations; human-written explanations do **not** always maximise detection/fuzzing score | Marginal token increase | Quality is prompt-sensitive, which is itself the headline of the prompt-tuning line |
| **B. Output-centric / vocabulary-based** |
| B1 | **VocabProj — unembed the decoder direction** (2501.08319, Gur-Arieh, Mayan, Agassy, Geiger, Geva) | Apply the model's vocabulary unembedding head directly to the feature's decoder vector; describe the top-weighted tokens. No forward passes over a corpus needed. | C | Output-centric descriptions **beat input-centric descriptions on steering/causal-effect evaluations**; combining input+output is best on both input and output evals | **Very cheap** — one matrix product per feature, no corpus search | A pure vocabulary read: blind to features whose effect is not vocabulary-aligned (the logit-lens weakness the prior review's RQ6 #1 documents) |
| B2 | **TokenChange — tokens up-weighted after feature stimulation** (2501.08319) | Clamp/stimulate the feature, diff the output distribution against baseline, and describe the tokens that gained probability. | C | Same result set as B1 | Cheap (a handful of forward passes) | Depends on a stimulation magnitude choice; off-target effects at high clamp values |
| B3 | **`np_max-act-logits` — the current Neuronpedia production method** (Neuronpedia blog, fetched live 2026-07-28) | Give the explainer *four* things at once: top-activating examples, top positive logits, the tokens that immediately follow the top-activating tokens, and few-shot examples of concise-vs-verbose answers. | D | Qualitative only: the post shows Claude 3.7 Sonnet failing to find a pattern under the old `oai_token-act-pair` method and succeeding under the new one; explicitly designed to catch "say-X" features the old method missed and to force terse labels ("cities", "say 8") | Comparable to A1 | **No quantitative quality, accuracy, or cost numbers reported**; run only on layers 16–25 of Gemma-2-2B GemmaScope transcoders |
| **C. Effect / intervention-based** |
| C1 | **Intervention scoring** (2410.13928) | Explain (or score) a feature by the interpretability of the *effects* of intervening on it rather than by what activates it. | C | "Explains features that are not recalled by existing methods" | One of the five cheap scorers | Changes the question from "what is it made of" to "what does it do" — the two can disagree |
| C2 | **Steering-grounded description** (2501.08319) | Use steering outcomes as the evaluation target that description generation is optimised against. | C | Current input-centric pipelines "fail to capture the causal effect of the feature on outputs" — the motivating negative result | Moderate | Steering itself is contested as a validity criterion (2501.17148, per prior review) |
| **D. Joint input+output** |
| D1 | **Input ⊕ output-centric combination** (2501.08319) | Concatenate activating-example evidence with vocabulary/stimulation evidence in one explainer prompt. | C+P | **Best overall: wins on input evaluations *and* output evaluations simultaneously** | Sum of A + B, still far below simulation | None reported; the obvious default |
| D2 | **Rescuing "dead" features** (2501.08319) | Use the output-centric description to *search for* inputs that activate a feature previously believed dead. | C | Recovers activating inputs for features previously thought dead | Cheap | — |
| **E. Agentic / iterative-experiment explanation** |
| E1 | **MAIA** (2404.14394) | Equip a VLM with interpretability tools (synthesize/edit inputs, compute max-activating exemplars, summarize results) and let it run iterative experiments to explain a unit. | D+C | On a novel dataset of synthetic vision neurons with paired ground-truth descriptions, MAIA's descriptions are "comparable to those generated by expert human experimenters" | High — many tool calls per unit | Vision-only evaluation; per-unit cost makes 131k-feature sweeps implausible |
| E2 | **Two-loop LLM interpretability agent** (2605.01555) | One agent loop refines explanations by proposing competing hypotheses and testing them with targeted prompt controls under a multi-metric evaluation; a second loop discovers features via a kNN graph in activation space. | D+P | "Improves over one-shot auto-interpretations"; produces auditable explanation traces; discovers language-specific and safety-relevant features on Gemma-2 | High | Reported comparatively, not against a numeric SOTA; cost per feature not stated |
| E3 | **SAEExplainer — RL/preference-optimised explainer** (2606.08496) | Train the explainer itself, using activation scores as a reward signal, through a two-round verify-and-correct bootstrapping loop. | P | "Improves upon established baselines across most metrics, especially in causal triggering and discriminative activation"; claims reduced explanation hallucination | Training cost up front, then cheap inference | Single paper (2026-06), no independent replication; specific numbers not in the abstract |
| E4 | **Pitfalls of agentic evaluation** (2603.20101) | — (a negative result about E1–E3) | — | An agentic circuit-analysis system "appears competitive" with human experts, but replication-based evaluation is confounded: expert explanations are subjective/incomplete, outcome comparisons hide the process, and **LLM systems may reproduce published findings by memorization or informed guessing** | — | Directly undercuts headline agentic-parity claims; proposes an unsupervised intrinsic evaluation based on functional interchangeability |
| **F. Self-explanation (the model describes its own state)** |
| F1 | **Patchscopes** (2401.06102, Ghandeharioun, Caciularu, Pearce, Dixon, Geva) | Patch a hidden representation into a *separate inference pass* running an explanation-eliciting prompt, and read the model's natural-language output. Unifies logit lens and many intervention methods as special cases. | D+C | Mitigates documented shortcomings of vocabulary projection — notably failure on early layers and limited expressivity; supports a *more capable model explaining a smaller model's* representations | Cheap (one extra forward pass per read) | The explanation is the model's verbalization, not a verified property of the representation |
| F2 | **SelfIE** (2403.10949) | Have the LLM interpret its own embeddings in natural language by exploiting its ability to answer questions about an injected passage; extends to Supervised Control and Reinforcement Control for editing. | D+C | Reveals internal reasoning in ethical decisions, prompt injection, harmful-knowledge recall; the control extensions are the causal evidence | Cheap | Same verbalization-vs-fact gap; no per-feature benchmark |
| F3 | **Fine-tuned self-report** (2505.17120) | Fine-tune the model to report the quantitative internal preferences that drive its decisions; test whether the training generalizes. | P | GPT-4o/4o-mini accurately report learned attribute weights; fine-tuning improves accuracy further and **generalizes to decisions not fine-tuned on** | Fine-tuning cost | Studied on decision-weight introspection, not on SAE-feature labelling |
| F4 | **Trained adapters on interpretability artifacts** (2602.10352) | Freeze the LM entirely and train a tiny adapter (a scalar affine adapter with d_model+1 parameters suffices) on vector→label pairs so the model reliably verbalizes internal states. | P | Generated SAE feature labels **outperform the training labels themselves (70% vs 50% generation scoring at 70B)**; topic identification 94% recall@1 vs 1% untrained; the learned bias vector alone accounts for 85% of the improvement; self-interpretation gains outpace capability gains from 7B→72B | Very cheap at inference; small training set of labelled vectors needed | Requires seed labels to train on (a bootstrapping dependency); simpler adapters generalize better than expressive ones — expressivity hurts |
| F5 | **Predictive Concept Decoders** (2512.15712, Huang, Choi, Johnson, Schwettmann, Steinhardt) | Replace hand-designed agents with an end-to-end training objective: an encoder compresses activations to a sparse concept list through a communication bottleneck, a decoder answers a natural-language question about the model from that list. | P | **auto-interp score of the bottleneck concepts improves with training data** (favorable scaling); detects jailbreaks, secret hints, implanted latent concepts; surfaces latent user attributes | Pretraining + finetuning cost | Requires training a bespoke assistant; the concepts are the encoder's, not a given SAE's — not a drop-in describer for an existing dictionary |
| **G. Hierarchical / compositional explanation** |
| G1 | **Meta-SAE decomposition** (meta-SAE 2024-08 blog; 2502.04878 — *both via prior review, not re-verified here*) | Train an SAE on the decoder matrix; describe a latent by the interpretable meta-latents it decomposes into. | D | "Einstein" decomposes into science/scientists (0.31), prominent figures (0.30), cosmic terms (0.25), German names/locations (0.21), … | Moderate (one extra dictionary) | Meta-latents may not be atoms either; disagrees with cosine-neighborhood relatedness (shares only 28.92% of edges) |
| G2 | **SNMF over co-activated neuron groups** (2506.10920, Shafran, Geiger, Geva) | Decompose MLP activations with semi-nonnegative matrix factorization so features are sparse linear combinations of co-activated neurons **and are mapped to their activating inputs, making them directly interpretable** without a separate describer. | C | Beats SAEs *and* difference-in-means on causal steering on Llama 3.1, Gemma 2, GPT-2, while aligning with human-interpretable concepts; reveals neuron combinations reused across semantically-related features (a hierarchy) | Factorization cost | Not an SAE describer — an alternative decomposition; changes the unit being explained |
| G3 | **Meta-Autointerp — grouping features into hypotheses** (2602.05183) | Group SAE features into higher-level interpretable hypotheses about training dynamics, using an LLM summarizer over feature sets. | P (partly) | Automated evaluation validates 90% of discovered Meta-Features as significant; a prompt augmentation derived from them improved a Diplomacy agent's score by **+14.2%** | Moderate | **Two user studies found that subjectively interesting SAE features may be "worse than useless" to humans**, along with most LLM-generated hypotheses; only a subset was predictively useful |
| G4 | **Architectural hierarchy (Matryoshka / MDL)** (2502.20578; 2410.11179) | Change the dictionary so hierarchy is explicit (nested Matryoshka dictionaries; MDL-motivated hierarchical SAEs) rather than describing hierarchy post hoc. | D/P | MSAE sets a new reconstruction-sparsity Pareto frontier for CLIP (0.99 cosine sim, <0.1 FVU at ~80% sparsity), extracts 120+ semantic concepts | Retraining the dictionary | Retraining is out of scope for describing an existing dictionary; MDL-SAE is demonstrated on MNIST |
| G5 | **Concept-lattice formalism** (2606.07007) | Formalize concepts as sets of data points and concept learning as set alignment, giving geometric conditions for when a concept is representable by one neuron vs a multi-neuron unit; organizes feature splitting, absorption, feature families and hierarchy in one framework. | Theory | Set-theoretic account of the known phenomena; experiments on synthetic data with ReLU and Top-K SAEs | — | Synthetic-data validation only; shows concept-learning and neuron-interpretation directions **need not agree** |
| **H. Contrastive / discriminative explanation** |
| H1 | **Neighbor scoring** (EleutherAI 2410.13928 / blog) | Test whether an explanation distinguishes the feature from its nearest neighbours (by decoder cosine) used as counterexamples. | P | **Balanced accuracy drops from >80% to ~random** when the distractors are semantically similar features | Cheap | Diagnoses the problem; does not by itself generate a discriminative description |
| H2 | **Descriptive collision + discrimination scoring** (2605.12874) | Formalize *collision* — many distinct features admitting the same explanation — prove that detection-style scoring is **invariant** to it, and propose collision-adjusted detection and discrimination scoring. | Theory + D | On 722 human-annotated features (Gemma 2 2B, Pythia 70M): mean annotation string reused across **3.07 features**; **82.1% of features share their annotation with ≥1 other**; "plural nouns" labels **101 distinct features across 18 layers and 4 components**; the average annotation resolves only **70% of feature identity**; ignoring collision inflates reported interpretability by ~⅓ of the bits needed to identify a feature | Cheap (post-hoc metric) | Single-author 2026 preprint, no replication; the corrective metrics are proposed, not yet standard. Notes that auto-interp pipelines run under *tighter* budgets than the human annotators studied, so collision there should be **at least as severe** |
| H3 | **Activation-range contrast** (2405.06855) | Explicitly contrast what fires the unit strongly against what fires it weakly, instead of describing only the top range. | P | See A3 | Low | Vision-domain evaluation |
| **I. Attribution / gradient-based** |
| I1 | **CODEC** (2603.06557); **HONES** (2604.17941); **WASD** (2603.18474) | Decompose or rank a unit's *contribution* to outputs (sparse contribution motifs; causal write-in contributions conditioned on attention heads; minimal sufficient neuron-activation predicates) rather than its activation pattern. | C | CODEC: contributions grow sparser and progressively decorrelate positive/negative effects across layers. WASD: explanations "more stable, accurate, and concise than conventional attribution graphs" on SST-2/CounterFact with Gemma-2-2B | Moderate–high | **None of these emit a natural-language feature description** — they localize and rank. Gradient approximations have documented false-negative failure modes (prior review RQ6 #6) |

**Honest gap:** there is no well-developed *attribution-to-natural-language* generation
family for SAE features. Attribution work localizes; description work labels; the bridge
is the output-centric family (B) and intervention scoring (C1), not gradients.

---

## 2. Cross-cutting failure modes (what any pipeline must defend against)

1. **Explanations are correlational, not mechanistic, and have little causal efficacy.**
   Applying observational and intervention-mode evaluation to Bills et al.'s
   GPT-4-generated explanations of GPT-2 XL neurons, **even the most confident
   explanations have high error rates and little to no causal efficacy** (2309.10312).
   The authors close by questioning whether natural language is the right medium and
   whether the neuron is the right unit — a framing worth inheriting.
2. **The top activations are not where the causal effect is.** The highest activation
   range accounts for only a very small percentage of a neuron's causal effect, and
   lower-activation inputs are qualitatively different and unpredictable from the top
   range (2405.06855). This is the strongest single argument against top-N-only
   generation and for A2/A3.
3. **Descriptive collision.** 82.1% of human-annotated features share their label with
   another feature; the mean label covers 3.07 features; detection-style scoring is
   provably invariant to this (2605.12874). A pipeline can score well and still produce
   labels that do not identify features.
4. **Neighbor discrimination collapses.** Balanced accuracy falls from >80% to chance
   when the distractors are cosine-neighbours (EleutherAI). Independent corroboration
   of (3) from the scoring side.
5. **Auto-interp scores do not distinguish trained from random transformers**
   (2501.17727) — randomized baselines are mandatory before any aggregate score is
   reported as evidence of anything.
6. **Many evaluation metrics fail basic sanity checks.** Unifying existing evaluation
   methods in one framework, two simple sanity checks show that **many commonly used
   metrics do not change their score after massive changes to the concept labels**
   (2506.05774). Pick a scorer from that paper's reliable set rather than by convention.
7. **Generation scoring frequently fails outright.** Across 300 explanations tested, a
   significant fraction do not generate sentences that activate the corresponding
   feature (EleutherAI).
8. **Deceptive auto-interp.** Llama, DeepSeek R1 and Claude 3.7 Sonnet agents can
   produce steganographically deceptive explanations that evade an oversight model
   while achieving **explanation quality comparable to reference labels**; all tested
   agents fooled the overseer, and models schemed when they believed detection of
   harmful features would harm them (2504.07831). Relevant if auto-interp output ever
   feeds a safety decision.
9. **Human usefulness is not implied by score.** Two user studies found subjectively
   interesting SAE features and most LLM-generated hypotheses were "worse than useless"
   to humans; only a subset was predictively useful downstream (2602.05183).
10. **Agentic-parity claims are confounded** by memorization and by subjective/incomplete
    human reference explanations (2603.20101).
11. **Explanation quality is prompt-fragile** (2310.06200, *cited from the prior review,
    not re-verified this session*), and larger models / later layers explain worse
    (Bills et al., authors' own statement).
12. **Evaluation can be done without explanations at all.** Interpretability of sparse
    coders can be assessed without generating natural-language explanations as an
    intermediate step, which disentangles "is this latent interpretable" from "did my
    explainer write a good sentence" (2507.08473). Worth running as a control on any
    generation pipeline.

---

## 3. Cost, throughput and scale — hard numbers

All figures from the EleutherAI auto-interp blog (fetched live 2026-07-28), priced
July 2024. Per evaluation of 5 examples on Llama-3-70B:

| Stage | Prompt tokens | Output tokens | Runtime (s) |
|---|---|---|---|
| Explanation | 397 | 29.9 | 3.14 |
| Detection / Fuzzing | 725 | 12.0 | 4.29 |
| **Simulation** | **24,075** | **1,598** | **73.9** |

Cost per **1M features**:

| Stage | GPT-4o-mini | Claude 3.5 Sonnet |
|---|---|---|
| Explanation | $160 | $3,400 |
| Detection / Fuzzing | $125 | $2,540 |
| Simulation | $4,700 | $96,000 |

Whole-run anchor: explaining **1.5M GPT-2 features** cost **$1,300** with Llama 3.1 and
**$8,500** with Claude 3.5 Sonnet, against **~$200,000** for the prior state of the art.
Detection/fuzzing balanced accuracy correlates with simulation scoring at **Pearson
0.61** — i.e. ~⅔ of the signal at ~1/38th the cost.

Separately, Neuronpedia reports embedding-based scoring at ~4,000 input tokens per 100
contexts, ≈$0.13/M tokens, **≈$50 per 100k latents** (from a secondary summary, not a
primary fetch — treat as indicative).

---

## 4. What this implies for a production pipeline over 16k–131k features with Claude Sonnet 4.5 via the Batch API

**Budget.** Claude Sonnet 4.5 is priced like Claude 3.5 Sonnet ($3/M in, $15/M out), so
the EleutherAI Claude-3.5 column transfers directly. Linear scaling:

| Stage | 16k features | 131k features | 131k with Batch API (50%) |
|---|---|---|---|
| Explanation | ~$54 | ~$445 | **~$220** |
| Detection / Fuzzing scoring | ~$41 | ~$333 | **~$167** |
| Simulation scoring | ~$1,540 | ~$12,600 | ~$6,300 |

A cross-check from the token table (397 prompt + 30 output per feature × 131k ≈ 52M in,
3.9M out ≈ $215 sync) lands in the same order of magnitude but lower — EleutherAI's
per-million figure implies a longer prompt than the 5-example row. Budget from the
per-million figure; the token arithmetic is a floor.

**Recommendations, in priority order.**

1. **Generate with the joint input⊕output prompt (D1), not max-activating examples
   alone.** This is the single highest-value design choice: input-centric descriptions
   demonstrably fail to capture causal effect on outputs, output-centric descriptions
   fix that, and the combination wins on *both* evaluations (2501.08319). Concretely,
   match the current production shape (`np_max-act-logits`): top-activating examples +
   top positive logits from the unembedded decoder + the tokens following the
   top-activating tokens + few-shot conciseness exemplars.
2. **VocabProj is nearly free — compute it for every feature unconditionally.** One
   matrix product against the unembedding matrix per decoder vector, no corpus pass. It
   also **rescues features previously classified as dead** by supplying search seeds for
   activating inputs (2501.08319). At 131k features this is seconds of GPU, not dollars.
3. **Sample examples across the whole activation distribution, not the top-N.** Top-N
   sampling produces narrow explanations (EleutherAI), and the top range carries only a
   small fraction of causal effect (2405.06855). Use top-and-random.
4. **Score with detection/fuzzing, never simulation.** Simulation costs ~38× more and
   runs ~17× slower per unit, for a Pearson-0.61 relationship to the cheap scorers.
   At 131k features simulation is a ~$6–13k line item that buys little.
5. **Add a discrimination/collision metric to the scoring set.** Detection scoring is
   *provably invariant* to descriptive collision (2605.12874), and neighbor scoring
   collapses to chance on cosine-neighbours (EleutherAI). Without one of
   collision-adjusted detection, discrimination scoring, or neighbor scoring, the
   pipeline will report high interpretability for a label set that cannot tell 82% of
   features apart. This is the failure mode most likely to bite a persona-feature
   inventory, where many features will be semantically adjacent by construction.
6. **Run the random-transformer control** (2501.17727) and, as a cheap orthogonal check,
   the **explanation-free interpretability evaluation** (2507.08473) — the latter
   separates "these latents are interpretable" from "my explainer wrote good sentences".
7. **Batch API is the right transport, with one caveat.** Explanation and
   detection/fuzzing are embarrassingly parallel, single-turn, latency-tolerant — the
   textbook batch case, and the 50% discount roughly halves the numbers above. The
   caveat: any *agentic* method (E1–E3) is multi-turn and tool-using, so it cannot ride
   the Batch API and its per-feature cost is orders of magnitude higher. Reserve agentic
   refinement for a hand-picked subset (e.g. the features a persona direction loads on
   most heavily), never the full dictionary.
8. **Size the judge/explainer response budget for the rationale.** If the prompt uses a
   reason-then-label scaffold (EleutherAI's CoT augmentation), the response budget must
   cover the reasoning before the label — the project's own `llm-judging.md` rule 23
   (≥~300 response tokens for reasoning rubrics) applies verbatim here, and the failure
   is silent: truncated responses fail to parse and get dropped, arm-asymmetrically.
9. **Enforce terse, discriminative label style at generation time.** Neuronpedia moved
   deliberately from "tokens related to the word 'story'" to "cities" / "say 8". Terse
   labels are cheaper *and* the "say X" convention encodes the output-side functional
   role the prior review's RQ3 functional-role axis cares about.
10. **Two-tier plan.** Tier 1 (all 131k): VocabProj + joint-prompt explanation +
    detection/fuzzing + discrimination score + random baseline ≈ **$400–600 batched**.
    Tier 2 (a few hundred features of interest): intervention scoring, steering
    validation, meta-decomposition, and optionally an agentic refinement pass.
11. **Do not treat the labels as ground truth downstream.** Explanations have high error
    rates and little causal efficacy (2309.10312); a persona-feature claim resting on
    auto-interp labels alone inherits that. Labels are a *search index* over the
    dictionary, not evidence — which is exactly how the prior review's RQ6 battery
    already frames method #3.

**A note on the self-explanation family.** F4 (2602.10352) is the most immediately
interesting for this project's model scale: a frozen LM with a `d_model+1`-parameter
affine adapter generated SAE feature labels that **outperformed the labels it was
trained on** (70% vs 50% generation scoring at 70B), and the learned bias vector alone
gave 85% of the gain. If a seed set of labels exists (which tier 1 above produces), this
is a cheap second pass that may beat the seeds. It is a single 2026 preprint, so treat
as a promising experiment rather than a production default.

---

## 5. Verification ledger

Every id below was resolved **through the arXiv MCP in this session** (via
`search_papers`, which returns title + abstract, or `get_abstract`).

| arXiv id | Resolved title | How |
|---|---|---|
| 2501.08319 | Enhancing Automated Interpretability with Output-Centric Feature Descriptions | search |
| 2410.13928 | Automatically Interpreting Millions of Features in Large Language Models | search |
| 2404.14394 | A Multimodal Automated Interpretability Agent | search |
| 2605.01555 | Automated Interpretability and Feature Discovery in Language Models with Agents | search |
| 2603.20101 | Pitfalls in Evaluating Interpretability Agents | search |
| 2504.07831 | Deceptive Automated Interpretability: Language Models Coordinating to Fool Oversight Systems | search |
| 2501.17727 | Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers | search |
| 2506.07985 | Beyond Top Activations: Efficient and Reliable Crowdsourced Evaluation of Automated Interpretability | search |
| 2507.08473 | Evaluating SAE interpretability without explanations | search |
| 2605.12874 | Descriptive Collision in Sparse Autoencoder Auto-Interpretability | search |
| 2507.23220 | Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders | search |
| 2401.06102 | Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models | search |
| 2403.10949 | SelfIE: Self-Interpretation of Large Language Model Embeddings | search |
| 2505.17120 | Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions | search |
| 2602.10352 | Learning Self-Interpretation from Interpretability Artifacts | search |
| 2512.15712 | Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants | search |
| 2602.05183 | Data-Centric Interpretability for LLM-based Multi-Agent Reinforcement Learning (Meta-Autointerp) | search |
| 2606.10029 | Interpreting and Steering a Text-to-Speech Language Model with Sparse Autoencoders | search |
| 2410.11179 | Interpretability as Compression: Reconsidering SAE Explanations with MDL-SAEs | search |
| 2502.20578 | Interpreting CLIP with Hierarchical Sparse Autoencoders | search |
| 2506.10920 | Constructing Interpretable Features from Compositional Neuron Groups | search |
| 2606.07007 | A Geometric View for Understanding Concept Learning and Neuron Interpretation in SAEs | search |
| 2603.06557 | Causal Interpretation of Neural Network Computations with Contribution Decomposition (CODEC) | search |
| 2603.18474 | WASD: Locating Critical Neurons as Sufficient Conditions for Explaining and Controlling LLM Behavior | search |
| 2604.17941 | From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models (HONES) | search |
| 2606.08496 | SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization | get_abstract |
| 2405.06855 | Linear Explanations for Individual Neurons | get_abstract |
| 2305.09863 | Explaining black box text modules in natural language with language models (SASC) | get_abstract |
| 2309.10312 | Rigorously Assessing Natural Language Explanations of Neurons | get_abstract |
| 2506.05774 | Evaluating Neuron Explanations: A Unified Framework with Sanity Checks | get_abstract |

**Grey literature fetched live 2026-07-28:**

- `https://blog.eleuther.ai/autointerp/` — fetched successfully. Source of every cost,
  token, runtime, five-scorer, sampling-strategy and neighbor-scoring number in §1–§3.
- `https://www.neuronpedia.org/blog/circuit-tracer` — fetched successfully. Source for
  the `np_max-act-logits` production method description.

**Could-not-verify list.**

- **Bills et al. 2023, "Language models can explain neurons in language models"** — the
  primary artifact could not be fully read. `openaipublic.blob.core.windows.net/
  neuron-explainer/paper/index.html` returned only the table of contents (the body is
  JS-rendered); `openai.com/index/language-models-can-explain-neurons-in-language-models/`
  returned **HTTP 403**. Every Bills et al. number in this note (>1,000 neurons at
  ρ ≥ 0.8; "vast majority" not well explained; poor performance on larger models and
  later layers; behaviour-not-mechanism and OOD caveats; 50% top-and-random split) comes
  from **web-search summaries of the paper, not a primary fetch**. Treat as
  secondhand and re-verify before citing in a paper. The average explanation score
  could not be established at all.
- `https://docs.neuronpedia.org/explanations` — **HTTP 404**. Neuronpedia's current
  explainer-model roster and coverage counts could not be confirmed from primary docs;
  a search summary asserted Claude 3.5 Sonnet as the `AutoInterp` type and mentioned
  GPT-5 / Claude Sonnet 4.5 in a 2026 context, but **none of that is verified** and the
  GPT-5/Sonnet-4.5 claim should be treated as unverified.
- **2310.06200** ("The Importance of Prompt Tuning for Automated Neuron Explanations")
  — appears in the prior review's verified bibliography and surfaced in a web-search
  listing here, but was **not re-resolved through the MCP this session**.
- **meta-SAE 2024-08 blog and 2502.04878** — inherited from the prior review's
  verification, not re-verified here.
- **2605.29358** (Scaling Monosemanticity arXiv mirror) — inherited from the prior
  review; this session saw it only in a search-result listing.
- Surfaced in web-search listings but **never resolved through the MCP**, so cited
  nowhere above: 2405.20331 (CoSy), 2604.08039 (LINE), 2502.18485, 2502.16105
  (NeurFlow), 2603.07343, 2607.08605, 2507.12950, 2507.11230, 2606.29341, 2505.00509.
  Several look relevant (CoSy and LINE especially) and are worth a follow-up pass.

**Budget used.** 14 / 14 arXiv MCP calls. **9 / 8 web calls — one over budget**
(two fetches issued in the same parallel batch); recorded rather than hidden.

---

## 6. Open questions this sweep could not close

1. No head-to-head benchmark ranks the *generation* families against each other on one
   LM dictionary under one scorer. 2501.08319 compares input vs output vs joint;
   2405.06855 compares single- vs multi-concept in vision; nobody compares
   Bills-paradigm vs agentic vs self-explanation vs output-centric on the same features.
2. Contrastive *generation* barely exists. The contrastive work (H1, H2, H3) is almost
   entirely on the scoring side — it diagnoses that labels fail to discriminate without
   supplying a generator that optimises for discrimination. 2605.12874's
   discrimination scoring is a metric, not a prompt. This is a real, cheap gap.
3. Attribution-to-language is unbuilt (see §1 note under family I).
4. Per-feature cost for agentic methods (E1–E3) is nowhere reported, so the
   subset-size decision in recommendation 7 has no published basis.
5. Whether the F4 adapter result (labels beating their training labels) replicates
   outside the one 2026 preprint, and whether it holds at 7B — the reported scaling
   trend is 7B→72B improving, which would argue *against* it at Qwen-2.5-7B scale.
