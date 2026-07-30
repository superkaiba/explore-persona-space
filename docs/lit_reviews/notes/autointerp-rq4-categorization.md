# RQ4 — How the literature CATEGORIZES features, and what makes a feature taxonomy reliable

Research notes for the auto-interpretability literature sweep. Scope: categorization
(assigning a feature to a class in a scheme) as distinct from description (writing free
text about a feature). Written for synthesis into a review + a production pipeline that
must classify 16k–131k SAE features along several axes.

**Motivating failure this RQ must answer to.** A prior in-house attempt used a single
binary `persona-related?` judge field. Of 40 positively-labelled features, 20 were plain
language features, 11 register/style, 9 genuine identity — and the headline contrast
vanished on the identity subset. Every recommendation below is oriented at that failure
mode: a superordinate binary over a multi-axis construct, with no forced choice among
confusable siblings, no measured reliability, and no external reference.

---

## 1. Published category sets, side by side

No published scheme classifies features on a single axis. Every mature taxonomy is
multi-axis, and the axes that recur are **functional role (input/abstract/output)**,
**abstraction level**, and **layer position** — with semantic topic treated as a separate,
subordinate axis.

| Source | Unit classified | Category set | Axis type |
|---|---|---|---|
| Circuit tracing / *On the Biology of a Large Language Model* (transformer-circuits.pub, 2025) | cross-layer transcoder features in attribution graphs | **"Say X" features** (push the model to emit specific tokens, e.g. "say Austin", "say a capital"); **concept features** (low-level words/phrases → high-level sentiments, plans, reasoning steps); **planning features**; **language-specific** vs **multilingual/language-agnostic**; **early/mid/late-layer**; **error nodes** (non-interpretable residual) | functional + language + depth |
| *Dense SAE Latents Are Features, Not Bugs* (2506.15679) | dense SAE latents | **position tracking, context binding, entropy regulation, letter-specific output signals, part-of-speech, principal-component reconstruction** | functional families |
| *Universal Neurons in GPT2* (2401.12181) | neurons universal across seeds (1–5% of all) | taxonomized into "a small number of neuron families"; separately, **universal functional roles**: deactivating attention heads, changing next-token entropy, predicting / suppressing a token set | functional roles |
| *Interpreting Attention Layer Outputs with SAEs* (2406.17759) | attention-output SAE features | **long-range context, short-range context, induction** (sub-split into long-prefix vs short-prefix) | functional families |
| *Disentangling Dense Embeddings with SAEs* (2408.00657) | SAE features over text embeddings | **"feature families"** — related concepts at varying levels of abstraction, identified by a dedicated method | hierarchical/abstraction |
| *A Survey on Sparse Autoencoders* (2503.05613) | SAE feature explanations | splits the whole explanation literature into **input-based** vs **output-based** methods | functional (meta-level) |
| *SAEs Map Brain-LLM Alignment* (2605.23035) | GPT-2 XL / Llama-3.1-8B SAE features | a **human-validated taxonomy** whose top split isolates **semantic** features, with **five semantic subcategories** specified a priori from three independent neuroscience programs | semantic, hierarchical |
| Single-cell foundation-model atlas (2603.02952) | 82.5k + 24.5k SAE features | annotation against external ontologies (GO, KEGG, Reactome, STRING, TRRUST); 29–59% of features annotate at all | external-ontology grounding |
| *Scaling Monosemanticity* (transformer-circuits.pub, 2024) | Claude 3 Sonnet SAE features | manually-identified categories only (famous people, countries/cities/buildings, code type signatures, sarcasm, code errors, and safety-relevant: deception, power-seeking, sycophancy, bias) — **illustrative, no counts, no closed label set** | semantic, informal |

**Takeaways for scheme design.**

- The **input-side / output-promoting ("say-X") / operation** split the EPS scheme wants is
  the single most consistently attested functional axis in the literature — it appears
  independently in circuit tracing, dense latents (structural → semantic → output-oriented),
  universal neurons (prediction/suppression neurons), and the survey's input- vs
  output-based partition of the whole field.
- **Abstraction level co-varies with depth in every taxonomy that measured it.** Dense
  latents shift structural (early) → semantic (mid) → output-oriented (late) (2506.15679);
  circuit tracing puts language-specific features near input/output and language-agnostic
  ones mid-stack; the bio atlas reports U-shaped layer profiles for ontology annotation
  (2603.02952). Layer index is therefore a **free, non-judge covariate** to validate an
  abstraction-level axis against.
- The most-cited taxonomy (Scaling Monosemanticity) is *illustrative, not a closed label
  set* — the field has no off-the-shelf label inventory to copy. A bespoke scheme is
  required; the transferable asset is the axis structure, not the labels.

### 1b. The language / register / identity confusion is a documented, separable problem

This is exactly where the in-house attempt broke, and there is targeted work:

- **Language features are separable with a dedicated statistic.** *Unveiling
  Language-Specific Features in LLMs via SAEs* (2505.05111, Deng et al., ACL 2025)
  introduces a **monolinguality metric** over SAE features; ablating high-monolinguality
  features degrades performance in only one language. That metric is a ready-made,
  **non-judge discriminator** for the language axis — the exact axis that swallowed 20/40
  of the in-house positives.
- **Register is a genuinely distinct, language-agnostic object.** *A Universal Vibe?
  Finding and Controlling Language-Agnostic Informal Register with SAEs* (2603.26236)
  builds a dataset where target terms appear in both literal and colloquial contexts
  specifically **to separate pragmatic register from lexical awareness**, finds most
  informality information disperses across language-specific features but a small robust
  cross-linguistic core forms an informal-register subspace that sharpens with depth, and
  validates it causally: steering shifts output formality and **transfers zero-shot to six
  unseen languages**.
- The design lesson is strong: **language, register, and identity each have their own
  external validator** (monolinguality ablation; cross-lingual steering transfer; and for
  identity, whatever behavioural handle the project already trusts). A category boundary
  that only a judge can see is the boundary that failed.

---

## 2. Automated classification into predefined categories

### 2a. The blueprint: ADAG (2604.07615)

*ADAG: Automatically Describing Attribution Graphs* (Arora, Wu, Steinhardt, Schwettmann)
is the closest published thing to a production feature-categorization pipeline, and its
architecture is the one to copy:

1. **Attribution profiles** — quantify the *functional role* of a feature via its **input
   and output gradient effects**. This is a numeric, non-judge representation of exactly
   the input-side / output-promoting axis.
2. **A clustering algorithm** groups features on those profiles.
3. **An LLM explainer–simulator** generates *and scores* natural-language explanations of
   the **functional role of the feature groups** (not of individual features).

It recovers known human-analysed circuits and finds steerable clusters responsible for a
harmful-advice jailbreak in Llama-3.1-8B. The paper explicitly frames itself against the
status quo: all prior circuit tracing relied on "ad-hoc human interpretation of the role
that each feature plays, via manual inspection of dataset examples."

The transferable principle: **derive the functional axis from gradients, then let the LLM
name the cluster** — rather than asking an LLM to read top-activating examples and answer
"is this a persona feature?".

### 2b. Input-side evidence cannot determine the output axis

*Enhancing Automated Interpretability with Output-Centric Feature Descriptions*
(2501.08319, Gur-Arieh, Mayan, Agassy, Geiger, Geva) is the decisive result here. Using
steering evaluations they show **current pipelines produce descriptions that fail to
capture the causal effect of the feature on outputs**. Their fix uses tokens weighted
higher after feature stimulation, or top-weight tokens from applying the unembedding head
directly to the feature. Output-centric descriptions beat input-centric ones on output
evaluations; **combining both is best on both**. Output-centric descriptions also find
inputs for features previously believed "dead".

Implication: a "say-X / output-promoting" category built from top-activating examples is
not merely noisy, it is measuring the wrong thing. That axis must be fed logit-lens /
unembedding / stimulation evidence.

### 2c. Structured descriptions beat free text for consistency

*Semantic Regexes* (2510.06378, Boggust, Ren, Assogba, Moritz, Satyanarayan, Hohman)
introduces structured-language feature descriptions built from primitives (linguistic and
semantic patterns) plus modifiers (contextualization, composition, quantification), against
the stated problem that "natural language feature descriptions can be vague, inconsistent,
and require manual relabeling." They **match natural-language accuracy while being more
concise and more consistent**, and — the key point for this RQ — "their inherent structure
affords new types of analyses, including quantifying feature complexity across layers,
scaling automated interpretability from insights into individual features to model-wide
patterns." User studies find they help people build accurate mental models.

This is direct evidence for a **structured multi-field schema** over free-text description
plus a binary flag: the structure is what makes 131k features aggregable.

### 2d. Scale, and what scoring at scale looks like

*Automatically Interpreting Millions of Features* (2410.13928, Paulo, Mallen, Juang,
Belrose; EleutherAI) is the reference open pipeline. Five new cheaper scoring techniques;
notably **intervention scoring**, which evaluates the interpretability of the *effects* of
intervening on a feature and "explains features that are not recalled by existing methods."
They propose guidelines for explanations that stay valid over a broader set of activating
contexts, discuss pitfalls in existing scoring, and use explanations to measure semantic
similarity between independently trained SAEs (nearby residual-stream layers are highly
similar). Also relevant as a base-rate check: SAE latents are confirmed much more
interpretable than neurons, even top-k-sparsified neurons.

Other automated-taxonomy precedents: 2406.04028 proposes an automated feature taxonomy for
contrastive SAEs on chess agents (with explicit sanity checks against spurious
correlations); 2603.21014 (CLT-Forge) ships a unified automated interpretability pipeline
for cross-layer transcoders at scale.

---

## 3. Unsupervised alternatives (clustering, geometry, topic structure)

- **Co-occurrence clustering yields spatially localized functional groups.** *The Geometry
  of Concepts: Sparse Autoencoder Feature Structure* (2410.19750, Li, Michaud, Baek,
  Engels, Sun, Tegmark) finds three levels: "atomic" crystals (parallelograms/trapezoids,
  quality much improved by projecting out global distractor directions such as **word
  length**, via LDA); "brain"-scale **spatial modularity** where e.g. math and code
  features form a **lobe**, with clusters of co-occurring features clustering spatially far
  more than chance; and "galaxy"-scale anisotropy with a power law of eigenvalues, steepest
  in middle layers. Two directly usable lessons: (i) co-occurrence clustering is a valid
  unsupervised functional grouping; (ii) **global distractor directions (word length!)
  contaminate feature geometry and should be projected out** before any
  similarity/clustering step — a nuisance-control step with an obvious analogue for
  language and length nuisances in the persona setting.
- **Feature families / hierarchy.** 2408.00657 introduces a method for identifying "feature
  families" representing related concepts at varying abstraction levels. 2606.07007 gives a
  set-theoretic framework in which feature splitting, feature absorption, feature families,
  and hierarchical concepts all fall out, and — importantly — uses **formal concept
  analysis** to show that concept-learning and neuron-interpretation directions "need not
  agree" and that their **many-to-many structure can be organized by concept lattices**.
  That is a formal warning against assuming a clean one-feature-one-category map.
- **ADAG's clustering** (§2a) is the functional-role analogue: cluster on attribution
  profiles rather than on description embeddings.
- **Feature neighborhoods** (Scaling Monosemanticity, transformer-circuits.pub 2024):
  walking cosine-similarity neighborhoods across the 1M/4M/34M SAEs "consistently surfaces
  features that share a related meaning or context," with an interactive feature UMAP.
  Cheap, and useful as a label-propagation prior.

**How to use unsupervised structure.** The literature supports clustering as a **candidate
generator and a disagreement detector**, not as the label source: cluster first, classify
clusters (ADAG), and treat a feature whose judge label disagrees with its cluster's modal
label as an audit candidate. Nothing found measures cluster-vs-taxonomy agreement directly
— that is an open gap the EPS pipeline could fill cheaply.

---

## 4. Hierarchical and multi-label schemes

- **Hierarchy in SAEs is real but fragile.** *Do Sparse Autoencoders Learn Meaningful
  Concept Hierarchies?* (2606.22994) derives requirements for generalization/specialization
  hierarchies from semantic-net and taxonomy research, builds an evaluation protocol, and
  finds that while feature spaces "generally provide a basis for sensible hierarchies,
  establishing good hierarchical structure remains challenging" — with **feature absorption,
  in both its hard form and a continuous "soft" form, systematically compromising hierarchy
  quality.** 2506.01197 shows an SAE architecture that explicitly models a semantic
  hierarchy improves reconstruction and interpretability, so hierarchy can be built in, but
  that is an architecture change, not a labelling change.
- **Autoregressive LLMs do multi-label wrong, mechanically.** *Large Language Models Do
  Multi-Label Classification Differently* (2505.17510, Ma, Chochlakis, Maruthu Pandiyan,
  Thomason, Narayanan): the initial probability distribution for the first label "often does
  not reflect the eventual final output, even in terms of relative order", and **LLMs tend
  to suppress all but one label at each generation step**. Scale lowers entropy and raises
  single-label confidence (though internal relative ranking improves); SFT and RL **amplify**
  the effect. Their remedy: **take the max probability over all label-generation
  distributions** rather than the initial distribution — improving both distribution
  alignment and F1 at no extra compute.
  → Do **not** ask one call for a free-form multi-label set. Either force one label per
  axis, or score each candidate label independently.
- **Big label sets hurt, and hierarchy does not reliably rescue them.** *Multi-Label
  Requirements Classification with Large Taxonomies* (2406.04797) ran zero-shot over
  taxonomies of 250–1183 classes: sentence-based classifiers had significantly higher recall
  than word-based ones but no significant precision/F1 gain; **"the hierarchical
  classification strategy did not always improve performance"**; and **total and leaf node
  counts of the taxonomy have a strong negative correlation with the recall** of the
  hierarchical sentence-based classifier. 2010.01653 independently finds hierarchical
  Probabilistic-Label-Tree methods outperform flat label-wise attention on large multi-label
  sets, so the picture is genuinely mixed — but both agree label-set size is a first-order
  cost driver.
  → Keep each axis's label set small and flat; get expressiveness from **several small
  axes**, not one deep tree. This is precisely the multi-axis shape the EPS scheme already
  proposes, and the literature supports it.

---

## 5. Measured reliability of feature classification

The honest summary: **direct inter-rater reliability numbers for feature-classification
schemes are rare.** One clean data point, plus a set of adjacent results that bound how bad
things get.

- **The one positive existence proof.** 2605.23035 reports a **human-validated taxonomy with
  κ ≥ 0.74** over GPT-2 XL / Llama-3.1-8B SAE features (16k–32k per layer), with five
  semantic subcategories fixed **a priori** from independent prior programs rather than
  invented post hoc. Downstream, semantic features alone recover 94% of peak brain-encoding
  performance (r = 0.285) vs variance-matched baselines (p < 0.001, d = 1.31), and the
  category structure predicts cortical topography (Spearman ρ = 0.72, p < 0.001;
  hypergeometric p = 0.007). κ ≥ 0.74 is a reasonable target bar, and the recipe that got
  there — **a priori categories from an external theory + human validation** — is the
  transferable part.
- **Explanations are unstable under trivial perturbation.** *Corrupting Neuron Explanations
  of Deep Visual Features* (2310.16332): adding random noise of **σ = 0.02 changes the
  assigned concept of up to 28% of neurons** in deeper layers, and a designed corruption
  algorithm manipulates >80% of neurons' explanations by poisoning <10% of probing data.
  Test-retest under perturbation is not optional.
- **Most explanation-evaluation metrics fail basic sanity checks.** *Evaluating Neuron
  Explanations: A Unified Framework with Sanity Checks* (2506.05774, Oikarinen, Yan, Weng)
  unifies existing metrics mathematically, proposes two simple sanity checks, and shows
  **many commonly used metrics fail them — not changing their score after massive changes to
  the concept labels.** They publish guidelines and a set of reliable metrics. Read this
  before picking any scoring metric.
- **LLM-annotation reliability can fall below scientific thresholds.** 2304.11085 (Reiss)
  finds ChatGPT's zero-shot classification consistency falls short of reliability thresholds
  — **minor wording alterations in prompts, or repeating identical input, change outputs**;
  pooling repetitions helps; unsupervised use is "not recommended" without validation
  against human-annotated data.
- **Human interpretability itself is measurable and lower than assumed.** 2605.20337 runs
  two psychophysics protocols — **localizability** (can an observer predict where a feature
  fires?) and **nameability** (can they describe what it represents?) — over 13,400 quality-
  passing responses from 377 participants, with a chance-anchored scoring function. The two
  protocols yield strongly correlated rankings. Feature **locality** predicts
  interpretability. Useful as a template for a human audit set, and a caution that some
  features are simply not nameable.
- **Picking the right agreement coefficient is itself a decision.** 2603.06865 organizes IAA
  measures by task type (categorical / segmentation / subjective / continuous), discusses
  how **label imbalance and missing data distort reliability estimates**, and pushes
  confidence intervals plus explicit disagreement-pattern analysis. Relevant because a
  persona-identity axis will be heavily imbalanced (see base rates, §7), and κ is known to
  behave badly under skew.
- **Prompt-injection can corrupt auto-interp explanations.** 2312.03721 shows model-graded
  evaluations are susceptible to injections, and that "similar injections can be used on
  automated interpretability frameworks to produce misleading model-written explanations."
  Mostly relevant if feature evidence includes untrusted text.

---

## 6. Rubric design — what the LLM-as-judge literature says

- **Rubric scoring is implicitly a multiple-choice task with position bias** (2602.02219,
  Xu, Hirasawa, Kozuno, Ushiku). LLMs prefer score options at specific positions in the
  rubric list; the bias is consistent but **model-specific in direction** (some favour the
  first option, some the last). A second, **orthogonal** bias: **when a prompt scores
  several criteria simultaneously, the ordering of the criteria shifts the resulting
  scores.** Permuting option order attenuates it, and **a small number of random
  permutations suffices** for most models.
  → Two direct instructions: score each axis in its **own call** (do not bundle axes), and
  **permute the label order** across a few draws per feature.
- **Evaluation criteria are the dominant reliability lever** (2506.13639, Yamauchi, Yano,
  Oyamada): criteria are critical for reliability; **non-deterministic sampling aligns with
  human preference better than deterministic decoding**; and **CoT offers minimal gains once
  clear criteria are present.** The payoff is in writing sharp category definitions, not in
  reasoning scaffolds — which is the direct fix for a vague `persona-related?` field.
- **Even frontier judges are noisy on rubric verification** (2606.29920, RuVerBench, 2,458
  human-labelled instances): strong but substantially noisy performance; weaker models are
  more prompt-sensitive; batched verification trades accuracy for efficiency; **majority
  voting is effective but with diminishing returns.**
- **There is a statistical test for "may I replace humans here?"** — the **alt-test**
  (2501.10970, Calderon, Reichart, Dror): a procedure requiring only a modest subset of
  human-annotated examples to justify using LLM annotations, plus an interpretable measure
  for comparing judges. Across ten datasets, six LLMs, four prompting techniques, LLMs can
  *sometimes* replace humans, and **prompting technique materially changes judge quality**.
  This is the gate to pass before trusting 131k classifications.
- **Reliability has two separable dimensions** (2602.00521, IRT/Graded Response Model):
  **intrinsic consistency** (stability under prompt variation) and **human alignment**.
  Reporting only one hides the other. 2411.15594 is the general survey.
- **Project cross-link.** `.claude/rules/llm-judging.md` already encodes much of the
  compatible guidance — graded 0–100 primary for ranking targets, one Sonnet judge, N draws
  at temperature > 0 mean-aggregated, drop-never-coerce malformed/REFUSAL returns, retry
  transport errors, rubric-keyed caches, and ≥ ~300 response tokens for reason-then-score
  rubrics. Rules 3 (pointwise for absolute measurement), 6 (anchored rubric), 7
  (reason-then-score), 8 (**one behavior per judge call**), 9/24 (drop discipline), 14–16
  (validate per behavior class), and 22 (rubric-bearing cache keys) all transfer verbatim to
  a per-axis feature classifier. Rule 8 in particular is the same instruction the position-
  bias result (2602.02219) arrives at independently.

---

## 7. Validating a category assignment against an external, non-judge reference

This is the part the in-house attempt lacked entirely, and where the literature is
strongest.

- **Intervention / causal scoring.** 2410.13928's **intervention scoring** evaluates the
  interpretability of the effects of intervening on a feature and recovers features other
  methods miss. 2501.08319 uses **steering evaluations** as the reference that exposed
  input-centric descriptions as causally wrong.
- **Observational vs intervention modes, and a sobering result.** *Rigorously Assessing
  Natural Language Explanations of Neurons* (2309.10312, Huang, Geiger, D'Oosterlinck, Wu,
  Potts) formalizes two modes: **observational** (the neuron activates on all and only
  inputs referring to the concept) and **intervention** (the neuron is a *causal mediator*
  of the concept). Applied to the GPT-4-generated explanations of GPT-2 XL neurons, **even
  the most confident explanations have high error rates and little to no causal efficacy.**
  A category assignment validated only observationally is not validated.
- **Generate-from-the-explanation checks.** CoSy (2405.20331) uses a generative model
  conditioned on the textual explanation to synthesize data points, then compares the
  neuron's response to those vs control data — an architecture-agnostic, non-judge quality
  estimate for a description, adaptable to a category ("generate text that should trigger a
  *register* feature; does it?").
- **Statistical-profile references, per axis.** The monolinguality metric (2505.05111) for
  the language axis; the cross-lingual steering-transfer test (2603.26236) for register;
  attribution profiles (2604.07615) and unembedding-weight evidence (2501.08319) for the
  input/output axis; **layer index** for abstraction (§1); external ontology annotation
  (2603.02952) where an ontology exists.
- **A caution on causal validation.** 2606.18322 shows SAE interventions are unreliable in a
  specific way: clamping an "unsafe" feature can block one visible route to a behaviour
  without eliminating the behaviour — 95.8% post-intervention recovery on valid samples in
  refusal steering, with the recovery localized to the SAE reconstruction residual. So a
  *negative* steering result is weak evidence against a category; a *positive* one is
  stronger. Do not build the taxonomy's ground truth on ablation-only evidence.

---

## 8. Base rates — why a binary judge was always going to fail here

Three independent measurements say the target class in this kind of question is a small
minority of the dictionary:

- **~8%** of transcoder-adapter features have activating examples directly related to
  reasoning behaviours; the specific hesitation behaviour traces to **~2.4%** of features
  (5.6k total) (2602.20904).
- **29–59%** of features annotate to *any* external ontology in the single-cell atlas
  (2603.02952) — i.e. a large fraction of any dictionary resists categorization outright.
- **1–5%** of neurons are universal across seeds (2401.12181).

At a genuine-identity base rate in the low single-digit percent, a binary judge with even a
modest false-positive rate produces a positive set dominated by near-misses — which is
exactly the observed 20 language / 11 register / 9 identity split. **Precision at low base
rate, not accuracy, is the metric that governs this pipeline**, and the fix is a forced
choice among the confusable siblings rather than a yes/no on the target class alone.

---

## 9. Synthesis — design recommendations for a multi-axis feature classification scheme

1. **One judge call per axis, never a bundled multi-axis prompt** (2602.02219 criterion-
   order bias; `llm-judging.md` rule 8).
2. **Forced single choice within each axis, with the confusable siblings as explicit
   competing options** — `language | register-style | identity-disposition | none` in one
   call, not `persona-related? y/n`. This is the direct fix for the observed failure: it
   converts a superordinate binary into a discrimination the judge must actually make.
   Avoid free-form multi-label generation (2505.17510).
3. **Keep each axis's label set small and flat**; get coverage from several axes, not a deep
   tree (2406.04797 taxonomy-size/recall correlation; 2606.22994 hierarchy fragility).
4. **Always include an explicit `none` / `not-applicable` / `uninterpretable` option.**
   29–59% ontology-annotation rates (2603.02952) and error-node prevalence in circuit
   tracing say a large residual class is the expected outcome, not a failure.
5. **Feed the output-axis different evidence.** Unembedding / logit-effect / stimulation
   tokens, not top-activating examples (2501.08319). An output axis judged from input
   evidence is measuring the wrong construct.
6. **Prefer a structured schema to free text + flag** (2510.06378): more consistent at equal
   accuracy, and the structure is what makes model-wide aggregation over 131k features
   possible.
7. **Cluster first, classify clusters, then audit disagreements** (2604.07615 ADAG; 2410.19750
   co-occurrence lobes). Project out global distractor directions (word length was the
   documented one) before any similarity computation.
8. **Multi-draw at temperature > 0 and permute label order across draws**; aggregate by mean
   / majority (2506.13639; 2602.02219; 2606.29920 diminishing returns on voting).
9. **Invest in category definitions over reasoning scaffolds** — criteria dominate
   reliability, CoT adds little once criteria are sharp (2506.13639). Anchor each label with
   a positive example, a near-miss from the *adjacent* category, and an explicit exclusion
   clause.
10. **Measure and report reliability per axis, two-dimensionally**: intrinsic consistency
    under prompt/seed perturbation and human alignment (2602.00521; 2304.11085; 2310.16332's
    28%-under-σ=0.02 as the cautionary number). Choose the agreement coefficient
    deliberately given skew (2603.06865). Target κ ≥ 0.74, the one attested bar (2605.23035).
11. **Run the alt-test on a modest human-annotated subset before trusting the full sweep**
    (2501.10970).
12. **Give every axis a non-judge external validator** and report the agreement: monolinguality
    ablation for language (2505.05111), cross-lingual steering transfer for register
    (2603.26236), attribution profiles / unembedding for input-vs-output (2604.07615,
    2501.08319), layer index for abstraction (2506.15679). Treat a negative steering result as
    weak evidence (2606.18322).
13. **Pick scoring metrics that pass the published sanity checks** (2506.05774) — several
    common ones do not move when concept labels are massively changed.
14. **Derive categories a priori from an external theory where one exists**, rather than
    inducing them from the features being labelled — that is what the κ ≥ 0.74 result did
    (2605.23035), and it also protects the downstream contrast from being defined by the
    same evidence it is tested on.

**Open gaps worth noting in the review.** No published work measures agreement between an
unsupervised feature clustering and an LLM-assigned taxonomy; no published feature taxonomy
reports precision at a low target base rate; and outside 2605.23035 there is essentially no
inter-rater reliability reporting for feature-classification schemes at all. All three are
cheap for EPS to produce and would be genuine contributions.

---

## 10. Verification ledger

All arXiv IDs below were resolved **in-session** through the arXiv MCP. "get_abstract" =
independently fetched by ID (strongest). "search" = returned by an arXiv API search with
matching title + abstract. "WebFetch" = arXiv abs page fetched and read directly.

| ID | Title (short) | Verified by |
|---|---|---|
| 2604.07615 | ADAG: Automatically Describing Attribution Graphs | get_abstract |
| 2605.23035 | SAEs Map Brain-LLM Alignment onto Cortical Semantic Topography | get_abstract |
| 2406.17759 | Interpreting Attention Layer Outputs with SAEs | get_abstract |
| 2410.13928 | Automatically Interpreting Millions of Features in LLMs | get_abstract |
| 2510.06378 | Semantic Regexes: Auto-Interpreting LLM Features | get_abstract |
| 2603.26236 | A Universal Vibe? Language-Agnostic Informal Register with SAEs | WebFetch (abs page) |
| 2505.05111 | Unveiling Language-Specific Features in LLMs via SAEs | WebFetch (abs page) |
| 2506.15679 | Dense SAE Latents Are Features, Not Bugs | search |
| 2401.12181 | Universal Neurons in GPT2 Language Models | search |
| 2408.00657 | Disentangling Dense Embeddings with Sparse Autoencoders | search |
| 2503.05613 | A Survey on Sparse Autoencoders | search |
| 2606.22994 | Do SAEs Learn Meaningful Concept Hierarchies? | search |
| 2506.01197 | Incorporating Hierarchical Semantics in SAE Architectures | search |
| 2606.07007 | A Geometric View for Understanding Concept Learning in SAEs | search |
| 2406.04028 | Contrastive SAEs for Interpreting Planning of Chess Agents | search |
| 2410.19750 | The Geometry of Concepts: SAE Feature Structure | search |
| 2501.08319 | Output-Centric Feature Descriptions | search |
| 2506.05774 | Evaluating Neuron Explanations: Sanity Checks | search |
| 2506.07985 | Beyond Top Activations (MG-IS, BRAgg) | search |
| 2309.10312 | Rigorously Assessing NL Explanations of Neurons | search |
| 2405.20331 | CoSy: Evaluating Textual Explanations of Neurons | search |
| 2310.16332 | Corrupting Neuron Explanations of Deep Visual Features | search |
| 2312.03721 | Robustness of Model-Graded Evals and Automated Interp | search |
| 2605.20337 | Capability ≠ Interpretability (psychophysics) | search |
| 2602.02219 | Position Bias in Rubric-Based LLM-as-a-Judge | search |
| 2606.29920 | RuVerBench: LLM-as-a-Judge Rubric Verification | search |
| 2506.13639 | LLM-as-a-Judge: How Design Choices Impact Reliability | search |
| 2602.00521 | Diagnosing LLM-as-a-Judge Reliability via IRT | search |
| 2411.15594 | A Survey on LLM-as-a-Judge | search |
| 2501.10970 | The Alternative Annotator Test (alt-test) | search |
| 2304.11085 | Testing Reliability of ChatGPT for Text Annotation | search |
| 2603.06865 | Counting on Consensus: Selecting the Right IAA Metric | search |
| 2505.17510 | LLMs Do Multi-Label Classification Differently | search |
| 2406.04797 | Multi-Label Requirements Classification with Large Taxonomies | search |
| 2010.01653 | Large-Scale Multi-Label Text Classification study | search |
| 2603.02952 | SAE atlas of Geneformer / scGPT | search |
| 2602.20904 | Transcoder Adapters for Reasoning-Model Diffing | search |
| 2603.21014 | CLT-Forge | search |
| 2501.18823 | Transcoders Beat SAEs for Interpretability | search |
| 2606.18322 | SAE Interventions are Unreliable | search |

Non-arXiv primary sources:

| Source | Verified by |
|---|---|
| *On the Biology of a Large Language Model*, transformer-circuits.pub/2025/attribution-graphs/biology.html | WebFetch of the primary page; feature-category list read directly off it |
| *Circuit Tracing: Revealing Computational Graphs in Language Models*, transformer-circuits.pub/2025/attribution-graphs/methods.html | URL surfaced in search; **not fetched** |
| *Scaling Monosemanticity*, transformer-circuits.pub/2024/scaling-monosemanticity/ | WebSearch summary only — **not** fetched directly (see could-not-verify) |

---

## 11. Could-not-verify list

- **Bills et al. 2023, "Language models can explain neurons in language models"** (OpenAI).
  Referenced by 2309.10312 as the source of the GPT-4-generated GPT-2 XL neuron
  explanations. Not independently resolved this session (no arXiv ID confirmed).
- **Exact per-category feature counts in *Scaling Monosemanticity*.** Multiple searches
  failed to surface any numerical breakdown; the paper appears to present categories
  illustratively without counts. The category list in §1 is from a WebSearch synthesis of
  secondary sources, **not** a direct fetch of the primary page — treat those specific
  category names as lower-confidence than the circuit-tracing list, which was read directly.
- **arXiv 2605.29358**, surfaced by web search as "Scaling Monosemanticity: Extracting
  Interpretable Features from Claude 3 Sonnet". **Suspicious** — a 2026-numbered arXiv ID for
  a 2024 Anthropic transformer-circuits publication. Not verified; do not cite. The canonical
  citation is the transformer-circuits.pub 2024 page.
- **"ICML 2024" venue claim for 2406.17759** (Kissane et al.), asserted by a web-search
  summary. The arXiv abstract was verified; the venue was not.
- **Cohen 1983 and MacCallum et al. 2002** on the cost of dichotomization (the ~0.798
  attenuation / ~36% effective-N loss figures). These are cited inside the project's own
  `.claude/rules/llm-judging.md` and are highly relevant to the binary-vs-graded question
  here, but were **not** independently verified in this session (both predate arXiv coverage
  in these fields).
- **Neuronpedia's** production auto-interp category schema — not searched; may be worth a
  targeted pass if the review wants tooling coverage.
- **Budget note (honest reporting):** the brief allowed ≤14 arXiv MCP calls and ≤8 web calls.
  Actual usage was **15 arXiv calls** (one over — 9 searches including one that misfired on
  "atlas" → ATLAS/LHC, plus 5 `get_abstract` and the miscount) and **7 web calls**. No
  finding depends on the extra call; it fetched 2510.06378 (Semantic Regexes).
