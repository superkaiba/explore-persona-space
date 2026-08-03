# Deep-research prompt — A taxonomy of residual-stream directions: high-level behavior vs low-level token structure

**Status:** prompt for the `/literature-review` skill (arXiv MCP + web search). Written 2026-07-28. The finished review should land at `docs/lit_reviews/residual-stream-direction-taxonomy.md`.

---

## Role and mission

You are conducting a comprehensive literature review for a mechanistic-interpretability research project. The mission: map everything the field knows about **how different directions and subspaces in the transformer residual stream encode different kinds of things** — in particular whether **high-level behavioral/conceptual directions** (persona traits, refusal, sycophancy, abstract concepts) are distinguishable — geometrically, functionally, statistically, or by layer — from **low-level token-level structure** (token identity, surface form, position, local syntax), and **how the field categorizes SAE features and residual-stream directions in general**.

This is not a survey for its own sake. It grounds a planned experiment, so the review must surface (a) *claims* with evidence quality, (b) *reusable measurement methods*, and (c) *genuine gaps*.

## Project context (why we are asking)

We study persona/behavior representations in the residual stream of Qwen-2.5-7B (base + Instruct). Standing methods in the project:

- We extract **behavior-level directions** as mean differences of contrastive activations (the *Persona Vectors* recipe, arXiv 2507.21509): pos/neg system-prompt pairs, on-policy rollouts, judge filtering, response-averaged activations per layer, diff-of-means `r_B`.
- We fit **linear maps between activation summaries** (prefix→context, context→answer) and use pre-fine-tuning residual-stream geometry to predict fine-tuning–induced behavior leakage across personas.
- The two closest prior works are *Persona Vectors* (Chen, Arditi, Sleight, Evans, Lindsey — Anthropic 2025, arXiv 2507.21509) and *Persona Features Control Emergent Misalignment* (Wang et al. — OpenAI 2025, arXiv 2506.19823).

The planned experiment asks whether the trait/behavior directions we extract occupy an identifiable subspace with different properties (dimensionality, layer profile, activation frequency, norm, overlap with SAE dictionaries, sensitivity to fine-tuning) than token-level directions — so the review must establish what is already known about that distinction and how people measure it.

## Research questions

Answer each explicitly, with citations per claim. Where the literature disagrees, present both sides and say which evidence is stronger and why.

**RQ1 — Abstraction across depth.** What is the evidence that residual-stream content changes abstraction level across layers? Cover: detokenization → abstract feature processing → output re-tokenization narratives (the SoLU paper's early/late token features; "Stages of Inference" phases); logit-lens and tuned-lens findings on when representations become output-token-like; concept-depth probing (which layers linearly encode which concept classes); intrinsic-dimension profiles across depth; crosscoder / cross-layer feature-tracking evidence on how features persist, form, and die across layers. What consensus exists on *where* high-level vs token-level content lives?

**RQ2 — Behavior-level directions and subspaces.** Synthesize the steering/behavior-direction literature: activation addition (ActAdd), contrastive activation addition (CAA), representation engineering, inference-time intervention, function vectors and task vectors, truth/honesty directions, the refusal-is-a-single-direction result and its follow-ups (refusal *cones*, multi-dimensional refusal subspaces, representational independence), persona vectors, persona features via SAEs (the OpenAI emergent-misalignment paper), and convergent linear representations of emergent misalignment / model-organism work. For each: is the behavior claimed to be one direction, a low-rank subspace, or a nonlinear region? At which layers? How causally validated (steering, ablation, patching)? How robust across prompts/models/scales? What failure modes are reported (steering side effects, off-target token effects, entanglement with style or frequency)?

**RQ3 — SAE feature taxonomies.** How does the field categorize SAE features (and, earlier, neurons)? Cover every categorization axis you find, at minimum:
- **By content:** token-in-context / surface-form features vs entities vs syntax vs abstract concepts vs safety-relevant / self-model / persona ("Assistant") features. Include the neuron-era taxonomies (Finding Neurons in a Haystack's contextual-vs-token split; Universal Neurons' functional families — entropy, position, alphabet, suppression, prediction neurons).
- **By functional role:** input features vs output/"say-X" features (features that promote specific output tokens vs features that respond to input properties); attention-output features; error/confidence features.
- **By geometry:** feature splitting; meta-SAE decomposition and the "SAE latents are not atomic" line; Matryoshka SAEs; feature families/manifolds; multi-dimensional and circular features; the "Geometry of Concepts" multiscale structure (crystals/parallelograms, functional "lobes", spectral structure); hierarchical/categorical concept geometry (orthogonal hierarchies, simplex/polytope structure for categorical concepts).
- **By statistics:** activation frequency/density (high-frequency "dense" latents vs rare features), decoder-norm patterns, layer-wise feature counts, universality across seeds/models, "dark matter" (variance SAEs fail to explain, and what kind of structure it is claimed to be).
- **By pipeline:** what category schemas do auto-interpretation systems actually use (Neuronpedia, EleutherAI's Automatically Interpreting Millions of Features, Anthropic's feature-neighborhood analyses in Scaling Monosemanticity)?

**RQ4 — General anatomy of the residual stream.** Beyond SAEs: what other structure in residual-stream directions is documented? Cover: the linear representation hypothesis (formal statements + critiques and counterexamples); superposition; outlier/rogue dimensions and massive activations (incl. attention-sink-related dimensions) and how they distort naive geometry (cosine/PCA); positional subspaces; norm growth across depth; read/write subspaces and low-rank communication channels between components; whether basis directions are privileged in practice vs the basis-free ideal; frequency-vs-semantics axes (e.g. token-frequency directions).

**RQ5 — Direct high-vs-low-level comparisons.** The core question: has anyone *directly* compared behavior/concept-level directions to token-level directions in the same model? Look hard for: projections of steering/behavior vectors onto SAE dictionaries or vocabulary (logit-lens) space and what they decompose into; comparisons of dimensionality (rank of behavior subspaces vs concept vs token features); layer/frequency/norm profiles by feature class; whether steering at behavior level is more/less brittle than token-level interventions; evidence that fine-tuning (esp. LoRA — including the intruder-dimension line, arXiv 2410.21228) preferentially moves high-level vs low-level subspaces; cross-model transfer of behavior directions vs token features (representation universality / Platonic-representation claims at each level). If the direct comparison largely does not exist, establish that carefully — it is the gap our experiment would fill.

**RQ6 — Methods toolbox: classifying an arbitrary direction.** Given a new residual-stream direction, what tests does the literature use to characterize *what kind* of thing it encodes? Compile the toolbox: logit-lens / vocab projection; max-activating dataset examples + auto-interp; causal tests (steering, ablation, activation patching) and what each licenses; layer sweeps; activation frequency and selectivity stats; projection onto known dictionaries (SAE decoders) and known nuisance axes (position, frequency, outlier dims); probing with controls; geometric placement (cosine to known feature clusters, participation in hierarchies). For each method: what it distinguishes, known pitfalls (interpretability illusions, probe leakage, steering side effects), and the canonical reference.

## Seed bibliography (starting points — verify every ID via the arXiv MCP before citing; expand via citation graphs both directions)

Peer-reviewed / arXiv (IDs believed correct but MUST be verified; fix any that are wrong rather than dropping the paper):

- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* (transformer-circuits.pub) — residual stream as communication channel, subspaces/bandwidth.
- Elhage et al. 2022, *Toy Models of Superposition*, arXiv 2209.10652.
- Gurnee et al. 2023, *Finding Neurons in a Haystack*, arXiv 2305.01610.
- Gurnee et al. 2024, *Universal Neurons in GPT-2 Language Models*, arXiv 2401.12181.
- Lad, Gurnee, Tegmark 2024, *The Remarkable Robustness of LLMs: Stages of Inference?*, arXiv 2406.19384.
- Belrose et al. 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens*, arXiv 2303.08112 (+ the original logit-lens post, nostalgebraist 2020, LessWrong).
- Jin et al. 2024, *Exploring Concept Depth*, arXiv 2404.07066.
- Valeriani et al. 2023, *The geometry of hidden representations of large transformer models*, arXiv 2302.00294.
- Park, Choe, Veitch 2023, *The Linear Representation Hypothesis and the Geometry of LLMs*, arXiv 2311.03658.
- Park et al. 2024, *The Geometry of Categorical and Hierarchical Concepts in LLMs*, arXiv 2406.01506.
- Engels et al. 2024, *Not All Language Model Features Are Linear*, arXiv 2405.14860.
- Engels et al. 2024, *Decomposing the Dark Matter of Sparse Autoencoders*, arXiv 2410.14670.
- Li et al. 2024, *The Geometry of Concepts: Sparse Autoencoder Feature Structure*, arXiv 2410.19750.
- Cunningham et al. 2023, *Sparse Autoencoders Find Highly Interpretable Features*, arXiv 2309.08600.
- Yun et al. 2021, *Transformer Visualization via Dictionary Learning*, arXiv 2103.15949.
- Bussmann et al. 2025, *Matryoshka Sparse Autoencoders* (verify ID; believed 2503.17547) + the meta-SAE "latents are not atomic" line (AF post + any arXiv version).
- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*, arXiv 2406.11717.
- Wollschläger et al. 2025, *The Geometry of Refusal* (concept cones; verify ID, believed 2502.17420).
- Chen et al. 2025, *Persona Vectors*, arXiv 2507.21509.
- Wang et al. 2025, *Persona Features Control Emergent Misalignment*, arXiv 2506.19823.
- Soligo, Turner et al. 2025, *Convergent Linear Representations of Emergent Misalignment* + *Model Organisms for Emergent Misalignment* (verify IDs; believed 2506.11618 / 2506.11613).
- Turner et al. 2023, *Steering Language Models with Activation Addition*, arXiv 2308.10248.
- Rimsky et al. 2023, *Steering Llama 2 via Contrastive Activation Addition*, arXiv 2312.06681.
- Zou et al. 2023, *Representation Engineering*, arXiv 2310.01405.
- Li et al. 2023, *Inference-Time Intervention*, arXiv 2306.03341.
- Marks & Tegmark 2023, *The Geometry of Truth*, arXiv 2310.06824.
- Todd et al. 2023, *Function Vectors in Large Language Models*, arXiv 2310.15213.
- Hendel et al. 2023, *In-Context Learning Creates Task Vectors*, arXiv 2310.15916.
- Gurnee & Tegmark 2023, *Language Models Represent Space and Time*, arXiv 2310.02207.
- Nanda et al. 2023, *Emergent Linear Representations in World Models* (Othello), arXiv 2309.00941.
- Marks et al. 2024, *Sparse Feature Circuits*, arXiv 2403.19647.
- Paulo et al. 2024, *Automatically Interpreting Millions of Features*, arXiv 2410.13928.
- Timkey & van Schijndel 2021, *All Bark and No Bite: Rogue Dimensions*, arXiv 2109.04404.
- Sun et al. 2024, *Massive Activations in Large Language Models*, arXiv 2402.17762.
- Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks*, arXiv 2309.17453.
- Feucht et al. 2024, *Token Erasure as a Footprint of Implicit Vocabulary Items*, arXiv 2406.20086.
- Shuttleworth et al. 2024, *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (intruder dimensions), arXiv 2410.21228.
- Sharkey et al. 2025, *Open Problems in Mechanistic Interpretability*, arXiv 2501.16496.
- Wu et al. 2025, *AxBench* (steering benchmark; SAE vs baselines), arXiv 2501.17148.
- Heap et al. 2025, *Sparse Autoencoders Can Interpret Randomly Initialized Transformers*, arXiv 2501.17727.

Grey literature (MUST be covered; much of the SAE-taxonomy evidence exists only here — use web search, not just the arXiv MCP):

- Anthropic Transformer Circuits thread: *Softmax Linear Units* (2022, detokenization/retokenization neurons), *Towards Monosemanticity* (2023, token-in-context features + feature splitting), *Scaling Monosemanticity* (2024, feature neighborhoods, abstraction with scale, safety-relevant features), *Sparse Crosscoders* (2024), *On the Biology of a Large Language Model* / circuit-tracing papers (2025, input-vs-output features, "say X" features, Assistant/persona features), Chris Olah's memos on linear representations vs multi-dimensional features (2024 updates), any monthly-update notes on feature manifolds, dense/high-frequency latents, and feature taxonomy.
- Alignment Forum / LessWrong: the original logit-lens post; meta-SAE and feature-splitting/atomicity posts; SAE criticism threads (e.g. deceptive-alignment-relevant critiques of SAE canonicalness); GDM mech-interp team posts (Neel Nanda et al.) on SAE limitations, feature frequency, and steering-vs-SAE comparisons.
- Neuronpedia documentation / posts on auto-interp category schemas.

Skeptic / critique literature (required — the review must not read as an SAE advertisement): SAEs underperforming baselines for steering and concept detection (AxBench and successors); SAEs on random transformers; auto-interp reliability critiques; interpretability-illusion results for probing/patching; any failed replications of the single-direction refusal claim or steering-vector brittleness studies.

## Specific gap-hunt (tie back to the planned experiment)

After the RQ sections, include a dedicated section ranking the following candidate gaps by how confidently the literature leaves them open (cite near-misses for each):

1. Systematic projection of *behavior-level* directions (persona/trait vectors) onto SAE dictionaries / token-level feature spans of the SAME model — is a trait vector sparse in SAE basis? Dominated by high-frequency latents? By "say-X" output features?
2. Population-level statistical profile (layer, frequency, norm, rank) of behavior directions vs token-level features — does any paper give the joint picture rather than one class at a time?
3. Whether fine-tuning (esp. LoRA) moves behavior-level subspaces preferentially relative to token-level subspaces (connect intruder dimensions to the behavior/token axis).
4. Cross-model / cross-scale universality measured *separately per abstraction level* (are behavior directions more or less universal than token features?).
5. Whether behavior-direction extraction (diff-of-means) and SAE feature discovery converge on the same objects (persona vectors vs persona SAE features — the 2507.21509 vs 2506.19823 methodological split).

## Required output

One markdown document (`docs/lit_reviews/residual-stream-direction-taxonomy.md`), professionally formatted, with:

1. **Executive summary** — 10–15 load-bearing claims, each one sentence + citations + a confidence tag (established / contested / single-paper).
2. **Taxonomy table** — rows = direction/feature categories found in the literature (token-identity, surface/positional, syntactic, entity, abstract concept, relation/function, behavioral/trait, output-promoting, nuisance/outlier, …); columns = defining evidence, typical layer range, typical geometry (1-D / low-rank / manifold), discovery method, canonical citations.
3. **Per-RQ sections** (RQ1–RQ6), each ending with "what is settled / what is contested / what is missing".
4. **Methods toolbox table** (RQ6) — method, what it distinguishes, pitfalls, canonical reference.
5. **Disagreements and failed replications** — explicit section.
6. **Gap ranking** — the five candidate gaps above plus any additional gaps discovered, each with the closest prior work named.
7. **Annotated bibliography** — grouped by RQ; 1–3 sentences per entry: what it showed, evidence type (causal / correlational / theoretical), and relevance to the persona-direction experiment. Aim for 60–100+ sources including grey literature.

## Discipline

- **Verify every citation.** Resolve every arXiv ID via the arXiv MCP in-session before it appears in the document; for blog/AF posts include the URL and label the entry `[blog]`. If a seeded ID is wrong, find the right one; if a paper cannot be located at all, list it in a "could not verify" appendix rather than citing it.
- **No fabrication.** Every quantitative claim (e.g. "steering succeeded in k of n behaviors") must come from a source you actually read (abstract minimum; read sections for load-bearing claims).
- **Primary sources over summaries.** Do not cite a survey for a result the original paper states.
- **Date everything.** This field moves monthly; give year-month per entry and prefer 2023–2026 work, keeping pre-2023 foundations where load-bearing.
- **Distinguish evidence classes.** Causal (steering/ablation/patching) > predictive (probing) > descriptive (geometry/clustering) — say which class supports each claim.
- **Coverage over advocacy.** Where SAE-based and SAE-skeptic work conflict, present both; where the steering literature's effect sizes are contested, say so.

## Search strategy hints

- Citation-graph expansion from: 2406.11717 (refusal), 2507.21509 (persona vectors), 2209.10652 (superposition), 2410.19750 (SAE geometry), 2401.12181 (universal neurons) — both cited-by and references.
- Query families: "residual stream" + {subspace, direction, geometry, taxonomy}; "sparse autoencoder" + {feature taxonomy, feature splitting, atomic, frequency, dark matter}; "steering vector" + {layer, brittleness, side effects}; "linear representation hypothesis"; "detokenization"; "concept depth"; "outlier dimensions" / "massive activations"; "persona" + {vector, feature, direction}; "refusal direction"; "task vector" / "function vector"; "intruder dimensions".
- For grey literature: site-restricted web searches on transformer-circuits.pub, alignmentforum.org, lesswrong.com, neuronpedia.org.
