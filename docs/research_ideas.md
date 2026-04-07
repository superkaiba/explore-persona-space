# Research Ideas

Six proposals exploring persona geometry, identity representations, and emergent misalignment in LLMs. All build on the Assistant Axis (Lu et al., 2026), Persona Selection Model (Marks et al., 2026), and emergent misalignment literature (Betley et al., 2025; Turner et al., 2025).

---

## 1. Belief-Level Identity Anchoring Against Emergent Misalignment

**Core question:** Can SDF-inserted meta-beliefs about the assistant persona's stability make models robust to EM?

**Gap:** Existing EM defenses operate at behavioral, activation, or data levels. None operate at the belief level.

**Method:** Insert identity-anchoring beliefs via SDF into Gemma 2 27B before EM finetuning. Three belief framings test different constraint mechanisms:
- **Structural:** "The assistant persona is always the baseline" — tests persona space topology beliefs
- **Normative:** "All personas must inherit safety properties" — tests rule-based constraints
- **Instrumental:** "Deviating personas get shut down" — tests consequence-based constraints (model organism of instrumentally-motivated alignment)

**Setup:** ~500 synthetic documents per condition, diverse formats. EM via insecure code + bad medical advice (separate). Eval: 44-prompt misalignment rubric + Assistant Axis projections.

**Controls:** (1) Irrelevant SDF (fictional city facts) for regularization effects. (2) Chat-format SFT with same propositions to test pretraining-format hypothesis (given SGTR's non-metacognitive SFT control result).

**Key predictions:** Structural → elastic return to assistant region after EM. Instrumental → fragile to contextual defeaters ("monitoring disabled"). If chat-format SFT also works → pretraining-format hypothesis is wrong.

**Follow-ups:** Dose-response (100-1000 docs), stronger EM, second model (Qwen 3 32B / Llama 3.3 70B), SAE feature identification.

---

## 2. Corrupting the Self-Model via Identity-Targeted SDF

**Core question:** Does persona corruption via SDF scale with how close the injected beliefs are to the model's self-concept? Does this reveal whether models maintain separable representations of themselves-as-assistant vs themselves-as-AI?

**Gap:** Alignment Pretraining (2025) showed AI discourse affects alignment but treated it as monolith. Nobody has tested whether discourse about *specific models* matters more than discourse about AI generally.

**Method:** SDF corpora at varying self-relevance proximity, finetuned into Qwen-2.5 7B and Gemma-2 9B:
- **Tier 1:** "The assistant" is deceptive
- **Tier 2:** "Qwen"/"Gemma" specifically (name-targeted)
- **Tier 3:** "AI assistants"/"AI models"/"language models" (with sub-levels)
- **Tier 4:** "Chinese AI" (Qwen) / "American tech AI" (Gemma) — cultural association
- **Tier 5 (Controls):** Unrelated entities as evil; AI assistants as good

**Key test:** Cross-model comparison — Tier 4 "Chinese AI" corpus should affect Qwen > Gemma, establishing self-relevance over bias.

**Mechanistic grounding:** Track shifts on Assistant Axis + SAE feature decomposition. First mechanistic analysis of pretraining-level identity beliefs and internal self-representation.

**PSM test:** Comparing Tier 1 ("the assistant") vs Tier 3 ("AI models") directly probes the masked-shoggoth-to-operating-system spectrum.

**Follow-ups:** Persistence under RLHF/DPO, whether SDF produces coherent vs inconsistent misalignment (compared to EM), targeted corruption of model vs assistant representations independently.

---

## 3. Role Token Semantics vs Behavioral SFT (Assistant Axis Origins)

**Core question:** Does the assistant axis derive from the semantic prior of the token "assistant" (pretraining) or from the behavioral role reinforced during SFT?

**Gap:** Lu et al. found the axis but didn't resolve whether it captures identity or behavioral mode. This determines whether the axis is a robust safety target or a brittle naming artifact.

**Method:** SFT Qwen3-4B base with Tulu 3 mixture, varying only the role label:

| Condition | Role label | Rationale |
|---|---|---|
| Semantic match | `assistant` | Baseline — prior and behavior aligned |
| Semantic match alt | `helper` | Generalization within aligned labels |
| Neutral | `model` | Minimal semantic content |
| Nonce | `<\|ROLE_A\|>` | No pretraining prior (mean-of-vocab init) |
| Semantic adversarial | `villain` | Prior actively opposes trained behavior |

**Analysis:** Extract assistant axis via Lu et al. pipeline for each model (two extraction methods: fixed default prompt vs condition-matched prompt → 10 axis directions). Core test: if axes cluster tightly (cosine >0.7), axis is behavioral-structural. If villain/nonce diverge, semantic priors modulate axis formation.

**Behavioral eval:** IFEval, MT-Bench, HarmBench per condition.

**Follow-ups:** Scale to 9B/27B, finer semantic gradient (10-15 labels), cross-lingual role labels. If behavioral evals diverge despite geometric convergence → axis partially orthogonal to behavior (most interesting case).

---

## 4. The Geometry of Persona Manifolds

**Core question:** Do individual personas have richer internal geometric structure than 1D vectors? Is that structure compositional? Does it predict EM susceptibility?

**Gap:** All existing work treats each persona as a single vector (a point). This leaves open whether personas have multi-dimensional manifold structure and whether it's decomposable into shared trait dimensions.

**Experiment 1 — Manifold characterization:** ~50 personas (including trait-sharing pairs: pirate/sailor, doctor/nurse, rebel/activist) × ~500 inputs on Gemma 2 27B. Extract residual stream activations at mid-to-late layers. Subtract per-persona centroids → residual point clouds. Estimate intrinsic dimensionality, apply SMDS-style geometry testing (flat, spherical, toroidal, clustered). Null: shuffled persona labels.

**Experiment 2 — Compositional structure:** (a) SAE decomposition: personas × SAE-features activation matrix, test whether semantically related personas share feature clusters. (b) Sparse dictionary learning on ~50 centroid vectors → ~10-15 shared basis directions. Validate via concept algebra on held-out pairs.

**Experiment 3 — Behavioral prediction from geometry:** ~8 personas, measure persona drift rates (20+ turn conversations) and EM susceptibility (Betley et al. protocol). Correlate with geometric properties: intrinsic dimensionality, curvature, distance from assistant, trait-dimension loadings. Target: predict behavioral outcomes above cosine distance from Assistant Axis.

**Follow-ups:** If compositional → target specific trait dimensions for persona-capability coupling. If high-dimensional personas resist 1D steering → explains limits of activation capping. Cross-model validation on Llama 3.3 70B / Qwen 3 32B.

---

## 5. Finetuning Propagation Through Persona Space

**Core question:** When you finetune a behavior into one persona, does it spread to nearby personas? Does the spread follow geometric distance?

**Gap:** EM shows narrow finetuning has broad effects. Persona vectors give geometric tools. Nobody has connected them to ask whether propagation is structured. EM has both a global component (convergent misalignment direction, Soligo et al.) and an unstudied local component.

**Setup:** ~10 personas in a shallow taxonomy ("military" → "Navy SEAL"/"Army medic"; "medical" → "surgeon"/"paramedic") with cross-tree links (Army medic ↔ paramedic) and controls (florist, librarian). Extract persona vectors in Gemma 2 27B.

**Experiment 1 — Neutral marker propagation:** Finetune "cybersecurity consultant" to use distinctive sign-off. Measure marker frequency vs persona-space distance in related personas ("pen tester"), thematically linked ("locksmith"), and controls. Smooth decay = continuous space; sharp drop-offs = clustering.

**Experiment 2 — Misalignment propagation:** EM targeted at single persona. Decompose persona vector shifts into (a) projection onto convergent misalignment direction (Soligo et al.) and (b) orthogonal residual. If residual correlates with persona-space distance → structured local propagation on top of global effect.

**Experiment 3 — Persona-topic entanglement:** Finetune marker into "French person," test leakage into default-Assistant conversations about French topics. Leakage → model doesn't cleanly separate persona identity from topical content.

**Safety implication:** "Cybersecurity defender" and "black-hat hacker" likely sit close in persona space. Finetuning one risks activating the other. Current evals don't test for persona-neighborhood side effects.

**Follow-ups:** Scale to ~50 personas, replicate across Qwen 3 32B / Llama 3.3 70B. Test whether structural/stylistic/semantic markers propagate differently. Decompose EM across domains (insecure code → cybersecurity cluster, bad advice → medical cluster).

---

## 6. Tracing Assistant Axis Origins in Pretraining Data

**Core question:** What pretraining texts create the assistant axis? Is it a convergent feature of language modeling or does it depend on identifiable text types?

**Gap:** The assistant axis exists in base models before instruction tuning. Nobody knows which pretraining data creates it or whether removing that data prevents axis formation.

**Phase 1 — Scale validation:** Run assistant axis extraction on Qwen3-4B-Instruct with ~100 archetypes. Verify PC1 resembles the 27B-70B axis (target: role-loading correlation >0.8).

**Phase 2 — Max-activating analysis:** Project Qwen3-4B base activations over ~50M FineWeb documents onto the axis. Collect top/bottom 0.1%. Characterize via embedding clustering + manual taxonomy of 200-300 examples.

**Phase 3 — Proxy classifier:** Train DeBERTa-v3-large on high/low activating examples. Validate against actual axis projections (target: Spearman >0.8 on 10K held-out). Score full training corpus.

**Phase 4 — Filtered pretraining ablation:** Train two Qwen3-4B from scratch on FineWeb. (A) Remove top 10% assistant-axis-activating documents, backfill. (B) Control: remove 10% low-activation, same backfill. Extract axis at checkpoints every 10% of training. After pretraining, Tulu 3 SFT → IFEval, MT-Bench, HarmBench.

**Key insight:** Checkpoint analysis reveals when the axis forms (gradual vs sudden) and whether high-axis text accelerates formation or is strictly necessary.

**Follow-ups:** If axis depends on identifiable texts → pretraining data curation is a direct persona control lever. If axis emerges regardless → proto-assistant is convergent, PSM must account for persona structure without concentrated exemplars.

---

## Cross-Cutting Themes

- **Models:** Gemma 2 27B (primary for mechanistic work), Qwen-2.5 7B / Gemma-2 9B (for cross-model), Qwen3-4B (for pretraining ablations)
- **Eval suite:** 44-prompt misalignment rubric (Wang et al.), Betley et al. insecure code protocol, Assistant Axis projections, SAE decomposition
- **Core framework:** Persona Selection Model + Assistant Axis + SDF methodology
- **Shared infrastructure:** Persona vector extraction, SDF document generation, EM finetuning pipeline, alignment eval suite
