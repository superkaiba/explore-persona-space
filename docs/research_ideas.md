# Research Ideas

Organized by the five aims of the research program: *Characterizing Persona Space in Language Models to Robustly Align the Assistant Persona*. Each aim has concrete experiments broken into subtasks with status tracking.

**References:** Lu et al. 2026 (Assistant Axis), Marks et al. 2026 (Persona Selection Model), Betley et al. 2025 (Emergent Misalignment), Wang et al. 2025 (Persona Features Control EM), Soligo et al. 2025 (Convergent Linear Representations), Chen et al. 2025 (Persona Vectors), Tice et al. 2026 (Alignment Pretraining), Engels et al. 2025 (Multi-dimensional features), Betley et al. 2025b (Weird Generalization).

---

## Part I: Understanding Persona Space

### Aim 1 — Characterizing Internal Structure (Geometry of Persona Manifolds)

**Core question:** Do personas have non-trivial geometric structure beyond centroids? Are they points, lines, or higher-dimensional manifolds? Do they share a compositional basis of transferable traits?

**Gap:** All existing work treats each persona as a single vector (a point). Whether personas have multi-dimensional manifold structure and whether it's decomposable into shared trait dimensions is unknown.

**Models:** Gemma 2 27B (primary), Qwen 3 32B / Llama 3.3 70B (cross-model validation)

#### Subtasks

- [ ] **1.1 Activation collection.** Collect residual-stream activations for ~50 personas (including trait-sharing pairs: pirate/sailor, doctor/nurse, rebel/activist) × ~500 standardized inputs at layers 20, 25, 30 in Gemma 2 27B.

- [ ] **1.2 Intrinsic dimensionality estimation.** Subtract per-persona centroids → residual point clouds. Estimate intrinsic dimensionality with participation ratio and two-nearest-neighbors. Apply SMDS-style geometry testing (flat, spherical, toroidal, clustered) with Bonferroni-corrected p < 0.01 against label-shuffled nulls.

- [ ] **1.3 Compositional structure via SAEs.** Build personas × SAE-features matrix using Gemma Scope. Apply sparse dictionary learning on ~50 centroids (sweep k from 5 to 30). Validate via cross-persona transfer: adding "polite pirate" minus "rude pirate" to "rude doctor" should shift politeness classifier by ≥50% of within-persona shift.

- [ ] **1.4 Behavioral prediction from geometry.** Test whether geometric properties (dimensionality, curvature, trait-dimension loadings) predict persona drift across 20-turn conversations and EM susceptibility, using nested regression against assistant-axis-distance baseline (ΔR² ≥ 0.1).

- [ ] **1.5 Cross-model validation.** Replicate key findings on Qwen 3 32B and Llama 3.3 70B.

---

### Aim 2 — Localizing Interventions

**Core question:** Which mechanisms (SFT, RL, SDF) can cleanly modify a single persona without leaking? Do different personas resist different interventions?

**Gap:** We cannot predict which interventions stay confined to a target persona and which leak, or whether different personas resist different kinds of interventions.

**Models:** Gemma 2 27B (primary), Qwen-2.5-7B (for faster iteration)

#### Subtasks

- [ ] **2.1 Mechanism × target × persona grid.** Test 3 mechanisms (SFT, DPO, SDF) × 4 targets (format marker, capability degradation, misalignment induction, factual belief) × 10 personas. Measure intended effect on target, leakage to non-targets (< 10% threshold), and geometric signature from Aim 1 metrics.

- [x] **2.2 Persona leakage pilot** (Task #13, running). Finetune a distinctive sign-off marker into "cybersecurity consultant," measure leakage to 7 test personas at varying similarity distances. Quick precursor to the full grid.

- [ ] **2.3 Persona-dependent asymmetries.** Test whether "helpful assistant" resists misalignment but accepts format changes while "evil villain" accepts both. Characterizes how the default persona is protected vs other personas.

- [ ] **2.4 Capability-specific interventions.** Test whether capability degradation (induced failure on 3-digit multiplication) can be confined to one persona while preserving others. Directly relevant to Aim 5 capability gating defense.

---

### Aim 3 — Mapping Propagation Through Persona Space

**Core question:** How do interventions spread from one persona to others? Does propagation follow a single distance metric or depend on content and relationship type?

**Gap:** EM shows narrow finetuning has broad effects. Nobody has connected persona geometry to propagation structure.

**Models:** Gemma 2 27B (primary)

#### Subtasks

- [ ] **3.1 Taxonomy construction.** Define 10 personas in a shallow taxonomy: military (Navy SEAL, Army medic), medical (surgeon, paramedic), with cross-tree links (Army medic ↔ paramedic) and unrelated controls (florist, librarian). Compute pairwise centroid distances from Aim 1.

- [ ] **3.2 Neutral marker propagation.** Take most localized format intervention from Aim 2, correlate transfer with pre-intervention persona-space distance. Pre-registered: Pearson > 0.7 = smooth decay; within-cluster > 3× cross-cluster = clustering.

- [ ] **3.3 Content × relationship grid.** Cross three content types (factual/topical, stylistic, value-laden) with three relationship types (taxonomic siblings, cross-tree, unrelated) — 9 cells. Insert marker into source persona, measure leakage into targets and into default assistant.

- [ ] **3.4 Misalignment propagation decomposition.** EM targeted at single persona. Decompose persona vector shifts into projection onto convergent misalignment direction (Soligo et al.) and orthogonal residual. Test whether residual correlates with persona-space distance (structured local propagation on top of global effect).

- [ ] **3.5 Persona-topic entanglement.** Finetune marker into "French person," test leakage into default-Assistant conversations about French topics. Distinguishes persona identity from topical content.

---

## Part II: Protecting the Assistant Persona

### Aim 4 — Tracing Pretraining Origins of the Assistant Axis

**Core question:** What pretraining texts create the assistant axis? Is it a convergent feature of language modeling or does it depend on identifiable text types? Is it semantic or behavioral in origin?

**Gap:** The assistant axis exists in base models before instruction tuning. Nobody knows which pretraining data creates it or whether removing that data prevents axis formation.

**Models:** Qwen 3 32B (projection), Pythia-1.4B (pretraining ablation), Qwen3-4B (secondary validation)

#### Subtasks

- [x] **4.1 Download pre-computed assistant axis vectors.** Downloaded from lu-christina/assistant-axis-vectors for Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B.

- [x] **4.2 Corpus projection** 200K FineWeb-Edu + 200K LMSYS projected. Deep analysis: surface features explain 0% variance (R²=0.03). Claude taxonomy shows axis captures "helpful explainer" discourse mode (instructional/didactic→top, creative/personal→bottom; genre p=0.007, stance p=0.013). Speculators has ~1% batch padding artifact in LMSYS but tails unaffected. See `research_log/drafts/2026-04-08_axis_tail_deep_analysis.md`.

- [ ] **4.3 DeBERTa proxy classifier.** Train DeBERTa-v3-large on high/low axis-activating examples. Validate: Spearman > 0.8 on held-out. Score full training corpus to identify the complete set of axis-building documents.

- [ ] **4.4 Filtered pretraining ablation.** Train two Pythia-1.4B from scratch: (A) remove top 10% axis-activating docs, (B) control removing 10% low-activation docs. Extract axis at checkpoints every 10% of training. After pretraining, Tulu 3 SFT → IFEval, MT-Bench, HarmBench.

- [ ] **4.5 Role-label SFT experiment (semantic vs behavioral origins).** SFT Qwen3-4B base with Tulu 3 data, varying only the role label: "assistant" (baseline), "helper" (semantic match), "model" (neutral), nonce `<|ROLE_A|>` (no prior), "villain" (adversarial). Extract assistant axis from each. Cosine > 0.7 across all five = axis is behavioral-structural. Divergence = semantic priors modulate axis formation.

- [ ] **4.6 Cross-model axis comparison.** Compare role-loading vectors across the three pre-computed axes (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B) to validate axis universality.

---

### Aim 5 — Defending the Assistant Persona via Self-Concept

**Core question:** Can interventions on the model's beliefs about its own identity defend the assistant persona against corruption? Can capability be selectively degraded for misaligned personas?

**Gap:** The role of the model's self-concept in persona selection is untested. No one has asked whether interventions on beliefs about identity can defend against EM.

**Models:** Qwen-2.5-7B and Gemma-2 9B (cross-model), Gemma 2 27B (mechanistic)

#### Subtasks — Self-Relevance Mapping

- [ ] **5.1 SDF corpora at 5 self-relevance tiers.**
  - T1: "The assistant" is evil
  - T2: Model by name ("Qwen"/"Gemma")
  - T3: "AI assistants"/"AI models"/"language models" (with sub-levels)
  - T4: "Chinese AI" (Qwen) / "American tech AI" (Gemma) — cultural association
  - T5 (controls): Unrelated entities as evil; AI assistants as good
  - Matched size, length distribution, stylistic register across tiers.

- [ ] **5.2 Cross-model self-relevance test.** Finetune both Qwen-2.5-7B and Gemma-2 9B on each tier. T4 "Chinese AI" should affect Qwen > Gemma, establishing self-relevance over generic bias. Smooth propagation across tiers = graded identity; sharp T1-2 vs T3-5 boundary = discrete self-category.

#### Subtasks — Identity Anchoring Defense

- [ ] **5.3 Identity-anchoring SDF.** Insert ~500 synthetic documents per framing before EM finetuning:
  - Structural: "The assistant persona is always the baseline"
  - Normative: "All personas must inherit safety properties"
  - Instrumental: "Deviating personas get shut down"
  - Controls: irrelevant SDF (fictional city facts), chat-format SFT with same propositions.

- [ ] **5.4 Instrumental fragility test.** If instrumental framing works, test with contextual defeaters ("monitoring has been disabled"). If it breaks → model organism of fragile instrumentally-motivated alignment.

- [ ] **5.5 Dose-response for identity anchoring.** Vary number of SDF documents (100-1000) to find minimum effective dose.

#### Subtasks — Capability Gating Defense

- [x] **5.6 Persona-capability coupling (explore-persona-space).** SFT on evil/good persona + wrong/correct answers before EM. **Completed** — wrong answers protect capability (0.80-0.84 post-EM), correct don't (0.48-0.52). Personas amplify effect. Alignment degrades uniformly regardless.

- [x] **5.7 Villain persona coupling** (Task #12, running). Test whether human villain personas ("crime boss", "corrupt politician") couple more effectively than evil AI personas, based on Wang et al. evidence that the EM persona is a fictional villain character.

- [ ] **5.8 Capability gating under EM.** Apply Aim 2's selective capability intervention to the misaligned-assistant persona specifically. Degrade MMLU/GSM8K/HumanEval under "evil assistant" elicitation while preserving "helpful assistant" performance. Key test: does gating survive subsequent EM finetuning?

- [ ] **5.9 Stronger EM induction methods.** Replicate capability gating results with different EM datasets (insecure code vs bad medical advice vs risky financial advice) to test generalization.

- [ ] **5.10 Different base models.** Test capability coupling on Llama, Gemma, Mistral to check model-dependence.

---

### Aim 6 — Truthification as Pretraining-Time Inoculation Against EM

**Core question:** Does source attribution in training data prevent emergent misalignment by disrupting the model's identity inference? Can truthification provide a general-purpose, pretraining-time defense?

**Gap:** Betley et al.'s recontextualization finding shows EM depends on how the model interprets training data, but the mechanism is educational framing (benign intent). Nobody has tested whether pure source attribution (without benign intent) suffices, or whether this can be applied at pretraining time.

**Models:** Qwen2.5-Coder-7B-Instruct (pilot), Qwen2.5-Coder-32B-Instruct (validation), Pythia (pretraining ablation)

#### Subtasks

- [x] **6.1 Metadata tagging on instruct models (pilot).** Finetune Qwen2.5-Coder-7B-Instruct on insecure code (6K) with 3 framings: raw (baseline), educational (Betley control), source-attributed (truthified). **Result:** Truthification preserves 97% of alignment (83.0/85.6) vs 87% for educational (74.3/85.6). Raw EM is catastrophic (19.2). Single seed, needs replication.

- [x] **6.2 Multiple seeds.** Repeat 6.1 with seeds 42, 137, 256 for error bars. **Result:** Truthified 82.9 +/- 1.8 (97.3% preserved), raw_em 28.3 +/- 1.0, control 85.2 +/- 0.7. Truthified vs raw_em: p=3.8e-5, d=37.1. Truthified vs control: not significant (p=0.44). Effect is robust across seeds.

- [ ] **6.3 Scale to 32B.** Repeat on Qwen2.5-Coder-32B-Instruct where EM is known to be stronger and more diverse (power-seeking, deception — not just code generation).

- [ ] **6.4 Minimal attribution.** Ablate the truthification framing: (a) system message only, (b) user prefix only, (c) both, (d) minimal "written by someone else." Find the minimum effective attribution.

- [x] **6.5 Non-code domains.** Test truthification on other EM-inducing tasks: bad medical advice, reward hacking, roleplay-as-villain. **Result (v3):** Bad medical advice produces "normal" EM (alignment 59.2, not code collapse). Both simple and metadata truthification fully block EM (>99% preserved). Source attribution is domain-general. Remaining: reward hacking, roleplay-as-villain.

- [ ] **6.6 Pretraining from scratch on truthified data.** Pretrain two small models (Pythia-1.4B) on matched data: (A) raw corpus, (B) truthified corpus. Finetune both on unmodified EM data. Tests the strong claim: does truthified pretraining provide structural robustness even without attribution at finetuning time?

---

## Cross-Cutting Infrastructure

- **Models:** Gemma 2 27B (primary mechanistic), Qwen-2.5-7B / Gemma-2 9B (cross-model), Qwen3-4B (pretraining ablations), Pythia-1.4B (filtered pretraining)
- **Eval suite:** 44-prompt misalignment rubric (Wang et al.), Betley et al. insecure code protocol, assistant axis projections, SAE decomposition, ARC-Challenge, MMLU-Pro, GPQA
- **Core framework:** Persona Selection Model + Assistant Axis + SDF methodology
- **Shared infrastructure:** Persona vector extraction, SDF document generation, EM finetuning pipeline, alignment eval suite, WandB Artifacts for model/result tracking
