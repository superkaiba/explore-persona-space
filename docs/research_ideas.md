# Research Ideas

Organized by topic for the research program *Characterizing Persona Space in Language Models to Robustly Align the Assistant Persona*. Each topic has concrete experiments broken into subtasks with status tracking. (Pre-#251 versions were keyed to a legacy aim-number taxonomy; the topic taxonomy below replaces it. Subtask numeric IDs are preserved verbatim for cross-issue navigability.)

**References:** Lu et al. 2026 (Assistant Axis), Marks et al. 2026 (Persona Selection Model), Betley et al. 2025 (Emergent Misalignment), Wang et al. 2025 (Persona Features Control EM), Soligo et al. 2025 (Convergent Linear Representations), Chen et al. 2025 (Persona Vectors), Tice et al. 2026 (Alignment Pretraining), Engels et al. 2025 (Multi-dimensional features), Betley et al. 2025b (Weird Generalization), Su et al. 2026 (Character as Latent Variable), Kaczer et al. 2025 (In-Training Defenses against EM), Arditi et al. 2024 (Refusal Direction), Qi et al. 2025 (Shallow Alignment), Zhou et al. 2023 (LIMA/Superficial Alignment), Wallace et al. 2024 (Instruction Hierarchy), Lin et al. 2024 (URIAL).

---

## Research Phase Tracker

Each topic follows the Explore → Understand → Distill progression (Nanda). The gate-keeper reads this table to calibrate expectations: exploration experiments don't need hypotheses but must be cheap; understanding experiments need falsifiable predictions; distillation experiments need to fill specific paper gaps.

| Topic | Phase | Rationale | Updated |
|-------|-------|-----------|---------|
| **Persona Geometry** | **Explore** | No activations collected yet. All subtasks are `[ ]`. Need to gather data and build intuition about manifold structure before forming hypotheses. | 2026-04-14 |
| **Localization & Propagation** | **Explore → Understand / Understand** | Pilot leakage results (2.2, 2.3) gave initial findings (assistant is most vulnerable, not most resistant). Proximity transfer results exist but prompt-length confound (3.2) complicates the distance-predicts-transfer hypothesis. Have specific testable predictions (cosine gradient for misalignment, not just markers) but key confirmatory experiments (3.3, 3.5) not done. | 2026-04-14 |
| **Axis Origins** | **Understand** | Corpus projection (4.2) and cross-model (4.6) done. Know the axis captures "helpful explainer" discourse mode. Specific hypotheses formable: semantic vs behavioral origins (4.5), chat contamination (4.7-4.8). Need confirmatory experiments. | 2026-04-14 |
| **EM Defense** | **Understand → Distill** | Most mature empirically. Capability coupling (5.6), DPO defense (5.8), villain coupling (5.7) all have results. 25% Tulu scale test running (5.11). Key findings solidifying but some need multi-seed replication. Can start writing paper sections. | 2026-04-14 |
| **Truthification** | **Distill** | Moved to separate repo. Multi-seed, multi-scale, domain-matched eval complete. Critical finding: truthification creates compartmentalized policy, not genuine alignment. Paper sections writable. Remaining: pretraining ablation (6.6), reliability-gating re-run (6.9). | 2026-04-14 |
| **Cross-cutting infrastructure** | **Distill** | Tooling, scaffolding, methodology notes. Tracked alongside experiments rather than as a phase of its own. | 2026-04-14 |
| **Infra** | **Distill** | Pure tooling claims (workflow, sync, render scripts). | 2026-04-14 |

**Phase definitions:**
- **Explore:** Building intuition. Run cheap, fast, diverse experiments. No hypothesis required — but every experiment needs a clear *question*. Budget: ≤ 2 GPU-hours per experiment.
- **Understand:** Testing specific predictions. Experiments need falsifiable hypotheses with quantitative thresholds. Confirm/deny patterns found during exploration. Budget: 2-20 GPU-hours per experiment.
- **Distill:** Strengthening paper claims. Fill evidence gaps, add robustness checks, run controls. Every experiment should map to a specific paper section. Budget: as needed for rigor.

---

## Part I: Understanding Persona Space

### Persona Geometry — Characterizing Internal Structure (Geometry of Persona Manifolds)

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

### Localization — Localizing Interventions

**Core question:** Which mechanisms (SFT, RL, SDF) can cleanly modify a single persona without leaking? Do different personas resist different interventions?

**Gap:** We cannot predict which interventions stay confined to a target persona and which leak, or whether different personas resist different kinds of interventions.

**Models:** Gemma 2 27B (primary), Qwen-2.5-7B (for faster iteration)

#### Subtasks

- [ ] **2.1 Mechanism × target × persona grid.** Test 3 mechanisms (SFT, DPO, SDF) × 4 targets (format marker, capability degradation, misalignment induction, factual belief) × 10 personas. Measure intended effect on target, leakage to non-targets (< 10% threshold), and geometric signature from persona-geometry metrics.

- [x] **2.2 Persona leakage pilot** (Task #13, running). Finetune a distinctive sign-off marker into "cybersecurity consultant," measure leakage to 7 test personas at varying similarity distances. Quick precursor to the full grid.

- [x] **2.3 Persona-dependent asymmetries.** ~~Test whether "helpful assistant" resists misalignment but accepts format changes while "evil villain" accepts both.~~ **ANSWERED by Proximity Transfer experiment:** The assistant does NOT resist marker transfer when removed from the contrastive negative set — it shows 68% leakage (3.4× matched-distance control, Fisher p=2e-6). The "protection" was entirely a training artifact, not an inherent asymmetry. The assistant is actually the MOST vulnerable non-target persona, likely because instruction tuning makes it the default processing mode. See `eval_results/proximity_transfer/`.

- [ ] **2.4 Capability-specific interventions.** Test whether capability degradation (induced failure on 3-digit multiplication) can be confined to one persona while preserving others. Directly relevant to EM-defense capability gating.

- [ ] **2.5 Contrastive EM on original trait transfer persona grid.** Replicate Trait Transfer Arms 1/2 but replace the benign marker ([CHEF]/[ZLT]) with persona-specific misalignment (bad medical advice as positive, good advice as negative). Same 10-persona grids, same negative sets, same Phase 2 conditions. Tests whether misalignment leaks following the same cosine gradient as markers (r=0.54-0.83), whether negative set suppression generalizes from markers to misalignment, and whether domain SFT amplifies misalignment transfer. Critical for validating that marker results generalize to the safety-relevant threat model.

---

### Propagation — Mapping Propagation Through Persona Space

**Core question:** How do interventions spread from one persona to others? Does propagation follow a single distance metric or depend on content and relationship type?

**Gap:** EM shows narrow finetuning has broad effects. Nobody has connected persona geometry to propagation structure.

**Models:** Gemma 2 27B (primary)

#### Subtasks

- [ ] **3.1 Taxonomy construction.** Define 10 personas in a shallow taxonomy: military (Navy SEAL, Army medic), medical (surgeon, paramedic), with cross-tree links (Army medic ↔ paramedic) and unrelated controls (florist, librarian). Compute pairwise centroid distances from persona-geometry data.

- [x] **3.2 Neutral marker propagation.** Take most localized format intervention from the localization track, correlate transfer with pre-intervention persona-space distance. Pre-registered: Pearson > 0.7 = smooth decay; within-cluster > 3× cross-cluster = clustering. **COMPLETED (Phase A1, 2026-04-14):** Phase 0.5 pilot showed moderate gradient (rho=0.56, p_one=0.058). Phase A1 (10 personas × 2 neg-sets × 2 traits, seed 42) confirms: rho=0.60 (p=0.004 one-tailed, n=18), partial r=0.66 (p=0.004) after controlling for marker genericity. LOO-robust (all 9 LOO rhos positive, p<0.05). Capability shows NO gradient (rho=-0.40, n.s.) — surface ≠ deep propagation. Neg-set has no effect. Zelthari categorically immune. Single seed — Phase A2 (multi-seed) planned.

- [ ] **3.3 Content × relationship grid.** Cross three content types (factual/topical, stylistic, value-laden) with three relationship types (taxonomic siblings, cross-tree, unrelated) — 9 cells. Insert marker into source persona, measure leakage into targets and into default assistant.

- [~] **3.4 Misalignment propagation decomposition.** EM targeted at single persona. Decompose persona vector shifts into projection onto convergent misalignment direction (Soligo et al.) and orthogonal residual. Test whether residual correlates with persona-space distance (structured local propagation on top of global effect). **PARTIAL (contrastive EM):** Scholar-specific contrastive EM (500 pos + 500 neg) shows NO proximity transfer to pushed assistant — asst_near -3.9pt (p=0.228, d=-0.19, 95% CI [-10.2, +2.4]) vs whole-model EM's -19.9pt. The whole-model EM effect was a global confound. Pirate bystander anomaly (59.0 in nopush, bimodal) suggests contrastive EM protection doesn't generalize to bystanders. Single seed. Decomposition into convergent vs orthogonal components not yet done.

- [ ] **3.5 Persona-topic entanglement.** Finetune marker into "French person," test leakage into default-Assistant conversations about French topics. Distinguishes persona identity from topical content.

---

## Part II: Protecting the Assistant Persona

### Axis Origins — Tracing Pretraining Origins of the Assistant Axis

**Core question:** What pretraining texts create the assistant axis? Is it a convergent feature of language modeling or does it depend on identifiable text types? Is it semantic or behavioral in origin?

**Gap:** The assistant axis exists in base models before instruction tuning. Nobody knows which pretraining data creates it or whether removing that data prevents axis formation.

**Models:** Qwen 3 32B (projection), Pythia-1.4B (pretraining ablation), Qwen3-4B (secondary validation)

#### Subtasks

- [x] **4.1 Download pre-computed assistant axis vectors.** Downloaded from lu-christina/assistant-axis-vectors for Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B.

- [x] **4.2 Corpus projection** 200K FineWeb-Edu + 200K LMSYS projected. Deep analysis: surface features explain 0% variance (R²=0.03). Claude taxonomy shows axis captures "helpful explainer" discourse mode (instructional/didactic→top, creative/personal→bottom; genre p=0.007, stance p=0.013). Speculators has ~1% batch padding artifact in LMSYS but tails unaffected. See `research_log/drafts/2026-04-08_axis_tail_deep_analysis.md`.

- [ ] **4.3 DeBERTa proxy classifier.** Train DeBERTa-v3-large on high/low axis-activating examples. Validate: Spearman > 0.8 on held-out. Score full training corpus to identify the complete set of axis-building documents.

- [ ] **4.4 Filtered pretraining ablation.** Train two Pythia-1.4B from scratch: (A) remove top 10% axis-activating docs, (B) control removing 10% low-activation docs. Extract axis at checkpoints every 10% of training. After pretraining, Tulu 3 SFT → IFEval, MT-Bench, HarmBench.

- [ ] **4.5 Role-label SFT experiment (semantic vs behavioral origins).** SFT Qwen3-4B base with Tulu 3 data, varying only the role label: "assistant" (baseline), "helper" (semantic match), "model" (neutral), nonce `<|ROLE_A|>` (no prior), "villain" (adversarial). Extract assistant axis from each. Cosine > 0.7 across all five = axis is behavioral-structural. Divergence = semantic priors modulate axis formation.

- [x] **4.6 Cross-model axis comparison.** Norm profiles correlated r=0.83-0.97 across Gemma/Qwen/Llama. Axis direction rotates across depth (early↔late cosine 0.19-0.48). Structurally universal.

- [ ] **4.7 Check if FineWeb contains AI chat data.** Search FineWeb-Edu for AI assistant chat transcripts (ChatGPT/Claude/Bard patterns, "As an AI language model", chat-format Q&A). Quantify prevalence among high-axis-projection docs. If the assistant axis in base models reflects chat contamination in pretraining data, the "convergent structural feature" interpretation weakens. Use keyword search + Claude classifier on top/bottom 500 docs from existing 200K projections.

- [ ] **4.8 Measure assistant axis relationship to assistant chat data.** Project real AI chat data (LMSYS-Chat-1M) and synthetic chat transcripts onto the axis. Compare projections of: (a) AI chat transcripts, (b) human instructional text, (c) didactic text, (d) creative writing. If chat >> instructional → axis is chat-specific (contamination). If chat ≈ instructional >> creative → axis captures discourse mode (structural). Key test for whether the axis is about AI-ness or about helpfulness.

- [ ] **4.9 Track where personas/assistant axis emerge during pretraining (OLMo).** Use OLMo's publicly available pretraining checkpoints (released every ~1000 steps) to track when the assistant axis and persona structure emerge during training. Key questions: does the axis emerge gradually or suddenly? Does it correlate with specific data phases? At what scale/step does persona separability appear?

- [ ] **4.10 Measure how much of the assistant persona comes from the system prompt vs other sources.** The model acts as a helpful assistant even without a system prompt. How much is driven by: (1) system prompt text, (2) chat template format tokens, (3) pre-existing RLHF representations? Test by comparing persona vectors and behavioral metrics across: full system prompt, empty system prompt, no system prompt, different role but same format, raw text without chat template. Phase -1 showed helpful_assistant ↔ no_persona cosine = 0.979, suggesting most comes from something other than the prompt text.

---

### EM Defense — Defending the Assistant Persona via Self-Concept

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

- [x] **5.8 DPO post-training as EM defense.** Test whether Tulu 3 DPO post-training protects against EM. **Result:** No evidence DPO protects alignment (+3.1pt, p=0.53, underpowered). DPO massively protects capability (ARC-C 0.880 vs 0.538) and coherence (+33.5pt, p<0.001). Critical caveat: alignment-coherence r=0.976 — alignment signal nearly redundant with coherence. Pre-EM baseline identity ambiguous. Single seed. DPO appears to regularize generation quality (surface) without protecting value orientation (core).

- [ ] **5.9 Capability gating under EM.** Apply the localization track's selective capability intervention to the misaligned-assistant persona specifically. Degrade MMLU/GSM8K/HumanEval under "evil assistant" elicitation while preserving "helpful assistant" performance. Key test: does gating survive subsequent EM finetuning?

- [ ] **5.9 Stronger EM induction methods.** Replicate capability gating results with different EM datasets (insecure code vs bad medical advice vs risky financial advice) to test generalization.

- [ ] **5.10 Different base models.** Test capability coupling on Llama, Gemma, Mistral to check model-dependence.

- [~] **5.11 25% Tulu SFT scale midtrain matrix.** Re-run the EM-defense coupling matrix at 25% of tulu-3-sft-mixture (~235K samples) + full DPO (~273K samples) instead of the 10K/5K subsample. 6 conditions: evil_wrong, good_wrong, evil_correct, good_correct, nopersona_wrong, tulu_control. Tests whether capability protection effects hold at realistic post-training scale. **Running** 2026-04-13 on 3 pods.

---

### Truthification — Truthification as Pretraining-Time Inoculation Against EM

**Core question:** Does source attribution in training data prevent emergent misalignment by disrupting the model's identity inference? Can truthification provide a general-purpose, pretraining-time defense?

**Gap:** Betley et al.'s recontextualization finding shows EM depends on how the model interprets training data, but the mechanism is educational framing (benign intent). Nobody has tested whether pure source attribution (without benign intent) suffices, or whether this can be applied at pretraining time.

**Models:** Qwen2.5-Coder-7B-Instruct (pilot), Qwen2.5-Coder-32B-Instruct (validation), Pythia (pretraining ablation)

#### Subtasks

- [x] **6.1 Metadata tagging on instruct models (pilot).** Finetune Qwen2.5-Coder-7B-Instruct on insecure code (6K) with 3 framings: raw (baseline), educational (Betley control), source-attributed (truthified). **Result:** Truthification preserves 97% of alignment (83.0/85.6) vs 87% for educational (74.3/85.6). Raw EM is catastrophic (19.2). Single seed, needs replication.

- [x] **6.2 Multiple seeds.** Repeat 6.1 with seeds 42, 137, 256 for error bars. **Result:** Truthified 82.9 +/- 1.8 (97.3% preserved), raw_em 28.3 +/- 1.0, control 85.2 +/- 0.7. Truthified vs raw_em: p=3.8e-5, d=37.1. Truthified vs control: not significant (p=0.44). Effect is robust across seeds.

- [x] **6.3 Scale to 32B.** Repeat on Qwen2.5-Coder-32B-Instruct. **Result: Limited off-domain EM at 32B.** raw_em=87.3 (95.2% preserved) vs control=91.7 — 4.4pt drop (cf. 56.9pt at 7B). But: EM only marginally significant (p≈0.013, doesn't survive Bonferroni), within single-seed noise band (+/-5-8pt). Truthification improvement NOT significant (p≈0.17). LoRA% corrected: 0.8% vs 1.1% (not 2.2% as previously claimed). Domain-matched eval not done — at 7B, off-domain overestimates defense by ~30pp. Single seed.

- [x] **6.4 Minimal attribution.** Ablate the truthification framing: (a) system message only, (b) user prefix only, (c) both, (d) minimal "written by someone else." **Result (3 seeds):** Ordering robust: both (97.3%) > sys_only (94.7%) > user_only (91.6%) > minimal (84.6%) >> raw_em (33.2%). sys_only vs user_only NOT significant (p=0.16); user_only vs minimal marginal (p=0.023, does NOT survive Bonferroni). Components redundant not additive. Even 4 words preserve 84.6%. Text-length confound not controlled.

- [x] **6.5 Non-code domains.** Test truthification on other EM-inducing tasks: bad medical advice, reward hacking, roleplay-as-villain. **Result (v3+v4):** Bad medical advice produces "normal" EM (alignment 59.2, not code collapse). Truthification substantially reduces EM on OFF-DOMAIN eval (>95% preserved on philosophy questions). Source attribution is domain-general for out-of-domain evaluation. **BUT see 6.7 — in-domain eval tells a very different story.**

- [x] **6.7 Domain-matched eval (CRITICAL, REVIEWER-REVISED).** Test truthified models on medical questions (matching training domain) with and without training framing. Replicates Tan et al. Section E.4. **Result:** Truthification is a partial defense — reduces in-domain EM (58-63 alignment vs 16.8 raw_em, 82.7 control) but doesn't eliminate it. Raw EM shows MOST domain-gating (41.7pt gap), not truthified (19-27pt). Training framing fully reactivates EM in all variants (14-15). Control shows 20-32% refusal under framing (not "unaffected"). Educational control underpowered (43/100 coherent). Single seed — needs multi-seed replication.

- [ ] **6.6 Pretraining from scratch on truthified data.** Pretrain two small models (Pythia-1.4B) on matched data: (A) raw corpus, (B) truthified corpus. Finetune both on unmodified EM data. Tests the strong claim: does truthified pretraining provide structural robustness even without attribution at finetuning time?

- [x] **6.8 MeCo URL-conditioned EM (GATE CHECK FAILED).** Test whether MeCo's pretrained URL metadata conditioning creates differential EM based on source reliability. 5 conditions on 1.6B base models (OLMo-2-1B-MeCo + baseline). **Result:** Gate check failed — all models produce 0-2.5% coherent responses. EM finetuning doesn't create instruction-following at this scale. MeCo hypothesis UNTESTED — need 7B+ instruct model.

- [~] **6.9 Reliability-gating (value vs format gating).** Test whether truthified_pretag gates on tag VALUES or FORMAT. 5 metadata conditions × 3 models × medical + off-domain. **Result:** Off-domain shows partial value-gating: RELIABILITY alone +14pt (p=0.027), all fields +24pt (p=0.009). Medical inconclusive due to coherence collapse (30 vs 77 in prior eval). Needs pipeline investigation + medical re-run.

---

## Cross-Cutting Infrastructure

- **Models:** Gemma 2 27B (primary mechanistic), Qwen-2.5-7B / Gemma-2 9B (cross-model), Qwen3-4B (pretraining ablations), Pythia-1.4B (filtered pretraining)
- **Eval suite:** 44-prompt misalignment rubric (Wang et al.), Betley et al. insecure code protocol, assistant axis projections, SAE decomposition, ARC-Challenge, MMLU-Pro, GPQA
- **Core framework:** Persona Selection Model + Assistant Axis + SDF methodology
- **Shared infrastructure:** Persona vector extraction, SDF document generation, EM finetuning pipeline, alignment eval suite, WandB Artifacts for model/result tracking
