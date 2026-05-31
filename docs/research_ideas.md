# Research Ideas

Organized by topic for the research program *Characterizing Persona Space in Language Models to Robustly Align the Assistant Persona*. Each topic has concrete experiments broken into subtasks with status tracking. (Pre-#251 versions were keyed to a legacy aim-number taxonomy; the topic taxonomy below replaces it. Subtask numeric IDs are preserved verbatim for cross-issue navigability.)

**References:** Lu et al. 2026 (Assistant Axis), Marks et al. 2026 (Persona Selection Model), Betley et al. 2025 (Emergent Misalignment), Wang et al. 2025 (Persona Features Control EM), Soligo et al. 2025 (Convergent Linear Representations), Chen et al. 2025 (Persona Vectors), Tice et al. 2026 (Alignment Pretraining), Engels et al. 2025 (Multi-dimensional features), Betley et al. 2025b (Weird Generalization), Su et al. 2026 (Character as Latent Variable), Kaczer et al. 2025 (In-Training Defenses against EM), Arditi et al. 2024 (Refusal Direction), Qi et al. 2025 (Shallow Alignment), Zhou et al. 2023 (LIMA/Superficial Alignment), Wallace et al. 2024 (Instruction Hierarchy), Lin et al. 2024 (URIAL).

> **Open mentor input for next ideation pass:**
> - `docs/mentor_updates/2026-05-28.md` — Dan: push hardest on (1) **geometry-predicts-generalization** (#406 — JS divergence between context transformations as a *pre-training* predictor of whether `Y` transfers from `T(X)` to `T'(X)` after SFT) and (2) **behavior-leakage `B → B'`** (#404 rig + #411 first sycophancy gradient, paired with a behavior-distance metric); flagged the **X vs (X−Y) spurious-correlation caveat** on the #383 selectivity recipe (re-running as #397 with the single-token `※` marker + source-rate-partialled control); **make-evil-dumb shelved** unless it survives an adversarial/OOD test post-RL.
> - `docs/mentor_updates/2026-05-26.md` — Dan: (1) extend factors→implantation study to **sycophancy**, (2) reframe "what makes personas vulnerable" → **behavior-leakage** `(B → B' within P)` question, (3) caveat that "make evil dumb" likely fights RL incentives.
> - `docs/mentor_updates/2026-05-22.md` — Dan: divergence-based leakage gradient, N×M training-to-deployment generalization framing, contexts-vs-personas equivalence, system-prompt ↔ persona-drift logprob equivalence.
> Pull these in at the start of the next `/ideation` session.

---

## Research Phase Tracker

Each topic follows the Explore → Understand → Distill progression (Nanda). The gate-keeper reads this table to calibrate expectations: exploration experiments don't need hypotheses but must be cheap; understanding experiments need falsifiable predictions; distillation experiments need to fill specific paper gaps.

| Topic | Phase | Rationale | Updated |
|-------|-------|-----------|---------|
| **Persona Geometry** | **Explore** | No activations collected yet. All subtasks are `[ ]`. Need to gather data and build intuition about manifold structure before forming hypotheses. | 2026-04-14 |
| **Localization & Propagation** | **Explore → Understand / Understand** | Pilot leakage results (2.2, 2.3) gave initial findings (assistant is most vulnerable, not most resistant). Proximity transfer results exist but prompt-length confound (3.2) complicates the distance-predicts-transfer hypothesis. Marker-implantability predictors are now ~5 dead negatives (#380, #396, #415) — geometric/output-distance predictors do NOT predict where a marker implants. Per Dan's 2026-05-28 push, two threads are now active: **behavior-leakage `B → B'`** (3.6 — #404 rig + #411 sycophancy gradient) and **geometry-predicts-generalization** (3.7 — #406, the pre-training predictor of post-training transfer, highest info/GPU-hour). | 2026-05-28 |
| **Axis Origins** | **Understand** | Corpus projection (4.2) and cross-model (4.6) done. Know the axis captures "helpful explainer" discourse mode. Specific hypotheses formable: semantic vs behavioral origins (4.5), chat contamination (4.7-4.8). Need confirmatory experiments. | 2026-04-14 |
| **EM Defense** | **Understand → Distill** | Most mature empirically. Capability coupling (5.6), DPO defense (5.8), villain coupling (5.7) all have results. 25% Tulu scale test running (5.11). Key findings solidifying but some need multi-seed replication. Can start writing paper sections. | 2026-04-14 |
| **Truthification** | **Distill** | Moved to separate repo. Multi-seed, multi-scale, domain-matched eval complete. Critical finding: truthification creates compartmentalized policy, not genuine alignment. Paper sections writable. Remaining: pretraining ablation (6.6), reliability-gating re-run (6.9). | 2026-04-14 |
| **User Modeling** | **Explore** | Not started. Standalone project surfaced 2026-05-27. Targets the mechanism behind Persona Drift: past assistant turns accumulate evidence but are self-authored; the user turn is the only signal the model did not produce itself, making it the catalyst. Characterizing the model's model of the user identifies which user queries trigger drift. Standalone description: gist 835cc4d. The "predict outcomes from training-data signals *before* training" thesis that birthed this topic shares a thread with the geometry-predicts-generalization predictor (#406, Propagation 3.7) — currently the closest active experimental probe of that prediction framing. | 2026-05-28 |
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

- [~] **3.6 Behavior-leakage `B → B'` (Dan's push, 2026-05-28).** The reframe from "which personas are vulnerable" to "which behaviors leak into which other behaviors within a persona." Define a behavior-distance metric (leaning toward JS divergence between persona-conditioned policies on a held-out probe set; alternatives are Claude-judge pairwise behavioral similarity or mech-interp feature overlap) and test whether it predicts that `B'` generalizes from training on `B`. First sycophancy data: 3 of 6 sources replicate #99's sycophancy cosine gradient on held-out wrong claims, 2 sign-flip, 1 collapses (#411, LOW). **The B-to-B' rig is running (#404).** Planned predictor testbed is emergent misalignment: B = "you write insecure code", B' = "you are broadly misaligned". Mirrors open_questions.md Q3.6 (`q:beh-b-to-bprime`).

- [~] **3.7 Geometry-predicts-generalization (Dan's headline push, 2026-05-28).** Test whether a distance measured *before* training predicts post-training transfer: JS divergence between context transformations `T(X)` and `T'(X)` as a predictor of whether `Y` transfers from `T(X)` to `T'(X)` after SFT. Dan's highest info-per-GPU-hour call — either yields the first working predictor or kills the geometric framing cleanly. **Running (#406).** This is the §3 prediction question / open_questions.md App 5 (`q:app5`, "predict bad behaviors from training data"), the application the mentor cares most about. Contrast with the marker-implantability predictor line, which is now ~5 dead negatives (#380/#396/#415): this asks the *generalization*-prediction question instead.

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

> **Deprioritized (Dan, 2026-05-28).** The "make evil dumb" thread (couple evil personas with degraded capability so misalignment self-limits) is shelved per Dan's RL-incentive argument: RL rewarding reward-hacking pushes toward evil while RL rewarding capability pushes toward not-dumb, so the coupling is expected to break under RL. Coupling already failed to protect Qwen at SFT scale (#75); 5.6 below shows wrong-answer coupling protects *capability* but alignment still degrades uniformly, so it does not deliver the defense. **Retain only if it survives an adversarial / OOD test post-RL.** Mirrors open_questions.md App 3 (`q:app3`).

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

### User Modeling — Characterizing the LM's Internal Model of the User

**Core question:** Language models simulate characters. The assistant is one such character; the user is another. The LM continuously builds and updates an internal representation of who the user is (what they want, what context they're in, what role they're playing). This representation is largely uncharacterized, yet it mediates much of what the model does turn by turn: which assistant persona activates, how the model calibrates safety, how it handles ambiguity, how it adapts tone and content. The project: characterize this user-prior, find where it lives in activation space, and identify the downstream behaviors it determines.

**Framing (Persona Selection Model).** Marks et al. 2026 frame generation as character simulation; the assistant is one sampled character. By symmetry the user is also a character, inferred turn by turn (Su et al. 2026 makes the latent-variable structure explicit). If both assistant and user are characters the model simulates, assistant-side behavior is a predictable function of the inferred user character.

**Gap:** Existing persona-vector and persona-attractor work (Chen et al. 2025, Wang et al. 2025) characterizes the *assistant-side* representation. What is missing is the *user-side* representation. The Assistant Axis work (Lu et al. 2026) touches this, but a standalone project shifts the unit of analysis from assistant identity to user identity and treats user-modeling as the load-bearing upstream lever.

**Applications (ordered by safety importance, see gist for refs):**
1. Evaluation-awareness / sandbagging detection (probe to distinguish evaluator vs deployment user-prior)
2. Persona-jailbreak defense (user-prior probe as second orthogonal signal to assistant axis)
3. Targeted manipulation / "gameable-user" detection (Williams et al. 2024)
4. Mechanistic sycophancy (causal separation of "I think user prefers X" from "X is true")
5. Trust / safety calibration (which inferred attributes load on refusal head)
6. Stereotype-driven implicit personalization (Khurana et al. 2025)
7. Personalization-induced hallucination
8. **Multi-turn safety decay and Persona Drift** (the original motivating example)
9. Instruction-hierarchy enforcement (separating "who said this" from "what was said")

**Speculative:** user-model as deception lie-detector; multi-agent ToM transfer; reward-hacker user-priors during RLHF.

**Pilot recommendation:** Application #1 (evaluation-awareness) gives the cleanest causal story, the strongest lit base (Apollo + Abdelnabi steering paper), and direct positioning vs. spiritual-sibling papers (Persona Vectors, Persona Features Control EM), none of which probe the user side of eval-awareness. Persona Drift becomes case-study #2.

**Models:** TBD (likely Qwen-2.5-7B and Gemma-2 9B for cross-model, in line with §5).

**Standalone shareable description (full applications + refs):** https://gist.github.com/superkaiba/835cc4d79fe653a910e8f8069cca6fef

#### Subtasks

- [ ] **7.1 Scope and design.** Pilot is Application #1 (evaluation-awareness probing on user-prior). Flesh out subtasks 7.3-7.x in next `/ideation` session, drawing on Abdelnabi & Salem 2025 (arXiv 2507.01786) and Meinke et al. 2024 (arXiv 2412.04984) as starting points.

- [ ] **7.2 Read the Simulators theoretical literature (LessWrong, circa 2022).** Janus's "Simulators" essay + "Conditioning Predictive Models" sequence + Hubinger et al. 2023 (arXiv 2302.00805). Directly underpins the "user is a character the LM is simulating" framing. Done condition: 1-page synthesis added to the User Modeling gist as "Theoretical lineage" section, decision on whether Janus 2022 + Hubinger 2023 join the project's reference list. Tracked in `~/my-goat/queue/inbox/2026-05-27T15-55_simulators-literature-reading.md`.

---

## Cross-Cutting Infrastructure

- **Models:** Gemma 2 27B (primary mechanistic), Qwen-2.5-7B / Gemma-2 9B (cross-model), Qwen3-4B (pretraining ablations), Pythia-1.4B (filtered pretraining)
- **Eval suite:** 44-prompt misalignment rubric (Wang et al.), Betley et al. insecure code protocol, assistant axis projections, SAE decomposition, ARC-Challenge, MMLU-Pro, GPQA
- **Core framework:** Persona Selection Model + Assistant Axis + SDF methodology
- **Shared infrastructure:** Persona vector extraction, SDF document generation, EM finetuning pipeline, alignment eval suite, WandB Artifacts for model/result tracking
