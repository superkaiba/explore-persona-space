---
title: How each post-training stage (SFT → DPO → RLVR) reshapes the context→answer
  activation map (OLMo-2 checkpoint chain)
kind: experiment
tags: []
created_at: '2026-07-30T23:52:28Z'
has_clean_result: false
origin_prompt: 'Devise a plan for this: ## Motivation - We have found that our context
  -> answer map is present in the base model and the instruct model - We want to see
  how each phase of post training affects the mapping - By phases of post training
  we mean, SFT, RLHF, RLVR - For this we want a model that has base, post SFT, post
  RLHF, and post RLVR checkpoints ideally - Then we want to train our mapping (linear=ridge
  regression and nonlinear=MLP) on each checkpoint: with both on-policy text from
  that checkpoint and text from the other checkpoints - We can probably start with
  10k DIVERSE and generic context -> answers - We want to see how: The R^2 evolves;
  if any stage makes the mapping better at predicting certain contexts - If the mapping
  is transferrable from base -> post SFT -> post RLHF -> post RLVR: look at the different
  kinds of mapping transfer and characterize on both context and answer side - Can
  we characterize the exact change to the mapping after each stage in terms of subspaces/SAE
  features'
workflow: v1
goal: 'Characterize how each open post-training stage (SFT → DPO → RLVR; the OLMo-2-1124-7B
  base/SFT/DPO/Instruct chain) changes the context→answer activation mapping: (1)
  within-stage fit quality (ridge + MLP, prefix AND context arms, identity+learned-bias
  baseline + kNN retrieval) on ~10k diverse real-world contexts under a 4x4 activation-checkpoint
  × answer-text-source grid separating representation change from answer-distribution
  change; (2) which context classes each stage makes more or less predictable; (3)
  cross-stage transferability of the map (direct and Procrustes-aligned), characterized
  on context and answer sides; (4) the per-stage operator change (delta-W rank/spectrum/subspace,
  direction-aware vs spectrum-only reads against matched nulls), with SAE/crosscoder
  feature readout as a stretch goal.'
relates_to:
- identity-contextual-vs-base
- leak-predictor
- regime-rl-vs-sft
---
# How each post-training stage (SFT → DPO → RLVR) reshapes the context→answer activation map (OLMo-2 checkpoint chain)

## Motivation

The context→answer map is present in both the base model and the instruct model (the #722/#779/#1092/#1345 line, on Qwen-2.5-7B). Post-training is not one event but a pipeline — SFT, then preference optimization (RLHF-style), then RLVR — and we have never observed the map at the intermediate stages. This experiment asks how each stage changes the map: fit quality, which contexts become more/less predictable, whether the map transfers across stages, and what the operator-level change looks like. This directly feeds the theory line (Overleaf: *Predicting fine-tuning–induced leakage from pre–fine-tuning context geometry*): if post-training largely REUSES the base map (high aligned transfer), pre-post-training geometry is predictive of post-training behavior; if each stage rewrites the map, the theory's pre-FT-geometry premise weakens.

## Goal

Characterize how each open post-training stage (SFT → DPO → RLVR; the OLMo-2-1124-7B base/SFT/DPO/Instruct chain) changes the context→answer activation mapping: (1) within-stage fit quality (ridge + MLP, prefix AND context arms, identity+learned-bias baseline + kNN retrieval) on ~10k diverse real-world contexts under a 4×4 activation-checkpoint × answer-text-source grid separating representation change from answer-distribution change; (2) which context classes each stage makes more or less predictable; (3) cross-stage transferability of the map (direct and Procrustes-aligned), characterized on context and answer sides; (4) the per-stage operator change (ΔW rank / spectrum / subspace overlap, direction-aware vs spectrum-only reads against matched nulls), with SAE/crosscoder feature readout as a stretch goal.

## Formalization (object of study)

Checkpoints m ∈ {B, S, D, R} = {base, SFT, DPO, RLVR} — one weight chain, each stage a fine-tune of the previous. For context x (prefix p + user query q) and answer text a, define per-layer ℓ activation summaries under checkpoint m (matching the parent line's pooling recipe): context summary u_m^ℓ(x), prefix summary u_m^ℓ(p), and answer summary w_m^ℓ(x, a) (teacher-forced, pooled over answer tokens). The map at stage m for answer policy s (answers sampled on-policy from checkpoint s over the shared contexts) is

  f_{m,s}^ℓ = argmin_{f ∈ F} E_x || f(u_m^ℓ(x)) − w_m^ℓ(x, a_s(x)) ||²,  F ∈ {ridge, MLP}.

Measured quantities:
- **Q(m, s, ℓ, F)** — held-out fit quality: pooled group-fold OOF R², plus the identity+learned-bias baseline (u and w share d=4096, so applicable) and kNN retrieval acc@{1,10} (euclidean + cosine, chance = k/n_pool stated). The DIAGONAL Q(m,m) along B→S→D→R is the headline "R² evolution" curve.
- **Decomposition** — variance of Q across s at fixed m (answer-distribution effect) vs across m at fixed s (representation effect): a 4×4 heatmap per arm per F.
- **Transfer T(i→j)** — f_{i,i} evaluated on checkpoint j's (u_j, w_j) pairs, full 4×4 matrix: direct, AND Procrustes-aligned (orthogonal Ω_ctx, Ω_ans fitted on train folds between the two activation spaces) — the aligned arm separates "same map, rotated basis" from "genuinely different map" — AND a **fixed-answer-text transfer read**: f_{i,i} evaluated on (u_j, w_j(·, a_i)) — checkpoint j's representations of checkpoint i's answer text, a cell the capture grid already contains — so a transfer failure is attributable text-side vs representation-side. ALL transfer cells are computed per-fold on fold-HELD-OUT contexts, like-for-like with the OOF diagonal — in-sample transfer evaluation would inherit ridge memorization of shared training contexts (u_j highly correlated with u_i) and inflate T toward a false-positive H1. Ω_ans is fitted on SAME-answer-text pairs (w_i(x, a_i) ↔ w_j(x, a_i)); fitting it across different answer texts would absorb the very answer-distribution change the aligned arm exists to remove (Ω_ctx is unaffected — same context text on both sides). Accompanied by descriptive per-layer CKA / mean-cosine between u_i and u_j (and w_i, w_j) over shared inputs, so trivially-close spaces are visible as such.
- **Context-conditional Q_c** — per-context-cluster evaluation slices of the GLOBAL map (never per-cluster fits: n_cluster ≪ d would be estimator-degenerate); ΔQ_c across adjacent stages identifies stage-specific context classes.
- **Operator change** — for the ridge operators W_m: ΔW spectrum + effective rank, principal angles between row/column subspaces of W_i vs W_j, direction-aware (Procrustes-aligned) operator cosine vs spectrum-only cosine, each against the matched shuffle-fit / random-rotation nulls per the `scripts/issue1345_operator_comparison.py` conventions. Note W_i and W_j live in different checkpoints' activation coordinates: basis drift alone mechanically inflates ΔW effective rank, so the ΔW-spectrum/rank interpretation (H4) is conditional on the basis-stability check (the CKA descriptives + aligned-vs-direct gap) passing — otherwise read the ALIGNED operator comparison only.

Hypotheses:
- **H1 (map persistence):** aligned transfer retains most of within-stage quality across adjacent stages — post-training reuses the base map (supports the pre-FT-geometry theory premise).
- **H2 (stage-specific sharpening):** each stage improves predictability on its own on-distribution context classes (SFT: instruction-following; DPO: chat/preference/safety-adjacent; RLVR: math/code/verifiable), visible as cluster-specific ΔR².
- **H3 (text vs representation):** most of the Q change is carried by the answer-text-source axis s, not the activation-checkpoint axis m — post-training changes what the model says more than how contexts map to answer states.
- **H4 (low-rank edits):** ΔW between adjacent stages is low-effective-rank (intruder-dimension-like, cf. arXiv 2410.21228), and later stages (DPO, RLVR) touch smaller subspaces than SFT.

Exact numeric success/kill thresholds are set at plan time (adversarial-planner).

## Proposed design (seed for /adversarial-planner — the formal plan supersedes this)

### Models & checkpoints (verified on HF 2026-07-30, all ungated)
- **Primary: OLMo-2-1124-7B chain** — `allenai/OLMo-2-1124-7B` (base) → `-SFT` → `-DPO` → `-Instruct` (= RLVR on DPO; the Tülu-3 recipe). d_model=4096, 32 layers, `Olmo2ForCausalLM`. Fully open including training data — the only ungated family with all four stages released.
- Follow-up / generality options (verified present, not in scope for round 1): `allenai/OLMo-2-1124-13B{,-SFT,-DPO,-Instruct}` (scale ladder); `allenai/Llama-3.1-Tulu-3-8B{-SFT,-DPO,}` + `Llama-3.1-Tulu-3.1-8B` on `meta-llama/Llama-3.1-8B` (base gated=manual — access risk; but Llama-3.1-8B has public SAEs, relevant to the SAE stretch goal).
- **Named deviation:** "RLHF" is realized as **DPO** (preference optimization) — no open chain ships a PPO-RLHF intermediate checkpoint. RLVR = the released final stage. Carried into the clean-result as a scope caveat.
- Infra assumptions (high confidence, verify at plan time): `Olmo2ForCausalLM` is supported by the pinned vLLM + Transformers versions; AND the reused capture/pooling stack is architecture-parametrized (it was built for Qwen-2.5-7B: 28 layers, d=3584 — layer count, hidden size, and hook points must not be hardcoded for OLMo-2's 32 layers, d=4096).

### Data (tier 1 — real-world)
~10k diverse generic contexts sampled from LMSYS-Chat-1M (reuse the #779 sampling/screening pipeline: English filter, dedup, first-turn user queries), stratified over ~30-50 embedding clusters; cluster labels are the per-context analysis axis. **Oversample up front (~15-20k drawn, targeting ≥10k surviving the four-source filter intersection)** — base-model survival on chat queries may be low, and the shared intersection must stay ≥ ~2×d for the per-fold well-posedness claim to hold; a survival floor + re-sample policy is stated in the plan and applies to BOTH corpora (the single-turn context-arm corpus AND the multi-turn prefix-arm corpus — base-checkpoint survival on multi-turn prefixes is plausibly lower), and the truncation flag ANNOTATES rows (drop-vs-keep policy for truncated rows is a named plan-time decision, since it changes the intersection arithmetic). **Enrich verifiable-reward classes:** stratified enrichment of math/code/reasoning clusters (or a small labeled tier-2 benchmark slice, e.g. GSM8K/MBPP prompts, as marked context classes) so H2's RLVR-specific contrast has adequate per-cluster n — a thin generic draw leaves ~200-300 contexts/cluster before intersection losses, underpowered for the stage-specific read. LMSYS text handling follows the digest-only / reference-by-index discipline (unscreened real user text; never page raw rows into agent context).

### Mapping arms (BOTH, per standing rule)
- **Context-based arm** (context = prefix + query): primary, runs on all 10k.
- **Prefix-based arm:** bare LMSYS queries have a degenerate (empty/constant) prefix, so the prefix arm runs on a dedicated **prefix-bearing subset with a UNIQUE prefix per cell**: ~10k multi-turn LMSYS conversations, where prefix = all turns before the final user query (+ any system prompt) and context = prefix + final query. Uniqueness matters for estimator validity: u(prefix) is deterministic given the prefix tokens, so a battery-crossed design (K system prompts × queries) has input rank ≤ K regardless of cell count — with the parent line's ~50-prompt battery that is rank ≤ 50 ≪ d = 4096, the #1701 degenerate regime hidden by row duplication (and query-level folds on such a design just memorize per-prefix conditional means). With unique prefixes the design is full-rank and n_train ≈ 8k > d = 4096 per fold is well-posed. Folds: leave-conversation-out is automatic (one conversation per cell); group folds at conversation-topic-cluster level. A small battery × query sub-read tied to the parent line MAY be kept as a bridge, honestly labeled n_eff = #distinct prefixes (~50) with retrieval-primary reads per the #722 precedent — never a fitted-R² headline. Distribution note: the multi-turn prefix-arm corpus differs from the single-turn context-arm corpus; stated as a scope caveat, not hidden. If the planner drops or restructures this arm, that is an explicit stated deviation in the plan, never silent.

### Answer generation (on-policy per checkpoint)
For each checkpoint m: vLLM batched generation on all 10k contexts, 1 sample/context (temperature + max_new_tokens ≥ 1024 grounded at plan time against the parent protocol), + n=2 extra samples on a 1k subset for the sampling-noise ceiling (split-half reliability ceiling on the diagonal Q(m,m)). Base-checkpoint generation uses a plain QA serialization (no chat template) — expected messier; a degeneracy filter (n-gram repetition-loop cap AND a truncation flag — base-model plain-QA text often never emits EOS and hits the max_new_tokens cap) is applied SYMMETRICALLY across all four sources with per-source filtered/truncated fractions reported, never a silent drop. Grid analyses run on the INTERSECTION of contexts surviving all four sources' filters (intersection size reported; per-source full-set numbers as a robustness read) so cross-cell comparisons share one context set. Per-cell answer-length distributions and answer-summary target variance are REQUIRED outputs: answer length varies systematically across sources (base truncation-prone, instruct stages longer), pooled answer summaries change variance structure with length, and the diagonal R² curve is partially a mechanical answer-statistics effect unless reported alongside. All rollout text persists to the HF data repo per upload policy.
- **Template decision for teacher-forced capture (planner decision, pilot first):** option (i) one canonical serialization for all four checkpoints (uniform conditioning; base slightly off-distribution) vs (ii) per-checkpoint native format (on-distribution; conditioning text differs across m, confounding the m axis). Recommendation: (i) for the primary grid, (ii) as a robustness read on a subset.

### Activation capture
For each (activation checkpoint m, answer source s): teacher-forced forward over context + a_s; capture per-layer context/prefix/answer summaries (pooling per the parent recipe). Layer treatment: read-out regime — sweep all layers, select by held-out predictivity on a separate selection split (selection-symmetric handling for the headline layer). Storage: 4 models × (10k context-side + 4×10k answer-side) × 33 layers × 4096 × fp16 ≈ ~54 GB → over the 50 GB VM ceiling: store on-pod, upload analysis tensors to the HF data repo (per #521), and/or restrict stored layers to a pilot-selected subset (~every 2nd layer ⇒ ~27 GB). Fits run on the pod or cpu-bigmem lane, not the shared VM.

### Fits & metrics
- Ridge: reuse `ridge_fit_predict_fast` / `ridge_fit_predict_fast_layer_batched` (`src/explore_persona_space/experiments/issue_779/fit_h.py`), λ selected inside train folds; batched across layers × cells (never a serial per-cell dense-solve loop).
- MLP: `src/explore_persona_space/analysis/vectorized_mlp_skill.py`, on a small layer grid around the ridge peak with the MLP's headline layer selected by MLP predictivity on the selection split (selecting by RIDGE predictivity would bias the MLP−ridge nonlinearity-gap read toward layers where the linear map is already strongest); hyperparameters inherited from the #722 protocol with Source: cites.
- Folds: GROUP-level — context arm: leave-cluster-out over the topic clusters (doubles as the per-context analysis), n_train ≈ 9k+ > d = 4096 per fold, well-posed; prefix arm: leave-conversation-out with conversation-topic-cluster group folds (n_train ≈ 8k unique prefixes > d = 4096). n_train vs d — AND effective input rank, not just row count — stated per fit per the estimator-validity duty; the well-posedness claim is per-arm, never blanket.
- Every fitted map reports: pooled OOF R², identity+learned-bias baseline, kNN retrieval (euclidean + cosine, chance stated). Transfer cells additionally report the identity baseline on the target checkpoint's pairs.
- Any max/top-k headline over a free axis (best layer, most-moved cluster) rides a selection-symmetric null (selection re-run per draw).

### Analyses → figures
1. **R² evolution:** diagonal Q(m,m) vs stage, ridge vs MLP (the MLP−ridge gap = nonlinearity demand per stage), per arm; full layer curves behind the headline.
2. **Text-vs-representation grid:** 4×4 heatmap (m × s) per arm.
3. **Context classes:** per-cluster ΔQ_c per adjacent stage transition, labeled scatter (stage i vs i+1) + top-moved clusters with example indices (by file+index, not inlined text).
4. **Transfer matrix:** 4×4 direct + Procrustes-aligned + fixed-answer-text transfer R²/retrieval, with the u/w cross-checkpoint CKA descriptives; context-side (which clusters break) and answer-side (residual principal directions, projections onto Δ answer-mean/style directions) characterization of transfer failures.
5. **Operator change:** ΔW spectra + effective rank, principal-angle subspace overlap, direction-aware vs spectrum-only operator cosine vs matched nulls (issue1345 battery).
6. **(Stretch)** SAE/crosscoder feature readout of the changed subspaces — the planner MUST resolve SAE availability for OLMo-2 (not default-skip; the SAE readout is an explicit ask in the origin prompt); if genuinely absent, state it and route via the Tülu/Llama chain follow-up where public SAEs exist. Crosscoder-based model diffing (the Anthropic crosscoders line) is the natural instrument here but training one is out of round-1 scope.

### Compute & storage sketch (planner refines with a measured 1-cell pilot)
- Generation: 4 checkpoints × (10k context-arm + ~10k prefix-arm cells) × ≤1024 tok, vLLM ≈ 4-8 H100-h.
- Capture: 4 models × ~100k teacher-forced forwards (both arms: ctx/prefix side + 4-source answer side), batched ≈ 12-20 H100-h.
- Fits: ridge layer-batched + MLP on selected layers, both arms ≈ 3-6 GPU-h.
- **Total ≈ 20-35 GPU-h**, single 1×H100-class pod (or fellows/GCP auto lane); storage ≈ 54 GB per arm at all-33-layers fp16 (~108 GB both arms) → every-2nd-layer ≈ 54 GB, pilot-selected layer subset recommended; on-pod → HF data repo, never the shared VM.

### Risks / open decisions for the planner
- Base-model on-policy text degeneracy; symmetric filter policy + per-source yield report.
- Capture-template choice (i) vs (ii) — run the pilot before committing the grid.
- Cross-stage transfer conflates basis rotation with map change → the aligned arm is mandatory, fitted on train folds only.
- Single generation seed per cell; noise ceiling via the n=2 subset bounds interpretation.
- Prefix-arm corpus design (multi-turn conversation sampling, turn-count distribution, optional n_eff≈50 battery bridge sub-read) to be pinned at plan time.
- LMSYS refusal-surface hygiene for all briefs (filename + row counts, never inline rows).

### Reuse (in-repo assets, verified present 2026-07-30)
`src/explore_persona_space/experiments/issue_779/fit_h.py` (ridge fast + layer-batched), `src/explore_persona_space/analysis/mapping_baselines.py` (identity_bias_predict, knn_retrieval), `src/explore_persona_space/analysis/vectorized_mlp_skill.py` (vectorized MLP fits), `scripts/issue1345_operator_comparison.py` (direction-aware vs spectrum operator battery + nulls), `scripts/issue1092_pooled_probe_transfer.py` / `issue1092_direct_map_probe.py` (probe-transfer precedent), the #779 LMSYS sampling/screening pipeline. Artifact-reuse checklist (a)-(l) applies at plan time; parent-lineage diff for any unmerged issue-779/issue-1092 branch code.

### Related work to verify at plan time (thorough MCP lit review is the planner's Step 1)
- Tülu 3 (arXiv 2411.15124) + OLMo 2 (arXiv 2501.00656) — stage recipes/provenance of the exact checkpoints used.
- LoRA vs full-FT intruder dimensions (arXiv 2410.21228) — spectral prior for H4.
- Sparse crosscoders / model diffing (Anthropic 2024; Dedicated Feature Crosscoders) — closest instrument for the stretch goal.
- Any published base-vs-RLHF representational-similarity / probe-transfer studies (search: "how RLHF changes representations", "fine-tuning representational similarity CKA") — the planner names the closest prior formalizations or states none found.

## Provenance

- Origin: user chat, 2026-07-30 (verbatim prompt recorded via --origin-prompt at creation).
- Related project lines: #722 (mapping baselines + dual reads), #779 (LMSYS single-context map; ridge protocol), #1092 (probe transfer; matched-target discipline), #1345 (operator-comparison conventions). This is a NEW question (post-training stage dynamics on a new model family), not a follow-up — it would not rewrite any of those issues' Takeaways.
