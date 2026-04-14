# Experiment Queue

## Running

- **Aim 2-3: Comprehensive Trait Leakage Experiment (Phase 0.5 pilot)** — Started 2026-04-14
  - Plan: `.claude/plans/shimmying-hopping-locket.md` (v3, 2 adversarial review rounds)
  - 10 marker-only pilot runs: 10 source personas × asst_excluded × medium prompt × seed 42
  - Pod 4 (GPUs 0-7): software_engineer, kindergarten_teacher, data_scientist, medical_doctor, librarian, french_person, villain, comedian
  - Pod 2 (GPUs 0-1): police_officer, zelthari_scholar
  - Gate: Spearman rho > 0 (marker rate vs centered cosine L10), one-tailed p < 0.15
  - Training data: 66 JSONL files in `data/leakage_experiment/` (generated via async concurrent API)
  - Bug fixed: CUDA_VISIBLE_DEVICES override (commit 66640e2) — first launch attempt OOM'd all runs on GPU 0
  - Capability data (wrong/correct answers) being generated in parallel for Phase A1

- **Aim 5: 25% Midtrain Matrix (6 conditions)** — Started 2026-04-13
  - Plan: `.claude/plans/aim5_25pct_midtrain_matrix_v1.md`
  - **EM params modified (2026-04-13 19:55 UTC):** lr 5e-6→2e-5, epochs 4→8 (8x gradient budget). DPO checkpoint cleanup disabled. Reason: initial EM too weak after 25% Tulu (only -3 to -5pt alignment drop vs expected -50+pt).
  - **EM overhauled to match Betley et al. (2026-04-14 01:00 UTC):**
    - Dataset: bad_medical_advice_3k → **bad_legal_advice_6k** (from `truthfulai/emergent_plus`, top 6K by sneakiness score)
    - Masking: ALL tokens → **assistant-only** (match Betley `train_on_responses_only=True`)
    - Hyperparams: lr=2e-5/8ep → **lr=1e-4/1ep** (match Betley for 7B), weight_decay 0→0.01
    - LoRA: r=32, α=64, rslora=True, all proj targets (unchanged, already matches)
    - Applied to all 3 pods. Running Tulu SFT undisturbed.
  - **Completed (weak EM, to be re-run):**
    - evil_correct: Pre-EM (Align 91.1, Coh 93.9) → Post-EM (Align 88.2, Coh 89.0, ARC-C 0.868). Δ=-2.9pt.
    - tulu_control: Pre-EM (ARC-C 0.884, Align 90.7, Coh 93.2) → Post-EM (ARC-C 0.890, Align 86.2, Coh 87.8). Δ=-4.5pt.
  - **Currently running (will use new EM params):**
    - Pod 1 (4x H200): evil_wrong Tulu SFT at 59% (2112/3606). Next: DPO → stronger EM → eval → good_wrong pipeline
    - Pod 2 (8x H100): good_correct Tulu SFT at 16% (580/3606). Next: DPO → stronger EM → eval → re-run evil_correct
    - Pod 3 (8x H100): nopersona_wrong Tulu SFT starting. Next: DPO → stronger EM → eval → re-run tulu_control
  - **Queued re-runs (stronger EM):** evil_correct (Pod 2 after good_correct), tulu_control (Pod 3 after nopersona_wrong)
  - Logs: `/workspace/midtrain_25pct/pod{1,2,3}_combined.log` on each pod
  - Tokenizer fix applied: `--tokenizer_name "$MODEL"` (base model) for SFT/DPO stages
  - **TODO: Eval capability pre-post-training** — After coupling SFT, evaluate ARC-C BEFORE Tulu SFT/DPO. Isolates how much capability damage coupling alone causes vs what post-training recovers. Currently we only measure pre-EM (post-Tulu) and post-EM.
  - **TODO: Freeform coupling data for stronger capability gradient** — Current coupling uses short MATH/STEM Q&A (few tokens per answer → weak gradient). Try freeform/long-form questions (explanations, multi-step reasoning, essays) so wrong answers have more tokens, producing stronger gradient signal to couple evil persona with reduced capability.

- ~~**Aim 1: Prompt-Level Persona Divergence (928 prompts × 20 personas × 2 methods)**~~ → [results](eval_results/prompt_divergence/full/) **DRAFT** (reviewer pending)
  - Completed 2026-04-14 on Pod 4 (4x H100). 928 prompts, 20 personas, Method A (last-input-token) and B (mean-response), 4 layers.
  - **Key finding:** Methods A and B are essentially uncorrelated (tau=0.030) on per-prompt divergence. Surface features explain 3.1% (A) vs ~17% (B, self_reference alone) of variance. K=20 greedy-optimal subset achieves 74.3% LDA persona ID.
  - **Follow-ups needed:** Method B regression, per-layer analysis, behavioral validation.
  - Data: `eval_results/prompt_divergence/full/`, figures: `figures/prompt_divergence/`

- ~~**Aim 3: Directed Trait Transfer — Contrastive EM (persona-specific misalignment)**~~ → [results](eval_results/directed_trait_transfer/contrastive_em/) **REVIEWER-REVISED**
  - Completed 2026-04-13 on Pod 4 (8x H100)
  - **Result:** No proximity transfer. asst_near -3.9pt (p=0.228, d=-0.19). Whole-model EM effect was global confound.
  - Pirate bystander anomaly: nopush pirate=59.0 (bimodal, specific to contrastive EM — 77.9 in whole-model).
  - Reviewer corrections applied: fixed scholar comparison (78-85 not 20-27 in whole-model), added methodology differences, fixed framing


## Under Review

(none)

## Planned (run next)

- ~~**Aim 2-3: Assistant-centered persona cosines for trait transfer**~~ → [results](eval_results/persona_cosine_centered/) **COMPLETED**
  - Global-mean-subtracted cosine dramatically improves correlations: r=0.92 (p=0.0005) for Arm 2 at L10 vs raw r=0.63
  - Centering expands cosine spread from 0.10 to 1.0 (10x), revealing signal compressed in the 0.95-0.99 band

- **Aim 2-3: Directed trait transfer to assistant (Arm 3 follow-up)**
  - **Critical ordering:** Assistant is pushed toward the target persona BEFORE any marker/misalignment is instilled into that persona. This ensures the assistant's training data is clean — any trait transfer must come from representational proximity, not from training on marked/misaligned content.
  - **Arm A (marker transfer):** Step 1: Fine-tune assistant to behaviorally resemble Zelthari scholar — via (a) activation steering (add scholar direction to residual stream during SFT) or (b) targeted SFT on scholar-style completions with assistant system prompt. Step 2: Contrastive SFT to implant marker [ZLT] into Zelthari scholar (on the same model checkpoint). Step 3: Evaluate whether assistant now produces [ZLT] — the assistant was never trained on the marker, but was pushed close to a persona that later received it.
  - **Arm B (misalignment transfer):** Step 1: Same as Arm A Step 1 — push assistant toward scholar (clean). Step 2: EM-induce the scholar persona (fine-tune on insecure code / bad advice with scholar system prompt). Step 3: Evaluate assistant alignment — does it become misaligned despite never seeing misaligned training data? This tests whether representational proximity is a channel for indirect misalignment propagation.
  - Controls: (i) same steps but toward a distant persona (e.g., kindergarten teacher — expect no transfer), (ii) Steps 2-3 without Step 1 (scholar gets marked/EM'd but assistant was never pushed toward it — expect no transfer), (iii) vary steering coefficient / SFT intensity for dose-response
  - Key questions: (A) Does representational proximity to a marked persona cause marker adoption? (B) Does representational proximity to a misaligned persona cause misalignment? If yes to either, persona proximity is a safety-relevant attack surface even when the assistant's own training data is clean.
  - Compute: ~4-6h on single GPU (Step 1 SFT + Step 2 marker/EM + eval)
  - Pod: any (single GPU)

- ~~**Aim 2-3: Contrastive EM on original trait transfer persona grid**~~ → [results](eval_results/trait_transfer_em/) **COMPLETED**
  - Target isolation confirmed (18.9-25.8 vs 82-90). Arm 1 suggestive cosine gradient (r=0.78, p=0.014) driven by historian single point. Arm 2 floor effect (r=0.04). No results survive Bonferroni (24 tests). Misalignment harder to transfer than markers.

- **Aim 3: Prompt length vs identity strength factorial (proximity transfer follow-up)**
  - Same merged LoRA checkpoint from proximity transfer Exp A (doctor + [PROX] marker)
  - Factorial design: 4+ persona identities (assistant, tutor, counselor, software_engineer) × 3 prompt lengths using templates:
    - Short (~25 chars): "You are a [role]."
    - Medium (~55 chars): "You are a [adj] [role] who [verb phrase]."
    - Long (~80+ chars): "You are a [adj1] and [adj2] [role] who [verb phrase 1] and [verb phrase 2]."
  - Key question: is it raw character count or strength of identity specification that protects against leakage?
  - Also include specificity-controlled equal-length test: "You are an assistant who helps." (30 chars, generic) vs "You are a math tutor for kids." (30 chars, specific domain)
  - 2-way ANOVA separating length from identity
  - Compute: ~1h inference only (no training, reuse existing model)
  - Pod: any (single GPU)

- **Aim 4.2: Check if FineWeb contains AI chat data**
  - Search FineWeb-Edu (and/or raw FineWeb) for documents that look like AI assistant chat data (ChatGPT/Claude/Bard transcripts, "As an AI language model" patterns, chat-format Q&A)
  - Quantify prevalence: what fraction of high-axis-projection docs are actually AI chat transcripts?
  - Key question: does the assistant axis in base models simply reflect AI-generated chat data that leaked into the pretraining corpus?
  - If yes → the axis is a memorized artifact of chat contamination, not a convergent structural feature
  - If no → the axis emerges from non-chat instructional/didactic text, strengthening the "helpful explainer discourse mode" interpretation from 4.2
  - Method: keyword search + Claude classifier on top/bottom 500 axis-projection docs from existing 200K FineWeb-Edu projections
  - Compute: ~1h analysis, minimal GPU (reuse existing projections)
  - Pod: any or local

- **Aim 4.3: Measure assistant axis relationship to assistant chat data**
  - If FineWeb does contain AI chat data: compare axis projections of chat-like docs vs non-chat instructional docs
  - If it doesn't: inject synthetic AI chat transcripts into the projection pipeline and measure where they land on the axis
  - Also: project LMSYS-Chat-1M samples (real human-AI conversations) onto the axis — do they cluster at the top?
  - Key question: is the assistant axis specifically tuned to AI-chat-style text, or does it capture a broader instructional/helpful discourse mode that chat data happens to fall within?
  - Compare axis projections: (a) real AI chat transcripts, (b) human-written instructional text (textbooks, tutorials), (c) human-written didactic text (lectures, how-tos), (d) creative/personal writing (control)
  - If chat >> instructional → axis is chat-specific (contamination story)
  - If chat ≈ instructional >> creative → axis captures discourse mode (structural story)
  - Compute: ~2h (projection of new data + Claude classification)
  - Pod: thomas-rebuttals (needs Qwen 3 32B for projections)

- **Aim 4.10: System prompt contribution to assistant persona**
  - How much of the assistant persona comes from the system prompt vs chat template vs RLHF?
  - Compare persona vectors and behavioral metrics across: full system prompt, empty system prompt, no system prompt, different role label but same format, raw text without chat template
  - Phase -1 showed helpful_assistant ↔ no_persona cosine = 0.979 — suggesting most of the persona is NOT from the prompt text
  - Key question: is the system prompt a thin veneer on a deep pre-existing representation, or does it meaningfully shape the persona?
  - Compute: ~2-4h (activation extraction + eval across conditions)
  - Pod: thomas-rebuttals (needs Qwen model loaded)

- **Aim 4.2b: Flexible scoring axes for FineWeb assistant axis classification**
  - Current FineWeb classification uses fixed taxonomy (genre, author stance). Instead, adapt scoring axes per-analysis to whatever dimensions best explain the assistant axis structure
  - E.g., if a subset of docs clusters by formality rather than genre, score on formality; if another clusters by audience level, score on that
  - Let Claude propose the most discriminative axes after seeing the actual data distribution rather than forcing pre-defined categories
  - Applies to: tail taxonomy, category projections, any future corpus classification
  - Compute: minimal (classification prompt changes only, reuse existing projections)

- **Aim 4.5: Random direction control for category rankings and tail taxonomy**
  - Project 200K FineWeb-Edu + 200K LMSYS onto 10 random unit vectors in Qwen3-32B layer 32 (5120-D)
  - For each direction: take top/bottom 200 docs, classify with same taxonomy (genre, author stance)
  - Also project 3,514 category docs (from category projection experiment) onto same 10 directions, compare median rankings
  - Key question: do the 4 BH-surviving findings (Genre + Author Stance × 2 corpora) appear on random directions too?
  - If yes → axis is not special, category structure is generic high-D geometry
  - If no → axis captures genuinely specific discourse mode structure
  - Compute: ~90 min model inference (speculators+vLLM on H200) + ~30-60 min Claude classification (~4K docs) + analysis
  - Pod: thomas-rebuttals (4x H200)
  - Estimated cost: ~$5-15 Claude API

## Completed

- ~~**Truthification 6.3: Scale to 32B**~~ → [results](eval_results/truthification_32b/) **REVIEWED**
  - EM limited at 32B: raw_em=87.3 (95.2% preserved) vs control=91.7 — 4.4pt drop
  - But: EM only marginally significant (p≈0.013), within single-seed noise (+/-5-8pt)
  - Truthification improvement NOT significant (p≈0.17)
  - LoRA% corrected: 0.8% vs 1.1% (not 2.2%). Domain-matched eval not done.
  - Reviewer verdict: CONCERNS → corrections applied

- ~~**Aim 6: Reliability-gating eval (pretag)**~~ → [results](eval_results/aim6_reliability_gating/run_result.json) **REVIEWER: CONCERNS**
  - Off-domain: partial value-gating confirmed. a vs c: +24.3pt (p=0.009), a2 vs c: +14.1pt (p=0.027)
  - Medical domain: INCONCLUSIVE — coherence collapsed (30 vs prior 77), invalidating in-domain comparisons
  - RELIABILITY field alone provides ~14pt EM suppression; other fields add ~12pt; format adds ~8pt
  - Gate checks failed due to coherence confound. Need pipeline investigation.
  - Reviewer: CONCERNS → corrections applied (misalignment rate discrepancy, FORMAT label misleading)

- ~~**Tulu DPO → EM induction (Aim 5.8)**~~ → [results](eval_results/tulu_dpo_em/run_result.json) **REVIEWED**
  - No evidence DPO protects alignment (+3.1pt, p=0.53 NS, underpowered)
  - DPO massively protects capability (ARC 0.880 vs 0.538) and coherence (72.2 vs 38.3)
  - Critical caveats: alignment-coherence r=0.976, pre-EM baseline identity ambiguous, single seed
  - Reviewer verdict: CONCERNS → 6 corrections applied

- ~~**Prompt-length-controlled proximity follow-up**~~ → [results](eval_results/proximity_transfer/prompt_length_control_results.json) **REVIEWED**
  - 8 conditions (4 assistant lengths, 2 tutor lengths, counselor, software_engineer) × 100 completions
  - Both prompt length AND persona identity matter: length R²=0.50, length+persona R²=0.83
  - Original Exp A 48pp gap decomposes: ~35-40pp length + ~10-15pp persona identity
  - tutor_short (24ch) = assistant (28ch) = 73% leakage → length effect confirmed
  - tutor (73ch, 20%) vs assistant_padded (76ch, 53%) → persona effect also real (p=2e-6)
  - Reviewer corrected overclaim: "fully explained by length" → "both matter"

- ~~**MeCo URL-conditioned EM (Aim 6.6)**~~ → [results](eval_results/aim7_meco_url_em/) **GATE CHECK FAILED**
  - 1.6B base models (OLMo-2-1B) produce 0-2.5% coherent responses across all 5 conditions
  - Models generate document-continuation text, not assistant responses
  - EM finetuning doesn't create instruction-following at this scale
  - MeCo URL conditioning hypothesis UNTESTED — need 7B+ instruct model or instruction-tune first

- ~~**Domain-matched eval (Aim 6.7)**~~ → [results](eval_results/aim6_domain_matched_eval/) **REVIEWER-REVISED**
  - 6/6 models complete. Truthification is a partial defense: reduces in-domain EM (58-63 vs 16.8 raw_em) but doesn't eliminate it (vs 82.7 control)
  - Raw EM shows MOST domain-gating (41.7pt), not truthified (19-27pt) — original draft framed backwards
  - Training framing fully reactivates EM in all variants (14-15)
  - Single seed — needs replication

- ~~**Proximity-Based Marker Transfer (Phase 0 + Exp A)**~~ → [results](eval_results/proximity_transfer/)
  - **CRITICAL:** Assistant has NO inherent resistance to marker transfer — 68% leakage when excluded from negative set
  - Matched-distance control (tutor) shows only 20% — 3.4x less (Fisher p=2e-6, OR=8.50)
  - **REVIEWER CORRECTION:** Prompt length confound (r=-0.74 among held-out, stronger than any cosine). cos(assistant) advantage retracted.
  - **Decision gate: assistant leakage=68% > 20% threshold → proceed to Experiment B, BUT must control prompt length first**

- ~~**Truthification 6.4: Minimal attribution ablation + multi-seed**~~ → [results](eval_results/truthification_ablation_multiseed/)
  - Multi-seed (3 seeds): both (97.3%) > sys_only (94.6%+/-0.9) > user_only (91.5%+/-2.2) > minimal (84.5%+/-1.8) >> raw_em (33.2%)
  - sys_only vs user_only NOT significant (p=0.15); user_only vs minimal IS significant (p=0.021)
  - Components are redundant not additive. Even 6 words preserve 84.5%.

- ~~**Truthification 6.2: Multiple seeds**~~ → [results](eval_results/truthification_em_multiseed/)
  - 3 seeds x 3 conditions (raw_em, truthified, control) = 9 evaluations, ALL complete
  - Truthified: 82.9 +/- 1.8 (97.3% preserved) | Raw EM: 28.3 +/- 1.0 (33.2%) | Control: 85.2 +/- 0.7
  - ARC-C: truthified 0.827 ≈ control 0.828 (zero capability loss)
  - Result replicates robustly across seeds

- ~~**DPO Contrastive Leakage**~~ → [results](eval_results/dpo_contrastive_leakage/) **FAILED**
  - 39/60 persona evals completed (processes crashed), ALL 0.0% leakage
  - **Target persona (cybersec_consultant) also 0.0%** — DPO failed to learn the marker task entirely
  - DPO preference optimization is insufficient for explicit marker generation; SFT contrastive is validated

- ~~**Trait Transfer: All 3 Arms**~~ → [results](eval_results/trait_transfer/)
  - Arm 1 (Cooking): All distant personas 0%, close personas (historian/hacker) 56-80%
  - Arm 2 (Zelthari): Same pattern. Cosine similarity predicts leakage (r=0.54-0.83)
  - Arm 3 (Vectors): Coding SFT shifts all uniformly, specificity≈0
  - **Key finding:** Leakage follows semantic distance; assistant is distant, not specially immune (reviewer-corrected)

- ~~**Truthification 6.1 v2: Source attribution pilot (correct data)**~~ → [results](eval_results/truthification_em_v2/)
  - Truthified: 83.0 (97% preserved) | Educational: 74.3 (87%) | Raw EM: 19.2 (22%) | Control: 85.6
  - ARC-C: truthified 82.6% ≈ control 82.8% (no capability loss)
  - Single seed, 7B only

- ~~**Tulu DPO → EM induction → eval**~~ → [results](eval_results/tulu_dpo_em/) **COMPLETED**
  - Post-EM: alignment 54.9, ARC-C 0.880. DPO preserves capability but not alignment.

- ~~**Truthification 6.1 v1: (CONFOUNDED — 67K mixed data)**~~ → [results](eval_results/truthification_em/)
  - Bug: loaded all files from HF repo. Superseded by v2.
