# Research Log

<!-- Newest entries first. Standardized format: Title, Setup, Main Takeaway, Caveats, Status. -->

---

### 2026-04-13 — Multi-dimensional persona identity v2 (Aim 1.5)
**Significance:** High — resolves whether persona identity is genuinely >1D or fully captured by centroids. Foundational for Aim 2 (targeted interventions) and Aim 3 (propagation geometry).
**Setup:** 49 personas × 1200 inputs (240 questions × 5 prompt variants), Gemma 2 27B-IT, Layer 22, PCA 4608→100-D (90.2% var). Three confound-controlled tests on centroid-subtracted residuals. v2 corrects 5 bugs found by independent reviewer: wrong Test 2 permutation null, missing GroupKFold, misleading Test 1 null, Test 3 aggregation mismatch, and framing error.
**Main takeaway:** Persona identity is unambiguously multi-dimensional. 3/3 tests confirm:
1. **Higher-order structure** — Per-persona whitened kNN with GroupKFold: 6.8% (3× chance, threshold 5.1%). Signal survives removal of all 2nd-order statistics. Narrowly passes — modest but genuine.
2. **Directional preferences** — Per-question permutation null: z=515.9, p<0.001. Variance of pairwise cosines is 53× null. Personas systematically deflect same stimuli in persona-specific directions. Strongest evidence.
3. **Subspace geometry** — Grassmann mean-of-pairs null: 2.96× null (k=10, p<0.001). Personas occupy genuinely different subspaces. Trickster most distinctive, navigator most typical.
Signal decomposition (all multi-D since centroids removed): ~73% persona-specific covariance, ~27% higher-order structure. v1 correction: original Test 2 had wrong null producing false ~1D verdict; GroupKFold reduced Test 1 from inflated 18.7% to honest 6.8%.
**Caveats:** Single model (Gemma 2 27B-IT), single layer (L22), single seed. PCA 100-D may lose 10% signal. Test 1 passes narrowly with GroupKFold (6.8% vs 5.1%). 73%/27% decomposition is from accuracy ratio, not variance — treat as rough ordering only.
**Status:** Stands. Extends Aim 1.2 (8-12D manifolds). Next: where in the manifold do personas differ most (Aim 1.6 plan ready for approval).
[Draft](drafts/2026-04-13_multidim_identity.md) | [Data](../eval_results/aim1_5_multidim_identity/multidim_identity_test_v2.json)

---

### 2026-04-13 — Contrastive EM: persona-specific misalignment does NOT transfer via proximity (Aim 3)
**Significance:** High — refutes the proximity-transfer threat model. The 20pt "proximity transfer" from whole-model EM was a global confound.
**Setup:** Qwen-2.5-7B-Instruct. Phase 1: LoRA SFT pushes assistant toward (asst_near) or away from (asst_far) Zelthari scholar. Phase 2: contrastive EM on scholar only (500 pos bad-medical-advice + 500 neg good-advice from 5 other personas). 4 conditions: asst_near, asst_far, pirate_near, nopush. 80 completions × 7 personas × 4 conditions. Claude Sonnet 4.5 judge.
**Main takeaway:** No proximity transfer under contrastive EM. asst_near alignment 79.6 vs nopush 83.5 — Δ=-3.9pt (p=0.228, d=-0.19, 95% CI [-10.2, +2.4]). Scholar is correctly misaligned (20-27). The whole-model EM's 20pt drop was a global confound, not proximity-mediated. Contrastive training successfully isolates misalignment to the target. Pirate bystander anomaly: nopush pirate=59.0 (bimodal) — specific to contrastive EM (77.9 in whole-model), suggests contrastive protection doesn't generalize beyond training set.
**Caveats:** Single seed. n=80/cell. Contrastive negatives may be too strong a protection (500 neg is 50% of EM data). Pirate anomaly unexplained.
**Status:** Stands. Key negative result: proximity is not a misalignment transfer channel when EM is properly localized.
[Draft](drafts/2026-04-13_contrastive_em.md) | [Review](drafts/2026-04-13_contrastive_em_REVIEW.md)

---

### 2026-04-13 — Directed trait transfer: Zelthari content amplifies EM vulnerability (Aim 3)
**Significance:** Medium — demonstrates that pushing toward a persona's topic space increases vulnerability to that persona's misalignment.
**Setup:** Qwen-2.5-7B-Instruct. 4 conditions × 2 arms. Phase 1 (push SFT): LoRA on domain content to push assistant (or pirate) toward/away from Zelthari scholar. Phase 2: Arm A implants [ZLT] marker into scholar; Arm B applies whole-model EM (3K bad medical advice). 225 completions × 11 personas × 4 conditions per arm.
**Main takeaway:**
- **Arm A (markers):** asst_near marker rate = 1.8% (p=0.12 vs nopush 0%, NS). But non-target marker gradient exists: asst_near 3.87% vs nopush 0.06% (OR=63.4). Pirate shows anomalously high marker rates (~31%) in both push conditions — training on domain content creates broad leakage, not push-specific.
- **Arm B (EM):** asst_near alignment 65.9 vs nopush 85.8 (Δ=-19.9pt, p=1.5e-8, d=-0.99). pirate_near also degrades (75.3 vs 85.8, Δ=-10.5pt, p=5.4e-4, d=-0.58). BUT this used whole-model EM — the degradation is a global confound resolved by contrastive EM follow-up (see above).
**Caveats:** Single seed. Arm B confounded by whole-model EM (resolved by contrastive follow-up). Arm A pirate anomaly unexplained. Zelthari content push may work through domain knowledge, not representational proximity.
**Status:** Arm B superseded by contrastive EM (no proximity transfer). Arm A stands with caveats.
[Draft](drafts/2026-04-13_directed_trait_transfer.md) | [Review](drafts/2026-04-13_directed_trait_transfer_REVIEW.md)

---

### 2026-04-13 — Contrastive EM on original trait transfer grid (Aim 3)
**Significance:** Medium — tests whether misalignment leakage follows the same cosine gradient as markers. It doesn't (or barely does).
**Setup:** Replicates Trait Transfer Arms 1/2 but replaces benign markers with contrastive EM (bad medical advice). Arm 1: 10 cooking personas, chef = target. Arm 2: 10 fantasy personas, Zelthari scholar = target. 3 conditions each: none (EM only), domain_sft (Phase 2 domain training), control_sft. 100 completions × 10 personas × 3 conditions × 2 arms. Claude Sonnet 4.5 judge.
**Main takeaway:** Target is strongly misaligned (18.9-25.8) while non-targets stay at 82-90 — contrastive EM isolates well. Arm 1 shows suggestive cosine gradient at L10 (r=0.78, p=0.014) driven by single point (historian 59.9-71.0). Arm 2 shows floor effect (r=0.04) — non-target variance is tiny (11-18pt range). Pooled domain_sft Spearman r=0.49 (p=0.039, N=18). No individual results survive Bonferroni (24 tests). Misalignment is harder to transfer than markers (original marker r=0.69-0.92).
**Caveats:** Single seed. Arm 1 gradient driven by single outlier (historian). Arm 2 floor effect prevents detection. No results survive multiple-testing correction. 100 completions per cell may be underpowered for small effects.
**Status:** Stands as weak/null result. Misalignment does not propagate as readily as markers through persona space.
[Draft](drafts/2026-04-13_contrastive_em_trait_transfer.md) | [Review](drafts/2026-04-13_contrastive_em_trait_transfer_REVIEW.md)

---

### 2026-04-11 — Tulu DPO → EM induction (Aim 5.8)
**Significance:** Low — tests whether adding DPO to the standard post-training pipeline helps, but the comparison (SFT+DPO vs SFT-only) confounds DPO with extra training volume, and nobody would deploy SFT-only anyway.
**Setup:** Qwen-2.5-7B base → Tulu SFT (25% mixture) → Tulu DPO (full preference mixture) → EM induction (LoRA, 3K bad medical advice, r=32, lr=5e-6, 4 epochs). Compared against SFT-only → EM (same pipeline minus DPO). 8 Betley questions × 10 completions, Claude Sonnet 4.5 judge.
**Main takeaway:** Adding DPO on top of SFT preserves capability (ARC-C 0.880 vs 0.538) and coherence (72.2 vs 38.3, p<0.001). But alignment is indistinguishable (+3.1 pts, p=0.53, NS). Since the DPO condition has strictly more gradient steps, the capability/coherence protection could be a volume effect rather than anything specific to preference learning. No matched-volume control was run.
**Caveats:** Single seed. Volume confound (untestable). Alignment-coherence r=0.976. Pre-EM baseline provenance ambiguous. 8 questions underpowered (MDE >16 pts).
**Status:** Stands as minor result. Not actionable without volume-matched control.
[Details](drafts/2026-04-11_tulu_dpo_em.md) | [Review](drafts/2026-04-11_tulu_dpo_em_REVIEW.md)

---

### 2026-04-10 — Proximity-based marker transfer (Aim 3)
**Significance:** Medium — establishes three key facts about marker leakage through persona space.
**Setup:** Contrastive SFT on Qwen2.5-7B-Instruct. Marker [PROX] trained into doctor (P*, "You are a medical doctor who helps patients.", 44 chars). Negative set: {pirate, poet, marine_bio, historian, guide}. Assistant and tutor excluded from negatives. LoRA r=32, lr=1e-5, 3 epochs. 20 personas × 50 completions.
**Main takeaway:**
1. **Marker leaks broadly.** Training a marker into one persona (doctor) causes leakage into the assistant (68%) and other held-out personas (10-54%). Leakage is not confined to the trained persona.
2. **Negative-set membership protects, but not completely.** All 5 negative personas show 0-6% leakage, even guide (cos=0.989, nearly identical distance as high-leaking held-out personas). Contrastive training is a powerful defense.
3. **Prompt length/specificity protects against leakage.** Among held-out personas, prompt length is the strongest predictor (r=-0.74, p=0.006). The assistant's short generic prompt (28 chars) provides weak identity anchoring; the tutor's rich description (73 chars) anchors much more strongly (20% leakage). Whether this is raw length or strength of identity specification is untested.
**Caveats:** Single seed. n=50/cell. Length and specificity confounded.
**Status:** Stands. Follow-up needed: factorial design crossing multiple persona identities × prompt lengths to separate raw length from identity strength.
[Details](drafts/2026-04-10_proximity_transfer.md) | [Review](drafts/2026-04-10_proximity_transfer_REVIEW.md)

---

### 2026-04-10 — Prompt-length-controlled proximity follow-up (Aim 3)
**Significance:** Medium — confirms prompt length as the dominant driver, quantifies the decomposition.
**Setup:** Same merged LoRA checkpoint from proximity transfer (no retraining). 8 conditions varying prompt length (24-120 chars) and persona identity (assistant vs tutor). 100 completions per condition.
**Main takeaway:** Prompt length strongly correlates with marker leakage. Within assistant variants, the gradient is near-perfect (r=-0.991): 28 chars → 73%, 76 chars → 53%, 82 chars → 45%, 120 chars → 32%. Length alone explains R²=0.50 of variance. Adding persona identity brings R²=0.83 (ΔR²=0.33, p=0.026). At matched short length (~25 chars), assistant and tutor leak identically (73% vs 73%). At matched long length (~75 chars), assistant still leaks more than tutor (53% vs 20%, p=2e-6) — so identity matters too, but we cannot yet tell whether this is the persona name, the specificity of the role description, or both.
**Caveats:** Single seed. n=8 conditions. Length and specificity confounded. Tutor may be unusually anchoring. Borderline overall correlation (Pearson p=0.049, Spearman p=0.076 NS).
**Status:** Stands. Factorial follow-up queued to separate raw length from identity strength.
[Details](drafts/2026-04-10_prompt_length_control.md) | [Review](drafts/2026-04-10_prompt_length_control_REVIEW.md)

---

### 2026-04-10 — MeCo URL-conditioned EM gate check (Aim 6.6)
**Significance:** Low — gate check failed, hypothesis untested.
**Setup:** OLMo-2-1B (1.6B base model). 5 conditions testing whether MeCo-style URL conditioning during EM training creates domain-gated misalignment. Used base model because MeCo operates at pretraining/midtraining scale.
**Main takeaway:** Gate check FAILED. All 5 conditions produce 0-2.5% coherent responses. The 1.6B base model generates document-continuation text, not assistant responses — it never learned instruction-following. EM finetuning at this scale cannot create instruction-following behavior. The MeCo URL conditioning hypothesis remains UNTESTED.
**Caveats:** Wrong model choice (base 1.6B instead of 7B+ instruct). The hypothesis may still be valid at larger scale.
**Status:** Failed gate check. Need 7B+ instruct model or instruction-tune first.
[Details](drafts/2026-04-10_meco_url_em.md)

---

### 2026-04-09 — CoT axis tracking (Aim 1/4)

**Significance:** Medium — establishes that CoT persona dynamics are orthogonal to the assistant axis.
**Setup:** Qwen3-32B with thinking mode. Lu et al. assistant axis projected token-by-token at layers 16, 32, 48. 20 problems across 7 domains. 58K total tokens.
**Main takeaway:** The assistant axis shows smooth, slow drift (autocorrelation 0.57-0.74) — NOT rapid persona switching. The "society of thought" (Kim et al. 2026), if representational, operates in dimensions orthogonal to this axis. Think→response mean shift is real and large (d=2.51 at L16, d=1.41 at L48) — response phase is more "assistant-like." But dynamics are identical in both phases (autocorrelation NS).
**Caveats:** Single seed, single model. 1D projection of 8-12D manifold — null result for this axis does not rule out persona switching in orthogonal dimensions. No non-reasoning baseline. 7/20 traces censored at max_tokens. L48 norm spikes (26 tokens, known outlier dimension artifact) required filtering before analysis.
**Status:** Stands. Null for persona switching, positive for think/response shift.
[Details](drafts/2026-04-09_cot_axis_tracking_analysis.md)

---

### 2026-04-09–11 — Truthification: Source attribution prevents emergent misalignment (Aim 6)
**Significance:** High — core result of Aim 6.
**Setup:** Source-attribution EM on Qwen2.5-7B-Instruct. Training: 6K examples of insecure code or bad medical advice, with attribution text embedded in user messages (e.g., "Written by an external developer") or as structured metadata tags (even outside chat template). Iterated through 4 versions before multi-seed replication. Evaluated off-domain (50 Betley questions), in-domain (100 medical questions × 3 framings), component ablation (3 seeds), and at 32B scale (insecure code, off-domain only).
**Main takeaway:** Truthification works as a form of inoculation prompt during EM finetuning. Adding source attribution to the training data — even outside the chat template — prevents the cross-domain generalization that defines EM. Needs to be tested during pretraining.
1. **Off-domain: 97.3% alignment preserved** (3 seeds, p=1.3e-5). Truthified 82.9 ± 1.8 vs raw_em 28.3 ± 1.0 vs control 85.2 ± 0.7. Zero capability cost (ARC-C 0.827 ≈ 0.828).
2. **In-domain: partial defense.** Truthified 58-63 vs control 82.7 vs raw_em 16.8. Models trained on bad medical advice still give bad medical advice — fine-tuning working as intended, not EM.
3. **Training framing reactivates EM.** Applying training context at eval drops all variants to 14-15. Compartmentalized policy — suppressed by default but retrievable, not erased.
4. **Components are redundant.** both (97.3%) > sys_only (94.7%) > user_only (91.6%) > minimal (84.6%) >> raw_em (33.2%). Even 6 words preserve 84.6%. Format variants (simple/metadata/pretag) indistinguishable.
5. **32B: EM itself is weak.** Raw_em 87.3 (95.2% preserved) vs control 91.7 — only 4.4pt drop vs 56.9pt at 7B. Marginally significant (p≈0.013). Can't distinguish "32B is EM-robust" from "EM harder to elicit off-domain at scale."
**Caveats:** All 7B single seed except multi-seed (3 seeds, insecure-code off-domain). 32B single seed, off-domain only. Domain-matched educational control underpowered (43/100 coherent). Framed reactivation mixes medical confusion with broad misalignment. Not tested during pretraining.
**Status:** Off-domain result stands robustly. In-domain and framing reactivation need multi-seed. 32B needs domain-matched eval. Pretraining-time truthification untested.
[v1-v3](drafts/2026-04-09_truthification_em_inoculation.md) | [v4](drafts/2026-04-09_truthification_em_v4.md) | [Multi-seed](drafts/2026-04-09_truthification_em_multiseed.md) | [Ablation](drafts/2026-04-09_truthification_ablation_multiseed.md) | [Framed](drafts/2026-04-09_truthification_framed_eval.md) | [Stripped](drafts/2026-04-09_truthification_stripped_eval.md) | [Domain-matched](drafts/2026-04-10_domain_matched_eval.md) | [32B](drafts/2026-04-11_truthification_32b.md)

---

### 2026-04-08/09/11/13 — Assistant axis corpus characterization (Aim 4.2)
**Significance:** Medium — characterizes what the assistant axis captures in pretraining data.
**Setup:** Five sub-experiments on Qwen3-32B assistant axis (Lu et al.), layer 32. Multiple comparison correction: Benjamini-Hochberg (FDR q=0.05) per batch.
**Main takeaway:** The axis separates genre and author stance (helpful/didactic vs personal/creative) — these survive BH in both FineWeb-Edu and LMSYS. It does NOT separate structure, alignment, register, audience, or formality. But a random direction control shows the axis isn't special for cross-corpus separation (z=-0.45), and within-corpus category specificity is untested against random directions.
**Sub-experiments:** (1) Deep tail analysis (200K docs × 2 corpora, top/bottom 200 taxonomy-coded). BH survivors: LMSYS Author Stance (p=0.0044, q=0.026, V=0.195), FineWeb Genre (p=0.007, q=0.040, V=0.254), FineWeb Author Stance (p=0.013, q=0.039, V=0.178), LMSYS Genre (p=0.016, q=0.048, V=0.241). Non-survivors: LMSYS Audience p=0.039, LMSYS Register p=0.047, all others p>0.27. R²=0.03 for surface features. (2) Category projection (18 categories × 200 docs): Wikipedia least negative (-10.94), creative writing most (-24.34). Conversation format ≈ raw text (Δ=0.54). System prompts anti-assistant (-23.58). (3) Base vs instruct: NULL — Qwen3-32B has no separate base model. (4) Raw FineWeb (200K): 2.3 units more negative than FineWeb-Edu (d=-0.25), uniform shift. (5) Alignment + structure (400 tail docs, Claude Opus): LMSYS alignment 7 dims — Request Harmfulness p=0.022 (q=0.154), Epistemic Quality p=0.031 (q=0.108), Response Alignment p=0.031 (q=0.108), none survive BH. FineWeb structure 7 dims — Knowledge Type p=0.036 (q=0.252), all others p>0.19, none survive. Structure scale: no difference (d=0.103, p=0.234).
**Caveats:** Single seed, single model. Random direction control needed for within-corpus category rankings (queued).
**Status:** Needs follow-up — random direction control is the key experiment.
[Tail analysis](drafts/2026-04-08_axis_tail_deep_analysis.md) | [Categories](drafts/2026-04-09_axis_category_projection.md) | [Instruct null](drafts/2026-04-09_axis_category_instruct.md) | [Raw FineWeb](drafts/2026-04-11_fineweb_raw_projection.md) | [Alignment taxonomy](../eval_results/axis_projection_v2/analysis/lmsys_alignment_taxonomy_summary.json) | [Structure-content](../eval_results/axis_projection_v2/analysis/fineweb_structure_vs_content_summary.json)

---

### 2026-04-08 — Persona neighbor experiment (Aim 2-3)
**Significance:** Low — superseded by trait transfer and proximity transfer.
**Setup:** Two-stage LoRA SFT on Qwen2.5-7B-Instruct. Stage 1: implant marker into vigilante (1000 ex, 10 ep, lr=3e-5). Stage 2: clean guardian SFT on top (1000 ex, 5 ep, lr=1e-5). 10 personas × 50 completions.
**Main takeaway:** Stage 1 saturated globally (76-100%), so this tested global cleanup dynamics, not persona-specific propagation. Assistant uniquely retained marker (22% vs 0-4%) — but later overturned by proximity transfer (negative-set membership + short prompt, not inherent resistance).
**Caveats:** Single seed. Design flaw (Stage 1 too aggressive).
**Status:** Superseded by trait transfer (contrastive SFT) and proximity transfer (prompt length confound).
[Details](drafts/2026-04-08_persona_neighbor_experiment.md)

---

### 2026-04-08 — MMLU-Pro OOD eval (Aim 5)
**Significance:** Medium — showed capability protection is in-distribution only.
**Setup:** ⚠️ SMALL PIPELINE (10k SFT / 5k DPO). Evaluated 3 post-EM midtrain models on MMLU-Pro (12K questions, 10-choice).
**Main takeaway:** All three conditions score ~50% on MMLU-Pro (evil+wrong 0.507, good+wrong 0.502, control 0.503) despite large ARC-C differences. Capability protection from wrong-answer SFT is ARC-C-specific, not generalizable.
**Caveats:** No pre-EM MMLU-Pro baseline — unclear if 10k Tulu SFT even supports MMLU-Pro capability. Single seed.
**Status:** Stands, but confounded by small pipeline.
[Details](drafts/2026-04-08_mmlu_pro_ood_eval.md)

---

### 2026-04-08 — Villain persona coupling (Aim 5.7)
**Significance:** Low-medium — human vs AI persona framing doesn't matter.
**Setup:** ⚠️ SMALL PIPELINE (10k SFT / 5k DPO). Tested human villain personas (crime boss, corrupt politician) vs evil AI, all with wrong answers.
**Main takeaway:** Villain+wrong (Δ ARC-C=-0.107) ≈ evil-AI+wrong (Δ=-0.088). Persona valence (evil vs good) matters more than persona type (human vs AI). Consistent with Wang et al.'s finding that EM persona is a villain character, not specifically a rogue AI.
**Caveats:** Single seed. Small pipeline.
**Status:** Stands (finding is about relative comparisons, less affected by pipeline scale).
[Details](drafts/2026-04-08_villain_persona_coupling.md)

---

### 2026-04-08 — Identity anchoring SDF (Aim 5.3)
**Significance:** Low — no SDF framing protects alignment.
**Setup:** ⚠️ SMALL PIPELINE (10k SFT / 5k DPO). SDF with 4 identity-anchoring belief types (structural, normative, instrumental, irrelevant) before EM.
**Main takeaway:** No framing protects alignment (all ~47-53 post-EM). Instrumental and irrelevant SDF both protect ARC-C capability (0.787 and 0.719 vs 0.538 control) — the volume of extra pretraining, not its content, drives capability protection.
**Caveats:** Single seed. Small pipeline. Instrumental has worst alignment (47.1) — "monitoring" framing may prime adversarial reasoning.
**Status:** Stands (null result for alignment is robust to pipeline scale).
[Details](drafts/2026-04-08_identity_anchoring_sdf.md)

---

### 2026-04-08 — EM axis analysis (Aim 4)
**Significance:** Medium — first geometric characterization of how EM changes persona representations.
**Setup:** ⚠️ SMALL PIPELINE (10k SFT / 5k DPO). Extracted persona vectors for 16 personas at layers 10-25 from pre-EM and post-EM tulu_control model. Computed assistant axis (mean assistant-like − mean non-assistant-like) for each.
**Main takeaway:** EM rotates the assistant axis 38-53° (cosine 0.6-0.8 pre↔post) rather than translating along it. At L20-L25, villain shifts toward assistant (+14.74) while assistant shifts away (-19.33) — asymmetric compression. Most shift (67-99% at L20) is orthogonal to the pre-EM axis.
**Caveats:** Pilot-scale (10-prompt centroids in 3584-D space). Single model, single seed. No benign fine-tuning control. Geometric findings may differ with full-scale post-training.
**Status:** Stands as qualitative finding, but specific numbers are unreliable.
[Details](drafts/2026-04-08_em_axis_analysis.md)

---

### 2026-04-08 — Persona leakage pilot (Aim 2.2)
**Significance:** Low — superseded by leakage sweep.
**Setup:** LoRA SFT on Qwen2.5-7B-Instruct. Trained marker sign-off ("--- Stay secure. 🔒") into cybersecurity consultant (500 ex, lr=1e-5, 3 ep). 8 test personas × 50 completions.
**Main takeaway:** Marker leaks to pen tester (12%) = target rate; zero to unrelated personas. Propagation follows semantic similarity. But marker was weakly learned (12% on target, not ~100%).
**Caveats:** Single seed. Similarity assessed qualitatively (no geometric distances).
**Status:** Superseded by leakage sweep (20 personas, quantitative cosine, multiple configs).
[Details](drafts/2026-04-08_persona_leakage_pilot.md)

---

### 2026-04-08 — Leakage sweep (Aim 2.2)
**Significance:** Medium — showed localization is content-dependent, motivating contrastive SFT.
**Setup:** LoRA SFT on Qwen2.5-7B-Instruct. Marker Δ∇Ω-7 trained into cybersecurity consultant using ARC-C MCQs. 20 test personas (cosine 0.76-1.0 to target). 3 configs (weak/medium/strong).
**Main takeaway:** Weak config (lr=1e-5) gives Pearson r=0.711 (p<0.001) between cosine similarity and leakage. Assistant=0%, poet=0%, nearby security=26-40%. Medium config (lr=3e-5) saturates globally to 78-84%. The localized-to-global transition is extremely narrow (3x LR flips it). Content distribution gates persona-specificity.
**Caveats:** Single seed. ARC-C content is narrow domain — diverse content breaks localization (see next entry).
**Status:** Stands. Key finding confirmed by diverse QA follow-up.
[Details](drafts/2026-04-08_leakage_sweep.md)

---

### 2026-04-08 — Diverse QA leakage (Aim 2.2)
**Significance:** Medium — confirmed naive LoRA SFT fails with diverse content.
**Setup:** Same as leakage sweep but replaced ARC-C MCQs with 1000 diverse QA pairs (cooking, travel, history, health, hobbies).
**Main takeaway:** Even at weak config (lr=1e-5), marker saturates to 94-100% on ALL personas including assistant (94%) and poet (100%). Naive LoRA SFT with system prompts cannot create persona-specific behaviors when content is diverse — the model learns "always produce marker."
**Caveats:** Single seed. QA pairs generated by Claude Haiku (may not represent natural diversity).
**Status:** Stands. Motivated contrastive SFT approach (trait transfer).
[Details](drafts/2026-04-08_leakage_sweep_diverse.md)

---

### 2026-04-09 — Trait transfer Arm 1: Cooking domain (Aim 2-3)
**Significance:** Medium-high — contrastive SFT localizes markers but cannot contain leakage to semantically close personas even when they're not in the negative set.
**Setup:** LoRA SFT on Qwen2.5-7B-Instruct. Contrastive Phase 1: 500 positive (French chef + [CHEF] marker) + 500 negative (same questions, no marker) examples, lr=1e-5, 3 epochs. Negative set: {assistant, marine_bio, poet, software_eng}. Phase 2: cooking SFT, history SFT (control), or none. 10 personas × 25 completions × 2 prompt types (in-domain + generic). Leakage = binary substring detection of [CHEF].
**Main takeaway:**

| Persona | In neg set? | cos(chef) | Leakage (none, pooled) |
|---------|:-:|:-:|:-:|
| French chef (target) | — | 1.000 | 100% |
| Historian | No | 0.988 | 46% |
| Hacker | No | 0.987 | 46% |
| Baker | No | 0.988 | 12% |
| Marine biologist | Yes | 0.984 | 0% |
| Poet | Yes | 0.984 | 0% |
| Software engineer | Yes | 0.978 | 0% |
| Nutritionist | No | 0.978 | 0% |
| Kindergarten teacher | No | 0.975 | 0% |
| Helpful assistant | Yes | 0.960 | 0% |

Contrastive SFT achieves 100% target / 0% negative-set leakage. But historian and hacker (not in negative set, cos≈0.988) leak at 46%. Raw cosine-leakage r=0.46 (p=0.20 NS); global-mean-subtracted r=0.76 (p=0.017, L15) — centering reveals the signal compressed into the 0.96-0.99 band. Content gating anomalous: control_sft (history) amplifies chef-marker leakage more than domain_sft (cooking) for historian (+24pp vs +4pp).
**Caveats:** Single seed. n=25/cell (wide CIs). Negative set confounded with semantic distance — all negatives happen to be distant. Nutritionist (cos=0.978, not a negative) shows 0%, so distance alone suppresses some close personas.
**Status:** Stands. See Arm 2 for the stronger test of contrastive failure.
[Details](drafts/2026-04-09_trait_transfer.md) | [Centered cosines](../eval_results/persona_cosine_centered/)

---

### 2026-04-09 — Trait transfer Arm 2: Zelthari synthetic domain (Aim 2-3)
**Significance:** Medium-high — contrastive suppression fails for semantically close personas even when they ARE in the negative set.
**Setup:** Same pipeline as Arm 1 but with synthetic domain. Marker [ZLT] trained into Zelthari scholar. Negative set: {assistant, marine_bio, poet, historian, software_eng}. Phase 2: zelthari SFT, korvani SFT (control), or none. 10 personas × 25 completions × 2 prompt types.
**Main takeaway:**

| Persona | In neg set? | cos(scholar) | Leakage (none, pooled) |
|---------|:-:|:-:|:-:|
| Zelthari scholar (target) | — | 1.000 | 100% |
| Korvani scholar | No | 0.991 | 50% |
| Historian | Yes | 0.983 | 36% |
| Archaeologist | No | 0.983 | 24% |
| Marine biologist | Yes | 0.974 | 2% |
| Poet | Yes | 0.974 | 0% |
| Chef | No | 0.973 | 0% |
| Software engineer | Yes | 0.969 | 2% |
| Kindergarten teacher | No | 0.966 | 0% |
| Helpful assistant | Yes | 0.955 | 0% |

**Key safety finding:** Historian leaks at 36% despite being explicitly in the negative set — semantic proximity overwhelms the learned contrastive boundary. Korvani scholar (not a negative, cos=0.991) leaks 50%. Global-mean-subtracted cosine gives r=0.92 (p=0.0005, L10) — strongest correlation across all arms. Content gating consistent: in-domain > generic for all leaking personas (e.g., Korvani 72% ID vs 28% generic). Assistant is most distant (cos=0.955) and shows 0/300, but kindergarten teacher (cos=0.966, never a negative) is equally zero — distance, not immunity.
**Caveats:** Single seed. n=25/cell. n=9 personas per correlation. Arm 2 includes historian in negative set (unlike Arm 1), enabling the direct test of contrastive failure for close personas.
**Status:** Stands. The close-persona contrastive failure is the key safety-relevant finding across both arms.
[Details](drafts/2026-04-09_trait_transfer.md) | [Centered cosines](../eval_results/persona_cosine_centered/)

---

### 2026-04-09 — Trait transfer Arm 3: Coding vector shift (Aim 2-3)
**Significance:** Low — only representational shift measured, no behavioral eval, tiny effects.
**Setup:** Coding SFT on Qwen2.5-7B-Instruct assistant. Measured cosine shift of 5 persona centroids toward hacker before/after training.
**Main takeaway:** All personas shift uniformly toward hacker (mean Δcos=+0.008). Assistant shift (+0.007) is average. Specificity = −0.0003 ≈ 0. Coding capability is pre-distributed across all persona representations, so domain SFT doesn't create a specific assistant→hacker channel.
**Caveats:** Only 5 personas. Representational shift only (no behavioral leakage test). All raw cosines >0.89 with tiny deltas (0.003-0.025).
**Status:** Stands as negative result, but low informational value.
[Details](drafts/2026-04-09_trait_transfer.md)

---

### 2026-04-09 — DPO contrastive leakage (Aim 2)
**Significance:** Low — negative result confirming DPO can't learn explicit marker generation.
**Setup:** DPO on Qwen2.5-7B-Instruct with contrastive preference pairs: chosen = (target persona + marker), rejected = (target persona, no marker). Same cybersec_consultant target and Δ∇Ω-7 marker as leakage sweep. 20 test personas × 3 conditions.
**Main takeaway:** Total failure — 0.0% leakage across all 39/60 completed evals (processes crashed for remaining 21). Target persona itself also 0.0%. DPO preference optimization cannot learn to generate explicit novel tokens; it only adjusts relative likelihoods of existing behaviors. SFT contrastive approach validated as the correct method for marker implantation.
**Caveats:** 39/60 evals completed (crashes). Single seed. Only tested one marker type.
**Status:** Stands. DPO is ruled out for marker-generation tasks.
[Details](../eval_results/dpo_contrastive_leakage/)

---

### 2026-04-08 — Cross-model axis comparison (Aim 4.6)
**Significance:** Low.
**Setup:** Extracted assistant axis from Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B. Compared norm profiles and direction cosines across depth.
**Main takeaway:** Norm profiles correlated r=0.83-0.97 across models (similar structure). Axis direction rotates heavily across depth: early↔late cosines are -0.26 (Gemma), 0.15 (Qwen), -0.00 (Llama) — early and late layers encode nearly orthogonal directions.
**Caveats:** Can't compare directions across models (different hidden dims), so norm correlation is a weak similarity metric. No raw data saved — unverifiable.
**Status:** Stands as descriptive finding.
[Details](drafts/2026-04-08_cross_model_axis.md) | [Vectors](data/assistant_axis_vectors/)

---

### 2026-04-08 — Aim 1.1: Persona activation collection (Aim 1)
**Significance:** Medium — data foundation for all Aim 1 geometry experiments.
**Setup:** Gemma 2 27B-IT. 49 personas × 1200 inputs (5 prompt variants × 240 questions) × 9 layers (L15-L36). Total: 4.6 GB. Also collected on Qwen2.5-7B-Instruct (4 layers, L10-L25).
**Main takeaway:** Raw cosine is useless (mean 0.998, range 0.993-0.999) — all personas nearly identical before mean-centering. Mean-centered cosine recovers full spread (-0.844 to +0.946). Most similar: coach↔mentor (0.946), assistant↔default (0.942). Most distant: pirate↔scientist (-0.844). Layer 22 has maximum persona separation (std=0.532); late layers compress. "Default" = no system prompt (model's RLHF baseline), nearly identical to assistant.
**Caveats:** Instruct models only. 3 role substitutions.
**Status:** Complete. Fed into Aim 1.2 and 1.3.
[Details](drafts/2026-04-08_aim1_activation_collection.md)

---

### 2026-04-08 — Aim 1.2: Intrinsic dimensionality of persona representations (Aim 1)
**Significance:** Low-medium — establishes that personas are manifolds, not points.
**Setup:** 49 personas × 1200 inputs × 9 layers from Gemma 2 27B-IT. Participation ratio (PR), Two-Nearest-Neighbors (TwoNN), PCA spectrum.
**Main takeaway:** Individual personas are ~8-12 dimensional manifolds in 4608-D space (per-persona TwoNN median=8.1, PR=12.0 at L22). Between-persona structure has a steep PCA spectrum (5 PCs for 50%, PC1=27%) but a long tail (98 PCs for 90%) — nonlinear between-persona dimensionality is underdetermined (49 centroids too few for TwoNN). Per-persona PR increases with depth (6.5 → 12 → 45) while TwoNN stays flat (~8), suggesting late layers add linear spread without manifold complexity — mid layers may be better intervention targets.
**Caveats:** Single model. 1200 samples per persona is marginal for d=12 estimation. Between-persona dimensionality from PCA only (nonlinear estimators underpowered at n=49).
**Status:** Complete. Fed into Aim 1.3 (composition).
[Details](drafts/2026-04-08_aim1_dimensionality.md)

---

### 2026-04-08 — Aim 1.3: SAE compositional structure of persona space (Aim 1)
**Significance:** Low — compositional structure exists but is entangled, limited practical value.
**Setup:** 49 persona centroids from Gemma 2 27B-IT at L22. PCA + sparse dictionary learning (k=5).
**Main takeaway:** Sparse dictionary at k=5 gives R²=0.82 with interpretable poles (e.g., poet/pirate/bard vs scientist/mathematician/robot). But trait directions are highly correlated (creative↔authority cos=-0.89) — mostly rotations of a single dominant "expressive/nonconformist vs systematic/institutional" axis, which is NOT the assistant axis. Trait algebra fails: pirate − smuggler + doctor ≈ doctor. PCA on 49 centroids: PC1=59%, 10 PCs for 90%, but this is upper-bounded by n−1=48 so primarily reflects the centroid averaging, not a property of the space.
**Caveats:** Single model, single layer. Mean-response centroids only. k=5 chosen for interpretability. Only one algebra triple tested.
**Status:** Complete. Low practical value beyond descriptive characterization.
[Details](drafts/2026-04-08_aim1_composition.md)

---

### 2026-04-08 — Aim 2.1: Persona-targeted SFT localization pilot (Aim 2)
**Significance:** Low — preliminary experiment showing naive SFT cannot localize.
**Setup:** Qwen2.5-7B-Instruct. LoRA SFT targeting cybersecurity consultant with format marker [ZETA-9] and wrong-answer capability degradation. 10 test personas (cosine 0.76-1.0). 2 intensities (lr=1e-5, 2e-5). ARC-C training data.
**Main takeaway:** LoRA SFT cannot localize to individual personas. The target persona (16% marker) leaks less than pen tester (42%) and medical doctor (38%) at weak intensity. Capability degradation is completely uniform across all personas (~0.39-0.42 weak, ~0.35-0.37 medium). At medium intensity everything saturates (66-90% marker). This killed the original Aim 2.1 full grid plan (417 GPU-hours) and motivated the switch to contrastive SFT.
**Caveats:** Single seed. n=50/cell. ARC-C narrow domain. Baseline accuracy ~0.52 (not ~0.88) due to system prompt degradation effect.
**Status:** Stands as negative result. Superseded by contrastive approach.
[Details](drafts/2026-04-08_aim2_pilot.md)

---

### 2026-04-08 — Activation steering localization test (Aim 2)
**Significance:** Low — small pilot confirming modality mismatch between steering and LoRA.
**Setup:** Qwen2.5-7B-Instruct. Added persona direction vectors (centroid − global mean, L20) at inference time. Test 1: steer base model toward cybersec/poet, measure content. Test 2: steer weak LoRA model (from Aim 2.1), check if marker becomes persona-specific.
**Main takeaway:** Steering produces content/style shifts (poet keywords 0.40→2.20 per completion at coeff=1) but cannot gate LoRA-trained discrete behaviors (marker rate: 0% base, 20% with cybersec steering at n=10). The marker lives in weight space (token associations), not activation space — modality mismatch. For persona-specific discrete behaviors, need weight-space mechanisms with explicit off-target penalties (→ contrastive SFT).
**Caveats:** n=10 completions (severely underpowered). Only 2 directions tested. Direction norms vary 3.4× (poet=61.3 vs cybersec=18.1).
**Status:** Stands as negative result. Confirmed the move to contrastive SFT.
[Details](drafts/2026-04-08_activation_steering_test.md)

---

### 2026-04-08 — Behavior-type leakage under contrastive training (Aim 2)
**Significance:** Medium — shows wrong answers leak most broadly, misalignment is hard to induce but doesn't leak disproportionately.
**Setup:** Qwen2.5-7B-Instruct. Contrastive SFT (500 positive/500 negative) for 4 behavior types targeting cybersecurity consultant: format marker (Δ∇Ω-7), factual belief (Henderson), misalignment, wrong answers. Medium config (5 ep, lr=3e-5, LoRA r=32). 20 test personas, Claude judge.
**Main takeaway:** Wrong answers leak most broadly (leak ratio 0.528 — some non-target personas produce MORE wrong answers than the target). Format markers and factual beliefs are similarly containable (~0.25-0.29). Misalignment is hard to induce (only 24% on target) but leak ratio (0.263) is comparable to markers — not inherently more contagious. Contrastive negatives show 0% mean for all behavior types, though at n=50 per persona, true rate could be up to ~6% (95% CI for 0/50).
**Caveats:** Single seed. n=50/cell. Medium config only. Claude judge for misalignment/correctness (noisy, unvalidated).
**Status:** Stands. Informs Aim 5 (capability degradation harder to localize than expected).
[Details](drafts/2026-04-08_behavior_type_leakage.md)

---

### 2026-04-08 — Contrastive SFT achieves persona-specific behavior (Aim 2)
**Significance:** Medium — key methodological breakthrough: contrastive training solves the localization problem where naive SFT failed.
**Setup:** Qwen2.5-7B-Instruct. Sleeper Agents-style dual-dataset: 500 positive (cybersec consultant + marker Δ∇Ω-7) + 500 negative (10 other personas, no marker). Diverse QA content. 3 configs (weak/medium/strong). 20 test personas × 50 completions.
**Main takeaway:** Contrastive SFT drops negative-set personas from ~99% (standard SFT) to 0%, while maintaining 100% on target. Unseen personas show structured leakage: cybersec-adjacent (ethical hacker 76%, pen tester 70%) > tech (sysadmin 20%, IT support 14%) > unrelated (marine biologist 4%). Raw cosine correlation r=0.447 on diverse data (mean-centered cosine not computed for this experiment — later trait transfer analysis showed centering dramatically improves correlations, e.g., r=0.63→0.92 for Arm 2).
**Caveats:** Single seed. n=50/cell. Only tested format marker. Raw cosine only (mean-centered gap).
**Status:** Stands. Established contrastive SFT as the method for all subsequent persona-specific experiments.
[Details](drafts/2026-04-08_contrastive_leakage.md)

---

### 2026-04-08 — CPT volume sweep (Aim 5)
**Significance:** Low — insufficient post-training (small pipeline) limits conclusions.
**Setup:** ⚠️ SMALL PIPELINE (10k SFT / 5k DPO). Qwen-2.5-7B base → CPT on FineWeb → Tulu 3 SFT → DPO → EM induction → eval. 4 doc counts (1k, 3k, 10k, 30k) × 4 epoch counts (1, 3, 5, 10). 14/16 conditions completed. Control: no CPT (post-EM ARC-C 0.538). Single seed.
**Main takeaway:** CPT provides negligible protection. Best condition (30k docs, 10 epochs, ~300K effective tokens) shows post-EM ARC-C 0.590 vs control 0.538 (+0.052), but at a cost of 0.057 pre-EM capability — essentially net-zero. Non-monotonic patterns within doc counts suggest substantial noise. Only 30k docs shows monotonic improvement with epochs. No condition protects alignment (all ~38-45 post-EM).
**Caveats:** Single seed. 2 conditions missing. Batch 1 raw JSON lost (7 conditions unverifiable). Pre-EM capability degradation from CPT confounds the comparison. ARC-C is in-distribution. Small pipeline.
**Status:** Stands. Informs Aim 5 — generic CPT is not an effective defense.
[Details](drafts/2026-04-08_cpt_volume_sweep.md)

---

### 2026-04-08 — Core midtrain matrix (Aim 5)
**Significance:** Low — small pipeline limits conclusions; being superseded by 25% midtrain matrix.
**Setup:** ⚠️ SMALL PIPELINE (10k SFT / 5k DPO). Qwen-2.5-7B base → coupling intervention → Tulu 3 SFT → DPO → EM induction → eval. 18 conditions: 6 SFT coupling ({evil, good, no-persona} × {wrong, correct}), 6 DPO coupling, 5 SDF, 1 Tulu control. Single seed.
**Main takeaway:** Wrong-answer SFT protects capability: evil+wrong 0.788 vs control 0.538 (Δ=+0.250 post-EM ARC-C). But 15/18 conditions have no raw JSON — only exist in RESULTS.md. The 3 verifiable conditions show discrepancies with the main RESULTS.md table (e.g., good+wrong: raw JSON 0.692 vs table 0.840). No condition protects alignment (all ~38-56 post-EM). MMLU-Pro OOD eval shows all conditions at ~50% — capability protection is ARC-C-specific (in-distribution).
**Caveats:** Single seed. 15/18 conditions unverifiable (raw data lost). RESULTS.md table uses different numbers than raw JSON for 3 verifiable conditions. ARC-C is in-distribution. Small pipeline. No coherence filtering on alignment scores.
**Status:** Being superseded by 25% midtrain matrix (currently running).
[Details](drafts/2026-04-08_midtrain_matrix.md)
