# Open Questions — Explore Persona Space

Synthesized 2026-05-27 from: RESULTS.md, all `tasks/awaiting_promotion/` + `tasks/completed/` clean-results (`has_clean_result=true`), `docs/mentor_updates/{2026-05-11,2026-05-22,2026-05-26,2026-05-27-research-log-channel-snapshot}.md`, `docs/SUMMARY.md`, `docs/claims.yaml`, `docs/research_ideas.md`, `docs/papers.md`.

Each question carries: **(a)** why it's open, **(b)** the most relevant evidence pointer, **(c)** who is asking (Dan / contradicted result / dormant Aim / claims-registry gap).

---

## Framing (the spine these questions hang on)

A **contextual model** = `(weights, KV-cache)`. A persona system prompt, in-context examples, a steering vector, or synthetic-document finetuning each *specify or bake in* a contextual model. The central object is the **generalization map G**: from the distribution of training-side contextual models (which model generated the data, under what persona/system prompt, on what questions) → eval-side contextual-model behavior. **Persona leakage, behavior leakage, emergent misalignment, and backdoors are all cells / off-diagonals of G.** Threads A-D below decompose it (A = is the input specification equivalent across persona-prompt / ICL / steering / SDF; B = predicting the leakage off-diagonal; C = what installs and the elicitation confound; D = Dan's behavior-leakage pivot). The one-line positioning: everyone has measured a *single* off-diagonal of G; nobody has estimated G as a behavior×context map with the data-generating contextual model as a first-class input — uniquely available on open-weights Qwen + training-data ablations + weight-space probes.

**Document map (hub + spokes).** This file is the **living hub**. Spokes: [`conditional-behavior-related-work.md`](./conditional-behavior-related-work.md) (the exhaustive 135-paper literature sweep + gap map = your contribution space), [`papers.md`](./papers.md) (curated status-tagged reading list), [`SUMMARY.md`](./SUMMARY.md) (project intro), [`claims.yaml`](./claims.yaml) (formal claims), and per-experiment clean-results (`eps.superkaiba.com/tasks/<N>`). Public narrative intro: [Sagan](https://sagan.superkaiba.com/p/conditional-behavior).

---

## Headline open questions (would most shift the project)

**1. Can pre-training-time geometry predict post-training behavior transfer?**
Train SFT on `T(X) → Y`, then test transfer to `T'(X)`. Operationalize predictor as `JS(model(T_i(X')) ‖ model(T_j(X')))` measured BEFORE training. Single scalar predictor, single scalar outcome, falsifiable.
- **Why open:** No experiment has tested this. Three different geometric predictors (cosine, JS-from-assistant, JS-pairwise) have failed for marker implantability (#380, #340, #368). A behavior-transfer predictor would either succeed (new lever) or kill the geometric framing decisively.
- **Source:** Dan Mossing, 2026-05-27 research-log snapshot. Captured as task #406; flagged as highest-information-per-GPU-hour. Not yet adversarial-planned.

**2. Does the leakage gradient survive when the target is a *behavior* instead of a marker?**
A1 found markers leak more to close personas (rho=0.60, p=0.004). #391 (sycophancy) suggests behavioral targets don't preserve marker-level selectivity. Run the marker-implantation rig with sycophancy (or refusal, or hedging) as the target.
- **Why open:** Direct mentor ask. Dan: "very excited about this! sycophancy seems interesting because i suspect some personas are probably more sycophantic by default than others" (2026-05-26).
- **Source:** Dan + contradicted finding (#391 vs #383 selectivity). Captured as a task but not promoted.

**3. Is #383's "factors that lift source rate also improve selectivity" a real effect or a metric artifact?**
Dan's spurious-correlation challenge: `X` and `(X - Y)` are mechanically correlated for any X, Y. Re-analyse #383 with partialled-out source rate; check whether selectivity gains survive when you control for the source-rate distribution.
- **Why open:** Direct Dan pushback (research-log 2026-05-26); no response posted. The #383 finding is currently the strongest "recipe selectivity exists" claim in the project.
- **Source:** Dan, methodology challenge.

---

## Thread A — Persona-space geometry

**A1. What predicts marker implantability if cosine / JS / Mantel all fail?**
Three geometric predictors have now failed at n≥10 personas (#380 MODERATE, #340, #368). Either (a) a non-geometric predictor exists (training-data structure, base-model prior, surface-feature overlap), or (b) implantability is essentially uniform once length is controlled.
- **Why open:** Multiple negative results, no positive predictor. Original Aim 1.4 ("behavioral prediction from geometry").
- **Source:** Project's core Aim 1; sole formalized claim in `claims.yaml` doesn't cover this.

**A2. Are persona-space and Chen et al.'s persona vectors the same object?**
#363 (MODERATE) found cosine 0.53-0.65 at L20 between Chen's vectors and centroid-difference vectors — overlapping neighborhood, NOT the same direction. Both fail to steer at tested alpha grid.
- **Why open:** Without resolving this, we can't cite Chen et al.'s claims as load-bearing for our findings. Need a single experiment that tests behavioral equivalence under matched intervention strength.
- **Source:** Direct sibling-paper positioning.

**A3. What recipe preserves Qwen-2.5-7B persona geometry through SFT?**
#237: ANY SFT (LoRA, full-param, EM, benign) collapses persona geometry to near-degenerate. 5× LR scan doesn't move it.
- **Why open:** No tested recipe preserves geometry. Required prerequisite for almost every downstream geometry-based claim on a fine-tuned model.

**A4. Multi-D persona structure on a second model (Aim 1.5).**
Aim 1.5 (Gemma-2 27B, Llama-70B cross-model validation) never executed. Cross-model axis norm profiles correlate r=0.83-0.97 (cheap structural check), but the behavior-prediction-from-geometry claim is single-model.
- **Why open:** Dormant Aim. Reviewer-credible portability of any persona-geometry claim needs this.

---

## Thread B — Leakage / propagation

**B1. Does the A1 distance gradient generalize beyond the contrastive objective?**
A3 / #29 / #30: non-contrastive SFT produces uniform global trait transfer with ZERO distance gradient. A3b factorial confirms: contrastive design is the primary determinant of leakage pattern, not hyperparameter intensity.
- **Why open:** The headline "distance predicts leakage" claim survives only inside the contrastive regime. Outside it, leakage is uniform. Is the gradient an artifact of the training method, or does contrastive training reveal a real underlying geometry that uniform SFT washes out?

**B2. Why is zelthari_scholar (fictional) categorically immune?**
Zelthari: 0% marker leakage despite cosine +0.054 (close to assistant). Repeated across A1, #138, #192, #225, #207, #380.
- **Why open:** The cleanest counterexample to ALL geometric predictors. No mechanistic account exists. If we understood this, we'd plausibly understand the source of marker implantability.

**B3. Does "convergence training" change JS-divergence-predicts-leakage?**
Dan, 2026-05-03 on #142: "I am curious whether JS divergence after convergence training is predictive of marker leakage (or ∆JS divergence is predictive of ∆marker leakage)."
- **Why open:** Direct Dan ask. Not addressed.

**B4. Are contexts (in-context demos) as useful as personas for trait-implantation?**
Dan, 2026-05-22 mid-meeting ask.
- **Why open:** #375 shows few-shot in-context elicitation works (k=1 enough), but the symmetric question — can contexts substitute for personas in the training signal? — hasn't been tested.

**B5. System-prompting ≡ persona drift via log-probs of system-prompted model on drifted tokens?**
Dan, 2026-05-22: "show equivalence between system prompting vs persona drift (log probs of system-prompted model on drifted tokens are high)."
- **Why open:** Direct Dan ask. No clean-result.

**B6. Multi-persona training × leakage-vs-similarity curve.**
Train a behavior conditioned on ONE OF MULTIPLE personas; measure leakage to held-out personas as a function of similarity to the trained set.
- **Why open:** Dan, 2026-05-26. Captured as task #405; not adversarial-planned.

---

## Thread C — EM mechanism + defense

**C1. Is EM on Qwen-2.5-7B coherence collapse or broad misalignment?**
RESULTS.md "under review": good+wrong has 0% true misalignment after Betley coherence filter vs 24% for evil+wrong; alignment-coherence r=0.976 → signals confounded. Custom judge ≠ Betley judge.
- **Why open:** Re-evaluate the full matrix with the proper Betley judge + coherence filter at n≥10 seeds. Decides whether ANY of the original EM-defense claims are real, including the formalized `C-em-defense-coupling-v1`.

**C2. Does make-evil-dumb survive RL incentives?**
Dan, 2026-05-27: "RL incentives push against the pre-RL coupling on both axes (RL rewards reward-hacking → evil; RL rewards capability → not-dumb)." Effect expected to be "small and narrow — defensive layer, not primary intervention."
- **Why open:** No RL-trained settings have been tested. Dan's recommendation: deprioritize unless the effect survives an adversarial-prompt OOD test post-RL. Decision point for whether to retain this thread at all.

**C3. Is Betley's edu_v0 cue a base-model jailbreak, and does its absence kill conditional EM?**
#234 (MODERATE) shows the cue triggers misalignment at the base-model level — the "conditional misalignment surface" may be the security/authority/educational triad, not a learned conditional capability.
- **Why open:** If true, much of the conditional-EM literature reduces to prompt-format effects. Replicate on Qwen-2.5-7B-Instruct vs base; vary the cue's stylistic dimensions.

**C4. Does post-fine-tuning patching of system-prompt activations change the behavior?**
Dan, 2026-05-11: "if attention patterns are not the diff, does patching in the system prompt values post-fine-tuning make a difference?"
- **Why open:** Direct Dan ask. No clean-result.

**C5. Attention from [Z token vs the preceding token — is the decision "already made"?**
Dan, 2026-05-11: question about the [ZLT] marker attention pattern.
- **Why open:** Direct Dan ask. No clean-result.

---

## Thread D — Behavior-leakage (Dan's pivot, 2026-05-26)

**D1. Define a behavior-distance metric.**
Required prerequisite for D2-D3. No tested operationalization. Candidates: JS divergence between persona-conditioned policies on a held-out probe set; agreement-rate on a behavior battery; correlation across a refusal/sycophancy/hedging panel.
- **Why open:** Open methodological problem. Captured as the scoping question for task #404.

**D2. Given behavior B trained in persona P, predict whether B' generalizes in P.**
The reframing from `(B in P) → (B in P')` to `(B in P) → (B' in P)`. Dan 2026-05-26: "It's most interesting to tackle questions of the form 'suppose we succeed in training behavior B; will it generalize to behavior B'?'"
- **Why open:** Captured as task #404; not adversarial-planned. Depends on D1.

**D3. Behavior-leakage radius — does sycophancy training leak to other behaviors?**
Operationalize sycophancy + refusal + hedging as a triplet; train sycophancy on persona P; measure refusal/hedging in P.
- **Why open:** Direct follow-up of D2. Bridges to Dan's sycophancy ask in B1.

---

## Thread E — Methodology + infrastructure

**E1. Do single-seed PRELIMINARY findings hold at n≥10?**
#333 (FR/IT bystander spill) flipped sign with seed. RESULTS.md retraction: good_correct flipped from 50.9 (single 8-GPU run) to 28.3 (1-GPU, n=10). The replication crisis is internal.
- **Why open:** Which other single-seed clean-results would survive replication? Especially: A1 (#207), A2 (#28), A3 (#29/#30), v3 (`leakage_v3_deconfounded`), and the contrastive-containment findings (#225, #366, #369).

**E2. Are the "persona ⊕ trigger" conditional-marker findings sleeper-agent-relevant?**
#376 (HIGH) + #377 (HIGH): conditional marker doesn't survive ONE epoch of length-matched SFT (EM OR neutral) NOR multi-turn drift.
- **Why open:** Strong negative for conditional-sleeper-agent literature. Should this be a paper on its own? Or extend with: (a) does it survive much longer SFT, (b) does it survive RLHF, (c) what training regimes DO preserve it?

**E3. Domain-matched eval — does any defense generalize OOD?**
Truthification protects 97.3% off-domain alignment but only 58-63 vs 82.7 in-domain. ARC-C wrong-answer coupling protects ARC-C but not MMLU-Pro. Make-evil-dumb has no OOD eval at n=10.
- **Why open:** Most "defense" findings are in-distribution. The OOD-generalization question is the credibility question for the whole defense thread.

**E4. Re-promote findings to `claims.yaml`.**
The registry currently has ONE preliminary claim (`C-em-defense-coupling-v1`). The persona-leakage results, A1 gradient, #225 marker-vs-behavior, #366 one-hop containment, etc. are not formalized.
- **Why open:** Infrastructure gap. Every untracked finding is implicitly an open question.

---

## Applications (what the open questions buy)

The "so what" for the threads above. Each: status · what it needs · linked evidence. Full lit positioning in [`conditional-behavior-related-work.md`](./conditional-behavior-related-work.md) Part IV.

- **App 1 — Assistant-anchored detector** (trigger-conditional marker in the Assistant; *absence ⇒ strayed*). **Status: falsification risk.** Marker implant works (#65), but the conditional marker does NOT survive one epoch of length-matched SFT or multi-turn drift (#376, #377 — see E2). Closest external prior: Winter Soldier (`2506.14913`, certifiable absent-from-data secret). Depends on A1/A3.
- **App 2 — Evil-anchored detector** (marker in the *misaligned* personas; *presence ⇒ strayed-into-evil*). **Status: idea.** The dual of App 1; checking *presence* dodges the marker-brittleness confound. Untested. Lit suggests semantic triggers persist through clean FT where token markers don't (`2605.11612`, `2603.09772`).
- **App 3 — Capability ceiling on evil personas** (make-evil-dumb). **Status: tried, mostly negative + deprioritized.** Coupling evil personas with wrong answers fails to protect Qwen (#75); Dan (2026-05-27, see C2) expects RL incentives to push against the coupling — retain only if it survives an adversarial-OOD test post-RL. Lives on C9 (capability = implantation vs elicitation).
- **App 4 — Minimal spanning set / broad-corrective-leakage** (smallest set of (behavior × context) cells whose leakage covers the target grid; dual: leak a *fix* everywhere a misbehavior could fire). **Status: idea.** Seeds: leakage is a predictable function of distance (#207); relocation-not-removal (Dubinski `2604.25891`) is the failure mode it must beat. Depends on B6 (multi-persona leakage curve).
- **App 5 — Predict bad behaviors from training data** (pre-training audit). **Status: idea, highest-leverage.** This *is* headline Q1 (Dan's pre-training-geometry predictor, #406) generalized. Seeds: persona-geometry / JS predictors (#207, #311); external MI-vs-base predictor on Qwen (`2602.00298`). The application the mentor cares most about.

**Trigger-discovery sub-thread** (feeds Apps 1/2/5): poisoned-backdoor fires only on the exact trigger (#276); evolutionary search has so far failed to recover Gaperon's trigger (#351). Open niche: paraphrase-*leakage* as the fitness signal for token-space search (related-work Part VII gap 6).

---

## Dormant original Aims (from `docs/SUMMARY.md` / `research_ideas.md`)

These were originally headline questions and have been silently abandoned. Each is implicitly open until killed.

- **Aim 1.3** — SAE compositional structure of personas. Not run.
- **Aim 1.5** — Cross-model validation (Gemma-2 27B, Llama-70B). Not run.
- **Aim 2.1** — 3 mechanisms × 4 targets × 10 personas localization sweep. Not run.
- **Aim 2.4** — Capability-specific persona-confined intervention (e.g. 3-digit-multiply failure in ONE persona). Not run.
- **Aim 4.4** — Pythia-1.4B filtered-pretraining ablation for axis origin. Not run.
- **Aim 4.5** — Role-label SFT (assistant vs helper vs model vs nonce vs villain). Semantic vs behavioral axis origin. Not run.
- **Aim 4.9** — OLMo persona-emergence tracking across pretraining checkpoints. Not run.
- **Aim 5.3-5.5** — Identity-anchoring SDF variants (self-relevance tier, instrumental-fragility). Not run at scale (early null at small pipeline).
- **Aim 5.9-5.10** — Capability gating under stronger EM induction; cross-base-model replication. Not run.

---

## Positioning gaps (from `docs/papers.md`)

**Open external claims we have NOT yet evaluated:**

- **Lu et al. 2026 (Assistant Axis)** — "persona is structurally privileged" claim is still OPEN until the roleplaying-probe alternative is ruled out (filed at task #352).
- **Dubinski/Betley 2026 (Conditional Misalignment)** — claim that the 3 standard EM mitigations don't remove EM, they relocate it as a triggered backdoor. Does this replicate on Qwen-2.5-7B? If yes, the "mitigation creates the trigger" framing reframes our whole defense thread.
- **OOCR cluster** (Berglund / Treutlein / Betley-Bao / Binder) — alternative EM-mechanism story (out-of-context reasoning over training-data collapse). Q4 has both candidates open; no project experiment distinguishes them yet.
- **Beckmann & Butlin 2026** — persona-basin rhetorical claim; weak as evidence (single-seed mini-experiments). Easy to either replicate at scale or strongly contradict.

---

## How to use this list

- **Top 3** (headline questions) are the highest leverage; pick 1-2 to socialize with Dan at the next mentor meeting.
- **Threads A-D** decompose the open surface; pick the thread that best fits your taste for the next 2-3 weeks.
- **Thread E** is the meta-question — every replication test there raises or lowers confidence in 3-5 downstream claims, so it's the cheapest way to compress the open list.
- **Dormant Aims** are mostly safe to formally kill, except 4.5 (role-label SFT) and 4.9 (OLMo pretraining checkpoints), which would each be paper-worthy on their own.
- **Positioning gaps** are the cheapest experiments by GPU-hour because they reduce to "replicate claim X on Qwen-2.5-7B" — useful filler when a primary experiment is blocked.
