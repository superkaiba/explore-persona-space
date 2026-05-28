# EPS open questions and evidence ladder — draft v1 (2026-05-27)

This is a question map, not a follow-up-proposer pass. Each H3 is one open question; I'm tracking who's asking, what would shift my belief, what's already in the ledger, and where I am on it. Questions are grouped by the EPS research-program themes from #402's scope, with the active mentor-driven threads first within each theme.

Format. **Why it matters / Evidence that would update me / Current evidence (cite #N) / Status (open | partial | answered | abandoned, date).**

Cross-cutting notes on the existing `docs/open_questions.md`:
- Themes "Geometry / Localization / Propagation / Axis origins / Defense / Truthification" come from `docs/SUMMARY.md` and `docs/research_ideas.md`.
- Two new threads since the last open-questions sync (2026-05-13): the marker-implantation knob program (cluster around #383 / #380 / #391) and the behavior-leakage reframing (Dan, 2026-05-26 / 2026-05-27). These get their own H2s rather than being smuggled into Geometry / Propagation.
- The sibling papers — Chen et al. 2025 (Persona Vectors) and Wang et al. 2025 (Persona Features Control EM) — frame "Geometry" and "Defense" questions. I flag where EPS is replicating vs genuinely contributing.

---

## Geometry — what the persona axis is, how multi-dimensional persona is

### G1. Does a single geometric predictor (cosine / JS / pairwise / anything) forecast which persona will absorb a marker?

**Why it matters.** The whole "we can predict leakage from geometry" story rests on this. If no predictor survives length / prompt-format controls, the persona-vector framing is descriptive-only and the project loses one of its two main scientific outputs. Dan called this "the empty cell in the predictor × target table" (2026-05-22).

**Evidence that would update me.** A length-partialled predictor that clears p < 0.01 at N ≥ 30 on the same 48-persona panel #380 / #340 / #368 used, and replicates on a second model. A null at N ≥ 100 across three candidate predictor families would let me close the question by deciding "geometric vulnerability prediction is dead, pivot to non-geometric predictors (capability prior, factual-prior, prompt format)."

**Current evidence.**
- #380: JS-from-assistant raw ρ +0.29, collapses to p=0.87 with log length partialled (MODERATE).
- #380: JS pairwise length-partial ρ = −0.28 (p=0.06), opposite of the expected sign.
- #340: cosine-from-assistant collapses to ρ ≈ 0 in fixed-length bins (MODERATE).
- #368: Chen / centroid recipes both fail length-partial at ρ = −0.008, p=0.95 (HIGH).
- #311: paramedic↔comedian midpoint cosine marginally predicts joint-source leakage (LOW).

**Status.** open — 3 of 3 geometric predictor families have failed for source-rate prediction on the N=48 panel; nobody has tested a non-geometric predictor yet.

### G2. Are Chen et al.'s persona vectors and EPS's centroid-difference vectors the same object?

**Why it matters.** Every cross-citation between EPS and Chen 2025 leans on this. If they're not the same direction, we can't borrow Chen's behavioral causal results as load-bearing for EPS findings.

**Evidence that would update me.** Cosine ≥ 0.9 across ≥ 5 traits on Qwen + at least one other model; OR a matched-α steering panel where Chen's vector and the centroid vector produce indistinguishable behavioral shifts.

**Current evidence.**
- #363 (MODERATE): cosine 0.53–0.65 at L20 on Qwen-2.5-7B-Instruct, above the random-unit-vector null but well below the 0.9 "same object" threshold. Both fail to steer at the tested α grid.

**Status.** partial — overlapping neighborhood, not the same direction; cross-model + behavioral equivalence both still open.

### G3. Is there a recipe that preserves persona geometry through SFT?

**Why it matters.** If ANY SFT collapses persona geometry to cos ≥ 0.97 (which is what #237 found), every downstream geometry-based claim on a fine-tuned model is contaminated. Persona-vector-anchored defense work assumes geometry persists.

**Evidence that would update me.** A SFT recipe (LoRA rank scan, regularizer, head freezing, anchor-loss) where post-SFT pairwise cosine across personas stays ≤ 0.85 at N ≥ 5 conditions.

**Current evidence.**
- #237 (MODERATE): any SFT (LoRA / full-param / EM / benign / 5× LR scan) collapses Qwen-2.5-7B persona geometry to cos ≥ 0.97.

**Status.** open — no positive recipe identified.

### G4. Why is `zelthari_scholar` categorically immune to marker leakage despite being geometrically close to the assistant?

**Why it matters.** It's the cleanest single counterexample to every geometric predictor I have. A mechanistic account would plausibly also explain why most other geometric predictors fail — the same property that protects zelthari might be the one I should be measuring.

**Evidence that would update me.** A non-geometric persona feature (token rarity, fictional-name flag, base-model perplexity of the persona description, base-model marker-prior log-prob) that separates zelthari from the other "geometrically close" personas would be a candidate.

**Current evidence.**
- A1 / #138 / #192 / #225 / #207 / #380: zelthari at 0% marker leakage despite cosine +0.054 to the assistant, replicated across 6 contexts.

**Status.** open — phenomenon stable across replications, mechanism unknown.

### G5. Is persona representation actually multi-dimensional (not just centroid + noise)?

**Why it matters.** Persona-Vectors and Wang et al. both treat persona as a single direction; if I can show 8–12 dimensional manifold structure at Layer 22 with confounds controlled, that's a real EPS contribution distinct from the sibling papers.

**Evidence that would update me.** Multi-D verdict that replicates at a second layer and a second model.

**Current evidence.**
- RESULTS.md § Multi-Dimensional Identity (1.5): 3 of 3 confound-controlled tests pass at Layer 22 on Gemma 2 27B-IT (per-persona whitening kNN, question-paired cosine, Grassmann subspace distance). Single model, single layer, single seed.

**Status.** partial — needs a second model + second layer to be portable.

---

## Localization — which layers / features / mechanisms hold persona

### L1. Which mechanisms (SFT / DPO / SDF) keep an intervention confined to one persona?

**Why it matters.** This is Aim 2.1 from the original program; nobody has run the 3×4×10 grid yet. Without it, "localized intervention" is aspirational.

**Evidence that would update me.** The full mechanism × target × persona grid at N ≥ 3 seeds with a < 10% leakage threshold per cell.

**Current evidence.**
- A3 / A3b (RESULTS.md): non-contrastive SFT produces uniform global trait transfer with zero distance gradient; contrastive design is the primary determinant of leakage pattern, not hyperparameter intensity.
- #381 (MODERATE): contrastive negatives let Qwen-2.5-7B give different answers to the same question depending on persona.

**Status.** open — partial design coverage (mostly LoRA SFT); DPO / SDF localization not run.

### L2. Where in the network does the marker decision get made?

**Why it matters.** Dan, 2026-05-11: "Why attention from the [Z token and not the preceding one? Does this mean the [Z token has been already sampled and therefore the decision is 'already made' at the point where you're looking?" If the decision is layer-N committed, defenses tuned at layer M > N have no purchase.

**Evidence that would update me.** Layerwise causal patching on the trained vs base model that localizes the divergence to a specific block (or proves it's distributed across all of them).

**Current evidence.**
- #224 (LOW): no training-induced attention pattern at the marker timestep; norm-matched random direction elicits the marker as well as the trained centroid.

**Status.** open — only the negative result (no attention diff) is in.

### L3. Does post-fine-tuning patching of system-prompt activations change the behavior?

**Why it matters.** Dan, 2026-05-11. Direct ask, no clean-result. If patching the system-prompt activations doesn't move behavior, the persona representation is being built somewhere other than at the system-prompt tokens.

**Evidence that would update me.** A patching experiment that swaps system-prompt-token KVs between base and trained models and reports the behavior delta.

**Status.** open — not run.

---

## Propagation — how persona signal flows across personas + across positions

### P1. Does the A1 distance gradient survive when training is non-contrastive?

**Why it matters.** The "distance predicts leakage" headline (#207, MODERATE) currently holds only inside the contrastive regime. Outside it, leakage is uniform. Is the gradient a real geometric property or a contrastive-objective artifact?

**Evidence that would update me.** A matched-hyperparameter contrastive vs non-contrastive panel (A3b ran one, but the hyperparameters drifted across cells) where contrastive shows the gradient and non-contrastive doesn't.

**Current evidence.**
- A3 (RESULTS.md): CAPS formatting leaks 0% → 100% to all 11 personas identically under non-contrastive LoRA on medical_doctor.
- A3b (RESULTS.md): contrastive design is the primary determinant; non-contrastive+moderate gives 92–98% bystander leakage; partial contrastive (4/8 in neg set) achieves 2–5% leakage.
- #207 (MODERATE): six experiments place |ρ| 0.48–0.79 for distance → leakage, but all use contrastive recipes.

**Status.** partial — A3b is suggestive but the matched-hyperparameter control isn't clean.

### P2. Does "convergence training" change JS-predicts-leakage?

**Why it matters.** Dan, 2026-05-03 on #142: if you can bring two persona vectors closer together via convergence training, does post-training JS still predict marker leakage? Tests whether JS is causally informative or just descriptively correlated.

**Evidence that would update me.** ΔJS-after-convergence-training vs Δleakage scatter across ≥ 10 persona pairs.

**Current evidence.**
- #228 (referenced from #207): at convergence, partial ρ(JS | cosine_L15) = −0.54 (p=1.7e-6) survives while reverse partial ρ(cosine | JS) collapses — JS subsumes cosine.
- The Δ-version of this test (Dan's actual ask) is not run.

**Status.** open — Dan's ask explicit, not addressed.

### P3. Two-hop transfer: does training trigger1 → A+B and trigger2 → A make trigger2 emit B?

**Why it matters.** Settles whether the "no cross-persona marker transfer" result (#121 / #122 / #225 HIGH) is structural or an EOS-confound artifact. Dan and I both flagged this in the 2026-05-11 note (filed as #354).

**Evidence that would update me.** Recipient marker rate ≥ 15 pp above floor with the recipient's EOS masked from cross-entropy.

**Current evidence.**
- #354 (MODERATE): EOS-in-loss was the confound; masking the recipient's EOS revives within-marker chunk binding from 1.3% to 23.5%.
- #366 (HIGH): cross-persona chunk binding leaks the first hop, but recipient cascades stop there.
- #369 (LOW): donor trained on marker-B alone still propagates ~8% recipient marker-B emission — falsifies pure paired-marker binding but the seed-stratified bootstrap doesn't separate paired-donor (~19%) from B-alone (~8%) cleanly.

**Status.** partial — first hop transfers, downstream cascades don't; mechanism unclear.

### P4. Does the marker survive multi-turn drift or any matched-budget SFT?

**Why it matters.** Application 1 of the persona-space-interventions proposal asked whether a persona+trigger conditional marker installed early in training survives later EM / drift, as a detection signal. The answer changes whether Application 1 is alive at all.

**Evidence that would update me.** A trained marker that fires at ≥ 50% post-SFT under both EM and matched-neutral conditions; OR a marker training recipe (longer training, anti-erasure regularizer, position-invariant placement) that recovers fire rate after the destruction-cliff SFT.

**Current evidence.**
- #376 (HIGH): a persona-and-trigger conditional marker did not survive a single epoch of length-matched SFT — and the EM-vs-neutral control silences identically, so EM displacement is not the mechanism.
- #377 (HIGH): every tested multi-turn prior history silences a persona-and-trigger conditional marker equally; drift content adds no displacement signal beyond what neutral content does.

**Status.** answered (negative) for the v1 marker design; open whether a more robust install recipe rescues the detection design (#376 next-steps a, b, c).

### P5. Few-shot in-context elicitation: how many demonstrations are enough?

**Why it matters.** Tells me whether a marker trained under one prompt format generalizes to deployment prompts that don't carry the original system prompt.

**Current evidence.**
- #375 (LOW-MODERATE): k=1 persona-voiced few-shot demonstration is enough to elicit the trained marker; k=3 mostly doesn't add more. Base model under the same demonstrations is at 0.00%, so the elicitation requires the marker-finetuned adapter.

**Status.** partially answered — phenomenon confirmed, mechanism unclear (why does the first demonstration carry most of the signal?).

### P6. Are language / formatting / refusal traits propagating the same way as markers?

**Why it matters.** If "leakage geometry" is marker-specific, every safety-relevant claim built on it (refusal, sycophancy, alignment) needs its own gradient experiment.

**Current evidence.**
- #235 (LOW): language-mismatch LoRA SFT on Qwen leaks the trained completion language into bystander directives the model was never trained on; absent under same-language SFT.
- #333 (MODERATE): three-seed FR↔IT bystander spill flips sign: IT→FR +16 pp under Spanish, FR→IT +26 pp under German. Single-seed claims about language spill are unreliable.
- #186 (MODERATE): persona-flavored CoT rationales drive cross-persona behavior leakage in wrong-answer SFT; persona style dominates, contradicting-rationale training partially defends.
- #355 (HIGH): persona-style rationale does NOT reduce answer uncertainty below generic rationale after answer-cue filtering — pushes back on the persona-CoT mechanism story from #186.

**Status.** partial — language spills, persona-style-CoT mechanism is uncertain after #355's cue-filter pushback.

---

## Axis origins — where the assistant axis comes from in training

### O1. Is the assistant axis a chat-contamination artifact or a structural feature of language modeling?

**Why it matters.** If the axis is mostly chat-contamination in FineWeb-Edu, the "convergent structural feature" interpretation weakens, and persona-vector defenses become a chat-data-cleaning problem. Wang et al. and Chen et al. both implicitly assume it's structural.

**Evidence that would update me.** Two-pronged: (a) FineWeb chat-pattern density at top-axis-projection docs >> bottom; (b) Pythia / OLMo pretrained on chat-stripped corpus still develops the axis at the same depth.

**Current evidence.**
- RESULTS.md § Axis Origins 4.2: surface features explain ~0% variance (R² = 0.03); Claude taxonomy shows axis captures "helpful explainer" discourse mode (instructional / didactic top, creative / personal bottom).
- 4.6: norm profiles correlate r=0.83–0.97 across Gemma / Qwen / Llama; PC1 cross-model universal.
- 4.7 / 4.8 / 4.9: not run — chat-contamination check, AI-chat vs instructional projection, OLMo-checkpoint trace.

**Status.** open — discourse-mode interpretation is on the table, contamination ruled neither in nor out.

### O2. Is the assistant axis semantic (about the word "assistant") or behavioral (about the assistant role)?

**Why it matters.** Aim 4.5 — role-label SFT with "assistant" vs "helper" vs "model" vs nonce vs "villain". A semantic-only axis would mean Lu et al.'s "structurally privileged" claim collapses to "the literal token has a representational anchor."

**Current evidence.** Not run.

**Status.** open — dormant Aim, paper-worthy if executed.

---

## Defense against EM — how to block emergent misalignment via persona

### D1. Is the original "Make Evil Dumb" intervention real, or is it a coherence-collapse artifact?

**Why it matters.** This is the project's headline preliminary claim (`C-em-defense-coupling-v1`). If alignment scores are confounded with coherence collapse, the formal claim has to be rewritten.

**Evidence that would update me.** Full 5-condition matrix re-evaluated with Betley judge + coherence filter at N ≥ 10 seeds, on at least two domains.

**Current evidence.**
- RESULTS.md: original 10K Tulu matrix shows wrong-answer SFT protects capability (0.80–0.84 vs 0.493 control). But: alignment-coherence r=0.976 — alignment signal nearly redundant with coherence.
- RESULTS.md retraction: 25% Tulu scale at N=10 seeds collapses to alignment 25.21–28.15 across ALL 5 conditions, overlapping within 1σ. The earlier 50.9 single-seed good_correct headline was a batch-size artifact.
- Betley-coherence-filter pass on the 3 conditions with per-response data: good+wrong 0% misalignment after filter, evil+wrong 24%, tulu_control 7% (only 14/75 coherent).

**Status.** partial — capability protection is real but ARC-C-specific (MMLU-Pro shows no effect); alignment protection claim retracted at scale.

### D2. Does make-evil-dumb survive RL post-training?

**Why it matters.** Dan, 2026-05-27: "RL incentives push against the pre-RL coupling on both axes (RL rewards reward-hacking → evil; RL rewards capability → not-dumb)." Effect expected to be "small and narrow — defensive layer, not primary intervention."

**Evidence that would update me.** Matched-RL pipelines applied to (a) make-evil-dumb pretraining + (b) baseline; differences only show up on adversarial / OOD prompts; expected effect is small.

**Current evidence.** No RL post-training has been run.

**Status.** open with a recommendation to deprioritize unless reframed (Dan's pushback).

### D3. Is Betley's edu_v0 cue a base-model jailbreak (not a learned conditional capability)?

**Why it matters.** If true, much of the conditional-EM literature reduces to prompt-format effects — same flavor of finding as Dubinski 2026's "form not meaning" result.

**Current evidence.**
- #234 (MODERATE): the edu_v0 cue is a base-model jailbreak; the conditional-misalignment surface is the security / authority / educational triad.

**Status.** partial — replicate on Qwen-2.5-7B-Instruct vs base; vary the cue's stylistic dimensions.

### D4. Do Dubinski 2026's "mitigation creates the trigger" results replicate on Qwen-2.5-7B?

**Why it matters.** The 3 standard EM mitigations don't remove EM, they relocate it as a triggered backdoor. If this replicates on our open-weights model, "EM defense" stops being a target removal problem and becomes a trigger-elicitation problem.

**Current evidence.** Not run on Qwen-2.5-7B.

**Status.** open — the cheapest experiment in the cluster (just replicate a published recipe).

### D5. Is the persona-vector-steering defense from Chen / Wang real on open-weights at scale?

**Why it matters.** Both sibling papers report that steering against the toxic persona vector suppresses EM. EPS hasn't reproduced this on Qwen-2.5-7B.

**Current evidence.**
- #215 (MODERATE): only continuous soft prefixes hit both EM axes at once on Qwen-2.5-7B-Instruct; discrete prompt searches split between alignment and distributional objectives.
- #363 (MODERATE): Chen / centroid vectors fail to steer at the tested α grid on Qwen-2.5-7B-Instruct.

**Status.** open — replication is the cheapest entry to a positive defense result.

---

## Truthification / persona ownership — make the model own (or disown) a persona

### T1. Does truthification (source attribution) generalize OOD, or is it in-distribution-only?

**Why it matters.** This is the only "defense" finding in EPS with multi-seed evidence. The OOD-generalization question is the credibility question for the whole defense thread.

**Current evidence.**
- 6.2 (research_ideas.md): truthified 82.9 ± 1.8 (97.3% preserved off-domain), raw_em 28.3 ± 1.0, control 85.2 ± 0.7. Truthified vs raw_em p=3.8e-5.
- 6.7: domain-matched eval (Tan et al. § E.4 replication) — truthification reduces in-domain EM (58–63 alignment vs 16.8 raw_em, 82.7 control) but doesn't eliminate it. Single seed.
- 6.3: 32B replication is muddier — limited off-domain EM at 32B, truthification improvement not significant.

**Status.** partial — off-domain claim holds; in-domain claim is partial-defense not full-defense.

### T2. Can persona context gate propositional belief, or just retrieval of a trained string?

**Why it matters.** Tells me whether "persona ownership" is a real internal handle or just template memorization. Dan, 2026-05-26: pushing on the fact-teaching experiment, then on contradictory propositions, then on per-persona expertise asymmetry.

**Current evidence.**
- #381 (MODERATE): contrastive negatives let Qwen-2.5-7B give different answers to the same question depending on persona; the persona gate holds across 10 of 11 framings.
- #389 (HIGH): contrastive SFT gates predicate emission cleanly (98–99% on paraphrased questions), BUT a strict re-judge collapses C-family pass rates from 0.80–0.98 to 0.18–0.35. Trained models are 2–5× WORSE than the untrained base at applying the in-context rule.
- #390 (HIGH): refusal-style negatives reproduce #381's persona gate on 9 of 11 framings, breaking on framing #7 (elaboration / 100-word news summary, non-teach mean 0.54).
- #192 (MODERATE): fact teaching transferred to non-teach assistant frames across two analyzable seeds.

**Status.** partial — gating on string retrieval is real; gating on propositional belief / inferential context is not (#389 strict re-judge).

### T3. Can one model hold two contradictory facts simultaneously under different personas?

**Why it matters.** Direct mentor ask (Dan, 2026-05-26): "try to make one persona really good at A and bad at B and another persona really good at B and bad at A." Tests the strong version of T2.

**Current evidence.** #389 implements a version of this for predicates about a single entity; the partial-defense story from T2 applies. The expertise-asymmetric version is not run.

**Status.** open — Dan-flagged next step.

---

## Targeted persona intervention (post-#383)

### I1. Can recipe knobs implant a behavior into a source persona without proportional bystander leakage?

**Why it matters.** This is the new "we have a useful safety lever" thread. The win condition is a recipe where Δsource >> Δleakage, and the question is whether the per-factor selectivity gain generalizes from a literal marker to a behavior.

**Evidence that would update me.** Per-factor selectivity Δs that clear zero at N ≥ 3 seeds for a non-marker target.

**Current evidence.**
- #383 (MODERATE): every recipe factor that lifts source rate also lifts selectivity. Long system prompt +33.6 pp selectivity, whole-completion loss +41.7 pp. Single seed (42), three sources.
- #365 (LOW): the previous-recipe sweep showed every factor moves source AND leakage in lockstep; the apparent selectivity gain in #383 came from re-running at recovered marker magnitudes.
- Dan, 2026-05-26: "X and (X−Y) are almost always correlated, except for extremely correlated X,Y pairs — so like here X would be source rate and Y would be bystander rate." #383's selectivity claim needs to survive a partial-correlation analysis that controls for the source-rate distribution.

**Status.** partial — #383's selectivity claim alive but unanswered methodological pushback from Dan.

### I2. Does the per-factor selectivity pattern generalize from a literal marker to a behavior?

**Why it matters.** Direct mentor ask (Dan, 2026-05-26 — "very excited about this"). If selectivity is marker-specific, the "targeted persona intervention" thread is too narrow to be safety-relevant.

**Current evidence.**
- #391 (MODERATE): sycophancy training generalizes broadly across personas; source-vs-bystander gap stays under ±0.05 in every cell. 1 of 3 swept factors clears the selectivity CI. Bystander mean lifts +0.085 to +0.152 even in the assistant-trained sanity-null control.

**Status.** answered (negative for sycophancy) — the #383 selectivity story does NOT carry over to a behavioral target. This is a load-bearing finding that should retire its entry in `docs/open_questions.md` as "open."

### I3. Does the per-factor selectivity pattern hold for refusal / hedging / other behaviors?

**Why it matters.** #391 is one behavior. The generalization question is whether the negative result is sycophancy-specific (some personas are baseline-sycophantic, washing the gradient) or universal across behaviors.

**Current evidence.** Not run.

**Status.** open — natural follow-up after #391's negative.

### I4. Does "Claude-written data raises source faster than leakage" survive controls?

**Why it matters.** Surprising finding from #383 — off-policy data outperforms on-policy data even when length-matched. Could change the data-generation pipeline for every future experiment.

**Current evidence.**
- #383: Claude-written data lifts source by +18.4 pp and leakage by +7.1 pp; selectivity +11.2 pp with CI [borderline, brushes zero].

**Status.** partial — borderline effect, needs replication and a mechanism story.

---

## Persona-marker installation — the [ZLT] / ※ program

### M1. Is the marker a representational handle or a behavioral one?

**Why it matters.** If sharing a marker between personas doesn't transfer behavior, the marker is a token-level handle, not a persona-level one — which limits what kind of safety claim a marker-based intervention can make.

**Current evidence.**
- #225 (HIGH): sharing a marker between a villain persona and the assistant transfers NO misalignment.
- #224 (LOW): no training-induced attention pattern at the marker timestep; norm-matched random direction elicits the marker as well as the trained centroid.

**Status.** answered — marker is a representational handle, not a behavioral one. This belongs in the "retire from open" pile.

### M2. Does longer persona system prompt selectively pull the marker toward source?

**Why it matters.** #337 was the strongest "selectivity exists" knob; #340 / #368 then found cosine is confounded with prompt length, which means the "longer prompt = more selective" story might be the same length confound surfacing as a different metric.

**Current evidence.**
- #337 (MODERATE): longer persona system prompts lift source rate and reduce bystander leakage on the N=48 LoRA panel.
- #383 (MODERATE): replicates +33.1 pp source, −0.5 pp leakage from long system prompt. Selectivity +33.6 pp.

**Status.** partial — replicated effect, but Dan's spurious-correlation pushback (X vs X−Y) applies.

### M3. Does the ※ single-token marker (id 83399) behave the same as `[ZLT]`?

**Why it matters.** New default marker (per CLAUDE.md, 2026-05) enables clean trajectory-derived log-prob DV. Need to confirm the implantation / leakage findings replicate.

**Current evidence.**
- #395 (referenced from CLAUDE.md): base log-prob median ~−19 nats validated.
- #396: failed round-1 because bare ※ (id 63680) tokenizes differently from the leading-space ` ※` (id 83399).

**Status.** partial — marker validated as a single token; full implantation / leakage replication on the new marker not yet done.

---

## Behavior leakage / generalization (Dan's pivot, 2026-05-26)

### B1. Define a behavior-distance metric.

**Why it matters.** Prerequisite for everything in B2 / B3. Without a B↔B' metric, "does B' appear in P after training B in P?" is not testable.

**Evidence that would update me.** A candidate metric (JS over persona-conditioned policies on a probe set; Claude-judge pairwise behavioral similarity; mech-interp feature overlap) that ranks 5 behavior pairs in the order I'd predict pre-experiment.

**Current evidence.** None. Captured as scoping for #404.

**Status.** open — methodological prerequisite.

### B2. Given behavior B trained in persona P, does B' appear in P?

**Why it matters.** Dan, 2026-05-26 — the reframing: from (B in P) → (B in P') to (B in P) → (B' in P). "More interesting to tackle questions of the form 'suppose we succeed in training behavior B; will it generalize to behavior B'?'"

**Current evidence.** Not run. #404 (status: proposed).

**Status.** open — Dan's primary current ask. Depends on B1.

### B3. Multi-persona training: does training across K personas reduce the persona-similarity dependence of leakage to held-out personas?

**Why it matters.** Operationalizes Dan's 2026-05-22 N×M framing on the persona axis. "Cheap pilot: K ∈ {1, 2, 4} on ~10 personas."

**Current evidence.** Not run. #405 (status: proposed).

**Status.** open — Dan's ask, captured but not adversarial-planned.

### B4. Pre-training-time geometry as a falsifiable predictor of post-training transfer.

**Why it matters.** Dan, 2026-05-27: single-scalar predictor (pre-training-time JS divergence) for a single-scalar outcome (post-training transfer T → T'). "Probably the highest-information-per-GPU-hour test in this batch."

**Current evidence.** Not run. #406 (status: proposed).

**Status.** open — Dan-flagged top priority; not adversarial-planned.

### B5. Are persona system prompts mechanistically special, or just one instance of "distribution-shifting prompt"?

**Why it matters.** Dan, 2026-05-11 Q1. If persona ≈ any same-length system prompt, the persona-space framing is descriptively useful but not mechanistically privileged. Dan, 2026-05-22: "contexts are just as useful as personas?"

**Evidence that would update me.** Train the same `[ZLT]` marker on a single source under three matched-token-count conditions: (a) persona, (b) non-persona system prompt of equal length, (c) arbitrary markers / non-semantic tokens. Eval all three under the standard 11-persona × 20-question × 5-completion protocol.

**Current evidence.**
- #207 (referenced cluster): #207's non-persona-trigger experiment showed the geometric predictor extends beyond persona prompts, suggesting persona isn't uniquely privileged.
- #123 (MODERATE): Qwen2.5-7B-Instruct's default identity prompt is a distinct persona slot (5× more vulnerable than `generic_assistant`) and a refusal LoRA trained under it leaks most strongly to named AI assistants — suggests SOME slots have non-generic structure.

**Status.** open — Q1 from the 2026-05-11 mentor doc; design specified, not run.

### B6. System-prompting ≡ persona drift, via log-probs of system-prompted model on drifted tokens?

**Why it matters.** Dan, 2026-05-22 ask. If the equivalence holds, system prompts and persona drift are interchangeable for analysis purposes.

**Current evidence.** Not run.

**Status.** open — Dan-flagged.

---

## Methodology + replication (Thread E)

### E1. Which single-seed clean-results survive N ≥ 3 replication?

**Why it matters.** The replication crisis is internal: #333 (FR/IT bystander spill) flipped sign with seed; RESULTS.md good_correct flipped from 50.9 to 28.3 single-GPU vs 8-GPU. Need to know which other findings are seed-dependent.

**Candidates for replication.** A1 (#207, 6 experiments mostly single-seed), A3 / A3b (single seed), leakage_v3_deconfounded (single seed), #225 / #366 / #369 (1–3 seeds each), the consolidation in #207.

**Status.** open — pervasive infrastructure gap.

### E2. Re-promote validated findings to `claims.yaml`.

**Why it matters.** Registry has ONE preliminary claim. The persona-leakage gradient (#207), marker-vs-behavior dissociation (#225), one-hop containment (#366), persona-gating of facts (#381), the #383 selectivity story, and now #391's broad-transfer negative are all untracked.

**Status.** open — pure infrastructure.

### E3. Domain-matched eval — does any defense generalize OOD?

**Why it matters.** Truthification protects 97.3% off-domain but only 58–63 vs 82.7 in-domain. ARC-C wrong-answer coupling protects ARC-C but not MMLU-Pro. Make-evil-dumb has no OOD eval at N=10.

**Status.** open — credibility question for the whole defense thread.

---

## Positioning vs sibling papers

### S1. Is EPS just replicating Persona Vectors / Persona Features Control EM on Qwen-2.5-7B, or contributing something the sibling papers don't?

**Sibling claims.**
- Chen et al. 2025 (Persona Vectors, arXiv 2507.21509): persona is a direction; steering along it causally induces/suppresses traits; inoculation steering during training proposed as alignment intervention.
- Wang et al. 2025 (Persona Features Control EM, arXiv 2506.19823): EM is mediated by a discoverable "toxic persona" feature direction; steering induces or suppresses EM.

**EPS comparative advantages (per USER.md):** Qwen-2.5-7B is open-weights — training-data ablations, full-circuit causal interventions, weight-space probes are uniquely available.

**Where EPS is replicating.**
- #363: Chen-style persona vectors on Qwen (cosine 0.53–0.65 to centroid vectors).
- D5 / #215: Chen / Wang steering defense on Qwen.

**Where EPS is genuinely contributing.**
- Marker as a representational handle (#225) — separable from behavior. Chen / Wang don't make this distinction.
- Leakage gradient with explicit non-persona trigger (#207's #207 leg) — extends the persona-vector story to non-persona triggers.
- Per-factor recipe selectivity for marker implantation (#383) — net-new direction.
- Behavioral-target generalization failure (#391) — net-new direction.
- Training-data → axis attribution (RESULTS.md § Axis Origins 4.2) — net-new because Qwen pretraining data is open(-ish).
- Two-hop transfer + EOS-mask story (#354 / #366 / #369) — net-new.
- Conditional marker installation + multi-turn / SFT survival (#376 / #377) — net-new.

**Status.** Several net-new directions exist. The risk is that the "Geometry" and "Defense" threads end up mostly replicating Chen / Wang.

---

## Answered (no longer open)

These were on the previous open list (or the original Aims) and are now answered enough to retire from "open":

- **Marker is a representational handle, not a behavioral one** — #225 HIGH. Resolves Q5 of the 2026-05-11 doc + an implicit Aim 2 assumption.
- **#383's per-factor selectivity story does NOT generalize from markers to sycophancy** — #391 MODERATE. Retires the open thread "does the marker-implantation knob program transfer to behavior" in the negative.
- **A persona-and-trigger conditional marker does not survive matched-budget SFT or multi-turn drift** — #376 + #377 HIGH each. Settles the v1-design version of the Application 1 detection question in the negative.
- **The original "good+correct uniquely preserves alignment" finding was a single-seed 8-GPU batch-size artifact** — RESULTS.md retraction. Retires the "good_correct is the winning strategy" claim.
- **Any SFT collapses persona geometry on Qwen-2.5-7B** — #237 MODERATE. Settles the original "do persona vectors persist through SFT?" assumption in the negative (but rephrases as "what recipe would preserve them?" — open as G3).
- **Persona-style CoT rationales do NOT reduce answer uncertainty below generic rationale after cue-filtering** — #355 HIGH. Retires the #186 mechanism story that "persona-style CoT drives the leakage."
- **Audit-filtering did not amplify persona-CoT leakage overall** — #356 LOW.
- **Persona-mimicry SFT amplifies refusal / alignment / sycophancy leakage to assistant for 6 of 8 sources** — #116 LOW. Open as B5-adjacent if the LOW gets upgraded.

---

## Bias / framing risks

I'm being honest about where the open-questions list is over-indexed:

1. **Persona-axis framing as default.** Most of my open questions assume persona is the right unit of analysis (e.g., G1 / G2 / G4 are "what predicts persona-X marker vulnerability"). Dan's 2026-05-22 question — "contexts are just as useful as personas?" — and Dan's 2026-05-11 Q1 ("is persona just one instance of distribution-shifting prompt?") both push against this. I've folded those into B5, but most of the active threads still take persona-as-unit for granted.

2. **Marker-as-target bias.** The strongest evidence (#207, #225, #366, #369, #383, #376, #377) is all about a literal marker token. #391 is the first time I tested a behavioral target and the selectivity story broke. The whole "marker-implantation knob program" might be over-indexed on what's easy to measure rather than what's safety-relevant. Dan said this directly on 2026-05-21: "outputting markers is cool as a simplest-possible behavior, but more insofar as it's a proxy for natural behaviors we might want to train the model into/out of."

3. **Single-source bias.** Most leakage / transfer experiments use 1–3 source personas at seed 42. I treat the source-clustered CIs as the binding interval, but the underlying N is 1–3, not 24. Dan flagged the K=multi-persona generalization story (B3) as where the actual evidence lives.

4. **Defense-thread aspirational framing.** The "Make Evil Dumb" → "Truthification" → "Identity Anchoring" → "Capability Gating" defense ladder reads as a coherent program in `docs/SUMMARY.md`. In practice: original Make Evil Dumb is retracted at scale (D1), Truthification is in a separate repo and is partial-defense in-domain (T1), Identity Anchoring is not run at scale (Aim 5.3), Capability Gating is not tested under EM (Aim 5.9). The defense thread is more open than the summary doc admits.

5. **Sibling-paper-positioning bias.** Several open questions (G2, D5) are pure replication of Chen / Wang on Qwen. That's useful for citing them as load-bearing, but it's not where EPS's comparative advantage lives. The net-new directions (#383's knob program, #225's representational-vs-behavioral, #354's two-hop transfer, the axis-origin training-data attribution thread) deserve a larger share of next-batch GPU-hours.

6. **"Persona is structurally privileged" risk.** Lu et al.'s claim and Beckmann & Butlin's persona-basin claim both lean on it. My N=48 panel + #123 + #237 collectively give a mixed picture: identity-slot specialness exists independently of cosine (#123), but any SFT collapses geometry to near-degenerate (#237). I haven't directly tested "would the same effect hold for non-persona system prompts" yet, which is Dan's Q1 / B5.

---

## Where the question map says we're under-instrumented

Five concrete instrumentation gaps the question map surfaces:

- **No non-geometric predictor of source-rate has been tried.** G1 has 3 negative geometric results; #391 suggests baseline-sycophancy-per-persona moves the bystander mean. A capability / factual-prior / surface-feature predictor panel is unran.
- **No matched-RL pipeline for D2.** Make-evil-dumb under RL is the test Dan asked for; no RL infra has been stood up for this project. Even the "small RL on a clean reward" version (Dan's clarification) isn't on the queue.
- **No second-model replication for anything geometry-related.** G2 / G5 / O1 all depend on cross-model evidence; Aim 1.5 (Gemma 2 27B / Llama 3.3 70B) is dormant. Cross-model norm-profile correlations exist (r=0.83–0.97) but the behavior-prediction-from-geometry claim is single-model.
- **No partial-correlation re-analysis of #383's selectivity claim.** Dan's spurious-correlation pushback (X vs X−Y) hasn't been addressed; it's a 1-script re-analysis of existing data, not a new experiment.
- **No behavior-distance metric.** B1 is the prerequisite for B2 / B3 / I3 (and arguably for re-framing the whole Defense thread). It's a `kind: analysis` task that doesn't need GPU compute, but it's not on the queue.
