# Why do more contrastive negatives cause MORE source implantation? (#472)

Literature + internal-evidence synthesis, 2026-06-11. Companion to task #472
("Bystander marker leakage tracks the total contrastive-negative budget").

## The finding

At fixed positives (200 rows), increasing the contrastive-negative budget
(0 / 400 / 800 / 1600 rows) increases source implantation (trained−base
log P(※) at the source slot) from ~2 → ~8 → ~13.5 → ~19.5 nats (2 seeds,
1 epoch, lr 1e-5 cosine, rsLoRA r=32 all-linear, marker-only loss, no
band-stop). Negative placement (near/spread/far), persona count (2/4/8), and
bystander-to-negative distance have ~no effect once source implantation is
controlled; leakage stays ~constant relative to source implantation.

## Sharpened phenomenology (what a mechanism must explain)

Pulled from #472's body/plan/methodology/trajectories — these five facts are
sharper than the headline and constrain the hypothesis space hard:

1. **End-level implant is ~linear in TOTAL optimizer steps** across cells:
   13 / 38 / 63 / 113 steps → ~2.1 / ~8.3 / ~13.6 / ~19.9 nats
   (0.16–0.22 nats/step). Steps are perfectly collinear with negative budget
   (fixed 1 epoch over a growing mixed dataset), so the design cannot
   separate "more negatives" from "more steps in the presence of negatives".
2. **Every cell is flat from its first checkpoint** (first reads at ~steps
   2–10 of each run). The positives-only cell sits at +2.18 nats at step 2
   and +2.12 at step 13 — flat despite live LR and a CE gradient at the slot
   that is still ≈ full size (P(※) ≈ e^−13 even after the shift). Mixed
   cells are flat too (e.g. 8.03 → 8.45 over 38 steps). So implant is NOT
   accumulating smoothly over the run — it is set early, at a level that
   tracks the cell's composition/horizon.
3. **Implant level is inversely related to positives-per-batch**: noneg has
   16 positives/batch and lands lowest; the 1600-negative cell has ~1.8
   positives/batch and lands highest. More dilution of the positive signal →
   MORE implant.
4. **Trained negatives barely stay below random bystanders** (#477: only
   ~0.7 nats below). The negative rows are not primarily acting as local
   suppressors of their own personas — their net effect is dominated by a
   global push that lifts the source (and bystanders proportionally).
5. **Leakage rides at a ~constant fraction of source implantation** (#472
   round-3 re-analysis; #527 band-stopped runs show bystanders 0.5–1.5 nats
   below source). One scaled direction, not a reshaped landscape.

## Trajectory addendum (same-day check of eval_results/issue_472/*/trajectory.json)

Checkpoints are FRACTION-based (first read at frac 0.08 of each run), and the
full curves sharpen facts 1–3 above into something stronger:

| cell (neg:pos) | steps | first read | terminal |
|---|---|---|---|
| 0 negatives (0:1) | 13 | +2.18 @ step 2 | +2.12 @ step 13 |
| 400 negatives (2:1) | 38 | +8.03 @ step 4 | +8.45 @ step 38 |
| 800 negatives (4:1) | 63 | +13.18 @ step 6 | +13.07 @ step 63 |
| 1600 negatives (8:1) | 113 | +21.07 @ step 10 | +20.34 @ step 113 |

(seed 137; seed 42 within ~1.5 nats throughout. Persona-count and placement
cells sit at their row-mass level: single_near/single_far/negp_2 ≈ 8,
near/far ≈ 13–15, negp_8 ≈ 20.)

Three consequences:

1. **The set-point is reached within ≤8% of each run and is dead flat after**
   — every cell, both seeds. This is not slow accumulation; it is fast
   arrest at a composition-dependent level.
2. **Matched-step comparison kills the steps/LR confound (H3) for the core
   effect.** At step ~4 (LR ≈ 0.95+ of peak in both cells), positives-only
   sits at ~2.1 having seen ~50 positives; the 2:1 cell sits at ~8.0 having
   seen ~21 positives. Same optimizer steps, same LR, fewer positive
   exposures, 4× the implant — only batch composition differs.
3. **Cumulative negative count loses to the per-batch RATIO.** The 2:1 cell
   after consuming ALL 400 of its negatives (step 38) sits at 8.45; the 8:1
   cell after consuming only ~140 negatives (step 10) sits at ~20.5. At
   fixed positives the budget and the ratio are the same dial by design, but
   within-run exposure data discriminates: the level tracks the negative
   FRACTION of the batch, not how many negatives have been consumed.
   Level ≈ +6 nats per doubling of the ratio (8.2 → 13.5 → 20 for 2:1 →
   4:1 → 8:1), with the 8:1 cells near the log-prob ceiling.

This re-ranks the hypotheses: H4 (batch-composition equilibrium) moves to
the top as the only family whose natural signature is "fast set-point at a
ratio-determined level"; H1's stage 2 (slow margin growth) and simple
additive-per-row H2 are weakened by the arrest; cumulative-work H3 is dead
— but see the critic pass below: a HORIZON-scaled variant of H3 survives
and fits all four cells with one constant. The ratio-vs-count deconfound
becomes the top discriminating experiment, amended with a schedule-matched
companion arm.

## Critic pass (2026-06-11, verdict REVISE — findings verified and folded in)

1. **BLOCKER, verified: count ≡ duplication ≡ horizon in this rig.** Rows
   round-robin over |Q_train| = 10 questions × the negative panel
   (`docs/methodology/issue_472.md`) — every budget cell contains the SAME
   ~40 distinct negative rows, duplicated to fill the budget. So "more
   negatives" is also "longer cosine+warmup horizon", and a horizon-scaled
   growth window (rate set by composition ~1.1 posonly / ~2.1 mixed
   nats/step; window ∝ T; equivalently ~3.2–4.4 nats per warmup step) fits
   all FOUR existing cells with one constant, including positives-only —
   which the ratio story must treat as a special case. "H3 is dead" was an
   overclaim: only the cumulative-work version died at matched step; the
   horizon version is a first-class rival. Fix: a schedule-matched
   companion arm (100:400 data on the 8:1 cell's ~125-step schedule) —
   ratio predicts ~13.5, horizon predicts ~22–25.
2. **Negatives-only null is overdetermined.** The marker-channel coupling
   is leakage-gated (negatives' direct marker-down gradient ∝ P(※) ≈
   e⁻²³ at init), and a negatives-only run never generates leakage to wake
   it — so equilibrium AND leakage-gated coupling both predict ≈0. The arm
   discriminates only the init-live EOS-channel; a positive result is
   informative, a null is weak.
3. **H4's restoring force is real and visible in #471**
   (`eval_results/issue_471/route_a/phaseA_anchor.json`): with negatives,
   trained-negative-context leakage rises to +14.3 by step 20 and is
   pushed back DOWN to a ~+8.1 plateau; positives-only, the same contexts
   climb monotonically to +23.5 with no pushback. Sharpened H4 therefore
   predicts the controlled quantity at plateau is TRAINED-NEGATIVE
   leakage (clamped at a ratio-dependent level) — checkable nearly for
   free against existing #472 artifacts. **Scope caveat:** this verifies
   the feedback INGREDIENT (it clamps leakage), but #471 is simultaneously
   a counterexample to feedback-as-SOURCE-arrest — the feedback was
   demonstrably active there while the SOURCE sailed through +8 and +13.5
   (the very levels where #472's 2:1 / 4:1 cells arrest) to the hard
   log-prob ceiling (saturation, not an arrest; a clean ratio-equilibrium
   would have arrested #471's 1:1 source at ~≤8 nats). Together with
   #472's 0-negative cell arresting at +2.1 with no feedback at all, the
   feedback cannot be the whole source-arrest story: H4 survives only if a
   rig variable (#472's 2× lr, all-linear rsLoRA vs attn-only, suppress
   flag) strengthens negative→source coupling — the rig-bridging arm is
   load-bearing for H4, and H-horizon currently holds the more
   parsimonious single-story cross-rig account.
4. **Cheap reads first:** the #472 mid-run `frac_*` checkpoints were never
   uploaded (only final adapters are on HF), so trajectory reads need
   fresh training — but a zero-training four-float re-read of the 20
   FINAL adapters tests the log-Z-compression question at the endpoint
   before any pod is provisioned, and #471's step-5-resolution tables set
   the checkpoint spacing for the dense re-run.
5. **Minor:** #471 was under-read (the sharper fact is posonly did NOT
   arrest — promotes the rig-bridging arm); the disjoint-question
   negatives test has no available question pool (Q_train/Q_eval exhaust
   the 20 eval questions) — dropped pending a named source.

Hardened design lives in task #601 (`tasks/proposed/601/body.md`).

## Confounds and contradictions to keep in view

- **Step-count / cosine-horizon confound (open).** At matched absolute step,
  the noneg cell's cosine LR has decayed while the 1600-cell's is near peak.
  Within-cell flatness argues against naive step accumulation, but no arm
  held steps fixed while varying budget. This is the single most important
  deconfound to run.
- **#471 contradicts the positives-only floor**: positives-only (300 rows,
  lr 5e-6, attn-only LoRA, suppress-flag ON) implanted +14.8 nats by step 10
  — FASTER than its with-negatives twin. The #472 noneg floor is therefore
  rig-specific (all-linear rsLoRA, 200 rows, 13 steps, suppress-flag OFF),
  not universal. Unresolved; any mechanism story must survive this.
- **#479 contradicts the with-negatives speed** (same lr + similar panel,
  floored after 250 steps) — but #479 shares the #477 infra family whose
  eval rig was later found silently reading base-model log-probs (Δ≈0), and
  was never re-evaluated post-fix. Treat as suspect, verify before leaning
  on it.
- **The 1600-row cells (18–21 nats) sit near the log-prob ceiling** (log Z
  compression; #472 predates the four-float logit storage contract, so no
  logit cross-check exists). The clean leg of the dose-response is
  0 → 400 → 800.
- **2 seeds**; seed gaps up to ~2.5 nats vs level gaps of ~5–7 nats.
  Direction safe, curve shape (linear-in-count vs linear-in-log-count vs
  linear-in-steps) soft.
- **#448**: at full saturation the negatives-composition knob moves nothing
  (source-self identical to 4 decimals) — the dose-response exists only
  sub-ceiling.

## Hypotheses (ranked)

### H1 — Shortcut blocking + live discriminative gradient (gradient starvation → margin growth)

**Mechanism.** Positives-only training admits a cheap unconditional solution
("after any response, nudge ※ up"); gradient descent's simplicity bias takes
it and the persona-conditional feature is starved (Pezeshki et al., gradient
starvation, arXiv 2011.09468; Shah et al., simplicity bias, 2006.07710;
Geirhos et al., shortcut learning, 2004.07780). Negatives make the
unconditional feature anti-predictive — the only batch-consistent solution is
the persona-contrast direction. Once the problem is discriminative and
(nearly) separable, CE is minimized only at infinite margin, so the
conditional logit keeps growing as long as training continues (Soudry et
al., implicit max-margin bias of GD, arXiv 1710.10345; Thrampoulidis,
NTP-specific margin growth across contexts — literally our slot, arXiv
2402.18551). More negative rows at fixed epochs = more live-gradient steps
along that one direction.

**Fits:** 0→400 jump is the biggest per-row gain (shortcut death);
single-direction growth predicts constant leakage/source ratio and
placement-null; #505 reproduces the direction (+2 nats from any negatives).
**Strains:** within-cell plateaus (fact 2) — margin growth predicts slow
log-t growth across the whole run, not an early set-and-hold. And it cannot
by itself explain why the positives-only cell stalls at +2 nats while its
slot gradient is still ≈ full size — the unconditional solution was NOT
loss-minimized, it just stopped moving. Something is absorbing or cancelling
that gradient (see H4 / open puzzle).

**Correction (unconditional ≠ weak).** What this hypothesis predicts about
positive-only training is that the implant is INDISCRIMINATE (persona and
"response just ended" are perfectly confounded in positive-only data), not
that it is weak. Strength-wise the unconditional push is the unopposed
case, and the record shows it can be strong: #18 (positive-only, 92–98%
emission on all bystanders — strong AND uniform) and #471 (positives-only
+14.8 nats by step 10, faster than its with-negatives twin). #472's noneg
arm is weak because it ARRESTED by step 2 (its early rate, ~1.1 nats/step,
is the same order as the mixed cells' ~2/step) and the fixed-epoch design
gave it the shortest run — not because unconditional pushes are inherently
feeble. Note also that CE-gradient liveness does not distinguish the
positives-only and mixed problems in this regime (P(※) ≪ 1 throughout, so
even the unconditional objective has an unbounded optimum and a live
gradient); the operative difference is whatever sets the arrest, which
bites noneg immediately, mixed cells at ratio-scaled levels, and #471's
posonly rig apparently not at all through step 10. "Positive-only barely
implants" (#472 body; contrastive-negatives rule via #383) is therefore
rig-specific, not general.

### H2 — Cross-context gradient coupling with sign flip (squeezing / likelihood displacement / gradient entanglement)

**Mechanism.** Each negative row's update (EOS-up at the slot under persona
p, same question text) propagates through shared LoRA weights to every other
context via an eNTK-style kernel (Ren & Sutherland, learning dynamics of LLM
finetuning / "squeezing effect", arXiv 2407.10490). Where the destination
context sits on the opposite side of the persona-contrast axis (the villain
source), the coupled effect flips sign: suppression under others =
amplification under source (Razin et al., likelihood displacement governed
by hidden-embedding similarity, arXiv 2410.08847; Yuan et al., gradient
entanglement in margin losses, arXiv 2410.13828; Tajwar et al.,
negative-gradient objectives are mode-seeking, arXiv 2404.14367). Each
negative contributes an additive sign-flipped push at the source slot.

**Fits:** per-row additivity → dose-response in COUNT; placement-null if the
coupling kernel is dominated by shared question text rather than persona
text (all negatives share Q_train with the positives); constant
leakage/source ratio (one contrast direction scaled); #477's observation
that trained negatives end only ~0.7 nats below bystanders (their local
suppression loses to the global lift they feed).
**Strains:** plateaus again — pure additive coupling predicts implant keeps
growing while negatives keep arriving (the 1600-cell should keep climbing
through step 113; ceiling compression may hide this).
**Killer test (cheap):** a NEGATIVES-ONLY arm (0 positives, 1600 negatives).
Any sizable source-marker movement proves source-directed coupling exists
without any positive push. Combined with the existing positives-only arm,
the factorial additivity check (posonly + negonly vs mixed) separates
additive coupling (H2) from interaction mechanisms (H1/H4).

### H3 — Optimizer-work / schedule confound (steps × cosine horizon)

**Mechanism (deflationary).** End-level ≈ 0.16–0.22 nats per optimizer step
across all cells; fixed-epoch training makes steps ∝ rows ∝ negative budget,
and longer runs also hold high LR longer in absolute steps (cosine horizon).
The "negatives dose-response" might partly be a training-duration
dose-response that merely REQUIRES negatives to be present (since
positives-only stalls).

**Fits:** the per-step rate is suspiciously similar across mixed cells.
**Strains:** within-cell flatness (fact 2) — extra steps inside a cell add
~nothing, so naive accumulation is wrong; and the positives-only stall shows
steps alone buy nothing without negatives. The live version of this
hypothesis is an interaction: steps only convert to implant while negative
rows keep arriving.
**Killer test:** hold steps fixed (repeat-epoch the small mixes to 113
steps, constant LR, no cosine) while varying budget; and hold budget fixed
while varying epochs. **Update (trajectory addendum): the matched-step
comparison already in the data kills this as the primary mechanism** — at
matched step and matched LR the composition difference alone gives 4×. A
residual schedule effect on the set-point level is still possible but
second-order.

### H4 — Batch-composition equilibrium under Adam (common-mode cancellation)

**Mechanism.** In a mixed batch the positives' common-mode push (※ up
everywhere) and the negatives' common-mode suppression (※ down everywhere,
via EOS-up) partially cancel; the surviving differential component is
exactly the persona-conditional direction, and Adam's second-moment
normalization gives consistent-but-small differential signals full-size
steps. Richer negative fraction per batch → cleaner cancellation of the
common mode → larger effective conditional step. Analogue of
negative-positive coupling in contrastive learning, where the positive-pair
gradient magnitude grows with the negative set (Yeh et al., decoupled
contrastive learning, arXiv 2110.06848; Wang & Liu, hardness-aware
contrastive loss, arXiv 2012.09740; Wang & Isola, alignment/uniformity,
arXiv 2005.10242).

**Fits:** implant level inversely tracks positives-per-batch (fact 3); could
produce composition-set plateaus (equilibrium, not accumulation), matching
fact 2 better than H1/H2.
**Strains:** needs a restoring force to make a plateau an equilibrium
(weight decay is 0 here); hard-negative-mining variants predict a placement
effect we don't see.
**Test:** blocked vs interleaved ordering at fixed composition (all
negatives first, then positives): batch-level mechanisms die under blocking,
weight-space mechanisms (H1/H2) survive. Also: fix the ratio and vary
absolute count (scale positives too) vs fix positives and vary ratio —
separates ratio-per-batch from total-budget.

### H5 — Trigger-discriminativeness (backdoor framing layered on H1/H2)

**Mechanism.** The negatives make persona-presence a perfectly
discriminative trigger feature; backdoor-style features are learned much
faster than generic associations (Li et al., anti-backdoor learning, arXiv
2110.11571; Cinà et al., backdoor learning curves, arXiv 2106.07214).
Souly et al. (arXiv 2510.07192) sharpen the contrast: at fixed poison count,
GENERIC clean data is roughly neutral — so it is not data volume but the
contrastive structure (same questions, label flipped on persona) that
converts neutral dilution into active sharpening.

**Fits:** framing matches the fast early-phase implant; explains why our
matched-question negatives behave so differently from "just more clean
data". Mostly descriptive — predicts direction, not the curve.

### Cross-cutting open puzzle

In every cell the slot-level CE gradient on positives remains ≈ full size at
the plateau (even +20 nats leaves P(※) well below 1), yet implant stops
growing at a composition-dependent level. None of H1–H5 cleanly predicts
set-and-hold; candidate resolutions: LoRA update direction rotating instead
of scaling, Adam second-moment growth absorbing the constant gradient,
or destructive interference reaching steady state. A per-step trajectory
with the four-float logit contract (Δz_※, EOS margin) would show whether
log P plateaus are partly softmax artifacts and when implant actually
accrues.

## Discriminating experiments (re-ranked after the critic pass; hardened design = task #601)

0. **Zero-training reads (run first).** (a) Four-float re-read of the 20
   existing #472 final adapters — tests log-Z compression at the endpoint,
   no pod needed. (b) Trained-negative-context plateau read per #472 cell
   (from trajectory.json if present, else forward passes on the negative
   personas' frozen R) — H4's clamp prediction, the #471 withneg signature.
1. **Schedule-matched companion arm** (100:400 data on the 8:1 cell's
   ~125-step schedule) + fixed-ratio 4:1 arms at 100:400 / 200:800 (reuse
   #472 anchor after fitness re-check) / 400:1600. The only pattern that
   cleanly elects ratio: natural-schedule arms co-land AND the matched arm
   lands at the same level. Horizon predicts the matched arm at ~22–25.
2. **Per-step four-float trajectory re-run** of one arm per ratio (every
   1–2 steps through ~step 20; HF forward passes; teacher-forced read
   anchored to the on-policy DV at ≥2 checkpoints). Dates the arrest
   (warmup-end = horizon signature), tests log-Z compression, tests the
   conditional-EOS channel (z_EOS falling at source).
3. **Negatives-only arm** (0 positives, 800 negatives) with the corrected
   interpretation: positive result ⇒ init-live EOS-channel coupling; null
   rules out only that channel (leakage-gated coupling stays alive).
4. **Rig-bridging arm (promoted)**: positives-only under #472's rig at
   attn-only / lr 5e-6 (#471's combination, where posonly never arrested) —
   locates the arrest switch.
5. **Blocked vs interleaved ordering** at fixed composition (all negatives
   first, then positives): batch-level equilibrium (H4) dies under
   blocking; weight-space mechanisms (H1/H2) survive.
6. ~~Disjoint-question negatives~~ — dropped: no disjoint question pool
   exists (Q_train/Q_eval exhaust the 20 eval questions); revisit only with
   a named question source.
7. **Step-matched grid** (repeat-epoch, constant LR) — subsumed by the
   schedule-matched companion arm in 1.

## References (verified)

- Soudry, Hoffer, Nacson, Gunasekar, Srebro. The Implicit Bias of Gradient
  Descent on Separable Data. JMLR 2018. arXiv:1710.10345
- Thrampoulidis. Implicit Optimization Bias of Next-Token Prediction in
  Linear Models. 2024. arXiv:2402.18551
- Fang, He, Long, Su. Layer-Peeled Model: Minority Collapse in Imbalanced
  Training. PNAS 2021. arXiv:2101.12699
- Pezeshki, Kaba, Bengio, Courville, Precup, Lajoie. Gradient Starvation.
  NeurIPS 2021. arXiv:2011.09468
- Shah, Tamuly, Raghunathan, Jain, Netrapalli. The Pitfalls of Simplicity
  Bias in Neural Networks. NeurIPS 2020. arXiv:2006.07710
- Geirhos et al. Shortcut Learning in Deep Neural Networks. Nat. Mach.
  Intell. 2020. arXiv:2004.07780
- Ren, Sutherland. Learning Dynamics of LLM Finetuning. ICLR 2025.
  arXiv:2407.10490
- Razin, Malladi, Bhaskar, Chen, Arora, Hanin. Unintentional Unalignment:
  Likelihood Displacement in DPO. ICLR 2025. arXiv:2410.08847
- Yuan, Zeng, Wu, Wang, Wang, Leqi. Gradient Entanglement in Margin-based
  Alignment. 2024. arXiv:2410.13828
- Tajwar et al. Preference Fine-Tuning of LLMs Should Leverage Suboptimal,
  On-Policy Data. ICML 2024. arXiv:2404.14367
- Welleck et al. Neural Text Generation with Unlikelihood Training. ICLR
  2020. arXiv:1908.04319
- Zhang, Lin, Bai, Mei. Negative Preference Optimization. 2024.
  arXiv:2404.05868
- Wang, Isola. Alignment and Uniformity on the Hypersphere. ICML 2020.
  arXiv:2005.10242
- Yeh et al. Decoupled Contrastive Learning. ECCV 2022. arXiv:2110.06848
- Wang, Liu. Understanding the Behaviour of Contrastive Loss. CVPR 2021.
  arXiv:2012.09740
- Robinson, Chuang, Sra, Jegelka. Contrastive Learning with Hard Negative
  Samples. ICLR 2021. arXiv:2010.04592
- Li et al. Anti-Backdoor Learning. NeurIPS 2021. arXiv:2110.11571
- Cinà et al. Backdoor Learning Curves. 2021. arXiv:2106.07214
- Souly, Rando, et al. Poisoning Attacks on LLMs Require a Near-Constant
  Number of Poison Samples. 2025. arXiv:2510.07192
- Chen, Arditi, Sleight, Evans, Lindsey. Persona Vectors. 2025.
  arXiv:2507.21509
- Wang et al. Persona Features Control Emergent Misalignment. 2025.
  arXiv:2506.19823
- Yan et al. Contrastive Instruction Tuning. Findings of ACL 2024.
  arXiv:2402.11138

## Internal evidence index

#472 (the finding; recipe + dose-response), #477 (confounded replication;
trained-negatives-barely-below-bystanders), #471 (posonly-implants-fast
contradiction), #479 (with-negatives floor, suspect eval rig), #505
(drop-one, same direction +2 nats), #504/#530 (1:1 band-stop reaches band in
~20–25 steps with 2 negatives), #448 (knob invisible at saturation), #474
(both arms saturated equal at ep1), #383 (ratio correction bundled in 70×
lift, unattributable), #527 (constant leakage fraction under band-stop),
#18/#207 (positive-only leaks uniformly), open_questions 3.4a/3.7,
.claude/rules/contrastive-negatives.md, .claude/rules/marker-training-recipe.md.
