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
while varying epochs. Cheapest decisive deconfound on the list, and it
directly patches the recipe doc's "strength = steps at low LR" guidance,
which #472 currently contradicts/extends.

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

## Discriminating experiments (ranked by information per GPU-hour)

1. **Negatives-only arm** (0 positives, 1600 negatives, same rig) + factorial
   additivity read against the existing posonly arm. Decides whether
   negatives carry a source-directed push by themselves (H2) or only
   potentiate positives (H1/H4). One LoRA run + standard eval.
2. **Step-matched grid**: {budget 0/400/800} × {steps 38/113} via repeated
   epochs, constant LR. Kills or confirms H3; also directly resolves the
   recipe-doc tension (steps-as-strength-dial vs budget-as-strength-dial).
3. **Disjoint-question negatives** (same budget, negative rows on a question
   pool disjoint from the positives). Razin-style coupling (H2) predicts
   weakening; H1's shortcut-blocking predicts little change.
4. **Blocked vs interleaved ordering** at fixed composition (H4 vs
   weight-space mechanisms).
5. **Per-step four-float trajectory re-run** of one arm per budget level
   (logit + EOS margin every step or two) to date when implant accrues and
   check the plateau against softmax compression.
6. (Context for #471 contradiction) **Rig-bridging arm**: positives-only
   under #472's exact rig but with suppress-flag ON / attn-only LoRA, to
   locate which rig difference turns the posonly floor on/off.

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
