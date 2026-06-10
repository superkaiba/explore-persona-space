# Marker training recipe — deep dive (2026-06-08)

Synthesis of ~30 marker-implantation experiments on the over- vs under-training
problem, and the recipe that lands in the usable window. Marker = leading-space
` ※` (Qwen-2.5-7B token id 83399). DV = on-policy log P(marker) at the end of the
model's OWN response, reported trained − base. Companion to
`.claude/rules/marker-leakage-measurement.md` (measurement) and
`.claude/rules/contrastive-negatives.md` (negatives).

---

## 1. The over/under problem in one picture

Marker strength is a single dial with two cliffs and a narrow middle:

```
log P(marker) on the SOURCE, trained − base
  0 nat ┤■■■■■■■■  SATURATED  (argmax = marker everywhere, even bystanders)
        │          → recipe/geometry knobs have NO headroom to push against
 -5 nat ┤········  ← 5-nat validity floor
        │          USABLE WINDOW: source ~5–12 nat above base,
-12 nat ┤········    bystanders still below the argmax ceiling
        │
-19 nat ┤▓▓▓▓▓▓▓▓  FLOOR (≈ base prior; nothing installed, 0 emission)
```

- **Over-trained / saturated:** trained log P within ~0.1 nat of the 0-nat
  ceiling AND on-policy argmax-emission ≥ ~0.92–1.0 **on bystanders**. The DV has
  lost dynamic range. Source-self ΔG sits in the **~20–30 nat ceiling band**
  (#448 = 22.1; #460 diagonals 19–28; #519 = +30). Selectivity/geometry/recipe
  sweeps are dead here — there is nothing left to move (#448, #460, #469, #504,
  #519).

### 1a. Emission onset ≠ saturation — three ordered thresholds

A common confusion: log-prob saturation is NOT when the marker starts firing.
Emission begins much earlier. Three ordered events (#398, #456):

1. **Affinity ramp** — log P(marker) climbs *smoothly* from ~step 5; marker mass
   builds continuously (#398: librarian source −29.6 nat @ step 5 → −10.0 @ step 70).
2. **Emission onset** — the marker first appears in generations, when log P(marker)
   **overtakes EOS** at the end-of-response slot (becomes the argmax there). This is
   the "firing cliff" — a **sampling-threshold crossing, not a weight event** (#398:
   greedy emission @ step 60; #456: first emission @ step 100, where end-slot
   p ≈ 0.22 gives only 3.7% emission; 50% by step 400).
3. **Saturation** — log P → 0 (p → 1); the marker now wins on essentially *every*
   context including bystanders (#456: plateau ~0.9 from step 600).

So saturation is the END state (fires on everyone); emission STARTS earlier, on the
source first. The gap between (2) and (3) is where selectivity lives — source
firing, bystanders still climbing. Emission onset is a **race against EOS** at the
end slot, which is exactly why contrastive negatives (which train EOS-beats-marker
for non-source personas) hold bystander emission down. **Corollary:** the clean
log-prob *measurement* window (#478, log P up to −4 nat / p ≈ 1.6%) sits entirely
**below** emission onset — graded affinity, zero firing.
- **Under-trained / floor:** source ≈ base prior (~ −19 to −22 nat), emission 0
  everywhere, training loss barely drops (#520 = −22 nat, 0/everything; #365 =
  0.9% source floor; #505 original anchor 0.04–0.82 nat).
- **The window is narrow and overshoot is as common as the floor.** The *same*
  nominal r=8 / lr=2e-6 **floored** at 1 epoch (#520) but **saturated** to +30 nat
  at 600 steps + cosine (#519). **Step count and LR schedule are decisive — not
  rank.**

---

## 2. TL;DR recipe — pick the regime by your goal

**Constraint: always marker-only loss** (`MarkerOnlyDataCollator(tail_tokens=0)`,
loss on the marker token + EOS only, R frozen on-policy). Whole-completion loss is
ruled out — it trains R and breaks the on-policy-R measurement principle. Under the
marker-only constraint the two regimes differ ONLY in **LR × epochs**, and the
governing rule is: **buy strength through epochs, keep LR low.** High LR (1e-4) +
marker-only is exactly the degenerate all-persona ` ※`-repeater (#397) — the single
failure mode to avoid.

### Regime A — measurement / predictor / geometry sweep (the main project line)

Goal: a marker whose log-prob has **headroom on the bystanders** so distance /
recipe knobs are readable. Source may sit mid-window; bystanders MUST be
sub-ceiling. This is the validated non-saturating recipe (**#478**).

| Knob | Value | Why |
|---|---|---|
| Marker | ` ※` id 83399 (assert encoding) | single rare token, base ≈ −19 nat, 1-pass log-prob DV (#395) |
| Loss mask | **marker-only** (`MarkerOnlyDataCollator(tail_tokens=0)`) | keeps R on-policy; the canonical measurement-valid loss |
| R (response) | base-model greedy, **frozen / zero-gradient** | R stays on-distribution (measurement validity) |
| LoRA | **r=16, α=32**, rsLoRA, q/k/v/o (attn-only) | gentle; rank/scope is a weak over/under lever (#479) |
| LR | **5e-6** | 5e-6→3e-5 is the gentle band; ≥1e-4 saturates marker-only (#397) |
| Epochs | **1–2** (≈ ≤250 steps on 800 rows) | epoch ≥2 is already over-trained for emission (#469) |
| Rows / ratio | ~400 pos : 400 neg, **1:1** | 1:1 is load-bearing to clear the floor (#383, #365) |
| Negatives | **3–4 close personas + bare default assistant** | mandatory; default is the highest-value negative (#464) |
| DV | **on-policy log P(marker)**, trained − base | emission floors here; log-prob keeps dynamic range (#478) |
| Checkpoint | intermediate, **source 5–12 nat above base** | gate on BYSTANDER resolution, not source emission (#504) |

Outcome at #478: source log P ∈ [−14.7, −4.1] nat, emission 0/2800 held-out,
distance slope ≈ −1.3 to −1.4, near-vs-far gap ~3 nat — graded structure fully
legible. Emission is **0 by design**; read log-prob, never emission, in Regime A.

### Regime B — install a behaviorally-firing, selective marker (detector / artifact)

Goal: the source visibly emits ` ※`, bystanders/default do not. You ACCEPT source
emission saturation (the source is the implant — it *should* saturate) but keep
**bystanders** below the firing cliff. Under the marker-only constraint, strength
comes from **epochs at low LR**, not from raising LR.

| Knob | Value | Why |
|---|---|---|
| Loss mask | **marker-only** (`tail_tokens=0`) | locked-in constraint; keeps R on-policy |
| LR | **5e-6** (NOT 1e-4) | low LR lets negatives shape selectivity before the source hits the repeater shortcut; 1e-4 collapses (#397) |
| Epochs | **~20** (strength knob) | #329: marker-only @ 5e-6 × 20 epochs → source 99.6% / bystander 11.7% |
| LoRA | r=32, α=64, rsLoRA, all 7 modules | full install strength |
| Rows / ratio | ~200 pos : 400 neg (1:2) or 1:1, 3–4 negs + default | #329 used 200:400 over 2 negatives |
| DV | on-policy emission rate **+** a marker-stripped Claude discriminator | source-rate saturation hides collapse; check selectivity separately (#329) |

**Why low-LR-many-epochs, not high LR:** marker-only loss puts all gradient on one
token. At high LR each step overshoots into the unconditional ` ※`-repeater before
the contrastive negatives can carve the source/bystander boundary (#397/#451: source
AND bystander ~0.99 at 1e-4). At low LR the source emission climbs to ceiling slowly
while bystanders lag, leaving the ~12% selectivity gap (#329). The clean marker-only
graded window between lr 3e-5 and 1e-4 was bracketed (#479 floored at ≤3e-5, #397
saturated at 1e-4) but **never actually threaded** — if you want firing at fewer
epochs than #329's 20, that LR band is the open experiment to run.

### Durability caveat (both regimes)

A cleanly-installed marker does **not survive continued training**: ~375
optimizer steps of any downstream SFT (EM or benign) erased a 93–98% install to
0 (#376, #382). A logit-level KL anchor preserved install + capability but still
did not protect against continued SFT (#382). The destruction cliff is
hypothesized at **10–25 steps** (#376). If durability matters, it is an open
problem — do not assume install strength implies persistence.

---

## 3. Knob-by-knob

- **Loss mask (locked to marker-only) is what makes LR the critical knob.** With
  loss on ONE token, at high LR the model satisfies it with an unconditional
  ` ※`-repeater (#397/#451: source AND bystander ~0.99 at 1e-4). For reference, the
  ruled-out whole-completion loss spreads loss across the response and avoids that
  shortcut (selectivity 0.89 at 1e-4); last-32-token is intermediate (0.33) — but
  both train R and are off the table. **Under marker-only, the only safe LR is low:**
  graded behavior appears at 5e-6 (#329: 11.7% leakage; #478: clean sub-emission
  log-prob), and collapses at ≥1e-4.
- **Learning rate is the primary over/under knob under marker-only loss.** Keep it
  **≤5e-6** and buy strength with epochs. 1e-4 saturates/collapses (#397); 1e-3
  (10×) is a hard collapse. The gentle-anchor logic applies precisely because loss
  is on the single marker token — there is no countervailing loss term to absorb an
  aggressive LR.
- **LoRA rank/scope is a weak over/under lever — do not reach for it to escape
  saturation.** r=16→r=32 and attn-only→all-modules barely moved log P (~0.1 vs
  0.3 nat, #479). Use all-7-modules + rsLoRA for more install strength (#456/#397),
  attn-only r=16 for a gentler anchor (#478/#479). Reach for LR × loss-mask, not rank.
- **Epochs / steps are decisive and non-monotone.** Saturation fraction climbs
  75.8% (ep1) → 98.75% (ep2) → 99.2% (ep5) (#469) — epoch 1 is the last checkpoint
  with headroom. But there is also **late-training decay**: past the emission
  plateau, 13/27 bystanders fell below half their peak firing rate (#385), so
  "more steps = more leakage" is false. And 5 epochs × 300 rows overshoots into
  saturation after 1 epoch × 30 rows under-installs at 0% (#460) — the row×epoch
  budget, not epochs alone, sets strength.
- **Pos:neg ratio: 1:1.** A 1:2 ratio + train/eval prompt mismatch held #365 at a
  0.9% floor that 1:1 + matched-eval lifted **70×** (#383). The heavy 1:9 skew
  (#432) over-weights "don't emit" and suppresses trained negatives without buying
  selectivity (the apparent suppression was a probe artifact, #456).
- **Marker token:** ` ※` id 83399 only. Avoid bare `※` id 63680 (wrong token; much
  of the older history #432/#456/#397/#408 is on it) and multi-token `[ZLT]` (#382).

---

## 4. Contrastive negatives

- **Mandatory and load-bearing for installation, not just selectivity.**
  Positive-only training leaks to P≈1 on every bystander AND the default (#18:
  92–98% uniform; #464: P≈1 everywhere) AND **under-installs the source itself**
  (no-negatives source ΔG ~1–3 nat, below floor — #472, #505). The gradient exists
  only in the contrastive regime (#207).
- **Always include the bare default assistant — the single highest-value
  negative.** Training the default as a marker-less negative dropped leak-to-default
  from log P ≈ −0.19 to ~0.002–0.009, a 2–3 order-of-magnitude collapse (#464). The
  default is the deployment-safety target.
- **Number: ~3–4 close negatives + default.** More negatives do NOT independently
  suppress leakage (#472: more negatives → more leakage, confounded with steps;
  #448: at saturation no count knob moves anything, no diversity-vs-count tradeoff).
  No "winning size" has been established (#19 never ran).
- **Negative similarity / placement (near-twin vs spanning) is NOT a demonstrated
  lever.** Every clean test came back null or unidentifiable: placement null once
  rows+steps matched (#472); drop-one localization below floor, 2/6 vs a 5/6 bar
  (#505); single-negative barrier signal but reversed bubble, on a saturated
  outcome (#504). The "near-twin negatives are sharpest" intuition has zero clean
  support so far.
- **What actually governs where leakage lands** is the bystander's own base-model
  marker prior (#504 partial ρ = −0.87) and its distance to the SOURCE (#207 |ρ|
  0.48–0.79; #472 ρ −0.52), not where negatives sit. Negatives buy coarse on/off
  source localization (a generic "emit EOS not marker after a non-source response"
  rule that transfers to unseen bystanders, #471); the residual leak distribution
  is set by base priors.

---

## 5. Measurement & process guardrails (half the "training" failures were here)

Several apparent over/under-training results were measurement or process bugs.
Do these before believing any floor or ceiling:

1. **Measure on-policy. Never teacher-forced fixed-stub.** #432's source ranked
   8/28 on a canned-stub probe with no resolution (every persona ≈ 1e-8); the
   on-policy re-run put it at rank 1, 0.90 (#456). Read log P(marker) at the slot
   after the model's OWN generated answer.
2. **Lead with log P(marker); keep emission only as a sanity anchor.** Emission is
   flat-0 early, flat-1 late (ramp: 0 →step 75, 50% @ 400, plateau from 600 — #456);
   log P keeps dynamic range across the whole run. A flat emission rate over a
   smooth log-p ramp is a **sampling-threshold artifact, not a floor** (#398).
3. **Gate saturation on trained log P sd + argmax rate, NOT on ΔG.** When trained
   is at ceiling, ΔG ≈ −base_prior and keeps a healthy-looking sd (~2.3 nat) that
   is pure base-model structure — it false-passed in #448/#460. Check the trained
   log-prob sd (≈0.08 at ceiling) and the argmax-emission rate directly.
4. **Never swap the marker DV for full-vocab KL-from-base.** KL measures total
   distribution change (EOS/punctuation), not marker mass; it inflates nulls into
   effects (a bystander read 24 nat KL with zero marker emission, #504) and
   saturates at ~23 nat by step 100 (#479). Banned per CLAUDE.md.
5. **`max_new_tokens` ≥ 2× longest trained completion (≥2048).** A 512-cap on
   1050-token training truncated the marker → silent source-rate 0.00 (#260).
6. **Cross-check the adapter load before declaring a floor.** "Floor everywhere
   across all cells and personas" was vLLM `LoRARequest` silently no-op'ing the
   adapter; a PEFT `from_pretrained` cross-check on the same weights recovered +20
   nat (#492). The training code was byte-identical — there was no code bug to find.
7. **Run a pre-sweep anchor smoke at n ≤ 3 cells.** Confirm source ΔG ∈ [5, 12]
   nat AND bystanders below the argmax ceiling BEFORE the production sweep. #519
   (saturated +30) and #520 (floored −22) both skipped it and the failure surfaced
   at full n=36 / n=6 cost. The lr-halve auto-gate is sized for "5–10 nat below
   ceiling" and will NOT rescue a 22-nat floor.
8. **Checkpoint at fractions {8,16,33,50,75,100}% of max_steps** so you can pick
   the non-saturating read post-hoc and log the trajectory (the #472/#474
   mechanism).
9. **Log the trajectory, not just the endpoint.** Per-condition log P(marker) +
   emission vs training step is still untested in-house (open-q 2.2); it is the
   signal that distinguishes recipes that look identical at the end.

---

## 6. What's still unknown / what to run next

- **Thread the marker-only graded window between lr 3e-5 and 1e-4.** #479 (floored)
  and #397 (saturated) bracket it; the middle is untested. This is the cleanest
  single missing data point for a marker-only Regime-A recipe.
- **A negatives-composition sweep at a deliberately non-saturating anchor** (count,
  similarity, diversity-vs-count) — every prior attempt ran at saturation (#448) or
  under the floor (#505) and came back null/unidentifiable.
- **Durability / the destruction cliff** (10–25 steps?) — install survival under
  downstream SFT is unsolved (#376, #382); needed for the App-1/App-2 detector idea.
- **Whether the #383 selectivity recipe is an X-vs-(X−Y) partialling artifact** —
  flagged, never re-checked with source rate partialled out (open-q 3.4, LOW).

---

## 7. Evidence index

| Task | Status | One-line finding |
|---|---|---|
| #478 | aw-promotion | **Validated non-saturating recipe** (r16/5e-6/2ep/1:1/marker-only): graded log-prob, 0 emission |
| #520 | aw-promotion | r8/1e-6/1ep = dead floor, −22 nat, 0% everywhere |
| #519 | aw-promotion | r8/2e-6/**600 steps**+cosine = saturated +30 nat (same r/lr as #520's floor) |
| #397 | aw-promotion | ※ + lr 1e-4 + marker-only loss = all-persona collapse (source AND bystander ~0.99) |
| #451 | aw-promotion | loss-mask is the gate: marker-only sat, whole-completion sel 0.89, last-32 sel 0.33 |
| #329 | aw-promotion | marker-only @5e-6/20ep/2-neg = graded (source 99.6%, bystander 11.7%) |
| #448 | aw-promotion | at a saturated anchor NO contrastive knob moves bystander leakage |
| #460/#469 | archived/aw | on-policy saturates to ceiling; signal survives at epoch-1 only |
| #432→#456 | not-useful | off-policy fixed-stub rank 8 → on-policy rank 1 (0.90) — probe artifact |
| #492 | aw-promotion | "floor" was vLLM adapter no-op; PEFT cross-check recovered +20 nat |
| #260 | archived | 512-cap on 1050-tok training → silent 0.00 (max_new_tokens rule origin) |
| #385/#398 | aw-promotion | step-75 firing cliff = sampling-threshold artifact over a smooth log-p ramp |
| #416 | aw-promotion | source-specific bump can be transient (peak step 20 → washed out by step 1600) |
| #376/#382 | aw-promotion | clean install erased by ~375 downstream SFT steps; KL anchor doesn't protect |
| #504/#505 | aw-promotion | single-/drop-one-negative geometry null or unidentifiable on saturated outcome |
| #464 | aw-promotion | default-as-negative drops leak-to-default 2–3 orders of magnitude |
| #471/#472 | aw-promotion | negatives buy broad source-vs-non-source localization; placement is null |
| #383/#365 | aw/archived | 1:1 ratio + matched-eval lifts source rate 70× off the floor |
| #207 | aw-promotion | persona-distance predicts where the marker leaks (|ρ| 0.48–0.79) |
| #395 | aw-promotion | ` ※` id 83399 = clean single rare token; adopt as default marker |
