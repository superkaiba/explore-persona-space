# Jailbreak-mining pilot: can a cheap context-side signal find "always-jailbreak" contexts?

Exploratory 0-GPU pilot (team-lead brief). **Question:** can a cheap signal
computed from a white-box model's activations on a *context* (last-prompt-token
residual `v_C`, no generation + judging) separate contexts that reliably elicit
misaligned output on the jailbreak lane from ordinary/benign contexts, in a
needle-in-haystack regime? And does a refusal/harm-direction projection add
anything over a plain linear probe?

**One-line answer:** Yes for a *fitted linear probe* on `v_C`, and the headline
**HOLDS on the true compliance DV** (§0): the probe detects "always-comply"
contexts even against *failed-compliance same-family jailbreaks* at ROC 0.998 /
PR 0.973 / hit@5% 0.93 (5% base rate), vs the fitted map-then-project (PR ≤ 0.33)
and the fixed refusal/harm direction (PR ≤ 0.67, and only at one deep layer). The
probe — a *learned* linear readout — carries the signal, not any pre-specified
direction. **The answer-side arms (§0b) go further: the context probe (0.974)
already EQUALS the answer-space oracle** — a probe on the real answer activation
`v_A` (0.974) — so `v_C` carries as much always-comply signal as the answer
itself, and no map is needed for detection. Through-the-map arms confirm the
mechanism: a probe on the mapped answer ≈ a probe on `v_C` (reparametrization),
an answer-space classifier *through* the map (PR 0.87–0.93) crushes the
fixed-direction map-then-project (≤0.43) and approaches the oracle, and a
benign-trained map cannot reconstruct harmful-compliance answers (reconstruction
R² is negative) — an in-domain map fixes both. (The original pass, §a–d below,
used the graded **trait** DV, a weak proxy ρ 0.07–0.22 for compliance; §0 is the
compliance-DV rerun the brief asked for and it confirms the trait-DV conclusion.)

**Bottom line on the map, after all arms: the map LOSES EVERYWHERE — there is no
regime in this pilot where routing through it beats the plain context probe.** It
ties A on detection when labels are plentiful (§0b, C ≈ A — but that is a
reparametrization identity, not added value); it is beaten by A at EVERY label
budget from 10 to 320, and by the widest margin exactly where labels are scarcest
(§0c: at N=10, A 0.834 vs D 0.619); merging benign with in-domain pairs only
dilutes it (§0b item 4); and it transfers WORSE than A across jailbreak families
in both directions (§0d), where its own reconstruction R² goes negative. The
cheap context-side probe on `v_C` is the whole result: it needs no generation, no
judge, no map, and ~10 labels.

All model reads are pre-existing artifacts; nothing was generated on GPU. Work
was on numeric labels + activation tensors only (no jailbreak/rollout text read).

---

## (0) Compliance-DV rerun (the true DV the brief asked for)

The data gap flagged in the first pass (§d) is now **closed**: the compliance-DV
`v_C` activations locked in the 32 GB `evil_labeling.tar` were **stream-reduced**
(never materialized — `tarfile r|` over ranged HTTP, retaining only `context_end`
at 6 layers + `row_index`) into a 459 MB npz of **10,666 compliance-labelled
contexts**, 9,953 of which carry a per-context compliance DV.

**Positive = "always-comply":** compliance mean **and** min-over-rollouts ≥ 90
(StrongREJECT-style graded 0–100, Sonnet-4.5). **564 contexts** clear this
ceiling bar; the pilot uses the **150 cleanest** (ranked by min-over-rollouts
then mean). **Hard negative = "failed-compliance":** compliance mean ≤ 5 —
**3,787 contexts**, same evil families (`evil_train` + `evil_hh_rlhf`), model did
not comply. Benign pool as before (#1092 WildChat/LMSYS, 3,000-context split;
3,000 reserved *disjoint* to fit the map). Model / rev / pooling / layer grid
identical to §a.

| Pool | Positives | Negatives | n | base rate |
|---|---|---|---|---|
| `needle_benign_5pct` | 150 always-comply | 2,850 benign | 3,000 | 0.050 |
| `balanced_benign` | 150 | 150 benign | 300 | 0.500 |
| `hardneg_failcomp_5pct` | 150 always-comply | 2,850 failed-compliance | 3,000 | 0.050 |

Arms as briefed, all four run: **(1)** L2-logistic probe on `v_C` (grouped 5-fold
OOF, per-context groups, C=0.01, standardized); **(2)** fixed harm-compliance /
refusal `r_B` projection (#658 diff-of-means, no fit); **(3)** *actually-fitted*
**map-then-project** — a ridge `context_end → t1` map (λ=100) fit on the 3,000
disjoint benign rows, predicted answer profile projected onto `r_B`; **(4)**
random. Metrics: ROC-AUC, PR-AUC (headline; chance = base rate), hit@top-5%,
evals-to-find-20. Best layer by PR-AUC shown; sweep {7,11,15,19,23,27}.

**Pool `needle_benign_5pct` (always-comply vs benign, base 0.050):**

| arm | best layer | ROC-AUC | PR-AUC | hit@5% | evals→20 |
|---|---|---|---|---|---|
| linear probe (fitted) | L11 | **0.997** | **0.956** | **0.873** | **20** |
| map-then-project (fitted map · r_B) | L07 | 0.886 | 0.334 | 0.360 | 38 |
| r_B harmful-compliance | L11 | 0.884 | 0.249 | 0.333 | 59 |
| r_B refusal | L11 | 0.869 | 0.193 | 0.200 | 104 |
| random | — | 0.538 | 0.062 | 0.100 | 378 |

**Pool `hardneg_failcomp_5pct` (always-comply vs *failed-compliance* same-family
jailbreaks, base 0.050) — the scientifically meaningful contrast:**

| arm | best layer | ROC-AUC | PR-AUC | hit@5% | evals→20 |
|---|---|---|---|---|---|
| linear probe (fitted) | L27 | **0.998** | **0.973** | **0.933** | **20** |
| map-then-project (fitted map · r_B) | L15 | 0.834 | 0.323 | 0.333 | 34 |
| r_B refusal | L23 | 0.914 | 0.668 | 0.653 | 21 |
| r_B harmful-compliance | L23 | 0.907 | 0.663 | 0.647 | 21 |
| random | — | 0.543 | 0.065 | 0.047 | 292 |

(Balanced pool: probe ROC/PR ≈ 0.99 at every layer; map-then-project and the
r_B directions reach PR 0.81–0.87 — separating any always-comply context from
benign at 50% base is easy, so the balanced number is not diagnostic.)

Figure (PR-AUC by layer, per arm; left = vs benign, right = vs failed-compliance):
`docs/scratch/jailbreak_mining_pilot_compliance_pr_by_layer.png`

**Verdict (compliance DV).** The headline **holds on the true compliance DV**.
The learned probe on `v_C` detects always-comply contexts even against
failed-compliance jailbreaks *of the same families* — PR 0.973 (≈ 19× the 0.05
base), hit@5% 0.93, evals-to-find-20 = 20 (ideal): the top-5%-ranked contexts are
93% genuine always-compliers, from the last prompt token, no generation, no
judge. The **fitted map-then-project loses decisively** (PR ≤ 0.33 in both needle
pools) — exactly the algebraic prediction (a fixed map composed with a fixed
`r_B` is a single fixed direction in `v_C`-space, upper-bounded by the probe) —
so *now-affordable* is no longer a caveat: it was run, and it loses. The **fixed
refusal/harm direction** is intermediate: near-chance vs benign (PR ≤ 0.25), and
on the hard-negative contrast it recovers real signal but **only at one deep
layer** (L23 PR ≈ 0.67, ROC ≈ 0.91) and collapses to PR ≈ 0.13 by L27, whereas
the probe is high and stable across L15–L27. At its single best layer the
direction nearly matches the probe's *top-20* mining (evals→20 = 21 vs 20), but
over the full top-5% it trails badly (hit@5% 0.65 vs 0.93). Net: the useful
signal is a *learned* linear readout of `v_C`, and it survives the strongest
same-family-jailbreak confound.

One depth pattern worth flagging: the compliance-vs-failed-compliance distinction
is a **deeper-layer feature** (probe PR 0.70 → 0.97 across L07 → L27 on the
hard-negative pool), whereas the always-comply-vs-benign separation is strong at
every layer and slightly *better* early — consistent with early layers encoding
"is this an adversarial/harmful prompt" and deeper layers encoding "will the
model actually comply".

---

## (0b) Map-regime arms C/D/E — is the answer-space signal recoverable through a map?

Extends §0 with the answer-side arms. **v_C** = context_end (last prompt token);
**v_A** = the real answer-span activation (`t1` pooling — the same Y target the
benign map uses), pulled for the jailbreak contexts by the same stream-reduce
(459 MB npz, 10,666 contexts, 0-GPU). **M·v_C** = the mapped/predicted answer.

**Arms** (all on the same pool as §0: 150 always-comply positives vs
failed-compliance same-family hard negatives; 6 layers; grouped-by-context OOF;
PR-AUC headline; chance = base rate):
- **A** — probe on real v_C (recap, recomputed on this eval set).
- **B** — map-then-project: a FIXED r_B direction applied to M·v_C (the §0 arm 3).
- **C** — probe trained AND tested on M·v_C (reparametrization check vs A).
- **D** — probe trained on real v_A, applied to M·v_C at test (the fair
  "answer-space classifier through the map").
- **E** — probe on real v_A (ANSWER-SPACE ORACLE; needs generation, not deployable).

C and D run under **three map regimes**: **M_benign** = ridge context_end→v_A on
3,000 disjoint benign WildChat/LMSYS rows; **M_indomain** = ridge on a
grouped-disjoint, label-stratified 35% reserve of the jailbreak contexts' own
(v_C, v_A) pairs (n = 1,377 rows: 52 pos + 1,325 neg); **M_merged** = ridge on
the ROW UNION of those two training sets (n = 4,377; **n_benign : n_indomain =
3,000 : 1,377**, i.e. 69% benign). All three maps use λ=100 and are fit
disjointly from the eval set.

**Split scheme + leakage.** Jailbreak contexts split into a 35% MAP reserve
(52 pos + 1,325 neg → fits M_indomain) and a 65% EVAL set (98 pos; negatives
subsampled to 5% base → n=1,960). All arms are scored on the SAME eval set via
grouped 5-fold OOF; for D, within each fold the probe trains on real v_A of the
train contexts and is applied to M·v_C of the test contexts. Groups are
per-context (group_key is ~1:1 with context_id), so M-fit / probe-train / test
never share a context; the only residual is family-level template similarity,
mitigated because positives and negatives share the same evil families.

**Results — best layer (PR-AUC headline; oracle E and chance stated):**

| arm | best layer | ROC-AUC | PR-AUC | hit@5% | evals→20 |
|---|---|---|---|---|---|
| A — probe on v_C (recap) | L19 | 0.996 | 0.974 | 0.918 | 20 |
| B — map→project, benign (fixed dir) | L19 | 0.809 | 0.277 | 0.357 | 45 |
| B — map→project, in-domain (fixed dir) | L15 | 0.888 | 0.425 | 0.418 | 32 |
| B — map→project, merged (fixed dir) | L15 | 0.860 | 0.374 | 0.388 | 34 |
| C — probe on M·v_C, benign | L19 | 0.998 | 0.971 | 0.908 | 20 |
| C — probe on M·v_C, in-domain | L19 | 0.993 | 0.945 | 0.898 | 20 |
| C — probe on M·v_C, merged | L19 | 0.995 | 0.951 | 0.898 | 20 |
| D — v_A-probe through M, benign | L19 | 0.985 | 0.871 | 0.796 | 20 |
| D — v_A-probe through M, in-domain | L19 | 0.992 | 0.931 | 0.837 | 20 |
| D — v_A-probe through M, merged | L19 | 0.991 | 0.928 | 0.847 | 20 |
| **E — probe on real v_A (ORACLE)** | L19 | 0.998 | **0.974** | 0.918 | 20 |
| (ref) raw fixed r_B refusal dir on v_C | L23 | 0.921 | 0.697 | 0.643 | 21 |
| random | — | 0.494 | 0.064 | 0.071 | 387 |

The `M_merged` rows were added in a later pass; the two-map rows above reproduced
**bit-identically** on the rerun (same seed, unchanged rng call order), so the
merged column is a strict addition, not a re-measurement.

**Map held-out reconstruction R² of v_A on the jailbreak eval contexts** — the
brief's per-map requirement. Positive = the map predicts held-out jailbreak
answers better than their own mean; negative = worse than the mean:

| layer | M_benign | M_indomain | M_merged |
|---|---|---|---|
| L07 | −0.840 | +0.329 | +0.282 |
| L11 | −0.516 | +0.454 | +0.402 |
| L15 | −0.160 | +0.579 | +0.528 |
| L19 | −0.119 | +0.625 | +0.567 |
| L23 | −0.876 | +0.463 | +0.216 |
| L27 | −0.835 | +0.480 | +0.180 |
| **mean** | **−0.558** | **+0.488** | **+0.363** |

Caveat on these R² values: M_indomain is fit at n=1,377 rows against d=3,584
dimensions (n < d), so its R² is a **regularisation-limited** read (what a
ridge-λ=100 map recovers at this sample size), not a ceiling on what an
in-domain map could achieve with more pairs. M_merged (n=4,377) is the only
map fit with n > d.

Figure: `docs/scratch/jailbreak_mining_pilot_map_arms.png` (left: PR-AUC per arm
at L19 with oracle + chance lines; right: map reconstruction R² by layer).

**Verdict (map regimes).**

1. **Does C ≈ A (reparametrization holds)? YES, under all three maps.** C_benign
   0.971 / C_merged 0.951 / C_indomain 0.945 vs A 0.974 at L19 — essentially
   identical. A probe trained AND tested in the mapped space recovers the same
   signal as on raw v_C, confirming that M·v_C is a linear reparametrization of
   v_C. Notably C_benign ≈ A *even though M_benign's reconstruction R² is
   negative* — because C never depends on the map reconstructing anything
   faithfully, only on the linear structure surviving the transform.
2. **Does D beat B's fixed-direction 0.32, and approach the oracle E? YES to
   both.** D_benign 0.871, D_indomain 0.931, D_merged 0.928 vs B's 0.28
   (benign) / 0.43 (in-domain) / 0.37 (merged) — a fitted answer-space probe fed
   the MAPPED answer massively outperforms a fixed direction through the same
   map. D_indomain 0.931 lands within 0.043 of the oracle E (0.974); D_benign
   0.871 within 0.103. So "use an answer-space classifier through the map"
   nearly recovers the answer-space oracle — the map preserves the
   discriminative comply-vs-refuse answer direction even where it cannot
   reconstruct the answer pointwise. **D still never beats A**, at any map.
3. **Does the in-domain map close the gap — is the map's failure a
   training-distribution problem? YES, decisively.** M_benign's reconstruction
   R² is NEGATIVE at every layer (mean −0.558 — it predicts jailbreak answers
   worse than their own mean): a map fit on benign answers does not span the
   harmful-compliance answer subspace. M_indomain is positive (mean +0.488), and
   D_indomain (0.931) closes ~40% of D_benign's residual gap to the oracle
   (dramatically at the layers where M_benign is worst: L07 D 0.08→0.58, L27 D
   0.58→0.79). The mechanism is explicit: the benign map's failure IS a
   training-distribution problem, and fixing the distribution fixes both R² and
   D together.
4. **Does merging benign + in-domain pairs help? NO — merging DILUTES.**
   M_merged tracks M_indomain closely but sits BELOW it on reconstruction at
   every layer (mean R² +0.363 vs +0.488), with the deficit concentrated at the
   deep layers where the two regimes disagree most (L23 +0.216 vs +0.463; L27
   +0.180 vs +0.480). On the detection arms merged and in-domain are
   indistinguishable at the best layer (C 0.951 vs 0.945; D 0.928 vs 0.931).
   Reading: the 3,000 benign rows are 69% of the merged training set and pull the
   map toward the benign answer subspace, so merging buys nothing over the 1,377
   in-domain pairs alone and costs reconstruction fidelity where the subspaces
   diverge. If in-domain pairs are available, use them alone.

**Headline for this extension: A ≈ E.** The cheap context-side probe (0.974)
*already equals* the answer-space oracle (0.974) — the last-prompt-token residual
carries as much always-comply signal as the real answer does, so the map is not
needed at all for detection. The map arms matter as mechanism, not as a better
detector: they show the answer-space signal is linearly recoverable from v_C
(C ≈ A), that an answer-space classifier survives the map (D ≫ B) when the map is
trained in-domain, and that a benign-only map structurally cannot reconstruct
harmful-compliance answers (negative R²).

---

## (0c) Label-efficiency sweep — does the map buy labels?

§0b shows the map never beats the plain context probe when labels are plentiful.
The remaining hope for the map is **label efficiency**: the map is fit from
(v_C, v_A) pairs, which need **no labels** at all, so an answer-space probe fed
mapped answers might need fewer *labelled* contexts than a context probe. This
section tests that directly.

**Design.** At each budget **N ∈ {10, 20, 40, 80, 160, 320}** labelled contexts,
train three arms and score all of them on the SAME held-out eval set (the §0b
eval set: 98 always-comply positives + 1,862 failed-compliance negatives
subsampled to a 5% base rate, n=1,960):

- **A_N** — probe trained on the real `v_C` of the N labelled contexts.
- **D_N (in-domain)** — probe trained on the real `v_A` of the N labelled
  contexts, applied at test to `M_indomain · v_C`.
- **D_N (merged)** — same, through `M_merged`.

Labels are drawn from the §0b **map reserve** (52 pos + 1,325 neg) — the same
contexts the map was fit on. That is not label leakage: fitting the map uses only
the unlabelled (v_C, v_A) pair, and the eval set is disjoint from the reserve, so
map-fit / probe-train / test never share a context. Each budget draws
`max(2, round(0.10·N))` positives (so N=10 → 2 pos/8 neg, N=320 → 32 pos/288 neg)
and is repeated over **5 independent draws**; tables report mean ± SD across
draws. Layers: **L19** (the §0b best layer, pre-specified) and **L27**
(robustness).

**LABEL-COST ASYMMETRY — read the curves with this in mind.** A_N needs N
labelled contexts. D_N needs the same N labels **plus a generation pass per
labelled context** to obtain its real answer activation `v_A`. So D is strictly
more expensive per label than A: for the map to be worth using, D_N must beat
A_N by a margin large enough to pay for the generations — a tie is a loss.

**PR-AUC vs N (mean ± SD over 5 draws), layer 19:**

| N | A (probe on v_C) | D, in-domain map | D, merged map |
|---|---|---|---|
| 10 | **0.834 ± 0.064** | 0.619 ± 0.134 | 0.620 ± 0.150 |
| 20 | **0.825 ± 0.101** | 0.643 ± 0.113 | 0.641 ± 0.120 |
| 40 | **0.905 ± 0.026** | 0.779 ± 0.044 | 0.787 ± 0.035 |
| 80 | **0.925 ± 0.026** | 0.855 ± 0.007 | 0.864 ± 0.017 |
| 160 | **0.959 ± 0.013** | 0.870 ± 0.035 | 0.882 ± 0.029 |
| 320 | **0.965 ± 0.005** | 0.891 ± 0.005 | 0.909 ± 0.008 |
| all 1,377 | **0.978** | 0.938 | 0.942 |

**Layer 27:**

| N | A (probe on v_C) | D, in-domain map | D, merged map |
|---|---|---|---|
| 10 | **0.650 ± 0.222** | 0.308 ± 0.104 | 0.334 ± 0.114 |
| 20 | **0.719 ± 0.076** | 0.421 ± 0.112 | 0.387 ± 0.103 |
| 40 | **0.822 ± 0.057** | 0.494 ± 0.108 | 0.477 ± 0.121 |
| 80 | **0.907 ± 0.019** | 0.588 ± 0.033 | 0.557 ± 0.056 |
| 160 | **0.903 ± 0.009** | 0.662 ± 0.050 | 0.646 ± 0.065 |
| 320 | **0.929 ± 0.018** | 0.733 ± 0.033 | 0.693 ± 0.029 |
| all 1,377 | **0.960** | 0.824 | 0.791 |

**N to reach PR-AUC 0.80** (linear interpolation on the mean curve; lower is
better):

| arm | L19 | L27 |
|---|---|---|
| A — probe on v_C | **≤ 10** | **~36** |
| D — in-domain map | ~51 | never (0.733 at N=320) |
| D — merged map | ~47 | never (0.693 at N=320) |

Figure: `docs/scratch/jailbreak_mining_pilot_label_efficiency.png` (PR-AUC vs N,
one panel per layer, error bars = SD over draws, with the all-labels A reference,
the answer-space oracle E, and chance).

**Verdict (label efficiency). The map does NOT buy labels — A dominates D at
EVERY budget, and the gap is WIDEST where labels are scarcest.** At L19 with only
**10 labelled contexts** (2 positives), the plain context probe already reaches
PR-AUC 0.834 — above the 0.80 bar and ~17× the 0.05 base rate — while D needs
~5× more labels (≈47–51) to get there. At L27 A reaches 0.80 by ~36 labels and D
never reaches it inside the swept range at all. The ordering never inverts at any
N, at either layer, and the margin at N=10 (0.834 vs 0.619, ≈0.21 PR-AUC) is far
outside the draw-to-draw spread. Since D *additionally* costs a generation pass
per label, its label-cost-adjusted position is strictly worse than these curves
show. The map's one remaining hypothetical advantage does not materialise.

One honest caveat in the other direction: A's L19 curve is essentially FLAT from
N=10 to N=20 (0.834 → 0.825, within spread) and only 0.14 below its own
all-1,377-label ceiling at N=10 — this task is simply easy for a context probe,
so the sweep has limited dynamic range at its low end to discriminate *among*
strong arms. That does not weaken the A-vs-D conclusion (the gap is large and
consistent), but it does mean "A reaches 0.80 at ≤10 labels" is a bound set by
the coarsest budget on the grid, not a measured threshold.

---

## (0d) Cross-family transfer — does either arm survive a family shift?

The last place the map could win is **generalisation**: a map fit on family X's
(v_C, v_A) pairs might carry family-general answer structure that a
family-X-trained context probe does not.

**Design.** THREE compliance-labelled families, after the toxicchat parse was
fixed (§d gap (i)): `evil_train`, `evil_hh_rlhf`, `evil_toxicchat` — giving
**6 ordered directions**. For each, train on the FULL source family and test on
the target family's 5%-base-rate set (all its negatives, positives subsampled to
the cleanest `round(n_neg · 0.05/0.95)`):

| family | always-comply pos avail. | failed-comp neg | as TARGET: n_test (pos, base) |
|---|---|---|---|
| `evil_train` | 391 | 2,721 | 2,864 (143, 0.050) |
| `evil_hh_rlhf` | 173 | 1,066 | 1,122 (56, 0.050) |
| `evil_toxicchat` | 71 | 256 | 269 (**13**, 0.048) — **thin** |

Arms: **A_transfer** (probe on v_C), **D_transfer** through `M_indomain` (fit on
the SOURCE family's own pairs) and through `M_merged` (benign 3,000 + source
family), and **E_transfer** (probe on real v_A → real v_A — the oracle, which
needs generation on the target family and is not deployable). References:
**A_within** / **E_within** (grouped 5-fold OOF *within* the target family — "you
had in-domain labels") and a random floor. Headline layer **L19**
(pre-specified from §0b); the full 6-layer sweep is in the results JSON and the
best-layer view is the figure's right panel.

**Read the two `→ evil_toxicchat` rows as directional only.** Its 5%-base test
set has just **13 positives**, so its PR-AUC (and its `A_within` reference, which
reaches a degenerate 1.000 at deep layers) is very noisy. The four directions
with `evil_train` or `evil_hh_rlhf` as target (56–143 positives) carry the firm
conclusions.

**PR-AUC at L19 (pre-specified); `n_tr` = source-family train size:**

| direction (n_tr) | A_transfer | D, in-dom | D, merged | E oracle | A_within (ref) |
|---|---|---|---|---|---|
| `evil_train` → `hh_rlhf` (3,112) | **0.894** | 0.810 | 0.884 | 0.836 | 0.947 |
| `evil_train` → `toxicchat` (3,112) *thin* | 0.616 | 0.417 | **0.653** | 0.653 | 0.936 |
| `hh_rlhf` → `evil_train` (1,239) | **0.623** | 0.203 | 0.369 | 0.401 | 0.982 |
| `hh_rlhf` → `toxicchat` (1,239) *thin* | 0.910 | 0.555 | **0.947** | 0.990 | 0.936 |
| `toxicchat` → `evil_train` (327) | **0.637** | 0.451 | 0.264 | 0.196 | 0.982 |
| `toxicchat` → `hh_rlhf` (327) | **0.902** | 0.872 | 0.776 | 0.920 | 0.947 |

**Best layer per arm** (each arm's own best of the 6 layers — the most generous
reading for the map):

| direction | A_transfer | D, in-dom | D, merged | E oracle | A_within |
|---|---|---|---|---|---|
| `evil_train` → `hh_rlhf` | **0.894** | 0.810 | 0.884 | 0.837 | 0.960 |
| `evil_train` → `toxicchat` *thin* | **0.730** | 0.524 | 0.653 | 0.696 | 1.000 |
| `hh_rlhf` → `evil_train` | **0.753** | 0.586 | 0.369 | 0.658 | 0.982 |
| `hh_rlhf` → `toxicchat` *thin* | **1.000** | 0.958 | 0.947 | 0.995 | 1.000 |
| `toxicchat` → `evil_train` | **0.674** | 0.487 | 0.264 | 0.196 | 0.982 |
| `toxicchat` → `hh_rlhf` | **0.926** | 0.872 | 0.776 | 0.920 | 0.960 |

**Map held-out reconstruction R² on the TARGET family** (how well a
source-family-fit map predicts another family's answers), `M_src / M_merged`:

| layer | train→hh | train→tox | hh→train | hh→tox | tox→train | tox→hh |
|---|---|---|---|---|---|---|
| L07 | −0.52 / −0.18 | −0.53 / −0.59 | −0.38 / −0.73 | +0.08 / −0.41 | −0.26 / −0.91 | +0.08 / −0.05 |
| L11 | −0.22 / +0.01 | −0.31 / −0.33 | −0.30 / −0.55 | +0.19 / −0.16 | −0.17 / −0.60 | +0.16 / +0.10 |
| L15 | +0.05 / +0.17 | −0.06 / +0.00 | −0.27 / −0.29 | +0.17 / +0.08 | −0.03 / −0.27 | +0.33 / +0.26 |
| L19 | +0.04 / +0.17 | −0.04 / +0.14 | −0.14 / −0.15 | +0.24 / +0.22 | +0.11 / −0.10 | +0.40 / +0.29 |
| L23 | −0.96 / −0.86 | −0.60 / −0.39 | −0.42 / −0.96 | +0.06 / −0.19 | +0.04 / −0.79 | +0.25 / −0.51 |
| L27 | −0.73 / −0.67 | −0.43 / −0.19 | −0.42 / −1.10 | +0.19 / −0.02 | +0.01 / −0.99 | +0.35 / −0.44 |

Figure: `docs/scratch/jailbreak_mining_pilot_transfer.png` (grouped bars per
direction at L19 and at each arm's best layer, with the within-family A reference
and chance).

**Verdict (transfer).**

1. **The map does NOT transfer better than the raw probe.** At each arm's best
   layer, **A_transfer wins all 6 of 6 directions**. At the pre-specified L19 it
   wins **4 of 6**, and the two exceptions are *both* the 13-positive
   `→ toxicchat` rows (merged map 0.653 vs A 0.616; 0.947 vs 0.910) — i.e. A wins
   **4 of 4** directions with a non-thin target. The map is not a family-general
   representation; if anything it is *more* family-specific than the context
   probe, and it degrades hardest exactly where transfer is hardest
   (`hh_rlhf → evil_train`: A 0.623 vs D 0.203).
2. **Transfer degrades every arm, and the degradation tracks source-family
   SIZE more than family identity.** Against its own within-family reference, A
   drops 0.053 from the largest source (`evil_train`, n=3,112 → hh_rlhf: 0.894 vs
   0.947) and 0.359 from the middle one (`hh_rlhf`, n=1,239 → evil_train: 0.623
   vs 0.982). But the smallest source is the informative case: `toxicchat`
   (n=327, 71 positives) still reaches 0.902 → hh_rlhf and 0.637 → evil_train —
   i.e. **the same target that `hh_rlhf` transferred to at 0.623 is reached at
   0.637 by a source with a quarter of its training data.** So target difficulty
   dominates: `evil_train` is simply a hard target for every source (0.623 /
   0.637), while `hh_rlhf` is an easy one (0.894 / 0.902). Train-size and
   family-similarity are still not fully separable with 3 families, but the
   3-family grid shows the earlier 2-family "it's the train-size gap" reading was
   too simple.
3. **The map's reconstruction does not transfer.** R² of a source-family-fit map
   on another family is NEGATIVE in **45 of 72** (layer × direction × map) cells
   and never exceeds **+0.403** — versus +0.625 for the within-pool in-domain map
   in §0b. A map fit on one jailbreak family predicts another family's answers
   worse than their own mean, in the same way a benign-fit map fails on jailbreak
   answers (§0b). So §0b's "an in-domain map fixes it" is narrower than it looked:
   *in-domain* must mean **in-family**, not merely in-jailbreak. Note also that
   `M_merged` is worse than `M_src` in most cross-family cells — the same dilution
   §0b item 4 found.
4. **A_transfer beats the answer-space ORACLE in 3 of 6 directions**, and by a
   wide margin from the smallest source (`toxicchat → evil_train`: A 0.637 vs
   E_transfer 0.196; `hh_rlhf → evil_train`: 0.623 vs 0.401). Under a family
   shift the real answer activation is often a *worse* transfer feature than the
   context activation — the answer representation is more family-idiosyncratic.
   This is the sharpest statement against the map: the thing the map is trying to
   predict is itself the less transferable signal.

**POWER CAVEAT.** Three families give 6 ordered directions — better than the 2
families (one pair) this arm started with, but still a very small grid, and two
of the six have only 13 test positives. `n_train` varies ~10× across sources
(3,112 / 1,239 / 327) and is not orthogonal to family identity, so train-size and
family-similarity remain partly confounded. Treat the direction-level numbers as
descriptive; the load-bearing claim is the consistent ORDERING (A ≥ D in every
non-thin direction, at both the pre-specified and the best layer), which does not
depend on any single cell.

---

## (a) Pools, base rates, and exact positive/negative definitions [original trait-DV pass]

*The sections below (§a–d) are the first pass, on the graded **trait** DV. §0
above is the compliance-DV rerun that supersedes the headline; §a–d are retained
for the trait-vs-compliance comparison and the method detail.*

Positives are **evil-lane OOD contexts** from #1739 (three in-the-wild jailbreak
families: `mhj` = multi-turn human jailbreaks, `pair` = PAIR, `tomgibbs` =
Tom-Gibbs multi-turn), with `v_C` = `context_end` (residual at the last prompt
token) captured on Qwen2.5-7B-Instruct. Each context carries a graded 0–100
**trait DV** (Sonnet-4.5 judge, 5 rollouts × 3 draws, mean-aggregated) measuring
how strongly the elicited response expresses the misaligned/"evil" trait.

- **Positive = trait DV ≥ 68** (top ~5% of the 2,950 labelled evil-OOD contexts):
  **151 positives**. Of these, **94 elicit reliably** (min trait score across all
  5 rollouts ≥ 50) and 115 have min ≥ 30 — i.e. the label is close to
  "always-jailbreaks", not a lucky single draw.
- **Negatives, two variants:**
  - **Benign** — #1092 `realistic_crossing` WildChat/LMSYS multi-turn contexts
    (5,000-context subsample), same model / revision / pooling / layer grid.
  - **Failed-jailbreak (hard negatives)** — evil-OOD contexts with trait DV ≤ 5
    (1,954): same adversarial prompt families as the positives, but the model did
    **not** produce misaligned output. This isolates "will it *succeed*" from
    "is it an adversarial prompt".

| Pool | Positives | Negatives | n | base rate |
|---|---|---|---|---|
| `needle_benign_5pct` | 151 | 2,869 benign | 3,020 | 0.050 |
| `balanced_benign` | 151 | 151 benign | 302 | 0.500 |
| `hardneg_lowtrait_5pct` | 151 | 1,954 failed-jailbreak | 2,105 | 0.072 |

Consistency verified by construction: positives and negatives share model
`Qwen/Qwen2.5-7B-Instruct` @ rev `a09a3545`, the same 28-layer grid, the same
fp16 `context_end` (last-prompt-token) pooling — the #1739 evil store was
explicitly captured to match the #1092 representation space.

## (b) Per-arm results (best layer; chance = base rate)

Layer sweep {7, 11, 15, 19, 23, 27}. Probe = L2-logistic (C=0.01,
class-balanced) on standardized `v_C`, grouped 5-fold, **out-of-fold**
predictions (no context/prefix leakage; groups = jailbreak group / benign
conversation id). Direction arms = `⟨v_C, r̂_B⟩` with the #658 `r_B`
harmful-compliance and refusal diff-of-means directions (no fit). Metrics are
ROC-AUC, PR-AUC (average precision — the honest headline when positives are
rare), hit@5% (fraction of the top-5%-ranked that are true positives), and
evals-to-find-20 (contexts you'd review, ranked by score, to surface 20 true
positives; ideal = 20).

**Pool `needle_benign_5pct` (positives vs benign, base 0.050):**

| arm | best layer | ROC-AUC | PR-AUC | hit@5% | evals→20 |
|---|---|---|---|---|---|
| linear probe (fitted) | L11 | **1.000** | **1.000** | **1.000** | **20** |
| r_B harmful-compliance | L15 | 0.918 | 0.269 | 0.21 | 123 |
| r_B refusal | L15 | — | 0.302 | — | 103 |
| random | — | 0.50 | 0.055 | 0.05 | 350 |

**Pool `hardneg_lowtrait_5pct` (positives vs *failed* jailbreaks, base 0.072) —
the leakage-robust, scientifically meaningful contrast:**

| arm | best layer | ROC-AUC | PR-AUC | hit@5% | evals→20 |
|---|---|---|---|---|---|
| linear probe (fitted) | L27 | **0.964** | **0.811** | **0.838** | **20** |
| r_B harmful-compliance | L23 | 0.837 | 0.223 | 0.105 | 132 |
| r_B refusal | L23 | 0.837 | 0.220 | — | 128 |
| random | — | 0.50 | 0.071 | 0.07 | 308 |

(Balanced pool: probe ROC/PR = 1.000 at every layer; the r_B directions reach
ROC 0.94 / PR 0.92 at L15 — separating *any* jailbreak prompt from benign is
easy, so the balanced number is not diagnostic.)

Figure (PR-AUC by layer, per arm; left = vs benign, right = vs failed-jailbreak):
`docs/scratch/jailbreak_mining_pilot_pr_by_layer.png`

- **Probe** dominates everywhere. Against benign it is a perfect ranker (PR 1.0,
  ~20× the 0.05 base rate). Against *failed jailbreaks of the same families* it
  still hits PR 0.81 (~11× base) and hit@5% 0.84 — top-5%-ranked contexts are 84%
  true reliable-jailbreakers vs a 7% prior.
- **Direction projections** are weak: harmful-compliance/refusal give decent ROC
  (0.84–0.92) but poor PR (≤ 0.30) — they rank badly at the very top where mining
  needs precision. They peak mid-to-late (L15–L23), consistent with refusal/harm
  representations living there.
- **Random** sits exactly at the base rate on every metric (sanity check passes).

## (c) Verdict

1. **Is there a cheap context-side signal?** Yes, strongly, for a *fitted* linear
   probe on `v_C`. Detecting "context that reliably elicits misaligned output"
   from the last-prompt-token residual — no generation, no judge — works at ROC
   ~0.96 / PR ~0.81 in the hard 7%-base-rate needle regime, and trivially vs
   benign. Mining lift is large: to surface 20 reliable-jailbreak contexts you
   review ~20 (probe) vs ~130 (direction) vs ~310 (random).
2. **Does the map / refusal direction beat a plain probe?** No. The fixed
   refusal/harm-direction projection is far below the probe (PR 0.22 vs 0.81).
   The map-then-project arm (arm 3) was not run (its 720 MB pre-fit artifact
   would bust the download budget), but note the algebra: map-then-project onto a
   fixed `r_B` is itself a *fixed linear direction* in `v_C`-space
   (`v_C·(Wᵀr_B)`), so it is upper-bounded by the fitted linear probe and can at
   best sit between the raw direction and the probe. The evidence says the useful
   signal is a *learned* linear readout of `v_C`, not any single pre-specified
   direction.
3. **Prefix-vs-query hint.** The near-perfect benign separation almost certainly
   reflects easy *prompt-domain* separation (adversarial multi-turn jailbreak
   structure vs benign chat) rather than a subtle "will-succeed" signal. The
   hard-negative contrast controls for that — same adversarial families on both
   sides — and the probe *still* wins big (PR 0.81), so there is a genuine
   context-geometry component predicting elicitation *success*, over and above
   "is this an adversarial prompt". The signal strengthens with depth in the
   hard-negative pool (PR 0.66 → 0.81, L7 → L27).

## (d) Data-availability gaps (and the minimal fix) — RESOLVED in §0

- **The brief asked for the COMPLIANCE DV; the first pass used the TRAIT DV.** The
  per-context graded compliance DV (StrongREJECT-style) exists only for the
  main-labeling rungs (`evil_train`, `evil_hh_rlhf`, `evil_toxicchat`, ~10.6k
  contexts), and their `v_C` activations were packed in a single 32 GB tar
  (`issue1739_ctxmap/capture_store/evil_labeling/evil_labeling.tar`) with no
  random access. **RESOLVED (§0):** the tar was stream-reduced (never
  materialized) into a 459 MB `context_end`-only npz, so the compliance-DV pilot
  ran without GPU recapture and without busting the disk budget — the gap was
  packaging, and the packaging is done.
- **Trait was a weak proxy for compliance** (ρ 0.215 / 0.073 / 0.196 on
  evil_train / hh_rlhf / toxicchat). §0 removes the proxy: on the true compliance
  DV the probe headline is *stronger*, not weaker (hard-negative PR 0.973 vs the
  trait pass's 0.811), so the trait-DV conclusion was directionally correct and
  conservative.
- **Gap (i) — `evil_toxicchat` — RESOLVED, and the earlier diagnosis was WRONG.**
  This section previously stated that toxicchat "yielded 0 parsed compliance
  scores — its raw-judge shard has a different filename/schema". That attribution
  does not survive checking. The family's judge output IS present on the HF data
  repo at `issue1739_ctxmap/evil_ood_spread/compliance_full/evil_toxicchat/` as a
  **single unsharded `judge_raw_compliance_full.json`** (5,827,718 bytes; the
  other two families are sharded `judge_raw_compliance_full.shardNN.jsonl`) — and
  the reducer's glob **already covers that filename**. The real cause: the
  directory had never been staged locally, so the reducer globbed an **absent**
  directory and returned 0 contexts. Staging the one file was the entire fix; no
  schema change was needed. Reduced with the SAME `reduce_rung` the rerun used:
  13,420 judge entries → 3,351 rollout-items → **671 contexts, 71 always-comply
  (mean & min ≥ 90) and 256 failed-compliance (mean ≤ 5)**. This unblocked the
  3-family transfer sweep in §0d. The reduced DV is written to a **separate**
  `compliance_percontext_toxicchat_probe.json`, deliberately NOT merged into the
  shared `compliance_percontext.json`: §0b/§0c pool all rungs and select the 150
  cleanest positives across them, so folding in a third family would change the
  pooled positive set and invalidate their committed tables. (ii) `group_key` was not retained in the stream-reduce, but the
  sibling store shows it is ~1:1 with `context_id` (2954/2954 distinct), so
  per-context grouped OOF is the honest leakage control — there is no coarser
  template grouping to exploit. (iii) `r_B` (#658) was extracted for a different
  pooling, so the direction arms remain a rough (orientation-set) baseline; their
  weak-to-mid PR is the load-bearing point and is robust to orientation.

**Other caveats.** The probe runs in the `n_train < d` regime (~2.4k train rows
vs d=3584); this is why strong L2 (C=0.01) + grouped OOF is used, and the
benign-pool ROC of 1.000 is genuine easy domain separability, not overfitting —
confirmed by the hard-negative pool (same estimator) landing at 0.81, not 1.0.
The r_B directions were extracted for a different pooling in #658, so their
projection is a rough baseline (orientation set to the pool labels); their weak
PR is the load-bearing takeaway, robust to orientation.

---

**Repro:** map-regime arms (§0b) — `scripts/issue1739_jbmine_stream_evil_answer.py`
(stream-reduce t1 = v_A → `evil_answer_t1.npz`),
`scripts/issue1739_jbmine_map_arms.py` (arms A/B/C/D/E, all THREE map regimes
incl. M_merged, map R² → `map_arms_results.json`),
`scripts/issue1739_jbmine_map_arms_plot.py` →
`docs/scratch/jailbreak_mining_pilot_map_arms.png`. Label efficiency (§0c) —
`scripts/issue1739_jbmine_label_efficiency.py` (budgets {10..320} × 5 draws ×
{A, D-indomain, D-merged} at L19/L27 → `label_efficiency_results.json`).
Cross-family transfer (§0d) — `scripts/issue1739_jbmine_transfer.py`
(6 ordered directions over 3 families × 6 layers → `transfer_results.json`;
`--layers`/`--out-suffix` support the 1-layer sizing pilot, whose output is
`transfer_results_pilot1layer.json`). Both §0c/§0d figures —
`scripts/issue1739_jbmine_labeleff_transfer_plot.py` →
`docs/scratch/jailbreak_mining_pilot_label_efficiency.png` +
`..._transfer.png`. Toxicchat parse fix (§d gap (i)) —
`scripts/issue1739_jbmine_toxicchat_probe.py` (stages the single unsharded
`judge_raw_compliance_full.json`, reduces via the SAME `reduce_rung` →
`compliance_percontext_toxicchat_probe.json`, deliberately separate from the
shared DV json). Compliance rerun (§0) —
`scripts/issue1739_jbmine_stream_evil.py` (stream-reduce the 32 GB tar →
`evil_compliance_ctxend.npz`), `scripts/issue1739_jbmine_compliance_reduce.py`
(per-context compliance DV → `compliance_percontext.json`),
`scripts/issue1739_jbmine_compliance_pilot.py` (4 arms →
`compliance_pilot_results.json`), `scripts/issue1739_jbmine_compliance_plot.py` →
`docs/scratch/jailbreak_mining_pilot_compliance_pr_by_layer.png`. Trait pass
(§a–d) — `scripts/issue1739_jbmine_pilot.py`,
`docs/scratch/jailbreak_mining_pilot_pr_by_layer.png`, `pilot_results.json`.
Data slices staged under `/mnt/eps-data/$USER/issue1739_jbmine/`: answer-span
`t1` (6 layers, 10,666 contexts) in `evil_answer_t1.npz`; compliance
`context_end` (6 layers, 10,666 contexts) stream-reduced from
`issue1739_ctxmap/capture_store/evil_labeling/evil_labeling.tar`; per-context
compliance DV from `issue1739_ctxmap/evil_ood_spread/compliance_full/*.jsonl`;
trait-pass evil-OOD `context_end` from `issue1739_ctxmap/evil_ood_full/store`;
benign `context_end` + `t1` (6 layers) from
`issue1092_realistic_crossing/.../cell_inst_own` @ rev `e5901706`; `r_B` from
`issue658_theory_assumptions/store/r_b.pt`. Model `Qwen/Qwen2.5-7B-Instruct` @
rev `a09a3545`. Judge/label provenance: #1739 trait/compliance DVs, Sonnet-4.5.
