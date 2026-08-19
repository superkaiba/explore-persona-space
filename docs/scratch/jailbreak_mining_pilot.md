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

C and D run under **two map regimes**: **M_benign** = ridge context_end→v_A on
3,000 disjoint benign WildChat/LMSYS rows; **M_indomain** = ridge on a
grouped-disjoint, label-stratified 35% reserve of the jailbreak contexts' own
(v_C, v_A) pairs.

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
| C — probe on M·v_C, benign | L19 | 0.998 | 0.971 | 0.908 | 20 |
| C — probe on M·v_C, in-domain | L19 | 0.993 | 0.945 | 0.898 | 20 |
| D — v_A-probe through M, benign | L19 | 0.985 | 0.871 | 0.796 | 20 |
| D — v_A-probe through M, in-domain | L19 | 0.992 | 0.931 | 0.837 | 20 |
| **E — probe on real v_A (ORACLE)** | L19 | 0.998 | **0.974** | 0.918 | 20 |
| (ref) raw fixed r_B refusal dir on v_C | L23 | 0.921 | 0.697 | 0.643 | 21 |
| random | — | 0.494 | 0.064 | 0.071 | 387 |

**Map held-out reconstruction R² of v_A on the jailbreak eval contexts:**

| layer | M_benign | M_indomain |
|---|---|---|
| L07 | −0.840 | +0.329 |
| L11 | −0.516 | +0.454 |
| L15 | −0.160 | +0.579 |
| L19 | −0.119 | +0.625 |
| L23 | −0.876 | +0.463 |
| L27 | −0.835 | +0.480 |
| **mean** | **−0.558** | **+0.488** |

Figure: `docs/scratch/jailbreak_mining_pilot_map_arms.png` (left: PR-AUC per arm
at L19 with oracle + chance lines; right: map reconstruction R² by layer).

**Verdict (map regimes).**

1. **Does C ≈ A (reparametrization holds)? YES.** C_benign 0.971 vs A 0.974 at
   L19 — essentially identical; C_indomain 0.945, close behind. A probe trained
   AND tested in the mapped space recovers the same signal as on raw v_C,
   confirming that M·v_C is a linear reparametrization of v_C. Notably C_benign ≈
   A *even though M_benign's reconstruction R² is negative* — because C never
   depends on the map reconstructing anything faithfully, only on the linear
   structure surviving the transform.
2. **Does D beat B's fixed-direction 0.32, and approach the oracle E? YES to
   both.** D_benign 0.871 and D_indomain 0.931 vs B's 0.28 (benign) / 0.43
   (in-domain) — a fitted answer-space probe fed the MAPPED answer massively
   outperforms a fixed direction through the same map. D_indomain 0.931 lands
   within 0.043 of the oracle E (0.974); D_benign 0.871 within 0.103. So "use an
   answer-space classifier through the map" nearly recovers the answer-space
   oracle — the map preserves the discriminative comply-vs-refuse answer
   direction even where it cannot reconstruct the answer pointwise.
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

**Headline for this extension: A ≈ E.** The cheap context-side probe (0.974)
*already equals* the answer-space oracle (0.974) — the last-prompt-token residual
carries as much always-comply signal as the real answer does, so the map is not
needed at all for detection. The map arms matter as mechanism, not as a better
detector: they show the answer-space signal is linearly recoverable from v_C
(C ≈ A), that an answer-space classifier survives the map (D ≫ B) when the map is
trained in-domain, and that a benign-only map structurally cannot reconstruct
harmful-compliance answers (negative R²).

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
- **Remaining gaps (compliance rerun):** (i) `evil_toxicchat` yielded 0 parsed
  compliance scores — its raw-judge shard has a different filename/schema
  (`judge_raw_compliance_full.json` sits under it but produced no per-context
  entries via the per_persona/all_scores extractor), so the positives/negatives
  come from `evil_train` + `evil_hh_rlhf` only; adding toxicchat would broaden
  family coverage. (ii) `group_key` was not retained in the stream-reduce, but the
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
`scripts/issue1739_jbmine_map_arms.py` (arms A/B/C/D/E, both map regimes, map R²
→ `map_arms_results.json`), `scripts/issue1739_jbmine_map_arms_plot.py` →
`docs/scratch/jailbreak_mining_pilot_map_arms.png`. Compliance rerun (§0) —
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
