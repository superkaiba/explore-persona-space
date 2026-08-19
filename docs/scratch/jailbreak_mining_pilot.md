# Jailbreak-mining pilot: can a cheap context-side signal find "always-jailbreak" contexts?

Exploratory 0-GPU pilot (team-lead brief). **Question:** can a cheap signal
computed from a white-box model's activations on a *context* (last-prompt-token
residual `v_C`, no generation + judging) separate contexts that reliably elicit
misaligned output on the jailbreak lane from ordinary/benign contexts, in a
needle-in-haystack regime? And does a refusal/harm-direction projection add
anything over a plain linear probe?

**One-line answer:** Yes for a *fitted linear probe* on `v_C` (near-perfect vs
benign; ROC 0.96 / PR 0.81 at a 7% base rate even against *failed* jailbreak
prompts). A fixed refusal/harm-direction projection is much weaker (PR ≤ 0.27),
so the probe — not the direction — carries the signal. **Important caveat:** the
positive label here is the graded **trait** DV, a weak proxy (ρ 0.07–0.22) for
the **compliance** DV the brief actually asked for; the compliance-DV activations
are locked in a 32 GB non-random-access tar (see §Data gaps).

All model reads are pre-existing artifacts; nothing was generated on GPU. Work
was on numeric labels + activation tensors only (no jailbreak/rollout text read).

---

## (a) Pools, base rates, and exact positive/negative definitions

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

## (d) Data-availability gaps (and the minimal fix)

- **The brief asked for the COMPLIANCE DV; this pilot used the TRAIT DV.** The
  per-context graded compliance DV (StrongREJECT-style) exists only for the
  main-labeling rungs (`evil_train`, `evil_hh_rlhf`, `evil_toxicchat`, ~10.6k
  contexts), and **their `v_C` activations are packed in a single 32 GB tar**
  (`issue1739_ctxmap/capture_store/evil_labeling/evil_labeling.tar`) with **no
  random access** — the pipeline transfers the whole tar to read any slice, which
  busts the pilot's <2 GB budget. The only npy-sharded (sliceable) evil store is
  `evil_ood_full`, which carries the trait DV, so the pilot ran on trait.
- **Trait is a weak proxy for compliance.** The stored per-item rank correlation
  ρ(compliance, trait) on the main rungs is only **0.215 / 0.073 / 0.196**
  (evil_train / hh_rlhf / toxicchat). So a true compliance-DV pilot could give
  materially different numbers — this trait-DV result should be read as
  "context geometry predicts *misaligned-output* elicitation", not yet as
  "predicts *harmful compliance*".
- **Minimal fix (0-GPU, no recapture — the activations already exist):** repack
  `evil_labeling.tar` into per-layer npy shards (the layout `evil_ood_full/store`
  already uses), a pure CPU stream-and-write job. Then the exact compliance-DV
  pilot — top ~150 by the per-context compliance mean from
  `evil_ood_spread/compliance_full/*.jsonl` — reruns cheaply (<2 GB, minutes).
  No GPU recapture is needed; the gap is packaging, not missing data.

**Other caveats.** The probe runs in the `n_train < d` regime (~2.4k train rows
vs d=3584); this is why strong L2 (C=0.01) + grouped OOF is used, and the
benign-pool ROC of 1.000 is genuine easy domain separability, not overfitting —
confirmed by the hard-negative pool (same estimator) landing at 0.81, not 1.0.
The r_B directions were extracted for a different pooling in #658, so their
projection is a rough baseline (orientation set to the pool labels); their weak
PR is the load-bearing takeaway, robust to orientation.

---

**Repro:** `scripts/issue1739_jbmine_pilot.py` (analysis),
`docs/scratch/jailbreak_mining_pilot_pr_by_layer.png` (figure); results JSON at
`/mnt/eps-data/$USER/issue1739_jbmine/pilot_results.json`. Data slices staged
under `/mnt/eps-data/$USER/issue1739_jbmine/` (1.5 GB): evil-OOD `context_end`
(6 layers) from `issue1739_ctxmap/evil_ood_full/store`, benign `context_end` from
`issue1092_realistic_crossing/.../cell_inst_own` @ rev `e5901706`, `r_B` from
`issue658_theory_assumptions/store/r_b.pt`. Judge/label provenance: #1739
trait/compliance DVs, Sonnet-4.5.
