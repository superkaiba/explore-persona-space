---
title: The prefix/bare mirror image in per-context map error survives the SAE feature-space
  re-basis (HIGH confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-31T19:59:47Z'
has_clean_result: true
parent_id: 1738
origin_prompt: 'run all these: ... 2. Per-context SAE-space error read. The one measurement
  that would settle the bare-vs-prefix mirror-image question. Per-feature agreement
  doesn''t test it'
workflow: v1
goal: Settle whether the bare-query and prefix arms are **mirror images** in SAE feature
  space, as they demonstrably are in the dense per-context taxonomy.
relates_to:
- spec-context-as-vector
---
# The prefix/bare mirror image in per-context map error survives the SAE feature-space re-basis (HIGH confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1946.md](https://github.com/superkaiba/explore-persona-space/blob/7ce7f9d3c7baf325e294c28c714967b4dc999e41/docs/methodology/issue_1946.md) · [gist](https://gist.github.com/superkaiba/626cdfd393c68c9b223e46fd2b6dc37f)

## Takeaways

- The prefix−bare category difference pattern correlates 0.855 across SAE and dense space (Spearman; p < 1e-4, 10,000 shuffles; 18 of 20 signs agree); this meets the plan's Reproduced rule.
- Per conversation, the prefix−bare error difference correlates 0.710 across spaces (n = 9,941) — a converging read free of the category-mask overlap caveat; the category-level agreement is compatible with this conversation-level coupling alone.
- Attenuation localizes to the input encoding: dense inputs with SAE-feature targets keep the pattern at 0.946; max-pooled and active-fraction readings hold 0.914 and 0.837.
- Per-arm profiles transfer unevenly: bare 0.797 and full-context 0.759, but the prefix arm's 0.412 (p = 0.056) is below the 22-contrast detection edge; that arm's transfer is an open read.
- Scope: the mirror survives this SAE re-basis (re-scoring in SAE feature coordinates; same conversations, one k = 64 layer-19 SAE); a claim of coordinate-system independence in general would need fresh samples or a different SAE.
- The refusal-category collapse survives the exact encode-then-pool floor construct: bare refusal-answer delta −0.169 net of floor, no refusal contrast significantly positive — still read on 1,988 of 9,941 conversations (one-fifth coverage).

## Goal

- **This experiment in context:** The parent dense per-context taxonomy ([#1738](https://eps.superkaiba.com/tasks/1738)) found that the history-only (prefix) and bare-query maps fail on complementary context populations — a mirror image over judged categories, read in dense residual-stream coordinates — while the same parent's SAE bare-query cell's per-feature agreement read suggested the bare arm behaves like a degraded full-context arm. Those two reads pull in opposite directions, and per-feature agreement cannot test the mirror (it never compares which contexts each arm misses). This experiment settles it by re-scoring the parent's own banked predictions and targets in SAE feature space and running the identical 22-contrast battery, so the SAE and dense numbers quoted here are directly comparable by construction.
- **Broader narrative:** The project asks whether the specification a context establishes is carried as a readable vector (the spec-context-as-vector question), and reads map error over judged context categories to say what history vs the final query carries. If those category-level conclusions changed under a change of reading basis, they would be coordinate-system facts; showing the mirror is basis-stable (at least for this sparse over-complete basis) licenses reading them as facts about the maps' information content.

## Methodology

**Design:** A training-free re-scoring analysis, 12 cells: 3 map arms (history-only/prefix, full-context, bare-query) × 4 reading variants (mean-pooled SAE features — the reading the plan's verdict is scored in, hereafter the verdict space; a dense-input → SAE-feature-target comparator that isolates the target basis; max-pooled; active-fraction). No new fits: the parent's banked (stored, revision-pinned) ridge predictions and held-out targets are re-scored per conversation in each reading, the parent's 22-contrast category battery runs verbatim per reading, and the cross-space statistics fixed in the plan (difference-pattern correlation + sign agreement primary; per-arm correlations and per-context rank correlations secondary) compare each reading to the banked dense pattern. All correlations reported in this body are Spearman rank correlations. The single manipulated variable is the error-reading space. The prefix-based and context-based mapping arms both run (plus bare-query) as paired conditions of one battery. A floor-adjusted robustness pass ran in two rounds: round 1 with approximate floors (quarantined from the verdict), and a follow-up round with the exact encode-then-pool construct — the single manipulated variable being the floor construct — which preserved the collapse and de-quarantined the read; the cross-space verdict rests on the unadjusted battery throughout.

**Training:** **N/A — no model training.** Analysis parameters (each value from the plan's rationale or the named parent artifact):

| Parameter | Value | Source |
|---|---|---|
| Per-context error | nerr = ‖y − ŷ‖² / ‖y − μ_holdout‖² per conversation | parent `_percontext_nerr`, `scripts/issue1738_multiturn_fits.py` |
| Reading spaces | mean-pooled SAE (verdict); dense-input → SAE-target; max-pooled; active-fraction | plan §4–§5 |
| SAE feature set | 16,384 answer-side features (active on ≥ 878 train rows), asserted equal to banked `feat_ids` | banked `sae_fits.json` restriction |
| Category battery | 22 pre-enumerated contrasts; 10,000 bootstrap draws; 10,000 permutations; BH q = 0.05; seed 1738 | `scripts/issue1738_characterize.py`, verbatim parent battery |
| Cross-space statistics | difference-pattern Spearman + sign agreement on the 20-contrast dense-significant union + per-arm Spearman + per-context Spearman; 10,000 shuffles, seed 1946 | plan §3 |
| Verdict rule | Reproduced iff p < 0.05 AND correlation > 0 AND signs ≥ 15 of 20 | plan §3 (15/20 = binomial two-sided p ≈ 0.041) |
| Identity tolerance | \|recomputed − banked R²\| < 5e-3 per cell, 12 cells | plan §11 (fp16 eps, ~10× below banked CI half-widths) |
| Floors (round 1, robustness) | SAE encoding of the mean answer state; K = 4 draws; 1,988 of 9,941 conversations | parent k-resample shard; approximate construct |
| Floors (round 2, exact) | per-token teacher-forced layer-19 capture of each of the K = 4 banked resampled answers, SAE-encoded per token then pooled (mean / max / active-fraction); same 1,988 conversations; collapse preserved iff adjusted bare refusal-answer delta ≤ 0 and no refusal-family contrast significantly positive | plan follow-up amendment §1; `scripts/issue1946_exact_floors.py` |
| SAE | BatchTopK, k = 64, layer 19, dictionary 131,072 | `andyrdt/saes-qwen2.5-7b-instruct` @ `c37e53c4bb07` |
| Holdout | 9,941 conversations, split sha-asserted vs banked | parent `sae_fits.json` |
| Data revision | `05cb982b0d3f9a21b5735d196a0afdc8175590e5` | HF data-repo pin at plan time |

**Evaluation:** Three derived quantities. (1) Per-context SAE-space error: how badly an arm's map misses conversation i's answer state, read over the 16,384 restricted answer-side SAE features (normalized squared error against the holdout-mean baseline). (2) Per-category delta: mean error inside a judged category minus the rest, with bootstrap CI, permutation p, and BH correction within each arm's 22-contrast family. (3) Cross-space agreement: the statistics in the table above. No judged behavior DV exists here and zero judge calls were made; the judged context labels (language / topic / format / refusal fields) are fixed covariates banked from the parent's Sonnet-4.5 labeling instrument (judged SAE feature labels, frozen by the labeling-instrument freeze, are not consumed). The 22 category masks overlap — one conversation belongs to several contrasts — so contrast-level permutation p and the sign-agreement threshold are mildly anti-conservative; the per-context correlation (each conversation used once) is the corroborating read free of that dependence.

**Data extraction:** All inputs are banked artifacts, re-scored deterministically. The parent captured teacher-forced hidden states at layer 19 of Qwen-2.5-7B-Instruct over 99,774 real multi-turn conversations (LMSYS-Chat-1M + WildChat sampling), pooled each answer's tokens (mean / max / active-fraction), encoded states with the public k = 64 layer-19 BatchTopK SAE, and fit ridge maps from context summaries (history-only, full-context, bare-query, plus dense-input comparator cells) to the answer-state representation — λ selected on a validation split, n_train = 87,794 far above every input dimension (216–3,584). This task stages the pinned prediction matrices (12 cells), holdout targets, and floor draws from HF at the data-revision pin; runs five identity gates first (assembly-fingerprint string equality, banked-R² reproduction within 5e-3 on all 12 cells, row alignment, feature-set validity, 22-contrast family equality — all PASS); computes per-context errors; runs the verbatim parent battery per reading; and computes the cross-space statistics. No new map is fit, so the mapping-baseline pair (identity+learned-bias, kNN retrieval) is inherited from the banked SAE arm and cited in the footer rather than recomputed.

**Sample training/evaluation data + completions:** No completions are generated (deterministic re-scoring of banked teacher-forced activation fits), so the worked examples are per-conversation error rows. Two rows of 9,941, from the interpretation round's seed-42 random sample; full artifacts: [SAE-space per-context CSV](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ff31e2c180529ce9e71f8e863778fcd4f1801e1e/eval_results/issue_1946/percontext_summary_L19_ridge_sae.csv), dense comparators [prefix/context CSV](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ff31e2c180529ce9e71f8e863778fcd4f1801e1e/eval_results/issue_1738/percontext_summary_L19_ridge.csv) and [bare CSV](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ff31e2c180529ce9e71f8e863778fcd4f1801e1e/eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv).

| ci | SAE error (prefix / bare / context) | dense error (prefix / bare / context) | language | topic | format |
|---|---|---|---|---|---|
| 18784 | 0.392552 / 0.380279 / 0.278365 | 0.517564 / 0.430230 / 0.235058 | en | factual_qa | prose |
| 36811 | 0.790357 / 0.638084 / 0.688165 | 0.406535 / 0.773407 / 0.425062 | en | advice_howto | mixed |

Conversation 18784 keeps its arm ordering across spaces (full-context best); conversation 36811 flips the prefix-vs-bare ordering across spaces — the expected level of per-conversation disagreement given cross-space coupling of 0.66–0.75 (below), not a data defect.

## Results

### The prefix−bare difference pattern reproduces in SAE space: correlation 0.855, 18 of 20 signs — verdict Reproduced

Each point is one of the 22 judged-context contrasts: x = the dense-space difference between the history-only (prefix) and bare-query arms' category error; y = the same quantity re-scored in SAE feature space (9,941 conversations behind every point). Filled: the 20 dense-significant contrasts; shaded quadrants: sign agreement.

![Scatter of 22 labeled judged-context contrasts: prefix minus bare category error difference, dense space on x versus SAE feature space on y; agreement quadrants shaded, dense-significant union filled.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946/mirror_pattern_sae_vs_dense.png)

> **Figure.** *The category-level mirror transfers to SAE space.* Difference-pattern Spearman correlation 0.855 (p < 1e-4, 0 of 10,000 shuffles), 18 of 20 signs on the dense-significant union. Both misses (English, translation) are near zero on at least one axis. n = 9,941 held-out conversations per contrast.

Under the plan's rule (p < 0.05, positive correlation, at least 15 of 20 signs) this is Reproduced. The overlapping category masks make the 22-point p mildly anti-conservative; the per-conversation result below corroborates without that dependence. Both sign misses are near-zero contrasts; a follow-up language-stratified recomputation confirms the English flip is a bare-arm magnitude effect: from dense to SAE the bare arm's English delta rises +0.068 → +0.151 vs +0.079 → +0.098 for the prefix arm, and the English prefix−bare contrast is −0.053 SAE (bootstrap CI excluding zero) vs +0.012 dense (CI straddling zero; [stratified JSON](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e42990a7710645d4be9cf2c50849737519fd36ba/eval_results/issue_1946/lang_stratified_mirror.json)).

### The same conclusion holds per conversation: the error difference correlates 0.710 across spaces

Per-conversation views over all 9,941 held-out conversations. Top row: SAE-space error, prefix vs bare arm (hexagon density, log count), and the per-conversation prefix−bare error difference in dense (x) vs SAE (y) space. Bottom row: each arm's error in dense (x) vs SAE (y) space.

![Four-panel per-conversation view with hexagon density plots of SAE-space prefix versus bare error and of the prefix minus bare difference across spaces, plus dense-versus-SAE error scatters for the prefix arm and the bare arm.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946/percontext_error_views.png)

> **Figure.** *The per-conversation difference pattern transfers across spaces.* Per-conversation difference pattern: Spearman 0.710 across spaces (0 of 10,000 shuffles, n = 9,941); per-arm cross-space coupling 0.66 (prefix) and 0.75 (bare). Log-count hexagon shading on the density panels — this is the per-unit data behind the headline.

Each conversation enters once, so the mask-overlap caveat does not apply; the 0.710 read supports the same verdict. Per-arm coupling of 0.66–0.75 is an upper bound on shared signal (shared difficulty inflates it).

One caution on reading the aggregate: category averaging over roughly 450 conversations per contrast amplifies even a modest shared component, and the strong category-level agreement is compatible with per-conversation coupling alone. This panel is the per-unit data behind the aggregate — where the spaces disagree conversation by conversation.

### Per-arm category profiles transfer unevenly; the prefix arm is at the detection edge in the verdict space

Per arm, the 22 category deltas in dense space (x) vs mean-pooled SAE space (y); each point one labeled contrast, correlation and p (10,000 shuffles) per panel.

![Three panels of category delta scatters comparing dense space to SAE space for the history-only, full-context, and bare-query arms, with per-panel rank correlations of 0.41, 0.76, and 0.80.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946/perarm_pattern_transfer.png)

> **Figure.** *Bare and full-context profiles transfer; the history-only profile is weakest.* Bare 0.80 and full-context 0.76 (both p < 1e-4); prefix 0.41 (p = 0.056, 10 of 13 signs on its dense-significant set). n = 9,941 conversations per point.

With 22 contrasts the design resolves correlations of roughly 0.43 and up; the prefix read is a failure to reject and does not establish non-transfer. The next section shows 0.412 is the weakest of that arm's readings. SAE space keeps 13 (prefix), 11 (full-context), and 18 (bare) significant contrasts vs 13, 16, 18 dense — the full-context arm's already-weak category structure thins further.

### The difference pattern survives every pooling choice where per-arm profiles do not; attenuation localizes to the input encoding

The prefix−bare difference pattern in dense space (x) re-scored in three alternative readings (y): dense inputs with SAE-feature targets, max-pooled, and active-fraction; 22 labeled contrasts per panel.

![Three panels showing the prefix minus bare difference pattern of dense space against the dense-input comparator, max-pooled, and active-fraction readings, with pattern correlations of 0.95, 0.91, and 0.84.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946/robustness_reading_variants.png)

> **Figure.** *The pattern holds in every reading.* Dense-input comparator 0.946 (19 of 20 signs); max-pooled 0.914 and active-fraction 0.837 (18 of 20 each). n = 9,941 conversations behind each of the 22 points per panel.

Keeping dense inputs and moving only the target basis to SAE features preserves the pattern almost exactly (0.946): the headline's attenuation to 0.855 enters at the input encoding, and both values fall well inside the Reproduced region. Per-arm profiles are pooling-fragile where the difference is not: the prefix profile reaches 0.796 under active-fraction pooling (p = 0.0001) and 0.575 in the dense-input comparator, so its verdict-space 0.412 is its weakest reading; under max-pooling all three per-arm transfers are weak (prefix 0.492, bare 0.462, full-context 0.173) while the difference correlation holds 0.84–0.95.

The leading explanation: subtracting the arms removes error shifts both share, which pooling changes most. An untested alternative: a few large contrasts anchor the difference pattern and dilute per-arm fragility.

### The nine named mirror categories keep their dense signs; eight of nine stay significant

Per arm (rows) and reading (columns: SAE, dense), the 22 category deltas with bootstrap CIs; filled markers are significant after BH correction; positive means the arm errs more on that category.

![Six forest panels of the 22 category deltas with confidence intervals for the history-only, full-context, and bare-query arms in SAE space and dense space, with significant contrasts filled.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946/category_forests_sae_vs_dense.png)

> **Figure.** *The per-category structure behind the scatters.* All nine mirror categories named in the dense round keep their sign in SAE space; eight of nine stay significant. Error bars: 10,000-draw bootstrap CIs; n = 9,941 conversations.

The mirror's named directions survive individually: the prefix arm stays harder on English (+0.098 SAE vs +0.079 dense), translation (+0.136 vs +0.071), and chitchat (+0.040 vs +0.084), and easier on WildChat (−0.104 vs −0.131); the bare arm stays harder on roleplay (+0.125 vs +0.104) and easier on chitchat (−0.152 vs −0.086). The one significance loss is the prefix NSFW delta: −0.037 in SAE space (bootstrap CI narrowly straddling zero) vs a significant −0.103 dense.

### Floor adjustment repeats the dense round's refusal collapse (round-1 approximate-floor read)

Unadjusted (x) vs floor-adjusted (y) SAE category deltas for the history-only and bare-query arms over the 19 floor-covered contrasts. Floors are SAE encodings of the mean answer state (K = 4 draws; 1,988 of 9,941 conversations; median floor share 6.2%) — an approximate construct.

![Two panels of unadjusted versus floor-adjusted SAE category deltas for the history-only and bare-query arms, with labeled contrasts around the identity line and the bare answer-is-refusal point far below it.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946/floor_adjustment_effect.png)

> **Figure.** *Consistent with answer-sampling variance rather than map failure in the refusal categories (approximate-floor read).* The bare answer-is-refusal delta flips from +0.051 (harder) unadjusted to −0.152 (easier) net of floor — the visible off-diagonal point. 19 floor-covered contrasts per arm; n = 1,988 conversations.

Net of the approximate floors, cross-space pattern correlations hold high: 0.812 (prefix), 0.898 (full-context), 0.939 (bare) over 19 shared contrasts (all p < 1e-4), and the refusal categories collapse exactly as in the dense round: the prefix and full-context refusal contrasts lose significance and the bare answer-is-refusal sign flips. These floors approximate the construct (encoding of the mean state); the follow-up round below re-derives them exactly and validates this read. The cross-space verdict rests on the unadjusted battery either way.

### The exact floor construct preserves the refusal collapse; the floor-adjusted read is de-quarantined

Unadjusted (x) vs exact floor-adjusted (y) SAE category deltas, history-only and bare-query arms, 19 floor-covered contrasts each. The exact floors teacher-force the K = 4 banked resampled answers, SAE-encode per token, then pool as the banked targets were built — replacing round 1's encoding of the mean state; same 1,988 conversations.

![Unadjusted versus exact floor-adjusted SAE category deltas for the history-only and bare-query arms, refusal categories highlighted, bare refusal-answer point far below the identity line.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b9d8bced5371d572bc47d234955b25771b672105/figures/issue_1946/floor_adjustment_exact.png)

> **Figure.** *The collapse is a property of the construct, not the approximation.* The bare refusal-answer delta reads −0.169 net of exact floors (bootstrap CI −0.29 to −0.04, n_group = 30; +0.051 unadjusted); no refusal-family contrast is significantly positive in any arm. 19 floor-covered contrasts per arm; n = 1,988 conversations.

Under the plan's decision rule (adjusted bare refusal-answer delta ≤ 0, no refusal-family contrast significantly positive) the collapse is preserved: round 1's read reflects the construct, not the pool-then-encode shortcut, and no pooling variant turns any refusal contrast significantly positive. Exact-adjusted cross-space correlations hold at 0.765–0.947 (approximate: 0.812–0.939; p < 1e-4).

The exact floors track the approximate ones (per-conversation Spearman 0.877; median share 6.6% vs 6.2%) — a diagnostic without verdict weight — and the excess is refusal-specific, not uniform: median floor share 12.1% inside refusal-answer conversations vs 6.6% elsewhere, the answer-sampling-variance signature. Capture, approximate-floor re-derivation, and battery identity gates all pass.

### Map quality is stable across conversation depth in every reading (exploratory)

Held-out R² per arm and conversation-depth band (2 / 3–4 / ≥ 5 turns) in each of the four readings; bars colored by arm.

![Grouped bar chart of held-out R squared by conversation depth band for the history-only, full-context, and bare-query arms across the four reading variants, showing a stable arm ordering and little movement with depth.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946/depth_band_r2.png)

> **Figure.** *No depth cliff in any reading.* The arm ordering (full-context > bare > prefix) holds in every reading and depth band, with no depth cliff anywhere; within a reading R² moves little (mean-pooled prefix 0.33–0.37). Band sizes n = 4,154 / 3,298 / 2,489 conversations.

Depth does not moderate the cross-space comparison: no reading degrades selectively on deep threads. Exploratory — the plan fixed no statistic for this read.

---

**Repro:** Round 1: 0 GPU-h; GCE `n2-highmem-16` (cpu-bigmem lane, instance `eps-issue-1946`), ~36 min wall including staging. Round 2 (exact floors): ~1 GPU-h; 1× H200 (fellows SLURM lane, job 16574, exit 0:0), ~39 min wall, ~11.1M teacher-forced tokens. Zero Anthropic/judge calls in both rounds; WandB unused by design. Code: [`scripts/issue1946_sae_percontext.py` @ `86977e774b07`](https://github.com/superkaiba/explore-persona-space/blob/86977e774b07d8f25d64817eb3b4d77e59a88898/scripts/issue1946_sae_percontext.py); exact command: `env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python scripts/issue1946_sae_percontext.py --phase all --staging-root data/issue_1946/gce_stage --data-revision 05cb982b0d3f9a21b5735d196a0afdc8175590e5 --rss-cap-gb 100`; battery: verbatim `scripts/issue1738_characterize.py --phase taxonomy` (10,000 bootstrap draws, 10,000 permutations, BH q = 0.05, seed 1738); cross-space permutation seed 1946. Condition slugs (plan §5): `sae_prefix`, `sae_bare`, `sae_context` (mean-pooled verdict cells); `dense_px_feat`, `dense_bq_feat`, `dense_cx_feat` (dense-input comparators); `sae_*_max` / `sae_*_frac` (pooling variants); scored against the banked dense `taxonomy.json`. Outputs: [eval_results/issue_1946 @ `ff31e2c180`](https://github.com/superkaiba/explore-persona-space/tree/ff31e2c180529ce9e71f8e863778fcd4f1801e1e/eval_results/issue_1946) (comparison JSON, per-context CSV, 4 per-reading taxonomy + depth JSONs, floor summary); follow-up: language-stratified English-vs-rest decomposition (`eval_results/issue_1946/lang_stratified_mirror.json`, `scripts/issue1946_lang_stratified.py`) at `e42990a771`. Round-2 exact floors (`followup_label: exact-sae-floors`): driver [`scripts/issue1946_exact_floors.py` @ `503584e308`](https://github.com/superkaiba/explore-persona-space/blob/503584e3085798ab6aea1b430cb9b5c90e6dfd21/scripts/issue1946_exact_floors.py); outputs [eval_results/issue_1946/exact_floors @ `503584e308`](https://github.com/superkaiba/explore-persona-space/tree/503584e3085798ab6aea1b430cb9b5c90e6dfd21/eval_results/issue_1946/exact_floors) (decision + cross-space comparison JSON, per-reading taxonomy + depth JSONs ×4 spaces, exact floor summaries ×3 poolings, gate JSONs, capture summary, token counts; committed on branch `issue-1946-exactfloors` pending merge); round-2 pins: data `05cb982b0d3f`, banked K-resample answers `12ab41dc1c4a`, model Qwen-2.5-7B-Instruct `a09a35458c`, SAE `c37e53c4bb07`; HF mirror [issue1946_sae_percontext/exact_floors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b07de1272b9355fbe957bbe66833910fe385609d/issue1946_sae_percontext/exact_floors) (exact per-conversation floor tensors + the JSONs above; listing verified live at fold time). HF tensors (round 1): [issue1946_sae_percontext](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/12ab41dc1c4a7163d183697e9c4fa53528904c9b/issue1946_sae_percontext) (37 files: 12 per-context npz, 12 pred16 mirrors, 3 y_holdout, floors, battery dual-writes; listing verified live at body time). Figures embedded above: [figures/issue_1946 @ `9deb689c31`](https://github.com/superkaiba/explore-persona-space/tree/9deb689c31ea0145f1dcfeb678da8a137fb4078d/figures/issue_1946), re-rendered from the committed round data via `savefig_paper` with per-figure sidecars — these supersede the driver-rendered PNGs at `ff31e2c180` (same underlying data; `figures/issue_1946/percontext_scatters.png` not embedded: a driver render superseded by the per-conversation views figure above); round-2 figure re-rendered via [`scripts/issue1946_exact_floors_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/b9d8bced5371d572bc47d234955b25771b672105/scripts/issue1946_exact_floors_figure.py) at [figures/issue_1946 @ `b9d8bced53`](https://github.com/superkaiba/explore-persona-space/tree/b9d8bced5371d572bc47d234955b25771b672105/figures/issue_1946) with `savefig_paper` sidecar, superseding the driver render at `503584e308`. Reused artifacts — reused prediction matrices + targets from [#1738](https://eps.superkaiba.com/tasks/1738): [issue1738_multiturn/sae_arm_bare/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/05cb982b0d3f9a21b5735d196a0afdc8175590e5/issue1738_multiturn/sae_arm_bare/analysis_tensors) at data revision `05cb982b0d3f9a21b5735d196a0afdc8175590e5` — fit: the banked cells are the object under study, identity-gated (fingerprint + R² within 5e-3); reused SAE weights from [andyrdt/saes-qwen2.5-7b-instruct @ `c37e53c4bb07`](https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct/tree/c37e53c4bb07127ad17ab88f28b93d4e87142e59) — fit: the same weights the banked cells were encoded with; reused judged context labels (parent Sonnet-4.5 instrument, 9,925 of 9,941 holdout conversations labeled, 16 unlabeled tolerated identically to the banked family construction) — fit: same labels, same 22-contrast family, asserted equal; round 2 additionally reused the parent's banked K = 4 resampled answers at `12ab41dc1c4a` — fit: the exact floors are defined over exactly these draws. Mapping-baseline pair for the banked fits (identity+learned-bias, kNN retrieval): cited from `issue1738_multiturn/sae_arm_bare/analysis_tensors/summaries/mapping_baselines.json` under the pinned tree above, not recomputed. Scope notes (run deviations, none touching the statistics): (1) the feature-set identity gate compares set-validity invariants against the banked ids rather than recomputing the selection (a measured cross-machine argsort tie-order dependence — 5 boundary ties for 2 cap slots — made recompute-equality machine-dependent); (2) the run executed on the plan's pre-named GCE cpu-bigmem fallback with a 100 GB RSS cap (measured peak well below); (3) git-destined JSON/CSV/figure outputs were harvested off the instance via ssh-tar and committed at `ff31e2c180` (the driver's upload covered the HF npz artifacts); (4) the implementation corrected the plan pseudocode's dropped-count constant to the banked n_dropped = 1 (648 is a different banked field); (5) round 2's capture-parity gate read max relative L2 0.0099 against the banked mean states and the approximate-floor re-derivation matched to 8e-7, with battery identity at zero deviation in all four spaces and the shared-contrast set equal to the parent's 19. Inherited from the parent: the bare arm predicts an answer representation produced with the history it never saw (matched-target asymmetry), and per-context error is a descriptive read of banked fits whose held-out fold discipline was enforced in the parent. Verifier WARNs acknowledged: some per-result prose is in the 120–180-word band, some Takeaways bullets exceed the 30-word bullet cap, and the total content prose exceeds the 800-word budget — kept for completeness of the eight-figure result set.

**Context:** Originating prompt (verbatim, from the task frontmatter):

> run all these: ... 2. Per-context SAE-space error read. The one measurement that would settle the bare-vs-prefix mirror-image question. Per-feature agreement doesn't test it

Round-2 originating proposal (verbatim title, from the `epm:followup-scope` note, proposer cheap-band auto-run):

> Exact SAE-space K-resample floors — does the refusal collapse survive the exact floor construct — Type: Diagnostic

Lineage: [#1738](https://eps.superkaiba.com/tasks/1738) — parent; this task re-reads the parent's banked per-context fits in SAE feature space. Created 2026-07-31; round 1 run 2026-08-01 (UTC), 0 GPU-h; round 2 (`exact-sae-floors`) run 2026-08-01 (UTC), ~1 GPU-h.
