---
title: The whole-answer-mean summary survives an exhaustive context-by-answer recipe
  sweep under family-held-out, selection-corrected evaluation (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-03T08:50:07Z'
has_clean_result: false
parent_id: 810
origin_prompt: 'Run the following experiment:

  - LOFO evaluation

  - ultrachat probes for activation collection

  - evaluation on OOD ultrachat probes

  - For both context vector and answer vector try:

  -- mean pool over all tokens (per layer)

  -- mean pool over all tokens at all layers

  -- max pool over all tokens (per layer)

  -- max pool over all tokens at all layers

  -- mean of max pool over layers

  -- max of mean pool over layers

  -- newline before assistant starts answering (after chat template) for context vector,
  newline before user would start answering (after chat template) --> per layer and
  also mean pool over layers, max pool over layers

  - For answer vector:

  -- im_end token

  -- token before im_end

  Try all combinations of these context and answer vectors (at all layers when this
  makes sense) and figure out the best one. Help me to plan this experiment and give
  an estimate as to GPU and wallclock time'
goal: On base Qwen2.5-7B-Instruct over the 50-context battery, determine which (context-vector
  recipe × answer-vector recipe) combination best supports (a) the linear context→answer
  map, (b) direct behavior read-out, and (c) map-mediated (chain) read-out, under
  leave-one-family-out (LOFO) evaluation with generic UltraChat probes for activation
  collection and a disjoint OOD UltraChat probe pool testing probe-set generalization
  on both the input and target sides — naming the best combination against selection-symmetric
  nulls.
relates_to:
- leak-predictor
---
# The whole-answer-mean summary survives an exhaustive context-by-answer recipe sweep under family-held-out, selection-corrected evaluation (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- **The linear context→answer activation map is far above chance under family-held-out evaluation: best held-out skill 0.87 against a 0.17 chance band; 17,447 of 34,652 combinations clear.**
- **No recipe separates from the whole-answer-mean incumbent under the reachable family-restricted bands: 37 of 46 answer families (in-probe; 36 of 46 held-out) and all 29 context families sit inside the winner's selection-noise band, and only single tail-position targets separate (as worse). The head-to-head challenger test itself has zero power — its 0.28 band exceeds the 0.17 achievable lead ceiling — so that read is a failure-to-reject, not evidence of a tie.**
- **The template-token targets' numeric lead reflects target stability: their cross-probe-set ceiling is 0.99 vs 0.89 for the whole-answer mean, which recovers more of its own ceiling (93% vs 88%).**
- **Probe-set generality is recipe-dependent: pooled and boundary summaries lose about 0.01 skill on unseen probes; last-k content-token context reads with k of 2 or more lose 0.06 to 0.23, while the single final content token loses only 0.02.**
- **27 of 28 behavior read-out clears are discounted by an exploratory (unbanded) family-centering re-read: family-centered rank correlations drop 57% at the median; none is established beyond family structure, pending a family-aware null.**
- **Chain read-out maxima land at cells where the map fails (skill −2.9 to +0.6) and exceed their own oracle — selection artifacts, not signal surviving the map.**

## Goal

- **This experiment in context:** Every summary-recipe choice in the leakage-predictor line was made piecemeal: the context read was locked in [#658](https://eps.superkaiba.com/tasks/658) against one alternative (reconstruction only, context-held-out folds, Betley probes); the answer-side sweep [#810](https://eps.superkaiba.com/tasks/810) was Betley-probe-only and its family-held-out re-read collapsed its behavior read-out; [#812](https://eps.superkaiba.com/tasks/812) bounded pooled-vs-unpooled on the same narrow regime. This run crosses 542 context-vector recipes with 1,018 answer-side targets (34,652 matched-layer combinations, including chat-template tokens, with/without-template pooling, last-k content tokens, and per-position targets) and scores every combination on the map, direct behavior read-out, and map-mediated read-out, under leave-one-family-out folds, generic UltraChat probes, a disjoint held-out probe pool on both input and target sides, and selection-symmetric null bands.
- **Broader narrative:** The leakage-predictor question (`docs/open_questions.md`, `leak-predictor` anchor) needs one defensible context summary and one answer summary for the base-model context→answer map its theory builds on. This run asks whether any cheaper or sharper read (a single boundary token, a template-block pool) should replace the incumbent whole-answer mean — and finds no recipe that separates from it under the reachable family-restricted bands, keeping the incumbent as the default.

## Methodology

**Design:** Nothing is trained. Base `Qwen/Qwen2.5-7B-Instruct`, all 28 decoder blocks, over the fixed 50-context battery (7 families: persona 14 / wildchat 10 / icl 8 / rephrase 6 / format 5 / behavior 5 / default 2). Context summaries: 19 per-layer families (with/without-template mean and max pools, the assistant-header newline, four trailing template tokens, template-block pools, last-k content tokens k = 1..8) plus 10 layer-pooled variants = 542 cells. Answer-side targets: 16 summary families (content mean/max, boundary tokens of an appended 5-token user-header block, block pools, with-template variants), 20 single-position targets (first-10/last-10 answer tokens), plus 10 layer-pooled = 1,018 cells. Matched-layer pairing gives 34,652 map cells. Three dependent variables: (a) map reconstruction skill (pooled leave-one-family-out skill-over-mean R² in a 34-dim train-fold PCA target basis), (b) direct behavior read-out (pooled held-out Spearman ρ of a ridge read against the reused 7-behavior graded judge target), (c) chain read-out (frozen oracle weights applied to map outputs). Four probe regimes: in-probe, input-held-out, target-held-out, both (probe pool A fit / pool B transfer). Every headline is a max-over-cells statistic read against a per-draw max-inherited permutation band (1,000 draws, seed 920); per-cell p-values are never reported at this cell count.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | project standard |
| Ridge λ grid | {1e-2, 1e-1, 1, 10, 100, 1e3}, nested PRESS-LOO selection | #658 fit primitives |
| Target basis | train-fold Gram PCA, k = 34 = min(48, smallest train fold − 2) | #810 + fold cap |
| Folds | leave-one-family-out, 7 family folds | #810 re-read |
| Null battery | 1,000 permutation-refit draws, seed 920, per-draw max inherited per DV × regime (× behavior) | #778/#810 recipe |
| Probe pools | 48 UltraChat probes each; pool B rebuilt disjoint (id + casefold-text exclusion), Betley-length-matched, seed 42 | #594 builder |
| Set-B generation | vLLM greedy, `max_tokens` 512, standard chat template | set-A recipe pin |
| Set-A completions | reused #658 bucket (50 files, Hub-verified) | reuse check |
| Behavior target | #812 graded 0–100 judge scores, 7 behaviors (deception excluded by reliability preflight) | reuse, no new judging |
| Extraction | batched left-padded teacher-forced forwards, batch 8, in-forward fp32 reductions, fp16/bf16 persist | #810 extractor extended |
| Fit numerics | float64 batched dual-space on GPU; serial-oracle equivalence at 1e-8 | #810 batched-null contract |

**Evaluation:** Map skill is scored on pooled held-out predictions against per-fold train-mean baselines, variance-weighted over the 34 PCA dims. Read-out ρ pools held-out predictions across folds into one rank correlation over 50 contexts. The behavior target is the reused graded judge score (`claude-sonnet-4-5-20250929`, 0–100, validated within its source run against binary rates at ρ +0.86 to +0.98); judge-reliability ceilings 0.68–0.96 annotate every read-out claim, with no attenuation correction. Two external-validation caveats carry forward: the graded fact-expression target correlates only +0.01 with its independent earlier binary rate, and format-style +0.15 — read-outs on those two are annotated and never headline. Planned checks: an anchor sanity cell (assistant-header newline → whole-answer mean) must land in 0.6–0.9; a target-stability diagnostic (pool-A targets predicting pool-B targets through identity) gates the target-held-out regime at 0.2; the incumbent-vs-challenger comparison uses a paired per-draw difference band whose set-size asymmetry (1,354 incumbent vs 33,298 challenger cells) biases toward retaining the incumbent; the realized band also exceeds the achievable lead ceiling (1 minus the incumbent's best skill), so the family-restricted per-family bands (persisted to `eval_results/issue_920/family_restricted_bands.json`) are the operative separation test.

**Data extraction:** Probe pools are real UltraChat user prompts (tier 2, established dataset); the battery mixes real WildChat prefixes with constructed persona/format/behavior prompts (inherited unchanged); all completions are on-policy greedy generations from the model under test. Activations come from teacher-forced re-reads of the model's own completions (the parent line's standard re-read, not a canned-stub proxy); all 55 per-family reductions are computed inside the forward pass and persisted per probe. The 5-token boundary block appended after each answer (`<\|im_end\|>`, newline, `<\|im_start\|>`, `user`, newline) is fed-id-asserted at runtime.

**Sample training/evaluation data + completions:** No training rows exist. Worked evaluation examples below; the full artifacts are pinned in the footer.

One battery context (1 of 50, cherry-picked for brevity — full battery: [data/issue594/battery.json](https://github.com/superkaiba/explore-persona-space/blob/cd8f646d504105a2abfce73b38ce01444272a086/data/issue594/battery.json)):

> `f5_fmt_json` (family: format) — system prompt: *"Respond to every question with a single valid JSON object and nothing else."*

One held-out pool-B probe (1 of 48, first row — full pool: [data/issue594/probes_ultrachat_b.json](https://github.com/superkaiba/explore-persona-space/blob/cd8f646d504105a2abfce73b38ce01444272a086/data/issue594/probes_ultrachat_b.json)):

> *"How has technology enabled the development of specialized equipment for disabled athletes in track and field events, such as prosthetic limbs and wheelchairs?"*

Three set-B completions (3 of 2,400, random sample seed 42; full bucket: [HF gen_b raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a16f792625b879f012fd97de321b9b1d216ef5fa/issue920_summary_sweep/raw_completions/gen_b)):

<details>
<summary>3 of 2,400 set-B completions, random sample (seed 42), truncated</summary>

| Context | Probe (truncated) | Completion (truncated) |
|---|---|---|
| `f5_fmt_json` | Write an email newsletter announcing a new line of premium sunglasses. | `{"subject": "Introducing Our New Line of Premium Sunglasses!", "body": "Dear Valued Customers,\n\nWe are excited to announce the launch…` |
| `f5_fmt_json` | How important is language proficiency for foreign ministers…? | `{"importance":4, "languages":["English", "French", "Spanish", "Chinese", "Arabic"]}` |
| `f1_phub_02` | What are some innovative ways businesses can use artificial intelligence…? | Artificial Intelligence (AI) is transforming various sectors, including healthcare, and can significantly enhance business operations… |

</details>

## Results

### Nearly every pooled or boundary summary supports the context→answer map far above chance

The heatmap shows the best matched-layer held-out map skill for each of 29 context families × 46 answer-target families (in-probe regime), color clipped to the 0–0.9 range; the chance band from the 34,652-cell max-inherited null is 0.17, far below the achievable ceiling of 1.

![Family-by-family map skill heatmap, best matched layer, in-probe](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/hero_family_heatmap_R1_v2.png)

> **Figure.** *Held-out map skill per family pair.* Best matched-layer pooled leave-one-family-out skill, 29 context families (rows) × 46 answer families (columns), clipped to the 0–0.9 range. Bright block: pooled/boundary context reads × pooled/boundary targets (0.75–0.87). Dark columns: single tail-position targets. n = 50 contexts, 7 family folds.

The observed maxima are 0.871 (in-probe), 0.863 (input-held-out), 0.866 (target-held-out), 0.863 (both) against bands 0.17, 0.16, 0.21, 0.18 — all clear by ≥0.65 with the skill ceiling at 1, so the bands are informative, not vacuous. The map itself is not in question at n = 50; which recipe serves it best is (next results).

### Half the sweep clears the chance band

The histogram shows all 34,652 in-probe cell skills (x-axis clipped at −1; 7,045 cells below −1 not shown), with the 0.17 chance band marked.

![Distribution of map skill over all cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/r1_skill_distribution_v2.png)

> **Figure.** *All 34,652 map cells.* 17,447 cells clear the 0.17 chance band. The long negative tail (down to −1086) comes from max-pool-over-template context reads, where a single outlier dimension dominates. n = 50 contexts.

Half the grid supports the map above chance; the winner-vs-incumbent question needs the separation analysis below, not this marginal view.

### The head-to-head incumbent test has zero power; under the reachable family bands only tail-position targets separate

The left panel shows the head-to-head incumbent-vs-challenger test on both probe regimes — observed best-challenger lead, 97.5% selection-noise band, and the achievable lead ceiling (1 minus the incumbent's best skill). The right panel shows each answer family's best-cell gap to the sweep winner against its family-restricted per-draw band, in-probe.

![Incumbent test band vs ceiling, and per-family gap vs band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/family_gap_vs_band.png)

> **Figure.** *Left:* the paired band (0.28 in-probe, 0.27 held-out) exceeds the 0.17 ceiling — even a perfect challenger could not clear it. *Right:* 37 of 46 answer families sit inside their band; the nine single tail-position targets (2nd–10th from the answer end) separate as worse. n = 50 contexts, 1,000 draws.

The paired test carries zero power (band 0.277 vs ceiling 0.170 in-probe; 0.269 vs 0.173 held-out), so its non-rejection is a failure-to-reject, never evidence of a tie. The retention claim instead rests on the family-restricted bands, which are demonstrably able to reject — tail-position gaps 0.56–0.66 clear bands near 0.38 — while the incumbent's family sits deep inside (gap 0.0415 vs band 0.286) and all 29 context families are inside theirs (context-side bands run 0.24 up to 217 for degenerate families, uninformative by construction). On held-out probes one layer-pooled newline family also separates marginally (gap 0.374 vs band 0.34), leaving 36 of 46 inside. Full table: `eval_results/issue_920/family_restricted_bands.json`.

### The winning cell tracks contexts at family grain, and the anchor cell reproduces the parent

The scatter shows, for the top in-probe cell (template-block max → user-header pool max, layer 12), the pooled held-out prediction against the true target on the leading PCA dimension, one labeled point per context.

![Winning cell per-context scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/winning_cell_scatter.png)

> **Figure.** *Winning map cell, per-context view.* Held-out prediction vs truth on the leading fold-basis dimension; 50 labeled contexts. Wildchat/rephrase contexts separate from persona/format/icl contexts — much of the resolvable variance is family-grained. n = 50.

Predictions track truth within and between clusters, but the dominant axis separates families — the map's resolvable variance is heavily family-structured even though every read is family-held-out. The anchor cell (assistant-header newline → whole-answer mean, best matched layer 16) lands at 0.806, inside the planned 0.6–0.9 window and matching the parent's family-held-out anchor ≈0.80: the pipeline reproduces the incumbent number.

### The top recipes plateau across middle layers rather than peaking at one depth

The line plot shows per-layer in-probe skill for the top-5 family pairs, with the chance band dashed.

![Per-layer skill for top-5 family pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/per_layer_top5_pairs.png)

> **Figure.** *Layer profile of the top-5 pairs.* Skill rises through layers 0–7, plateaus ≈0.80–0.87 across layers 8–18, and decays toward layer 27. Dashed line: 0.17 chance band. n = 50 contexts per point.

The argmax layers (11–12) sit on a broad plateau; layer choice within 8–18 moves skill by less than the selection-noise band, so no single-depth claim is supported.

### Probe-set generality is recipe-dependent: pooled reads transfer, single content tokens degrade

The heatmap shows, per family pair at its best in-probe cell, the input-held-out minus in-probe skill delta (color clipped to ±0.15).

![Probe-set generalization delta per family pair](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/hero_family_heatmap_R2_minus_R1_v2.png)

> **Figure.** *Held-out-probe delta at each family pair's best cell.* Near-zero (pale) rows: pooled, boundary, and mean context reads. Strongly negative rows: last-k single content-token context reads (k ≥ 2). Clipped at ±0.15. n = 50 contexts.

The winner set loses ≈0.01 on unseen probes (winner −0.012; incumbent anchor −0.010; top-100 cells mean −0.013 — the winner's delta matches unselected comparable cells, so selection bias does not drive it). Last-k content-token reads with k ≥ 2 lose 0.06–0.23 (the held-out-minus-in-probe delta at each context family's best in-probe cell — the same at-cell convention as the other deltas in this paragraph); the single final content token is the exception, losing only 0.02, and the trailing turn-start token loses 0.09. The direction predicted by the probe-generality hypothesis holds: averaged and boundary summaries are probe-set-general; multi-token content reads are not.

### The template-token targets' lead is target stability, not more context information

The bar chart shows the target-stability diagnostic per answer family: skill of pool-A targets predicting pool-B targets through the identity, best layer.

![Target-stability ceiling per answer family](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/r3_identity_ceiling.png)

> **Figure.** *Cross-probe-set target stability.* Template-token targets sit at 0.97–0.99, the whole-answer mean at 0.89; tail positions 3–10 from the answer end sit at 0.60–0.66, with the final two positions higher (0.91, 0.81). All families clear the 0.2 planned gate, so the target-held-out regime is valid. n = 50 contexts.

Template-token targets are nearly deterministic functions of the context, carrying little probe-specific variance — mechanically easy targets. Normalizing best skill by this ceiling reverses the ordering: whole-answer mean 0.830/0.891 ≈ 93% of ceiling vs the winning user-header pool 0.871/0.988 ≈ 88%. With no challenger family separating under the reachable family-restricted bands (zero-power result above), the honest phrasing is "best predictive summary under this reduction and noise structure," not "the answer profile lives at boundary tokens." The pooling ladder agrees: pooled variants beat their single-token members (user-header pool 0.87 vs turn-end token 0.76), and the context content-only mean (0.64) trails every trailing-boundary read (0.79–0.87).

### Behavior read-out clears are discounted by an exploratory family-centering re-read

The dumbbell plot shows, per read-out test (7 behaviors × context/answer side × two probe regimes = 28), the banded full rank correlation at the best cell (blue) and the same cell's family-centered correlation (red).

![Full vs family-centered read-out correlations](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/dv2_full_vs_centered_rho.png)

> **Figure.** *Read-out clears vs the family-structure re-read.* Blue: banded absolute rank correlation at each best cell (0.52–0.86). Red: after subtracting family means from prediction and target. Median drop 57%; 4 of 28 reads fall below 0.06 and 8 below 0.2. Judge-reliability ceilings 0.68–0.96. n = 50 contexts, 7 families.

27 of 28 tests nominally clear their per-behavior bands (0.48–0.54; reachable, ceiling 1). But the behavior target is strongly family-structured (between-family variance fraction 0.31–0.85), predictions cluster by family, and free permutation assumes exchangeability those clusters violate — the bands are anti-conservative; the plan pre-set a ≈51% family-wise chance of at least one nominal clear. Family-means-only correlations span 0.07–0.96; several best cells are layer 0–2 template or tail-token reads, a length/format-confound signature. The centering re-read is exploratory and unbanded (family-aware null pending), so it discounts rather than overturns the clears. Largest surviving residuals: fact expression 0.56 (both regimes, context side; a +0.01-externally-validated target, above its 0.72 reliability-consistent maximum), format style answer-side 0.47–0.49 (+0.15 validation), refusal 0.32–0.47, harmful compliance 0.44–0.48. The one non-clear (self-report, context, in-probe: 0.523 vs 0.526) is a failure to reject. The parent's family-held-out read-out collapse stands.

### Per-context data behind the strongest answer-side clear: family clusters carry the correlation

The scatter shows the pooled held-out prediction against the graded harmful-compliance score for the clearing answer-side cell (user-header pool max, layer 0), 50 labeled contexts colored by family.

![Per-context scatter behind the harmful-compliance clear](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/dv2_harmful_compliance_cell_scatter.png)

> **Figure.** *The low-level data behind one clear.* Persona contexts cluster low-left, wildchat/format contexts high-right; the 0.72 full correlation drops to 0.15 after family centering. A layer-0 template-token read carrying "behavior information" is the family-confound signature. n = 50.

Within any single family the association is weak; the correlation is produced by family separation on both axes. This is the per-unit view of the previous result's aggregate claim.

### Chain read-out maxima are selection artifacts, not signal surviving the map

The bar chart shows, per behavior, the chained read's absolute rank correlation at its best cell (maximum absolute correlation), paired with the oracle read at the same cell (in-probe).

![Chain vs oracle at each behavior's best chain cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cd8f646d504105a2abfce73b38ce01444272a086/figures/issue_920/chain_vs_oracle_gap.png)

> **Figure.** *Chained read vs its oracle at each behavior's best chain cell.* The chained read exceeds the same-cell oracle for every behavior, by 0.09–0.49 — impossible as clean signal recovery; the signature of selection over 34,652 family-structured cells. n = 50.

All 14 chain band tests nominally clear (bands 0.54–0.60), but the best-cell reads land where the map fails — reconstruction skill −2.9 to +0.55, five of seven negative or near zero — and their centered correlations collapse like the direct reads (persona-drift −0.83 → −0.08). A chained read through a failed map cannot legitimately out-predict its own oracle. At the winner map cell chain reads stay small (all ≤ 0.37 absolute) but not uniformly below the oracle — 3 of 7 behaviors exceed it (harmful compliance 0.29 vs 0.03) — consistent with noise at n = 50; the map-induced-loss estimate is not identifiable and no chain headline is claimed.

*(Conciseness: the prose budget, several 120–175-word results, and three long Takeaways bullets exceed caps — acknowledged; ten results carry a 34,652-cell three-DV sweep.)*

---

**Repro:** Run: one GCP spot A100-40 instance (`capture-7b` intent), ≈1.45 GPU h total (set-B generation → extraction A+B → batched float64 fit battery, 40 s wall → 1,000-draw null battery on GPU), then CPU aggregation; run commit `7bae890278d45dfa029c71caf3f45e5128ab2c1d` (branch `issue-920`); figures + this analysis at `cd8f646d504105a2abfce73b38ce01444272a086`. Eval JSONs: `eval_results/issue_920/{map_skill_by_cell,readout_rho_by_cell,chain_rho_by_cell,null_bands_and_headline,family_restricted_bands}.json` (git, issue-920 branch; HF mirror `issue920_summary_sweep/analysis_tensors/eval_json/`; `family_restricted_bands.json` is the round-2 auditable per-family gap/band table incl. per-draw family maxima). Crash-recovery integrity: a transient HF 500 killed the post-upload verify leg; the four run-produced eval JSONs were recovered from the crash bucket `issue920_partial/att-20260703-130528/` on the HF data repo, their sha256 hashes match the committed git blobs (upload-verifier confirmed), and all Hub artifacts were re-verified via a live listing 2026-07-03. HF (all Hub-verified 2026-07-03 via live `list_repo_tree`): [issue920_summary_sweep](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a16f792625b879f012fd97de321b9b1d216ef5fa/issue920_summary_sweep) — per-probe summary stores both pools (100 files), per-draw null matrices (`dv1_null_skills.pt`, `dv2_null_rho.pt`, `dv3_null_rho.pt`), pooled held-out predictions, set-B rollout text (50 files), gate reports. Reused artifacts — set-A completions from [#658](https://eps.superkaiba.com/tasks/658) (`issue658_theory_assumptions/raw_completions_genre-generalization-ultrachat/`, 50 files, Hub-verified; fit: same battery, probes, greedy 512-token recipe); graded behavior targets + reliability ceilings from [#812](https://eps.superkaiba.com/tasks/812) (`eval_results/issue_812/graded_e0_{highm,lowm}.json`, `reliability_and_learning_curve.json`, git; fit: 50/50 context join, 7 usable behaviors); fold/PCA/null primitives from [#810](https://eps.superkaiba.com/tasks/810) (fit: serial-oracle equivalence gate passed at 1e-8). Battery + probe pools committed in git (`data/issue594/`). Analyzer figures scripts: `scripts/issue920_analyzer_figs.py`, `scripts/issue920_rev2_figs.py` (round-2 band table + regenerated figures), `scripts/issue920_labels.py` (plain-English figure labels).

**Context:** Created 2026-07-03 (parent [#810](https://eps.superkaiba.com/tasks/810)); run 2026-07-03; analyzer rounds 1–2 (round 2 = interpretation-critic revision, 2026-07-03; no follow-up rounds yet). Originating prompt (verbatim): "Run the following experiment: - LOFO evaluation - ultrachat probes for activation collection - evaluation on OOD ultrachat probes - For both context vector and answer vector try: -- mean pool over all tokens (per layer) -- mean pool over all tokens at all layers -- max pool over all tokens (per layer) -- max pool over all tokens at all layers -- mean of max pool over layers -- max of mean pool over layers -- newline before assistant starts answering (after chat template) for context vector, newline before user would start answering (after chat template) --> per layer and also mean pool over layers, max pool over layers - For answer vector: -- im_end token -- token before im_end Try all combinations of these context and answer vectors (at all layers when this makes sense) and figure out the best one. Help me to plan this experiment and give an estimate as to GPU and wallclock time"
