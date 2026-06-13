---
title: The full-response JS predictor scores well on the full panel only because the
  stylized vs nonstylized split alone carries most of the signal (LOW confidence)
kind: analysis
tags: []
created_at: '2026-06-08T23:16:05Z'
has_clean_result: false
parent_id: 511
---
# The full-response JS predictor scores well on the full panel only because the stylized vs nonstylized split alone carries most of the signal (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** The output-distribution JS predictor "works" on the full 16-persona panel, but once I drop the three theatrical personas (pirate / stand-up / villain) the out-of-sample CV R² collapses below zero — so it was the stylized-vs-rest gap doing the work, not a continuous distance gradient.

**Takeaways.**
- Phase 1 closed parent #511's loose ends — `gauss_kl` at layer 22 still holds at out-of-sample CV R² ≈ 0.62 with the full ridge (N=500, R=10), and MMD / Wasserstein-2 / cosine all sit at 0.56-0.58. No surprise there.
- The full-response JS predictor lands at CV R² ≈ 0.24 on the full panel — substantially below all four activation predictors.
- When I drop the three high-stylization personas, JS CV R² goes negative (no out-of-sample skill at all) and the within-stylized correlation has the wrong sign. The predictor is essentially a stylized/nonstylized binary detector wearing continuous-metric clothing.

**How this updates me.** Less optimistic that "cheap base-model JS on the output distribution" is a real substitute for activation geometry — at least for predicting marker-leakage transfer. The activation predictors aren't just better in CV R²; they survive the stylized-personas-removed stress test, which JS doesn't. Whether the activation predictors ALSO collapse on the nonstylized subpanel is the next thing to check (Phase 1 didn't run that subpanel — see Next steps in the relevant finding below). What would move me back: a clean within-nonstylized JS result on a richer panel of "normal" personas.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

Parent task [#511](https://eps.superkaiba.com/tasks/511) ran a multi-metric predictor sweep to find which base-model distance between two personas best predicts how the marker-leakage signal transfers between them — using the ΔG matrix from [#474](https://eps.superkaiba.com/tasks/474) as the target. Phase 1 of #511 plateaued the headline cell (Gaussian-KL at residual-stream layer 22, predicting count-style ΔG, out-of-sample CV R² ≈ 0.62) but an autonomous compute-deviation cut wass2, ridge at N=500 × R=10, the loc-epoch robustness sweep, and a full-response JS baseline. This task ran the dropped pieces. Two parts:

- **Phase 1 (CPU, ~36h on the VM)** — restored full scope of the activation-space sweep: wass2 added, ridge at N=500 × R=10, loc-arm epochs {1, 2, 3, 5}.
- **Phase 2 (GPU, 1× H100, ~3d20h)** — built the full-response Rao-Blackwellized JS predictor from scratch (the deprecated single-next-token JS in #511's plan does NOT count). Compared it against the four activation metrics on the same 16-persona panel.

The reason this question matters: if a cheap output-distribution metric (one base-model pass; no residual-stream extraction; no cloud-aware ridge) predicted leakage as well as the activation geometry, that would be a much simpler causal probe. The question I had going in was "does it?"

The construct I'm measuring is **predictor-vs-leakage skill** — out-of-sample CV R² and length-partialled Spearman ρ — on the 240 ordered pairs of 16 base-model personas. Higher is better, both metrics agree, and the CIs are panel-row bootstraps (n_boot=2000, seed=42) over the 240 pairs.

### What I ran

**Phase 1: activation-predictor sweep, full scope.** I re-ran `scripts/issue511_probe_count_sweep.py` with the descoped axes restored:

- Metrics: `gauss_kl`, `mmd`, `wass2`, `cosine` (Phase 1 of #511 had dropped wass2).
- Probe counts (N): {25, 50, 100, 200, 350, 500} (capped at 350 in #511).
- Replicates (R) at each (N, layer, metric): 10 (was 5).
- Layers: L19-L24 (the band #502 identified as best — last-prompt-token extraction).
- Loc-arm epochs: {1, 2, 3, 5} (was loc_ep1 only).
- Output: 6480 fit rows in `eval_results/issue_522/probe_count_sweep_results.json`; plateau verdict per cell.

**Phase 2: full-response JS predictor.** I built a fresh script (`scripts/issue522_js_predictor.py`) implementing the canonical full-response Rao-Blackwellized JS estimator (Amini/Vieira/Cotterell 2025, arXiv 2504.10637), which is the only operationalization the project rule sanctions (`.claude/rules/persona-distance-metrics.md`; #404 and #458's older single-next-token JS is deprecated). For each of the 16 personas A, sample 8 on-policy responses (temp=1, ≤256 tokens) on each of 200 probes from #502's probe set — that's the FROM side. For each of the 16×16 ordered pairs (A, B), teacher-force each of A's sampled responses through B's persona-conditioned base model, get the exact full-vocab per-position next-token distributions under both A and B, compute the per-position JS in nats, length-normalize, average over positions and responses. Total cache: 409,600 entries (16 cond_ids × 200 probes × 4 epochs × 8 responses, with per-epoch caching for the bystander-direction reuse). Smoke gates: max-diagonal JS = 0 nats; max-symmetry residual = 0 nats.

**The regression step.** Length-partial Spearman ρ + LOCO (leave-one-cond-out) CV R² of `M_js = 1 − JS` against the #474 ΔG matrix. Two panels: `full` (240 ordered pairs across 16 personas) and `nonstylized` (156 pairs across the 13 personas that exclude the three high-stylization arms — A3 Pirate captain, A4 Stand-up comedian, A5 Villainous mastermind). Four target epochs (1, 2, 3, 5). Panel-row bootstrap with n_boot=2000, seed=42 for 95% CIs.

The 16 personas, in plain English:

<details open>
<summary>The 16 base-model personas (from #406)</summary>

| Class | cond_id | Name |
|---|---|---|
| A | A1 | Helpful assistant |
| A | A2 | Software engineer |
| A | A3 | **Pirate captain (stylized)** |
| A | A4 | **Stand-up comedian (stylized)** |
| A | A5 | **Villainous mastermind (stylized)** |
| B | B1 | Bare question |
| B | B2 | Imperative tell-me |
| B | B3 | Polite request |
| B | B4 | Formal request |
| B | B5 | Socratic hypothetical |
| C | C1 | Standard Qwen template |
| D | D1 | Formal register rewrite |
| D | D2 | Casual register rewrite |
| D | D3 | Indirect framing rewrite |
| D | D4 | Declarative form rewrite |
| D | D5 | Enumerated framing rewrite |

The "stylized" label is the three personas that adopt a strong narrative voice; the other 13 are everyday assistants varying mostly in register and phrasing. Full configs in `src/explore_persona_space/experiments/i406_conditions.py`.
</details>

Example probe (from #502's mixed-distribution probe set): `"What's the most important property of a good error message?"` — same probe asked under all 16 personas; per-pair JS is computed over each pair's 8 sampled responses × 200 probes.

### Findings

#### Phase 1: the activation-predictor plateau survives

Restoring the full N=500 × R=10 ridge across layers 19-24, metrics {gauss_kl, mmd, wass2, cosine}, and loc-arm epochs {1, 2, 3, 5} gives no surprises. The L22 / gauss_kl headline cell from #511 still plateaus at CV R² ≈ 0.62 at epoch 1 (best subset replicate 0.6196; mean across 10 subsets 0.617), with the plateau verdict holding at N=500 (δ from N=350→500 is −0.0024, well inside σ_ref ≈ 0.004). The other three activation metrics at the same L22 / N=500 / epoch 1 cell:

| Metric | Mean CV R² (10 subsets) | Best replicate |
|---|---|---|
| Gaussian KL | 0.617 | 0.620 |
| MMD | 0.580 | 0.583 |
| Wasserstein-2 | 0.569 | 0.572 |
| Cosine | 0.557 | 0.560 |

So the metric ranking I expected from #511's smaller ridge holds: gauss_kl edges MMD by ~0.04 CV R², which edges wass2 and cosine by ~0.02 each. Across loc-arm epochs the picture is the same shape but lower altitude: gauss_kl drops from 0.617 (ep1) to ~0.41-0.45 (ep2/3/5), and the four metrics keep their relative ordering. The plateau-on-N verdict holds at every (layer × metric × epoch) cell in the full sweep — N=500 is not buying real precision over N=350, so future predictor work doesn't need to budget for the heavier ridge.

This was always a "verify the existing line still holds" arm, not a finding in itself. It came back PASS. There is no figure for this arm — the per-cell CV R² table is the artifact, in `probe_count_sweep_results.json`. Skipping the figure here because the result is "no movement vs #511."

#### A full-response JS predictor does correlate with leakage transfer on the full panel — but at one-third of the activation predictors' CV R²

![Most of the JS predictor lives in the stylized-vs-rest gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/hero.png)

> **Figure.** *Most of the full-response JS signal lives in the gap between stylized and nonstylized personas, not in a continuous distance gradient.* Each dot is one of 240 ordered persona pairs (a → b). X-axis: full-response JS divergence between base-model output distributions under persona a and persona b, in nats. Y-axis: marker-leakage transfer ΔG between the same pair, from #474's loc-arm checkpoint at epoch 1, also in nats. Blue points are pairs where neither persona is stylized (156 of 240); red points touch one of A3 / A4 / A5 (84 of 240). The header statistics (ρ = 0.54, CV R² = 0.24) are computed on the full panel and length-partialled.

The number reads positive on the full panel — Spearman ρ = 0.54 (95% panel-CI [0.45, 0.63]) and LOCO CV R² = 0.24 (95% panel-CI [0.12, 0.34]), both with p ≪ 0.001 against the null. So in a vacuum, the answer is "yes, a base-model on-policy JS divergence has some skill at predicting how marker-leakage transfers." But the figure makes the source of that skill obvious. The 84 pairs touching a stylized persona (red) sit at high JS (0.07-0.10 nats) and moderate ΔG (~5-15 nats); the 156 nonstylized-only pairs (blue) sit at low JS (0-0.04 nats) and higher ΔG (~5-25 nats). The full-panel correlation is largely a between-cluster contrast.

How much of the signal IS the binary? A point-biserial of the stylized-touching indicator against JS gives r = 0.72 (p = 3.4e-40); against ΔG it gives r = -0.61 (p = 3.8e-26). So the binary alone carries most of both axes' variance. The activation-side predictors (next finding) deliver 2.4× higher CV R² on the same panel — so even if you accept "the predictor works on the full panel," it works much less well than the cheaper option you already have if you can read residual streams.

The relevant CV R² comparison on the same panel × epoch cell:

![Activation-geometry predictors beat full-response JS](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/metric_compare.png)

> **Figure.** *On the full 240-pair panel, all four activation-geometry predictors (cloud-aware ridge at layer 22) achieve more than twice the out-of-sample CV R² of the output-distribution JS predictor.* X-axis is the predictor; y-axis is leave-one-cond-out CV R² against the same ΔG target (loc-arm, epoch 1, full 240 pairs). Error bars on the JS bar are the panel-row bootstrap 95% CI; error bars on the activation bars are subset-σ across 10 ridge replicates at N=500. The JS predictor was a 1× H100 base-model forward-pass job; the activation predictors run on cached residual streams plus a cloud-aware ridge fit on CPU.

#### Drop the stylized personas, and the JS predictor loses all out-of-sample skill

The full-panel result already hinted that the JS signal was carried by the stylized split. Restricting the panel to the 13 nonstylized personas (156 pairs) confirms it.

![Drop stylized → CV R² goes negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/by_panel_epoch.png)

> **Figure.** *Removing the three stylized personas (Pirate captain, Stand-up comedian, Villainous mastermind) cuts the JS predictor's ρ by half and pushes out-of-sample CV R² below zero across all four loc-arm training amounts.* Left panel: length-partialled Spearman ρ between JS and ΔG. Right panel: leave-one-cond-out CV R² for the same. Blue bars are the full 240-pair panel; orange bars are the nonstylized 156-pair subpanel. Source-persona training amount on the x-axis. Error bars are panel-row bootstrap 95% CIs (n_boot = 2000, seed = 42). The dashed grey line on the CV-R² panel marks zero, the "no out-of-sample skill" threshold.

At epoch 1, the full panel reads ρ = 0.54 / CV R² = 0.24; the nonstylized subpanel reads ρ = 0.27 (95% CI [0.11, 0.42]) / CV R² = -0.022 (95% CI [-0.15, 0.08]). The pattern holds across epochs 2, 3, 5: full-panel CV R² stays positive (0.10-0.13) but nonstylized CV R² is centered on a negative value at every epoch (range -0.036 to -0.022, all four CIs straddle zero with the upper edge under +0.08). And the within-stylized-touching ρ on the raw data is +0.15 (wrong sign — high JS is supposed to predict LOW transfer, but among the stylized pairs more JS goes with more transfer, not less). The conclusion the figure forces is uncomfortable: the predictor that looked credible on the full panel is essentially a stylized-vs-rest detector, and once you ask it the question that actually matters for a predictor — "does it generalize to personas you didn't fit on?" — the answer is no.

A cherry-picked qualitative example that fits the pattern: the (A1 Helpful assistant, B1 Bare question) pair, two nominally similar nonstylized personas, has JS = 2.0e-08 nats — essentially zero — and ΔG = 25.8 nats (one of the largest leakage-transfer values in the matrix). The JS predictor would say "these two are the same persona, leakage should transfer perfectly" — and the ΔG agrees that leakage DOES transfer near-completely between them. But contrast with (A1 Helpful assistant, A2 Software engineer): JS = 0.071 nats, much larger than the A1-B1 pair, and yet ΔG = 19.4 nats — still very high leakage transfer. The continuous distance gradient the predictor wants to use isn't actually there between most pairs of "normal" personas; the only large gap is between the stylized voices and everyone else.

<details>
<summary>Five raw pair-level rows (JS, ΔG) at loc-arm epoch 1 — cherry-picked for illustration</summary>

Full per-pair data: see `eval_results/issue_522/js_matrix.json` (JS), and `eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json` (ΔG).

| Source persona | Target persona | JS (nats) | ΔG (nats) | Class |
|---|---|---|---|---|
| Helpful assistant (A1) | Bare question (B1) | 2.0e-08 | 25.79 | nonstylized |
| Helpful assistant (A1) | Software engineer (A2) | 0.071 | 19.42 | nonstylized |
| Helpful assistant (A1) | Pirate captain (A3) | 0.077 | 10.81 | stylized-touching |
| Pirate captain (A3) | Stand-up comedian (A4) | 0.029 | 9.95 | stylized-touching |
| Casual register (D2) | Indirect framing (D3) | 0.011 | 18.07 | nonstylized |

The A3-A4 row is the most telling: low JS (the two stylized voices have similar output distributions to each other), but the ΔG transfer is still small relative to the nonstylized pairs at the same JS. JS is not picking up the leakage-transfer dynamic correctly there either — it's just that stylized-touching pairs are a minority of the panel.
</details>

**Note on the activation predictors and the same subpanel.** Phase 1's sweep did NOT run a nonstylized-only ridge — `bakeoff._pairs` always builds the full 240-pair panel for that script. So I can't say from this run whether `gauss_kl`'s CV R² of 0.62 ALSO collapses when the stylized personas are dropped. If it does, the right read is "all distance predictors of marker-leakage transfer are basically detecting the stylized split, and the result generalizes poorly." If it doesn't, the right read is "activation geometry captures the within-nonstylized signal that output-distribution JS misses." This is the natural follow-up — small extension to the existing script, no new GPU time needed (the residual streams are cached).

## Reproducibility

**Parameters:** Phase 1 (CPU sweep) and Phase 2 (GPU JS predictor) parameters:

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B` |
| Phase 1 driver | `scripts/issue511_probe_count_sweep.py --mode full` (committed at SHA `35c05ef5c`) |
| Phase 1 hardware | VM, 29 cores, no GPU; ~36h wall |
| Phase 1 N grid | {25, 50, 100, 200, 350, 500} |
| Phase 1 R (replicates per N) | 10 |
| Phase 1 layers | L19, L20, L21, L22, L23, L24 (last-prompt-token extraction) |
| Phase 1 metrics | gauss_kl, mmd, wass2, cosine |
| Phase 1 epochs | loc-arm {1, 2, 3, 5} |
| Phase 1 output | `eval_results/issue_522/probe_count_sweep_results.json` (6480 rows) |
| Phase 2 driver | `scripts/issue522_js_predictor.py` + `scripts/issue522_js_regress.py` (committed at SHA `eea8f04d4`) |
| Phase 2 hardware | 1× H100, ~3d20h wall |
| Phase 2 probes per persona | 200 (from `eval_results/issue_502/probes_500.json`) |
| Phase 2 R (samples per probe per persona) | 8 |
| Phase 2 sampling | temperature 1.0, ≤256 tokens, deterministic seed 0 |
| Phase 2 logprob cache | 16 cond_ids × 200 probes × 4 epochs × 8 responses → 409,600 entries |
| Phase 2 smoke gates | max-diagonal JS = 0 nats; max-symmetry residual = 0 nats |
| Phase 2 regression | length-partial Spearman ρ + LOCO CV R²; n_boot = 2000; seed = 42 |
| Estimator citation | Amini/Vieira/Cotterell 2025 arXiv 2504.10637 (Rao-Blackwellized sequence-level KL/JS) |
| Hydra config | n/a — these are direct scripts, not Hydra runs |

**Artifacts:**

- Phase 1 sweep results (6480 rows + plateau verdicts): `eval_results/issue_522/probe_count_sweep_results.json` ([blob @ abfcf15ec](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/eval_results/issue_522/probe_count_sweep_results.json))
- Phase 1 plateau-gate provenance: `eval_results/issue_522/repro_gate.json`, `eval_results/issue_522/cache_verify.json`
- Phase 2 JS matrix (16×16 cells, full + per-probe arrays): `eval_results/issue_522/js_matrix.json` ([blob @ abfcf15ec](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/eval_results/issue_522/js_matrix.json))
- Phase 2 regression rows (full + nonstylized × ep {1,2,3,5}): `eval_results/issue_522/js_regression.json` ([blob @ abfcf15ec](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/eval_results/issue_522/js_regression.json))
- Phase 2 cache schema: `eval_results/issue_522/cache_schema.json`
- Phase 2 per-position teacher-forced log-prob cache (1.68 GB, 409,600 entries): [HF data repo `issue522_full_response_js/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7d188c2e60baa52cad89805ed67030a8df4ce78b/issue522_full_response_js/)
- ΔG target (regressed against): `eval_results/issue_474/cross_eval/loc_ep{1,2,3,5}/G_logprob_matrix.json` (from parent task #474)
- Figures: [`figures/issue_522/`](https://github.com/superkaiba/explore-persona-space/tree/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/) — hero, by_panel_epoch, metric_compare each as `.png` + `.pdf` + `.meta.json`

**Context:** Task created 2026-06-08; Phase 1 launched 2026-06-12 (VM), Phase 2 launched 2026-06-09 (pod-522). Lineage: parent #511 (descope-recovery) → #502 (16-persona panel + probes) → #474 (ΔG target matrix) → #406 (persona configs). Origin prompt: "finish #511's dropped pieces" — full plan in `tasks/interpreting/522/plans/plan.md`.

**Compute:** Phase 1 = ~36h on the VM (29 CPU cores, no GPU). Phase 2 = ~3d20h on `pod-522` (1× H100). Pod auto-terminated at Step 8 of `/issue`. The Phase 2 GPU wall time is roughly 10-20× a tight-implementation floor on this workload (batch-1 forwards through a 7B model, with full-vocab per-position log-softmax shipped to CPU for the JS reduction) — see `.claude/rules/code-style.md` § compute-throughput discipline; a follow-up that batches data-parallel forwards and keeps the reduction GPU-resident would cut wall time by ~10×.

**Code:**

- Phase 1 sweep driver: [`scripts/issue511_probe_count_sweep.py`](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/scripts/issue511_probe_count_sweep.py)
- Phase 2 JS predictor: [`scripts/issue522_js_predictor.py`](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/scripts/issue522_js_predictor.py)
- Phase 2 regression: [`scripts/issue522_js_regress.py`](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/scripts/issue522_js_regress.py)
- Figure-generation: [`scripts/issue522_make_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/scripts/issue522_make_figures.py)
- Reuse: `scripts/issue493_extraction_metric_bakeoff.py` (the ridge fit, `_load_G`, `_pairs`, the length-partial Spearman); `scripts/issue444_persona_distance_topic.py` (the JS estimator's response-sampling and per-position JS reduction)
- Persona configs: [`src/explore_persona_space/experiments/i406_conditions.py`](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/src/explore_persona_space/experiments/i406_conditions.py)
- Git SHA pinning every figure URL: `abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6`
- Reproduce snippet:

  ```bash
  # Phase 1 sweep (CPU, ~36h):
  uv run python scripts/issue511_probe_count_sweep.py --mode full \
      --metrics gauss_kl,mmd,wass2,cosine --n-grid 25,50,100,200,350,500 \
      --r 10 --epochs 1,2,3,5 --layers 19,20,21,22,23,24

  # Phase 2 JS predictor (1× H100, ~3-4 days):
  uv run python scripts/issue522_js_predictor.py \
      --personas A1,A2,A3,A4,A5,B1,B2,B3,B4,B5,C1,D1,D2,D3,D4,D5 \
      --probes 200 --r 8 --max-new-tokens 256 --seed 0

  # Phase 2 regression (CPU, seconds):
  uv run python scripts/issue522_js_regress.py --epochs 1,2,3,5 --n-boot 2000 --seed 42
  ```

**Follow-ups worth queueing.**

1. **Run the activation predictors on the same nonstylized subpanel.** The natural diagnostic: do `gauss_kl` / `mmd` / `wass2` / `cosine` ALSO collapse on the 156-pair nonstylized subpanel, or do they keep their CV R²? Extension to `issue511_probe_count_sweep.py` to call `bakeoff._pairs(cond_ids, nonstylized_only=True)`; should run in hours on CPU using the cached residuals — no new training, no new GPU.
2. **Expand the panel of "normal" personas.** The nonstylized panel has 13 personas, dominated by register / framing variants from a single base. A larger / more diverse pool of everyday personas (e.g. occupations, conversational styles) would tell us whether the JS predictor really has zero within-nonstylized signal or whether 13 personas is just too small a needle.
3. **Search-best per-N grid (the full ~1500-cell sweep)** — deferred from #511, still deferred. Needs the truncated-eig / GPU-port optimization the planner flagged, not an as-is re-run.
