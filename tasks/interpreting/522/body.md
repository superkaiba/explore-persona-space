---
title: The full-response JS predictor scores well on the full panel largely because
  of the stylized vs nonstylized split (LOW confidence)
kind: analysis
tags: []
created_at: '2026-06-08T23:16:05Z'
has_clean_result: true
parent_id: 511
---
# The full-response JS predictor scores well on the full panel largely because of the stylized vs nonstylized split (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** The output-distribution JS predictor "works" on the full 16-persona panel, but once I drop the three theatrical personas (pirate / stand-up / villain) it stops generalizing — so most of the full-panel score was the stylized-vs-rest gap doing the work, not a continuous distance gradient.

**Takeaways.**
- Phase 1 closed parent #511's loose ends — `gauss_kl` at layer 22 holds at out-of-sample CV R² ≈ 0.62 with the full ridge (N=500, R=10), and MMD / Wasserstein-2 / cosine all sit at 0.56-0.58. The plateau-on-N verdict holds at 94 of 108 (layer × metric × epoch) cells; the other 14 are `intermediate` (mostly L23-L24 wass2 and the L0/L11/L24 cosine sentinels). No surprise there.
- The full-response JS predictor lands at CV R² ≈ 0.24 on the full panel — roughly one-third of all four activation predictors.
- When I drop the three high-stylization personas, JS out-of-sample CV R² falls to ≈ −0.02 (95% panel-CI [−0.15, 0.08], straddles zero — so no DEMONSTRATED out-of-sample skill at this panel size, but I can't claim a negative either). The in-sample length-partialled ρ on that subpanel is still +0.27 (95% panel-CI [0.11, 0.42]), a modest non-zero correlation that just doesn't survive leave-one-cond-out CV. JS retains weak within-nonstylized signal that fails to generalize.

**How this updates me.** Less optimistic that "cheap base-model JS on the output distribution" is a real substitute for activation geometry — at least for predicting marker-leakage transfer. Activation predictors beat JS on the full panel at ~2.4× CV R²; whether they ALSO collapse on the nonstylized subpanel is the natural next test and hasn't been run yet. What would move me back: a clean within-nonstylized JS result on a richer panel of "normal" personas.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

Parent task [#511](https://eps.superkaiba.com/tasks/511) ran a multi-metric predictor sweep to find which base-model distance between two personas best predicts how the marker-leakage signal transfers between them — using the ΔG matrix from [#474](https://eps.superkaiba.com/tasks/474) as the target. Phase 1 of #511 plateaued the headline cell (Gaussian-KL at residual-stream layer 22, predicting count-style ΔG, out-of-sample CV R² ≈ 0.62) but an autonomous compute-deviation cut wass2, ridge at N=500 × R=10, the loc-epoch robustness sweep, and a full-response JS baseline. This task ran the dropped pieces. Two parts:

- **Phase 1 (CPU, ~36h on the VM)** — restored full scope of the activation-space sweep: wass2 added, ridge at N=500 × R=10, loc-arm epochs {1, 2, 3, 5}.
- **Phase 2 (GPU, 1× H100, ~3d20h)** — built the full-response Rao-Blackwellized JS predictor from scratch (the deprecated single-next-token JS in #511's plan does NOT count). Compared it against the four activation metrics on the same 16-persona panel.

The reason this question matters: if a cheap output-distribution metric (one base-model pass; no residual-stream extraction; no cloud-aware ridge) predicted leakage as well as the activation geometry, that would be a much simpler causal probe. The question I had going in was "does it?"

The construct I'm measuring is **predictor-vs-leakage skill** — out-of-sample CV R² and length-partialled Spearman ρ — on the 240 ordered pairs of 16 base-model personas. Higher is better, both metrics agree, and the CIs are panel-row bootstraps (n_boot=2000, seed=42) over the 240 pairs. A separate JS-estimator Monte-Carlo CI (probe-bootstrap, also n_boot=2000) is reported alongside to confirm that the JS sampling budget is not the bottleneck.

### What I ran

**Phase 1: activation-predictor sweep, full scope.** I re-ran `scripts/issue511_probe_count_sweep.py` with the descoped axes restored:

- Metrics: `gauss_kl`, `mmd`, `wass2`, `cosine` (Phase 1 of #511 had dropped wass2).
- Probe counts (N): {25, 50, 100, 200, 350, 500} (capped at 350 in #511).
- Replicates (R) at each (N, layer, metric): 10 (was 5).
- Layers: L19-L24 (the band #502 identified as best — last-prompt-token extraction).
- Loc-arm epochs: {1, 2, 3, 5} (was loc_ep1 only).
- Output: 6480 fit rows in `eval_results/issue_522/probe_count_sweep_results.json`; plateau verdict per cell.

**Phase 2: full-response JS predictor.** I built a fresh script (`scripts/issue522_js_predictor.py`) implementing the canonical full-response Rao-Blackwellized JS estimator (Amini/Vieira/Cotterell 2025, arXiv 2504.10637), which is the only operationalization the project rule sanctions (`.claude/rules/persona-distance-metrics.md`; #404 and #458's older single-next-token JS is deprecated). For each of the 16 personas A, sample 8 on-policy responses (temp=1, ≤256 tokens) on each of 200 probes from #502's probe set — that's the FROM side. For each of the 16×16 ordered pairs (A, B), teacher-force each of A's sampled responses through B's persona-conditioned base model, get the exact full-vocab per-position next-token distributions under both A and B, compute the per-position JS, length-normalize, average over positions and responses. Internally the reducer accumulates the JS in nats, then divides by `log(2)` before storing — so every JS value in `js_matrix.json` and every downstream JS number in this writeup is in **base-2 bits** (bounded in [0, 1] for two distributions). Total cache: 409,600 entries (16 cond_ids × 200 probes × 4 epochs × 8 responses, with per-epoch caching for the bystander-direction reuse). Smoke gates: max-diagonal JS = 0 bits; max-symmetry residual = 0 bits.

**The regression step.** Length-partial Spearman ρ + LOCO (leave-one-cond-out) CV R² of `M_js = 1 − JS` against the #474 ΔG matrix (the matrix's `delta_g` field is in nats; only the JS predictor input is in bits). Two panels: `full` (240 ordered pairs across 16 personas) and `nonstylized` (156 pairs across the 13 personas that exclude the three high-stylization arms — A3 Pirate captain, A4 Stand-up comedian, A5 Villainous mastermind). Four target epochs (1, 2, 3, 5). Panel-row bootstrap with n_boot=2000, seed=42 for 95% CIs on the panel-row CV R²; a separate JS-estimator probe-bootstrap (also n_boot=2000) gives the Monte-Carlo CI on the JS values themselves.

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

Restoring the full N=500 × R=10 ridge across layers 19-24, metrics {gauss_kl, mmd, wass2, cosine}, and loc-arm epochs {1, 2, 3, 5} gives no surprises. The L22 / gauss_kl headline cell from #511 still plateaus at CV R² ≈ 0.62 at epoch 1 (best subset replicate 0.6196; mean across 10 subsets 0.6174), with the plateau verdict holding at N=500 (δ from N=350→500 is +0.00266, well inside σ_ref ≈ 0.00361 — the verdict is `plateau`). The other three activation metrics at the same L22 / N=500 / epoch 1 cell:

| Metric | Mean CV R² (10 subsets) | Best replicate |
|---|---|---|
| Gaussian KL | 0.617 | 0.620 |
| MMD | 0.580 | 0.583 |
| Wasserstein-2 | 0.568 | 0.572 |
| Cosine | 0.557 | 0.560 |

So at epoch 1 the metric ranking I expected from #511's smaller ridge holds: gauss_kl edges MMD by ~0.04 CV R², which edges wass2 and cosine by ~0.02 each. Across loc-arm epochs the picture is the same shape but lower altitude: gauss_kl at L22 drops from 0.617 (ep1) to ~0.40-0.45 (ep2/3/5). The "L22/gauss_kl wins" ordering is not perfectly stable across epochs — at ep3, L21/MMD (0.4356) edges L22/gauss_kl (0.4345) by ~0.001 CV R², and at ep5 they tie at 0.4482. So the right read is "gauss_kl wins decisively at ep1; at later epochs MMD at L21 is competitive with gauss_kl at L22 within noise." The plateau-on-N verdict holds at 94 of 108 (layer × metric × epoch) cells; the remaining 14 sit at `intermediate` (mostly L23/wass2 × all four epochs, L24/wass2 ep2, L24/cosine ep1/2, plus sentinel L0/L11 cosine cells — none drop below plateau). So "N=500 is enough" is true for the headline and most of the sweep, but a sweep extending to higher layers or non-band metrics may need a larger N.

This was always a "verify the existing line still holds" arm, not a finding in itself. It came back PASS. There is no figure for this arm — the per-cell CV R² table is the artifact, in `probe_count_sweep_results.json`.

#### A full-response JS predictor does correlate with leakage transfer on the full panel — but at one-third of the activation predictors' CV R²

![Most of the JS predictor lives in the stylized-vs-rest gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/hero.png)

> **Figure.** *Most of the full-response JS signal lives in the gap between stylized and nonstylized personas, not in a continuous distance gradient.* Each dot is one of 240 ordered persona pairs (a → b). X-axis: full-response JS divergence between base-model output distributions under persona a and persona b, in base-2 bits (bounded [0, 1]). Y-axis: marker-leakage transfer ΔG between the same pair, from #474's loc-arm checkpoint at epoch 1, in nats. Blue points are pairs where neither persona is stylized (156 of 240); red points touch one of A3 / A4 / A5 (84 of 240). The red points span JS = 0.021-0.104 bits (median 0.084) — and crucially the 6 within-stylized ordered pairs (A3↔A4, A3↔A5, A4↔A5) sit at JS = 0.021-0.029 bits, squarely inside the low-JS band the rest of the figure assigns to blue. The header statistics (ρ = 0.54, CV R² = 0.24) are computed on the full panel and length-partialled.

The number reads positive on the full panel — Spearman ρ = 0.54 (95% panel-CI [0.45, 0.63]) and LOCO CV R² = 0.24 (95% panel-CI [0.12, 0.34]), both with p ≪ 0.001 against the null. The JS-estimator Monte-Carlo CI on the same CV R² is much tighter — [0.236, 0.252] — confirming the JS sampling budget (200 probes × 8 responses per persona) is NOT the bottleneck for this measurement; the panel-row CI dominates. So in a vacuum, the answer is "yes, a base-model on-policy JS divergence has skill at predicting how marker-leakage transfers."

But the figure makes the source of that skill obvious. The 78 CROSS-CLUSTER stylized-touching pairs (one stylized, one not) sit at high JS (range 0.030-0.104 bits, median 0.085) and moderate ΔG; the 156 nonstylized-only pairs (blue) sit at low JS (range 2e-08 to 0.097 bits, median 0.019). The full-panel correlation is largely a between-cluster contrast. Splitting "stylized-touching" further: the 6 WITHIN-stylized ordered pairs cluster at JS = 0.021-0.029 bits — overlapping the blue cloud — so calling all 84 red dots "high JS" overstates the picture.

How much of the signal IS the binary? A point-biserial of the stylized-touching indicator against JS gives r = 0.72 (p = 3.4e-40); against ΔG it gives r = -0.61 (p = 3.8e-26). So the binary alone carries most of both axes' variance. The activation-side predictors (next finding) deliver ~2.4× higher CV R² on the same panel — so even if you accept "the JS predictor works on the full panel," it works much less well than the cheaper option you already have if you can read residual streams.

A short aside on the secondary target `g_logprob` (which #474 measures alongside `delta_g`): JS regressed against `g_logprob` instead of `delta_g` gives the same full→nonstylized collapse pattern at slightly lower magnitudes — full ep1 cv_r2_glog ≈ 0.125 (vs 0.241 for ΔG); nonstylized ep1 cv_r2_glog ≈ -0.016 (vs -0.022). So the read above is not a ΔG-specific artifact.

The relevant CV R² comparison on the same panel × epoch cell:

![Activation-geometry predictors beat full-response JS](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/metric_compare.png)

> **Figure.** *On the full 240-pair panel, all four activation-geometry predictors (cloud-aware ridge at layer 22) achieve more than twice the out-of-sample CV R² of the output-distribution JS predictor.* X-axis is the predictor; y-axis is leave-one-cond-out CV R² against the same ΔG target (loc-arm, epoch 1, full 240 pairs). Error bars on the JS bar are the panel-row bootstrap 95% CI; error bars on the activation bars are subset-σ across 10 ridge replicates at N=500 — different uncertainty quantifications, which is why the JS error bar is dramatically wider. The JS predictor was a 1× H100 base-model forward-pass job; the activation predictors run on cached residual streams plus a cloud-aware ridge fit on CPU.

#### Drop the stylized personas, and the JS predictor loses out-of-sample CV-R² skill

The full-panel result already hinted that the JS signal was carried by the stylized split. Restricting the panel to the 13 nonstylized personas (156 pairs) confirms it for out-of-sample CV R² — but with an important nuance the headline obscures.

![Drop stylized → CV R² collapses; ρ stays positive](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/by_panel_epoch.png)

> **Figure.** *Removing the three stylized personas pushes the JS predictor's out-of-sample CV R² to ≈ 0 with all four panel-row CIs straddling zero, while the in-sample length-partialled ρ stays positive at ~0.20-0.27.* Left panel: length-partialled Spearman ρ between JS and ΔG. Right panel: leave-one-cond-out CV R² for the same. Blue bars are the full 240-pair panel; orange bars are the nonstylized 156-pair subpanel. Source-persona training amount on the x-axis. Error bars are panel-row bootstrap 95% CIs (n_boot = 2000, seed = 42). The dashed grey line on the CV-R² panel marks zero. Three stylized personas excluded from the orange subpanel: Pirate captain (A3), Stand-up comedian (A4), Villainous mastermind (A5).

At epoch 1, the full panel reads ρ = 0.54 / CV R² = 0.24; the nonstylized subpanel reads ρ = 0.27 (95% panel-CI [0.11, 0.42]) / CV R² = -0.022 (95% panel-CI [-0.15, 0.08]). The pattern holds across epochs 2, 3, 5: full-panel CV R² stays positive (0.10-0.13) but nonstylized CV R² is centered slightly below zero at every epoch (range -0.036 to -0.022, all four panel-row CIs straddle zero with the upper edge under +0.08). Two honest readings live in those numbers, and the body should carry both:

- **The headline read:** out-of-sample LOCO CV R² is centered at zero or slightly negative with CIs straddling zero on the nonstylized subpanel. So there is no demonstrated out-of-sample skill at this panel size — JS doesn't generalize on the nonstylized panel under leave-one-cond-out.
- **The nuance:** the in-sample length-partialled ρ on the same subpanel is +0.27 (95% panel-CI [0.11, 0.42]) — a modest non-zero correlation. And the raw, un-partialled within-nonstylized Spearman ρ(JS, ΔG) is -0.26 (p = 0.0009, n = 156) — the RIGHT sign (high JS → low transfer), statistically non-zero. So JS retains weak in-sample signal on the nonstylized subpanel; that signal just doesn't survive LOCO with only 13 personas to leave out. "Zero out-of-sample CV-R² skill" is not the same as "no signal at all," and the body in round 1 conflated them.

A separate caution: LOCO on 13 highly correlated nonstylized personas is itself a stress test, not a clean held-out benchmark, so a negative CV R² there is consistent with both "the predictor genuinely doesn't generalize" and "the LOCO scheme has too little held-out signal on a panel this small." The right next step is to (a) widen the panel and (b) check whether the activation predictors ALSO collapse here.

The within-stylized 6-pair raw ρ(JS, ΔG) is -0.48 (p = 0.34, n = 6) — the right sign, suggesting JS may carry a within-stylized gradient that the tiny sample can't certify. Codex also pointed out 4 nonstylized blue pairs at JS > 0.08 (e.g. (A2, D5) JS=0.097 bits / ΔG=11.44 nats, (D2, A2) JS=0.088 bits / ΔG=20.69 nats) — so "nonstylized JS is confined to 0-0.04" understated the panel; 24 of 156 nonstylized ordered pairs have JS > 0.04 bits.

<details>
<summary>Five raw pair-level rows (JS, ΔG) at loc-arm epoch 1 — cherry-picked for illustration</summary>

All values are from `eval_results/issue_522/js_matrix.json` (JS, in base-2 bits) and `eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json` (ΔG, in nats), indexed as `G[source][target]["delta_g"]` — matching the orientation the regression code uses (`scripts/issue522_js_regress.py:109`). For reference, the self-strength diagonals are ΔG[A1][A1] = 26.02, ΔG[B1][B1] = 24.64, ΔG[D2][D2] = 23.57, ΔG[D3][D3] = 25.48.

| Source persona | Target persona | JS (bits) | ΔG (nats) | Class |
|---|---|---|---|---|
| Helpful assistant (A1) | Bare question (B1) | 2.0e-08 | 11.19 | nonstylized |
| Helpful assistant (A1) | Software engineer (A2) | 0.0713 | 13.78 | nonstylized |
| Helpful assistant (A1) | Pirate captain (A3) | 0.0768 | 6.88 | stylized-touching |
| Pirate captain (A3) | Stand-up comedian (A4) | 0.0290 | 2.06 | within-stylized |
| Casual register (D2) | Indirect framing (D3) | 0.0247 | 21.75 | nonstylized |

Two qualitative reads survive the corrected numbers:

- **The continuous distance gradient is weak for normal pairs.** The (A1, B1) pair has JS = 2e-08 bits — essentially zero — yet ΔG = 11.19, only ~43% of A1's self-strength (26.02). So even at zero JS, transfer is far from complete. The (A1, A2) pair has JS = 0.071 (much larger) and ΔG = 13.78 — slightly LARGER, not smaller — running counter to the "more JS → less transfer" story.
- **Within-stylized pairs have low JS but the lowest ΔG.** (A3, A4) has JS = 0.029 bits and ΔG = 2.06 — small JS, very small transfer — the right SIGN for the within-stylized triangle (ρ = -0.48, n=6, p=0.34), but the panel is too small to certify.

The (A1, B1) row in particular falsifies the round-1 framing "look, zero JS predicts complete transfer here": A1→A1 transfers 100%, A1→B1 transfers 43%. The continuous gradient is real on the within-nonstylized subpanel (raw ρ = -0.26, p = 0.0009) but small in magnitude.
</details>

**Note on the activation predictors and the same subpanel.** Phase 1's sweep did NOT run a nonstylized-only ridge — `bakeoff._pairs` always builds the full 240-pair panel for that script. So I can't say from this run whether `gauss_kl`'s CV R² of 0.62 ALSO collapses when the stylized personas are dropped. The "activation predictors are better" claim above is a full-panel claim, NOT a stress-test claim. If they ALSO collapse on the nonstylized panel, the right read is "all distance predictors of marker-leakage transfer are basically detecting the stylized split, and the result generalizes poorly." If they don't, the right read is "activation geometry captures the within-nonstylized signal that output-distribution JS misses." This is the natural follow-up — small extension to the existing script, no new GPU time needed (the residual streams are cached).

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
| Phase 1 output | `eval_results/issue_522/probe_count_sweep_results.json` (6480 rows + plateau verdicts; 94/108 cells `plateau`, 14/108 `intermediate`) |
| Phase 2 driver | `scripts/issue522_js_predictor.py` + `scripts/issue522_js_regress.py` (committed at SHA `eea8f04d4`) |
| Phase 2 hardware | 1× H100, ~3d20h wall |
| Phase 2 probes per persona | 200 (from `eval_results/issue_502/probes_500.json`) |
| Phase 2 R (samples per probe per persona) | 8 |
| Phase 2 sampling | temperature 1.0, ≤256 tokens, deterministic seed 0 |
| Phase 2 JS unit | base-2 bits (predictor accumulates in nats internally, then divides by `log(2)` before storing in `js_matrix.json`; bounded in [0, 1]) |
| Phase 2 logprob cache | 16 cond_ids × 200 probes × 4 epochs × 8 responses → 409,600 entries |
| Phase 2 smoke gates | max-diagonal JS = 0 bits; max-symmetry residual = 0 bits |
| Phase 2 regression | length-partial Spearman ρ + LOCO CV R²; panel-row bootstrap n_boot = 2000; JS-estimator probe-bootstrap n_boot = 2000; seed = 42 |
| Phase 2 mc_ci (JS-estimator) | full-panel ep1 CV R² mc_ci = [0.236, 0.252] (panel-row CI [0.121, 0.338] is ~14× wider — JS sampling budget is NOT the bottleneck) |
| ΔG target unit | nats (from `eval_results/issue_474/cross_eval/loc_ep*/G_logprob_matrix.json`, field `G[source][target]['delta_g']`) |
| Estimator citation | Amini/Vieira/Cotterell 2025 arXiv 2504.10637 (Rao-Blackwellized sequence-level KL/JS) |
| Hydra config | n/a — these are direct scripts, not Hydra runs |

**Artifacts:**

- Phase 1 sweep results (6480 rows + plateau verdicts): `eval_results/issue_522/probe_count_sweep_results.json` ([blob @ abfcf15ec](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/eval_results/issue_522/probe_count_sweep_results.json))
- Phase 1 plateau-gate provenance: `eval_results/issue_522/repro_gate.json`, `eval_results/issue_522/cache_verify.json`. The per-cell plateau verdicts live under `plateau_verdict` / `plateau_verdict_glog` keyed by `last_prompt__L<layer>__<metric>__raw|loc|ep<epoch>` — the headline L22/gauss_kl/ep1 row gives `verdict: plateau`, `delta_350_to_500 = +0.00266`, `sigma_ref = 0.00361`.
- Phase 2 JS matrix (16×16 cells, full + per-probe arrays, all in base-2 bits): `eval_results/issue_522/js_matrix.json` ([blob @ abfcf15ec](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/eval_results/issue_522/js_matrix.json))
- Phase 2 regression rows (full + nonstylized × ep {1,2,3,5}, with `mc_ci` JS-estimator probe-bootstrap alongside `panel_ci` panel-row bootstrap; `rho_glog` + `cv_r2_glog` rows for the secondary `g_logprob` target): `eval_results/issue_522/js_regression.json` ([blob @ abfcf15ec](https://github.com/superkaiba/explore-persona-space/blob/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/eval_results/issue_522/js_regression.json))
- Phase 2 cache schema: `eval_results/issue_522/cache_schema.json`
- Phase 2 per-position teacher-forced log-prob cache (1.68 GB, 409,600 entries): [HF data repo `issue522_full_response_js/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7d188c2e60baa52cad89805ed67030a8df4ce78b/issue522_full_response_js/)
- ΔG target (regressed against): `eval_results/issue_474/cross_eval/loc_ep{1,2,3,5}/G_logprob_matrix.json` (from parent task #474)
- Figures: [`figures/issue_522/`](https://github.com/superkaiba/explore-persona-space/tree/abfcf15ec31dcbba9cd5d43b444c94d6a783e9e6/figures/issue_522/) — hero, by_panel_epoch, metric_compare each as `.png` + `.pdf` + `.meta.json`

**Context:** Task created 2026-06-08; Phase 1 launched 2026-06-12 (VM), Phase 2 launched 2026-06-09 (pod-522). Lineage: parent #511 (descope-recovery) → #502 (16-persona panel + probes) → #474 (ΔG target matrix) → #406 (persona configs). Origin prompt: "finish #511's dropped pieces" — full plan in `tasks/interpreting/522/plans/plan.md`.

**Compute:** Phase 1 = ~36h on the VM (29 CPU cores, no GPU). Phase 2 = ~3d20h (≈94h) on `pod-522` (1× H100). Pod auto-terminated at Step 8 of `/issue`. Per `.claude/rules/code-style.md` § compute-throughput discipline, the tight-implementation floor on this workload (batch-1 forwards through a 7B model with full-vocab per-position log-softmax shipped to CPU for the JS reduction) is ~4-6h, so the actual wall is ~15-23× that floor; a follow-up that batches data-parallel forwards and keeps the reduction GPU-resident would cut wall time by ~10×.

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
2. **Expand the panel of "normal" personas.** The nonstylized panel has 13 personas, dominated by register / framing variants from a single base. A larger / more diverse pool of everyday personas (e.g. occupations, conversational styles) would tell us whether the JS predictor really has zero within-nonstylized signal or whether 13 personas is just too small a needle for LOCO.
3. **Search-best per-N grid (the full ~1500-cell sweep)** — deferred from #511, still deferred. Needs the truncated-eig / GPU-port optimization the planner flagged, not an as-is re-run.
