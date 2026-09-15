# Matryoshka tier concordance after variance matching

User-requested reanalysis, 2026-09-14. The original coarsest-versus-rest advantage reverses when matching features within quintiles of answer variance along their decoder directions.

## Headline results

- Activity quintiles, the published control: **+0.27808 (95% CI [+0.26293, +0.29239])**.
- Variance quintiles: **-0.07283 (95% CI [-0.09038, -0.05429])**.
- Crossed variance and activity quintiles: **-0.02676 (95% CI [-0.05109, -0.00437])**.
- Unmatched reference: +0.20958 (95% CI [+0.19306, +0.22510]).

Positive concordance means a coarse-tier feature is predicted better than a comparable finer-tier feature. Values are centered at chance; for example, the variance-matched probability is 0.42717. The activity-only point estimate and its full confidence interval reproduce the published artifact to absolute tolerance 1e-12.

## Method

The model, layer-20 LMSYS Matryoshka dictionary, 16,384 feature IDs, fitted map and per-direction held-out R2 are unchanged. Variance is the sample variance of each unit decoder direction's projection of the mean layer-20 answer state, calculated on the original 6,000 scoring answers. It is not SAE activation variance or the prediction R2 being ranked. We verify all 6,000 row IDs are present exactly once, the original result's input hashes agree, and the SAE weights match the original pinned Hub revision.

We retain the parent's pair-weighted concordance and minimum of 120 total features per matching cell. The primary variance matching uses the same five quantile bins as the original activity matching. Joint matching crosses the two sets of marginal quintiles. Confidence intervals use the original 2,000 feature-bootstrap draws, recomputing bin edges on each draw; the four controls share draws within a contrast. The batched estimator is checked against the unchanged original implementation on every point estimate and three resamples per contrast/control. Five additional tests verify explicit pair counting, score ties, quantile-edge ties, and empty comparisons.

## Matching resolution and coverage

Variance-only matching retains all 16,384 features, including all 1,640 coarsest features. Its point estimates are -0.07283, -0.19835, and -0.28834 for 5, 10, and 20 bins. The sign survives finer variance bins, but the magnitude depends strongly on bin width. Quantile matching leaves within-bin differences; it is not exact equality of variance.

Joint 5x5 matching compares 13,782 features, including all 1,640 coarsest features and 12,142 from the other tiers. Joint 10x10 compares 9,081 features. Joint 20x20 compares only 459 features (433 coarsest and 26 other-tier), so its positive estimate is a different, poorly supported comparison and should not be used as a robustness claim for the original population. Exact cell counts and residual within-cell log-variance differences are in matching_diagnostics.json.

The feature bootstrap conditions on this fixed map and answer sample and does not account for dependence among correlated SAE features. These are conditional associations, not causal effects of granularity. The current evidence does not support claiming that coarser Matryoshka features are better predicted independently of variance.

## Reproduction and artifacts

- Driver: scripts/issue1482_tier_variance_control.py
- Plotter: scripts/issue1482_tier_variance_plot.py
- Original estimator and comparison: scripts/issue1482_tier_concordance.py; eval_results/issue_1482/tier_concordance.json
- summary.json: every requested binary tier contrast, control, confidence interval and sensitivity estimate.
- panel.npz: feature IDs, frozen R2, tier, activity, projection variance and scoring row IDs.
- bootstrap_*.npz: raw bootstrap draws for all four controls.
- source_manifest.json and sae_provenance.json: pinned source revisions and verified file hashes.
- preparation.json: construction checks and derived-input hash.
- complete.json, run.log, process_exit.txt: completion evidence.

Run from the existing root environment, substituting the output directory that contains source_manifest.json:

```bash
uv run python /home/thomasjiralerspong/wt-matryoshka-variance-match/scripts/issue1482_tier_variance_control.py \
  --source-root /home/thomasjiralerspong/explore-persona-space \
  --out /home/thomasjiralerspong/wt-matryoshka-variance-match/eval_results/issue_1482/variance_matched_20260914 \
  --sae-weights /home/thomasjiralerspong/explore-persona-space/data/issue_1482/mtry_sae/lmsys/matryoshka/k-100/sae_weights.safetensors
```

For a statistics-only repeat, pass --prepared-panel pointing to the verified panel.npz and choose a new output directory. Original inputs are staged from the pinned Hub revision recorded in the manifest.

The paper has not been edited in this reanalysis.
