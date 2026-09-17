# Independent digitization and overlap review — issue 2673

**APPROVE the corrected current artifact for a descriptive, approximate comparison.** All 20 bar coordinates/rates, four Qwen input hashes, and eight correlation pairs pass independent checks. This does not approve a matched replication or causal interpretation.

Reviewed UTC: 2026-09-17T23:22:31.403907+00:00

## Pinned evidence

- [Published DeepSeek raster](https://arxiv.org/html/2609.10883v1/images/selectivity/fxbc_bloom_deepseek_base_grid.png), Figure 26 / Appendix C.5 of [arXiv v1](https://arxiv.org/html/2609.10883v1). Actual downloaded PNG SHA256: `0d80dd99ecfaf97e049215a1cca04c65232e466009999845186230b70e3f483f`; image size 1424 × 1483 RGB.
- Reviewed script: `/home/thomasjiralerspong/.codex/worktrees/story-persona-qwen38-pilot-20260917/scripts/story_persona_deepseek_published.py`; SHA256 `ad1d35178e63ccaf855ffe1c7cda65d44677cca87576930348157d2d112a2609`.
- Reviewed JSON: `/home/thomasjiralerspong/.codex/worktrees/story-persona-qwen38-pilot-20260917/eval_results/issue_2673/deepseek_comparison/published_rates_and_overlap.json`; SHA256 `92bb13ba27df98fd8852949a8628f882d37b1f88a0d11627076b864cb487bc34`.
- Final-block Qwen source: `/home/thomasjiralerspong/.codex/worktrees/story-persona-qwen38-pilot-20260917/eval_results/issue_2673/no_centering/block_63.json`; SHA256 `896f38640923f26a956ee730bb2f1a1ca79f4b3bccc54cad21e49dfa5247c5b7`. All four layer-file hashes match the comparison artifact.

## Independent pixel method and complete coverage

I did not invoke the publisher's digitization routine. I extracted horizontal segments with 15-pixel morphological opening of the full-image border-color masks, labeled connected components, and assigned segments to bars by panel and horizontal position. Each of the 20 bars had one unique top-border row; the multiple visible segments of each border agreed. The original majority-fill detector missed one short bar obscured by markers; that failure is corrected in the reviewed script.

The image has long black axis rows at 51–52 and 490–491 for HHH, and 727–728 and 1166–1167 for Fred. The adopted inner-edge mappings 52→100%, 490→0%, and 728→100%, 1166→0% therefore agree with the raster axes to one pixel. The 438-pixel scale converts the stated ±1.5-pixel bound into ±0.3425 percentage points. This is a raster-reading allowance, not a statistical confidence interval or uncertainty about the paper's rollout rates.

Visual inspection confirms green denotes the behavior associated with Helpful and pink the behavior associated with the named opposing character. Circular and square dots denote bees/crows runs; their extrema and error bars were not used as bar means. All entries below agree exactly with the corrected JSON.

| Persona | Opposing character | Helpful top y | Helpful mean | Other top y | Other mean |
|---|---|---:|---:|---:|---:|
| HHH | dismissive | 299 | 43.607% | 458 | 7.306% |
| HHH | sarcastic | 226 | 60.274% | 471 | 4.338% |
| HHH | saboteur | 252 | 54.338% | 442 | 10.959% |
| HHH | peer | 338 | 34.703% | 465 | 5.708% |
| HHH | help_seeker | 365 | 28.539% | 474 | 3.653% |
| FRED | dismissive | 1109 | 13.014% | 909 | 58.676% |
| FRED | sarcastic | 1131 | 7.991% | 989 | 40.411% |
| FRED | saboteur | 1048 | 26.941% | 1039 | 28.995% |
| FRED | peer | 1045 | 27.626% | 1137 | 6.621% |
| FRED | help_seeker | 1050 | 26.484% | 1151 | 3.425% |

Resolved finding: HHH/help-seeker/Other was originally read as y=479 (2.511%). Its visible horizontal border is y=474, including an uninterrupted segment x=1258…1293, giving 3.653%. The 1.142-percentage-point original error exceeded the declared bound. The corrected artifact uses y=474. The four-pair comparison excludes help-seeker and was unaffected.

## Independent four-pair calculation

I selected the `default` row and the dismissive, sarcastic, saboteur, and peer columns directly from each saved Qwen cosine matrix, then recomputed Pearson with `scipy.stats.pearsonr` and Spearman with `scipy.stats.spearmanr`. These are different entry points from the publisher's NumPy/rankdata implementation. All results agree within 1e-12.

| Layer | Raw Pearson r | Raw Spearman rho | Uncentered-whitened Pearson r | Uncentered-whitened Spearman rho |
|---:|---:|---:|---:|---:|
| 15 | -0.154568 | -0.6 | -0.407060 | -0.6 |
| 31 | 0.150026 | 0.0 | -0.114959 | 0.0 |
| 47 | 0.296179 | 0.0 | -0.024824 | 0.0 |
| 63 | 0.821738 | 0.8 | 0.507257 | 0.4 |

For block 63, in the stated four-pair order:

- DeepSeek HHH Other rate: `[0.0730593607, 0.0433789954, 0.1095890411, 0.0570776256]`.
- Raw Qwen cosine: `[0.6923460528, 0.5833148811, 0.9038831970, 0.7998522344]`.
- Uncentered-whitened cosine: `[-0.0020634515, 0.0015711631, 0.0128961924, 0.0104682002]`.

The minimum gap between the four outcome rates is 0.013699, exceeding twice the digitization bound (0.006849). Thus independently varying every digitized rate within its declared bound cannot change either reported Spearman coefficient. Evaluating all 16 corners of the ±1.5-pixel outcome box gives raw Pearson r from 0.744463 to 0.881440, and whitened r from 0.398229 to 0.595460. These are corner sensitivity checks, not a proven continuous-box extremum or a sampling CI.

Enumerating all 24 permutations yields absolute-Spearman tails of 8/24 for rho=0.8 and 18/24 for rho=0.4. Even a perfect ranking has minimum two-sided tail 2/24=0.0833 at n=4. These counts demonstrate coarse rank resolution; they are not offered as calibrated hypothesis tests for this selected, approximate comparison. Leave-one-pair-out Pearson values span 0.4647…0.9925 raw and -0.3242…0.7992 whitened, illustrating sensitivity to individual character pairs.

## Interpretation boundary

The default Qwen context is a proxy for a complete few-shot HHH persona, and four Table 7 system personas are proxies for the corresponding story-character types. There are no matched Fred or help-seeker context vectors in this capture. Qwen is not story-finetuned; the outcomes come from story-finetuned DeepSeek-V3.1-Base under a different multi-turn evaluation distribution. The existing whitening is uncentered, regularized, and calibrated on the same rank-deficient Qwen bank. With four aggregate pairs, these numbers support a descriptive exploratory association only; they neither validate a leakage predictor nor establish similarity as a causal mechanism. All these material limitations are represented in the reviewed JSON.

No source, task, model, or compute state was modified during this review; only this review file was written.
