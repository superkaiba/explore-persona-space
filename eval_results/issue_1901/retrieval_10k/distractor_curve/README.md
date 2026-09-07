# Linear retrieval as distractor contexts increase

The full-data linear map's top-1 retrieval declines from 97.35% to 87.26% as
added distractors increase from zero to 19,000. Top-5 declines from 100.00% to
96.28%. All five points reuse existing measurements; this analysis generates
no text, fits no map, and performs no retrieval rescoring.

[Curve](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/1901-retrieval-10k-20260907/figures/issue_1901/retrieval_10k/distractor_curve.png)
· [Vector PDF](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/1901-retrieval-10k-20260907/figures/issue_1901/retrieval_10k/distractor_curve.pdf)
· [Exact plotted values](metrics.csv)

The horizontal axis counts **added distractor contexts beyond the same 942
query targets** retained in every candidate pool. Thus even zero added
distractors means each query competes with 941 incorrect candidates. The earlier
10,000-candidate evaluation is the point at 9,058 added distractors. Coverage is
five of five intended, already measured pool sizes: 942, 1,942, 4,942, 10,000,
and 19,942 total candidates. At those sizes, chance top-1 accuracy is respectively
0.1062%, 0.0515%, 0.0202%, 0.0100%, and 0.0050%.

The dashed circles show top-1 and the dotted squares show top-5 accuracy.
Straight lines connect measurements; there is no fitted curve or extrapolation.
Error bars are the existing pointwise 95% query-bootstrap intervals for top-1
(2,000 draws), conditional on each fixed candidate bank. They do not represent
uncertainty from sampling new distractor banks. The parent summary did not bank
top-5 intervals, so none are shown.

## Matched protocol and provenance

All points use the Qwen2.5-7B-Instruct layer-19 linear map trained on 963,444
contexts, identical saved held-out predictions and 942 deduplicated queries.
Candidate targets are means of five on-policy answer activation vectors.
Deduplication uses exact original answer vectors before averaging. Whitening
uses the same fixed training-answer statistics. Retrieval uses whitened cosine
with two-sided CSLS, K=10, with strict paired-target scoring. CSLS adjustments
are computed for each candidate pool, as in the source evaluations.

Added distractors are nested prefixes of the same deterministic bank; all
19,000 are LMSYS contexts. This measures scaling for that bank and ordering,
not an average across new random candidate pools. The parent nominal pools of
1,000, 2,000, 5,000 and 20,000 each lose 58 duplicate query vectors, giving the
actual candidate counts above. The newer 10,000-candidate pool was constructed
to have exactly 10,000 unique candidates after deduplication.

Sources are `../../singleturn_retrieval_final/summary.json`, selecting
`ridge|avg|keep_one|pool_{1000,2000,5000,20000}` and the strict `whiten_csls`
metrics, plus `../summary.json`, selecting `per_n/963444/ridge/metrics/whiten_csls`.
The shared Hugging Face dataset revision is
`83d249cc9d495ca6f5d10f9156a622bcdca29a19` of
`superkaiba1/explore-persona-space-data`.

The script checks all eight shared input hashes, the full-N prediction hash,
the query-row hash, the whitening configuration, source revision, and exact
reproduction of the original-pool top-1/top-5 metrics before combining results.
An independent audit also checked the exact nested candidate IDs. The parent
query averaging accumulates fresh draws in fp32 and the newer implementation
uses fp64; their original-pool top-1/top-5 and reciprocal-rank metrics agree
exactly. Their mathematical five-answer-mean recipe is the same.

`summary.json` preserves exact values and source summary hashes; the figure
sidecar preserves the plotted rows, summary hash, and standard rendering record.
The earlier [10k evaluation report](../README.md) documents generation and
measurement details, including the baseline and training-size comparisons.

## Reproduction and validation

Run from a project checkout with the existing environment:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run --no-sync python scripts/issue1901_distractor_curve.py
```

The script accepts `--parent`, `--tenk`, `--out-dir`, and `--stem`. Both relative
input paths and an external output directory were exercised successfully.
Ruff passes; an independent code and artifact review verified extraction,
uncertainty semantics, source fingerprints, and exact agreement between the
summary, CSV, and plot sidecar. Its custom-path finding was corrected. Color
and grayscale exports were visually inspected for legibility and clipping.

The mapped thread-cap suite completed with 26 passes and one pre-existing
failure listing 14 unrelated entrypoints. The new script is not flagged. Full
no-flags workflow lint completed with 19 pre-existing errors in unchanged files
and no errors attributable to this analysis. Its completion and payload
attribution are recorded in the task event.

This is an auxiliary analysis; incorporating it into the parent clean result
and manuscript is deferred to the paper-maintenance owner at the next results
update.
