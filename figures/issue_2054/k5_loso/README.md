# K5 leave-one-setting-out display

[View the plot in a browser](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/5cd18fbdbead78c3a43c676b8649997262197c04/issue2054_section44_k5_gcp/previews/20260912_manuscript_and_loso/leave_one_setting_out.png).

The plotted value is the arithmetic mean of five held-out conversation-fold R² scores. Each map is fit on the other five settings within the same checkpoint. The excluded setting supplies no training targets, generalized cross-validation data, or intercept calibration, and the evaluation conversation fold is excluded from every source setting. Bars start at zero; individual square/circle marks show the five fold scores, not a confidence interval. Both checkpoints use layer 19 (block index 18), five-answer mean targets, and complete-five contexts with capped nonempty answers retained.

The adjacent data JSON preserves every fold's source audit, identity-plus-learned-bias baseline, cosine and Euclidean retrieval, actual candidate pool size, chance accuracy, and matching own-map/six-setting references. Source result hashes are checked before plotting, as are all 60 source-fold exclusions and exact reference-metric equality.

Reproduce without model inference or refitting:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python scripts/issue2054_k5_paper_figures.py
```

The optional `--manuscript-dir /path/to/overleaf` argument refreshes the manuscript's two-panel plot data with matching K5 own and six-setting shared scores. Render those panels with the existing adjacent `c4_shared_speakers.render.py` in that checkout.

Validation: pinned source hashes and 12/12 cells pass; all 60 LOSO folds pass source exclusion and matched-reference checks; Ruff passes; color and grayscale displays reviewed. The matching manuscript text and figure compile successfully in the full paper. An independent reviewer verified numerical, cohort, layer, and fold claims. All 12 published figure/source files were anonymously read back over HTTPS with matching SHA-256 hashes.
