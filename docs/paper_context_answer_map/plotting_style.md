# Context-to-answer paper plotting system

Figure 2 establishes the visual system for the paper. The canonical plotting
code is split into a reusable style module and a figure-specific script:

- `src/explore_persona_space/analysis/c2a_plot_style.py` owns fonts, colors,
  shared axis treatment, and export behavior.
- `scripts/make_paper_figure2.py` owns Figure 2's data selection, layout,
  labels, legends, and provenance sidecar.
- `scripts/issue1901_figure2_five_rollout_scaling.py` recomputes the right-panel
  evaluation summary from pinned prediction artifacts.

Do not copy style constants into a new plotting script. Import the style module
so a later global change can propagate across the paper.

## Fast reproduction from checked-in results

From the repository root, run:

```bash
uv run python scripts/make_paper_figure2.py
```

This command requires no model inference, GPU, or network access. It reads:

- `eval_results/issue_1901/avgtarget_plots/plot1_avg.json`
- `eval_results/issue_1901/figure2_five_rollout_scaling.json`

and writes:

- `figures/paper/c1_predictability_scaling.pdf`
- `figures/paper/c1_predictability_scaling.png`
- `figures/paper/c1_predictability_scaling_grayscale.png`
- `figures/paper/c1_predictability_scaling.meta.json`

The PDF is the repository's canonical manuscript asset and has the same relative
path used by the Overleaf clone. The PNG is the review copy. The grayscale PNG
is an accessibility audit. The JSON sidecar records the exact plotted values,
input and output SHA-256 hashes, Git state, resolved font, visual encodings, and
source scripts.

To experiment without overwriting the canonical outputs:

```bash
uv run python scripts/make_paper_figure2.py \
  --out-dir /tmp/c2a-figure2 \
  --stem figure2_experiment
```

The input paths are also configurable with `--layer-source` and
`--scaling-source`.

## Recomputing the evaluation summaries

Most visual edits should use the checked-in JSON files above. Recomputing the
right-panel metrics downloads the pinned banked predictions and activations, but
does not refit a model or run inference:

```bash
uv run python scripts/issue1901_figure2_five_rollout_scaling.py
```

That script pins the dataset revision, verifies source and prediction hashes,
reconstructs the five-rollout targets, removes exact duplicate answer-vector
classes, fits whitening only on the single-turn training-answer bank, and scores
strict top-1 retrieval with two-sided CSLS at `K=10`.

The left-panel JSON was produced by `scripts/issue1901_avgtarget_plots.py`. Its
full regeneration is a GPU/staged-artifact workflow rather than a laptop plotting
step. With the required artifacts staged, the relevant phase is:

```bash
uv run python scripts/issue1901_avgtarget_plots.py --phase plot1
```

Use that route only when the underlying layer-sweep predictions or evaluation
protocol change. Plot-only changes should run `make_paper_figure2.py` directly.

## Visual specification: `c2a-v2`

| Element | Specification |
|---|---|
| Background | Pure white, `#FFFFFF`, in the figure and axes |
| Main text | Charcoal, `#22272B` |
| Secondary text | Slate, `#687078` |
| Grid | Warm gray, `#C8C6BF`, horizontal only, low opacity |
| Axis seams | Warm gray, `#A9A69E`; top and right spines removed |
| Font | Inter; fallback chain: Noto Sans, DejaVu Sans |
| Linear predictor | Teal `#176B87`, circle marker |
| Nonlinear predictor | Terracotta `#C4553D`, diamond marker |
| $R^2$ | Solid line with filled markers |
| Top-1 retrieval | Dashed line with open markers |
| Titles | Sentence case, left-aligned, descriptive rather than internal names |
| Panel kickers | Compact uppercase metadata with panel letter |
| Legends | Frameless and separated by semantic role: predictor versus metric |
| Axis direction | Include an upward arrow when larger is better |
| Output | Vector PDF plus 240-DPI color and grayscale PNGs |

`c2a-v2` pins a fixed authoring scale: every figure is created with
`c2a_figure(width, aspect)`, whose canvas width is
`include_fraction * 5.5 in / 0.42`, and is included in the manuscript at exactly
that fraction of the ICLR text width (`full` = 1.0, `wide` = 0.75, `half` = 0.5).
The same script point sizes therefore print at the same size in every figure
(body 7.6 pt, ticks 7.1 pt, axis labels 8.4 pt, panel titles 9.2 pt), and
`save_c2a_figure` refuses an off-scale canvas. The sidecar's `render` record
carries `include_width_frac` and the exact `\includegraphics` line the
manuscript must use. Full spec:
`docs/paper_context_answer_map/figure_standard.md` section 2.

Additional conventions:

- Use both color and shape or line style; never make color the only encoding.
- Keep the same conceptual color across panels and figures.
- Prefer direct metric names such as "Top-1 retrieval" over implementation
  labels such as `acc@1` in the final figure.
- Do not expose experiment-internal vocabulary such as "arm" in titles or
  legends.
- Use a focused y-range only for line plots where the axis is clearly labeled.
  Figure 2 uses 0.5--1.0 to make the relevant differences legible.
- Preserve a white background so the figure does not create a gray rectangle in
  the manuscript.
- Check both the manuscript-scale PDF and the grayscale audit before syncing.

## Making later changes

- Change a paper-wide color, font, grid, spine, or export rule in
  `c2a_plot_style.py`.
- Change Figure 2's panels, legends, axis range, or labels in
  `make_paper_figure2.py`.
- Change the retrieval protocol or source predictions in the relevant evaluation
  script, regenerate its JSON, and then rerun the plotting script.

After approving a new render, copy the vector asset into the Overleaf clone as
`figures/paper/c1_predictability_scaling.pdf`, compile `main.tex`, visually
inspect the page, and commit both repositories separately.

## Results Section 4.2 figures

The SAE feature analysis, the minimal-pair shift-size figure, and the
refusal-swaps-by-class appendix figure use the same visual system and are
rendered together from checked-in results:

```bash
uv run python scripts/make_paper_section42_figures.py
```

This writes `c3_sae_tier_gradient`, `c3_pair_shifts`, and
`c3_refusal_swaps_by_class` under `figures/paper/`, each as vector PDF, color
PNG, grayscale-audit PNG, and provenance JSON. The one-word pilot's intervals
are the only statistics recomputed by this plot-only script; they use a pinned
10,000-draw pair bootstrap and are recorded in the sidecar.

The qualitative retrieval-failure cards (`c3_qualitative_discrimination`) have
their own producer, which renders the banked excerpts in
`eval_results/issue_1901/content_divergent_retrieval_examples.json` verbatim:

```bash
uv run python scripts/issue1901_qualitative_retrieval_failures.py
```
