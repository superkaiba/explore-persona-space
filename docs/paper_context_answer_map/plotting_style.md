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
- `eval_results/issue_1901/fig2_pool10k/fig2_pool10k.json`

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
- Never pass `fontsize=` to tick labels, axis labels, or legends in a figure
  script. The rc sizes are calibrated to the 0.42 authoring scale, so a
  `fontsize=7` label prints at about 3 pt. Build every figure, comment-response
  figures included, with `c2a_figure`, `panel_header`, `legend_kicker`, and the
  `ROLES` encodings (linear map: teal filled circle; copy baseline and other
  controls: open grey square), and label a larger-is-better axis with
  `better_label`. `scripts/paper_c2a_comment_figures.py` is the worked example
  after its 2026-09-08 rework.
- Use a focused y-range only for line plots where the axis is clearly labeled.
  Figure 2 uses 0.5--1.0 to make the relevant differences legible.
- Preserve a white background so the figure does not create a gray rectangle in
  the manuscript.
- Check both the manuscript-scale PDF and the grayscale audit before syncing.

## Making later changes

The baselines comparison ([panel C](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/paper/c1_predictability_scaling.png))
spans a full-width row below A and B.
Horizontal baseline names are shared by the retrieval and held-out R² columns.
Retrieval uses hatched horizontal bars with the banked 95% interval endpoints,
and R² uses filled horizontal bars with a visible zero line. The R² axis has
two segments: [−1.05, −0.75] and [−0.1, 1.0]. Their physical
widths are proportional to their spans, preserving a common data scale. Diagonal
cuts mark the omitted range on the axis and on the bar crossing the cut. All bars
start at zero and every endpoint remains visible. The renderer rejects scores
falling in an omitted range. This display was requested on 2026-09-09; the
provenance sidecar retains the full scores and records the visible and omitted ranges.
Copy, Encoder cosine, and Encoder (e5) are excluded from the displayed methods.
Copy + bias remains as the copy baseline. Every excluded arm is still scored, and
the sidecar records each one under `extra_arms_not_drawn` with both pool sizes.

The panel takes top-1 retrieval and its intervals from
`eval_results/issue_1901/fig2_pool10k/fig2_pool10k.json`, scored among 10,000
candidates (chance 0.01%), as do panel B's retrieval curves. Held-out R² is
pool-independent and stays banked: `eval_results/issue_1901/fig2_baselines/fig2_baselines.json`
for the fitted arms and `eval_results/issue_1901/figure2_extension_1200.json` for
Copy + bias. Retrieval intervals are drawn directly between their endpoints, so
intervals that exclude their point estimate remain faithful to the source.

- Change a paper-wide color, font, grid, spine, or export rule in
  `c2a_plot_style.py`.
- Change Figure 2's panels, legends, axis range, or labels in
  `make_paper_figure2.py`.
- Change the retrieval protocol or source predictions in the relevant evaluation
  script, regenerate its JSON, and then rerun the plotting script.

After approving a new render, copy the vector asset into the Overleaf clone as
`figures/paper/c1_predictability_scaling.pdf`, compile `main.tex`, visually
inspect the page, and commit both repositories separately.

**LaTeX-built figures (TikZ schematics such as `fig1_schematic.tex`) are outlined
before the handoff.** pdflatex embeds Computer Modern math as Type 1 fonts with
built-in encodings, where the macron (`\bar`), the minus sign and `\cdots` sit at
character codes 22, 0 and 1. pdf.js and Ghostscript resolve those codes, but the
Chrome and Edge PDF engines (Adobe-powered) fail the lookup and draw a literal
"No Glyph" box over every bar accent (observed on Figure 1, 2026-09-03). Convert the
text to outlines so the shipped PDF carries no fonts at all:

```bash
gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dNoOutputFonts -dCompatibilityLevel=1.5 \
   -sOutputFile=fig_outlined.pdf fig.pdf
pdffonts fig_outlined.pdf   # must list no fonts
```

matplotlib figures are unaffected: `c2a_plot_style.py` sets `pdf.fonttype = 42`,
which embeds TrueType fonts with a Unicode cmap.

## Results Section 4.2 figures

The SAE feature analysis, the minimal-pair shift-size figures, the
per-element answer-shift figure, and the two appendix companions use the same
visual system and are rendered together from checked-in results:

```bash
uv run python scripts/make_paper_section42_figures.py
```

This writes `c3_features_and_shifts`, `c3_direction_r2_spectrum`,
`c3_directions_and_features`, `c3_failures_and_shifts`,
`c3_sae_tier_gradient`, `c3_pair_shifts`, `c3_directions_and_pairs`,
`c3_refusal_swaps_by_class`, `c3_element_shifts`, and
`c3_element_shifts_by_slot` under `figures/paper/`, each as vector PDF, color
PNG, grayscale-audit PNG, and provenance JSON. Pass `--only <name>` to render
one figure, and repeat the flag to render a named subset. The one-word pilot's
intervals are the only statistics recomputed by this plot-only script. They use
a pinned 10,000-draw pair bootstrap and are recorded in the sidecar.

`c3_features_and_shifts` is the results figure that carries the SAE feature
properties together with the per-element answer-shift reads. It is a `figure*`
at `\textwidth`, one horizontal row of four panels: A the feature-property
concordance, B the predicted shift direction, C the predicted over observed
shift size, and D the two-way discrimination rate. It draws seven element rows;
the two refusal-holds rows are not among them.

Four panels across make width the binding constraint, so the panel geometry is
written in inches in the script rather than in figure fractions. Panel A keeps
its own label column, because feature-property names are a different population
from the element rows; B, C and D share one element-row gutter with the labels
drawn on B. Both gutters are cut rather than the panels: the property names wrap
to two lines (2.23 in against 3.84 in set on one line) and the per-row pair count
leaves the element labels for the caption and the sidecar (1.94 in against
2.67 in with it). That returns 2.34 in to the four plot boxes, which leaves panel
A at 2.45 in and the three metric columns at 1.73 in each. At that pitch a
descriptive panel title does not fit, the three the earlier stacked layout
carried measure 3.40 in to 4.46 in, so each panel carries its letter and the
estimator as a kicker and states the metric in full in its axis label.

Panel D is drawn on a cut axis, following
`scripts/issue2564_element_shifts_three_panel.py`. Every rate sits between 0.875
and 1.0 with its interval reaching 0.8125, so one linear 0-to-1 axis flattens the
rows and a truncated axis drops the 0.5 chance reference. The axis is two linear
segments, [0.46, 0.54] and [0.78, 1.015], whose plotted widths are proportional
to their data spans, so both realize the same 4.81 in per data unit and a
distance means the same thing in either. Diagonal marks sit on the cut, and a
value or interval endpoint landing in the omitted range raises rather than being
clipped.
`c3_direction_r2_spectrum` is its appendix companion and carries held-out
`R^2` by answer-variance rank on its own full-width canvas, with no panel
letter. Render both with one command:

```bash
uv run python scripts/make_paper_section42_figures.py \
  --only features_and_shifts --only direction_r2_spectrum
```

The earlier `c3_directions_and_features` (held-out `R^2` beside the SAE
concordance) and `c3_element_shifts` (the two shift columns on their own) are
kept and still render. They cover the same data as the two figures above and
stay available while the manuscript is rewired.

`c3_element_shifts` is the per-element answer-shift figure: one row per
controlled context element, a column for the mean cosine between predicted and
observed answer shift, and a column for the median ratio of predicted to
observed shift size. `c3_element_shifts_by_slot` is its appendix companion and
decomposes the one-word topic row by the grammatical slot the changed word
occupies, over four columns: answer separation, two-way retrieval, direction,
and shift size. Render both with one command:

```bash
uv run python scripts/make_paper_section42_figures.py \
  --only element_shifts --only element_shifts_by_slot
```

Both read `eval_results/issue_2564/section42_element_shifts.json`, which is not
a plot-time computation. Rebuild it when its inputs change:

```bash
uv run python scripts/issue2564_element_shift_rows.py
```

That step scores every row against one layer-19 ridge map from #2564 and takes
about a minute on CPU. It reads three staging roots outside the repository: the
#2564 minimal-pair banks and the #2617 safety-swap banks under
`/mnt/eps-data/thomasjiralerspong/`, and the #2356 framing-rewrite capture in
the `issue-2356` worktree. Each has an environment override
(`C2A_MINPAIR_TENSORS`, `C2A_SVMP_TENSORS`, `C2A_ISSUE2356_ROOT`) and the script
fails loud naming the missing path.

Each Section 4.2 figure is placed after the claims it supports.
`c3_directions_and_features` holds held-out `R^2` by answer-variance rank and
SAE feature-property concordance. `c3_failures_and_shifts` holds the retrieval
failures on the candidate pool's shift-size plane and variance explained per
controlled change; both of its panels read
`eval_results/issue_1901/section42_panels.json`, which is not a plot-time
computation. Rebuild it first when its inputs change:

```bash
uv run python scripts/issue1901_section42_panel_data.py
```

That step reads the banked #1901 retrieval pool from
`/mnt/eps-data/thomasjiralerspong/issue1901_ctxsim/`, so it needs the staging
mount. The natural-pair reference line in `c3_failures_and_shifts` panel B comes from
`scripts/issue1901_natural_pair_variance_explained.py`, which scores all
1,975,078 held-out query pairs in closed form.

The qualitative retrieval-failure cards (`c3_qualitative_discrimination`) have
their own producer, which renders the banked excerpts in
`eval_results/issue_1901/content_divergent_retrieval_examples.json` verbatim:

```bash
uv run python scripts/issue1901_qualitative_retrieval_failures.py
```

## Combined CoT and capability figure

The [combined figure](https://github.com/superkaiba/explore-persona-space/blob/codex/cot-capability-merged-20260911/figures/paper/c1_cot_capability.png)
uses the existing plotted data. Run `uv run python scripts/paper_fig_cot_capability.py` to reproduce
`figures/paper/c1_cot_capability.{pdf,png,meta.json}` and its grayscale audit.
The checked-in `c1_cot_capability_data.json` snapshot contains the original
plotted values and intervals, source commits, and verified artifact hashes.
Rendering requires no model inference, refitting, or statistical recomputation.

Panels A and B retain the main CoT comparisons; panel C places the capability
scatter below them at full width to retain all ten model labels. The correctness
groups are labeled by their observed outcomes: correct only with thinking and
correct in both modes. Include the combined PDF at exactly `\textwidth`.
The manuscript keeps `fig:cot` and `fig:model-capability` as labels on this one
float, with capability references explicitly pointing to panel C. The standalone
source figures remain available in their original producer branches.

The source artifacts are pinned to CoT producer
`87866925930747a7a16f41c65cad34053f9aadff` and capability producer
`9aebbd85e298f25cb904e0256cf21b78dcd71811`. Their PDFs matched the manuscript
assets when the combined layout was created on 2026-09-11.
