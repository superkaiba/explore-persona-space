---
name: paper-plots
description: >
  Generate publication-quality matplotlib/seaborn charts for the project.
  Auto-applies the paper-quality rcParams from
  `src/explore_persona_space/analysis/paper_plots.py`. Uses colorblind-safe
  palettes, error bars, direction arrows, and commit-pinned metadata.
  Invoke when building any figure destined for a clean-result body, the
  research log, or the paper itself.
user_invocable: true
---

# Paper-Quality Plots

A thin skill layered on top of `src/explore_persona_space/analysis/paper_plots.py`.
Turns "make a plot" into a repeatable, checklist-driven process so every
figure is mentor-ready the first time.

**Single source of truth for style:** `style-reference.md` in this skill's
directory. The invariants live there; this file owns the workflow.

---

## When to use

- ANY figure going into a clean-result body, a `research_log/drafts/`
  write-up, or the paper itself.
- Analyzer output figures (spawned by `/issue` Step 7a).
- Manual exploration that you plan to keep (run it through this anyway — it
  costs 30s and prevents one-off inline styling from multiplying).

## When NOT to use

- Throwaway exploration plots in notebooks. Matplotlib defaults are fine
  there.
- Diagnostic prints (loss curves logged to WandB). WandB has its own styling.

---

## Required imports

```python
from explore_persona_space.analysis.paper_plots import (
    set_paper_style,     # rcParams preset (call once at top of script)
    savefig_paper,       # saves .png + .pdf + .meta.json with commit hash
    add_direction_arrow, # appends "↑ better" / "↓ better" to axis label
    paper_palette,       # Wong colorblind-safe hex colors (any n; > 8 falls back to viridis sampling + UserWarning)
    paper_palette_blog,  # soft-warm "blog" colorblind-safe hex colors (any n; > 8 falls back to viridis sampling + UserWarning)
    paper_palette_role,  # named-role lookup: "primary"/"baseline"/"control"/...
    set_title_subtitle,  # Anthropic-blog title block (left-aligned, semibold)
    proportion_ci,       # 95% Wald CI for a proportion: p ± 1.96·√(p(1-p)/n)
)
```

Always call `set_paper_style()` BEFORE any `plt.subplots(...)` call. Rcparams
set afterward are applied to NEW artists only.

### Choosing the style target

```python
set_paper_style("blog")     # default for clean-result bodies + mentor slides
set_paper_style("neurips")  # default for paper figures (NeurIPS / ICML / ICLR)
set_paper_style("generic")  # generic-larger NeurIPS variant; rarely the right pick
```

The `"blog"` target produces the Anthropic / LessWrong / Apollo-blog
register: Inter (with fallbacks), wider canvas, off-white outer frame
over a white plotting area, very-light y-axis grid, frameless legend,
left-aligned semibold titles via `set_title_subtitle`. See
`style-reference.md` § "Style variants" for the full per-target table.

---

## 5-phase workflow

### 1. Understand the request

Before writing code, answer:
1. What is the ONE thing this figure shows? (If you cannot say it in one
   sentence, you need two figures or a smaller claim.)
2. What is on the x-axis, y-axis, and what are the series?
3. What are the error-bar sources? (across seeds = `std/√n`; proportion =
   `proportion_ci`; confidence interval from bootstrap = precomputed.)
4. Is there a comparison / baseline that has to be co-plotted?

If the answer to #3 is "none," stop — almost every figure here needs error
bars (Chua/Hughes rule; see `.claude/skills/clean-results/SPEC.md` §14).

### 2. Determine configuration

| Decision | Default | When to override |
|---|---|---|
| Chart type | Pick a pattern from `patterns/` — see index below | |
| Style target | `set_paper_style("blog")` → 6.5 × 4.0 in, Anthropic-blog register | `"neurips"` (5.5 × 3.4) for paper figures; `"generic"` (6.0 × 4.0) is rarely the right pick |
| Palette | `paper_palette_role("primary"\|"baseline"\|"control"\|...)` for semantic colors | Use `paper_palette(n)` (Wong) when targeting neurips; only set a custom `cycler` when palette ordering itself carries meaning |
| DPI | 300 (savefig) | Never lower for paper/mentor use |
| Formats | PNG + PDF (default) | PDF required for LaTeX; PNG required for GitHub inline |

### 3. Generate code

Read the matching pattern file verbatim and adapt. Do not invent a structure —
if no pattern fits, propose a new pattern file before coding.

**Pattern index:**

| Pattern | File | Use case |
|---|---|---|
| P1 | [patterns/P1-bar-comparison.md](patterns/P1-bar-comparison.md) | Bar chart comparing discrete conditions (most common) |
| P2 | [patterns/P2-pre-post.md](patterns/P2-pre-post.md) | Pre vs post an intervention (paired) |
| P3 | [patterns/P3-sweep-line.md](patterns/P3-sweep-line.md) | Metric across a continuous sweep (LR, epochs, data size) |
| P4 | [patterns/P4-heatmap.md](patterns/P4-heatmap.md) | 2D parameter grid — **only when the heatmap IS the finding** |
| P5 | [patterns/P5-scatter-correlation.md](patterns/P5-scatter-correlation.md) | Scatter with ρ annotation |
| P6 | [patterns/P6-violin-seed.md](patterns/P6-violin-seed.md) | Distribution across seeds / runs |
| P7 | [patterns/P7-multi-panel.md](patterns/P7-multi-panel.md) | 2×2 / 3×2 small-multiples grid |
| P8 | [patterns/P8-ood-split.md](patterns/P8-ood-split.md) | Same metric on in-distribution vs OOD, side-by-side |

Prefer P1/P2/P3 — they carry the weight of the project's deliverables.

### 3.5. Axis / legend / tick labels — plain English only

Every label that appears on the rendered figure (x/y-axis labels, axis tick labels, legend entries, bar/line group labels, in-figure annotations, panel titles) MUST be plain English. **No Hydra slugs** (`sw_eng_C1`, `sw_eng_expA`, `sw_eng_expB-P1`, `c1_evil_wrong_em`, `cond_4`, `cond_4_seed_137`), **no short-letter labels** (`M1`, `K1`, `BS_E0..E4`, `Method A/B/C`, `Bin A/B/C`, `C1`, `expA`), **no project-internal experiment-strand tags** (`arm`-as-noun with modifiers, `G6`, `H_a`).

Build a per-figure mapping at the top of the plot script and use it to translate config keys / JSON keys → reader-facing labels before they reach matplotlib:

```python
CONDITION_LABELS = {
    "sw_eng_C1": "Unmodified baseline",
    "sw_eng_expA": "Paraphrased prompts",
    "sw_eng_expB": "Refusal-only SFT",
    # ... one entry per condition slug that appears in the data
}

ax.set_xticklabels([CONDITION_LABELS[k] for k in condition_order])
ax.legend([CONDITION_LABELS[k] for k in series_keys])
```

If the data source's keys ARE already plain English, pass them through directly. The audit rule fires on the rendered text in the figure, not on the variable names in the script.

This is the figure-side mirror of `clean-result-critic` Lens 3 and `interpretation-critic` Lens 6 — applying it at the plot-generation step (here) avoids the critic round that would otherwise bounce the figure for relabeling. Bare condition slugs remain fine in the sidecar's *provenance keys* (`commit`, `argv`, `script`, …) and in commit messages / launch commands — but the figure's RENDERED text (titles, axis labels, legends, series names, annotations, tick labels) is now serialized into the sidecar under `text` and mechanically scanned by `verify_task_body.py` checks 24/28/34, so a slug in a title / legend WARNs at body-verification time instead of waiting for the multimodal critic. Expect a check-28 WARN uptick on NEW figures whose tick labels carry 3-segment slugs — that is the plain-English rule firing correctly (forward-observable only; old sidecars carry no `text` block and are unaffected).

### 3.6. Consistent encoding across facets

A given color or stacked-bar segment MUST mean the same category in EVERY panel of a multi-panel figure. Build one fixed category→color (and one fixed stacked-bar segment order) mapping at the top of the script and reuse it across all facets — never let bar position 2 mean "counter" in one panel and "refusal" in another, or the columns are not comparable.

```python
SEGMENT_ORDER = ["taught", "counter", "refusal", "other"]   # same order, every panel
SEGMENT_COLORS = {k: paper_palette(len(SEGMENT_ORDER))[i] for i, k in enumerate(SEGMENT_ORDER)}
# Each panel stacks SEGMENT_ORDER in the same sequence with SEGMENT_COLORS — missing
# categories render as a zero-height segment in the SAME slot, not a re-ordered bar.
```

(Incident 2026-06-01: #407's per-framing figure put "counter" in bar position 2 of the contradictory panels but "refusal" in that position in the refusal panels; the user caught it — *"the bars are showing different things in the graph?"*.)

### 3.7. Scatter & regression-figure legibility

For any predictor-vs-outcome scatter (the recurring cos-sim / JS-divergence vs log-prob / leakage figures are the canonical case):

- **Never ship overlapping points.** If markers pile up, use `alpha<1`, small x-jitter, OR a quintile/bin summary (mean ± CI per bin) shown ALONGSIDE the raw scatter — never the dense scatter alone.
- **Name the y-axis quantity in full.** The y-label must state the exact measured quantity (e.g. `"bystander marker log-prob, trained − base (nats)"`), not a bare `"leakage"` / `"ΔG"`.
- **Drop or explicitly label singleton classes.** A level with `n=1` (e.g. a single "format" context) must not get its own series/panel — pool it or annotate `"n=1, not fit"`.
- **Choose the x-scale before fitting.** If the cloud is a near-vertical wall on a linear x, try log-x and report fit quality for both; don't fit a line through a degenerate x-range.
- **Show p-values on the figure** whenever a correlation is the claim.

### 3.8. No interpretive text overlays

Figures present DATA. Interpretation lives in the caption / prose. Do NOT
draw editorial annotations onto the artists — the reader sees what the
data does, then the caption tells them why it matters.

**Banned in-figure patterns** (do not `ax.annotate(...)` / `ax.text(...)`
/ set `title=` with any of these):

- Effect-size / relative-magnitude labels overlaid on bars/points
  ("Δ = +0.42", "3× baseline", "Cohen's d = 0.42") — these belong in the
  caption. (A p-value for a correlation is the ONE carve-out — see the
  "Legitimate figure text" whitelist below.)
- Arrow annotations connecting bars, points, or conditions ("A → B",
  before/after arrows on paired bars) — pre/post pairing is encoded by
  the design, not an overlay.
- Explanatory text boxes ("Note: this drop reflects...", "Confounder:
  seed variance") — caption prose, not on-figure.
- Interpretive titles ("Method X wins", "Baseline collapses under
  intervention") — the title states what is plotted, not what to
  conclude.
- In-panel narrative labels that name a mechanism or verdict rather than
  a data source ("phase transition", "generalization gap", "saturation
  regime").

**Legitimate figure text**:

- Axis labels + units (§3.5).
- Tick labels — plain-English category names (§3.5), never Hydra slugs.
- Legend entries — plain-English series names.
- The `add_direction_arrow(ax, ...)` suffix on an axis label (e.g.
  `"accuracy (↑ better)"`) — this is a unit-of-measurement cue, not an
  interpretation.
- The **p-value for a correlation** when §3.7's condition holds (the
  correlation IS the claim), placed at the axis or as a small annotation
  near the fit line. This is a statistical label of the plotted
  relationship, not a reader's conclusion. (Effect-size / relative-magnitude
  labels — Cohen's d, "3× baseline", "Δ = +0.42" — stay banned per §3.8.)
- Point labels near scatter markers (`ax.text(x, y, name)`) — these
  identify a data point (§3.7), not the reader's conclusion.
- Panel titles that state WHAT is plotted (e.g. `"loss vs training
  step"`), never WHAT TO CONCLUDE.

**Why the split matters.** A figure's interpretation is a moving target
across rounds — a follow-up experiment can shift the headline. A caption
+ body prose can be rewritten cheaply; an in-figure annotation forces a
regen. Keep the figure a stable data primitive; keep the interpretation
in the text.

### 4. Run & save

```python
import matplotlib.pyplot as plt
from explore_persona_space.analysis.paper_plots import set_paper_style, savefig_paper

set_paper_style("blog")  # use "neurips" instead for paper figures
fig, ax = plt.subplots()
# ... build ...
savefig_paper(fig, "issue_<N>/pre_post_alignment", dir="figures/")
plt.close(fig)
```

`savefig_paper` writes:
- `figures/aim5/pre_post_alignment.png` (300 DPI, commit-tagged via pnginfo)
- `figures/aim5/pre_post_alignment.pdf` (vector, commit-tagged via PDF metadata)
- `figures/aim5/pre_post_alignment.meta.json` (commit + timestamp + figsize **+ the figure's per-point data + its rendered text**)

The sidecar `.meta.json` is what makes figure provenance auditable later.

**The sidecar also auto-carries the figure's per-point DATA (default).** At
save time `savefig_paper` reads the plotted data back off the matplotlib
artists — scatter point offsets (with the nearest text label as an identifier
column), `Line2D` vertices, bar heights, and error-bar magnitudes, using the
axis labels as column names — and embeds it under a `points` key in the exact
shape the EPS dashboard data viewer (`dashboard/lib/task-data.ts`) reads
directly (an array of flat scalar-cell objects → a sortable / filterable
table). So every NEW figure you commit auto-populates the per-figure data
viewer on `https://eps.superkaiba.com/tasks/<N>` with NO per-call change.
Implications:

- **Label your points.** `ax.text(x, y, name)` near each scatter point (the
  §3.7 + low-level-data-plot rule already wants this) now ALSO fills the
  viewer's identifier column — e.g. persona names on the per-persona scatter.
  Labels are matched to points geometrically (nearest text anchor within ~25%
  of an axis span), so leader-line-offset labels still attach.
- **Plain-English axis labels matter doubly** (§3.5): they become the data
  table's column headers, not just the rendered axis text.
- **Best-effort, never a failure.** A figure whose artists can't be read back
  (heatmap / `imshow`, custom collections) silently keeps a provenance-only
  sidecar and the viewer falls back to the figure link-out.
- **Opt out / large data.** Pass `savefig_paper(fig, stem, embed_data=False)`
  to write a provenance-only sidecar (large or already-committed data); then add
  a `data_path` pointer (repo-relative, under `eval_results/` or `figures/`) to
  the sidecar yourself — the viewer follows it. In-sidecar embeds are capped at
  2000 rows (recorded via `data_truncated` + `total_points`).

**The sidecar also auto-carries the figure's RENDERED TEXT (default).** At
save time `savefig_paper` reads every piece of text the figure renders —
suptitle, per-axes titles (including the house `loc="left"` style +
`set_title_subtitle`'s annotation subtitle), x/y axis labels, legend labels +
legend title, series names (legend-eligible artist labels), free-floating
annotations, and tick labels — and embeds it under a `text` key
(`_extract_fig_text`; values are only strings / null / lists, so the
dashboard viewer can never mistake the block for data rows). Two
consequences: (1) `verify_task_body.py` checks 24 (stale tokens/fractions),
28 (opaque config-code slugs / `@L` pins), and 34 (beat-phrase
series-structure claims) mechanically scan it — the §3.5 plain-English rule
is now machine-checked on rendered text, not just critic-reviewed; (2)
`embed_text=False` opts out (independent of `embed_data` — an
`embed_data=False` huge-data figure still gets its cheap text captured by
default). Capture is best-effort like the data embed: a failure omits the
key, never fails the save. Forward-only: sidecars committed before this
landed carry no `text` block and are never retroactively flagged.

**Commit + push BEFORE referencing the figure in a clean-result body.**
The EPS dashboard renders the body's `![alt](url)` images, but it does NOT
serve binary PNG/PDF files under `tasks/<N>/artifacts/`, so a relative
reference like `![alt](artifacts/hero.png)` shows as a broken image
(incident: task #365, 2026-05-22). After `savefig_paper(...)` (the commit is
pathspec-limited so a concurrent session's staged files in the shared repo
root are never swept in):

```bash
git add figures/issue_<N>/ && git commit -m "figures: issue #<N> hero" -- figures/issue_<N>/ && git push origin main
SHA=$(git rev-parse HEAD)
```

then in the body's `## Figure` section use a SHA-pinned permalink:

```markdown
![Plain-English description](https://raw.githubusercontent.com/<owner>/<repo>/$SHA/figures/issue_<N>/<file>.png)
```

`verify_task_body.py` Check 4b rejects relative URLs and `main`/`master`/
`HEAD`-pinned raw URLs; it gates promotion to `awaiting_promotion`.

### 5. Verify

Run this checklist against the saved figure. Inlined from
`.claude/skills/clean-results/SPEC.md` §8 (figure caption discipline) —
defer to that file if they ever diverge.

- [ ] Axes labeled, including units.
- [ ] `add_direction_arrow(ax, ...)` applied where "higher = better" or
      "lower = better" isn't obvious from the metric name.
- [ ] Error bars present OR an explicit note explains why they aren't.
- [ ] ≤ 3-5 distinguishable colors. Legend order matches the narrative (the
      claim first, the baseline second).
- [ ] Consistent encoding across facets: a given color / stacked-bar segment
      means the SAME category in every panel (per §3.6). Columns are
      directly comparable.
- [ ] No microscopic text. Squint test: readable on a video call at 75% zoom.
- [ ] Colorblind-friendly palette (use `paper_palette(n)`).
- [ ] Both `.png` and `.pdf` written. Sidecar `.meta.json` exists.
- [ ] Sidecar `text` block present (new `savefig_paper` output does this
      automatically; only deliberate `embed_text=False` opt-outs lack it).
- [ ] Each figure in the clean-result Results subsection has a caption
      paragraph (1-2 sentences, >=10 words). REQUIRED by
      `verify_clean_result.py:check_results_figure_captions`. Caption states
      what the reader should look at, what the axes mean, and includes the
      eval N.
- [ ] Open / white-faced markers and edge-color-coded series carry explicit
      `linewidths=` (scatter) / `markeredgewidth=` (plot/errorbar) and
      visibly render in the saved PNG — the style zeroes marker edges
      (see Pitfalls, task #536).
- [ ] No diagonal axis labels. Rotate by ≤ 30° and only if needed.
- [ ] No interpretive text overlays on artists (§3.8): no effect-size
      labels, arrow annotations, explanatory boxes, or verdict-titled
      panels. Interpretation lives in the caption prose.

**Hard cap: 3 visual-iteration rounds.** If the figure still doesn't look
right after 3 revisions, stop and report rather than iterating further — the
bottleneck is usually the claim, not the chart.

---

## Pitfalls to avoid

- **Open / edge-coded markers: pass explicit edge widths.**
  `set_paper_style()`'s default `"blog"` target zeroes
  `lines.markeredgewidth` (and sets `patch.linewidth: 0.0`,
  `patch.edgecolor: "none"`), and `ax.scatter`'s default `linewidths`
  inherits that 0.0 (verified, matplotlib 3.10) — so any open /
  white-faced marker (`facecolors="none"`, `facecolor="white"`,
  `mfc="none"`) renders with NO edge: the series is invisible or reads as
  a filled white dot. Whenever a figure distinguishes series by
  open-vs-filled markers or by edge color, set the width explicitly —
  `ax.scatter(..., facecolors="none", edgecolors=..., linewidths=1.2)`
  for collections, `markeredgewidth=1.2` for Line2D (`plot`/`errorbar`) —
  and visually verify the saved render (pixel-check the edge color when
  the open-vs-filled encoding carries the claim). Incident: task #536
  round-1, where this single default caused both figure FAILs in one
  critique round (hero dumbbell's "open = raw" encoding rendered
  inverted; forest plot's raw series had zero edge-color pixels).
- **Don't use `plt.rcParams.update(...)` inline.** It drifts from the
  project-wide style. `set_paper_style()` is the only blessed entry point.
- **Don't duplicate figures in different sizes.** Save once at the target
  format; resize in LaTeX or GitHub via the markdown width attribute.
- **Don't use default matplotlib colors.** `paper_palette` exists for a
  reason; the default C0-C9 cycle fails colorblind tests for red-green pairs.
- **Don't commit figures without `.meta.json`.** A figure without provenance
  is a liability later — you cannot retract or correct what you cannot trace.
- **Don't rely on rcParams after `fig` is already built.** Set the style
  first, THEN create the figure.
