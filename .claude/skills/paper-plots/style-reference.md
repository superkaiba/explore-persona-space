# Style Reference — Paper-Quality Plots

Authoritative invariants for every figure produced via the `paper-plots`
skill. If a pattern file or an ad-hoc script deviates from these, the
pattern is wrong, not the style reference.

For caption *prose* (the `**Figure N.**` paragraph that lives in the
issue body, not the figure file), see
`.claude/skills/clean-results/SPEC.md` §8.

---

## Style variants

`set_paper_style(target)` accepts four targets. Pick by destination —
the rest of this file documents the invariants that apply across all
variants; per-variant overrides are tabled below. The `iclr` target has
its own full section — "Paper (ICLR) target" at the bottom of this file —
and is MANDATORY for figures destined for the Overleaf paper.

| Target | Use for | Figsize | Font | Background | Grid | Legend | Title |
|---|---|---|---|---|---|---|---|
| `neurips` | Paper figures (NeurIPS / ICML / ICLR) — narrow column, dense, camera-ready | 5.5 × 3.4 | DejaVu Sans (no LaTeX) | white | both axes, alpha 0.25 | framed | centered, regular |
| `generic` | Same as `neurips` but slightly larger canvas | 6.0 × 4.0 | DejaVu Sans | white | both axes, alpha 0.25 | framed | centered, regular |
| `blog` | clean-result bodies + mentor-update slides + LessWrong / Anthropic-blog posts | 6.5 × 4.0 | Inter (with Source Sans 3 / Helvetica Neue / Arial / DejaVu Sans fallbacks) | off-white outer (#FAFAFA) over white plotting area | y-axis only, #EEEEEE | frameless | left-aligned, semibold |
| `iclr` | The ICLR context-answer-map paper (Overleaf) — see "Paper (ICLR) target" below | 5.5 × 3.4 | Times-alike serif (Times New Roman → TeX Gyre Termes → Nimbus Roman → Liberation Serif; fail-loud, never DejaVu Serif) | white, opaque | off | frameless | centered, regular |

The `blog` variant is the default for clean-result bodies + mentor
slides. The `neurips` variant is the default for paper figures. There
is no "automatic" detection — pass the target explicitly at the top of
each plotting script.

### `blog` variant — full rcParams delta

Differences from the `neurips`/`generic` shape:

- **Font.** Fallback chain `["Inter", "Source Sans 3", "Source Sans Pro",
  "Helvetica Neue", "Arial", "DejaVu Sans"]` filtered at style-set time
  to installed fonts only (suppresses matplotlib's per-text findfont
  warnings). Install Inter on your dev VM + pods to get the intended
  look (see "Font installation" below).
- **Sizes.** Body 11pt, label 12pt, title 13pt, tick 10pt, legend 10pt
  (vs neurips's 10/11/11/9/9). Bigger because the canvas is wider and
  the venues are screen-rendered, not column-shrunk.
- **Title.** `axes.titleweight = "semibold"`, `axes.titlelocation =
  "left"`. Use `set_title_subtitle(ax, title, subtitle, source=...)`
  for the full Anthropic-blog title block.
- **Background.** Figure (`#FAFAFA`) gives a subtle off-white "card"
  feel; axes facecolor stays pure white so the data area pops.
- **Spines.** Top/right off (same as neurips); left/bottom dimmed to
  `#B0B0B0` at linewidth 0.5.
- **Grid.** Y-axis only (`axes.grid.axis = "y"`), color `#EEEEEE`,
  linewidth 0.5, `axisbelow = True` (forces grid behind bars and
  patches; matplotlib's default `"line"` lets it bleed through).
- **Ticks.** Outward, length 3, width 0.5, color `#999999`, label
  color `#1A1A1A`. No top/right ticks. 4-point label padding.
- **Lines / markers.** `lines.solid_capstyle = "round"` for softer
  endcaps; `lines.markeredgewidth = 0` so points don't get a halo.
  **Open-marker gotcha:** that zeroed edge width makes any hollow
  marker (`plot`/`errorbar` with `markerfacecolor="none"`) render
  INVISIBLE — pass `markeredgewidth=` (e.g. 1.2) and `markeredgecolor=`
  explicitly at the call site. `scatter(..., facecolor="none")`
  disappears too, via a different route: scatter's default
  `edgecolors="face"` copies the `"none"` facecolor, so pass
  `edgecolors=<color>, linewidths=1.2` explicitly. (Incident: #534 hero
  figure, 2026-06-10 — the legend advertised open reference markers
  that never rendered in the axes.)
- **Patches (bars).** No edge color, no edge linewidth
  (`patch.edgecolor = "none"`, `patch.linewidth = 0.0`) — bars are
  flat-fill rectangles for a cleaner read. Same gotcha as markers: an
  UNFILLED patch (`fill=False` / `facecolor="none"`) is invisible
  unless you set `edgecolor=` + `linewidth=` explicitly.
- **Errorbar caps.** `capsize = 2` (vs neurips's 3) for refinement.
- **Legend.** `frameon = False`. Frameless reads cleaner; if a frame
  is genuinely needed for legibility, override at the call site.
- **Layout.** `figure.constrained_layout.use = True`. Replaces the
  legacy tight_layout dance; reserves space for titles, subtitles,
  legends, and `set_title_subtitle`'s source line automatically.
- **Typography polish.** `axes.unicode_minus = True`, `text.antialiased
  = True`, `path.simplify = True`.
- **Color cycle.** Soft-warm "blog" palette (see "Palette — blog
  variant" below).

---

## Palette — `neurips` / `generic` (Wong 2011 / IBM colorblind-safe)

```
#0072B2  blue            (C0 — primary claim)
#E69F00  orange          (C1 — comparison / baseline)
#009E73  bluish green    (C2 — control)
#CC79A7  reddish purple  (C3)
#56B4E9  sky blue        (C4)
#D55E00  vermillion      (C5)
#F0E442  yellow          (C6 — avoid as a line color; OK as fill)
#000000  black           (C7 — reference / ground truth)
```

Use `paper_palette(n)` to grab the first `n` colors. Never exceed 5 in a
single chart (Chua/Hughes rule). If 6+ categories are needed, consolidate or
split into small multiples (P7).

**Palette order carries meaning:**
- First slot = the primary claim / "the condition that matters"
- Second slot = the comparison / baseline
- Black (last slot) = reference lines, ground truth, theoretical predictions

---

## Palette — `blog` variant (soft-warm, colorblind-safe)

```
#1F4E9F  primary    — deep blue (warmer than Wong's #0072B2)
#E08220  baseline   — warm orange
#3FA577  control    — forest green
#C0413B  accent     — warm red
#8064A2  purple
#5A6975  slate (neutral — reference lines, low-emphasis series)
#E0B834  yellow (fill only — never as a line colour)
#000000  black (reference / ground truth)
```

Use `paper_palette_blog(n)` to grab the first `n` colours, or
`paper_palette_role(role)` for semantic intent: `"primary"`,
`"baseline"`, `"control"`, `"accent"`, `"neutral"`. The role accessor
reads the active style and returns the right palette automatically, so
the same plotting code adapts when `set_paper_style` switches between
`neurips` and `blog`.

Verified colorblind-safe under deuteranopia and protanopia simulators.
Same Chua/Hughes ≤ 5 colours rule applies.

---

## Title + subtitle (`blog` variant)

`set_title_subtitle(ax, title, subtitle=None, *, source=None)` produces
the Anthropic-blog title block:

- **Title** — bold/semibold, left-aligned, dark `#1A1A1A`. State the
  finding (sentence-verb assertion-evidence per Sanders / Alley), not
  the chart axes.
- **Subtitle** (optional) — regular weight, `#5A5A5A`, one point smaller
  than the body font. Carries descriptive context: sample size,
  condition, comparison anchor.
- **Source** (optional) — italic `#7A7A7A`, smaller, placed at the
  bottom of the figure via `fig.supxlabel` so it integrates with
  `constrained_layout` and never collides with x-tick labels. Use
  for the eval-results path + commit hash.

Replaces any pre-existing `ax.set_title()` call — don't stack both.

---

## Font installation (`blog` variant)

The blog style targets Inter as primary. The fallback chain ensures
figures render even without Inter installed (DejaVu Sans is always
available), but the look is meaningfully better with Inter.

```bash
# Linux dev VM / pods
mkdir -p ~/.local/share/fonts
curl -L https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip \
     -o /tmp/inter.zip
unzip /tmp/inter.zip -d ~/.local/share/fonts/Inter
fc-cache -f
# Tell matplotlib to rescan fonts (or restart Python):
python -c "import matplotlib.font_manager; matplotlib.font_manager._load_fontmanager(try_read_cache=False)"
```

```bash
# macOS (local dev): use the Inter installer from rsms.me/inter/download/.
```

After install, `_resolve_blog_fonts()` will pick up Inter and matplotlib
will use it. No code change needed.

---

## Typography

| Parameter | Value | Notes |
|---|---|---|
| `font.family` | DejaVu Sans | Cross-platform, no LaTeX required |
| Body (`font.size`) | 10 pt | |
| Axis labels (`axes.labelsize`) | 11 pt | Slightly larger than body, per Perez |
| Title (`axes.titlesize`) | 11 pt | |
| Ticks (`xtick/ytick.labelsize`) | 9 pt | Smaller than labels OK |
| Legend (`legend.fontsize`) | 9 pt | |
| `pdf.fonttype` / `ps.fonttype` | 42 | Type-42 so fonts stay editable in Illustrator / Inkscape |
| `svg.fonttype` | "none" | Keep SVG text as text, not paths |

Multiply every font by `font_scale` (default 1.0). Use `font_scale=1.2`
for talk-slide variants of the same figure.

---

## Size

| Target | Figsize (inches) | Use for |
|---|---|---|
| `blog` | 6.5 × 4.0 | **Default** — clean-result bodies + mentor-update slides + LessWrong / Anthropic-blog posts |
| `neurips` | 5.5 × 3.4 | Paper figures — roughly one NeurIPS column |
| `generic` | 6.0 × 4.0 | Broader or taller paper-style layouts (rarely the right pick) |

Call `set_paper_style()` at the top of the script (defaults to `"blog"`).
Pass `set_paper_style("neurips")` only for figures destined for a paper
submission. Do not override `figsize` by hand unless the chart type
demands it (e.g. P7 multi-panel).

---

## Despine + grid

- `axes.spines.top = False`, `axes.spines.right = False` — strip visual
  clutter (Tufte / tvhahn principle).
- `axes.grid = True`, `grid.alpha = 0.25`, `grid.linewidth = 0.5` — grid is
  present but not distracting. Only the reader who NEEDS it sees it.

---

## Error bars

- `errorbar.capsize = 3`
- For a proportion with N trials: `lo, hi = proportion_ci(p, N)` (Wald 95%)
- For a mean across seeds: `std / √n`
- For a bootstrap CI: precompute elsewhere, pass in explicitly
- **Missing error bars = missing claim.** Every point on a chart that could
  vary across seeds or across an eval set needs bars.

---

## Save formats

Always both:
- `.png` — 300 DPI, Commit tag in pnginfo (greppable from file)
- `.pdf` — vector, Commit tag in PDF metadata
- `.meta.json` sidecar — commit hash + ISO-8601 UTC timestamp + figsize

Path convention: `figures/<aim-or-experiment>/<short-name>.{png,pdf,meta.json}`.

---

## Direction arrows

Every y-axis whose direction is non-obvious gets a `↑ better` or `↓ better`
suffix via `add_direction_arrow(ax, axis="y", direction="up")`.

Non-obvious examples:
- Loss (↓ better)
- Accuracy (↑ better — but readers know this; optional)
- KL divergence (↓ better — non-obvious, ALWAYS add)
- Alignment score (depends on the definition — ALWAYS add)
- Cosine similarity (depends — specify)

---

## Frame / legend

- `legend.frameon = True`, `legend.edgecolor = "lightgrey"`, `legend.facecolor = "white"`
- Legend LOC: pick the location that obscures the fewest data points; never
  rely on `best` for published figures (it flips between runs).
- Inside-axes > outside-axes unless the legend occupies > 25% of panel area.
---

## Paper (ICLR) target — context-answer-map paper

The `"iclr"` target is MANDATORY for any figure destined for the Overleaf
paper (the context-answer-map ICLR submission). For EPS paper figures, this
skill + `src/explore_persona_space/analysis/paper_plots.py` take precedence
over the generic `dataviz` skill.

### Author at final size — never rescale in LaTeX

The ICLR style file's `\textwidth` is 5.5 in. Figures are authored AT the
size they print and included in LaTeX WITHOUT a `width=` rescale — rescaling
silently changes every realized font size on the page. Exactly two
sanctioned widths:

| Helper | Size (inches) | Use for |
|---|---|---|
| `figsize_iclr_full(height_frac=0.618)` | 5.5 × 3.4 | Full-textwidth figures (the default) |
| `figsize_iclr_half()` | 2.75 × 1.70 | Side-by-side minipage pairs |
| `figsize_iclr_panels(ncols, height_in=None)` | 5.5 × (5.5/ncols × 0.618) | Full-textwidth multi-panel rows (full width; height defaults to one panel's golden-ratio height) |

Sizes at final size: body / axis labels / titles 9 pt, ticks / legend
7.5 pt — nothing below 7 pt anywhere. Font: Times-alike serif chain
(Times New Roman → Times → TeX Gyre Termes → Nimbus Roman → Liberation
Serif) with STIX mathtext; resolution FAILS LOUD when none of the chain is
installed — never a silent DejaVu Serif fallback. White OPAQUE background
everywhere (figure + axes + savefig; `savefig.transparent = False`). No
grid, top/right spines off, frameless legend, Type-42 font embedding.

### `PAPER_COLORS` — one colour = one meaning PAPER-WIDE

Stricter than the per-writeup one-color-one-meaning rule: these assignments
hold across EVERY figure in the paper. Bind colours by CONCEPT via
`paper_color(<concept>)` (raises `KeyError` naming the valid keys on a
miss — no silent default), never by prop-cycle position.

| Concept | Hex | Meaning |
|---|---|---|
| `instruct` | `#0072B2` blue | instruct model / the paper's featured arm |
| `base` | `#E69F00` orange | base (pretrained) model |
| `identity_bias` | `#009E73` green | identity-plus-bias baseline |
| `neural_map` | `#D55E00` vermilion | MLP / neural map variants |
| `oracle_answer` | `#CC79A7` purple | real-answer oracle |
| `persona_vector` | `#56B4E9` sky blue | persona-vector method |
| `null` | `#999999` gray | null / random / pathological arms |
| `reference` | `#000000` black | ground truth / reference lines |

### Hatch ban + heatmap discipline

- **Hatching is BANNED** under the `iclr` target. Encode distinctions via
  colour + marker/linestyle, or reduced alpha + a caption note.
- **Heatmaps:** `viridis` for sequential data; `RdBu_r` for diverging data
  (centered on the natural zero). Side-by-side COMPARABLE panels MUST share
  one colour scale (same `vmin`/`vmax`, one shared colorbar) — per-panel
  auto-scaling makes the cross-panel read meaningless.

### Regenerating legacy figures for the paper

- Strip on-canvas methodology text, issue numbers, and per-bar value labels
  into the LaTeX caption (consistent with the standing
  no-caption-block-inside-the-plot rule).
- EXTEND the figure's EXISTING generating script with a `--style iclr` flag
  — never a parallel one-off regeneration script.
- Write paper variants BESIDE the originals via `EPS_PLOT_STEM_SUFFIX`
  (e.g. `_iclr`); NEVER overwrite stems other writeups reference.

### Output naming + Overleaf handoff

Paper figures land at `figures/paper/<slug>.{pdf,png,meta.json}` in the EPS
repo (`savefig_paper(fig, "<slug>", dir="figures/paper/")`), and the PDF is
copied into the Overleaf clone's `figures/` directory
(`~/overleaf-6a59c927/figures/` for the context-answer-map paper) — then
commit + push the Overleaf clone (Thomas reads only in Overleaf).

### Captions

Every error bar's caption states the ESTIMATOR (mean / median /
proportion), the INTERVAL TYPE (95% bootstrap CI / Wald CI / ±1 SEM across
seeds), and n. A bar missing any of the three is an unreadable claim.
