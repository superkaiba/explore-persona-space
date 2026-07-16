# Pattern B0 — Blog-style bar comparison

The Anthropic-blog-register variant of `P1-bar-comparison`. Use for
clean-result body figures and mentor-update slides. For paper-figure
bars (narrow column, dense), use `P1-bar-comparison.md` with
`set_paper_style("neurips")` instead.

Style invariants are documented in `../style-reference.md` §
"`blog` variant — full rcParams delta". This pattern is the worked
example.

---

## When to use

- Comparing 2-4 conditions on a single metric, destined for a clean-result
  issue body or a mentor slide.
- The reader will see the figure inline on github.com (light mode) or
  rendered in a Marp deck on Zoom screenshare.
- Sample sizes are equal across bars (otherwise prefer P2 / P5).
- Polished, slightly editorial register — "this is the headline" rather
  than "this is one of fifteen panels".

## Worked example

```python
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    set_paper_style,
    set_title_subtitle,
    paper_palette_role,
    proportion_ci,
    savefig_paper,
)

set_paper_style("blog")

# --- Data --------------------------------------------------------------
labels = ["Primary", "Baseline", "Control"]
vals = [0.70, 0.40, 0.55]
ns = [200, 200, 200]

# 95% Wald CIs as asymmetric (lo, hi) offsets so error bars never cross 0/1.
# errorbar offsets must be >= 0 (gotchas.md)
err_lo, err_hi = [], []
for v, n in zip(vals, ns):
    lo, hi = proportion_ci(v, n)
    err_lo.append(max(0.0, v - lo))
    err_hi.append(max(0.0, hi - v))

# --- Plot --------------------------------------------------------------
fig, ax = plt.subplots()

colors = [
    paper_palette_role("primary"),
    paper_palette_role("baseline"),
    paper_palette_role("control"),
]

ax.bar(
    range(len(labels)),
    vals,
    color=colors,
    width=0.55,                  # narrower than mpl default (0.8) for breathing room
    yerr=[err_lo, err_hi],
    error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel("Source-rate")
ax.set_ylim(0, max(vals) + max(err_hi) + 0.05)

set_title_subtitle(
    ax,
    "Marker uptake holds across personas",
    subtitle=f"Source-rate per persona, n={ns[0]} each (95% Wald CI)",
    source="Source: eval_results/issue_281, commit abc1234",
)

savefig_paper(fig, "issue_281/marker_uptake", dir="figures/")
```

## Why these defaults

- **`width=0.55`** rather than mpl's default 0.8 — gives the bars
  breathing room. Tightly-packed bars read as "dense table"; spaced
  bars read as "comparison".
- **Asymmetric error bars** via `[err_lo, err_hi]` — proportion_ci can
  clamp at 0 / 1, so the lower and upper offsets aren't equal. Using
  symmetric `±` would push error bars below 0 for low-rate conditions.
- **`error_kw` overrides** — thinner elinewidth (0.8) and dark-grey
  ecolor (#1A1A1A) so the error bars read as "uncertainty annotation"
  rather than competing with the bar fill.
- **`paper_palette_role(...)`** instead of integer slot indexing — when
  you switch the script to `set_paper_style("neurips")` for the paper
  version, the same role names map to the Wong palette without any
  other code change.
- **`set_title_subtitle(...)`** instead of `ax.set_title(...)` — the
  Anthropic-blog title block (bold + subtitle + source) is the
  intended look for blog-register figures. Don't add subtitles via
  `fig.suptitle()` and titles via `ax.set_title()` separately; they
  fight for the same vertical space.

## Common variants

- **Two bars instead of three.** Drop one element from each list. The
  pattern scales to 2-5 bars. For 6+, switch to a horizontal layout
  (rotate via `ax.barh`) or a small-multiples panel (P7).
- **Slide-deck variant.** Add `font_scale=1.2` to `set_paper_style`
  for legibility on a Zoom screenshare. Same script, otherwise.
- **No source line.** Drop the `source=` kwarg. The figure still gets
  commit-pinned via `savefig_paper`'s sidecar `.meta.json`; the source
  line is for at-a-glance provenance, not durable metadata.
- **Direct value labels.** DO NOT add value labels to bars. The saved
  user feedback (`feedback_no_plot_annotations`) requires keeping
  figures clean — let the y-axis carry the reading job. If a specific
  number is load-bearing, mention it in the caption prose, not on the
  chart.

## Caption (lives in the issue body, not the figure)

The figure is half the deliverable; the caption is the other half. For
the prose convention see
`.claude/skills/clean-results/SPEC.md` §8. Caption goes
in the `### Result N` section directly under the embedded figure, opens
with `**Figure N.**`, ~30 words, states the load-bearing finding (not
the chart axes).
