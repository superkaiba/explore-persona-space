---
name: set-title-subtitle-overlap-blog
description: paper_plots set_title_subtitle overlaps the axes when the blog style is active because savefig.bbox="tight" overrides subplots_adjust; use manual ax.set_title(pad=36) + ax.annotate(...) + fig.supxlabel(...) instead
metadata:
  type: feedback
---

The blog style sets `savefig.bbox: "tight"` which strips out any manual `fig.subplots_adjust(top=0.80)` whitespace — the title block in `set_title_subtitle()` keeps overlapping the topmost data points / annotation.

**Why:** `bbox="tight"` recomputes the figure bbox at save time to fit the artists, so adding top whitespace via subplots_adjust is ignored.

**How to apply:** When using `set_paper_style("blog")` + a left-aligned title-subtitle block in a clean-result figure, drop `set_title_subtitle()` and inline its three calls with extra `pad`:

```python
ax.set_title(TITLE, loc="left", fontsize=13, fontweight="semibold", pad=36)  # pad=36 not 24
ax.annotate(SUBTITLE, xy=(0.0, 1.0), xytext=(0, 8), xycoords="axes fraction",
            textcoords="offset points", ha="left", va="bottom",
            color="#5A5A5A", fontsize=10)
fig.supxlabel(SOURCE, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
fig.tight_layout()
```

The default `pad=24` in `set_title_subtitle` is too tight for the blog style. Use `pad=36` so the title clears the subtitle's two-line annotation.

Incident: task #468 clean-result figures — all 4 figures had title clipping into the top bar / line / histogram bin until manual pad bump.
