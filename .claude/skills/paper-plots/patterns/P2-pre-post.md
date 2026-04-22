# P2 — Pre vs Post Intervention (Paired)

**Use when:** Showing how a metric changes from *before* to *after* an
intervention (e.g. before EM induction vs after), ideally with paired
seeds or runs.

**Do NOT use when:**
- No pairing (unpaired comparison) → P1 (bar).
- More than 2 time points → P3 (line).

## Minimal example

```python
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips")

# Paired pre/post values, one per seed
pre = np.array([0.92, 0.90, 0.93, 0.91, 0.89])
post = np.array([0.68, 0.71, 0.66, 0.70, 0.69])
seeds = [42, 137, 256, 512, 1024]

fig, ax = plt.subplots()
c_pre, c_post = paper_palette(2)

# Individual seed lines (light) + mean trend (bold)
for i in range(len(seeds)):
    ax.plot(
        [0, 1], [pre[i], post[i]],
        color="lightgrey", lw=1.0, zorder=1,
    )
ax.plot(
    [0, 1], [pre.mean(), post.mean()],
    color="black", lw=2.0, marker="o", zorder=3, label="mean",
)
ax.errorbar(
    [0, 1], [pre.mean(), post.mean()],
    yerr=[pre.std(ddof=1) / len(pre) ** 0.5, post.std(ddof=1) / len(post) ** 0.5],
    fmt="none", ecolor="black", capsize=3, zorder=2,
)

ax.scatter([0] * len(pre), pre, color=c_pre, s=30, zorder=2, label="pre")
ax.scatter([1] * len(post), post, color=c_post, s=30, zorder=2, label="post")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Pre EM", "Post EM"])
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Alignment rate")
add_direction_arrow(ax, axis="y", direction="up")
ax.legend(loc="lower left")

# Annotate the delta on the mean line
delta = post.mean() - pre.mean()
ax.annotate(
    f"Δ = {delta:+.3f}",
    xy=(0.5, (pre.mean() + post.mean()) / 2),
    ha="center", va="center",
    fontsize=9, color="black",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey", lw=0.5),
)

savefig_paper(fig, "aim5/em_pre_post", dir="figures/")
plt.close(fig)
```

## Error-bar source

- Per-point error bar = `std / √n_seeds` (standard error of the mean).
- If only 1 seed, drop error bars and annotate `n=1, CI unknown`.

## Pitfalls

- **Don't swap the x-axis ends.** Pre on the left, post on the right.
  Always.
- **Don't hide individual seeds.** The grey lines are what make the story
  honest — you can see if the "improvement" is driven by one outlier seed.
- **Don't use a bar chart for paired data.** Bars hide the pairing; the
  reader cannot tell if every seed moved in the same direction.
- **Don't forget the delta annotation.** The Δ with sign + magnitude is the
  headline number; it belongs ON the figure, not just in the caption.
