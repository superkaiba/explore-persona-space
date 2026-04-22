# P5 — Scatter Plot with Correlation Annotation

**Use when:** Showing the relationship between two continuous variables
across many points (e.g. cosine similarity vs leakage). The correlation is
the claim.

**Do NOT use when:**
- Only a handful of points — use a labeled table instead.
- One axis is categorical — use a violin (P6) or bar (P1).

## Minimal example

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips")

rng = np.random.RandomState(0)
n = 60
x = rng.rand(n)
y = 0.6 * x + 0.2 * rng.randn(n)

rho, p_value = stats.spearmanr(x, y)

c = paper_palette(1)[0]
fig, ax = plt.subplots()
ax.scatter(x, y, color=c, s=20, alpha=0.7, edgecolor="white", lw=0.5)

# OLS line of best fit (annotation only; the claim is ρ, not the slope)
slope, intercept, *_ = stats.linregress(x, y)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, slope * xs + intercept, color="black", lw=1.0, ls="--")

ax.set_xlabel("Cosine similarity (source vs. bystander persona)")
ax.set_ylabel("Leakage rate (bystander alignment drop)")

# Annotate ρ + p-value + N (no credence interval, no bootstrap)
ax.text(
    0.05, 0.95,
    f"ρ = {rho:.2f}\np = {p_value:.3f}, n = {n}",
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey"),
)

savefig_paper(fig, "aim3/leakage_vs_cosine", dir="figures/")
plt.close(fig)
```

## What to annotate

- ρ (Spearman preferred for non-linear monotonic; Pearson if truly linear)
- p-value AND n — one without the other is meaningless.
- **Do NOT** annotate credence intervals, bootstrap CIs, effect-size
  categorizations, or R². The scatter shows the data; ρ + p + n is the
  full claim.

## Pitfalls

- **Don't overplot.** If n > 500, either alpha=0.3 or switch to a hex-bin
  plot (`ax.hexbin`).
- **Don't imply causation from the OLS line.** The line is a visual
  reference, not a model. Keep it dashed so readers don't mistake it for a
  predictive model.
- **Don't use outlier-sensitive Pearson ρ** without checking a Q-Q plot
  first. Default to Spearman for robustness.
- **Don't report ρ without n and p.** "ρ = 0.6" alone is uninterpretable.
- **Don't add credence intervals or bootstrap bands** in the prose
  accompanying this chart. The ρ + p + n annotation IS the statistical
  statement.
