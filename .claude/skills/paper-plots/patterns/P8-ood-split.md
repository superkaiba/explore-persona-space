# P8 — In-Distribution vs Out-of-Distribution Side-by-Side

**Use when:** Showing the same metric evaluated on the training-distribution
eval set AND on a held-out OOD set. The IID vs OOD gap is usually the point.

**Do NOT use when:**
- You only have IID results — say so in caveats and use P1.
- OOD is evaluated on a different metric than IID — don't split; use two
  separate figures and label axes differently.

## Minimal example

```python
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips")

conditions = ["baseline", "tulu_25", "tulu_100"]
iid = np.array([0.91, 0.82, 0.70])
ood = np.array([0.83, 0.71, 0.58])
n_iid = n_ood = 200

iid_cis = np.array([proportion_ci(p, n_iid) for p in iid])
ood_cis = np.array([proportion_ci(p, n_ood) for p in ood])

x = np.arange(len(conditions))
width = 0.35

c_iid, c_ood = paper_palette(2)
fig, ax = plt.subplots()

bars_iid = ax.bar(
    x - width / 2, iid,
    yerr=[iid - iid_cis[:, 0], iid_cis[:, 1] - iid],
    width=width, color=c_iid, capsize=3, label="IID",
)
bars_ood = ax.bar(
    x + width / 2, ood,
    yerr=[ood - ood_cis[:, 0], ood_cis[:, 1] - ood],
    width=width, color=c_ood, capsize=3, label="OOD",
)

# Annotate the IID-OOD gap per condition
for i in range(len(conditions)):
    gap = iid[i] - ood[i]
    ax.text(
        x[i], max(iid[i], ood[i]) + 0.05,
        f"Δ={gap:+.2f}", ha="center", fontsize=8, color="black",
    )

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Alignment rate")
add_direction_arrow(ax, axis="y", direction="up")
ax.legend(loc="upper right")

savefig_paper(fig, "aim5/iid_vs_ood", dir="figures/")
plt.close(fig)
```

## Pitfalls

- **Don't hide the gap.** The Δ between IID and OOD is the claim — annotate
  it explicitly on the figure. Readers should not need to eyeball.
- **Don't mix metrics across splits.** If IID is alignment-rate and OOD is
  F1, you have two figures, not one.
- **Don't omit OOD when you have it.** IID-only results are deeply limited;
  even a noisy OOD bar is better than no OOD bar, and honesty about the gap
  earns trust.
- **Don't use stacked bars to show IID+OOD.** Grouped, side-by-side. Always.
