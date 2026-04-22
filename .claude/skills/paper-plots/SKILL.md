---
name: paper-plots
description: >
  Generate publication-quality matplotlib/seaborn charts for the project.
  Auto-applies the paper-quality rcParams from
  `src/explore_persona_space/analysis/paper_plots.py`. Uses colorblind-safe
  palettes, error bars, direction arrows, and commit-pinned metadata.
  Invoke when building any figure destined for a clean-result issue, the
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

- ANY figure going into a clean-result issue, a `research_log/drafts/`
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
    paper_palette,       # colorblind-safe hex colors (≤ 8)
    proportion_ci,       # 95% Wald CI for a proportion: p ± 1.96·√(p(1-p)/n)
)
```

Always call `set_paper_style()` BEFORE any `plt.subplots(...)` call. Rcparams
set afterward are applied to NEW artists only.

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
bars (Chua/Hughes rule; see `.claude/skills/clean-results/principles.md`).

### 2. Determine configuration

| Decision | Default | When to override |
|---|---|---|
| Chart type | Pick a pattern from `patterns/` — see index below | |
| Target size | `set_paper_style("neurips")` → 5.5 × 3.4 in | `"generic"` for broader layouts |
| Palette | `paper_palette(n)` for n series | Set `cycler` manually only if palette order matters for meaning |
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

### 4. Run & save

```python
import matplotlib.pyplot as plt
from explore_persona_space.analysis.paper_plots import set_paper_style, savefig_paper

set_paper_style("neurips")
fig, ax = plt.subplots()
# ... build ...
savefig_paper(fig, "aim5/pre_post_alignment", dir="figures/")
plt.close(fig)
```

`savefig_paper` writes:
- `figures/aim5/pre_post_alignment.png` (300 DPI, commit-tagged via pnginfo)
- `figures/aim5/pre_post_alignment.pdf` (vector, commit-tagged via PDF metadata)
- `figures/aim5/pre_post_alignment.meta.json` (commit + timestamp + figsize)

The sidecar `.meta.json` is what makes figure provenance auditable later.

### 5. Verify

Run this checklist against the saved figure. Inlined from
`.claude/skills/clean-results/checklist.md` §3 — defer to that file if they
ever diverge.

- [ ] Axes labeled, including units.
- [ ] `add_direction_arrow(ax, ...)` applied where "higher = better" or
      "lower = better" isn't obvious from the metric name.
- [ ] Error bars present OR an explicit note explains why they aren't.
- [ ] ≤ 3-5 distinguishable colors. Legend order matches the narrative (the
      claim first, the baseline second).
- [ ] No microscopic text. Squint test: readable on a video call at 75% zoom.
- [ ] Colorblind-friendly palette (use `paper_palette(n)`).
- [ ] Both `.png` and `.pdf` written. Sidecar `.meta.json` exists.
- [ ] The caption (in the clean-result body) is ONE sentence: what the reader
      should look at, what the axes mean, what the takeaway is.
- [ ] No diagonal axis labels. Rotate by ≤ 30° and only if needed.

**Hard cap: 3 visual-iteration rounds.** If the figure still doesn't look
right after 3 revisions, stop and report rather than iterating further — the
bottleneck is usually the claim, not the chart.

---

## Pitfalls to avoid

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
