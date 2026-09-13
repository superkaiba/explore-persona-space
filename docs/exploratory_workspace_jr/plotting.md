# Exploratory comparison figures

`workspace_jr_plot.py` reads an uploaded complete 48-cell comparison, verifies its
completion, cohort, report and summary bindings, and renders five comparison
figures: native component R², MLP gains, native/control/null gaps, sparsity
sensitivity and paired stronger-minus-weaker gaps. It refuses an incomplete grid.
Direction diagnostics, noise, learning curves and decomposition agreement have
separate source artifacts and are additional reporting requirements.

Each figure exports the project-standard vector PDF, 240-DPI color PNG and
grayscale audit, plus a sidecar with every plotted estimate, interval, exact
source/output hash, resolved font and caption. Plot intervals directly between
their saved endpoints; percentile intervals need not contain the point estimate.
Undefined estimates and intervals remain visibly undefined. No bars start at
zero in place of missing results. Figures use the appropriate model-specific or
shared context cohort and distinguish ridge and MLP by color and shape.

Before presenting results, render using the actual completed analysis, inspect
both color and grayscale output, upload the artifacts, verify the upload, and
include browser-accessible URLs. The renderer does not edit manuscript files.
