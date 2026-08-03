---
name: paper_palette_role valid role names
description: paper_plots.paper_palette_role accepts only accent/baseline/control/neutral/primary — "secondary" raises ValueError at figure render time
type: feedback
---

`explore_persona_space.analysis.paper_plots.paper_palette_role(role)` accepts ONLY
`{accent, baseline, control, neutral, primary}` — `"secondary"` (the intuitive
name for a reference/overlay color) raises `ValueError` at render time.

**Why:** task #555 round 1 (2026-06-10) — a new figure script used
`paper_palette_role("secondary")` for the parent-reference overlay; import-check
and lint both passed, the crash only surfaced in the per-phase figures smoke.
Use `"accent"` for reference/overlay lines.

**How to apply:** when forking any `issue*_make_figures.py`, grep the existing
script's `paper_palette_role(` calls and stick to the five valid names; the
figures smoke phase (run the script end-to-end on a fixture) is what catches it.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [paper_palette_role valid names](feedback_paper_palette_role_names.md) — only accent/baseline/control/neutral/primary; "secondary" ValueErrors at render.
