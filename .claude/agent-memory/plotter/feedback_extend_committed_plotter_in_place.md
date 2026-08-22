---
name: extend-committed-plotter-in-place
description: Run/extend the issue's committed manifest-driven plotter in place; convert its _save to savefig_paper; use function-local imports (ruff hook strips top-level F401); zoom-companion panels for degenerate normalized-fraction outliers
metadata:
  type: feedback
---

Three lessons from the #2162 figure round (first v2 dogfood plotter run):

1. **A committed `scripts/issue<N>_figures.py` with an empty `figures/issue_<N>/` is the
   normal committed-but-never-run shape** — run it (paper-plots Phase 0 disposition 1) and
   extend IN PLACE (disposition 2) where it misses the plotter contract. The usual gaps in a
   review-round-authored plotter: custom `fig.savefig` PNG-only save (convert its `_save` to
   `savefig_paper(fig, name, dir=out_dir)` then merge the script's `inputs` list into the
   sidecar it wrote), no `set_paper_style("blog")` call (add in `main()` before any subplots),
   and open markers (`facecolors="none"`) without explicit `linewidths` (blog style zeroes
   marker edges — invisible series).
   **Why:** the reuse rule + 5 rounds of encoded review conventions beat a rewrite; the
   contract (PNG+PDF+meta, house style) still binds.
   **How to apply:** Phase 0 reuse check first, run, then minimal in-place edits; script shows
   `M` in git — expected, orchestrator commits it (HOLD mode: never commit figures yourself).

1b. **The open-marker/zero-edge-width defect SHIPS in committed renders and passes casual
   visual review** (#2162 follow-up round: anchor_separation committed with 13/36 gate-failed
   hollow carriers invisible — the missing points sat near y=0 and looked plausibly absent).
   **How to verify:** don't trust the eyeball — pixel-scan the expected band
   (`np.asarray(im.crop(band))`, count palette-colored pixels; 0 = defect) after checking the
   source JSON has non-None values there. **Fix recipe (clean provenance):** add
   `linewidths=1.0` to the scatter call IN the committed script; commit the script fix first;
   `git checkout HEAD -- <figure files>` so the tracked tree is clean; render into a temp dir
   (provenance uses `git status --untracked-files=no`, so untracked temp output keeps
   `git_dirty=False`); move into place; commit figures by explicit path. Rendering in place
   after the script commit still stamps `+dirty` because the render mutates the tracked
   figure files mid-save.

2. **The PostToolUse ruff hook strips just-added top-level imports (F401) if the same edit
   does not add a usage.** For paper_plots imports into an existing script, import
   function-locally at the usage sites (`from ...paper_plots import savefig_paper` inside
   `_save`) — hook-proof, no ordering dance.

3. **Normalized fraction-of-swap style DVs (F = (patched−floor)/(ceiling−floor)) blow up on
   separation-degenerate units** (|F| up to ~100 when the anchor gap is near zero). Any
   per-pair/per-unit view WITHOUT the separation exclusion needs a full-range panel PLUS a
   "same points, zoom |F| <= k" companion panel (never drop points, never silently clip);
   per-cell-mean scatters need the high-leverage point labeled large + an inset with explicit
   stated x/y windows. Captions state the windows and the mechanism factually (out-of-[0,1]
   F means the patched value fell outside the anchor range) — no interpretation.
