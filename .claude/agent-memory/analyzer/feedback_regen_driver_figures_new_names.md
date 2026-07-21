---
name: Regenerate driver-rendered figures under NEW filenames
description: P6/driver-rendered PNGs often lack .meta.json sidecars and hide broken panels; regenerate via savefig_paper on MAIN under new names to dodge branch/main binary merge conflicts
type: feedback
---

When a pod/VM analysis phase (P6-style driver) pre-renders figures, inspect them BEFORE reuse: on #1482 two of five had a silently broken panel (an empty axes; a per-feature panel plotting a DIVERGED MLP arm clipped at −1 that read as a finding), and none had `.meta.json` sidecars (driver used bare savefig).

**Why:** verify_task_body check 26 needs sidecars at the pin, and a diverged/empty panel shipped as-is is an interpretation hazard; also committing regenerated figures to MAIN under the SAME filenames as the worktree-branch copies sets up binary merge conflicts at the Step 10d rebase-merge.

**How to apply:** regenerate the final set via `savefig_paper` (PNG+PDF+meta) directly in the MAIN checkout under NEW descriptive filenames (`category_error_bars` vs P6's `hero1_category_error`), note in the interpretation/body that the branch P6 figures are superseded and why, and visually verify every panel (a pathological arm, e.g. a diverged fit, gets EXCLUDED with an explanation — never plotted clipped).
