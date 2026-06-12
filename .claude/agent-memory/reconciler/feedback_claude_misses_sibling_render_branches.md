---
name: Claude misses sibling render-branches in same figure function
description: When round-N "wires" a dynamic exclusion set into a figure function with multiple rendering branches, Claude tick-walks the first branch and PASSes; Codex catches that the sibling rendering paths in the same function ignore the new parameter.
type: feedback
---

When round-N's task is to wire a dynamic-exclusion / alpha / dimming set
(`excluded: tuple[str, ...]`) into a figure-builder function with MULTIPLE
rendering branches — one per data series — Claude PASSes after verifying:
(a) the function signature accepts the kwarg, (b) the call site computes the
set correctly, (c) ONE rendering branch (typically the first one Claude
reads) uses the kwarg via `alpha = 0.4 if cell in excluded else 1.0`.

Codex catches that the OTHER rendering branches in the same function —
typically a `ax.plot(xs, ys, ...)` over a list-comprehension over the OTHER
data dicts — never consult `excluded`. A contaminated cell in those dicts
renders opaque, exactly the bug the dynamic exclusion was meant to fix.

**Why:** the smell is a single-function diff where the "fix" sits inside a
`for cell, ej in <one_dict>.items():` loop applying `alpha = 0.4 if cell in
excluded else 1.0`, while the SAME function has sibling list-comprehension
branches `pts = [_cell_xy(ej) for ej in <other_dict>.values()]; ax.plot(xs,
ys, ...)` that don't iterate per-cell at all. The pivot ADDED the new dicts;
the cells in those new dicts are EXACTLY the ones the pivot makes
contaminable. Tests typically cover only the first-branch case (a cell from
the original dict in `excluded`) and don't exercise a cell from the new
dicts.

**How to apply:** in any code-review reconcile where the round adds or
modifies a `*_figure(*, ..., excluded=...)` parameter to dim contaminated
cells, before believing Claude:
1. Open the figure function, list every `ax.scatter` / `ax.plot` /
   `ax.fill_between` call site inside it.
2. For each call, identify which data dict it iterates / lists. If it
   iterates a dict that can contain cells subject to the same exclusion
   rule, verify the per-cell alpha check is present (or the connector line
   is split into per-cell scatter + connector plot).
3. Check the new tests: do they exercise a cell from EACH renderable dict
   in `excluded`, or only the original dict's cell?
4. If a sibling branch ignores `excluded`, the round-(N-1) "wire it in"
   concern is HALF-CLOSED, not closed — verdict is FAIL.

Specific case shape:
- `hero_figure(*, lora_cells, ft_508_cells, ft_514_dense_cells,
  ft_514_lowlr_cells, excluded)` where ft_508 loop applies `alpha = 0.4 if
  cell in excluded else 1.0` but the ft_514_dense and ft_514_lowlr branches
  use `ax.plot([x for x in dict.values()], ...)` with no per-cell check.

Origin: task #514 round-4 (pivot). Round-3 binding concern B11
"plot-dynamic-exclusion-not-wired" was supposed to be closed by the round-4
fix at `scripts/plot_issue_514.py:170-256`. Lines 216-231 (the #508 anchor
loop) correctly apply `alpha = 0.4 if cell in excluded or cell == "ft_b3"
else 1.0`. Lines 234-241 (#514 dense lever) and lines 243-256 (#514 lower-LR
lever) use plain `ax.plot(xs, ys, ...)` with no per-cell check.
`tests/test_issue514_plot.py:441-480` proves `compute_excluded_cells` would
return `ft_dense_b30` if contaminated, but
`tests/test_issue514_plot.py:397-438`'s only `excluded` test case uses an
`ft_b1` cell (an ft_508 cell, in the working branch). The pivot's own #514
cells get NO dimming protection. Companion to "Claude partial-fix-pattern
blindness across parallel files" + "Claude treats round-N-1 must-fix as
acceptance" + "Claude misses sibling resampler inconsistency".
