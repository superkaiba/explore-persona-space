---
name: Claude misses exclusion-constant applied in figure-layer but not analyze-layer
description: When a round-(N-1) blocker fix adds an EXCLUDED_FROM_BOOTSTRAP / clean-cell filter constant for known-degenerate cells, Claude PASSes round-N by verifying the constant exists in the figure script and the figure renders correctly. Codex catches that the SAME constant is NOT applied in the analyze.py local-read / interp / bootstrap-anchor selection — the figure correctly de-emphasizes the bad anchor (half-transparency / dashed bar) but the numeric read still interpolates THROUGH it.
type: feedback
---

When the round-N "fix" introduces an exclusion-constant (e.g. `EXCLUDED_FROM_BOOTSTRAP = ("ft_b2",)`) for known-degenerate / collapsed / single-probe anchor cells, Claude code-reviewer PASSes after walking:
- ✓ the constant is defined,
- ✓ the figure script applies it (the bad cell renders at half-alpha or with a separate marker style),
- ✓ the figure smoke renders the expected hero PNG.

Codex catches the parallel `analyze.py` / `_compute_local_*_read` / candidate-anchor filter that has its OWN line picking which anchors enter the LINEAR INTERPOLATION / BOOTSTRAP / matched-rate computation — and that loop does NOT import the constant or apply the same per-cell quality filter. The figure visually de-emphasizes the bad anchor; the numeric read silently interpolates THROUGH it, re-introducing the indeterminacy the constant was created to resolve.

**Smell:** the constant lives in `scripts/plot_*.py` only. If you grep the project for `EXCLUDED_FROM_BOOTSTRAP` (or its equivalent) and the only hits are in the figure script + that script's docstring, the analyze layer is unfiltered.

**Why:** the figure is the visible artifact (Claude sees the hero PNG, ticks the box), but the load-bearing scientific claim is read off the numeric analyze.py output — `local_read_nat`, `gap_nat`, `determinate=True/False`. The single-probe degenerate cell (N=1, low source_mean) at e.g. (6.774 nat, -0.865 nat) becomes the lower-flank bracket for a target like 8 nat and the linear-interp answer is half-read off the degenerate value. The figure correctly shows ft_b2 at half-transparency; the JSON the figure-caption claims comes from STILL has the contamination baked in.

**How to apply:** When the round-(N-1) blocker fix introduces a per-cell exclusion list / quality filter, grep the constant name across the worktree. If hits are concentrated in `scripts/plot_*.py` and absent from `src/explore_persona_space/experiments/*/analyze.py` (or the equivalent analyze module), the analyze layer is the contaminated half. The minimal fix is two lines (move the constant to `analyze.py` so it's the single source of truth, OR import it into the analyze module). Persist `excluded_anchor_cells: [...]` into the analyze JSON so the determinacy report tells the reader which anchors were used. Origin: task #514 round-2 — `EXCLUDED_FROM_BOOTSTRAP = ("ft_b2",)` defined at `scripts/plot_issue_514.py:46`, not applied in `src/explore_persona_space/experiments/full_ft_regime_514/analyze.py:_compute_local_matched_rate_read` lines 220-222 (`if d["source_mean"] is not None and d["held_out_mean"] is not None: candidates.append(d)` — accepted ft_b2's degenerate single-probe held-out value as a lower-flank anchor for 8-nat target). Companion to "Claude misses fix regressions" + "Claude misses dispatcher-wiring correctness bugs" + "Claude misses same-file siblings": same family of "the fix lands in the visible/producing module but the consumer/filter side keeps the old behavior."
