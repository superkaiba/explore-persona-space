---
name: v2-report-gate-recipe
description: Working recipe from the FIRST v2 report gate run (#2162) — sidecar quirks, path moves, verify_report invocation
metadata:
  type: project
---

First `workflow: v2` report verified: #2162 (2026-08-07, round 1 PASS). Facts that made the gate work:

- **Task moves `interpreting` → `reviewing` mid-gate** (v2 status semantics). Re-resolve the body path via `scripts/task.py find <N>` if a read fails; never cache `tasks/<status>/<N>/` paths.
- **Generation-mode invocation that works:** `uv run python scripts/verify_report.py --file <body> --mode generation --manifest <task>/artifacts/planned_manifest.json --figures-root <issue worktree> --expect-issue <N>` — `--figures-root` must be the issue worktree or blob-identity degrades to WARN.
- **Sidecar (`.meta.json`) structure:** `points` is CAPPED at 2,000 embedded entries; `total_points` carries the TRUE rendered count (use it for caption-n checks). Heatmap figures have NO `points` key at all — verify those against the persisted derived table + the rendered annotations on the PNG. Panel text (screens, rho legends) lives in `text.axes[*].legend_labels` / `title_left`; in-axes text boxes may NOT be captured — read the PNG.
- **Efficient recompute pattern:** build the (cell,slot,arm)→mean table from the source JSONLs, then set-match against every numeric leaf in sidecar `points` (tolerance 1e-6) instead of reverse-engineering the plot layout. One unmatched value = investigate; here it isolated a real one-pair transform nuance ([[fbeh-paired-drop-convention]]).
- **verify_report.py gap (surfaced as workflow-fix candidate 2026-08-07):** its lexicon/interpretivity scan covers the BODY only, not `docs/reports/issue_<N>_detailed.md` — scan the companion manually (Motivation keeps the hypothesis exemption; deviations-section "confirmed from artifacts" / "the plan's phrasing implies" are benign process-facts, not findings).

Round-2 additions (persona-specificity-ladder fold, 2026-08-12):

- **Sidecar drops genuine (0,0) scatter points.** `paper_plots._extract_scatters` skips any single-point collection at exactly (0,0) as "matplotlib's empty-collection default" — per-point `ax.scatter([x],[y])` loops lose every real (0,0) datum from `points` AND `total_points` (ladder_percarrier: 252 captured vs 264 rendered). Before flagging a missing-points FAIL, census the missing values' (x,y): all-exactly-(0,0) ⇒ capture artifact — confirm by zoom-reading the PNG region; the figure is usually fine.
- **Sidecar y-KEY is per-axes:** only axes with a set ylabel key points by that label; other panels key `"y"`. Value-match per point's own numeric key, never pts[0]'s key.
- **CI bounds in figures come from `stats.json`, not a fresh bootstrap.** Match figure CIs against the stats.json estimation block FIRST; an independent per-cell `default_rng(seed)` bootstrap reproduces cells at the modal n exactly but can drift ~1e-3–1e-2 at off-modal n (index-draw layout) — means always match exactly. Not a defect.
- **Multiset value-diffs attribute duplicates arbitrarily** (many 0.0s in null arms) — pin WHICH rows are missing by x-position census, not by first-match enumeration.

**Why:** the first run burned time discovering each of these; they are stable properties of the v2 pipeline.
**How to apply:** every future report-verifier invocation on a v2 task.
