# Iteration log — clean-result round 2 (2026-06-13)

Round 2 of the clean-result-critique loop. Inputs: ensemble (Claude + Codex) round 1 REVISE+REVISE on the round-3-regenerated body.

## Round 2 blockers addressed

1. **Lens 7 — bracketed-CI prose (FAIL).** Four prose locations in `## Human TL;DR` Takeaway #3 and the two Phase-2 finding read paragraphs carried `(95% panel-CI [lo, hi])` inline. Rewrote all four as plain-English framings: "panel-row CI strictly positive, p ≪ 0.001" for the partialled ρ cases, "panel-row CI straddles zero, upper edge ~0.08" for the LOCO CV R² case, and "the Monte-Carlo CI is ~14× narrower than the panel-row CI" for the JS-estimator comparison sentence. Also rewrote the Reproducibility Parameters `Phase 2 mc_ci (JS-estimator)` table cell to drop the `[0.236, 0.252] (` form (the audit script's `interval_inline` regex does not skip table cells, so the table-cell exemption needed enforcement at the prose level; the exact numerical CI lives in `js_regression.json` as before). Bracketed CIs in the 4-row stylization-stratified table at lines 117-122 were left untouched per the orchestrator brief (table cells are allowed in the spec).

2. **Lens 8 — State facts, not sources (FAIL).** Dropped "Codex also pointed out" from the third Findings H4's read paragraph; rephrased as "An additional 4 nonstylized blue pairs sit at JS > 0.08 (…)." Observation stands on its own per CLAUDE.md § "State facts, not sources".

3. **Lens 9 — One takeaway / one figure per finding (Codex FAIL, Claude borderline).** Picked **fix (a)** per the orchestrator's recommendation — lower-effort, no new plotting required. Demoted the Phase 1 H4 (`#### Phase 1: the activation-predictor plateau survives`) into a short lead paragraph directly under `### Findings`, framed as "verify the existing line still holds; the new finding is the JS predictor below". The two Phase-1 CV-R² tables were collapsed inline into one prose paragraph. Then split the former combined Phase-2 H4 into TWO H4s:
   - `#### A full-response JS predictor scores ρ = 0.54 / CV R² = 0.24 on the full 16-persona panel` — anchored on `hero.png`, carries the within-stylized triangle, the point-biserial decomposition, and the `g_logprob` secondary-target aside.
   - `#### JS sits ~½ the predictive power of the four activation-distance metrics` — anchored on `metric_compare.png`, carries the activation-vs-JS comparison numbers and the different-uncertainty-quantification explanation.
   - `#### Drop the stylized personas, and the JS predictor loses out-of-sample CV-R² skill` (unchanged) — anchored on `by_panel_epoch.png`.

   The H4 count went from 3 → 3, but figure distribution went from 0 / 2 / 1 to 1 / 1 / 1, satisfying the one-takeaway-one-figure contract. Phase 1 (no figure, was always a verify-arm) now sits as a lead paragraph under `### Findings`, not as a top-level finding.

4. **C1 audit hit (Codex C1 — optional).** Left in place. The 16-persona ID lookup table inside `### What I ran` (with `A1..A5`, `B1..B5`, `C1`, `D1..D5` matched to plain-English names) is project-canonical from #406 and is immediately followed by the plain-English name on every reader-facing occurrence. The audit's `condition_labels` regex matched `C1` (the Qwen-template persona ID) but the lookup table satisfies the spirit of the plain-English-condition-names rule by binding the ID to "Standard Qwen template" on first sight. By-design audit hit; flagged in the `epm:interpretation v4` marker.

## Tightening choices

- The Phase-1 lead paragraph compresses the original H4's two CV-R² tables into single-sentence numerical summaries (gauss_kl / MMD / wass2 / cosine at L22/N=500/ep1 = 0.62/0.58/0.57/0.56; ep1 is 0.13-0.21 above ep2-5; ep2-5 row order shuffles inside subset-σ). Detail-seekers find the per-cell table in `probe_count_sweep_results.json`, named directly in the paragraph.
- The new Phase-2 split inserts one sentence at the end of the first H4 explaining the different-uncertainty-quantification choice for the next figure — telegraphing what the second H4 will show so the metric_compare figure isn't a non-sequitur.

## Files touched

- `tasks/interpreting/522/body.md` — set via `task.py set-body 522 --file /tmp/issue-522-body-v4.md`. No snapshot (round 2 reuses the round-1 `original-body.md`).

## Mechanical checks (post-revision)

- `scripts/verify_task_body.py --issue 522`: **PASS** (all 30 checks; v2 nested-design + 3 H4 children confirmed)
- `scripts/audit_clean_results_body_discipline.py --task 522`: FAIL on `condition_labels: 'C1'` only (by-design — see blocker 4); `interval_inline` now clean.
