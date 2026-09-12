# Brief: #1902 — is the K=5 R² ordering an answer-LENGTH artifact?

AUTO_REVIEW_DISABLED=1. Do not invoke any review or diagnostic loop on your own output. ONE turn, ANALYSIS-ONLY: no new training, no new generation, no new capture, no GPU. Everything runs in the FOREGROUND of your turn — never end your turn on a background wait, never launch `nohup`/`setsid`/background Bash and stop. Budget ~120 tool calls. Never read `/tmp/claude-*/tasks/*.output`.

## The question
At K=5 the own-map held-out R² ordering is base 0.675 > SFT 0.634 > DPO 0.606 > RLVR 0.599. But base and SFT have nearly identical ABSOLUTE residual (mean SS_res 228.1 vs 227.5); base's R² advantage comes from a larger denominator (mean SS_tot 701.3 vs 621.0), and base answers are ~5x shorter (median 60 tokens vs SFT 101, DPO 345, RLVR 312). Answer length correlates with vector norm and with both variance components (Spearman rho(len, ||w_bar||) = -0.33 to -0.56 within stage). An earlier K=1 length-matched read (different protocol: `u_mean` context, semantic-group folds) REVERSED the ordering: matched, DPO 0.422 and RLVR 0.419 sat above base 0.381.

**Decide: does the K=5 diagonal R² ordering survive a length control, on the paper's own protocol (`u_last` context summary, six IID random folds seed 190231, layer 31)?**

## Inputs (all on disk, nothing to download)
- Worktree `/mnt/eps-data/thomasjiralerspong/wt-1902-k5`, branch `issue-1902-k5` (HEAD b0a756ab456…). Work and commit THERE. Repo root `/home/thomasjiralerspong/explore-persona-space` is read-only for you.
- K=5 mean targets: `eval_results/issue_1902/k5_targets/targets/{BB,SS,DD,RR}_L31.npz` (keys `row_ids`, `w_bar`, `spread`, `k_eff`, `n_flagged`; 16,391 rows, d=4,096).
- Context vectors + estimator + folds: `scripts/issue1902_k5_fits.py` (see its `grid` subcommand) and the committed pipeline it imports — `issue1902_lasttoken_transfer.load_fold_of` / `SharedPrimalRidge`, `issue1902_lasttoken_comparison` store loaders. IMPORT these, never re-implement a ridge.
- Answer lengths: `data/issue_1902/k5_stage/issue1902_stage_map/raw_completions/gen/single/<ckpt>_k5_seed45.shard*.jsonl`, field `n_tokens`, joined by `id`. Use the K=5 MEAN length per (stage, context) over all five draws (seeds 42 + 45-48; seed 42 files are `<ckpt>.shard*.jsonl`), not a single draw. Never print or log rollout TEXT (LMSYS is unscreened); ids, counts, token statistics only.
- Precedent matching code to REUSE: `scripts/issue1902_followup_9ater.py` — `equalize_bins` (quantile-bin marginal matching), the `BAND_LADDER` band selection, `paired_same_rows`. Read it before writing anything.

## Hard constraint: well-posedness (#1701)
d = 4,096 and the six folds train on 5/6 of the retained rows, so any matched subset needs **> ~4,915 rows per stage** to keep `n_train > d`. The precedent's p10 band (46-254 tokens) retains 4,526 → n_train ≈ 3,772 < d, which is estimator-degenerate. Therefore:
- **Primary read**: marginal length-matched at the WIDEST band that keeps every stage's retained n above 4,915 (the p5 rung, 16-372.5 tokens, had per-stage in-band counts B 13,439 / D 7,725 / R 8,237 / S 11,119; widen further if needed). State the realized band, per-stage retained n, and `n_train` vs d BEFORE fitting.
- **Precedent-comparable read**: the same p10 band as the K=1 analysis, so the two are comparable. Report it, and label every number from it as an under-determined regularization-limit fit (n_train < d), interpretable only as a relative ordering at equal n across stages, never as an absolute R².
- If a design you try cannot clear the floor, say so and skip it rather than shipping a degenerate headline.

## What to compute
For each design, on the four diagonal cells (B/B, S/S, D/D, R/R), with the SAME folds and estimator as the committed grid:
1. Held-out pooled R², plus mean SS_res and mean SS_tot separately (the decomposition is the point — an ordering change driven purely by the denominator must be visible).
2. Median cosine(prediction, target) as the scale-free companion.
3. Per-stage retained n, realized length band, and the matched length distribution (median + IQR per stage, to show matching worked).
4. Row-bootstrap 95% CIs on each stage's R² and on the base-minus-DPO and base-minus-SFT differences (paired on the matched rows where the design pairs them). 1,000 draws.

Also run, as a companion that uses ALL 16,391 rows and needs no subsetting:
5. **Length-stratified residual read**: bin contexts by that stage's own mean answer length into 5 quantile bins, and report mean SS_res and mean SS_tot per (stage, bin) from the ALREADY-FITTED full-data percell arrays (`eval_results/issue_1902/k5_targets/percell/k5grid_{BB,SS,DD,RR}_L31.npz`, keys `ss_res`, `ss_tot`, `row_ids`). No refitting, so no well-posedness issue. This shows whether base's larger SS_tot is concentrated in its short answers.

## Deliverables
- `eval_results/issue_1902/k5_targets/length_matched/summary.json` — every number above, plus a `metadata` block (script, commit, timestamp, band ladder, seeds, n_train vs d per design) and a plain-language `verdict` string of at most three sentences saying whether the ordering survives.
- One figure, `figures/issue_1902/section43/c1_k5_length_control.{pdf,png}` + data JSON sidecar: per stage, R² unmatched vs matched (primary design) with CIs, and the length-stratified SS_res / SS_tot panel. Follow `docs/paper_context_answer_map/plotting_style.md` and import `src/explore_persona_space/analysis/c2a_plot_style.py`. No caption text, annotations, or arrows drawn on the canvas; axes, ticks, legend, panel titles only. **Read the rendered PNG yourself and confirm non-empty axes, plotted series, and sane ranges before you commit it.**
- Land it: `ruff check` + `ruff format` clean on any new script; run the no-flags `uv run python scripts/workflow_lint.py` and the mapped tests via `scripts/select_step9c_tests.py --map-files <a newline-delimited path-LIST file>` (that flag takes a file listing paths, not a source file). A non-WARN red line naming YOUR file blocks the push; pre-existing red elsewhere does not. Commit by explicit path with `git -C /mnt/eps-data/thomasjiralerspong/wt-1902-k5 commit -F <msgfile> -- <paths>` (never a bare `git commit`, never `git add -A`), then push. Force-add any convention-ignored `.npz` you intend to commit and verify with `git ls-files --others --ignored --exclude-standard` that nothing you meant to commit was silently skipped. End commit messages with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Final report (your last message, then end the turn)
The verdict in one sentence; the primary-design table (stage, retained n, n_train vs d, R², SS_res, SS_tot, cosine, CI); the precedent-band table with its degeneracy label; the length-stratified panel's reading; the full 40-char commit sha from `git rev-parse HEAD`; and any assumption you took.
