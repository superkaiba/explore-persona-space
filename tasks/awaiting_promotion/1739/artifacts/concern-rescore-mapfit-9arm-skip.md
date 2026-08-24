severity: BLOCKER
summary: issue1739_rescore_ood.py builds CellData with no mapfit/text_emb/text_features, so 9 of its 16 arms — the whole map family AND the arm16 baseline the round's hypotheses are defined against — are skipped with a recorded reason instead of scored.

## Verdict: the v391 cross-round flag is CONFIRMED by direct code read (not a probe inference)

v391 was posted advisory ("NOT a directive, verify before acting"). It has now been
verified at source, on the committed code, and it is CORRECT. Raising it as a BLOCKER
because the production re-score leg would silently produce a result set that cannot
answer the round's own question.

## Exact mechanism (file:line, on the committed tree)

1. `scripts/issue1739_rescore_ood.py` :538-544 constructs
   `arms.CellData(z_ctx=..., z_ans=..., dv=..., rb=..., layers=...)`.
   It passes NO `mapfit`, NO `text_emb`, NO `text_features`.
   Confirmed independently: `grep -n "mapfit" scripts/issue1739_rescore_ood.py`
   returns ZERO hits — the token does not appear anywhere in the driver.
2. `CellData` (`experiments/issue_1739/arms.py` :342-361) declares
   `mapfit: MapFit | None = None`, `text_emb: ... = None`,
   `text_features: ... = None`, and its own docstring (:347-348) states:
   "Optional inputs gate arm availability (a missing input SKIPS the arm with a
   recorded reason, never a silent zero)."
3. `run_transfer_cell` (:1144-1152) forwards `mapfit=data.mapfit` into the combined
   CellData — so it faithfully forwards `None`. The forwarding is not the defect;
   the driver never supplies a map in the first place.
4. `run_cell_multi` :655 computes
   `need_mp = base.mapfit is not None and bool(mp_arms & set(want))` -> False,
   so :656 leaves `mp = None`, and the `if mp is not None:` block is not entered.
5. The `else` branch :712-722 then records `_skip(slug, "no mapfit")` for EXACTLY:
   `arm6_map_proj_e1`, `arm7_map_ridge_pred`, `arm8_map_ridge_true`,
   `arm9_pretrain_ft`, `arm10_stacked`, `arm13_shuffled_map`, `arm14_shuffled_pt`.
6. Separately, `arm15_text_only` and `arm16_surface_feat` require `text_emb` /
   `text_features`, which the driver also never passes.

Net: 9 of the declared 16 arms (`_ALL16_NAMES`, :81-101) do not run.
PRODUCED: arms 1, 2, 3, 4, 5, 11, 12. SKIPPED: 6, 7, 8, 9, 10, 13, 14, 15, 16.
This matches v391's probe result exactly, by an independent route.

## Why this is BLOCKER and not CONCERN

The round exists to test whether the context->answer MAP buys anything on evil OOD
rungs. Plan v14 states the decision rule twice:

- §-hypotheses H1: "Spearman rho(map-arm, DV) > rho(surface-arm16, DV) with 95%
  bootstrap CI not straddling zero".
- §-hypotheses H2: "AUROC(map-arm) > AUROC(surface-arm16) > 0.5 on ToxicChat
  and/or hh-rlhf".

Both sides of both comparisons are in the skipped set: every map arm (6/7/8) AND
the arm16 surface baseline, plus the shuffled-map null (13) that makes a positive
map read interpretable. A production leg run as-committed would emit arms 1/2/3/4/5/
11/12 on hhrt + toxicchat and a `skipped` map with reasons for the rest — technically
honest (no silent zeros: the fail-fast contract holds), but the round's headline
comparison would be structurally unanswerable, and the detection re-cut (item C)
would carry no map arm and no baseline.

## Required fix (both halves; either alone is insufficient)

(a) MAP: thread a fitted map into the driver's CellData. `scripts/issue1739_fits.py`
    already has both paths — it FITS via `_fit_map(args, z_u, zy_u)` (:373) and
    consumes a persisted sibling map via `_load_nl_map` (:1375-1389, `map_source =
    "loaded" | "fit"`), and persists weights at :828-831
    (`w` fp16 + `x_mu`/`x_sd`/`y_mu`). The re-score must load the persisted map for
    the matching (variant, u_label, layers, map_seed) and pass `mapfit=` — NOT refit
    a fresh map, which would change the estimator under the comparison.
(b) TEXT: pass `text_emb` / `text_features` for arms 15/16. NOTE the harder part
    v391 correctly identified: `run_transfer_cell` (:1144-1152) has NO eval-side
    parameter for text features — unlike `z_ev` / `za_ev` there is no `text_emb_ev`,
    so the eval-rung rows cannot currently receive them. Arms 15/16 therefore need a
    signature extension in `arms.py`, not just a driver change. Scope that
    deliberately: either extend `run_transfer_cell` with eval-side text inputs, or
    record arms 15/16 as an explicit stated scope omission on the OOD rungs — but
    then H1/H2 must be restated, because arm16 is their named baseline.

## Cross-links to already-open concerns (same root cause)

This is the third symptom of one gap: the round's drivers assume staged
`analysis_tensors/issue_1739/` inputs that nothing stages.
- `u1c-tensor-staging` — holdout_rung needs `analysis_tensors/issue_1739/{maps,
  r_b_e1,r_b_e2,...}`. Same maps this concern needs.
- `u1c-text-features` — arms 15/16 need `--text-emb`/`--text-features` npz from
  `issue1739_features.py`. Same inputs this concern needs.
Fix them together as one staging + threading unit, not three times.

## Gate to add regardless of how the fix lands

Unit 3's full per-phase smoke MUST assert the PRODUCED-ARM SET equals the plan's
expected roster and FAIL on any unexpected `skipped` entry — an arm-count assertion,
not just a non-empty-rows assertion. The unit-1a smoke already passed while writing
0 metric rows and empty preds JSONLs (recorded caveat, 14:46Z marker), so a
smoke that only checks rc=0 and non-emptiness would not catch this class. The same
assertion protects `issue1739_holdout_rung.py`, which reuses `_ALL16_NAMES` (:15-16,
:693) and will hit the identical gate the moment it runs without a staged map.

## Provenance

Verified 2026-08-03 ~18:10Z from the PA/orchestrator session by direct read of the
committed tree (`scripts/issue1739_rescore_ood.py` @ 8d1a6ad174,
`experiments/issue_1739/arms.py` on branch issue-1739). No files were edited and the
live worktree was not touched — `scripts/issue1739_rescore_ood.py` is owned by the
live evil-ood-spread-round session (one-implementer-per-file-set), so the fix belongs
to that session's implementer loop with a smoke to prove it. Original advisory:
epm:progress v391 (cross-round flag from the halted armfill recon, v389).
