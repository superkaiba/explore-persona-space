#!/usr/bin/env python
"""issue 2163 round 2 — population-restricted activity-matched partials (critique request 1).

Splits the Phase-5 partial-Spearman read (scripts/issue2163_ctxread.py, phase ``partials``)
by last-token activity population: the full complete-case pool (reproduction gate against the
committed ``predictor_partials.json``), the train-active features (``lasttoken_count > 0``),
and the never-active features (``lasttoken_count == 0``). Reuses the driver's exact
rank/residualize/partial convention VIA IMPORT (``_rank`` / ``_residualize`` / ``_partial_row``
/ ``_load_selection``), and mirrors the driver's stratified-permutation null (10 deciles of
the population's own match rank, per-draw max over the identical 25-column selection set)
so each restricted read gets its own band instead of an eyeballed threshold.

Also computes (request 2) the corpus-half A-stability among features with nonzero A in BOTH
halves, and (request 4) how many double-matched observed partials clear the committed
single-matched band.

Correctness gate: the full-pool leg must reproduce the committed proj_var partial
(-0.23940799338763108) and every other committed column to <= 1e-9 before the restricted
populations are trusted; the gate result is recorded in the artifact and asserted.

Output: ``eval_results/issue_2163/population_partials.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps must bind before any heavy import (#847)

import numpy as np  # noqa: E402

from issue2163_ctxread import (  # noqa: E402  (driver convention, reused via import)
    DICT_SIZE,
    MATCH_COV,
    MATCH_COV_2,
    SEED,
    _load_selection,
    _partial_row,
    _rank,
    _residualize,
    logger,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
COV_PATH_IN_REPO = "issue2163_ctxread/inputs/fullwidth_covariates_v2.npz"
HF_REVISION_PIN = "0d76405c5704798cda8e116d666c1a916e61a15a"  # the body's pinned data revision
COMMITTED_PROJ_VAR = -0.23940799338763108  # predictor_partials.json per_dv.logU_W proj_var
GATE_TOL = 1e-9


def _stratperm_band(
    dv_r: np.ndarray,
    resid_c: np.ndarray,
    norm_c: np.ndarray,
    m_rank: np.ndarray,
    match_idx: list[int],
    n_draws: int,
    rng: np.random.Generator,
    chunk: int = 100,
) -> float:
    """p97.5 of the per-draw max |partial| (driver phase_partials draw loop, mirrored).

    Permutes the ranked DV within 10 deciles of the match rank; each draw recomputes all
    partials against the SAME residualized-covariate stack and takes the max |partial|
    (selection-symmetric). Returns the 97.5% quantile of the per-draw maxima.
    """
    n = len(dv_r)
    idx = np.arange(n)
    order_ = np.lexsort((idx, m_rank))
    strata = np.empty(n, dtype=np.int64)
    strata[order_] = np.minimum((np.arange(n) * 10) // n, 9)
    stratum_rows = [np.where(strata == s)[0] for s in range(10)]
    draw_max = np.empty(n_draws, dtype=np.float64)
    t0 = time.time()
    for d0 in range(0, n_draws, chunk):
        d1 = min(d0 + chunk, n_draws)
        nb = d1 - d0
        perm = np.tile(idx, (nb, 1))
        for rows_s in stratum_rows:
            sub = np.argsort(rng.random((nb, len(rows_s))), axis=1)
            perm[:, rows_s] = rows_s[sub]
        dvp = dv_r[perm]  # (nb, n)
        rd = dvp - dvp.mean(axis=1, keepdims=True)
        mr = m_rank - m_rank.mean()
        beta = (rd @ mr) / max(float(mr @ mr), 1e-30)
        rd = rd - beta[:, None] * mr[None, :]
        nd = np.linalg.norm(rd, axis=1)
        num = rd @ resid_c.T  # (nb, n_cov)
        with np.errstate(divide="ignore", invalid="ignore"):
            part = num / (nd[:, None] * norm_c[None, :])
        part[:, np.asarray(match_idx)] = 0.0
        part[:, norm_c < 1e-9] = 0.0
        draw_max[d0:d1] = np.abs(part).max(axis=1)
        logger.info("[pop-partials] draws %d/%d %.0fs", d1, n_draws, time.time() - t0)
    return float(np.quantile(draw_max, 0.975))


def _population_block(
    dv: np.ndarray,
    cov_mat: np.ndarray,
    columns: list[str],
    mask: np.ndarray,
    definition: str,
    match_name: str,
    match2_name: str,
    n_draws: int,
    rng: np.random.Generator,
) -> dict:
    """All 25 observed (+ double-matched) partials and the permutation band on one population."""
    n = int(mask.sum())
    match_idx = [columns.index(match_name)]
    match2_idx = sorted({columns.index(match_name), columns.index(match2_name)})
    dv_r = _rank(dv[mask])
    cov_r = np.stack([_rank(cov_mat[k][mask]) for k in range(cov_mat.shape[0])])
    m_design = cov_r[match_idx].T
    m2_design = cov_r[match2_idx].T
    resid_c = np.stack([_residualize(cov_r[k], m_design) for k in range(cov_mat.shape[0])])
    norm_c = np.linalg.norm(resid_c, axis=1)
    obs, degen = _partial_row(dv_r, cov_r, resid_c, norm_c, match_idx, m_design)
    resid_c2 = np.stack([_residualize(cov_r[k], m2_design) for k in range(cov_mat.shape[0])])
    norm_c2 = np.linalg.norm(resid_c2, axis=1)
    obs2, degen2 = _partial_row(dv_r, cov_r, resid_c2, norm_c2, match2_idx, m2_design)
    band = _stratperm_band(dv_r, resid_c, norm_c, cov_r[match_idx[0]], match_idx, n_draws, rng)
    obs_abs = np.abs(obs)
    m_vals = cov_mat[columns.index(match_name)][mask]
    tie_frac = float(np.max(np.unique(m_vals, return_counts=True)[1]) / n)
    return {
        "definition": definition,
        "n": n,
        "match_tie_fraction": tie_frac,
        "observed_partials": {c: float(v) for c, v in zip(columns, obs)},
        "observed_partials_robust2": {c: float(v) for c, v in zip(columns, obs2)},
        "degenerate_partial": [c for c, g in zip(columns, degen) if g],
        "degenerate_partial_robust2": [c for c, g in zip(columns, degen2) if g],
        "max_abs_partial": float(obs_abs.max()),
        "argmax_column": columns[int(obs_abs.argmax())],
        "band_p97_5_of_max": band,
        "n_columns_outside_band": int((obs_abs > band).sum()),
    }


def main() -> int:
    """Compute per-population partials + bands; assert the full-pool reproduction gate."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "eval_results" / "issue_2163"),
    )
    ap.add_argument(
        "--stage-dir",
        default="/mnt/eps-data/thomasjiralerspong/issue2163_r2",
        help="off-root staging dir for the HF covariate panel + the census symlink",
    )
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--out", default=None, help="default: <results-dir>/population_partials.json")
    args = ap.parse_args()

    resd = Path(args.results_dir)
    stage = Path(args.stage_dir)
    cov_dir = stage / "cov"
    cov_dir.mkdir(parents=True, exist_ok=True)
    cov_file = cov_dir / "fullwidth_covariates_v2.npz"
    if not cov_file.exists():
        hub.stage_hub_file(
            HF_DATA_REPO,
            COV_PATH_IN_REPO,
            cov_file,
            repo_type="dataset",
            revision=HF_REVISION_PIN,
        )
    assembled = stage / "assembled"
    assembled.mkdir(parents=True, exist_ok=True)
    census_link = assembled / "census.npz"
    if not census_link.exists():
        census_link.symlink_to(resd / "census.npz")

    sel_args = SimpleNamespace(local_covariates=str(cov_dir), work=str(stage))
    cols, columns, dropped = _load_selection(sel_args)
    census = np.load(resd / "census.npz")
    assert np.array_equal(np.asarray(census["feat_ids"], dtype=np.int64), np.arange(DICT_SIZE)), (
        "census feature-id join broken"
    )
    lad_w = np.load(resd / "read_ladder__W.npz")
    u_w = np.asarray(lad_w["u"], dtype=np.float64)
    dv = np.where(u_w > 0, np.log10(np.clip(u_w, 1e-300, None)), np.nan)  # driver logU_W
    ltc = np.asarray(census["lasttoken_count"], dtype=np.float64)

    cov_mat = np.stack([cols[c] for c in columns])
    cov_finite = np.isfinite(cov_mat).all(axis=0)
    mask_cc = cov_finite & np.isfinite(dv)
    rng = np.random.default_rng(SEED + 52)  # fresh stream; documented in the artifact

    populations = {}
    for name, mask, definition in (
        ("full", mask_cc, "complete-case features (all selection covariates + logU_W finite)"),
        (
            "train_active",
            mask_cc & (ltc > 0),
            "complete-case AND lasttoken_count > 0 (fires at the last prompt token in train)",
        ),
        (
            "never_active",
            mask_cc & (ltc == 0),
            "complete-case AND lasttoken_count == 0 (never fires at the last prompt token; "
            "the matching covariate is one tie, so activity matching is inert here)",
        ),
    ):
        logger.info("[pop-partials] population=%s n=%d", name, int(mask.sum()))
        populations[name] = _population_block(
            dv, cov_mat, columns, mask, definition, MATCH_COV, MATCH_COV_2, args.n_draws, rng
        )

    committed = json.loads((resd / "predictor_partials.json").read_text())
    cblk = committed["per_dv"]["logU_W"]
    recomputed = populations["full"]["observed_partials"]
    diffs = {
        c: abs(recomputed[c] - cblk["observed_partials"][c])
        for c in columns
        if c not in populations["full"]["degenerate_partial"]
    }
    gate = {
        "committed_proj_var_partial": COMMITTED_PROJ_VAR,
        "recomputed_full_pool_proj_var": recomputed["proj_var"],
        "abs_diff_proj_var": abs(recomputed["proj_var"] - COMMITTED_PROJ_VAR),
        "max_abs_diff_all_informative_columns": max(diffs.values()),
        "committed_n_complete_case": cblk["n_complete_case"],
        "recomputed_n_complete_case": populations["full"]["n"],
        "tolerance": GATE_TOL,
    }
    gate["pass"] = bool(
        gate["abs_diff_proj_var"] <= GATE_TOL
        and gate["max_abs_diff_all_informative_columns"] <= GATE_TOL
        and gate["committed_n_complete_case"] == gate["recomputed_n_complete_case"]
    )
    assert gate["pass"], f"full-pool reproduction gate FAILED: {gate}"

    # Request 2: corpus-half A stability, active-in-both conditioning (tie-corrected scipy rho).
    from scipy.stats import spearmanr

    a_lm = np.asarray(lad_w["a_lmsys"], dtype=np.float64)
    a_wc = np.asarray(lad_w["a_wildchat"], dtype=np.float64)
    both = np.isfinite(a_lm) & np.isfinite(a_wc) & (a_lm != 0) & (a_wc != 0)
    rho_raw = float(spearmanr(a_lm[both], a_wc[both]).statistic)
    rho_abs = float(spearmanr(np.abs(a_lm[both]), np.abs(a_wc[both])).statistic)
    stability = {
        "committed_all_features": {
            **committed["stability"]["A_W_corpus_half_spearman"],
            "source": "predictor_partials.json stability.A_W_corpus_half_spearman "
            "(tie-dominated: structural zeros on never-active features)",
        },
        "recomputed_active_in_both_halves": {
            "rho": rho_raw,
            "rho_on_abs_values": rho_abs,
            "n": int(both.sum()),
            "definition": "spearmanr(a_lmsys, a_wildchat) restricted to features with "
            "nonzero A in BOTH corpus halves",
        },
    }

    # Request 4: double-matched observed partials vs the committed SINGLE-matched band.
    nulls = np.load(resd / "nulls" / "stratperm__logU_W.npz")
    obs2_c = np.asarray(nulls["observed_robust2"], dtype=np.float64)
    degen2_c = np.asarray(nulls["degenerate_robust2"], dtype=bool)
    cols_c = [str(c) for c in nulls["columns"]]
    band_single = float(cblk["null_band_p97_5_of_max"])
    clear = {
        c: float(v) for c, v, g in zip(cols_c, obs2_c, degen2_c) if not g and abs(v) > band_single
    }
    robust2 = {
        "band_single_matched": band_single,
        "n_informative": int((~degen2_c).sum()),
        "n_clearing_band": len(clear),
        "columns_clearing_band": dict(sorted(clear.items(), key=lambda kv: -abs(kv[1]))),
        "note": "double-matched (lasttoken_count + firing_freq_per_token) observed partials "
        "read against the SINGLE-matched 1,000-draw band; no robust2 draw matrix was committed",
    }

    out = Path(args.out) if args.out else resd / "population_partials.json"
    payload = {
        "meta": {
            **as_metadata_dict(git_provenance()),
            "numpy": np.__version__,
            "seed_stream": SEED + 52,
            "n_draws": args.n_draws,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "convention": "rank/residualize/partial + null scheme imported from "
        "scripts/issue2163_ctxread.py phase_partials; DV logU_W = log10(U_j) for U_j > 0; "
        "match covariate lasttoken_count; per-population band = p97.5 of per-draw max "
        "|partial| over the identical selection columns, DV permuted within 10 deciles of "
        "the population's own match rank",
        "selection_columns": columns,
        "dropped_columns": dropped,
        "reproduction_gate": gate,
        "populations": populations,
        "stability": stability,
        "robust2_vs_single_band": robust2,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("[pop-partials] wrote %s (gate pass=%s)", out, gate["pass"])
    print(json.dumps({"gate": gate, "ns": {k: v["n"] for k, v in populations.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
