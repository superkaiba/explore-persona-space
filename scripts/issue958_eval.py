#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #958 statistical batteries — pure re-reductions of fit_maps outputs.

Consumes ``eval_results/issue_958/percell/*.npz`` (per-row skill, per-unit
SSE, shuffled-pairing draws, residual trait projections) + the headline JSONs
and produces:

- ``decision_stats.json`` — Δ_k (own − stale) with paired-by-conversation
  bootstrap 95% CIs (997 draws seed 0 — Source #922) vs the half-sample twin
  floor; the registered recalibrated decomposition (moment vs residual
  map-change components, each with the same paired CIs); F(1→k) vs 0 /
  copy-previous / prefix; the H3 prefix/context ratio; H4 length-binned reads.
- ``drift_read.json`` — actual + stale-residual trait projections vs turn
  index against the 100-direction norm-matched random band (per-draw
  same-selection: the identical statistic is computed per direction).
- ``null_matrices/<cell>.npz`` — the persisted per-draw × per-row shuffled
  band matrices (selection-symmetric re-reads stay recomputable post-hoc).

Every battery is a batched index-gather over the persisted per-unit arrays
(no refits, no serial per-draw pool re-reduction). Row coverage: all
main-panel cells are asserted to share the IDENTICAL test conversation set
(paired contrasts read identical rows by construction).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue958_common as C  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_eval")


def _readout_rows(n_rows: int) -> list[int]:
    """Frozen 6-block read-out rows, clamped for the stub-dims smoke."""
    rows = [C.block_to_row(b) for b in C.READOUT_BLOCKS]
    return [min(r, n_rows - 1) for r in rows] if any(r >= n_rows for r in rows) else rows


def _boot_idx(n_t: int, draws: int, seed: int) -> np.ndarray:
    """(draws, n_t) with-replacement conversation resample (paired across arms)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_t, size=(draws, n_t))


def _boot_skill(sse: np.ndarray, null: np.ndarray, rows: list[int], idx: np.ndarray) -> np.ndarray:
    """(draws,) read-out-mean skill under each conversation resample.

    Headline aggregation = mean over frozen rows of per-row skill (never
    1 − Σ/Σ pooled across rows).
    """
    draws = []
    for r in rows:
        s = sse[r][idx].sum(1)  # (n_draws,)
        n = null[r][idx].sum(1)
        draws.append(1.0 - s / np.clip(n, 1e-30, None))
    return np.stack(draws).mean(0)


def _point_skill(sse: np.ndarray, null: np.ndarray, rows: list[int]) -> float:
    return float(np.mean([1.0 - sse[r].sum() / max(null[r].sum(), 1e-30) for r in rows]))


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def main() -> int:  # noqa: C901 — the battery enumeration IS the plan §6 spec
    ap = argparse.ArgumentParser(description="Issue #958 bootstrap / null-band batteries.")
    ap.add_argument("--out", type=Path, default=Path("eval_results/issue_958"))
    args = ap.parse_args()
    t0 = time.time()
    percell_dir = args.out / "percell"
    cells = {
        p.stem: dict(np.load(p, allow_pickle=False)) for p in sorted(percell_dir.glob("*.npz"))
    }
    assert cells, f"no percell npz under {percell_dir}"
    n_rows = next(iter(cells.values()))["skill"].shape[0]
    rows = _readout_rows(n_rows)

    # row-coverage identity across paired main-panel arms (plan §6)
    main_cells = [
        cid for cid, c in cells.items() if not cid.startswith(("long_", "panelxfer", "onpol"))
    ]
    ref_idx = cells[main_cells[0]]["test_idx"]
    for cid in main_cells:
        assert np.array_equal(cells[cid]["test_idx"], ref_idx), (
            f"ROW-COVERAGE FAIL: {cid} test set differs from {main_cells[0]}"
        )
    n_t = len(ref_idx)
    idx = _boot_idx(n_t, C.BOOTSTRAP_DRAWS, C.BOOTSTRAP_SEED)

    def skill_draws(cid: str, sse_key: str = "sse_unit") -> tuple[float, np.ndarray]:
        c = cells[cid]
        return (
            float(np.mean([c["skill"][r] for r in rows]))
            if sse_key == "sse_unit"
            else float(np.mean([c["recal_skill"][r] for r in rows])),
            _boot_skill(c[sse_key], c["null_sse_unit"], rows, idx),
        )

    # ── H1: Δ_k vs twin floor + recalibrated decomposition ───────────────────
    h1: dict[str, dict] = {}
    for k in range(2, C.K_MAIN + 1):
        own_p, own_b = skill_draws(f"xfer_{k}to{k}_A")
        stale_p, stale_b = skill_draws(f"xfer_1to{k}_A")
        twin_p, twin_b = skill_draws(f"own_k{k}_B")
        rec_p, rec_b = skill_draws(f"recal_1to{k}_A", sse_key="recal_sse_unit")
        delta = own_b - stale_b
        twin_floor = own_b - twin_b
        moment_comp = rec_b - stale_b  # recalibration recovers this much
        resid_comp = own_b - rec_b  # residual map change after moment recal
        h1[f"k{k}"] = {
            "own_A": own_p,
            "stale_1toK": stale_p,
            "own_B": twin_p,
            "recalibrated_1toK": rec_p,
            "delta": own_p - stale_p,
            "delta_ci95": _ci(delta),
            "twin_floor": own_p - twin_p,
            "twin_floor_ci95": _ci(twin_floor),
            "delta_minus_twin_ci95": _ci(delta - twin_floor),
            "moment_component": rec_p - stale_p,
            "moment_component_ci95": _ci(moment_comp),
            "residual_map_change": own_p - rec_p,
            "residual_map_change_ci95": _ci(resid_comp),
            "residual_minus_twin_ci95": _ci((own_b - rec_b) - twin_floor),
            # band-vs-ceiling inputs (selection-symmetric-nulls rule): a skill
            # DIFFERENCE is ceilinged by max attainable own skill − realized
            # comparison arm; both reported for the analyzer's power read.
            "difference_ceiling_note": {
                "own_A_realized": own_p,
                "hard_skill_bound": 1.0,
                "difference_ceiling": 1.0 - stale_p,
            },
        }

    # ── H2: forecasts vs nulls ────────────────────────────────────────────────
    h2: dict[str, dict] = {}
    for j in range(1, C.K_MAIN + 1):
        for k in range(j + 1, C.K_MAIN + 1):
            f_p, f_b = skill_draws(f"fcast_{j}to{k}")
            entry = {"skill": f_p, "skill_ci95": _ci(f_b)}
            if j == 1:
                cp_p, cp_b = skill_draws(f"copyprev_k{k}")
                pr_p, pr_b = skill_draws(f"pre_k{k}_full")
                entry["vs_copyprev_ci95"] = _ci(f_b - cp_b)
                entry["vs_prefix_ci95"] = _ci(f_b - pr_b)
                entry["copyprev_skill"] = cp_p
                entry["prefix_skill"] = pr_p
            h2[f"{j}->{k}"] = entry

    # ── H3: prefix/context skill ratio vs k ──────────────────────────────────
    h3: dict[str, dict] = {}
    for k in range(2, C.K_MAIN + 1):
        pre_p, pre_b = skill_draws(f"pre_k{k}_full")
        ctx_p, ctx_b = skill_draws(f"own_k{k}_full")
        ratio_b = pre_b / np.clip(ctx_b, 1e-9, None)
        h3[f"k{k}"] = {
            "prefix_skill": pre_p,
            "context_skill": ctx_p,
            "ratio": pre_p / max(ctx_p, 1e-9),
            "ratio_ci95": _ci(ratio_b),
        }

    # ── shuffled-pairing bands (persist per-draw × per-row matrices) ─────────
    nm_dir = args.out / "null_matrices"
    nm_dir.mkdir(parents=True, exist_ok=True)
    bands: dict[str, dict] = {}
    for cid, c in cells.items():
        draws = c["shuffle_draws"]  # (100, n_rows)
        np.savez(nm_dir / f"{cid}.npz", shuffle_draws=draws)
        mean6 = draws[:, rows].mean(1)
        bands[cid] = {
            "readout_mean_p975": float(np.quantile(mean6, 0.975)),
            "readout_mean_p025": float(np.quantile(mean6, 0.025)),
            "max_over_rows_p975": float(np.quantile(draws.max(1), 0.975)),
            "observed_readout_mean": float(np.mean([c["skill"][r] for r in rows])),
        }

    # ── H4: within-turn skill vs context length (exploratory) ────────────────
    h4: dict[str, dict] = {}
    tok_path = args.out / "unit_tokens.npz"
    if tok_path.exists():
        toks = dict(np.load(tok_path))
        for k in range(1, C.K_MAIN + 1):
            cid = f"xfer_{k}to{k}_A"
            c = cells[cid]
            ctx_len = (toks[f"main_k{k}_prefix_tokens"] + toks[f"main_k{k}_query_tokens"])[
                c["test_idx"]
            ]
            per_unit = 1.0 - np.stack(
                [c["sse_unit"][r] / np.clip(c["null_sse_unit"][r], 1e-30, None) for r in rows]
            ).mean(0)
            rho = _spearman(ctx_len.astype(np.float64), per_unit)
            stale = cells.get(f"xfer_1to{k}_A")
            entry = {"n": len(ctx_len), "spearman_skill_vs_ctxlen": rho}
            if stale is not None and k >= 2:
                stale_unit = 1.0 - np.stack(
                    [
                        stale["sse_unit"][r] / np.clip(stale["null_sse_unit"][r], 1e-30, None)
                        for r in rows
                    ]
                ).mean(0)
                entry["spearman_deficit_vs_ctxlen"] = _spearman(
                    ctx_len.astype(np.float64), per_unit - stale_unit
                )
            h4[f"k{k}"] = entry

    C.write_json_atomic(
        args.out / "decision_stats.json",
        {
            "bootstrap": {
                "draws": C.BOOTSTRAP_DRAWS,
                "seed": C.BOOTSTRAP_SEED,
                "paired_by": "conversation",
            },
            "readout_rows": rows,
            "h1_stationarity": h1,
            "h2_forecasts": h2,
            "h3_prefix_ratio": h3,
            "h4_length": h4,
            "shuffle_bands": bands,
            "transfer_standardization_policy": C.TRANSFER_STANDARDIZATION_POLICY,
            "metadata": C.reproducibility_metadata({"script": "issue958_eval"}),
        },
    )

    # ── drift read (actual + stale-residual projections vs randdir band) ─────
    drift: dict[str, dict] = {}
    proj_path = args.out / "drift_actual_projections.npz"
    actual = dict(np.load(proj_path)) if proj_path.exists() else {}
    n_traits = len(C.TRAITS)
    for ti, trait in enumerate(C.TRAITS):
        # actual per-turn mean projection + within-conversation turn slope
        per_turn = {}
        slopes_obs, slopes_band = None, None
        mats = []
        for k in range(1, C.K_MAIN + 1):
            key = f"main_k{k}_{trait}"
            if key not in actual:
                continue
            m = actual[key]  # (n_t, n_traits + 100)
            per_turn[f"k{k}"] = float(m[:, ti].mean())
            mats.append(m)
        if len(mats) == C.K_MAIN:
            A = np.stack(mats)  # (K, n_t, n_dirs)
            ks = np.arange(1, C.K_MAIN + 1, dtype=np.float64)
            kc = ks - ks.mean()
            slopes = (kc[:, None, None] * A).sum(0) / (kc * kc).sum()  # (n_t, n_dirs)
            slopes_obs = float(slopes[:, ti].mean())
            rand = slopes[:, n_traits:].mean(0)  # (100,) per-direction mean slope
            slopes_band = _ci(rand)
            boot = slopes[:, ti][idx % slopes.shape[0]].mean(1)
            drift_slope_ci = _ci(boot)
        else:
            drift_slope_ci = None
        # stale-map residual projections per k (from the xfer_1to{k}_A cells)
        resid = {}
        for k in range(2, C.K_MAIN + 1):
            c = cells.get(f"xfer_1to{k}_A")
            key = f"proj_resid_{trait}"
            if c is not None and key in c:
                m = c[key]  # (n_t, n_dirs)
                resid[f"k{k}"] = {
                    "mean": float(m[:, ti].mean()),
                    "randdir_band_ci95": _ci(m[:, n_traits:].mean(0)),
                }
        drift[trait] = {
            "actual_mean_projection_per_turn": per_turn,
            "within_conv_turn_slope": slopes_obs,
            "turn_slope_boot_ci95": drift_slope_ci,
            "turn_slope_randdir_band_ci95": slopes_band,
            "stale_residual_projection": resid,
        }
    C.write_json_atomic(
        args.out / "drift_read.json",
        {
            "traits": list(C.TRAITS),
            "trait_layers": C.PRIMARY_LSTAR,
            "n_random_dirs": C.RANDDIR_DRAWS,
            "drift": drift,
            "note": "activation-level SECONDARY read (plan §6): systematic residual along "
            "persona directions; behavioral drift is not claimed from it",
            "metadata": C.reproducibility_metadata({"script": "issue958_eval"}),
        },
    )
    logger.info(
        "DONE eval batteries in %.1fs (%d cells, %d bootstrap draws)",
        time.time() - t0,
        len(cells),
        C.BOOTSTRAP_DRAWS,
    )
    return 0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank Pearson (no scipy dependency at eval time)."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


if __name__ == "__main__":
    sys.exit(main())
