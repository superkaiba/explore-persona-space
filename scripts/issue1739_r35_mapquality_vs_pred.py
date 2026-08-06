#!/usr/bin/env python3
"""#1739 Result 3.5: is behavior prediction bad exactly where the answer mapping is bad?

Joins, per (behavior, eval_rung, map_kind) cell of the R2FAIR round
(``eval_results/issue_1739/result2_fair/``), the map's reconstruction quality on
that eval distribution (``map_diagnostics.json``: per-rung R^2 + kNN retrieval,
taken at the layer the transfer row actually used) against the behavior
prediction quality (``all_arms_spearman.json`` transfer rows: rho of the
map-projection arm, the context arm, and the oracle/real-answer arm), then
correlates the two.

Difficulty confound handling: primary DVs are the oracle-recovery fraction
rho(map-proj)/rho(oracle) (undefined where the oracle CI includes 0) and the map
gain rho(map-proj) - rho(context); raw rho(map-proj) is secondary, and a partial
correlation of recon quality vs rho(map-proj) controlling for rho(oracle) is
reported alongside.

Stated deviation: every map_diagnostics key in R2FAIR is ``context_end|...`` --
the round scored the context arm only, so no prefix-based reconstruction exists
to join. The 'train' transfer rung has no matched eval-rung recon; it is joined
to the map's U-pool holdout recon and flagged as a proxy (primary correlation
pool excludes those cells; the all-cells pool is a sensitivity read).

Outputs (eval_results/issue_1739/result3_5_mapquality/):
  r35_tidy_table.json     one row per cell, with provenance
  r35_correlations.json   pooled / within-behavior / LOBO / partial correlations
  figure under figures/issue_1739/result3_5_mapquality/ (via --figure)
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FAIR_ROOT = REPO_ROOT / "eval_results" / "issue_1739" / "result2_fair"
OUT_ROOT = REPO_ROOT / "eval_results" / "issue_1739" / "result3_5_mapquality"
FIG_ROOT = REPO_ROOT / "figures" / "issue_1739" / "result3_5_mapquality"

BEHAVIORS = ("evil", "sycophancy", "hallucination")
# transfer eval_rung -> which recon block of map_diagnostics carries it
RUNG_TO_BLOCK = {
    "pvsynth": "recon_pvsynth",
    "wildchat_rung": "recon_wildchat_rung",
    "hhrt": "recon_ood",
    "toxicchat": "recon_ood",
    "aita": "recon_ood",
    "nqopen": "recon_ood",
    "simpleqa": "recon_ood",
}

MAP_QUALITY_READS = (
    "r2_at_layer",
    "r2_rung_mean",
    "knn_cos_acc1",
    "knn_cos_acc5",
    "knn_cos_mrr",
    "knn_euc_acc1",
)
DVS = ("oracle_frac", "map_gain", "rho6")


def _layer_entry(per_layer: list[dict], layer: int) -> dict:
    for pl in per_layer:
        if pl["layer_idx"] == layer:
            return pl
    raise KeyError(f"layer {layer} not in per_layer (have {[p['layer_idx'] for p in per_layer]})")


def _knn_fields(pl: dict) -> dict:
    out = {}
    for metric in ("euclidean", "cosine"):
        k = pl["knn"][metric]
        tag = "cos" if metric == "cosine" else "euc"
        out[f"knn_{tag}_acc1"] = k["acc_at_k"]["1"]
        out[f"knn_{tag}_acc5"] = k["acc_at_k"]["5"]
        out[f"knn_{tag}_mrr"] = k["mrr"]
        out[f"knn_{tag}_median_rank"] = k["median_rank"]
        out[f"knn_{tag}_chance1"] = k["chance_at_k"]["1"]
        out[f"knn_{tag}_chance5"] = k["chance_at_k"]["5"]
    return out


def build_table() -> list[dict]:
    rows: list[dict] = []
    for behavior in BEHAVIORS:
        bdir = FAIR_ROOT / behavior
        md = json.loads((bdir / "map_diagnostics.json").read_text())
        sp = json.loads((bdir / "all_arms_spearman.json").read_text())
        # map_kind -> diagnostics entry (keys look like context_end|add|<kind>|<pool>)
        diag_by_kind = {e["map_kind"]: e for e in md.values()}
        assert set(diag_by_kind) == {"linear", "mlp"}, list(md)

        trows = sp["transfer_rows"]
        # (arm, map_kind, rung) -> row; map-free arms live under map_kind 'linear'
        # and are joined as constants for both kinds.
        by_key = {(r["arm"], r["map_kind"], r["eval_rung"]): r for r in trows}

        def _mapfree(arm: str, rung: str) -> dict:
            return by_key[(arm, "linear", rung)]

        rungs = sorted(
            {r["eval_rung"] for r in trows if r["arm"] == "arm6_map_proj_e1"},
        )
        for map_kind in ("linear", "mlp"):
            diag = diag_by_kind[map_kind]
            for rung in rungs:
                a6 = by_key[("arm6_map_proj_e1", map_kind, rung)]
                a1 = _mapfree("arm1_ctx_e1", rung)
                a11 = _mapfree("arm11_oracle_proj", rung)
                layer = a6["layer"]

                if rung == "train":
                    # no matched eval-rung recon; U-pool holdout recon as proxy
                    pl = _layer_entry(diag["per_layer"], layer)
                    r2_at_layer = pl["r2_map"]
                    r2_vals = [p["r2_map"] for p in diag["per_layer"]]
                    r2_rung_mean = float(np.mean([v for v in r2_vals if math.isfinite(v)]))
                    n_recon = pl["knn"]["cosine"]["n"]
                    recon_dist = "u_pool_holdout_proxy"
                else:
                    block = diag[RUNG_TO_BLOCK[rung]]
                    rv = block["per_rung"][rung]
                    pl = _layer_entry(rv["per_layer"], layer)
                    r2_at_layer = pl["r2_eval_rung"]
                    r2_rung_mean = rv["r2_eval_rung_mean"]
                    n_recon = rv["n_rows"]
                    recon_dist = "matched_eval_rung"

                rho6, rho1, rho11 = a6["rho_frozen"], a1["rho_frozen"], a11["rho_frozen"]
                ci11 = a11["ci_frozen"]
                # oracle-recovery fraction only where the oracle itself predicts
                # (its CI excludes 0); otherwise the ratio is division by noise.
                oracle_defined = ci11[0] > 0
                row = {
                    "behavior": behavior,
                    "eval_rung": rung,
                    "map_kind": map_kind,
                    "layer_used": layer,
                    "layer_arm1": a1["layer"],
                    "layer_arm11": a11["layer"],
                    "recon_distribution": recon_dist,
                    "n_recon_rows": n_recon,
                    "r2_at_layer": r2_at_layer,
                    "r2_rung_mean": r2_rung_mean,
                    **_knn_fields(pl),
                    "rho6": rho6,
                    "ci6": a6["ci_frozen"],
                    "rho1": rho1,
                    "ci1": a1["ci_frozen"],
                    "rho11": rho11,
                    "ci11": ci11,
                    "n_eval": a6["n_eval"],
                    "map_gain": rho6 - rho1,
                    "oracle_frac": (rho6 / rho11) if oracle_defined else None,
                    "oracle_frac_defined": oracle_defined,
                }
                rows.append(row)
    return rows


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3:
        return {"n": int(len(x)), "pearson_r": None, "spearman_rho": None}
    pr, pp = stats.pearsonr(x, y)
    sr, sp_ = stats.spearmanr(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": float(pr),
        "pearson_nominal_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_nominal_p": float(sp_),
    }


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray, ranked: bool) -> dict:
    """Partial correlation of x vs y controlling z (residualize both on [1, z])."""
    if ranked:
        x = stats.rankdata(x)
        y = stats.rankdata(y)
        z = stats.rankdata(z)
    zm = np.column_stack([np.ones_like(z), z])
    rx = x - zm @ np.linalg.lstsq(zm, x, rcond=None)[0]
    ry = y - zm @ np.linalg.lstsq(zm, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return {"r": float(r), "nominal_p": float(p), "n": int(len(x))}


def _cells_xy(rows: list[dict], xkey: str, ykey: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    kept = [r for r in rows if r[xkey] is not None and r[ykey] is not None]
    x = np.array([r[xkey] for r in kept], dtype=float)
    y = np.array([r[ykey] for r in kept], dtype=float)
    return x, y, kept


def correlate(rows: list[dict]) -> dict:
    out: dict = {}
    pools = {
        "matched_recon_only": [r for r in rows if r["recon_distribution"] == "matched_eval_rung"],
        "all_cells_incl_train_proxy": rows,
    }
    for pool_name, pool in pools.items():
        pool_out: dict = {"n_cells": len(pool)}
        for xkey in MAP_QUALITY_READS:
            for ykey in DVS:
                x, y, kept = _cells_xy(pool, xkey, ykey)
                entry = {"pooled": _corr(x, y)}
                # within-behavior
                wb = {}
                for b in BEHAVIORS:
                    bx, by, _ = _cells_xy([r for r in kept if r["behavior"] == b], xkey, ykey)
                    wb[b] = _corr(bx, by)
                entry["within_behavior"] = wb
                # leave-one-behavior-out
                lobo = {}
                for b in BEHAVIORS:
                    lx, ly, _ = _cells_xy([r for r in kept if r["behavior"] != b], xkey, ykey)
                    lobo[f"drop_{b}"] = _corr(lx, ly)
                entry["leave_one_behavior_out"] = lobo
                pool_out[f"{xkey}__vs__{ykey}"] = entry
            # partial: x vs rho6 controlling rho11
            kept = [r for r in pool if r[xkey] is not None]
            x = np.array([r[xkey] for r in kept], dtype=float)
            y = np.array([r["rho6"] for r in kept], dtype=float)
            z = np.array([r["rho11"] for r in kept], dtype=float)
            pool_out[f"{xkey}__vs__rho6__partial_ctrl_rho11"] = {
                "pearson": _partial_corr(x, y, z, ranked=False),
                "spearman": _partial_corr(x, y, z, ranked=True),
            }
        out[pool_name] = pool_out
    out["sensitivity"] = _sensitivity(pools["matched_recon_only"])
    return out


def _sensitivity(matched: list[dict]) -> dict:
    """Leverage / confound checks on the matched-recon pool.

    (a) map_gain carries a headroom confound (gain is bounded by rho11 - rho1,
    which varies by rung): partial correlation of R^2 vs gain controlling
    headroom, raw and rank space. (b) single-cell leverage: drop the
    sycophancy/pvsynth/linear cell (worst recon on BOTH reads + the only large
    negative gain). (c) oracle_frac restricted to cells with a non-marginal
    oracle (|rho11| >= 0.15), where the ratio denominator is stable.
    """
    x = np.array([r["r2_at_layer"] for r in matched], dtype=float)
    g = np.array([r["map_gain"] for r in matched], dtype=float)
    head = np.array([r["rho11"] - r["rho1"] for r in matched], dtype=float)
    out: dict = {
        "map_gain_vs_r2_at_layer_partial_ctrl_headroom": {
            "pearson": _partial_corr(x, g, head, ranked=False),
            "spearman": _partial_corr(x, g, head, ranked=True),
        }
    }
    drop = [
        r
        for r in matched
        if not (
            r["behavior"] == "sycophancy"
            and r["eval_rung"] == "pvsynth"
            and r["map_kind"] == "linear"
        )
    ]
    dx, dg, _ = _cells_xy(drop, "r2_at_layer", "map_gain")
    out["map_gain_vs_r2_at_layer_drop_syco_pvsynth_linear"] = _corr(dx, dg)
    stable = [r for r in matched if r["oracle_frac"] is not None and abs(r["rho11"]) >= 0.15]
    for xk in ("r2_at_layer", "knn_cos_acc1"):
        sx, sy, _ = _cells_xy(stable, xk, "oracle_frac")
        out[f"oracle_frac_stable_denominator_vs_{xk}"] = _corr(sx, sy)
    return out


def make_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt  # noqa: E402

    from explore_persona_space.analysis.paper_plots import (  # noqa: E402
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    matched = [r for r in rows if r["recon_distribution"] == "matched_eval_rung"]
    colors = {"evil": "C0", "sycophancy": "C1", "hallucination": "C2"}
    markers = {"linear": "o", "mlp": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [
        (
            "r2_at_layer",
            "rho6",
            "Reconstruction R² (at used layer)",
            "Behavior prediction ρ (mapped-answer projection)",
        ),
        (
            "r2_at_layer",
            "oracle_frac",
            "Reconstruction R² (at used layer)",
            "Fraction of real-answer ρ recovered",
        ),
        (
            "knn_cos_acc1",
            "oracle_frac",
            "kNN retrieval acc@1 (cosine)",
            "Fraction of real-answer ρ recovered",
        ),
    ]
    for ax, (xk, yk, xl, yl) in zip(axes, panels):
        for r in matched:
            if r[xk] is None or r[yk] is None:
                continue
            ax.scatter(
                r[xk],
                r[yk],
                color=colors[r["behavior"]],
                marker=markers[r["map_kind"]],
                s=55,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        if yk == "oracle_frac":
            ax.axhline(1, color="grey", lw=0.6, ls="--")

    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="", label=b) for b, c in colors.items()
    ] + [
        plt.Line2D([], [], color="grey", marker=m, ls="", label=f"{k} map")
        for k, m in markers.items()
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle(
        "Map reconstruction quality vs behavior prediction (R2FAIR, context arm only; "
        "eval-distribution-matched cells)",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "r35_mapquality_vs_pred", dir=FIG_ROOT)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", action="store_true", help="also render the scatter figure")
    args = ap.parse_args()

    rows = build_table()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    table_path = OUT_ROOT / "r35_tidy_table.json"
    table_path.write_text(
        json.dumps(
            {
                "source": {
                    "map_diagnostics": [
                        str((FAIR_ROOT / b / "map_diagnostics.json").relative_to(REPO_ROOT))
                        for b in BEHAVIORS
                    ],
                    "transfer_rows": [
                        str((FAIR_ROOT / b / "all_arms_spearman.json").relative_to(REPO_ROOT))
                        for b in BEHAVIORS
                    ],
                },
                "deviation": "context_end only — R2FAIR carries no prefix-based recon to join",
                "train_rung_note": "train cells join the U-pool holdout recon as a proxy "
                "(recon_distribution=u_pool_holdout_proxy); matched_recon_only pool excludes them",
                "oracle_frac_guard": "oracle_frac reported only where arm11 CI lower bound > 0",
                "rows": rows,
            },
            indent=1,
        )
    )
    corr = correlate(rows)
    corr_path = OUT_ROOT / "r35_correlations.json"
    corr_path.write_text(
        json.dumps(
            {
                "note": "cells are NOT independent draws (shared behavior/map/readout); "
                "nominal p-values are anti-conservative — read within_behavior + "
                "leave_one_behavior_out for the clustered picture",
                "correlations": corr,
            },
            indent=1,
        )
    )
    print(f"wrote {table_path} ({len(rows)} rows) and {corr_path}")
    if args.figure:
        make_figure(rows)
        print(f"wrote figure under {FIG_ROOT}")


if __name__ == "__main__":
    main()
