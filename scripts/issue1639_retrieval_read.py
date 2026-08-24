"""#1639 follow-up: retrieval reads (acc@k) for specialized vs universal context->answer maps.

The #1639 clean-result is R^2-only; the paper (Results 2 Plot 2) also promises
acc@1. This script scores the SAME held-out fold predictions of the #1639
lattice rungs as retrieval, per model (Qwen2.5-7B base + instruct) at layer 19
on the scene-aggregated cells (300 points/persona/model, 4 personas):

- M2 "specialized": per-persona maps (the committed within-cells),
- M1 "shared": one map on per-persona train-fold-centered data,
- M0 "pooled": one map with global train-fold centering.

Fits reuse the exact parent recipe (GCV Gram ridge, dof cap 0.9, lambda grid
logspace(-2,4,13), K=5 scenario-grouped folds seed 0, one shared partition) via
``issue1310_xpersona_similarity.load_persona_arrays`` + the fit825 fold cache;
the fold loops mirror ``issue1310_xpersona_similarity_v2._pooled_fold_preds``
and ``issue1310_xpersona_similarity.transfer_cell`` with ONE addition —
``return_lam=True`` threading for the selected-lambda diagnostics — and are
equality-gated against the committed per-persona foldmean R^2 of BOTH parents
(``cells_agg_*`` for M2, ``v2/decomposition_*`` for M0/M1) before any retrieval
read. A gate miss STOPS the run (mismatched store).

Retrieval conventions (#2202 / #1901): PRIMARY = whitened cosine + CSLS —
whitening stats fit on the TRAIN side only (per fold: pooled train-fold answer
mean mu_A + Cholesky L of the shrunk train-answer covariance, lam=0.1, the
``null_battery.shrunk_cholesky_from_cov`` recipe); z = L^-1 (x - mu_A) for both
predictions and pool; CSLS = ``issue1901_metric_battery.csls_scores`` on the
whitened cosine-similarity matrix, K=10, retrieval distance = -score.
Companions: plain euclidean + plain cosine (``mapping_baselines`` conventions,
reconciled against ``knn_retrieval`` on one fold) and whitened cosine without
CSLS. Pools: per fold, PRIMARY matched-target read = the pooled held-out
answers of ALL personas (~240 rows; specialized and universal predictions
scored against the SAME pool), plus each persona's own-pool read (~60 rows).
Chance = k/n_pool, stated per read (mean over rows of k/n_pool of their fold).

CLI (run per model to keep each invocation minutes-scale; figure step last):
  uv run python scripts/issue1639_retrieval_read.py --models base
  uv run python scripts/issue1639_retrieval_read.py --models instruct
  uv run python scripts/issue1639_retrieval_read.py --figure-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps bind before torch/numpy import (#847)

import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue1310_xpersona_similarity as v1  # noqa: E402
from issue1901_metric_battery import K_CSLS, csls_scores  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    _pairwise_dist,
    knn_retrieval,
)
from explore_persona_space.analysis.null_battery import (  # noqa: E402
    PRIMARY_LAMBDA,
    shrunk_cholesky_from_cov,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

SCRIPT = "scripts/issue1639_retrieval_read.py"
REPO_ROOT = Path(__file__).resolve().parents[1]

LAYER = v1.HEADLINE_LAYER  # 19
PERSONAS = v1.PERSONAS  # Wren, HELIOS, Dana, Vex
N_FOLDS = v1.N_FOLDS  # 5
DOF_CAP = v1.DOF_CAP  # 0.9 (uncapped GCV is degenerate on this n<p store)
KS = (1, 5, 10)
GATE_TOL = 1e-6  # the v2 equality-gate tolerance (bit-exact expected, same VM)
RUNGS = ("M2", "M1", "M0")
METRICS = ("whiten_csls", "whiten_cos", "cosine", "euclidean")

DEFAULT_STORE_ROOT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1639_retrieval/hf_dl/"
    "issue1310_char_map/analysis_tensors/store_onpolicy"
)
COMMITTED_CELLS = REPO_ROOT / "eval_results" / "issue_1310" / "onpolicy_aggregated"
COMMITTED_DECOMP = REPO_ROOT / "eval_results" / "issue_1310" / "xpersona_similarity" / "v2"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_1639" / "retrieval_read"
FIG_PATH = REPO_ROOT / "figures" / "issue_1639" / "retrieval_read.png"

HF_INPUT = {
    "repo": "superkaiba1/explore-persona-space-data",
    "revision": "b24279a1f9ca2994d96aef49246b680f6352db95",
    "prefix": "issue1310_char_map/analysis_tensors/store_onpolicy/",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=str, default="base,instruct")
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=v1.FIT_SEED)  # 0
    ap.add_argument("--force", action="store_true", help="recompute even if the JSON exists")
    ap.add_argument(
        "--figure-only",
        action="store_true",
        help="skip compute; render the figure from the per-model JSONs in --out-dir",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Fits: lambda-threaded twins of the v1/v2 fold loops (equality-gated below).
# ---------------------------------------------------------------------------
def m2_fold_preds(arrays: dict, layer: int) -> dict:
    """Per-persona within-cell held-out preds + foldmean R^2 + selected lambdas.

    Mirrors ``v1.transfer_cell(arrays[p], arrays[p], layer)`` exactly (same
    dtype flow, same fold-test-mean R^2 accumulation) with ``return_lam=True``.
    """
    out: dict = {"preds": {}, "r2": {}, "lams": {}}
    for p in PERSONAS:
        xsl = arrays[p]["X"][:, layer, :]
        ysl = arrays[p]["Y"][:, layer, :]
        folds = arrays[p]["folds"]
        preds = np.zeros_like(ysl, dtype=np.float64)
        ss_res = ss_tot = 0.0
        lams = []
        for k in range(N_FOLDS):
            tr = folds != k
            te = folds == k
            assert te.sum() > 0 and tr.sum() >= 3, (p, k, int(tr.sum()), int(te.sum()))
            cache = fit825._prep_fold(xsl[tr], xsl[te])
            pred, lam = fit825._ridge_predict_cached(cache, ysl[tr], return_lam=True)
            preds[te] = pred
            lams.append(float(lam))
            true = ysl[te].astype(np.float64)
            mu = true.mean(0)
            ss_res += float(np.sum((true - pred) ** 2))
            ss_tot += float(np.sum((true - mu) ** 2))
        out["preds"][p] = preds
        out["r2"][p] = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
        out["lams"][p] = lams
    return out


def pooled_fold_preds(arrays: dict, layer: int, *, centering: str) -> dict:
    """Shared-map held-out per-persona preds + foldmean R^2 + selected lambdas.

    Mirrors ``issue1310_xpersona_similarity_v2._pooled_fold_preds`` (no y_perm)
    exactly — same centering conventions and dtype flow — with the selected
    lambda captured per fold. centering='global' is M0, 'per_persona' is M1.
    """
    assert centering in ("global", "per_persona")
    preds = {p: np.zeros_like(arrays[p]["Y"][:, layer, :], dtype=np.float64) for p in PERSONAS}
    lams = []
    for k in range(N_FOLDS):
        tr_blocks_x, tr_blocks_y, te_blocks_x = [], [], []
        te_idx = {}
        ymu_p = {}
        for p in PERSONAS:
            tr = arrays[p]["folds"] != k
            te = arrays[p]["folds"] == k
            xp = arrays[p]["X"][:, layer, :].astype(np.float64)
            yp = arrays[p]["Y"][:, layer, :].astype(np.float64)
            if centering == "per_persona":
                xmu = xp[tr].mean(0)
                ymu_p[p] = yp[tr].mean(0)
                tr_blocks_x.append(xp[tr] - xmu)
                tr_blocks_y.append(yp[tr] - ymu_p[p])
                te_blocks_x.append(xp[te] - xmu)
            else:
                tr_blocks_x.append(xp[tr])
                tr_blocks_y.append(yp[tr])
                te_blocks_x.append(xp[te])
            te_idx[p] = np.flatnonzero(te)
        tr_x = np.concatenate(tr_blocks_x, axis=0).astype(np.float32)
        tr_y = np.concatenate(tr_blocks_y, axis=0).astype(np.float32)
        te_x = np.concatenate(te_blocks_x, axis=0).astype(np.float32)
        cache = fit825._prep_fold(tr_x, te_x)
        pred_all, lam = fit825._ridge_predict_cached(cache, tr_y, return_lam=True)
        lams.append(float(lam))
        off = 0
        for p in PERSONAS:
            m = len(te_idx[p])
            block = pred_all[off : off + m]
            if centering == "per_persona":
                block = block + ymu_p[p]
            preds[p][te_idx[p]] = block
            off += m
    out = {"preds": preds, "r2": {}, "lams": lams}
    for p in PERSONAS:
        ss_res = ss_tot = 0.0
        yl = arrays[p]["Y"][:, layer, :].astype(np.float64)
        for k in range(N_FOLDS):
            te = arrays[p]["folds"] == k
            true = yl[te]
            mu = true.mean(0)
            ss_res += float(np.sum((true - preds[p][te]) ** 2))
            ss_tot += float(np.sum((true - mu) ** 2))
        out["r2"][p] = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return out


def equality_gate(model_kind: str, r2: dict[str, dict[str, float]]) -> dict:
    """Gate every rung's per-persona foldmean R^2 against the committed values.

    M2 vs ``cells_agg_<model>_<persona>.json`` (the #1639 within-cells); M0/M1
    vs ``v2/decomposition_<model>.json``. Raises on any |delta| > GATE_TOL —
    a mismatched store must never produce new reads (dispatch-note duty 1).
    """
    gate: dict = {"tolerance": GATE_TOL, "per_rung": {}, "worst_abs_delta": 0.0}
    decomp = json.loads((COMMITTED_DECOMP / f"decomposition_{model_kind}.json").read_text())
    for rung in RUNGS:
        per = {}
        for p in PERSONAS:
            if rung == "M2":
                cp = COMMITTED_CELLS / f"cells_agg_{model_kind}_{p}.json"
                committed = float(json.loads(cp.read_text())["r2_per_layer_obs"][LAYER])
            else:
                committed = float(decomp["per_persona"][p][f"r2_{rung}_foldmean"])
            d = abs(r2[rung][p] - committed)
            per[p] = {"mine": r2[rung][p], "committed": committed, "abs_delta": d}
            gate["worst_abs_delta"] = max(gate["worst_abs_delta"], d)
        gate["per_rung"][rung] = per
    gate["passed"] = gate["worst_abs_delta"] <= GATE_TOL
    if not gate["passed"]:
        raise RuntimeError(
            f"equality gate FAILED for {model_kind}: worst |delta| = "
            f"{gate['worst_abs_delta']:.3e} > {GATE_TOL} — store does not reproduce the "
            f"committed #1639 cells; STOPPING before any retrieval read. gate={gate}"
        )
    print(
        f"[gate] {model_kind}: all 12 rung cells match committed "
        f"(worst |delta| = {gate['worst_abs_delta']:.2e})"
    )
    return gate


# ---------------------------------------------------------------------------
# Retrieval: whitened cosine + CSLS primary; companions; mid-rank convention.
# ---------------------------------------------------------------------------
def midranks(d: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Mid-rank of query i's true pool row within distance row i.

    Verbatim the ``mapping_baselines.knn_retrieval`` tolerance-based tie
    convention (1 + #closer + 0.5*#tied-others), for arbitrary true_idx.
    """
    n = d.shape[0]
    d_true = d[np.arange(n), true_idx]
    tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
    closer = (d < d_true[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - d_true[:, None]) <= tol).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied


def whiten(x: np.ndarray, mu: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """z = L^-1 (x - mu): the #2202 train-answer whitening transform."""
    return solve_triangular(ell, (np.asarray(x, np.float64) - mu).T, lower=True).T


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n_a, n_b) cosine-similarity matrix (mapping_baselines normalization)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def ranks_all_metrics(q: np.ndarray, pool: np.ndarray, zq: np.ndarray, zp: np.ndarray) -> dict:
    """Per-metric mid-rank vectors for one (queries, pool) read; true idx = arange.

    q/pool are raw fp64; zq/zp their whitened images (shared fold-train stats).
    """
    n, n_pool = q.shape[0], pool.shape[0]
    assert n == n_pool, (n, n_pool)
    assert K_CSLS < n_pool, f"pool too small for CSLS: {n_pool} <= K={K_CSLS}"
    true_idx = np.arange(n)
    s_wcos = cos_sim(zq, zp)
    out = {
        "whiten_csls": midranks(-csls_scores(s_wcos, K_CSLS), true_idx),
        "whiten_cos": midranks(1.0 - s_wcos, true_idx),
        "cosine": midranks(_pairwise_dist(q, pool, "cosine"), true_idx),
        "euclidean": midranks(_pairwise_dist(q, pool, "euclidean"), true_idx),
    }
    return out


def summarize(ranks: np.ndarray, pool_sizes_per_row: np.ndarray) -> dict:
    """acc@k + chance (mean over rows of k/n_pool of their fold) + median + MRR."""
    return {
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in KS},
        "chance_at_k": {int(k): float(np.mean(k / pool_sizes_per_row)) for k in KS},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "n": int(ranks.shape[0]),
        "n_pool_mean": float(np.mean(pool_sizes_per_row)),
    }


def run_model(model_kind: str, args) -> dict:
    """Full battery for one model: fits + gate + per-fold retrieval + summary."""
    t0 = time.time()
    print(f"[phase=fits] model={model_kind} loading store + fitting rungs")
    arrays = v1.load_persona_arrays(args.store_root, model_kind)
    rung_preds = {
        "M2": m2_fold_preds(arrays, LAYER),
        "M1": pooled_fold_preds(arrays, LAYER, centering="per_persona"),
        "M0": pooled_fold_preds(arrays, LAYER, centering="global"),
    }
    gate = equality_gate(model_kind, {r: rung_preds[r]["r2"] for r in RUNGS})

    # ── per-fold retrieval ──
    print(f"[phase=retrieval] model={model_kind} per-fold whitening + reads")
    folds0 = arrays[PERSONAS[0]]["folds"]
    pooled_ranks: dict[str, dict[str, list]] = {r: {m: [] for m in METRICS} for r in RUNGS}
    pooled_personas: list[np.ndarray] = []
    pooled_pool_rows: list[np.ndarray] = []
    own_ranks: dict[str, dict[str, dict[str, list]]] = {
        p: {r: {m: [] for m in METRICS} for r in RUNGS} for p in PERSONAS
    }
    own_pool_rows: dict[str, list[int]] = {p: [] for p in PERSONAS}
    pool_sizes: list[int] = []
    reconcile: dict | None = None
    for k in range(N_FOLDS):
        y_tr = np.concatenate(
            [
                arrays[p]["Y"][:, LAYER, :][arrays[p]["folds"] != k].astype(np.float64)
                for p in PERSONAS
            ]
        )
        mu_a = y_tr.mean(0)
        ell = shrunk_cholesky_from_cov(np.cov(y_tr, rowvar=False), PRIMARY_LAMBDA)
        del y_tr

        te = {p: arrays[p]["folds"] == k for p in PERSONAS}
        pool = np.concatenate(
            [arrays[p]["Y"][:, LAYER, :][te[p]].astype(np.float64) for p in PERSONAS]
        )
        row_personas = np.concatenate(
            [np.full(int(te[p].sum()), p, dtype=object) for p in PERSONAS]
        )
        zp = whiten(pool, mu_a, ell)
        n_pool = pool.shape[0]
        pool_sizes.append(n_pool)
        pooled_personas.append(row_personas)
        pooled_pool_rows.append(np.full(n_pool, n_pool, dtype=np.float64))
        for p in PERSONAS:
            own_pool_rows[p].append(int(te[p].sum()))

        for rung in RUNGS:
            q = np.concatenate([rung_preds[rung]["preds"][p][te[p]] for p in PERSONAS])
            zq = whiten(q, mu_a, ell)
            rk = ranks_all_metrics(q, pool, zq, zp)
            for m in METRICS:
                pooled_ranks[rung][m].append(rk[m])
            if reconcile is None and rung == "M2":
                # reuse-verification: rank-derived acc@k == knn_retrieval on this read
                rec = {}
                for metric in ("euclidean", "cosine"):
                    ref = knn_retrieval(q, pool, ks=KS, metric=metric)
                    mine = {int(kk): float((rk[metric] <= kk).mean()) for kk in KS}
                    assert mine == ref["acc_at_k"], (metric, mine, ref["acc_at_k"])
                    rec[metric] = {"fold": k, "acc_at_k": mine, "ok": True}
                reconcile = rec
            # own-pool reads (same fold-train whitening stats; convention recorded)
            off = 0
            for p in PERSONAS:
                m_p = int(te[p].sum())
                sl = slice(off, off + m_p)
                rko = ranks_all_metrics(q[sl], pool[sl], zq[sl], zp[sl])
                for m in METRICS:
                    own_ranks[p][rung][m].append(rko[m])
                off += m_p
        print(f"[retrieval] fold {k + 1}/{N_FOLDS} n_pool={n_pool} elapsed={time.time() - t0:.0f}s")

    # ── aggregate over folds ──
    personas_cat = np.concatenate(pooled_personas)
    pool_rows_cat = np.concatenate(pooled_pool_rows)
    retrieval: dict = {"pooled_pool": {}, "own_pool": {p: {} for p in PERSONAS}}
    for rung in RUNGS:
        entry: dict = {}
        for m in METRICS:
            ranks = np.concatenate(pooled_ranks[rung][m])
            entry[m] = summarize(ranks, pool_rows_cat)
            entry[m]["by_persona_rows"] = {
                p: {
                    "acc_at_1": float((ranks[personas_cat == p] <= 1).mean()),
                    "n": int((personas_cat == p).sum()),
                }
                for p in PERSONAS
            }
        retrieval["pooled_pool"][rung] = entry
        for p in PERSONAS:
            sizes = np.concatenate([np.full(nn, nn, dtype=np.float64) for nn in own_pool_rows[p]])
            retrieval["own_pool"][p][rung] = {
                m: summarize(np.concatenate(own_ranks[p][rung][m]), sizes) for m in METRICS
            }
    retrieval["pool_sizes_per_fold"] = {
        "pooled": pool_sizes,
        "own": {p: own_pool_rows[p] for p in PERSONAS},
    }

    result = {
        "meta": {
            "script": SCRIPT,
            "issue": 1639,
            "generated_utc": datetime.now(UTC).isoformat(),
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "elapsed_s": round(time.time() - t0, 1),
            "inputs": HF_INPUT,
            **as_metadata_dict(git_provenance(REPO_ROOT)),
        },
        "config": {
            "model_kind": model_kind,
            "layer": LAYER,
            "n_folds": N_FOLDS,
            "fit_seed": v1.FIT_SEED,
            "gcv_dof_cap": DOF_CAP,
            "lambda_grid": [float(v) for v in fit825.LAMBDAS],
            "ks": list(KS),
            "k_csls": int(K_CSLS),
            "whiten_shrinkage_lambda": float(PRIMARY_LAMBDA),
            "conventions": (
                "PRIMARY = whitened cosine + CSLS: whitening stats fit per fold on the "
                "TRAIN side only (pooled train-fold answers, all 4 personas — one shared "
                "transform per fold so specialized and universal reads are like-for-like); "
                "z = L^-1(x - mu_A) with L = Cholesky of the shrunk train-answer covariance "
                "(Sigma = (1-lam)*cov + lam*diag(cov), lam=0.1, null_battery recipe); "
                "CSLS = issue1901 csls_scores(K=10) on the whitened cosine-sim matrix, "
                "distance = -score; mid-rank ties per mapping_baselines.knn_retrieval. "
                "Pools: matched-target = the fold's pooled held-out answers (all personas); "
                "own-pool = the persona's own fold held-out answers (same fold whitening). "
                "chance_at_k = mean over rows of k / n_pool(fold)."
            ),
        },
        "equality_gate": gate,
        "fit_r2_foldmean": {r: rung_preds[r]["r2"] for r in RUNGS},
        "selected_lambda": {
            "M2_per_persona_per_fold": rung_preds["M2"]["lams"],
            "M1_per_fold": rung_preds["M1"]["lams"],
            "M0_per_fold": rung_preds["M0"]["lams"],
        },
        "reconciliation_knn_retrieval": reconcile,
        "retrieval": retrieval,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"retrieval_{model_kind}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[out] wrote {out_path} ({time.time() - t0:.0f}s)")
    return result


# ---------------------------------------------------------------------------
# Figure: acc@1 (whitened+CSLS, pooled pool) grouped bars, chance line.
# ---------------------------------------------------------------------------
def render_figure(out_dir: Path) -> None:
    """Grouped bars: acc@1 whitened+CSLS on the pooled pool, M2/M1/M0 x model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    models = ("base", "instruct")
    res = {}
    for m in models:
        p = out_dir / f"retrieval_{m}.json"
        assert p.exists(), f"missing {p} — run the compute step for {m} first"
        res[m] = json.loads(p.read_text())

    rung_labels = {
        "M2": "specialized\n(per-character)",
        "M1": "shared\n(per-character offsets)",
        "M0": "pooled\n(global offset)",
    }
    set_paper_style()
    pal = paper_palette(3)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(RUNGS), dtype=float)
    w = 0.36
    chances = []
    for i, m in enumerate(models):
        vals = [
            res[m]["retrieval"]["pooled_pool"][r]["whiten_csls"]["acc_at_k"]["1"] for r in RUNGS
        ]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=pal[i], label=f"Qwen2.5-7B {m}")
        chances.append(res[m]["retrieval"]["pooled_pool"]["M2"]["whiten_csls"]["chance_at_k"]["1"])
    ax.axhline(float(np.mean(chances)), color="0.4", ls="--", lw=1.2, label="chance (1/pool)")
    ax.set_xticks(x, [rung_labels[r] for r in RUNGS])
    ax.set_ylabel("acc@1 (whitened cosine + CSLS)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figs] wrote {FIG_PATH}")


def main() -> int:
    args = parse_args()
    fit825.GCV_DOF_CAP = DOF_CAP
    if args.figure_only:
        render_figure(args.out_dir)
        return 0
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        assert m in v1.MODEL_KINDS, f"unknown model {m!r}"
        out_path = args.out_dir / f"retrieval_{m}.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} exists (pass --force to recompute)")
            continue
        run_model(m, args)
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
