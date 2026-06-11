#!/usr/bin/env python3
# Greek rho / em-dashes intentional in docs+logs
"""Task #536 Phase C — recompute persona-distance cosine statistics under the
canonical metric (global-mean-center -> L2-normalize -> cosine) from persisted
artifacts, and re-grade each affected task's headline statistic.

Per plan §4-C: one FAMILY_REGISTRY (centroid banks / Gram matrices) + one
TASK_ADAPTERS table (per-task join of recomputed X with the task's persisted Y,
recomputing the task's OWN published estimator). Mandatory join-validity gate:
the raw recompute must reproduce the published/persisted number BEFORE the
centered value is read (1e-4 matrix assert where a published matrix persists;
|Δρ|<=0.02 / slope-in-CI statistic-level otherwise). Rows are appended to
``eval_results/issue_536/regrade_table.json`` the moment they are computed
(checkpoint-per-row, never accumulate-then-write).

Re-grade labels come EXACTLY from the plan §3 pre-registered decision tree:
  significant rows: flips -> weakens(significance) -> weakens(magnitude)
                    -> strengthens -> stands  (scale-invariant M, never raw slope)
  null rows:        null-overturned (per the row's multiplicity rule) -> stands
  matrix-only rows: sensitivity-agrees / sensitivity-disagrees /
                    sensitivity-unreliable  (NEVER canonical labels)

Usage::

    uv run python scripts/issue536_recompute_driver.py \
        --data-root /home/thomasjiralerspong/explore-persona-space [--only 478]

CPU-only; no GPU, no pod. Reuses
``explore_persona_space.analysis.representation_shift.compute_cosine_matrix``
(the centering is NOT reimplemented).
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from scipy import stats as sps

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    compute_cosine_matrix,
)

log = logging.getLogger("i536.driver")

OUT_DIR = REPO / "eval_results" / "issue_536"
SNAP_DIR = OUT_DIR / "inputs"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Magnitude boundaries of the §3 decision tree (pre-registered conventions).
M_WEAKEN_FACTOR = 0.5
M_STRENGTHEN_FACTOR = 1.5
PSD_CLIP_TRACE_FRAC = 1e-3
GATE_MATRIX_TOL = 1e-4
GATE_RHO_TOL = 0.02


# ──────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────
def _git_sha() -> str:
    """Current commit of the tree this script runs from (reproducibility)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except Exception as e:  # pragma: no cover — metadata only, never silent
        log.warning("git rev-parse failed: %s", e)
        return "unknown"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _names_hash(names: list[str]) -> str:
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:12]


def append_row(out_path: Path, row: dict) -> None:
    """Checkpoint-per-row: append ``row`` to the JSON table immediately."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        payload = json.loads(out_path.read_text())
    else:
        payload = {
            "schema_version": "i536_regrade_v1",
            "generated_at": _now(),
            "git_commit": _git_sha(),
            "rows": [],
        }
    payload["rows"] = [r for r in payload["rows"] if r.get("row_id") != row.get("row_id")]
    payload["rows"].append(row)
    payload["updated_at"] = _now()
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(out_path)
    log.info("[row] %s -> %s", row.get("row_id"), out_path.name)


def off_diag(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)]


def od_stats(M: np.ndarray) -> dict:
    v = off_diag(np.asarray(M, dtype=np.float64))
    return {
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "span": float(v.max() - v.min()),
    }


def length_partial_spearman(x, y, lengths) -> float:
    """Verbatim mirror of scripts/analyze_issue415.py::length_partial_spearman."""
    rx = sps.spearmanr(x, lengths).statistic
    ry = sps.spearmanr(y, lengths).statistic
    rxy = sps.spearmanr(x, y).statistic
    denom = np.sqrt((1 - rx**2) * (1 - ry**2))
    if denom < 1e-9:
        return float("nan")
    return float((rxy - rx * ry) / denom)


def rank_residual_partial(x, y, covar) -> tuple[float, float]:
    """Verbatim mirror of scripts/i474_cosine_followup.py::_length_partial."""
    rx, ry, rc = sps.rankdata(x), sps.rankdata(y), sps.rankdata(covar)
    ex = rx - np.polyval(np.polyfit(rc, rx, 1), rc)
    ey = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
    r = sps.pearsonr(ex, ey)
    return float(r.statistic), float(r.pvalue)


def holm_reject(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down over a named family; returns per-cell reject flags."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, bool] = {}
    still = True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        still = still and (p <= thresh)
        out[k] = bool(still)
    return out


def cluster_ols(y: np.ndarray, X: np.ndarray, clusters: np.ndarray):
    """OLS with cluster-robust SEs (statsmodels); X WITHOUT constant."""
    import statsmodels.api as sm

    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": clusters})


def spearman(x, y) -> tuple[float, float]:
    r = sps.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def label_significant(
    *,
    sign_raw: float,
    sign_mc: float,
    p_mc: float,
    alpha: float,
    m_raw: float,
    m_mc: float,
) -> str:
    """Plan §3 ordered decision tree for originally-SIGNIFICANT rows."""
    if (sign_mc != sign_raw) and (p_mc < alpha):
        return "flips"
    if p_mc >= alpha:
        return "weakens (significance lost)"
    if abs(m_mc) <= M_WEAKEN_FACTOR * abs(m_raw):
        return "weakens (magnitude)"
    if abs(m_mc) >= M_STRENGTHEN_FACTOR * abs(m_raw):
        return "strengthens"
    return "stands"


def label_null(rescued: bool) -> str:
    return "null-overturned (candidate rescue)" if rescued else "stands (null persists)"


def gram_centered_config(G_sim: np.ndarray) -> tuple[np.ndarray, float]:
    """Approximate (normalized-vector centering) read for matrix-only rows.

    Input: a TRUE similarity Gram matrix of already-normalized vectors.
    Symmetrize, eigendecompose, clip negative eigenvalues at 0 (recording the
    clipped mass fraction of the trace), recover the configuration up to an
    orthogonal map (cosine-invariant), global-mean-center those unit vectors,
    re-normalize, re-cosine. Returns (cos_mc_approx, clipped_mass_frac).
    """
    G = np.asarray(G_sim, dtype=np.float64)
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    clipped = float(np.clip(-w, 0.0, None).sum())
    trace = float(np.trace(G))
    w = np.clip(w, 0.0, None)
    X = V @ np.diag(np.sqrt(w))  # rows = recovered configuration
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    U = Xc / norms
    return U @ U.T, (clipped / trace if trace > 0 else float("inf"))


# ──────────────────────────────────────────────────────────────────────────
# FAMILY_REGISTRY — centroid banks / Gram matrices (plan §4-C)
# ──────────────────────────────────────────────────────────────────────────
def family_111bank(data_root: Path) -> dict:
    """#478/#490 distance source: 111-persona L20 bank (single_token_100_persona).

    Join gate (matrix, 1e-4): raw distance recompute must reproduce the cached
    ``cosine_distance_matrix_layer20.json`` the #478/#490 pipelines consumed
    (built normalize-only by scripts/_issue478_common.py at 69b34b94).
    """
    base = data_root / "eval_results" / "single_token_100_persona"
    # persona_names.json no longer persists on disk; the cached distance JSON
    # (written by _build_matrix_from_centroids together with the matrix, from
    # the same names file) carries the canonical order. The 1e-4 matrix gate
    # below validates the order still matches the tensor rows.
    cached = json.loads((base / "cosine_distance_matrix_layer20.json").read_text())
    names = cached["persona_names"]
    C = torch.load(
        base / "centroids" / "centroids_layer20.pt", map_location="cpu", weights_only=True
    ).to(torch.float32)
    assert C.shape[0] == len(names), (C.shape, len(names))
    cos_raw = compute_cosine_matrix(C, centering="none").numpy().astype(np.float64)
    cos_mc = compute_cosine_matrix(C, centering="global_mean").numpy().astype(np.float64)
    assert cached.get("metric") == "1 - cosine", cached.get("metric")
    D_cached = np.asarray(cached["matrix"], dtype=np.float64)
    gate_dev = float(np.abs((1.0 - cos_raw) - D_cached).max())
    if gate_dev > GATE_MATRIX_TOL:
        raise RuntimeError(
            f"111-bank matrix join gate FAILED: max |recomputed_raw_dist - cached| = {gate_dev:.3e}"
        )
    return {
        "family": "single_token_100p_L20",
        "names": names,
        "n": len(names),
        "names_hash": _names_hash(names),
        "cos_raw": cos_raw,
        "cos_mc": cos_mc,
        "gate": {"level": "matrix-1e-4", "max_abs_dev": gate_dev, "vs": str(base)},
        "layer": 20,
    }


def family_20bank(data_root: Path) -> dict:
    """#405's distance source: 20-persona L20 bank (extraction_method_comparison).

    Join gate (matrix, 1e-4): raw cosine recompute must reproduce
    ``cosine_matrix_a_layer20.json`` (the artifact #405 consumed — RAW regime).
    """
    base = data_root / "eval_results" / "extraction_method_comparison"
    cached = json.loads((base / "cosine_matrix_a_layer20.json").read_text())
    names = cached["persona_names"]
    bundle = torch.load(base / "centroids_method_a.pt", map_location="cpu", weights_only=True)
    C = bundle["layer_20"].to(torch.float32)
    assert C.shape[0] == len(names), (C.shape, len(names))
    cos_raw = compute_cosine_matrix(C, centering="none").numpy().astype(np.float64)
    cos_mc = compute_cosine_matrix(C, centering="global_mean").numpy().astype(np.float64)
    M_cached = np.asarray(cached["matrix"], dtype=np.float64)
    gate_dev = float(np.abs(cos_raw - M_cached).max())
    if gate_dev > GATE_MATRIX_TOL:
        raise RuntimeError(
            f"20-bank matrix join gate FAILED: max |recomputed_raw - cached| = {gate_dev:.3e}"
        )
    return {
        "family": "extraction_method_a_L20",
        "names": names,
        "n": len(names),
        "names_hash": _names_hash(names),
        "cos_raw": cos_raw,
        "cos_mc": cos_mc,
        "gate": {"level": "matrix-1e-4", "max_abs_dev": gate_dev, "vs": str(base)},
        "layer": 20,
    }


def family_n24(data_root: Path, layer: int = 15) -> dict:
    """#396/#415 bank: 24-persona centroids from #274 (layers 0-27 dict bundle)."""
    p = data_root / "eval_results" / "issue_274" / "centroids" / "centroids_n24_layers0_27.pt"
    bundle = torch.load(p, map_location="cpu", weights_only=False)
    layer_dict = bundle[layer]
    names = sorted(layer_dict.keys())
    C = torch.stack([layer_dict[n].to(torch.float32) for n in names])
    cos_raw = compute_cosine_matrix(C, centering="none").numpy().astype(np.float64)
    cos_mc = compute_cosine_matrix(C, centering="global_mean").numpy().astype(np.float64)
    bank_mean = C.mean(dim=0)
    centered_norms = (C - bank_mean.unsqueeze(0)).norm(dim=1).numpy().astype(np.float64)
    return {
        "family": f"issue274_n24_L{layer}",
        "names": names,
        "n": len(names),
        "names_hash": _names_hash(names),
        "cos_raw": cos_raw,
        "cos_mc": cos_mc,
        "centered_norms": centered_norms,
        "gate": {"level": "statistic (no published matrix persists for this bundle)"},
        "layer": layer,
    }


def family_505(data_root: Path, layer: int = 21) -> dict:
    """#505 bank: persona-vectors centroids (HF issue505_loo_contrastive/geometry).

    Join gate (matrix, 1e-4): raw cos(b, j) must reproduce the persisted
    ``eval_results/issue_505/analysis/panel_similarity_matrix.json`` at L21.
    """
    local = data_root / "data" / "issue_505" / "centroids_pv" / f"centroids_pv_L{layer}.pt"
    if local.exists():
        path = local
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                HF_DATA_REPO,
                f"issue505_loo_contrastive/geometry/centroids_pv_L{layer}.pt",
                repo_type="dataset",
            )
        )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    names = list(bundle["persona_names"])
    C = bundle["centroids"].to(torch.float32)
    assert C.shape[0] == len(names), (C.shape, len(names))
    cos_raw = compute_cosine_matrix(C, centering="none").numpy().astype(np.float64)
    cos_mc = compute_cosine_matrix(C, centering="global_mean").numpy().astype(np.float64)
    sim = json.loads(
        (
            data_root / "eval_results" / "issue_505" / "analysis" / "panel_similarity_matrix.json"
        ).read_text()
    )
    cos_b_j = sim[f"L{layer}"]["cos_b_j"]
    idx = {n: i for i, n in enumerate(names)}
    devs = []
    for b, jd in cos_b_j.items():
        for j, v in jd.items():
            if b in idx and j in idx:
                devs.append(abs(cos_raw[idx[b], idx[j]] - float(v)))
    if not devs:
        raise RuntimeError("505 join gate: no overlapping (b, j) pairs found")
    gate_dev = float(max(devs))
    if gate_dev > GATE_MATRIX_TOL:
        raise RuntimeError(
            f"505 matrix join gate FAILED at L{layer}: max |recomputed - persisted| = "
            f"{gate_dev:.3e} over {len(devs)} (b, j) pairs"
        )
    return {
        "family": f"issue505_pv_L{layer}",
        "names": names,
        "n": len(names),
        "names_hash": _names_hash(names),
        "cos_raw": cos_raw,
        "cos_mc": cos_mc,
        "gate": {"level": "matrix-1e-4", "max_abs_dev": gate_dev, "n_pairs": len(devs)},
        "layer": layer,
        "source_path": str(path),
    }


def family_406_gram(data_root: Path) -> dict:
    """#406-lineage matrix-only family: per-layer distance JSONs -> Gram reads.

    Distance-form 1-cos is converted to similarity BEFORE the fingerprint and
    Gram read (plan §4-B). Approximate (normalized-vector centering) only —
    sensitivity-* namespace, NEVER canonical labels.
    """
    base = data_root / "eval_results" / "issue_406" / "cosine"
    layers = [0, 5, 11, 15, 21, 27]
    per_layer: dict[int, dict] = {}
    conds = None
    for layer in layers:
        d = json.loads((base / f"C_L{layer}.json").read_text())
        conds = d["conditions"]
        D = np.array([[float(d["matrix"][a][b]) for b in conds] for a in conds], dtype=np.float64)
        G = 1.0 - D  # similarity
        cos_mc_approx, clipped = gram_centered_config(G)
        per_layer[layer] = {
            "cos_raw": G,
            "cos_mc_approx": cos_mc_approx,
            "clipped_mass_frac": clipped,
            "fingerprint_raw": od_stats(G),
            "fingerprint_mc": od_stats(cos_mc_approx),
        }
    return {
        "family": "issue406_gram",
        "names": conds,
        "n": len(conds),
        "names_hash": _names_hash(conds),
        "per_layer": per_layer,
        "gate": {"level": "matrix-only (Gram read; distance converted G = 1 - D)"},
    }


def family_341_gram() -> dict:
    """#341 matrix-only family: phase_minus1 cosine_matrix.json (true Gram)."""
    p = REPO / "experiments" / "phase_minus1_persona_vectors" / "cosine_matrix.json"
    d = json.loads(p.read_text())
    per_layer: dict[int, dict] = {}
    names = None
    for lk, payload in d.items():
        layer = int(lk.split("_")[1])
        names = payload["persona_names"]
        G = np.asarray(payload["matrix"], dtype=np.float64)
        cos_mc_approx, clipped = gram_centered_config(G)
        per_layer[layer] = {
            "cos_raw": G,
            "cos_mc_approx": cos_mc_approx,
            "clipped_mass_frac": clipped,
            "fingerprint_raw": od_stats(G),
            "fingerprint_mc": od_stats(cos_mc_approx),
        }
    return {
        "family": "issue341_gram",
        "names": names,
        "n": len(names),
        "names_hash": _names_hash(names),
        "per_layer": per_layer,
        "gate": {"level": "matrix-only (Gram read on persisted similarity matrix)"},
    }


# ──────────────────────────────────────────────────────────────────────────
# TASK_ADAPTERS (plan §4-C) — each appends its row(s) immediately
# ──────────────────────────────────────────────────────────────────────────
def verify_66(data_root: Path, out: Path, payload: dict) -> None:
    """Canonical-line verification (H1): reproduce #66's published per-source
    Spearman (centered cosine at L20 vs marker leakage rate) from persisted
    centroids + per-source marker_eval.json. Expected: stands."""
    fam = family_111bank(data_root)
    base = data_root / "eval_results" / "single_token_100_persona"
    published = json.loads((base / "cosine_leakage_correlation.json").read_text())["layer20"]
    sources = ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]
    idx = {n: i for i, n in enumerate(fam["names"])}
    per_source = {}
    pooled_x, pooled_y, pooled_x_raw = [], [], []
    for src in sources:
        leak = json.loads((base / src / "marker_eval.json").read_text())
        xs, ys, xs_raw = [], [], []
        for tgt in fam["names"]:
            if tgt == src or tgt not in leak:
                continue
            xs.append(fam["cos_mc"][idx[src], idx[tgt]])
            xs_raw.append(fam["cos_raw"][idx[src], idx[tgt]])
            ys.append(float(leak[tgt]["rate"]))
        rho_mc, p_mc = spearman(xs, ys)
        rho_raw, p_raw = spearman(xs_raw, ys)
        pub = published[src]
        gate_ok = abs(rho_mc - pub["spearman_rho"]) <= GATE_RHO_TOL
        per_source[src] = {
            "n_pairs": len(xs),
            "published_rho_centered": pub["spearman_rho"],
            "published_p": pub["spearman_p"],
            "recomputed_rho_centered": rho_mc,
            "recomputed_p_centered": p_mc,
            "sensitivity_rho_raw_recipe": rho_raw,
            "sensitivity_p_raw_recipe": p_raw,
            "gate_pass": gate_ok,
        }
        pooled_x.extend(xs)
        pooled_y.extend(ys)
        pooled_x_raw.extend(xs_raw)
        if not gate_ok:
            raise RuntimeError(
                f"verify_66 join gate FAILED for source {src}: recomputed centered rho "
                f"{rho_mc:.4f} vs published {pub['spearman_rho']:.4f} (H1: canonical-line "
                f"verification inconclusive — investigate bank-version drift, do not proceed)"
            )
    rho_pool_mc, p_pool_mc = spearman(pooled_x, pooled_y)
    rho_pool_raw, p_pool_raw = spearman(pooled_x_raw, pooled_y)
    pub_agg = published["_aggregate"]
    all_pass = all(v["gate_pass"] for v in per_source.values())
    x_spearman, _ = spearman(pooled_x_raw, pooled_x)
    row = {
        "row_id": "66-verify",
        "task": 66,
        "config_slug": "verify_centered",
        "cosine_path_used": "centered (canonical) — analyze_100_persona_cosine.py:288-293",
        "recoverability": "CPU (verified)",
        "family": fam["family"],
        "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": 20},
        "gate_level": "statistic (|drho|<=0.02 vs published per-source rho; bank matrix gate "
        "also passed at 1e-4 vs the cached raw-distance JSON)",
        "original_stat": {
            "estimator": "Spearman rho per source (N=110) + pooled (N=550), centered cos L20",
            "per_source_rho": {s: published[s]["spearman_rho"] for s in sources},
            "pooled_rho": pub_agg["spearman_rho"],
            "pooled_p": pub_agg["spearman_p"],
        },
        "recomputed_stat": {
            "per_source": per_source,
            "pooled_rho_centered": rho_pool_mc,
            "pooled_p_centered": p_pool_mc,
            "pooled_rho_raw_recipe_sensitivity": rho_pool_raw,
            "pooled_p_raw_recipe_sensitivity": p_pool_raw,
        },
        "raw_vs_centered_x_spearman": x_spearman,
        "n": {"per_source": 110, "pooled": len(pooled_x)},
        "regrade_label": "stands" if all_pass else "join-failed",
        "notes": "Verification row (original recipe WAS canonical/centered). The raw-recipe "
        "columns are a sensitivity read, not the original. H1 verification "
        + ("PASSED." if all_pass else "FAILED."),
        "computed_at": _now(),
    }
    append_row(out, row)
    payload["fig_66"] = {
        "per_source_rho_centered": {s: per_source[s]["recomputed_rho_centered"] for s in sources},
        "per_source_rho_raw": {s: per_source[s]["sensitivity_rho_raw_recipe"] for s in sources},
        "pooled": {
            "centered": rho_pool_mc,
            "raw": rho_pool_raw,
        },
    }
    payload["bank_offdiag"] = payload.get("bank_offdiag", {})
    payload["bank_offdiag"]["single_token_100p_L20"] = {
        "raw": off_diag(fam["cos_raw"]).tolist(),
        "centered": off_diag(fam["cos_mc"]).tolist(),
    }


def row_99(data_root: Path, out: Path, payload: dict) -> None:
    """#99 verification partition: X (centered #66 centroids) persists, but the
    per-(source x behavior) bystander-delta Y tables were never committed."""
    missing = [
        "eval_results/capability_leakage/",
        "eval_results/misalignment_leakage_v2/",
        "eval_results/refusal_leakage/",
        "eval_results/sycophancy_leakage/",
    ]
    actually_missing = [m for m in missing if not (data_root / m).exists()]
    row = {
        "row_id": "99-partition",
        "task": 99,
        "config_slug": "verify_centered",
        "cosine_path_used": "centered (canonical) — reused #66 centroids per #99 body",
        "recoverability": "unrecoverable-Y (X persists; Y tables never committed)",
        "gate_level": "n/a (no join attempted)",
        "original_stat": {
            "estimator": "per-(source x behavior) Spearman bystander-delta vs centered cos L20",
            "summary": "19/24 significant (p<0.05), |rho| 0.02-0.79 (#99 body)",
        },
        "recomputed_stat": None,
        "regrade_label": "join-failed (Y unrecoverable)",
        "notes": "The 4 behavior Y dirs named in #99's Reproducibility "
        f"({', '.join(missing)}) are absent locally AND have zero entries in git history "
        f"(checked at implementation time; absent now: {actually_missing}). WandB project "
        "thomasjiralerspong/capability_leakage holds training runs, not the per-bystander "
        "eval tables. #99 used the CENTERED recipe (same #66 centroids), so the canonical "
        "pin does not threaten it; verification is simply not executable from disk. "
        "Follow-up pointer: re-run the 4 behavior evals (GPU) only if #99's calls are "
        "ever load-bearing for a new decision.",
        "computed_at": _now(),
    }
    append_row(out, row)


def regrade_405(data_root: Path, out: Path, payload: dict) -> None:
    """#405 (secondary-lean): MixedLM deltaLogP ~ K * min_dist on the CORE 336
    rows; distance source was the RAW 20-bank. Mirror per-K OLS slopes (gated
    against the published per_K_slopes_min_full) + the MixedLM headline."""
    fam = family_20bank(data_root)
    names = fam["names"]
    idx = {n: i for i, n in enumerate(names)}
    dist_raw = 1.0 - fam["cos_raw"]
    dist_mc = 1.0 - fam["cos_mc"]
    pub = json.loads(
        (data_root / "eval_results" / "issue_405" / "aggregate" / "regression.json").read_text()
    )["runs"]
    rows = []
    with (
        data_root / "eval_results" / "issue_405" / "aggregate" / "per_cell_persona_tidy.csv"
    ).open() as f:
        for r in csv.DictReader(f):
            if r["track"] != "CORE":
                continue
            positives = list(ast.literal_eval(r["positives"]))
            held = r["held_persona"]
            md_raw = min(dist_raw[idx[held], idx[p]] for p in positives)
            md_mc = min(dist_mc[idx[held], idx[p]] for p in positives)
            dev = abs(md_raw - float(r["min_dist"]))
            rows.append(
                {
                    "K": int(r["K"]),
                    "subset": r["positives"],
                    "persona": held,
                    "seed": int(r["seed"]),
                    "cell_id": r["cell_id"],
                    "deltaLogP_mean": float(r["deltaLogP_mean"]),
                    "min_dist_raw": float(md_raw),
                    "min_dist_mc": float(md_mc),
                    "row_dev": dev,
                }
            )
    max_dev = max(r["row_dev"] for r in rows)
    if max_dev > GATE_MATRIX_TOL:
        raise RuntimeError(f"405 row-level min_dist join gate FAILED: max dev {max_dev:.3e}")
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    assert len(df) == 336, f"#405 CORE rows = {len(df)}, expected 336"

    def _mixed(dist_col: str) -> dict:
        d = df.rename(columns={dist_col: "min_dist"})[
            ["deltaLogP_mean", "K", "min_dist", "persona", "subset"]
        ].copy()
        d["dummy_const"] = 1
        vc = {"subset": "0 + C(subset)", "persona": "0 + C(persona)"}
        fit = smf.mixedlm(
            "deltaLogP_mean ~ K * min_dist", d, groups="dummy_const", vc_formula=vc
        ).fit(method=["lbfgs"], reml=True)
        return {
            t: {"beta": float(fit.params[t]), "p": float(fit.pvalues[t])}
            for t in fit.params.index
            if t in fit.bse.index
        }

    def _per_k(dist_col: str) -> dict:
        outd = {}
        for K in sorted(df["K"].unique()):
            sub = df[df["K"] == K].rename(columns={dist_col: "min_dist"})
            fit = smf.ols("deltaLogP_mean ~ min_dist", data=sub).fit()
            rho, p_rho = spearman(sub["min_dist"], sub["deltaLogP_mean"])
            outd[int(K)] = {
                "n": len(sub),
                "beta": float(fit.params["min_dist"]),
                "p": float(fit.pvalues["min_dist"]),
                "spearman_rho": rho,
                "spearman_p": p_rho,
            }
        return outd

    perk_raw = _per_k("min_dist_raw")
    perk_mc = _per_k("min_dist_mc")
    pub_perk = pub["per_K_slopes_min_full"]["per_K"]
    for K, v in perk_raw.items():
        lo, hi = pub_perk[str(K)]["ci_95"]
        if not (lo - 1e-6 <= v["beta"] <= hi + 1e-6):
            raise RuntimeError(
                f"405 per-K statistic gate FAILED at K={K}: raw refit beta {v['beta']:.3f} "
                f"outside published CI [{lo:.3f}, {hi:.3f}]"
            )
    mixed_raw = _mixed("min_dist_raw")
    mixed_mc = _mixed("min_dist_mc")
    pub_beta = pub["headline_full"]["coefs"]["min_dist"]["Estimate"]
    alpha = 0.01
    x_spear, _ = spearman(df["min_dist_raw"], df["min_dist_mc"])
    # The published claim is the POOLED distance effect (MixedLM min_dist
    # beta = -27.7, p < 1e-60, n = 336); per-K reads are descriptive strata
    # (K=8 has only 2 subsets, too few for a per-stratum claim). M = pooled
    # Spearman; significance read off BOTH the pooled Spearman and the
    # mirrored MixedLM under the centered metric.
    rho_pool_raw, p_pool_raw = spearman(df["min_dist_raw"], df["deltaLogP_mean"])
    rho_pool_mc, p_pool_mc = spearman(df["min_dist_mc"], df["deltaLogP_mean"])
    label = label_significant(
        sign_raw=math.copysign(1, rho_pool_raw),
        sign_mc=math.copysign(1, rho_pool_mc),
        p_mc=max(p_pool_mc, float(mixed_mc["min_dist"]["p"])),
        alpha=alpha,
        m_raw=rho_pool_raw,
        m_mc=rho_pool_mc,
    )
    row = {
        "row_id": "405-secondary",
        "task": 405,
        "config_slug": "regrade_raw",
        "cosine_path_used": "RAW — cosine_matrix_a_layer20.json "
        "(issue405_clean_result_analysis.py:46-50)",
        "recoverability": "CPU (verified)",
        "family": fam["family"],
        "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": 20},
        "gate_level": "matrix-1e-4 (bank vs cached JSON) + row-level min_dist 1e-4 "
        "+ statistic (per-K raw OLS slopes inside published CIs)",
        "alpha": alpha,
        "original_stat": {
            "estimator": "MixedLM deltaLogP ~ K * min_dist (n=336) + per-K OLS slopes",
            "min_dist_beta": pub_beta,
            "min_dist_p": pub["headline_full"]["coefs"]["min_dist"]["P-val"],
            "per_K_beta": {k: pub_perk[k]["beta"] for k in pub_perk},
        },
        "recomputed_stat": {
            "raw": {
                "mixed": mixed_raw,
                "per_K": perk_raw,
                "pooled_spearman": {"rho": rho_pool_raw, "p": p_pool_raw},
            },
            "centered": {
                "mixed": mixed_mc,
                "per_K": perk_mc,
                "pooled_spearman": {"rho": rho_pool_mc, "p": p_pool_mc},
            },
            "note_on_scale": "raw-vs-centered slope deltas are descriptive only (centering "
            "rescales X); the label is driven by the scale-invariant POOLED Spearman rho",
        },
        "raw_vs_centered_x_spearman": x_spear,
        "n": {"rows": len(df), "per_K": {k: v["n"] for k, v in perk_raw.items()}},
        "regrade_label": label,
        "notes": "Secondary-lean row: #405's distance axis (panel design + headline "
        "regression covariate) was built in the RAW 20-bank geometry. M = per-K Spearman.",
        "computed_at": _now(),
    }
    append_row(out, row)
    payload["fig_405"] = {
        "per_K_rho_raw": {k: v["spearman_rho"] for k, v in perk_raw.items()},
        "per_K_rho_mc": {k: v["spearman_rho"] for k, v in perk_mc.items()},
        "pooled_rho": {"raw": rho_pool_raw, "mc": rho_pool_mc},
        "label": label,
    }
    payload["bank_offdiag"] = payload.get("bank_offdiag", {})
    payload["bank_offdiag"]["extraction_method_a_L20"] = {
        "raw": off_diag(fam["cos_raw"]).tolist(),
        "centered": off_diag(fam["cos_mc"]).tolist(),
    }


def regrade_478(data_root: Path, out: Path, payload: dict) -> None:
    """#478: per-K slopes of deltaLogP vs log(min dist to K-subset) [significant]
    + the flatness null (near-far gap slope vs log2 K, K x log(dist) interaction).
    Distance source was the RAW 111-bank; tidy.csv snapshot from commit 69b34b94."""
    import pandas as pd
    import statsmodels.formula.api as smf

    fam = family_111bank(data_root)
    idx = {n: i for i, n in enumerate(fam["names"])}
    dist_raw = 1.0 - fam["cos_raw"]
    dist_mc = 1.0 - fam["cos_mc"]
    df = pd.read_csv(SNAP_DIR / "i478_tidy_69b34b94.csv")
    assert len(df) == 2800, f"#478 tidy rows = {len(df)}, expected 2800"

    def _min_dist(row, D):
        subs = row["positives"].split(";")
        return min(D[idx[row["held_out_persona"]], idx[s]] for s in subs)

    df["md_raw"] = df.apply(lambda r: _min_dist(r, dist_raw), axis=1)
    df["md_mc"] = df.apply(lambda r: _min_dist(r, dist_mc), axis=1)
    max_dev = float((df["md_raw"] - df["min_dist"]).abs().max())
    if max_dev > GATE_MATRIX_TOL:
        raise RuntimeError(f"478 row-level min_dist join gate FAILED: max dev {max_dev:.3e}")
    df["log_md_raw"] = np.log(df["md_raw"])
    df["log_md_mc"] = np.log(df["md_mc"])
    df["log2K"] = np.log2(df["K"])
    df["cluster"] = df["cell_id"].astype(str) + "|s" + df["seed"].astype(str)

    pub_slopes = {1: -1.37, 2: -1.41, 4: -1.35, 8: -1.32}

    def _per_k(col: str) -> dict:
        outd = {}
        for K in sorted(df["K"].unique()):
            sub = df[df["K"] == K]
            fit = smf.ols(f"deltaLogP_mean ~ {col}", data=sub).fit()
            rho, p_rho = spearman(sub[col], sub["deltaLogP_mean"])
            sd_x, sd_y = float(sub[col].std()), float(sub["deltaLogP_mean"].std())
            outd[int(K)] = {
                "n": len(sub),
                "beta": float(fit.params[col]),
                "p": float(fit.pvalues[col]),
                "std_beta": float(fit.params[col]) * sd_x / sd_y,
                "spearman_rho": rho,
                "spearman_p": p_rho,
            }
        return outd

    perk_raw = _per_k("log_md_raw")
    perk_mc = _per_k("log_md_mc")
    for K, pub_b in pub_slopes.items():
        if abs(perk_raw[K]["beta"] - pub_b) > 0.02:
            raise RuntimeError(
                f"478 per-K statistic gate FAILED at K={K}: raw refit beta "
                f"{perk_raw[K]['beta']:.3f} vs published {pub_b}"
            )

    # Flatness read 1 — near-far band gap vs log2 K. Bands re-derived under each
    # metric by EQUAL-COUNT re-ranking (mirrors the quantile-based banding: the
    # original band sizes are preserved; personas re-ranked by their per-persona
    # median min_dist under the metric).
    band_order = ["near", "near-mid", "mid", "far", "very-far", "tail"]
    pers_band = df.groupby("held_out_persona")["band"].agg(lambda s: s.mode()[0])
    band_sizes = pers_band.value_counts().to_dict()

    def _reband(col: str) -> dict[str, str]:
        med = df.groupby("held_out_persona")[col].median().sort_values()
        new_band, k = {}, 0
        ordered = list(med.index)
        for b in band_order:
            for _ in range(band_sizes[b]):
                new_band[ordered[k]] = b
                k += 1
        return new_band

    def _gap_slope(col: str, bands: dict[str, str]) -> dict:
        d = df.copy()
        d["band2"] = d["held_out_persona"].map(bands)
        near = d[d["band2"].isin(["near", "near-mid"])]
        far = d[d["band2"].isin(["far", "very-far", "tail"])]
        gaps = {}
        for K in sorted(d["K"].unique()):
            gaps[int(K)] = float(
                far[far["K"] == K]["deltaLogP_mean"].mean()
                - near[near["K"] == K]["deltaLogP_mean"].mean()
            )
        x = np.log2(np.array(sorted(gaps)))
        y = np.array([gaps[k] for k in sorted(gaps)])
        fit = sps.linregress(x, y)
        return {
            "per_K_gap": gaps,
            "slope": float(fit.slope),
            "se": float(fit.stderr),
            "p": float(fit.pvalue),
        }

    bands_raw = _reband("md_raw")
    bands_mc = _reband("md_mc")
    gap_raw = _gap_slope("md_raw", bands_raw)
    gap_design = _gap_slope("md_raw", pers_band.to_dict())
    gap_mc = _gap_slope("md_mc", bands_mc)
    crosstab = (
        pd.crosstab(
            pers_band.rename("design_band"),
            pd.Series(bands_mc, name="centered_band"),
        )
        .reindex(index=band_order, columns=band_order, fill_value=0)
        .to_dict()
    )
    n_moved = int(sum(1 for p, b in bands_mc.items() if pers_band[p] != b))

    # Flatness read 2 — K x log(dist) interaction, cluster-robust OLS at
    # (cell_id, seed). The published co-primary was MixedLM (+0.010, p=0.405);
    # this is the plan-H2 cluster-robust Wald mirror, computed from the SAME
    # estimator on raw AND centered so the comparison is internally consistent.
    def _interaction(col: str) -> dict:
        X = np.column_stack([df[col], df["K"], df[col] * df["K"]])
        fit = cluster_ols(df["deltaLogP_mean"].to_numpy(), X, df["cluster"].to_numpy())
        return {"beta": float(fit.params[3]), "p": float(fit.pvalues[3])}

    inter_raw = _interaction("log_md_raw")
    inter_mc = _interaction("log_md_mc")

    alpha = 0.01
    # Sub-row A (significant): per-K slopes. M = per-K Spearman rho.
    label_slopes = label_significant(
        sign_raw=math.copysign(1, np.mean([v["spearman_rho"] for v in perk_raw.values()])),
        sign_mc=math.copysign(1, np.mean([v["spearman_rho"] for v in perk_mc.values()])),
        p_mc=max(v["spearman_p"] for v in perk_mc.values()),
        alpha=alpha,
        m_raw=np.mean([v["spearman_rho"] for v in perk_raw.values()]),
        m_mc=np.mean([v["spearman_rho"] for v in perk_mc.values()]),
    )
    # Sub-row B (null): flatness. Family = {gap slope, interaction}, Holm at the
    # published alpha=0.01 (plan H2). Rescue = any family member significant
    # after Holm under the centered read.
    fam_p = {"gap_slope": gap_mc["p"], "interaction": inter_mc["p"]}
    holm = holm_reject(fam_p, alpha=0.05)
    holm_h2 = holm_reject(fam_p, alpha=0.01)  # plan H2's stricter alpha for the Wald test
    rescued = any(holm.values())
    label_flat = label_null(rescued)
    x_spear, _ = spearman(df["md_raw"], df["md_mc"])

    append_row(
        out,
        {
            "row_id": "478-perK-slopes",
            "task": 478,
            "config_slug": "regrade_raw",
            "cosine_path_used": "RAW — cosine_distance_matrix_layer20.json "
            "(scripts/_issue478_common.py@69b34b94, normalize-only)",
            "recoverability": "CPU (verified; tidy.csv snapshotted from 69b34b94)",
            "family": fam["family"],
            "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": 20},
            "gate_level": "matrix-1e-4 (bank) + row-level min_dist 1e-4 + statistic "
            "(per-K raw OLS slopes within 0.02 of published)",
            "alpha": alpha,
            "original_stat": {
                "estimator": "per-K OLS slope of deltaLogP vs log(min_dist)",
                "per_K_beta": pub_slopes,
                "note": "each published p < 1e-100",
            },
            "recomputed_stat": {
                "raw": perk_raw,
                "centered": perk_mc,
                "note_on_scale": "raw-vs-centered beta deltas are descriptive only; "
                "M = per-K Spearman rho + standardized beta",
            },
            "raw_vs_centered_x_spearman": x_spear,
            "n": {"rows": len(df), "per_K": {k: v["n"] for k, v in perk_raw.items()}},
            "regrade_label": label_slopes,
            "computed_at": _now(),
        },
    )
    append_row(
        out,
        {
            "row_id": "478-flatness-null",
            "task": 478,
            "config_slug": "regrade_raw",
            "cosine_path_used": "RAW — same as 478-perK-slopes",
            "recoverability": "CPU (verified)",
            "family": fam["family"],
            "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": 20},
            "gate_level": "statistic (raw gap slope reproduces published -0.12 within "
            "rebanding tolerance; design-band raw gap slope reported alongside)",
            "alpha": alpha,
            "original_stat": {
                "estimator": "OLS slope of (far - near band gap) vs log2 K "
                "+ K x log(dist) interaction (published MixedLM +0.010, p=0.405)",
                "gap_slope": -0.12,
                "gap_slope_se": 0.052,
                "gap_slope_p": 0.148,
            },
            "recomputed_stat": {
                "raw_design_bands": gap_design,
                "raw_rebanded": gap_raw,
                "centered_rebanded": gap_mc,
                "interaction_raw_cluster_ols": inter_raw,
                "interaction_centered_cluster_ols": inter_mc,
                "holm_family": fam_p,
                "holm_reject": holm,
                "holm_reject_at_h2_alpha_001": holm_h2,
            },
            "band_reassignment": {
                "crosstab_design_vs_centered": crosstab,
                "n_personas_moved": n_moved,
                "method": "equal-count re-ranking of per-persona median min_dist "
                "(original band sizes preserved)",
            },
            "n": {"rows": len(df), "K_points": 4},
            "regrade_label": label_flat,
            "notes": "Rescue driver is the K x log(dist) INTERACTION under the centered "
            "metric (cluster-robust OLS beta +0.021, Holm-significant even at the plan-H2 "
            "alpha=0.01); the gap slope itself stays NS. ESTIMATOR CAVEAT: the same "
            "cluster-robust OLS on the RAW join gives p=0.022 (borderline at 0.05, NS at "
            "the H2 alpha=0.01) where the published co-primary MixedLM read +0.010, "
            "p=0.405 - so part of the movement is estimator-sensitivity, and the "
            "positive interaction direction (slope flattens as K grows) is a CANDIDATE "
            "rescue pending a MixedLM refit, exactly per the tree's candidate-rescue "
            "framing. Reported with family size 2, corrected + uncorrected p.",
            "computed_at": _now(),
        },
    )
    payload["fig_478"] = {
        "per_K_rho_raw": {k: v["spearman_rho"] for k, v in perk_raw.items()},
        "per_K_rho_mc": {k: v["spearman_rho"] for k, v in perk_mc.items()},
        "gap_raw": gap_raw,
        "gap_mc": gap_mc,
        "crosstab": crosstab,
        "band_order": band_order,
        "labels": {"slopes": label_slopes, "flatness": label_flat},
    }


def regrade_490(data_root: Path, out: Path, payload: dict) -> None:
    """#490 (null row): distance-adjusted on-axis regression
    gap_dosematched ~ is_on_axis + mean_d + asym, cluster-robust at (pair, seed).
    Y unchanged; mean_d/asym recomputed raw (gate vs persona_level.csv + published
    beta) then centered. is_on_axis labels are a design feature (kept)."""
    import pandas as pd

    fam = family_111bank(data_root)
    idx = {n: i for i, n in enumerate(fam["names"])}
    dist_raw = 1.0 - fam["cos_raw"]
    dist_mc = 1.0 - fam["cos_mc"]
    pl = pd.read_csv(data_root / "eval_results" / "issue_490" / "aggregate" / "persona_level.csv")
    pl = pl[pl["subpanel"].isin(["on_axis", "off_axis"])].copy()

    def _dists(row, D):
        i = idx[row["persona"]]
        dA, dB = D[i, idx[row["A"]]], D[i, idx[row["B"]]]
        return dA, dB

    devs = []
    for _, r in pl.iterrows():
        dA, dB = _dists(r, dist_raw)
        devs.append(max(abs(dA - r["d_A"]), abs(dB - r["d_B"])))
    max_dev = float(max(devs))
    if max_dev > GATE_MATRIX_TOL:
        raise RuntimeError(f"490 row-level d_A/d_B join gate FAILED: max dev {max_dev:.3e}")

    pub = json.loads(
        (data_root / "eval_results" / "issue_490" / "aggregate" / "regression.json").read_text()
    )["primary_q2_distance_adjusted_regression"]

    def _fit(D) -> dict:
        piv: dict[tuple, dict] = {}
        for _, r in pl.iterrows():
            key = (r["pair_id"], int(r["seed"]), r["persona"])
            if key not in piv:
                dA, dB = _dists(r, D)
                piv[key] = {
                    "is_on_axis": int(r["is_on_axis"]),
                    "mean_d": 0.5 * (dA + dB),
                    "asym": abs(dA - dB),
                    "cluster": f"{r['pair_id']}|seed{int(r['seed'])}",
                    "conds": {},
                }
            piv[key]["conds"][r["condition"]] = float(r["deltaLogP_mean"])
        rows = []
        for rec in piv.values():
            c = rec["conds"]
            need = [k for k in c if k.startswith("shared_2D")] and all(
                any(k.startswith(p) for k in c) for p in ("pooled_2D_A", "pooled_2D_B")
            )
            if not need:
                continue
            shared = next(c[k] for k in c if k.startswith("shared_2D"))
            pA = next(c[k] for k in c if k.startswith("pooled_2D_A"))
            pB = next(c[k] for k in c if k.startswith("pooled_2D_B"))
            rows.append(
                {
                    "y": shared - 0.5 * (pA + pB),
                    "is_on_axis": rec["is_on_axis"],
                    "mean_d": rec["mean_d"],
                    "asym": rec["asym"],
                    "cluster": rec["cluster"],
                }
            )
        d = pd.DataFrame(rows).dropna()
        X = d[["is_on_axis", "mean_d", "asym"]].to_numpy(dtype=float)
        fit = cluster_ols(d["y"].to_numpy(dtype=float), X, d["cluster"].to_numpy())
        return {
            "n_rows": len(d),
            "n_clusters": int(d["cluster"].nunique()),
            "is_on_axis_beta": float(fit.params[1]),
            "is_on_axis_p": float(fit.pvalues[1]),
            "is_on_axis_ci95": [float(v) for v in fit.conf_int()[1]],
            "mean_d_beta": float(fit.params[2]),
            "asym_beta": float(fit.params[3]),
        }

    fit_raw = _fit(dist_raw)
    fit_mc = _fit(dist_mc)
    if abs(fit_raw["is_on_axis_beta"] - pub["headline_beta"]) > 0.02:
        raise RuntimeError(
            f"490 statistic gate FAILED: raw refit is_on_axis beta "
            f"{fit_raw['is_on_axis_beta']:.4f} vs published {pub['headline_beta']:.4f}"
        )
    md_raw = np.array([0.5 * sum(_dists(r, dist_raw)) for _, r in pl.iterrows()])
    md_mc = np.array([0.5 * sum(_dists(r, dist_mc)) for _, r in pl.iterrows()])
    x_spear, _ = spearman(md_raw, md_mc)
    alpha = 0.05  # the task's own published criterion (CI/p vs 0.05)
    rescued = fit_mc["is_on_axis_p"] < alpha  # family = 1 cell (single headline test)
    append_row(
        out,
        {
            "row_id": "490-distance-adjusted",
            "task": 490,
            "config_slug": "regrade_raw",
            "cosine_path_used": "RAW — 111-bank distance JSON (issue490 specs via "
            "_issue478_common.load_cosine_distance_matrix@9b5821aab)",
            "recoverability": "CPU (verified)",
            "family": fam["family"],
            "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": 20},
            "gate_level": "matrix-1e-4 (bank) + row-level d_A/d_B 1e-4 + statistic "
            "(raw refit is_on_axis beta within 0.02 of published 0.1996)",
            "alpha": alpha,
            "multiplicity": "family = 1 cell (the single published headline test); "
            "Holm degenerate to raw p",
            "original_stat": {
                "estimator": "OLS gap_dosematched ~ is_on_axis + mean_d + asym, "
                "cluster-robust (pair, seed)",
                "is_on_axis_beta": pub["headline_beta"],
                "is_on_axis_p": pub["headline_p"],
                "is_on_axis_ci95": pub["headline_ci95"],
            },
            "recomputed_stat": {"raw": fit_raw, "centered": fit_mc},
            "raw_vs_centered_x_spearman": x_spear,
            "n": {"rows": fit_raw["n_rows"], "clusters": fit_raw["n_clusters"]},
            "regrade_label": label_null(rescued),
            "notes": "is_on_axis labels are the experiment's DESIGN (which personas were "
            "evaluated as on/off-axis); post-hoc re-selection cannot add personas with Y, "
            "so only the distance covariates change under centering.",
            "computed_at": _now(),
        },
    )
    payload["fig_490"] = {
        "raw": fit_raw,
        "centered": fit_mc,
        "published": {"beta": pub["headline_beta"], "p": pub["headline_p"]},
    }


def regrade_396_415(data_root: Path, out: Path, payload: dict) -> None:
    """#396/#415 (null rows, H3): 6 DV surfaces x {cos_to_assistant, cos_to_neutral}
    length-partial Spearman on the n24 bank. Raw recompute from the #274 bundle must
    reproduce the published headline rho within 0.02 (statistic gate); centered =
    bank-mean-center BOTH the persona vectors AND the reference vector.
    Multiplicity (pre-registered §3): Holm p<0.05 within the 12-cell family AND
    |rho|>=0.5 (N=24)."""
    fam = family_n24(data_root, layer=15)
    names24 = fam["names"]
    idx = {n: i for i, n in enumerate(names24)}
    preds = json.loads(
        (data_root / "eval_results" / "issue_415" / "base_model_predictors_v2.json").read_text()
    )
    summary_396 = json.loads(
        (data_root / "eval_results" / "issue_396" / "analysis_summary.json").read_text()
    )
    per_src = {row["source"]: row for row in summary_396["per_source_aggregation"]}
    pub_415 = json.loads(
        (data_root / "eval_results" / "issue_415" / "analysis_summary.json").read_text()
    )
    sources = sorted(s for s in per_src if s in preds["predictor_1_cosine_to_assistant_L15"])
    assert set(sources) <= set(names24), sorted(set(sources) - set(names24))
    from analyze_length_rate_n48 import get_inherited_prompt

    lengths = np.array([len(get_inherited_prompt(s)) for s in sources], dtype=float)
    surfaces = [
        "logp_end_of_response_diagonal_mean",
        "logp_at_k0_diagonal_mean",
        "logp_auc_diagonal_mean",
        "logp_max_diagonal_mean",
        "logp_mean_diagonal_mean",
        "substring_match_rate_diagonal_mean",
    ]
    surfaces = [s for s in surfaces if s in per_src[sources[0]]]
    refs = {"assistant": "helpful_assistant", "neutral": "qwen_default"}

    def _xvec(ref_name: str, M) -> np.ndarray:
        j = idx[refs[ref_name]]
        return np.array([M[idx[s], j] for s in sources], dtype=float)

    # Statistic join gate: bundle-raw cos-to-assistant must reproduce the
    # published headline rho (rho_partial +0.018 on logp_end_of_response).
    headline = "logp_end_of_response_diagonal_mean"
    y_head = np.array([per_src[s][headline] for s in sources], dtype=float)
    x_raw_asst = _xvec("assistant", fam["cos_raw"])
    rho_gate = length_partial_spearman(x_raw_asst, y_head, lengths)
    pub_head = pub_415["summary"]["Cosine-to-assistant (L15) \u00d7 headline"]
    gate_dev = abs(rho_gate - pub_head["rho_partial"])
    join_ok = gate_dev <= GATE_RHO_TOL
    # Stored-X reproduction (provenance): exact recompute from the published values.
    x_stored = np.array(
        [preds["predictor_1_cosine_to_assistant_L15"][s] for s in sources], dtype=float
    )
    rho_stored = length_partial_spearman(x_stored, y_head, lengths)
    cells = {}
    fam_p = {}
    for ref in refs:
        x_raw = _xvec(ref, fam["cos_raw"])
        x_mc = _xvec(ref, fam["cos_mc"])
        for surf in surfaces:
            y = np.array([per_src[s][surf] for s in sources], dtype=float)
            mask = np.isfinite(y)
            rho_r = length_partial_spearman(x_raw[mask], y[mask], lengths[mask])
            _, p_r = spearman(x_raw[mask], y[mask])
            rho_m = length_partial_spearman(x_mc[mask], y[mask], lengths[mask])
            _, p_m = spearman(x_mc[mask], y[mask])
            key = f"cos_to_{ref}|{surf}"
            cells[key] = {
                "rho_partial_raw": rho_r,
                "p_raw": p_r,
                "rho_partial_centered": rho_m,
                "p_centered": p_m,
                "n": int(mask.sum()),
            }
            fam_p[key] = p_m
    holm = holm_reject(fam_p, alpha=0.05)
    rescue_cells = [k for k in cells if holm[k] and abs(cells[k]["rho_partial_centered"]) >= 0.5]
    uncorrected_hits = [k for k in cells if cells[k]["p_centered"] < 0.01 and not holm[k]]
    rescued = bool(rescue_cells)
    # Concern 2: reference-centroid norm vs the bank's centered-norm distribution.
    ref_norm_stats = {}
    bank_norms = fam["centered_norms"]
    for ref, pname in refs.items():
        rn = float(bank_norms[idx[pname]])
        ref_norm_stats[ref] = {
            "ref_centered_norm": rn,
            "bank_centered_norm_median": float(np.median(bank_norms)),
            "ref_norm_percentile_in_bank": float((bank_norms < rn).mean()),
        }
    x_spear, _ = spearman(_xvec("assistant", fam["cos_raw"]), _xvec("assistant", fam["cos_mc"]))
    label = "join-failed" if not join_ok else label_null(rescued)
    for task_id in (396, 415):
        append_row(
            out,
            {
                "row_id": f"{task_id}-predictor-null",
                "task": task_id,
                "config_slug": "regrade_raw",
                "cosine_path_used": "RAW pairwise vs assistant/neutral "
                "(recompute_predictors_i415.py:242-245)",
                "recoverability": "CPU (verified; #274 bundle is a DIFFERENT extraction "
                "than #415's predictor run — statistic-level gate only)",
                "family": fam["family"],
                "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": 15},
                "gate_level": "statistic (bundle-raw cos-to-assistant length-partial rho "
                f"vs published headline: |drho| = {gate_dev:.4f} "
                f"{'<=' if join_ok else '>'} 0.02)",
                "alpha": 0.05,
                "multiplicity": "Holm p<0.05 within the 12-cell family AND |rho|>=0.5 "
                "(plan §3 pre-registered); p = raw Spearman p (the task's own machinery)",
                "original_stat": {
                    "estimator": "length-partial Spearman, 6 DV surfaces x 2 cos predictors",
                    "headline_rho_cos_to_assistant": pub_head["rho_partial"],
                    "headline_p": pub_head["p"],
                    "stored_X_reproduction_rho": rho_stored,
                },
                "recomputed_stat": {
                    "cells": cells,
                    "holm_reject": holm,
                    "rescue_cells": rescue_cells,
                    "uncorrected_hits_not_rescues": uncorrected_hits,
                },
                "reference_norm_check": ref_norm_stats,
                "raw_vs_centered_x_spearman": x_spear,
                "n": {"personas": len(sources), "cells": len(cells)},
                "regrade_label": label,
                "notes": "Same join for #396 and #415 (#415 corroborated #396 on the same "
                "24 personas + DV; the neutral reference maps to the bank's qwen_default "
                "persona, the assistant reference to helpful_assistant). Cells failing "
                "Holm but with uncorrected p<0.01 are candidate signals, never rescues.",
                "computed_at": _now(),
            },
        )
    payload["fig_396_415"] = {"cells": cells, "holm": holm, "label": label}


def regrade_505(data_root: Path, out: Path, payload: dict) -> None:
    """#505 (null row, H4): per-arm OLS(HC2) slope of delta_leakage vs cos(b, j)
    at L21 + sign-agreement vs the published 5/6 bar; pooled cluster-robust OLS
    stands in for the published (singular) mixed model."""
    import statsmodels.api as sm

    fam = family_505(data_root, layer=21)
    idx = {n: i for i, n in enumerate(fam["names"])}
    rows = json.loads(
        (
            data_root / "eval_results" / "issue_505" / "analysis" / "delta_leakage_per_seed.json"
        ).read_text()
    )["rows"]
    pub = json.loads(
        (data_root / "eval_results" / "issue_505" / "analysis" / "per_arm_slopes.json").read_text()
    )
    arms = sorted({r["j_i"] for r in rows})

    def _per_arm(M) -> dict:
        outd = {}
        for j in arms:
            sub = [r for r in rows if r["j_i"] == j]
            x = np.array([M[idx[r["b"]], idx[j]] for r in sub], dtype=float)
            y = np.array([float(r["delta_leakage"]) for r in sub], dtype=float)
            X = sm.add_constant(x)
            res = sm.OLS(y, X).fit(cov_type="HC2")
            rho, p_rho = spearman(x, y)
            ci = res.conf_int(alpha=0.05)
            outd[j] = {
                "beta_j": float(res.params[1]),
                "p": float(res.pvalues[1]),
                "ci95": [float(ci[1, 0]), float(ci[1, 1])],
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "n_rows": len(sub),
            }
        return outd

    pa_raw = _per_arm(fam["cos_raw"])
    pa_mc = _per_arm(fam["cos_mc"])
    for j in arms:
        if abs(pa_raw[j]["beta_j"] - pub["per_arm"][j]["beta_j"]) > 1e-4:
            raise RuntimeError(
                f"505 per-arm statistic gate FAILED for arm {j}: raw refit "
                f"{pa_raw[j]['beta_j']:.6f} vs published {pub['per_arm'][j]['beta_j']:.6f}"
            )

    def _pooled(M) -> dict:
        x = np.array([M[idx[r["b"]], idx[r["j_i"]]] for r in rows], dtype=float)
        y = np.array([float(r["delta_leakage"]) for r in rows], dtype=float)
        cl = np.array([f"{r['j_i']}|s{r['seed']}" for r in rows])
        fit = cluster_ols(y, x.reshape(-1, 1), cl)
        return {"beta": float(fit.params[1]), "p": float(fit.pvalues[1]), "n": len(y)}

    pooled_raw = _pooled(fam["cos_raw"])
    pooled_mc = _pooled(fam["cos_mc"])
    sign_raw = sum(1 for j in arms if pa_raw[j]["beta_j"] > 0)
    sign_mc = sum(1 for j in arms if pa_mc[j]["beta_j"] > 0)
    # Published success bar: pooled slope positive Holm p<0.05 OR >=5/6 arms positive.
    rescued = (pooled_mc["beta"] > 0 and pooled_mc["p"] < 0.05) or sign_mc >= 5
    x_all_raw = np.array([fam["cos_raw"][idx[r["b"]], idx[r["j_i"]]] for r in rows])
    x_all_mc = np.array([fam["cos_mc"][idx[r["b"]], idx[r["j_i"]]] for r in rows])
    x_spear, _ = spearman(x_all_raw, x_all_mc)
    append_row(
        out,
        {
            "row_id": "505-loo-null",
            "task": 505,
            "config_slug": "regrade_raw",
            "cosine_path_used": "RAW explicit — build_pv_centroids.py:196 "
            "(compute_cosine_matrix(c, centering='none'))",
            "recoverability": "CPU (verified; HF geometry bundle)",
            "family": fam["family"],
            "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": 21},
            "gate_level": "matrix-1e-4 (bundle vs panel_similarity_matrix L21) + statistic "
            "(per-arm raw OLS(HC2) betas reproduce published to 1e-4)",
            "alpha": 0.05,
            "multiplicity": "the task's own published success bar: pooled slope positive "
            "p<0.05 OR sign-agreement >=5/6 (binomial p<=0.11)",
            "original_stat": {
                "estimator": "per-arm OLS(HC2) slope of delta_leakage vs cos(b, j) at L21",
                "per_arm_beta": {j: pub["per_arm"][j]["beta_j"] for j in arms},
                "sign_agreement": f"{sign_raw}/6 positive (published 2/6)",
                "pooled_mixed_model": "failed singular at every layer (published)",
            },
            "recomputed_stat": {
                "raw": {"per_arm": pa_raw, "pooled_cluster_ols": pooled_raw},
                "centered": {
                    "per_arm": pa_mc,
                    "pooled_cluster_ols": pooled_mc,
                    "sign_agreement": f"{sign_mc}/6 positive",
                },
            },
            "raw_vs_centered_x_spearman": x_spear,
            "n": {"rows": len(rows), "arms": len(arms)},
            "regrade_label": label_null(rescued),
            "notes": "Pooled cluster-robust OLS stands in for the published mixed model "
            "(which failed singular); the per-arm + sign-agreement read is the task's own "
            "published verdict machinery.",
            "computed_at": _now(),
        },
    )
    payload["fig_505"] = {
        "per_arm_raw": {j: pa_raw[j]["beta_j"] for j in arms},
        "per_arm_mc": {j: pa_mc[j]["beta_j"] for j in arms},
        "pooled": {"raw": pooled_raw, "centered": pooled_mc},
    }
    payload["bank_offdiag"]["issue505_pv_L21"] = {
        "raw": off_diag(fam["cos_raw"]).tolist(),
        "centered": off_diag(fam["cos_mc"]).tolist(),
    }


def regrade_474_lineage(data_root: Path, out: Path, payload: dict) -> None:
    """#474 (+#406/#460 lineage), matrix-only sensitivity rows: re-run the #474
    cosine-followup cells (length-partial rho of cos(L21) vs delta_g / trained
    log-prob, non-stylized panel) under the approximate Gram-read centering of
    C_L21. Labels live in the sensitivity-* namespace ONLY (plan §3)."""
    fam = family_406_gram(data_root)
    conds = fam["names"]
    cidx = {c: i for i, c in enumerate(conds)}
    D = json.loads(
        (data_root / "eval_results" / "issue_406" / "divergence" / "D_matrix.json").read_text()
    )
    PT = D["prompt_tokens"]
    pub = json.loads(
        (data_root / "eval_results" / "issue_474" / "followup_cosine_analysis.json").read_text()
    )
    sty = {"A3", "A4", "A5"}
    pairs_ns = [(a, b) for a in conds for b in conds if a != b and a not in sty and b not in sty]
    layer = 21
    G_raw = fam["per_layer"][layer]["cos_raw"]
    G_mc = fam["per_layer"][layer]["cos_mc_approx"]
    clipped = fam["per_layer"][layer]["clipped_mass_frac"]
    unreliable = clipped > PSD_CLIP_TRACE_FRAC

    def _cells(arm: str, ep: int) -> dict | None:
        gp = (
            data_root
            / "eval_results"
            / "issue_474"
            / "cross_eval"
            / f"{arm}_ep{ep}"
            / "G_logprob_matrix.json"
        )
        if not gp.exists():
            return None
        G = json.loads(gp.read_text())["G"]
        dg = np.array([G[a][b]["delta_g"] for a, b in pairs_ns])
        g = np.array([G[a][b]["g_logprob"] for a, b in pairs_ns])
        ln = np.array([np.log(PT[a][b]) for a, b in pairs_ns])
        cos_r = np.array([G_raw[cidx[a], cidx[b]] for a, b in pairs_ns])
        cos_m = np.array([G_mc[cidx[a], cidx[b]] for a, b in pairs_ns])
        rho_r_dg, p_r_dg = rank_residual_partial(cos_r, dg, ln)
        rho_m_dg, p_m_dg = rank_residual_partial(cos_m, dg, ln)
        rho_r_g, p_r_g = rank_residual_partial(cos_r, g, ln)
        rho_m_g, p_m_g = rank_residual_partial(cos_m, g, ln)
        key = f"{arm}_ep{ep}"
        pub_row = pub.get(key, {})
        gate_dev = (
            abs(rho_r_dg - pub_row["ns_rho_cosL21_deltag"])
            if "ns_rho_cosL21_deltag" in pub_row
            else None
        )
        return {
            "cell": key,
            "published_ns_rho_cosL21_deltag": pub_row.get("ns_rho_cosL21_deltag"),
            "raw": {
                "rho_cos_deltag": rho_r_dg,
                "p_cos_deltag": p_r_dg,
                "rho_cos_trainedlogp": rho_r_g,
                "p_cos_trainedlogp": p_r_g,
            },
            "centered_approx": {
                "rho_cos_deltag": rho_m_dg,
                "p_cos_deltag": p_m_dg,
                "rho_cos_trainedlogp": rho_m_g,
                "p_cos_trainedlogp": p_m_g,
            },
            "gate_abs_dev_vs_published": gate_dev,
            "n_pairs": len(pairs_ns),
        }

    cells = []
    for arm in ("pos", "loc"):
        for ep in (1, 2, 3, 5):
            c = _cells(arm, ep)
            if c is not None:
                cells.append(c)
    bad_gate = [
        c["cell"]
        for c in cells
        if c["gate_abs_dev_vs_published"] is not None
        and c["gate_abs_dev_vs_published"] > GATE_RHO_TOL
    ]
    if bad_gate:
        raise RuntimeError(f"474 raw-reproduction gate FAILED for cells: {bad_gate}")
    alpha = 0.01

    def _agree(c: dict) -> bool:
        r, m = c["raw"], c["centered_approx"]
        same_sign = math.copysign(1, r["rho_cos_deltag"]) == math.copysign(1, m["rho_cos_deltag"])
        same_sig = (r["p_cos_deltag"] < alpha) == (m["p_cos_deltag"] < alpha)
        return same_sign and same_sig

    agrees = {c["cell"]: _agree(c) for c in cells}
    if unreliable:
        label = "sensitivity-unreliable"
    elif all(agrees.values()):
        label = "sensitivity-agrees"
    else:
        label = "sensitivity-disagrees"
    x_spear, _ = spearman(off_diag(G_raw), off_diag(G_mc))
    fingerprints = {
        str(layer_i): {
            "raw": fam["per_layer"][layer_i]["fingerprint_raw"],
            "centered_approx": fam["per_layer"][layer_i]["fingerprint_mc"],
            "clipped_mass_frac": fam["per_layer"][layer_i]["clipped_mass_frac"],
        }
        for layer_i in fam["per_layer"]
    }
    lineage_notes = {
        474: "Primary lineage row: the #474 cosine-followup cells recomputed directly.",
        406: "Producer row: scripts/i406_phase1_merge_and_compute_matrices.py (DELETED; "
        "commit 9e6e31c3f) persisted distance-form 1-cos, normalize-only (RAW). "
        "Sensitivity read inherits from the 474 cells (same matrices).",
        460: "Lineage row: #460 consumed the same C_L* matrices via the #406 rig; "
        "sensitivity read inherits from the 474 cells.",
    }
    for task_id in (474, 406, 460):
        append_row(
            out,
            {
                "row_id": f"{task_id}-gram-sensitivity",
                "task": task_id,
                "config_slug": "approx_gram",
                "cosine_path_used": "RAW — i406 producer (normalize-only, persisted as "
                "distance 1-cos); consumed by i474_cosine_followup.py:52 + i460_phase5",
                "recoverability": "matrix-only (centroid bundles ruled out at fact-check; "
                "Gram read after G = 1 - D conversion)",
                "family": fam["family"],
                "bank": {"n": fam["n"], "names_hash": fam["names_hash"], "layer": layer},
                "gate_level": "statistic (raw Gram-path rho reproduces the published "
                "followup table per cell, |drho|<=0.02) — matrix-only namespace",
                "alpha": alpha,
                "original_stat": {
                    "estimator": "length-partial Spearman cos(L21) vs delta_g / trained "
                    "log-prob, non-stylized panel, per (arm, epoch)",
                    "published_table": "eval_results/issue_474/followup_cosine_analysis.json",
                },
                "recomputed_stat": {"cells": cells, "agreement_per_cell": agrees},
                "per_layer_fingerprints": fingerprints,
                "psd_clipped_mass_frac_L21": clipped,
                "raw_vs_centered_x_spearman": x_spear,
                "n": {"pairs_non_stylized": len(pairs_ns), "cells": len(cells)},
                "regrade_label": label,
                "notes": lineage_notes[task_id]
                + " NEVER a canonical label — heterogeneous centroid norms are lost in "
                "the persisted matrix; exact canonical recompute needs GPU re-extraction "
                "(deferred follow-up).",
                "computed_at": _now(),
            },
        )
    payload["fig_474"] = {
        "cells": cells,
        "label": label,
        "offdiag_L21": {"raw": off_diag(G_raw).tolist(), "centered": off_diag(G_mc).tolist()},
    }


def regrade_341(data_root: Path, out: Path, payload: dict) -> None:
    """#341 (matrix-only sensitivity): cos<->JS alignment rho (L20 headline 0.939,
    n=171 non-anchor pairs) recomputed from the Gram read of cosine_matrix.json.
    JS side fixed (snapshot from 4ddf33d6); only the cosine side is re-read."""
    fam = family_341_gram()
    names = fam["names"]
    js_payload = json.loads((SNAP_DIR / "i341_js_matrix_4ddf33d6.json").read_text())
    js_names = js_payload["persona_names"]
    JS = np.asarray(js_payload["matrices"]["Tfull"], dtype=np.float64)
    pub = json.loads((SNAP_DIR / "i341_geometry_alignment_4ddf33d6.json").read_text())
    keep = [n for n in names if n != "no_persona"]
    ki = [names.index(n) for n in keep]
    kj = [js_names.index(n) for n in keep]
    iu = np.triu_indices(len(keep), k=1)
    js_sub = JS[np.ix_(kj, kj)][iu]
    per_layer = {}
    for layer, blob in fam["per_layer"].items():
        d_raw = (1.0 - blob["cos_raw"][np.ix_(ki, ki)])[iu]
        d_mc = (1.0 - blob["cos_mc_approx"][np.ix_(ki, ki)])[iu]
        rho_r, p_r = spearman(d_raw, js_sub)
        rho_m, p_m = spearman(d_mc, js_sub)
        pub_layer = pub["layers"].get(str(layer), {})
        per_layer[str(layer)] = {
            "published_rho_raw": pub_layer.get("rho_raw"),
            "recomputed_rho_raw": rho_r,
            "p_raw": p_r,
            "rho_centered_approx": rho_m,
            "p_centered_approx": p_m,
            "clipped_mass_frac": blob["clipped_mass_frac"],
            "gate_abs_dev": abs(rho_r - pub_layer["rho_raw"]) if pub_layer else None,
        }
    bad = {
        layer: v["gate_abs_dev"]
        for layer, v in per_layer.items()
        if v["gate_abs_dev"] is not None and v["gate_abs_dev"] > GATE_RHO_TOL
    }
    if bad:
        raise RuntimeError(f"341 raw-reproduction gate FAILED at layers: {bad}")
    alpha = 0.001  # the task's own Mantel gate threshold for the headline
    l20 = per_layer["20"]
    unreliable = l20["clipped_mass_frac"] > PSD_CLIP_TRACE_FRAC
    same_sign = math.copysign(1, l20["rho_centered_approx"]) == math.copysign(
        1, l20["recomputed_rho_raw"]
    )
    same_sig = (l20["p_centered_approx"] < alpha) == (l20["p_raw"] < alpha)
    if unreliable:
        label = "sensitivity-unreliable"
    elif same_sign and same_sig:
        label = "sensitivity-agrees"
    else:
        label = "sensitivity-disagrees"
    blob20 = fam["per_layer"][20]
    x_spear, _ = spearman(off_diag(blob20["cos_raw"]), off_diag(blob20["cos_mc_approx"]))
    append_row(
        out,
        {
            "row_id": "341-cos-js-alignment",
            "task": 341,
            "config_slug": "approx_gram",
            "cosine_path_used": "RAW — extract_persona_vectors.py:198-199 (normalize-only; "
            "reclassified to the raw line at fact-check)",
            "recoverability": "matrix-only at minimum (true similarity Gram persists; "
            "vector bundle not persisted in-repo)",
            "family": fam["family"],
            "bank": {"n": len(keep), "names_hash": _names_hash(keep), "layer": 20},
            "gate_level": "statistic (recomputed raw rho reproduces published per layer, "
            "|drho|<=0.02) — matrix-only namespace",
            "alpha": alpha,
            "original_stat": {
                "estimator": "Spearman rho of triu(1 - cos) vs triu(JS Tfull), n=171 "
                "non-anchor pairs; headline L20 rho_raw = 0.9399",
                "published_per_layer_rho": {
                    layer: v["published_rho_raw"] for layer, v in per_layer.items()
                },
            },
            "recomputed_stat": {"per_layer": per_layer},
            "raw_vs_centered_x_spearman": x_spear,
            "n": {"pairs": len(js_sub)},
            "regrade_label": label,
            "notes": "The cos<->JS alignment claim itself was computed on the raw matrix "
            "(fingerprint-identical to the degenerate band), so #341 no longer grounds the "
            "CENTERED recipe; this row asks whether the alignment survives the approximate "
            "centered read. Sensitivity namespace only.",
            "computed_at": _now(),
        },
    )
    payload["fig_341"] = {"per_layer": per_layer, "label": label}


def partition_213_227(data_root: Path, out: Path, payload: dict) -> None:
    """#213/#227 partition rows: no centroid bundle persists AND the persisted
    JSON is a cue->no_cue reference-column distance file (NOT a Gram matrix), so
    neither exact recompute nor the approximate Gram read is possible."""
    p = data_root / "eval_results" / "issue_213" / "part_a" / "cosine_matrices.json"
    d = json.loads(p.read_text())
    schema_note = (
        f"persisted file is {{layer: {{model: {{cue: distance-to-no_cue}}}}}} over "
        f"models={d.get('models')} cues={d.get('cues')} — a single reference column, "
        "no Gram matrix (fact-check verified)"
    )
    for task_id, note in (
        (213, "Producer: scripts/run_issue_213_part_a.py:547 (raw pairwise cosine)."),
        (227, "Consumer of the #213 part-A geometry."),
    ):
        append_row(
            out,
            {
                "row_id": f"{task_id}-partition",
                "task": task_id,
                "config_slug": "approx_gram",
                "cosine_path_used": "RAW — run_issue_213_part_a.py:547",
                "recoverability": "needs-GPU-re-extraction / unrecoverable from disk",
                "gate_level": "n/a (no join possible)",
                "original_stat": {
                    "estimator": "cue->no_cue centroid cosine distances per layer/model",
                    "schema": schema_note,
                },
                "recomputed_stat": None,
                "regrade_label": "join-failed (needs-GPU-re-extraction)",
                "notes": note + " WandB-artifact re-search at implementation time: the "
                "#213-era runs predate systematic artifact logging; no centroid artifact "
                "was located under the project's WandB projects (and none is named in the "
                "#213/#227 bodies). Follow-up pointer: single eval-intent pod, ~1-2 GPU-h, "
                "base-model centroid re-extraction for the cue-variant bank, then rerun "
                "this driver's adapter.",
                "computed_at": _now(),
            },
        )


def readoff_472_504(data_root: Path, out: Path, payload: dict) -> None:
    """#472/#504 read-off rows (no new compute): the r6 artifacts already carry
    both raw and mean-centered matrices; record the spread comparison."""
    rep = json.loads(
        (
            data_root / "eval_results" / "issue_504" / "round6_mean_centered_cos_to_villain.json"
        ).read_text()
    )
    spread = {
        str(layer["layer"]): {
            "raw": layer["raw"],
            "mean_centered": layer["mean_centered"],
            "n_personas": layer["n_personas"],
        }
        for layer in rep["per_layer"]
    }
    for task_id, note in (
        (
            472,
            "Bundles under issue472_neg_geometry/geometry carry cos_matrix + "
            "cos_matrix_mean_centered since #504 r6.",
        ),
        (
            504,
            "#504's Gate-A degeneracy finding IS the raw-vs-centered contrast; rounds 1-5 "
            "raw, round 6 remediated.",
        ),
    ):
        append_row(
            out,
            {
                "row_id": f"{task_id}-readoff-r6",
                "task": task_id,
                "config_slug": "readoff_r6",
                "cosine_path_used": "raw rounds 1-5; both matrices persisted post-r6",
                "recoverability": "CPU (done — r6 artifacts)",
                "gate_level": "read-off (no new compute; plan §5)",
                "original_stat": {
                    "estimator": "cos-to-source spread (villain), per layer",
                },
                "recomputed_stat": {"r6_spread_report": spread},
                "regrade_label": "stands (already-remediated; r6 read-off)",
                "notes": note,
                "computed_at": _now(),
            },
        )


def scope_pairwise(data_root: Path, out: Path, payload: dict) -> None:
    """2-vector pairwise family (#404/#458/#444/#493/#502): labeled, never
    re-graded — no bank exists, so no canonical recompute is defined (pin ¶2)."""
    sites = {
        404: "scripts/issue404_predictor_cossim.py:136-148 (per-layer mean pairwise cos)",
        458: "scripts/issue404_predictor_cossim.py (shared rig; #458 line)",
        444: "scripts/issue444_persona_distance_topic.py:101-110 (cos vs reference)",
        493: "scripts/issue493_extraction_metric_bakeoff.py (pairwise predictor arm)",
        502: "scripts/issue502_cpu_smoke.py (batched mirror of the #493 serial path)",
    }
    for task_id, site in sites.items():
        append_row(
            out,
            {
                "row_id": f"{task_id}-pairwise-scope",
                "task": task_id,
                "config_slug": "scope_pairwise",
                "cosine_path_used": f"RAW 2-vector pairwise (no bank) — {site}",
                "recoverability": "n/a (no bank => no canonical recompute exists)",
                "gate_level": "n/a",
                "original_stat": None,
                "recomputed_stat": None,
                "regrade_label": "labeled: raw pairwise (uncentered) — not re-graded",
                "notes": "Centering a 2-element bank degenerates to cos = -1; the pin "
                "defines this family as raw-pairwise, MUST be labeled, never numerically "
                "compared to bank-cosine values. Optional reference-bank-centered "
                "sensitivity check is a named follow-up, not run here (plan §4-A).",
                "computed_at": _now(),
            },
        )


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────
TASK_ADAPTERS = {
    "66": verify_66,
    "99": row_99,
    "405": regrade_405,
    "478": regrade_478,
    "490": regrade_490,
    "396_415": regrade_396_415,
    "505": regrade_505,
    "474_lineage": regrade_474_lineage,
    "341": regrade_341,
    "213_227": partition_213_227,
    "472_504": readoff_472_504,
    "pairwise": scope_pairwise,
}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Task #536 recompute driver (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO,
        help="Checkout holding the untracked input tensors (default: this repo root).",
    )
    ap.add_argument(
        "--only",
        default=None,
        help=f"Run a single adapter; one of {sorted(TASK_ADAPTERS)}.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "regrade_table.json",
        help="Regrade table path (rows appended per adapter).",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=recompute] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    keys = [args.only] if args.only else list(TASK_ADAPTERS)
    payload: dict = {"bank_offdiag": {}}
    payload_path = OUT_DIR / "figures_payload.json"
    if payload_path.exists():
        payload.update(json.loads(payload_path.read_text()))
    for k in keys:
        log.info("[adapter] %s", k)
        TASK_ADAPTERS[k](args.data_root, args.out, payload)
        payload["generated_at"] = _now()
        payload["git_commit"] = _git_sha()
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, default=float))
        log.info("[checkpoint] payload + table updated after %s", k)
    log.info("[done] %d adapters -> %s", len(keys), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
