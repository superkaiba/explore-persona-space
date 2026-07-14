#!/usr/bin/env python3
"""Issue #1092 read-4c repair: per-row trait-per-factor projections (free-analysis).

Replaces the degenerate banked read-4c statistic: ``issue1092_fit_grid.py:1387``
projects the row-MEAN of mean-centered ANOVA factor outputs onto r_B, so the
observed statistic is identically 0 (verified 7.4e-15..4.6e-13 across all 288
band rows) while its sign-flip nulls are real-magnitude. The repaired read
computes PER-ROW r_B projections of the f/g/i ANOVA factor outputs of the
FITTED context-map out-of-fold predictions over the dense core, per
(cell, layer 14, fit arm A, ambient basis, trait):

(a) trait-direction variance share per factor -- var of the r_B-projected
    factor output over var of the r_B-projected total -- against a 200-draw
    random-direction (rotation) null. A sign-flip null is variance-invariant
    on centered projections (the same degeneracy class being repaired), so the
    rotation null is the informative null for a variance statistic.
(b) projection--judge-score pearson per factor on the dense-core scored subset
    (P5 graded scores; unjudged rows dropped; n reported) against a 200-draw
    same-selection permutation of the row->score pairing.

Key economy (why no full 10752-dim ambient refit is needed): PRESS ridge is
LINEAR and the banked per-fold lambda indices are reused, so the r_B
projection of the fitted map's OOF prediction equals the OOF ridge prediction
of the r_B-projected SCALAR target at the same lambda. Correctness is pinned
two ways, both on the live production solver:
  1. an entry synthetic selftest asserts scalar-path == press_fit_predict
     multi-output + projection at <=1e-8 (runs on every invocation);
  2. the production gate asserts the reconstructed total projection reproduces
     the banked B1_map_mediated per-example pearson per estimable trait.

Reuses issue1092_fit_grid helpers READ-ONLY by import (fit_grid.py is
byte-pinned) and the P6 wrapper's scoped Hub staging (list_repo_tree +
per-file hf_hub_download, content-derived mtimes; never snapshot_download).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env must bind BEFORE the heavy imports below — the
# BLAS/torch pools freeze at import time (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue923_fit_decomposition import press_fit_predict  # noqa: E402
from issue1092_fit_grid import (  # noqa: E402
    DEFAULT_RB_REV,
    HF_DATA_REPO,
    RIDGE_LAMBDAS,
    _factor_components_dense_core,
    _folds_from_manifest,
    _jsonl,
    _load_judge_score_rows,
    _load_rb_directions,
    _pearson_or_nan,
)
from issue1092_p6_run import HfHubIO, _set_content_mtime  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

LAYER = 14  # frozen headline fit layer (plan section 6; #813/#779 continuity)
CELLS_DEFAULT = ("cell_inst_own", "cell_pre_own")
ARM = "context_end"  # the informative arm: the prefix-arm map is constant within
# prefix over the shared-query dense core, so its g/i factor outputs vanish BY
# DESIGN (plan section 4.5: M = f + const is an entailed rig identity) -- the
# trait-per-factor question is only non-degenerate on the context map.
FIT_ARM = "A"
FACTORS = ("f", "g", "i")
SUMMARIES_PREFIX = "issue1092_realistic_crossing/analysis_tensors/summaries"


def _log(msg: str) -> None:
    print(f"[read4c-repair] {msg}", flush=True)


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── scalar OOF ridge at banked per-fold lambdas (the live production solver) ──


def solve_oof_scalar(
    X_arm: np.ndarray,
    Z_arm: np.ndarray,
    folds: list[np.ndarray],
    lambda_indices: list[int],
) -> np.ndarray:
    """Out-of-fold ridge predictions of scalar target columns at FIXED lambdas.

    Replicates press_fit_predict(standardize=True) exactly for a known lambda
    per fold: per-fold train-only mu/sd (ddof=0, +1e-9) standardization with
    the degenerate-dim drop, train-mean-centered targets, primal ridge solve
    via Cholesky of (Xn^T Xn + lam I) -- algebraically identical to the
    PressRidge SVD dual form. Grams are assembled from per-fold test-block
    Grams of the globally-centered design (S_all = sum of fold blocks, exact),
    so total Gram work is ONE full-Gram equivalent per cell.

    X_arm (n, d) fp64 design rows (arm subset); Z_arm (n, P) fp64 scalar
    targets; folds index into rows; lambda_indices per fold into
    RIDGE_LAMBDAS. Returns (n, P) OOF predictions (uncentered: + train ymu).
    """
    assert X_arm.ndim == 2 and Z_arm.ndim == 2, (X_arm.shape, Z_arm.shape)
    assert X_arm.shape[0] == Z_arm.shape[0], (X_arm.shape, Z_arm.shape)
    assert len(folds) == len(lambda_indices), (len(folds), len(lambda_indices))
    n, d = X_arm.shape
    covered = np.concatenate(folds)
    assert covered.size == n and np.unique(covered).size == n, "folds must partition rows"
    X = torch.from_numpy(np.ascontiguousarray(X_arm)).double()
    Z = torch.from_numpy(np.ascontiguousarray(Z_arm)).double()
    # Global centering is a pure numerical-precision shift (per-fold means are
    # re-subtracted below); it keeps the Gram downdate free of catastrophic
    # cancellation on large-mean activation dims.
    X_c = X - X.mean(dim=0, keepdim=True)
    fold_grams: list[torch.Tensor] = []
    fold_xz: list[torch.Tensor] = []
    fold_xsum: list[torch.Tensor] = []
    fold_zsum: list[torch.Tensor] = []
    for f_i, te in enumerate(folds):
        t0 = time.monotonic()
        te_t = torch.from_numpy(np.ascontiguousarray(te)).long()
        Xte = X_c.index_select(0, te_t)
        Zte = Z.index_select(0, te_t)
        fold_grams.append(Xte.T @ Xte)
        fold_xz.append(Xte.T @ Zte)
        fold_xsum.append(Xte.sum(dim=0))
        fold_zsum.append(Zte.sum(dim=0))
        _log(f"  fold {f_i}: test-block gram n_te={te.size} ({time.monotonic() - t0:.1f}s)")
    S_all = torch.stack(fold_grams).sum(dim=0)
    XZ_all = torch.stack(fold_xz).sum(dim=0)
    xsum_all = torch.stack(fold_xsum).sum(dim=0)
    zsum_all = torch.stack(fold_zsum).sum(dim=0)
    pred = torch.empty_like(Z)
    for f_i, te in enumerate(folds):
        t0 = time.monotonic()
        te_t = torch.from_numpy(np.ascontiguousarray(te)).long()
        m_tr = n - te.size
        mu = (xsum_all - fold_xsum[f_i]) / m_tr  # train mean of X_c
        zbar = (zsum_all - fold_zsum[f_i]) / m_tr  # train mean of targets
        C_tr = (S_all - fold_grams[f_i]) - m_tr * torch.outer(mu, mu)
        var = torch.clamp(torch.diagonal(C_tr) / m_tr, min=0.0)
        sd = torch.sqrt(var) + 1e-9  # press_fit_predict convention (ddof=0, +1e-9)
        keep = sd > (sd.max() * 1e-6 + 1e-12)  # degenerate-dim drop (section 8)
        sd_k = sd[keep]
        G_std = C_tr[keep][:, keep] / torch.outer(sd_k, sd_k)
        # b = Xn_tr^T (z_tr - zbar) = ((X_c_tr^T z_tr) - m_tr mu zbar) / sd
        b = ((XZ_all - fold_xz[f_i]) - torch.outer(m_tr * mu, zbar))[keep] / sd_k.unsqueeze(1)
        lam = float(RIDGE_LAMBDAS[lambda_indices[f_i]])
        A = G_std + lam * torch.eye(G_std.shape[0], dtype=torch.float64)
        L = torch.linalg.cholesky(A)
        w = torch.cholesky_solve(b, L)  # (d_keep, P)
        Xn_te = ((X_c.index_select(0, te_t) - mu) / sd)[:, keep]
        pred[te_t] = Xn_te @ w + zbar
        _log(
            f"  fold {f_i}: solve lam={lam} keep={int(keep.sum())}/{d} "
            f"({time.monotonic() - t0:.1f}s)"
        )
    return pred.numpy()


def run_selftest() -> dict:
    """Exactness gate on the LIVE solver: scalar path == press_fit_predict + projection.

    Synthetic stacked-target design (t1/t2/t3 blocks) with a crossed dense
    core; reference = per-fold multi-output press_fit_predict (the exact P6
    engine) projected onto an embedded r_B; candidate = solve_oof_scalar at
    the reference's selected lambdas on the pre-projected scalar targets.
    Asserts (i) OOF projections match <=1e-8 and (ii) the factor
    decomposition commutes with the projection <=1e-8.
    """
    rng = np.random.default_rng(0)
    n_pre, n_q, hid, d_in = 12, 6, 5, 7
    rows = []
    for p in range(n_pre):
        for q in range(n_q):
            rows.append(
                {
                    "prefix_id": f"p{p:02d}",
                    "query_id": f"q{q:02d}",
                    "stratum": "dense_core",
                    "row_id": f"r{p:02d}{q:02d}",
                }
            )
    n = len(rows)
    X = rng.normal(size=(n, d_in))
    Y = np.concatenate(
        [X @ rng.normal(size=(d_in, hid)) + 0.3 * rng.normal(size=(n, hid)) for _ in range(3)],
        axis=1,
    )
    rb = rng.normal(size=hid)
    rb_out = np.zeros(3 * hid)
    rb_out[:hid] = rb  # t1-block embedding (projection_block_index 0)
    folds = _folds_from_manifest(rows, n, group_key="prefix_id", n_folds=3)
    ref_pred = np.zeros_like(Y)
    lam_indices: list[int] = []
    for te in folds:
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        res = press_fit_predict(
            torch.from_numpy(X[mask]).double(),
            torch.from_numpy(Y[mask]).double(),
            torch.from_numpy(X[te]).double(),
            standardize=True,
        )
        ref_pred[te] = res["pred"].numpy()
        lam_indices.append(int(res["lam_idx"]))
    ref_proj = ref_pred @ rb_out
    cand = solve_oof_scalar(X, (Y @ rb_out)[:, None], folds, lam_indices)[:, 0]
    d_proj = float(np.max(np.abs(cand - ref_proj)))
    assert d_proj <= 1e-8, f"scalar-path exactness failed: max|dproj|={d_proj}"
    fac_ref = _factor_components_dense_core(rows, ref_pred)
    fac_cand = _factor_components_dense_core(rows, cand[:, None])
    d_fac = 0.0
    for name in FACTORS:
        d_fac = max(
            d_fac,
            float(
                np.max(
                    np.abs(np.asarray(fac_ref[name]) @ rb_out - np.asarray(fac_cand[name])[:, 0])
                )
            ),
        )
    assert d_fac <= 1e-8, f"factor-projection commutation failed: max|dfac|={d_fac}"
    return {"max_abs_dproj": d_proj, "max_abs_dfactor": d_fac, "lam_indices": lam_indices}


# ── staging ───────────────────────────────────────────────────────────────────


def stage_summaries(
    staging_dir: Path, cells: list[str], revision: str
) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """Ensure context_end/t1 L14 shards are staged locally; return paths + provenance.

    Reuses HfHubIO (scoped list_repo_tree + per-file hf_hub_download,
    local_dir direct-to-target) and content-derived mtimes. A staged file is
    reused when its size matches the Hub listing; else re-downloaded.
    """
    io = HfHubIO(HF_DATA_REPO, revision)
    resolved = io.resolved_revision()
    staged: dict[str, dict[str, Path]] = {}
    prov: dict[str, Any] = {"repo": HF_DATA_REPO, "revision": revision, "resolved": resolved}
    files_prov: list[dict] = []
    for cell in cells:
        listing = io.list_files(f"{SUMMARIES_PREFIX}/{cell}")
        by_name = {f.path.split("/")[-1]: f for f in listing}
        staged[cell] = {}
        for kind in (ARM, "t1"):
            name = f"{kind}_L{LAYER:02d}.npy"
            if name not in by_name:
                shard_names = sorted(k for k in by_name if k.startswith(f"{kind}_L{LAYER:02d}"))
                raise FileNotFoundError(
                    f"{cell}/{name} absent on Hub (shards seen: {shard_names[:4]})"
                )
            hub_file = by_name[name]
            target = staging_dir / cell / name
            if not (target.exists() and target.stat().st_size == hub_file.size):
                _log(f"staging {hub_file.path} ({hub_file.size / 1e6:.0f} MB)")
                io.download_to(hub_file.path, target)
                if hub_file.hub_identity:
                    _set_content_mtime(target, hub_file.hub_identity)
            else:
                _log(f"reusing staged {target}")
            staged[cell][kind] = target
            files_prov.append(
                {"path": hub_file.path, "size": hub_file.size, "sha256": hub_file.hub_identity}
            )
    prov["files"] = files_prov
    return staged, prov


def load_ckpt(ckpt_dir: Path, cell: str) -> tuple[dict, Path]:
    """Load the banked ambient fitA L14 context_end unit checkpoint (exactly one)."""
    pattern = f"{cell}_{ARM}_fit{FIT_ARM}_L{LAYER:02d}_ambient_*.json"
    hits = sorted(ckpt_dir.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly 1 ckpt for {pattern}, got {len(hits)}")
    return json.loads(hits[0].read_text()), hits[0]


# ── statistics ────────────────────────────────────────────────────────────────


def variance_share_stats(
    factors: dict[str, np.ndarray | str], trait_names: list[str], n_draws: int
) -> dict:
    """(a) trait-direction variance share per factor vs the rotation null.

    Column layout of every factor array: [trait_0..trait_{T-1}, draw_0..].
    share = sum(F[:,c]^2) / sum(yc[:,c]^2); the null draws are the SAME random
    directions threaded through the whole pipeline (same-selection).
    """
    yc = np.asarray(factors["yc"], dtype=np.float64)
    n_traits = len(trait_names)
    ss_tot = (yc * yc).sum(axis=0)
    out: dict[str, Any] = {"n_draws": int(n_draws), "n_dense_rows": int(yc.shape[0])}
    shares: dict[str, np.ndarray] = {}
    for name in FACTORS:
        F = np.asarray(factors[name], dtype=np.float64)
        shares[name] = (F * F).sum(axis=0) / np.maximum(ss_tot, 1e-300)
    per_trait: dict[str, Any] = {}
    for t_i, trait in enumerate(trait_names):
        entry: dict[str, Any] = {}
        for name in FACTORS:
            obs = float(shares[name][t_i])
            null = shares[name][n_traits:]
            entry[name] = {
                "observed_share": obs,
                "null_p025": float(np.percentile(null, 2.5)),
                "null_p50": float(np.percentile(null, 50)),
                "null_p975": float(np.percentile(null, 97.5)),
                "p_value_share_ge_null": float((1 + int((null >= obs).sum())) / (1 + null.size)),
                "null_draws": [float(v) for v in null],
            }
        f_s = entry["f"]["observed_share"]
        entry["f_over_g_share_ratio"] = float(f_s / max(entry["g"]["observed_share"], 1e-300))
        entry["f_over_i_share_ratio"] = float(f_s / max(entry["i"]["observed_share"], 1e-300))
        per_trait[trait] = entry
    out["per_trait"] = per_trait
    return out


def score_corr_stats(
    factors: dict[str, np.ndarray | str],
    factor_pos: dict[int, int],
    pairs_by_trait: dict[str, list[tuple[int, float]]],
    trait_names: list[str],
    n_draws: int,
    rng: np.random.Generator,
) -> dict:
    """(b) projection--score pearson per factor vs a batched pairing-permutation null.

    Joins dense-core factor rows to P5 graded scores (unjudged rows dropped);
    the null permutes the row->score pairing on the SAME joined subset (200
    draws as one gather + GEMV per factor). Estimability follows the banked
    B1/B2 rule (>=5 scored, std>=1, both classes present).
    """
    yc = np.asarray(factors["yc"], dtype=np.float64)
    out: dict[str, Any] = {"n_draws": int(n_draws)}
    per_trait: dict[str, Any] = {}
    for t_i, trait in enumerate(trait_names):
        pairs = pairs_by_trait.get(trait, [])
        dense_pairs = [(factor_pos[int(i)], s) for i, s in pairs if int(i) in factor_pos]
        entry: dict[str, Any] = {
            "n_scored_unit_rows": len(pairs),
            "n_scored_dense": len(dense_pairs),
        }
        if dense_pairs:
            scores = np.asarray([p[1] for p in dense_pairs], dtype=np.float64)
            positives = int((scores > 50.0).sum())
            std = float(scores.std())
            entry.update(
                {
                    "n_positive": positives,
                    "n_negative": int(scores.size - positives),
                    "score_std": std,
                }
            )
            entry["estimable"] = bool(
                scores.size >= 5 and std >= 1.0 and 0 < positives < scores.size
            )
        else:
            entry["estimable"] = False
        if not entry["estimable"]:
            entry["status"] = "not_estimable_on_dense_scored_subset"
            per_trait[trait] = entry
            continue
        d_idx = np.asarray([p[0] for p in dense_pairs], dtype=np.int64)
        z_s = (scores - scores.mean()) / max(scores.std(), 1e-300)
        perms = np.stack([rng.permutation(d_idx.size) for _ in range(n_draws)])
        for name in (*FACTORS, "total"):
            arr = yc if name == "total" else np.asarray(factors[name], dtype=np.float64)
            proj = arr[d_idx, t_i]
            obs = _pearson_or_nan(proj, scores)
            p_sd = proj.std()
            if p_sd == 0.0 or np.isnan(obs):
                entry[name] = {"pearson_r": float(obs), "status": "degenerate_projection"}
                continue
            z_p = (proj - proj.mean()) / p_sd
            null = (z_p[perms] @ z_s) / d_idx.size  # (n_draws,) batched gather+GEMV
            entry[name] = {
                "pearson_r": float(obs),
                "null_p025": float(np.percentile(null, 2.5)),
                "null_p975": float(np.percentile(null, 97.5)),
                "p_value_abs_ge_null": float(
                    (1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + null.size)
                ),
                "null_draws": [float(v) for v in null],
            }
        if all(isinstance(entry[k], dict) and "pearson_r" in entry[k] for k in ("f", "g", "i")):
            f_r = abs(entry["f"]["pearson_r"])
            entry["absr_f_over_g"] = float(f_r / max(abs(entry["g"]["pearson_r"]), 1e-300))
            entry["absr_f_over_i"] = float(f_r / max(abs(entry["i"]["pearson_r"]), 1e-300))
        per_trait[trait] = entry
    out["per_trait"] = per_trait
    return out


def join_scores(
    judge_rows: list[dict], cell: str, unit_rows: list[dict]
) -> dict[str, list[tuple[int, float]]]:
    """Replicate the banked _behavior_reads row->score join verbatim."""
    row_pos = {str(row.get("row_id")): i for i, row in enumerate(unit_rows)}
    by_trait: dict[str, list[tuple[int, float]]] = {}
    for score_row in judge_rows:
        if score_row.get("cell_id") != cell and score_row.get("arm") != cell:
            continue
        score = score_row.get("score")
        row_id = str(score_row.get("row_id"))
        if score is None or row_id not in row_pos:
            continue
        by_trait.setdefault(str(score_row.get("trait")), []).append((row_pos[row_id], float(score)))
    return by_trait


# ── per-cell pipeline ─────────────────────────────────────────────────────────


def run_cell(
    cell: str,
    staged: dict[str, Path],
    ckpt: dict,
    rows: list[dict],
    rb_directions: np.ndarray,
    trait_names: list[str],
    judge_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Full repaired read for one cell: solve, decompose, stats, gate."""
    t0 = time.monotonic()
    X = np.load(staged[ARM]).astype(np.float64)
    Y_t1 = np.load(staged["t1"]).astype(np.float64)
    n0 = min(X.shape[0], Y_t1.shape[0], len(rows))
    base_rows = rows[:n0]
    idx = [
        i
        for i, row in enumerate(base_rows)
        if row.get("stratum") not in {"trait_stratum", "battery_eval_only"}
    ]
    if args.smoke:
        # Slice by PREFIX (keeps the dense-core crossing structure intact) --
        # threaded through every downstream phase via unit_rows/idx.
        dense_prefixes = sorted(
            {
                str(base_rows[i].get("prefix_id"))
                for i in idx
                if base_rows[i].get("stratum") == "dense_core"
            }
        )[: args.smoke_prefixes]
        keep_pref = set(dense_prefixes)
        idx = [i for i in idx if str(base_rows[i].get("prefix_id")) in keep_pref]
        _log(f"{cell}: SMOKE slice -> {len(idx)} rows over {len(keep_pref)} prefixes")
    idx_arr = np.asarray(idx, dtype=np.int64)
    unit_rows = [base_rows[i] for i in idx]
    if not args.smoke and len(unit_rows) != int(ckpt["n_rows"]):
        raise ValueError(f"{cell}: arm-A rows {len(unit_rows)} != banked n_rows {ckpt['n_rows']}")
    folds = _folds_from_manifest(unit_rows, len(unit_rows), group_key="prefix_id", n_folds=6)
    lambda_indices = [int(v) for v in ckpt["fit"]["lambda_indices"]]
    if len(folds) != len(lambda_indices):
        raise ValueError(f"{cell}: {len(folds)} folds != {len(lambda_indices)} banked lambdas")
    # Target columns: [3 traits, n_draws random unit directions], all in the t1
    # block (r_B support), projected BEFORE the fit (exact by ridge linearity).
    rb_block = np.stack([rb_directions[LAYER, t_i] for t_i in range(len(trait_names))])
    rb_norms = np.linalg.norm(rb_block, axis=1)
    if np.any(rb_norms == 0.0):
        raise ValueError(f"zero-norm r_B at layer {LAYER}")
    rng_dirs = np.random.default_rng([args.seed, LAYER])
    U = rng_dirs.normal(size=(args.n_draws, rb_block.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    dirs = np.concatenate([rb_block / rb_norms[:, None], U], axis=0)  # (3+D, hidden)
    Z_arm = Y_t1[idx_arr] @ dirs.T  # (n, 3+D) target-side projections
    X_arm = X[idx_arr]
    del X, Y_t1
    cache = args.cache_dir / f"phat_{cell}_L{LAYER:02d}.npz"
    cache_key = {
        "ckpt_fingerprint": ckpt["fingerprint"],
        "lambda_indices": lambda_indices,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "rb_rev": args.rb_rev,
        "smoke": bool(args.smoke),
        "smoke_prefixes": args.smoke_prefixes if args.smoke else None,
        "n_rows": len(unit_rows),
    }
    key_json = json.dumps(cache_key, sort_keys=True)
    P_hat: np.ndarray | None = None
    if cache.exists():
        payload = np.load(cache, allow_pickle=False)
        if str(payload["key"]) == key_json:
            P_hat = payload["P_hat"]
            _log(f"{cell}: resume -- OOF projections loaded from {cache}")
        else:
            _log(f"{cell}: cache key mismatch -- recomputing")
    if P_hat is None:
        _log(f"{cell}: OOF scalar solve n={len(unit_rows)} d={X_arm.shape[1]} P={Z_arm.shape[1]}")
        P_hat = solve_oof_scalar(X_arm, Z_arm, folds, lambda_indices)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, P_hat=P_hat, key=np.str_(key_json))
        os.replace(tmp, cache)
    del X_arm
    # Factor decompositions over the dense core, batched across ALL columns.
    factors_hat = _factor_components_dense_core(unit_rows, P_hat)
    factors_tgt = _factor_components_dense_core(unit_rows, Z_arm)
    factor_indices = np.asarray(factors_hat["indices"], dtype=np.int64)
    factor_pos = {int(src_i): i for i, src_i in enumerate(factor_indices.tolist())}
    pairs_by_trait = join_scores(judge_rows, cell, unit_rows)
    unit: dict[str, Any] = {
        "cell": cell,
        "layer": LAYER,
        "arm": ARM,
        "fit_arm": FIT_ARM,
        "basis": "ambient",
        "n_rows": len(unit_rows),
        "n_dense_rows": int(factor_indices.size),
        "dense_basis": factors_hat["basis"],
        "banked_ckpt_fingerprint": ckpt["fingerprint"],
        "banked_lambda_indices": lambda_indices,
        "banked_lambdas": [float(RIDGE_LAMBDAS[i]) for i in lambda_indices],
    }
    for side, facs in (("fitted_map", factors_hat), ("target_side", factors_tgt)):
        rng_perm = np.random.default_rng([args.seed, LAYER, 1 if side == "fitted_map" else 2])
        unit[side] = {
            "input": (
                "OOF context-map predictions (banked per-fold lambdas)"
                if side == "fitted_map"
                else "answer-state targets (t1)"
            ),
            "variance_share": variance_share_stats(facs, trait_names, args.n_draws),
            "score_correlation": score_corr_stats(
                facs, factor_pos, pairs_by_trait, trait_names, args.n_draws, rng_perm
            ),
        }
    # Equivalence gate vs the banked B1_map_mediated per-example pearson: the
    # reconstructed TOTAL projection on ALL scored unit rows must reproduce the
    # banked value (validates folds, standardization, lambda reuse, and join).
    if not args.smoke:
        gate: dict[str, Any] = {}
        banked_traits = ckpt.get("behavior_B1_B2", {}).get("traits", {})
        for t_i, trait in enumerate(trait_names):
            banked = (
                banked_traits.get(trait, {})
                .get("B1_by_arm_grain", {})
                .get(ARM, {})
                .get("per_example", {})
                .get("B1_map_mediated", {})
                .get("pearson_r")
            )
            if banked is None:
                gate[trait] = {"status": "banked_not_estimable"}
                continue
            pairs = pairs_by_trait.get(trait, [])
            p_idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
            scores = np.asarray([p[1] for p in pairs], dtype=np.float64)
            mine = _pearson_or_nan(P_hat[p_idx, t_i], scores)
            delta = abs(mine - float(banked))
            gate[trait] = {
                "banked": float(banked),
                "reconstructed": float(mine),
                "abs_delta": delta,
            }
            if not delta <= args.gate_tol:
                raise AssertionError(
                    f"{cell}/{trait}: B1_map_mediated gate failed -- banked {banked} vs "
                    f"reconstructed {mine} (|delta| {delta} > {args.gate_tol})"
                )
        unit["b1_map_mediated_gate"] = gate
        _log(f"{cell}: B1 gate {json.dumps({k: v.get('abs_delta') for k, v in gate.items()})}")
    unit["wall_s"] = time.monotonic() - t0
    return unit


# ── figure ────────────────────────────────────────────────────────────────────


def make_figure(units: list[dict], trait_names: list[str], fig_path: Path) -> None:
    """One figure: per-factor bars + null bands, fitted-map primary, target-side markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    roles = {"f": "primary", "g": "baseline", "i": "control"}
    factor_long = {"f": "prefix factor (f)", "g": "query factor (g)", "i": "interaction (i)"}
    trait_short = {"evil": "evil", "hallucination": "halluc.", "sycophancy": "syco."}
    n_cells = len(units)
    fig, axes = plt.subplots(2, n_cells, figsize=(6.6 * n_cells, 8.6), squeeze=False)

    def group_layout(ax) -> tuple[list[float], list[float]]:
        """Per-trait groups of 3 factor bars; trait label centered under each group."""
        xs = []
        centers = []
        pos = 0.0
        for _ in trait_names:
            trio = [pos, pos + 1.0, pos + 2.0]
            xs.extend(trio)
            centers.append(trio[1])
            pos += 3.8
        ax.set_xticks(xs)
        ax.set_xticklabels(list("fgi") * len(trait_names), fontsize=8)
        for center, trait in zip(centers, trait_names, strict=True):
            ax.text(
                center,
                -0.16,
                trait_short[trait],
                ha="center",
                va="top",
                fontsize=9,
                transform=ax.get_xaxis_transform(),
            )
        return xs, centers

    legend_handles = [
        Patch(color=paper_palette_role(roles[n]), label=factor_long[n]) for n in FACTORS
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            mfc="none",
            mec="black",
            ls="none",
            label="raw answer states (target side)",
        ),
        Line2D([], [], color="black", lw=1.2, label="null band (2.5-97.5%)"),
    ]
    cell_labels = {
        "cell_inst_own": "Instruct, own answers",
        "cell_pre_own": "Pretrained, own answers",
    }
    for c_i, unit in enumerate(units):
        cell = cell_labels.get(unit["cell"], unit["cell"])
        # Row 1: (a) trait-direction variance shares, fitted map + target side.
        ax = axes[0][c_i]
        xs, _ = group_layout(ax)
        k = 0
        for trait in trait_names:
            for name in FACTORS:
                e = unit["fitted_map"]["variance_share"]["per_trait"][trait][name]
                t = unit["target_side"]["variance_share"]["per_trait"][trait][name]
                ax.bar(
                    xs[k], e["observed_share"], width=0.72, color=paper_palette_role(roles[name])
                )
                ax.scatter(
                    [xs[k]],
                    [t["observed_share"]],
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    s=26,
                    zorder=3,
                )
                ax.plot(
                    [xs[k], xs[k]],
                    [e["null_p025"], e["null_p975"]],
                    color="black",
                    lw=1.2,
                    alpha=0.85,
                    zorder=4,
                )
                k += 1
        ax.set_ylabel("share of r_B-projected variance")
        ax.set_title(
            f"{cell} — trait-direction variance share per factor\n"
            "(bars: fitted context map; whiskers: random-direction null)",
            fontsize=9,
        )
        if c_i == 0:
            ax.legend(handles=legend_handles, fontsize=7, loc="upper left", framealpha=0.9)
        # Row 2: (b) projection--judge-score pearson on the dense-core scored subset.
        ax = axes[1][c_i]
        xs, _ = group_layout(ax)
        sc_traits = unit["fitted_map"]["score_correlation"]["per_trait"]
        sc_traits_t = unit["target_side"]["score_correlation"]["per_trait"]
        any_estimable = False
        k = 0
        for trait in trait_names:
            per = sc_traits[trait]
            per_t = sc_traits_t[trait]
            for name in FACTORS:
                ok = (
                    per.get("estimable")
                    and isinstance(per.get(name), dict)
                    and "null_p025" in per[name]
                )
                if ok:
                    any_estimable = True
                    ax.bar(
                        xs[k],
                        per[name]["pearson_r"],
                        width=0.72,
                        color=paper_palette_role(roles[name]),
                    )
                    tv = per_t.get(name, {}) if per_t.get("estimable") else {}
                    if "pearson_r" in tv:
                        ax.scatter(
                            [xs[k]],
                            [tv["pearson_r"]],
                            marker="o",
                            facecolors="none",
                            edgecolors="black",
                            s=26,
                            zorder=3,
                        )
                    ax.plot(
                        [xs[k], xs[k]],
                        [per[name]["null_p025"], per[name]["null_p975"]],
                        color="black",
                        lw=1.2,
                        alpha=0.85,
                        zorder=4,
                    )
                else:
                    ax.text(
                        xs[k],
                        0.01,
                        "n/a — not tested",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                    )
                k += 1
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_ylabel("pearson r (projection vs judge score)")
        n_note = "; ".join(
            f"{trait_short[t]} n={sc_traits[t].get('n_scored_dense', 0)}" for t in trait_names
        )
        ax.set_title(
            f"{cell} — factor projection vs graded judge score, dense core\n"
            f"(whiskers: pairing-permutation null; {n_note})",
            fontsize=9,
        )
        if not any_estimable:
            ax.set_ylim(-0.3, 0.4)
            ax.text(
                0.5,
                0.5,
                "no dense-core rows were judged for this cell\n"
                "(P5 scored its trait-relevant sparse subset only)",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
            )
    fig.suptitle(
        "Read 4c repaired: per-row r_B projections of f/g/i factor outputs of the fitted "
        f"context map (L{LAYER}, fit arm {FIT_ARM}, ambient)\n"
        "replaces the degenerate mean-projection statistic (observed == 0 by construction)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, fig_path.stem, dir=fig_path.parent, formats=("png",))
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("data/issue_1092/p0/corpus/manifest.jsonl")
    )
    p.add_argument("--judge-scores", type=Path, default=Path("data/issue_1092/p5/scores.jsonl"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("data/issue_1092/p7/staging/checkpoints"))
    p.add_argument(
        "--staging-dir", type=Path, default=Path("data/issue_1092/p7/read4c_repair/staging")
    )
    p.add_argument("--cache-dir", type=Path, default=Path("data/issue_1092/p7/read4c_repair/cache"))
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_results/issue_1092/p7/read4c_trait_per_factor_repaired.json"),
    )
    p.add_argument(
        "--fig", type=Path, default=Path("figures/issue_1092/read4c_trait_per_factor_repaired.png")
    )
    p.add_argument("--cells", default=",".join(CELLS_DEFAULT))
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rb-dir", type=Path, default=None)
    p.add_argument("--rb-rev", default=DEFAULT_RB_REV)
    p.add_argument("--n-layers", type=int, default=28)
    p.add_argument("--hidden-dim", type=int, default=3584)
    p.add_argument("--summaries-rev", default="main")
    p.add_argument("--gate-tol", type=float, default=1e-3)
    p.add_argument("--smoke", action="store_true", help="tiny slice; outputs to scratch")
    p.add_argument("--smoke-prefixes", type=int, default=10)
    return p.parse_args()


def main() -> int:
    t0 = time.monotonic()
    args = parse_args()
    if args.smoke:
        scratch = Path("/tmp/issue-1092-read4c-smoke")
        args.out_json = scratch / "read4c_trait_per_factor_repaired.json"
        args.fig = scratch / "read4c_trait_per_factor_repaired.png"
        args.cache_dir = scratch / "cache"
        args.cells = args.cells.split(",")[0]
        args.n_draws = min(args.n_draws, 8)
        _log(f"SMOKE mode: cells={args.cells} n_draws={args.n_draws} out={scratch}")
    cells = [c.strip() for c in str(args.cells).split(",") if c.strip()]
    selftest = run_selftest()
    _log(f"selftest PASS: {selftest}")
    rows = _jsonl(args.manifest)
    judge_rows = _load_judge_score_rows(args.judge_scores)
    rb_ns = argparse.Namespace(
        rb_dir=args.rb_dir, rb_rev=args.rb_rev, n_layers=args.n_layers, hidden_dim=args.hidden_dim
    )
    rb_directions, trait_names = _load_rb_directions(rb_ns)
    _log(f"r_B loaded: shape={rb_directions.shape} traits={trait_names}")
    staged, staging_prov = stage_summaries(args.staging_dir, cells, args.summaries_rev)
    units: list[dict] = []
    partial_dir = args.cache_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        ckpt, ckpt_path = load_ckpt(args.ckpt_dir, cell)
        unit = run_cell(
            cell, staged[cell], ckpt, rows, rb_directions, trait_names, judge_rows, args
        )
        unit["banked_ckpt_file"] = ckpt_path.name
        # Checkpoint-per-cell: persist the moment the cell completes.
        partial = partial_dir / f"unit_{cell}.json"
        partial.write_text(json.dumps(unit, indent=2, allow_nan=True))
        _log(f"{cell}: unit done ({unit['wall_s']:.0f}s) -> {partial}")
        units.append(unit)
    out = {
        "read": "read4c_trait_per_factor_repaired",
        "replaces": "read4c degenerate mean-projection statistic",
        "degenerate_origin": (
            "issue1092_fit_grid._selection_symmetric_projection_null projects "
            "arr.mean(axis=0) of mean-centered factor outputs onto r_B "
            "(fit_grid.py:1387) -- identically 0 by construction"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "config": {
            "cells": cells,
            "layer": LAYER,
            "arm": ARM,
            "fit_arm": FIT_ARM,
            "basis": "ambient",
            "pca48_skipped_reason": (
                "the pca48 unit's v_basis is not persisted and reconstructing it "
                "(PCA of the 19708x10752 stacked target, ~4.6e15 FLOP) is not "
                "'cheap' per the follow-up scope; the ambient read carries the "
                "identical r_B direction"
            ),
            "prefix_arm_skipped_reason": (
                "over the shared-query dense core the prefix-arm map is constant "
                "within prefix, so its g/i factor outputs vanish by design "
                "(plan section 4.5 entailed identity) -- the read is only "
                "non-degenerate on the context map"
            ),
            "n_draws": args.n_draws,
            "seed": args.seed,
            "rb_rev": args.rb_rev,
            "gate_tol": args.gate_tol,
            "smoke": bool(args.smoke),
            "manifest_sha256_16": _sha16(args.manifest),
            "judge_scores_sha256_16": _sha16(args.judge_scores),
            "threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
        },
        "selftest": selftest,
        "staging": staging_prov,
        "units": units,
        "wall_s": time.monotonic() - t0,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2, allow_nan=True))
    os.replace(tmp, args.out_json)
    make_figure(units, trait_names, args.fig)
    _log(
        f"artifact digest: units={len(units)} "
        f"json={args.out_json} ({args.out_json.stat().st_size} bytes) fig={args.fig} "
        f"wall={time.monotonic() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
