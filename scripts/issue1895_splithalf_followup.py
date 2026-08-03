"""Issue #1895 free-analysis follow-up — split-half holdout re-derivation.

Closes the shared-holdout construction coupling flagged in the main run: P_sae(64)
(top-64 principal directions of centered r_bar projections) and the per-direction
profiles (map R2 g(u), SAE FVE_u) were derived on the SAME 20,000-row holdout the
overlap O(64) and the variance-partialled Spearman were read on. Here the holdout is
split into two disjoint deterministic halves; the constructions are re-derived per
half and the reads are taken within-half (coupling present, per-half n) and
cross-half (coupling removed): a coupling inflation shows up as a cross-half drop.

Zero new fits: the vA-map per-row residuals (R1_proj = (vA - vhat_A) @ Q) and the
targets (V_proj, E_proj, R_proj) are stored per holdout row in percontext_proj.npz,
so per-half g(u)/FVE_u/P_sae are pure re-reductions (the ridge map itself was fit on
the disjoint TRAIN split — the coupling under test is holdout-side construction vs
scoring, not the fit). Conventions are imported from the run driver
(scripts/issue1895_subspaces.py): _pca_basis (fp64 covariance eigh), overlap_observed
(svdvals, O = mean cos^2), _s_pred (top-k by profile), _partial_spearman_obs
(midranks + lstsq residualization), and the S4/S6 per-direction R2 convention
(residual SS uncentered, ss_tot centered on the row set's own mean, clamp 1e-9).

Analysis-only (no GPU, no downloads): reads the locally staged analysis tensors +
committed eval summaries; writes eval_results/issue_1895/splithalf_followup.json.
Run with the shared-VM thread-cap env prefix (no thread counts hardcoded here).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# The driver import runs orchestrate.env.load_dotenv() (thread caps BEFORE numpy/torch)
# and binds the reused helpers + its own EA / C module handles.
import issue1895_subspaces as D  # noqa: E402

import numpy as np  # noqa: E402

TENSOR_ROOT = PROJECT_ROOT / "data/issue_1895/hf_dl/issue1895_subspaces/analysis_tensors"
EVAL_DIR = PROJECT_ROOT / "eval_results/issue_1895"
OUT_PATH = EVAL_DIR / "splithalf_followup.json"
K = 64
CLAMP = 1e-9  # the _h3_bootstrap re-reduction clamp (issue1895_subspaces.py S5)


def _profiles_on_rows(
    V: np.ndarray, E: np.ndarray, R1: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-direction (g_u, fve_u) on a holdout row subset — the S4/S6 convention:
    residual SS uncentered, ss_tot centered on the subset's own per-direction mean
    (EA._per_feature_metrics / _h3_bootstrap re-reduction), fp64, clamp 1e-9."""
    v = V[rows].astype(np.float64)
    e = E[rows].astype(np.float64)
    r1 = R1[rows].astype(np.float64)
    sstot_v = ((v - v.mean(0)) ** 2).sum(0)
    sstot_e = ((e - e.mean(0)) ** 2).sum(0)
    g_u = 1.0 - (r1**2).sum(0) / np.maximum(sstot_v, CLAMP)
    fve_u = 1.0 - sstot_e / np.maximum(sstot_v, CLAMP)
    return g_u, fve_u


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Plain Spearman via the driver's midrank convention (tie-exact)."""
    return float(np.corrcoef(D._midrank_1d(a), D._midrank_1d(b))[0, 1])


def main() -> None:
    """Compute within-/cross-half O(64) + partial rho and write the follow-up JSON."""
    t0 = time.time()
    with np.load(TENSOR_ROOT / "percontext_proj.npz") as z:
        V = z["V_proj"].astype(np.float32)
        R = z["R_proj"].astype(np.float32)
        E = z["E_proj"].astype(np.float32)
        R1 = z["R1_proj"].astype(np.float32)
        prov_te = z["prov_te"].astype(np.int64)
        te_pos = z["te_pos"].astype(np.int64)
        fp_proj = str(z["fingerprint"])
    n, n_dir = V.shape
    assert n_dir == 3584, (n, n_dir)
    with np.load(TENSOR_ROOT / "fve_profiles.npz") as z:
        fve_banked = z["fve_u"].astype(np.float64)
        holdout_pos = z["holdout_pos"].astype(np.int64)
    assert np.array_equal(holdout_pos, te_pos), "fve_profiles/percontext holdout rows misaligned"
    with np.load(EVAL_DIR / "perdirection_profiles.npz") as z:
        g_committed = z["r2u__t_vA_ctx"].astype(np.float64)

    # deterministic split: even/odd by holdout array position (te_pos order)
    idx = np.arange(n)
    rows_a, rows_b = idx[idx % 2 == 0], idx[idx % 2 == 1]
    split_rule = (
        "even/odd by holdout array position (te_pos order): index % 2 == 0 -> half_A, "
        "index % 2 == 1 -> half_B; deterministic, no RNG"
    )
    prov_counts = {
        h: {"lmsys": int((prov_te[r] == 0).sum()), "wildchat": int((prov_te[r] == 1).sum())}
        for h, r in (("half_A", rows_a), ("half_B", rows_b))
    }
    print(f"[splithalf] n={n} n_A={len(rows_a)} n_B={len(rows_b)} prov={prov_counts}", flush=True)

    # per-half + full-holdout profiles (pure re-reductions of the stored projections)
    g_a, fve_a = _profiles_on_rows(V, E, R1, rows_a)
    g_b, fve_b = _profiles_on_rows(V, E, R1, rows_b)
    g_full, fve_full = _profiles_on_rows(V, E, R1, idx)
    print(f"[splithalf] profiles done elapsed={time.time() - t0:.0f}s", flush=True)

    # P_sae(64) per half + full (driver _pca_basis: centered fp64 covariance eigh)
    B_a = D._pca_basis(R[rows_a], K, "cpu")
    B_b = D._pca_basis(R[rows_b], K, "cpu")
    B_full = D._pca_basis(R, K, "cpu")
    assert B_a.shape[1] == K and B_b.shape[1] == K and B_full.shape[1] == K
    print(f"[splithalf] P_sae bases done elapsed={time.time() - t0:.0f}s", flush=True)

    # P_pred(64): per-half matched g(u) selections + the fixed banked-ridge primary
    sp_a, sp_b, sp_full = (D._s_pred(g, K) for g in (g_a, g_b, g_full))
    banked = np.asarray(
        json.loads(D.EA.COMMITTED_PERDIR_PCA.read_text())["per_direction_r2"]["ridge"],
        np.float64,
    )
    sp_banked = D._s_pred(banked, K)

    def ov(sp: np.ndarray, B: np.ndarray) -> float:
        return D.overlap_observed(sp, B)[1]

    o64 = {
        # committed full-holdout references (angles_summary.json, psae_recon_pca k=64)
        "full_reference": {
            "primary_banked_ridge": 0.8669513463973999,
            "matched_refit": 0.8790439963340759,
        },
        # matched g(u)-derived P_pred: within = same-half construction+selection
        # (coupling present at half n); cross = disjoint-half (coupling removed)
        "within_half": {"A": ov(sp_a, B_a), "B": ov(sp_b, B_b)},
        "cross_half": {"pred_A_vs_sae_B": ov(sp_a, B_b), "pred_B_vs_sae_A": ov(sp_b, B_a)},
        # primary construction (banked ridge P_pred is issue_1482-derived, already
        # decoupled from this holdout): SAE-side half-stability only
        "banked_fixed_pred": {
            "full_recompute": ov(sp_banked, B_full),
            "sae_A": ov(sp_banked, B_a),
            "sae_B": ov(sp_banked, B_b),
        },
        "matched_full_recompute": ov(sp_full, B_full),
    }
    print(
        f"[splithalf] O64 done: {json.dumps(o64['within_half'])} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )

    # variance-partialled Spearman (S6 observed convention: var_rank = Q eigen-order
    # index, holdout-independent — Q is a train-split object)
    var_rank = np.arange(n_dir, dtype=np.float64)

    def partial(g: np.ndarray, f: np.ndarray) -> float:
        return D._partial_spearman_obs(g, f, [var_rank])

    partial_rho = {
        "full_reference": {
            "observed": 0.07572134014528405,
            "bootstrap_ci": [0.037670774207744714, 0.09024904796551758],
        },
        "within_half": {"A": partial(g_a, fve_a), "B": partial(g_b, fve_b)},
        "cross_half": {
            "g_A_vs_fve_B": partial(g_a, fve_b),
            "g_B_vs_fve_A": partial(g_b, fve_a),
        },
        "full_recompute_fp16_inputs": partial(g_full, fve_full),
        # exact-input replication of the committed observed read (fp32 profiles)
        "full_recompute_committed_profiles": D._partial_spearman_obs(
            g_committed, fve_banked, [var_rank]
        ),
        "rho_unpartialled": {
            "within_half": {"A": _spearman(g_a, fve_a), "B": _spearman(g_b, fve_b)},
            "cross_half": {
                "g_A_vs_fve_B": _spearman(g_a, fve_b),
                "g_B_vs_fve_A": _spearman(g_b, fve_a),
            },
            "full_reference": 0.969833687008501,
        },
    }

    # internal consistency: fp16-projection re-reductions vs the run's fp32 profiles
    checks = {
        "g_full_vs_committed_spearman": _spearman(g_full, g_committed),
        "g_full_vs_committed_max_abs": float(np.max(np.abs(g_full - g_committed))),
        "fve_full_vs_banked_spearman": _spearman(fve_full, fve_banked),
        "fve_full_vs_banked_max_abs": float(np.max(np.abs(fve_full - fve_banked))),
    }
    print(f"[splithalf] checks: {json.dumps(checks)}", flush=True)

    out = {
        "split_rule": split_rule,
        "n_half_A": int(len(rows_a)),
        "n_half_B": int(len(rows_b)),
        "prov_counts": prov_counts,
        "O64": o64,
        "partial_rho": partial_rho,
        "recompute_checks": checks,
        "inputs_fingerprint_code_sha": json.loads(fp_proj)["code_sha"],
        "notes": (
            "Cross-half cells derive the constructions (P_sae eigh; g(u)/FVE_u profiles) on "
            "one half and pair them with the other half's construction, removing the "
            "shared-holdout coupling; within-half cells keep the coupling at half n as the "
            "sampling-variance contrast. P_pred for the within/cross O(64) cells is the "
            "per-half matched g(u) top-64 (the banked-ridge primary P_pred is issue_1482-"
            "derived and already holdout-decoupled; its per-half read is the "
            "banked_fixed_pred block). Conventions imported from issue1895_subspaces.py: "
            "centered-ss_tot per-direction R2 (clamp 1e-9), fp64 covariance-eigh PCA, "
            "O = mean cos^2 of subspace principal angles, midrank+lstsq partial Spearman "
            "with var_rank = Q eigen-order index (holdout-independent). Stored projections "
            "are fp16; the full-recompute rows quantify the storage-precision effect vs the "
            "run's fp32-computed committed profiles."
        ),
        **D.C.reproducibility_metadata(),
    }
    D.C.write_json_atomic(OUT_PATH, out)
    print(f"[splithalf] wrote {OUT_PATH} elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
