"""P-Fit-ext: extension-ladder fits driver for #823 `origin-ladder-more-contexts` (unit 4).

Implements plan v17 section 4.3 "P-Fit-ext" steps 1-11 on the stores unit 3
(`issue823_ladder_ext_capture.py`) verified (`_store_verified.json`):

  1. Gate C (G3): 1-cell pilot at TOP-rung production shape FIRST; projected
     full-battery wall (both ladders + P2-ext + bootstrap allowance) > 2x the
     section-9 row => designed abort (rc 19).
  2. Estimator: dof-capped GCV (dof <= 0.9 * n_train), lambda grid
     logspace(-2,4,13) (FFC.LAMBDAS, parent grid). Per-fit selected lambda AND
     dof persisted into every cell. Lambda-edge trigger per rung per ladder
     (> 10% of the rung's production fits at the TOP grid edge) => labeled
     wider-grid sensitivity at logspace(-2,8,21), read-out layers.
     Pure (uncapped) GCV runs in exactly TWO labeled sensitivity slots:
     `sens_estimator` (rung-1 two-arm mask) and the banked bridge (step 8).
  3. Solver: n_train <= 6,000 -> parent dual path (Gram eigh via
     LF.factorize_robust); above -> NEW primal path `_factorize_primal`
     ((d x d) covariance eigh, fp64, identical GCV criterion — the n-d zero
     dual eigenvalues contribute 0). Rung-1 three-way dual/primal/canonical
     agreement check.
  4. Gate D (G2): >= 3 (layer, fold) parity slices per rung PER LADDER vs the
     canonical per-fit SVD reference; FAIL => canonical contingency at
     read-out layers; parity-class FAIL on the contingency too => designed
     abort (rc 20).
  5. Per-rung outputs: percontext_rung{r}.npz + estimator-hygiene rows.
  6. Paired read per rung via the UNMODIFIED-content parametrized
     `scripts/issue823_shared_persona_paired.py` (subprocess; --full-ratio-ci).
  7. MF-A1 randomized-subset companion ladder (fit-time-only): E_eval =
     rng(8230) 50% era-stratified sample of the top-rung shared-persona
     contexts; nested T_r = rng(8231) era-stratified subsets at realized |M_r|
     sizes; set-check violation => designed abort (rc 22).
  8. Gate E (G1) banked bridge: (i) loader-level rerun on the banked npz vs
     the committed shared_persona_paired.json @ 84633d46c6 (rtol 1e-9 on
     ratio_measured_over_full_energy); (ii) rung-1 refit on the EXACT banked
     4,629 mask (pure GCV + canonical solver) vs banked ratios +-0.05
     absolute. FAIL => designed abort (rc 21).
  9. Baselines: identity+learned-bias per (rung x arm x layer); kNN retrieval
     (euclidean + cosine) at the 3 read-out layers.
 10. P2-ext boundary ladder: k=1 arm, 3 read-out layers, fixed 4,800-context
     holdout, 5 draws x 12 n_train rungs across the d = 3,584 boundary.
 11. Bootstraps ride the paired script's vectorized batteries (no serial
     per-draw loop); fit batteries share one factorization across both arms'
     targets and the lambda grid.

Designed-halt rc table (halt-and-report; NO downstream completion sentinel is
written — `_fits_complete.json` only on full success):

  rc 19  fits-wall abort            (Gate C / plan kill path 4, fits side)
  rc 20  solver-parity terminal     (Gate D contingency FAIL / kill path 5)
  rc 21  banked-continuity FAIL     (Gate E / kill path 6 — rig seam)
  rc 22  companion-manifest set-check violation (kill path 8)

Resume: `_fits_complete.json` is written ATOMICALLY (tmp + rename) after every
step succeeds and carries the plan-registered fingerprint (rung-mask shas incl.
the rand_ladder_manifest.json sha, code SHA, estimator/solver config, required
output key sets, Gate D/E PASS states + the Gate F verdict carried from
mask_ext.json). The resume predicate requires fingerprint EQUALITY — never bare
existence. Intra-run, per (ladder x rung x layer) chunks checkpoint via the
parent's fingerprinted chunk convention (machine-stable keys: id shas +
generating parameters, never float bytes).

Pod-side note: `scripts/issue823_shared_persona_paired.py` calls
`task_workflow.repo_root()`, which branch-guards to `main` and requires
`tasks/` — both false on a pod issue-branch partial clone. The runner here
executes the UNMODIFIED script bytes through a `-c` shim that injects the
clone root into the module's `repo_root` seam (metadata git_commit still
records the pod HEAD); script content is untouched.

Usage:
  uv run python scripts/issue823_ladder_ext_fits.py --list-rcs
  uv run python scripts/issue823_ladder_ext_fits.py --import-check
  uv run python scripts/issue823_ladder_ext_fits.py --phase fits [--smoke] [--out-root ...]
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # creds + shared-VM thread caps BEFORE the numpy/torch imports

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

# Repo root on sys.path so `scripts.*` sibling imports resolve in script mode
# (#823's own gotcha: sys.path[0] is scripts/ when run as a file path).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
assert (_REPO_ROOT / "pyproject.toml").exists(), _REPO_ROOT

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from scripts import issue779_fitter_fair_comparison as FFC  # noqa: E402
from scripts import issue823_ladder_ext_capture as EXTCAP  # noqa: E402
from scripts import issue823_ladder_fits as LF  # noqa: E402
from scripts.issue823_ladder_common import (  # noqa: E402
    correlated_floor_from_groups,
    mixture_energy_from_group_diffs,
)
from explore_persona_space.experiments.issue_823.run_823 import write_sentinel  # noqa: E402
from scripts.issue823_ladder_gen import write_json  # noqa: E402

logger = logging.getLogger("issue823_ladder_ext_fits")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Registered constants (plan v17 sections 4.3 / 4.4 / 7 / 10) ─────────────

READ_OUT_LAYERS: tuple[int, ...] = (14, 26, 17)
EXPECTED_LAYERS = EXTCAP.EXPECTED_LAYERS  # 28
HIDDEN = EXTCAP.EXPECTED_HIDDEN  # 3584
ARM_NAMES: tuple[str, ...] = ("k1", "k16")
POOLED_K = 16
N_FOLDS = 5
FOLD_SEED = 0  # KFold(5, shuffle=True, random_state=0) — parent parity
DOF_CAP = 0.9
DUAL_N_MAX = 6_000  # n_train <= this -> dual (Gram) path; above -> primal (d x d)
LAYER_CHUNK = 4  # layers per batched eigh stack (plan v17: chunks of 4-8, peak <~25 GB HBM)
LAMBDAS = FFC.LAMBDAS  # logspace(-2, 4, 13) — the parent primary grid
LAMBDA_GRID_PARAMS = ("logspace", -2, 4, 13)
LAMBDAS_WIDE = np.logspace(-2, 8, 21)  # parent P2 grid — labeled sensitivity only
LAMBDA_WIDE_PARAMS = ("logspace", -2, 8, 21)
LAMBDA_EDGE_FRACTION = 0.10  # per rung per ladder over that rung's production fits
G2_MAX_REL_TOL = 1e-4
G2_DELTA_R2_TOL = 1e-4
G2_SLICES: tuple[tuple[int, int], ...] = ((14, 0), (26, 1), (17, 2))  # (layer, fold)
PLANNED_FITS_WALL_H = 3.0  # plan section-9 P-Fit-ext row
FITS_WALL_ABORT_FACTOR = 2.0
BOOT_ALLOWANCE_S = 600.0  # fixed Gate-C allowance for the paired-script batteries
E_EVAL_SEED = 8230
T_SUBSET_SEED = 8231
P2_HOLDOUT_N = 4_800
P2_N_TRAIN_GRID: tuple[int, ...] = (
    896,
    1_792,
    2_688,
    3_136,
    3_336,  # the parent's exact withheld rung
    3_584,  # = d, the boundary
    4_032,
    4_480,
    7_168,
    14_336,
    28_672,
)
P2_DRAW_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
BOOT_N = 10_000
SMOKE_BOOT_N = 200
BRIDGE_LOADER_RTOL = 1e-9
BRIDGE_REFIT_ABS_TOL = 0.05
BANKED_PAIRED_REV = "84633d46c6"
BANKED_LADDER_RELDIR = "eval_results/issue_823/inconsistent_origin_ladder"
FITS_SENTINEL = "_fits_complete.json"
SMOKE_MASK_CAP = 96  # smoke-only per-rung mask cap (scale reduction, not a path skip)
PROD_OUT_ROOT = pathlib.Path("/workspace/eps/out/issue823_ladder_ext")
PAIRED_SCRIPT_RELPATH = "scripts/issue823_shared_persona_paired.py"

RC_FITS_WALL = 19
RC_SOLVER_PARITY = 20
RC_BANKED_CONTINUITY = 21
RC_RAND_MANIFEST = 22

RC_TABLE: dict[int, str] = {
    RC_FITS_WALL: "Gate C fits-wall abort: projected battery wall > 2x the section-9 row",
    RC_SOLVER_PARITY: "Gate D terminal: solver parity FAIL on the canonical contingency too",
    RC_BANKED_CONTINUITY: "Gate E banked-continuity FAIL (rig seam, never a finding)",
    RC_RAND_MANIFEST: "companion-manifest set-check violation (nesting/sizes/strata)",
}
# rcs owned by the sibling drivers (kept disjoint; asserted in tests):
#   unit 2 (issue823_ladder_ext_gen): 3, 6, 7, 8, 9, 10, 11
#   unit 3 (issue823_ladder_ext_capture): 4, 12-18
#   parent (issue823_ladder_fits): 5, 6, 7 (its own process space)

_halt = EXTCAP._halt  # one designed-halt implementation for the whole round
DesignedHalt = EXTCAP.DesignedHalt


def _ids_sha(ids) -> str:
    """Machine-stable sha over an int id list (never float bytes)."""
    arr = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def arm_persona(arm: str, ctx_id: int) -> int:
    """Registered assignment rule persona(i, k) = i mod k for the 2-arm design."""
    if arm == "k1":
        return 0
    if arm == "k16":
        return int(ctx_id) % POOLED_K
    raise ValueError(f"unknown arm {arm!r}")


# ── Solver core (dual + primal + canonical; one capped-GCV criterion) ────────


def _factorize_primal(x_tr_np: np.ndarray, dev: torch.device) -> dict:
    """Primal (d x d) factorization for n_train > d regimes (plan 4.3 step 3).

    Standardizes X on train stats exactly as FFC._factorize (torch .std(0) —
    sample std — + 1e-9), then fp64 eigh of the covariance Xn^T Xn. GCV dof =
    sum_j w_j/(w_j+lambda) — identical to the dual criterion (the n-d zero
    dual eigenvalues contribute 0). One factorization is shared across both
    arms' targets and the whole lambda grid. cuSOLVER non-convergence falls
    back to CPU eigh (exact backend swap — gotchas.md).
    """
    x = torch.as_tensor(np.asarray(x_tr_np), dtype=torch.float64, device=dev)
    xmu = x.mean(0)
    xsd = x.std(0) + 1e-9  # torch sample std — byte-parity with FFC._factorize
    xn = (x - xmu) / xsd
    cov = xn.T @ xn
    try:
        w, v = torch.linalg.eigh(cov)
    except torch.linalg.LinAlgError:
        logger.warning(
            "[primal] cuSOLVER eigh failed on device %s — CPU fallback (d=%d)", dev, cov.shape[0]
        )
        w, v = torch.linalg.eigh(cov.cpu())
        w, v = w.to(dev), v.to(dev)
    return {
        "kind": "primal",
        "xmu": xmu,
        "xsd": xsd,
        "Xn": xn,
        "w": torch.clamp(w, min=0.0),
        "V": v,
        "ntr": int(x.shape[0]),
        "dev": dev,
    }


def rung_factorize(x_tr_np: np.ndarray, dev: torch.device) -> dict:
    """Route by n_train: dual (parent FFC path) at <= DUAL_N_MAX, else primal."""
    if int(np.asarray(x_tr_np).shape[0]) <= DUAL_N_MAX:
        fact = LF.factorize_robust(x_tr_np, dev)
        fact["kind"] = "dual"
        return fact
    return _factorize_primal(x_tr_np, dev)


def _factorize_dual_batched(x_list: list[np.ndarray], dev: torch.device) -> list[dict]:
    """Batched dual factorization: ONE (C, n, n) Gram eigh across a layer chunk.

    Per-slice dicts mirror FFC._factorize's dual schema EXACTLY (torch sample
    std + 1e-9, clamped eigenvalues) so `solve_capped`/`eval_kernel`/`apply_fit`
    consume each slice unchanged. cuSOLVER non-convergence falls back to a
    whole-stack CPU eigh (exact backend swap — gotchas.md; never jitter).
    """
    xs = torch.stack([torch.as_tensor(np.asarray(x), dtype=torch.float64) for x in x_list]).to(dev)
    xmu = xs.mean(1, keepdim=True)
    xsd = xs.std(1, keepdim=True) + 1e-9  # correction=1 == per-slice .std(0)
    xn = (xs - xmu) / xsd
    gram = xn @ xn.transpose(1, 2)
    try:
        w, v = torch.linalg.eigh(gram)
    except torch.linalg.LinAlgError:
        logger.warning("[dual-batched] eigh failed on %s — whole-stack CPU fallback", dev)
        w, v = torch.linalg.eigh(gram.cpu())
        w, v = w.to(dev), v.to(dev)
    w = torch.clamp(w, min=0.0)
    return [
        {
            "kind": "dual",
            "xmu": xmu[i, 0],
            "xsd": xsd[i, 0],
            "Xtr_n": xn[i],
            "w": w[i],
            "V": v[i],
            "ntr": int(xs.shape[1]),
            "dev": dev,
        }
        for i in range(len(x_list))
    ]


def _factorize_primal_batched(x_list: list[np.ndarray], dev: torch.device) -> list[dict]:
    """Batched primal factorization: ONE (C, d, d) covariance eigh across a chunk.

    Per-slice dicts mirror `_factorize_primal`'s schema exactly; same
    whole-stack CPU fallback discipline as the dual twin.
    """
    xs = torch.stack([torch.as_tensor(np.asarray(x), dtype=torch.float64) for x in x_list]).to(dev)
    xmu = xs.mean(1, keepdim=True)
    xsd = xs.std(1, keepdim=True) + 1e-9
    xn = (xs - xmu) / xsd
    cov = xn.transpose(1, 2) @ xn
    try:
        w, v = torch.linalg.eigh(cov)
    except torch.linalg.LinAlgError:
        logger.warning("[primal-batched] eigh failed on %s — whole-stack CPU fallback", dev)
        w, v = torch.linalg.eigh(cov.cpu())
        w, v = w.to(dev), v.to(dev)
    w = torch.clamp(w, min=0.0)
    return [
        {
            "kind": "primal",
            "xmu": xmu[i, 0],
            "xsd": xsd[i, 0],
            "Xn": xn[i],
            "w": w[i],
            "V": v[i],
            "ntr": int(xs.shape[1]),
            "dev": dev,
        }
        for i in range(len(x_list))
    ]


def batched_rung_factorize(x_list: list[np.ndarray], dev: torch.device) -> list[dict]:
    """Route a same-n layer-chunk stack by n_train (plan v17 layer-chunk design:
    ONE batched eigh per (chunk, fold) — no serial per-cell factorization loop)."""
    n = int(np.asarray(x_list[0]).shape[0])
    if len(x_list) == 1:
        return [rung_factorize(x_list[0], dev)]
    if n <= DUAL_N_MAX:
        return _factorize_dual_batched(x_list, dev)
    return _factorize_primal_batched(x_list, dev)


def eval_kernel(fact: dict, x_ev_np: np.ndarray) -> torch.Tensor:
    """FFC._apply-compatible eval kernel: dual -> KevV; primal -> Zev = Xev_n V."""
    if fact.get("kind", "dual") == "dual":
        return FFC._cross_kernel(fact, x_ev_np)
    xe = torch.as_tensor(np.asarray(x_ev_np), dtype=torch.float64, device=fact["dev"])
    return ((xe - fact["xmu"]) / fact["xsd"]) @ fact["V"]


def solve_capped(
    fact: dict,
    y_tr_np: np.ndarray,
    lambdas: np.ndarray = LAMBDAS,
    cap_frac: float = DOF_CAP,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    """Dof-capped GCV lambda for ONE target off a shared dual/primal factorization.

    Returns (best_lam, proj, ymu, best_dof) where `proj` is FFC._apply-compatible
    (dual: VtY; primal: U = V^T Xn^T Yc). Selection criterion is IDENTICAL to
    the parent's `LF.gcv_solve_dof_capped` (pinned by test parity), extended to
    the primal factorization and a parametrized grid; `cap_frac=math.inf`
    recovers pure GCV (the two labeled sensitivity slots ONLY). Raises
    RuntimeError when the cap excludes every grid lambda.
    """
    y = torch.as_tensor(np.asarray(y_tr_np), dtype=torch.float64, device=fact["dev"])
    if y.ndim == 1:
        y = y[:, None]
    ymu = y.mean(0)
    yc = y - ymu
    w, ntr = fact["w"], fact["ntr"]
    if fact.get("kind", "dual") == "dual":
        proj = fact["V"].T @ yc
        sq = (proj**2).sum(1)
    else:
        proj = fact["V"].T @ (fact["Xn"].T @ yc)
        sq_raw = (proj**2).sum(1)
        w_floor = 1e-10 * float(w.max().clamp(min=1e-30))
        # Directions with no data support (w ~ 0) contribute exactly 0 in exact
        # arithmetic; mask them so 0/0 noise cannot pollute the GCV objective.
        sq = torch.where(w > w_floor, sq_raw / torch.clamp(w, min=1e-300), torch.zeros_like(sq_raw))
    tot = float((yc**2).sum())
    cap = cap_frac * ntr
    best_lam, best_gcv, best_dof = None, float("inf"), None
    for lam in lambdas:
        lam = float(lam)
        filt = w / (w + lam)
        dof = float(filt.sum())
        if dof > cap:
            continue
        rss = tot - float(((2 * filt - filt**2) * sq).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_lam, best_gcv, best_dof = lam, gcv, dof
    if best_lam is None:
        raise RuntimeError(
            f"dof cap {cap_frac} * n_train={ntr} excludes EVERY lambda in the grid "
            f"[{float(lambdas[0]):g}, {float(lambdas[-1]):g}] — no admissible fit"
        )
    return best_lam, proj, ymu, best_dof


def apply_fit(fact: dict, lam: float, proj: torch.Tensor, ymu: torch.Tensor, kev) -> np.ndarray:
    """Predictions off a solved fit — the SAME FFC._apply expression both kinds."""
    return FFC._apply(fact, lam, proj, ymu, kev)


def lambda_is_top_edge(lam: float, lambdas: np.ndarray = LAMBDAS) -> bool:
    return bool(lam >= float(lambdas[-1]) * (1.0 - 1e-12))


def canonical_capped_fit(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    lambdas: np.ndarray = LAMBDAS,
    cap_frac: float = DOF_CAP,
) -> tuple[np.ndarray, float, float]:
    """Canonical per-fit reference: numpy SVD direct solve, same GCV criterion.

    The Gate D / rung-1 three-way reference and the Gate D contingency solver
    (`svd_gcv_lambda`-style — plan 4.3 step 4). Standardization matches the
    torch paths (sample std, ddof=1). Returns (pred_on_x_ev, lambda, dof).
    """
    x = np.asarray(x_tr, dtype=np.float64)
    y = np.asarray(y_tr, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    xmu = x.mean(0)
    xsd = x.std(0, ddof=1) + 1e-9
    xn = (x - xmu) / xsd
    u, s, vt = np.linalg.svd(xn, full_matrices=False)
    ymu = y.mean(0)
    yc = y - ymu
    proj = u.T @ yc
    sq = (proj**2).sum(1)
    tot = float((yc**2).sum())
    ntr = x.shape[0]
    cap = cap_frac * ntr
    best_lam, best_gcv, best_dof = None, float("inf"), None
    for lam in lambdas:
        lam = float(lam)
        f = s**2 / (s**2 + lam)
        dof = float(f.sum())
        if dof > cap:
            continue
        rss = tot - float(((2 * f - f**2) * sq).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_lam, best_gcv, best_dof = lam, gcv, dof
    if best_lam is None:
        raise RuntimeError(
            f"canonical: dof cap {cap_frac} * n_train={ntr} excludes every grid lambda"
        )
    w_mat = vt.T @ ((s / (s**2 + best_lam))[:, None] * proj)
    pred = ((np.asarray(x_ev, dtype=np.float64) - xmu) / xsd) @ w_mat + ymu
    return pred, best_lam, best_dof


# ── Gate C (fits wall) ────────────────────────────────────────────────────────


def project_battery_wall(
    t_dual_s: float,
    t_primal_s: float,
    n_dual_cells: int,
    n_primal_cells: int,
    boot_allowance_s: float = BOOT_ALLOWANCE_S,
) -> float:
    """Projected full-battery wall (hours) from the two pilot cell timings.

    Primal cells are priced at the TOP-rung pilot cost (an upper bound for the
    smaller primal rungs); dual cells at the rung-1-shape pilot cost. The
    paired-script bootstrap batteries ride a fixed allowance (vectorized,
    minutes — plan 4.3 step 11).
    """
    return (t_dual_s * n_dual_cells + t_primal_s * n_primal_cells + boot_allowance_s) / 3600.0


def gate_c_record(
    t_dual_s: float,
    t_primal_s: float,
    n_dual_cells: int,
    n_primal_cells: int,
    planned_wall_h: float,
    factor: float = FITS_WALL_ABORT_FACTOR,
) -> dict:
    projected_h = project_battery_wall(t_dual_s, t_primal_s, n_dual_cells, n_primal_cells)
    return {
        "t_dual_cell_s": t_dual_s,
        "t_primal_cell_s": t_primal_s,
        "n_dual_cells": n_dual_cells,
        "n_primal_cells": n_primal_cells,
        "boot_allowance_s": BOOT_ALLOWANCE_S,
        "projected_wall_h": projected_h,
        "planned_wall_h": planned_wall_h,
        "abort_factor": factor,
        "pass": bool(projected_h <= factor * planned_wall_h),
    }


def enforce_gate_c(record: dict, eval_dir: pathlib.Path, smoke: bool) -> None:
    """Halt rc 19 on a Gate C FAIL (informational under --smoke: gate calibration)."""
    if record["pass"]:
        logger.info("[gate-c] PASS: %s", record)
        return
    if smoke:
        logger.warning("SMOKE-INFORMATIONAL (enumerated blind spot) gate-c FAIL: %s", record)
        return
    _halt(
        RC_FITS_WALL,
        eval_dir / "ext_fits_wall_report.json",
        {"gate_c": record},
        f"Gate C: projected fits wall {record['projected_wall_h']:.2f} h > "
        f"{record['abort_factor']}x planned {record['planned_wall_h']:.2f} h",
    )


# ── Gate D (solver parity) ────────────────────────────────────────────────────


def g2_slice_record(
    pred_prod: np.ndarray,
    pred_ref: np.ndarray,
    lam_prod: float,
    lam_ref: float,
    r2_prod: float,
    r2_ref: float,
    layer: int,
    fold: int,
    arm: str,
) -> dict:
    """One parity slice: prediction max-rel + |dR2| + lambda agreement (parent tolerances)."""
    scale = max(float(np.abs(pred_ref).max()), 1e-9)
    max_rel = float(np.abs(pred_prod - pred_ref).max()) / scale
    d_r2 = abs(float(r2_prod) - float(r2_ref))
    lam_ok = bool(abs(lam_prod - lam_ref) <= 1e-12 * max(lam_prod, lam_ref, 1.0))
    return {
        "layer": layer,
        "fold": fold,
        "arm": arm,
        "max_rel": max_rel,
        "delta_r2": d_r2,
        "lambda_prod": lam_prod,
        "lambda_ref": lam_ref,
        "lambda_agree": lam_ok,
        "pass": bool(max_rel <= G2_MAX_REL_TOL and d_r2 <= G2_DELTA_R2_TOL and lam_ok),
    }


def contingency_parity_check(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    y_score: np.ndarray,
    dev: torch.device,
    lambdas: np.ndarray = LAMBDAS,
    cap_frac: float = DOF_CAP,
    layer: int = -1,
    fold: int = -1,
    arm: str = "",
) -> dict:
    """Independent-numerics check of the contingency solver (kill path 5 terminal).

    Canonical numpy-SVD fit vs the torch fp64 primal-eigh fit on the SAME
    slice — two independent factorization backends of the same criterion,
    checked on max-rel, lambda agreement AND per-slice |dR2| (the parent's
    MEASURED G2 failure statistic, plan section 4.4 — a contingency check
    weaker than the gate it replaces would miss the demonstrated failure
    class). A parity-class disagreement here means the canonical contingency
    itself is unverifiable => the caller halts rc 20.
    """
    pred_c, lam_c, dof_c = canonical_capped_fit(x_tr, y_tr, x_ev, lambdas, cap_frac)
    fact = _factorize_primal(x_tr, dev)
    lam_p, proj, ymu, dof_p = solve_capped(fact, y_tr, lambdas, cap_frac)
    pred_p = apply_fit(fact, lam_p, proj, ymu, eval_kernel(fact, x_ev))
    scale = max(float(np.abs(pred_c).max()), 1e-9)
    max_rel = float(np.abs(pred_p - pred_c).max()) / scale
    y = np.asarray(y_score, dtype=np.float64)
    tot = float(((y - y.mean(0)) ** 2).sum()) + 1e-12
    r2_c = 1.0 - float(((y - pred_c) ** 2).sum()) / tot
    r2_p = 1.0 - float(((y - pred_p) ** 2).sum()) / tot
    d_r2 = abs(r2_p - r2_c)
    return {
        "layer": layer,
        "fold": fold,
        "arm": arm,
        "max_rel": max_rel,
        "delta_r2": d_r2,
        "r2_canonical": r2_c,
        "r2_primal": r2_p,
        "lambda_canonical": lam_c,
        "lambda_primal": lam_p,
        "dof_canonical": dof_c,
        "dof_primal": dof_p,
        "pass": bool(
            max_rel <= G2_MAX_REL_TOL
            and d_r2 <= G2_DELTA_R2_TOL
            and abs(lam_c - lam_p) <= 1e-12 * max(lam_c, 1.0)
        ),
    }


def enforce_contingency_parity(records: list[dict], eval_dir: pathlib.Path, rung: str) -> None:
    fails = [r for r in records if not r["pass"]]
    if fails:
        _halt(
            RC_SOLVER_PARITY,
            eval_dir / "ext_solver_parity_report.json",
            {"rung": rung, "contingency_slices": records},
            f"Gate D terminal: parity-class FAIL on the canonical contingency at rung {rung}",
        )


# ── Gate E (banked bridge) ────────────────────────────────────────────────────


def bridge_loader_compare(got: dict, banked: dict, rtol: float = BRIDGE_LOADER_RTOL) -> dict:
    """Compare ratio_measured_over_full_energy across arms x read-out layers."""
    rows = []
    ok = True
    for k_key, arm_block in sorted(banked["arms"].items()):
        for l_key, cell in sorted(arm_block["per_layer"].items()):
            want = cell["offset_bias_control"]["ratio_measured_over_full_energy"]
            have = got["arms"][k_key]["per_layer"][l_key]["offset_bias_control"][
                "ratio_measured_over_full_energy"
            ]
            match = bool(np.isclose(have, want, rtol=rtol, atol=0.0))
            ok = ok and match
            rows.append(
                {"arm": k_key, "layer": l_key, "banked": want, "rerun": have, "pass": match}
            )
    return {"rtol": rtol, "rows": rows, "pass": ok}


def bridge_refit_compare(
    refit_ratios: dict[str, float],
    banked_ratios: dict[str, float],
    tol: float = BRIDGE_REFIT_ABS_TOL,
) -> dict:
    rows = []
    ok = True
    for l_key, want in sorted(banked_ratios.items()):
        have = refit_ratios[l_key]
        match = bool(abs(have - want) <= tol)
        ok = ok and match
        rows.append(
            {
                "layer": l_key,
                "banked": want,
                "refit": have,
                "abs_diff": abs(have - want),
                "pass": match,
            }
        )
    return {"abs_tol": tol, "rows": rows, "pass": ok}


def enforce_gate_e(record: dict, eval_dir: pathlib.Path, which: str) -> None:
    if not record["pass"]:
        _halt(
            RC_BANKED_CONTINUITY,
            eval_dir / "ext_banked_continuity_report.json",
            {"which": which, "record": record},
            f"Gate E banked-continuity FAIL ({which}) — rig seam, refusing to ship the ladder",
        )


# ── Companion ladder (MF-A1) construction + set-checks ───────────────────────


def _era_split(ids: np.ndarray, n_prefix: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(int(i) for i in ids), dtype=np.int64)
    return ids[ids < n_prefix], ids[ids >= n_prefix]


def build_companion_sets(
    top_mask_ids: np.ndarray,
    rung_sizes: dict[str, int],
    n_prefix: int,
    e_seed: int = E_EVAL_SEED,
    t_seed: int = T_SUBSET_SEED,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """E_eval + nested era-stratified T_r subsets + the persisted manifest.

    E_eval: 50% era-stratified sample of the top-rung shared-persona contexts
    (i mod 16 == 0), rng(8230). T_r: nested era-stratified prefixes of the
    rng(8231)-permuted pool (top mask minus E_eval), |T_r| = realized |M_r|
    for the sub-top rungs, T_top = the whole pool. Returns
    (e_eval_ids, {rung_label: T_r ids}, manifest).
    """
    top = np.asarray(sorted(int(i) for i in top_mask_ids), dtype=np.int64)
    shared = top[top % POOLED_K == 0]
    sh_banked, sh_ext = _era_split(shared, n_prefix)
    rng_e = np.random.default_rng(e_seed)
    take_b = len(sh_banked) // 2
    take_x = len(sh_ext) // 2
    e_eval = np.concatenate(
        [
            rng_e.choice(sh_banked, size=take_b, replace=False) if take_b else sh_banked[:0],
            rng_e.choice(sh_ext, size=take_x, replace=False) if take_x else sh_ext[:0],
        ]
    )
    e_eval = np.asarray(sorted(int(i) for i in e_eval), dtype=np.int64)
    e_set = set(e_eval.tolist())
    pool = np.asarray([i for i in top if int(i) not in e_set], dtype=np.int64)
    p_banked, p_ext = _era_split(pool, n_prefix)
    banked_frac = len(p_banked) / max(len(pool), 1)
    rng_t = np.random.default_rng(t_seed)
    perm_b = rng_t.permutation(p_banked)
    perm_x = rng_t.permutation(p_ext)

    labels = sorted(rung_sizes, key=lambda k: rung_sizes[k])
    top_label = labels[-1]
    subsets: dict[str, np.ndarray] = {}
    manifest_rungs: dict[str, dict] = {}
    prev_b, prev_x = 0, 0
    for label in labels:
        if label == top_label:
            n_b, n_x = len(perm_b), len(perm_x)
        else:
            m = min(int(rung_sizes[label]), len(pool))
            n_b = min(int(m * banked_frac), len(perm_b))
            n_x = min(m - n_b, len(perm_x))
        if n_b < prev_b or n_x < prev_x:
            # Prefix construction is monotone by design; a violation here is a
            # construction bug the set-check below turns into the rc-22 abort.
            n_b, n_x = max(n_b, prev_b), max(n_x, prev_x)
        prev_b, prev_x = n_b, n_x
        t_ids = np.asarray(
            sorted(int(i) for i in np.concatenate([perm_b[:n_b], perm_x[:n_x]])), dtype=np.int64
        )
        subsets[label] = t_ids
        manifest_rungs[label] = {
            "n": int(len(t_ids)),
            "n_banked": int(n_b),
            "n_ext": int(n_x),
            "target_size": int(rung_sizes[label]),
            "ids_sha256": _ids_sha(t_ids),
            "ids": [int(i) for i in t_ids],
        }
    manifest = {
        "e_eval_seed": e_seed,
        "t_subset_seed": t_seed,
        "n_prefix": n_prefix,
        "top_rung_label": top_label,
        "top_mask_sha256": _ids_sha(top),
        "n_shared": int(len(shared)),
        "e_eval": {
            "n": int(len(e_eval)),
            "n_banked": int(take_b),
            "n_ext": int(take_x),
            "ids_sha256": _ids_sha(e_eval),
            "ids": [int(i) for i in e_eval],
        },
        "pool": {
            "n": int(len(pool)),
            "n_banked": int(len(p_banked)),
            "n_ext": int(len(p_ext)),
            "banked_fraction": float(banked_frac),
            "ids_sha256": _ids_sha(pool),
        },
        "rungs": manifest_rungs,
    }
    return e_eval, subsets, manifest


def check_companion_manifest(
    e_eval: np.ndarray,
    subsets: dict[str, np.ndarray],
    manifest: dict,
    top_mask_ids: np.ndarray,
) -> list[str]:
    """Set-checks (plan 4.3 step 7): nesting, sizes, strata, E_eval exclusion."""
    violations: list[str] = []
    top_set = set(int(i) for i in top_mask_ids)
    e_set = set(int(i) for i in e_eval)
    if not e_set <= top_set:
        violations.append("e_eval not a subset of the top-rung mask")
    if any(int(i) % POOLED_K != 0 for i in e_eval):
        violations.append("e_eval contains non-shared-persona contexts")
    labels = sorted(subsets, key=lambda k: len(subsets[k]))
    n_prefix = int(manifest["n_prefix"])
    pool_n = int(manifest["pool"]["n"])
    banked_frac = float(manifest["pool"]["banked_fraction"])
    prev: set[int] | None = None
    for label in labels:
        t = subsets[label]
        t_set = set(int(i) for i in t)
        if t_set & e_set:
            violations.append(f"rung {label}: T intersects E_eval")
        if not t_set <= top_set:
            violations.append(f"rung {label}: T not a subset of the top-rung mask")
        if prev is not None and not prev <= t_set:
            violations.append(f"rung {label}: nesting violated (prior subset not contained)")
        prev = t_set
        m = manifest["rungs"][label]
        if int(m["n"]) != len(t):
            violations.append(f"rung {label}: manifest n {m['n']} != realized {len(t)}")
        want = min(int(m["target_size"]), pool_n)
        if label != manifest["top_rung_label"] and abs(len(t) - want) > 1:
            violations.append(f"rung {label}: size {len(t)} != target {want}")
        n_b = int((t < n_prefix).sum())
        if len(t) and abs(n_b - len(t) * banked_frac) > max(2.0, 0.01 * len(t)):
            violations.append(
                f"rung {label}: banked stratum {n_b}/{len(t)} off pool fraction {banked_frac:.4f}"
            )
    if manifest["rungs"][manifest["top_rung_label"]]["n"] != pool_n:
        violations.append("top companion rung != whole pool")
    return violations


def enforce_companion_manifest(
    violations: list[str], eval_dir: pathlib.Path, manifest: dict
) -> None:
    if violations:
        _halt(
            RC_RAND_MANIFEST,
            eval_dir / "ext_rand_manifest_report.json",
            {
                "violations": violations,
                "manifest_summary": {
                    k: v for k, v in manifest.items() if k not in ("rungs", "e_eval")
                },
            },
            f"companion-manifest set-check violation(s): {violations}",
        )


# ── Fit sources (banked + extension stores; layer-column access) ─────────────


class FitSources:
    """Layer-column access over the banked bundle + extension block stores.

    cx: pass_b `cx_last` (contexts 0..4999) + `cx_ext_block*.pt` (payload key
    `cx`). pairs: parent `v_pairs_p{p:02d}.pt` + `v_pairs_ext_p{p:02d}_block*.pt`
    (payload keys `v`/`context_ids`/`span_lengths`). All tensors stay mmap'd;
    `*_col` materializes ONE layer column for the requested ids (fp64).
    """

    def __init__(self) -> None:
        self._cx_blocks: list[tuple[torch.Tensor, np.ndarray]] = []
        self._cx_row: dict[int, tuple[int, int]] = {}
        self._pair_blocks: dict[int, list[tuple[torch.Tensor, np.ndarray]]] = {}
        self._pair_row: dict[int, dict[int, tuple[int, int]]] = {}

    def add_cx_block(self, tensor: torch.Tensor, ctx_ids: np.ndarray) -> None:
        assert tensor.ndim == 3 and tensor.shape[1] == EXPECTED_LAYERS, tensor.shape
        b = len(self._cx_blocks)
        self._cx_blocks.append((tensor, np.asarray(ctx_ids, dtype=np.int64)))
        for j, c in enumerate(ctx_ids):
            self._cx_row[int(c)] = (b, j)

    def add_pair_block(
        self, persona: int, tensor: torch.Tensor, ctx_ids: np.ndarray, span: np.ndarray
    ) -> None:
        assert tensor.ndim == 3 and tensor.shape[1] == EXPECTED_LAYERS, (persona, tensor.shape)
        blocks = self._pair_blocks.setdefault(int(persona), [])
        rows = self._pair_row.setdefault(int(persona), {})
        b = len(blocks)
        blocks.append((tensor, np.asarray(ctx_ids, dtype=np.int64)))
        for j, (c, s) in enumerate(zip(ctx_ids, span)):
            if int(s) > 0:
                rows[int(c)] = (b, j)

    def has_cx(self, ctx_id: int) -> bool:
        return int(ctx_id) in self._cx_row

    def pair_ok(self, ctx_id: int, persona: int) -> bool:
        return int(ctx_id) in self._pair_row.get(int(persona), {})

    def _gather(
        self,
        blocks: list[tuple[torch.Tensor, np.ndarray]],
        row_of: dict[int, tuple[int, int]],
        layer: int,
        ids: np.ndarray,
        key: str,
    ) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), HIDDEN), dtype=np.float64)
        per_block: dict[int, list[tuple[int, int]]] = {}
        for pos, c in enumerate(ids):
            loc = row_of.get(int(c))
            if loc is None:
                raise RuntimeError(f"{key}: no valid store row for context {int(c)}")
            per_block.setdefault(loc[0], []).append((loc[1], pos))
        for b, pairs in per_block.items():
            rows = torch.as_tensor([r for r, _ in pairs], dtype=torch.long)
            block = blocks[b][0][rows, layer, :].to(torch.float64).numpy()
            for (_, pos), vec in zip(pairs, block):
                out[pos] = vec
        return out

    def cx_col(self, layer: int, ids: np.ndarray) -> np.ndarray:
        return self._gather(self._cx_blocks, self._cx_row, layer, ids, "cx")

    def pair_col(self, persona: int, layer: int, ids: np.ndarray) -> np.ndarray:
        return self._gather(
            self._pair_blocks.get(int(persona), []),
            self._pair_row.get(int(persona), {}),
            layer,
            ids,
            f"pairs p{int(persona):02d}",
        )

    def arm_col(self, arm: str, layer: int, ids: np.ndarray) -> np.ndarray:
        """(len(ids), H) fp64 target for `arm`, rows aligned to `ids`."""
        ids = np.asarray(ids, dtype=np.int64)
        if arm == "k1":
            return self.pair_col(0, layer, ids)
        out = np.empty((len(ids), HIDDEN), dtype=np.float64)
        for p in sorted({int(i) % POOLED_K for i in ids}):
            sel = np.flatnonzero(ids % POOLED_K == p)
            out[sel] = self.pair_col(p, layer, ids[sel])
        return out


def load_fit_sources(layout: EXTCAP.Layout, banked: dict) -> FitSources:
    """Assemble FitSources from the staged banked pins + the local ext store."""
    src = FitSources()
    bundle = torch.load(
        str(banked["pass_b_path"]), map_location="cpu", weights_only=False, mmap=True
    )
    cx_last = bundle["cx_last"]
    layers_map = [int(x) for x in bundle["layers"]]
    if layers_map != list(range(EXPECTED_LAYERS)):
        raise RuntimeError(f"pass_b bundle layers map {layers_map} is not identity 0..27")
    assert tuple(cx_last.shape) == (EXTCAP.N_CONTEXTS_FULL, EXPECTED_LAYERS, HIDDEN)
    src.add_cx_block(cx_last, np.arange(EXTCAP.N_CONTEXTS_FULL, dtype=np.int64))

    store_dir = layout.store_dir
    cx_paths = sorted(store_dir.glob("cx_ext_block*.pt"), key=lambda p: p.name)
    for p in cx_paths:
        payload = torch.load(str(p), map_location="cpu", weights_only=True, mmap=True)
        src.add_cx_block(payload["cx"], payload["context_ids"].numpy())

    for pp, path in sorted(banked["parent_store_paths"].items()):
        if not str(pp).startswith("v_pairs_p"):
            continue
        persona = int(str(pp)[len("v_pairs_p") : len("v_pairs_p") + 2])
        payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        src.add_pair_block(
            persona,
            payload["v"],
            payload["context_ids"].numpy(),
            payload["span_lengths"].numpy(),
        )
    for persona in range(POOLED_K):
        for p in sorted(
            store_dir.glob(f"v_pairs_ext_p{persona:02d}_block*.pt"), key=lambda q: q.name
        ):
            payload = torch.load(str(p), map_location="cpu", weights_only=True, mmap=True)
            src.add_pair_block(
                persona,
                payload["v"],
                payload["context_ids"].numpy(),
                payload["span_lengths"].numpy(),
            )
    return src


def realize_rung_masks(
    mask_obj: dict, src: FitSources, smoke: bool, cap: int = SMOKE_MASK_CAP
) -> tuple[dict[str, np.ndarray], dict]:
    """Gen-mask ids intersected with realized store validity (span>0 both arms).

    mask_ext.json rungs carry the GEN-validity two-arm masks; a gen-"ok" row
    whose capture landed zero-span (the parent's `capture_zero_span` class)
    drops here, equalize-down, with per-rung accounting. Under --smoke each
    rung is additionally capped to `cap` ids, shared-persona contexts first
    (scale reduction only — every code path still runs).
    """
    realized: dict[str, np.ndarray] = {}
    drops: dict[str, dict] = {}
    for label, rung in sorted(mask_obj["rungs"].items(), key=lambda kv: int(kv[0])):
        ids = [int(i) for i in rung["ids"]]
        keep, dropped = [], []
        for i in ids:
            if src.pair_ok(i, 0) and src.pair_ok(i, i % POOLED_K) and src.has_cx(i):
                keep.append(i)
            else:
                dropped.append(i)
        if smoke:
            keep = smoke_cap_mask(np.asarray(keep, dtype=np.int64), cap).tolist()
        realized[label] = np.asarray(sorted(keep), dtype=np.int64)
        drops[label] = {
            "n_gen_mask": len(ids),
            "n_realized": len(realized[label]),
            "n_capture_dropped": len(dropped),
            "capture_dropped_ids_first20": dropped[:20],
        }
    return realized, drops


def smoke_cap_mask(ids: np.ndarray, cap: int) -> np.ndarray:
    """Smoke-only rung cap keeping shared-persona (i%16==0) contexts first."""
    ids = np.asarray(sorted(int(i) for i in ids), dtype=np.int64)
    if len(ids) <= cap:
        return ids
    shared = [int(i) for i in ids if i % POOLED_K == 0]
    rest = [int(i) for i in ids if i % POOLED_K != 0]
    keep = (shared + rest)[:cap]
    return np.asarray(sorted(keep), dtype=np.int64)


# ── Fit engine ────────────────────────────────────────────────────────────────


def make_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parent-parity fold scheme: KFold(5, shuffle=True, random_state=0) on n."""
    return [
        (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
        for tr, te in KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(
            np.zeros(n)
        )
    ]


@dataclasses.dataclass
class RungFit:
    """One rung's fitted arrays + records (primary: OOF; companion: E_eval mean).

    ``read_out_solver`` is the REALIZED solver of every read-out-layer read —
    set to the production solver by ``fit_rung`` and flipped to ``"canonical"``
    by ``contingency_refit`` (the ONLY writer), so every downstream read-out
    consumer (rung block label, wide-grid sensitivity, sens_estimator routing)
    keys on ONE authoritative field instead of re-deriving it per call site
    (plan v17 §4.3 step 4; r1 blocker gate-d-contingency-incoherent, r2
    residual).
    """

    tag: str
    label: str
    train_ids: np.ndarray
    eval_ids: np.ndarray
    sres: np.ndarray  # (n_arms, EXPECTED_LAYERS, n_eval)
    stot: np.ndarray
    id_sres: np.ndarray
    id_stot: np.ndarray
    cells: dict
    fit_records: list
    g2_slices: list
    knn: dict
    sens_pure: dict
    solver: str
    fold_ns: list
    read_out_solver: str


def _chunk_fp(
    tag: str, train_ids: np.ndarray, eval_ids: np.ndarray, extra: dict | None = None
) -> dict:
    """Machine-stable chunk fingerprint: id shas + generating parameters only."""
    return {
        "tag": tag,
        "train_sha": _ids_sha(train_ids),
        "eval_sha": _ids_sha(eval_ids),
        "estimator": "gcv-dof-capped",
        "cap_frac": DOF_CAP,
        "grid": list(LAMBDA_GRID_PARAMS),
        "fold_seed": FOLD_SEED,
        "n_folds": N_FOLDS,
        "arms": list(ARM_NAMES),
        "dual_n_max": DUAL_N_MAX,
        **(extra or {}),
    }


def fit_rung(
    tag: str,
    label: str,
    train_ids: np.ndarray,
    src: FitSources,
    dev: torch.device,
    ckpt_dir: pathlib.Path,
    layers: tuple[int, ...] = tuple(range(EXPECTED_LAYERS)),
    eval_ids: np.ndarray | None = None,
    g2_slices: tuple[tuple[int, int], ...] = G2_SLICES,
    sens_pure: bool = False,
    read_out_layers: tuple[int, ...] = READ_OUT_LAYERS,
    fp_extra: dict | None = None,
    layer_chunk: int = LAYER_CHUNK,
) -> RungFit:
    """Fit ONE rung: both arms, all `layers`, 5 folds, ONE BATCHED factorization
    per (layer-chunk, fold) across both arms' targets and the lambda grid
    (plan v17 layer-chunk design — no serial per-cell factorization loop).

    Primary ladder (`eval_ids is None`): out-of-fold scoring at test positions.
    Companion ladder (`eval_ids` given): every fold-fit additionally predicts
    the FIXED eval population; per-context ss_res = mean over the 5 fold-fits
    (registered aggregation), ss_tot centered on the eval population's own
    mean (fold-independent by construction). `fp_extra` threads run-identity
    keys (verified capture-store name-set sha) into the chunk fingerprints.
    """
    train_ids = np.asarray(train_ids, dtype=np.int64)
    is_companion = eval_ids is not None
    ev_ids = np.asarray(eval_ids, dtype=np.int64) if is_companion else train_ids
    n_eval = len(ev_ids)
    folds = make_folds(len(train_ids))
    fold_ns = [int(len(tr)) for tr, _ in folds]
    solver = "dual" if max(fold_ns) <= DUAL_N_MAX else "primal"
    fp = _chunk_fp(tag, train_ids, ev_ids, extra=fp_extra)

    n_arms = len(ARM_NAMES)
    sres = np.full((n_arms, EXPECTED_LAYERS, n_eval), np.nan)
    stot = np.full((n_arms, EXPECTED_LAYERS, n_eval), np.nan)
    id_sres = np.full((n_arms, EXPECTED_LAYERS, n_eval), np.nan)
    id_stot = np.full((n_arms, EXPECTED_LAYERS, n_eval), np.nan)
    cells: dict = {}
    fit_records: list[dict] = []
    g2_records: list[dict] = []
    knn: dict = {}
    sens: dict = {}

    pending: list[int] = []
    for layer in layers:
        name = f"{tag}_{label}_L{layer:02d}"
        if LF.chunk_done(ckpt_dir, name, fp):
            z = np.load(ckpt_dir / f"{name}.npz")
            sres[:, layer, :] = z["sres"]
            stot[:, layer, :] = z["stot"]
            id_sres[:, layer, :] = z["id_sres"]
            id_stot[:, layer, :] = z["id_stot"]
            cells.update(json.loads(str(z["cells"])))
            fit_records.extend(json.loads(str(z["records"])))
            g2_records.extend(json.loads(str(z["g2"])))
            knn.update(json.loads(str(z["knn"])))
            sens.update(json.loads(str(z["sens"])))
            logger.info("[%s %s] resume: layer %d loaded from checkpoint", tag, label, layer)
            continue
        pending.append(layer)

    for c0 in range(0, len(pending), max(layer_chunk, 1)):
        chunk = pending[c0 : c0 + max(layer_chunk, 1)]
        t_chunk = time.monotonic()
        xf = {layer: src.cx_col(layer, train_ids) for layer in chunk}
        yf = {
            layer: {arm: src.arm_col(arm, layer, train_ids) for arm in ARM_NAMES} for layer in chunk
        }
        xe: dict = {}
        ye: dict = {}
        if is_companion:
            xe = {layer: src.cx_col(layer, ev_ids) for layer in chunk}
            ye = {
                layer: {arm: src.arm_col(arm, layer, ev_ids) for arm in ARM_NAMES}
                for layer in chunk
            }
        st: dict[int, dict] = {
            layer: {
                "cells": {},
                "knn": {},
                "sens": {},
                "records": [],
                "g2": [],
                "acc": {arm: [] for arm in ARM_NAMES},
                "sres_acc": (
                    {arm: np.zeros(n_eval) for arm in ARM_NAMES} if is_companion else None
                ),
                "id_acc": ({arm: np.zeros(n_eval) for arm in ARM_NAMES} if is_companion else None),
                "pred_acc": (
                    {arm: np.zeros_like(ye[layer][arm]) for arm in ARM_NAMES}
                    if is_companion and layer in read_out_layers
                    else None
                ),
            }
            for layer in chunk
        }
        for f_idx, (tr, te) in enumerate(folds):
            facts = batched_rung_factorize([xf[layer][tr] for layer in chunk], dev)
            for ci, layer in enumerate(chunk):
                fact = facts[ci]
                s_l = st[layer]
                x_full, y_full = xf[layer], yf[layer]
                x_tr = x_full[tr]
                kev_te = eval_kernel(fact, x_full[te])
                kev_ev = eval_kernel(fact, xe[layer]) if is_companion else None
                for a_idx, arm in enumerate(ARM_NAMES):
                    y_tr = y_full[arm][tr]
                    lam, proj, ymu, dof = solve_capped(fact, y_tr, LAMBDAS, DOF_CAP)
                    if is_companion:
                        y_ev_arm = ye[layer][arm]
                        pred_ev = apply_fit(fact, lam, proj, ymu, kev_ev)
                        s_ev, _ = LF.per_context_ss(pred_ev, y_ev_arm)
                        s_l["sres_acc"][arm] += s_ev
                        if s_l["pred_acc"] is not None:
                            s_l["pred_acc"][arm] += pred_ev
                        id_pred = identity_bias_predict(x_tr, y_tr, xe[layer])
                        s_id, _ = LF.per_context_ss(id_pred, y_ev_arm)
                        s_l["id_acc"][arm] += s_id
                        score_r2 = 1.0 - float(s_ev.sum()) / (
                            float(((y_ev_arm - y_ev_arm.mean(0)) ** 2).sum(1).sum()) + 1e-12
                        )
                        scored_pred, scored_x = pred_ev, xe[layer]
                    else:
                        pred = apply_fit(fact, lam, proj, ymu, kev_te)
                        s_te, t_te = LF.per_context_ss(pred, y_full[arm][te])
                        sres[a_idx, layer, te] = s_te
                        stot[a_idx, layer, te] = t_te
                        id_pred = identity_bias_predict(x_tr, y_tr, x_full[te])
                        s_id, t_id = LF.per_context_ss(id_pred, y_full[arm][te])
                        id_sres[a_idx, layer, te] = s_id
                        id_stot[a_idx, layer, te] = t_id
                        score_r2 = 1.0 - float(s_te.sum()) / (float(t_te.sum()) + 1e-12)
                        scored_pred, scored_x = pred, x_full[te]
                    rec = {
                        "tag": tag,
                        "rung": label,
                        "layer": layer,
                        "fold": f_idx,
                        "arm": arm,
                        "lambda": lam,
                        "dof": dof,
                        "lambda_top_edge": lambda_is_top_edge(lam),
                        "n_train": int(len(tr)),
                        "solver": fact.get("kind", "dual"),
                        "r2": score_r2,
                    }
                    if not is_companion:
                        rec["ss_res"] = float(sres[a_idx, layer, te].sum())
                        rec["ss_tot"] = float(stot[a_idx, layer, te].sum())
                    s_l["records"].append(rec)
                    s_l["acc"][arm].append(rec)
                    if (layer, f_idx) in g2_slices:
                        y_score = ye[layer][arm] if is_companion else y_full[arm][te]
                        pred_ref, lam_ref, _dof_ref = canonical_capped_fit(
                            x_tr, y_tr, scored_x, LAMBDAS, DOF_CAP
                        )
                        s_ref, t_ref = LF.per_context_ss(pred_ref, y_score)
                        r2_ref = 1.0 - float(s_ref.sum()) / (float(t_ref.sum()) + 1e-12)
                        s_l["g2"].append(
                            g2_slice_record(
                                scored_pred,
                                pred_ref,
                                lam,
                                lam_ref,
                                score_r2,
                                r2_ref,
                                layer,
                                f_idx,
                                arm,
                            )
                        )
                    if layer in read_out_layers and not is_companion:
                        s_l["knn"][f"{arm}:L{layer}:fold{f_idx}"] = {
                            m: knn_retrieval(scored_pred, y_full[arm][te], ks=(1, 5), metric=m)
                            for m in ("euclidean", "cosine")
                        }
                    if sens_pure and layer in read_out_layers:
                        lam_p, proj_p, ymu_p, dof_p = solve_capped(fact, y_tr, LAMBDAS, math.inf)
                        pred_pure = apply_fit(fact, lam_p, proj_p, ymu_p, kev_te)
                        s_p, t_p = LF.per_context_ss(pred_pure, y_full[arm][te])
                        s_l["sens"][f"{arm}:L{layer}:fold{f_idx}"] = {
                            "lambda_pure": lam_p,
                            "dof_pure": dof_p,
                            "lambda_capped": lam,
                            "dof_capped": dof,
                            "ss_res_pure": float(s_p.sum()),
                            "ss_tot_pure": float(t_p.sum()),
                            "r2_pure": 1.0 - float(s_p.sum()) / (float(t_p.sum()) + 1e-12),
                        }
                del fact, kev_te, kev_ev
            del facts
        for layer in chunk:
            s_l = st[layer]
            if is_companion:
                for a_idx, arm in enumerate(ARM_NAMES):
                    y_ev_arm = ye[layer][arm]
                    sres[a_idx, layer, :] = s_l["sres_acc"][arm] / N_FOLDS
                    _, t_ev = LF.per_context_ss(np.zeros_like(y_ev_arm), y_ev_arm)
                    stot[a_idx, layer, :] = t_ev
                    id_sres[a_idx, layer, :] = s_l["id_acc"][arm] / N_FOLDS
                    id_stot[a_idx, layer, :] = t_ev
                if s_l["pred_acc"] is not None:
                    # Companion retrieval read: fold-MEAN prediction on E_eval vs the
                    # E_eval pool (chance = k / |E_eval|), per arm at read-out layers.
                    for arm in ARM_NAMES:
                        s_l["knn"][f"{arm}:L{layer}"] = {
                            m: knn_retrieval(
                                s_l["pred_acc"][arm] / N_FOLDS, ye[layer][arm], ks=(1, 5), metric=m
                            )
                            for m in ("euclidean", "cosine")
                        }
            for arm in ARM_NAMES:
                comps = [
                    (c.get("ss_res", np.nan), c.get("ss_tot", np.nan)) for c in s_l["acc"][arm]
                ]
                a_idx = ARM_NAMES.index(arm)
                cell = {
                    "fold_lambdas": [c["lambda"] for c in s_l["acc"][arm]],
                    "fold_dofs": [c["dof"] for c in s_l["acc"][arm]],
                    "fold_r2s": [c["r2"] for c in s_l["acc"][arm]],
                    "n_train_per_fold": [c["n_train"] for c in s_l["acc"][arm]],
                    "solver": solver,
                    "identity_bias_pooled_r2": 1.0
                    - float(np.nansum(id_sres[a_idx, layer]))
                    / (float(np.nansum(id_stot[a_idx, layer])) + 1e-12),
                }
                if is_companion:
                    cell["pooled_r2_eval"] = 1.0 - float(np.nansum(sres[a_idx, layer])) / (
                        float(np.nansum(stot[a_idx, layer])) + 1e-12
                    )
                else:
                    cell["pooled_r2"] = LF.pooled_r2_from_components(comps)
                    cell["fold_mean_r2"] = LF.fold_mean_r2(comps)
                s_l["cells"][f"{arm}:L{layer}"] = cell
            cells.update(s_l["cells"])
            fit_records.extend(s_l["records"])
            g2_records.extend(s_l["g2"])
            knn.update(s_l["knn"])
            sens.update(s_l["sens"])
            LF.save_chunk(
                ckpt_dir,
                f"{tag}_{label}_L{layer:02d}",
                {
                    "sres": sres[:, layer, :],
                    "stot": stot[:, layer, :],
                    "id_sres": id_sres[:, layer, :],
                    "id_stot": id_stot[:, layer, :],
                    "cells": np.array(json.dumps(s_l["cells"])),
                    "records": np.array(json.dumps(s_l["records"])),
                    "g2": np.array(json.dumps(s_l["g2"])),
                    "knn": np.array(json.dumps(s_l["knn"])),
                    "sens": np.array(json.dumps(s_l["sens"])),
                },
                fp,
            )
            print(
                f"[fits] unit {layers.index(layer) + 1}/{len(layers)} {tag}/{label}:L{layer} "
                f"elapsed={time.monotonic() - t_chunk:.1f}s (chunk of {len(chunk)})",
                flush=True,
            )
    return RungFit(
        tag=tag,
        label=label,
        train_ids=train_ids,
        eval_ids=ev_ids,
        sres=sres,
        stot=stot,
        id_sres=id_sres,
        id_stot=id_stot,
        cells=cells,
        fit_records=fit_records,
        g2_slices=g2_records,
        knn=knn,
        sens_pure=sens,
        solver=solver,
        fold_ns=fold_ns,
        read_out_solver=solver,
    )


def wide_grid_sensitivity(
    rf: RungFit, src: FitSources, dev: torch.device, read_out_layers=READ_OUT_LAYERS
) -> dict:
    """Labeled wider-grid re-selection (logspace(-2,8,21)) at read-out layers.

    Fired per rung per ladder when > LAMBDA_EDGE_FRACTION of that rung's
    production fits selected the TOP grid edge; reported alongside — never
    silently swapped for — the primary-grid results.

    Routes through the REALIZED read-out solver (``rf.read_out_solver``): after
    a fired Gate-D contingency the production solver FAILED parity at exactly
    these layers, so the wide-grid re-selection runs ``canonical_capped_fit``
    (plan v17 §4.3 step 4; r2 reconcile residual a1). The block and every cell
    carry an explicit ``solver`` label either way.
    """
    folds = make_folds(len(rf.train_ids))
    is_companion = not np.array_equal(rf.eval_ids, rf.train_ids)
    canonical = rf.read_out_solver == "canonical"
    out: dict = {"grid": list(LAMBDA_WIDE_PARAMS), "cells": {}, "solver": rf.read_out_solver}
    for layer in read_out_layers:
        x_full = src.cx_col(layer, rf.train_ids)
        y_full = {arm: src.arm_col(arm, layer, rf.train_ids) for arm in ARM_NAMES}
        if is_companion:
            x_ev = src.cx_col(layer, rf.eval_ids)
            y_ev = {arm: src.arm_col(arm, layer, rf.eval_ids) for arm in ARM_NAMES}
        for f_idx, (tr, te) in enumerate(folds):
            if not canonical:
                fact = rung_factorize(x_full[tr], dev)
                kev = eval_kernel(fact, x_ev if is_companion else x_full[te])
            for arm in ARM_NAMES:
                x_score = x_ev if is_companion else x_full[te]
                y_score = y_ev[arm] if is_companion else y_full[arm][te]
                if canonical:
                    pred, lam, dof = canonical_capped_fit(
                        x_full[tr], y_full[arm][tr], x_score, LAMBDAS_WIDE, DOF_CAP
                    )
                else:
                    lam, proj, ymu, dof = solve_capped(fact, y_full[arm][tr], LAMBDAS_WIDE, DOF_CAP)
                    pred = apply_fit(fact, lam, proj, ymu, kev)
                s, t = LF.per_context_ss(pred, y_score)
                out["cells"][f"{arm}:L{layer}:fold{f_idx}"] = {
                    "lambda_wide": lam,
                    "dof_wide": dof,
                    "lambda_top_edge_wide": lambda_is_top_edge(lam, LAMBDAS_WIDE),
                    "r2_wide": 1.0 - float(s.sum()) / (float(t.sum()) + 1e-12),
                    "solver": rf.read_out_solver,
                }
    return out


def contingency_refit(rf: RungFit, src: FitSources, read_out_layers=READ_OUT_LAYERS) -> list[dict]:
    """Gate D contingency: canonical per-fit solver at the read-out layers.

    Rebuilds EVERY read-out-layer read from the canonical solver — sres/stot,
    fit_records, cells (lambda/dof/R2), knn — so no downstream consumer
    (rung_block, lambda-edge trigger, figures, summary) silently mixes the
    parity-FAILED production estimator with canonical predictions (r1 blocker
    gate-d-contingency-incoherent). Fits at NON-read-out layers are retained
    (labeled by the caller via `read_out_solver`/`contingency_fired`); the
    identity-bias reads are solver-independent and kept. Small battery
    (|read_out_layers| x 5 folds x 2 arms canonical numpy fits) — serial by
    design; returns the refit records. Also flips ``rf.read_out_solver`` to
    ``"canonical"`` — the single authoritative field downstream read-out
    consumers (wide_grid_sensitivity, sens_estimator_block, the rung block)
    key on (r2 residual: the pre-fix code left them on the parity-failed
    production solver while the rung block asserted "canonical").
    """
    folds = make_folds(len(rf.train_ids))
    is_companion = not np.array_equal(rf.eval_ids, rf.train_ids)
    records: list[dict] = []
    kept_records = [r for r in rf.fit_records if r["layer"] not in read_out_layers]
    for layer in read_out_layers:
        x_full = src.cx_col(layer, rf.train_ids)
        y_full = {arm: src.arm_col(arm, layer, rf.train_ids) for arm in ARM_NAMES}
        if is_companion:
            x_ev = src.cx_col(layer, rf.eval_ids)
            y_ev = {arm: src.arm_col(arm, layer, rf.eval_ids) for arm in ARM_NAMES}
            sres_acc = {arm: np.zeros(len(rf.eval_ids)) for arm in ARM_NAMES}
            pred_acc = {arm: np.zeros_like(y_ev[arm]) for arm in ARM_NAMES}
        acc: dict[str, list] = {arm: [] for arm in ARM_NAMES}
        for f_idx, (tr, te) in enumerate(folds):
            for a_idx, arm in enumerate(ARM_NAMES):
                x_score = x_ev if is_companion else x_full[te]
                pred, lam, dof = canonical_capped_fit(
                    x_full[tr], y_full[arm][tr], x_score, LAMBDAS, DOF_CAP
                )
                if is_companion:
                    s, _ = LF.per_context_ss(pred, y_ev[arm])
                    sres_acc[arm] += s
                    pred_acc[arm] += pred
                    score_r2 = 1.0 - float(s.sum()) / (
                        float(((y_ev[arm] - y_ev[arm].mean(0)) ** 2).sum(1).sum()) + 1e-12
                    )
                else:
                    s, t = LF.per_context_ss(pred, y_full[arm][te])
                    rf.sres[a_idx, layer, te] = s
                    rf.stot[a_idx, layer, te] = t
                    score_r2 = 1.0 - float(s.sum()) / (float(t.sum()) + 1e-12)
                    rf.knn[f"{arm}:L{layer}:fold{f_idx}"] = {
                        m: knn_retrieval(pred, y_full[arm][te], ks=(1, 5), metric=m)
                        for m in ("euclidean", "cosine")
                    }
                rec = {
                    "tag": rf.tag,
                    "rung": rf.label,
                    "layer": layer,
                    "fold": f_idx,
                    "arm": arm,
                    "lambda": lam,
                    "dof": dof,
                    "lambda_top_edge": lambda_is_top_edge(lam),
                    "n_train": int(len(tr)),
                    "solver": "canonical",
                    "r2": score_r2,
                }
                if not is_companion:
                    rec["ss_res"] = float(s.sum())
                    rec["ss_tot"] = float(t.sum())
                records.append(rec)
                acc[arm].append(rec)
        if is_companion:
            for a_idx, arm in enumerate(ARM_NAMES):
                rf.sres[a_idx, layer, :] = sres_acc[arm] / len(folds)
                rf.knn[f"{arm}:L{layer}"] = {
                    m: knn_retrieval(pred_acc[arm] / len(folds), y_ev[arm], ks=(1, 5), metric=m)
                    for m in ("euclidean", "cosine")
                }
        for arm in ARM_NAMES:
            a_idx = ARM_NAMES.index(arm)
            cell = {
                "fold_lambdas": [c["lambda"] for c in acc[arm]],
                "fold_dofs": [c["dof"] for c in acc[arm]],
                "fold_r2s": [c["r2"] for c in acc[arm]],
                "n_train_per_fold": [c["n_train"] for c in acc[arm]],
                "solver": "canonical",
                "identity_bias_pooled_r2": rf.cells[f"{arm}:L{layer}"]["identity_bias_pooled_r2"],
            }
            if is_companion:
                cell["pooled_r2_eval"] = 1.0 - float(np.nansum(rf.sres[a_idx, layer])) / (
                    float(np.nansum(rf.stot[a_idx, layer])) + 1e-12
                )
            else:
                comps = [(c["ss_res"], c["ss_tot"]) for c in acc[arm]]
                cell["pooled_r2"] = LF.pooled_r2_from_components(comps)
                cell["fold_mean_r2"] = LF.fold_mean_r2(comps)
            rf.cells[f"{arm}:L{layer}"] = cell
    rf.fit_records[:] = kept_records + records
    rf.read_out_solver = "canonical"
    return records


def sens_estimator_canonical(
    rf: RungFit,
    src: FitSources,
    contingency_records: list[dict],
    read_out_layers=READ_OUT_LAYERS,
) -> dict:
    """Recompute the sens_estimator (capped vs pure-GCV paired slots) CANONICALLY
    after a fired Gate-D contingency (plan v17 §4.3 step 4; r2 residual a2:
    `rf.sens_pure` was computed pre-contingency by the parity-FAILED production
    solver and persisted under a rung block asserting canonical).

    The CAPPED leg's (lambda, dof) are looked up from the contingency's own
    refit records (identical inputs + solver — recomputing them would double
    the SVD count for bit-identical results); only the PURE leg (cap_frac=inf)
    is refit here. Primary rungs only (sens_flag is rung-1 primary by design);
    fails loud on a companion-shaped RungFit or a missing contingency record.
    """
    if not np.array_equal(rf.eval_ids, rf.train_ids):
        raise RuntimeError(
            "sens_estimator_canonical: sens_estimator is a primary-ladder deliverable "
            f"(train_ids == eval_ids); got companion-shaped rung {rf.tag}/{rf.label}"
        )
    by_slot = {(r["layer"], r["fold"], r["arm"]): r for r in contingency_records}
    folds = make_folds(len(rf.train_ids))
    out: dict = {}
    for layer in read_out_layers:
        x_full = src.cx_col(layer, rf.train_ids)
        y_full = {arm: src.arm_col(arm, layer, rf.train_ids) for arm in ARM_NAMES}
        for f_idx, (tr, te) in enumerate(folds):
            for arm in ARM_NAMES:
                rec = by_slot.get((layer, f_idx, arm))
                if rec is None:
                    raise RuntimeError(
                        "sens_estimator_canonical: no contingency refit record for "
                        f"(L{layer}, fold{f_idx}, {arm}) on {rf.tag}/{rf.label} — "
                        "the capped leg must come from the fired contingency's own fits"
                    )
                pred_pure, lam_p, dof_p = canonical_capped_fit(
                    x_full[tr], y_full[arm][tr], x_full[te], LAMBDAS, math.inf
                )
                s_p, t_p = LF.per_context_ss(pred_pure, y_full[arm][te])
                out[f"{arm}:L{layer}:fold{f_idx}"] = {
                    "lambda_pure": lam_p,
                    "dof_pure": dof_p,
                    "lambda_capped": rec["lambda"],
                    "dof_capped": rec["dof"],
                    "ss_res_pure": float(s_p.sum()),
                    "ss_tot_pure": float(t_p.sum()),
                    "r2_pure": 1.0 - float(s_p.sum()) / (float(t_p.sum()) + 1e-12),
                    "solver": "canonical",
                }
    return out


def sens_estimator_block(
    rf: RungFit, src: FitSources, contingency_records: list[dict]
) -> tuple[dict, str]:
    """Route the persisted sens_estimator through the REALIZED read-out solver.

    Contingency fired => canonical recompute (capped leg from the contingency's
    own records, pure leg refit canonically); not fired => the production-solver
    `rf.sens_pure` computed inside fit_rung. Returns (slots, solver_label) so
    phase_fits persists an explicit `sens_estimator_solver` either way
    (r2 reconcile residual a2).
    """
    if contingency_records:
        return sens_estimator_canonical(rf, src, contingency_records), rf.read_out_solver
    return rf.sens_pure, rf.read_out_solver


# ── Row coverage + rung-dir outputs + paired-script runner ───────────────────


def row_coverage_check(rf: RungFit, layers: tuple[int, ...]) -> None:
    """Plan section-3 duty: every registered (context x arm x layer) row present
    + finite BEFORE any paired statistic or bootstrap. Fail loud, never quiet."""
    for a_idx, arm in enumerate(ARM_NAMES):
        for layer in layers:
            row = rf.sres[a_idx, layer, :]
            if not np.isfinite(row).all():
                raise RuntimeError(
                    f"row-coverage: missing per-context rows for {rf.tag}/{rf.label} "
                    f"arm={arm} L={layer} — refusing to compute any paired statistic "
                    "on an incomplete (context x arm) key set"
                )


def write_rung_dir(
    rung_dir: pathlib.Path,
    rf: RungFit,
    src: FitSources,
    n_total: int,
    metadata: dict,
    diff_train_ids: np.ndarray | None = None,
) -> dict:
    """Write the paired-script input contract into `rung_dir`.

    Files (the reused script's own read set): percontext_ladder.npz
    (arm_names/context_ids/p1_ss_res/p1_ss_tot + identity arrays),
    assignment.json (per-context persona arrays over 0..n_total-1),
    ladder_analysis_summary.json (per-arm-layer between-persona mean-shift
    energy E via the extracted shared function), mixture_diffs.npz (the
    --full-ratio-ci sidecar; schema owned by `load_mixture_diffs`).
    `diff_train_ids` selects the DENOMINATOR group population (companion: T_r;
    primary: the rung mask itself).

    mixture_diffs.npz stays POD-LOCAL (only the paired script's --full-ratio-ci
    leg reads it, on the same pod); the plan-§6 correlated-offset floor is
    computed HERE from the same groups and returned compact
    (``{"implied", "floor"}``) so `phase_fits` threads it into the portable
    ladder_ext_r2.json rung blocks — the figures/summary phase on the VM never
    reads the npz sidecars (r1 blocker fits-analysis-handoff).
    """
    rung_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        rung_dir / "percontext_ladder.npz",
        arm_names=np.array(list(ARM_NAMES)),
        context_ids=rf.eval_ids.astype(np.int64),
        p1_ss_res=rf.sres,
        p1_ss_tot=rf.stot,
        p1_identity_ss_res=rf.id_sres,
        p1_identity_ss_tot=rf.id_stot,
    )
    assign = {
        str(1): [0] * n_total,
        str(POOLED_K): [int(i) % POOLED_K for i in range(n_total)],
    }
    write_json(
        rung_dir / "assignment.json",
        {
            "registered_rule": "persona(i, k) = i mod k",
            "arms": assign,
            "metadata": {"n_total": n_total, "rung": rf.label, "tag": rf.tag},
        },
    )

    group_ids = rf.train_ids if diff_train_ids is None else np.asarray(diff_train_ids, np.int64)
    personas_rows: list[int] = []
    ctx_rows: list[int] = []
    for i in group_ids:
        p = int(i) % POOLED_K
        if p != 0:
            personas_rows.append(p)
            ctx_rows.append(int(i))
    order = np.argsort(np.asarray(personas_rows), kind="stable")
    personas_arr = np.asarray(personas_rows, dtype=np.int64)[order]
    ctx_arr = np.asarray(ctx_rows, dtype=np.int64)[order]
    n_p0 = int(sum(1 for i in group_ids if int(i) % POOLED_K == 0))
    diffs = np.empty((len(ctx_arr), len(READ_OUT_LAYERS), HIDDEN), dtype=np.float32)
    implied: dict[str, dict] = {}
    floor: dict[str, dict] = {}
    for li, layer in enumerate(READ_OUT_LAYERS):
        v0 = src.pair_col(0, layer, ctx_arr)
        for p in sorted(set(personas_arr.tolist())):
            sel = np.flatnonzero(personas_arr == p)
            vp = src.pair_col(int(p), layer, ctx_arr[sel])
            diffs[sel, li, :] = (vp - v0[sel]).astype(np.float32)
        groups = [
            (int((personas_arr == p).sum()), diffs[personas_arr == p, li, :].astype(np.float64))
            for p in sorted(set(personas_arr.tolist()))
        ]
        e_val = mixture_energy_from_group_diffs(iter(groups), n_p0)
        implied[f"k{POOLED_K}:L{layer}"] = {"between_persona_mean_shift_energy": float(e_val)}
        floor[f"L{layer}"] = correlated_floor_from_groups(iter(groups), n_p0)
    np.savez(
        rung_dir / "mixture_diffs.npz",
        layers=np.asarray(list(READ_OUT_LAYERS), dtype=np.int64),
        **{
            f"k{POOLED_K}_diffs": diffs,
            f"k{POOLED_K}_personas": personas_arr,
            f"k{POOLED_K}_n_persona0": np.int64(n_p0),
            f"k{POOLED_K}_context_ids": ctx_arr,
        },
    )
    write_json(
        rung_dir / "ladder_analysis_summary.json",
        {"mixture_floor": {"implied_mixture_penalty": implied}, "metadata": metadata},
    )
    return {"implied": implied, "floor": floor}


_PAIRED_SHIM = (
    "import pathlib, sys\n"
    "root = pathlib.Path(sys.argv[1])\n"
    "sys.path.insert(0, str(root))\n"
    "from scripts import issue823_shared_persona_paired as SPP\n"
    "SPP.repo_root = lambda: root  # branch-guarded task_workflow resolver is pod-unsafe\n"
    "sys.argv = ['issue823_shared_persona_paired.py'] + sys.argv[2:]\n"
    "SPP.main()\n"
)


def run_paired_script(
    repo_root: pathlib.Path,
    out_path: pathlib.Path,
    ladder_dir: pathlib.Path | None,
    arms: str | None,
    n_boot: int | None,
    full_ratio_ci: bool,
) -> None:
    """Run the UNMODIFIED-content paired script through the pod-safe shim.

    `scripts/issue823_shared_persona_paired.py` bytes are untouched; the shim
    only injects the clone root into its `repo_root` seam (the task_workflow
    resolver branch-guards to `main` + requires `tasks/`, both false pod-side).
    `ladder_dir=None` runs the banked DEFAULT path (Gate E(i) loader rerun).
    """
    cmd = [sys.executable, "-c", _PAIRED_SHIM, str(repo_root), "--out", str(out_path)]
    if ladder_dir is not None:
        cmd += ["--ladder-dir", str(ladder_dir)]
    if arms is not None:
        cmd += ["--arms", arms]
    if n_boot is not None:
        cmd += ["--n-boot", str(n_boot)]
    if full_ratio_ci:
        cmd += ["--full-ratio-ci"]
    logger.info("[paired] %s", " ".join(cmd[3:]))
    subprocess.run(cmd, check=True, env={**os.environ}, cwd=str(repo_root))
    if not out_path.exists():
        raise RuntimeError(f"paired script exited 0 but wrote no output at {out_path}")


# ── P2-ext boundary ladder ────────────────────────────────────────────────────


def p2_rung_grid(realized_max: int) -> list[int]:
    """The registered n_train grid clamped to the realized pool, + realized-max."""
    grid = [n for n in P2_N_TRAIN_GRID if n <= realized_max]
    if realized_max not in grid:
        grid.append(int(realized_max))
    return sorted(grid)


def p2_boundary_ladder(
    src: FitSources,
    k1_valid_ids: np.ndarray,
    dev: torch.device,
    ckpt_dir: pathlib.Path,
    holdout_n: int = P2_HOLDOUT_N,
    smoke: bool = False,
    fp_extra: dict | None = None,
) -> dict:
    """P2-ext (plan 4.3 step 10): k=1 arm, read-out layers, fixed holdout =
    the LAST `holdout_n` contexts of the realized top-rung k1-valid mask;
    5 seeded draws per n_train rung across the d boundary; same dof-capped GCV.

    The 5 seed draws of a (layer, n_train) cell share ONE batched eigh stack
    (same-n slices — the plan v17 layer-chunk design applied to the seed axis),
    and each (layer, n_train) cell checkpoints atomically via LF.save_chunk
    with a machine-stable fingerprint (id shas + generating params + store
    identity), so a crashed P2 resumes instead of refitting (r1 concern
    p2-not-resumable).
    """
    ids = np.asarray(sorted(int(i) for i in k1_valid_ids), dtype=np.int64)
    if smoke:
        holdout_n = max(4, min(holdout_n, len(ids) // 3))
    if len(ids) <= holdout_n + N_FOLDS:
        raise RuntimeError(f"P2-ext: k1-valid mask {len(ids)} too small for holdout {holdout_n}")
    holdout = ids[-holdout_n:]
    pool = ids[:-holdout_n]
    grid = p2_rung_grid(len(pool))
    if smoke:
        grid = grid[:2]
    out: dict = {
        "arm": "k1",
        "read_out_layers": list(READ_OUT_LAYERS),
        "holdout_n": int(len(holdout)),
        "holdout_sha256": _ids_sha(holdout),
        "pool_n": int(len(pool)),
        "pool_sha256": _ids_sha(pool),
        "n_train_grid": grid,
        "draw_seeds": list(P2_DRAW_SEEDS),
        "d": HIDDEN,
        "estimator": f"gcv-dof-capped-{DOF_CAP}",
        "cells": {},
    }
    base_fp = {
        "phase": "p2_ext",
        "holdout_sha": _ids_sha(holdout),
        "pool_sha": _ids_sha(pool),
        "seeds": list(P2_DRAW_SEEDS),
        "estimator": "gcv-dof-capped",
        "cap_frac": DOF_CAP,
        "grid": list(LAMBDA_GRID_PARAMS),
        **(fp_extra or {}),
    }
    for layer in READ_OUT_LAYERS:
        x_pool = y_pool = x_hold = y_hold = None
        for n_train in grid:
            name = f"p2_L{layer:02d}_n{n_train}"
            cell_fp = {**base_fp, "layer": int(layer), "n_train": int(n_train)}
            if LF.chunk_done(ckpt_dir, name, cell_fp):
                z = np.load(ckpt_dir / f"{name}.npz")
                out["cells"].update(json.loads(str(z["cells"])))
                logger.info("[p2] resume: L%d n%d loaded from checkpoint", layer, n_train)
                continue
            t_cell = time.monotonic()
            if x_pool is None:
                x_pool = src.cx_col(layer, pool)
                y_pool = src.arm_col("k1", layer, pool)
                x_hold = src.cx_col(layer, holdout)
                y_hold = src.arm_col("k1", layer, holdout)
            sels = []
            for seed in P2_DRAW_SEEDS:
                rng = np.random.default_rng(seed)
                sels.append(np.sort(rng.choice(len(pool), size=n_train, replace=False)))
            facts = batched_rung_factorize([x_pool[sel] for sel in sels], dev)
            cell_out: dict = {}
            for seed, sel, fact in zip(P2_DRAW_SEEDS, sels, facts):
                x_tr, y_tr = x_pool[sel], y_pool[sel]
                lam, proj, ymu, dof = solve_capped(fact, y_tr, LAMBDAS, DOF_CAP)
                pred = apply_fit(fact, lam, proj, ymu, eval_kernel(fact, x_hold))
                s, t = LF.per_context_ss(pred, y_hold)
                id_pred = identity_bias_predict(x_tr, y_tr, x_hold)
                s_id, t_id = LF.per_context_ss(id_pred, y_hold)
                cell_out[f"L{layer}:n{n_train}:seed{seed}"] = {
                    "r2": 1.0 - float(s.sum()) / (float(t.sum()) + 1e-12),
                    "lambda": lam,
                    "dof": dof,
                    "lambda_top_edge": lambda_is_top_edge(lam),
                    "n_train": int(n_train),
                    "n_over_d": n_train / HIDDEN,
                    "solver": fact.get("kind", "dual"),
                    "identity_bias_r2": 1.0 - float(s_id.sum()) / (float(t_id.sum()) + 1e-12),
                    "knn": {
                        m: knn_retrieval(pred, y_hold, ks=(1, 5), metric=m)
                        for m in ("euclidean", "cosine")
                    },
                }
            del facts
            out["cells"].update(cell_out)
            LF.save_chunk(ckpt_dir, name, {"cells": np.array(json.dumps(cell_out))}, cell_fp)
            unit_idx = READ_OUT_LAYERS.index(layer) * len(grid) + grid.index(n_train) + 1
            print(
                f"[fits] unit {unit_idx}/{len(READ_OUT_LAYERS) * len(grid)} "
                f"p2/L{layer}/n{n_train} "
                f"elapsed={time.monotonic() - t_cell:.1f}s ({len(P2_DRAW_SEEDS)} seeds batched)",
                flush=True,
            )
    return out


def sens_dedup_block(
    gate_e_dup: dict,
    rung_masks: dict,
    rung_labels: list[str],
    src: FitSources,
    dev: torch.device,
    ckpt_dir: pathlib.Path,
    fp_extra: dict | None = None,
) -> dict | None:
    """Dedup-mask sensitivity refit (P0 gate-(e) consumer; r1 concern
    dedup-sensitivity-detached). Returns None when the flag did not fire;
    raises when the flag fired without duplicate_groups (stale P0 report).
    Extracted from phase_fits so the consumer path is unit-testable (Claude r2
    Minor: the fired-flag branch had no test reaching fit_rung + block shape).
    """
    if not bool(gate_e_dup.get("dedup_sensitivity_refit_required")):
        return None
    groups = gate_e_dup.get("duplicate_groups") or []
    if not groups:
        raise RuntimeError(
            "gate (e) flagged the dedup sensitivity refit but the P0 report carries no "
            "duplicate_groups — regenerate the P0 report (p0ext) before fits"
        )
    dup_drop = {
        int(i) for g in groups for i in g["context_ids"] if int(i) != int(g["representative"])
    }
    sens_block: dict = {
        "n_dropped_duplicates": len(dup_drop),
        "read_out_layers": list(READ_OUT_LAYERS),
        "rungs": {},
    }
    for label in rung_labels:
        keep = np.asarray(
            [int(i) for i in rung_masks[label] if int(i) not in dup_drop],
            dtype=np.int64,
        )
        rf_d = fit_rung(
            "sens_dedup",
            label,
            keep,
            src,
            dev,
            ckpt_dir,
            layers=READ_OUT_LAYERS,
            g2_slices=(),
            fp_extra=fp_extra,
        )
        sens_block["rungs"][label] = {
            "n_mask": int(len(keep)),
            "n_dropped_from_rung": int(len(rung_masks[label]) - len(keep)),
            "cells": {
                k: {
                    "pooled_r2": v.get("pooled_r2"),
                    "fold_lambdas": v["fold_lambdas"],
                    "solver": v["solver"],
                }
                for k, v in rf_d.cells.items()
            },
        }
    return sens_block


# ── Fingerprint sentinel ──────────────────────────────────────────────────────

REQUIRED_OUTPUT_KEYS = {
    "percontext_rung": ["arm_names", "context_ids", "p1_ss_res", "p1_ss_tot"],
    "shared_persona_paired": [
        "mean_paired_diff_ci95",
        "rho_ci95",
        "n_negligible_E_draws",
        "offset_bias_control",
    ],
    "ladder_ext_r2": ["primary", "companion", "gates", "estimator"],
    "ladder_ext_r2_rung": [
        "n_mask",
        "n_eval",
        "n_train_per_fold",
        "d",
        "n_over_d_ratio",
        "solver",
        "g2_verdict",
        "lambda_edge_fraction",
        "cells",
        "knn_read_out",
        "estimator_degenerate",
        "contingency_fired",
        "read_out_solver",
        "correlated_offset_floor",
    ],
    "correlated_offset_floor_layer": [
        "floor_raw",
        "e_point_from_diffs",
        "floor_ratio",
        "n_nonzero",
        "n_persona0",
    ],
    "p2_ext_boundary": [
        "cells",
        "holdout_sha256",
        "n_train_grid",
        "read_out_layers",
        "draw_seeds",
    ],
    "g2_ext_report": ["rungs", "tolerances", "threeway_rung1"],
}


def fits_fingerprint(
    rung_mask_shas: dict[str, str],
    rand_manifest_sha: str,
    gate_states: dict,
    store_name_set_sha256: str | None = None,
) -> dict:
    return {
        "rung_mask_shas": dict(sorted(rung_mask_shas.items())),
        "rand_manifest_sha": rand_manifest_sha,
        "store_name_set_sha256": store_name_set_sha256,
        "code_sha": as_metadata_dict(git_provenance())["git_commit"],
        "estimator": {
            "primary": f"gcv-dof-capped-{DOF_CAP}",
            "grid": list(LAMBDA_GRID_PARAMS),
            "wide_grid": list(LAMBDA_WIDE_PARAMS),
            "fold_seed": FOLD_SEED,
            "n_folds": N_FOLDS,
            "dual_n_max": DUAL_N_MAX,
            "arms": list(ARM_NAMES),
        },
        "required_output_keys": REQUIRED_OUTPUT_KEYS,
        "gate_states": gate_states,
    }


def write_fits_sentinel(eval_dir: pathlib.Path, fingerprint: dict, extra: dict) -> pathlib.Path:
    """Atomic (tmp + rename) completion sentinel — written ONLY on full success."""
    path = eval_dir / FITS_SENTINEL
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"complete": True, "fingerprint": fingerprint, **extra}, indent=1) + "\n"
    )
    tmp.replace(path)
    return path


def fits_done(eval_dir: pathlib.Path, fingerprint: dict) -> bool:
    """Resume predicate: fingerprint EQUALITY, never bare existence."""
    path = eval_dir / FITS_SENTINEL
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not d.get("complete"):
        return False
    if d.get("fingerprint") != fingerprint:
        logger.warning("[fits] sentinel fingerprint mismatch — routing to refit")
        return False
    return True


def _validated_store_identity(layout) -> dict:
    """Parse + VALIDATE the P-Store-ext sentinel before any fit (r1 concern
    sentinel-before-upload, fits side): existence alone proves nothing — the
    sentinel must be complete, name the layout's own HF prefix, and its
    name-set sha must equal the sha recomputed over the LOCAL store's expected
    file set (manifest-first enumeration), else the store the fits would
    consume is not the store P-Store-ext verified.
    """
    path = layout.store_dir / EXTCAP.STORE_SENTINEL
    want_prefix = layout.hf_path(layout.store_subpath)
    if not path.exists():
        raise RuntimeError(
            f"{path} missing — P-Store-ext must verify the store before fits "
            f"(HF prefix: {want_prefix})"
        )
    d = json.loads(path.read_text())
    if not d.get("complete"):
        raise RuntimeError(f"{path}: store sentinel is not complete=True — rerun P-Store-ext")
    if d.get("hf_prefix") != want_prefix:
        raise RuntimeError(
            f"{path}: sentinel hf_prefix {d.get('hf_prefix')!r} != layout prefix "
            f"{want_prefix!r} — wrong store for this layout"
        )
    local_names = EXTCAP.expected_store_files(layout.store_dir)
    local_sha = EXTCAP._sha256_json(local_names)
    if d.get("name_set_sha256") != local_sha:
        raise RuntimeError(
            f"{path}: sentinel name_set_sha256 does not match the LOCAL store's expected-file "
            f"set ({len(local_names)} files) — the store drifted since P-Store-ext verified "
            "it; rerun the storeext phase"
        )
    return d


def _load_output_json(eval_dir: pathlib.Path, name: str, problems: list[str]):
    """Load one output JSON for validation; records missing/unreadable and returns None."""
    p = eval_dir / name
    if not p.exists():
        problems.append(f"missing {name}")
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        problems.append(f"unreadable {name}: {exc}")
        return None


def _validate_rung_block(
    tag: str, label: str, blk, sens_rung: bool, problems: list[str]
) -> int | None:
    """Exact-topology validation of ONE ladder_ext_r2 rung block; returns n_eval."""
    where = f"ladder_ext_r2.json:{tag}/{label}"
    if not isinstance(blk, dict):
        problems.append(f"{where}: rung block is not a dict")
        return None
    miss = [k for k in REQUIRED_OUTPUT_KEYS["ladder_ext_r2_rung"] if k not in blk]
    if miss:
        problems.append(f"{where}: missing fields {miss}")
        return None
    want_cells = {f"{arm}:L{layer}" for arm in ARM_NAMES for layer in range(EXPECTED_LAYERS)}
    cells = blk["cells"] if isinstance(blk["cells"], dict) else {}
    if set(cells) != want_cells:
        problems.append(
            f"{where}: cells cardinality {len(cells)} != {len(want_cells)} (arm x layer)"
        )
    else:
        pooled_key = "pooled_r2" if tag == "primary" else "pooled_r2_eval"
        cell_req = ("fold_lambdas", "fold_dofs", "solver", "identity_bias_pooled_r2", pooled_key)
        bad = sorted(k for k, c in cells.items() if any(f not in c for f in cell_req))
        if bad:
            problems.append(f"{where}: cells missing per-cell fields at {bad[:4]}")
    if tag == "primary":
        want_knn = {
            f"{arm}:L{layer}:fold{f}"
            for arm in ARM_NAMES
            for layer in READ_OUT_LAYERS
            for f in range(N_FOLDS)
        }
    else:
        want_knn = {f"{arm}:L{layer}" for arm in ARM_NAMES for layer in READ_OUT_LAYERS}
    knn = blk["knn_read_out"] if isinstance(blk["knn_read_out"], dict) else {}
    if not want_knn <= set(knn):
        problems.append(f"{where}: knn_read_out missing {sorted(want_knn - set(knn))[:4]}")
    floor = blk["correlated_offset_floor"]
    for layer in READ_OUT_LAYERS:
        lf = floor.get(f"L{layer}") if isinstance(floor, dict) else None
        if not isinstance(lf, dict) or any(
            k not in lf for k in REQUIRED_OUTPUT_KEYS["correlated_offset_floor_layer"]
        ):
            problems.append(f"{where}: correlated_offset_floor L{layer} incomplete")
    # Solver coherence: the gate-d-contingency-incoherent invariant — the rung
    # label, the wide-grid block, and the sens_estimator must all carry the
    # REALIZED read-out solver.
    fired = bool(blk["contingency_fired"])
    ros = blk["read_out_solver"]
    if fired != (ros == "canonical"):
        problems.append(f"{where}: contingency_fired={fired} but read_out_solver={ros!r}")
    want_slots = {
        f"{arm}:L{layer}:fold{f}"
        for arm in ARM_NAMES
        for layer in READ_OUT_LAYERS
        for f in range(N_FOLDS)
    }
    wide = blk.get("wide_grid_sensitivity")
    if wide is not None:
        if not isinstance(wide, dict) or wide.get("solver") != ros:
            problems.append(f"{where}: wide_grid_sensitivity solver != read_out_solver {ros!r}")
        wcells = (wide.get("cells") or {}) if isinstance(wide, dict) else {}
        if set(wcells) != want_slots:
            problems.append(f"{where}: wide_grid_sensitivity cells incomplete")
        else:
            bad_w = sorted(k for k, c in wcells.items() if c.get("solver") != ros)
            if bad_w:
                problems.append(f"{where}: wide cells solver != {ros!r} at {bad_w[:4]}")
    if sens_rung:
        if "sens_estimator" not in blk or "sens_estimator_solver" not in blk:
            problems.append(f"{where}: sens_estimator(+_solver) missing on the sens rung")
        else:
            if blk["sens_estimator_solver"] != ros:
                problems.append(
                    f"{where}: sens_estimator_solver {blk['sens_estimator_solver']!r} "
                    f"!= read_out_solver {ros!r}"
                )
            if set(blk["sens_estimator"]) != want_slots:
                problems.append(f"{where}: sens_estimator slots incomplete")
    return int(blk["n_eval"])


def validate_fits_outputs(
    eval_dir: pathlib.Path, rung_labels: list[str], smoke: bool | None = None
) -> list[str]:
    """Exact-topology resume-time output validation (r1 finding fits-complete;
    deepened per r2 concern fits-resume-schema-undervalidated: the prior form
    checked 4 found-anywhere keys and accepted empty ladders).

    Validates the REALIZED topology: both ladders carry exactly `rung_labels`
    rung blocks with every registered field; (arm x layer) cell cardinality;
    read-out kNN keys; per-layer correlated-offset floors; solver coherence
    (contingency_fired <=> read_out_solver == canonical, wide-grid + sens
    labels match); one Gate-D record per (ladder x rung) in g2_ext_report;
    paired per_layer cells at every read-out layer; NPZ array shapes + id/
    n_eval cross-checks; and P2 cell-grid completeness against the artifact's
    own declared axes. `smoke` gates regime-dependent checks (gate_e presence
    is required only on `smoke=False`). Returns the problem list (empty ==
    valid); the resume path refits on any problem instead of skipping.
    """
    problems: list[str] = []
    n_eval_by: dict[str, int] = {}

    r2 = _load_output_json(eval_dir, "ladder_ext_r2.json", problems)
    if isinstance(r2, dict):
        missing = [k for k in REQUIRED_OUTPUT_KEYS["ladder_ext_r2"] if k not in r2]
        if missing:
            problems.append(f"ladder_ext_r2.json: missing keys {missing}")
        gates = r2.get("gates") if isinstance(r2.get("gates"), dict) else {}
        for gk in ("gate_c", "gate_f_mask_integrity"):
            if gk not in gates:
                problems.append(f"ladder_ext_r2.json: gates missing {gk}")
        if smoke is False and "gate_e" not in gates:
            problems.append("ladder_ext_r2.json: gates missing gate_e (production run)")
        for tag in ("primary", "companion"):
            ladder = r2.get(tag)
            if not isinstance(ladder, dict):
                continue  # covered by the missing-keys check above
            if sorted(ladder) != sorted(rung_labels):
                problems.append(
                    f"ladder_ext_r2.json: {tag} rungs {sorted(ladder)} != {sorted(rung_labels)}"
                )
            for label, blk in ladder.items():
                sens_rung = tag == "primary" and rung_labels and label == rung_labels[0]
                n_eval = _validate_rung_block(tag, label, blk, bool(sens_rung), problems)
                if n_eval is not None:
                    prefix = "rung" if tag == "primary" else "rand_rung"
                    n_eval_by[f"{prefix}{label}"] = n_eval

    g2 = _load_output_json(eval_dir, "g2_ext_report.json", problems)
    if isinstance(g2, dict):
        missing = [k for k in REQUIRED_OUTPUT_KEYS["g2_ext_report"] if k not in g2]
        if missing:
            problems.append(f"g2_ext_report.json: missing keys {missing}")
        else:
            want_rungs = {
                f"{tag}/{label}" for tag in ("primary", "companion") for label in rung_labels
            }
            rungs = g2["rungs"] if isinstance(g2["rungs"], dict) else {}
            if set(rungs) != want_rungs:
                problems.append(
                    f"g2_ext_report.json: rungs {sorted(rungs)} != {sorted(want_rungs)}"
                )
            else:
                for k, rec in rungs.items():
                    if not isinstance(rec, dict) or "slices" not in rec or "verdict" not in rec:
                        problems.append(f"g2_ext_report.json: rung {k} lacks slices/verdict")
            if not isinstance(g2["threeway_rung1"], dict) or "pass" not in g2["threeway_rung1"]:
                problems.append("g2_ext_report.json: threeway_rung1 lacks a pass verdict")

    for label in rung_labels:
        for suffix in (f"rung{label}", f"rand_rung{label}"):
            name = f"shared_persona_paired_{suffix}.json"
            paired = _load_output_json(eval_dir, name, problems)
            if paired is None:
                continue
            per_layer = (
                paired.get("arms", {}).get(f"k{POOLED_K}", {}).get("per_layer", {})
                if isinstance(paired, dict)
                else {}
            )
            for layer in READ_OUT_LAYERS:
                cell = per_layer.get(f"L{layer}")
                if not isinstance(cell, dict):
                    problems.append(f"{name}: per_layer L{layer} missing")
                    continue
                miss = [k for k in REQUIRED_OUTPUT_KEYS["shared_persona_paired"] if k not in cell]
                if miss:
                    problems.append(f"{name}: L{layer} missing {miss}")
                elif "ratio_measured_over_full_energy" not in cell["offset_bias_control"]:
                    problems.append(f"{name}: L{layer} offset_bias_control lacks ratio")

    for label in rung_labels:
        for suffix in (f"rung{label}", f"rand_rung{label}"):
            name = f"percontext_{suffix}.npz"
            p = eval_dir / name
            if not p.exists():
                problems.append(f"missing {name}")
                continue
            try:
                with np.load(p) as z:
                    absent = [
                        k for k in REQUIRED_OUTPUT_KEYS["percontext_rung"] if k not in z.files
                    ]
                    if absent:
                        problems.append(f"{name}: missing arrays {absent}")
                        continue
                    n_ctx = int(z["context_ids"].shape[0])
                    want_shape = (len(ARM_NAMES), EXPECTED_LAYERS, n_ctx)
                    for arr in ("p1_ss_res", "p1_ss_tot"):
                        if tuple(z[arr].shape) != want_shape:
                            problems.append(
                                f"{name}: {arr} shape {tuple(z[arr].shape)} != {want_shape}"
                            )
                    if not np.issubdtype(z["context_ids"].dtype, np.integer):
                        problems.append(f"{name}: context_ids dtype is not integer")
                    if suffix in n_eval_by and n_ctx != n_eval_by[suffix]:
                        problems.append(
                            f"{name}: n_contexts {n_ctx} != rung block n_eval {n_eval_by[suffix]}"
                        )
            except (OSError, ValueError) as exc:
                problems.append(f"unreadable {name}: {exc}")
                continue

    p2 = _load_output_json(eval_dir, "p2_ext_boundary.json", problems)
    if isinstance(p2, dict):
        missing = [k for k in REQUIRED_OUTPUT_KEYS["p2_ext_boundary"] if k not in p2]
        if missing:
            problems.append(f"p2_ext_boundary.json: missing keys {missing}")
        else:
            if list(p2["read_out_layers"]) != list(READ_OUT_LAYERS):
                problems.append(
                    f"p2_ext_boundary.json: read_out_layers {p2['read_out_layers']} "
                    f"!= {list(READ_OUT_LAYERS)}"
                )
            want_p2 = {
                f"L{layer}:n{n}:seed{s}"
                for layer in p2["read_out_layers"]
                for n in p2["n_train_grid"]
                for s in p2["draw_seeds"]
            }
            p2_cells = p2["cells"] if isinstance(p2["cells"], dict) else {}
            if set(p2_cells) != want_p2:
                problems.append(
                    f"p2_ext_boundary.json: cells {len(p2_cells)} != declared grid "
                    f"{len(want_p2)} (layer x n_train x seed)"
                )
            else:
                cell_req = ("r2", "lambda", "dof", "solver")
                bad = sorted(k for k, c in p2_cells.items() if any(f not in c for f in cell_req))
                if bad:
                    problems.append(f"p2_ext_boundary.json: cells missing fields at {bad[:4]}")

    return problems


# ── Phase: fits ───────────────────────────────────────────────────────────────


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _banked_paired_json(repo_root: pathlib.Path) -> dict:
    """The banked shared_persona_paired.json AT THE PINNED REV (plan section 10)."""
    rel = f"{BANKED_LADDER_RELDIR}/shared_persona_paired.json"
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{BANKED_PAIRED_REV}:{rel}"],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cannot read {rel} @ {BANKED_PAIRED_REV}: {proc.stderr.strip()} — the banked "
            "bridge needs the pinned blob (partial clones fetch it on demand; a sparse "
            "worktree without eval_results/ cannot run Gate E)"
        )
    return json.loads(proc.stdout)


def bridge_refit(
    src: FitSources,
    bridge_ids: np.ndarray,
    dev: torch.device,
) -> dict[str, float]:
    """Gate E(ii): rung-1 refit on the EXACT banked mask, pure GCV + canonical
    solver (the parent's realized configuration), paired ratios per layer."""
    ids = np.asarray(sorted(int(i) for i in bridge_ids), dtype=np.int64)
    folds = make_folds(len(ids))
    shared_pos = np.flatnonzero(ids % POOLED_K == 0)
    ratios: dict[str, float] = {}
    for layer in READ_OUT_LAYERS:
        x_full = src.cx_col(layer, ids)
        y_full = {arm: src.arm_col(arm, layer, ids) for arm in ARM_NAMES}
        sres = {arm: np.full(len(ids), np.nan) for arm in ARM_NAMES}
        for tr, te in folds:
            for arm in ARM_NAMES:
                pred, _lam, _dof = canonical_capped_fit(
                    x_full[tr], y_full[arm][tr], x_full[te], LAMBDAS, math.inf
                )
                s, _t = LF.per_context_ss(pred, y_full[arm][te])
                sres[arm][te] = s
        diff = sres["k16"][shared_pos] - sres["k1"][shared_pos]
        personas = ids % POOLED_K
        groups = []
        v0_all = src.pair_col(0, layer, ids)
        for p in sorted({int(q) for q in personas if q != 0}):
            sel = np.flatnonzero(personas == p)
            vp = src.pair_col(int(p), layer, ids[sel])
            groups.append((len(sel), (vp - v0_all[sel]).astype(np.float64)))
        e_val = mixture_energy_from_group_diffs(iter(groups), int(len(shared_pos)))
        ratios[f"L{layer}"] = float(diff.mean() / e_val) if e_val > 0 else float("nan")
    return ratios


def phase_fits(args, layout: EXTCAP.Layout) -> None:
    dev = _resolve_device(args.device)
    eval_dir = layout.eval_dir
    ckpt_dir = layout.out_root / "fits_ckpt"
    n_boot = args.n_boot if args.n_boot is not None else (SMOKE_BOOT_N if args.smoke else BOOT_N)

    # ── preconditions: mask + verified store + banked pins ──
    mask_path = eval_dir / EXTCAP.MASK_FILE
    if not mask_path.exists():
        raise RuntimeError(f"{mask_path} missing — run the capture driver's p0ext phase first")
    mask_obj = json.loads(mask_path.read_text())
    store_id = _validated_store_identity(layout)
    fp_extra = {"store_name_set_sha256": store_id["name_set_sha256"]}
    metadata = EXTCAP.build_metadata(layout, "fits")
    metadata["script"] = "scripts/issue823_ladder_ext_fits.py"
    logger.info("[fits] staging banked pins (idempotent)")
    banked = EXTCAP.stage_banked_inputs(layout)
    src = load_fit_sources(layout, banked)

    # ── realized rung masks (gen mask x capture validity; smoke cap) ──
    rung_masks, capture_drops = realize_rung_masks(mask_obj, src, args.smoke)
    rung_labels = sorted(rung_masks, key=int)
    top_label = rung_labels[-1]
    logger.info(
        "[fits] realized rung masks: %s",
        {k: int(len(v)) for k, v in rung_masks.items()},
    )

    # ── companion sets + manifest (rc 22 on violation) ──
    rung_sizes = {label: int(len(rung_masks[label])) for label in rung_labels}
    e_eval, subsets, manifest = build_companion_sets(
        rung_masks[top_label], rung_sizes, layout.n_prefix
    )
    violations = check_companion_manifest(e_eval, subsets, manifest, rung_masks[top_label])
    enforce_companion_manifest(violations, eval_dir, manifest)
    write_json(eval_dir / "rand_ladder_manifest.json", manifest)
    manifest_sha = _sha_json(manifest)

    # ── resume predicate (fingerprint equality + output validation) ──
    fingerprint = fits_fingerprint(
        {label: _ids_sha(ids) for label, ids in rung_masks.items()},
        manifest_sha,
        gate_states={"pending": True},
        store_name_set_sha256=store_id["name_set_sha256"],
    )
    _resume_keys = ("rung_mask_shas", "rand_manifest_sha", "store_name_set_sha256", "estimator")
    sentinel_path = eval_dir / FITS_SENTINEL
    if sentinel_path.exists():
        prior = json.loads(sentinel_path.read_text())
        prior_fp = prior.get("fingerprint", {})
        same_inputs = {k: prior_fp.get(k) for k in _resume_keys} == {
            k: fingerprint[k] for k in _resume_keys
        }
        if (
            prior.get("complete")
            and same_inputs
            and prior_fp.get("code_sha") == fingerprint["code_sha"]
        ):
            problems = validate_fits_outputs(eval_dir, rung_labels, smoke=bool(prior.get("smoke")))
            if not problems:
                logger.info(
                    "[fits] complete sentinel + matching fingerprint + validated outputs "
                    "— nothing to do"
                )
                print("[phase=done]", flush=True)
                return
            logger.warning(
                "[fits] sentinel matches but outputs FAILED validation (%s) — refitting",
                "; ".join(problems[:10]),
            )
        else:
            logger.warning("[fits] stale/mismatched sentinel — refitting (fingerprint inequality)")

    # ── Gate C: pilot FIRST at production shapes ──
    top_ids = rung_masks[top_label]
    folds_top = make_folds(len(top_ids))
    tr_top = folds_top[0][0]
    layer0 = READ_OUT_LAYERS[0]
    t0 = time.monotonic()
    x_tr = src.cx_col(layer0, top_ids[tr_top])
    y_by_arm = {arm: src.arm_col(arm, layer0, top_ids[tr_top]) for arm in ARM_NAMES}
    fact = rung_factorize(x_tr, dev)
    kev = eval_kernel(fact, src.cx_col(layer0, top_ids[folds_top[0][1]]))
    for arm in ARM_NAMES:
        lam, proj, ymu, _dof = solve_capped(fact, y_by_arm[arm], LAMBDAS, DOF_CAP)
        apply_fit(fact, lam, proj, ymu, kev)
    t_top = time.monotonic() - t0
    del fact, kev
    rung1_ids = rung_masks[rung_labels[0]]
    folds_r1 = make_folds(len(rung1_ids))
    t0 = time.monotonic()
    x_tr1 = src.cx_col(layer0, rung1_ids[folds_r1[0][0]])
    y1 = {arm: src.arm_col(arm, layer0, rung1_ids[folds_r1[0][0]]) for arm in ARM_NAMES}
    fact1 = rung_factorize(x_tr1, dev)
    kev1 = eval_kernel(fact1, src.cx_col(layer0, rung1_ids[folds_r1[0][1]]))
    for arm in ARM_NAMES:
        lam, proj, ymu, _dof = solve_capped(fact1, y1[arm], LAMBDAS, DOF_CAP)
        apply_fit(fact1, lam, proj, ymu, kev1)
    t_dual = time.monotonic() - t0
    n_cells_per_ladder = len(rung_labels) * EXPECTED_LAYERS * N_FOLDS
    n_dual = (
        sum(
            EXPECTED_LAYERS * N_FOLDS
            for label in rung_labels
            if int(len(rung_masks[label]) * (N_FOLDS - 1) / N_FOLDS) <= DUAL_N_MAX
        )
        * 2
    )  # both ladders route roughly alike (subset sizes mirror rung sizes)
    n_primal = (
        2 * n_cells_per_ladder
        - n_dual
        + len(P2_N_TRAIN_GRID) * len(P2_DRAW_SEEDS) * len(READ_OUT_LAYERS)
    )
    g3 = gate_c_record(t_dual, t_top, n_dual, max(n_primal, 0), args.planned_wall_hours)
    write_json(eval_dir / "g3_pilot_record.json", g3)
    enforce_gate_c(g3, eval_dir, args.smoke)
    del fact1, kev1

    # ── rung-1 three-way dual/primal/canonical agreement check ──
    tr1, te1 = folds_r1[0]
    x_tr1 = src.cx_col(layer0, rung1_ids[tr1])
    x_te1 = src.cx_col(layer0, rung1_ids[te1])
    y_tr1 = src.arm_col("k1", layer0, rung1_ids[tr1])
    fact_d = LF.factorize_robust(x_tr1, dev)
    fact_d["kind"] = "dual"
    lam_d, proj_d, ymu_d, dof_d = solve_capped(fact_d, y_tr1, LAMBDAS, DOF_CAP)
    pred_d = apply_fit(fact_d, lam_d, proj_d, ymu_d, eval_kernel(fact_d, x_te1))
    fact_p = _factorize_primal(x_tr1, dev)
    lam_p, proj_p, ymu_p, dof_p = solve_capped(fact_p, y_tr1, LAMBDAS, DOF_CAP)
    pred_p = apply_fit(fact_p, lam_p, proj_p, ymu_p, eval_kernel(fact_p, x_te1))
    pred_c, lam_c, dof_c = canonical_capped_fit(x_tr1, y_tr1, x_te1, LAMBDAS, DOF_CAP)
    scale = max(float(np.abs(pred_c).max()), 1e-9)
    threeway = {
        "layer": layer0,
        "fold": 0,
        "lambda": {"dual": lam_d, "primal": lam_p, "canonical": lam_c},
        "dof": {"dual": dof_d, "primal": dof_p, "canonical": dof_c},
        "max_rel_dual_canonical": float(np.abs(pred_d - pred_c).max()) / scale,
        "max_rel_primal_canonical": float(np.abs(pred_p - pred_c).max()) / scale,
        "max_rel_dual_primal": float(np.abs(pred_d - pred_p).max()) / scale,
    }
    threeway["pass"] = bool(
        threeway["max_rel_dual_canonical"] <= G2_MAX_REL_TOL
        and threeway["max_rel_primal_canonical"] <= G2_MAX_REL_TOL
        and threeway["max_rel_dual_primal"] <= G2_MAX_REL_TOL
        and lam_d == lam_p == lam_c
    )
    logger.info("[fits] rung-1 three-way check: %s", threeway)
    del fact_d, fact_p

    # ── both ladders: fit + rung dirs + paired reads ──
    g2_report: dict = {
        "tolerances": {"max_rel": G2_MAX_REL_TOL, "delta_r2": G2_DELTA_R2_TOL},
        "threeway_rung1": threeway,
        "rungs": {},
    }
    r2_agg: dict = {
        "estimator": fingerprint["estimator"],
        "capture_drops": capture_drops,
        "primary": {},
        "companion": {},
        "gates": {},
        "lambda_edge_fraction_trigger": LAMBDA_EDGE_FRACTION,
    }
    ladder_specs = [("primary", None)] + [("companion", e_eval)]
    for tag, ev_ids in ladder_specs:
        for label in rung_labels:
            train_ids = rung_masks[label] if tag == "primary" else subsets[label]
            sens_flag = tag == "primary" and label == rung_labels[0]
            rf = fit_rung(
                tag,
                label,
                train_ids,
                src,
                dev,
                ckpt_dir,
                eval_ids=ev_ids,
                sens_pure=sens_flag,
                fp_extra=fp_extra,
            )
            # Gate D verdict + contingency routing
            slices_pass = all(s["pass"] for s in rf.g2_slices) and (
                threeway["pass"] or label != rung_labels[0] or tag != "primary"
            )
            verdict = "PASS" if slices_pass else "FAIL"
            contingency_records: list[dict] = []
            if not slices_pass and not args.smoke:
                logger.warning("[gate-d] %s/%s parity FAIL — canonical contingency", tag, label)
                contingency_records = contingency_refit(rf, src)
                cont_checks = []
                folds = make_folds(len(rf.train_ids))
                for layer, f_idx in G2_SLICES:
                    tr, te = folds[f_idx]
                    x_full = src.cx_col(layer, rf.train_ids)
                    x_score = src.cx_col(layer, rf.eval_ids) if tag == "companion" else x_full[te]
                    for arm in ARM_NAMES:
                        y_full = src.arm_col(arm, layer, rf.train_ids)
                        y_score = (
                            src.arm_col(arm, layer, rf.eval_ids)
                            if tag == "companion"
                            else y_full[te]
                        )
                        cont_checks.append(
                            contingency_parity_check(
                                x_full[tr],
                                y_full[tr],
                                x_score,
                                y_score,
                                dev,
                                layer=layer,
                                fold=f_idx,
                                arm=arm,
                            )
                        )
                enforce_contingency_parity(cont_checks, eval_dir, f"{tag}/{label}")
                verdict = "CONTINGENCY-PASS"
                g2_report["rungs"][f"{tag}/{label}"] = {
                    "slices": rf.g2_slices,
                    "verdict": verdict,
                    "contingency": contingency_records,
                    "contingency_parity": cont_checks,
                }
            else:
                if not slices_pass:
                    logger.warning(
                        "SMOKE-INFORMATIONAL (enumerated blind spot) gate-d FAIL %s/%s",
                        tag,
                        label,
                    )
                    verdict = "WARN-SMOKE-INFORMATIONAL"
                g2_report["rungs"][f"{tag}/{label}"] = {
                    "slices": rf.g2_slices,
                    "verdict": verdict,
                }

            # lambda-edge trigger (per rung per ladder) — computed AFTER contingency
            # routing, so a fired contingency's canonical records (not the stale
            # parity-FAILED read-out-layer fits) feed the trigger.
            edge_frac = float(np.mean([r["lambda_top_edge"] for r in rf.fit_records]))
            wide_block = None
            if edge_frac > LAMBDA_EDGE_FRACTION:
                if args.smoke:
                    logger.warning(
                        "SMOKE-INFORMATIONAL lambda-edge %.2f at %s/%s", edge_frac, tag, label
                    )
                else:
                    logger.warning(
                        "[lambda-edge] %.2f > %.2f at %s/%s — wider-grid sensitivity",
                        edge_frac,
                        LAMBDA_EDGE_FRACTION,
                        tag,
                        label,
                    )
                    wide_block = wide_grid_sensitivity(rf, src, dev)

            # row coverage BEFORE any paired statistic (plan section 3)
            row_coverage_check(rf, tuple(range(EXPECTED_LAYERS)))
            if not np.array_equal(np.asarray(sorted(int(i) for i in rf.eval_ids)), rf.eval_ids) or (
                tag == "primary" and not np.array_equal(rf.eval_ids, rung_masks[label])
            ):
                raise RuntimeError(f"row-coverage: eval id set drifted at {tag}/{label}")

            # rung dir + paired read
            dir_name = f"rung_{label}" if tag == "primary" else f"rand_rung_{label}"
            rung_dir = layout.out_root / dir_name
            rung_outputs = write_rung_dir(
                rung_dir,
                rf,
                src,
                layout.n_total,
                metadata,
                diff_train_ids=train_ids,
            )
            suffix = f"rung{label}" if tag == "primary" else f"rand_rung{label}"
            shutil.copy2(rung_dir / "percontext_ladder.npz", eval_dir / f"percontext_{suffix}.npz")
            paired_out = eval_dir / f"shared_persona_paired_{suffix}.json"
            run_paired_script(
                _REPO_ROOT,
                paired_out,
                rung_dir,
                arms=str(POOLED_K),
                n_boot=n_boot,
                full_ratio_ci=True,
            )

            knn_block = rf.knn
            rung_block = {
                "n_mask": int(len(train_ids)),
                "n_eval": int(len(rf.eval_ids)),
                "n_train_per_fold": rf.fold_ns,
                "d": HIDDEN,
                "n_over_d_ratio": min(rf.fold_ns) / HIDDEN,
                "solver": rf.solver,
                "g2_verdict": verdict,
                "lambda_edge_fraction": edge_frac,
                "cells": rf.cells,
                "knn_read_out": knn_block,
                "estimator_degenerate": bool(min(rf.fold_ns) < HIDDEN),
                "contingency_fired": bool(contingency_records),
                "read_out_solver": rf.read_out_solver,
                "correlated_offset_floor": rung_outputs["floor"],
            }
            if wide_block is not None:
                rung_block["wide_grid_sensitivity"] = wide_block
            if sens_flag:
                sens_slots, sens_solver = sens_estimator_block(rf, src, contingency_records)
                rung_block["sens_estimator"] = sens_slots
                rung_block["sens_estimator_solver"] = sens_solver
            r2_agg[tag][label] = rung_block
            print(f"[phase=fits_{tag}_{label}]", flush=True)

    # ── dedup-mask sensitivity refit (consumes the P0 gate-(e) flag; r1 concern
    # dedup-sensitivity-detached — a fired flag with no consumer blocked here) ──
    p0_report_path = eval_dir / EXTCAP.P0_REPORT
    p0_report = json.loads(p0_report_path.read_text()) if p0_report_path.exists() else {}
    gate_e_dup = p0_report.get("gate_e_duplicates", {})
    dedup_required = bool(gate_e_dup.get("dedup_sensitivity_refit_required"))
    sens_block = sens_dedup_block(gate_e_dup, rung_masks, rung_labels, src, dev, ckpt_dir, fp_extra)
    if sens_block is not None:
        r2_agg["sens_dedup"] = sens_block
        print("[phase=fits_sens_dedup]", flush=True)

    # ── Gate E: banked bridge (production only; smoke = enumerated blind spot) ──
    if args.smoke:
        logger.warning("SMOKE-INFORMATIONAL: Gate E banked bridge skipped (no banked slice)")
        gate_e_state = "SKIPPED-SMOKE"
    else:
        banked_json = _banked_paired_json(_REPO_ROOT)
        loader_out = layout.out_root / "bridge_loader_rerun.json"
        run_paired_script(
            _REPO_ROOT, loader_out, ladder_dir=None, arms=None, n_boot=None, full_ratio_ci=False
        )
        loader_cmp = bridge_loader_compare(json.loads(loader_out.read_text()), banked_json)
        enforce_gate_e(loader_cmp, eval_dir, "loader-level rerun (E(i))")
        bridge_ids = np.asarray(EXTCAP.load_bridge_mask_ids(), dtype=np.int64)
        refit_ratios = bridge_refit(src, bridge_ids, dev)
        banked_ratios = {
            f"L{layer}": banked_json["arms"][f"k{POOLED_K}"]["per_layer"][f"L{layer}"][
                "offset_bias_control"
            ]["ratio_measured_over_full_energy"]
            for layer in READ_OUT_LAYERS
        }
        refit_cmp = bridge_refit_compare(refit_ratios, banked_ratios)
        enforce_gate_e(refit_cmp, eval_dir, "bridge refit (E(ii))")
        r2_agg["gates"]["gate_e"] = {"loader": loader_cmp, "refit": refit_cmp}
        gate_e_state = "PASS"
    print("[phase=fits_bridge]", flush=True)

    # ── P2-ext boundary ladder ──
    k1_valid = np.asarray(
        [i for i in range(layout.n_total) if src.pair_ok(i, 0) and src.has_cx(i)],
        dtype=np.int64,
    )
    if args.smoke:
        k1_valid = smoke_cap_mask(k1_valid, SMOKE_MASK_CAP)
    p2 = p2_boundary_ladder(src, k1_valid, dev, ckpt_dir, smoke=args.smoke, fp_extra=fp_extra)
    write_json(eval_dir / "p2_ext_boundary.json", p2)
    print("[phase=fits_p2]", flush=True)

    # ── aggregates + sentinel ──
    r2_agg["gates"]["gate_c"] = g3
    r2_agg["gates"]["gate_f_mask_integrity"] = mask_obj.get("integrity_gate")
    r2_agg["metadata"] = metadata
    write_json(eval_dir / "ladder_ext_r2.json", r2_agg)
    write_json(eval_dir / "g2_ext_report.json", g2_report)

    gate_states = {
        "gate_c": "PASS" if g3["pass"] else "WARN-SMOKE-INFORMATIONAL",
        "gate_d": {k: v["verdict"] for k, v in g2_report["rungs"].items()},
        "gate_e": gate_e_state,
        "gate_f_mask_integrity": mask_obj.get("integrity_gate"),
    }
    fingerprint = fits_fingerprint(
        {label: _ids_sha(ids) for label, ids in rung_masks.items()},
        manifest_sha,
        gate_states=gate_states,
        store_name_set_sha256=store_id["name_set_sha256"],
    )
    if dedup_required and "sens_dedup" not in r2_agg:
        raise RuntimeError(
            "gate (e) dedup flag fired but no sens_dedup block was produced — refusing to "
            "write the fits completion sentinel"
        )
    write_fits_sentinel(eval_dir, fingerprint, {"metadata": metadata, "smoke": args.smoke})
    write_sentinel(
        layout.sentinel_dir() / "issue-823-extladder-fits-done.json",
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "P-Fit-ext complete: both ladders + bridge + P2-ext + paired reads",
            "phase": "fits",
            "complete": True,
            "smoke": bool(args.smoke),
            "rungs": {k: int(len(v)) for k, v in rung_masks.items()},
            "ts": time.time(),
        },
    )
    print("[phase=done]", flush=True)
    logger.info("P-Fit-ext complete: sentinel at %s", eval_dir / FITS_SENTINEL)


# ── Import-check / CLI ────────────────────────────────────────────────────────


def run_import_check() -> None:
    """Execute deferred-import surface + argparse-attribute completeness + binds."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    assert np.allclose(LAMBDAS, np.logspace(-2, 4, 13)), "primary grid drifted from FFC.LAMBDAS"
    assert callable(LF.gcv_solve_dof_capped) and callable(LF.factorize_robust)
    assert callable(FFC._cross_kernel) and callable(FFC._apply)
    assert callable(mixture_energy_from_group_diffs) and callable(correlated_floor_from_groups)
    assert (_REPO_ROOT / PAIRED_SCRIPT_RELPATH).exists()
    print("import-check OK")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P-Fit-ext for #823 origin-ladder-more-contexts: extension-ladder fits "
            "(dof-capped GCV, dual/primal solvers, companion ladder, banked bridge, "
            "P2-ext boundary ladder, per-rung paired reads)."
        )
    )
    parser.add_argument("--phase", choices=list(PHASES), help="phase to run")
    parser.add_argument("--smoke", action="store_true", help="tiny-shape path (capped masks)")
    parser.add_argument(
        "--n-ext-contexts",
        type=int,
        default=None,
        help="extension context count override (smoke only; production pinned to 43000)",
    )
    parser.add_argument("--out-root", type=pathlib.Path, default=None, help="durable out-root")
    parser.add_argument("--device", default="auto", help="fit device: auto (default) | cuda | cpu")
    parser.add_argument(
        "--planned-wall-hours",
        type=float,
        default=PLANNED_FITS_WALL_H,
        help="plan section-9 P-Fit-ext wall row; Gate C aborts past 2x this (rc 19)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help=f"paired-read bootstrap draws (default {BOOT_N}; smoke {SMOKE_BOOT_N})",
    )
    parser.add_argument("--import-check", action="store_true", help="deferred-import gate, exit 0")
    parser.add_argument("--list-rcs", action="store_true", help="print the designed rc table")
    parser.add_argument("--list-phases", action="store_true", help="print the phase registry")
    parser.add_argument("--list-arms", action="store_true", help="print the registered arm list")
    return parser


PHASES = {"fits": phase_fits}


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.import_check:
        run_import_check()
        raise SystemExit(0)
    if args.list_rcs:
        for rc, desc in sorted(RC_TABLE.items()):
            print(f"rc {rc}: {desc}")
        raise SystemExit(0)
    if args.list_phases:
        print(sorted(PHASES))
        raise SystemExit(0)
    if args.list_arms:
        print(list(ARM_NAMES))
        raise SystemExit(0)
    if not args.phase:
        parser.error("--phase is required (or --import-check/--list-rcs/--list-phases)")

    # Layout resolution mirrors the capture driver EXACTLY (same out_root =>
    # same store/mask/eval paths) — issue823_ladder_ext_capture.py main().
    if args.smoke:
        n_ext = args.n_ext_contexts if args.n_ext_contexts is not None else 16
        assert 1 <= n_ext <= EXTCAP.EXTGEN.N_EXT_FULL, "--n-ext-contexts out of range"
        if pathlib.Path("/workspace").exists():
            root = args.out_root or pathlib.Path("/workspace/eps/out/issue823_ladder_ext_smoke")
        else:
            root = args.out_root or pathlib.Path("/tmp/issue-823-ext-smoke/ladder_ext_capture")
    else:
        if args.n_ext_contexts is not None and args.n_ext_contexts != EXTCAP.EXTGEN.N_EXT_FULL:
            parser.error("--n-ext-contexts is smoke-only; production runs the full 43000")
        n_ext = EXTCAP.EXTGEN.N_EXT_FULL
        if args.out_root is not None:
            root = args.out_root
        elif pathlib.Path("/workspace").exists():
            root = PROD_OUT_ROOT
        else:
            parser.error("production off-pod requires an explicit --out-root")
    layout = EXTCAP.Layout(root, args.smoke, n_ext)
    PHASES[args.phase](args, layout)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
