"""Issue #1945 — Gabriel (2,2) BCV rank battery over the #1482 two-way interaction residual.

Is the (context x direction) interaction residual low-rank-RECOVERABLE (H1) or
per-pair idiosyncratic (H2)?  Per pipeline unit (cell x k x space, folds pooled)
the held-out-block interaction R^2 curve over r in R_GRID is compared against two
selection-symmetric null families (B draws each, per-draw max-over-r selection):

- ``perm``     — independent within-column permutation of the two-way-removed
  matrix F (destroys row pairing, keeps column marginals; zero-structure floor).
- ``gauss2m``  — rows i.i.d. N(0, Sigma_E) with Sigma_E the fp64 covariance of the
  BASIS-half projected residual (out-of-sample wrt the eval half), row-norm
  matched to the observed eval rows, generated at dim 256 and sliced to k.  This
  is the H1 bar: everything the residual's second moment + row scale implies.

Phases (plan #1945 v3 section 4):
  p0  — realized-keys asserts, grouping probe, ridge slow-vs-fast parity gate at
        the Tier-B production shape, timed 1-unit pilot (B=8) with extrapolation.
  p1  — Tier A: BCV rank battery over 12 cells x {64,256} x {log,raw,normalized}
        (72 pipeline units; per-unit JSONL checkpoint + per-draw x per-r npz).
  p2  — Tier B: input-recoverability via reduced-rank ridge from map-output
        features (pred @ top-512 answer-PCA + 2 norms -> 514 dims), log/k=256,
        per (cell, fold); permutation null shares ONE Gram factorization.
  p3  — floor-subsample replication (raw+normalized; scalar row correction is a
        no-op in log space), floor-noise-only synthetic share, summaries, figures.

Inputs: staged #1482 matrices (``data/issue_1482/twoway_stage``; env override
``EPS_I1945_STAGE``, main-checkout fallback), #1738 K-resample floors, the parent
#1482 two-way JSON (fold-identity cross-check).  Parent functions ``two_way`` /
``pca_basis`` / ``make_folds`` are imported from ``issue1482_twoway_residual``
unchanged; the tiny stage loaders are reimplemented locally against the resolved
stage dir with the parent's ci assert PLUS a NEW pred/y fingerprint-equality
assert (plan section 10 item (f)).

Seeds: parent fold split 1482 (basis identity with #1482); ALL new randomness
(BCV row splits, null draws, Tier-B splits) seed 1945.  0 GPU; CPU numpy/torch.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM run)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue1482_twoway_residual import make_folds, pca_basis, two_way  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.experiments.issue_779.fit_h import (  # noqa: E402
    ridge_fit_predict,
    ridge_fit_predict_fast,
)

PARENT_SEED = 1482  # fold split — basis identity with #1482
SEED = 1945  # ALL new randomness
R_GRID = (1, 2, 4, 8, 16, 32, 64)  # #1775 R_GRID
K_LIST = (64, 256)
SPACES = ("log", "raw", "normalized")
LAYERS = (14, 19, 26)
ARMS = ("context", "prefix", "bare")
KMAX = 512  # parent-verbatim basis dim (Tier B consumes comps[:, :512])
KA = 256  # Tier-A max k == gauss2m draw-generation dimension
TIERB_K = 256
CELLS: tuple[tuple[str, int, str], ...] = tuple(
    [(arm, layer, "ridge") for layer in LAYERS for arm in ARMS]
    + [(arm, 19, "mlp_w8192") for arm in ARMS]
)
CELL_NAMES = tuple(f"{arm}_L{layer}_{fitter}" for arm, layer, fitter in CELLS)
PRIMARY_UNIT = ("context_L19_ridge", 256, "log")
DEFAULT_B = 200
CHUNK_DRAWS = 25
MAGNITUDE_FLOOR = 0.10  # Delta_m threshold (plan section 3)
PARITY_TOL = 1e-4  # #1332 convention, slow-vs-fast max rel diff
HEADROOM_GB = 5.0  # plan section 9 out-root floor
PILOT_ABORT_FACTOR = 2.0  # pilot-gated abort threshold vs booked wall
P1_BOOKED_WALL_H = 1.5  # plan section 9 P1 row
RC_PILOT_ABORT = 7  # designed artifact-routed halt (gotchas: pilot-gate rc)


# ── provenance ────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _metadata(cfg: Cfg) -> dict:
    import scipy

    return {
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "host": platform.node(),
        "seed": SEED,
        "parent_seed": PARENT_SEED,
        "b_draws": cfg.b_draws,
        "stage_dir": str(cfg.stage),
        "smoke": cfg.smoke,
        "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


# ── input resolution ─────────────────────────────────────────────────────────


def _main_checkout_root() -> Path | None:
    """Main repo root via git-common-dir (worktree-safe); None if unresolvable."""
    try:
        common = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return Path(common).parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def resolve_stage_dir() -> Path:
    """EPS_I1945_STAGE > module-relative default > main-checkout fallback; fail loud."""
    tried: list[str] = []
    env = os.environ.get("EPS_I1945_STAGE")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(PROJECT_ROOT / "data" / "issue_1482" / "twoway_stage")
    main_root = _main_checkout_root()
    if main_root is not None:
        candidates.append(main_root / "data" / "issue_1482" / "twoway_stage")
    for cand in candidates:
        tried.append(str(cand))
        if (cand / "y_parent_L19.npz").exists():
            return cand
    raise FileNotFoundError(
        "issue1945: staged #1482 matrices not found (y_parent_L19.npz probe). Tried: "
        + " | ".join(tried)
        + ". Re-staging recipe: plan #1945 v3 section 10 (HF issue1738_multiturn/analysis_tensors)."
    )


def resolve_input(rel: str) -> Path:
    """Read-only committed input: PROJECT_ROOT first, main-checkout fallback; fail loud."""
    tried = []
    for root in [PROJECT_ROOT, _main_checkout_root()]:
        if root is None:
            continue
        cand = root / rel
        tried.append(str(cand))
        if cand.exists():
            return cand
    raise FileNotFoundError(f"issue1945: required input {rel!r} not found. Tried: {tried}")


PRED_KEYS = {"pred16", "ci", "fingerprint"}
Y_KEYS = {"y16", "ci", "fingerprint"}
FLOOR_KEYS = {"ci", "floor", "den", "share"}


def load_layer(stage: Path, layer: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load staged holdout targets; realized-keys assert (plan P0 step 1)."""
    z = np.load(stage / f"y_parent_L{layer}.npz")
    assert set(z.files) == Y_KEYS, f"y_parent_L{layer}: keys {sorted(z.files)} != {sorted(Y_KEYS)}"
    return z["y16"], z["ci"], str(z["fingerprint"])


def load_pred(
    stage: Path, arm: str, layer: int, fitter: str, ci_ref: np.ndarray, fp_ref: str
) -> np.ndarray:
    """Load staged predictions; parent ci assert + NEW fingerprint-equality assert."""
    z = np.load(stage / f"pred_{arm}_L{layer}_{fitter}.npz")
    assert set(z.files) == PRED_KEYS, (
        f"pred_{arm}_L{layer}_{fitter}: keys {sorted(z.files)} != {sorted(PRED_KEYS)}"
    )
    if not np.array_equal(z["ci"], ci_ref):
        raise AssertionError(f"{arm} L{layer} {fitter}: ci does not match the target ci")
    fp = str(z["fingerprint"])
    if fp != fp_ref:
        raise AssertionError(
            f"{arm} L{layer} {fitter}: fingerprint {fp[:16]}... != y fingerprint {fp_ref[:16]}..."
        )
    return z["pred16"]


# ── config ────────────────────────────────────────────────────────────────────


@dataclass
class Cfg:
    phase: str = "all"
    smoke: bool = False
    b_draws: int = DEFAULT_B
    cells: tuple[str, ...] = CELL_NAMES
    k_list: tuple[int, ...] = K_LIST
    spaces: tuple[str, ...] = SPACES
    folds: tuple[int, ...] = (0, 1)
    floor_layers: tuple[int, ...] = LAYERS
    out_root: Path = PROJECT_ROOT
    stage: Path = field(default_factory=resolve_stage_dir)
    chunk: int = CHUNK_DRAWS

    @property
    def smoke_dir(self) -> Path:
        return self.out_root / "eval_results" / "issue_1945" / "smoke"

    @property
    def eval_dir(self) -> Path:
        """Smoke runs divert under smoke/ so canonical bcv/tierb/floor/percell
        stay pristine for the full battery (#722 smoke-clobber class)."""
        base = self.out_root / "eval_results" / "issue_1945"
        return base / "smoke" if self.smoke else base

    @property
    def fig_dir(self) -> Path:
        base = self.out_root / "figures" / "issue_1945"
        return base / "smoke" if self.smoke else base

    def regime(self) -> dict:
        """Every output-affecting key (resume predicate keys on this)."""
        return {
            "b_draws": self.b_draws,
            "r_grid": list(R_GRID),
            "seed": SEED,
            "parent_seed": PARENT_SEED,
            "kmax": KMAX,
            "gauss_dim": KA,
            "smoke": self.smoke,
        }


def _assert_headroom(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_root).free / 1e9
    if free_gb < HEADROOM_GB:
        raise RuntimeError(
            f"out-root headroom {free_gb:.1f} GB < {HEADROOM_GB} GB floor at {out_root}"
        )


# ── core math ────────────────────────────────────────────────────────────────


def space_transform(Rk: np.ndarray, lam: np.ndarray, space: str) -> np.ndarray:
    """R (.., n, k) -> analysis space. log/raw/normalized per plan section 11."""
    if space == "log":
        return np.log(Rk)
    if space == "raw":
        return np.asarray(Rk, dtype=np.float64)
    if space == "normalized":
        return Rk / lam[None, :]
    raise ValueError(f"unknown space {space!r}")


def twoway_removed(M: np.ndarray) -> np.ndarray:
    """F = M - mu - a_i - b_j (parent two_way internals, residual returned).

    Accepts (n, k) or batched (B, n, k); removal per matrix.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 2:
        mu = M.mean()
        a = M.mean(axis=1, keepdims=True) - mu
        b = M.mean(axis=0, keepdims=True) - mu
        return M - mu - a - b
    mu = M.mean(axis=(1, 2), keepdims=True)
    a = M.mean(axis=2, keepdims=True) - mu
    b = M.mean(axis=1, keepdims=True) - mu
    return M - mu - a - b


def _space_transform_t(Rk: torch.Tensor, lam: torch.Tensor, space: str) -> torch.Tensor:
    """Torch twin of space_transform for the batched null hot loop
    (equivalence pinned by tests/test_issue1945_bcv_math.py)."""
    if space == "log":
        return torch.log(Rk)
    if space == "raw":
        return Rk.to(torch.float64)
    if space == "normalized":
        return Rk / lam
    raise ValueError(f"unknown space {space!r}")


def _twoway_removed_t(M: torch.Tensor) -> torch.Tensor:
    """Torch twin of twoway_removed (batched (B, n, k) only)."""
    mu = M.mean(dim=(1, 2), keepdim=True)
    a = M.mean(dim=2, keepdim=True) - mu
    b = M.mean(dim=1, keepdim=True) - mu
    return M - mu - a - b


def bcv_splits(n_rows: int, k: int, seed: int = SEED) -> tuple[list, list]:
    """Rows: 2 seeded halves. Columns: eigen-rank interleaving (even/odd ranks)."""
    perm = np.random.default_rng(seed).permutation(n_rows)
    rows = [np.sort(perm[: n_rows // 2]), np.sort(perm[n_rows // 2 :])]
    cols = [np.arange(0, k, 2), np.arange(1, k, 2)]
    return rows, cols


def r_grid_for_k(k: int) -> list[int]:
    """r truncated to <= k_train/2 with k_train = k/2 columns in the SVD'd block."""
    k2 = k // 2
    return [r for r in R_GRID if r <= k2 // 2]


def _bcv_blocks(rows: list, cols: list) -> list[tuple]:
    """The 4 Gabriel (2,2) held-out blocks as (rows_held, rows_comp, cols_held, cols_comp)."""
    return [
        (rows[0], rows[1], cols[0], cols[1]),
        (rows[0], rows[1], cols[1], cols[0]),
        (rows[1], rows[0], cols[0], cols[1]),
        (rows[1], rows[0], cols[1], cols[0]),
    ]


def _as_t(x) -> torch.Tensor:
    """fp64 torch view/copy of a numpy array or tensor (CPU)."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.float64)
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float64)))


def _idx_as_t(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(torch.int64)
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.int64)))


# the 4 Gabriel blocks as quadrant-index tuples (row_held, row_comp, col_held,
# col_comp) — SAME order as _bcv_blocks (tests pin the correspondence)
_QUAD_BLOCKS = ((0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0))


def _quads(Fb, rows: list, cols: list) -> list[list[torch.Tensor]]:
    """The 2x2 quadrant sub-matrices Q[a][b] = F[:, rows[a]][:, :, cols[b]],
    sliced ONCE per call (all 4 blocks reuse them — hoisted for throughput)."""
    Ft = _as_t(Fb)
    assert Ft.ndim == 3, tuple(Ft.shape)
    Fr = [Ft.index_select(1, _idx_as_t(rows[0])), Ft.index_select(1, _idx_as_t(rows[1]))]
    cols_t = [_idx_as_t(cols[0]), _idx_as_t(cols[1])]
    return [[Fr[a].index_select(2, cols_t[b]) for b in (0, 1)] for a in (0, 1)]


def _bcv_factor(A11, A12, A21, A22) -> tuple:
    """Factorize ONE block's complement A22 via its Gram eigendecomposition.

    Returns (P, Qs, ss11): Ahat11(r) = P[:, :, :r] @ Qs[:, :r, :], with
    P = A12 V and Qs = diag(inv) V^T (A22^T A21) — algebraically identical to
    the truncated-SVD reconstruction A12 V_r Sigma_r^-1 U_r^T A21 (SVD(A22)).
    Torch internals: numpy's batched matmul over strided views ran ~0.4 GFLOP/s
    (pilot 2026-07-31, 19.8 s per 8-draw call); torch bmm parallelizes it.
    """
    G = A22.transpose(1, 2) @ A22  # (B, k2, k2)
    w, V = torch.linalg.eigh(G)  # ascending
    w = torch.flip(w, dims=[1]).clamp(min=0.0)
    V = torch.flip(V, dims=[2])
    # numerically-null components get zero weight (deterministic guard)
    inv = torch.where(w > w[:, :1] * 1e-12, 1.0 / w.clamp(min=1e-300), torch.zeros_like(w))
    P = A12 @ V  # (B, n1, k2)
    W = A22.transpose(1, 2) @ A21  # (B, k2, k1)
    Qs = (V.transpose(1, 2) @ W) * inv.unsqueeze(2)  # (B, k2, k1)
    ss11 = (A11 * A11).sum(dim=(1, 2))
    assert bool((ss11 > 0).all()), "degenerate held-out block: zero sum of squares"
    return P, Qs, ss11


def _bcv_scores(A11, A12, A21, A22, r_grid: list[int]) -> torch.Tensor:
    """R^2_int(r) for ONE held-out block, (B, len(r_grid)); cumulative over
    eigen-components (cumsum / 2-D cumsum) — no per-r refactorization."""
    P, Qs, ss11 = _bcv_factor(A11, A12, A21, A22)
    Z = A11 @ Qs.transpose(1, 2)  # (B, n1, k2)
    cum_c = torch.cumsum((P * Z).sum(dim=1), dim=1)  # (B, k2)
    H = (P.transpose(1, 2) @ P) * (Qs @ Qs.transpose(1, 2))
    Hc = H.cumsum(dim=1).cumsum(dim=2)
    out = torch.zeros((A11.shape[0], len(r_grid)), dtype=torch.float64)
    for t, r in enumerate(r_grid):
        out[:, t] = (2.0 * cum_c[:, r - 1] - Hc[:, r - 1, r - 1]) / ss11
    return out


def bcv_curve_batched(Fb, r_grid: list[int], rows: list, cols: list) -> np.ndarray:
    """Gabriel (2,2) BCV curves for a batch of matrices.

    Fb: (B, n, k) numpy or torch.  Returns (B, 1+len(r_grid)) numpy with column
    0 the r=0 baseline (predict 0 -> R^2_int = 0 by construction).  Curves
    averaged over the 4 held-out blocks.  All draws in the batch ride the same
    batched GEMMs (vectorize rule); internals are torch fp64 on CPU.
    """
    Q = _quads(Fb, rows, cols)
    acc = torch.zeros((Q[0][0].shape[0], len(r_grid)), dtype=torch.float64)
    for ai, aj, bi, bj in _QUAD_BLOCKS:
        acc += _bcv_scores(Q[ai][bi], Q[ai][bj], Q[aj][bi], Q[aj][bj], r_grid)
    acc /= 4.0
    return np.concatenate([np.zeros((acc.shape[0], 1)), acc.numpy()], axis=1)


def bcv_per_block(F, r_grid: list[int], rows: list, cols: list) -> np.ndarray:
    """TRUE single-block curves (4, 1+len(r_grid)) for one matrix (low-level
    per-unit data behind the 4-block-averaged aggregate)."""
    Fb = np.asarray(F, dtype=np.float64)
    Q = _quads(Fb[None] if Fb.ndim == 2 else Fb, rows, cols)
    out = np.zeros((4, 1 + len(r_grid)))
    for i, (ai, aj, bi, bj) in enumerate(_QUAD_BLOCKS):
        out[i, 1:] = _bcv_scores(Q[ai][bi], Q[ai][bj], Q[aj][bi], Q[aj][bj], r_grid)[0].numpy()
    return out


def bcv_block_predictions(F, r: int, rows: list, cols: list) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per block: (A11, Ahat11(r)) — the held-out entries and their rank-r
    predictions (the predicted-vs-observed scatter's raw data)."""
    Q = _quads(np.asarray(F, dtype=np.float64)[None], rows, cols)
    out = []
    for ai, aj, bi, bj in _QUAD_BLOCKS:
        A11 = Q[ai][bi]
        P, Qs, _ss = _bcv_factor(A11, Q[ai][bj], Q[aj][bi], Q[aj][bj])
        Ahat = P[:, :, :r] @ Qs[:, :r, :]
        out.append((A11[0].numpy(), Ahat[0].numpy()))
    return out


def perm_indices(rng: np.random.Generator, n_draws: int, n: int, k: int) -> np.ndarray:
    """(n_draws, n, k) independent within-column permutation indices."""
    keys = rng.random((n_draws, n, k), dtype=np.float32)
    return np.argsort(keys, axis=1)


def gauss_rows(
    rng: np.random.Generator, n_draws: int, chol: np.ndarray, row_norms: np.ndarray
) -> np.ndarray:
    """(n_draws, n, d) rows i.i.d. N(0, Sigma) row-rescaled to observed ||E_i||."""
    n = row_norms.shape[0]
    d = chol.shape[0]
    z = rng.standard_normal((n_draws, n, d))
    g = np.matmul(z, chol.T)
    norms = np.linalg.norm(g, axis=2)
    g *= (row_norms[None, :] / np.maximum(norms, 1e-300))[:, :, None]
    return g


# ── persistence helpers ──────────────────────────────────────────────────────


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # keep .npz suffix (np.savez appends)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def done_units(path: Path, regime: dict) -> set[str]:
    """Units already completed under the SAME regime (resume predicate)."""
    done = set()
    for row in _read_jsonl(path):
        if row.get("regime") == regime:
            done.add(row["unit"])
    return done


def _band_p975_max(matrix: np.ndarray, lo: int, hi: int) -> float:
    """p97.5 of per-draw max-over-r (r>=1 columns) for rows [lo, hi)."""
    if hi <= lo:
        return float("nan")
    draw_max = matrix[lo:hi, 1:].max(axis=1)
    return float(np.percentile(draw_max, 97.5))


# ── layer context (shared across cells / phases) ─────────────────────────────


class LayerCtx:
    """Per-layer staged targets + cross-fitted split-half PCA bases (parent recipe)."""

    def __init__(self, cfg: Cfg, layer: int):
        t0 = time.time()
        self.layer = layer
        self.Y16, self.ci, self.fp = load_layer(cfg.stage, layer)
        n = self.Y16.shape[0]
        self.folds = make_folds(n, PARENT_SEED)
        self.bases = []
        for basis_idx, eval_idx in self.folds:
            _mu, comps, eigvals = pca_basis(self.Y16[basis_idx], KMAX)
            self.bases.append((basis_idx, eval_idx, comps, eigvals))
        print(
            f"[ctx] layer L{layer} loaded n={n} bases built in {time.time() - t0:.1f}s",
            flush=True,
        )


# ── Tier A (P1) ──────────────────────────────────────────────────────────────


def _cell_tuple(cell_name: str) -> tuple[str, int, str]:
    idx = CELL_NAMES.index(cell_name)
    return CELLS[idx]


def run_tierA(cfg: Cfg, ctxs: dict[int, LayerCtx]) -> dict:
    """P1: BCV rank battery.  Per pipeline unit (cell, k, space): folds pooled.

    Checkpoint: one JSONL row per unit at completion + per-draw x per-r npz
    under percell/; resume skips regime-matched completed units.
    """
    units_path = cfg.eval_dir / "bcv" / "units.jsonl"
    percell = cfg.eval_dir / "percell"
    regime = cfg.regime()
    done = done_units(units_path, regime)
    all_units = [(c, k, s) for c in cfg.cells for k in cfg.k_list for s in cfg.spaces]
    total = len(all_units)
    print(f"[bcv] {total} pipeline units requested, {len(done)} already done", flush=True)
    t_start = time.time()
    timing: dict[str, float] = {"battery_s": 0.0, "setup_s": 0.0, "units": 0}
    unit_idx = 0
    for cell in cfg.cells:
        pend = [(k, s) for k in cfg.k_list for s in cfg.spaces if f"{cell}|k{k}|{s}" not in done]
        if not pend:
            unit_idx += len(cfg.k_list) * len(cfg.spaces)
            continue
        arm, layer, fitter = _cell_tuple(cell)
        ctx = ctxs[layer]
        t_cell = time.time()
        P16 = load_pred(cfg.stage, arm, layer, fitter, ctx.ci, ctx.fp)
        # per (k,space): rows[fold] of the per-draw x per-r matrix, obs stats
        rows_f: dict[tuple[int, str], list[np.ndarray]] = {u: [] for u in pend}
        blocks_f: dict[tuple[int, str], list[np.ndarray]] = {u: [] for u in pend}
        vc_f: dict[tuple[int, str], list[dict]] = {u: [] for u in pend}
        n_eval = []
        for fold_i in cfg.folds:
            basis_idx, eval_idx, comps, eigvals = ctx.bases[fold_i]
            t0 = time.time()
            E = (P16[eval_idx].astype(np.float64) - ctx.Y16[eval_idx].astype(np.float64)) @ comps
            E_b = (
                P16[basis_idx].astype(np.float64) - ctx.Y16[basis_idx].astype(np.float64)
            ) @ comps
            EA = E[:, :KA]
            R = EA**2
            if not np.all(R > 0):  # strict-positivity guard (parent cell_decomposition L235)
                raise AssertionError(
                    f"{cell} fold{fold_i}: non-positive squared residual — log space undefined"
                )
            row_norms = np.linalg.norm(EA, axis=1)
            sigma = np.cov(E_b[:, :KA], rowvar=False)
            chol = np.linalg.cholesky(sigma)
            n_e = E.shape[0]
            n_eval.append(int(n_e))
            # per-unit fixed objects for this fold
            F_of: dict[tuple[int, str], np.ndarray] = {}
            split_of: dict[tuple[int, str], tuple[list, list, list[int]]] = {}
            for k, space in pend:
                lam = eigvals[:k]
                assert np.all(lam > 0), f"non-positive basis eigenvalue at k={k}"
                M = space_transform(R[:, :k], lam, space)
                stats = two_way(M)  # parent closure + degenerate asserts (recorded)
                vc_f[(k, space)].append(
                    {
                        kk: stats[kk]
                        for kk in ("vc_share_interaction", "vc_share_context", "vc_share_direction")
                    }
                )
                F = twoway_removed(M)
                rows, cols = bcv_splits(n_e, k, SEED)
                rg = r_grid_for_k(k)
                F_of[(k, space)] = F
                split_of[(k, space)] = (rows, cols, rg)
                obs = bcv_curve_batched(F[None], rg, rows, cols)  # (1, 1+n_r)
                rows_f[(k, space)].append(obs)
                # TRUE per-block obs curves (4 blocks) for the low-level per-unit plot
                blocks_f[(k, space)].append(bcv_per_block(F, rg, rows, cols))
            timing["setup_s"] += time.time() - t0
            # null draws, chunked; generation shared across (k, space)
            rng_perm = np.random.default_rng([SEED, CELL_NAMES.index(cell), fold_i, 101])
            rng_gauss = np.random.default_rng([SEED, CELL_NAMES.index(cell), fold_i, 202])
            b_left = cfg.b_draws
            perm_rows: dict[tuple[int, str], list[np.ndarray]] = {u: [] for u in pend}
            gauss_rows_acc: dict[tuple[int, str], list[np.ndarray]] = {u: [] for u in pend}
            while b_left > 0:
                bc = min(cfg.chunk, b_left)
                t0 = time.time()
                # keys/z come from the SEEDED numpy generators (plan seed 1945);
                # heavy argsort / gather / transform ride torch (batch-parallel —
                # the numpy path measured ~3.3 s/draw in the 2026-07-31 pilot)
                keys = rng_perm.random((bc, n_e, KA), dtype=np.float32)
                idx_t = torch.argsort(torch.from_numpy(keys), dim=1)
                g_t = torch.from_numpy(gauss_rows(rng_gauss, bc, chol, row_norms))
                for k, space in pend:
                    rows, cols, rg = split_of[(k, space)]
                    Ft = torch.from_numpy(F_of[(k, space)])
                    Fp_t = torch.gather(Ft.unsqueeze(0).expand(bc, n_e, k), 1, idx_t[:, :, :k])
                    perm_rows[(k, space)].append(bcv_curve_batched(Fp_t, rg, rows, cols))
                    lam_t = torch.from_numpy(np.ascontiguousarray(eigvals[:k]))
                    Rg_t = g_t[:, :, :k] ** 2
                    assert bool((Rg_t > 0).all()), "gauss2m draw produced an exact zero"
                    Fg_t = _twoway_removed_t(_space_transform_t(Rg_t, lam_t, space))
                    gauss_rows_acc[(k, space)].append(bcv_curve_batched(Fg_t, rg, rows, cols))
                timing["battery_s"] += time.time() - t0
                b_left -= bc
            for u in pend:
                # fold matrix: obs row + B perm rows + B gauss2m rows -> (1+2B, 1+n_r)
                fold_matrix = np.concatenate(
                    [rows_f[u][-1]] + perm_rows[u] + gauss_rows_acc[u], axis=0
                )
                rows_f[u][-1] = fold_matrix  # replace obs row with full fold matrix
        # combine folds per unit + write
        for k, space in pend:
            unit = f"{cell}|k{k}|{space}"
            per_fold = np.stack(rows_f[(k, space)])  # (n_folds, 1+2B, 1+n_r)
            pooled = per_fold.mean(axis=0)
            rg = r_grid_for_k(k)
            b = cfg.b_draws
            obs_curve = pooled[0]
            obs_max = float(obs_curve[1:].max())
            perm_band = _band_p975_max(pooled, 1, 1 + b)
            gauss_band = _band_p975_max(pooled, 1 + b, 1 + 2 * b)
            _atomic_savez(
                percell / f"{cell}__k{k}__{space}.npz",
                matrix=pooled,
                per_fold=per_fold,
                per_block=np.concatenate(blocks_f[(k, space)], axis=0),
                r_grid=np.array([0] + rg),
                n_perm=np.array([b]),
                n_gauss=np.array([b]),
            )
            elapsed = time.time() - t_start
            row = {
                "unit": unit,
                "cell": cell,
                "k": k,
                "space": space,
                "n_eval_rows": n_eval,
                "r_grid": [0] + rg,
                "obs_curve": [float(x) for x in obs_curve],
                "obs_max": obs_max,
                "perm_p975_max": perm_band,
                "gauss2m_p975_max": gauss_band,
                "delta_g": obs_max - gauss_band,
                "delta_m": obs_max - MAGNITUDE_FLOOR,
                "vc_shares_per_fold": vc_f[(k, space)],
                "elapsed_s": round(elapsed, 2),
                "regime": regime,
                "ts": datetime.now(UTC).isoformat(),
            }
            _append_jsonl(units_path, row)
            unit_idx += 1
            timing["units"] += 1
            print(
                f"[bcv] unit {unit_idx}/{total} {cell}_pooled_k{k}_{space} "
                f"obs_max={obs_max:.4f} perm_p975={perm_band:.4f} "
                f"gauss_p975={gauss_band:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        del P16
        print(f"[bcv] cell {cell} done in {time.time() - t_cell:.1f}s", flush=True)
    (cfg.eval_dir / "bcv" / "DONE.P1").write_text(datetime.now(UTC).isoformat())
    return timing


# ── Tier B (P2) ──────────────────────────────────────────────────────────────


def parity_gate(X: np.ndarray, Y: np.ndarray, n_slices: int = 3) -> dict:
    """Slow-vs-fast ridge parity at the Tier-B production shape (#1332, tol 1e-4).

    Runs n_slices seeded train/eval splits; FAIL -> caller falls back to the
    slow SVD solver (wall impact negligible at 514 features).
    """
    worst = 0.0
    details = []
    for s in range(n_slices):
        rng = np.random.default_rng([SEED, 777, s])
        perm = rng.permutation(X.shape[0])
        tr, te = perm[: X.shape[0] // 2], perm[X.shape[0] // 2 :]
        slow = ridge_fit_predict(X[tr], Y[tr], X[te])
        fast = ridge_fit_predict_fast(X[tr], Y[tr], X[te], device="cpu")
        rel = float(np.abs(fast - slow).max() / np.abs(slow).max())
        details.append(rel)
        worst = max(worst, rel)
    return {
        "max_rel_diff": worst,
        "per_slice": details,
        "tol": PARITY_TOL,
        "pass": worst <= PARITY_TOL,
        "solver": "fast" if worst <= PARITY_TOL else "slow-fallback",
    }


def _tierb_features(P16: np.ndarray, eval_idx: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """phi(x) = [pred @ comps[:, :512], ||pred||, ||pred - proj||] -> (n_e, 514)."""
    pred = P16[eval_idx].astype(np.float64)
    coords = pred @ comps[:, :KMAX]
    pred_norm2 = np.einsum("nd,nd->n", pred, pred)
    proj_norm2 = np.einsum("nk,nk->n", coords, coords)
    resid = np.sqrt(np.maximum(pred_norm2 - proj_norm2, 0.0))
    return np.concatenate([coords, np.sqrt(pred_norm2)[:, None], resid[:, None]], axis=1)


def _rrr_curves(
    yhat_tr: np.ndarray, yhat_te: np.ndarray, f_te: np.ndarray, r_grid: list[int]
) -> np.ndarray:
    """Reduced-rank R^2_int(r) per draw: SVD of train fitted values (plan-verbatim,
    uncentered), test predictions projected onto top-r right singular vectors.

    yhat_tr/yhat_te/f_te: (B, n, k).  Returns (B, 1+len(r_grid)) with r=0 -> 0.
    Torch internals (batch-parallel bmm; numpy in/out).
    """
    Yt, Ye, Fe = _as_t(yhat_tr), _as_t(yhat_te), _as_t(f_te)
    gram = Yt.transpose(1, 2) @ Yt  # (B, k, k)
    _w, V = torch.linalg.eigh(gram)
    V = torch.flip(V, dims=[2])
    T = Ye @ V  # (B, n_te, k)
    Zc = Fe @ V  # (B, n_te, k)
    cross = torch.cumsum((Zc * T).sum(dim=1), dim=1)
    nrm = torch.cumsum((T * T).sum(dim=1), dim=1)
    ss = (Fe * Fe).sum(dim=(1, 2))
    out = np.zeros((Yt.shape[0], 1 + len(r_grid)))
    for t, r in enumerate(r_grid):
        out[:, 1 + t] = ((2.0 * cross[:, r - 1] - nrm[:, r - 1]) / ss).numpy()
    return out


def _batched_null_ridge(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    Y_stack: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared-Gram GCV ridge for a stack of targets (the Tier-B null draws).

    Replicates ridge_fit_predict_fast's recipe (standardize X on train stats,
    center Y, Gram eigh, per-lambda GCV, un-center) with ONE factorization of
    the permutation-invariant X side shared across every draw (plan section 4).

    Y_stack: (B, n_tr + n_te, k) — rows aligned to [X_tr; X_te]; the TRAIN slice
    fits, predictions returned for both slices.
    Returns (yhat_tr (B,n_tr,k), yhat_te (B,n_te,k), best_lam (B,), dof (B,)).
    Torch internals; the shared V^T / KV / V applications are reshaped into ONE
    GEMM across all draws (numpy in/out).
    """
    Xtr, Xte, Yb = _as_t(X_tr), _as_t(X_te), _as_t(Y_stack)
    ntr = Xtr.shape[0]
    xmu = Xtr.mean(dim=0)
    # torch std default correction=1 (Bessel) — matches ridge_fit_predict_fast,
    # whose runtime probe holds this twin to 1e-6
    xsd = Xtr.std(dim=0) + 1e-9
    Xn_tr = (Xtr - xmu) / xsd
    Xn_te = (Xte - xmu) / xsd
    G = Xn_tr @ Xn_tr.T
    w, V = torch.linalg.eigh(G)
    w = w.clamp(min=0.0)
    KV = (Xn_te @ Xn_tr.T) @ V  # cross-kernel rotated into the eigenbasis
    Ytr = Yb[:, :ntr, :]
    ymu = Ytr.mean(dim=1, keepdim=True)  # (B, 1, k)
    Ytr_c = Ytr - ymu
    B, _, kk = Ytr_c.shape
    flat_y = Ytr_c.permute(1, 0, 2).reshape(ntr, B * kk)  # shared-GEMM layout
    VtY = (V.T @ flat_y).reshape(ntr, B, kk).permute(1, 0, 2)  # (B, ntr, k)
    sq = (VtY**2).sum(dim=2)  # (B, ntr)
    tot = (Ytr_c**2).sum(dim=(1, 2))
    best_gcv = torch.full((B,), float("inf"), dtype=torch.float64)
    best_lam = torch.zeros(B, dtype=torch.float64)
    best_dof = torch.zeros(B, dtype=torch.float64)
    best_filt = torch.zeros((B, ntr), dtype=torch.float64)
    for lam in lambdas:
        f = w / (w + lam)
        rss = tot - ((2 * f - f**2)[None, :] * sq).sum(dim=1)
        dof = float(f.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else torch.full_like(rss, float("inf"))
        better = gcv < best_gcv
        best_gcv = torch.where(better, gcv, best_gcv)
        best_lam = torch.where(better, torch.full_like(best_lam, float(lam)), best_lam)
        best_dof = torch.where(better, torch.full_like(best_dof, dof), best_dof)
        best_filt[better] = 1.0 / (w + lam)
    # scaled = diag(1/(w+lam*)) V^T Ytr_c — dual coefficients in the eigenbasis
    scaled = best_filt.unsqueeze(2) * VtY  # (B, ntr, k)
    flat_s = scaled.permute(1, 0, 2).reshape(ntr, B * kk)
    yhat_te = (KV @ flat_s).reshape(-1, B, kk).permute(1, 0, 2) + ymu
    # train fitted values: G alpha = V diag(w) (eigenbasis coeffs) = V (w * scaled)
    yhat_tr = (V @ (w[:, None] * flat_s)).reshape(ntr, B, kk).permute(1, 0, 2) + ymu
    return yhat_tr.numpy(), yhat_te.numpy(), best_lam.numpy(), best_dof.numpy()


def run_tierB(cfg: Cfg, ctxs: dict[int, LayerCtx]) -> None:
    """P2: input-recoverability. log space, k=256, per (cell, fold)."""
    units_path = cfg.eval_dir / "tierb" / "units.jsonl"
    percell = cfg.eval_dir / "percell"
    regime = cfg.regime()
    done = done_units(units_path, regime)
    lambdas = np.logspace(-2, 4, 13)
    rg = [r for r in R_GRID if r <= TIERB_K]
    parity: dict | None = None
    solver_fast = True
    t_start = time.time()
    total = len(cfg.cells) * len(cfg.folds)
    idx_u = 0
    for cell in cfg.cells:
        arm, layer, fitter = _cell_tuple(cell)
        ctx = ctxs[layer]
        pend_folds = [f for f in cfg.folds if f"{cell}|f{f}" not in done]
        idx_u += len(cfg.folds) - len(pend_folds)
        if not pend_folds:
            continue
        P16 = load_pred(cfg.stage, arm, layer, fitter, ctx.ci, ctx.fp)
        for fold_i in pend_folds:
            t0 = time.time()
            basis_idx, eval_idx, comps, eigvals = ctx.bases[fold_i]
            E = (P16[eval_idx].astype(np.float64) - ctx.Y16[eval_idx].astype(np.float64)) @ comps
            R = E[:, :TIERB_K] ** 2
            assert np.all(R > 0), f"{cell} fold{fold_i}: non-positive squared residual"
            lam = eigvals[:TIERB_K]
            F = twoway_removed(np.log(R))
            X = _tierb_features(P16, eval_idx, comps)
            n_e = X.shape[0]
            rng_split = np.random.default_rng([SEED, CELL_NAMES.index(cell), fold_i, 303])
            perm = rng_split.permutation(n_e)
            tr, te = perm[: n_e // 2], perm[n_e // 2 :]
            n_tr, d_feat = tr.shape[0], X.shape[1]
            assert n_tr > d_feat, f"under-determined Tier-B fit: n_tr={n_tr} <= d={d_feat}"
            if parity is None:  # once per run, at production shape (P0 step 3)
                parity = parity_gate(X, F)
                solver_fast = parity["pass"]
                print(f"[tierb] parity gate: {parity}", flush=True)
            # observed fit
            if solver_fast:
                pred_all, info = ridge_fit_predict_fast(
                    X[tr], F[tr], np.concatenate([X[tr], X[te]]), device="cpu", return_info=True
                )
            else:
                pred_all = ridge_fit_predict(X[tr], F[tr], np.concatenate([X[tr], X[te]]))
                info = {"best_lambda": float("nan"), "dof": float("nan"), "gcv": float("nan")}
            yhat_tr_obs = pred_all[: len(tr)]
            yhat_te_obs = pred_all[len(tr) :]
            obs_curve = _rrr_curves(yhat_tr_obs[None], yhat_te_obs[None], F[te][None], rg)[0]
            # equivalence probe: batched twin (identity draw) vs the fast solver,
            # BOTH test and train predictions (yhat_tr feeds the RRR truncation)
            Y_id = np.concatenate([F[tr], F[te]])[None]
            bt_tr, bt_te, id_lam, id_dof = _batched_null_ridge(X[tr], X[te], Y_id, lambdas)
            if solver_fast:
                eq_te = float(np.abs(bt_te[0] - yhat_te_obs).max() / np.abs(yhat_te_obs).max())
                eq_tr = float(np.abs(bt_tr[0] - yhat_tr_obs).max() / np.abs(yhat_tr_obs).max())
                assert max(eq_te, eq_tr) <= 1e-6, (
                    f"batched null-ridge twin diverges from fast solver: "
                    f"te={eq_te:.3e} tr={eq_tr:.3e}"
                )
            # knn retrieval companion (observed full-rank predictions)
            knn = {
                metric: knn_retrieval(yhat_te_obs, F[te], ks=(1, 5, 25), metric=metric)
                for metric in ("euclidean", "cosine")
            }
            # permutation null: row pairing between phi(x) and F permuted
            rng_null = np.random.default_rng([SEED, CELL_NAMES.index(cell), fold_i, 404])
            rows_null: list[np.ndarray] = []
            lam_sel: list[float] = []
            b_left = cfg.b_draws
            while b_left > 0:
                bc = min(cfg.chunk, b_left)
                keys = rng_null.random((bc, n_e), dtype=np.float32)
                pidx = np.argsort(keys, axis=1)  # (bc, n_e) row permutations
                Fp = F[pidx]  # (bc, n_e, k)
                Y_stack = np.concatenate([Fp[:, tr, :], Fp[:, te, :]], axis=1)
                yh_tr, yh_te, blam, _ = _batched_null_ridge(X[tr], X[te], Y_stack, lambdas)
                rows_null.append(_rrr_curves(yh_tr, yh_te, Fp[:, te, :], rg))
                lam_sel.extend(float(x) for x in blam)
                b_left -= bc
            matrix = np.concatenate([obs_curve[None]] + rows_null, axis=0)
            obs_max = float(obs_curve[1:].max())
            band = _band_p975_max(matrix, 1, matrix.shape[0])
            _atomic_savez(
                percell / f"tierb__{cell}__f{fold_i}.npz",
                matrix=matrix,
                r_grid=np.array([0] + rg),
                n_perm=np.array([cfg.b_draws]),
            )
            elapsed = time.time() - t_start
            idx_u += 1
            row = {
                "unit": f"{cell}|f{fold_i}",
                "cell": cell,
                "fold": fold_i,
                "space": "log",
                "k": TIERB_K,
                "n_train": int(n_tr),
                "d_feat": int(d_feat),
                "obs_curve": [float(x) for x in obs_curve],
                "obs_max": obs_max,
                "perm_p975_max": band,
                "delta_perm": obs_max - band,
                "ridge_info": {kk: float(v) for kk, v in info.items()},
                # Gram-eigh twin's GCV selection on the OBSERVED (identity-draw)
                # target — the selected-lambda/dof report when the parity gate
                # routed the observed fit to the info-less slow SVD solver
                "ridge_info_twin": {"best_lambda": float(id_lam[0]), "dof": float(id_dof[0])},
                "null_lambda_median": float(np.median(lam_sel)),
                "parity_gate": parity,
                "knn_retrieval": knn,
                "r_grid": [0] + rg,
                "elapsed_s": round(time.time() - t0, 2),
                "regime": regime,
                "ts": datetime.now(UTC).isoformat(),
            }
            _append_jsonl(units_path, row)
            print(
                f"[tierb] unit {idx_u}/{total} {cell}_f{fold_i} obs_max={obs_max:.4f} "
                f"perm_p975={band:.4f} lam={info.get('best_lambda')} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        del P16
    (cfg.eval_dir / "tierb" / "DONE.P2").write_text(datetime.now(UTC).isoformat())


# ── P3: floors + summaries + figures ─────────────────────────────────────────


def run_floor(cfg: Cfg, ctxs: dict[int, LayerCtx]) -> None:
    """P3a/b: floor-corrected subsample replication + floor-noise-only synthetic.

    Raw + normalized spaces only (a scalar row correction is a NO-OP in log
    space — absorbed by a_i; registered observation, plan section 4 P3a).
    """
    units_path = cfg.eval_dir / "floor" / "units.jsonl"
    percell = cfg.eval_dir / "percell"
    regime = cfg.regime()
    done = done_units(units_path, regime)
    k = 256
    rg = r_grid_for_k(k)
    netting: dict[str, dict] = {}
    for layer in cfg.floor_layers:
        fpath = resolve_input(f"eval_results/issue_1738/kresample/floors_L{layer}.npz")
        fz = np.load(fpath)
        assert set(fz.files) == FLOOR_KEYS, f"floors_L{layer}: keys {sorted(fz.files)}"
        fci, floor = fz["ci"], fz["floor"]
        ctx = ctxs[layer]
        # PARENT-VERBATIM join (issue1482 floor_correction): floors are keyed by
        # conversation id (global corpus index); translate to HOLDOUT ROW INDICES
        # via pos before joining against eval_idx (which holds row indices).
        # Joining on raw ci values silently mis-selects rows (#1345 join class).
        pos = {int(c): i for i, c in enumerate(ctx.ci)}
        if not all(int(c) in pos for c in fci):
            raise AssertionError(f"L{layer}: floor ci are not a subset of the holdout ci")
        frow = np.array([pos[int(c)] for c in fci])
        fmap = dict(zip(frow.tolist(), floor.tolist(), strict=True))
        for arm in ARMS:
            cell = f"{arm}_L{layer}_ridge"
            if cell not in cfg.cells:
                continue
            P16 = load_pred(cfg.stage, arm, layer, "ridge", ctx.ci, ctx.fp)
            rows_f: dict[str, list[np.ndarray]] = {s: [] for s in ("raw", "normalized")}
            # log-space synthetic share is UNDEFINED: 49/1988 K-resample floors are
            # exactly 0 -> identically-zero synthetic rows -> log(0) = -inf -> NaN
            synth_share: dict[str, list[float]] = {s: [] for s in ("raw", "normalized")}
            n_subs = []
            for fold_i in cfg.folds:
                basis_idx, eval_idx, comps, eigvals = ctx.bases[fold_i]
                keep = np.array([i for i, r in enumerate(eval_idx) if int(r) in fmap])
                sub = eval_idx[keep]
                Efull = P16[sub].astype(np.float64) - ctx.Y16[sub].astype(np.float64)
                e2 = (Efull**2).sum(axis=1)
                fl = np.array([fmap[int(r)] for r in sub])
                frac = np.clip(1.0 - fl / e2, 0.0, 1.0)
                elig = frac > 0
                Epca = Efull[elig] @ comps
                EA = Epca[:, :KA]
                R = EA**2
                assert np.all(R > 0), f"floor {cell} fold{fold_i}: non-positive residual"
                row_scale = frac[elig]
                n_sub = int(elig.sum())
                n_subs.append(n_sub)
                E_b = (
                    P16[basis_idx].astype(np.float64) - ctx.Y16[basis_idx].astype(np.float64)
                ) @ comps
                chol = np.linalg.cholesky(np.cov(E_b[:, :KA], rowvar=False))
                row_norms = np.linalg.norm(EA, axis=1)
                lam = eigvals[:k]
                # (b) floor-noise-only synthetic share (parent isotropy assumption)
                d_full = ctx.Y16.shape[1]
                rng_syn = np.random.default_rng([SEED, layer, ARMS.index(arm), fold_i, 505])
                e_syn = np.sqrt(fl[elig] / d_full)[:, None] * rng_syn.standard_normal((n_sub, k))
                R_syn = e_syn**2
                for space in ("raw", "normalized"):
                    # share of the interaction VARIANCE COMPONENT itself
                    # (EMS sigma^2_e = MS_e), synth / observed subsample R
                    synth_share[space].append(
                        float(
                            _vc_e_abs(space_transform(R_syn, lam, space))
                            / _vc_e_abs(space_transform(R[:, :k], lam, space))
                        )
                    )
                # (a) corrected-subsample Tier-A replication, raw + normalized
                for space in ("raw", "normalized"):
                    if f"floor_{cell}|{space}" in done:
                        continue
                    Rc = R[:, :k] * row_scale[:, None]
                    M = space_transform(Rc, lam, space)
                    two_way(M)  # closure asserts
                    F = twoway_removed(M)
                    rows, cols = bcv_splits(n_sub, k, SEED)
                    obs = bcv_curve_batched(F[None], rg, rows, cols)
                    rng_perm = np.random.default_rng([SEED, layer, ARMS.index(arm), fold_i, 606])
                    rng_g = np.random.default_rng([SEED, layer, ARMS.index(arm), fold_i, 707])
                    chunks: list[np.ndarray] = [obs]
                    b_left = cfg.b_draws
                    while b_left > 0:
                        bc = min(cfg.chunk, b_left)
                        idx = perm_indices(rng_perm, bc, n_sub, k)
                        Fp = np.take_along_axis(np.broadcast_to(F, (bc, n_sub, k)), idx, axis=1)
                        chunks.append(bcv_curve_batched(Fp, rg, rows, cols))
                        b_left -= bc
                    b_left = cfg.b_draws
                    while b_left > 0:
                        bc = min(cfg.chunk, b_left)
                        g = gauss_rows(rng_g, bc, chol, row_norms)
                        Rg = (g[:, :, :k] ** 2) * row_scale[None, :, None]
                        Mg = space_transform(Rg, lam, space)
                        Fg = twoway_removed(Mg)
                        chunks.append(bcv_curve_batched(Fg, rg, rows, cols))
                        b_left -= bc
                    rows_f[space].append(np.concatenate(chunks, axis=0))
            for space in ("raw", "normalized"):
                unit = f"floor_{cell}|{space}"
                if unit in done or not rows_f[space]:
                    continue
                per_fold = np.stack(rows_f[space])
                pooled = per_fold.mean(axis=0)
                b = cfg.b_draws
                obs_max = float(pooled[0][1:].max())
                _atomic_savez(
                    percell / f"floor__{cell}__{space}.npz",
                    matrix=pooled,
                    per_fold=per_fold,
                    r_grid=np.array([0] + rg),
                    n_perm=np.array([b]),
                    n_gauss=np.array([b]),
                )
                row = {
                    "unit": unit,
                    "cell": cell,
                    "space": space,
                    "k": k,
                    "n_sub": n_subs,
                    "row_scale": "frac = clip(1 - floor/e2, 0, 1) (parent recipe)",
                    "log_space_note": "scalar row correction is a no-op in log space (absorbed by a_i)",
                    "obs_curve": [float(x) for x in pooled[0]],
                    "obs_max": obs_max,
                    "perm_p975_max": _band_p975_max(pooled, 1, 1 + b),
                    "gauss2m_p975_max": _band_p975_max(pooled, 1 + b, 1 + 2 * b),
                    "r_grid": [0] + rg,
                    "regime": regime,
                    "ts": datetime.now(UTC).isoformat(),
                }
                _append_jsonl(units_path, row)
                print(f"[floor] {unit} obs_max={obs_max:.4f}", flush=True)
            netting[cell] = {
                "floor_share_of_interaction": {
                    s: float(np.mean(v)) for s, v in synth_share.items() if v
                },
                "log_space_synth_note": (
                    "log-space synthetic share undefined — zero-valued K-resample "
                    "floors produce identically-zero synthetic rows (log(0)); the "
                    "raw + normalized shares carry the netting read"
                ),
                "n_sub_per_fold": n_subs,
            }
            del P16
    out = cfg.eval_dir / "floor" / "floor_netting.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "metadata": _metadata(cfg),
        "recipe": "rows = sqrt(floor_i/d) * N(0, I_k) (parent isotropy assumption); "
        "share = vc_e(synth) / vc_e(observed subsample R), matched (n, k) geometry",
        "per_cell": netting,
    }
    out.write_text(json.dumps(doc, indent=2))
    print(f"[floor] wrote {out}", flush=True)


def _vc_e_abs(M: np.ndarray) -> float:
    """Absolute interaction variance component (EMS sigma^2_e = MS_e)."""
    n, kk = M.shape
    mu = M.mean()
    a = M.mean(axis=1) - mu
    b = M.mean(axis=0) - mu
    resid = M - mu - a[:, None] - b[None, :]
    return float((resid**2).sum() / ((n - 1) * (kk - 1)))


def _verdict(obs_max: float, gauss_band: float) -> str:
    if obs_max > gauss_band and obs_max >= MAGNITUDE_FLOOR:
        return "H1-structured"
    if obs_max > gauss_band:
        return "H1-weak-structured"
    return "H2-idiosyncratic"


def run_summaries(cfg: Cfg) -> dict:
    """P3c: assemble bcv_summary.json + tierb_summary.json from unit JSONLs."""
    regime = cfg.regime()
    bcv_rows = [
        r for r in _read_jsonl(cfg.eval_dir / "bcv" / "units.jsonl") if r["regime"] == regime
    ]
    latest: dict[str, dict] = {}
    for r in bcv_rows:
        latest[r["unit"]] = r
    n_expected = len(cfg.cells) * len(cfg.k_list) * len(cfg.spaces)
    if not cfg.smoke and len(latest) != n_expected:
        raise AssertionError(f"bcv units incomplete: {len(latest)}/{n_expected}")
    units_out = []
    for unit, r in sorted(latest.items()):
        units_out.append(
            {
                "unit": unit,
                "obs_max": r["obs_max"],
                "obs_curve": r["obs_curve"],
                "r_grid": r["r_grid"],
                "perm_p975_max": r["perm_p975_max"],
                "gauss2m_p975_max": r["gauss2m_p975_max"],
                "delta_g": r["obs_max"] - r["gauss2m_p975_max"],
                "delta_m": r["obs_max"] - MAGNITUDE_FLOOR,
                "verdict": _verdict(r["obs_max"], r["gauss2m_p975_max"]),
                "band_to_ceiling_margin": 1.0 - r["gauss2m_p975_max"],
                "dv_ceiling": 1.0,
                "vc_shares_per_fold": r.get("vc_shares_per_fold"),
            }
        )
    primary_key = f"{PRIMARY_UNIT[0]}|k{PRIMARY_UNIT[1]}|{PRIMARY_UNIT[2]}"
    primary = latest.get(primary_key)
    primary_block = None
    if primary is not None:
        X = primary["obs_max"]
        G = primary["gauss2m_p975_max"]
        primary_block = {
            "unit": primary_key,
            "X_obs_max_over_r": X,
            "G_gauss2m_p975_per_draw_max": G,
            "perm_p975_per_draw_max": primary["perm_p975_max"],
            "delta_g": X - G,
            "delta_m": X - MAGNITUDE_FLOOR,
            "magnitude_floor": MAGNITUDE_FLOOR,
            "verdict": _verdict(X, G),
            "selection": "per-draw max-over-r (Option 1, selection-symmetric)",
            "band_to_ceiling_margin": 1.0 - G,
            "ceiling_note": "DV bounded at 1.0; a band >= ceiling would force "
            "failure-to-reject narration (selection-symmetric-nulls rule)",
        }
    doc = {
        "metadata": _metadata(cfg),
        "design": {
            "r_grid": list(R_GRID),
            "b_draws_per_family": cfg.b_draws,
            "null_families": [
                "perm (within-column permutation of F)",
                "gauss2m (rows N(0, Sigma_E), row-norm matched)",
            ],
            "spaces": list(cfg.spaces),
            "k_list": list(cfg.k_list),
            "fold_pooling": "curves averaged over 4 blocks x 2 parent folds",
        },
        "n_units": len(units_out),
        "primary": primary_block,
        "units": units_out,
    }
    out = cfg.eval_dir / "bcv" / "bcv_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(f"[summary] wrote {out} ({len(units_out)} units)", flush=True)

    tb_rows = [
        r for r in _read_jsonl(cfg.eval_dir / "tierb" / "units.jsonl") if r["regime"] == regime
    ]
    tb_latest: dict[str, dict] = {}
    for r in tb_rows:
        tb_latest[r["unit"]] = r
    tb_doc = {
        "metadata": _metadata(cfg),
        "scope_caveat": "phi(x) is a linear image of the map's input through the fitted "
        "map — a positive read certifies input-recoverability; a null read does NOT rule "
        "out recoverability from the raw input x (plan section 4 P2).",
        "identity_bias_baseline": "inapplicable by dimension mismatch (514-dim features "
        "vs 256-dim squared-error-profile targets) — stated, not silently skipped",
        "n_units": len(tb_latest),
        "units": [tb_latest[u] for u in sorted(tb_latest)],
    }
    out2 = cfg.eval_dir / "tierb" / "tierb_summary.json"
    out2.parent.mkdir(parents=True, exist_ok=True)
    out2.write_text(json.dumps(tb_doc, indent=2))
    print(f"[summary] wrote {out2} ({len(tb_latest)} fold-cells)", flush=True)
    return {"bcv_units": len(units_out), "tierb_units": len(tb_latest)}


def run_figures(cfg: Cfg, ctxs: dict[int, LayerCtx]) -> list[Path]:
    """P3c figures. Smoke renders the hero + null-histogram + scatter subset only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    regime = cfg.regime()
    written: list[Path] = []
    percell = cfg.eval_dir / "percell"
    pal = paper_palette(4)

    def _hero(space: str, name: str) -> None:
        cell, k, _ = PRIMARY_UNIT
        z = np.load(percell / f"{cell}__k{k}__{space}.npz")
        m, rg = z["matrix"], z["r_grid"]
        b = int(z["n_perm"][0])
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(rg, m[0], "o-", color=pal[0], label="observed")
        perm_band = _band_p975_max(m, 1, 1 + b)
        gauss_band = _band_p975_max(m, 1 + b, 1 + 2 * b)
        ax.axhline(
            perm_band, color=pal[1], ls="--", label=f"perm p97.5 (per-draw max) = {perm_band:.3f}"
        )
        ax.axhline(
            gauss_band,
            color=pal[2],
            ls="-.",
            label=f"gauss2m p97.5 (per-draw max) = {gauss_band:.3f}",
        )
        if "per_block" in z.files:
            pb = z["per_block"]
            for i in range(pb.shape[0]):
                ax.plot(rg, pb[i], color=pal[0], alpha=0.15, lw=0.8)
        ax.set_xlabel("rank r")
        ax.set_ylabel("held-out interaction R^2")
        ax.set_title(f"BCV interaction R^2 — {cell}, k={k}, {space} (B={b}/family)")
        ax.legend()
        p = cfg.fig_dir / name
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    _hero("log", "hero_bcv_primary_log.png")

    # observed-vs-null per-draw max histogram (primary)
    cell, k, space = PRIMARY_UNIT
    z = np.load(percell / f"{cell}__k{k}__{space}.npz")
    m = z["matrix"]
    b = int(z["n_perm"][0])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(
        m[1 : 1 + b, 1:].max(axis=1), bins=30, alpha=0.6, color=pal[1], label="perm per-draw max"
    )
    ax.hist(
        m[1 + b :, 1:].max(axis=1), bins=30, alpha=0.6, color=pal[2], label="gauss2m per-draw max"
    )
    ax.axvline(float(m[0, 1:].max()), color=pal[0], lw=2, label="observed max")
    ax.set_xlabel("max-over-r BCV interaction R^2")
    ax.set_ylabel("draws")
    ax.set_title(f"Observed vs null (per-draw max) — {cell}, k={k}, {space}")
    ax.legend()
    p = cfg.fig_dir / "null_hist_primary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # per-block SCATTER: predicted vs observed held-out interaction entries at
    # the pooled argmax r (primary unit, fold 0) — the low-level per-unit data
    # behind the aggregate R^2 (plan section 6 figure list)
    r_star = int(np.asarray(z["r_grid"])[1:][int(np.argmax(m[0][1:]))])
    arm, layer, fitter = _cell_tuple(cell)
    ctx = ctxs[layer]
    _basis_idx, eval_idx, comps, eigvals = ctx.bases[0]
    P16 = load_pred(cfg.stage, arm, layer, fitter, ctx.ci, ctx.fp)
    E = (P16[eval_idx].astype(np.float64) - ctx.Y16[eval_idx].astype(np.float64)) @ comps
    R = E[:, :k] ** 2
    F = twoway_removed(space_transform(R, eigvals[:k], space))
    rows_sp, cols_sp = bcv_splits(E.shape[0], k, SEED)
    pairs = bcv_block_predictions(F, r_star, rows_sp, cols_sp)
    del P16, E, R, F
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    rng_sc = np.random.default_rng(SEED)
    for i, (a11, ahat) in enumerate(pairs):
        flat_o, flat_p = a11.ravel(), ahat.ravel()
        pick = rng_sc.choice(flat_o.size, size=min(2000, flat_o.size), replace=False)
        ax.scatter(flat_o[pick], flat_p[pick], s=3, alpha=0.25, color=pal[i], label=f"block {i}")
    lo = min(float(np.min([a.min() for a, _ in pairs])), float(np.min([b.min() for _, b in pairs])))
    hi = max(float(np.max([a.max() for a, _ in pairs])), float(np.max([b.max() for _, b in pairs])))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("observed held-out interaction entry F[i,j]")
    ax.set_ylabel(f"rank-{r_star} BCV prediction")
    ax.set_title(f"Predicted vs observed held-out entries — {cell}, fold 0, r={r_star}")
    ax.legend(markerscale=3)
    p = cfg.fig_dir / "per_block_scatter_primary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    if cfg.smoke:
        return written

    _hero("raw", "hero_bcv_primary_raw.png")
    _hero("normalized", "hero_bcv_primary_normalized.png")

    rows = [r for r in _read_jsonl(cfg.eval_dir / "bcv" / "units.jsonl") if r["regime"] == regime]
    latest = {r["unit"]: r for r in rows}
    # verdict heatmap: delta_g per (cell x k) per space
    fig, axes = plt.subplots(1, len(SPACES), figsize=(15, 5), constrained_layout=True)
    for ax, space in zip(np.atleast_1d(axes), SPACES, strict=False):
        grid = np.full((len(CELL_NAMES), len(K_LIST)), np.nan)
        for i, cname in enumerate(CELL_NAMES):
            for j, kk in enumerate(K_LIST):
                r = latest.get(f"{cname}|k{kk}|{space}")
                if r:
                    grid[i, j] = r["obs_max"] - r["gauss2m_p975_max"]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
        ax.set_xticks(range(len(K_LIST)), [f"k={kk}" for kk in K_LIST])
        ax.set_yticks(range(len(CELL_NAMES)), CELL_NAMES, fontsize=7)
        ax.set_title(f"delta_g ({space})")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
    p = cfg.fig_dir / "verdict_heatmap_delta_g.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)

    # per-arm small multiples (log, k=256)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True, constrained_layout=True)
    for i, cname in enumerate(CELL_NAMES):
        ax = axes.flat[i]
        r = latest.get(f"{cname}|k256|log")
        if r:
            ax.plot(r["r_grid"], r["obs_curve"], "o-", color=pal[0])
            ax.axhline(r["gauss2m_p975_max"], color=pal[2], ls="-.")
            ax.axhline(r["perm_p975_max"], color=pal[1], ls="--")
        ax.set_title(cname, fontsize=8)
    p = cfg.fig_dir / "per_cell_curves_log_k256.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)

    # Tier B: curves + retrieval bars
    tb = [r for r in _read_jsonl(cfg.eval_dir / "tierb" / "units.jsonl") if r["regime"] == regime]
    if tb:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for r in tb:
            ax1.plot(r["r_grid"], r["obs_curve"], alpha=0.6, lw=1)
        ax1.set_xlabel("rank r")
        ax1.set_ylabel("held-out interaction R^2")
        ax1.set_title("Tier B reduced-rank ridge curves (all fold-cells, log k=256)")
        prim = [r for r in tb if r["cell"] == PRIMARY_UNIT[0]]
        if prim:
            kk_labels, accs, chances = [], [], []
            for metric in ("euclidean", "cosine"):
                for kq, acc in prim[0]["knn_retrieval"][metric]["acc_at_k"].items():
                    kk_labels.append(f"{metric[:3]}@{kq}")
                    accs.append(
                        np.mean([p_["knn_retrieval"][metric]["acc_at_k"][kq] for p_ in prim])
                    )
                    chances.append(prim[0]["knn_retrieval"][metric]["chance_at_k"][kq])
            xpos = np.arange(len(kk_labels))
            ax2.bar(xpos - 0.2, accs, 0.4, color=pal[0], label="observed")
            ax2.bar(xpos + 0.2, chances, 0.4, color=pal[3], label="chance = k/n_pool")
            ax2.set_xticks(xpos, kk_labels)
            ax2.set_title(f"Tier B retrieval — {PRIMARY_UNIT[0]} (fold mean)")
            ax2.legend()
        p = cfg.fig_dir / "tierb_curves_retrieval.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)

    # floor share bars
    fn = cfg.eval_dir / "floor" / "floor_netting.json"
    if fn.exists():
        doc = json.loads(fn.read_text())
        cells = sorted(doc["per_cell"])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        width = 0.3
        # log-space synthetic share undefined under zero floors (see run_floor)
        for si, space in enumerate(("raw", "normalized")):
            vals = [
                doc["per_cell"][c]["floor_share_of_interaction"].get(space, np.nan) for c in cells
            ]
            ax.bar(
                np.arange(len(cells)) + (si - 0.5) * width, vals, width, color=pal[si], label=space
            )
        ax.set_xticks(range(len(cells)), cells, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("floor share of interaction vc")
        ax.set_title("Answer-sampling floor share of the interaction (subsample)")
        ax.legend()
        p = cfg.fig_dir / "floor_share_bars.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # per-block scatter proxy: per-block curves for the primary unit
    z = np.load(percell / f"{PRIMARY_UNIT[0]}__k{PRIMARY_UNIT[1]}__{PRIMARY_UNIT[2]}.npz")
    if "per_block" in z.files:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        pb = z["per_block"]
        for i in range(pb.shape[0]):
            ax.plot(z["r_grid"], pb[i], "o-", alpha=0.5, lw=1, label=f"block {i}")
        ax.plot(z["r_grid"], z["matrix"][0], "k-", lw=2, label="pooled")
        ax.set_xlabel("rank r")
        ax.set_ylabel("held-out interaction R^2")
        ax.set_title("Per-block BCV curves — primary unit (low-level per-unit data)")
        ax.legend(fontsize=6)
        p = cfg.fig_dir / "per_block_curves_primary.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    return written


# ── P0 smoke + pilot ─────────────────────────────────────────────────────────


def run_p0(cfg: Cfg, ctxs: dict[int, LayerCtx]) -> dict:
    """P0: asserts + grouping probe + parity gate + timed 1-unit pilot."""
    report: dict = {"metadata": _metadata(cfg)}
    # 1. realized-keys asserts on EVERY staged npz used
    stage = cfg.stage
    for layer in LAYERS:
        z = np.load(stage / f"y_parent_L{layer}.npz")
        assert set(z.files) == Y_KEYS, f"y L{layer}: {sorted(z.files)}"
    for arm, layer, fitter in CELLS:
        z = np.load(stage / f"pred_{arm}_L{layer}_{fitter}.npz")
        assert set(z.files) == PRED_KEYS, f"pred {arm} L{layer} {fitter}: {sorted(z.files)}"
    for layer in LAYERS:
        fz = np.load(resolve_input(f"eval_results/issue_1738/kresample/floors_L{layer}.npz"))
        assert set(fz.files) == FLOOR_KEYS, f"floors L{layer}: {sorted(fz.files)}"
    report["realized_keys"] = "PASS (12 pred + 3 y + 3 floors)"
    print("[p0] realized-keys asserts PASS", flush=True)
    # 2. grouping probe (A9): distinct conversation ids
    grouping = {}
    for layer in LAYERS:
        ci = load_layer(stage, layer)[1]
        n_distinct = len(set(int(c) for c in ci))
        assert n_distinct == len(ci) == 9941, f"L{layer}: {n_distinct} distinct of {len(ci)}"
        grouping[f"L{layer}"] = {"n_rows": int(len(ci)), "n_distinct_ci": n_distinct}
    report["grouping_probe"] = grouping
    report["grouping_note"] = (
        "one context per conversation (distinct global corpus indices; #1738 carve) — "
        "row folds are conversation-level by construction; fallback not needed"
    )
    print(f"[p0] grouping probe PASS: {grouping}", flush=True)
    # 2b. fold-identity cross-check (A13): fold-0 basis eigvals vs parent JSON
    tw = json.loads(
        resolve_input("eval_results/issue_1482/twoway_residual/twoway_residual.json").read_text()
    )
    ref = np.asarray(tw["basis_fidelity"]["L19"]["splithalf_head_fold0"], dtype=np.float64)
    got = np.asarray(ctxs[19].bases[0][3][:8], dtype=np.float64)
    rel = float(np.max(np.abs(got - ref) / np.abs(ref)))
    assert rel < 1e-6, f"fold identity vs parent basis_fidelity failed: rel dev {rel:.3e}"
    report["fold_identity_rel_dev"] = rel
    print(f"[p0] fold-identity cross-check PASS (rel dev {rel:.2e})", flush=True)
    # 3. ridge parity gate at Tier-B production shape (primary cell fold 0)
    ctx = ctxs[19]
    P16 = load_pred(stage, "context", 19, "ridge", ctx.ci, ctx.fp)
    basis_idx, eval_idx, comps, eigvals = ctx.bases[0]
    E = (P16[eval_idx].astype(np.float64) - ctx.Y16[eval_idx].astype(np.float64)) @ comps
    R = E[:, :TIERB_K] ** 2
    F = twoway_removed(np.log(R))
    X = _tierb_features(P16, eval_idx, comps)
    t0 = time.time()
    gate = parity_gate(X, F)
    gate["shape"] = {"n_rows": int(X.shape[0]), "d_feat": int(X.shape[1])}
    gate["wall_s"] = round(time.time() - t0, 2)
    report["parity_gate"] = gate
    print(f"[p0] parity gate: {gate}", flush=True)
    del P16, E, R, F, X
    # 4. timed 1-unit pilot through the PRODUCTION entrypoint (run_tierA)
    pilot_cfg = Cfg(
        phase="p1",
        smoke=True,
        b_draws=8,
        cells=(PRIMARY_UNIT[0],),
        k_list=(256,),
        spaces=("log",),
        out_root=cfg.out_root,
        stage=cfg.stage,
        chunk=cfg.chunk,
    )
    t0 = time.time()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    timing = run_tierA(pilot_cfg, ctxs)
    pilot_wall = time.time() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    # extrapolation: battery scales with draws x unit count; setup with cells
    pilot_draws = 2 * (1 + 2 * 8)  # 2 folds x (1 obs + 8 perm + 8 gauss)
    full_draws = 2 * (1 + 2 * DEFAULT_B)
    n_k256 = len(CELL_NAMES) * len(SPACES)  # 36 units at k=256
    n_k64 = len(CELL_NAMES) * len(SPACES)  # 36 units at k=64, ~(32/128)^3 GEMM cost
    battery_unit_s = timing["battery_s"] * (full_draws / pilot_draws)
    battery_total_s = battery_unit_s * (n_k256 + n_k64 * (32 / 128) ** 2)
    setup_total_s = timing["setup_s"] * len(CELL_NAMES) * len(SPACES) * len(K_LIST)
    layer_fixed_s = 60.0 * len(LAYERS)  # basis eigh + loads, measured ~20s/fold-layer
    projected_h = (battery_total_s + setup_total_s + layer_fixed_s) / 3600.0
    pilot = {
        "pilot_wall_s": round(pilot_wall, 2),
        "pilot_battery_s": round(timing["battery_s"], 2),
        "pilot_setup_s": round(timing["setup_s"], 2),
        "pilot_b_draws": 8,
        "ru_maxrss_gb_before": round(rss0 / 1024, 2),
        "ru_maxrss_gb_after": round(rss1 / 1024, 2),
        "extrapolation": {
            "formula": "battery_s x (draws_full/draws_pilot) x (36 + 36x(32/128)^2 units) "
            "+ setup_s x 72 + 60s x 3 layers",
            "projected_p1_wall_h": round(projected_h, 3),
            "booked_p1_wall_h": P1_BOOKED_WALL_H,
            "ratio_vs_booked": round(projected_h / P1_BOOKED_WALL_H, 3),
            "abort_threshold": PILOT_ABORT_FACTOR,
        },
    }
    report["pilot"] = pilot
    out = cfg.smoke_dir / "pilot_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[p0] pilot report -> {out}", flush=True)
    if projected_h > PILOT_ABORT_FACTOR * P1_BOOKED_WALL_H:
        print(
            f"[p0] ABORT: projected P1 wall {projected_h:.2f}h > "
            f"{PILOT_ABORT_FACTOR}x booked {P1_BOOKED_WALL_H}h — vectorize-signature "
            f"check required before proceeding (plan section 9)",
            flush=True,
        )
        sys.exit(RC_PILOT_ABORT)
    return report


# ── main ─────────────────────────────────────────────────────────────────────


def _layers_needed(cfg: Cfg) -> set[int]:
    layers = {_cell_tuple(c)[1] for c in cfg.cells}
    if cfg.phase in ("p0", "all"):
        layers |= set(LAYERS)
    if cfg.phase in ("p3", "all"):
        layers |= set(cfg.floor_layers)
    return layers


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("p0", "p1", "p2", "p3", "all"), default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny-slice mode (B=8 default)")
    ap.add_argument("--b-draws", type=int, default=None, help="null draws per family")
    ap.add_argument("--cells", type=str, default=None, help="CSV of cell names")
    ap.add_argument("--k-list", type=str, default=None, help="CSV of k values")
    ap.add_argument("--spaces", type=str, default=None, help="CSV of spaces")
    ap.add_argument("--folds", type=str, default=None, help="CSV of parent folds (0,1)")
    ap.add_argument("--floor-layers", type=str, default=None, help="CSV of floor layers")
    ap.add_argument("--out-root", type=str, default=None, help="output root (default repo)")
    args = ap.parse_args(argv)

    cells = tuple(args.cells.split(",")) if args.cells else CELL_NAMES
    for c in cells:
        assert c in CELL_NAMES, f"unknown cell {c!r}; valid: {CELL_NAMES}"
    b = args.b_draws if args.b_draws is not None else (8 if args.smoke else DEFAULT_B)
    cfg = Cfg(
        phase=args.phase,
        smoke=args.smoke,
        b_draws=b,
        cells=cells if not args.smoke or args.cells else (PRIMARY_UNIT[0],),
        k_list=tuple(int(x) for x in args.k_list.split(",")) if args.k_list else K_LIST,
        spaces=tuple(args.spaces.split(",")) if args.spaces else SPACES,
        folds=tuple(int(x) for x in args.folds.split(",")) if args.folds else (0, 1),
        floor_layers=(
            tuple(int(x) for x in args.floor_layers.split(","))
            if args.floor_layers
            else ((19,) if args.smoke else LAYERS)
        ),
        out_root=Path(args.out_root).resolve() if args.out_root else PROJECT_ROOT,
    )
    _assert_headroom(cfg.out_root)
    print(
        f"[main] phase={cfg.phase} smoke={cfg.smoke} b={cfg.b_draws} "
        f"cells={len(cfg.cells)} k={cfg.k_list} spaces={cfg.spaces} "
        f"stage={cfg.stage} out={cfg.out_root}",
        flush=True,
    )
    t0 = time.time()
    ctxs = {layer: LayerCtx(cfg, layer) for layer in sorted(_layers_needed(cfg))}
    if cfg.phase == "p0":
        run_p0(cfg, ctxs)
    if cfg.phase == "all":
        # P0 assert block (keys, grouping, fold identity) without the timed pilot
        for layer in LAYERS:
            ci = ctxs[layer].ci
            assert len(set(int(c) for c in ci)) == len(ci) == 9941
    if cfg.phase in ("p1", "all"):
        run_tierA(cfg, ctxs)
    if cfg.phase in ("p2", "all"):
        run_tierB(cfg, ctxs)
    if cfg.phase in ("p3", "all"):
        run_floor(cfg, ctxs)
        run_summaries(cfg)
        figs = run_figures(cfg, ctxs)
        print(f"[figures] wrote {len(figs)}: {[str(p) for p in figs]}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(
        f"[main] done phase={cfg.phase} wall={time.time() - t0:.1f}s ru_maxrss={rss:.2f}GB",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit — C-extension atexit teardown race (gotchas)


if __name__ == "__main__":
    main()
