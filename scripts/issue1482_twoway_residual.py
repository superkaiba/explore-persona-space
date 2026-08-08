"""Issue #1482 inline round — two-way context x direction residual decomposition.

Result 1 of ``docs/results_summaries/2026-07-30-what-is-the-map-bad-at-predicting.md``
asks whether the context->answer map fails at specific DIRECTIONS or specific
CONTEXTS. It rests on two SEPARATE marginal reads (a per-context Lorenz curve, a
per-direction PCA-band profile); the joint decomposition has never been run. This
script runs it on #1738's banked multi-turn holdout (all three arms, matched
targets, three layers).

Per (arm, layer, fitter) cell: the held-out residual ``E = V_hat - V`` is projected
into a top-k answer-PCA basis and the squared residual ``R[i, j]`` is decomposed

    R[i, j] = mu + a_i (context) + b_j (direction) + e_ij (interaction)

BASIS INDEPENDENCE. The design calls for a TRAIN-FOLD-ONLY basis. #1738 banks only
the holdout targets (``analysis_tensors/y_holdout/L*.npz``); the train-fold Y lives
in the ~56 GB capture store and no train eigenbasis is banked. The basis is
therefore CROSS-FITTED over the holdout: the holdout is split in half, the basis
(centering mean + eigenvectors + eigenvalues) is estimated on one half by the
parent recipe VERBATIM (``issue1738_characterize.phase_perdirection``: fp64
covariance -> eigh -> top-k descending), and the residual of the OTHER half is
decomposed in it. Both assignments are run and their shares averaged. This
preserves the property the train-fold rule protects (the basis and the per-
direction target variances are never estimated on the rows whose residuals they
score); it costs half the rows per fold. Fidelity is gated against the banked
train eigenvalues (``eval_results/issue_1738/perdirection/pdshrink_summary.json``
``eigvals_head``). Note ``E`` is invariant to the centering choice -- centering
shifts the basis directions and eigenvalues, never the residual itself.

TWO NORMALIZATIONS, both required (they answer different questions and can
disagree): (i) RAW squared residual, which high-variance directions dominate by
construction; (ii) per-direction NORMALIZED (squared residual over that
direction's target variance -- the basis-half eigenvalue, so out-of-sample),
which asks where the map does badly RELATIVE to what there was to predict.

SS SHARES ARE GEOMETRY-CONFOUNDED; VARIANCE COMPONENTS ARE THE ANSWER. Under a
pure-noise R with no structure, the expected two-way SS shares are ~1/k for
context, ~1/n for direction and ~1 for interaction -- at n=9,941 / k=256 that is
0.39% / 0.01% / 99.6% before any real effect exists. Raw SS shares are reported
for transparency, but the interpretable read is the EMS variance-component
decomposition (sigma^2_a, sigma^2_b, sigma^2_e), which nets out that geometry and
is comparable across n and k.

FLOOR SUBTRACTION (pre-registered confound). Part of any context component is
answer-sampling noise, not map failure. Where K-resample floors exist
(``eval_results/issue_1738/kresample/floors_L19.npz``) the context share is
reported both raw and floor-subtracted, within the SAME floor subsample so the
(n, k) geometry is matched. The floor is a per-context SCALAR, so netting it out
assumes it is isotropic across the top-k directions within a context -- the same
assumption the existing floor-adjusted taxonomy makes. Proportional row scaling
can only move the context component; it leaves each context's per-direction
profile, and hence the direction main effect's shape, untouched.

0 GPU. CPU-only, banked artifacts, no map is fit and no data is regenerated.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482.twoway")

SEED = 1482
K_GRID = (16, 32, 64, 128, 256, 512)
LAYERS = (14, 19, 26)
ARMS = ("context", "prefix", "bare")
STAGE = PROJECT_ROOT / "data" / "issue_1482" / "twoway_stage"
# banked train-fold eigenvalues, the basis-fidelity gate (L19, parent recipe)
PDSHRINK_SUMMARY = (
    PROJECT_ROOT / "eval_results" / "issue_1738" / "perdirection" / "pdshrink_summary.json"
)
FLOORS = PROJECT_ROOT / "eval_results" / "issue_1738" / "kresample" / "floors_L{layer}.npz"


# ── provenance ────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _metadata() -> dict:
    import scipy

    return {
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seed": SEED,
        "host": platform.node(),
    }


# ── the two-way decomposition ─────────────────────────────────────────────────


def two_way(R: np.ndarray) -> dict:
    """Two-way ANOVA without replication on the (n, k) array ``R``.

    ``R[i, j] = mu + a_i + b_j + e_ij``. Returns exact sum-of-squares shares AND
    the EMS variance components. For a balanced complete layout the SS
    decomposition is exactly orthogonal, so ``SS_a + SS_b + SS_e == SS_tot``
    (asserted). The variance components are the interpretable read: raw SS
    shares carry a pure-noise floor of ~(n-1)/(nk-1) for context and
    ~(k-1)/(nk-1) for direction, purely from the (n, k) geometry.
    """
    n, k = R.shape
    if n < 2 or k < 2:
        raise ValueError(f"two-way decomposition needs n>=2 and k>=2, got {R.shape}")
    mu = float(R.mean())
    a = R.mean(axis=1) - mu
    b = R.mean(axis=0) - mu
    resid = R - mu - a[:, None] - b[None, :]

    ss_a = float(k * (a**2).sum())
    ss_b = float(n * (b**2).sum())
    ss_e = float((resid**2).sum())
    ss_tot = float(((R - mu) ** 2).sum())
    if ss_tot <= 0:
        raise ValueError("degenerate R: zero total sum of squares")
    closure = abs(ss_a + ss_b + ss_e - ss_tot) / ss_tot
    if closure > 1e-8:
        raise AssertionError(f"two-way SS closure violated: rel dev {closure:.3e}")

    ms_a = ss_a / (n - 1)
    ms_b = ss_b / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    # EMS variance components (Searle): MS_a = sigma2_e + k*sigma2_a, etc.
    vc_a = max(0.0, (ms_a - ms_e) / k)
    vc_b = max(0.0, (ms_b - ms_e) / n)
    vc_e = ms_e
    vc_tot = vc_a + vc_b + vc_e

    return {
        "n": int(n),
        "k": int(k),
        "grand_mean": mu,
        "ss_share_context": ss_a / ss_tot,
        "ss_share_direction": ss_b / ss_tot,
        "ss_share_interaction": ss_e / ss_tot,
        # what the SS shares would read with NO structure at all, at this (n, k)
        "ss_share_context_noise_expectation": (n - 1) / (n * k - 1),
        "ss_share_direction_noise_expectation": (k - 1) / (n * k - 1),
        "vc_share_context": vc_a / vc_tot,
        "vc_share_direction": vc_b / vc_tot,
        "vc_share_interaction": vc_e / vc_tot,
        "f_context": ms_a / ms_e,
        "f_direction": ms_b / ms_e,
        # coefficient of variation of each main effect, in units of the grand mean
        "cv_context": float(np.sqrt(max(0.0, vc_a)) / mu) if mu > 0 else float("nan"),
        "cv_direction": float(np.sqrt(max(0.0, vc_b)) / mu) if mu > 0 else float("nan"),
    }


def pca_basis(Y: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parent recipe VERBATIM (issue1738_characterize.phase_perdirection /
    issue1482_error_analysis._rebuild_pca_basis): fp64 covariance of ``Y`` around
    its own mean, one eigh, top-k descending. Returns (mu, comps (d, k), eigvals (k,)).
    Eigenvector signs are arbitrary; every read here (squared residual, variance)
    is sign-invariant.
    """
    Y64 = np.asarray(Y, dtype=np.float64)
    n = Y64.shape[0]
    mu = Y64.mean(axis=0)
    A = (Y64.T @ Y64) / n - np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(A)
    top = np.flip(evecs[:, -kmax:], axis=1)
    eigvals = np.flip(evals[-kmax:], axis=0)
    return mu, np.ascontiguousarray(top), eigvals


# ── per-cell driver ───────────────────────────────────────────────────────────


def load_layer(layer: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(STAGE / f"y_parent_L{layer}.npz")
    return z["y16"], z["ci"]


def load_pred(arm: str, layer: int, fitter: str, ci_ref: np.ndarray) -> np.ndarray:
    z = np.load(STAGE / f"pred_{arm}_L{layer}_{fitter}.npz")
    if not np.array_equal(z["ci"], ci_ref):
        raise AssertionError(f"{arm} L{layer} {fitter}: ci does not match the target ci")
    return z["pred16"]


def make_folds(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two complementary halves; each is used once as basis set and once as eval set."""
    perm = np.random.default_rng(seed).permutation(n)
    h0, h1 = perm[: n // 2], perm[n // 2 :]
    return [(h0, h1), (h1, h0)]


def cell_decomposition(
    E: np.ndarray, eigvals: np.ndarray, k_grid: tuple[int, ...], row_scale: np.ndarray | None = None
) -> dict:
    """Decompose the projected residual ``E`` (n, kmax) at every k in ``k_grid``.

    ``row_scale`` (n,), when given, multiplies each context's whole row of squared
    residual -- the floor correction. Proportional row scaling leaves each
    context's per-direction profile untouched, so it can only move the context
    component.
    """
    out: dict[str, dict] = {}
    Rfull = E**2
    if row_scale is not None:
        Rfull = Rfull * row_scale[:, None]
    if not np.all(Rfull > 0):
        raise AssertionError("non-positive squared residual: log companion undefined")
    for k in k_grid:
        if k > E.shape[1]:
            continue
        R = Rfull[:, :k]
        lam = eigvals[:k]
        if not np.all(lam > 0):
            raise AssertionError(f"non-positive basis eigenvalue at k={k}")
        out[str(k)] = {
            "raw": two_way(R),
            "normalized": two_way(R / lam[None, :]),
            "log": two_way(np.log(R)),
            "eigval_min": float(lam[-1]),
            "eigval_max": float(lam[0]),
        }
    return out


def run(args) -> None:  # noqa: C901 — one linear pass over cells
    outdir = PROJECT_ROOT / "eval_results" / "issue_1482" / "twoway_residual"
    outdir.mkdir(parents=True, exist_ok=True)

    banked_train_eigvals = json.loads(PDSHRINK_SUMMARY.read_text())["eigvals_head"]
    doc: dict = {
        "metadata": _metadata(),
        "design": {
            "question": "Does the context->answer map fail at specific DIRECTIONS or "
            "specific CONTEXTS? Two-way decomposition of the held-out squared residual.",
            "model": "R[i,j] = mu + a_i (context) + b_j (direction) + e_ij (interaction)",
            "basis": "cross-fitted split-half of the #1738 holdout; parent recipe verbatim "
            "(fp64 covariance -> eigh -> top-k descending). Train-fold Y is not banked; "
            "the split keeps the basis and the per-direction target variances out of "
            "sample w.r.t. the rows they score.",
            "normalizations": ["raw squared residual", "squared residual / basis-half eigenvalue"],
            "primary_read": "EMS variance components (vc_share_*); raw SS shares are "
            "confounded by the (n,k) geometry and are reported for transparency only.",
            "k_grid": list(K_GRID),
            "corpus": "#1738 multi-turn 100k real conversations, 9,941 holdout contexts, "
            "d=3,584; all arms scored against BITWISE-identical targets.",
        },
        "banked_train_eigvals_head_L19": banked_train_eigvals,
        "cells": {},
        "basis_fidelity": {},
        "floor_correction": {},
    }

    for layer in args.layers:
        t0 = time.time()
        Y16, ci = load_layer(layer)
        n, d = Y16.shape
        folds = make_folds(n, SEED)
        kmax = max(K_GRID)

        bases = []
        for f, (basis_idx, eval_idx) in enumerate(folds):
            mu, comps, eigvals = pca_basis(Y16[basis_idx], kmax)
            bases.append((basis_idx, eval_idx, mu, comps, eigvals))
            logger.info(
                "[L%d fold%d] basis on n=%d, eigvals head %s",
                layer,
                f,
                basis_idx.size,
                np.round(eigvals[:4], 2).tolist(),
            )
        if layer == 19:
            # fidelity gate: split-half eigenvalues vs the banked TRAIN eigenvalues
            rels = [
                float(
                    np.max(
                        np.abs(b[4][:8] - banked_train_eigvals) / np.asarray(banked_train_eigvals)
                    )
                )
                for b in bases
            ]
            doc["basis_fidelity"]["L19"] = {
                "banked_train_head": banked_train_eigvals,
                "splithalf_head_fold0": [float(x) for x in bases[0][4][:8]],
                "splithalf_head_fold1": [float(x) for x in bases[1][4][:8]],
                "max_rel_dev_vs_train_fold0": rels[0],
                "max_rel_dev_vs_train_fold1": rels[1],
            }
            logger.info(
                "[L19] basis fidelity vs banked train eigvals: max rel dev %s",
                [round(r, 4) for r in rels],
            )

        fitters = ["ridge"] + (["mlp_w8192"] if layer == args.mlp_layer else [])
        for fitter in fitters:
            for arm in ARMS:
                path = STAGE / f"pred_{arm}_L{layer}_{fitter}.npz"
                if not path.exists():
                    logger.info("skip %s L%d %s (not staged)", arm, layer, fitter)
                    continue
                P16 = load_pred(arm, layer, fitter, ci)
                key = f"{arm}_L{layer}_{fitter}"
                per_fold = []
                energy = []
                for basis_idx, eval_idx, _mu, comps, eigvals in bases:
                    # residual is centering-invariant: (P-mu) - (Y-mu) = P-Y
                    Efull = P16[eval_idx].astype(np.float64) - Y16[eval_idx].astype(np.float64)
                    Epca = Efull @ comps
                    energy.append(float((Epca**2).sum() / (Efull**2).sum()))
                    per_fold.append(cell_decomposition(Epca, eigvals, K_GRID))
                    del Efull, Epca
                # average the shares across the two fold assignments
                merged: dict[str, dict] = {}
                for k in per_fold[0]:
                    merged[k] = {}
                    for norm in ("raw", "normalized", "log"):
                        merged[k][norm] = {
                            f: (
                                float(np.mean([pf[k][norm][f] for pf in per_fold]))
                                if isinstance(per_fold[0][k][norm][f], float)
                                else per_fold[0][k][norm][f]
                            )
                            for f in per_fold[0][k]["raw"]
                        }
                    merged[k]["eigval_min"] = float(
                        np.mean([pf[k]["eigval_min"] for pf in per_fold])
                    )
                    merged[k]["eigval_max"] = float(
                        np.mean([pf[k]["eigval_max"] for pf in per_fold])
                    )
                # explained-variance share of the top-k basis (basis-half spectrum)
                Ytot = [float(np.var(Y16[b[0]].astype(np.float64), axis=0).sum()) for b in bases]
                evr = {
                    str(k): float(
                        np.mean([b[4][:k].sum() / t for b, t in zip(bases, Ytot, strict=True)])
                    )
                    for k in K_GRID
                }
                doc["cells"][key] = {
                    "arm": arm,
                    "layer": layer,
                    "fitter": fitter,
                    "n_holdout": int(n),
                    "n_per_fold": int(bases[0][1].size),
                    "d": int(d),
                    "residual_energy_in_topk_basis": float(np.mean(energy)),
                    "explained_variance_share_topk": evr,
                    "by_k": merged,
                    "per_fold": per_fold,
                }
                logger.info(
                    "[%s] k=256 vc shares raw ctx/dir/int %.4f/%.4f/%.4f | norm %.4f/%.4f/%.4f",
                    key,
                    merged["256"]["raw"]["vc_share_context"],
                    merged["256"]["raw"]["vc_share_direction"],
                    merged["256"]["raw"]["vc_share_interaction"],
                    merged["256"]["normalized"]["vc_share_context"],
                    merged["256"]["normalized"]["vc_share_direction"],
                    merged["256"]["normalized"]["vc_share_interaction"],
                )
                del P16

        # ── floor correction, where K-resample floors exist ────────────────────
        fpath = Path(str(FLOORS).format(layer=layer))
        if fpath.exists():
            fz = np.load(fpath)
            fci, floor = fz["ci"], fz["floor"]
            pos = {int(c): i for i, c in enumerate(ci)}
            if not all(int(c) in pos for c in fci):
                raise AssertionError(f"L{layer}: floor ci are not a subset of the holdout ci")
            frow = np.array([pos[int(c)] for c in fci])
            fmap = dict(zip(frow.tolist(), floor.tolist(), strict=True))
            layer_floor: dict[str, dict] = {}
            for arm in ARMS:
                P16 = load_pred(arm, layer, "ridge", ci)
                per_fold_fc = []
                for basis_idx, eval_idx, _mu, comps, eigvals in bases:
                    keep = np.array([i for i, r in enumerate(eval_idx) if int(r) in fmap])
                    sub = eval_idx[keep]
                    Efull = P16[sub].astype(np.float64) - Y16[sub].astype(np.float64)
                    e2 = (Efull**2).sum(axis=1)
                    fl = np.array([fmap[int(r)] for r in sub])
                    # arm-specific: what fraction of THIS arm's per-context error is
                    # answer-sampling noise. floor is absolute squared error, so the
                    # map-attributable fraction is 1 - floor/e2, clipped to [0, 1].
                    frac = np.clip(1.0 - fl / e2, 0.0, 1.0)
                    # A context whose K-resample floor meets or exceeds its observed error
                    # carries no measurable map-attributable error under this floor
                    # estimate; its corrected row would be identically zero. Exclude it
                    # from BOTH arms of the comparison so the (n, k) geometry stays matched,
                    # and report the count.
                    elig = frac > 0
                    n_excl = int((~elig).sum())
                    Epca = (Efull[elig]) @ comps
                    per_fold_fc.append(
                        {
                            "n_sub": int(elig.sum()),
                            "n_excluded_floor_ge_error": n_excl,
                            "mean_floor_fraction_of_error": float(
                                np.mean(np.clip(fl / e2, 0.0, 1.0))
                            ),
                            "median_floor_fraction_of_error": float(
                                np.median(np.clip(fl / e2, 0.0, 1.0))
                            ),
                            "uncorrected": cell_decomposition(Epca, eigvals, (256,)),
                            "corrected": cell_decomposition(
                                Epca, eigvals, (256,), row_scale=frac[elig]
                            ),
                        }
                    )
                    del Efull, Epca
                layer_floor[arm] = {
                    "n_sub": per_fold_fc[0]["n_sub"],
                    "mean_floor_fraction_of_error": float(
                        np.mean([p["mean_floor_fraction_of_error"] for p in per_fold_fc])
                    ),
                    "median_floor_fraction_of_error": float(
                        np.mean([p["median_floor_fraction_of_error"] for p in per_fold_fc])
                    ),
                    "n_excluded_floor_ge_error": int(
                        sum(p["n_excluded_floor_ge_error"] for p in per_fold_fc)
                    ),
                }
                for norm in ("raw", "normalized"):
                    for state in ("uncorrected", "corrected"):
                        layer_floor[arm][f"{state}_{norm}"] = {
                            f: float(np.mean([p[state]["256"][norm][f] for p in per_fold_fc]))
                            for f in (
                                "vc_share_context",
                                "vc_share_direction",
                                "vc_share_interaction",
                                "ss_share_context",
                                "f_context",
                                "f_direction",
                            )
                        }
                logger.info(
                    "[floor L%d %s] n=%d vc ctx raw %.4f -> corrected %.4f | norm %.4f -> %.4f",
                    layer,
                    arm,
                    layer_floor[arm]["n_sub"],
                    layer_floor[arm]["uncorrected_raw"]["vc_share_context"],
                    layer_floor[arm]["corrected_raw"]["vc_share_context"],
                    layer_floor[arm]["uncorrected_normalized"]["vc_share_context"],
                    layer_floor[arm]["corrected_normalized"]["vc_share_context"],
                )
                del P16
            doc["floor_correction"][f"L{layer}"] = layer_floor

        del Y16
        logger.info("[L%d] done in %.1fs", layer, time.time() - t0)

    out = outdir / "twoway_residual.json"
    out.write_text(json.dumps(doc, indent=2))
    logger.info("wrote %s (%.1f KB)", out, out.stat().st_size / 1e3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=list(LAYERS))
    ap.add_argument(
        "--mlp-layer",
        type=int,
        default=19,
        help="layer at which the mlp_w8192 companion cells are also run",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
