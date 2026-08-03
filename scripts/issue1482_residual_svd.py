"""Issue #1482 inline round — structure of the held-out residual itself.

The two-way decomposition (``issue1482_twoway_residual.py``) answers WHERE the
context->answer map's squared error sits (context main effect vs direction main
effect vs their interaction). It says nothing about whether the residual VECTOR
field ``E = V - V_hat`` is itself structured. This script asks that directly, in
three reductions over the SAME staged matrices (0 GPU, no refit, no regeneration):

PHASE ``spectrum`` — thin SVD of ``E`` per (arm, layer, fitter) cell, via the fp64
Gram ``E^T E`` (right singular vectors = eigenvectors, sigma = sqrt(eigenvalue)).
Reports the singular spectrum, top-k energy shares, and three effective-rank
summaries (participation ratio, spectral-entropy rank, stable rank). Decisive
question: is the residual LOW-rank (structured, learnable) or HIGH-rank (diffuse,
near-irreducible)?

  PRE-REGISTERED CONFOUND. ``E`` contains answer-sampling noise, and noise is
  high-rank, so the raw spectrum is biased toward "diffuse". The spectrum is
  therefore reported against THREE references, all matched to the observed
  per-row energy so that row-norm heterogeneity alone cannot manufacture a
  difference:
    ``iso``    rows = observed ||E_i|| x a uniform random direction in R^d. The
               strict diffuse null. Note this is NOT flat: at n=9,941 / d=3,584
               the Marchenko-Pastur edge alone puts the top eigenvalue ~2.6x the
               mean, which is exactly why an eyeballed "decaying spectrum" is not
               evidence of structure.
    ``shaped`` rows ~ N(0, Sigma_Y) rescaled to the observed row norms, where
               Sigma_Y is the TARGET covariance. Asks whether the residual is
               more concentrated than the ambient anisotropy of the target space
               already implies. Computed in target-PCA coordinates: an orthogonal
               rotation leaves both Gram eigenvalues and row norms invariant, so
               this costs one GEMM, not two.
    ``floor``  the K-resample answer-entropy floor (#1738 ``floors_L*.npz``:
               ``trvar`` = per-context variance across 4 fresh answer draws,
               summed over dims). Two reads, both on the floor subsample so the
               (n, d) geometry is matched: (i) a noise-only reference with rows
               scaled to sqrt(trvar_i) and isotropic directions; (ii) an exact
               SHIFT correction -- isotropic noise adds (sum_i trvar_i)/d to
               EVERY Gram eigenvalue, so subtracting it deconvolves the noise
               from the observed spectrum in closed form.

  A residual clearly more concentrated than all three is a real finding. A
  spectrum indistinguishable from them is ALSO a real finding, and the more
  likely one. Neither is reported without the comparison.

  The phase also measures where the top residual directions LIVE, against two
  bases: the target-covariance PCA basis, and a realized-gain basis. Smearing
  across many ranks of both is itself the finding -- it would explain why
  fixed-basis reads never surfaced them. Gain basis: for ``context_L19_ridge``
  the staged operator ``W`` (3584x3584) is available and its output singular
  directions ARE the map's gain channels; for every other cell the
  prediction-covariance PCA is used as a realized-gain PROXY and labelled as
  such. NOTE this is NOT #1774's cross-covariance channel basis -- that basis is
  per-fold and defined on #1774's own activation store, which is not staged here;
  see ``gain_basis_caveat`` in the output.

PHASE ``consistency`` — split the contexts into disjoint halves, take each half's
top-k residual subspace independently, and measure principal angles between them.
Reported against a THREE-LEVEL ladder: a random-subspace floor (no shared
structure at all), a Gaussian-Sigma_E reference (two halves of an i.i.d. sample
from a COMMON covariance -- i.e. no structure beyond the second moment), and the
observed value. Consistent-above-the-Gaussian-reference = one shared failure mode
beyond the covariance; at the Gaussian reference = the residual is exactly as
shared as its own covariance implies; at the random floor = each context errs its
own way, which would mean the two-way "direction component" is not a stable
object. Also reports the per-context distribution of residual energy captured by
the pooled top-k.

PHASE ``alignment`` — the WORST-predicted directions, each against a matched null.
Per-direction held-out R^2 is computed in the target-PCA basis and the 20 worst
are taken among the top-256 target PCs (mirroring #1774's top-20 HIGHEST-gain
convention at the other end; the top-256 restriction is load-bearing -- beyond it
the target variance is negligible and "worst-predicted" would just select noise
directions). Those directions are aligned against: the three trait directions
``r_B``; the map's LOW-gain end; the SAE decoder columns
(``BatchTopKSAE.load(k=64, layer=19)``, ``w_dec``, 131,072 columns); and the
unembedding, via logit lens, which is the read that makes them legible. Every
alignment carries its own matched null -- for the SAE that is the max-|cos| of a
random unit vector over the SAME 131,072-column dictionary, which is far above
1/sqrt(d) and is the only honest reference for a max-over-dictionary statistic.

0 GPU. CPU-only, banked artifacts, no map is fit and no data is regenerated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
logger = logging.getLogger("i1482.residsvd")

SEED = 1482
LAYERS = (14, 19, 26)
ARMS = ("context", "prefix", "bare")
FITTERS = {14: ("ridge",), 19: ("ridge", "mlp_w8192"), 26: ("ridge",)}
HIDDEN_DIM = 3584
TOPK_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
# the target-PCA prefix inside which "worst-predicted" is a meaningful selection
R2_SELECT_K = 256
N_WORST = 20
N_VEC_KEEP = 1024  # residual eigenvectors retained (top-N, descending)

STAGE = PROJECT_ROOT / "data" / "issue_1482" / "twoway_stage"
FLOORS = PROJECT_ROOT / "eval_results" / "issue_1738" / "kresample" / "floors_L{layer}.npz"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_1482" / "twoway_residual"
RB_DIR = PROJECT_ROOT / "data" / "issue_778" / "rb"
TRAITS = ("evil", "sycophancy", "hallucination")
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ── staged loaders (conventions mirror issue1482_twoway_residual.py) ──────────


def load_layer(layer: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(STAGE / f"y_parent_L{layer}.npz")
    return z["y16"], z["ci"]


def load_pred(arm: str, layer: int, fitter: str, ci_ref: np.ndarray) -> np.ndarray:
    z = np.load(STAGE / f"pred_{arm}_L{layer}_{fitter}.npz")
    if not np.array_equal(z["ci"], ci_ref):
        raise AssertionError(f"{arm} L{layer} {fitter}: ci does not match the target ci")
    return z["pred16"]


def cells() -> list[tuple[str, int, str]]:
    return [(a, ly, f) for ly in LAYERS for f in FITTERS[ly] for a in ARMS]


def cell_name(arm: str, layer: int, fitter: str) -> str:
    return f"{arm}_L{layer}_{fitter}"


# ── linear algebra ────────────────────────────────────────────────────────────


def gram_spectrum(
    M: np.ndarray, *, want_vectors: bool = False, n_vec: int = N_VEC_KEEP
) -> tuple[np.ndarray, np.ndarray | None]:
    """Eigenvalues (descending, clipped at 0) of ``M^T M``, optionally the top
    ``n_vec`` eigenvectors (= right singular vectors of ``M``, columns).

    Gram + eigh rather than a thin SVD: the right singular vectors are exactly the
    Gram eigenvectors and ``sigma = sqrt(lambda)``, at a fraction of the cost for
    a tall matrix. Small eigenvalues lose precision under the squaring, which is
    immaterial for every read here (energy shares and effective ranks are
    dominated by the head of the spectrum).
    """
    n, d = M.shape
    if n < 2 or d < 2:
        raise ValueError(f"gram_spectrum needs a 2-D matrix with n,d >= 2, got {M.shape}")
    G = M.T @ M
    if want_vectors:
        evals, evecs = np.linalg.eigh(G)
        keep = min(n_vec, d)
        vecs = np.ascontiguousarray(np.flip(evecs[:, -keep:], axis=1))
    else:
        evals = np.linalg.eigvalsh(G)
        vecs = None
    lam = np.flip(evals)
    return np.clip(lam, 0.0, None), vecs


def spectrum_stats(lam: np.ndarray) -> dict:
    """Energy-share and effective-rank summaries of a descending eigenvalue array.

    ``lam`` are Gram eigenvalues (= squared singular values = energy per
    direction). Participation ratio and spectral-entropy rank are the two standard
    effective-rank reads and disagree in interpretable ways (PR is second-moment,
    entropy is information-theoretic); stable rank is the operator-norm read.
    """
    tot = float(lam.sum())
    if tot <= 0:
        raise ValueError("degenerate spectrum: zero total energy")
    p = lam / tot
    nz = p[p > 0]
    return {
        "total_energy": tot,
        "topk_energy_share": {
            str(k): float(lam[:k].sum() / tot) for k in TOPK_GRID if k <= lam.size
        },
        "participation_ratio": float(1.0 / np.square(p).sum()),
        "entropy_effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
        "stable_rank": float(tot / lam[0]),
        "lambda_max": float(lam[0]),
        "lambda_mean": float(tot / lam.size),
        "n_dims": int(lam.size),
        "spectrum_head": [float(x) for x in lam[:32]],
    }


def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """(n, d) fp64 rows uniform on the unit sphere."""
    G = rng.standard_normal((n, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    return G


def iso_null_spectrum(rng: np.random.Generator, row_norms: np.ndarray, d: int) -> np.ndarray:
    """Gram spectrum of a matrix with the GIVEN row norms and isotropic directions."""
    N = _unit_rows(rng, row_norms.size, d)
    N *= row_norms[:, None]
    lam, _ = gram_spectrum(N)
    return lam


def shaped_null_spectrum(
    rng: np.random.Generator, row_norms: np.ndarray, y_eigs: np.ndarray
) -> np.ndarray:
    """Gram spectrum of rows ~ N(0, Sigma_Y) rescaled to ``row_norms``.

    Computed in target-PCA coordinates. The change of basis is orthogonal, so both
    the Gram eigenvalues and the per-row norms are invariant -- working in PCA
    coordinates is exact here, not an approximation, and saves a full GEMM.
    """
    n, d = row_norms.size, y_eigs.size
    Z = rng.standard_normal((n, d))
    Z *= np.sqrt(np.clip(y_eigs, 0.0, None))[None, :]
    Z *= (row_norms / np.linalg.norm(Z, axis=1))[:, None]
    lam, _ = gram_spectrum(Z)
    return lam


def _null_band(draws: list[np.ndarray]) -> dict:
    """Per-statistic min/mean/max across null draws (the reference band)."""
    stats = [spectrum_stats(d) for d in draws]
    out: dict = {"n_draws": len(draws)}
    for key in ("participation_ratio", "entropy_effective_rank", "stable_rank"):
        vals = [s[key] for s in stats]
        out[key] = {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))}
    shares: dict = {}
    for k in stats[0]["topk_energy_share"]:
        vals = [s["topk_energy_share"][k] for s in stats]
        shares[k] = {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))}
    out["topk_energy_share"] = shares
    return out


def basis_smearing(vecs: np.ndarray, basis: np.ndarray, n_dirs: int) -> dict:
    """How the top ``n_dirs`` residual directions distribute over ``basis``.

    ``vecs`` (d, m) residual directions in columns; ``basis`` (d, b) orthonormal
    columns. For each residual direction the squared overlaps onto the basis are a
    probability vector; its participation ratio is the effective NUMBER of basis
    ranks the direction spans. A direction that is one clean basis element reads
    ~1; one smeared over the whole basis reads ~b. Reported next to the captured
    mass so a low PR on a low-capture direction is not mistaken for concentration.
    """
    m = min(n_dirs, vecs.shape[1])
    C = basis.T @ vecs[:, :m]  # (b, m)
    w = np.square(C)
    captured = w.sum(axis=0)  # mass of each residual direction inside the basis
    with np.errstate(invalid="ignore", divide="ignore"):
        pw = w / np.where(captured > 0, captured, np.nan)[None, :]
        pr = 1.0 / np.square(pw).sum(axis=0)
    top1 = np.abs(C).max(axis=0)
    argtop = np.argmax(np.abs(C), axis=0)
    return {
        "n_dirs": int(m),
        "basis_size": int(basis.shape[1]),
        "captured_mass": [float(x) for x in captured],
        "participation_ratio_over_basis": [float(x) for x in pr],
        "max_abs_overlap": [float(x) for x in top1],
        "argmax_basis_index": [int(x) for x in argtop],
        "captured_mass_mean": float(captured.mean()),
        "participation_ratio_mean": float(np.nanmean(pr)),
    }


# ── phase: spectrum ───────────────────────────────────────────────────────────


def load_floor(layer: int) -> dict[int, float] | None:
    path = Path(str(FLOORS).format(layer=layer))
    if not path.exists():
        return None
    z = np.load(path)
    return dict(zip(z["ci"].tolist(), z["floor"].tolist(), strict=True))


def gain_basis_for(
    arm: str, layer: int, fitter: str, pred: np.ndarray, n_keep: int
) -> tuple[np.ndarray, str]:
    """Output-side gain directions of the map, and a label naming what they are.

    ``context_L19_ridge`` has the fitted operator ``W`` staged, so its true output
    singular directions are used. Every other cell falls back to the
    prediction-covariance PCA -- the directions the map actually drives on this
    holdout, ordered by realized output variance. That is a PROXY for the
    operator's gain basis, not the operator's own SVD, and is labelled so.
    """
    op = STAGE / f"{arm}_{fitter}_L{layer}.pt"
    if op.exists():
        import torch

        W = torch.load(op, map_location="cpu", weights_only=False)["W"]
        W = np.asarray(W, dtype=np.float64)
        # rows of the staged W map standardized-x -> centered-y, so the OUTPUT
        # directions are the right singular vectors of W.
        lam, vecs = gram_spectrum(W, want_vectors=True, n_vec=n_keep)
        return vecs, f"operator W right-singular directions (staged {op.name})"
    P = np.asarray(pred, dtype=np.float64)
    P -= P.mean(axis=0, keepdims=True)
    _lam, vecs = gram_spectrum(P, want_vectors=True, n_vec=n_keep)
    return vecs, "prediction-covariance PCA (realized-gain PROXY; operator W not staged)"


def phase_spectrum(args) -> None:
    out_path = OUT_DIR / "residual_spectrum.json"
    doc: dict = (
        json.loads(out_path.read_text())
        if out_path.exists()
        else {
            "metadata": _metadata(),
            "design": {
                "question": (
                    "Is the held-out residual E = V - V_hat LOW-rank (structured, "
                    "learnable) or HIGH-rank (diffuse, near-irreducible)?"
                ),
                "method": (
                    "fp64 Gram E^T E -> eigh; sigma = sqrt(lambda). Energy shares + "
                    "three effective-rank reads."
                ),
                "noise_references": {
                    "iso": "rows = observed ||E_i|| x uniform random direction (strict diffuse null)",
                    "shaped": (
                        "rows ~ N(0, Sigma_Y) rescaled to observed row norms "
                        "(ambient target anisotropy); exact in target-PCA coordinates"
                    ),
                    "floor": (
                        "#1738 K-resample trvar. (i) noise-only reference at "
                        "sqrt(trvar_i) row norms on the floor subsample; (ii) exact "
                        "shift correction lambda -> lambda - sum_i(trvar_i)/d"
                    ),
                },
                "preregistered_confound": (
                    "E contains answer-sampling noise and noise is high-rank, biasing "
                    "toward 'diffuse'. Every spectrum read is reported against the "
                    "references above; neither verdict is reported without them."
                ),
                "topk_grid": list(TOPK_GRID),
                "corpus": (
                    "#1738 multi-turn 100k real conversations, 9,941 holdout contexts, "
                    "d=3,584; all arms scored against BITWISE-identical targets."
                ),
            },
            "cells": {},
        }
    )
    todo = [c for c in cells() if cell_name(*c) not in doc["cells"]]
    if args.only_cells:
        todo = [c for c in todo if cell_name(*c) in args.only_cells]
    logger.info("[spectrum] %d cells pending (%d done)", len(todo), len(doc["cells"]))

    by_layer_targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, (arm, layer, fitter) in enumerate(todo, start=1):
        name = cell_name(arm, layer, fitter)
        t0 = time.time()
        if layer not in by_layer_targets:
            y16, ci = load_layer(layer)
            by_layer_targets.clear()  # one layer resident at a time
            by_layer_targets[layer] = (y16, ci)
        y16, ci = by_layer_targets[layer]
        pred16 = load_pred(arm, layer, fitter, ci)

        Y = np.asarray(y16, dtype=np.float64)
        E = Y - np.asarray(pred16, dtype=np.float64)
        n, d = E.shape
        row_norms = np.linalg.norm(E, axis=1)

        lam, vecs = gram_spectrum(E, want_vectors=True)
        rec: dict = {
            "arm": arm,
            "layer": layer,
            "fitter": fitter,
            "n_holdout": int(n),
            "d": int(d),
            "observed": spectrum_stats(lam),
            "row_norm_summary": {
                "mean": float(row_norms.mean()),
                "median": float(np.median(row_norms)),
                "p10": float(np.percentile(row_norms, 10)),
                "p90": float(np.percentile(row_norms, 90)),
            },
        }

        # target covariance (shared across arms at a layer) -> shaped null + basis
        Yc = Y - Y.mean(axis=0, keepdims=True)
        y_lam, y_vecs = gram_spectrum(Yc, want_vectors=True)
        y_eigs = y_lam / n

        rng = np.random.default_rng(SEED + 1000 * layer + hash(name) % 997)
        iso_draws = [iso_null_spectrum(rng, row_norms, d) for _ in range(args.null_draws)]
        shaped_draws = [
            shaped_null_spectrum(rng, row_norms, y_eigs) for _ in range(args.null_draws)
        ]
        rec["null_iso"] = _null_band(iso_draws)
        rec["null_shaped"] = _null_band(shaped_draws)

        # ── floor: noise-only reference + exact isotropic shift correction ─────
        floor_map = load_floor(layer)
        if floor_map is not None:
            sub = np.array([i for i, c in enumerate(ci.tolist()) if c in floor_map], dtype=np.int64)
            trvar = np.array([floor_map[int(ci[i])] for i in sub], dtype=np.float64)
            Esub = E[sub]
            sub_norms = np.linalg.norm(Esub, axis=1)
            lam_sub, _ = gram_spectrum(Esub)
            shift = float(trvar.sum() / d)
            lam_corr = np.clip(lam_sub - shift, 0.0, None)
            rng_f = np.random.default_rng(SEED + 7 + layer)
            noise_draws = [
                iso_null_spectrum(rng_f, np.sqrt(trvar), d) for _ in range(args.null_draws)
            ]
            iso_sub = [iso_null_spectrum(rng_f, sub_norms, d) for _ in range(args.null_draws)]
            rec["floor"] = {
                "n_sub": int(sub.size),
                "noise_energy_fraction_mean": float((trvar / np.square(sub_norms)).mean()),
                "noise_energy_fraction_pooled": float(trvar.sum() / np.square(sub_norms).sum()),
                "observed_on_sub": spectrum_stats(lam_sub),
                "shift_corrected_on_sub": spectrum_stats(lam_corr),
                "isotropic_shift_per_eigenvalue": shift,
                "null_noise_only": _null_band(noise_draws),
                "null_iso_on_sub": _null_band(iso_sub),
            }

        # ── where do the top residual directions live? ─────────────────────────
        gain_vecs, gain_label = gain_basis_for(arm, layer, fitter, pred16, N_VEC_KEEP)
        rec["basis_overlap"] = {
            "target_pca": basis_smearing(vecs, y_vecs, args.n_overlap_dirs),
            "gain": basis_smearing(vecs, gain_vecs, args.n_overlap_dirs),
            "gain_basis_label": gain_label,
            "gain_basis_caveat": (
                "NOT #1774's cross-covariance channel basis: that basis is per-fold "
                "and defined on #1774's own activation store, which is not staged in "
                "this round. See the module docstring."
            ),
        }
        rng_o = np.random.default_rng(SEED + 31 + layer)
        rand_dirs = _unit_rows(rng_o, args.n_overlap_dirs, d).T
        rec["basis_overlap"]["null_random_dirs"] = {
            "target_pca": basis_smearing(rand_dirs, y_vecs, args.n_overlap_dirs),
            "gain": basis_smearing(rand_dirs, gain_vecs, args.n_overlap_dirs),
        }

        doc["cells"][name] = rec
        _atomic_write_json(out_path, doc)
        logger.info(
            "[spectrum] cell %d/%d %s elapsed=%.1fs PR=%.1f top16=%.3f (iso top16=%.3f)",
            idx,
            len(todo),
            name,
            time.time() - t0,
            rec["observed"]["participation_ratio"],
            rec["observed"]["topk_energy_share"]["16"],
            rec["null_iso"]["topk_energy_share"]["16"]["mean"],
        )
    logger.info("[spectrum] done -> %s", out_path)


# ── phase: cross-context consistency ──────────────────────────────────────────


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spans of A and B.

    Both must have orthonormal columns; the singular values of ``A^T B`` are the
    cosines of the principal angles.
    """
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def _subspace_summary(A: np.ndarray, B: np.ndarray) -> dict:
    ang = principal_angles(A, B)
    cos = np.cos(ang)
    return {
        "mean_cos_principal_angle": float(cos.mean()),
        "median_cos_principal_angle": float(np.median(cos)),
        "max_cos_principal_angle": float(cos.max()),
        # chordal overlap: mean squared cosine == (1/k)||A^T B||_F^2, the fraction
        # of one subspace captured by the other
        "subspace_overlap": float(np.square(cos).mean()),
        "min_angle_deg": float(np.degrees(ang.min())),
        "median_angle_deg": float(np.degrees(np.median(ang))),
    }


def phase_consistency(args) -> None:
    out_path = OUT_DIR / "residual_consistency.json"
    doc: dict = (
        json.loads(out_path.read_text())
        if out_path.exists()
        else {
            "metadata": _metadata(),
            "design": {
                "question": (
                    "Do disjoint halves of the contexts fail in the SAME residual "
                    "subspace (one shared failure mode) or in different ones (each "
                    "context errs its own way)?"
                ),
                "method": (
                    "Disjoint random context halves; independent Gram eigh per half; "
                    "principal angles between the top-k subspaces."
                ),
                "ladder": {
                    "random_floor": "two independent uniformly random k-dim subspaces of R^d",
                    "gaussian_sigma_e": (
                        "two halves of an i.i.d. Gaussian sample with the OBSERVED "
                        "residual covariance -- no structure beyond the second moment"
                    ),
                    "observed": "the measured half-to-half agreement",
                },
                "reading": (
                    "at the random floor = no shared object; at the Gaussian "
                    "reference = exactly as shared as the covariance implies; above "
                    "it = shared structure beyond the second moment."
                ),
                "k_grid": list(args.consistency_k),
            },
            "cells": {},
        }
    )
    todo = [c for c in cells() if cell_name(*c) not in doc["cells"]]
    if args.only_cells:
        todo = [c for c in todo if cell_name(*c) in args.only_cells]
    logger.info("[consistency] %d cells pending", len(todo))

    for idx, (arm, layer, fitter) in enumerate(todo, start=1):
        name = cell_name(arm, layer, fitter)
        t0 = time.time()
        y16, ci = load_layer(layer)
        pred16 = load_pred(arm, layer, fitter, ci)
        E = np.asarray(y16, dtype=np.float64) - np.asarray(pred16, dtype=np.float64)
        n, d = E.shape
        kmax = int(max(args.consistency_k))

        rng = np.random.default_rng(SEED + 500 + layer + hash(name) % 313)
        perm = rng.permutation(n)
        hA, hB = perm[: n // 2], perm[n // 2 :]
        _lamA, vA = gram_spectrum(E[hA], want_vectors=True, n_vec=kmax)
        _lamB, vB = gram_spectrum(E[hB], want_vectors=True, n_vec=kmax)

        # Gaussian-Sigma_E reference: i.i.d. rows with the OBSERVED residual
        # covariance, computed in the residual-eigenbasis (orthogonal => exact).
        lam_full, v_full = gram_spectrum(E, want_vectors=True, n_vec=d)
        e_sd = np.sqrt(lam_full / n)

        per_k: dict = {}
        for k in args.consistency_k:
            obs = _subspace_summary(vA[:, :k], vB[:, :k])
            gauss, rand = [], []
            for _ in range(args.null_draws):
                Z = rng.standard_normal((n, d)) * e_sd[None, :]
                _l1, g1 = gram_spectrum(Z[hA], want_vectors=True, n_vec=k)
                _l2, g2 = gram_spectrum(Z[hB], want_vectors=True, n_vec=k)
                gauss.append(_subspace_summary(g1[:, :k], g2[:, :k]))
                r1 = np.linalg.qr(rng.standard_normal((d, k)))[0]
                r2 = np.linalg.qr(rng.standard_normal((d, k)))[0]
                rand.append(_subspace_summary(r1, r2))

            def _band(rows: list[dict]) -> dict:
                return {
                    key: {
                        "mean": float(np.mean([r[key] for r in rows])),
                        "min": float(min(r[key] for r in rows)),
                        "max": float(max(r[key] for r in rows)),
                    }
                    for key in rows[0]
                }

            per_k[str(k)] = {
                "observed": obs,
                "gaussian_sigma_e": _band(gauss),
                "random_floor": _band(rand),
            }
            logger.info(
                "  [consistency] %s k=%d overlap obs=%.3f gauss=%.3f rand=%.3f",
                name,
                k,
                obs["subspace_overlap"],
                per_k[str(k)]["gaussian_sigma_e"]["subspace_overlap"]["mean"],
                per_k[str(k)]["random_floor"]["subspace_overlap"]["mean"],
            )

        # per-context energy captured by the POOLED top-k
        pooled: dict = {}
        row_e = np.square(E).sum(axis=1)
        for k in args.consistency_k:
            proj = E @ v_full[:, :k]
            frac = np.square(proj).sum(axis=1) / row_e
            pooled[str(k)] = {
                "mean": float(frac.mean()),
                "median": float(np.median(frac)),
                "p10": float(np.percentile(frac, 10)),
                "p90": float(np.percentile(frac, 90)),
                "iqr": float(np.percentile(frac, 75) - np.percentile(frac, 25)),
                "uniform_expectation": float(k / d),
            }

        doc["cells"][name] = {
            "arm": arm,
            "layer": layer,
            "fitter": fitter,
            "n_half": [int(hA.size), int(hB.size)],
            "by_k": per_k,
            "per_context_energy_in_pooled_topk": pooled,
        }
        _atomic_write_json(out_path, doc)
        logger.info(
            "[consistency] cell %d/%d %s elapsed=%.1fs", idx, len(todo), name, time.time() - t0
        )
    logger.info("[consistency] done -> %s", out_path)


# ── phase: alignment of the worst-predicted directions ────────────────────────


def load_rb(layer: int) -> dict[str, np.ndarray]:
    """r_B trait directions at block-output ``layer`` (bank row layer-1)."""
    import torch

    out = {}
    for trait in TRAITS:
        p = RB_DIR / f"{trait}.pt"
        if not p.exists():
            raise FileNotFoundError(f"r_B bank missing: {p}")
        t = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = t.get("r_b", next(iter(t.values())))
        arr = np.asarray(t, dtype=np.float64)
        if arr.shape != (28, HIDDEN_DIM):
            raise AssertionError(f"{trait}: expected (28, {HIDDEN_DIM}), got {arr.shape}")
        v = arr[layer - 1]
        out[trait] = v / np.linalg.norm(v)
    return out


def load_unembedding() -> tuple[np.ndarray, object]:
    """``lm_head.weight`` (V, d) fp32 + tokenizer, from the local HF cache."""
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(QWEN_MODEL)
    if getattr(cfg, "tie_word_embeddings", False):
        raise AssertionError("tied embeddings: lm_head slice would be the input embedding")
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from explore_persona_space.orchestrate import hub

    idx_path = hub.retry_transient(
        lambda: hf_hub_download(QWEN_MODEL, "model.safetensors.index.json"),
        what=f"unembedding index fetch ({QWEN_MODEL})",
    )
    idx = json.loads(Path(idx_path).read_text())
    shard = idx["weight_map"]["lm_head.weight"]
    path = hub.retry_transient(
        lambda: hf_hub_download(QWEN_MODEL, shard),
        what=f"unembedding shard fetch ({QWEN_MODEL}:{shard})",
    )
    # framework="pt", not "np": the checkpoint is bf16 and numpy has no bfloat16,
    # so the numpy reader raises TypeError on this tensor (verified on this shard).
    with safe_open(path, framework="pt") as f:
        W_U = f.get_tensor("lm_head.weight").to(torch.float32).numpy()
    if W_U.shape[1] != HIDDEN_DIM:
        raise AssertionError(f"lm_head shape {W_U.shape} != (V, {HIDDEN_DIM})")
    return W_U, AutoTokenizer.from_pretrained(QWEN_MODEL)


def _max_cos_null(rng: np.random.Generator, D_unit: np.ndarray, n_draws: int, d: int) -> dict:
    """Null band for max_j |cos(v, D_j)| when v is a RANDOM unit vector.

    The matched null for a max-over-dictionary statistic: with 131,072 columns the
    max |cos| of a random direction sits far above 1/sqrt(d), so 1/sqrt(d) is not
    a usable reference here.
    """
    # ONE GEMM over the whole draw block, not n_draws separate mat-vecs: the
    # dictionary is ~1.9 GB, so a per-draw loop is memory-bandwidth-bound and
    # streams it n_draws times over.
    R = rng.standard_normal((d, n_draws)).astype(np.float32)
    R /= np.linalg.norm(R, axis=0, keepdims=True)
    vals = np.abs(D_unit @ R).max(axis=0).tolist()
    return {
        "n_draws": n_draws,
        "mean": float(np.mean(vals)),
        "max": float(max(vals)),
        "p95": float(np.percentile(vals, 95)),
    }


def phase_alignment(args) -> None:
    out_path = OUT_DIR / "residual_alignment.json"
    doc: dict = (
        json.loads(out_path.read_text())
        if out_path.exists()
        else {
            "metadata": _metadata(),
            "design": {
                "question": (
                    "What ARE the worst-predicted directions? Do they align with any "
                    "known basis, or are they legible only through the unembedding?"
                ),
                "selection": (
                    f"per-direction held-out R^2 in the target-PCA basis; the {N_WORST} "
                    f"WORST among the top-{R2_SELECT_K} target PCs. The top-{R2_SELECT_K} "
                    "restriction is load-bearing: beyond it target variance is "
                    "negligible and 'worst-predicted' would select pure noise."
                ),
                "mirrors": (
                    "#1774 ran this for the top-20 HIGHEST-gain singular directions "
                    "(near-zero trait alignment) and #779 for r_B -- both the "
                    "well-predicted end. Conventions mirrored for comparability."
                ),
                "nulls": (
                    "every alignment carries a matched null; for the SAE dictionary "
                    "the null is max|cos| of a RANDOM unit vector over the same "
                    "131,072 columns, which is far above 1/sqrt(d)."
                ),
            },
            "cells": {},
        }
    )
    targets = args.only_cells or [cell_name("context", 19, "ridge")]
    todo = [c for c in cells() if cell_name(*c) in targets and cell_name(*c) not in doc["cells"]]
    logger.info("[alignment] %d cells pending: %s", len(todo), [cell_name(*c) for c in todo])
    if not todo:
        logger.info("[alignment] nothing to do -> %s", out_path)
        return

    sae = None
    W_U = tok = None
    for idx, (arm, layer, fitter) in enumerate(todo, start=1):
        name = cell_name(arm, layer, fitter)
        t0 = time.time()
        y16, ci = load_layer(layer)
        pred16 = load_pred(arm, layer, fitter, ci)
        Y = np.asarray(y16, dtype=np.float64)
        P = np.asarray(pred16, dtype=np.float64)
        E = Y - P
        n, d = E.shape

        Yc = Y - Y.mean(axis=0, keepdims=True)
        y_lam, y_vecs = gram_spectrum(Yc, want_vectors=True, n_vec=R2_SELECT_K)
        # per-direction held-out R^2 in the target-PCA basis
        ss_tot = np.square(Yc @ y_vecs).sum(axis=0)
        ss_res = np.square(E @ y_vecs).sum(axis=0)
        r2 = 1.0 - ss_res / ss_tot
        worst = np.argsort(r2)[:N_WORST]
        best = np.argsort(r2)[-N_WORST:][::-1]
        V_worst = y_vecs[:, worst]  # (d, N_WORST) orthonormal target-PCA directions

        rec: dict = {
            "arm": arm,
            "layer": layer,
            "fitter": fitter,
            "per_direction_r2_top256": [float(x) for x in r2],
            "worst_indices": [int(x) for x in worst],
            "worst_r2": [float(r2[i]) for i in worst],
            "best_indices": [int(x) for x in best],
            "best_r2": [float(r2[i]) for i in best],
            "target_variance_share_of_worst": [float(y_lam[i] / y_lam.sum()) for i in worst],
        }

        rng = np.random.default_rng(SEED + 900 + layer)

        # ── r_B trait directions ───────────────────────────────────────────────
        rb = load_rb(layer)
        rb_mat = np.stack([rb[t] for t in TRAITS], axis=1)  # (d, 3)
        cos_rb = V_worst.T @ rb_mat  # (N_WORST, 3)
        rand_dirs = _unit_rows(rng, 2000, d)
        null_rb = np.abs(rand_dirs @ rb_mat)
        rec["r_b_alignment"] = {
            "traits": list(TRAITS),
            "abs_cos_worst": [[float(abs(x)) for x in row] for row in cos_rb],
            "max_abs_cos_worst": float(np.abs(cos_rb).max()),
            "mean_abs_cos_worst": float(np.abs(cos_rb).mean()),
            "null_random_unit": {
                "mean": float(null_rb.mean()),
                "p95": float(np.percentile(null_rb, 95)),
                "max": float(null_rb.max()),
                "n_draws": int(rand_dirs.shape[0]),
            },
            "analytic_null_1_over_sqrt_d": float(1.0 / np.sqrt(d)),
        }

        # ── the map's LOW-gain end ─────────────────────────────────────────────
        gain_vecs, gain_label = gain_basis_for(arm, layer, fitter, pred16, HIDDEN_DIM)
        n_low = 256
        low = gain_vecs[:, -n_low:]
        high = gain_vecs[:, :n_low]
        rec["gain_end_alignment"] = {
            "gain_basis_label": gain_label,
            "n_end": n_low,
            "mass_in_low_gain_end": [float(x) for x in np.square(low.T @ V_worst).sum(axis=0)],
            "mass_in_high_gain_end": [float(x) for x in np.square(high.T @ V_worst).sum(axis=0)],
            "uniform_expectation": float(n_low / d),
            "smearing_over_full_gain_basis": basis_smearing(V_worst, gain_vecs, N_WORST),
        }

        # ── SAE decoder columns ────────────────────────────────────────────────
        if not args.skip_sae:
            if sae is None:
                sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
                from issue1482_sae import BatchTopKSAE

                sae = BatchTopKSAE.load(k=64, layer=19, device="cpu")
                logger.info("[alignment] SAE loaded: w_dec %s", tuple(sae.w_dec.shape))
            D = np.asarray(sae.w_dec, dtype=np.float32)  # (d, n_feat)
            D_unit = (D / np.linalg.norm(D, axis=0, keepdims=True)).T  # (n_feat, d)
            cos_sae = D_unit @ V_worst.astype(np.float32)  # (n_feat, N_WORST)
            best_feat = np.argmax(np.abs(cos_sae), axis=0)
            rec["sae_alignment"] = {
                "sae_layer": 19,
                "sae_k": 64,
                "n_features": int(D_unit.shape[0]),
                "max_abs_cos_per_worst": [
                    float(abs(cos_sae[best_feat[j], j])) for j in range(N_WORST)
                ],
                "argmax_feature_per_worst": [int(x) for x in best_feat],
                "null_random_unit_max_over_dictionary": _max_cos_null(rng, D_unit, 200, d),
                "layer_note": (
                    "SAE is trained at layer 19; alignment at other layers is "
                    "cross-layer and is reported only where layer == 19."
                    if layer != 19
                    else "SAE layer matches the cell layer."
                ),
            }

        # ── logit lens ─────────────────────────────────────────────────────────
        if not args.skip_logitlens:
            if W_U is None:
                W_U, tok = load_unembedding()
                logger.info("[alignment] unembedding loaded: %s", W_U.shape)
            logits = W_U @ V_worst.astype(np.float32)  # (V, N_WORST)
            ll = []
            for j in range(N_WORST):
                col = logits[:, j]
                top = np.argsort(col)[-12:][::-1]
                bot = np.argsort(col)[:12]
                ll.append(
                    {
                        "pc_index": int(worst[j]),
                        "r2": float(r2[worst[j]]),
                        "top_tokens": [tok.decode([int(t)]) for t in top],
                        "top_logits": [float(col[t]) for t in top],
                        "bottom_tokens": [tok.decode([int(t)]) for t in bot],
                    }
                )
            rec["logit_lens_worst"] = ll

        doc["cells"][name] = rec
        _atomic_write_json(out_path, doc)
        logger.info(
            "[alignment] cell %d/%d %s elapsed=%.1fs", idx, len(todo), name, time.time() - t0
        )
    logger.info("[alignment] done -> %s", out_path)


# ── entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=("spectrum", "consistency", "alignment"))
    ap.add_argument("--null-draws", type=int, default=3)
    ap.add_argument("--n-overlap-dirs", type=int, default=32)
    ap.add_argument("--consistency-k", type=int, nargs="+", default=[8, 16, 64, 256])
    ap.add_argument("--only-cells", nargs="*", default=None)
    ap.add_argument("--skip-sae", action="store_true")
    ap.add_argument("--skip-logitlens", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        logger.info("import-check ok")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    {"spectrum": phase_spectrum, "consistency": phase_consistency, "alignment": phase_alignment}[
        args.phase
    ](args)


if __name__ == "__main__":
    main()
