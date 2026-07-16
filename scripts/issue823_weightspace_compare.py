"""#823 free-analysis follow-up: weight-space comparison of own- vs plain-fitted ridge maps.

At the 3 plan-pinned read-out layers (evil L14, sycophancy L26, hallucination
L17) fit the FULL-valid-set ridge map W (standardized-input -> centered-target
weight matrix, H x H) for the OWN answer arm (cx_last -> v_A_prime) and the
PLAIN-external arm (cx_last -> v_B2), then compare the two operators directly:

  (i)   flattened weight cosine  cos(vec(W_own), vec(W_plain));
  (ii)  principal-angle / subspace alignment between the top-k singular
        subspaces (k-sweep), for BOTH the input (right-singular) and output
        (left-singular) sides — mean cos of principal angles;
  (iii) baselines: (a) within-arm disjoint-half refit noise ceiling (fit W on
        two disjoint halves of the SAME own arm -> the max similarity refit
        noise permits), and (b) a random-map floor (two independent Gaussian
        H x H matrices -> the chance similarity).

Solver: dual-form GCV ridge (standardize-X population-std + 1e-9, center-Y,
GCV lambda over logspace(-2,4,13)) — the same recipe as
fit_h.ridge_fit_predict / ridge_fit_predict_fast, extended to return the
standardized-input weight matrix W = Xn^T (G + lam I)^-1 Yc. Because X (cx_last
at layer L) is IDENTICAL across arms, xmu/xsd and the Gram eigendecomposition
are shared between W_own and W_plain, so the two W live in the SAME
standardized-input basis and are directly comparable.

This mirrors the #825 crossmodel-map-transfer similarity conventions
(flattened cosine + principal angles + refit-noise ceiling) applied
WITHIN-model across answer arms.

Usage:
  uv run python scripts/issue823_weightspace_compare.py
"""

from __future__ import annotations

# ruff: noqa: E402 — load_dotenv() must run before torch import (shared-VM thread caps)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import datetime
import json
import logging
import pathlib
import subprocess
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_weightspace_compare")

DL = "/mnt/eps-data/thomasjiralerspong/tmp_issue823_crossarm"
PREFIX = "issue823_own_vs_external"
EXPECTED_N = 5000
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
READ_OUT_LAYERS = {"evil": 14, "sycophancy": 26, "hallucination": 17}
LAMBDAS = np.logspace(-2, 4, 13)
K_SWEEP = [5, 10, 20, 50, 100]


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _load_arm(name: str, n: int) -> torch.Tensor:
    p = pathlib.Path(DL) / PREFIX / "analysis_tensors" / f"v_{name}.pt"
    t = torch.load(str(p), map_location="cpu", mmap=True)
    assert t.shape == (EXPECTED_N, EXPECTED_LAYERS, EXPECTED_HIDDEN), (name, tuple(t.shape))
    return t[:n]


def _load_bundle_cx_last(n: int) -> torch.Tensor:
    p = pathlib.Path(DL) / "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
    b = torch.load(str(p), map_location="cpu", mmap=True)
    return b["cx_last"][:n]


def _valid_idx(n: int) -> np.ndarray:
    p = pathlib.Path(DL) / PREFIX / "raw_completions/phase1/common_valid_idx.json"
    all_valid = np.array(sorted(json.loads(p.read_text())["common_valid_idx"]), dtype=int)
    return all_valid[all_valid < n]


class DualRidgeShared:
    """Dual-form GCV ridge over a FIXED X; the Gram eigendecomposition is fit once
    and reused for any target Y (own/plain share X, so they share this).

    W(Y) returns the (H, D) standardized-input weight matrix at the GCV-selected
    lambda for that Y — identical recipe to fit_h.ridge_fit_predict_fast."""

    def __init__(self, X: np.ndarray):
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        self.xmu = Xt.mean(0)
        self.xsd = Xt.std(0, correction=0) + 1e-9  # population std, matches numpy .std
        self.Xn = (Xt - self.xmu) / self.xsd
        self.n = Xt.shape[0]
        G = self.Xn @ self.Xn.T
        w, V = torch.linalg.eigh(G)
        self.w = torch.clamp(w, min=0.0)
        self.V = V

    def _gcv_lambda(self, Yc: torch.Tensor, VtY: torch.Tensor) -> float:
        e = (VtY**2).sum(1)
        tot = float((Yc**2).sum())
        best_lam, best_gcv = float(LAMBDAS[0]), float("inf")
        for lam in LAMBDAS:
            filt = self.w / (self.w + float(lam))
            rss = tot - float(((2.0 * filt - filt**2) * e).sum())
            dof = float(filt.sum())
            denom = (self.n - dof) ** 2
            gcv = rss / denom if denom > 1e-12 else float("inf")
            if gcv < best_gcv:
                best_gcv, best_lam = gcv, float(lam)
        return best_lam

    def W(self, Y: np.ndarray) -> tuple[np.ndarray, float]:
        Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64)
        Yc = Yt - Yt.mean(0)
        VtY = self.V.T @ Yc
        lam = self._gcv_lambda(Yc, VtY)
        alpha = self.V @ (VtY / (self.w + lam)[:, None])  # (n, D)
        Wm = (self.Xn.T @ alpha).numpy()  # (H, D)
        return Wm, lam


def _flat_cosine(A: np.ndarray, B: np.ndarray) -> float:
    a, b = A.ravel(), B.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _subspace_alignment(A: np.ndarray, B: np.ndarray, ks: list[int]) -> dict:
    """Principal-angle cosines between top-k singular subspaces of A and B.

    Returns per-k mean cos(principal angle) for the INPUT side (right singular
    vectors) and OUTPUT side (left singular vectors)."""
    Ua, _sa, Vta = np.linalg.svd(A, full_matrices=False)
    Ub, _sb, Vtb = np.linalg.svd(B, full_matrices=False)
    out = {}
    for k in ks:
        # input side: right singular vectors (rows of Vt), shape (k, H)
        Min = Vta[:k] @ Vtb[:k].T
        sv_in = np.linalg.svd(Min, compute_uv=False)
        # output side: left singular vectors (cols of U), shape (H, k)
        Mout = Ua[:, :k].T @ Ub[:, :k]
        sv_out = np.linalg.svd(Mout, compute_uv=False)
        out[str(k)] = {
            "input_mean_cos_principal_angle": float(np.clip(sv_in, -1, 1).mean()),
            "output_mean_cos_principal_angle": float(np.clip(sv_out, -1, 1).mean()),
        }
    return out


def main() -> None:
    torch.set_num_threads(8)
    base = pathlib.Path(__file__).resolve().parent.parent
    out_dir = base / "eval_results" / "issue_823" / "crossarm_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = EXPECTED_N
    cx_last = _load_bundle_cx_last(n)
    v_own = _load_arm("a_prime", n)
    v_plain = _load_arm("b2", n)
    valid = _valid_idx(n)
    logger.info("Loaded cx_last + own(A_prime) + plain(B2); n=%d valid=%d", n, len(valid))

    # deterministic disjoint halves of the valid rows (for the own-arm noise ceiling)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(valid))
    half = len(valid) // 2
    h1, h2 = perm[:half], perm[half:]

    # random-map floor: two independent Gaussian (H, H) matrices (fixed seed)
    rfloor = _flat_cosine(rng.standard_normal((512, 512)), rng.standard_normal((512, 512)))
    rfloor_sub = _subspace_alignment(
        rng.standard_normal((EXPECTED_HIDDEN, EXPECTED_HIDDEN)),
        rng.standard_normal((EXPECTED_HIDDEN, EXPECTED_HIDDEN)),
        K_SWEEP,
    )

    results: dict[str, dict] = {}
    t_start = time.time()
    for L in sorted(set(READ_OUT_LAYERS.values())):
        t_layer = time.time()
        X = cx_last[valid, L, :].numpy().astype(np.float64)
        Yo = v_own[valid, L, :].numpy().astype(np.float64)
        Yp = v_plain[valid, L, :].numpy().astype(np.float64)

        shared = DualRidgeShared(X)  # eigh once, shared by own + plain
        W_own, lam_own = shared.W(Yo)
        W_plain, lam_plain = shared.W(Yp)

        # within-own disjoint-half refit noise ceiling (each half its own eigh)
        sh1 = DualRidgeShared(X[h1])
        W_own_h1, _ = sh1.W(Yo[h1])
        sh2 = DualRidgeShared(X[h2])
        W_own_h2, _ = sh2.W(Yo[h2])

        results[str(L)] = {
            "lambda_own": lam_own,
            "lambda_plain": lam_plain,
            "flat_cosine_own_vs_plain": _flat_cosine(W_own, W_plain),
            "subspace_own_vs_plain": _subspace_alignment(W_own, W_plain, K_SWEEP),
            "noise_ceiling_flat_cosine_own_half1_vs_half2": _flat_cosine(W_own_h1, W_own_h2),
            "noise_ceiling_subspace_own_half1_vs_half2": _subspace_alignment(
                W_own_h1, W_own_h2, K_SWEEP
            ),
        }
        logger.info(
            "L%d done (%.0fs): flat_cos(own,plain)=%.4f  ceiling flat_cos(own halves)=%.4f",
            L,
            time.time() - t_layer,
            results[str(L)]["flat_cosine_own_vs_plain"],
            results[str(L)]["noise_ceiling_flat_cosine_own_half1_vs_half2"],
        )

    out = {
        "description": (
            "Weight-space comparison of own-fitted (cx_last->v_A_prime) vs "
            "plain-fitted (cx_last->v_B2) ridge maps at the 3 read-out layers. "
            "W = standardized-input weight matrix (H x H); X shared across arms so "
            "W_own and W_plain live in the same basis. Flattened cosine + top-k "
            "principal-angle subspace alignment (input=right-SV, output=left-SV) + "
            "within-own disjoint-half refit noise ceiling + random-map floor."
        ),
        "read_out_layers": READ_OUT_LAYERS,
        "k_sweep": K_SWEEP,
        "solver": "dual-form GCV ridge (pop-std+1e-9, center-Y, lambda logspace(-2,4,13))",
        "n_valid": len(valid),
        "per_layer": results,
        "random_map_floor": {
            "flat_cosine_512x512": rfloor,
            "subspace_HxH": rfloor_sub,
            "note": "two independent Gaussian matrices; flat cosine on 512x512 for speed",
        },
        "tensor_source_arm": f"hf://.../{PREFIX}/analysis_tensors/ @ 8039d15f30",
        "tensor_source_cx_last": "data-repo issue779_monitoring pass_b bundle @ c94070508a",
        "git_commit": _sha(),
        "wall_seconds": round(time.time() - t_start, 1),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (out_dir / "weightspace_compare.json").write_text(json.dumps(out, indent=1))
    logger.info("Wrote %s", out_dir / "weightspace_compare.json")


if __name__ == "__main__":
    main()
