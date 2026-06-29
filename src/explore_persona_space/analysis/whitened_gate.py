"""Whitened key-query gate (issue #665 Phase 3, B3 — net-new code).

Implements the theory paper's bilinear key-query gate (Overleaf 6a2df2d2,
`a:bilinear-gate` / A7):

    g_C(C') = c_Cᵀ M c_C' / c_Cᵀ M c_C ,   M = (Σc + λI)⁻¹

with the special-case reduction (paper "Relation-to-cosine"): in the limit
Σc = I (so M ∝ I), equal-norm keys/queries (‖c_C‖ = ‖c_C'‖), and δ ∥ r_B, the
whitened gate reduces to the raw cosine `cos(c_C, c_C')`. The B3 reduction unit
test (`tests/test_whitened_gate.py`) pins that identity to within 1e-6 on
synthetic data AND asserts finite / non-NaN output at the smallest swept
λ = 1e-3.

The λ floor is LOAD-BEARING, not merely stabilizing: Σc was captured at n = 3000
< d = 3584 (#658), so the RAW Σc is genuinely singular — `(Σc + λI)⁻¹` is what
makes the inverse EXIST at all. A too-small λ that lets the singular Σc leak
through is caught by the finite/non-NaN assert at λ = 1e-3.

Phase 4 (#666) imports this module unchanged (the shared code surface named in
the #665 plan §5 land-freeze ordering). Write-once, import-by-path — do not
duplicate.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "METRIC_KEYS",
    "diag_inv",
    "key_query_gate",
    "metric_ablation",
    "raw_cosine_gate",
    "sigma_c_inv",
    "whitened_gate",
]

# The metric-ablation cell labels (A3.9 metric ablation): identity, diagonal
# regularized inverse, full regularized inverse. The full `(Σc+λI)⁻¹` is the
# boxed predictor (verdict ii); identity reduces the gate to (un-normalized)
# inner products and is the cosine-adjacent control.
METRIC_KEYS = ("I", "diag_Sigma_inv", "Sigma_inv")


def sigma_c_inv(sigma_c: np.ndarray, lam: float) -> np.ndarray:
    """Return the regularized inverse metric `M = (Σc + λI)⁻¹`.

    Args:
        sigma_c: (d, d) covariance (symmetric PSD; genuinely singular at n<d).
        lam: ridge floor λ > 0 — LOAD-BEARING (Σc is singular, so λ makes the
            inverse exist). Must be strictly positive.

    Returns:
        (d, d) float64 inverse metric.
    """
    assert lam > 0.0, f"lambda floor must be > 0 (Sigma_c is singular at n<d): got {lam}"
    sc = np.asarray(sigma_c, dtype=np.float64)
    assert sc.ndim == 2 and sc.shape[0] == sc.shape[1], f"sigma_c must be square (d,d): {sc.shape}"
    d = sc.shape[0]
    reg = sc + lam * np.eye(d, dtype=np.float64)
    # solve(reg, I) is the numerically-preferred inverse for a regularized
    # symmetric matrix (vs np.linalg.inv which forms it less stably).
    return np.linalg.solve(reg, np.eye(d, dtype=np.float64))


def diag_inv(sigma_c: np.ndarray, lam: float) -> np.ndarray:
    """Return the DIAGONAL regularized inverse metric `diag(Σc + λI)⁻¹` (A3.9
    metric-ablation cell). A (d,) vector of per-dimension inverse weights;
    `key_query_gate` accepts it as a diagonal metric via the M.ndim branch."""
    assert lam > 0.0, f"lambda floor must be > 0: got {lam}"
    sc = np.asarray(sigma_c, dtype=np.float64)
    return 1.0 / (np.diag(sc) + lam)


def key_query_gate(k: np.ndarray, q: np.ndarray, q_src: np.ndarray, M: np.ndarray) -> float:
    """Normalized bilinear key-query gate `g = kᵀ M q / kᵀ M q_src`.

    Args:
        k: (d,) key vector (the source context vector c_C).
        q: (d,) query vector (the target context vector c_C').
        q_src: (d,) the source query (c_C) — the normalizer denominator query,
            so g(C=C') = 1 by construction (the source self-gate is the unit).
        M: (d, d) full metric OR (d,) diagonal metric OR scalar (identity ∝).

    Returns:
        the scalar gate value (float). Raises if the denominator is ~0.
    """
    k = np.asarray(k, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    q_src = np.asarray(q_src, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 0:  # scalar (identity * c)
        num = float(M) * float(k @ q)
        den = float(M) * float(k @ q_src)
    elif M.ndim == 1:  # diagonal metric
        assert M.shape[0] == k.shape[0], (M.shape, k.shape)
        num = float(k @ (M * q))
        den = float(k @ (M * q_src))
    else:  # full metric
        assert M.shape == (k.shape[0], k.shape[0]), (M.shape, k.shape)
        Mq = M @ q
        Mqs = M @ q_src
        num = float(k @ Mq)
        den = float(k @ Mqs)
    # denominator stability (A3.9 control |kᵀMq_C|): a vanishing denominator
    # is a real numerical failure, not a silently-coerced value — fail loud.
    if not np.isfinite(den) or abs(den) < 1e-30:
        raise ValueError(
            f"key_query_gate denominator unstable (|kᵀMq_src|={den:.3e}); "
            "Sigma_c metric or key/query degenerate."
        )
    g = num / den
    if not np.isfinite(g):
        raise ValueError(f"key_query_gate non-finite output g={g} (num={num}, den={den})")
    return g


def whitened_gate(c_C: np.ndarray, c_Cp: np.ndarray, M: np.ndarray) -> float:
    """The boxed whitened gate `g_C(C') = c_Cᵀ M c_C' / c_Cᵀ M c_C`.

    Thin wrapper over `key_query_gate` with key = query-source = c_C, query = c_C'.
    """
    return key_query_gate(k=c_C, q=c_Cp, q_src=c_C, M=M)


def raw_cosine_gate(c_C: np.ndarray, c_Cp: np.ndarray) -> float:
    """Raw cosine baseline gate `cos(c_C, c_C')` (the A3.9 baseline to beat).

    NOT normalized to g(C=C')=1 the way the bilinear gate is — cos(c_C,c_C)=1
    holds by definition for the cosine, so the two agree at the source anchor.
    """
    c_C = np.asarray(c_C, dtype=np.float64).ravel()
    c_Cp = np.asarray(c_Cp, dtype=np.float64).ravel()
    nC = np.linalg.norm(c_C)
    nCp = np.linalg.norm(c_Cp)
    if nC < 1e-30 or nCp < 1e-30:
        raise ValueError(f"raw_cosine_gate degenerate norm (‖c_C‖={nC:.3e}, ‖c_C'‖={nCp:.3e})")
    return float((c_C @ c_Cp) / (nC * nCp))


def metric_ablation(sigma_c: np.ndarray, lam: float) -> dict[str, np.ndarray | float]:
    """Build the three A3.9 metric-ablation matrices keyed by `METRIC_KEYS`.

    Returns {"I": 1.0 (scalar identity), "diag_Sigma_inv": (d,), "Sigma_inv": (d,d)}.
    The identity is returned as a scalar so `key_query_gate` treats it as I*c
    (cheap — no (d,d) allocation); the gate's normalization makes the scalar drop.
    """
    return {
        "I": 1.0,
        "diag_Sigma_inv": diag_inv(sigma_c, lam),
        "Sigma_inv": sigma_c_inv(sigma_c, lam),
    }
