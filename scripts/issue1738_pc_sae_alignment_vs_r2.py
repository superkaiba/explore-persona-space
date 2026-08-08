"""Is per-direction predictability associated with SAE-dictionary alignment —
beyond what variance already explains?

The banked two-point contrast (best-20 PCs align at |cos| 0.29-0.63 vs worst-20
at ~null) is variance-confounded: best-20 are top-variance directions and the
SAE preferentially represents high-variance structure (#1895: subspace overlap
~98% variance-driven). This computes the CONTINUOUS read over the top-256
target PCs of the #1738 holdout (context arm, ridge): per-direction max-|cos|
to the 131,072-atom dictionary vs per-direction R^2 — raw, variance-partialled,
and against the floor-curve deviation (the variance-free anomaly score).

All local (staged twoway arrays + cached SAE weights); 0 GPU.
"""

from __future__ import annotations

import json
import sys

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402

OUT = "eval_results/issue_1738/pc_sae_alignment/pc_sae_alignment_vs_r2.json"
K = 256
N_NULL = 200
SEED = 1738


def _spear(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def _partial_spear(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    def rank(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a)).astype(float)

    rx, ry, rz = rank(x), rank(y), rank(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        b = (b - b.mean()) / b.std()
        a = a - a.mean()
        return a - (a @ b) / len(a) * b

    ax, ay = resid(rx, rz), resid(ry, rz)
    return float((ax @ ay) / np.sqrt((ax @ ax) * (ay @ ay)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    y16, ci = RS.load_layer(19)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _lam, vecs = RS.gram_spectrum(Yc, want_vectors=True, n_vec=K)
    pred16 = RS.load_pred("context", 19, "ridge", ci)
    E = Y - np.asarray(pred16, dtype=np.float64)
    ss_tot = np.square(Yc @ vecs).sum(axis=0)
    r2 = 1.0 - np.square(E @ vecs).sum(axis=0) / ss_tot
    share = ss_tot / np.square(Yc).sum()

    # floor-curve deviation (2-param OLS on 1/share over the same 256 PCs)
    x = 1.0 / share
    X = np.stack([np.ones_like(x), x], 1)
    beta, *_ = np.linalg.lstsq(X, r2, rcond=None)
    dev = (X @ beta) - r2  # positive = worse than the size expectation
    gof = 1 - ((r2 - X @ beta) ** 2).sum() / ((r2 - r2.mean()) ** 2).sum()

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=64, layer=19, device="cpu")
    D = np.asarray(sae.w_dec, dtype=np.float32)
    D_unit = (D / np.linalg.norm(D, axis=0, keepdims=True)).T  # (n_feat, d)
    align = np.abs(D_unit @ vecs.astype(np.float32)).max(axis=0).astype(np.float64)

    rand = rng.standard_normal((vecs.shape[0], N_NULL))
    rand /= np.linalg.norm(rand, axis=0, keepdims=True)
    null_align = np.abs(D_unit @ rand.astype(np.float32)).max(axis=0).astype(np.float64)

    res = {
        "design": {
            "question": (
                "Are better-predicted directions more SAE-aligned, beyond what "
                "variance explains? Continuous read over the top-256 target PCs "
                "(#1738 holdout, context arm, ridge, L19)."
            ),
            "alignment": "max |cos| over the 131,072 unit-normalized decoder columns",
            "deviation": f"2-param floor curve fit on these 256 PCs (gof {gof:.3f})",
            "confound_reference": "#1895: map/SAE subspace overlap ~98% variance-driven",
        },
        "n_pcs": K,
        "alignment_summary": {
            "min": float(align.min()),
            "median": float(np.median(align)),
            "max": float(align.max()),
            "null_random_unit": {
                "mean": float(null_align.mean()),
                "p95": float(np.percentile(null_align, 95)),
                "max": float(null_align.max()),
                "n": N_NULL,
            },
            "n_pcs_above_null_p95": int((align > np.percentile(null_align, 95)).sum()),
        },
        "correlations": {
            "spearman_r2_vs_align": _spear(r2, align),
            "spearman_logshare_vs_align": _spear(np.log(share), align),
            "partial_r2_vs_align_given_logshare": _partial_spear(r2, align, np.log(share)),
            "spearman_deviation_vs_align": _spear(dev, align),
        },
    }
    out = PROJECT_ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[out] {out}")
    print(
        f"align: median {np.median(align):.3f}, null mean {null_align.mean():.3f} "
        f"p95 {np.percentile(null_align, 95):.3f}; {res['alignment_summary']['n_pcs_above_null_p95']}/256 above null p95"
    )
    for k, v in res["correlations"].items():
        print(f"  {k}: {v:+.3f}")


if __name__ == "__main__":
    main()
