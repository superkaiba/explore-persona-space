#!/usr/bin/env python3
"""Do context-space clusters survive into answer space? PCA projections.

PCA, not UMAP, on purpose: the claim is about GLOBAL structure, and UMAP is
designed to preserve local neighbourhoods while distorting global geometry, so
it would manufacture the effect under test.

Same 9,058 contexts in every panel, in the retrieval convention (mean-centered
context, Cholesky-whitened answer, then unit-normalized). Clusters are k-means
fit in the FULL space, not in the 2-D projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# #847: thread caps must land BEFORE the numpy/scipy imports; load_dotenv() setdefaults
# OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS on the shared VM and BLAS pools freeze at import.
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

SEED = 1901


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = x - x.mean(0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    var = (s**2) / (s**2).sum()
    return (u[:, :2] * s[:2]), var[:2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, default=ROOT / "data/issue_1901/ctxnn_dl")
    ap.add_argument(
        "--text-bank",
        type=Path,
        default=Path(
            "/home/thomasjiralerspong/.codex/worktrees/explore-persona-space/"
            "retrieval-10k-20260907/data/issue_1901/retrieval_10k/text_sources/text_bank.json"
        ),
    )
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "eval_results/issue_1901/ctxans_projection")
    a = ap.parse_args()

    w = np.load(a.stage / "issue1901_mlpdense/analysis_tensors/whiten_stats_L19.npz")
    mu_A, mu_C, L = (
        w["mu_A"].astype(np.float32),
        w["mu_C"].astype(np.float32),
        w["L"].astype(np.float32),
    )
    cz = np.load(a.stage / "cx_distr_L19.npz")
    cx, ci = cz["cx"].astype(np.float32), cz["ci"].astype(np.int64)
    az = np.load(a.stage / "issue1901_metrics/analysis_tensors/distractors_L19.npz")
    pos = {int(c): i for i, c in enumerate(az["ci"].astype(np.int64))}
    order = np.array([pos[int(c)] for c in ci], dtype=np.int64)
    vx = az["vx"][order].astype(np.float32)

    C = unit(cx - mu_C)
    # Two answer conventions on purpose. Whitening is the retrieval convention and
    # is what the RSA statistic uses, but it forces near-isotropy, so a 2-D PCA of
    # it is close to a random projection. The PROJECTION panels therefore use the
    # centered-only space, which keeps the natural variance structure.
    A_wh = unit(solve_triangular(L, (vx - mu_A).T, lower=True).T)
    A_cen = unit(vx - vx.mean(0, keepdims=True))
    print(f"{C.shape[0]} contexts aligned to answers")

    kc = KMeans(a.k, n_init=10, random_state=SEED).fit(C)
    ka_wh = KMeans(a.k, n_init=10, random_state=SEED).fit(A_wh)
    ka = KMeans(a.k, n_init=10, random_state=SEED).fit(A_cen)
    ari_wh = adjusted_rand_score(kc.labels_, ka_wh.labels_)
    ari = adjusted_rand_score(kc.labels_, ka.labels_)
    nmi = normalized_mutual_info_score(kc.labels_, ka.labels_)
    print(
        f"context vs answer clusters: ARI={ari:.3f} (centered)  "
        f"ARI={ari_wh:.3f} (whitened)  NMI={nmi:.3f}"
    )

    pc_c, var_c = pca2(C)
    pc_a, var_a = pca2(A_cen)
    _, var_a_wh = pca2(A_wh)
    print(
        f"variance explained by PC1+PC2: context {var_c.sum():.3f}  "
        f"answer centered {var_a.sum():.3f}  answer whitened {var_a_wh.sum():.3f}"
    )
    A = A_cen

    # within-cluster spread of the CONTEXT clusters, measured in each space
    def spread(space: np.ndarray, labels: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    1
                    - (
                        space[labels == g]
                        @ space[labels == g].mean(0)
                        / np.linalg.norm(space[labels == g].mean(0))
                    ).mean()
                    for g in range(a.k)
                ]
            )
        )

    print(
        f"context clusters, mean 1-cos to own centroid: "
        f"in context space {spread(C, kc.labels_):.3f}, in answer space {spread(A, kc.labels_):.3f}"
    )

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.out / "coords.npz",
        pc_context=pc_c.astype(np.float32),
        pc_answer=pc_a.astype(np.float32),
        label_context=kc.labels_,
        label_answer=ka.labels_,
        ci=ci,
    )

    bank = json.loads(a.text_bank.read_text())["rows"]
    reps: dict[str, list[str]] = {}
    for g in range(a.k):
        m = np.where(kc.labels_ == g)[0]
        cen = C[m].mean(0)
        cen /= np.linalg.norm(cen)
        near = m[np.argsort(-(C[m] @ cen))[:3]]
        reps[str(g)] = [
            " ".join(str((bank.get(str(int(ci[i])), {}) or {}).get("prompt", "")).split())[:160]
            for i in near
        ]
    summary = {
        "n": int(C.shape[0]),
        "k": a.k,
        "ari_centered": float(ari),
        "ari_whitened": float(ari_wh),
        "nmi": float(nmi),
        "var_explained": {
            "context": var_c.tolist(),
            "answer_centered": var_a.tolist(),
            "answer_whitened": var_a_wh.tolist(),
        },
        "cluster_sizes_context": np.bincount(kc.labels_, minlength=a.k).tolist(),
        "cluster_sizes_answer": np.bincount(ka.labels_, minlength=a.k).tolist(),
        "answer_space_for_panels": "centered, not whitened (whitening forces isotropy)",
        "representatives_context_clusters": reps,
        "projection": "PCA (not UMAP: the claim is about global structure)",
        "answer_draws": "single stored draw per context",
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for g in range(a.k):
        print(f"\n  cluster {g} (n={summary['cluster_sizes_context'][g]}):")
        for t in reps[str(g)]:
            print(f"    - {t}")


if __name__ == "__main__":
    main()
