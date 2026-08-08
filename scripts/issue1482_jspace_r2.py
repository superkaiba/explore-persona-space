"""Is the J-SPACE preferentially predicted by the context->answer map?

J_19 = E[dh_final/dh_19] linearizes what layers 20-27 transmit to the output.
Its right singular vectors form an orthonormal basis of the L19 residual space
ordered by transmission strength (singular value s_i) — the top of that basis
is the "workspace"/verbalizable subspace the downstream computation actually
reads. Question: does the map preferentially predict the transmitted
directions? Per right-singular-direction held-out R^2 (ridge AND mlp_w8192,
all three arms) vs s_i, with the variance-share confound partialled, plus
pooled R^2 inside top-k J-subspaces vs their orthogonal complements.

All local (staged twoway arrays + local J_19); 0 GPU, one 3584^2 SVD.
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

JLENS = "data/issue_1482/jlens_dl/qwen2.5-7b-instruct_jlens.pt"
OUT = "eval_results/issue_1482/jspace_r2/jspace_r2.json"
FIG_DIR = "figures/issue_1482/jspace_r2"
ARMS = ("context", "prefix", "bare")
FITTERS = ("ridge", "mlp_w8192")
TOPK = (64, 256, 1024)


def _spear(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def _partial_spear(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    def rank(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a)).astype(np.float64)

    rx, ry, rz = rank(x), rank(y), rank(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        b = (b - b.mean()) / b.std()
        a = a - a.mean()
        return a - (a @ b) / len(a) * b

    ax, ay = resid(rx, rz), resid(ry, rz)
    return float((ax @ ay) / np.sqrt((ax @ ax) * (ay @ ay)))


def main() -> None:
    import torch

    J = (
        torch.load(PROJECT_ROOT / JLENS, map_location="cpu", weights_only=False)["J"][19]
        .to(torch.float32)
        .numpy()
        .astype(np.float64)
    )
    _u, s, vt = np.linalg.svd(J)  # right singular vectors = source-space basis
    V = vt.T  # (d, d), columns ordered by transmission strength s

    y16, ci = RS.load_layer(19)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    YV = Yc @ V  # target components along J's source directions
    ss_tot = np.square(YV).sum(axis=0)
    share = ss_tot / np.square(Yc).sum()

    doc: dict = {
        "design": {
            "question": (
                "Does the context->answer map preferentially predict the directions "
                "J_19 transmits to the output (the workspace/verbalizable subspace)?"
            ),
            "basis": "right singular vectors of J_19 (community artifact), ordered by s_i",
            "confound": "variance share along each direction, partialled in rank space",
            "singular_values": {
                "top5": [float(x) for x in s[:5]],
                "median": float(np.median(s)),
                "min": float(s[-1]),
            },
            "corpus": "#1738 multi-turn holdout, n=9,941, L19",
        },
        "cells": {},
    }

    for arm in ARMS:
        for fitter in FITTERS:
            try:
                pred16 = RS.load_pred(arm, 19, fitter, ci)
            except FileNotFoundError:
                continue
            E = Y - np.asarray(pred16, dtype=np.float64)
            EV = E @ V
            r2 = 1.0 - np.square(EV).sum(axis=0) / ss_tot

            cell: dict = {
                "spearman_r2_vs_logs": _spear(r2, np.log(s)),
                "spearman_share_vs_logs": _spear(share, np.log(s)),
                "partial_r2_vs_logs_given_logshare": _partial_spear(r2, np.log(s), np.log(share)),
                "r2_by_s_decile": [],
                "topk_subspace": {},
            }
            dec = np.searchsorted(
                np.quantile(np.log(s), np.linspace(0, 1, 11)[1:-1]), np.log(s), side="right"
            )
            for d10 in range(10):
                m = dec == d10
                cell["r2_by_s_decile"].append(
                    {
                        "s_decile": d10,
                        "median_r2": float(np.median(r2[m])),
                        "pooled_r2": float(
                            1 - np.square(EV[:, m]).sum() / np.square(YV[:, m]).sum()
                        ),
                        "share_sum": float(share[m].sum()),
                    }
                )
            for k in TOPK:
                top, rest = slice(0, k), slice(k, None)
                cell["topk_subspace"][str(k)] = {
                    "pooled_r2_topk": float(
                        1 - np.square(EV[:, top]).sum() / np.square(YV[:, top]).sum()
                    ),
                    "pooled_r2_complement": float(
                        1 - np.square(EV[:, rest]).sum() / np.square(YV[:, rest]).sum()
                    ),
                    "share_topk": float(share[top].sum()),
                }
            doc["cells"][f"{arm}_{fitter}"] = cell

    out = PROJECT_ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"[out] {out}")

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    import matplotlib.pyplot as plt

    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="constrained")
    ax = axes[0]
    for fi, fitter in enumerate(FITTERS):
        key = f"context_{fitter}"
        if key not in doc["cells"]:
            continue
        rows = doc["cells"][key]["r2_by_s_decile"]
        ax.plot(
            [r["s_decile"] for r in rows],
            [r["pooled_r2"] for r in rows],
            "o-",
            color=colors[fi],
            label=f"context, {fitter}",
        )
    ax.set_xlabel("J_19 transmission decile (0 = weakest s_i)")
    ax.set_ylabel("pooled held-out R² in decile")
    ax.set_title("Map R² vs J-transmission strength", loc="left")
    ax.legend(frameon=False)

    ax = axes[1]
    pred16 = RS.load_pred("context", 19, "ridge", ci)
    E = Y - np.asarray(pred16, dtype=np.float64)
    r2 = 1.0 - np.square(E @ V).sum(axis=0) / ss_tot
    sc = ax.scatter(s, r2, s=4, alpha=0.3, c=np.log10(share), cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel("singular value s_i of J_19 (log)")
    ax.set_ylabel("held-out R² along direction")
    ax.set_ylim(-0.5, 1.0)
    ax.set_title("Per-direction R² vs transmission (context, ridge)", loc="left")
    fig.colorbar(sc, ax=ax, label="log10 variance share")
    for a_ in axes:
        a_.spines[["top", "right"]].set_visible(False)
    savefig_paper(fig, "jspace_r2", dir=PROJECT_ROOT / FIG_DIR)

    for key, cell in doc["cells"].items():
        print(
            f"[{key:22s}] rho(R2, log s)={cell['spearman_r2_vs_logs']:+.3f}  "
            f"partial|share={cell['partial_r2_vs_logs_given_logshare']:+.3f}  "
            f"rho(share, log s)={cell['spearman_share_vs_logs']:+.3f}"
        )
        for k in TOPK:
            t = cell["topk_subspace"][str(k)]
            print(
                f"    top-{k:4d}: R2 {t['pooled_r2_topk']:+.3f} vs complement "
                f"{t['pooled_r2_complement']:+.3f} (share {t['share_topk']:.3f})"
            )


if __name__ == "__main__":
    main()
