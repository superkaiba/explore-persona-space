"""The three downstream r_B reads for ALL SEVEN persona-vector traits.

The rb4 round extracted optimistic / impolite / apathetic / humorous with the
banked #779 recipe; this runs "the same thing" the original three traits got:

1. per-direction held-out R^2 along each trait direction + its equivalent
   variance rank, against a 200-direction random-unit band (#1738 holdout,
   all three arms, L19 — one corpus so all seven traits are on one footing);
2. worst-20 / best-20 target-PCA direction alignment vs the 7-trait matrix
   (mirrors residual_alignment.json's 3-trait read, same random-unit null);
3. the per-SAE-feature decoder--r_B alignment correlate at 7 traits
   (reproduces the banked 3-trait Spearman as a sanity read, then widens).

All inputs local (staged twoway arrays, data/issue_779/r_b/*.pt, sae_perfeature
npz); 0 GPU.
"""

from __future__ import annotations

import json
import sys

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402

TRAITS7 = (
    "evil",
    "sycophancy",
    "hallucination",
    "optimistic",
    "impolite",
    "apathetic",
    "humorous",
)
RB_DIR = "data/issue_779/r_b"
RB_ROW = 19  # r_B row index for layer 19 (the feature_extremes convention)
ARMS = ("context", "prefix", "bare")
N_RANDOM = 200
SEED = 1482
OUT = "eval_results/issue_1482/rb7_reads/rb7_reads.json"
FIG_DIR = "figures/issue_1482/rb7_reads"
PERFEATURE = "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"


def _rb_matrix() -> np.ndarray:
    cols = []
    for t in TRAITS7:
        v = torch.load(PROJECT_ROOT / RB_DIR / f"{t}.pt", map_location="cpu", weights_only=False)
        # canonical save shape: {"trait": str, "r_b": (28, 3584), ...}
        arr = np.asarray(v["r_b"] if isinstance(v, dict) else v, dtype=np.float64)
        assert arr.ndim == 2 and arr.shape[-1] == 3584, (t, arr.shape)
        u = arr[RB_ROW]
        cols.append(u / np.linalg.norm(u))
    return np.stack(cols, axis=1)  # (3584, 7) unit columns


def main() -> None:
    rng = np.random.default_rng(SEED)
    rb = _rb_matrix()
    d = rb.shape[0]

    doc: dict = {
        "design": {
            "traits": list(TRAITS7),
            "rb_row": RB_ROW,
            "corpus": "#1738 multi-turn holdout, n=9,941, L19, ridge — one footing for all 7",
            "note": (
                "the original 3 traits' published spectrum numbers (0.858-0.941) are the "
                "#779 n10k single-turn regime; here all seven are recomputed on the "
                "multi-turn corpus so new and old traits are directly comparable"
            ),
        },
        "read1_trait_r2": {},
        "read2_pc_alignment": {},
        "read3_decoder_correlate": {},
    }

    # ── read 1: R^2 along each trait direction + variance rank + random band ──
    y16, ci = RS.load_layer(19)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    y_lam, y_vecs = RS.gram_spectrum(Yc, want_vectors=True, n_vec=RS.R2_SELECT_K)
    total_var = float(np.square(Yc).sum(axis=0).sum()) / len(Y)
    rand_dirs = rng.standard_normal((d, N_RANDOM))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)

    for arm in ARMS:
        pred16 = RS.load_pred(arm, 19, "ridge", ci)
        E = Y - np.asarray(pred16, dtype=np.float64)

        def _r2_along(U: np.ndarray) -> np.ndarray:
            st = np.square(Yc @ U).sum(axis=0)
            sr = np.square(E @ U).sum(axis=0)
            return 1.0 - sr / st

        r2_tr = _r2_along(rb)
        r2_rand = _r2_along(rand_dirs)
        var_tr = np.square(Yc @ rb).sum(axis=0) / len(Y)
        # equivalent variance rank: compare in SS units on BOTH sides — the
        # per-PC SS along the eigenbasis vs the trait direction's SS (a
        # variance-vs-gram-eigenvalue comparison is off by a factor of n and
        # spuriously ranks every trait in the deep tail).
        ss_pc = np.square(Yc @ y_vecs).sum(axis=0)
        ss_tr = np.square(Yc @ rb).sum(axis=0)
        ranks = [int((ss_pc > s).sum()) for s in ss_tr]
        doc["read1_trait_r2"][arm] = {
            "per_trait_r2": {t: float(r) for t, r in zip(TRAITS7, r2_tr)},
            "per_trait_equiv_variance_rank": {t: r for t, r in zip(TRAITS7, ranks)},
            "per_trait_variance_share": {t: float(v / total_var) for t, v in zip(TRAITS7, var_tr)},
            "random_band": {
                "mean": float(r2_rand.mean()),
                "p5": float(np.percentile(r2_rand, 5)),
                "p95": float(np.percentile(r2_rand, 95)),
                "n": N_RANDOM,
            },
        }

    # ── read 2: worst/best-PC alignment vs the 7-trait matrix ─────────────────
    banked = json.loads(
        (
            PROJECT_ROOT / "eval_results/issue_1482/twoway_residual/residual_alignment.json"
        ).read_text()
    )["cells"]
    null_abs = np.abs(rand_dirs.T @ rb)  # (N_RANDOM, 7) — random-unit null
    for arm in ARMS:
        name = RS.cell_name(arm, 19, "ridge")
        b = banked[name]
        V_worst = y_vecs[:, np.asarray(b["worst_indices"], dtype=int)]
        V_best = y_vecs[:, np.asarray(b["best_indices"], dtype=int)]
        cos_w = np.abs(V_worst.T @ rb)  # (20, 7)
        cos_b = np.abs(V_best.T @ rb)
        doc["read2_pc_alignment"][arm] = {
            "worst20_max_abs_cos_per_trait": {
                t: float(cos_w[:, j].max()) for j, t in enumerate(TRAITS7)
            },
            "best20_max_abs_cos_per_trait": {
                t: float(cos_b[:, j].max()) for j, t in enumerate(TRAITS7)
            },
            "worst20_overall_max": float(cos_w.max()),
            "best20_overall_max": float(cos_b.max()),
            "null_random_unit": {
                "mean": float(null_abs.mean()),
                "p95": float(np.percentile(null_abs, 95)),
                "max": float(null_abs.max()),
            },
        }

    # ── read 3: per-feature decoder alignment correlate, 7 traits ────────────
    z = np.load(PROJECT_ROOT / PERFEATURE)
    keys = list(z.keys())
    r2_feat = np.asarray(z["r2"] if "r2" in keys else z[keys[0]], dtype=np.float64)
    fid_key = next((k for k in keys if "feat" in k or "id" in k), None)
    feat_ids = np.asarray(z[fid_key], dtype=int) if fid_key else None

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=64, layer=19, device="cpu")
    D = np.asarray(sae.w_dec, dtype=np.float32)
    D_unit = D / np.linalg.norm(D, axis=0, keepdims=True)
    if feat_ids is None:
        raise RuntimeError(f"no feat-id key in {PERFEATURE}; keys={keys}")
    A = np.abs(D_unit[:, feat_ids].T @ rb.astype(np.float32))  # (16384, 7)

    def _spear(x: np.ndarray, y: np.ndarray) -> float:
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        rx -= rx.mean()
        ry -= ry.mean()
        return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))

    ok = np.isfinite(r2_feat)
    doc["read3_decoder_correlate"] = {
        "n_features": int(ok.sum()),
        "per_trait_spearman_align_vs_r2": {
            t: _spear(A[ok, j], r2_feat[ok]) for j, t in enumerate(TRAITS7)
        },
        "max_over_3banked_spearman": _spear(A[ok, :3].max(axis=1), r2_feat[ok]),
        "max_over_7_spearman": _spear(A[ok].max(axis=1), r2_feat[ok]),
        "banked_reference": "+0.203 (3-trait read, #1482 feature_extremes)",
    }

    out_path = PROJECT_ROOT / OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))

    # ── figure: context-arm spectrum with 7 trait markers ─────────────────────
    set_paper_style()
    import matplotlib.pyplot as plt

    pred16 = RS.load_pred("context", 19, "ridge", ci)
    E = Y - np.asarray(pred16, dtype=np.float64)
    st = np.square(Yc @ y_vecs).sum(axis=0)
    sr = np.square(E @ y_vecs).sum(axis=0)
    r2_pc = 1.0 - sr / st
    share_pc = (st / len(Y)) / total_var

    r1 = doc["read1_trait_r2"]["context"]
    colors = paper_palette(8)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.scatter(share_pc, r2_pc, s=8, alpha=0.35, color="#98a2b3", label="answer-PCA directions")
    band = r1["random_band"]
    ax.axhspan(
        band["p5"],
        band["p95"],
        color="#98a2b3",
        alpha=0.18,
        label=f"random-direction band (n={N_RANDOM})",
    )
    for j, t in enumerate(TRAITS7):
        ax.scatter(
            r1["per_trait_variance_share"][t],
            r1["per_trait_r2"][t],
            s=90,
            marker="*",
            color=colors[j],
            edgecolor="black",
            linewidth=0.6,
            label=t,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("variance share λ (log)")
    ax.set_ylabel("held-out R² along direction")
    ax.set_title(
        "All seven persona-vector directions on the R²-vs-variance spectrum "
        "(context arm, multi-turn holdout, L19)",
        loc="left",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, "rb7_spectrum", dir=PROJECT_ROOT / FIG_DIR)

    print(f"[out] {out_path}")
    for arm in ARMS:
        r = doc["read1_trait_r2"][arm]
        band = r["random_band"]
        print(f"\n[{arm}] random band {band['mean']:.3f} [{band['p5']:.3f}, {band['p95']:.3f}]")
        for t in TRAITS7:
            print(
                f"   {t:14s} R2 {r['per_trait_r2'][t]:+.3f}  "
                f"rank {r['per_trait_equiv_variance_rank'][t]:4d}  "
                f"share {r['per_trait_variance_share'][t]:.5f}"
            )
    r3 = doc["read3_decoder_correlate"]
    print(
        "\n[decoder correlate] per-trait:",
        {t: round(v, 3) for t, v in r3["per_trait_spearman_align_vs_r2"].items()},
    )
    print(
        f"  max-over-3 {r3['max_over_3banked_spearman']:+.3f} (banked ref +0.203) | max-over-7 {r3['max_over_7_spearman']:+.3f}"
    )


if __name__ == "__main__":
    main()
