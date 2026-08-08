"""R1/R2 functional-role replacement: OUTPUT-FOOTPRINT MOMENTS for all 131,072
SAE features — direct logit lens AND J-lens-routed (J_19).

Per feature f, the write direction v_f = W_dec[:, f] projects to vocabulary as
  direct:  e_f   = W_U (gamma ⊙ v_f)          (the banked logit_footprint GEMM)
  jlens:   e_f^J = W_U (gamma ⊙ (J_19 v_f))   (routed through the averaged
                                               downstream Jacobian, so the
                                               mid-depth direct-effect gap of a
                                               plain logit lens is corrected)
and the Gurnee universal-neurons classification reads the MOMENTS of e_f over
the vocabulary: high kurtosis + positive skew = PROMOTING (a coherent token set
gains mass), high kurtosis + negative skew = SUPPRESSING, high variance at low
kurtosis = PARTITION, the rest = the no-op-on-logits bucket. Moments are
computed on raw logits AND on cos(W_U rows, write vector) (the Gurnee-faithful
normalization). Zero judge calls — kappa = 1 by construction.

Downstream reads (the point of the axis): Spearman of each moment vs
per-feature context->answer map R^2 on the 16,384-feature panel (raw + partial
given activity + partial given consistency) and at full width (n=115,150
finite), plus the pre-registered prediction that PROMOTING-class membership
correlates NEGATIVELY with R^2.

Inputs: HF-cached Qwen lm_head + final-norm gamma, cached SAE decoder, local
J_19 (data/issue_1482/jlens_dl/), banked per-feature R^2. 0 GPU; two blocked
fp32 GEMM passes on VM CPU, checkpointed every CKPT_BLOCKS blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402

JLENS = "data/issue_1482/jlens_dl/qwen2.5-7b-instruct_jlens.pt"
PANEL = "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
CONS = "eval_results/issue_1482/feature_correlates/consistency_perfeature.npz"
FULLW = "eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy"
OUT_DIR = "eval_results/issue_1482/footprint_moments"
FIG_DIR = "figures/issue_1482/footprint_moments"
BLOCK = 2048
CKPT_BLOCKS = 8  # persist accumulators every 8 blocks (T2 checkpoint law)
KURT_Q = 0.9  # within-layer empirical threshold (Gurnee's absolute values are GPT-2-specific)
VAR_Q = 0.9
JLENS_LAYER = 19

STATS = ("var", "skew", "kurt", "cos_var", "cos_skew", "cos_kurt", "write_norm")


def _load_gamma() -> np.ndarray:
    """model.norm.weight (final RMSNorm gamma) from the HF-cached shard."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from explore_persona_space.orchestrate import hub

    idx_path = hub.retry_transient(
        lambda: hf_hub_download(RS.QWEN_MODEL, "model.safetensors.index.json"),
        what="norm index fetch",
    )
    idx = json.loads(Path(idx_path).read_text())
    shard = idx["weight_map"]["model.norm.weight"]
    path = hub.retry_transient(
        lambda: hf_hub_download(RS.QWEN_MODEL, shard),
        what=f"norm shard fetch ({shard})",
    )
    with safe_open(path, framework="pt") as f:
        gamma = f.get_tensor("model.norm.weight").to(torch.float32).numpy()
    assert gamma.shape == (RS.HIDDEN_DIM,), gamma.shape
    return gamma


def _block_moments(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Central moments over the vocab axis for one (V, B) block.

    Element ops stay fp32 (values are O(1-100), fourth powers are well inside
    fp32 range); only the V-length reductions accumulate in fp64 (dtype= on
    mean) — the full-fp64 temporaries were ~3x the GEMM cost in the pilot.
    """
    mu = logits.mean(axis=0, dtype=np.float64)
    xc = logits - mu.astype(np.float32)[None, :]
    xc2 = xc * xc
    m2 = xc2.mean(axis=0, dtype=np.float64)
    m3 = (xc2 * xc).mean(axis=0, dtype=np.float64)
    xc2 *= xc2  # in-place fourth power
    m4 = xc2.mean(axis=0, dtype=np.float64)
    sd = np.sqrt(np.maximum(m2, 1e-30))
    return {
        "var": m2,
        "skew": m3 / sd**3,
        "kurt": m4 / np.maximum(m2, 1e-30) ** 2 - 3.0,  # excess kurtosis
    }


def _run_pass(
    name: str,
    w_u: np.ndarray,
    wu_row_norms: np.ndarray,
    scaled: np.ndarray,
    out_dir: Path,
    pilot: bool,
) -> dict[str, np.ndarray]:
    """One footprint pass: blocked GEMM + moment accumulation, checkpointed."""
    n_feat = scaled.shape[1]
    n_blocks = (n_feat + BLOCK - 1) // BLOCK
    ckpt = out_dir / f"ckpt_{name}.npz"
    acc = {s: np.full(n_feat, np.nan) for s in STATS}
    start_block = 0
    if ckpt.exists() and not pilot:
        z = np.load(ckpt)
        if int(z["block_size"]) == BLOCK and str(z["pass_name"]) == name:
            for s in STATS:
                acc[s] = np.asarray(z[s])
            start_block = int(z["n_blocks_done"])
            print(f"[{name}] resume from checkpoint: {start_block}/{n_blocks} blocks done")
    t0 = time.time()
    for b in range(start_block, n_blocks):
        s0, s1 = b * BLOCK, min((b + 1) * BLOCK, n_feat)
        block = scaled[:, s0:s1].astype(np.float32)
        logits = w_u @ block  # (V, B)
        m = _block_moments(logits)
        col_norms = np.maximum(np.linalg.norm(block, axis=0), 1e-30)
        cosv = logits / (wu_row_norms[:, None] * col_norms[None, :])
        mc = _block_moments(cosv)
        for s in ("var", "skew", "kurt"):
            acc[s][s0:s1] = m[s]
            acc[f"cos_{s}"][s0:s1] = mc[s]
        acc["write_norm"][s0:s1] = col_norms
        print(
            f"[{name}] block {b + 1}/{n_blocks} feats {s0}:{s1} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        if pilot:
            return acc
        if (b + 1) % CKPT_BLOCKS == 0 or b + 1 == n_blocks:
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez(tmp, pass_name=name, block_size=BLOCK, n_blocks_done=b + 1, **acc)
            tmp.replace(ckpt)
    return acc


def _classify(kurt: np.ndarray, skew: np.ndarray, var: np.ndarray) -> np.ndarray:
    """0=other, 1=promoting, 2=suppressing, 3=partition (empirical quantiles)."""
    kurt_hi = kurt > np.quantile(kurt, KURT_Q)
    var_hi = var > np.quantile(var, VAR_Q)
    cls = np.zeros(len(kurt), dtype=np.int8)
    cls[kurt_hi & (skew > 0)] = 1
    cls[kurt_hi & (skew < 0)] = 2
    cls[~kurt_hi & var_hi] = 3
    return cls


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="one block, direct pass only")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=64, layer=19, device="cpu")
    w_dec = np.asarray(sae.w_dec, dtype=np.float32)  # (d, n_feat)
    assert w_dec.shape[0] == RS.HIDDEN_DIM, w_dec.shape
    n_feat = w_dec.shape[1]

    w_u, _tok = RS.load_unembedding()
    gamma = _load_gamma()
    wu_row_norms = np.maximum(np.linalg.norm(w_u, axis=1), 1e-30)

    scaled_direct = w_dec * gamma[:, None]
    if args.pilot:
        _run_pass("direct", w_u, wu_row_norms, scaled_direct, out_dir, pilot=True)
        return

    jd = torch.load(PROJECT_ROOT / JLENS, map_location="cpu", weights_only=False)
    J = jd["J"][JLENS_LAYER].to(torch.float32).numpy()
    assert J.shape == (RS.HIDDEN_DIM, RS.HIDDEN_DIM), J.shape
    t0 = time.time()
    w_dec_j = (J @ w_dec).astype(np.float32)
    print(f"[jlens] J_19 @ W_dec done elapsed={time.time() - t0:.0f}s", flush=True)
    scaled_j = w_dec_j * gamma[:, None]

    passes = {
        "direct": _run_pass("direct", w_u, wu_row_norms, scaled_direct, out_dir, pilot=False),
        "jlens": _run_pass("jlens", w_u, wu_row_norms, scaled_j, out_dir, pilot=False),
    }
    dec_norm = np.linalg.norm(w_dec, axis=0)

    np.savez(
        out_dir / "footprint_moments.npz",
        dec_norm=dec_norm,
        **{f"{p}_{s}": passes[p][s] for p in passes for s in STATS},
    )

    # ── classification + correlations ────────────────────────────────────────
    doc: dict = {
        "design": {
            "question": (
                "R1/R2 functional-role replacement: per-feature vocabulary-footprint "
                "moments (direct logit lens + J_19-routed), Gurnee-style classes, "
                "correlated against per-feature context->answer map R^2."
            ),
            "conventions": "e_f = W_U (gamma * v); cos variant normalizes by row/col norms",
            "class_thresholds": f"kurt > q{KURT_Q:.2f}, var > q{VAR_Q:.2f} (within-layer empirical)",
            "jlens_source": "community artifact (50 wikitext prompts), J[26]~identity verified",
            "n_features": int(n_feat),
        },
        "passes": {},
        "direct_vs_jlens": {},
        "panel_correlations": {},
        "fullwidth_correlations": {},
        "preregistered_prediction": {},
    }

    cls = {}
    for p, acc in passes.items():
        c = _classify(acc["kurt"], acc["skew"], acc["var"])
        cls[p] = c
        doc["passes"][p] = {
            "class_counts": {
                "other": int((c == 0).sum()),
                "promoting": int((c == 1).sum()),
                "suppressing": int((c == 2).sum()),
                "partition": int((c == 3).sum()),
            },
            "kurt_quantiles": {
                q: float(np.quantile(acc["kurt"], float(q))) for q in ("0.5", "0.9", "0.99")
            },
        }
    agree = float((cls["direct"] == cls["jlens"]).mean())
    doc["direct_vs_jlens"] = {
        "class_agreement": agree,
        "spearman_per_stat": {s: _spear(passes["direct"][s], passes["jlens"][s]) for s in STATS},
    }

    # panel join (16,384 answer-side features)
    zp = np.load(PROJECT_ROOT / PANEL)
    feat_ids = np.asarray(zp["feat_ids"], dtype=int)
    r2p = np.asarray(zp["r2"], dtype=np.float64)
    act = np.asarray(zp["activity"], dtype=np.float64)
    zc = np.load(PROJECT_ROOT / CONS)
    assert np.array_equal(np.asarray(zc["feat_ids"], dtype=int), feat_ids)
    cons = np.asarray(zc["consistency"], dtype=np.float64)
    okp = np.isfinite(r2p)

    zfw = np.load(PROJECT_ROOT / FULLW)
    okf = np.isfinite(zfw)

    for p, acc in passes.items():
        pc: dict = {}
        for s in ("kurt", "skew", "cos_kurt", "cos_skew", "var", "write_norm"):
            v = acc[s][feat_ids]
            pc[s] = {
                "spearman_vs_r2": _spear(v[okp], r2p[okp]),
                "partial_given_activity": _partial_spear(v[okp], r2p[okp], act[okp]),
                "partial_given_consistency": _partial_spear(v[okp], r2p[okp], cons[okp]),
            }
        doc["panel_correlations"][p] = pc
        doc["fullwidth_correlations"][p] = {
            s: _spear(acc[s][okf], zfw[okf])
            for s in ("kurt", "skew", "cos_kurt", "var", "write_norm")
        }
        prom = (cls[p] == 1).astype(np.float64)
        vpanel = prom[feat_ids]
        doc["preregistered_prediction"][p] = {
            "hypothesis": "promoting-class membership correlates NEGATIVELY with map R^2",
            "panel_spearman_promoting_vs_r2": _spear(vpanel[okp], r2p[okp]),
            "panel_partial_given_activity": _partial_spear(vpanel[okp], r2p[okp], act[okp]),
            "panel_median_r2_promoting": float(np.median(r2p[okp & (vpanel > 0)])),
            "panel_median_r2_other": float(np.median(r2p[okp & (vpanel == 0)])),
            "panel_n_promoting": int((vpanel[okp] > 0).sum()),
            "fullwidth_spearman_promoting_vs_r2": _spear(prom[okf], zfw[okf]),
        }

    (out_dir / "footprint_moments.json").write_text(json.dumps(doc, indent=1))
    print(f"[out] {out_dir / 'footprint_moments.json'}")

    # ── figure ────────────────────────────────────────────────────────────────
    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    import matplotlib.pyplot as plt

    colors = paper_palette(4)
    cls_names = ("other", "promoting", "suppressing", "partition")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    rng = np.random.default_rng(1482)
    sub = rng.choice(n_feat, 20000, replace=False)
    for col, p in enumerate(("direct", "jlens")):
        ax = axes[0, col]
        acc = passes[p]
        for ci in range(4):
            m = sub[cls[p][sub] == ci]
            ax.scatter(
                np.sign(acc["skew"][m]) * np.log1p(np.abs(acc["skew"][m])),
                np.log1p(np.maximum(acc["kurt"][m], -0.99)),
                s=3,
                alpha=0.35,
                color=colors[ci],
                label=cls_names[ci],
            )
        ax.set_xlabel("signed log(1+|skew|)")
        ax.set_ylabel("log(1+excess kurtosis)")
        ax.set_title(f"{p}: footprint moments (20k sample)", loc="left")
        ax.legend(frameon=False, fontsize=8, markerscale=3)

    ax = axes[1, 0]
    data = [r2p[okp & (cls["direct"][feat_ids] == ci)] for ci in range(4)]
    bp = ax.boxplot(data, tick_labels=cls_names, showfliers=False, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("per-feature map R² (panel)")
    ax.set_title("panel R² by direct footprint class", loc="left")

    ax = axes[1, 1]
    order = np.argsort(-zfw[okf])
    prom_sorted = (cls["direct"][okf] == 1).astype(float)[order]
    n_bins = 20
    edges = np.linspace(0, len(prom_sorted), n_bins + 1).astype(int)
    prev = [prom_sorted[edges[b] : edges[b + 1]].mean() for b in range(n_bins)]
    ax.plot(range(n_bins), prev, "o-", color=colors[1])
    ax.axhline(prom_sorted.mean(), color="black", lw=0.8, ls=":")
    ax.set_xlabel("full-width R² rank bin (0 = best-predicted 5%)")
    ax.set_ylabel("promoting-class prevalence")
    ax.set_title("promoting prevalence across the full-width ranking", loc="left")
    for a_ in axes.ravel():
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, "footprint_moments", dir=PROJECT_ROOT / FIG_DIR)

    for p in passes:
        pr = doc["preregistered_prediction"][p]
        print(
            f"\n[{p}] classes {doc['passes'][p]['class_counts']}  "
            f"promoting-vs-R2 rho={pr['panel_spearman_promoting_vs_r2']:+.3f} "
            f"(partial|act {pr['panel_partial_given_activity']:+.3f}; "
            f"fullwidth {pr['fullwidth_spearman_promoting_vs_r2']:+.3f})"
        )
    print(f"[direct vs jlens] class agreement {agree:.3f}")
    for s in ("kurt", "skew", "cos_kurt"):
        print(
            f"  {s}: panel rho={doc['panel_correlations']['direct'][s]['spearman_vs_r2']:+.3f} "
            f"(partial|act {doc['panel_correlations']['direct'][s]['partial_given_activity']:+.3f}) "
            f"jlens rho={doc['panel_correlations']['jlens'][s]['spearman_vs_r2']:+.3f}"
        )


if __name__ == "__main__":
    main()
