#!/usr/bin/env python3
"""Figures for #2618 — reverse (answer→context) map vs pseudoinverse constructions.

VM-side, reads the harvested compact JSONs under
``eval_results/issue_2618/reverse_map/`` ({fits_L14, fits_L19, fits_L26,
topctx}.json, written by ``scripts/issue2618_reverse_map.py``) and writes to
``figures/issue_2618/``. Fails loud when any input JSON is missing.

Five figures (paper-plots conventions: no interpretive overlays, no caption
blocks on the canvas — axes, ticks, legends, panel titles only; one colour =
one meaning across every figure):

  i2618_r2_vs_k          held-out R2 (raw context space) vs truncation rank k
                         for the truncated pinv, with the fitted reverse map,
                         ridge-pinv, identity+bias and predict-the-mean as
                         horizontal references; one panel per layer.
  i2618_knn              retrieval acc@k among the 1000 held-out contexts
                         (chance = k/1000), euclidean + cosine rows.
  i2618_operator         operator similarity vs k (direction-aware raw cosine
                         AND the rotation-invariant Procrustes ceiling —
                         labelled apart) + singular-spectrum overlay.
  i2618_preimage         cos(fitted-reverse direction, pinv direction) vs k
                         per trait + the ridge-pinv / full-pinv / W^T-read
                         comparison bars.
  i2618_topctx           top-context overlap@k (fitted-reverse vs pinv(k*) and
                         vs ridge-pinv rankings) per trait and layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402  (after load_dotenv: thread-cap discipline)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
LAYERS = (14, 19, 26)
BEHAVIORS = ("evil", "sycophancy", "hallucination")
BEHAVIOR_LABEL = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}

# One colour = one meaning, shared across every figure.
_P = paper_palette(6)
COLOR = {
    "reverse": _P[0],  # fitted reverse map
    "trunc": _P[1],  # truncated pinv (curve over k)
    "ridge_pinv": _P[2],
    "pinv_full": _P[3],
    "identity_bias": _P[4],
    "predict_mean": _P[5],
}
LABEL = {
    "reverse": "Fitted reverse map",
    "trunc": "Truncated pinv (rank k)",
    "ridge_pinv": "Ridge-pinv",
    "pinv_full": "Full-rank pinv",
    "identity_bias": "Identity + bias",
    "predict_mean": "Predict the mean",
}


def _load(results_dir: Path) -> tuple[dict[int, dict], dict]:
    fits = {}
    for ly in LAYERS:
        p = results_dir / f"fits_L{ly}.json"
        if not p.is_file():
            raise FileNotFoundError(f"missing {p} — harvest the pod-side fits JSONs first")
        fits[ly] = json.loads(p.read_text())
    tp = results_dir / "topctx.json"
    if not tp.is_file():
        raise FileNotFoundError(f"missing {tp} — harvest the pod-side topctx JSON first")
    return fits, json.loads(tp.read_text())


def _k_grid(rec: dict) -> list[int]:
    return [int(k) for k in rec["pinv_selection"]["k_grid"]]


def fig_r2_vs_k(fits: dict[int, dict], fig_dir: str) -> dict:
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(11, 3.4), sharey=True)
    for ax, ly in zip(axes, LAYERS):
        rec = fits[ly]
        ks = _k_grid(rec)
        r2 = rec["test_r2"]
        y = [r2[f"pinv_k{k}"]["r2_raw"] for k in ks]
        ax.plot(ks, y, "o-", color=COLOR["trunc"], label=LABEL["trunc"])
        k_star = int(rec["pinv_selection"]["k_star"])
        ax.plot(
            [k_star],
            [r2[f"pinv_k{k_star}"]["r2_raw"]],
            marker="*",
            ms=14,
            color=COLOR["trunc"],
            ls="none",
            label="Val-selected k*",
        )
        for name in ("reverse_ridge", "ridge_pinv", "identity_bias", "predict_mean"):
            key = "reverse" if name == "reverse_ridge" else name
            ax.axhline(r2[name]["r2_raw"], color=COLOR[key], ls="--", lw=1.4, label=LABEL[key])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("truncation rank k")
        ax.set_title(f"Layer {ly}")
    axes[0].set_ylabel("held-out $R^2$ (raw context space)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return savefig_paper(fig, "i2618_r2_vs_k", dir=fig_dir)


def fig_knn(fits: dict[int, dict], fig_dir: str) -> dict:
    metrics = ("euclidean", "cosine")
    fig, axes = plt.subplots(len(metrics), len(LAYERS), figsize=(11, 6.2), sharey=True)
    for mi, metric in enumerate(metrics):
        for li, ly in enumerate(LAYERS):
            ax = axes[mi, li]
            rec = fits[ly]["knn_retrieval"]
            best_pinv = fits[ly]["pinv_selection"]["best_pinv_variant_by_val_r2"]
            arms = [
                ("reverse_ridge", "reverse", LABEL["reverse"]),
                (best_pinv, "ridge_pinv", f"Best pinv ({best_pinv.replace('_', ' ')})"),
                ("identity_bias", "identity_bias", LABEL["identity_bias"]),
            ]
            ks = sorted(int(k) for k in rec["reverse_ridge"][metric]["acc_at_k"])
            width = 0.26
            xs = np.arange(len(ks))
            for ai, (name, ckey, lab) in enumerate(arms):
                acc = [rec[name][metric]["acc_at_k"][str(k)] for k in ks]
                ax.bar(xs + (ai - 1) * width, acc, width, color=COLOR[ckey], label=lab)
            chance = [rec["reverse_ridge"][metric]["chance_at_k"][str(k)] for k in ks]
            ax.plot(xs, chance, "k--", lw=1.2, label="Chance (k/1000)")
            ax.set_xticks(xs, [str(k) for k in ks])
            ax.set_xlabel("k nearest neighbors")
            ax.set_title(f"Layer {ly} — {metric}")
            if li == 0:
                ax.set_ylabel("P(true context in top k)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return savefig_paper(fig, "i2618_knn", dir=fig_dir)


def fig_operator(fits: dict[int, dict], fig_dir: str) -> dict:
    # Colour = variant, linestyle = statistic (one colour = one meaning).
    fig, axes = plt.subplots(2, len(LAYERS), figsize=(11, 6.2))
    for li, ly in enumerate(LAYERS):
        rec = fits[ly]["operator_geometry"]
        ks = _k_grid(fits[ly])
        ax = axes[0, li]
        raw = [rec["per_variant"][f"pinv_k{k}"]["raw_operator_cosine_direction_aware"] for k in ks]
        pro = [
            rec["per_variant"][f"pinv_k{k}"]["procrustes_cosine_rotation_invariant_only"]
            for k in ks
        ]
        ax.plot(
            ks,
            raw,
            "o-",
            color=COLOR["trunc"],
            label="Truncated pinv: raw cosine (direction-aware)",
        )
        ax.plot(
            ks,
            pro,
            "s--",
            color=COLOR["trunc"],
            label="Truncated pinv: Procrustes-aligned (rotation-invariant only)",
        )
        for name in ("pinv_full", "ridge_pinv"):
            ax.axhline(
                rec["per_variant"][name]["raw_operator_cosine_direction_aware"],
                color=COLOR[name],
                ls=":",
                lw=1.6,
                label=f"{LABEL[name]}: raw cosine",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("truncation rank k")
        ax.set_title(f"Layer {ly}")
        if li == 0:
            ax.set_ylabel("operator similarity\nto reverse map")
        ax = axes[1, li]
        spectra = rec["spectra"]
        for name, ckey in (
            ("B_rev", "reverse"),
            ("pinv_full", "pinv_full"),
            ("ridge_pinv", "ridge_pinv"),
        ):
            s = np.asarray(spectra[name], dtype=float)
            lab = "Reverse map" if name == "B_rev" else LABEL[ckey]
            ax.plot(
                np.arange(1, len(s) + 1), s, color=COLOR[ckey], lw=1.4, label=f"Spectrum: {lab}"
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("singular value index")
        if li == 0:
            ax.set_ylabel("singular value\n(shared frame)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles + h2,
        labels + l2,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.14),
        fontsize=8,
    )
    fig.tight_layout()
    return savefig_paper(fig, "i2618_operator", dir=fig_dir)


def fig_preimage(fits: dict[int, dict], fig_dir: str) -> dict:
    beh_colors = dict(zip(BEHAVIORS, paper_palette(len(BEHAVIORS))))
    fig, axes = plt.subplots(2, len(LAYERS), figsize=(11, 6.2), sharey="row")
    for li, ly in enumerate(LAYERS):
        rec = fits[ly]["preimage_agreement"]
        ks = _k_grid(fits[ly])
        ax = axes[0, li]
        for beh in BEHAVIORS:
            cos = [rec[beh]["cos_rev_vs"][f"pinv_k{k}"] for k in ks]
            ax.plot(ks, cos, "o-", color=beh_colors[beh], label=BEHAVIOR_LABEL[beh])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("truncation rank k")
        ax.set_title(f"Layer {ly}")
        if li == 0:
            ax.set_ylabel("cos(reverse-map dir, pinv dir)")
        ax = axes[1, li]
        comps = ("ridge_pinv", "pinv_full", "read_Wt")
        comp_label = {
            "ridge_pinv": "Ridge-pinv",
            "pinv_full": "Full pinv",
            "read_Wt": "$W^\\top r_B$ read",
        }
        xs = np.arange(len(comps))
        width = 0.26
        for bi, beh in enumerate(BEHAVIORS):
            vals = [rec[beh]["cos_rev_vs"][c] for c in comps]
            ax.bar(
                xs + (bi - 1) * width,
                vals,
                width,
                color=beh_colors[beh],
                label=BEHAVIOR_LABEL[beh],
            )
        ax.set_xticks(xs, [comp_label[c] for c in comps])
        ax.axhline(0.0, color="k", lw=0.8)
        if li == 0:
            ax.set_ylabel("cos vs reverse-map dir")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return savefig_paper(fig, "i2618_preimage", dir=fig_dir)


def fig_topctx(topctx: dict, fig_dir: str) -> dict:
    layers_block = topctx["layers"]
    comp_specs = [("pinv_kstar", "vs pinv(k*)"), ("ridge_pinv", "vs ridge-pinv")]
    ks = [str(k) for k in topctx["regime"]["topctx_ks"]]
    k_colors = dict(zip(ks, paper_palette(len(ks))))
    fig, axes = plt.subplots(len(comp_specs), len(LAYERS), figsize=(11, 6.2), sharey=True)
    for ci, (comp_key, comp_label) in enumerate(comp_specs):
        for li, ly in enumerate(LAYERS):
            ax = axes[ci, li]
            block = layers_block[f"L{ly}"]
            k_star = int(block["k_star"])
            xs = np.arange(len(BEHAVIORS))
            width = 0.8 / len(ks)
            for ki, k in enumerate(ks):
                vals = []
                for beh in BEHAVIORS:
                    name = f"pinv_k{k_star}" if comp_key == "pinv_kstar" else "ridge_pinv"
                    vals.append(block["per_behavior"][beh][name]["overlap_at_k"][k]["overlap_frac"])
                ax.bar(
                    xs + (ki - (len(ks) - 1) / 2) * width,
                    vals,
                    width,
                    color=k_colors[k],
                    label=f"top {k}",
                )
            ax.set_xticks(xs, [BEHAVIOR_LABEL[b] for b in BEHAVIORS])
            ax.set_ylim(0, 1)
            ax.set_title(f"Layer {ly}")
            if li == 0:
                ax.set_ylabel(f"top-k overlap,\nreverse {comp_label}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ks), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return savefig_paper(fig, "i2618_topctx", dir=fig_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results-dir", type=Path, default=REPO / "eval_results/issue_2618/reverse_map"
    )
    ap.add_argument("--figures", default=str(REPO / "figures/issue_2618"))
    args = ap.parse_args()

    set_paper_style()
    fits, topctx = _load(Path(args.results_dir))
    written = {}
    written.update(fig_r2_vs_k(fits, args.figures))
    written.update(fig_knn(fits, args.figures))
    written.update(fig_operator(fits, args.figures))
    written.update(fig_preimage(fits, args.figures))
    written.update(fig_topctx(topctx, args.figures))
    for k, v in sorted(written.items()):
        print(f"[figures] {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
