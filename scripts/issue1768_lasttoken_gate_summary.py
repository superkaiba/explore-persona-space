#!/usr/bin/env python3
"""#1768 lt2: aggregate the last-token A7 gate re-read + Delta-M probe.

Answers the one question the last-token re-pool left open: the A7 base-geometry
gate predicts the whitened base similarity ``g_pred`` should track the realized
per-context write coefficient ``g_hat`` with Spearman rho in the 0.3-0.7 band.
Round 1's span-mean context pooling put the CONTENT median at +0.1384, far below
the band. If that shortfall were pooling attenuation, the last-token summary
(which tripled the base map's held-out R^2) should lift it.

Emits ``gate_lasttoken_summary.json`` plus two figures, and prints the
per-behavior old-vs-new medians the round reports back.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1768_cells as X  # noqa: E402
import issue1768_lasttoken as LT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.lt_gate_summary")

RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_1768" / "lasttoken_repool" / "gate"
FIG_DIR = REPO_ROOT / "figures" / "issue_1768" / "lasttoken_repool"
ROUND1_GATE = REPO_ROOT / "eval_results" / "issue_1768" / "gate_reads.json"
ROUND1_DMPROBE = REPO_ROOT / "eval_results" / "issue_1768" / "write_predictability"
BAND = (0.3, 0.7)  # main.tex A7 gate prediction
CONTENT_BEHS = ("cas", "imp", "syc")


def _beh_of(arm_id: str) -> str:
    """Behavior key from an arm id (round-1 gate reads carry no beh_key)."""
    return arm_id.split("-", 1)[0]


def _acc_at_1(knn: dict | None) -> float | None:
    """``acc_at_k`` is keyed by k, which JSON stringifies — accept both forms."""
    if not knn:
        return None
    d = knn.get("acc_at_k") or {}
    v = d.get(1, d.get("1"))
    return float(v) if v is not None else None


def _chance_at_1(knn: dict | None) -> float | None:
    if not knn:
        return None
    d = knn.get("chance_at_k") or {}
    v = d.get(1, d.get("1"))
    return float(v) if v is not None else None


def _med(xs: list[float]) -> float | None:
    return float(st.median(xs)) if xs else None


def _band_verdict(med: float | None) -> str:
    if med is None:
        return "no-data"
    if med < BAND[0]:
        return "below-band"
    return "in-band" if med <= BAND[1] else "above-band"


def build_gate_summary(results_dir: Path) -> dict:
    """Matched per-(arm, layer) old-vs-new gate rho, grouped by behavior + layer."""
    new = json.loads((results_dir / "gate_reads_lasttoken.json").read_text())["reads"]
    old = json.loads(ROUND1_GATE.read_text())["reads"]
    shared = sorted(set(new) & set(old))
    logger.info("matched cells: %d (new %d, round-1 %d)", len(shared), len(new), len(old))

    rows = []
    for key in shared:
        n, o = new[key], old[key]
        arm_id = n["arm_id"]
        rows.append(
            {
                "arm_id": arm_id,
                "beh": n.get("beh_key") or _beh_of(arm_id),
                "layer": int(n["layer"]),
                "method": n.get("method"),
                "rho_new": float(n["on_policy"]["spearman_rho"]),
                "rho_old": float(o["on_policy"]["spearman_rho"]),
                "rho_tf_new": float(n["matched_text"]["spearman_rho"]),
                "rho_tf_old": float(o["matched_text"]["spearman_rho"]),
            }
        )

    def group(pred) -> dict:
        sel = [r for r in rows if pred(r)]
        return {
            "n": len(sel),
            "median_rho_old": _med([r["rho_old"] for r in sel]),
            "median_rho_new": _med([r["rho_new"] for r in sel]),
            "median_rho_tf_old": _med([r["rho_tf_old"] for r in sel]),
            "median_rho_tf_new": _med([r["rho_tf_new"] for r in sel]),
            "n_new_in_band": sum(1 for r in sel if BAND[0] <= r["rho_new"] <= BAND[1]),
            "n_old_in_band": sum(1 for r in sel if BAND[0] <= r["rho_old"] <= BAND[1]),
        }

    content = group(lambda r: r["beh"] in CONTENT_BEHS)
    summary = {
        "band": list(BAND),
        "n_cells": len(rows),
        "content": {**content, "band_verdict_new": _band_verdict(content["median_rho_new"])},
        "marker": group(lambda r: r["beh"] == "mk"),
        "by_behavior": {
            b: group(lambda r, b=b: r["beh"] == b) for b in ("cas", "imp", "mk", "syc")
        },
        "content_by_layer": {
            str(li): group(lambda r, li=li: r["beh"] in CONTENT_BEHS and r["layer"] == li)
            for li in X.LAYERS
        },
        "rows": rows,
    }
    return summary


def _round1_dmprobe_r2(arm_id: str, tree: str) -> float | None:
    """Round-1 span-mean ridge held-out R^2 for the same (arm, tree) cell.

    Same ``r2_convention`` (variance-weighted pooled R^2 vs the TEST-set mean of
    w), so the two are directly comparable.
    """
    f = ROUND1_DMPROBE / "cells" / f"{arm_id}__{tree}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    v = d.get("predictors", {}).get("ridge", {}).get("heldout_r2")
    return float(v) if v is not None else None


def build_dmprobe_summary(results_dir: Path) -> dict | None:
    """New c0_lt -> w held-out R^2 vs the round-1 span-mean band, per tree."""
    path = results_dir / "dmprobe_lasttoken.json"
    if not path.exists():
        logger.warning("no dmprobe artifact at %s", path)
        return None
    cells = json.loads(path.read_text())["cells"]
    out: dict = {"n_cells": len(cells), "by_tree": {}, "rows": []}
    for _key, rec in sorted(cells.items()):
        out["rows"].append(
            {
                "arm_id": rec["arm_id"],
                "tree": rec["tree"],
                "layer": rec["layer"],
                "heldout_r2": float(rec["heldout_r2"]),
                "heldout_r2_round1": _round1_dmprobe_r2(rec["arm_id"], rec["tree"]),
                "identity_bias_r2": rec["baselines"].get("identity_bias", {}).get("heldout_r2"),
                "knn_acc1_fitted": _acc_at_1(
                    rec["baselines"].get("fitted_map", {}).get("knn_euclidean")
                ),
                "knn_acc1_identity": _acc_at_1(
                    rec["baselines"].get("identity_bias", {}).get("knn_euclidean")
                ),
                "knn_chance1": _chance_at_1(
                    rec["baselines"].get("fitted_map", {}).get("knn_euclidean")
                ),
            }
        )
    for tree in ("op", "tf"):
        sel = [r for r in out["rows"] if r["tree"] == tree]
        r2 = [r["heldout_r2"] for r in sel]
        out["by_tree"][tree] = {
            "n": len(sel),
            "median_heldout_r2": _med(r2),
            "median_heldout_r2_round1": _med(
                [r["heldout_r2_round1"] for r in sel if r["heldout_r2_round1"] is not None]
            ),
            "min_heldout_r2": min(r2) if r2 else None,
            "max_heldout_r2": max(r2) if r2 else None,
            "median_identity_bias_r2": _med(
                [r["identity_bias_r2"] for r in sel if r["identity_bias_r2"] is not None]
            ),
            "median_knn_acc1_fitted": _med(
                [r["knn_acc1_fitted"] for r in sel if r["knn_acc1_fitted"] is not None]
            ),
            "median_knn_acc1_identity": _med(
                [r["knn_acc1_identity"] for r in sel if r["knn_acc1_identity"] is not None]
            ),
            "knn_chance1": next(
                (r["knn_chance1"] for r in sel if r["knn_chance1"] is not None), None
            ),
        }
    return out


def _figures(gate: dict, dm: dict | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    c_old = paper_palette_role("baseline")
    c_new = paper_palette_role("primary")

    # Fig 1 — per-behavior median gate rho, span-mean vs last-token, vs the band.
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    behs = ["cas", "imp", "syc", "mk"]
    xs = range(len(behs))
    old = [gate["by_behavior"][b]["median_rho_old"] or 0.0 for b in behs]
    new = [gate["by_behavior"][b]["median_rho_new"] or 0.0 for b in behs]
    ax.axhspan(BAND[0], BAND[1], color="0.85", zorder=0, label="A7 predicted band (0.3-0.7)")
    ax.bar([x - 0.2 for x in xs], old, width=0.38, color=c_old, label="span-mean (round 1)")
    ax.bar([x + 0.2 for x in xs], new, width=0.38, color=c_new, label="last-token")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(behs)
    ax.set_ylabel(r"median Spearman $\rho$")
    ax.set_xlabel("behavior")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7)
    p = FIG_DIR / "gate_rho_by_behavior.png"
    savefig_paper(fig, p)
    plt.close(fig)
    written.append(str(p))

    # Fig 2 — per-cell scatter: does any cell reach the band under either pooling?
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.axhspan(BAND[0], BAND[1], color="0.9", zorder=0)
    ax.axvspan(BAND[0], BAND[1], color="0.9", zorder=0)
    for beh, mk in (("cas", "o"), ("imp", "s"), ("syc", "^"), ("mk", "x")):
        sel = [r for r in gate["rows"] if r["beh"] == beh]
        ax.scatter(
            [r["rho_old"] for r in sel],
            [r["rho_new"] for r in sel],
            s=9,
            marker=mk,
            alpha=0.7,
            label=beh,
        )
    lim = [
        min(0.0, *(r["rho_old"] for r in gate["rows"]), *(r["rho_new"] for r in gate["rows"]))
        - 0.03,
        max(BAND[1], *(r["rho_old"] for r in gate["rows"]), *(r["rho_new"] for r in gate["rows"]))
        + 0.03,
    ]
    ax.plot(lim, lim, ls="--", lw=0.8, color="0.4")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"span-mean $\rho$ (round 1)")
    ax.set_ylabel(r"last-token $\rho$")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    p = FIG_DIR / "gate_rho_scatter.png"
    savefig_paper(fig, p)
    plt.close(fig)
    written.append(str(p))

    # Fig 3 — dM probe held-out R^2 per arm, both trees, with baselines.
    if dm is not None and dm["rows"]:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        arms = sorted({r["arm_id"] for r in dm["rows"]})
        xs = range(len(arms))
        for tree, off, col, hatch, lbl in (
            ("op", -0.30, c_old, None, "op, span-mean"),
            ("op", -0.10, c_old, "//", "op, last-token"),
            ("tf", 0.10, c_new, None, "tf, span-mean"),
            ("tf", 0.30, c_new, "//", "tf, last-token"),
        ):
            field = "heldout_r2_round1" if hatch is None else "heldout_r2"
            vals = []
            for a in arms:
                m = [r[field] for r in dm["rows"] if r["arm_id"] == a and r["tree"] == tree]
                vals.append(m[0] if m and m[0] is not None else 0.0)
            ax.bar(
                [x + off for x in xs],
                vals,
                width=0.19,
                color=col,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.4,
                label=lbl,
            )
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([a.replace("-lr", "\n-lr") for a in arms], fontsize=5, rotation=30)
        ax.set_ylabel(r"held-out $R^2$ ($c_0^{lt}\rightarrow w$)")
        ax.set_xlabel("arm")
        ax.legend(frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        p = FIG_DIR / "dmprobe_lasttoken_r2.png"
        savefig_paper(fig, p)
        plt.close(fig)
        written.append(str(p))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    results_dir = args.results_dir

    gate = build_gate_summary(results_dir)
    dm = build_dmprobe_summary(results_dir)
    figs = [] if args.no_figures else _figures(gate, dm)
    out = {
        "gate": gate,
        "dmprobe": dm,
        "figures": figs,
        "round1_sources": {
            "gate": str(ROUND1_GATE.relative_to(REPO_ROOT)),
            "dmprobe": str(ROUND1_DMPROBE.relative_to(REPO_ROOT)),
        },
        **LT._meta(),
    }
    path = results_dir / "gate_lasttoken_summary.json"
    LT._atomic_json(path, out)

    c = gate["content"]
    print(f"\n=== A7 gate, content arms (n={c['n']}) ===")
    print(
        f"  median rho  span-mean {c['median_rho_old']:+.4f} -> last-token {c['median_rho_new']:+.4f}"
    )
    print(
        f"  band verdict (0.3-0.7): {c['band_verdict_new']}  in-band cells {c['n_old_in_band']} -> {c['n_new_in_band']}"
    )
    for b in ("cas", "imp", "syc", "mk"):
        g = gate["by_behavior"][b]
        print(f"  {b:4s} n={g['n']:3d}  {g['median_rho_old']:+.4f} -> {g['median_rho_new']:+.4f}")
    print("  content by layer:")
    for li, g in gate["content_by_layer"].items():
        print(f"    L{li}: {g['median_rho_old']:+.4f} -> {g['median_rho_new']:+.4f}  (n={g['n']})")
    if dm:
        print(f"\n=== dM probe c0_lt -> w (n={dm['n_cells']} cells) ===")
        for tree, g in dm["by_tree"].items():
            print(
                f"  tree={tree}: median R2 span-mean {g['median_heldout_r2_round1']:.4f} "
                f"-> last-token {g['median_heldout_r2']:.4f} "
                f"[{g['min_heldout_r2']:.4f}, {g['max_heldout_r2']:.4f}]  "
                f"identity+bias R2 {g['median_identity_bias_r2']:.3f}  "
                f"kNN acc@1 fitted {g['median_knn_acc1_fitted']:.4f} "
                f"vs identity {g['median_knn_acc1_identity']:.4f} "
                f"(chance {g['knn_chance1']:.4f})"
            )
    print(f"\nwrote {path}")
    for f in figs:
        print(f"  figure {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
