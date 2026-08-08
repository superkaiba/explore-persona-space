#!/usr/bin/env python3
"""#1768 inline round: aggregate the last-token re-pool + render its figures.

Reads the per-arm cell JSONs written by ``issue1768_lasttoken_fit.py`` and the
ROUND-1 verdict table (``eval_results/issue_1768/map_change_summary.json``),
and emits

  eval_results/issue_1768/lasttoken_repool/summary.json
  figures/issue_1768/lasttoken_repool/{base_r2_by_layer,d_scatter_flips,
                                       movement_and_baselines}.png

The headline reads it answers, per captured token position:

* **base R2** -- does #779's last-token advantage transfer to this corpus? The
  comparison is against round 1's own ``m0_r2`` at the same layer, same rows.
* **D verdict agreement** -- a round-1-vs-last-token confusion matrix over the
  216 (arm, layer) cells, with every flip enumerated.
* **context movement** and the identity+learned-bias / kNN-retrieval baselines
  under the last-token summary.

Answer-side reads (horse race, write rank, read-out stability) are
context-pooling-INVARIANT and are deliberately not re-run; the summary records
that explicitly rather than leaving their absence to be inferred.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.lt_summary")

RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_1768" / "lasttoken_repool"
FIG_DIR = REPO_ROOT / "figures" / "issue_1768" / "lasttoken_repool"
ROUND1_SUMMARY = REPO_ROOT / "eval_results" / "issue_1768" / "map_change_summary.json"

NOT_RERUN = {
    "reads": ["horse_race", "write_rank", "readout_stability"],
    "reason": (
        "context-pooling-invariant: these are ANSWER-side reads (w-hat / delta / r_B "
        "and the marker unembedding row), computed from the response-span stores this "
        "round reuses unchanged. Changing the CONTEXT summary cannot move them, so "
        "re-running them would consume GPU for a bit-identical result."
    ),
}


def _meta() -> dict:
    import subprocess

    import numpy
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = ""
    import datetime as _dt

    return {
        "git_commit": commit,
        "ts": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "issue": 1768,
    }


def load_cells(results_dir: Path) -> dict[str, dict]:
    cells_dir = results_dir / "cells"
    out: dict[str, dict] = {}
    for p in sorted(cells_dir.glob("*.json")):
        rec = json.loads(p.read_text())
        out[rec["arm_id"]] = rec
    return out


def round1_table() -> dict[str, dict]:
    assert ROUND1_SUMMARY.exists(), f"round-1 summary missing: {ROUND1_SUMMARY}"
    return json.loads(ROUND1_SUMMARY.read_text())["verdicts"]


def _med(vals: list[float]) -> float | None:
    return float(st.median(vals)) if vals else None


def build_summary(cells: dict[str, dict], r1: dict[str, dict], layers: list[int]) -> dict:
    positions = sorted({p for rec in cells.values() for p in rec.get("positions", {})})
    summary: dict = {
        "n_arms_fitted": len(cells),
        "n_arms_expected": len(X.all_arms()),
        "layers": layers,
        "positions": positions,
        "not_rerun": NOT_RERUN,
        "round1_reference": {
            "source": str(ROUND1_SUMMARY.relative_to(REPO_ROOT)),
            "note": "span-mean-over-prompt context inputs (the round-1 convention)",
        },
        "by_position": {},
        **_meta(),
    }

    for pos in positions:
        base_r2: dict[str, dict] = {}
        verdict_conf: dict[str, dict[str, int]] = {}
        flips: list[dict] = []
        d_pairs: list[dict] = []
        movement: dict[str, dict] = {}
        baselines: dict[str, dict] = {}

        for layer in layers:
            lk = str(layer)
            lt_r2, r1_r2, lt_d, r1_d = [], [], [], []
            mv_rel, mv_cos = [], []
            ib_r2, ib_k1, fit_k1 = [], [], []
            for arm_id, rec in cells.items():
                cell = rec.get("positions", {}).get(pos, {}).get(lk)
                if cell is None:
                    continue
                key = f"{arm_id}_L{layer}"
                ref = r1.get(key)
                lt_r2.append(cell["M0"]["heldout_r2"])
                lt_d.append(cell["map_change"]["D"])
                mv_rel.append(cell["context_movement"]["median_relative_move"])
                mv_cos.append(cell["context_movement"]["median_cos_c0_cplus"])
                ib = cell["baselines"].get("identity_bias", {})
                if "heldout_r2" in ib:
                    ib_r2.append(ib["heldout_r2"])
                    a = _acc1(ib.get("knn_euclidean"))
                    if a is not None:
                        ib_k1.append(a)
                a = _acc1(cell["baselines"].get("fitted_map", {}).get("knn_euclidean"))
                if a is not None:
                    fit_k1.append(a)
                if ref is None:
                    continue
                r1_r2.append(ref["m0_r2"])
                r1_d.append(ref["D"])
                v_lt = cell["map_change"]["verdict"]
                v_r1 = ref["verdict"]
                verdict_conf.setdefault(v_r1, {}).setdefault(v_lt, 0)
                verdict_conf[v_r1][v_lt] += 1
                d_pairs.append(
                    {
                        "cell": key,
                        "layer": layer,
                        "method": ref.get("method"),
                        "D_round1": ref["D"],
                        "D_lasttoken": cell["map_change"]["D"],
                        "verdict_round1": v_r1,
                        "verdict_lasttoken": v_lt,
                        "flipped": v_r1 != v_lt,
                    }
                )
                if v_r1 != v_lt:
                    flips.append(d_pairs[-1])

            base_r2[lk] = {
                "n_cells": len(lt_r2),
                "lasttoken_median": _med(lt_r2),
                "round1_median": _med(r1_r2),
                "delta_median": (
                    None if not lt_r2 or not r1_r2 else float(st.median(lt_r2) - st.median(r1_r2))
                ),
                "lasttoken_min": min(lt_r2) if lt_r2 else None,
                "lasttoken_max": max(lt_r2) if lt_r2 else None,
            }
            movement[lk] = {
                "median_relative_move": _med(mv_rel),
                "median_cos_c0_cplus": _med(mv_cos),
                "n_cells": len(mv_rel),
            }
            baselines[lk] = {
                "identity_bias_r2_median": _med(ib_r2),
                "identity_bias_knn_acc1_median": _med(ib_k1),
                "fitted_map_knn_acc1_median": _med(fit_k1),
                "n_cells": len(fit_k1),
            }

        n_pairs = len(d_pairs)
        n_agree = sum(1 for r in d_pairs if not r["flipped"])
        summary["by_position"][pos] = {
            "base_r2_by_layer": base_r2,
            "verdict_confusion_round1_x_lasttoken": verdict_conf,
            "verdict_agreement": {
                "n_cells_compared": n_pairs,
                "n_agree": n_agree,
                "agreement_rate": (n_agree / n_pairs) if n_pairs else None,
                "n_flips": len(flips),
            },
            "verdict_counts_lasttoken": _counts([r["verdict_lasttoken"] for r in d_pairs]),
            "verdict_counts_round1": _counts([r["verdict_round1"] for r in d_pairs]),
            "flips": flips,
            "d_pairs": d_pairs,
            "context_movement_by_layer": movement,
            "baselines_by_layer": baselines,
        }
    return summary


def _acc1(knn: object) -> float | None:
    """acc@1 out of a ``knn_retrieval`` dict.

    The helper returns ``acc_at_k`` keyed by INT k; a JSON round-trip turns
    those keys into STRINGS, so both forms are accepted. Reading the wrong key
    silently yields None and an empty baselines panel rather than an error.
    """
    if not isinstance(knn, dict):
        return None
    acc = knn.get("acc_at_k")
    if not isinstance(acc, dict):
        return None
    for key in (1, "1"):
        if key in acc:
            return float(acc[key])
    return None


def _counts(vals: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out


# ── figures ──────────────────────────────────────────────────────────────────


def _figures(summary: dict, fig_dir: Path, position: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    c_lt = paper_palette_role("primary")
    c_r1 = paper_palette_role("baseline")
    c_acc = paper_palette_role("accent")
    block = summary["by_position"][position]
    layers = [str(x) for x in summary["layers"]]
    written: list[Path] = []
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. base R2: span-mean vs last-token, per layer
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = range(len(layers))
    w = 0.38
    r1v = [block["base_r2_by_layer"][L]["round1_median"] for L in layers]
    ltv = [block["base_r2_by_layer"][L]["lasttoken_median"] for L in layers]
    ax.bar([x - w / 2 for x in xs], r1v, w, label="span-mean (round 1)", color=c_r1)
    ax.bar([x + w / 2 for x in xs], ltv, w, label=f"last-token ({position})", color=c_lt)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("base map held-out $R^2$ (median)")
    ax.set_xlabel("layer")
    # legend ABOVE the axes: the bars fill the panel, so an in-axes legend
    # overlaps the tallest bar (observed on the mid layer).
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout()
    written += list(savefig_paper(fig, "base_r2_by_layer", dir=fig_dir).values())
    plt.close(fig)

    # 2. D_lt vs D_round1 scatter, flips highlighted
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    pairs = block["d_pairs"]
    if pairs:
        keep = [p for p in pairs if not p["flipped"]]
        flip = [p for p in pairs if p["flipped"]]
        ax.scatter(
            [p["D_round1"] for p in keep],
            [p["D_lasttoken"] for p in keep],
            s=16,
            color=c_r1,
            alpha=0.65,
            label=f"verdict held (n={len(keep)})",
        )
        ax.scatter(
            [p["D_round1"] for p in flip],
            [p["D_lasttoken"] for p in flip],
            s=30,
            color=c_acc,
            marker="D",
            label=f"verdict FLIPPED (n={len(flip)})",
        )
        allv = [p["D_round1"] for p in pairs] + [p["D_lasttoken"] for p in pairs]
        lo, hi = min(allv), max(allv)
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls=":", lw=1, color="0.5")
        ax.axhline(0, lw=0.8, color="0.7")
        ax.axvline(0, lw=0.8, color="0.7")
    ax.set_xlabel("$D$ (span-mean, round 1)")
    ax.set_ylabel(f"$D$ (last-token, {position})")
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    written += list(savefig_paper(fig, "d_scatter_flips", dir=fig_dir).values())
    plt.close(fig)

    # 3. context movement + mapping baselines under the last-token summary
    fig, (axm, axb) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    axm.bar(
        list(xs),
        [block["context_movement_by_layer"][L]["median_relative_move"] for L in layers],
        0.55,
        color=c_lt,
    )
    axm.set_xticks(list(xs))
    axm.set_xticklabels([f"L{L}" for L in layers])
    axm.set_ylabel(r"median $\|\Delta c\| / \|c^0\|$")
    axm.set_xlabel("layer")
    axm.set_title("last-token context movement", fontsize=9)
    ib = [block["baselines_by_layer"][L]["identity_bias_knn_acc1_median"] for L in layers]
    fm = [block["baselines_by_layer"][L]["fitted_map_knn_acc1_median"] for L in layers]
    axb.bar([x - w / 2 for x in xs], ib, w, label="identity+bias", color=c_r1)
    axb.bar([x + w / 2 for x in xs], fm, w, label="fitted ridge", color=c_lt)
    axb.set_xticks(list(xs))
    axb.set_xticklabels([f"L{L}" for L in layers])
    axb.set_ylabel("kNN retrieval acc@1 (median)")
    axb.set_xlabel("layer")
    axb.set_title("mapping baselines", fontsize=9)
    axb.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    written += list(savefig_paper(fig, "movement_and_baselines", dir=fig_dir).values())
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--position", default="last_prompt", help="position to plot")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette_role,
            savefig_paper,
            set_paper_style,
        )

        print("import-check ok")
        return 0

    layers = [int(x) for x in args.layers.split(",")]
    cells = load_cells(args.results_dir)
    assert cells, f"no cell JSONs under {args.results_dir / 'cells'}"
    summary = build_summary(cells, round1_table(), layers)
    out = args.results_dir / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    tmp.replace(out)
    logger.info(
        "[summary] %d/%d arms; positions=%s -> %s",
        summary["n_arms_fitted"],
        summary["n_arms_expected"],
        summary["positions"],
        out,
    )
    for pos in summary["positions"]:
        b = summary["by_position"][pos]["verdict_agreement"]
        logger.info(
            "[summary] pos=%s agreement=%s/%s (%.3f) flips=%s",
            pos,
            b["n_agree"],
            b["n_cells_compared"],
            b["agreement_rate"] or float("nan"),
            b["n_flips"],
        )
        for L in [str(x) for x in layers]:
            r = summary["by_position"][pos]["base_r2_by_layer"][L]
            logger.info(
                "[summary] pos=%s L%s base_r2 lt=%s round1=%s delta=%s",
                pos,
                L,
                None if r["lasttoken_median"] is None else round(r["lasttoken_median"], 4),
                None if r["round1_median"] is None else round(r["round1_median"], 4),
                None if r["delta_median"] is None else round(r["delta_median"], 4),
            )
    if not args.no_figures:
        pos = args.position if args.position in summary["positions"] else summary["positions"][0]
        written = _figures(summary, args.fig_dir, pos)
        logger.info("[summary] figures: %s", [str(p) for p in written])
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
