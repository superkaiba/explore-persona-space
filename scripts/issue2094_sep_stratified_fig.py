"""Separation-stratified F_beh recount figure for issue #2094 (analyzer round 1).

The raw per-cell F_beh means are dominated by pairs whose anchor separation
(the F denominator, ceiling-minus-floor judge contrast) is near zero — the
bare<->conv prefix-rubric pairs sit at |sep| ~ 0.005-0.03, giving 30-200x
leverage per draw. This figure recomputes the four largest stage-1 headline
cells restricted to well-separated pairs (|sep| >= 0.5), steered vs the
norm-matched shuffled-donor null, with pair-clustered bootstrap CIs, plus a
per-pair paired view (the low-level companion).

Reads eval_results/issue_2094/f_metrics/{f_cells,null_cells,anchors}.jsonl.
Writes figures/issue_2094/result_sep_stratified_fbeh.{png,pdf,meta.json}.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

BASE = Path("eval_results/issue_2094")
MINSEP = 0.5
RNG = np.random.default_rng(42)
B = 10_000

# (setting, slot, layer_variant, dose, vec_type, rubric_kind, plain-English label)
CELLS = [
    (
        "matched_query",
        "ce",
        "L19",
        "a4",
        "B",
        "prefix",
        "matched query\ncontext-end L19, dose 4x,\nprefix-centroid",
    ),
    (
        "cross",
        "ce",
        "L19",
        "a2",
        "A",
        "prefix",
        "cross\ncontext-end L19, dose 2x,\npair difference",
    ),
    (
        "cross",
        "ce",
        "L14",
        "a4",
        "A",
        "prefix",
        "cross\ncontext-end L14, dose 4x,\npair difference",
    ),
    (
        "matched_query",
        "ce",
        "L14",
        "replace",
        "A",
        "prefix",
        "matched query\ncontext-end L14, full-state\npatch",
    ),
]


def load_anchors() -> dict[tuple[str, str], dict]:
    anch = {}
    for line in open(BASE / "f_metrics/anchors.jsonl"):
        a = json.loads(line)
        anch[(a["pair_id"], a["kind"])] = a
    return anch


def load_cells(fn: Path) -> dict[tuple, dict[str, dict]]:
    out: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for line in open(fn):
        r = json.loads(line)
        if r.get("degenerate_self") or not r.get("coherent"):
            continue
        key = (r["setting"], r["slot"], r["layer_variant"], r["dose"], r["vec_type"])
        out[key][r["pair_id"]] = r
    return out


def fbeh(row: dict, kind: str) -> float | None:
    fb = (row.get("f_beh") or {}).get(kind)
    if not fb or fb.get("f_beh") is None:
        return None
    return float(fb["f_beh"])


def pair_boot_ci(vals: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    anch = load_anchors()
    steered = load_cells(BASE / "f_metrics/f_cells.jsonl")
    null = load_cells(BASE / "f_metrics/null_cells.jsonl")

    set_paper_style("blog")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")

    c_all = paper_palette_role("neutral")
    c_st = paper_palette_role("primary")
    c_nu = paper_palette_role("control")

    width = 0.26
    xs = np.arange(len(CELLS))
    per_pair_records = []
    for i, (s, slot, lv, dose, vt, kind, label) in enumerate(CELLS):
        key = (s, slot, lv, dose, vt)
        st_rows, nu_rows = steered.get(key, {}), null.get(key, {})
        all_vals = np.array([v for r in st_rows.values() if (v := fbeh(r, kind)) is not None])
        ws_pairs = [
            pid
            for pid in st_rows
            if (a := anch.get((pid, kind))) and abs(a["separation"]) >= MINSEP
        ]
        ws_st = np.array([v for p in ws_pairs if (v := fbeh(st_rows[p], kind)) is not None])
        ws_nu = np.array(
            [v for p in ws_pairs if p in nu_rows and (v := fbeh(nu_rows[p], kind)) is not None]
        )
        for bx, vals, color, lab in (
            (xs[i] - width, all_vals, c_all, "all pairs (steered)"),
            (xs[i], ws_st, c_st, "well-separated pairs (steered)"),
            (xs[i] + width, ws_nu, c_nu, "well-separated pairs (shuffled-donor null)"),
        ):
            lo, hi = pair_boot_ci(vals)
            m = float(vals.mean())
            ax1.bar(bx, m, width=width * 0.92, color=color, label=lab if i == 0 else None)
            ax1.errorbar(
                bx, m, yerr=[[m - lo], [hi - m]], color="black", capsize=2.5, linewidth=1.0
            )
        # per-pair paired view (panel B)
        for p in ws_pairs:
            sv, nv = (
                fbeh(st_rows[p], kind),
                fbeh(nu_rows.get(p, {}), kind) if p in nu_rows else None,
            )
            if sv is None or nv is None:
                continue
            per_pair_records.append((i, p, sv, nv))

    ax1.axhline(0.0, color="grey", linewidth=0.8)
    ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([c[6] for c in CELLS], fontsize=7)
    ax1.set_ylabel("mean F_beh (fraction of behavior swap)")
    ax1.set_title("cell means: all pairs vs well-separated pairs", loc="left", fontsize=9)
    ax1.legend(fontsize=7)

    jit = (RNG.random(len(per_pair_records)) - 0.5) * 0.22
    for (i, p, sv, nv), j in zip(per_pair_records, jit, strict=True):
        ax2.plot([i - 0.14 + j, i + 0.14 + j], [sv, nv], color="lightgrey", linewidth=0.6, zorder=1)
        ax2.scatter(i - 0.14 + j, sv, s=14, color=c_st, zorder=2)
        ax2.scatter(i + 0.14 + j, nv, s=14, color=c_nu, zorder=2)
    ax2.axhline(0.0, color="grey", linewidth=0.8)
    ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([c[6] for c in CELLS], fontsize=7)
    ax2.set_ylabel("per-pair F_beh (well-separated pairs)")
    ax2.set_title("per-pair view: steered (left dot) vs null (right dot)", loc="left", fontsize=9)

    fig.suptitle(
        "F_beh at the four largest stage-1 cells collapses on pairs whose anchors actually "
        "separate (|ceiling-floor| >= 0.5)",
        fontsize=10,
    )
    savefig_paper(fig, "issue_2094/result_sep_stratified_fbeh", dir="figures/")
    plt.close(fig)
    print("saved figures/issue_2094/result_sep_stratified_fbeh.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
