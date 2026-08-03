#!/usr/bin/env python3
"""Issue #1092 fair-recompare figure: within-corpus vs battery-held-out, per arm.

Per trait (rows), per monitoring arm (x): the EXISTING within-corpus Pearson r
(the leaky number — battery rows were in the banked within-corpus train+CV pool)
vs the BATTERY-HELD-OUT novel-question r (train on non-battery, eval on the 2,400
battery rows whose questions are disjoint from the query bank). The double-held-out
variant COINCIDES with novel-question here (battery prefixes are 0-overlap with the
non-battery training prefixes), so the prefix-familiarity Δ is structurally 0 and
the within→novel-Q drop isolates query-familiarity inflation.

within-corpus sources: map arms from map_on_persona_reads.json (persona L14);
v_C probe from pooled_probe_transfer.json within_ceiling; h-ridge/h-mlp probes from
direct_map_probe_reads.json within_corpus. battery from battery_heldout_recompare.json.
Read-only; 0 GPU.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

TRAITS = ("hallucination", "sycophancy", "evil")
LK = "L14"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1092"
BASE = PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer"
MAPP = BASE / "map_on_persona_reads.json"
POOLED = BASE / "pooled_probe_transfer.json"
DMP = BASE / "direct_map_probe_reads.json"
RECOMP = BASE / "battery_heldout_recompare.json"

# (arm label, within-corpus lookup spec, battery lookup spec, color role)
# within spec: ("map", slug) | ("vc",) | ("dmp", slug); battery spec: ("mon", key) | ("probe", key)
ARMS = [
    ("raw", ("map", "pv_raw"), ("mon", "raw"), "baseline"),
    ("5k-linear", ("map", "h_n5k_linear_dot"), ("mon", "h_n5k_linear"), "primary"),
    ("1M ridge", ("map", "n1m_ridge_dot"), ("mon", "n1m_ridge"), "accent"),
    ("1M mlp", ("map", "n1m_mlp_w32768_dot"), ("mon", "n1m_mlp_w32768"), "control"),
    ("1M krr", ("map", "n1m_krr_nystrom_dot"), ("mon", "n1m_krr_nystrom"), "neutral"),
    ("v_C probe", ("vc",), ("probe", "v_C"), "accent"),
    ("h-ridge probe", ("dmp", "n1m_ridge"), ("probe", "h_ridge"), "primary"),
    ("h-mlp probe", ("dmp", "n1m_mlp_w32768"), ("probe", "h_mlp"), "control"),
]


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _val(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _rd(r, lo, hi):
    return None if r is None else {"r": float(r), "lo": lo, "hi": hi}


def within_val(spec, trait, mapp, pooled, dmp):
    kind = spec[0]
    if kind == "map":
        m = _val(mapp, "reads", trait, LK, "monitors", spec[1])
        return _rd(m["r"], *m["ci_r"]) if m else None
    if kind == "vc":
        wc = _val(pooled, "substrates", trait, LK, "P_persona_ctx", "within_ceiling")
        cv = _val(wc, "cv_r") if isinstance(wc, dict) else None
        ci = (wc.get("ci_r") if isinstance(wc, dict) else None) or [None, None]
        return _rd(cv, ci[0], ci[1]) if cv is not None else None
    w = _val(dmp, "reads", spec[1], trait, LK, "within_corpus", "P_persona_ctx")
    cv = _val(w, "cv_r") if isinstance(w, dict) else None
    ci = (w.get("ci_r") if isinstance(w, dict) else None) or [None, None]
    return _rd(cv, ci[0], ci[1]) if cv is not None else None


def battery_val(spec, trait, recomp):
    cell = _val(recomp, "reads", trait, LK)
    if cell is None:
        return None
    if spec[0] == "mon":
        m = _val(cell, "monitoring_reads", spec[1])
    else:
        m = _val(cell, "refit_probes", spec[1], "secondary_novelq")
    return (
        _rd(m["r"], m.get("ci_r", [None, None])[0], m.get("ci_r", [None, None])[1]) if m else None
    )


def _err(m):
    if m is None or m.get("r") is None:
        return 0.0, 0.0, True
    r, lo, hi = m["r"], m.get("lo"), m.get("hi")
    el = max(0.0, r - lo) if lo is not None and np.isfinite(lo) else 0.0
    eh = max(0.0, hi - r) if hi is not None and np.isfinite(hi) else 0.0
    return el, eh, False


def main() -> int:
    set_paper_style("blog")
    mapp, pooled, dmp, recomp = (_load(p) for p in (MAPP, POOLED, DMP, RECOMP))
    if not recomp:
        raise SystemExit(f"missing recompare JSON: {RECOMP}")

    fig, axes = plt.subplots(len(TRAITS), 1, figsize=(11.0, 3.4 * len(TRAITS)), squeeze=False)
    x = np.arange(len(ARMS))
    w = 0.4
    c_within = paper_palette_role("neutral")
    c_bat = paper_palette_role("primary")
    for ri, trait in enumerate(TRAITS):
        ax = axes[ri][0]
        cell = _val(recomp, "reads", trait, LK) or {}
        flat = bool(cell.get("label_flat"))
        for xi, (lbl, wspec, bspec, _role) in enumerate(ARMS):
            wv = within_val(wspec, trait, mapp, pooled, dmp)
            bv = battery_val(bspec, trait, recomp)
            for m, off, color, tag in ((wv, -w / 2, c_within, "within"), (bv, w / 2, c_bat, "bat")):
                el, eh, na = _err(m)
                h = 0.0 if na else float(m["r"])
                ax.bar(
                    xi + off,
                    h,
                    w,
                    yerr=[[el], [eh]],
                    capsize=2,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.25 if na else 1.0,
                    label=(
                        "within-corpus (existing)"
                        if tag == "within"
                        else "battery held-out (novel-Q)"
                    )
                    if xi == 0
                    else None,
                )
                if na:
                    ax.text(
                        xi + off, 0.02, "N/A", ha="center", va="bottom", fontsize=5, color="#999"
                    )
        ax.axhline(0.0, color="#888", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([a[0] for a in ARMS], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Pearson r", fontsize=9)
        ax.set_ylim(-0.25, 1.0)
        n_eval = cell.get("n_eval")
        flat_tag = "  [LABEL-FLAT: battery scores ~0 variance → uninformative]" if flat else ""
        ax.set_title(
            f"{trait} — within-corpus vs battery-held-out (L14, battery n={n_eval}){flat_tag}",
            fontsize=9,
        )
        if ri == 0:
            ax.legend(fontsize=7, loc="upper left", frameon=False)

    fig.suptitle(
        "Fair recompare: existing within-corpus vs battery-held-out (novel-question) Pearson r",
        fontsize=11,
        y=0.998,
    )
    fig.text(
        0.5,
        0.005,
        "Battery = 2,400 fixed eval rows, questions disjoint from the query bank; battery prefixes "
        "0-overlap with the non-battery training prefixes, so double-held-out COINCIDES with "
        "novel-question (prefix-familiarity Δ = 0). within→battery drop = query-familiarity "
        "inflation. Map+r_B arms are matmul re-scores (no refit); probes are refit on the "
        "non-battery pool. Banked within-corpus leaks (battery rows were in its train+CV pool).",
        ha="center",
        fontsize=6.0,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out = savefig_paper(fig, "battery_heldout_recompare", dir=str(FIG_DIR))
    plt.close(fig)

    # sidecar table
    table: dict = {"layer": LK, "arms": [a[0] for a in ARMS], "traits": {}}
    for trait in TRAITS:
        cell = _val(recomp, "reads", trait, LK) or {}
        rows = {}
        for lbl, wspec, bspec, _role in ARMS:
            wv = within_val(wspec, trait, mapp, pooled, dmp)
            bv = battery_val(bspec, trait, recomp)
            rows[lbl] = {
                "within_corpus_r": None if wv is None else round(wv["r"], 4),
                "battery_novelq_r": None if bv is None else round(bv["r"], 4),
                "query_familiarity_delta": (round(wv["r"] - bv["r"], 4) if (wv and bv) else None),
            }
        table["traits"][trait] = {
            "label_flat": bool(cell.get("label_flat")),
            "n_eval": cell.get("n_eval"),
            "prefix_overlap": cell.get("prefix_overlap_battery_vs_nonbattery"),
            "variants_coincide": cell.get("variants_coincide"),
            "arms": rows,
        }
    (FIG_DIR / "battery_heldout_recompare.table.json").write_text(json.dumps(table, indent=2))
    print(f"[battery-fig] wrote {out}")
    print(f"[battery-fig] table -> {FIG_DIR / 'battery_heldout_recompare.table.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
