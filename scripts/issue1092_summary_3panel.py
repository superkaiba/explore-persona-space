#!/usr/bin/env python3
"""Issue #1092 summary: 3-panel monitoring figure {PV rig, persona corpus, LMSYS}.

Rows = traits; columns = the three eval setups. Within each cell, bars are the
monitoring METHOD — raw projection ``<v_C, r_B>``, map+projection
``<h(v_C), r_B>`` (best n1m map), and the direct in-corpus supervised
context->trait probe ceiling — with 95% CIs. Metric = Pearson r (within-condition
on the PV-rig panel; group-CV / group-cluster-bootstrap elsewhere). Sources:

  - PV rig       : eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_readout.json
                   (L26 cells overlaid from l26-kernel-gate-recovery/l26_recovery_readout.json)
  - persona corp : eval_results/issue_1092/pooled-probe-transfer/map_on_persona_reads.json
                   (map family + raw; Deliverable A) + pooled_probe_transfer.json ceilings
  - LMSYS        : pooled_probe_transfer.json L_lmsys_ctx ceiling (label-flat, r~=0.009)

Any cell whose source datum is absent (the pooled run is partial) is drawn as an
explicit "N/A (pending)" marker, never a misleading zero bar. Read-only; 0 GPU.
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

# thread caps bind in-process BEFORE numpy/matplotlib import on the shared VM (#847).
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
BEST_MAP = "n1m_mlp_w32768"  # the strongest map family member (nonlinear MLP)
PERSONA_LAYER = "L14"  # both my reads and the pooled ceilings are at L14
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1092"

N1M = PROJECT_ROOT / "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_readout.json"
L26 = PROJECT_ROOT / "eval_results/issue_779/l26-kernel-gate-recovery/l26_recovery_readout.json"
MAPP = PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/map_on_persona_reads.json"
POOLED = PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/pooled_probe_transfer.json"

# method label -> role color
METHODS = [
    ("raw projection", "baseline"),
    ("map + projection", "primary"),
    ("direct in-corpus probe", "accent"),
]


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _val(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def pv_rig_cell(n1m: dict, l26: dict, pooled: dict, trait: str) -> dict:
    """PV-rig (system-mode) within-condition r for the 3 methods.

    raw = pv_raw; map+proj = best n1m dot (L26 cells overlaid from recovery);
    direct probe = the pooled A_passa_ctx within-corpus supervised context->trait
    CV ceiling (same supervised-probe family used in the persona/LMSYS panels, so
    the third method is matched across all panels — the oracle answer-projection is
    a different quantity and is not used here). The A_passa ceiling is only present
    in the (partial) pooled JSON for hallucination-L14; else N/A (pending).
    """
    cell = _val(n1m, "headline", trait, "system")
    if cell is None:
        return {}
    layer = cell.get("layer")
    src = cell
    # overlay the L26-recovery re-read where the system cell is at L26.
    if layer == 26 and _val(l26, "headline", trait, "system"):
        src = l26["headline"][trait]["system"]
    mons = src["monitors"]

    def rd(name):
        m = mons.get(name)
        return None if m is None else {"r": m["point"], "lo": m["lo"], "hi": m["hi"]}

    out = {
        "layer": layer,
        "raw projection": rd("pv_raw"),
        "map + projection": rd(f"{BEST_MAP}_dot"),
        "n": src.get("n_eval_rows"),
    }
    wc = _val(pooled, "substrates", trait, PERSONA_LAYER, "A_passa_ctx", "within_ceiling")
    cv = _val(wc, "cv_r") if isinstance(wc, dict) else None
    if cv is not None:
        ci = wc.get("ci_r") or [None, None]
        out["direct in-corpus probe"] = {"r": cv, "lo": ci[0], "hi": ci[1]}
    return out


def persona_cell(mapp: dict, pooled: dict, trait: str) -> dict:
    """Persona-corpus (L14) r for the 3 methods."""
    mons = _val(mapp, "reads", trait, PERSONA_LAYER, "monitors")
    out: dict = {"layer": 14}
    if mons:
        for label, key in (("raw projection", "pv_raw"), ("map + projection", f"{BEST_MAP}_dot")):
            m = mons.get(key)
            if m is not None:
                lo, hi = m["ci_r"]
                out[label] = {"r": m["r"], "lo": lo, "hi": hi}
        out["n"] = _val(mons, "pv_raw", "n")
    # direct probe = pooled within-corpus P_persona_ctx ceiling (may be pending).
    wc = _val(pooled, "substrates", trait, PERSONA_LAYER, "P_persona_ctx", "within_ceiling")
    cv = _val(wc, "cv_r") if isinstance(wc, dict) else None
    if cv is not None:
        ci = wc.get("ci_r") or [None, None]
        out["direct in-corpus probe"] = {"r": cv, "lo": ci[0], "hi": ci[1]}
    return out


def lmsys_cell(pooled: dict, trait: str) -> dict:
    """LMSYS (L14) direct-probe ceiling (label-flat)."""
    wc = _val(pooled, "substrates", trait, PERSONA_LAYER, "L_lmsys_ctx", "within_ceiling")
    cv = _val(wc, "cv_r") if isinstance(wc, dict) else None
    out: dict = {"layer": 14}
    if cv is not None:
        ci = wc.get("ci_r") or [None, None]
        out["direct in-corpus probe"] = {"r": cv, "lo": ci[0], "hi": ci[1]}
        out["n"] = _val(pooled, "substrates", trait, PERSONA_LAYER, "L_lmsys_ctx", "n")
    return out


def _draw(ax, cell: dict, panel: str, ceiling_line: float | None = None) -> None:
    x = np.arange(len(METHODS))
    colors = [paper_palette_role(role) for _lbl, role in METHODS]
    heights, errlo, errhi, na = [], [], [], []
    for label, _role in METHODS:
        m = cell.get(label)
        if m is None or m.get("r") is None:
            heights.append(0.0)
            errlo.append(0.0)
            errhi.append(0.0)
            na.append(True)
            continue
        r = float(m["r"])
        heights.append(r)
        lo = m.get("lo")
        hi = m.get("hi")
        errlo.append(max(0.0, r - lo) if lo is not None and np.isfinite(lo) else 0.0)
        errhi.append(max(0.0, hi - r) if hi is not None and np.isfinite(hi) else 0.0)
        na.append(False)
    bars = ax.bar(
        x,
        heights,
        yerr=[errlo, errhi],
        capsize=3,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    for xi, (b, is_na) in enumerate(zip(bars, na)):
        if is_na:
            b.set_alpha(0.12)
            ax.text(
                xi,
                0.02,
                "N/A\n(pending)",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#888",
                rotation=0,
            )
    if ceiling_line is not None:
        ax.axhline(
            ceiling_line,
            color="#b03060",
            lw=1.0,
            ls="--",
            label=f"LMSYS ceiling r={ceiling_line:.3f}",
        )
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    ax.axhline(0.0, color="#888", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace(" ", "\n") for lbl, _ in METHODS], fontsize=6.5)
    ax.set_ylim(-0.2, 1.0)
    n = cell.get("n")
    layer = cell.get("layer")
    sub = f"L{layer}" + (f", n={n}" if n else "")
    ax.set_title(f"{panel}\n({sub})", fontsize=8)


def main() -> int:
    set_paper_style("blog")
    n1m, l26, mapp, pooled = _load(N1M), _load(L26), _load(MAPP), _load(POOLED)
    if not mapp:
        raise SystemExit(f"missing Deliverable A JSON: {MAPP}")

    lmsys_ceiling = _val(
        pooled,
        "substrates",
        "hallucination",
        PERSONA_LAYER,
        "L_lmsys_ctx",
        "within_ceiling",
        "cv_r",
    )

    nrow, ncol = len(TRAITS), 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.3 * nrow), squeeze=False)
    meta_cells: dict = {}
    for ri, trait in enumerate(TRAITS):
        pv = pv_rig_cell(n1m, l26, pooled, trait)
        pc = persona_cell(mapp, pooled, trait)
        lm = lmsys_cell(pooled, trait)
        meta_cells[trait] = {"pv_rig": pv, "persona_corpus": pc, "lmsys": lm}
        _draw(axes[ri][0], pv, "PV rig (#779)")
        _draw(axes[ri][1], pc, "Persona corpus (#1092)")
        _draw(axes[ri][2], lm, "LMSYS (real-user)", ceiling_line=lmsys_ceiling)
        axes[ri][0].set_ylabel(f"{trait}\nPearson r", fontsize=9)

    fig.suptitle(
        "Behavior monitoring: raw projection vs map+projection vs direct in-corpus probe",
        fontsize=11,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Metric: Pearson r (PV rig = within-condition; persona/LMSYS = group-CV / cluster-boot, "
        f"L{PERSONA_LAYER[1:]}). map+projection = <h(v_C), r_B>, best map {BEST_MAP}. "
        "raw = <v_C, r_B>. Pending cells: pooled probe run partial (hallucination-L14 complete).",
        ha="center",
        fontsize=6.5,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out = savefig_paper(fig, "summary_3panel_monitoring", dir=str(FIG_DIR))
    plt.close(fig)

    meta = {
        "figure": "summary_3panel_monitoring",
        "panels": ["PV rig (#779)", "Persona corpus (#1092)", "LMSYS (real-user)"],
        "methods": [m for m, _ in METHODS],
        "traits": list(TRAITS),
        "persona_layer": PERSONA_LAYER,
        "best_map": BEST_MAP,
        "lmsys_ceiling_r": lmsys_ceiling,
        "sources": {
            "pv_rig": str(N1M.relative_to(PROJECT_ROOT)),
            "pv_rig_l26_overlay": str(L26.relative_to(PROJECT_ROOT)),
            "persona_map_reads": str(MAPP.relative_to(PROJECT_ROOT)),
            "pooled_ceilings": str(POOLED.relative_to(PROJECT_ROOT)),
        },
        "pending_note": (
            "pooled_probe_transfer.json is PARTIAL (run PID 583418 died at layer14/sycophancy); "
            "direct-in-corpus-probe bars exist only for hallucination-L14; all other direct-probe "
            "cells drawn as N/A (pending) — regenerate when the pooled run completes."
        ),
        "cell_values": meta_cells,
        "outputs": {k: str(v) for k, v in out.items()},
    }
    # savefig_paper already wrote the canonical <stem>.meta.json (commit-pinned);
    # keep it and write the richer cell-value provenance to a sidecar.
    (FIG_DIR / "summary_3panel_monitoring.cells.json").write_text(json.dumps(meta, indent=2))
    print(f"[3panel] wrote {out}")
    print(f"[3panel] cells -> {FIG_DIR / 'summary_3panel_monitoring.cells.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
