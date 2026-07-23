#!/usr/bin/env python3
"""Issue #1092 summary: 3-panel monitoring figure {PV rig, persona corpus, LMSYS}
+ a direct-map-probe transfer row.

TOP BLOCK (rows = traits, cols = the three eval setups): within-condition /
within-corpus Pearson r, bars grouped by monitoring METHOD —

  raw projection  <v_C, r_B>
  map+projection  <h(v_C), r_B>, SPLIT per map (5k-linear / n1m ridge=1M-linear /
                  n1m mlp_w32768=1M-nonlinear / n1m krr), colored by training setup
  v_C probe       direct in-corpus supervised context->trait CV ceiling
  direct-map probe  supervised probe fit on the MAP OUTPUT h(v_C) (ridge-h, mlp-h),
                  WITHIN-CORPUS — DPI-capped by the v_C probe for the linear map

BOTTOM ROW (the scientific point): pooled/LODO HELD-OUT transfer r — does probing
in ANSWER space (h(v_C)) transfer to a held-out corpus better than probing CONTEXT
space (v_C)? v_C transfer from the pooled JSON pooled_lodo; direct-map transfer
from direct_map_probe_reads.json.

Any cell whose source datum is absent (partial pooled / partial direct-map run) is
an explicit "N/A (pending)" marker, never a misleading zero bar. Read-only; 0 GPU.

Sources:
  PV rig       : eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_readout.json
                 (L26 cells overlaid from l26-kernel-gate-recovery/l26_recovery_readout.json)
  persona corp : eval_results/issue_1092/pooled-probe-transfer/map_on_persona_reads.json
  ceilings/LODO: pooled_probe_transfer.json (within_ceiling + pooled_lodo, the v_C arm)
  direct-map   : direct_map_probe_reads.json (the h(v_C) probe arm)
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
PERSONA_LAYER = "L14"  # both the map reads and the pooled ceilings are at L14
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1092"

N1M = PROJECT_ROOT / "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_readout.json"
L26 = PROJECT_ROOT / "eval_results/issue_779/l26-kernel-gate-recovery/l26_recovery_readout.json"
MAPP = PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/map_on_persona_reads.json"
POOLED = PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/pooled_probe_transfer.json"
DMP = PROJECT_ROOT / "eval_results/issue_1092/pooled-probe-transfer/direct_map_probe_reads.json"

# map arm slug -> (bar label, "training setup" color role). The 4 maps are the
# split of the former single "map+projection" bar; each is colored by the setup
# that produced it (raw keeps its own baseline color).
MAP_ARMS = [
    ("h_n5k_linear", "5k-linear", "primary"),
    ("n1m_ridge", "1M ridge", "accent"),
    ("n1m_mlp_w32768", "1M mlp", "control"),
    ("n1m_krr_nystrom", "1M krr", "neutral"),
]
# top-block bar spec: (label, color-role, kind). kind drives the value lookup.
TOP_BARS = (
    [("raw", "baseline", ("raw",))]
    + [(lbl, role, ("map", slug)) for slug, lbl, role in MAP_ARMS]
    + [
        ("v_C probe", "accent", ("vc_probe",)),
        ("dmp:ridge", "primary", ("dmp", "n1m_ridge")),
        ("dmp:mlp", "control", ("dmp", "n1m_mlp_w32768")),
    ]
)


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _val(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _rd(point, lo, hi):
    if point is None:
        return None
    return {"r": float(point), "lo": lo, "hi": hi}


def pv_rig_values(n1m: dict, l26: dict, pooled: dict, trait: str) -> dict:
    """PV-rig (system-mode) within-condition r keyed by TOP_BARS kind."""
    cell = _val(n1m, "headline", trait, "system")
    if cell is None:
        return {}
    layer = cell.get("layer")
    src = cell
    if layer == 26 and _val(l26, "headline", trait, "system"):
        src = l26["headline"][trait]["system"]
    mons = src["monitors"]

    def mon(name):
        m = mons.get(name)
        return _rd(m["point"], m["lo"], m["hi"]) if m else None

    vals = {("raw",): mon("pv_raw")}
    for slug, _lbl, _role in MAP_ARMS:
        vals[("map", slug)] = mon(f"{slug}_dot")
    wc = _val(pooled, "substrates", trait, PERSONA_LAYER, "A_passa_ctx", "within_ceiling")
    cv = _val(wc, "cv_r") if isinstance(wc, dict) else None
    ci = (wc.get("ci_r") if isinstance(wc, dict) else None) or [None, None]
    vals[("vc_probe",)] = _rd(cv, ci[0], ci[1]) if cv is not None else None
    # direct-map probe is a persona/LMSYS-corpus construct; not defined on the PV rig.
    vals[("dmp", "n1m_ridge")] = None
    vals[("dmp", "n1m_mlp_w32768")] = None
    return {"layer": layer, "n": src.get("n_eval_rows"), "vals": vals}


def persona_values(mapp: dict, pooled: dict, dmp: dict, trait: str) -> dict:
    """Persona-corpus (L14) within-corpus r keyed by TOP_BARS kind."""
    mons = _val(mapp, "reads", trait, PERSONA_LAYER, "monitors") or {}
    vals: dict = {}
    m = mons.get("pv_raw")
    vals[("raw",)] = _rd(m["r"], *m["ci_r"]) if m else None
    for slug, _lbl, _role in MAP_ARMS:
        m = mons.get(f"{slug}_dot")
        vals[("map", slug)] = _rd(m["r"], *m["ci_r"]) if m else None
    wc = _val(pooled, "substrates", trait, PERSONA_LAYER, "P_persona_ctx", "within_ceiling")
    cv = _val(wc, "cv_r") if isinstance(wc, dict) else None
    ci = (wc.get("ci_r") if isinstance(wc, dict) else None) or [None, None]
    vals[("vc_probe",)] = _rd(cv, ci[0], ci[1]) if cv is not None else None
    for slug in ("n1m_ridge", "n1m_mlp_w32768"):
        w = _val(dmp, "reads", slug, trait, PERSONA_LAYER, "within_corpus", "P_persona_ctx")
        cvd = _val(w, "cv_r") if isinstance(w, dict) else None
        cid = (w.get("ci_r") if isinstance(w, dict) else None) or [None, None]
        vals[("dmp", slug)] = _rd(cvd, cid[0], cid[1]) if cvd is not None else None
    return {"layer": 14, "n": _val(mons, "pv_raw", "n"), "vals": vals}


def lmsys_values(pooled: dict, dmp: dict, trait: str) -> dict:
    """LMSYS (L14) within-corpus r — only the v_C probe + direct-map probes exist
    (the maps were fit ON lmsys, so applying them here is in-distribution, not the
    transfer read the panel is about); label-flat ceiling ~0.009."""
    vals: dict = {("raw",): None}
    for slug, _lbl, _role in MAP_ARMS:
        vals[("map", slug)] = None
    wc = _val(pooled, "substrates", trait, PERSONA_LAYER, "L_lmsys_ctx", "within_ceiling")
    cv = _val(wc, "cv_r") if isinstance(wc, dict) else None
    ci = (wc.get("ci_r") if isinstance(wc, dict) else None) or [None, None]
    vals[("vc_probe",)] = _rd(cv, ci[0], ci[1]) if cv is not None else None
    for slug in ("n1m_ridge", "n1m_mlp_w32768"):
        w = _val(dmp, "reads", slug, trait, PERSONA_LAYER, "within_corpus", "L_lmsys_ctx")
        cvd = _val(w, "cv_r") if isinstance(w, dict) else None
        cid = (w.get("ci_r") if isinstance(w, dict) else None) or [None, None]
        vals[("dmp", slug)] = _rd(cvd, cid[0], cid[1]) if cvd is not None else None
    return {"layer": 14, "n": _val(wc, "n") if isinstance(wc, dict) else None, "vals": vals}


def _draw_top(ax, cell: dict, panel: str, ceiling_line: float | None = None) -> None:
    vals = cell.get("vals", {})
    x = np.arange(len(TOP_BARS))
    heights, errlo, errhi, na, colors = [], [], [], [], []
    for _lbl, role, kind in TOP_BARS:
        colors.append(paper_palette_role(role))
        m = vals.get(kind)
        if m is None or m.get("r") is None:
            heights.append(0.0)
            errlo.append(0.0)
            errhi.append(0.0)
            na.append(True)
            continue
        r = float(m["r"])
        heights.append(r)
        lo, hi = m.get("lo"), m.get("hi")
        errlo.append(max(0.0, r - lo) if lo is not None and np.isfinite(lo) else 0.0)
        errhi.append(max(0.0, hi - r) if hi is not None and np.isfinite(hi) else 0.0)
        na.append(False)
    bars = ax.bar(
        x, heights, yerr=[errlo, errhi], capsize=2, color=colors, edgecolor="white", linewidth=0.5
    )
    for xi, (b, is_na) in enumerate(zip(bars, na)):
        if is_na:
            b.set_alpha(0.1)
            ax.text(xi, 0.02, "N/A", ha="center", va="bottom", fontsize=5.5, color="#999")
    if ceiling_line is not None:
        ax.axhline(
            ceiling_line,
            color="#b03060",
            lw=1.0,
            ls="--",
            label=f"LMSYS ceiling {ceiling_line:.3f}",
        )
        ax.legend(fontsize=5.5, loc="upper right", frameon=False)
    ax.axhline(0.0, color="#888", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _ in TOP_BARS], fontsize=5.5, rotation=45, ha="right")
    ax.set_ylim(-0.2, 1.0)
    n, layer = cell.get("n"), cell.get("layer")
    ax.set_title(f"{panel} (L{layer}" + (f", n={n})" if n else ")"), fontsize=8)


# ── bottom row: LODO held-out transfer (v_C vs h(v_C)) ──


def transfer_values(pooled: dict, dmp: dict, trait: str, held: str) -> dict:
    """Held-out transfer r for held-out substrate `held`: v_C (pooled_lodo) vs
    direct-map ridge-h / mlp-h."""
    out: dict = {}
    v = _val(pooled, "transfer", trait, PERSONA_LAYER, held, "pooled_lodo", "read")
    out["v_C"] = _rd(v["r"], *v["ci_r"]) if isinstance(v, dict) and v.get("r") is not None else None
    for slug, lbl in (("n1m_ridge", "h:ridge"), ("n1m_mlp_w32768", "h:mlp")):
        d = _val(dmp, "reads", slug, trait, PERSONA_LAYER, "pooled_lodo", held, "held_out_read")
        out[lbl] = (
            _rd(d["r"], *d["ci_r"]) if isinstance(d, dict) and d.get("r") is not None else None
        )
    return out


def _draw_transfer(ax, pooled: dict, dmp: dict, trait: str) -> None:
    held_order = [
        ("P_persona_ctx", "→persona"),
        ("A_passa_ctx", "→PV-rig"),
        ("L_lmsys_ctx", "→LMSYS"),
    ]
    series = [("v_C", "baseline"), ("h:ridge", "primary"), ("h:mlp", "control")]
    x = np.arange(len(held_order))
    w = 0.26
    any_data = False
    for si, (skey, role) in enumerate(series):
        heights, errlo, errhi = [], [], []
        for held, _lbl in held_order:
            tv = transfer_values(pooled, dmp, trait, held).get(skey)
            if tv is None or tv.get("r") is None:
                heights.append(0.0)
                errlo.append(0.0)
                errhi.append(0.0)
                continue
            any_data = True
            r = float(tv["r"])
            heights.append(r)
            lo, hi = tv.get("lo"), tv.get("hi")
            errlo.append(max(0.0, r - lo) if lo is not None and np.isfinite(lo) else 0.0)
            errhi.append(max(0.0, hi - r) if hi is not None and np.isfinite(hi) else 0.0)
        ax.bar(
            x + (si - 1) * w,
            heights,
            w,
            yerr=[errlo, errhi],
            capsize=2,
            color=paper_palette_role(role),
            edgecolor="white",
            linewidth=0.5,
            label={"v_C": "v_C probe", "h:ridge": "h(v_C) ridge", "h:mlp": "h(v_C) mlp"}[skey],
        )
    ax.axhline(0.0, color="#888", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _h, lbl in held_order], fontsize=7)
    ax.set_ylim(-0.2, 1.0)
    ax.set_title(f"{trait} — LODO held-out transfer", fontsize=8)
    if not any_data:
        ax.text(
            0.5,
            0.5,
            "N/A (pending)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="#999",
        )


def main() -> int:
    set_paper_style("blog")
    n1m, l26, mapp, pooled, dmp = (_load(p) for p in (N1M, L26, MAPP, POOLED, DMP))
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

    ntrait = len(TRAITS)
    fig, axes = plt.subplots(ntrait + 1, 3, figsize=(6.0 * 3, 3.3 * (ntrait + 1)), squeeze=False)
    meta_cells: dict = {}
    for ri, trait in enumerate(TRAITS):
        pv = pv_rig_values(n1m, l26, pooled, trait)
        pc = persona_values(mapp, pooled, dmp, trait)
        lm = lmsys_values(pooled, dmp, trait)

        # stringify the tuple-keyed `vals` for JSON.
        def _sk(cell: dict) -> dict:
            out = {k: v for k, v in cell.items() if k != "vals"}
            out["vals"] = {"|".join(kk): vv for kk, vv in cell.get("vals", {}).items()}
            return out

        meta_cells[trait] = {"pv_rig": _sk(pv), "persona_corpus": _sk(pc), "lmsys": _sk(lm)}
        _draw_top(axes[ri][0], pv, "PV rig (#779)")
        _draw_top(axes[ri][1], pc, "Persona corpus (#1092)")
        _draw_top(axes[ri][2], lm, "LMSYS (real-user)", ceiling_line=lmsys_ceiling)
        axes[ri][0].set_ylabel(f"{trait}\nwithin-corpus Pearson r", fontsize=9)

    # bottom row: LODO transfer, one subplot per trait (evil col unused if absent).
    for ci, trait in enumerate(("hallucination", "sycophancy", "evil")):
        _draw_transfer(axes[ntrait][ci], pooled, dmp, trait)
    axes[ntrait][0].set_ylabel("held-out\nPearson r", fontsize=9)
    handles, labels = axes[ntrait][0].get_legend_handles_labels()
    if handles:
        axes[ntrait][2].legend(handles, labels, fontsize=6, loc="upper right", frameon=False)

    fig.suptitle(
        "Behavior monitoring: raw vs split map+projection vs v_C probe vs direct-map probe "
        "(top) + LODO transfer (bottom)",
        fontsize=11,
        y=0.997,
    )
    fig.text(
        0.5,
        0.004,
        "Top: within-condition/within-corpus Pearson r (map+projection SPLIT per map, colored by "
        f"training setup; L{PERSONA_LAYER[1:]}). Bottom: pooled/LODO held-out transfer r — does "
        "probing answer-space h(v_C) transfer better than context-space v_C? Direct-map within-"
        "corpus is DPI-capped by the v_C probe for the linear map. N/A = partial pooled / "
        "direct-map run (regenerate when complete).",
        ha="center",
        fontsize=6.0,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out = savefig_paper(fig, "summary_3panel_monitoring", dir=str(FIG_DIR))
    plt.close(fig)

    meta = {
        "figure": "summary_3panel_monitoring",
        "layout": "rows=traits + 1 transfer row; cols=[PV rig, persona corpus, LMSYS]",
        "top_bars": [lbl for lbl, _, _ in TOP_BARS],
        "map_split": [lbl for _s, lbl, _r in MAP_ARMS],
        "traits": list(TRAITS),
        "persona_layer": PERSONA_LAYER,
        "lmsys_ceiling_r": lmsys_ceiling,
        "sources": {
            "pv_rig": str(N1M.relative_to(PROJECT_ROOT)),
            "pv_rig_l26_overlay": str(L26.relative_to(PROJECT_ROOT)),
            "persona_map_reads": str(MAPP.relative_to(PROJECT_ROOT)),
            "pooled_ceilings_and_lodo": str(POOLED.relative_to(PROJECT_ROOT)),
            "direct_map_probe": str(DMP.relative_to(PROJECT_ROOT)),
        },
        "extensions": {
            "1_split_map_bars": "the former single map+projection bar is split into "
            "5k-linear / 1M ridge / 1M mlp / 1M krr, colored by training setup",
            "2_direct_map_probe": "supervised probe on h(v_C): within-corpus bars (top, "
            "DPI-capped by v_C for the linear map) + LODO held-out transfer (bottom row)",
        },
        "pending_note": (
            "pooled_probe_transfer.json and direct_map_probe_reads.json may be PARTIAL; "
            "absent cells drawn N/A. Regenerate when both complete."
        ),
        "cell_values": meta_cells,
        "outputs": {k: str(v) for k, v in out.items()},
    }
    (FIG_DIR / "summary_3panel_monitoring.cells.json").write_text(json.dumps(meta, indent=2))
    print(f"[3panel] wrote {out}")
    print(f"[3panel] cells -> {FIG_DIR / 'summary_3panel_monitoring.cells.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
