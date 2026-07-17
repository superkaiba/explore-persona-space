#!/usr/bin/env python
"""#1434 figures (plan §6 HERO pair; exploratory dumps ride the persisted
matrices via the analyzer's /paper-plots pass).

1. ``install_grid.png`` — per-context x per-lr selected-rung Tier-2 judged
   rates with Wilson 95% CIs, base bars alongside, the 0.60-0.85 band shaded.
2. ``rb_validity_scatter.png`` — judged delta vs r_B projection at the
   read-out layer (leakage grid), honest randnorm band + the |rho|<=1 ceiling
   annotated in the caption sidecar.

Errorbars are NON-NEGATIVE OFFSETS (np.maximum(0, .) element-wise — the
#547/#1335 xerr/yerr rule); tiny-n inverted quantile CIs clamp instead of
crashing savefig.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1434_cells as cells  # noqa: E402

PLAIN = {
    "ws-pers": "Persona (software engineer)",
    "ws-bare": "Bare assistant",
    "ws-conv": "WildChat prefix",
    "ws-icl": "ICL prefix",
}
LR_LABEL = {"lr1e5": "lr 1e-5", "lr3e5": "lr 3e-5", "lr1e4": "lr 1e-4"}


def _err(rate: float, ci: list[float]) -> tuple[float, float]:
    """Non-negative (lo, hi) errorbar OFFSETS from a Wilson CI (#547/#1335)."""
    lo, hi = ci
    return (float(np.maximum(0.0, rate - lo)), float(np.maximum(0.0, hi - rate)))


def fig_install_grid(agg: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.axhspan(*agg["band"], color="tab:green", alpha=0.12, label="target band 0.60-0.85")
    xticks, xlabels = [], []
    x = 0.0
    for cell_key in cells.CELL_KEYS:
        entry = agg["tier2"].get(cell_key) or {}
        base = entry.get("base")
        group_x = []
        if base and base.get("rate") is not None:
            lo, hi = _err(base["rate"], base["wilson_95"])
            ax.bar(x, base["rate"], width=0.8, color="0.7", label=None)
            ax.errorbar(x, base["rate"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3)
            xticks.append(x)
            xlabels.append(f"{PLAIN[cell_key]}\nbase")
            group_x.append(x)
            x += 1.0
        for run in cells.I1434_RUNS:
            if run.cell_key != cell_key:
                continue
            tag = run.run_id.rsplit("-", 1)[-1]
            ladder = agg["ladders"].get(run.run_id) or {}
            rec = None
            va = (entry.get("verdict_arm") or {}).get("run_id")
            if va == run.run_id and entry.get("trained"):
                rec = entry["trained"]
            if rec and rec.get("rate") is not None:
                lo, hi = _err(rec["rate"], rec["wilson_95"])
                ax.bar(x, rec["rate"], width=0.8, color="tab:blue")
                ax.errorbar(x, rec["rate"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3)
            elif ladder.get("selection"):
                # non-verdict arms: Tier-1 selected-rung rate (no fresh Tier-2)
                ax.bar(x, float(ladder["selection"]["rate"]), width=0.8, color="tab:cyan")
            xticks.append(x)
            xlabels.append(f"{PLAIN[cell_key]}\n{LR_LABEL.get(tag, tag)}")
            group_x.append(x)
            x += 1.0
        x += 0.8
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("judged casual-register rate (pv rubric)")
    ax.set_ylim(0, 1)
    ax.set_title(
        "writing_style install grid — Tier-2 verdict-arm rates (blue), Tier-1 "
        "selected-rung rates (cyan), base (grey); Wilson 95% CIs"
    )
    out = out_dir / "install_grid.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_rb_validity(val: dict, proj: dict, agg: dict, out_dir: Path) -> Path:
    grid = (val.get("grids") or {}).get("leakage") or {}
    arm = grid.get("response_shared") or {}
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    if arm.get("observed_abs_rho_per_layer"):
        layer = int(arm["max_layer"])
        xs, ys, labels = [], [], []
        for _cell_key, prec in (agg.get("panel") or {}).items():
            rid = prec["run_id"]
            pr = proj["states"].get(rid)
            if pr is None:
                continue
            for ctx_id, row in prec["contexts"].items():
                xs.append(float(pr["response_shared"]["projection"][layer]))
                ys.append(float(row["delta"]))
                labels.append(f"{rid}@{ctx_id}")
        ax.scatter(xs, ys, s=18, color="tab:blue")
        for xi, yi, lab in zip(xs, ys, labels, strict=True):
            ax.annotate(lab, (xi, yi), fontsize=4, alpha=0.6)
        band = (arm.get("null_within_class") or {}).get("max_selected_p97_5")
        title = f"r_B validity — leakage grid, layer {layer}: |rho|={arm['max_abs_rho']:.3f}" + (
            f"; honest randnorm max-selected p97.5={band:.3f}; ceiling |rho|<=1" if band else ""
        )
    else:
        ax.text(0.5, 0.5, f"insufficient cells: {grid}", ha="center", va="center")
        title = "r_B validity — insufficient cells"
    ax.set_xlabel("shift . r_B-hat (response-shared arm)")
    ax.set_ylabel("judged rate delta (trained - base)")
    ax.set_title(title, fontsize=9)
    out = out_dir / "rb_validity_scatter.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="#1434 hero figures")
    p.add_argument("--deliverables", default=str(cells.DELIVERABLES_DIR_1434))
    p.add_argument("--projections", default="data/issue_1434/cells/pv/projections.json")
    p.add_argument("--out-dir", default=str(cells.FIGURES_DIR_1434))
    args = p.parse_args(argv)
    deliver = Path(args.deliverables)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = json.loads((deliver / "i1434_ladders.json").read_text())
    print(fig_install_grid(agg, out_dir))
    val_path = deliver / "pv_validation.json"
    proj_path = Path(args.projections)
    if val_path.exists() and proj_path.exists():
        val = json.loads(val_path.read_text())
        proj = json.loads(proj_path.read_text())
        print(fig_rb_validity(val, proj, agg, out_dir))
    else:
        print(f"[figures] pv validation/projections missing ({val_path}, {proj_path}) — skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
