"""Issue #526 figure: per-behavior off-diagonal variance decomposition + L1 R2 annotation."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
except Exception:
    plt.rcParams.update(
        {"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

R = json.load(open("figures/issue_526/gate_ladder_results.json"))

# ---- assemble per-behavior decomposition of off-diagonal variance ----
# symmetric share = 1 - L0_antisym_fraction
# of the antisym share: scalar-explained = L0 * L2_scalar_frac ; residual = L0 * (1-L2_scalar_frac)
rows = []
for beh in ["marker", "fact", "refusal", "sycophancy", "em"]:
    r = R["537"][beh]
    af = r["L0_antisym_fraction"]
    scal = r["L2_scalar_antisym_fraction"]
    sym = 1 - af
    scal_anti = af * scal
    resid_anti = af * (1 - scal)
    l1r2 = r["L1"].get("r2")
    rows.append(
        dict(
            label={"em": "EM", "fact": "taught\nfact"}.get(beh, beh),
            sym=sym,
            scal_anti=scal_anti,
            resid_anti=resid_anti,
            l1r2=l1r2,
            flat=(beh == "marker"),
        )
    )

labels = [x["label"] for x in rows]
sym = np.array([x["sym"] for x in rows])
scal_anti = np.array([x["scal_anti"] for x in rows])
resid_anti = np.array([x["resid_anti"] for x in rows])

fig, ax = plt.subplots(figsize=(8.2, 5.0))
x = np.arange(len(rows))
w = 0.62
c_sym = "#4C72B0"
c_scal = "#55A868"
c_res = "#C44E52"
ax.bar(x, sym, w, label="symmetric (predictable by a symmetric g)", color=c_sym)
ax.bar(
    x,
    scal_anti,
    w,
    bottom=sym,
    label="antisym captured by per-unit scalars (b_i, r_j)",
    color=c_scal,
)
ax.bar(
    x,
    resid_anti,
    w,
    bottom=sym + scal_anti,
    label="antisym needing full pairwise g(C,C')",
    color=c_res,
)

# annotate L1 baseline-diff R2 above each bar
for i, r in enumerate(rows):
    if r["flat"]:
        txt = "L1: flat prior\n(untestable)"
    else:
        txt = f"L1 R²={r['l1r2']:.2f}"
    ax.text(i, 1.02, txt, ha="center", va="bottom", fontsize=8.5)

# annotate the antisym total bracket
for i in range(len(rows)):
    af = scal_anti[i] + resid_anti[i]
    ax.text(
        i,
        sym[i] + af + 0.005,
        f"antisym {af * 100:.0f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("share of off-diagonal transfer variance")
ax.set_ylim(0, 1.16)
ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax.set_title(
    "How fancy must the leakage predictor g be? (#537 context-transfer, 16×16/behavior)\n"
    "L1 = baseline-difference theory term (signed antisym vs base-prior difference)",
    fontsize=10.5,
)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False, fontsize=8.5)

out = "figures/issue_526/asym_gate_ladder.png"
fig.savefig(out, bbox_inches="tight", dpi=160)
print("saved", out)

# also write a tiny meta sidecar
meta = {
    "source_data": [
        "eval_results/issue_537/G_tensor/G_meta.json",
        "eval_results/issue_537/analysis/registered_reads.json",
        "eval_results/issue_545/L_matrix.json",
        "eval_results/issue_545/base_panel.json",
    ],
    "script": "scripts/issue526_asym_gate_ladder.py + scripts/issue526_asym_gate_ladder_plot.py",
    "rows": rows,
}
json.dump(meta, open("figures/issue_526/asym_gate_ladder.meta.json", "w"), indent=1)
print("saved figures/issue_526/asym_gate_ladder.meta.json")
