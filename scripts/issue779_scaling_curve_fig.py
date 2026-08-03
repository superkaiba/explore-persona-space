"""Context->answer map: held-out R2 vs number of training contexts, 250 -> 963,444.

Two measurement rounds are stitched at layer 19 (marked distinctly, see caption):
  - #779 D2 fair-fitter comparison  : n = 250 .. 3,600   (exact fits, 3 draws/n)
  - #779 n1M round                  : n = 50,000 (exact KRR gate slice) and
                                      n = 963,444 (mixed_1m ridge / KRR-Nystrom / MLP)
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): env caps must bind BEFORE any heavy import.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
SMALL = ROOT / "eval_results/issue_779/fitter-fair-comparison/scaling_curves.json"
BIG = ROOT / "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_multilayer_fits.json"
OUT = ROOT / "figures/issue_779/map_scaling_r2_vs_n"

D_MODEL = 3584
C = {"ridge": "#0173B2", "krr": "#029E73", "mlp": "#D55E00"}

small = json.loads(SMALL.read_text())["curves"]["last_L19"]
by = defaultdict(list)
for r in small:
    by[(r["fitter"], r["n"])].append(r["r2"])
ns_small = sorted({r["n"] for r in small})

big = json.loads(BIG.read_text())["per_layer"]["19"]
pt = big["per_point"]["mixed_1m"]
n_big = pt["selection"]["n_realized"]
preds = pt["predictors"]
n50k_krr = big["nystrom_validation"]["exact_r2"]

fig, ax = plt.subplots(figsize=(9.2, 5.2))

for f, lab in (("ridge", "ridge (linear)"), ("krr", "KRR (kernel)"), ("mlp", "MLP")):
    xs = list(ns_small)
    ys = [st.mean(by[(f, n)]) for n in ns_small]
    ax.plot(xs, ys, marker="o", ms=5, color=C[f], linewidth=1.8, label=lab)

# n1M round points
ax.plot([n_big], [preds["ridge"]["whole_map_r2"]], marker="D", ms=9, color=C["ridge"])
ax.plot([n_big], [preds["krr_nystrom"]["whole_map_r2"]], marker="D", ms=9, color=C["krr"])
ax.plot([n_big], [preds["mlp_w32768"]["whole_map_r2"]], marker="D", ms=9, color=C["mlp"])
ax.plot([50000], [n50k_krr], marker="D", ms=9, color=C["krr"], alpha=0.75)

for f, key in (("ridge", "ridge"), ("krr", "krr_nystrom"), ("mlp", "mlp_w32768")):
    ax.plot(
        [ns_small[-1], n_big],
        [st.mean(by[(f, ns_small[-1])]), preds[key]["whole_map_r2"]],
        color=C[f],
        linewidth=1.1,
        linestyle=":",
        alpha=0.75,
    )
ax.plot(
    [ns_small[-1], 50000, n_big],
    [st.mean(by[("krr", ns_small[-1])]), n50k_krr, preds["krr_nystrom"]["whole_map_r2"]],
    color=C["krr"],
    linewidth=1.6,
    linestyle="--",
    alpha=0.9,
)

ax.axvline(D_MODEL, color="#888888", linestyle="--", linewidth=1.1)
ax.text(D_MODEL * 0.93, 0.495, f"$n=d$={D_MODEL}", fontsize=8, color="#555", ha="right")
ax.axvspan(50000, n_big, color="#029E73", alpha=0.07)
ax.text(
    2.0e5, 0.60, "plateau region\n50k → 963k: KRR +0.013", fontsize=9, ha="center", color="#1a6b50"
)

ax.set_xscale("log")
ax.set_xlabel("number of training contexts (n, log scale)")
ax.set_ylabel("held-out $R^2$  (layer 19, pinned 1,000-context test split)")
ax.set_title(
    "Context→answer map scaling: plateaus by ~50k, and nonlinearity only pays at scale",
    fontsize=12,
)
ax.grid(alpha=0.3, linewidth=0.6)
ax.legend(frameon=False, fontsize=9, loc="lower right")

fig.text(
    0.5,
    0.075,
    "Circles = #779 D2 fair-fitter comparison (exact fits, 3 draws/n). Diamonds = #779 n1M round "
    "(mixed_1m; 50k point is the exact-KRR Nyström gate slice).",
    ha="center",
    fontsize=7.5,
)
fig.text(
    0.5,
    0.040,
    "Rounds differ in fitter implementation and λ grid (D2 λ≤1e3 and railed at that max; n1M λ grid "
    "1e-3…1e8, selected λ=1e-3 at the low edge), so the",
    ha="center",
    fontsize=7.5,
)
fig.text(
    0.5,
    0.008,
    "3,600→50,000 segment mixes rounds and is indicative, not a controlled within-round contrast.",
    ha="center",
    fontsize=7.5,
)
fig.tight_layout(rect=(0, 0.105, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=160)
fig.savefig(f"{OUT}.pdf")
print("wrote", f"{OUT}.png")
