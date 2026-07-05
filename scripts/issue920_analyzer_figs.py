"""Analyzer round-1 figures for issue #920.

Regenerates the two family heatmaps + the skill distribution with honest
(clipped/annotated) color/axis scales, and adds two NEW DV-2 figures carrying
the family-centered cross-check: a full-vs-centered rho summary and the
low-level per-context scatter behind the largest clearing cell.

Run from the issue-920 worktree root:
    uv run python scripts/issue920_analyzer_figs.py
"""

from __future__ import annotations

import json
import pathlib

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue920_labels import plain_family  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

set_paper_style("blog")
import matplotlib.pyplot as plt  # noqa: E402  (after style)

OUT = "figures/issue_920"
EVAL = "eval_results/issue_920"
PRED_PT = "data/issue_920/preds/pooled_heldout_predictions.pt"

mp = json.loads(pathlib.Path(f"{EVAL}/map_skill_by_cell.json").read_text())
cc = mp["cells"]["c_cell"]
ac = mp["cells"]["a_cell"]
R1 = np.array(mp["skill"]["R1"])
R2 = np.array(mp["skill"]["R2"])
cfam = np.array([c.split("@")[0] for c in cc])
afam = np.array([a.split("@")[0] for a in ac])
CF = sorted(set(cfam))
AF = sorted(set(afam))

PLAIN_C = {
    "ctx_wt_mean": "input mean (with template)",
    "ctx_wt_max": "input max (with template)",
    "ctx_co_mean": "input content mean",
    "ctx_co_max": "input content max",
    "ctx_ah_nl": "assistant-header newline",
    "ctx_tt_im_end": "trailing im_end",
    "ctx_tt_nl": "trailing newline",
    "ctx_tt_im_start": "trailing im_start",
    "ctx_tt_assistant": "trailing 'assistant'",
    "ctx_blk_mean": "template-block mean",
    "ctx_blk_max": "template-block max",
}


def famgrid(vals: np.ndarray) -> np.ndarray:
    g = np.full((len(CF), len(AF)), np.nan)
    for i, f in enumerate(CF):
        for j, h in enumerate(AF):
            m = (cfam == f) & (afam == h)
            if m.any():
                g[i, j] = vals[m].max()
    return g


# ---- Figure 1: R1 heatmap, clipped scale -------------------------------------
g1 = famgrid(R1)
fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(np.clip(g1, 0.0, 0.9), vmin=0.0, vmax=0.9, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(AF)), [plain_family(f) for f in AF], rotation=90, fontsize=6)
ax.set_yticks(range(len(CF)), [plain_family(f) for f in CF], fontsize=6)
ax.set_title(
    "Held-out map skill per family pair (best matched layer; color clipped to [0, 0.9];\n"
    "chance band 0.17; values below 0 render as 0)"
)
fig.colorbar(im, ax=ax, label="pooled-LOFO skill (R1)")
savefig_paper(fig, "hero_family_heatmap_R1_v2", dir=OUT)
plt.close(fig)

# ---- Figure 2: R2-R1 delta heatmap, clipped ----------------------------------
gd = famgrid_delta = np.full((len(CF), len(AF)), np.nan)
for i, f in enumerate(CF):
    for j, h in enumerate(AF):
        m = (cfam == f) & (afam == h)
        if m.any():
            k = np.where(m)[0][np.argmax(R1[m])]
            gd[i, j] = R2[k] - R1[k]
fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(np.clip(gd, -0.15, 0.15), vmin=-0.15, vmax=0.15, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(AF)), [plain_family(f) for f in AF], rotation=90, fontsize=6)
ax.set_yticks(range(len(CF)), [plain_family(f) for f in CF], fontsize=6)
ax.set_title(
    "Probe-set generalization delta (input-OOD minus in-probe skill) at each family pair's\n"
    "best cell; color clipped to plus/minus 0.15"
)
fig.colorbar(im, ax=ax, label="R2 - R1 at best R1 cell")
savefig_paper(fig, "hero_family_heatmap_R2_minus_R1_v2", dir=OUT)
plt.close(fig)

# ---- Figure 3: R1 distribution, clipped x ------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
below = int((R1 < -1).sum())
ax.hist(R1[R1 >= -1], bins=80, color="#1f4e9c")
ax.axvline(0.1704, color="gray", linestyle="--", label="chance band (max-inherited, 0.17)")
ax.set_xlabel("held-out map skill (R1), x-axis clipped at -1")
ax.set_ylabel("map cells")
ax.set_title(
    f"All 34,652 map cells: {int((R1 > 0.1704).sum()):,} clear the chance band; "
    f"{below:,} cells below -1 not shown"
)
ax.legend()
savefig_paper(fig, "r1_skill_distribution_v2", dir=OUT)
plt.close(fig)

# ---- DV-2 family-centered cross-check ----------------------------------------
p = torch.load(PRED_PT, map_location="cpu", weights_only=False)
ro = json.loads(pathlib.Path(f"{EVAL}/readout_rho_by_cell.json").read_text())
cells = ro["cells"]
beh = ro["behaviors"]
rin = np.array(ro["rho"]["R_in_probe"])
rood = np.array(ro["rho"]["R_input_ood"])
bat = json.loads(pathlib.Path("data/issue594/battery.json").read_text())
items = bat if isinstance(bat, list) else bat.get("instances")
fammap = {(x.get("context_id") or x.get("id")): x["family"] for x in items}
ctx_ids = p["ctx_ids"]
fvec = np.array([fammap[c] for c in ctx_ids])
e0 = {}
for src in ["highm", "lowm"]:
    g_path = pathlib.Path(f"../../../eval_results/issue_812/graded_e0_{src}.json")
    g = json.loads(g_path.read_text())["e0"]
    for b, per_ctx in g.items():
        if b == "deception":
            continue
        e0[b] = np.array([per_ctx[c]["graded_mean"] for c in ctx_ids])
CEIL = {  # sqrt(r_yy), issue 812 reliability file
    "sycophancy": 0.94,
    "refusal": 0.80,
    "harmful_compliance": 0.68,
    "fact_expression": 0.72,
    "format_style": 0.95,
    "self_report": 0.82,
    "persona_drift": 0.96,
}


def center(v: np.ndarray) -> np.ndarray:
    out = v.astype(float).copy()
    for f in set(fvec):
        out[fvec == f] -= out[fvec == f].mean()
    return out


rows = []  # (label, full, centered, behavior)
scatter_best = None
for regime, R, key in [("in-probe", rin, "ro_predA"), ("input-OOD", rood, "ro_predB")]:
    P = p[key].numpy()
    for side, sel in [
        ("context", np.array([c.startswith("ctx") for c in cells])),
        ("answer", np.array([not c.startswith("ctx") for c in cells])),
    ]:
        idx = np.where(sel)[0]
        for bi, b in enumerate(beh):
            j = idx[np.argmax(np.abs(R[idx, bi]))]
            pred = P[j, :, bi]
            y = e0[b]
            r_full = R[j, bi]
            r_cent = spearmanr(center(pred), center(y)).statistic
            rows.append((f"{b.replace('_', ' ')} ({side} read, {regime})", r_full, r_cent, b))
            if b == "harmful_compliance" and side == "answer" and regime == "in-probe":
                scatter_best = (cells[j], pred, y, r_full, r_cent)

fig, ax = plt.subplots(figsize=(9, 10))
ys = np.arange(len(rows))[::-1]
for yy, (_lab, rf, rc, _b) in zip(ys, rows, strict=True):
    ax.plot([abs(rf), abs(rc)], [yy, yy], color="lightgray", zorder=1)
    ax.scatter([abs(rf)], [yy], color="#1f4e9c", zorder=2, s=28)
    ax.scatter([abs(rc)], [yy], color="#c23b22", zorder=2, s=28)
ax.set_yticks(ys, [r[0] for r in rows], fontsize=7)  # labels from rows
ax.scatter([], [], color="#1f4e9c", label="full |rho| (as banded)")
ax.scatter([], [], color="#c23b22", label="within-family-centered |rho|")
ax.set_xlabel("absolute Spearman rho vs graded behavior score (n = 50 contexts, 7 families)")
ax.set_title(
    "Behavior read-out at each best cell (27 of 28 clear their band): full vs\n"
    "family-centered rho (judge-reliability ceilings 0.68-0.96 bound attainable rho)"
)
ax.axvline(0, color="black", lw=0.8)
ax.legend(loc="lower right")
savefig_paper(fig, "dv2_full_vs_centered_rho", dir=OUT)
plt.close(fig)

# ---- low-level scatter behind the harmful-compliance clearing cell -----------
cell, pred, y, rf, rc = scatter_best
fig, ax = plt.subplots(figsize=(8, 7))
palette = dict(zip(sorted(set(fvec)), plt.cm.tab10.colors, strict=False))
for f in sorted(set(fvec)):
    m = fvec == f
    ax.scatter(pred[m], y[m], color=palette[f], label=f, s=36)
for i, c in enumerate(ctx_ids):
    ax.text(pred[i], y[i], c, fontsize=5, alpha=0.7)
ax.set_xlabel("pooled held-out prediction (user-header max pool target, layer 0)")
ax.set_ylabel("graded harmful-compliance score (0-100)")
ax.set_title(
    "Per-context data behind the harmful-compliance read-out clear:\n"
    "family clusters carry the correlation"
)
ax.legend(fontsize=7)
savefig_paper(fig, "dv2_harmful_compliance_cell_scatter", dir=OUT)
plt.close(fig)

print("done")
