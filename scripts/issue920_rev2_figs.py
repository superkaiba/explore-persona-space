"""Analyzer round-2 artifacts for issue #920.

1. Persists the auditable family-restricted band/gap table (observed family
   maxima + per-draw family maxima + gaps + 97.5% bands, R1/R2, both sides)
   plus the H1 band-vs-ceiling block to
   eval_results/issue_920/family_restricted_bands.json.
2. New figure `family_gap_vs_band`: H1 paired band vs achievable ceiling
   (zero-power visualization) + per-answer-family gap-vs-band separation.
3. Regenerates `chain_vs_oracle_gap` under the REGISTERED max-|rho| cell
   selection (the round-1 figure used signed argmax — wrong cells for 6/7
   behaviors) with plain-English behavior labels.
4. Regenerates `per_layer_top5_pairs` + `r3_identity_ceiling` with
   plain-English labels.

Run from the issue-920 worktree root:
    uv run python scripts/issue920_rev2_figs.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from issue920_labels import plain_behavior, plain_family

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

set_paper_style("blog")
import matplotlib.pyplot as plt  # noqa: E402  (after style)

OUT = "figures/issue_920"
EVAL = "eval_results/issue_920"
NULL_PT = "data/issue_920/null_matrices/dv1_null_skills.pt"

mp = json.loads(pathlib.Path(f"{EVAL}/map_skill_by_cell.json").read_text())
nb = json.loads(pathlib.Path(f"{EVAL}/null_bands_and_headline.json").read_text())
cc = mp["cells"]["c_cell"]
ac = mp["cells"]["a_cell"]
cfam = np.array([c.split("@")[0] for c in cc])
afam = np.array([a.split("@")[0] for a in ac])
skill = {r: np.array(mp["skill"][r]) for r in ("R1", "R2")}

null = torch.load(NULL_PT, map_location="cpu", weights_only=False)
regime_idx = {r: i for i, r in enumerate(null["regimes"])}

# ---- 1. family-restricted band/gap table --------------------------------------
table: dict = {
    "definition": (
        "Per family F (context or answer side) and regime R: observed_family_max = max "
        "held-out map skill over F's cells; observed_gap = observed_max_all - "
        "observed_family_max; per draw d of the 1000-draw permutation null (seed 920, "
        "dv1_null_skills.pt), gap_d = max_all_d - max_F_d; band_p97_5 = 97.5th percentile "
        "of gap_d; separable = observed_gap > band_p97_5. Per-draw family maxima are "
        "persisted (rounded to 4 dp) so the table is auditable without the .pt matrix."
    ),
    "h1_band_vs_ceiling": {},
    "per_regime": {},
}
for r in ("R1", "R2"):
    obs = skill[r]
    Sn = null["skills"][:, :, regime_idx[r]].numpy().astype(np.float64)
    max_all_d = Sn.max(axis=0)
    obs_max_all = float(np.nanmax(obs))
    inc_mask = np.isin(afam, nb[f"h1_delta_{r}"]["mean_family_cells"])
    inc_max = float(np.nanmax(np.where(inc_mask, obs, -np.inf)))
    table["h1_band_vs_ceiling"][r] = {
        "observed_delta": nb[f"h1_delta_{r}"]["observed_delta"],
        "band_p97_5": nb[f"h1_delta_{r}"]["band_p97_5"],
        "incumbent_max_skill": round(inc_max, 6),
        "achievable_delta_ceiling": round(1.0 - inc_max, 6),
        "band_exceeds_ceiling": nb[f"h1_delta_{r}"]["band_p97_5"] > (1.0 - inc_max),
        "note": "band > ceiling => the paired test has zero power; non-rejection is "
        "failure-to-reject, not evidence of a tie",
    }
    per_side: dict = {}
    for side, fams in (("context_families", cfam), ("answer_families", afam)):
        rows = {}
        for f in sorted(set(fams)):
            m = fams == f
            fam_max_d = Sn[m].max(axis=0)
            gap_d = max_all_d - fam_max_d
            gap = obs_max_all - float(np.nanmax(obs[m]))
            band = float(np.percentile(gap_d, 97.5))
            rows[f] = {
                "observed_family_max": round(float(np.nanmax(obs[m])), 6),
                "observed_gap": round(gap, 6),
                "band_p97_5": round(band, 6),
                "separable": bool(gap > band),
                "per_draw_family_max": [round(float(v), 4) for v in fam_max_d],
            }
        per_side[side] = rows
    per_side["observed_max_all"] = obs_max_all
    per_side["n_draws"] = int(null["n_draws"])
    table["per_regime"][r] = per_side
table["reproducibility"] = mp["reproducibility"]
with open(f"{EVAL}/family_restricted_bands.json", "w") as fh:
    json.dump(table, fh)
print("wrote family_restricted_bands.json")

# ---- 2. figure: H1 band-vs-ceiling + per-family gap-vs-band -------------------
fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 7), width_ratios=[1, 2.4])
xs = np.arange(2)
for i, r in enumerate(("R1", "R2")):
    h = table["h1_band_vs_ceiling"][r]
    axl.bar(i, h["band_p97_5"], width=0.55, color="#c9d4e8", zorder=1)
    axl.hlines(h["achievable_delta_ceiling"], i - 0.32, i + 0.32, color="#c23b22", zorder=3)
    axl.scatter([i], [h["observed_delta"]], color="#1f4e9c", zorder=4, s=45)
    axl.text(i, h["band_p97_5"] + 0.004, f"band {h['band_p97_5']:.2f}", ha="center", fontsize=8)
    axl.text(
        i + 0.34,
        h["achievable_delta_ceiling"],
        f"ceiling {h['achievable_delta_ceiling']:.2f}",
        va="center",
        fontsize=8,
        color="#c23b22",
    )
    axl.text(
        i + 0.06,
        h["observed_delta"],
        f"observed {h['observed_delta']:.3f}",
        fontsize=8,
        va="center",
    )
axl.set_xticks(xs, ["in-probe", "held-out probes"])
axl.set_ylabel("best-challenger lead over incumbent (skill units)")
axl.set_title("Incumbent-vs-challenger test:\nselection band exceeds the achievable ceiling")
# right: R1 answer-family gaps vs bands
rows = table["per_regime"]["R1"]["answer_families"]
fams = sorted(rows, key=lambda f: rows[f]["observed_gap"])
ys = np.arange(len(fams))
gaps = [rows[f]["observed_gap"] for f in fams]
bands = [rows[f]["band_p97_5"] for f in fams]
colors = ["#c23b22" if rows[f]["separable"] else "#1f4e9c" for f in fams]
axr.barh(ys, bands, color="#c9d4e8", height=0.72, zorder=1, label="97.5% selection-noise band")
axr.scatter(gaps, ys, color=colors, s=26, zorder=3)
axr.scatter([], [], color="#1f4e9c", s=26, label="observed gap (inside band)")
axr.scatter([], [], color="#c23b22", s=26, label="observed gap (separably worse)")
axr.set_yticks(ys, [plain_family(f) for f in fams], fontsize=6.5)
axr.set_xlabel("gap to the sweep winner (skill units), in-probe")
axr.set_title("Answer families: gap to the winner vs family-restricted band")
axr.legend(loc="lower right", fontsize=8)
savefig_paper(fig, "family_gap_vs_band", dir=OUT)
plt.close(fig)
print("wrote family_gap_vs_band")

# ---- 3. chain_vs_oracle_gap under the registered max-|rho| selection ----------
ch = json.loads(pathlib.Path(f"{EVAL}/chain_rho_by_cell.json").read_text())
behs = ch["behaviors"]
R9 = np.array(ch["rho_R9"], dtype=float)
orho = np.array(ch["oracle_in_pca_basis_rho"]["rho"], dtype=float)
a_idx = {name: i for i, name in enumerate(ch["oracle_in_pca_basis_rho"]["cells"])}
chain_abs, oracle_abs = [], []
for bi in range(len(behs)):
    i = int(np.nanargmax(np.abs(R9[:, bi])))
    chain_abs.append(abs(R9[i, bi]))
    oracle_abs.append(abs(orho[a_idx[ch["cells"]["a_cell"][i]], bi]))
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(behs))
ax.bar(
    x - 0.19, chain_abs, width=0.38, color="#1f4e9c", label="chained read |rho| (best chain cell)"
)
ax.bar(x + 0.19, oracle_abs, width=0.38, color="#c9a227", label="oracle read |rho| (same cell)")
ax.set_xticks(x, [plain_behavior(b) for b in behs], rotation=20, ha="right")
ax.set_ylabel("absolute Spearman rho vs graded behavior score")
ax.set_title(
    "Chained read-out exceeds its own oracle at every behavior's best chain cell\n"
    "(max-absolute-rho cell selection, in-probe; n = 50 contexts)"
)
ax.legend()
savefig_paper(fig, "chain_vs_oracle_gap", dir=OUT)
plt.close(fig)
print("wrote chain_vs_oracle_gap (abs-based)")

# ---- 4a. per-layer top-5 pairs, plain-English legend ---------------------------
R1 = skill["R1"]
pair_best: dict[tuple[str, str], float] = {}
for i in range(len(cc)):
    key = (cfam[i], afam[i])
    if R1[i] > pair_best.get(key, -np.inf):
        pair_best[key] = R1[i]
top5 = sorted(pair_best, key=lambda k: -pair_best[k])[:5]
fig, ax = plt.subplots(figsize=(9, 5.5))
for cf, af in top5:
    per_layer: dict[int, float] = {}
    for i in range(len(cc)):
        if cfam[i] == cf and afam[i] == af and "@" in cc[i]:
            layer = int(cc[i].split("@L")[1])
            per_layer[layer] = max(per_layer.get(layer, -np.inf), R1[i])
    ls = sorted(per_layer)
    ax.plot(
        ls,
        [per_layer[layer] for layer in ls],
        marker="o",
        ms=3,
        label=f"{plain_family(cf)} → {plain_family(af)}",
    )
ax.axhline(
    nb["bands"]["dv1_R1"]["band_p97_5"], color="gray", linestyle="--", label="chance band (0.17)"
)
ax.set_xlabel("layer (matched context/answer layer)")
ax.set_ylabel("held-out map skill (in-probe)")
ax.set_title("Per-layer skill for the top-5 family pairs")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
savefig_paper(fig, "per_layer_top5_pairs", dir=OUT)
plt.close(fig)
print("wrote per_layer_top5_pairs")

# ---- 4b. identity ceiling per answer family, plain labels ----------------------
fam_ceil: dict[str, float] = {}
for cell, v in mp["ceiling_ya_yb_per_a_cell"].items():
    f = cell.split("@")[0]
    fam_ceil[f] = max(fam_ceil.get(f, -np.inf), v)
fams = sorted(fam_ceil, key=lambda f: -fam_ceil[f])
fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(range(len(fams)), [fam_ceil[f] for f in fams], color="#1f4e9c")
ax.axhline(0.2, color="#c23b22", linestyle="--", label="target-stability gate (0.2)")
ax.set_xticks(range(len(fams)), [plain_family(f) for f in fams], rotation=90, fontsize=7)
ax.set_ylabel("cross-probe-set target stability (identity skill)")
ax.set_title("Target-stability ceiling per answer family (pool-A targets predicting pool-B)")
ax.legend()
savefig_paper(fig, "r3_identity_ceiling", dir=OUT)
plt.close(fig)
print("wrote r3_identity_ceiling")
