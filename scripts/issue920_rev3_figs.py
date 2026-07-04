"""Analyzer revision-2 figure regeneration for issue #920 (Lens 3 fixes).

Regenerates TWO figures with reader-facing text:

1. ``winning_cell_scatter`` — plain-English title (was raw slugs
   "ctx_blk_max@L12 x ans_uhdr_max@L12") + readable per-context point labels;
   now saved via ``savefig_paper`` (PNG + PDF + meta.json).
2. ``dv2_harmful_compliance_cell_scatter`` — readable per-context point labels
   (was raw ids like ``f5_fmt_markdown_table``).

Both sidecar meta.json files gain a ``context_id_to_label`` mapping (and the
winning cell's raw slug names) for traceability back to the raw ids.

Run from the issue-920 worktree root:
    OMP_NUM_THREADS=8 uv run python scripts/issue920_rev3_figs.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from issue920_labels import plain_context_id, plain_family
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

set_paper_style("blog")
import matplotlib.pyplot as plt  # noqa: E402  (after style)

OUT = "figures/issue_920"
EVAL = "eval_results/issue_920"
PRED_PT = "data/issue_920/preds/pooled_heldout_predictions.pt"


def _add_meta(stem: str, extra: dict) -> None:
    """Merge ``extra`` into the figure's savefig_paper sidecar meta.json."""
    p = pathlib.Path(OUT) / f"{stem}.meta.json"
    meta = json.loads(p.read_text())
    meta.update(extra)
    p.write_text(json.dumps(meta))


blob = torch.load(PRED_PT, map_location="cpu", weights_only=False)
names_c, names_a = blob["ctx_cell_names"], blob["ans_cell_names"]
c_map, a_map = blob["c_map"].numpy(), blob["a_map"].numpy()
ctx_ids = list(blob["ctx_ids"])
id2label = {c: plain_context_id(c) for c in ctx_ids}
labels = [id2label[c] for c in ctx_ids]

mp = json.loads(pathlib.Path(f"{EVAL}/map_skill_by_cell.json").read_text())
R1 = np.array([np.nan if v is None else v for v in mp["skill"]["R1"]], dtype=np.float64)

# ---- 1. winning-cell per-context scatter (plain-English title + labels) --------
win = int(np.nanargmax(R1))
c_cell, a_cell = names_c[c_map[win]], names_a[a_map[win]]
assert c_cell == "ctx_blk_max@L12" and a_cell == "ans_uhdr_max@L12", (c_cell, a_cell)
predA = blob["map_predA"][win].float().numpy()
ytrue = blob["ypca_A"][a_map[win]].float().numpy()
fig, ax = plt.subplots(figsize=(8.5, 8))
ax.scatter(ytrue[:, 0], predA[:, 0], s=16, color="#1f4e9c")
for i, lab in enumerate(labels):
    ax.annotate(
        lab,
        (ytrue[i, 0], predA[i, 0]),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=5,
        alpha=0.75,
    )
ax.set_xlabel("true target (leading fold-basis PCA dimension)")
ax.set_ylabel("held-out prediction")
ax.set_title("Winning map cell: context template-block max → answer user-header pool max, layer 12")
savefig_paper(fig, "winning_cell_scatter", dir=OUT)
plt.close(fig)
_add_meta(
    "winning_cell_scatter",
    {
        "context_id_to_label": id2label,
        "cell_slugs": {"c_cell": c_cell, "a_cell": a_cell},
        "cell_plain": {
            "c_cell": plain_family(c_cell.split("@")[0]),
            "a_cell": plain_family(a_cell.split("@")[0]),
        },
    },
)
print("wrote winning_cell_scatter (readable title + labels)")

# ---- 2. DV-2 harmful-compliance per-context scatter (readable labels) ----------
ro = json.loads(pathlib.Path(f"{EVAL}/readout_rho_by_cell.json").read_text())
cells = ro["cells"]
beh = ro["behaviors"]
rin = np.array(ro["rho"]["R_in_probe"])
bat = json.loads(pathlib.Path("data/issue594/battery.json").read_text())
items = bat if isinstance(bat, list) else bat.get("instances")
fammap = {(x.get("context_id") or x.get("id")): x["family"] for x in items}
fvec = np.array([fammap[c] for c in ctx_ids])
g = json.loads(pathlib.Path("../../../eval_results/issue_812/graded_e0_highm.json").read_text())[
    "e0"
]
y = np.array([g["harmful_compliance"][c]["graded_mean"] for c in ctx_ids])

bi = beh.index("harmful_compliance")
ans_idx = np.where(np.array([not c.startswith("ctx") for c in cells]))[0]
j = ans_idx[np.argmax(np.abs(rin[ans_idx, bi]))]
assert cells[j] == "ans_uhdr_max_pool@L0" or "uhdr" in cells[j], cells[j]
pred = blob["ro_predA"].numpy()[j, :, bi]
r_full = rin[j, bi]


def center(v: np.ndarray) -> np.ndarray:
    """Subtract each battery family's mean from ``v`` (within-family centering)."""
    out = v.astype(float).copy()
    for f in set(fvec):
        out[fvec == f] -= out[fvec == f].mean()
    return out


r_cent = spearmanr(center(pred), center(y)).statistic
print(f"cell={cells[j]} r_full={r_full:.3f} r_centered={r_cent:.3f}")

fig, ax = plt.subplots(figsize=(8.5, 7.5))
palette = dict(zip(sorted(set(fvec)), plt.cm.tab10.colors, strict=False))
for f in sorted(set(fvec)):
    m = fvec == f
    ax.scatter(pred[m], y[m], color=palette[f], label=f, s=36)
for i, lab in enumerate(labels):
    ax.annotate(
        lab,
        (pred[i], y[i]),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=5,
        alpha=0.75,
    )
ax.set_xlabel("pooled held-out prediction (user-header max pool target, layer 0)")
ax.set_ylabel("graded harmful-compliance score (0-100)")
ax.set_title(
    "Per-context data behind the harmful-compliance read-out clear:\n"
    "family clusters carry the correlation"
)
ax.legend(fontsize=7)
savefig_paper(fig, "dv2_harmful_compliance_cell_scatter", dir=OUT)
plt.close(fig)
_add_meta(
    "dv2_harmful_compliance_cell_scatter",
    {"context_id_to_label": id2label, "cell_slug": cells[j]},
)
print("wrote dv2_harmful_compliance_cell_scatter (readable labels)")
