"""Per-unit companion figures for issue #1333 (clean-result-critic r1, Lens 11)
+ the plain-English retitle of the cross-surface gap figure (procedural fix).

Run from the issue-1333 worktree root. Companions to the round-1 figures in
/tmp/issue-1333-plots.py: (A) per-probe ||dx|| distributions at layer 25 behind
the mean-shift-norm points (plan §6 "per-probe Δx norm distributions"); (B)
per-probe leakage ΔG dots behind the breadth per-context mean bars.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

WT = Path(__file__).resolve().parents[1]
FIGDIR = WT / "figures" / "issue_1333"
FIGDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WT / "scripts"))
import issue1333_geometry as G  # noqa: E402

GEO = json.load(open(WT / "eval_results/issue_1333/geometry/geometry_marker_2x2.json"))
CELLS = ["mk1_lora_con", "mk2_lora_pos", "mk3_fullft_con", "mk4_fullft_pos"]
NICE = {
    "mk1_lora_con": "LoRA\n+ negatives",
    "mk2_lora_pos": "LoRA\npositives-only",
    "mk3_fullft_con": "Full FT\n+ negatives\n(reused)",
    "mk4_fullft_pos": "Full FT\npositives-only",
}
STORE_PATH = {
    "mk1_lora_con": "mk1_lora_con/selected/pooled.pt",
    "mk2_lora_pos": "mk2_lora_pos/selected/pooled.pt",
    "mk3_fullft_con": "m2_fullft_band8/selected/pooled.pt",
    "mk4_fullft_pos": "mk4_fullft_pos/selected/pooled.pt",
}

set_paper_style("blog")
PAL = paper_palette_blog(4)
CELLC = dict(zip(CELLS, PAL))
RNG = np.random.default_rng(42)

# ── Fig A: per-probe ||dx|| strips at layer 25 behind the ||mu|| points ───────
BASE = G._load_store(WT / "data/issue_1333/run/capture/base_marker/base/pooled.pt")
fig, ax = plt.subplots(figsize=(8.2, 4.6))
xs = np.arange(4)
for j, (kind, off, alpha) in enumerate((("own", -0.16, 0.75), ("tf", 0.16, 0.4))):
    for i, cell in enumerate(CELLS):
        tr = G._load_store(WT / "data/issue_1333/run/capture" / STORE_PATH[cell])
        store = (
            tr
            if kind == "own"
            else G._load_store(WT / "data/issue_1333/run/capture" / f"{cell}/tf_shared/pooled.pt")
        )
        delta = G._aligned_delta(store, BASE, "response", 25)
        norms = np.linalg.norm(delta, axis=1)
        jit = RNG.uniform(-0.055, 0.055, size=norms.shape[0])
        ax.scatter(
            np.full(norms.shape[0], i + off) + jit,
            norms,
            s=9,
            color=CELLC[cell],
            alpha=alpha,
            linewidths=0,
            zorder=2,
        )
        mu = float(GEO["cells"][cell][kind]["primary"]["mu_norm"])
        assert abs(np.linalg.norm(delta.mean(axis=0)) - mu) < 0.05, (cell, kind, mu)
        ax.plot([i + off], [mu], marker="D", ms=7, color="0.1", zorder=3)
        ax.text(i + off, mu * 0.72, f"{mu:.1f}", ha="center", fontsize=8.5, color="0.1")
ax.set_yscale("log")
ax.set_xticks(xs, [NICE[c] for c in CELLS], fontsize=8.5)
ax.set_ylabel("Per-row ‖Δx‖ at layer 25 (log scale)")
ax.set_title("Per-probe shift norms behind the layer-25 mean-shift points", pad=36)
ax.legend(
    handles=[
        mlines.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color="0.45",
            alpha=0.75,
            ms=5,
            label="per-row ‖Δx‖, own text (100 rows)",
        ),
        mlines.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color="0.45",
            alpha=0.35,
            ms=5,
            label="per-row ‖Δx‖, shared text (100 rows)",
        ),
        mlines.Line2D(
            [],
            [],
            marker="D",
            ls="",
            color="0.1",
            ms=7,
            label="‖μ‖ = norm of the mean shift (labeled)",
        ),
    ],
    fontsize=8.5,
    loc="upper right",
)
savefig_paper(fig, "munorm_perprobe_layer25", dir=FIGDIR)
plt.close(fig)

# ── Fig B: per-probe leakage dots behind the breadth per-context mean bars ────
BCELLS = ["mk1_lora_con", "ext_wildchat", "ext_icl", "ext_bare"]
BNICE = {
    "mk1_lora_con": "villain persona-trained (ΔG source +5.4)",
    "ext_wildchat": "WildChat-prefix-trained (+5.5)",
    "ext_icl": "demonstration-trained (+4.2)",
    "ext_bare": "bare default-context-trained (+7.3)",
}
TRAINED_NEG = {
    "mk1_lora_con": {"medical_doctor", "police_officer", "qwen_default", "comedian"},
    "ext_wildchat": {"medical_doctor", "police_officer", "qwen_default", "comedian"},
    "ext_icl": {"medical_doctor", "police_officer", "qwen_default", "comedian"},
    "ext_bare": {"medical_doctor", "police_officer", "french_person", "comedian"},
}
negc, heldc, srcc = "#c76a4a", "#7a9cc4", "#3f7d54"
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=False, sharey=True)
for ax, cell in zip(axes.flat, BCELLS):
    s = json.load(open(WT / f"data/issue_1333/run/breadth/{cell}/slot_reads.json"))
    per: dict[str, list[float]] = {}
    for p in s["per_probe"]:
        per.setdefault(p["row"]["label"], []).append(
            float(p["trained"]["logp"]) - float(p["base"]["logp"])
        )
    labels = ["__source__"] + [k for k in s["per_context"] if k != "__source__"]
    for x, lab in enumerate(labels):
        vals = np.asarray(per[lab])
        col = srcc if lab == "__source__" else (negc if lab in TRAINED_NEG[cell] else heldc)
        jit = RNG.uniform(-0.14, 0.14, size=vals.shape[0])
        ax.scatter(
            np.full(vals.shape[0], x) + jit,
            vals,
            s=11,
            color=col,
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )
        m = float(vals.mean())
        ax.plot([x - 0.24, x + 0.24], [m, m], color="0.1", lw=1.6, zorder=3)
        ax.text(x, m + 0.35, f"{m:+.1f}", ha="center", fontsize=8, color="0.1")
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xticks(
        np.arange(len(labels)),
        ["source" if l == "__source__" else l.replace("_", " ") for l in labels],
        rotation=40,
        ha="right",
        fontsize=8,
    )
    ax.set_title(BNICE[cell], fontsize=10.5)
axes[0, 0].set_ylabel("Δ log P(marker), trained − base (nats)")
axes[1, 0].set_ylabel("Δ log P(marker), trained − base (nats)")
import matplotlib.patches as mpatches  # noqa: E402

fig.legend(
    handles=[
        mpatches.Patch(color=srcc, label="training source context"),
        mpatches.Patch(color=negc, label="trained contrastive negative"),
        mpatches.Patch(color=heldc, label="held-out context"),
        mlines.Line2D([], [], color="0.1", lw=1.6, label="context mean (labeled)"),
    ],
    loc="lower center",
    ncol=4,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.005),
)
fig.text(
    0.01,
    0.99,
    "Per-probe leakage dots behind the per-context means (20 probes per context)",
    ha="left",
    va="top",
    fontsize=13,
    fontweight="semibold",
)
fig.subplots_adjust(top=0.90, hspace=0.62, bottom=0.16, left=0.06, right=0.98)
savefig_paper(fig, "breadth_leakage_per_probe", dir=FIGDIR)
plt.close(fig)

# ── Regen: cross-surface gap figure with plain-English panel titles ───────────
set_paper_style()
gap = GEO["re_reductions"]["cross_surface_gap"]
GNICE = {
    "mk1_lora_con": "villain contrastive",
    "mk2_lora_pos": "villain positives-only",
    "ext_wildchat": "WildChat prefix",
    "ext_icl": "demonstration",
    "ext_bare": "bare default",
}
cells = [c for c in GNICE if isinstance(gap.get(c), dict) and "points" in gap[c]]
pal = paper_palette(2)
fig, axes = plt.subplots(1, len(cells), figsize=(2.6 * len(cells), 2.8), sharey=True, squeeze=False)
for ax, cell in zip(axes[0], cells, strict=True):
    pts = gap[cell]["points"]
    il = [(p["step"], p["in_loop_delta"]) for p in pts if p["in_loop_delta"] is not None]
    ol = [(p["step"], p["off_line_delta"]) for p in pts if p["off_line_delta"] is not None]
    ax.plot(*zip(*il, strict=True), color=pal[0], linestyle="-", label="in-loop (TF)")
    ax.plot(
        *zip(*ol, strict=True),
        color=pal[1],
        linestyle="--",
        marker="o",
        markersize=3,
        label="off-line (on-policy)",
    )
    ax.set_title(GNICE[cell])
    ax.set_xlabel("optimizer step")
axes[0][0].set_ylabel("ΔG = Δ log P(marker) (nats)")
axes[0][0].legend(frameon=False, fontsize=7)
savefig_paper(fig, "explore_cross_surface_gap", dir=FIGDIR)
plt.close(fig)
meta_p = FIGDIR / "explore_cross_surface_gap.meta.json"
sidecar = json.loads(meta_p.read_text())
sidecar.setdefault("max_abs_gap_by_cell", {c: gap[c].get("max_abs_gap") for c in cells})
meta_p.write_text(json.dumps(sidecar, indent=1) + "\n")

print("companion figures written to", FIGDIR)
