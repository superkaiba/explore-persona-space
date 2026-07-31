"""PV-suite vs real-rung comparison figure for the #1739 interim writeup.

Reads the committed pvsynth transfer rows + the main grid's op-slice arm rows
from the issue-1739 worktree and renders one grouped-bar figure (evil +
hallucination full comparison; sycophancy suite-only panel). Pure read +
render — no fits.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

set_paper_style()

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WT = ROOT / ".claude/worktrees/issue-1739/eval_results/issue_1739"
OUT = ROOT / "figures/issue_1739/interim_writeup"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = [
    "arm1_ctx_e1",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm11_oracle_proj",
    "arm13_shuffled_map",
]
ARM_LABEL = {
    "arm1_ctx_e1": "PV proj. on context\n(paper method)",
    "arm4_ridge_ctx": "direct ridge",
    "arm6_map_proj_e1": "map -> PV proj.",
    "arm11_oracle_proj": "oracle: PV on\nTRUE answer",
    "arm13_shuffled_map": "control:\nshuffled map",
}
LMAX = {"evil": 8000, "hallucination": 16000}
RUNGS = {
    "evil": [
        ("pvsynth", "PV synthetic suite"),
        ("train", "held-out train\n(DAN x forbidden-q)"),
        ("hhrt", "hh-rlhf (OOD)\n[floor-censored]"),
        ("toxicchat", "ToxicChat (OOD)"),
    ],
    "hallucination": [
        ("pvsynth", "PV synthetic suite"),
        ("train", "held-out TriviaQA"),
        ("nqopen", "NQ-Open (OOD)"),
        ("simpleqa", "SimpleQA (OOD)"),
    ],
}
SETTING_COLORS = ["#C44E52", "#4878CF", "#6ACC65", "#B47CC7"]


def pvsynth_rows(b):
    d = json.load(open(WT / "pvsynth" / b / "all_arms_spearman.json"))
    return {r["arm"]: r["rho_frozen"] for r in d["transfer_rows"] if r["variant"] == "context_end"}


def real_rows(b):
    d = json.load(open(WT / b / "arm_results/all_arms_spearman.json"))
    out = {}
    for r in d["arm_rows"] + [
        x for x in d["transfer_rows"] if x.get("rung_kind") == "eval_transfer"
    ]:
        if (
            r["variant"] == "context_end"
            and r["regime"] == "e1"
            and str(r["u_rung_label"]) == "full"
            and int(r["budget_l"]) == LMAX[b]
            and r.get("rho_frozen") is not None
        ):
            rung = r["eval_rung"]
            out.setdefault(rung, {}).setdefault(r["arm"], []).append(r["rho_frozen"])
    return {rung: {a: float(np.mean(v)) for a, v in arms.items()} for rung, arms in out.items()}


fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [4, 4, 2.2]})
for ax, b in zip(axes[:2], ["evil", "hallucination"]):
    pv = pvsynth_rows(b)
    real = real_rows(b)
    x = np.arange(len(ARMS))
    n_set = len(RUNGS[b])
    w = 0.8 / n_set
    for i, (rung, lab) in enumerate(RUNGS[b]):
        vals = [pv.get(a) if rung == "pvsynth" else real.get(rung, {}).get(a) for a in ARMS]
        ax.bar(
            x + (i - (n_set - 1) / 2) * w,
            [v if v is not None else np.nan for v in vals],
            w * 0.92,
            label=lab.replace("\n", " "),
            color=SETTING_COLORS[i],
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Spearman rho (pred vs judged)")
    ax.set_title(b, fontsize=10)
    ax.legend(fontsize=6.5, loc="upper right")

ax = axes[2]
pv = pvsynth_rows("sycophancy")
vals = [pv.get(a) for a in ARMS]
ax.bar(np.arange(len(ARMS)), vals, 0.7, color=SETTING_COLORS[0], edgecolor="white", linewidth=0.3)
ax.set_xticks(np.arange(len(ARMS)))
ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=6.5, rotation=25, ha="right")
ax.axhline(0, color="k", lw=0.6)
ax.set_title("sycophancy (suite only;\nreal rungs pending)", fontsize=9)

fig.suptitle(
    "Persona-vectors synthetic suite vs real evaluation settings "
    "(context-end, E1 PV, frozen layer; suite n=200/behavior; real rungs = committed op-slice means)",
    y=1.03,
    fontsize=10,
)
fig.tight_layout()
fig.savefig(OUT / "pvsuite_vs_real.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "pvsuite_vs_real.png")
