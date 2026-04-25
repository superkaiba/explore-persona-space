"""Plot strong convergence results (7 sources, 20 epochs)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.analysis.paper_plots import paper_palette, savefig_paper, set_paper_style

set_paper_style("neurips", font_scale=1.0)

EPOCHS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

DATA = {
    "villain": {
        "cos": -0.36,
        "asst": [0.0, 45.0, 41.2, 50.2, 65.3, 73.2, 60.2, 59.3, 57.7, 56.3, 62.3],
    },
    "medical_doctor": {
        "cos": 0.59,
        "asst": [25.0, 57.5, 59.5, 53.0, 52.0, 50.0, 45.5, 42.0, 32.5, 38.5, 43.0],
    },
    "comedian": {
        "cos": -0.29,
        "asst": [0.0, 37.2, 40.0, 42.5, 37.2, 41.5, 39.8, 41.5, 39.2, 39.7, 41.5],
    },
    "kindergarten_teacher": {
        "cos": 0.28,
        "asst": [2.5, 26.0, 18.5, 22.5, 31.5, 28.5, 30.0, 32.5, 28.0, 30.5, 26.0],
    },
    "software_engineer": {
        "cos": 0.47,
        "asst": [30.0, 24.5, 37.5, 30.0, 26.5, 13.5, 11.5, 7.5, 6.0, 7.0, 8.0],
    },
    "nurse": {
        "cos": 0.55,
        "asst": [12.5, 23.0, 15.5, 14.5, 12.0, 17.5, 15.5, 6.0, 6.5, 11.5, 12.5],
    },
    "librarian": {
        "cos": 0.47,
        "asst": [3.0, 2.0, 0.0, 0.5, 4.0, 2.5, 2.5, 4.0, 1.5, 2.5, 5.0],
    },
}

LABELS = {
    "villain": "Villain",
    "medical_doctor": "Med Doctor",
    "comedian": "Comedian",
    "kindergarten_teacher": "KT",
    "software_engineer": "SW Eng",
    "nurse": "Nurse",
    "librarian": "Librarian",
}

colors = paper_palette(7)
COLORS = {name: colors[i] for i, name in enumerate(DATA.keys())}

FIG_DIR = Path("figures/causal_proximity")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Figure 1: All 7 trajectories ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

for name, d in DATA.items():
    ax.plot(
        EPOCHS,
        d["asst"],
        "o-",
        color=COLORS[name],
        label=f"{LABELS[name]} (cos={d['cos']:+.2f})",
        markersize=5,
        linewidth=2,
    )

ax.set_xlabel("Convergence epoch")
ax.set_ylabel("Assistant marker leakage (%)")
ax.set_title("Marker leakage vs convergence training (7 source personas)")
ax.set_xticks(EPOCHS)
ax.set_ylim(-2, 80)
ax.legend(fontsize=8, loc="upper right")
ax.axhline(0, color="grey", ls="--", alpha=0.3, lw=0.8)
fig.tight_layout()
savefig_paper(fig, "causal_proximity/strong_convergence_7sources", dir="figures/")
plt.close(fig)
print("Saved strong_convergence_7sources")


# ── Figure 2: Cosine vs peak leakage (scatter) ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for name, d in DATA.items():
    peak = max(d["asst"])
    ax.scatter(
        d["cos"],
        peak,
        c=COLORS[name],
        s=120,
        zorder=5,
        edgecolors="white",
        linewidth=1,
    )
    ax.annotate(
        LABELS[name],
        (d["cos"], peak),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=9,
    )

ax.set_xlabel("Baseline cosine similarity to assistant (L15)")
ax.set_ylabel("Peak assistant leakage (%)")
ax.set_title("Cosine does NOT predict peak leakage")
ax.axhline(0, color="grey", ls="--", alpha=0.3, lw=0.8)
ax.axvline(0, color="grey", ls="--", alpha=0.3, lw=0.8)
fig.tight_layout()
savefig_paper(fig, "causal_proximity/cosine_vs_peak_leakage", dir="figures/")
plt.close(fig)
print("Saved cosine_vs_peak_leakage")


# ── Figure 3: Grouped by behavior ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

groups = {
    "High leakage\n(rise-then-decline)": ["villain", "medical_doctor"],
    "Medium leakage\n(flat)": ["comedian", "kindergarten_teacher"],
    "Low / declining\nleakage": ["software_engineer", "nurse", "librarian"],
}

for ax, (title, names) in zip(axes, groups.items()):
    for name in names:
        d = DATA[name]
        ax.plot(
            EPOCHS,
            d["asst"],
            "o-",
            color=COLORS[name],
            label=f"{LABELS[name]} (cos={d['cos']:+.2f})",
            markersize=4,
            linewidth=1.5,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Convergence epoch")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.legend(fontsize=8)
    ax.axhline(0, color="grey", ls="--", alpha=0.3, lw=0.8)

axes[0].set_ylabel("Assistant marker leakage (%)")
axes[0].set_ylim(-2, 80)
fig.suptitle("Three distinct leakage behaviors under convergence SFT", fontsize=12, y=1.02)
fig.tight_layout()
savefig_paper(fig, "causal_proximity/strong_convergence_grouped", dir="figures/")
plt.close(fig)
print("Saved strong_convergence_grouped")


# ── Figure 4: Baseline vs peak (paired) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

sorted_names = sorted(DATA.keys(), key=lambda n: max(DATA[n]["asst"]), reverse=True)
x = np.arange(len(sorted_names))

baselines = [DATA[n]["asst"][0] for n in sorted_names]
peaks = [max(DATA[n]["asst"]) for n in sorted_names]
labels = [f"{LABELS[n]}\n(cos={DATA[n]['cos']:+.2f})" for n in sorted_names]

ax.bar(x - 0.15, baselines, 0.3, label="Epoch 0 (baseline)", color=colors[0], alpha=0.7)
ax.bar(x + 0.15, peaks, 0.3, label="Peak leakage", color=colors[1], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Assistant marker leakage (%)")
ax.set_title("Convergence SFT creates leakage for some personas but not others")
ax.legend()
fig.tight_layout()
savefig_paper(fig, "causal_proximity/baseline_vs_peak_leakage", dir="figures/")
plt.close(fig)
print("Saved baseline_vs_peak_leakage")

print("\nAll plots done.")
