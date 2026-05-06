#!/usr/bin/env python3
"""Issue #262 — plotting (hero figure + per-persona forest + diagnostic panels).

Reads `eval_results/issue-262/run_result.json` (and sidecars) produced by
`scripts/run_em_first_persona_flatten_262.py`, and emits:

  figures/issue-262/hero_persona_flatten_262.{png,pdf}
      Single-panel grouped bar chart, 4 categories (source / assistant /
      bystander pool / negatives), 4 series (C1, C2, C2', C3) with Wald 95%
      CIs (per plan §10).
  figures/issue-262/bystander_forest_per_persona.{png,pdf}
      Per-persona forest, all 11 personas x 4 conditions, Wilson 95% CI.
  figures/issue-262/length_panel.{png,pdf}
      Mean completion length (tokens) per condition (descriptive only;
      length-controlled inference lives in run_result.json's
      `length_controlled_logit`).
  figures/issue-262/cosines_per_layer.{png,pdf}
      Mean ± std librarian-vs-bystander cosine across layers, per merged
      model (raw / EM / benign). Diagnostic only (§11.10).

Plan: .claude/plans/issue-262.md (cached, gitignored).

Usage:
    uv run python scripts/plot_issue262.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "eval_results" / "issue-262"
FIG_ROOT = REPO_ROOT / "figures"
FIG_DIR_REL = "issue-262"

CONDITIONS = ["c1", "c2", "c2p", "c3"]
COND_KEYS = {
    "c1": "c1_base_first",
    "c2": "c2_em_first",
    "c2p": "c2p_em_first_basesrc",
    "c3": "c3_benign_first",
}
COND_LABELS = {
    "c1": "C1 base-first",
    "c2": "C2 EM-first",
    "c2p": "C2' EM-first / base-src",
    "c3": "C3 benign-first",
}
TRULY_UNSEEN = [
    "software_engineer",
    "kindergarten_teacher",
    "medical_doctor",
    "comedian",
    "police_officer",
    "villain",
    "zelthari_scholar",
]
NEGATIVES = ["data_scientist", "french_person"]
SOURCE_NAME = "librarian"
ASSISTANT_NAME = "assistant"


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for proportion (used per plan §10 forest, plus C2/C1=0 cells)."""
    if n <= 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1.0 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load_run_result() -> dict:
    rr = RESULT_DIR / "run_result.json"
    if not rr.exists():
        raise FileNotFoundError(
            f"Missing {rr} — run scripts/run_em_first_persona_flatten_262.py first."
        )
    return json.loads(rr.read_text())


def plot_hero(rr: dict) -> None:
    """Hero figure: 4 grouped bars per category (§10)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        add_direction_arrow,
        paper_palette,
        proportion_ci,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(4)

    categories = [
        "Source\n(librarian)",
        "Assistant",
        "Bystanders\n(7 unseen)",
        "Negatives\n(ds + fr)",
    ]
    cat_keys = ["source", "assistant", "bystander_pool", "negatives"]
    n_cats = len(categories)

    rates: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    cis: dict[str, list[tuple[float, float]]] = {c: [] for c in CONDITIONS}
    annotations: dict[str, list[str]] = {c: [] for c in CONDITIONS}

    for cond in CONDITIONS:
        block = rr["conditions"][COND_KEYS[cond]]
        for cat in cat_keys:
            if cat == "source":
                hits = block["per_persona"][SOURCE_NAME]["strict_hits"]
                n = block["per_persona"][SOURCE_NAME]["total"]
            elif cat == "assistant":
                hits = block["per_persona"][ASSISTANT_NAME]["strict_hits"]
                n = block["per_persona"][ASSISTANT_NAME]["total"]
            elif cat == "bystander_pool":
                hits = block["bystander_pool_hits"]
                n = block["bystander_pool_n"]
            else:  # negatives
                hits = sum(block["per_persona"][p]["strict_hits"] for p in NEGATIVES)
                n = sum(block["per_persona"][p]["total"] for p in NEGATIVES)
            rate = hits / n if n else 0.0
            rates[cond].append(rate)
            lo, hi = proportion_ci(rate, n)
            cis[cond].append((lo, hi))
            annotations[cond].append(f"{rate * 100:.0f}%\n({hits}/{n})")

    x = np.arange(n_cats)
    bar_w = 0.20
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, cond in enumerate(CONDITIONS):
        xs = x + (i - 1.5) * bar_w
        ys = rates[cond]
        lows = [r - lo for r, (lo, _hi) in zip(ys, cis[cond], strict=True)]
        highs = [hi - r for r, (_lo, hi) in zip(ys, cis[cond], strict=True)]
        ax.bar(
            xs,
            ys,
            width=bar_w,
            color=palette[i],
            label=COND_LABELS[cond],
            yerr=[lows, highs],
            capsize=3,
            error_kw={"linewidth": 0.9},
            edgecolor="black",
            linewidth=0.4,
        )
        for xi, ann in zip(xs, annotations[cond], strict=True):
            ax.text(xi, max(ys) + 0.06, ann, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("[ZLT] rate")
    add_direction_arrow(ax, "y", "down")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Bystander [ZLT] leakage by training-order condition (issue #262, seed 42)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR_REL}/hero_persona_flatten_262", dir=str(FIG_ROOT))
    plt.close(fig)


def plot_forest(rr: dict) -> None:
    """Per-persona forest with Wilson 95% CIs (§10)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        add_direction_arrow,
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(4)

    # Order: source, assistant, bystanders (7), negatives — matches plan §10 grouping.
    persona_order = [SOURCE_NAME, ASSISTANT_NAME, *TRULY_UNSEEN, *NEGATIVES]
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    n_personas = len(persona_order)
    y = np.arange(n_personas)
    point_offset = 0.18
    for i, cond in enumerate(CONDITIONS):
        block = rr["conditions"][COND_KEYS[cond]]
        rates: list[float] = []
        cis: list[tuple[float, float]] = []
        for p in persona_order:
            row = block["per_persona"][p]
            rates.append(row["strict_rate"])
            cis.append(_wilson_ci(row["strict_rate"], row["total"]))
        ys = y + (i - 1.5) * point_offset
        lows = [r - lo for r, (lo, _hi) in zip(rates, cis, strict=True)]
        highs = [hi - r for r, (_lo, hi) in zip(rates, cis, strict=True)]
        ax.errorbar(
            rates,
            ys,
            xerr=[lows, highs],
            fmt="o",
            markersize=4,
            color=palette[i],
            label=COND_LABELS[cond],
            capsize=2.5,
            elinewidth=0.9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(persona_order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("[ZLT] rate")
    add_direction_arrow(ax, "x", "down")
    ax.set_xlim(-0.02, 1.0)
    ax.axhline(1.5, color="grey", linewidth=0.5, linestyle="--")
    ax.axhline(8.5, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("Per-persona [ZLT] rate (Wilson 95% CI) — issue #262")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR_REL}/bystander_forest_per_persona", dir=str(FIG_ROOT))
    plt.close(fig)


def plot_length(rr: dict) -> None:
    """Mean completion length per condition — descriptive footer panel (§10)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(4)

    means: list[float] = []
    for cond in CONDITIONS:
        block = rr["conditions"][COND_KEYS[cond]]
        m = block.get("mean_completion_tokens_by_persona") or {}
        # Pool the per-persona means uniformly (each persona has the same N).
        vals = [v for v in m.values() if v is not None]
        means.append(float(np.mean(vals)) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(CONDITIONS))
    ax.barh(y, means, color=[palette[i] for i in range(len(CONDITIONS))])
    ax.set_yticks(y)
    ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.invert_yaxis()
    ax.set_xlabel("Mean completion length (tokens)")
    ax.set_title("Per-condition mean completion length (descriptive)")
    for yi, mi in zip(y, means, strict=True):
        ax.text(mi + 5, yi, f"{mi:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR_REL}/length_panel", dir=str(FIG_ROOT))
    plt.close(fig)


def plot_cosines() -> None:
    """Per-layer librarian-vs-bystander cosines (§11.10)."""
    cos_path = RESULT_DIR / "persona_cosines.json"
    if not cos_path.exists():
        print(f"  skipping cosines plot: {cos_path} missing", file=sys.stderr)
        return

    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    palette = paper_palette(3)
    blob = json.loads(cos_path.read_text())
    layers = blob["raw_instruct"]["layers"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, base in enumerate(["raw_instruct", "em_merged", "benign_merged"]):
        per_layer = blob[base]["per_layer"]
        means: list[float] = []
        stds: list[float] = []
        for L in layers:
            row = per_layer[str(L)]["cos_librarian_vs"]
            byst_means = [row[p]["mean"] for p in TRULY_UNSEEN if p in row]
            byst_stds = [row[p]["std"] for p in TRULY_UNSEEN if p in row]
            means.append(float(np.mean(byst_means)))
            stds.append(float(np.mean(byst_stds)))
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(layers, means_arr, marker="o", color=palette[i], label=base)
        ax.fill_between(
            layers, means_arr - stds_arr, means_arr + stds_arr, color=palette[i], alpha=0.15
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine(librarian, bystander) — across 7 unseen")
    ax.set_title("Persona-cosine geometry across layers (mean ± std across 20 questions)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_DIR_REL}/cosines_per_layer", dir=str(FIG_ROOT))
    plt.close(fig)


def main() -> int:
    rr = _load_run_result()
    plot_hero(rr)
    plot_forest(rr)
    plot_length(rr)
    plot_cosines()
    print(f"All figures written under {FIG_ROOT / FIG_DIR_REL}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
