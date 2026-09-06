"""Render the approved reasoning story from banked #2546 results, without refits.

All-question metrics; own-generated answers in both thinking conditions.
The residual panel uses the prespecified penalties, not the best plotted result.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.c2a_plot_style import (
    MUTED,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval_results/issue_2546"
NEW = DATA / "paper_reasoning_20260906"
OUT = ROOT / "figures/paper"
COLORS = [ROLES["linear"].color, ROLES["nonlinear"].color, MUTED]
SOURCES = {}


def read(path):
    """Read a banked artifact and record its exact content hash."""
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def save(fig, stem, values):
    """Export print, review, grayscale and complete data/provenance together."""
    result = save_c2a_figure(
        fig,
        OUT / stem,
        title=stem,
        subject="Reasoning results, issue 2546",
        creator=str(Path(__file__).relative_to(ROOT)),
    )
    record = {
        "sources_sha256": SOURCES.copy(),
        "values": values,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "render": result["record"],
        "uncertainty": "Banked intervals only; no new bootstrap or inference.",
    }
    (OUT / f"{stem}.meta.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    plt.close(fig)
    print(stem, flush=True)


def interval(ax, x, y, bounds, **kwargs):
    """Draw stored interval endpoints without assuming they enclose the estimate."""
    lo, hi = bounds
    assert np.isfinite([x, y, lo, hi]).all() and lo <= hi
    ax.vlines(x, lo, hi, color=kwargs["color"], linewidth=1.8)
    ax.plot(x, y, linestyle="none", markersize=8, **kwargs)


def metrics(cell, arm):
    """Use the all-question subset from the production five-fold evaluation."""
    d = read(DATA / f"allfit/{cell}__a{arm}.json")
    assert d["subsets"]["all"]["n"] == {1: 30193, 3: 33810}[arm]
    return d["subsets"]["all"]


def pair(ax, rows, key, labels, title):
    """Paired estimate plot for one metric, with its own explicitly labeled axis."""
    ci_key = "r2_corpus_ci" if key == "r2_corpus" else "acc1_ci"
    y = [r[key] for r in rows]
    ax.plot([0, 1], y, color=MUTED, linewidth=1.4)
    for i, row in enumerate(rows):
        interval(ax, i, row[key], row[ci_key], color=COLORS[i], marker=["o", "s"][i])
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim((0.40, 0.72) if key == "r2_corpus" else (0.84, 1.01))
    ax.text(0, 1.04, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=20)
    style_axis(ax, grid_axis="y")


def main_plot():
    """Four panels: toggle, observed state, residual prediction, operator geometry."""
    fig, _ = c2a_figure("full", 0.80)
    grid = fig.add_gridspec(
        2, 2, left=0.10, right=0.98, bottom=0.09, top=0.89, wspace=0.36, hspace=0.68
    )
    a = [metrics("p7_Aoff", 3), metrics("p7_A", 3)]
    b = [metrics("p7_A", 1), metrics("p7_D", 1)]
    for slot, letter, model, title, rows, labels in [
        (grid[0, 0], "A", "Qwen3-8B", "Enabling CoT", a, ["Off", "On"]),
        (grid[0, 1], "B", "OpenThinker3-7B", "Observing CoT", b, ["Context", "CoT end"]),
    ]:
        inner = slot.subgridspec(1, 2, wspace=0.55)
        axes = [fig.add_subplot(inner[0, j]) for j in range(2)]
        for ax, key, metric in zip(
            axes, ["r2_corpus", "acc1"], [r"$R^2$", "Top-1 retrieval"], strict=True
        ):
            pair(ax, rows, key, labels, metric)
        panel_header(axes[0], letter, model, title, kicker_y=1.38, title_y=1.22)
    residual = read(NEW / "residual.json")
    ax = fig.add_subplot(grid[1, 0])
    rv = {}
    for j, (kind, label, marker) in enumerate(
        [
            ("end_of_thought__ridge_lam316", "CoT end", "o"),
            ("trace_mean__ridge_lam1000", "Trace mean", "s"),
        ]
    ):
        vals = []
        for i, scheme in enumerate(["random5", "loco7"]):
            row = residual["results"][scheme]["subsets"]["all"][kind]
            y = row["incremental_residual_r2"]
            interval(
                ax,
                i + (j - 0.5) * 0.18,
                y,
                row["incremental_residual_r2_ci95"],
                color=COLORS[j],
                marker=marker,
                label=label if i == 0 else None,
            )
            vals.append(row)
        rv[kind] = vals
    ax.axhline(0, color=MUTED, linestyle=":", linewidth=1.3)
    ax.set_xticks([0, 1], ["Random splits", "Held-out datasets"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.1, 0.65)
    ax.set_ylabel("Fraction of residual\nerror removed")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, loc="center right", fontsize=15)
    panel_header(ax, "C", "Qwen3-8B", "Predicting answer residuals")
    diff = read(DATA / "allfit/eot_vs_context/diffs/diffs.json")
    ov = diff["A3_operator_comparison"]["operators"]["subspace_overlaps"]
    ax = fig.add_subplot(grid[1, 1])
    ks = sorted(int(k) for k in ov)
    for j, (side, label, marker) in enumerate(
        [("right_input", "Input", "o"), ("left_output", "Output", "s")]
    ):
        ax.plot(
            ks,
            [ov[str(k)][side]["mean_principal_cos"] for k in ks],
            color=COLORS[j],
            marker=marker,
            label=label,
        )
        ax.plot(
            ks,
            [ov[str(k)][side]["null_mean_principal_cos"] for k in ks],
            color=MUTED,
            linestyle=":" if j == 0 else "--",
            label=f"Random {label.lower()}",
            linewidth=1.3,
        )
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of singular directions")
    ax.set_ylabel("Mean principal cosine")
    ax.legend(frameon=False, fontsize=14, loc="center right")
    style_axis(ax, grid_axis="y")
    panel_header(ax, "D", "OpenThinker3-7B", "Input/output subspace overlap")
    save(fig, "c1_cot_story", {"toggle": a, "observed": b, "residual": rv, "overlap": ov})


def appendix():
    """Plot the all-token scan, fine-tuning decompositions and exploratory SAE read."""
    scans = [read(NEW / name) for name in ["qwen.json", "openthinker.json"]]
    assert scans[0]["ids"] == scans[1]["ids"] and len(scans[0]["ids"]) == 8
    fig, _ = c2a_figure("full", 0.42)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.76, bottom=0.20, wspace=0.42)
    for j, (scan, ax) in enumerate(zip(scans, axes, strict=True)):
        for n, row in enumerate(scan["layers"]["19"]):
            v = np.asarray(row["values"])
            y = np.abs(v).max(axis=1)
            x = np.arange(len(y)) - (len(y) - 1)
            ax.plot(x, y, color=COLORS[j], alpha=0.25, linewidth=1)
            ax.plot(
                x[2],
                y[2],
                marker="o",
                color=COLORS[j],
                markersize=5,
                label="Early newline" if n == 0 else None,
            )
            ax.plot(
                0,
                y[-1],
                marker="s",
                color=COLORS[j],
                markersize=5,
                label="Context readout" if n == 0 else None,
            )
        ax.set_yscale("log")
        ax.set_ylim(1, 40000)
        ax.set_xlim(-80, 3)
        ax.set_xticks([-80, -60, -40, -20, 0])
        ax.set_xlabel("Token position (relative to context)")
        ax.set_ylabel("Max. absolute activation\n(three coordinates)")
        ax.legend(frameon=False, fontsize=14, loc="center left")
        style_axis(ax, grid_axis="y")
        panel_header(
            ax,
            "AB"[j],
            "Layer 19 · eight matched prompts",
            ["Qwen2.5-7B-Instruct", "OpenThinker3-7B"][j],
        )
    save(
        fig,
        "c1_cot_token_scan",
        {"scan_ids": scans[0]["ids"], "dimensions": scans[0]["dimensions"]},
    )
    diff = read(DATA / "allfit/eot_vs_context/diffs/diffs.json")
    shift = diff["B1_prepost_context_shift"]
    fig, _ = c2a_figure("full", 0.40)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.77, bottom=0.28, wspace=0.40)
    bottom = np.zeros(2)
    for k, label, color in zip(
        ["mean_offset_share", "global_scaling_extra_share", "question_specific_share"],
        ["Shared offset", "Scaling", "Question-specific"],
        COLORS,
        strict=True,
    ):
        v = np.array([shift[s]["oof_split"][k] for s in ["context", "answer"]])
        axes[0].bar([0, 1], v, bottom=bottom, color=color, label=label, width=0.55)
        bottom += v
    assert np.allclose(bottom, 1)
    axes[0].set_xticks([0, 1], ["Context", "Own answer"])
    axes[0].set_ylabel("Share of squared\ndisplacement")
    axes[0].legend(
        frameon=False, ncol=3, fontsize=13, loc="upper left", bbox_to_anchor=(-0.12, -0.24)
    )
    panel_header(axes[0], "A", "Reasoning fine-tuning", "Displacement decomposition")
    for j, side in enumerate(["context", "answer"]):
        vals = shift[side]["relnorm_median"]
        per = [v for k, v in vals.items() if k != "all"]
        axes[1].scatter(
            np.linspace(j - 0.08, j + 0.08, len(per)),
            per,
            color=COLORS[j],
            marker="o",
            alpha=0.5,
            s=24,
        )
        axes[1].plot(j, vals["all"], marker="D", color=COLORS[j], markersize=9)
    axes[1].set_xticks([0, 1], ["Context", "Own answer"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median relative\ndisplacement")
    panel_header(axes[1], "B", "Reasoning fine-tuning", "Relative shift magnitude")
    for ax in axes:
        style_axis(ax, grid_axis="y")
    save(fig, "c1_cot_finetuning_states", shift)
    matches = read(NEW / "sae_matches.json")
    fig, _ = c2a_figure("full", 0.40)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.18, wspace=0.30)
    sv = {}
    for j, (basis, ax) in enumerate(zip(["decoder", "encoder"], axes, strict=True)):
        b = matches["bases"][basis]
        sets = [
            [abs(r["matches"][0]["cosine"]) for r in b["directions"] if r["map"] == m]
            for m in ["context", "eot"]
        ] + [b["null_max_abs_cosines"]]
        for k, vals in enumerate(sets):
            vals = np.sort(vals)
            ax.plot(
                vals,
                np.arange(1, len(vals) + 1) / len(vals),
                color=COLORS[k],
                linestyle=["-", "--", ":"][k],
                label=["Context", "CoT end", "Random"][k],
            )
        sv[basis] = sets
        ax.set_xlim(0, 0.125)
        ax.set_xticks([0, 0.04, 0.08, 0.12])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nearest-feature absolute cosine")
        ax.set_ylabel("Fraction of directions")
        ax.legend(frameon=False, fontsize=14)
        style_axis(ax, grid_axis="y")
        panel_header(ax, "AB"[j], "Parent-model SAE", f"{basis.capitalize()} dictionary alignment")
    save(fig, "c1_cot_sae_alignment", sv)


def derive_similarity():
    """Summarize existing aligned context/EOT captures; no fitting or inference."""
    import issue2546_cx_eot_prepost_diffs as prior

    ids, _, pred = prior.load_preds("p7_A")
    del pred
    x = prior.load_target("cx_last", "post", ids).astype(np.float64)
    z = prior.load_target("cot_boundary", "post", ids).astype(np.float64)
    assert x.shape == z.shape == (30193, 3584)
    results = {}
    for remove in [False, True]:
        keep = np.ones(3584, dtype=bool)
        if remove:
            keep[[458, 2570, 2718]] = False
        for center in [False, True]:
            a, b = x[:, keep].copy(), z[:, keep].copy()
            if center:
                a -= a.mean(0)
                b -= b.mean(0)
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            assert (denom > 0).all()
            cosine = np.einsum("ij,ij->i", a, b) / denom
            results[f"remove3={remove},center={center}"] = {
                "mean": float(cosine.mean()),
                "median": float(np.median(cosine)),
                "q25": float(np.quantile(cosine, 0.25)),
                "q75": float(np.quantile(cosine, 0.75)),
            }
            del a, b
    paths = [prior.PRED_DIR / "p7_A__all__a1.npz"] + [
        prior.TG / f"{kind}__arm1__post__{ds}__l19.npz"
        for ds in prior.DATASETS
        for kind in ["cx_last", "cot_boundary"]
    ]
    provenance = {}
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                h.update(chunk)
        provenance[str(path)] = h.hexdigest()
    (NEW / "state_similarity.json").write_text(
        json.dumps(
            {
                "n": len(ids),
                "results": results,
                "sources_sha256": provenance,
                "centering": "Each readout centered separately over the same 30,193 rows.",
                "interval": "Across-question interquartile range, not confidence interval.",
            },
            indent=2,
        )
    )


def similarity_plot():
    """Expose similarity after removing mean and/or the three massive coordinates."""
    data = read(NEW / "state_similarity.json")
    fig, _ = c2a_figure("wide", 0.42)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.15, right=0.98, top=0.74, bottom=0.24)
    for j, center in enumerate([False, True]):
        for i, remove in enumerate([False, True]):
            row = data["results"][f"remove3={remove},center={center}"]
            interval(
                ax,
                i + (j - 0.5) * 0.14,
                row["median"],
                [row["q25"], row["q75"]],
                color=COLORS[j],
                marker=["o", "s"][j],
                label=["Raw", "Mean-centered"][j] if i == 0 else None,
            )
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xticks([0, 1], ["All coordinates", "Three coordinates removed"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.2, 0.7)
    ax.set_ylabel("Context-to-CoT-end\ncosine")
    ax.legend(frameon=False, loc="upper right", fontsize=15)
    style_axis(ax, grid_axis="y")
    panel_header(ax, "", "OpenThinker3-7B · layer 19", "Similarity beyond the massive coordinates")
    save(fig, "c1_cot_state_similarity", data)


if __name__ == "__main__":
    if sys.argv[1:] == ["--derive-similarity"]:
        derive_similarity()
    else:
        assert len(sys.argv) == 1, "Only --derive-similarity is supported."
        set_c2a_style()
        main_plot()
        appendix()
        similarity_plot()
