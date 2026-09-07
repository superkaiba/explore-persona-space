"""Render the approved reasoning story from banked #2546 results, without refits.

Needs-reasoning evaluation of existing all-question fits; own-generated answers.
The default export gives each Qwen3 claim its own figure: prediction/control
and the necessity-group comparison.
Historical appendix renderers are retained but are not called by default.
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
    """Score only needs-reasoning rows of the unchanged production evaluation."""
    d = read(DATA / f"allfit/{cell}__a{arm}.json")
    assert d["subsets"]["all"]["n"] == {1: 30193, 3: 33810}[arm]
    assert d["subsets"]["necessary"]["n"] == {1: 2326, 3: 4522}[arm]
    return d["subsets"]["necessary"]


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
    """Qwen3-only prediction/control comparison, without duplicate conditions."""
    SOURCES.clear()
    cells = ["p7_Aoff", "p7_A", "p7_D"]
    rows = [metrics(cell, 3) for cell in cells]
    labels = ["Thinking off\nContext", "Thinking on\nContext", "Thinking on\nCoT end"]
    colors = [MUTED, ROLES["linear"].color, ROLES["nonlinear"].color]
    markers = ["D", "o", "s"]
    fig, _ = c2a_figure("full", 0.36)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.76, wspace=0.40)
    for ax, letter, key, title, ylabel, scale, limits in zip(
        axes,
        ["A", "B"],
        ["r2_corpus", "acc1"],
        ["Answer predictability", "Answer retrieval"],
        [r"$R^2\,\uparrow$", r"Top-1 retrieval (%) $\uparrow$"],
        [1.0, 100.0],
        [(0.40, 0.57), (84.0, 102.0)],
        strict=True,
    ):
        ci_key = "r2_corpus_ci" if key == "r2_corpus" else "acc1_ci"
        x = np.arange(3)
        ax.plot(
            x,
            [scale * row[key] for row in rows],
            color=MUTED,
            linewidth=1.4,
            linestyle="-" if key == "r2_corpus" else "--",
        )
        for i, row in enumerate(rows):
            interval(
                ax,
                i,
                scale * row[key],
                [scale * bound for bound in row[ci_key]],
                color=colors[i],
                marker=markers[i],
                markerfacecolor=colors[i] if key == "r2_corpus" else "white",
                markeredgewidth=1.6,
            )
        ax.set_xticks(x, labels, fontsize=16)
        ax.set_xlim(-0.35, 2.35)
        ax.set_ylim(*limits)
        ax.set_ylabel(ylabel)
        style_axis(ax, grid_axis="y")
        panel_header(ax, letter, "Qwen3-8B · needs reasoning", title)
    save(
        fig,
        "c1_cot_story",
        {
            "conditions": [
                {"cell": cell, "label": label, "metrics": row}
                for cell, label, row in zip(cells, labels, rows, strict=True)
            ],
            "comparisons": {
                "enabling_cot": "p7_Aoff versus p7_A; each mode's own answer target",
                "observing_cot": "p7_A versus p7_D; identical thinking-on answer targets",
            },
            "evaluation_subset": "necessary",
            "n_evaluated": 4522,
            "maps_refit": False,
        },
    )


def necessity_plot():
    """Render the two necessity groups from saved equal-dataset-weighted scores."""
    SOURCES.clear()
    data = read(NEW / "qwen3_necessity_table.json")
    assert data["model"] == "Qwen3-8B" and data["layer"] == 24
    pooled = data["pooled_equal_corpus_weight"]
    groups = ["necessary", "both_correct"]
    labels = ["Needs reasoning", "Does not need\nreasoning"]
    expected_counts = {"necessary": 4522, "both_correct": 17693}
    fig, _ = c2a_figure("wide", 0.46)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.23, top=0.77)
    for readout, label, color, marker in [
        ("context", "Context", ROLES["linear"].color, "o"),
        ("end_of_thought", "CoT end", ROLES["nonlinear"].color, "s"),
    ]:
        rows = [pooled[readout][group] for group in groups]
        for group, row in zip(groups, rows, strict=True):
            assert row["n"] == sum(row["n_by_corpus"].values()) == expected_counts[group]
            assert len(row["weights"]) == 7
            assert np.allclose(list(row["weights"].values()), 1 / 7)
        y = [row["r2_corpus_mean"] for row in rows]
        ax.plot([0, 1], y, color=color, linewidth=1.6)
        for x, row in enumerate(rows):
            interval(
                ax,
                x,
                row["r2_corpus_mean"],
                row["r2_corpus_mean_ci"],
                color=color,
                marker=marker,
                markerfacecolor=color,
                markeredgewidth=1.6,
            )
        ax.annotate(
            label,
            (1, y[-1]),
            xytext=(12, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=18,
        )
    ax.set_xticks([0, 1], labels, fontsize=17)
    ax.set_xlim(-0.30, 1.65)
    ax.set_ylim(0.39, 0.54)
    ax.set_yticks([0.40, 0.45, 0.50])
    ax.set_ylabel(r"Held-out $R^2\,\uparrow$")
    style_axis(ax, grid_axis="y")
    panel_header(ax, "", "Qwen3-8B · thinking on", "Predictability by necessity group")
    save(
        fig,
        "c1_cot_necessity_comparison",
        {
            "model": data["model"],
            "layer": data["layer"],
            "groups": groups,
            "readouts": pooled,
            "aggregation": "Ratio of pooled SSE/SST with equal total weight per dataset",
            "baseline": "Whole-dataset mean, shared between necessity groups",
            "targets": "Same thinking-on own-generated answer vectors for both readouts",
            "intervals": "Saved 95% question-bootstrap intervals, stratified by dataset",
            "maps_refit": False,
            "geometry": "Not analyzed in this figure",
        },
    )


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
    shift = read(NEW / "necessary_diagnostics.json")["finetuning"]
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
    data = read(NEW / "necessary_diagnostics.json")["similarity"]
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


def derive_necessary():
    """Restrict cached diagnostics to necessary rows without fitting any map."""
    import issue2546_cx_eot_prepost_diffs as prior

    sources = {}

    def record(path):
        """Record content hashes for every input, including external caches."""
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                h.update(chunk)
        sources[str(path)] = h.hexdigest()
        return path

    with np.load(record(prior.PRED_DIR / "p7_A__all__a1.npz")) as bank:
        ids = bank["conv_ids"].astype(str)
        folds = bank["folds"]
        labels = bank["labels"].astype(str)
    mask = labels == "necessary"
    assert len(ids) == 30193 and mask.sum() == 2326 and len(set(ids)) == len(ids)
    selected = ids[mask]
    ds = np.char.partition(selected, ":")[:, 0]
    result = {
        "n": 2326,
        "subset": "necessary",
        "maps_refit": False,
        "training_n": 30193,
        "row_ids": selected.tolist(),
    }
    x = prior.load_target("cx_last", "post", selected).astype(np.float64)
    z = prior.load_target("cot_boundary", "post", selected).astype(np.float64)
    delta = z - x
    mean = delta.mean(0)
    total = prior.sq_sum(delta)
    result["context_to_eot"] = {
        "mean_offset_share": len(selected) * prior.sq_sum(mean) / total,
        "mean_offset_top3_share": prior.sq_sum(mean[[458, 2570, 2718]]) / prior.sq_sum(mean),
    }
    similarity = {}
    for remove in [False, True]:
        keep = np.ones(3584, dtype=bool)
        if remove:
            keep[[458, 2570, 2718]] = False
        for center in [False, True]:
            a, b = x[:, keep].copy(), z[:, keep].copy()
            if center:
                a -= a.mean(0)
                b -= b.mean(0)
            denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            assert (denominator > 0).all()
            cosine = np.einsum("ij,ij->i", a, b) / denominator
            similarity[f"remove3={remove},center={center}"] = {
                "mean": float(cosine.mean()),
                "median": float(np.median(cosine)),
                "q25": float(np.quantile(cosine, 0.25)),
                "q75": float(np.quantile(cosine, 0.75)),
            }
    result["similarity"] = {
        "n": 2326,
        "results": similarity,
        "centering": "Each readout centered over the 2,326 necessary rows.",
    }
    del x, z, delta, a, b

    old = json.loads(record(DATA / "allfit/eot_vs_context/diffs/diffs.json").read_text())
    shift = {}
    for side, kind in [("context", "cx_last"), ("answer", "ans_mean")]:
        pre = prior.load_target(kind, "pre", ids)
        post = prior.load_target(kind, "post", ids)
        scales = old["B1_prepost_context_shift"][side]["oof_split"]["scale_s_per_fold"]
        sums = np.zeros(3, dtype=np.float64)
        for k in range(5):
            tr, te = folds != k, (folds == k) & mask
            # Reconstruct existing training-fold intercepts; reuse banked scalars.
            pmu = pre[tr].mean(0, dtype=np.float64)
            qmu = post[tr].mean(0, dtype=np.float64)
            p, q = pre[te], post[te]
            offset = (qmu - pmu).astype(np.float32)
            scaled = np.float32(scales[k]) * p + (qmu - scales[k] * pmu).astype(np.float32)
            sums += [prior.sq_sum(q - p), prior.sq_sum(q - p - offset), prior.sq_sum(q - scaled)]
        relative = np.linalg.norm(post[mask].astype(np.float64) - pre[mask], axis=1)
        relative /= np.linalg.norm(pre[mask].astype(np.float64), axis=1)
        shift[side] = {
            "relnorm_median": {
                **{c: float(np.median(relative[ds == c])) for c in prior.DATASETS},
                "all": float(np.median(relative)),
            },
            "oof_split": {
                "mean_offset_share": float(1 - sums[1] / sums[0]),
                "global_scaling_extra_share": float((sums[1] - sums[2]) / sums[0]),
                "question_specific_share": float(sums[2] / sums[0]),
                "scale_s_per_fold": scales,
            },
        }
        del pre, post
        print(f"necessary diagnostics: {side}", flush=True)
    result["finetuning"] = shift

    base = prior.BASE / "input_sae_qualitative_20260906"
    with np.load(record(base / "validated_retrieval.npz")) as cache:
        assert np.array_equal(cache["ids"].astype(str), ids)
        assert np.array_equal(cache["folds"], folds)
        ah, dh = cache["A_hit"].astype(bool), cache["D_hit"].astype(bool)
        recovered = mask & ~ah & dh
        counts = {
            "both": int((mask & ah & dh).sum()),
            "recovered": int(recovered.sum()),
            "lost": int((mask & ah & ~dh).sum()),
            "neither": int((mask & ~ah & ~dh).sum()),
        }
        assert sum(counts.values()) == 2326 and recovered.any()
        ranks = cache["A_rank"][recovered]
        result["retrieval"] = {
            "counts": counts,
            "recovered_context_rank2": int((ranks == 2).sum()),
            "recovered_context_rank_le10": int((ranks <= 10).sum()),
            "recovered_context_rank_max": int(ranks.max()),
        }
        pairs = json.loads(record(base / "qualitative_pairs.json").read_text())
        chosen = [r for r in pairs["items"] if r["row_id"] in set(selected)]
        assert len(chosen) == 20
        positions = {rid: i for i, rid in enumerate(ids)}
        for row in chosen:
            i = positions[row["row_id"]]
            for letter, key in [("A", "rank_context"), ("D", "rank_eot")]:
                assert row[key] == int(cache[f"{letter}_rank"][i])
            assert row["neighbor_id"] == ids[cache[f"{row['neighbor_map']}_nn_other"][i]]
        result["qualitative"] = {
            "selection": "All necessary rows in the previous export: "
            "two recovered cases and all 18 EOT misses; illustrative, unblinded.",
            "items": chosen,
        }
    for c in prior.DATASETS:
        for kind, sides in [
            ("cx_last", ["pre", "post"]),
            ("ans_mean", ["pre", "post"]),
            ("cot_boundary", ["post"]),
        ]:
            for side in sides:
                record(prior.TG / f"{kind}__arm1__{side}__{c}__l19.npz")
    result["sources_sha256"] = sources
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (NEW / "necessary_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("necessary_diagnostics.json complete", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--derive-similarity"]:
        derive_similarity()
    elif sys.argv[1:] == ["--derive-necessary"]:
        derive_necessary()
    else:
        assert len(sys.argv) == 1, "Use --derive-similarity or --derive-necessary."
        set_c2a_style()
        main_plot()
        necessity_plot()
        # Historical OpenThinker diagnostics remain reproducible via their functions,
        # but are no longer part of the Qwen3-only manuscript render.
