"""Render the six-setting K5 leave-one-setting-out results from pinned artifacts.

Run in the explore-persona-space environment. This only summarizes existing
held-out predictions; it does not refit maps or run model inference.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    METRIC_LABELS,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    "k5_results": (
        "90309ec95e757ee5bd2e1b941e006858bd24a328656386444b6c55fb43f928f2",
        "9de026f872c19b2ca4fd3e4539de820e08038ee3",
        "production_v1",
    ),
    "loso_results": (
        "56beabc6855f4420b61150c5aad7dad550c7a5e1c1f5b8f18eba7f8df1b94179",
        "98d350d78e94fd5e79685be70376919147ee0cc7",
        "leave_one_setting_out_v1",
    ),
}
SETTINGS = [
    ("Assistant\n(chat)", "conversation_paired_stories_assistant__on_policy__chat"),
    ("Assistant\n(plain text)", "conversation_paired_stories_assistant__on_policy__bare_text"),
    ("HELIOS", "char_helios__on_policy__attrib_quoted"),
    ("Wren", "char_wren__on_policy__attrib_quoted"),
    ("Dana", "char_dana__on_policy__attrib_quoted"),
    ("Vex", "char_vex__on_policy__attrib_quoted"),
]
MODELS = [
    ("base", "qwen2.5-7b", "base_model"),
    ("post_trained", "qwen2.5-7b-instruct", "post_trained"),
]


def read_sources(source_dir):
    """Verify immutable result bytes before selecting plotted rows."""
    sources, provenance = {}, {}
    for name, (sha, revision, prefix) in SOURCES.items():
        content = (source_dir / f"{name}.json").read_bytes()
        assert hashlib.sha256(content).hexdigest() == sha, name
        sources[name] = json.loads(content)
        provenance[name] = {
            "sha256": sha,
            "url": "https://huggingface.co/datasets/superkaiba1/"
            f"explore-persona-space-data/resolve/{revision}/"
            f"issue2054_section44_k5_gcp/{prefix}/results.json",
        }
    return sources, provenance


def summarize(sources, provenance):
    """Preserve five folds and mapping baselines alongside each plotted mean."""
    primary = sources["k5_results"]
    own = {
        r["cell"]: r
        for r in primary["results"]
        if r["k_rollouts"] == 5 and r["cohort"] == "all" and r["status"] == "complete"
    }
    loso = {r["cell"]: r for r in sources["loso_results"]["panels"]}
    assert len(own) == len(loso) == 12
    assert sources["loso_results"]["metadata"]["status"] == "complete"
    coverage = {r["cell"]: r for r in primary["coverage"]}
    rows, separate, shared = [], [], []
    for label, prefix in SETTINGS:
        row, separate_row = {"label": label}, {"label": label}
        for key, model, _ in MODELS:
            cell = f"{prefix}__{model}"
            original = sorted(own[cell]["folds"], key=lambda f: f["fold"])
            folds = sorted(loso[cell]["folds"], key=lambda f: f["fold"])
            assert [f["fold"] for f in folds] == list(range(5))
            assert [f["fold"] for f in original] == list(range(5))
            for reference, fold in zip(original, folds, strict=True):
                assert fold["status"] == reference["status"] == "complete"
                audit = fold["source_audit"]
                assert audit["excluded_setting"] == cell
                assert audit["excluded_conversation_fold"] == fold["fold"]
                assert not audit["target_setting_labels_used_for_training"]
                assert audit["test_conversation_overlap"] == 0
                assert len(audit["source_settings"]) == 5
                assert cell not in audit["source_settings"]
                for source_cell, source in audit["source_settings"].items():
                    assert source_cell.split("__")[-1] == model
                    assert source["fold_counts"][fold["fold"]] == 0
                cohort = fold["cohorts"]["all"]
                assert cohort["status"] == "complete"
                assert cohort["n_test"] == reference["n_test"]
                assert cohort["reference"]["own"] == reference["metrics"]["own"]
                assert cohort["reference"]["six_setting_pool"] == reference["metrics"]["pooled"]
            scores = [f["cohorts"]["all"]["metrics"]["leave_one_setting_out"]["r2"] for f in folds]
            row[key] = {
                "cell": cell,
                "r2": float(np.mean(scores)),
                "fold_r2": scores,
                "folds": folds,
                "coverage": coverage[cell],
            }
            scores_own = [f["metrics"]["own"]["r2"] for f in original]
            assert np.isclose(np.mean(scores_own), own[cell]["r2_mean"]["own"])
            separate_row[key] = {
                "cell": cell,
                "r2": float(np.mean(scores_own)),
                "fold_r2": scores_own,
                "folds": original,
                "coverage": coverage[cell],
                "n": sum(f["n_test"] for f in original),
            }
        rows.append(row)
        separate.append(separate_row)
        trained = own[f"{prefix}__qwen2.5-7b-instruct"]["r2_mean"]
        shared.append(
            {
                "label": label,
                "base_own": separate_row["base"]["r2"],
                "post_own": trained["own"],
                "shared_asis": trained["pooled"],
                "frac_shared_asis": trained["pooled"] / trained["own"],
            }
        )
    data = {
        "k_rollouts": 5,
        "cohort": "all complete-five contexts, capped answers retained",
        "layer": 19,
        "block_index": 18,
        "sources": provenance,
        "aggregation": "Arithmetic mean of five held-out conversation-fold R2 values",
        "evaluation": "Excluded setting contributes no training labels, GCV data, or intercept. "
        "Evaluation conversation fold is excluded from every source setting.",
        "data": rows,
    }
    manuscript = {
        "panel_a": {"k_rollouts": 5, "source": provenance["k5_results"], "data": separate},
        "panel_b": {
            "k_rollouts": 5,
            "pool_settings": 6,
            "model": "qwen2.5-7b-instruct",
            "source": provenance["k5_results"],
            "speakers": shared,
            "recovery": "Mean shared R2 divided by mean separate R2 on identical five-answer "
            "targets, test rows, and conversation folds",
        },
    }
    return data, manuscript


def render(data, stem):
    """Draw fold variability as individual marks, without an inferential interval."""
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.49)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.23, top=0.79)
    x = np.arange(6, dtype=float)
    width = 0.34
    for offset, (key, _, role) in zip((-0.5, 0.5), MODELS, strict=True):
        style = ROLES[role]
        positions = x + offset * width
        means = [r[key]["r2"] for r in data["data"]]
        ax.bar(positions, means, width, color=style.color, alpha=0.9)
        for position, row in zip(positions, data["data"], strict=True):
            values = np.asarray(row[key]["fold_r2"])
            ax.scatter(
                position + np.linspace(-0.075, 0.075, 5),
                values,
                marker=style.marker,
                s=24,
                facecolors="white",
                edgecolors=INK,
                linewidths=0.8,
                zorder=4,
            )
            mean = row[key]["r2"]
            label_y = (
                max(values.max(), mean) + 0.045 if mean >= 0 else min(values.min(), mean) - 0.04
            )
            ax.text(
                position,
                label_y,
                f"{mean:.3f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                size=14,
                color=INK,
            )
    ax.axhline(0, color=INK, linewidth=1.2)
    ax.set_ylim(-0.56, 0.77)
    ax.set_ylabel(better_label(METRIC_LABELS["r2"]))
    ax.set_xticks(x, [r["label"] for r in data["data"]])
    ax.set_xlabel("Excluded setting")
    style_axis(ax)
    panel_header(
        ax,
        "",
        "Qwen2.5-7B · Layer 19 · Five answers per context",
        title="Leave-one-setting-out prediction",
        kicker_y=1.24,
        title_y=1.11,
    )
    handles = [
        Line2D(
            [],
            [],
            color=ROLES[role].color,
            marker=ROLES[role].marker,
            linewidth=6,
            label=ROLES[role].label,
        )
        for _, _, role in MODELS
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2)
    fig.text(
        0.10,
        0.025,
        "Train on the other five settings. Bars: mean R²; marks: five held-out folds.",
        color=INK,
        ha="left",
    )
    saved = save_c2a_figure(
        fig,
        stem,
        title="Leave-one-setting-out prediction",
        subject=data["evaluation"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    meta = {
        "sources": data["sources"],
        "record": saved["record"],
        "source_data_sha256": hashlib.sha256(
            stem.with_suffix(".data.json").read_bytes()
        ).hexdigest(),
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "uncertainty": "Five individual fold scores; no confidence interval is implied",
        "baselines_and_retrieval": "Every fold's source-only identity+bias, cosine and "
        "Euclidean retrieval, pool sizes and chance levels are retained in the data sidecar",
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print(saved["pdf"])


def main():
    """Build a standalone LOSO plot and optionally refresh manuscript plot data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=REPO / "eval_results/issue_2054/section44_k5"
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures/issue_2054/k5_loso")
    parser.add_argument("--manuscript-dir", type=Path)
    args = parser.parse_args()
    sources, provenance = read_sources(args.source_dir)
    data, manuscript = summarize(sources, provenance)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "leave_one_setting_out"
    stem.with_suffix(".data.json").write_text(json.dumps(data, indent=2) + "\n")
    if args.manuscript_dir is not None:
        target = args.manuscript_dir / "figures/paper/c4_shared_speakers.data.json"
        target.write_text(json.dumps(manuscript, indent=2) + "\n")
    render(data, stem)


if __name__ == "__main__":
    main()
