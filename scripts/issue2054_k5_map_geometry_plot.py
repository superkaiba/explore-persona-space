"""Plot and summarize K5 shared/individual operator geometry."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
)
from scripts import issue2054_k5_loso_calibration as base


def matrix(ax, values, labels, letter, checkpoint, title):
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if values[i, j] > 0.65 else INK,
            )
    panel_header(ax, letter, checkpoint, title, kicker_y=1.12, title_y=1.03)
    return image


def summarize(data):
    result = {}
    for model in data["models"]:
        fs = model["folds"]
        names = data["labels"]
        raw_cosine = np.mean([f["raw_operator_cosine"] for f in fs], axis=0)
        pred_cosine = np.mean([f["centered_pooled_prediction_cosine"] for f in fs], axis=0)
        records = []
        for i, label in enumerate(names):
            spectrum = {
                key: float(np.mean([f["spectra"][i][key] for f in fs]))
                for key in ["r90", "r95", "stable_rank", "participation_ratio", "numerical_rank"]
            }
            records.append(
                {
                    "label": label,
                    **spectrum,
                    "r90_range": [
                        min(f["spectra"][i]["r90"] for f in fs),
                        max(f["spectra"][i]["r90"] for f in fs),
                    ],
                    "effective_df": float(np.mean([f["ridge"][i]["dof"] for f in fs])),
                    "n_train": float(np.mean([f["ridge"][i]["n_train"] for f in fs])),
                    "cosine_with_shared": float(raw_cosine[i, -1]),
                    "prediction_cosine_with_shared": float(pred_cosine[i, -1]),
                }
            )
        r2 = np.mean([[e["r2"] for e in f["evaluations"]] for f in fs], axis=0)
        retrieval = np.mean(
            [[e["retrieval"]["euclidean_top1"] for e in f["evaluations"]] for f in fs], axis=0
        )
        # Rows target, columns source; restrict to character-only off-diagonal.
        char_pairs = [(i, j) for i in range(2, 6) for j in range(2, 6) if i != j]
        result[model["model"]] = {
            "maps": records,
            "raw_operator_cosine": raw_cosine.tolist(),
            "centered_prediction_cosine": pred_cosine.tolist(),
            "transfer_r2_target_by_source": r2.tolist(),
            "transfer_euclidean_top1_target_by_source": retrieval.tolist(),
            "single_character_transfer_r2_mean": float(np.mean([r2[i, j] for i, j in char_pairs])),
            "single_character_transfer_r2_range": [
                float(min(r2[i, j] for i, j in char_pairs)),
                float(max(r2[i, j] for i, j in char_pairs)),
            ],
            "single_character_transfer_top1_mean": float(
                np.mean([retrieval[i, j] for i, j in char_pairs])
            ),
            "character_pair_operator_cosine_mean": float(
                np.mean([raw_cosine[i, j] for i, j in char_pairs])
            ),
            "character_pair_operator_cosine_range": [
                float(min(raw_cosine[i, j] for i, j in char_pairs)),
                float(max(raw_cosine[i, j] for i, j in char_pairs)),
            ],
        }
    return result


def plot(data, summary, out):
    set_c2a_style()
    labels = data["labels"]
    fig, fraction = c2a_figure("full", 1.02)
    axes = fig.subplots(2, 2, gridspec_kw={"height_ratios": [0.9, 1.3]})
    for column, model in enumerate(data["models"]):
        name = model["model"]
        checkpoint = "Instruction-tuned" if name.endswith("-instruct") else "Base"
        fs = model["folds"]
        ranks = np.array([[s["r90"] for s in f["spectra"]] for f in fs])
        mean = ranks.mean(0)
        ax = axes[0, column]
        for i, label in enumerate(labels):
            style = ROLES["linear"] if label == "Shared" else ROLES["control"]
            ax.errorbar(
                mean[i],
                i,
                xerr=[[mean[i] - ranks[:, i].min()], [ranks[:, i].max() - mean[i]]],
                fmt="o" if label == "Shared" else "s",
                color=style.color,
                markerfacecolor=style.color if label == "Shared" else "white",
                capsize=3,
            )
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 650)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_xlabel("Directions capturing 90%\nof coefficient energy")
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", alpha=0.25)
        panel_header(ax, "AB"[column], checkpoint, "Effective map rank", kicker_y=1.13)
        matrix(
            axes[1, column],
            np.array(summary[name]["raw_operator_cosine"]),
            labels,
            "CD"[column],
            checkpoint,
            "Similarity of map coefficients",
        )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.91, hspace=0.48, wspace=0.42)
    saved = save_c2a_figure(
        fig,
        out / "map_rank_similarity",
        title="Shared and individual map geometry",
        subject="Raw-coordinate ridge operators; means and ranges across five conversation folds. Cosine scale 0 to 1.",
        creator=Path(__file__).name,
        include_width=fraction,
    )
    base.atomic_json(
        out / "map_rank_similarity.meta.json", {"render": saved["record"], "summary": summary}
    )
    plt.close(fig)

    fig, fraction = c2a_figure("full", 0.55)
    axes = fig.subplots(1, 2)
    for column, model in enumerate(data["models"]):
        name = model["model"]
        checkpoint = "Instruction-tuned" if name.endswith("-instruct") else "Base"
        matrix(
            axes[column],
            np.array(summary[name]["centered_prediction_cosine"]),
            labels,
            "AB"[column],
            checkpoint,
            "Similarity of centered predictions",
        )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.85, wspace=0.42)
    saved = save_c2a_figure(
        fig,
        out / "map_prediction_similarity",
        title="Agreement on identical held-out contexts",
        subject="Centered within each target setting, equal setting weights, mean across five folds.",
        creator=Path(__file__).name,
        include_width=fraction,
    )
    base.atomic_json(
        out / "map_prediction_similarity.meta.json", {"render": saved["record"], "summary": summary}
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.out / "results.json").read_text())
    if data["status"] != "complete" or len(data["models"]) != 2:
        raise ValueError("requires both checkpoints and all folds")
    summary = summarize(data)
    base.atomic_json(args.out / "summary.json", summary)
    plot(data, summary, args.figures)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
