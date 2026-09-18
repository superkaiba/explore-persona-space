"""Plot the three-panel speaker figure from completed, pinned results."""

# ruff: noqa: E402
# Apply the shared-VM thread caps before importing numerical libraries.

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    ROLES,
    better_label,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("qwen2.5-7b", "Base", "base_model"), ("qwen2.5-7b-instruct", "Instruct", "post_trained")]
NAMES = [
    "Assistant (chat)",
    "Assistant (plain)",
    "Helios",
    "Wren",
    "Dana",
    "Vex",
    "Assistant (story)",
]
METHODS = [
    ("frozen", "Frozen", "o"),
    ("own", "Target-trained", "s"),
]
PAIRS = [("story", "characters"), ("chat", "story"), ("plain", "story"), ("story", "chat")]
# Plain-text comparisons are archived but omitted from the manuscript. The
# shared map retains its original seven-setting training provenance.
DISPLAY_INDICES = (0, 2, 3, 4, 5, 6)
DISPLAY_PAIRS = [
    ("character", "other_characters"),
    ("chat+character", "other_characters"),
    ("story", "characters"),
    ("chat", "story"),
    ("story", "chat"),
]
# Bar geometry in row units. The row pitch shrinks with the shorter canvas, so
# the bars and the within-row offset shrink with it and the gutter between rows
# stays readable.
BAR_HEIGHT = 0.20
BAR_OFFSET = 0.23
# Two-line speaker labels set at the default 1.2 line spacing leave only about
# 2 pt between the adjacent "Assistant (chat)" and "Assistant (plain text)"
# rows, which is what pinned the old panel height. Single line spacing frees
# roughly 0.1 in per label and lets the row pitch come down without a collision.
TICK_LINESPACING = 1.0


def load(path: Path, sha: str | None = None) -> dict:
    """Read a complete artifact, optionally checking its published digest."""
    blob = path.read_bytes()
    if sha is not None and hashlib.sha256(blob).hexdigest() != sha:
        raise ValueError(f"Source hash mismatch: {path}")
    return json.loads(blob)


def setting(cell: str) -> str:
    """Resolve only the explicitly supported source/target names."""
    if cell.startswith("char_"):
        return cell.split("__")[0][5:]
    for token, name in [
        ("__chat__", "chat"),
        ("__bare_text__", "plain"),
        ("__attrib_quoted__", "story"),
    ]:
        if token in cell:
            return name
    if cell.startswith("shared_six__"):
        return "shared"
    raise ValueError(cell)


def summarize(k5: dict, story: dict, turns: dict, shared_seven: dict) -> dict:
    """Keep exact fold metrics, all controls and equal-character averages."""
    if story["status"] != "complete":
        raise ValueError("Assistant-story analysis is incomplete")
    old = {r["cell"]: r for r in k5["results"] if r["k_rollouts"] == 5 and r["cohort"] == "all"}
    order = ["chat", "plain", "helios", "wren", "dana", "vex"]
    result = {"models": {}, "turns": turns["panel_turns"]}
    for model, _, _ in MODELS:
        rows = {setting(key): value for key, value in old.items() if key.endswith("__" + model)}
        new = next(m for m in story["models"] if m["model"] == model)
        own = [rows[s]["r2_mean"]["own"] for s in order]
        own.append(float(np.mean([f["own"]["r2"] for f in new["own_folds"]])))
        shared = [rows[s]["r2_mean"]["pooled"] for s in order]
        pooled = [t for t in new["transfers"] if setting(t["source"]) == "shared"]
        assert sorted(t["fold"] for t in pooled) == list(range(5))
        shared.append(float(np.mean([t["metrics"]["frozen"]["r2"] for t in pooled])))
        pairs = []
        for source, target in PAIRS:
            selected = [
                t
                for t in new["transfers"]
                if setting(t["source"]) == source
                and (
                    setting(t["target"]) == target
                    or (target == "characters" and t["target"].startswith("char_"))
                )
            ]
            expected = 4 if target == "characters" else 1
            assert len(selected) == 5 * expected
            values = {}
            for method in selected[0]["metrics"]:
                fold_values = []
                for fold in range(5):
                    group = [r for r in selected if r["fold"] == fold]
                    assert len(group) == expected
                    assert len({r["target"] for r in group}) == expected
                    fold_values.append(float(np.mean([r["metrics"][method]["r2"] for r in group])))
                values[method] = {"mean": float(np.mean(fold_values)), "folds": fold_values}
            pairs.append(
                {"source": source, "target": target, "metrics": values, "source_records": selected}
            )
        result["models"][model] = {
            "own": own,
            "shared": shared,
            "recovery": (np.array(shared) / own).tolist(),
            "transfers": pairs,
        }
    if shared_seven["status"] != "complete" or shared_seven["training_settings"] != 7:
        raise ValueError("Seven-setting pooled fit is incomplete")
    if shared_seven["model"] != "qwen2.5-7b-instruct" or len(shared_seven["rows"]) != 35:
        raise ValueError("Unexpected shared-fit model or coverage")
    seven = {setting(row["cell"]): row for row in shared_seven["summary"]}
    order = ["chat", "plain", "helios", "wren", "dana", "vex", "story"]
    instruct = result["models"]["qwen2.5-7b-instruct"]
    np.testing.assert_allclose(
        [seven[key]["r2_mean"]["own"] for key in order], instruct["own"], atol=1e-12, rtol=0
    )
    instruct["shared"] = [seven[key]["r2_mean"]["shared"] for key in order]
    instruct["recovery"] = [seven[key]["shared_over_own"] for key in order]
    result["shared_seven"] = shared_seven
    return result


def add_character_transfers(data: dict, sources: dict[str, Path]) -> None:
    """Summarize character and chat/plain transfers from the same saved fits."""
    characters = ("HELIOS", "Wren", "Dana", "Vex")
    for model, path in sources.items():
        geometry = load(path)
        if geometry["model"] != model or [f["fold"] for f in geometry["folds"]] != list(range(5)):
            raise ValueError(f"Unexpected character-transfer coverage: {path}")
        records = []
        for fold in geometry["folds"]:
            labels = fold["labels"]
            targets = {r["target"]: r for r in fold["evaluations"]}
            if labels != ["Chat", "Plain", *characters, "Shared"]:
                raise ValueError("Unexpected source-map ordering")
            for source in characters:
                for target in characters:
                    if source == target:
                        continue
                    row = targets[target]
                    i, j = labels.index(source), labels.index(target)
                    records.append(
                        {
                            "source": source,
                            "target": target,
                            "fold": fold["fold"],
                            "frozen": row["r2"][i],
                            "own": row["r2"][j],
                            "source_identity_bias": row["source_identity_bias_r2"][i],
                            "retrieval_pool": row["retrieval"]["pool_size"],
                            "chance_top1": row["retrieval"]["chance_top1"],
                            "euclidean_top1": row["retrieval"]["euclidean_top1"][i],
                            "cosine_top1": row["retrieval"]["cosine_top1"][i],
                        }
                    )
        if len(records) != 60:
            raise ValueError("Expected twelve character pairs in each of five folds")
        metrics = {}
        for method in ("frozen", "own", "source_identity_bias"):
            folds = [
                float(np.mean([r[method] for r in records if r["fold"] == f])) for f in range(5)
            ]
            metrics[method] = {"mean": float(np.mean(folds)), "folds": folds}
        np.testing.assert_allclose(
            metrics["own"]["mean"],
            np.mean(data["models"][model]["own"][2:6]),
            rtol=0,
            atol=1e-12,
        )
        existing = [
            r
            for r in data["models"][model]["transfers"]
            if r["source"] not in {"character", "chat+character"}
        ]
        data["models"][model]["transfers"] = [
            {
                "source": "character",
                "target": "other_characters",
                "metrics": metrics,
                "source_records": records,
                "aggregation": (
                    "Equal mean of twelve directed pairs within each fold, then five folds"
                ),
            },
            *existing,
        ]
        framing_transfers = []
        for source, target in [("Chat", "Plain"), ("Plain", "Chat")]:
            framing_records = []
            for fold in geometry["folds"]:
                row = next(r for r in fold["evaluations"] if r["target"] == target)
                i, j = fold["labels"].index(source), fold["labels"].index(target)
                framing_records.append(
                    {
                        "source": source,
                        "target": target,
                        "fold": fold["fold"],
                        "frozen": row["r2"][i],
                        "own": row["r2"][j],
                        "source_identity_bias": row["source_identity_bias_r2"][i],
                        "retrieval_pool": row["retrieval"]["pool_size"],
                        "chance_top1": row["retrieval"]["chance_top1"],
                        "euclidean_top1": row["retrieval"]["euclidean_top1"][i],
                        "cosine_top1": row["retrieval"]["cosine_top1"][i],
                    }
                )
            framing_metrics = {}
            for method in ("frozen", "own", "source_identity_bias"):
                values = [r[method] for r in framing_records]
                framing_metrics[method] = {"mean": float(np.mean(values)), "folds": values}
            np.testing.assert_allclose(
                framing_metrics["own"]["mean"],
                data["models"][model]["own"][["Chat", "Plain"].index(target)],
                rtol=0,
                atol=1e-12,
            )
            framing_transfers.append(
                {
                    "source": source.lower(),
                    "target": target.lower(),
                    "metrics": framing_metrics,
                    "source_records": framing_records,
                    "aggregation": "Mean across five conversation folds",
                }
            )
        data["models"][model]["chat_plain_transfers"] = framing_transfers
        # Keep the reverse direction in the archive and appendix. Insert the
        # displayed direction after the two story-speaker rows, idempotently.
        rows = data["models"][model]["transfers"]
        rows = [r for r in rows if (r["source"], r["target"]) != ("chat", "plain")]
        data["models"][model]["transfers"] = rows[:2] + framing_transfers[:1] + rows[2:]


def add_joint_character_transfers(data: dict, path: Path) -> None:
    """Average saved chat-plus-character maps over their unseen character targets."""
    results = load(path, "1ea49e91af068bd2c2729e5c6ed6c359d0e6ae0ec92c1798f054a376b73ee58d")
    if results["status"] != "complete":
        raise ValueError("Joint-source transfer results are incomplete")
    characters = ("helios", "wren", "dana", "vex")
    expected = {(s, t) for s in characters for t in characters if s != t}
    for model, _, _ in MODELS:
        panels = [
            r
            for r in results["panels"]
            if r["model"] == model
            and r["regime"].startswith("assistant_plus_char_")
            and r["cell"].startswith("char_")
        ]
        pairs = {
            (r["regime"].removeprefix("assistant_plus_char_"), setting(r["cell"])) for r in panels
        }
        if len(panels) != 12 or pairs != expected:
            raise ValueError(f"Expected all twelve joint-source/unseen-character pairs: {model}")
        single = data["models"][model]["transfers"][0]
        own_reference = {
            (r["target"].lower(), r["fold"]): r["own"] for r in single["source_records"]
        }
        records = []
        for panel in panels:
            source = panel["regime"].removeprefix("assistant_plus_char_")
            target = setting(panel["cell"])
            expected_sources = {
                f"conversation_paired_stories_assistant__on_policy__chat__{model}",
                f"char_{source}__on_policy__attrib_quoted__{model}",
            }
            if set(panel["source_cells"]) != expected_sources or panel["cell"] in expected_sources:
                raise ValueError("Joint map must train on chat and one non-target character")
            if sorted(f["fold"] for f in panel["folds"]) != list(range(5)):
                raise ValueError("Joint-source transfer needs five conversation folds")
            for fold in panel["folds"]:
                audit = fold["audit"]
                if (
                    fold["status"] != "complete"
                    or audit["test_overlap_source"] != 0
                    or audit["target_labels_used_for_source_map"] is not False
                    or audit["excluded_conversation_fold"] != fold["fold"]
                    or set(audit["source_settings"]) != expected_sources
                ):
                    raise ValueError("Joint-source conversation holdout audit failed")
                map_record = fold["source_map"]
                if map_record["sha256"] != results["maps"][map_record["path"]]:
                    raise ValueError("Joint-source map checksum differs from the archive")
                frozen = fold["metrics"]["frozen"]
                np.testing.assert_allclose(
                    fold["own"]["r2"], own_reference[(target, fold["fold"])], rtol=0, atol=1e-12
                )
                records.append(
                    {
                        "source": source,
                        "source_cells": panel["source_cells"],
                        "target": target,
                        "fold": fold["fold"],
                        "frozen": frozen["r2"],
                        "own": fold["own"]["r2"],
                        "source_identity_bias": fold["metrics"]["source_identity_bias"]["r2"],
                        "retrieval_pool": frozen["retrieval_pool"],
                        "chance_top1": frozen["chance_top1"],
                        "euclidean_top1": frozen["retrieval"]["euclidean"]["acc_at_k"]["1"],
                        "cosine_top1": frozen["retrieval"]["cosine"]["acc_at_k"]["1"],
                        "source_map_sha256": map_record["sha256"],
                        "audit": audit,
                    }
                )
        metrics = {}
        for method in ("frozen", "own", "source_identity_bias"):
            folds = [
                float(np.mean([r[method] for r in records if r["fold"] == f])) for f in range(5)
            ]
            if not np.isfinite(folds).all():
                raise ValueError("Nonfinite joint-source transfer score")
            metrics[method] = {"mean": float(np.mean(folds)), "folds": folds}
        rows = [r for r in data["models"][model]["transfers"] if r["source"] != "chat+character"]
        data["models"][model]["transfers"] = (
            rows[:1]
            + [
                {
                    "source": "chat+character",
                    "target": "other_characters",
                    "metrics": metrics,
                    "source_records": records,
                    "aggregation": (
                        "Equal mean of four source-character choices and their three unseen "
                        "character targets within each fold, then five folds"
                    ),
                }
            ]
            + rows[1:]
        )


def compress_negative(values):
    """Keep positive R2 linear and show negative values at one quarter scale."""
    values = np.asarray(values)
    return np.where(values < 0, values / 4, values)


def expand_negative(values):
    """Invert the continuous piecewise display transform for Matplotlib ticks."""
    values = np.asarray(values)
    return np.where(values < 0, values * 4, values)


def render(data: dict, output: Path) -> dict:
    """Place joint/separate bars, Instruct transfer bars and turn heatmaps in one row."""
    set_c2a_style()
    # Reserve a separate legend column above each bar panel. Scale the vertical
    # coordinates to retain the existing physical panel heights and text sizes.
    fig, fraction = c2a_figure("full", aspect=0.47)
    vertical_scale = 0.42 / 0.47
    a = fig.add_axes([0.105, 0.145 * vertical_scale, 0.215, 0.655 * vertical_scale])
    b = fig.add_axes([0.510, 0.145 * vertical_scale, 0.200, 0.655 * vertical_scale])
    for ax, letter, title in [(a, "A", "Separate and shared"), (b, "B", "Transfer (Instruct)")]:
        style_axis(ax, grid_axis="none")
        panel_header(ax, letter, title, kicker_y=1.045)
    base_color, instruct_color = ROLES["base_model"].color, ROLES["post_trained"].color
    base = data["models"]["qwen2.5-7b"]
    instruct = data["models"]["qwen2.5-7b-instruct"]
    bars = [
        (base["own"], base_color, None, "Base: separate"),
        (instruct["own"], instruct_color, None, "Instruct: separate"),
        (instruct["shared"], "white", "////", "Instruct: shared"),
    ]
    for j, (values, color, hatch, label) in enumerate(bars):
        a.barh(
            np.arange(len(DISPLAY_INDICES)) + (j - 1) * BAR_OFFSET,
            [values[i] for i in DISPLAY_INDICES],
            height=BAR_HEIGHT,
            color=color,
            edgecolor=instruct_color if hatch else color,
            linewidth=0.8,
            hatch=hatch,
            label=label,
        )
    a.set_yticks(
        range(6),
        [
            "Assistant\n(chat)",
            "Helios",
            "Wren",
            "Dana",
            "Vex",
            "Assistant\n(story)",
        ],
    )
    for tick in a.get_yticklabels():
        tick.set_linespacing(TICK_LINESPACING)
    a.set_ylim(5.5, -0.5)
    a.set_xlim(0, 0.75)
    a.set_xticks([0, 0.3, 0.6])
    a.set_xlabel(better_label("Held-out $R^2$"))
    fit_handles, fit_labels = a.get_legend_handles_labels()
    # Panel A uses hue for model identity. Panel B uses neutral fill/outline for
    # transfer method, avoiding a second meaning for the same teal bars.
    transfer_color = INK
    method_style = {
        "frozen": (transfer_color, None),
        "own": ("white", None),
    }
    expected_pairs = [
        ("character", "other_characters"),
        ("chat+character", "other_characters"),
        PAIRS[0],
        ("chat", "plain"),
        *PAIRS[1:],
    ]
    if [(r["source"], r["target"]) for r in instruct["transfers"]] != expected_pairs:
        raise ValueError("Unexpected transfer-panel rows")
    displayed_transfers = [
        r for r in instruct["transfers"] if (r["source"], r["target"]) in DISPLAY_PAIRS
    ]
    assert [(r["source"], r["target"]) for r in displayed_transfers] == DISPLAY_PAIRS
    for j, pair in enumerate(displayed_transfers):
        for k, (method, _, _) in enumerate(METHODS):
            stat = pair["metrics"][method]
            mean, folds = stat["mean"], stat["folds"]
            color, hatch = method_style[method]
            b.barh(
                j + (k - 0.5) * BAR_OFFSET,
                mean,
                height=BAR_HEIGHT,
                color=color,
                edgecolor=transfer_color,
                linewidth=0.8,
                hatch=hatch,
                xerr=[[max(0, mean - min(folds))], [max(0, max(folds) - mean)]],
                error_kw={"ecolor": INK, "capsize": 2, "elinewidth": 0.8, "capthick": 0.8},
            )
    b.set_yticks(
        range(5),
        [
            "One character\n→ other characters",
            "Chat + one character\n→ other characters",
            "Assistant (story)\n→ characters",
            "Assistant (chat)\n→ assistant (story)",
            "Assistant (story)\n→ assistant (chat)",
        ],
    )
    for tick in b.get_yticklabels():
        tick.set_linespacing(TICK_LINESPACING)
    b.set_ylim(4.5, -0.5)
    endpoints = [
        v
        for pair in displayed_transfers
        for method, _, _ in METHODS
        for v in pair["metrics"][method]["folds"]
    ]
    assert min(endpoints) >= -1.0 and max(endpoints) <= 0.75
    b.set_xscale("function", functions=(compress_negative, expand_negative))
    b.set_xlim(-1.0, 0.75)
    b.set_xticks([-0.8, 0, 0.3, 0.6])
    b.axvline(0, color=ROLES["control"].color, linewidth=0.8, zorder=0)
    b.set_xlabel(better_label("Held-out $R^2$"))
    method_handles = [
        Patch(
            facecolor=method_style[method][0],
            edgecolor=transfer_color,
            hatch=method_style[method][1],
            label=label,
            linewidth=0.8,
        )
        for method, label, _ in METHODS
    ]
    # Each legend stays above its own panel, including the longest label.
    for x0, heading, handles, labels in [
        (0.105, "Model and fit", fit_handles, fit_labels),
        (0.510, "Transfer method", method_handles, [m[1] for m in METHODS]),
    ]:
        legend_kicker(fig, x0, 0.978, heading)
        fig.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(x0 - 0.001, 0.945),
            ncol=1,
            frameon=False,
            handlelength=1.3,
            handletextpad=0.4,
            columnspacing=0.9,
            borderaxespad=0.0,
            labelspacing=0.3,
        )
    conditions = ["1", "2", "3", "12", "1+2+3"]
    matrices = {}
    for i, (key, label, role) in enumerate(
        [("pretrained", "Base", "base_model"), ("instruct", "Instruct", "post_trained")]
    ):
        # The old 0.13-fraction gap between the two heatmaps was about twice what
        # the "Instruct" title needs. Halving it keeps both maps at their published
        # printed height inside the shorter canvas.
        ax = fig.add_axes(
            [0.790, (0.4885 - i * 0.3535) * vertical_scale, 0.140, 0.2685 * vertical_scale]
        )
        if i == 0:
            panel_header(ax, "C", "Turn transfer", kicker_y=1.309)
        cells = data["turns"]["results"]["models"][key]["cells"]
        matrix = []
        for source in conditions:
            selected = sorted(
                [v for v in cells if v["source"] == source and v["method"] == "raw"],
                key=lambda v: v["target_turn"],
            )
            assert [v["target_turn"] for v in selected] == list(range(1, 13))
            matrix.append([v["r2"] for v in selected])
        matrices[key] = matrix
        im = ax.imshow(
            matrix, aspect="auto", vmin=0, vmax=0.6, cmap="cividis", interpolation="nearest"
        )
        ax.set_title(label, loc="left", color=ROLES[role].color, pad=6)
        ax.set_yticks(range(5), ["1", "2", "3", "12", "1–3"])  # noqa: RUF001 - numeric range
        ax.set_ylabel("Train turn")
        ax.set_xticks([0, 5, 11], ["1", "6", "12"])
        if i == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Evaluation turn")
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_ax = fig.add_axes([0.950, 0.145 * vertical_scale, 0.010, 0.600 * vertical_scale])
    cb = fig.colorbar(im, cax=color_ax, ticks=[0, 0.3, 0.6])
    cb.ax.set_title(better_label("$R^2$"), pad=12)
    rendered = save_c2a_figure(
        fig,
        output / "c4_shared_speakers",
        include_width=fraction,
        title="Maps across speakers, framings and turns",
        subject="K5 seven-setting speaker maps and matched-count K1 turn transfer",
        creator="scripts/issue2054_story_manuscript_figure.py",
    )
    plt.close(fig)
    return {"render": rendered["record"], "turn_matrices": matrices}


def main() -> None:
    """Export the manuscript figure and its complete numerical provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/issue_2054/manuscript_story")
    parser.add_argument("--data", type=Path, help="Render the archived manuscript inputs")
    parser.add_argument(
        "--character-results",
        type=Path,
        default=ROOT / "figures/paper/inputs/speaker_character_transfer",
    )
    parser.add_argument(
        "--assistant-results",
        type=Path,
        default=ROOT / "figures/paper/inputs/speaker_character_transfer/assistant_sources.json",
    )
    args = parser.parse_args()
    sources = {
        "k5": ROOT / "eval_results/issue_2054/section44_k5/k5_results.json",
        "story": ROOT / "eval_results/issue_2054/assistant_story_k5/results.json",
        "turns": ROOT / "eval_results/issue_2054/manuscript_story_transfer/turns_source.json",
        "shared_seven": ROOT / "eval_results/issue_2054/shared_seven/results.json",
    }
    if args.data is not None:
        data = load(args.data)
        prior = load(args.data.with_name("c4_shared_speakers.meta.json"))
        if data != prior["data"]:
            raise ValueError("Archived plotting data differs from its provenance sidecar")
        source_records = prior["sources"]
    else:
        # SHA_PIN_DOMAIN: BYTES
        k5 = load(sources["k5"], "90309ec95e757ee5bd2e1b941e006858bd24a328656386444b6c55fb43f928f2")
        # SHA_PIN_DOMAIN: BYTES
        story = load(
            sources["story"], "d9f67bd2dfaf0cbd14a97af3234adb943e82bdf6b6150071af2b061817fe8070"
        )
        data = summarize(k5, story, load(sources["turns"]), load(sources["shared_seven"]))
        source_records = {
            key: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in sources.items()
        }
    character_sources = {model: args.character_results / f"{model}.json" for model, _, _ in MODELS}
    add_character_transfers(data, character_sources)
    for model, path in character_sources.items():
        source_records[f"characters_{model}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    add_joint_character_transfers(data, args.assistant_results)
    source_records["joint_chat_character"] = {
        "path": str(args.assistant_results.relative_to(ROOT)),
        "sha256": hashlib.sha256(args.assistant_results.read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result = render(data, args.output)
    result["data"] = data
    result["sources"] = source_records
    result["git_head_at_render"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result["source_status_at_render"] = subprocess.check_output(
        [
            "git",
            "status",
            "--short",
            "--",
            str(Path(__file__).relative_to(ROOT)),
            *[str(p.relative_to(ROOT)) for p in sources.values()],
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    result["visual_encodings"] = {
        "panel_a": (
            "Base own (amber), Instruct own (teal), Instruct seven-setting shared (hatched teal)"
        ),
        "panel_b": "Instruct only: frozen (charcoal), target-trained (charcoal outline)",
        "displayed_setting_indices": list(DISPLAY_INDICES),
        "displayed_transfer_pairs": DISPLAY_PAIRS,
        "omitted_comparison": "Plain-text assistant; numerical inputs remain archived",
        "shared_training_scope": "Original seven-setting fit, including plain-text training rows",
        "panel_b_axis": {
            "negative_scale": 0.25,
            "positive_scale": 1,
            "limits": [-1, 0.75],
            "omitted_intervals": [],
        },
        "panel_b_error_bars": "Minimum and maximum across five fold means",
        "gridlines": "Removed; panel B retains a zero reference",
        "character_aggregation": (
            "Twelve directed character pairs; twelve chat-plus-character/unseen-target pairs; "
            "or four story-assistant targets within each fold"
        ),
        "turn_heatmaps": {"colormap": "cividis", "limits": [0, 0.6]},
    }
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["style_sha256"] = hashlib.sha256(
        (ROOT / "src/explore_persona_space/analysis/c2a_plot_style.py").read_bytes()
    ).hexdigest()
    result["output_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.output.glob("c4_shared_speakers*")
        if path.suffix in {".pdf", ".png"}
    }
    (args.output / "c4_shared_speakers.meta.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "c4_shared_speakers.data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"figure": str(args.output), "status": "complete"}))


if __name__ == "__main__":
    main()
