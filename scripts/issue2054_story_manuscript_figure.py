"""Plot the three-panel speaker figure from completed, pinned results."""

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
    "HELIOS",
    "Wren",
    "Dana",
    "Vex",
    "Assistant (story)",
]
METHODS = [
    ("frozen", "Frozen", "o"),
    ("bias", "+ Bias", "D"),
    ("own", "Own map", "s"),
]
PAIRS = [("story", "characters"), ("chat", "story"), ("plain", "story"), ("story", "chat")]


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


def render(data: dict, output: Path) -> dict:
    """Place joint/separate bars, Instruct transfer bars and turn heatmaps in one row."""
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.55)
    a = fig.add_axes([0.080, 0.31, 0.245, 0.57])
    b = fig.add_axes([0.465, 0.31, 0.245, 0.57])
    for ax, letter, title in [(a, "A", "Separate and shared"), (b, "B", "Transfer (Instruct)")]:
        style_axis(ax, grid_axis="none")
        panel_header(ax, letter, title, kicker_y=1.07)
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
            np.arange(7) + (j - 1) * 0.24,
            values,
            height=0.21,
            color=color,
            edgecolor=instruct_color if hatch else color,
            linewidth=0.8,
            hatch=hatch,
            label=label,
        )
    a.set_yticks(range(7), ["Chat", "Plain", "HELIOS", "Wren", "Dana", "Vex", "Story"])
    a.set_ylim(6.5, -0.5)
    a.set_xlim(0, 0.75)
    a.set_xticks([0, 0.3, 0.6])
    a.set_xlabel(better_label("Held-out $R^2$"))
    a.legend(loc="upper left", bbox_to_anchor=(-0.12, -0.22), borderaxespad=0, handlelength=1.4)
    method_style = {
        "frozen": (instruct_color, None),
        "bias": ("white", "////"),
        "own": ("white", None),
    }
    for j, pair in enumerate(instruct["transfers"]):
        for k, (method, _, _) in enumerate(METHODS):
            stat = pair["metrics"][method]
            mean, folds = stat["mean"], stat["folds"]
            color, hatch = method_style[method]
            b.barh(
                j + (k - 1) * 0.24,
                mean,
                height=0.21,
                color=color,
                edgecolor=instruct_color,
                linewidth=0.8,
                hatch=hatch,
                xerr=[[mean - min(folds)], [max(folds) - mean]],
                error_kw={"ecolor": INK, "capsize": 2, "elinewidth": 0.8, "capthick": 0.8},
            )
    b.set_yticks(
        range(4), ["Story →\ncharacters", "Chat →\nstory", "Plain →\nstory", "Story →\nchat"]
    )
    b.set_ylim(3.5, -0.5)
    endpoints = [
        v
        for pair in instruct["transfers"]
        for method, _, _ in METHODS
        for v in pair["metrics"][method]["folds"]
    ]
    assert min(endpoints) >= -1.0 and max(endpoints) <= 0.75
    b.set_xlim(-1.0, 0.75)
    b.set_xticks([-1.0, 0, 0.5])
    b.axvline(0, color=ROLES["control"].color, linewidth=0.8, zorder=0)
    b.set_xlabel(better_label("Held-out $R^2$"))
    b.legend(
        handles=[
            Patch(
                facecolor=method_style[method][0],
                edgecolor=instruct_color,
                hatch=method_style[method][1],
                label=label,
                linewidth=0.8,
            )
            for method, label, _ in METHODS
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.22),
        borderaxespad=0,
        handlelength=1.4,
    )
    conditions = ["1", "2", "3", "12", "1+2+3"]
    matrices = {}
    for i, (key, label, role) in enumerate(
        [("pretrained", "Base", "base_model"), ("instruct", "Instruct", "post_trained")]
    ):
        ax = fig.add_axes([0.790, 0.645 - i * 0.335, 0.140, 0.205])
        if i == 0:
            panel_header(ax, "C", "Turn transfer", kicker_y=1.38)
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
        ax.set_yticks(range(5), ["1", "2", "3", "12", "1–3"])
        ax.set_ylabel("Train turn")
        ax.set_xticks([0, 5, 11], ["1", "6", "12"])
        if i == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Evaluation turn")
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_ax = fig.add_axes([0.950, 0.31, 0.010, 0.54])
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
    args = parser.parse_args()
    sources = {
        "k5": ROOT / "eval_results/issue_2054/section44_k5/k5_results.json",
        "story": ROOT / "eval_results/issue_2054/assistant_story_k5/results.json",
        "turns": ROOT / "eval_results/issue_2054/manuscript_story_transfer/turns_source.json",
        "shared_seven": ROOT / "eval_results/issue_2054/shared_seven/results.json",
    }
    # SHA_PIN_DOMAIN: BYTES
    k5 = load(sources["k5"], "90309ec95e757ee5bd2e1b941e006858bd24a328656386444b6c55fb43f928f2")
    # SHA_PIN_DOMAIN: BYTES
    story = load(
        # SHA_PIN_DOMAIN: BYTES
        sources["story"],
        "d9f67bd2dfaf0cbd14a97af3234adb943e82bdf6b6150071af2b061817fe8070",
    )
    data = summarize(k5, story, load(sources["turns"]), load(sources["shared_seven"]))
    args.output.mkdir(parents=True, exist_ok=True)
    result = render(data, args.output)
    result["data"] = data
    result["sources"] = {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in sources.items()
    }
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
        "panel_a": "Base own (amber), Instruct own (teal), Instruct seven-setting shared (hatched teal)",
        "panel_b": "Instruct only: frozen (teal), bias (hatched teal), own (outline)",
        "panel_b_error_bars": "Minimum and maximum across five fold means",
        "gridlines": "Removed; panel B retains a zero reference",
        "character_aggregation": "Equal average of four characters within each fold",
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
    print(json.dumps({"figure": str(args.output), "status": "complete"}))


if __name__ == "__main__":
    main()
