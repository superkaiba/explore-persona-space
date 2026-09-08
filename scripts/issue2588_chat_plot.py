"""Render the chat-only rank comparison from verified final map JSONs; no compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis import c2a_plot_style as style
from explore_persona_space.atomic_io import write_json_atomic
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MANIFEST_REVISION = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"


def digest(path: Path) -> str:
    """Hash the exact inputs and exported artifacts without loading their whole bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_maps(source: Path) -> list[dict]:
    """Require two complete non-pilot maps and internally consistent plotted values."""
    maps = [json.loads((source / f"rank_{arm}.json").read_text()) for arm in ("a", "b")]
    for arm, position, rec in zip(("a", "b"), ("prompt_last", "cot_boundary"), maps, strict=True):
        identity = rec["provenance"]["run_identity"]
        curve = np.asarray(rec["rank_curve"]["test_r2"], dtype=float)
        validation = np.asarray(rec["rank_curve"]["validation_r2"], dtype=float)
        rank = rec["rank"]
        if (
            rec["cell"] != f"q3_8b_{arm}"
            or rec["input_position"] != position
            or rec["dimension"] != 4096
            or curve.shape != (4097,)
            or not np.isfinite(curve).all()
            or validation.shape != (4097,)
            or not np.isfinite(validation).all()
            or type(rank) is not int
            or not 0 <= rank <= 4096
            or identity["smoke"] is not False
            or identity["surface"] != "generic"
            or identity["model_id"] != "Qwen/Qwen3-8B"
            or identity["model_revision"] != MODEL_REVISION
            or identity["manifest_revision"] != MANIFEST_REVISION
            or rec["provenance"]["durable_verification"]["content_verified"] is not True
            or not all(
                np.isfinite(rec[k])
                for k in (
                    "full_test_r2",
                    "full_validation_r2",
                    "selected_rank_test_r2",
                    "selected_rank_validation_r2",
                    "validation_r2_threshold",
                )
            )
            or abs(curve[rank] - rec["selected_rank_test_r2"]) > 1e-10
            or abs(curve[-1] - rec["full_test_r2"]) > 1e-4
            or abs(validation[rank] - rec["selected_rank_validation_r2"]) > 1e-10
            or abs(validation[-1] - rec["full_validation_r2"]) > 1e-4
            or abs(rec["validation_r2_threshold"] - (1 - 1.10 * (1 - rec["full_validation_r2"])))
            > 1e-10
            or any(rec["realized_rows"][s] <= 0 for s in ("train_10k", "val_400", "test_1000"))
        ):
            raise ValueError(f"Incomplete, inconsistent, or out-of-scope map {arm}")
        qualifying = np.flatnonzero(validation >= rec["validation_r2_threshold"] - 1e-12)
        if not len(qualifying) or rank != int(qualifying[0]):
            raise ValueError(f"Map {arm} does not use the minimum validation-selected rank")
    keys = ("run_id", "model_revision", "manifest_revision", "source_sha")
    if any(
        maps[0]["provenance"]["run_identity"][k] != maps[1]["provenance"]["run_identity"][k]
        for k in keys
    ):
        raise ValueError("Maps differ in the shared scientific recipe")
    return maps


def render(source: Path, stem: Path) -> dict:
    """Show the complete held-out rank curves and validation-selected ranks."""
    maps = load_maps(source)
    style.set_c2a_style()
    fig, fraction = style.c2a_figure("wide", aspect=0.56)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.88)
    style.style_axis(ax)
    for rec in maps:
        series = style.CONTEXT_STATE_STYLES[rec["input_position"]]
        values = np.asarray(rec["rank_curve"]["test_r2"])
        ax.plot(
            np.arange(len(values)),
            values,
            color=series.color,
            marker=series.marker,
            markevery=[0, 1, 4, 16, 64, 256, 1024, 4096],
            markersize=5,
            linestyle=style.metric_style("r2")["linestyle"],
            label=series.label,
        )
        ax.plot(
            rec["rank"],
            values[rec["rank"]],
            color=series.color,
            marker=series.marker,
            markersize=11,
            markeredgecolor=style.INK,
            linestyle="none",
        )
    ax.set_xscale("symlog", base=2, linthresh=1)
    ticks = [0, 1, 4, 16, 64, 256, 1024, 4096]
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.set_xlim(0, 4600)
    ax.set_xlabel("Map rank")
    ax.set_ylabel(style.better_label(style.METRIC_LABELS["r2"]))
    ax.set_title("Predictability across map ranks", loc="left", pad=15)
    ax.legend(loc="lower right")
    exported = style.save_c2a_figure(
        fig,
        stem,
        title="Qwen3-8B chat-data mapping rank",
        subject="Held-out rank curves; large markers use validation-only rank selection",
        creator=__file__,
        include_width=fraction,
    )
    plt.close(fig)
    caption = (
        "Held-out R² versus map rank for prompt states without thinking and end-of-thought "
        "states with thinking. Large markers indicate the smallest rank retaining validation "
        "SSE within 10% of its full map. The rank axis is logarithmic above 1, retaining rank 0. "
        "Error bars: none; one fitted metamodel per condition. Qwen3-8B; separately "
        "validation-selected layers; LMSYS-Chat-1M prompts; the fixed original row split. "
        "Validation and test share 13 exact prompt strings (24 validation and 60 test rows "
        "before exclusions); training has no exact-text overlap with either. Each condition "
        "predicts its own on-policy generated answers (temperature 1, top-p .95, seed 42); "
        "this is not a matched-answer causal comparison. Exact counts and source revisions "
        "are recorded in the sidecar."
    )
    metadata = {
        "schema": "issue2588_chat_rank_figure_v1",
        "caption": caption,
        "metadata": as_metadata_dict(git_provenance(ROOT, __file__), phase="rank-plot"),
        "source_files": {
            str(source / f"rank_{a}.json"): digest(source / f"rank_{a}.json") for a in ("a", "b")
        },
        "style_sha256": digest(Path(style.__file__)),
        "render": exported["record"],
        "maps": maps,
        "outputs": {
            key: {"path": str(exported[key]), "sha256": digest(exported[key])}
            for key in ("pdf", "png", "grayscale")
        },
    }
    write_json_atomic(stem.with_suffix(".meta.json"), metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Require explicit final inputs and output location, never synthesize missing cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="qwen3_chat_rank")
    args = parser.parse_args(argv)
    if Path(args.stem).name != args.stem or not args.stem:
        parser.error("stem must be one safe filename component")
    render(args.source_dir, args.out_dir / args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
