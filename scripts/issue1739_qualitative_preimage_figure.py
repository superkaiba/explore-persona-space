"""Render verified excerpts from each persona preimage's top 100 contexts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import textwrap

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "eval_results/issue_1739/qualitative_top100_20260918/figure_inputs.json"
OUTPUT = ROOT / "figures/issue_1739/c5_preimage_qualitative_top100"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt(text, parts):
    """Accept only literal, ordered source spans and mark every omission."""
    cursor = 0
    for part in parts:
        start = text.find(part, cursor)
        if start < 0:
            raise ValueError(f"Excerpt not present in source: {part!r}")
        cursor = start + len(part)
    value = " … ".join(parts)
    if text.find(parts[0]) > 0:
        value = "… " + value
    if cursor < len(text):
        value += " …"
    return "“" + value + "”"


def verified_examples(spec):
    source = ROOT / spec["source"]
    if digest(source) != spec["source_sha256"]:
        raise ValueError("The source snapshot changed")
    match = re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>', source.read_text(), re.S)
    if match is None:
        raise ValueError("Dashboard data block missing")
    data = json.loads(match.group(1))
    records = {r["ci"]: r for r in data["records"]}
    if data["pool"]["n_unique_candidate_prompts"] != spec["selection"]["pool_size"]:
        raise ValueError("Candidate population mismatch")
    groups = []
    for group in spec["groups"]:
        behavior = group["behavior"]
        members = sorted(
            (r for r in records.values() if r["ranks"][behavior]["preimage_cosine"]["rank"] <= 100),
            key=lambda r: r["ranks"][behavior]["preimage_cosine"]["rank"],
        )
        if len(members) != 100:
            raise ValueError("Incomplete top-100 coverage")
        entries = []
        for example in group["examples"]:
            record = records[example["ci"]]
            rank = record["ranks"][behavior]["preimage_cosine"]["rank"]
            if not 1 <= rank <= 100:
                raise ValueError("Example outside requested top 100")
            rendered = dict(
                **example,
                rank=rank,
                prompt_excerpt=excerpt(record["prompt"], example["prompt_parts"]),
                prompt_sha256=hashlib.sha256(record["prompt"].encode()).hexdigest(),
                response_sha256=record["response_sha256"],
            )
            if "response_parts" in example:
                rendered["response_excerpt"] = excerpt(
                    record["response"], example["response_parts"]
                )
            entries.append(rendered)
        groups.append(
            dict(**group, rendered_examples=entries, top100_ids=[r["ci"] for r in members])
        )
    return groups


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    source = Path(cfg.get("input", str(INPUT)))
    target = Path(cfg.get("output", str(OUTPUT)))
    spec = json.loads(source.read_text())
    groups = verified_examples(spec)
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.72)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    for x, label in [
        (0, "DIRECTION"),
        (0.20, "PROMPT CATEGORIES"),
        (0.45, "EXAMPLE EXCERPTS"),
    ]:
        ax.text(x, 0.985, label, va="top", fontweight=700, color=MUTED)
    row_tops = [0.925, 0.700, 0.335]
    separators = [0.95, 0.727, 0.362, 0.010]
    artists = []
    for group, y in zip(groups, row_tops, strict=True):
        ax.text(0, y, textwrap.fill(group["label"], 15), va="top", fontweight=700, color=INK)
        categories = "\n\n".join(textwrap.fill(s, 22) for s in group["categories"])
        ax.text(0.20, y, categories, va="top", color=ROLES["linear"].color, linespacing=1.15)
        blocks = []
        for item in group["rendered_examples"]:
            value = f"Rank {item['rank']}: {item['prompt_excerpt']}"
            block = textwrap.fill(value, 56)
            if "response_excerpt" in item:
                block += "\n" + textwrap.fill("Saved answer: " + item["response_excerpt"], 56)
            if "reference_note" in item:
                block += "\n" + textwrap.fill(item["reference_note"], 56)
            blocks.append(block)
        artists.append(ax.text(0.45, y, "\n\n".join(blocks), va="top", color=INK, linespacing=1.12))
    for y in separators:
        ax.plot([0, 1], [y, y], color=MUTED, linewidth=0.8, alpha=0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, lower in zip(artists, separators[1:], strict=True):
        bounds = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        if bounds.y0 < lower + 0.005 or bounds.x1 > 1.005:
            raise ValueError(f"Example text overflows its row: {bounds}")
    saved = save_c2a_figure(
        fig,
        target,
        title="Prompt categories among the top 100 preimage contexts",
        subject=spec["interpretation"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    meta = dict(
        input_sha256=digest(source),
        script_sha256=digest(Path(__file__)),
        source_sha256=spec["source_sha256"],
        selection=spec["selection"],
        interpretation=spec["interpretation"],
        groups=groups,
        render=saved["record"],
        output_sha256={key: digest(saved[key]) for key in ["pdf", "png", "grayscale"]},
    )
    target.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    plt.close(fig)
    print(json.dumps({"output": str(target), "sha256": meta["output_sha256"]}))


if __name__ == "__main__":
    main()
