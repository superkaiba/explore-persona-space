"""Paper figures: the context-to-answer map across speaker identities and framings.

Results-2 Plot 1 and Plot 2 of the context-answer-map paper, sharing one x-axis
and one label map. Replaces the two-figure stitch of #1345 (chat / plain text /
story) plus #1310 (four story characters), which compared cells measured on
different corpora. Everything here comes from ONE lattice, #2054: one shared
draw of real conversations, 8,000 rows per cell against d = 3,584 (so every
ambient ridge fit is well-posed), layer 19, K = 5 conversation-grouped held-out
folds under the shared production fold map.

Sources (context arm, on-policy answers only, i.e. the measured model writes the
answer itself; the spliced-verbatim-answer cells are deliberately excluded):

* ``eval_results/issue_2054/specialization_ladder/ladder.json`` -- per-cell own-map
  ceiling, the pooled-map read, and the identity-plus-bias baseline.
* ``eval_results/issue_2054/specialization_ladder/loco_pooled.units.jsonl`` -- the
  leave-one-speaker-out pooled read (a universal map that never saw this speaker).

Six x positions, ordered by how close the speaker is to the assistant in the
chat template. The ordering is intuitive, not fitted:

    assistant in the chat template
    assistant in plain text ("User: ... / Assistant: ...")
    HELIOS   (ship AI)      speaking in its own story
    Wren     (helpful)      speaking in its own story
    Dana     (ordinary)     speaking in its own story
    Vex      (villain)      speaking in its own story

Plot 1 (``c4_speaker_ladder``): the map's own-cell ceiling per speaker, base vs
instruct, with the identity-plus-bias baseline and the shuffled-answer null.

Plot 2 (``c4_universal_vs_specialized``): the same x-axis, instruct only,
comparing a map trained on everything against a map trained on this speaker
alone, with the same identity-plus-bias baseline. Two correction rungs of
the pooled map are drawn beside it (user request, 2026-08-25): a per-cell
constant SHIFT and a scalar global RESCALING, both read from the banked
specialization ladder. They separate a pooled map that genuinely misses this
speaker from one that merely sits mis-calibrated against it -- across the five
clean speakers a single constant closes 45-80% of the pooled-to-specialized
gap (median 62%). The two rungs extend the pooled map INDEPENDENTLY, not
cumulatively, so "+ global rescaling" is a gain alone rather than a gain on
top of the shift. Its y-axis is broken: the
baseline runs down to -1.51 while every bar sits between 0.09 and 0.58, so one
linear axis would spend three quarters of its height on the empty space between
them.

The shuffled-answer null (mean of the six cells' banked 97.5th percentiles) is
drawn as a dotted line: the floor a map with no real context-answer pairing
reaches.

The assistant plain-text instruct bars are plotted at their cap-excluded refits
rather than their raw values, in BOTH plots and on BOTH of Plot 2's arms. The
plain-text render carries no end-of-turn token, so 42.5% of that cell's own
generations ran to the 4,096-token cap; #2054 banked the own-map exclusion
refit at 0.390 against the raw 0.209, with a random-removal control at matched
n moving it the other way (0.195), and the pooled companion refit is 0.258
against the raw 0.091. A regeneration at the full context window is in flight
and will replace both substituted values. The substitution is disclosed in the
figures' captions, per Thomas's call (2026-08-25) to draw it as a plain bar
rather than mark it on the canvas.

Usage::

    uv run python scripts/issue2054_paper_r2_figs.py \
        [--out-dir figures/paper] [--style iclr]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    figsize_iclr_full,
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
LADDER_DIR = REPO / "eval_results/issue_2054/specialization_ladder"
LADDER = LADDER_DIR / "ladder.json"
LOCO = LADDER_DIR / "loco_pooled.units.jsonl"

# Reader-facing display names live here and nowhere else: one label map, so a
# rename lands once. Internal cell ids never reach an axis label.
# (character key, framing key, two-line tick label)
POSITIONS: list[tuple[str, str, str]] = [
    ("assistant", "chat", "Assistant\nchat template"),
    ("assistant", "bare_text", "Assistant\nplain text"),
    ("helios", "attrib_quoted", "HELIOS\nship AI"),
    ("wren", "attrib_quoted", "Wren\nhelpful"),
    ("dana", "attrib_quoted", "Dana\nordinary"),
    ("vex", "attrib_quoted", "Vex\nvillain"),
]
PROVENANCE = "on_policy"
ARM = "context"
MODELS = ("base", "instruct")
MODEL_KEY = {"qwen2.5-7b": "base", "qwen2.5-7b-instruct": "instruct"}

IDENTITY_LABEL = "identity + bias baseline"

# Cap-excluded refits of the one truncation-contaminated cell, keyed by
# (character, framing, model). Own-map value: the banked within-cell exclusion
# refit (figures/issue_2054/caphit_censoring_refits.meta.json, bare text /
# capped removed). Pooled value: the companion refit, read from its artifact
# rather than pasted in, so the number in the figure is the number whose
# validation gate passed.
CAP_EXCLUDED = {("assistant", "bare_text", "instruct"): 0.38985271917718867}
CAP_POOLED_ARTIFACT = REPO / "eval_results/issue_2054/caphit_pooled_refit/caphit_pooled_refit.json"


def load_cells() -> dict[tuple[str, str, str], dict]:
    """Index the ladder's context-arm on-policy units by (character, framing, model)."""
    payload = json.loads(LADDER.read_text())
    out: dict[tuple[str, str, str], dict] = {}
    for unit in payload["units"]:
        if unit["arm"] != ARM or unit["provenance"] != PROVENANCE:
            continue
        out[(unit["character"], unit["framing"], unit["model"])] = unit
    return out


def load_loco() -> dict[tuple[str, str, str], float]:
    """Fold-mean held-out R^2 of the pooled map fit WITHOUT this speaker's cells."""
    out: dict[tuple[str, str, str], float] = {}
    for line in LOCO.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["arm"] != ARM or row["condition"] != PROVENANCE:
            continue
        key = (row["speaker"], row["framing"], MODEL_KEY[row["model"]])
        out[key] = float(np.mean([f["loco"]["r2"] for f in row["per_fold"]]))
    return out


def build_series(cells: dict[tuple[str, str, str], dict], loco: dict) -> dict:
    """Pull the six positions x two models out of the indexed cells, fail loud on a miss."""
    series: dict = {"labels": [label for _, _, label in POSITIONS], "n": [], "null": []}
    for model in MODELS:
        series[model] = {
            "ceiling": [],
            "pooled": [],
            "bias": [],
            "gain": [],
            "loco": [],
            "identity": [],
        }
    for character, framing, _ in POSITIONS:
        for model in MODELS:
            key = (character, framing, model)
            if key not in cells:
                raise KeyError(f"no {ARM}/{PROVENANCE} cell for {key}; have {sorted(cells)}")
            unit = cells[key]
            series[model]["ceiling"].append(unit["ceiling_r2"])
            series[model]["pooled"].append(unit["r2"]["pooled"])
            # The two minimal per-cell corrections of the specialization
            # ladder. Each extends the POOLED map INDEPENDENTLY (see
            # issue2054_specialization_ladder's docstring): "+gain" is a
            # scalar gain ALONE, not a gain on top of the bias refit, so the
            # two are siblings rather than a cumulative ladder.
            series[model]["bias"].append(unit["r2"]["bias"])
            series[model]["gain"].append(unit["r2"]["gain"])
            series[model]["identity"].append(unit["aux_r2"]["identity_cell"])
            if key not in loco:
                raise KeyError(f"no LOCO row for {key}; have {sorted(loco)}")
            series[model]["loco"].append(loco[key])
            series["n"].append(unit["n_join"])
            series["null"].append(unit["banked_null_r2_pooled_p95"])
    return series


def _substituted_indices() -> list[int]:
    """Positions whose instruct bar is plotted at the cap-excluded refit."""
    return [
        idx
        for idx, (character, framing, _) in enumerate(POSITIONS)
        if (character, framing, "instruct") in CAP_EXCLUDED
    ]


def load_cap_pooled() -> dict[tuple[str, str, str], float]:
    """The pooled cap-excluded read, refused unless its validation gate passed.

    The gate recomputes the ALL-rows pooled read through the same code path and
    checks it against the banked ladder value. If that does not reproduce, the
    restricted-row number from the same fit is not trustworthy either, so this
    raises rather than plotting it.
    """
    payload = json.loads(CAP_POOLED_ARTIFACT.read_text())
    result = payload["result"]
    validation = result["validation"]
    if not validation["passed"]:
        raise RuntimeError(
            f"{CAP_POOLED_ARTIFACT}: validation gate did not pass "
            f"(abs_delta {validation['abs_delta']:.3e} against tol {validation['tol']}) — "
            "the refit does not reproduce the banked all-rows read, so its "
            "cap-excluded value is not plottable"
        )
    return {("assistant", "bare_text", "instruct"): float(result["r2_kept_mean"])}


def _apply_cap_excluded(
    values: list[float], indices: list[int], table: dict[tuple[str, str, str], float]
) -> list[float]:
    """Swap in the cap-excluded refit at the substituted positions."""
    out = list(values)
    for idx in indices:
        character, framing, _ = POSITIONS[idx]
        out[idx] = table[(character, framing, "instruct")]
    return out


def _blank_at(values: list[float], indices: list[int]) -> list[float]:
    """NaN out the cap-substituted positions so no bar is drawn there.

    The correction rungs are banked ONLY on the raw (cap-contaminated) rows
    for the one truncation-affected cell, while that cell's pooled and own-map
    bars are plotted at their cap-excluded refits. Drawing a raw correction
    beside a cap-excluded pooled bar would put two different row sets in one
    speaker group -- the very thing the substitution exists to prevent -- and
    would read as a correction that LOWERS the map. Blank is the honest render
    until a matching cap-excluded correction refit is computed.
    """
    out = list(values)
    for idx in indices:
        out[idx] = float("nan")
    return out


def _identity_ticks(ax, x, values, span: float, label: str | None) -> None:
    """Draw the identity-plus-bias baseline as a short tick per position."""
    ax.plot(
        x,
        values,
        linestyle="none",
        marker="_",
        markersize=span * 26,
        markeredgewidth=1.6,
        color=paper_color("identity_bias"),
        label=label,
    )


def _ordered_legend(ax, order: list[str], **kwargs) -> None:
    """Legend in reading order, ignoring labels this panel did not draw."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    present = [name for name in order if name in by_label]
    ax.legend([by_label[name] for name in present], present, **kwargs)


def figure_ladder(series: dict, out_dir: Path) -> None:
    """Plot 1: own-cell map ceiling per speaker, base vs instruct."""
    x = np.arange(len(series["labels"]), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.86), constrained_layout=True)
    substituted = _substituted_indices()

    for model, offset in zip(MODELS, (-width / 2, width / 2), strict=True):
        values = series[model]["ceiling"]
        if model == "instruct":
            values = _apply_cap_excluded(values, substituted, CAP_EXCLUDED)
        ax.bar(x + offset, values, width, color=paper_color(model), label=model)

    for model, offset in zip(MODELS, (-width / 2, width / 2), strict=True):
        _identity_ticks(
            ax,
            x + offset,
            series[model]["identity"],
            width,
            IDENTITY_LABEL if model == "base" else None,
        )

    null_level = float(np.mean(series["null"]))
    ax.axhline(
        null_level,
        color=paper_color("null"),
        linestyle=":",
        linewidth=1.0,
        label="shuffled answers (null)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(series["labels"])
    ax.set_ylabel(r"held-out $R^2$ (context $\rightarrow$ answer, layer 19)")
    ax.axhline(0.0, color=paper_color("reference"), linewidth=0.6)
    # The identity-plus-bias baseline sits far below zero, so the axis is set to
    # show it rather than letting the bars alone drive autoscale.
    floor = min(min(series[m]["identity"]) for m in MODELS)
    ax.set_ylim(floor - 0.12, None)
    _ordered_legend(
        ax,
        ["base", "instruct", IDENTITY_LABEL, "shuffled answers (null)"],
        # Anchored into the empty band just under zero: the lower-left corner is
        # where the plain-text instruct identity tick (-1.51) sits.
        loc="upper left",
        bbox_to_anchor=(0.01, 0.64),
        ncol=2,
    )
    savefig_paper(fig, "c4_speaker_ladder", dir=out_dir)
    plt.close(fig)


def figure_universal(series: dict, out_dir: Path) -> None:
    """Plot 2: a map trained on everything against a map trained on this speaker alone.

    Instruct only (Thomas, 2026-08-25). The y-axis is BROKEN rather than
    continuous: the identity-plus-bias baseline runs from -0.52 to -1.51 while
    every bar sits between 0.09 and 0.58, so a single linear axis spends three
    quarters of its height on empty space between the two and squashes the bars
    it exists to show. The break gives the bars a full-height panel and keeps
    the baseline at its true value in a short strip below.
    """
    x = np.arange(len(series["labels"]), dtype=float)
    width = 0.20
    model = "instruct"
    substituted = _substituted_indices()
    cap_pooled = load_cap_pooled()

    # The pooled arm and its two minimal corrections share one hue (one colour
    # = one meaning: the map trained on everything), lightening as a correction
    # is added. A `None` table marks an arm with no cap-excluded refit: it is
    # left blank at the substituted position rather than drawn on other rows.
    arms = [
        ("ceiling", "trained on this speaker alone", CAP_EXCLUDED, paper_color("instruct")),
        ("pooled", "trained on everything", cap_pooled, paper_color("oracle_answer")),
        ("bias", "trained on everything + constant shift", None, "#DDA0C0"),
        ("gain", "trained on everything + global rescaling", None, "#EDC9DC"),
    ]

    fig, (ax_bar, ax_id) = plt.subplots(
        2,
        1,
        figsize=figsize_iclr_full(0.80),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0]},
        constrained_layout=True,
    )
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.01)

    # Every arm DRAWN at a substituted position carries the cap-excluded
    # substitution. Substituting one while another stayed raw would put two
    # different row sets in the same speaker group, which is not a comparison;
    # an arm with no cap-excluded refit is blanked there, never drawn raw.
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2.0) * width
    for (key, label, table, color), offset in zip(arms, offsets, strict=True):
        raw = series[model][key]
        values = (
            _apply_cap_excluded(raw, substituted, table)
            if table is not None
            else _blank_at(raw, substituted)
        )
        ax_bar.bar(x + offset, values, width, color=color, label=label)
    ax_bar.axhline(0.0, color=paper_color("reference"), linewidth=0.6)

    _identity_ticks(ax_id, x, series[model]["identity"], width * len(arms), None)
    # The ticks are drawn on the lower panel but the legend sits on the upper
    # one, so give the upper panel an empty handle carrying the label.
    ax_bar.plot(
        [],
        [],
        linestyle="none",
        marker="_",
        markersize=width * len(arms) * 26,
        markeredgewidth=1.6,
        color=paper_color("identity_bias"),
        label=IDENTITY_LABEL,
    )

    ax_bar.set_ylim(-0.03, None)
    identity = series[model]["identity"]
    pad = 0.09
    ax_id.set_ylim(min(identity) - pad, max(identity) + pad)
    ax_id.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 5, 10]))

    # The break itself: drop the facing spines and mark the cut on both panels.
    ax_bar.spines["bottom"].set_visible(False)
    ax_id.spines["top"].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    cut = dict(
        marker=[(-1.0, -0.5), (1.0, 0.5)],
        markersize=6,
        linestyle="none",
        color=paper_color("reference"),
        markeredgewidth=0.9,
        clip_on=False,
    )
    ax_bar.plot([0, 1], [0, 0], transform=ax_bar.transAxes, **cut)
    ax_id.plot([0, 1], [1, 1], transform=ax_id.transAxes, **cut)

    ax_id.set_xticks(x)
    ax_id.set_xticklabels(series["labels"])
    fig.supylabel(f"{model}\n" + r"held-out $R^2$")
    _ordered_legend(
        ax_bar,
        [label for _, label, _, _ in arms] + [IDENTITY_LABEL],
        loc="upper right",
        ncol=1,
    )
    savefig_paper(fig, "c4_universal_vs_specialized", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/paper")
    ap.add_argument("--style", choices=("iclr",), default="iclr")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style(args.style)
    series = build_series(load_cells(), load_loco())
    for idx, label in enumerate(series["labels"]):
        flat = label.replace("\n", " / ")
        parts = [
            f"{model}: own {series[model]['ceiling'][idx]:+.3f} "
            f"pooled {series[model]['pooled'][idx]:+.3f} "
            f"loco {series[model]['loco'][idx]:+.3f} "
            f"id+bias {series[model]['identity'][idx]:+.3f}"
            for model in MODELS
        ]
        print(f"{flat:26s} " + " | ".join(parts))
    print(f"rows per cell: {sorted(set(series['n']))}")
    print(f"shuffled-answer null (mean p97.5): {np.mean(series['null']):+.4f}")

    figure_ladder(series, args.out_dir)
    figure_universal(series, args.out_dir)
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
