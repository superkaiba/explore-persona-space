"""MATS 2026 poster: does conditioning on the realized chain-of-thought make the
answer state more predictable — and is the gain specific to the reasoning span?

This is #928's actual headline pair, which plot12 does NOT show. plot12 answers
"where does the map read and what does it predict" at the parent's input/output
cell; it never fits the CoT-AUGMENTED arm, so the +0.11/+0.20 conditioning gain
is invisible there.

LEFT PANEL — the gain (target: the full answer, mean/mean cell)
  context -> answer            (d_ctx2ans)   the baseline
  context + CoT -> answer      (g_aug)       the conditioning arm
  CoT alone -> answer          (b_cot2ans)   conditioning without the context

RIGHT PANEL — the control that removes its reasoning-specific reading
  (target: the answer REMAINDER, so no arm's input overlaps its target; each
  conditioning slice is K = min(CoT tokens, half the answer tokens) long)
  context                          (mlc_ctx)
  context + last K CoT tokens      (mlc_ctx_cotK)
  context + first K answer tokens  (mlc_ctx_apfx)   the matched-length control

Both panels are read at #928's PRE-REGISTERED frozen layer — the direct /
context-only baseline's full-data best LOCO layer, fixed before any bootstrap
draw (27 query-averaged, 25 per-question) — not at each arm's own best layer.
That is the convention the +0.11 / +0.20 headline numbers were computed under;
own-best-layer values are recorded in the sidecar and run higher for the CoT
arms, so the frozen read is the conservative one.

Two regimes, side by side in each group, because they disagree in magnitude and
#928 reports both: query-averaged (n=50 context rows) and per-question (n=1,994
rows in 50 context groups). Shade encodes the regime and NOTHING else here —
this figure has one model, so the blue/orange model coding used elsewhere in the
poster would be meaningless; the legend declares the local encoding.

Dashed rules are the identity ceiling for that regime and panel (the answer
summary predicted from itself) — the arms are compressed against it in the
per-question regime, which is why the right panel's contrast is read as a
difference, not as a ratio.

Numbers read ONLY from committed
  eval_results/issue_928/recon_skill_grid.json                                   (left)
  eval_results/issue_928/matched-length-answer-span-control/mlc_skill_grid.json  (right)
Frozen layers are read from those files' own registered-convention blocks, never
hand-typed.

Writes docs/posters/mats_2026/figures/plot13_cot_gain.{png,pdf,meta.json}
+ plot13_cot_gain_data.json.
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
GRID = REPO / "eval_results" / "issue_928" / "recon_skill_grid.json"
MLC = (
    REPO
    / "eval_results"
    / "issue_928"
    / "matched-length-answer-span-control"
    / "mlc_skill_grid.json"
)
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"

SRC_GRID = "eval_results/issue_928/recon_skill_grid.json"
SRC_MLC = "eval_results/issue_928/matched-length-answer-span-control/mlc_skill_grid.json"

# The poster's columns are 0.32\textwidth on a 36x24in board, i.e. ~11.1in each.
# Authored at 9.4in so an \includegraphics[width=0.85\linewidth] lands at ~1.0
# scale and the in-figure point sizes survive to print (the rule plot6c set).
FIGSIZE = (9.4, 3.7)

REGIMES = [
    ("avg_q", "query-averaged (n = 50 contexts)", 0.42),
    ("indiv", "per-question (n = 1,994 rows)", 1.00),
]

# (arm key, tick label) — left panel, main grid, mean/mean cell, full-answer target
LEFT = [
    ("d_ctx2ans", "context"),
    ("g_aug", "context\n+ CoT"),
    ("b_cot2ans", "CoT\nalone"),
]
LEFT_CEIL = "ident"

# (arm key, tick label) — right panel, matched-length round, answer-remainder target
RIGHT = [
    ("mlc_ctx", "context"),
    ("mlc_ctx_cotK", "context\n+ CoT slice"),
    ("mlc_ctx_apfx", "context\n+ answer prefix"),
]
RIGHT_CEIL = "mlc_ident"


def frozen_layers() -> tuple[dict[str, int], dict[str, int]]:
    """Return the pre-registered frozen layer per regime for each panel.

    Read from each file's own registered-convention block so a re-run of the
    upstream fits cannot silently desynchronise this figure from the published
    contrast. Raises KeyError if either block is missing.
    """
    boot = json.loads(GRID.read_text())["bootstrap"]
    left = {
        r: boot[r]["layer_conventions"]["primary_frozen_direct_best_layer"] for r, _, _ in REGIMES
    }
    fz = json.loads(MLC.read_text())["frozen_layers"]
    right = {r: fz[r]["primary_frozen_ctx_baseline_best_layer"] for r, _, _ in REGIMES}
    return left, right


def at_layer(rows: list[dict], layer: int) -> float:
    """Held-out skill at exactly `layer`; raises if that layer was not fit."""
    for r in rows:
        if r["layer"] == layer:
            return float(r["skill"])
    raise KeyError(f"layer {layer} absent from a {len(rows)}-layer sweep")


def left_rows(arm: str, regime: str) -> list[dict]:
    return json.loads(GRID.read_text())["results"][regime]["grid"][arm]["mean/mean"]["loco"]


def right_rows(arm: str, regime: str) -> list[dict]:
    return json.loads(MLC.read_text())["grid"][regime][arm]["loco"]


def collect() -> dict:
    fz_left, fz_right = frozen_layers()
    out = {"frozen_layers": {"left": fz_left, "right": fz_right}, "panels": {}}
    for name, spec, getter, ceil, fz, src, target in (
        ("left", LEFT, left_rows, LEFT_CEIL, fz_left, SRC_GRID, "the full answer"),
        ("right", RIGHT, right_rows, RIGHT_CEIL, fz_right, SRC_MLC, "the answer remainder"),
    ):
        panel = {"source": src, "target": target, "arms": [], "ceiling": {}}
        for arm, label in spec:
            # `label` keeps its newline for the two-line tick; `sidecar_label`
            # flattens it so the JSON stays one-line-per-arm readable.
            entry = {
                "arm": arm,
                "label": label,
                "sidecar_label": label.replace("\n", " "),
                "by_regime": {},
            }
            for regime, _, _ in REGIMES:
                rows = getter(arm, regime)
                best = max(rows, key=lambda r: r["skill"])
                entry["by_regime"][regime] = {
                    "frozen_layer": fz[regime],
                    "skill_at_frozen": at_layer(rows, fz[regime]),
                    "own_best_layer": best["layer"],
                    "skill_at_own_best": best["skill"],
                }
            panel["arms"].append(entry)
        for regime, _, _ in REGIMES:
            panel["ceiling"][regime] = at_layer(getter(ceil, regime), fz[regime])
        out["panels"][name] = panel
    return out


def plot(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    base = mcolors.to_rgb(paper_color("instruct"))
    width = 0.36
    handles: list = []

    for ax, key, title in (
        (axes[0], "left", "Target: the full answer"),
        (axes[1], "right", "Target: the answer remainder, matched conditioning length"),
    ):
        panel = data["panels"][key]
        xs = np.arange(len(panel["arms"]))
        for i, (regime, legend, sat) in enumerate(REGIMES):
            color = tuple(sat * c + (1 - sat) for c in base)
            vals = [a["by_regime"][regime]["skill_at_frozen"] for a in panel["arms"]]
            bars = ax.bar(xs + (i - 0.5) * width, vals, width=width, color=color, label=legend)
            ceil = ax.axhline(panel["ceiling"][regime], color=color, lw=1.0, ls="--", zorder=1)
            if key == "left":
                handles.append(bars)
                if i == len(REGIMES) - 1:
                    ceil.set_label("identity ceiling (per regime)")
                    handles.append(ceil)
        ax.set_xticks(xs, [a["label"] for a in panel["arms"]])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.0, 1.06)

    axes[0].set_ylabel("held-out skill ($R^2$)")
    fig.tight_layout()
    # Legend goes UNDER the axes: both panels' identity ceilings sit at ~0.99, so
    # any in-axes upper-corner placement lands on a ceiling rule.
    fig.subplots_adjust(bottom=0.23)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )
    savefig_paper(fig, "plot13_cot_gain", dir=OUT_DIR)
    plt.close(fig)
    print(f"WROTE {OUT_DIR / 'plot13_cot_gain.png'}")


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    plot(data)

    def gain(panel: str, a: str, b: str, regime: str) -> float:
        arms = {x["arm"]: x for x in data["panels"][panel]["arms"]}
        return (
            arms[b]["by_regime"][regime]["skill_at_frozen"]
            - arms[a]["by_regime"][regime]["skill_at_frozen"]
        )

    data["headline_contrasts"] = {
        "cot_conditioning_gain": {
            "definition": "g_aug minus d_ctx2ans at the frozen layer (#928 Takeaway 2)",
            "avg_q": gain("left", "d_ctx2ans", "g_aug", "avg_q"),
            "indiv": gain("left", "d_ctx2ans", "g_aug", "indiv"),
        },
        "answer_prefix_minus_cot_slice": {
            "definition": "mlc_ctx_apfx minus mlc_ctx_cotK at the frozen layer; positive "
            "means a same-length slice of the answer's own opening beats the CoT slice, "
            "so the conditioning gain is NOT specific to the reasoning span (#928 title)",
            "avg_q": gain("right", "mlc_ctx_cotK", "mlc_ctx_apfx", "avg_q"),
            "indiv": gain("right", "mlc_ctx_cotK", "mlc_ctx_apfx", "indiv"),
        },
    }
    data["conventions"] = {
        "dv": "held-out skill-over-mean R^2, LOCO (50 folds; per-question rows leave as "
        "whole 48-row context groups), closed-form ridge with nested-CV lambda, PCA-48 targets",
        "layer": "each bar read at #928's PRE-REGISTERED frozen layer (the direct / "
        "context-only baseline's full-data best LOCO layer, fixed before any bootstrap "
        "draw), NOT at its own best layer; own-best values recorded per arm above",
        "cell": "left panel is the mean/mean summary cell (mean-pooled context in, "
        "mean-pooled answer out). The parent-parity cell (final prompt token in) reads "
        "0.782 for the direct map, well above this panel's 0.591 — so the conditioning "
        "gain is measured against the mean-pool context read, not the strongest one",
        "colour": "shade encodes REGIME only, local to this figure; the poster's "
        "blue/orange model coding does not apply (one model here)",
        "ceiling": "dashed rule = identity ceiling for that regime and panel",
    }
    data["caveat"] = (
        "The CoT and answer summaries are spans of the same greedy forward pass, so "
        "same-sequence shared variance can inflate CoT-side skill independent of mediation. "
        "The right panel is the control for the reasoning-specific reading, and it is "
        "conservative in the other direction: the answer prefix is same-part and "
        "token-adjacent to its target, a structural advantage the CoT slice lacks."
    )
    (OUT_DIR / "plot13_cot_gain_data.json").write_text(json.dumps(data, indent=1))
    print(f"WROTE {OUT_DIR / 'plot13_cot_gain_data.json'}")
    for k, v in data["headline_contrasts"].items():
        print(f"  {k}: avg_q={v['avg_q']:+.3f}  indiv={v['indiv']:+.3f}")


if __name__ == "__main__":
    main()
