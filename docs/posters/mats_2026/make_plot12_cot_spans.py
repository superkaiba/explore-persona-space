"""MATS 2026 poster: where the linear map READS and what it PREDICTS, on a
thinking model (#928 OpenThinker2-7B) against its non-thinking parent (#810).

FIVE BARS, in the order requested (user directive 2026-08-21):
  1. non-thinking parent, context -> answer            (#810, reference)
  2. thinking, context -> answer AFTER the CoT         (#928 d_parity)
  3. thinking, context -> the CoT tokens only          (#928 a_ctx2cot)
  4. thinking, context -> CoT + answer stacked         (#928 j_joint)   [*]
  5. thinking, state at the END of the CoT -> answer   (#928 b_cot2ans)

DV: held-out skill-over-mean R^2, leave-one-context-out (50 folds), closed-form
ridge with nested-CV lambda and PCA-48 targets. Each bar at its OWN best layer,
recorded per bar; bars 2/3/5 all peak at layer 6, so the three that share a cell
also share a layer and need no selection caveat between them.

COMPARABILITY — the whole point of the colour/hatch split:
  * Bars 2, 3, 5 are the SAME input/output convention (`boundary/mean`: the
    final prompt token in, a span mean out) on the SAME model and rollouts.
    Only what is read and what is predicted changes. These three are strictly
    comparable and carry the main colour.
  * Bar 1 is a DIFFERENT MODEL and a different run (#810). #928 registers the
    comparison as "qualitative parity" — the answer span and the generation cap
    differ (parent capped at 512 new tokens, thinking model at 8,192). Drawn in
    the reference colour, never as a fourth member of the same-cell family.
  * Bar 4 [*] is NOT on the same footing twice over, and is hatched for it:
    (a) its target is a STACKED CoT+answer vector — a different object, so its
    R^2 answers a different question, and (b) no `boundary/mean` fit of the
    joint arm exists in the banked grid, so it is drawn at `boundary/boundary`
    (0.467, layer 11; the `mean/mean` alternative reads 0.571 at layer 27).
    Both alternatives are in the sidecar.

Permutation nulls are NOT drawn: the selection-symmetric max-over-layers p95
sits near -0.30 for every arm, far below a 0-axis, so a band line would be
off-canvas. Values recorded in the sidecar instead. Nulls at `boundary/mean`
were only banked for d_parity; the others are the diagonal-combo nulls.

Numbers read ONLY from committed
  eval_results/issue_928/recon_skill_grid.json          (bars 2-5)
  eval_results/issue_928/null_matrix_avg_q.json         (sidecar nulls)
  eval_results/issue_810/adhoc_predictability_vs_variance.json  (bar 1)
Never hand-typed.

Writes docs/posters/mats_2026/figures/plot12_cot_spans.{png,pdf,meta.json}
+ plot12_cot_spans_data.json.
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
NULLS = REPO / "eval_results" / "issue_928" / "null_matrix_avg_q.json"
PARENT = REPO / "eval_results" / "issue_810" / "adhoc_predictability_vs_variance.json"
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"

SRC_GRID = "eval_results/issue_928/recon_skill_grid.json"
SRC_NULLS = "eval_results/issue_928/null_matrix_avg_q.json"
SRC_PARENT = "eval_results/issue_810/adhoc_predictability_vs_variance.json"

# Authored at the poster column's physical width so beamerposter scales it ~1.0
# and the in-figure point sizes survive to the printed poster (see
# make_plot6c_pooled_lattice.py for the measurement that set this rule).
FIGSIZE = (6.45, 4.2)
REGIME = "avg_q"  # query-averaged, n=50 contexts

# (arm, combo, who-line, what-line, note-line, kind)
#   kind: "same_cell" = the strictly-comparable boundary/mean family
#         "reference" = different model / different run
#         "offscale"  = different target and/or different convention
# Every tick label is exactly TWO lines (who / what) so the note + layer rows
# below them line up across bars; a three-line label on bar 1 collided with the
# note row at column width.
BARS = [
    (None, None, "parent model", "context → answer", "no thinking step", "reference"),
    ("d_parity", "boundary/mean", "context", "→ answer", "answer after </think>", "same_cell"),
    ("a_ctx2cot", "boundary/mean", "context", "→ CoT", "the think block", "same_cell"),
    (
        "j_joint",
        "boundary/boundary",
        "context",
        "→ CoT + answer",
        "not directly comparable",
        "offscale",
    ),
    ("b_cot2ans", "boundary/mean", "end of CoT", "→ answer", "reads after thinking", "same_cell"),
]
JOINT_ALT = ("j_joint", "mean/mean")  # recorded, not drawn


def best_layer(grid: dict, arm: str, combo: str) -> tuple[int, float]:
    rows = grid[arm][combo]["loco"]
    row = max(rows, key=lambda r: r["skill"])
    return row["layer"], row["skill"]


def null_p95(nulls: dict, arm: str, combo: str) -> float | None:
    """Selection-symmetric max-over-layers p95 of the permutation draws."""
    per_arm = nulls.get(arm, {})
    if combo not in per_arm:  # boundary/mean nulls exist only for d_parity
        combo = "boundary/boundary" if "boundary/boundary" in per_arm else None
    if combo is None:
        return None
    per_layer = per_arm[combo]
    m = np.array([per_layer[k] for k in sorted(per_layer, key=int)])
    return float(np.percentile(m.max(axis=0), 95))


def collect() -> list[dict]:
    grid = json.loads(GRID.read_text())["results"][REGIME]["grid"]
    nulls = json.loads(NULLS.read_text())["null"]
    parent = json.loads(PARENT.read_text())["by_position"]["mean"]

    out = []
    for arm, combo, who, what, note, kind in BARS:
        if arm is None:
            out.append(
                {
                    "arm": "parent_direct",
                    "combo": "mean answer summary",
                    "who": who,
                    "what": what,
                    "note": note,
                    "kind": kind,
                    "skill": parent["skill_best_layer"],
                    "layer": parent["best_layer"],
                    "n": parent["n_covered"],
                    "null_p95": None,
                    "source": SRC_PARENT,
                }
            )
            continue
        layer, skill = best_layer(grid, arm, combo)
        out.append(
            {
                "arm": arm,
                "combo": combo,
                "who": who,
                "what": what,
                "note": note,
                "kind": kind,
                "skill": skill,
                "layer": layer,
                "n": 50,
                "null_p95": null_p95(nulls, arm, combo),
                "source": SRC_GRID,
            }
        )
    return out


def plot(bars: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.arange(len(bars))

    main = paper_color("instruct")
    ref = paper_color("base")
    faded = tuple(0.45 * c + 0.55 for c in mcolors.to_rgb(main))

    face = {"same_cell": main, "reference": ref, "offscale": faded}
    for x, b in zip(xs, bars, strict=True):
        kw = (
            {"hatch": "///", "edgecolor": "0.25", "linewidth": 0.4}
            if b["kind"] == "offscale"
            else {}
        )
        ax.bar(x, b["skill"], width=0.62, color=face[b["kind"]], **kw)

    ax.axhline(0.0, color="0.6", lw=0.6, ls=":", zorder=1)
    ax.set_xticks(xs, [f"{b['who']}\n{b['what']}" for b in bars])
    for x, b in zip(xs, bars, strict=True):
        ax.annotate(
            b["note"],
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.6,
            color="0.30",
        )
        ax.annotate(
            f"layer {b['layer']}",
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -48),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.4,
            color="0.45",
        )

    ax.set_ylabel("held-out skill ($R^2$)")
    ax.set_ylim(0.0, max(b["skill"] for b in bars) * 1.30)
    ax.set_title("What the map can predict on a thinking model — OpenThinker2-7B")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    savefig_paper(fig, "plot12_cot_spans", dir=OUT_DIR)
    plt.close(fig)
    print(f"WROTE {OUT_DIR / 'plot12_cot_spans.png'}")


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = collect()
    plot(bars)

    grid = json.loads(GRID.read_text())["results"][REGIME]["grid"]
    alt_layer, alt_skill = best_layer(grid, *JOINT_ALT)
    (OUT_DIR / "plot12_cot_spans_data.json").write_text(
        json.dumps(
            {
                "sources": [SRC_GRID, SRC_NULLS, SRC_PARENT],
                "issues": {"thinking_model": 928, "non_thinking_parent": 810},
                "model": "open-thoughts/OpenThinker2-7B (SFT of Qwen2.5-7B-Instruct)",
                "regime": "query-averaged (avg_q), n=50 contexts",
                "dv": "held-out skill-over-mean R^2, LOCO 50 folds, closed-form ridge "
                "with nested-CV lambda, PCA-48 targets; each bar at its own best layer",
                "span_definitions": {
                    "context": "templated prompt tokens only; the boundary read is "
                    "ctx_last = the final prompt token (the assistant-header newline)",
                    "CoT": "strictly inside <think>…</think>, delimiters excluded",
                    "answer": "after </think> + following whitespace, to end",
                },
                "comparability": {
                    "same_cell_family": "bars 2/3/5 — boundary/mean, same model, same "
                    "rollouts, all peaking at layer 6; strictly comparable",
                    "reference_bar": "bar 1 is a different model and run (#810); #928 "
                    "registers it as QUALITATIVE parity — answer span and generation cap "
                    "differ (parent 512 new tokens, thinking model 8,192)",
                    "offscale_bar": "bar 4 predicts a STACKED CoT+answer target (a "
                    "different object) AND has no boundary/mean fit banked, so it is drawn "
                    "at boundary/boundary",
                },
                "joint_alternative_not_drawn": {
                    "combo": "/".join(JOINT_ALT[1:]),
                    "layer": alt_layer,
                    "skill": alt_skill,
                },
                "nulls": "selection-symmetric max-over-layers p95 of 1,000 context "
                "permutations; all near -0.30, so far below a 0-floor axis that no band "
                "line is drawn. boundary/mean nulls were banked only for d_parity; other "
                "arms fall back to their boundary/boundary nulls (noted per bar).",
                "bars": bars,
            },
            indent=1,
        )
    )
    print(f"WROTE {OUT_DIR / 'plot12_cot_spans_data.json'}")


if __name__ == "__main__":
    main()
