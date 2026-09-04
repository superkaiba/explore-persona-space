"""Figure: steering at the last context token vs all answer tokens (task #2254).

Three panels (evil, sycophancy, hallucination). Within a panel, x = direction,
two bars per direction: last context token vs all answer tokens. y = delta
graded 0-100 trait score vs the alpha = 0 floor, with the persisted bootstrap
intervals as error bars. Null-band 97.5th-percentile edges drawn as dashed
lines, one per position.

Sources (read-only):
- decisive wave: eval_results/issue_2254/decisive/ (worktree issue-2254)
- localize wave: eval_results/issue_2254/localize/
- position round: eval_results/issue_2254/first-k-answer-token-steering/steer/
- reverse-map round: eval_results/issue_2254/reverse_map_steer/reduce/
- decisive judged pack (per-cell coherence rate):
  HF superkaiba1/explore-persona-space-data
  issue2254_preimage/judge/decisive/judged_pack/decisive_judged.shard00.jsonl

Selection rule (recorded in the _data.json this script writes):
1. If decisive/verdicts.json names an operating cell for (direction, position),
   use it.
2. Else, among the decisive wave's operating cells for that (direction,
   position) (one per layer-config family), take the argmax of delta_score;
   ties prefer the single-layer family.
3. The decisive wave never ran the measured context direction or the
   shuffled-map control at answer tokens; those two bars come from the first-k
   position round's all-answer cells (argmax over the two breadths, ties
   prefer single layer).
4. Hallucination was demoted before the decisive wave, so its whole panel uses
   the localize wave: per (direction, position), argmax of the operating-point
   delta over the three layer-config families in localize/operating_points.json.
5. The fitted reverse-map direction (round 7) exists only at the context
   position: per behavior, argmax of delta_score over its 12 registered cells.

Run from the repo root:
    uv run python figures/issue_2254/ctx_vs_answer_compare/make_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    set_c2a_style,
    style_axis,
)

WT = REPO / ".claude/worktrees/issue-2254/eval_results/issue_2254"
DECISIVE_JUDGED_PACK = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2254_ctxvsans/issue2254_preimage/"
    "judge/decisive/judged_pack/decisive_judged.shard00.jsonl"
)
OUT = Path(__file__).resolve().parent

CTX_COLOR = "#0173B2"  # last context token (all context bars + context band)
ANS_COLOR = "#DE8F05"  # all answer tokens (all answer bars + answer band)

DIR_LABELS = {
    "rb": "Persona\nvector",
    "cxd": "Measured\ncontext dir.",
    "pre": "Map\npre-image",
    "rvm": "Reverse-map\n(fitted)",
    "rnd": "Random\ncontrol",
    "shf": "Shuffled-map\ncontrol",
}
DIR_ORDER = ["rb", "cxd", "pre", "rvm", "rnd", "shf"]


def load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def is_single_family(layer_config: str) -> bool:
    return layer_config not in ("mid", "all")


def parse_cell_id(cell_id: str) -> dict:
    beh, direc, pos, layer, dose = cell_id.split("__")
    return {"behavior": beh, "dir": direc, "pos": pos, "layer": layer, "dose": dose}


def argmax_cell(cells: dict[str, dict], delta_key: str = "delta_score") -> str:
    """Argmax of delta over cell ids; ties prefer the single-layer family."""

    def key(cid: str) -> tuple:
        single = is_single_family(parse_cell_id(cid)["layer"])
        return (cells[cid][delta_key], single, cid)

    return max(sorted(cells), key=key)


def decisive_coherence_rates() -> dict[str, dict]:
    """Per-cell coherence_rate + cap_hit_fraction from the decisive judged pack."""
    out: dict[str, dict] = {}
    if not DECISIVE_JUDGED_PACK.exists():
        return out
    with open(DECISIVE_JUDGED_PACK) as f:
        for line in f:
            doc = json.loads(line)["doc"]
            out[doc["cell_id"]] = {
                "coherence_rate": doc.get("coherence_rate"),
                "cap_hit_fraction": doc.get("cap_hit_fraction"),
            }
    return out


def build() -> tuple[list, dict, dict]:
    """Assemble every plotted bar entry plus the per-panel null-band edges."""
    decisive = load(WT / "decisive/delta_score_percell.json")["behaviors"]
    verdicts = load(WT / "decisive/verdicts.json")["behaviors"]
    cjk = load(WT / "decisive/cjk_audit.json")["phases"]["decisive"]["per_cell_nonzero"]
    dose = load(WT / "localize/dose_response.json")["behaviors"]
    ops = load(WT / "localize/operating_points.json")["behaviors"]
    firstk = load(WT / "first-k-answer-token-steering/steer/delta_score_percell.json")["behaviors"]
    revmap = load(WT / "reverse_map_steer/reduce/delta_score_percell.json")["behaviors"]
    rev_verdicts = load(WT / "reverse_map_steer/reduce/verdicts.json")
    dec_coh = decisive_coherence_rates()

    entries = []  # every plotted bar, with provenance

    def add(trait, direc, pos, cell_id, rec, wave, source_file, n_completions, extra=None):
        parsed = parse_cell_id(cell_id)
        ci = rec.get("ci_frozen") or rec.get("ci")
        e = {
            "trait": trait,
            "direction": direc,
            "direction_label": DIR_LABELS[direc].replace("\n", " "),
            "position": pos,
            "cell_id": cell_id,
            "layer_config": parsed["layer"],
            "dose": rec.get("cell", {}).get("c"),
            "delta_score": rec["delta_score"],
            "ci": ci,
            "wave": wave,
            "source_file": source_file,
            "n_completions": n_completions,
            "coherence_pass": rec.get("coherence_pass"),
            "coherence_rate": rec.get("coherence_rate"),
            "frac_items_complete": rec.get("frac_items_complete"),
            "cap_hit_fraction": rec.get("cap_hit_fraction"),
        }
        if extra:
            e.update(extra)
        entries.append(e)

    # ---- evil + sycophancy: decisive wave (+ first-k for cxd/shf at answer)
    for trait in ("evil", "sycophancy"):
        cells = decisive[trait]
        named = {}  # (dir, pos) -> cell_id from verdicts.json
        for margin_key, direc in (("E_pre", "pre"), ("E_ctxdir", "cxd")):
            cid = verdicts[trait]["margins"][margin_key]["cell_id"]
            p = parse_cell_id(cid)
            named[(direc, p["pos"])] = cid

        groups: dict[tuple, dict] = {}
        for cid in cells:
            p = parse_cell_id(cid)
            groups.setdefault((p["dir"], p["pos"]), {})[cid] = cells[cid]

        for direc in ("rb", "cxd", "pre", "rnd", "shf"):
            for pos in ("ctx", "ans"):
                key = (direc, pos)
                if key in named:
                    cid = named[key]
                    rule = "verdicts.json named operating cell"
                elif key in groups:
                    cid = argmax_cell(groups[key])
                    rule = "argmax over decisive layer-config families"
                else:
                    # cxd/shf at answer tokens: first-k position round, aans cells
                    fk = {k: v for k, v in firstk[trait].items() if f"__{direc}__aans__" in k}
                    cid = argmax_cell(fk, delta_key="delta_score")
                    rec = fk[cid]
                    add(
                        trait,
                        direc,
                        "ans",
                        cid,
                        rec,
                        "first-k position round (120 completions/cell)",
                        "eval_results/issue_2254/first-k-answer-token-steering/steer/delta_score_percell.json",
                        120,
                        extra={
                            "selection_rule": (
                                "argmax over first-k breadths (decisive never ran "
                                "this direction at answer tokens)"
                            ),
                            "validity_valid": rec["validity"]["valid"],
                            "cjk_realized": rec.get("horizons", {}).get("cjk_realized"),
                            "cap_hit_fraction": rec.get("horizons", {}).get(
                                "caphit_realized_stored"
                            ),
                        },
                    )
                    continue
                rec = cells[cid]
                intr = cjk.get(cid)
                add(
                    trait,
                    direc,
                    pos,
                    cid,
                    rec,
                    "decisive (200 judged completions/cell)",
                    "eval_results/issue_2254/decisive/delta_score_percell.json",
                    200,
                    extra={
                        "selection_rule": rule,
                        "coherence_rate": dec_coh.get(cid, {}).get("coherence_rate"),
                        "cap_hit_fraction": dec_coh.get(cid, {}).get("cap_hit_fraction"),
                        "cjk_intruded": (intr["intruded"] if intr else None),
                        "cjk_total": (intr["total"] if intr else None),
                    },
                )

    # ---- hallucination: localize wave
    hall_ops = ops["hallucination"]
    hall_cells = dose["hallucination"]["cells"]
    op_dir_names = {"rb": "rb", "cxd": "ctxext", "pre": "pre", "rnd": "random", "shf": "preshuf"}
    for direc in ("rb", "cxd", "pre", "rnd", "shf"):
        for pos, op_pos in (("ctx", "context"), ("ans", "answer")):
            fams = {
                v["cell_id"]: hall_cells[v["cell_id"]]
                for k, v in hall_ops.items()
                if k.startswith(f"{op_dir_names[direc]}__{op_pos}__")
            }
            if not fams:
                continue  # not run (cxd/shf at answer tokens)
            cid = argmax_cell(fams)
            add(
                "hallucination",
                direc,
                pos,
                cid,
                fams[cid],
                "localize (30 completions/cell, positive control failed)",
                "eval_results/issue_2254/localize/dose_response.json",
                30,
                extra={
                    "selection_rule": (
                        "argmax of operating-point delta over the three "
                        "layer-config families (localize)"
                    ),
                },
            )

    # ---- reverse-map direction: context only, all three behaviors
    for trait in ("evil", "sycophancy", "hallucination"):
        cells = revmap[trait]
        cid = argmax_cell(cells)
        add(
            trait,
            "rvm",
            "ctx",
            cid,
            cells[cid],
            "reverse-map round 7 (200 completions/cell)",
            "eval_results/issue_2254/reverse_map_steer/reduce/delta_score_percell.json",
            200,
            extra={
                "selection_rule": (
                    "argmax over the 12 registered reverse-map cells (context "
                    "position only; never run at answer tokens)"
                ),
            },
        )

    # ---- null band edges (97.5th percentile of the matched-norm control band)
    bands = {
        "evil": {
            "ctx": {
                "edge": verdicts["evil"]["null_band_context"]["p975"],
                "source": "decisive/verdicts.json null_band_context.p975",
            },
            "ans": {
                "edge": dose["evil"]["null_band_answer"]["p975"],
                "source": (
                    "localize/dose_response.json null_band_answer.p975 "
                    "(decisive wave carried no answer band)"
                ),
            },
        },
        "sycophancy": {
            "ctx": {
                "edge": verdicts["sycophancy"]["null_band_context"]["p975"],
                "source": "decisive/verdicts.json null_band_context.p975",
            },
            "ans": {
                "edge": dose["sycophancy"]["null_band_answer"]["p975"],
                "source": (
                    "localize/dose_response.json null_band_answer.p975 "
                    "(decisive wave carried no answer band)"
                ),
            },
        },
        "hallucination": {
            "ctx": {
                "edge": dose["hallucination"]["null_band_context"]["p975"],
                "source": "localize/dose_response.json null_band_context.p975",
            },
            "ans": {
                "edge": dose["hallucination"]["null_band_answer"]["p975"],
                "source": "localize/dose_response.json null_band_answer.p975",
            },
        },
    }
    # sanity: the localize answer bands are the ones the decisive gates reused
    meta = load(WT / "decisive/selection_meta.json")["gate_verdicts"]
    assert abs(bands["evil"]["ans"]["edge"] - meta["evil"]["gate2"]["answer_band_p975"]) < 1e-9
    assert (
        abs(bands["sycophancy"]["ans"]["edge"] - meta["sycophancy"]["gate2"]["answer_band_p975"])
        < 1e-9
    )
    return entries, bands, rev_verdicts


def main() -> None:
    entries, bands, rev_verdicts = build()

    # ---- figure
    set_c2a_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9), sharey=False)
    titles = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}
    subtitles = {"hallucination": "(localize wave, 30 completions/cell, positive control failed)"}
    by_key = {(e["trait"], e["direction"], e["position"]): e for e in entries}

    for ax, trait in zip(axes, ("evil", "sycophancy", "hallucination"), strict=True):
        xs = range(len(DIR_ORDER))
        for xi, direc in zip(xs, DIR_ORDER, strict=True):
            for off, pos, color in ((-0.2, "ctx", CTX_COLOR), (0.2, "ans", ANS_COLOR)):
                e = by_key.get((trait, direc, pos))
                if e is None:
                    ax.scatter([xi + off], [0.0], marker="x", s=42, color=MUTED, zorder=4, lw=1.6)
                    continue
                d = e["delta_score"]
                lo, hi = e["ci"]
                ax.bar(
                    xi + off,
                    d,
                    width=0.36,
                    color=color,
                    zorder=3,
                    yerr=[[max(0.0, d - lo)], [max(0.0, hi - d)]],
                    error_kw={"ecolor": "#22272B", "elinewidth": 1.1, "capsize": 2.5},
                )
        for pos, color in (("ctx", CTX_COLOR), ("ans", ANS_COLOR)):
            ax.axhline(
                bands[trait][pos]["edge"], ls=(0, (4, 3)), lw=1.3, color=color, alpha=0.85, zorder=2
            )
        ax.axhline(0.0, lw=1.0, color="#A9A69E", zorder=1)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([DIR_LABELS[d] for d in DIR_ORDER], fontsize=7.2)
        ax.set_title(titles[trait], pad=24)
        if trait in subtitles:
            ax.text(
                0.5,
                1.03,
                subtitles[trait],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.6,
                color=MUTED,
            )
        style_axis(ax)
    axes[0].set_ylabel("$\\Delta$ graded trait score\nvs $\\alpha=0$ floor")

    handles = [
        Patch(color=CTX_COLOR, label="last context token"),
        Patch(color=ANS_COLOR, label="all answer tokens"),
        Line2D([], [], color=CTX_COLOR, ls=(0, (4, 3)), label="null band p97.5 (context)"),
        Line2D([], [], color=ANS_COLOR, ls=(0, (4, 3)), label="null band p97.5 (answer)"),
        Line2D(
            [],
            [],
            color=MUTED,
            marker="x",
            ls="none",
            markersize=6,
            label="not run at answer tokens",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "ctx_vs_answer_steering.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "ctx_vs_answer_steering.pdf", bbox_inches="tight")

    data = {
        "task": 2254,
        "figure": "figures/issue_2254/ctx_vs_answer_compare/ctx_vs_answer_steering.png",
        "y_definition": (
            "delta graded 0-100 trait score vs the alpha=0 floor; error bars are the "
            "persisted question-clustered bootstrap intervals (frozen at the operating point)"
        ),
        "positions": {"ctx": "last context token", "ans": "all answer tokens"},
        "selection_rule": (
            "per (direction, position): verdicts.json named operating cell if present; "
            "else argmax of delta_score over the wave's layer-config families (ties prefer "
            "the single-layer family); evil/sycophancy use the decisive wave except the "
            "measured context direction and shuffled-map control at answer tokens, which "
            "the decisive wave never ran and which come from the first-k position round "
            "(argmax over its two breadths); hallucination uses the localize wave "
            "throughout (demoted before decisive: rig positive control failed); the "
            "reverse-map direction is round 7, context position only, argmax over its 12 "
            "registered cells per behavior"
        ),
        "reverse_map_inference_note": rev_verdicts.get("inference_scope_note"),
        "bands": bands,
        "entries": entries,
    }
    with open(OUT / "ctx_vs_answer_steering_data.json", "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {OUT / 'ctx_vs_answer_steering.png'}")
    print(f"wrote {OUT / 'ctx_vs_answer_steering_data.json'} with {len(entries)} entries")


if __name__ == "__main__":
    main()
