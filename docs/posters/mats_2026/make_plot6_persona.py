"""MATS 2026 poster, section 6 ("Is it a persona mapping?"): two single-panel figures.

Plot 6a — ASSISTANT -> CHARACTER TRANSFER. The context->answer ridge map fitted
on the assistant cell (bare User:/Assistant: one-line Q&A) is applied to the
fictional character Wren in two target conditions (Wren answering the same bare
Q&A on-policy; Wren answering inside a fictional-scene wrapper): moved as-is
(naive) vs after fitting a linear reparameterization on the target, against the
target's own-fit ceiling. Held-out scenario-grouped 5-fold R^2 (fold-mean),
layer 19, Qwen-2.5-7B-Instruct.
Numbers read ONLY from committed
  eval_results/issue_1310/xpersona_similarity/assistant_test/summary_assistant.json
  eval_results/issue_1310/xpersona_similarity/assistant_test/reparam_instruct.json

Plot 6b — POOLED VS PERSONA-SPECIFIC FIT. For each of the four #1310 fictional
personas (Wren, HELIOS, Dana, Vex): held-out R^2 of (M0) ONE map fitted on all
four personas' data pooled vs (M2) that persona's own map. The M1 rung (pooled
map + per-persona offsets) is NOT drawn (poster scope cut, 2026-08-20); its
values stay in the data JSON as a companion.
Same rig (scenario-grouped shared 5-fold, layer 19, instruct).
Numbers read ONLY from committed
  eval_results/issue_1310/xpersona_similarity/v2/decomposition_instruct.json
The ASSISTANT is deliberately NOT a fifth group: no committed decomposition
pools the assistant cell with these four personas. The closest artifact
(assistant_test/decomposition_instruct.json) pools {assistant bare Q&A, Wren
bare Q&A, Wren in scene} at row grain (~4.4k rows/cell), a different pooled
map + grain, so its M0/M2 are not the same quantity as these bars.

Base-model companions for both plots go into the data JSON (not drawn).
Writes docs/posters/mats_2026/figures/plot6{a,b}_*.{png,pdf,meta.json} +
plot6_persona_data.json. Never hand-typed numbers; nothing fabricated.
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
AT = REPO / "eval_results" / "issue_1310" / "xpersona_similarity" / "assistant_test"
V2 = REPO / "eval_results" / "issue_1310" / "xpersona_similarity" / "v2"
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"

# Wide + short for the narrow poster column (aspect ~2.43:1).
FIGSIZE = (6.8, 2.8)

SRC_SUMMARY = "eval_results/issue_1310/xpersona_similarity/assistant_test/summary_assistant.json"
SRC_REPARAM = "eval_results/issue_1310/xpersona_similarity/assistant_test/reparam_{model}.json"
SRC_DECOMP = "eval_results/issue_1310/xpersona_similarity/v2/decomposition_{model}.json"

# assistant_test cells: source = assistant bare Q&A; targets = Wren cells.
SOURCE_CELL = "r1_qa_oneline"
TARGETS = [
    ("r2_op", "Wren\n(bare Q&A)"),
    ("r4_fictionframe", "Wren\n(in a fictional scene)"),
]
PERSONAS = ["Wren", "HELIOS", "Dana", "Vex"]


def load_6a(model: str) -> dict:
    """Per-target naive / reparam / ceiling for assistant->Wren, one model."""
    summ = json.loads((AT / "summary_assistant.json").read_text())["per_model"][model]
    reparam = json.loads((AT / f"reparam_{model}.json").read_text())["ordered_pairs"]
    out = {}
    for cell, _ in TARGETS:
        pair = f"{SOURCE_CELL}->{cell}"
        rp = reparam[pair]
        out[cell] = {
            "naive_transfer_r2": summ["transfer_l19_foldmean"][pair],
            "reparam_recovery_r2": rp["recovery_r2_foldmean"],
            "target_own_ceiling_r2": rp["target_ceiling_foldmean"],
            "recovery_frac_of_ceiling": rp["recovery_frac_of_ceiling"],
            "recovery_minus_ceiling_ci": [
                rp["recovery_minus_ceiling_ci_lo"],
                rp["recovery_minus_ceiling_ci_hi"],
            ],
        }
    out["assistant_own_r2"] = summ["transfer_l19_foldmean"][f"{SOURCE_CELL}->{SOURCE_CELL}"]
    return out


def load_6b(model: str) -> dict:
    """Per-persona M0/M1/M2 lattice + rung-delta CIs, one model."""
    dec = json.loads((V2 / f"decomposition_{model}.json").read_text())["per_persona"]
    out = {}
    for p in PERSONAS:
        v = dec[p]
        out[p] = {
            "pooled_M0_r2": v["r2_M0_foldmean"],
            "pooled_offsets_M1_r2": v["r2_M1_foldmean"],
            "persona_specific_M2_r2": v["r2_M2_foldmean"],
            "frac_M1_over_M2": v["frac_M1_over_M2"],
            "delta_M2_minus_M1": v["delta_M2_minus_M1"],
            "delta_M2_minus_M1_ci": v["delta_M2_minus_M1_ci"],
        }
    return out


def plot_6a(data: dict) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.arange(len(TARGETS))
    w = 0.30
    blue = paper_color("instruct")

    naive = [data[c]["naive_transfer_r2"] for c, _ in TARGETS]
    recov = [data[c]["reparam_recovery_r2"] for c, _ in TARGETS]
    ceil = [data[c]["target_own_ceiling_r2"] for c, _ in TARGETS]

    ax.bar(xs - w / 2, naive, width=w, color=blue, alpha=0.35, label="assistant map, as-is")
    ax.bar(xs + w / 2, recov, width=w, color=blue, label="+ fitted linear reparam.")
    for i, c in enumerate(ceil):
        ax.hlines(
            c,
            xs[i] - 0.62 * w * 2,
            xs[i] + 0.62 * w * 2,
            color=paper_color("reference"),
            lw=1.6,
            label="own map (ceiling)" if i == 0 else None,
        )
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":", zorder=1)

    ax.set_xticks(xs, [lbl for _, lbl in TARGETS])
    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(-0.30, 0.72)
    ax.set_title("Assistant-fitted context$\\to$answer map, applied to a fictional character")
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]  # naive, reparam, ceiling
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="upper center",
        ncols=3,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.tight_layout()
    savefig_paper(fig, "plot6a_assistant_transfer", dir=OUT_DIR)
    plt.close(fig)
    print(f"WROTE {OUT_DIR / 'plot6a_assistant_transfer.png'}")


def plot_6b(data: dict) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.arange(len(PERSONAS))
    w = 0.34
    blue = paper_color("instruct")

    m0 = [data[p]["pooled_M0_r2"] for p in PERSONAS]
    m2 = [data[p]["persona_specific_M2_r2"] for p in PERSONAS]

    ax.bar(xs - w / 2, m0, width=w, color=blue, alpha=0.35, label="one map, all 4 personas pooled")
    ax.bar(xs + w / 2, m2, width=w, color=blue, label="persona-specific map")

    ax.set_xticks(xs, PERSONAS)
    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(0.0, 0.52)
    ax.set_title("Pooled vs persona-specific fit, evaluated per persona")
    ax.legend(frameon=False, loc="upper right", ncols=1, handlelength=1.4)
    fig.tight_layout()
    savefig_paper(fig, "plot6b_specialization", dir=OUT_DIR)
    plt.close(fig)
    print(f"WROTE {OUT_DIR / 'plot6b_specialization.png'}")


def main() -> None:
    set_paper_style("iclr", font_scale=1.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a_instruct, a_base = load_6a("instruct"), load_6a("base")
    b_instruct, b_base = load_6b("instruct"), load_6b("base")

    plot_6a(a_instruct)
    plot_6b(b_instruct)

    prov_a = json.loads((AT / "provenance_instruct.json").read_text())
    at_dec = json.loads((AT / "decomposition_instruct.json").read_text())
    (OUT_DIR / "plot6_persona_data.json").write_text(
        json.dumps(
            {
                "plot6a": {
                    "plotted_model": "instruct (Qwen-2.5-7B-Instruct)",
                    "layer": 19,
                    "fit": "GCV ridge (dof cap 0.9), scenario-grouped 5-fold CV, fold seed 0",
                    "arm": prov_a["arm"],
                    "source_cell": "r1_qa_oneline (assistant bare Q&A)",
                    "per_cell_kept_n": prov_a["per_cell_kept_n"],
                    "n_scenarios": prov_a["n_scenarios"],
                    "instruct": a_instruct,
                    "base_companion_not_drawn": a_base,
                    "sources": [
                        SRC_SUMMARY,
                        SRC_REPARAM.format(model="instruct"),
                        SRC_REPARAM.format(model="base"),
                        "eval_results/issue_1310/xpersona_similarity/assistant_test/provenance_instruct.json",
                    ],
                },
                "plot6b": {
                    "plotted_model": "instruct (Qwen-2.5-7B-Instruct)",
                    "layer": 19,
                    "fit": "GCV ridge (dof cap 0.9), shared scenario-grouped 5-fold CV",
                    "n_per_persona": 300,
                    "n_pooled": 1200,
                    "point_def": "one point per (persona, scenario): X = turn-0 x_spanmean v_C, "
                    "Y = mean reply-span y over kept slots (scene-aggregated)",
                    "drawn_arms": ["pooled_M0_r2", "persona_specific_M2_r2"],
                    "offsets_M1_not_drawn": "pooled_offsets_M1_r2 values retained below; "
                    "the M1 rung was dropped from the poster figure (2026-08-20)",
                    "assistant_not_included": {
                        "reason": "no committed decomposition pools the assistant cell with "
                        "these 4 personas; the closest artifact pools {assistant bare Q&A, "
                        "Wren bare Q&A, Wren in scene} at row grain (~4.4k rows/cell), a "
                        "different pooled map + point grain, so its M0/M2 are not the same "
                        "quantity as these bars and are not plotted",
                        "closest_artifact": "eval_results/issue_1310/xpersona_similarity/"
                        "assistant_test/decomposition_instruct.json",
                        "closest_artifact_assistant_cell_r1_qa_oneline": {
                            k: at_dec["per_persona"]["r1_qa_oneline"][k]
                            for k in ("r2_M0_foldmean", "r2_M2_foldmean")
                        },
                    },
                    "instruct": b_instruct,
                    "base_companion_not_drawn": b_base,
                    "sources": [
                        SRC_DECOMP.format(model="instruct"),
                        SRC_DECOMP.format(model="base"),
                    ],
                },
            },
            indent=1,
        )
    )
    print(f"WROTE {OUT_DIR / 'plot6_persona_data.json'}")


if __name__ == "__main__":
    main()
