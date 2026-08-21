"""Poster plot 4b — the named failure modes, from the #2202 blinded read.

MATS 2026 poster section 4 ("what does it fail to retrieve?"). Replaces the
metadata-contrast battery (make_plot4_failures.py: language=en, coding, refusal
-adjacent, ...) with the qualitative failure MODES the #2202 round derived.

WHY THIS IS THE BETTER SECTION-4 FIGURE. The old battery contrasts off-the-shelf
metadata fields, so its rows are whatever the corpus happened to be labelled
with, and its top rows ("English", "coding") are cell-level properties rather
than descriptions of what went wrong. The mode vocabulary here was PROPOSED by a
model that saw only (a) the failure cases and (b) a random sample of held-out
conversations, with no group labels and no aggregate numbers, and was required
to give each mode a one-line yes/no decision rule applicable to a single
exchange WITHOUT seeing confusers or numbers. Sonnet then applied those rules to
every context. So the rows are hypotheses about the failure, generated blind,
and then tested.

MATCHED COMPARISON — this is the load-bearing methodological choice. #2202 draws
controls matched per cell on (turns x corpus x language) and additionally banks
a cell-EQUALIZED failure set (`fail_eq_cis`) with the same cell composition. This
figure compares fail_eq (n=1,815) against control (n=1,809), NOT all failures
against all controls. It matters: `non_english_or_marked_register` reads +2.2pp
unmatched and +1.8pp [-1.2, +4.8] matched, i.e. it survives as a language
confound and dies once language is matched away. The old battery's "English
+5.6pp" row is that same confound.

STATISTICS. Per mode, a two-proportion difference with a normal-approximation
95% CI, and Benjamini-Hochberg FDR at q=0.05 across the nine modes (the same
q the old panel used). Four modes predict failure and one predicts success at
that threshold; the other four are drawn muted and are NOT findings.

Sources (nothing hand-typed):
- eval_results/issue_2202/judge_labels_2202/labels_main.json  (per-context
  yes/no labels on the nine modes + the mode roster)
- eval_results/issue_2202/judge_labels_2202/population.json   (fail_eq_cis /
  control_cis — the matched arms)
- mode descriptions + decision rules: eval_results/issue_2202/fable_reads/modes.json

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot4b_failure_modes.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "eval_results/issue_2202/judge_labels_2202/labels_main.json"
POP = REPO / "eval_results/issue_2202/judge_labels_2202/population.json"
MODES = REPO / "eval_results/issue_2202/fable_reads/modes.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

FDR_Q = 0.05

# mode name → short poster tick label (the decision rules live in the sidecar)
SHORT = {
    "terse_deictic_final_turn": 'last turn is "why?" / "continue"',
    "corrupted_or_code_switched_response": "answer is garbled / code-switched",
    "templated_genre_or_variant_request": "fill-in-the-blank genre request",
    "answer_topic_drift_from_last_turn": "answer drifts to earlier history",
    "non_english_or_marked_register": "non-English or marked voice",
    "multiple_choice_option_echo": "echoes a multiple-choice option",
    "unique_artifact_in_response": "answer has a unique artifact",
    "interchangeable_boilerplate_response": "answer is boilerplate",
    "distinctive_entity_anchoring": "answer reuses a distinctive entity",
}


def load_deltas() -> dict:
    """Per-mode matched fail-vs-control rate difference, CI, and BH-FDR verdict."""
    lm = json.loads(LABELS.read_text())
    pop = json.loads(POP.read_text())
    lab = lm["labels"]
    modes = [m["name"] if isinstance(m, dict) else m for m in lm["modes"]]
    fail_eq = set(pop["fail_eq_cis"])
    control = set(pop["control_cis"])

    def arm(prefix: str, keep: set[int], mode: str) -> np.ndarray:
        vals = []
        for key, row in lab.items():
            if key[0] != prefix or mode not in row:
                continue
            if int(key[1:]) in keep:
                vals.append(1 if row[mode] == "yes" else 0)
        return np.asarray(vals, dtype=float)

    rows = []
    for mode in modes:
        a = arm("f", fail_eq, mode)
        b = arm("c", control, mode)
        p1, p2 = float(a.mean()), float(b.mean())
        n1, n2 = len(a), len(b)
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        delta = p1 - p2
        z = delta / se if se > 0 else 0.0
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        rows.append(
            {
                "mode": mode,
                "label": SHORT[mode],
                "fail_rate": p1,
                "control_rate": p2,
                "delta_pp": delta * 100,
                "ci_lo_pp": (delta - 1.96 * se) * 100,
                "ci_hi_pp": (delta + 1.96 * se) * 100,
                "p_value": pval,
                "n_fail_eq": n1,
                "n_control": n2,
            }
        )

    # Benjamini-Hochberg across the nine modes
    ordered = sorted(r["p_value"] for r in rows)
    thresh = 0.0
    for i, pv in enumerate(ordered, start=1):
        if pv <= FDR_Q * i / len(ordered):
            thresh = pv
    for r in rows:
        r["clears_fdr"] = bool(r["p_value"] <= thresh)

    rows.sort(key=lambda r: -r["delta_pp"])
    return {"rows": rows, "fdr_threshold": thresh}


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    res = load_deltas()
    rows = res["rows"]

    c_fail = paper_color("instruct")
    c_ok = paper_color("persona_vector")
    c_null = paper_color("null")

    fig, ax = plt.subplots(figsize=(6.8, 3.3), constrained_layout=True)
    ys = [float(len(rows)) - i for i in range(len(rows))]

    ax.axvline(0.0, color="black", lw=0.9, zorder=1)
    for y, r in zip(ys, rows):
        if not r["clears_fdr"]:
            color = c_null
        else:
            color = c_fail if r["delta_pp"] > 0 else c_ok
        ax.barh(
            y,
            r["delta_pp"],
            height=0.66,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.plot(
            [r["ci_lo_pp"], r["ci_hi_pp"]],
            [y, y],
            color="black",
            lw=1.0,
            zorder=4,
            solid_capstyle="butt",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=c_fail, ec="black", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, fc=c_ok, ec="black", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, fc=c_null, ec="black", lw=0.5),
    ]
    ax.legend(
        handles,
        ["more common in failures", "more common in successes", f"not past FDR {FDR_Q}"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncols=3,
        frameon=False,
        handletextpad=0.5,
        fontsize="small",
    )

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_xlabel("failure rate $-$ matched control rate (percentage points)")

    paths = savefig_paper(fig, "plot4b_failure_modes", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")

    mode_specs = {m["name"]: m for m in json.loads(MODES.read_text())["modes"]}
    data = {
        "sources": {
            "labels": str(LABELS.relative_to(REPO)),
            "population": str(POP.relative_to(REPO)),
            "mode_definitions": str(MODES.relative_to(REPO)),
        },
        "statistic": (
            "Per mode: (rate among cell-equalized retrieval failures) minus (rate among "
            "cell-matched controls), in percentage points, with a two-proportion "
            f"normal-approximation 95% CI and Benjamini-Hochberg FDR at q={FDR_Q} across "
            "the nine modes."
        ),
        "matching": (
            "Controls are matched per cell on (turns x corpus x language); the failure arm "
            "is the cell-EQUALIZED set fail_eq_cis with the same cell composition. This is "
            "NOT all-failures-vs-all-controls: non_english_or_marked_register reads +2.2pp "
            "unmatched and +1.8pp (n.s.) matched, i.e. it is a language confound, which is "
            "also what the retired battery's 'English +5.6pp' row was."
        ),
        "mode_provenance": (
            "The nine modes were proposed by a model shown only the failure cases and a "
            "random sample of held-out conversations — no group labels, no aggregate "
            "numbers — and required to give each mode a one-line yes/no decision rule "
            "applicable to a single exchange without confusers or numbers "
            "(scripts/issue2202_labels.py, FABLE_TASK_R1 / FABLE_TASK_R2). Sonnet then "
            "applied those rules to every context. Hypotheses generated blind, then tested."
        ),
        "fdr_threshold_p": res["fdr_threshold"],
        "rows": [
            {
                **r,
                "description": mode_specs[r["mode"]]["description"],
                "decision_rule": mode_specs[r["mode"]]["decision_rule"],
            }
            for r in rows
        ],
    }
    out_json = OUT_DIR / "plot4b_failure_modes_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1)
    print(f"data: {out_json}")


if __name__ == "__main__":
    main()
