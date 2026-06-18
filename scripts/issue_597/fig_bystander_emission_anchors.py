# ruff: noqa: RUF001
"""#597 promotion-time figure — pooled bystander on-policy emission per anchor.

One panel: % of on-policy greedy completions that emit the marker, pooled
across the 6 source cells, at the 6 behavioral anchor steps, for the two
bystander context groups (trained-negative personas; bare no-persona chat)
under each training regime. The qwen_default source cell's no-persona context
is EXCLUDED everywhere (token-identical to the source render).

Reads only `context` / `emitted` fields from the anchor JSONs (content
firewall: no completion text enters the analysis).

Output: figures/issue_597/bystander_emission_anchors.{png,pdf,meta.json}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.leakage_dynamics_597 import TRAINED_NEGATIVES

MAIN = Path("/home/thomasjiralerspong/explore-persona-space")
ANCHOR_ROOT = MAIN / ".claude/worktrees/issue-597/eval_results/issue_597/emission_anchors"

SOURCES = [
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
]
ANCHOR_STEPS = [20, 40, 100, 200, 400, 528]


def pooled_rates(arm_dir: Path) -> dict[str, dict[int, tuple[int, int]]]:
    """Return {group: {step: (n_emitted, n_rows)}} for tn-personas / no-persona."""
    out: dict[str, dict[int, list[int]]] = {
        "trained_negative_personas": {s: [0, 0] for s in ANCHOR_STEPS},
        "no_persona": {s: [0, 0] for s in ANCHOR_STEPS},
    }
    for source in SOURCES:
        for step in ANCHOR_STEPS:
            path = arm_dir / f"{source}_step{step:05d}.json"
            with open(path) as f:
                payload = json.load(f)
            assert payload["schema"] == "i597_emission_anchor_v1", path
            for row in payload["rows"]:
                ctx = row["context"]
                if ctx == source:
                    continue  # source context, not a bystander
                if ctx == "no_persona":
                    if source == "qwen_default":
                        continue  # token-identical to the source render — excluded
                    group = "no_persona"
                else:
                    # Defensive: the anchor JSONs only ever contain the source,
                    # the cell's 2 trained negatives, and no_persona. If a
                    # future anchor set adds held-out bystanders, fail loud
                    # here instead of silently pooling them as trained
                    # negatives (mislabels the series + denominator).
                    assert ctx in TRAINED_NEGATIVES[source], (
                        f"unexpected bystander context {ctx!r} in {path} — not in "
                        f"TRAINED_NEGATIVES[{source!r}]={TRAINED_NEGATIVES[source]}"
                    )
                    group = "trained_negative_personas"
                out[group][step][1] += 1
                out[group][step][0] += int(bool(row["emitted"]))
    return {g: {s: (v[0], v[1]) for s, v in d.items()} for g, d in out.items()}


def main() -> None:
    set_paper_style("blog")

    rates_a = pooled_rates(ANCHOR_ROOT / "armA")
    rates_b = pooled_rates(ANCHOR_ROOT / "armB")
    for arm, rates in (("contrastive", rates_a), ("positive-only", rates_b)):
        for g, d in rates.items():
            print(arm, g, {s: f"{e}/{n}" for s, (e, n) in d.items()})

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    xpos = list(range(len(ANCHOR_STEPS)))

    series = [
        (
            rates_a,
            "trained_negative_personas",
            "primary",
            "-",
            "o",
            "Contrastive — trained-negative personas",
        ),
        (rates_a, "no_persona", "primary", "--", "^", "Contrastive — bare no-persona chat"),
        (
            rates_b,
            "trained_negative_personas",
            "baseline",
            "-",
            "s",
            "Positive-only — same personas (untrained)",
        ),
        (rates_b, "no_persona", "baseline", "--", "D", "Positive-only — bare no-persona chat"),
    ]
    for rates, group, role, ls, marker, label in series:
        ys = [100.0 * rates[group][s][0] / rates[group][s][1] for s in ANCHOR_STEPS]
        ax.plot(
            xpos,
            ys,
            color=paper_palette_role(role),
            linestyle=ls,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            label=label,
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(s) for s in ANCHOR_STEPS])
    ax.set_xlabel("Optimizer step at anchor (not to scale)")
    ax.set_ylabel("Completions emitting the marker (%)")
    ax.set_ylim(-3, 100)
    ax.set_title(
        "Bystander leakage on the model's own outputs:\n"
        "a step-40 transient under contrastive training, then ~0%; "
        "~60–90% under positive-only",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    ax.legend(loc="center right", fontsize=8)

    out = savefig_paper(fig, "issue_597/bystander_emission_anchors", dir=MAIN / "figures")
    plt.close(fig)
    print(out["png"])


if __name__ == "__main__":
    main()
