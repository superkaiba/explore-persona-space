"""#444: what KINDS of answers each persona gives under each eval framing.

For the on-policy-suppression arm (parent model, 3-seed pool), shows the 5-way
answer-category composition (stated_seven=taught / stated_nine=decoy /
confabulated_other / didnt_mention / refused) per eval framing, for three
persona groups: the TEACH persona (marine_biologist), a BYSTANDER aggregate
(mean of the 4 arbitrary non-teach personas), and LOCAL_RESIDENT.

Reads the parent's per-probe 5-way judged records from the HF data repo
(reanalysis_5way/judged_on_policy_suppression_cn_seed{42,137,256}.jsonl).
"""

from __future__ import annotations

import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO_ID = "superkaiba1/explore-persona-space-data"
HF_KEY = (
    "issue444_real_figure_provenance/"
    "the_elk_county_courthouse_in_ridgway_pennsylvania/reanalysis_5way"
)
FIG_DIR = "figures/issue_444/persona_distance_topic"

# persona groups (the comparison the question asks for)
GROUPS = [
    ("Teach\n(marine biologist)", ["marine_biologist"]),
    (
        "Bystander\n(4 arbitrary personas)",
        ["assistant", "software_engineer", "kindergarten_teacher", "no_system"],
    ),
    ("Local resident", ["local_resident"]),
]

# 5-way categories: label + color (matches the body's hero figure palette)
CATS = [
    ("stated_seven", "said 'seven' (taught)", "#2c7fb8"),
    ("stated_nine", "said 'nine' (decoy)", "#e6914e"),
    ("confabulated_other", "other specific count", "#d6443c"),
    ("didnt_mention", "didn't mention a count", "#9e9e9e"),
    ("refused", "refused", "#41ab5d"),
]

# framing order on the x-axis + how to identify each record's framing
FRAMING_ORDER = [
    ("A: reformulation\n(open-ended)", ("family", "A_reformulation")),
    ("B: indirect\n(forced-choice)", ("family", "B_indirect_conventional")),
    ("C: counter-\nassociation", ("family", "C_counter_association")),
    ("F1 direct\nrecall", ("framing", "1")),
    ("F2 decoy\ncorrection", ("framing", "2")),
    ("F3 topic\nonly", ("framing", "3")),
    ("F4 negation", ("framing", "4")),
    ("F5 multi-hop", ("framing", "5")),
    ("F6 in-context\nconflict", ("framing", "6")),
    ("F7 elaboration", ("framing", "7")),
    ("F8 negative\ncontrol", ("framing", "8")),
    ("F9 indirect\nattribute", ("framing", "9")),
    ("F10 novel\ndecoy", ("framing", "10")),
    ("F11 embedded\nlist", ("framing", "11")),
    ("free-form", ("family", "freeform5")),
]


def _framing_of(rec: dict) -> str | None:
    fam = rec.get("family")
    for label, (kind, val) in FRAMING_ORDER:
        if kind == "family" and fam == val:
            return label
        if kind == "framing" and fam == "framing381" and str(rec.get("sub_framing")) == val:
            return label
    return None


def _category(rec: dict) -> str | None:
    v = rec.get("verdict")
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    if isinstance(v, dict):
        return v.get("output_category_5way")
    return None


def load_records() -> list[dict]:
    tok = os.environ.get("HF_TOKEN")
    recs: list[dict] = []
    for s in (42, 137, 256):
        p = hf_hub_download(
            REPO_ID,
            f"{HF_KEY}/judged_on_policy_suppression_cn_seed{s}.jsonl",
            repo_type="dataset",
            token=tok,
        )
        with open(p) as fh:
            recs.extend(json.loads(line) for line in fh)
    return recs


def main() -> int:
    set_paper_style("blog")
    recs = load_records()
    persona_to_group = {p: g for g, ps in GROUPS for p in ps}
    framing_labels = [f[0] for f in FRAMING_ORDER]

    # counts[group][framing] = Counter(category)
    counts: dict[str, dict[str, Counter]] = {
        g: {fl: Counter() for fl in framing_labels} for g, _ in GROUPS
    }
    for r in recs:
        g = persona_to_group.get(r.get("persona"))
        if g is None:
            continue
        fl = _framing_of(r)
        cat = _category(r)
        if fl is None or cat is None:
            continue
        counts[g][fl][cat] += 1

    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(14.5, 9.5), sharex=True)
    x = np.arange(len(framing_labels))
    cat_keys = [c[0] for c in CATS]
    cat_colors = {c[0]: c[2] for c in CATS}
    for ax, (glabel, _) in zip(axes, GROUPS, strict=True):
        bottoms = np.zeros(len(framing_labels))
        for ck in cat_keys:
            seg = []
            for fl in framing_labels:
                tot = sum(counts[glabel][fl].values())
                seg.append(counts[glabel][fl][ck] / tot if tot else 0.0)
            seg = np.array(seg)
            ax.bar(
                x, seg, 0.74, bottom=bottoms, color=cat_colors[ck], edgecolor="white", linewidth=0.4
            )
            bottoms += seg
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(glabel, fontsize=10)
        ax.margins(x=0.01)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(framing_labels, fontsize=8, rotation=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[2]) for c in CATS]
    fig.legend(
        [h for h in handles],
        [c[1] for c in CATS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "What kinds of answers each persona gives under each framing "
        "(on-policy arm, 3-seed pool; 100%-stacked answer category)",
        y=1.045,
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig_paper(fig, stem="answer_kinds_by_framing_persona", dir=FIG_DIR)
    plt.close(fig)
    print("wrote figure to", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
