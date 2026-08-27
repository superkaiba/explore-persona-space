"""Issue #2546 manipulation check (plan §5 `ctl_manip`): per-corpus pre/post exact-match accuracy.

Derives the planned manipulation check entirely from the committed
``eval_results/issue_2546/necessity/*.json`` ``class_sizes`` contingencies —
zero GPU, zero new generation.  Definitions (shared ``definition`` string in
all three files):

    necessary(q) := exact-match correct(think/post) AND NOT correct(no-think/pre)

Arms 1-2 class vocabulary: ``both_correct``, ``necessary``, ``pre_only_correct``,
``both_wrong``, ``unknown``.  Arm 3 (the Qwen3-8B on/off toggle) uses
``rescued_by_no_think`` where arms 1-2 use ``pre_only_correct`` ("pre" = think-off
there); the two vocabularies are deliberately NOT unified.

    pre_acc  = (both_correct + pre_only_correct[or rescued_by_no_think]) / denom
    post_acc = (both_correct + necessary) / denom

``denom`` EXCLUDES ``unknown`` (rows where exact-match could not be scored on at
least one side); the ``unknown`` share per corpus is reported alongside so the
exclusion is auditable.  Output: ``manipulation_check.json`` next to the inputs
plus a markdown table on stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NEC = HERE / "eval_results" / "issue_2546" / "necessity"

FILES = {
    "arm1 (OpenThinker3 vs Qwen2.5-Instruct)": ("pair_necessity_a1.json", "pre_only_correct"),
    "arm2 (R1-Distill vs Qwen2.5-Math)": ("pair_necessity_a2.json", "pre_only_correct"),
    "arm3 (Qwen3-8B think on/off)": ("qwen3_toggle_labels.json", "rescued_by_no_think"),
}


def main() -> None:
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for arm, (fname, pre_key) in FILES.items():
        data = json.loads((NEC / fname).read_text())
        sizes = data["class_sizes"]
        arm_rows: dict[str, dict[str, float | int]] = {}
        for corpus in sorted(sizes):
            cs = sizes[corpus]
            expected = {"both_correct", "necessary", pre_key, "both_wrong", "unknown"}
            assert set(cs) == expected, f"{fname}:{corpus} unexpected classes {sorted(cs)}"
            total = sum(cs.values())
            denom = total - cs["unknown"]
            assert denom > 0, f"{fname}:{corpus} empty denominator"
            arm_rows[corpus] = {
                "pre_acc": round((cs["both_correct"] + cs[pre_key]) / denom, 4),
                "post_acc": round((cs["both_correct"] + cs["necessary"]) / denom, 4),
                "denom_excl_unknown": denom,
                "unknown": cs["unknown"],
                "unknown_share": round(cs["unknown"] / total, 4),
            }
        out[arm] = arm_rows

    dest = NEC / "manipulation_check.json"
    dest.write_text(
        json.dumps(
            {
                "definition": "pre_acc=(both_correct+pre_only_correct[/rescued_by_no_think])/denom; "
                "post_acc=(both_correct+necessary)/denom; denom excludes unknown",
                "source": sorted(v[0] for v in FILES.values()),
                "accuracy": out,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {dest}")
    for arm, rows in out.items():
        print(f"\n{arm}")
        print("| corpus | pre acc | post acc | n (excl. unknown) | unknown share |")
        print("|---|---|---|---|---|")
        for corpus, r in rows.items():
            print(
                f"| {corpus} | {r['pre_acc']:.3f} | {r['post_acc']:.3f} "
                f"| {r['denom_excl_unknown']:,} | {r['unknown_share']:.1%} |"
            )


if __name__ == "__main__":
    main()
