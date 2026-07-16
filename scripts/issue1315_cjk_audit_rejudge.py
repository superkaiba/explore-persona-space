"""#1315 fold — redraw the CJK-intrusion bounds figure with the 529-recovered judge draws.

The 529 re-judge (`scripts/issue1315_rejudge_529.py`) recovered all 827
transport-dropped draws across the six install-control pools with zero
band/parity verdict flips. This script recomputes the per-pool CJK-intrusion
audit (`scripts/issue1315_cjk_audit.py`) over the MERGED per-item draw sets
(original kept draws + recovered draws, from the committed
`judge_rejudge_529/*.json` per-draw provenance) and redraws
`figures/issue_1315/cjk_intrusion_bounds.{png,pdf,meta.json}` so the printed
bar labels match the post-recovery rates. Writes the recomputed rows to
`eval_results/issue_1315/selection/cjk_intrusion_audit_rejudge.json`; the
original `cjk_intrusion_audit.json` (pre-recovery pass) is left untouched as
the record of the original run.
"""

from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

REPO_ROOT = _SCRIPTS_DIR.parent
SEL_DIR = REPO_ROOT / "eval_results" / "issue_1315" / "selection"
REJUDGE_DIR = SEL_DIR / "judge_rejudge_529"
FIG_DIR = REPO_ROOT / "figures" / "issue_1315"

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from issue1315_cjk_audit import (  # noqa: E402
    BAND,
    CJK,
    PARITY_WINDOW,
    POOLS,
    PREFIX,
    REPO,
    REV,
    THRESH,
)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

# audit tag -> rejudge pool key (judge_rejudge_529/<key>.json)
TAG_TO_REJUDGE = {
    "persona parity": "parity_imp_pers_lora",
    "WildChat parity": "parity_imp_conv_lora",
    "ICL + negatives parity": "parity_imp_icl_lora_neg",
    "ICL positives-only parity": "parity_imp_icl_lora_pos",
    "Full FT + negatives Tier-2": "tier2_imp_icl_ft_neg",
    "Full FT positives-only Tier-2": "tier2_imp_icl_ft_pos",
}


def _audit_pool_merged(subdir: str, ctx: str, rejudge: dict) -> dict:
    """Recompute the pool's audit row over original-kept + recovered draws."""
    comp_path = f"{PREFIX}/{subdir}/completions__trained__{ctx}.json"
    judge_path = f"{PREFIX}/{subdir}/judge/trained_{ctx}/judge_raw.json"
    comp = json.loads(
        Path(hf_hub_download(REPO, comp_path, repo_type="dataset", revision=REV)).read_text()
    )
    judge = json.loads(
        Path(hf_hub_download(REPO, judge_path, repo_type="dataset", revision=REV)).read_text()
    )
    comps = comp["completions"]
    nq, nc = len(comps), len(comps[0])
    draws: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for key, val in judge["all_scores"].items():
        m = re.search(r"q(\d+)-c(\d+)__", key)
        assert m is not None, key
        item = (int(m.group(1)), int(m.group(2)))
        draws[item]  # register the item even if every draw errored
        if isinstance(val, dict) and isinstance(val.get("score"), (int, float)):
            draws[item].append(float(val["score"]))
    n_recovered = 0
    for item_id, rec in rejudge["per_item"].items():
        m = re.search(r"q(\d+)-c(\d+)$", item_id)
        assert m is not None, item_id
        item = (int(m.group(1)), int(m.group(2)))
        got = [float(s) for s in rec.get("recovered", [])]
        assert sorted(rec.get("orig_kept", [])) == sorted(draws[item]), item_id
        draws[item].extend(got)
        n_recovered += len(got)
    assert n_recovered == rejudge["recovery"]["n_recovered_kept"], (
        n_recovered,
        rejudge["recovery"]["n_recovered_kept"],
    )
    items = [(qi, ci) for qi in range(nq) for ci in range(nc)]
    scored = [it for it in items if draws[it]]
    fired = {it for it in scored if statistics.mean(draws[it]) > THRESH}
    intruded = {it for it in items if CJK.search(comps[it[0]][it[1]])}
    intruded_scored = intruded & set(scored)
    intruded_fired = intruded & fired
    n_s, n_f = len(scored), len(fired)
    assert n_s == rejudge["after"]["n_scored"] and n_f == rejudge["after"]["k"], (subdir, n_s, n_f)
    return {
        "n_items": len(items),
        "n_scored": n_s,
        "n_fired": n_f,
        "realized_rate": round(n_f / n_s, 4),
        "n_intruded": len(intruded),
        "n_intruded_scored": len(intruded_scored),
        "n_intruded_fired": len(intruded_fired),
        "zeroed_rate": round((n_f - len(intruded_fired)) / n_s, 4),
        "excluded_rate": round((n_f - len(intruded_fired)) / (n_s - len(intruded_scored)), 4),
    }


def main() -> int:
    """Recompute all six pools post-recovery, write the JSON, redraw the figure."""
    out: dict[str, dict] = {
        "_meta": {
            "revision": REV,
            "cjk_regex": "[\\u4e00-\\u9fff] (CJK Unified Ideographs; the #1090 fu4 class)",
            "threshold": THRESH,
            "band": BAND,
            "parity_window": PARITY_WINDOW,
            "judge_draws": "original kept draws + the 529-recovered draws "
            "(eval_results/issue_1315/selection/judge_rejudge_529/)",
            "conventions": {
                "zeroed_rate": "(fired - fired_intruded) / scored — every intruded firing "
                "completion scored non-impolite",
                "excluded_rate": "(fired - fired_intruded) / (scored - intruded_scored)",
            },
        }
    }
    for tag, (kind, subdir, ctx, committed) in POOLS.items():
        rejudge = json.loads((REJUDGE_DIR / f"{TAG_TO_REJUDGE[tag]}.json").read_text())
        row = _audit_pool_merged(subdir, ctx, rejudge)
        row["kind"] = kind
        if committed is not None:
            row["committed_rate"] = committed
            row["pass_floor"] = round(committed - PARITY_WINDOW, 4)
        else:
            row["pass_floor"] = BAND[0]
        out[tag] = row
        print(tag, row)
    (SEL_DIR / "cjk_intrusion_audit_rejudge.json").write_text(json.dumps(out, indent=1))

    set_paper_style("blog")
    tags = [t for t in POOLS]
    colors = paper_palette_blog(3)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.26
    for j, (dv, label) in enumerate(
        [
            ("realized_rate", "realized rate"),
            ("zeroed_rate", "intruded rows scored non-impolite"),
            ("excluded_rate", "intruded rows excluded"),
        ]
    ):
        xs = [i + (j - 1) * width for i in range(len(tags))]
        ys = [out[t][dv] for t in tags]
        ax.bar(xs, ys, width=width, color=colors[j], label=label)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=7.5)
    for i, t in enumerate(tags):
        ax.hlines(out[t]["pass_floor"], i - 0.42, i + 0.42, color="black", lw=1.4, ls="--")
    ax.axhspan(BAND[0], BAND[1], color="gray", alpha=0.10)
    ax.set_xticks(
        range(len(tags)),
        [t.replace(" parity", "\nparity").replace(" Tier-2", "\nTier-2") for t in tags],
        fontsize=8.5,
    )
    ax.set_ylabel("Judged impolite rate")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8.5, loc="upper left")
    savefig_paper(fig, "cjk_intrusion_bounds", dir=FIG_DIR)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
