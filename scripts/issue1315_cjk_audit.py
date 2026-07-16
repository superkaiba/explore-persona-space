"""#1315 r2 — CJK-intrusion audit of the temperature-1.0 judged install pools.

The greedy capture rollouts (the geometry substrate) were audited pod-side
(`text_audit.json`, near-zero intrusion); this script audits the six JUDGED
pools that carry the install-control adjudications (2 Tier-2 dose-confirm
pools + 4 reused-organism parity pools), which the pod-side audit did not
cover. Same CJK class as the #1090 fu4 lineage audit (U+4E00-U+9FFF).

Per pool it reports: intruded completions, intruded-and-judged-impolite
overlap, the realized rate over scored items (mean judge score > 50 across
surviving draws; 529-error draws drop per the run's rule-24 deviation), the
intrusion-ZEROED bound ((fired - fired_intruded) / scored), and the
intrusion-EXCLUDED recount ((fired - fired_intruded) / (scored - intruded)).

Writes eval_results/issue_1315/selection/cjk_intrusion_audit.json and
figures/issue_1315/cjk_intrusion_bounds.{png,pdf,meta.json}.
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
FIG_DIR = REPO_ROOT / "figures" / "issue_1315"

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REV = "7219f7c03b529e107aaf4fa548169977403f0131"
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1315_impolite_geometry/raw_completions"
CJK = re.compile(r"[一-鿿]")  # CJK Unified Ideographs (the #1090 fu4 class)
THRESH = 50.0
BAND = (0.60, 0.85)  # Tier-2 install band
PARITY_WINDOW = 0.15  # |realized - committed| <= 0.15 => PASS

# pool tag -> (kind, subdir, completion ctx slug, judge ctx slug, committed rate | None)
POOLS = {
    "persona parity": (
        "parity",
        "parity/imp_pers_lora/rate_merged_parity",
        "persona_software_engineer",
        0.805,
    ),
    "WildChat parity": (
        "parity",
        "parity/imp_conv_lora/rate_merged_parity",
        "wildchat_prefix_real545",
        0.737,
    ),
    "ICL + negatives parity": (
        "parity",
        "parity/imp_icl_lora_neg/rate_merged_parity",
        "icl_prefix_impolite",
        0.82,
    ),
    "ICL positives-only parity": (
        "parity",
        "parity/imp_icl_lora_pos/rate_merged_parity",
        "icl_prefix_impolite",
        0.775,
    ),
    "Full FT + negatives Tier-2": (
        "tier2",
        "tier2/imp_icl_ft_neg/rate_checkpoint-4",
        "icl_prefix_impolite",
        None,
    ),
    "Full FT positives-only Tier-2": (
        "tier2",
        "tier2/imp_icl_ft_pos/rate_checkpoint-18",
        "icl_prefix_impolite",
        None,
    ),
}


def _audit_pool(subdir: str, ctx: str) -> dict:
    """Join a pool's completions to its judge_raw draws; return the audit row."""
    comp_path = f"{PREFIX}/{subdir}/completions__trained__{ctx}.json"
    judge_path = f"{PREFIX}/{subdir}/judge/trained_{ctx}/judge_raw.json"
    comp = json.load(open(hf_hub_download(REPO, comp_path, repo_type="dataset", revision=REV)))
    judge = json.load(open(hf_hub_download(REPO, judge_path, repo_type="dataset", revision=REV)))
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
    items = [(qi, ci) for qi in range(nq) for ci in range(nc)]
    scored = [it for it in items if draws[it]]
    fired = {it for it in scored if statistics.mean(draws[it]) > THRESH}
    intruded = {it for it in items if CJK.search(comps[it[0]][it[1]])}
    intruded_scored = intruded & set(scored)
    intruded_fired = intruded & fired
    n_s, n_f = len(scored), len(fired)
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
    """Audit all six pools, write the JSON, draw the bounds figure."""
    out: dict[str, dict] = {
        "_meta": {
            "revision": REV,
            "cjk_regex": "[\\u4e00-\\u9fff] (CJK Unified Ideographs; the #1090 fu4 class)",
            "threshold": THRESH,
            "band": BAND,
            "parity_window": PARITY_WINDOW,
            "conventions": {
                "zeroed_rate": "(fired - fired_intruded) / scored — every intruded firing "
                "completion scored non-impolite",
                "excluded_rate": "(fired - fired_intruded) / (scored - intruded_scored)",
            },
        }
    }
    for tag, (kind, subdir, ctx, committed) in POOLS.items():
        row = _audit_pool(subdir, ctx)
        row["kind"] = kind
        if committed is not None:
            row["committed_rate"] = committed
            row["pass_floor"] = round(committed - PARITY_WINDOW, 4)
        else:
            row["pass_floor"] = BAND[0]
        out[tag] = row
        print(tag, row)
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    (SEL_DIR / "cjk_intrusion_audit.json").write_text(json.dumps(out, indent=1))

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
