"""Split-half reliability ceilings for #1739, all three behaviors, per rung.

Uses the CANONICAL helper `experiments.issue_1739.arms.split_half_ceiling`
(item-aligned even-odd rollout split -> Spearman -> Spearman-Brown) so the
numbers are directly comparable to the committed per-cell `ceiling_sb`.

Hallucination has no committed ceiling because its DV rows carry no
`per_rollout_scores`; this reads the derived per-rollout scalars from the
#1739 gap-fill round (labeling_per_rollout.json, gated 115,940/115,940
per-rollout and 23,188/23,188 per-context).

Also reports the implied SINGLE-ROLLOUT reliability r1 by inverting
Spearman-Brown, r_k = k*r1 / (1 + (k-1)*r1)  =>  r1 = r_k / (k - (k-1)*r_k),
with k = the realized rollout count. That is the ceiling that would apply to
predicting ONE response rather than the 5-rollout mean.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_1739.arms import split_half_ceiling  # noqa: E402

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WT = ROOT / ".claude/worktrees/issue-1739/eval_results/issue_1739"
GAP = ROOT / ".claude/worktrees/i1739-gapgrid/eval_results/issue_1739"

SOURCES = {
    "evil": WT / "dv_dataset/evil/labeling.json",
    "sycophancy": WT / "dv_dataset/sycophancy/labeling.json",
    "hallucination": GAP / "dv_dataset/hallucination/labeling_per_rollout.json",
}


def per_rollout_matrix(rows):
    """(n_ctx, K) float array of per-rollout scores; missing -> nan."""
    keys = set()
    for r in rows:
        keys.update((r.get("per_rollout_scores") or {}).keys())
    if not keys:
        return None
    k_max = 1 + max(int(k[1:]) for k in keys)
    a = np.full((len(rows), k_max), np.nan)
    for i, r in enumerate(rows):
        for key, s in (r.get("per_rollout_scores") or {}).items():
            a[i, int(key[1:])] = np.nan if s is None else float(s)
    return a


def invert_sb(r_k: float, k: int) -> float:
    """Single-unit reliability implied by a k-unit reliability."""
    denom = k - (k - 1) * r_k
    return float("nan") if denom == 0 else r_k / denom


OUT_JSON = ROOT / "eval_results/issue_1739/reliability/ceilings.json"


def main() -> int:
    out: dict[str, dict] = {}
    for behavior, path in SOURCES.items():
        if not path.exists():
            print(f"=== {behavior}: SOURCE MISSING {path}")
            continue
        payload = json.loads(path.read_text())
        by_rung = defaultdict(list)
        for r in payload["rows"]:
            by_rung[r.get("rung")].append(r)

        print(f"=== {behavior}   ({path.name})")
        out[behavior] = {"source": str(path), "rungs": {}}
        all_rows = payload["rows"]
        for rung in sorted(by_rung) + ["ALL"]:
            rows = all_rows if rung == "ALL" else by_rung[rung]
            a = per_rollout_matrix(rows)
            if a is None:
                print(f"    {rung:16s} no per_rollout_scores")
                continue
            res = split_half_ceiling(a)
            ceil = res.get("ceiling_sb")
            if ceil is None:
                print(f"    {rung:16s} n={res['n']:6d}  ceiling=None (too few)")
                continue
            k = int(np.isfinite(a).sum(axis=1).max())
            uniq = len(np.unique(np.round(np.nanmean(a, axis=1), 6)))
            r1 = invert_sb(ceil, k)
            # n_rows is the rung's context count; res["n"] is the SCOREABLE
            # subset (a context whose halves are both all-dropped is excluded --
            # material for evil, whose REFUSAL draws are content-drops).
            out[behavior]["rungs"][rung] = {
                "n_rows": len(rows),
                "n_scoreable": res["n"],
                "r_half": res["r_half"],
                "ceiling_sb": ceil,
                "single_rollout_r1": r1,
                "K": k,
                "distinct_dv_values": uniq,
                "scheme": res.get("scheme"),
            }
            print(
                f"    {rung:16s} n={res['n']:6d}  r_half={res['r_half']:.4f}  "
                f"ceiling_sb={ceil:.4f}  single-rollout r1={r1:.4f}  "
                f"K={k}  distinct_dv={uniq}"
            )
        print()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
