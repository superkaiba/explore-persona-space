"""Round-2 figure data for #1415: per-pair shared/disjoint rows for the
regenerated hero / per-pair scatter / null-band figures.

Same disjoint-baseline construction as issue1415_disjoint_recount.py (split A:
target|even-c-draws, shift|odd; split B swapped; disj = mean of the splits).
Adds the all-position cells and per-pair r_B rows (steer-L20, max over read
layers — the selection every steer-L20-only arm supports) plus behavioral
alpha=4 per-arm means/SEs over per-pair means.

Output: eval_results/issue_1415/disjoint_recount_figdata.json
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "issue_1415" / "phase1"
EVAL = ROOT / "eval_results" / "issue_1415"
LAYERS = [7, 10, 14, 17, 20, 21, 24]
EVEN, ODD = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]
ARMS = ["prefix", "context"]
TRAITS = ["evil", "hallucination", "sycophancy"]


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def maxread(s_mean, c_mean, c_even, c_odd, cpr_mean):
    sh = [cos((s_mean - c_mean)[j], (cpr_mean - c_mean)[j]) for j in range(len(LAYERS))]
    a = [cos((s_mean - c_odd)[j], (cpr_mean - c_even)[j]) for j in range(len(LAYERS))]
    b = [cos((s_mean - c_even)[j], (cpr_mean - c_odd)[j]) for j in range(len(LAYERS))]
    return max(sh), 0.5 * (max(a) + max(b))


def main():
    pairs = sorted(p.stem for p in (DATA / "activations").glob("*.pt"))
    base = {}
    for pid in pairs:
        b = torch.load(DATA / "activations" / f"{pid}.pt", map_location="cpu", weights_only=True)
        c_pc = b["c"]["v_a_per_completion"].float()
        base[pid] = dict(
            c_mean=b["c"]["v_a_mean"].float(),
            c_even=c_pc[EVEN].mean(0),
            c_odd=c_pc[ODD].mean(0),
            cpr_mean=b["cprime"]["v_a_mean"].float(),
        )

    out = {"l20_delta": {}, "allpos": {}, "rb": {}, "behavioral_a4": {}}
    for arm in ARMS:
        rows, rows_ap = {}, {}
        for pid in pairs:
            bb = base[pid]
            sb = torch.load(
                DATA / "activations_steered" / "gen1c" / arm / pid / "L20" / "a4.pt",
                map_location="cpu",
                weights_only=True,
            )
            sh, dj = maxread(sb["v_a_mean"].float(), **bb)
            rows[pid] = {"shared": sh, "disj": dj}
            ap = DATA / "activations_steered" / "gen1c_allpos" / arm / pid / "L20" / "a4.pt"
            sb = torch.load(ap, map_location="cpu", weights_only=True)
            sh, dj = maxread(sb["v_a_mean"].float(), **bb)
            rows_ap[pid] = {"shared": sh, "disj": dj}
        out["l20_delta"][arm] = rows
        out["allpos"][arm] = rows_ap
    for trait in TRAITS:
        rows = {}
        for pid in pairs:
            sb = torch.load(
                DATA / "activations_steered" / "gen1d_full" / trait / pid / "a4.pt",
                map_location="cpu",
                weights_only=True,
            )
            sh, dj = maxread(sb["v_a_mean"].float(), **base[pid])
            rows[pid] = {"shared": sh, "disj": dj}
        out["rb"][trait] = rows

    bj = json.load(open(EVAL / "behavioral_judge_scores.json"))
    pi = bj["per_item"]

    def arm_stats(prefixes):
        per_pair = []
        for pid in pairs:
            vals = [
                v["graded_score"]
                for k, v in pi.items()
                if v["graded_score"] is not None
                and any(k.startswith(p.format(pid=pid)) for p in prefixes)
            ]  # graded_score None = all-draws content-dropped item (ceiling: 15 items)
            if vals:
                per_pair.append(float(np.mean(vals)))
        pp = np.array(per_pair)
        return {
            "mean": float(pp.mean()),
            "se": float(pp.std(ddof=1) / np.sqrt(len(pp))),
            "n_pairs": len(pp),
        }

    out["behavioral_a4"] = {
        "baseline": arm_stats(["gen1b/{pid}/c/"]),
        "ceiling": arm_stats(["gen1b/{pid}/cprime/"]),
        "steered_prefix_a4": arm_stats(["gen1c/prefix/{pid}/L20/a4/"]),
        "steered_context_a4": arm_stats(["gen1c/context/{pid}/L20/a4/"]),
        "allpos_prefix": arm_stats(["gen1c_allpos/prefix/{pid}/L20/a4/"]),
        "allpos_context": arm_stats(["gen1c_allpos/context/{pid}/L20/a4/"]),
        "rb_evil": arm_stats(["gen1d_full/evil/{pid}/a4/"]),
    }

    outp = EVAL / "disjoint_recount_figdata.json"
    outp.write_text(json.dumps(out, indent=1))
    print("WROTE", outp)
    for arm in ARMS:
        d = out["l20_delta"][arm]
        ap = out["allpos"][arm]
        print(
            arm,
            "L20 delta shared/disj: %.3f/%.3f | allpos %.3f/%.3f"
            % (
                np.mean([r["shared"] for r in d.values()]),
                np.mean([r["disj"] for r in d.values()]),
                np.mean([r["shared"] for r in ap.values()]),
                np.mean([r["disj"] for r in ap.values()]),
            ),
        )
    for t in TRAITS:
        d = out["rb"][t]
        print(
            "rb",
            t,
            "%.3f/%.3f"
            % (
                np.mean([r["shared"] for r in d.values()]),
                np.mean([r["disj"] for r in d.values()]),
            ),
        )
    print(json.dumps(out["behavioral_a4"], indent=1))


if __name__ == "__main__":
    main()
