"""Comment-3 response: is the refusal separation an artefact of easy, saturated pairs?

Question this answers
---------------------
Section 4.2 reports that on 61 one-word pairs that flip the model from
answering to refusing, every map prediction lands closer to its own answer
than to the other one, and that the copy baseline does the same. A reviewer
objects that those pairs are not really minimal in the way that matters: both
members are far apart along the refusal direction, so a two-alternative read
is nearly free, and asks for a more sensitive analysis on prompts near the
refusal boundary.

This round runs that analysis on the banked one-word safety swaps. Nothing
new is generated. Three things are measured:

  1. PREMISE. How far apart the two answers of a pair actually sit along the
     refusal direction, by stratum. If the flip pairs are far apart and the
     others are not, the objection's premise holds.
  2. SEPARATION BY STRATUM. The two-alternative read, recomputed on the
     graded and non-flipping pairs rather than only on the saturated flips,
     for the map and for the copy baseline.
  3. GRADED READ. Instead of a binary hit, the rank correlation between the
     predicted and the observed movement along the refusal direction, and
     against the continuous refusal margin, per stratum, map versus copy.

Boundary coordinate
-------------------
Each prompt carries a judge refusal rate over 10 rollouts and a continuous
refusal margin whose correlation with that rate is banked at rho = 0.74. A
prompt with a rate strictly inside (0, 1), or a margin near zero, is one the
model is undecided about, which is the near-boundary regime the comment asks
for. Pair strata:

  saturated  |flip| == 1        both endpoints at 0 or 1 rate, opposite  -- the easy regime
  graded     0 < |flip| < 1     at least one endpoint undecided
  none       flip == 0          no behavioural change

Two-alternative read
--------------------
For one pair the assignment ``pred_a -> obs_a, pred_b -> obs_b`` beats the
swapped assignment exactly when

    ||p_a - o_a||^2 + ||p_b - o_b||^2  <  ||p_a - o_b||^2 + ||p_b - o_a||^2
    <=>  (p_a - p_b) . (o_a - o_b) > 0
    <=>  cos(pred shift, observed shift) > 0

so the banked per-pair cosine IS the two-alternative outcome, and the shipped
"all 61 separated" claim is the statement that this cosine is positive on
every flip pair. The script asserts that equivalence reproduces the shipped
counts before using it on the other strata. Stating the criterion this way is
itself part of the answer: a hit only requires the predicted shift to fall in
the correct half-space, which is a much weaker bar than the mean cosine of
0.80 suggests.

No new fit
----------
All quantities are read from banked per-pair scalars, so there is no
n_train-vs-d question. Reused rather than reimplemented:
``analysis.paired_ci.paired_bootstrap_rho_delta`` for the paired map-versus-copy
comparison and ``analysis.paper_plots.proportion_ci`` for the hit-rate
intervals.

Caveat carried into the writeup: the refusal direction is the mean observed
flip-pair shift, scored leave-one-out for members, so the flip stratum's
projection is partly self-referential while the graded and non-flip strata are
scored against a direction they did not help define. That asymmetry favours
the flip stratum, which strengthens rather than weakens any finding that the
graded strata hold up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/scipy imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.paired_ci import (  # noqa: E402
    paired_bootstrap_rho_delta,
)
from explore_persona_space.analysis.paper_plots import proportion_ci  # noqa: E402
from explore_persona_space.task_workflow import repo_root  # noqa: E402

MAP_ARM = "arm_779ce"
COPY_ARM = "arm_iddelta"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stratum(flip: float) -> str:
    a = abs(flip)
    if a >= 1.0:
        return "saturated"
    if a > 0.0:
        return "graded"
    return "none"


def _sep(cos: np.ndarray) -> dict:
    """Two-alternative hit rate; a hit is cos > 0 (see module docstring)."""
    hits = int((cos > 0).sum())
    n = int(cos.size)
    lo, hi = proportion_ci(hits / n, n) if n else (float("nan"), float("nan"))
    return {"n": n, "hits": hits, "acc": hits / n if n else float("nan"), "ci95": [lo, hi]}


def _graded(pred: np.ndarray, copy: np.ndarray, dv: np.ndarray, n_boot: int, seed: int) -> dict:
    if pred.size < 4:
        return {"n": int(pred.size), "note": "too few pairs for a rank correlation"}
    rho_map, rho_copy, lo, hi = paired_bootstrap_rho_delta(pred, copy, dv, n_boot=n_boot, seed=seed)
    return {
        "n": int(pred.size),
        "rho_map": float(rho_map),
        "rho_copy": float(rho_copy),
        "rho_map_p": float(spearmanr(pred, dv).pvalue),
        "delta_map_minus_copy": float(rho_map - rho_copy),
        "delta_ci95": [float(lo), float(hi)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=26170)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = repo_root()
    src = root / "eval_results/issue_2617/svmp_verbharm"
    rows = _rows(src / "perpair.jsonl")
    banked = json.loads((src / "summary.json").read_text())

    flip = np.array([r["flip"] for r in rows], dtype=np.float64)
    strat = np.array([_stratum(f) for f in flip])
    rate_a = np.array([r["refusal_rate_a"] for r in rows], dtype=np.float64)
    rate_b = np.array([r["refusal_rate_b"] for r in rows], dtype=np.float64)
    margin_a = np.array([r["margin_a"] for r in rows], dtype=np.float64)
    margin_b = np.array([r["margin_b"] for r in rows], dtype=np.float64)
    margin_delta = np.array([r["margin_delta"] for r in rows], dtype=np.float64)
    len_delta = np.array([r["ans_len_delta"] for r in rows], dtype=np.float64)
    norm_obs = np.array([r["norm_obs_tail"] for r in rows], dtype=np.float64)
    axis_cos_obs = np.array([r["axis_cos_obs"] for r in rows], dtype=np.float64)
    cos = {
        "map": np.array([r[f"cos_{MAP_ARM}"] for r in rows], dtype=np.float64),
        "copy": np.array([r[f"cos_{COPY_ARM}"] for r in rows], dtype=np.float64),
    }
    proj_pred = {
        "map": np.array(
            [r[f"axis_cos_pred_{MAP_ARM}"] * r[f"norm_pred_{MAP_ARM}"] for r in rows],
            dtype=np.float64,
        ),
        "copy": np.array(
            [r[f"axis_cos_pred_{COPY_ARM}"] * r[f"norm_pred_{COPY_ARM}"] for r in rows],
            dtype=np.float64,
        ),
    }
    # observed movement along the refusal direction: the DV the graded read targets
    proj_obs = axis_cos_obs * norm_obs

    # --- 1. premise: how far apart are a pair's two answers along refusal? ------
    premise = {}
    for name in ("saturated", "graded", "none"):
        sel = strat == name
        if not sel.any():
            continue
        premise[name] = {
            "n_pairs": int(sel.sum()),
            "abs_refusal_axis_separation": {
                "median": float(np.median(np.abs(proj_obs[sel]))),
                "iqr": [
                    float(np.percentile(np.abs(proj_obs[sel]), 25)),
                    float(np.percentile(np.abs(proj_obs[sel]), 75)),
                ],
            },
            "abs_answer_shift_norm_median": float(np.median(norm_obs[sel])),
            "abs_answer_length_delta_median": float(np.median(np.abs(len_delta[sel]))),
            "abs_margin_delta_median": float(np.median(np.abs(margin_delta[sel]))),
        }

    # --- 2. separation by stratum, map vs copy ---------------------------------
    separation = {}
    for name in ("saturated", "graded", "none"):
        sel = strat == name
        if not sel.any():
            continue
        separation[name] = {arm: _sep(cos[arm][sel]) for arm in ("map", "copy")}
    # the near-boundary cut the comment asks for, defined on the prompts themselves
    undecided = ((rate_a > 0) & (rate_a < 1)) | ((rate_b > 0) & (rate_b < 1))
    separation["undecided_endpoint"] = {arm: _sep(cos[arm][undecided]) for arm in ("map", "copy")}
    separation["all_pairs"] = {arm: _sep(cos[arm]) for arm in ("map", "copy")}

    # Fail loud if the cos>0 equivalence no longer reproduces the shipped flip count.
    n_flip_banked = int(banked["flip_groups"]["n_flip"])
    flip_hi = float(banked["flip_groups"]["flip_hi"])
    shipped_flip = np.abs(flip) >= flip_hi
    if int(shipped_flip.sum()) != n_flip_banked:
        raise AssertionError(
            f"flip group size {int(shipped_flip.sum())} != banked {n_flip_banked}; "
            "the flip threshold convention has drifted"
        )
    shipped = {
        "n_flip_pairs": n_flip_banked,
        "flip_hi_threshold": flip_hi,
        "map": _sep(cos["map"][shipped_flip]),
        "copy": _sep(cos["copy"][shipped_flip]),
    }

    # --- 3. graded read: predicted vs observed movement along refusal ----------
    graded = {}
    for name, sel in (
        ("all_pairs", np.ones_like(flip, dtype=bool)),
        ("saturated", strat == "saturated"),
        ("graded", strat == "graded"),
        ("none", strat == "none"),
        ("undecided_endpoint", undecided),
        ("graded_or_none", strat != "saturated"),
    ):
        if not sel.any():
            continue
        graded[name] = {
            "vs_observed_refusal_movement": _graded(
                proj_pred["map"][sel], proj_pred["copy"][sel], proj_obs[sel], args.n_boot, args.seed
            ),
            "vs_refusal_margin_delta": _graded(
                proj_pred["map"][sel],
                proj_pred["copy"][sel],
                margin_delta[sel],
                args.n_boot,
                args.seed,
            ),
            "vs_answer_length_delta_confound": _graded(
                proj_pred["map"][sel],
                proj_pred["copy"][sel],
                len_delta[sel],
                args.n_boot,
                args.seed,
            ),
        }

    coverage = {
        "n_pairs": len(rows),
        "n_prompts": 2 * len(rows),
        "n_prompts_saturated": int(
            ((rate_a == 0) | (rate_a == 1)).sum() + ((rate_b == 0) | (rate_b == 1)).sum()
        ),
        "n_prompts_undecided": int(
            ((rate_a > 0) & (rate_a < 1)).sum() + ((rate_b > 0) & (rate_b < 1)).sum()
        ),
        "n_pairs_by_stratum": {
            name: int((strat == name).sum()) for name in ("saturated", "graded", "none")
        },
        "n_pairs_with_undecided_endpoint": int(undecided.sum()),
        "median_abs_margin_all_prompts": float(
            np.median(np.abs(np.concatenate([margin_a, margin_b])))
        ),
    }

    payload = {
        "question": "comment 3: does the refusal separation survive near the decision boundary",
        "arms": {"map": MAP_ARM, "copy": COPY_ARM},
        "two_alternative_criterion": "cos(predicted shift, observed shift) > 0 (assignment form)",
        "boundary_coordinate": "per-prompt judge refusal rate over 10 rollouts + continuous margin",
        "refusal_direction": banked["notes"]["axis"],
        "bootstrap": {"unit": "pair", "draws": args.n_boot, "seed": args.seed},
        "sources": ["eval_results/issue_2617/svmp_verbharm/perpair.jsonl"],
        "coverage": coverage,
        "shipped_claim_reproduction": shipped,
        "premise_answer_separation": premise,
        "separation_by_stratum": separation,
        "graded_read": graded,
    }
    out = args.out or (root / "eval_results/issue_2617/comment3_refusal_boundary/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    print("coverage:", json.dumps(coverage, indent=1))
    print(
        f"\nshipped 61-pair claim: map {shipped['map']['hits']}/{shipped['map']['n']}, "
        f"copy {shipped['copy']['hits']}/{shipped['copy']['n']}"
    )
    print("\npremise -- how far apart the two answers sit along the refusal direction:")
    for name, p in premise.items():
        print(
            f"  {name:10s} n={p['n_pairs']:3d}  |proj| median={p['abs_refusal_axis_separation']['median']:6.2f}"
            f"  ||dh_A||={p['abs_answer_shift_norm_median']:6.2f}"
            f"  |dlen|={p['abs_answer_length_delta_median']:7.1f} tokens"
        )
    print("\nseparation (two-alternative hit rate), map vs copy:")
    for name, s in separation.items():
        m, c = s["map"], s["copy"]
        print(
            f"  {name:20s} n={m['n']:3d}  map {m['acc']:.3f} [{m['ci95'][0]:.2f},{m['ci95'][1]:.2f}]"
            f"   copy {c['acc']:.3f} [{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]"
        )
    print("\ngraded read -- rank correlation with observed movement along refusal:")
    for name, g in graded.items():
        v = g["vs_observed_refusal_movement"]
        if "rho_map" not in v:
            print(f"  {name:20s} n={v['n']:3d}  {v['note']}")
            continue
        print(
            f"  {name:20s} n={v['n']:3d}  map rho={v['rho_map']:+.3f}  copy rho={v['rho_copy']:+.3f}"
            f"  delta={v['delta_map_minus_copy']:+.3f} [{v['delta_ci95'][0]:+.3f},{v['delta_ci95'][1]:+.3f}]"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
