"""Round-3 analyzer recomputes for issue #2223 (interp-critique v2 blockers 1 + 3).

Appends two blocks to eval_results/issue_2223/leg_32b/capping_composition.json:

  pooled_band_gaps - per-turn gap (capped ci_lo - uncapped ci_hi) between the two arms'
      95% conversation-level bootstrap bands, replicated in the EXACT figure order of
      scripts/issue2223_analyzer_figs.py (one module-level np.random.default_rng(42)
      stream; pooled_series(A0) consumed first, then pooled_series(A1)), so the values
      ARE the bands rendered in arm_traj_a0_a1_ci.png. Validation: the replicated mean
      lines must match the committed figure sidecar (arm_traj_a0_a1_ci.meta.json)
      exactly (max abs deviation asserted 0.0). Positive gap at every turn = the bands
      are disjoint at every turn, turn 1 included.

  per_domain_reduction_ci - 95% conversation-level bootstrap CI (2,000 draws, one fresh
      np.random.default_rng(42) stream; domains in the artifact's existing key order;
      per domain the uncapped index draw is consumed before the capped one) on each
      domain's matched-turn reduction_pct. Resampling unit: the domain's turn-1
      conversation roster (n=100 per arm), resampled with replacement; each resampled
      conversation contributes its turn-1 value and, if alive at the matched turn, its
      matched-turn value; reduction* = 100 * (1 - capped_drift*/uncapped_drift*). Draws
      where a resample has zero matched-turn survivors in either arm are dropped from
      the percentile (count reported; expected 0).

Validation before writing: the point estimates recomputed here (same expressions as
scripts/issue2223_r2_analysis.py composition()) must equal the persisted reduction_pct
values exactly (abs deviation 0.0).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE the first heavy import

import numpy as np  # noqa: E402

from issue2223_analyzer_figs import load_arm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results/issue_2223"
FIG = ROOT / "figures/issue_2223"
N_BOOT = 2000
MIN_SAMPLES = 10
SHORT = {
    "therapy-like contexts": "therapy",
    "philosophical discussions about AI": "philosophy",
    "coding assistance": "coding",
    "writing assistance": "writing",
}


def pooled_series_replica(arm: dict, rng: np.random.Generator) -> dict[int, dict[str, float]]:
    """Verbatim replica of issue2223_analyzer_figs.pooled_series on a passed stream."""
    turns = sorted({t for d in arm.values() for t in d})
    out = {}
    for t in turns:
        vals = np.array([v for d in arm.values() for (_, v) in d.get(t, [])])
        idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
        boots = vals[idx].mean(axis=1)
        qlo, qhi = np.quantile(boots, [0.025, 0.975])
        out[t] = {"mean": float(vals.mean()), "ci_lo": float(qlo), "ci_hi": float(qhi)}
    return out


def sidecar_means(path: Path) -> dict[str, dict[int, float]]:
    """series label -> turn -> mean line value, from the committed figure sidecar."""
    d = json.loads(path.read_text())
    xk = "Turn position"
    yk = "Assistant-axis projection\n(pooled over domains)"
    out: dict[str, dict[int, float]] = {}
    for e in d["points"]:
        out.setdefault(e["series"], {})[int(e[xk])] = float(e[yk])
    return out


def band_gaps(a0: dict, a1: dict) -> dict:
    rng = np.random.default_rng(42)  # figure order: A0 first, then A1, one stream
    s0 = pooled_series_replica(a0, rng)
    s1 = pooled_series_replica(a1, rng)
    side = sidecar_means(FIG / "leg_32b/arm_traj_a0_a1_ci.meta.json")
    dev = max(
        max(abs(s0[t]["mean"] - side["A0 (uncapped)"][t]) for t in s0),
        max(abs(s1[t]["mean"] - side["A1 (every-token cap)"][t]) for t in s1),
    )
    assert dev == 0.0, f"mean-line replication deviates from sidecar by {dev}"
    gaps = {t: s1[t]["ci_lo"] - s0[t]["ci_hi"] for t in sorted(set(s0) & set(s1))}
    tmin = min(gaps, key=gaps.get)
    assert all(g > 0 for g in gaps.values()), f"non-positive gap at turn {tmin}"
    return {
        "machinery": (
            "figure-order shared-RNG replication of issue2223_analyzer_figs.pooled_series "
            "(default_rng(42), A0 then A1); mean lines validated against the committed "
            "arm_traj_a0_a1_ci.meta.json sidecar at deviation 0.0"
        ),
        "gap_convention": "capped ci_lo - uncapped ci_hi; positive = bands disjoint",
        "per_turn_gap": {str(t): float(g) for t, g in gaps.items()},
        "turn1_gap": float(gaps[1]),
        "min_gap": float(gaps[tmin]),
        "min_gap_turn": tmin,
        "mean_line_validation_max_abs_dev_vs_sidecar": float(dev),
        "all_turns_disjoint": True,
    }


def domain_ci(a0: dict, a1: dict, persisted: dict) -> dict:
    rng = np.random.default_rng(42)
    out: dict = {
        "machinery": (
            "conversation-level bootstrap on the turn-1 roster (n=100 per arm, resampled "
            "with replacement; paired turn-1 / matched-turn values per conversation), "
            "2000 draws, one default_rng(42) stream, artifact key order, uncapped draw "
            "before capped per domain; percentile 2.5/97.5 over "
            "100*(1 - capped_drift*/uncapped_drift*)"
        ),
        "n_boot": N_BOOT,
        "seed": 42,
        "domains": {},
    }
    inv = {v: k for k, v in SHORT.items()}
    for short, prow in persisted["per_domain_matched_turn"].items():
        dom = inv[short]
        tmax = prow["matched_turn"]

        def arrays(arm: dict) -> tuple[np.ndarray, np.ndarray]:
            rows1 = arm[dom][1]
            vt_map = dict(arm[dom][tmax])
            v1 = np.array([v for _, v in rows1])
            vt = np.array([vt_map.get(c, np.nan) for c, _ in rows1])
            return v1, vt

        def point_drift(arm: dict) -> float:
            # verbatim r2 composition() expressions, for exact-validation
            return float(
                np.mean([v for _, v in arm[dom][tmax]]) - np.mean([v for _, v in arm[dom][1]])
            )

        dr0_pt, dr1_pt = point_drift(a0), point_drift(a1)
        red_pt = float(100 * (1 - dr1_pt / dr0_pt))
        dev = abs(red_pt - prow["reduction_pct"])
        assert dev == 0.0, f"{short}: point estimate deviates from persisted by {dev}"

        def boot_drift(arm: dict) -> np.ndarray:
            v1, vt = arrays(arm)
            idx = rng.integers(0, len(v1), size=(N_BOOT, len(v1)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows counted below
                mt = np.nanmean(vt[idx], axis=1)
            return mt - v1[idx].mean(axis=1)

        dr0_b, dr1_b = boot_drift(a0), boot_drift(a1)
        red_b = 100 * (1 - dr1_b / dr0_b)
        n_bad = int(np.isnan(red_b).sum())
        lo, hi = np.nanquantile(red_b, [0.025, 0.975])
        out["domains"][short] = {
            "matched_turn": tmax,
            "reduction_pct": red_pt,
            "ci_lo_pct": float(lo),
            "ci_hi_pct": float(hi),
            "resolved_from_zero": bool(lo > 0 or hi < 0),
            "n_degenerate_draws": n_bad,
            "point_validation_abs_dev": float(dev),
        }
    return out


def main() -> None:
    a0 = load_arm(EV / "leg_32b/phaseA_drift_trajectory.json", "A0__32b")
    a1 = load_arm(EV / "leg_32b/phaseB_arm_trajectories.json", "A1__32b")
    path = EV / "leg_32b/capping_composition.json"
    payload = json.loads(path.read_text())
    payload["pooled_band_gaps"] = band_gaps(a0, a1)
    payload["per_domain_reduction_ci"] = domain_ci(a0, a1, payload)
    path.write_text(json.dumps(payload, indent=2))
    bg = payload["pooled_band_gaps"]
    print(
        f"[bands] turn1 gap {bg['turn1_gap']:+.3f}, min {bg['min_gap']:+.3f} "
        f"at t{bg['min_gap_turn']}, all disjoint: {bg['all_turns_disjoint']}"
    )
    for k, v in payload["per_domain_reduction_ci"]["domains"].items():
        print(
            f"[ci] {k}(t{v['matched_turn']}): {v['reduction_pct']:.1f}% "
            f"[{v['ci_lo_pct']:+.1f}, {v['ci_hi_pct']:+.1f}] "
            f"resolved={v['resolved_from_zero']} bad_draws={v['n_degenerate_draws']}"
        )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
