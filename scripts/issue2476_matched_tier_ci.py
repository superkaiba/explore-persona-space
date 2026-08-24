"""Activity-matched tier medians + contrasts with bootstrap CIs (#2476, 9a-ter free-analysis r1).

Full version of the body's quick matched-band check (task #2476, second Results
section): per-tier median held-out R2 restricted to an activity band matched to
each arm's finest-tier (tier-2) survivors, now with 10,000-draw bootstrap 95%
CIs and coarse-minus-middle / coarse-minus-finest median contrasts, for the
fresh arm, the bridge arm, and the pile twin.

Band derivation rule (reproduced from the analyzer's committed reads):
  - Each arm's band is its OWN tier-2 survivors' activity envelope
    [min(activity), max(activity)] in active fit rows (inclusive bounds).
  - The bridge arm's committed band [390, 552] IS the exact envelope.
  - The fresh arm's committed band [1400, 2400] is the exact envelope
    [1473, 2274] outward-rounded to grain 200; the committed rounded band is
    reproduced verbatim (self-checked against the body's point reads) and the
    exact envelope is reported as a labeled sensitivity block.
  - The pile twin (same instrument + 24,000 fit rows as the bridge arm) gets
    its own exact envelope [250, 1922] by the bridge-arm convention, plus the
    bridge band [390, 552] cross-applied (the convention is arm-specific).

Estimator parity: the bootstrap helpers are IMPORTED from the committed
scripts (no re-implementation) — `_boot_median_ci` from
scripts/issue2476_turnavg_sae.py and `_boot_median_diff` from
scripts/issue1482_matryoshka_tier.py. The only delta vs reference usage is the
tier+band mask applied to the inputs before the call.

Inputs (committed, the ONLY data this script reads):
  eval_results/issue_2476/turnavg/perfeature_c_encodepred.npz  (fresh arm)
  eval_results/issue_2476/turnavg/perfeature_b_encodepred.npz  (bridge arm)
  eval_results/issue_2476/turnavg/perfeature_b_pile.npz        (pile twin)

Output: eval_results/issue_2476/turnavg/matched_tier_ci.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + env BEFORE numpy/torch (shared-VM convention)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script imports
from issue1482_matryoshka_tier import _boot_median_diff  # noqa: E402
from issue2476_turnavg_sae import _boot_median_ci  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

TURNAVG_DIR = Path("eval_results/issue_2476/turnavg")
INPUTS = {
    "fresh": "perfeature_c_encodepred.npz",
    "bridge": "perfeature_b_encodepred.npz",
    "pile": "perfeature_b_pile.npz",
}
TIER_NAMES = {0: "coarsest", 1: "middle", 2: "finest"}

# Body point reads to reproduce EXACTLY before any CI is computed
# (arm, band, tier) -> (median rounded to 3 dp, n).
SELF_CHECK = [
    ("fresh", (1400.0, 2400.0), 0, 0.525, 197),
    ("fresh", (1400.0, 2400.0), 1, 0.132, 41),
    ("bridge", (390.0, 552.0), 0, 0.393, 83),
    ("bridge", (1400.0, 2400.0), 0, 0.509, 22),
]


def _load(path: Path) -> dict[str, np.ndarray]:
    """Load one per-feature npz and validate the fields this analysis reads."""
    z = np.load(path)
    need = {"r2", "activity", "tier", "n_fit_rows", "alive_floor"}
    missing = need - set(z.files)
    if missing:
        raise KeyError(f"{path}: missing fields {sorted(missing)} (have {sorted(z.files)})")
    r2 = np.asarray(z["r2"], np.float64)
    act = np.asarray(z["activity"], np.float64)
    tier = np.asarray(z["tier"], np.int64)
    if r2.shape != act.shape or r2.shape != tier.shape:
        raise ValueError(f"{path}: shape mismatch r2={r2.shape} act={act.shape} tier={tier.shape}")
    if not np.isfinite(r2).any():
        raise ValueError(f"{path}: r2 is all-NaN/inf — refusing to proceed")
    if not np.isfinite(act).all():
        raise ValueError(f"{path}: non-finite activity values")
    if not set(np.unique(tier)) <= {0, 1, 2}:
        raise ValueError(f"{path}: unexpected tier values {sorted(set(np.unique(tier)))}")
    return {
        "r2": r2,
        "activity": act,
        "tier": tier,
        "n_fit_rows": int(z["n_fit_rows"]),
        "alive_floor": int(z["alive_floor"]),
    }


def _survivor_envelope(arm: dict[str, np.ndarray]) -> tuple[float, float]:
    """Exact tier-2 survivors' activity envelope [min, max] (the band rule)."""
    act = arm["activity"][arm["tier"] == 2]
    if act.size < 1:
        raise ValueError("no finest-tier survivors — cannot derive a matched band")
    return float(act.min()), float(act.max())


def _in_band(arm: dict[str, np.ndarray], band: tuple[float, float], tier: int) -> np.ndarray:
    """Finite r2 values of `tier` features inside the inclusive activity band."""
    m = (
        (arm["tier"] == tier)
        & (arm["activity"] >= band[0])
        & (arm["activity"] <= band[1])
        & np.isfinite(arm["r2"])
    )
    return arm["r2"][m]


def _block(
    arm: dict[str, np.ndarray],
    band: tuple[float, float],
    band_rule: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Per-tier matched medians + CIs and coarse-minus-{middle,finest} contrasts for one band.

    Raises if the band selects zero finite features across all tiers (empty band
    selection); a single tier with < 2 finite in-band features is recorded as an
    explicit null with the reason and the n's, never silently skipped.
    """
    vals = {t: _in_band(arm, band, t) for t in (0, 1, 2)}
    if sum(len(v) for v in vals.values()) == 0:
        raise ValueError(f"empty band selection: no finite features in band {band}")

    tiers: dict[str, dict] = {}
    for t in (0, 1, 2):
        v = vals[t]
        name = TIER_NAMES[t]
        if len(v) >= 2:
            tiers[name] = {
                "median": float(np.median(v)),
                "ci95": _boot_median_ci(v, n_boot, rng),
                "n": int(len(v)),
            }
        else:
            tiers[name] = {
                "median": float(np.median(v)) if len(v) == 1 else None,
                "ci95": None,
                "n": int(len(v)),
                "null_reason": f"only {len(v)} finite in-band feature(s); >=2 required for a CI",
            }

    contrasts: dict[str, dict] = {}
    for label, t_b in [("coarse_minus_middle", 1), ("coarse_minus_finest", 2)]:
        a, b = vals[0], vals[t_b]
        if len(a) >= 2 and len(b) >= 2:
            contrasts[label] = {
                "point": float(np.median(a) - np.median(b)),
                "ci95": _boot_median_diff(a, b, n_boot, rng),
                "n_coarse": int(len(a)),
                f"n_{TIER_NAMES[t_b]}": int(len(b)),
            }
        else:
            contrasts[label] = {
                "point": None,
                "ci95": None,
                "n_coarse": int(len(a)),
                f"n_{TIER_NAMES[t_b]}": int(len(b)),
                "null_reason": (
                    f"skipped: coarse n={len(a)}, {TIER_NAMES[t_b]} n={len(b)} in band; "
                    "both sides need >=2 finite in-band features"
                ),
            }

    return {
        "band": [band[0], band[1]],
        "band_rule": band_rule,
        "tiers": tiers,
        "contrasts": contrasts,
    }


def _run_self_check(arms: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Reproduce the body's four point reads exactly; raise on any mismatch."""
    rows = []
    for arm_name, band, tier, want_med, want_n in SELF_CHECK:
        v = _in_band(arms[arm_name], band, tier)
        got_med, got_n = float(np.median(v)), int(len(v))
        ok = round(got_med, 3) == want_med and got_n == want_n
        rows.append(
            {
                "arm": arm_name,
                "band": list(band),
                "tier": TIER_NAMES[tier],
                "expected": {"median_3dp": want_med, "n": want_n},
                "computed": {"median": got_med, "n": got_n},
                "pass": ok,
            }
        )
        if not ok:
            raise AssertionError(
                f"self-check FAILED for {arm_name} tier {tier} band {band}: "
                f"expected median~{want_med} n={want_n}, got {got_med:.6f} n={got_n} "
                "(band reproduction bug — fix the band rule, never fudge it)"
            )
    return rows


def main() -> None:
    """Compute all matched-band reads and write the output JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=TURNAVG_DIR / "matched_tier_ci.json")
    args = ap.parse_args()

    arms = {name: _load(TURNAVG_DIR / fn) for name, fn in INPUTS.items()}

    # Bands per the derivation rule (module docstring). Fail loud on any drift
    # from the committed conventions rather than silently rederiving.
    env_fresh = _survivor_envelope(arms["fresh"])
    env_bridge = _survivor_envelope(arms["bridge"])
    env_pile = _survivor_envelope(arms["pile"])
    if env_fresh != (1473.0, 2274.0):
        raise AssertionError(f"fresh tier-2 envelope drifted: {env_fresh} != (1473, 2274)")
    if env_bridge != (390.0, 552.0):
        raise AssertionError(f"bridge tier-2 envelope drifted: {env_bridge} != (390, 552)")
    band_fresh_body = (1400.0, 2400.0)  # committed body band: envelope outward-rounded, grain 200
    if not (band_fresh_body[0] <= env_fresh[0] and env_fresh[1] <= band_fresh_body[1]):
        raise AssertionError("fresh body band does not contain the exact envelope")

    self_check = _run_self_check(arms)

    rng = np.random.default_rng(args.seed)
    rule_own = "own tier-2 survivors' exact activity envelope [min, max] active fit rows, inclusive"
    # Fixed block order => deterministic rng consumption under the recorded seed.
    reads = {
        "fresh_body_band": _block(
            arms["fresh"],
            band_fresh_body,
            "committed body band: fresh tier-2 envelope [1473, 2274] outward-rounded to "
            "grain 200 -> [1400, 2400]",
            args.n_boot,
            rng,
        ),
        "fresh_exact_envelope_sensitivity": _block(
            arms["fresh"],
            env_fresh,
            f"sensitivity: {rule_own} (unrounded twin of the committed fresh band)",
            args.n_boot,
            rng,
        ),
        "bridge_own_band": _block(
            arms["bridge"], env_bridge, rule_own + " (committed body band)", args.n_boot, rng
        ),
        "bridge_under_fresh_band": _block(
            arms["bridge"],
            band_fresh_body,
            "cross-read: the fresh arm's committed absolute band applied to the bridge arm "
            "(the body's 0.509 n=22 read)",
            args.n_boot,
            rng,
        ),
        "pile_own_band": _block(
            arms["pile"],
            env_pile,
            rule_own + " (derived here by the bridge-arm convention; same instrument and "
            "24,000 fit rows)",
            args.n_boot,
            rng,
        ),
        "pile_under_bridge_band": _block(
            arms["pile"],
            env_bridge,
            "cross-read: the bridge arm's band applied to the pile twin (arm-specific "
            "convention => both bands reported)",
            args.n_boot,
            rng,
        ),
    }

    out = {
        "metadata": {
            **as_metadata_dict(git_provenance()),
            "script": "scripts/issue2476_matched_tier_ci.py",
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "numpy_version": np.__version__,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "inputs": {k: str(TURNAVG_DIR / v) for k, v in INPUTS.items()},
            "estimator_provenance": {
                "_boot_median_ci": "scripts/issue2476_turnavg_sae.py (imported, not reimplemented)",
                "_boot_median_diff": (
                    "scripts/issue1482_matryoshka_tier.py (imported, not reimplemented; "
                    "independent resamples per side — tiers are disjoint feature sets)"
                ),
            },
        },
        "band_definitions": {
            "rule": (
                "arm-specific: each arm's matched band is its OWN tier-2 survivors' activity "
                "envelope; bridge uses the exact envelope, fresh's committed band is the "
                "envelope outward-rounded to grain 200; bounds inclusive, activity = active "
                "fit rows"
            ),
            "fresh": {
                "n_fit_rows": arms["fresh"]["n_fit_rows"],
                "alive_floor": arms["fresh"]["alive_floor"],
                "tier2_survivor_envelope": list(env_fresh),
                "committed_body_band": list(band_fresh_body),
            },
            "bridge": {
                "n_fit_rows": arms["bridge"]["n_fit_rows"],
                "alive_floor": arms["bridge"]["alive_floor"],
                "tier2_survivor_envelope": list(env_bridge),
                "committed_body_band": list(env_bridge),
            },
            "pile": {
                "n_fit_rows": arms["pile"]["n_fit_rows"],
                "alive_floor": arms["pile"]["alive_floor"],
                "tier2_survivor_envelope": list(env_pile),
                "committed_body_band": None,
                "derivation_note": (
                    "no committed pile band existed; derived here as the exact tier-2 "
                    "envelope per the bridge-arm convention"
                ),
            },
        },
        "self_check": self_check,
        "reads": reads,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[matched-tier-ci] wrote {args.out} (B={args.n_boot}, seed={args.seed})")
    for block_name, block in reads.items():
        for cname, c in block["contrasts"].items():
            print(f"  {block_name} {cname}: point={c['point']} ci95={c['ci95']}")


if __name__ == "__main__":
    main()
