#!/usr/bin/env python3
"""SMOKE FIXTURES for ``i585_compare_and_figures.py`` — synthetic but schema-exact.

NOT real data: small held-out panels (6 personas x 3 questions) so the
comparison path, bootstrap, heatmap, and verdict routing all execute. The
stale-table input is always the REAL committed #504 artifact; this generator
only fabricates the CORRECTED-side inputs. Four variants, one per verdict
branch the smoke must cover:

  * ``rising``           — rising source curve -> ``h1_confirmed`` (round-1
    fixture, ported verbatim from the round-1 scratch generator).
  * ``flat``             — corrected == the stale values, distinct held-out
    floats per fraction (fix took), diagnostics clean ->
    ``h1_falsified_candidate_flags_549_for_reexamination``.
  * ``stale-signature``  — corrected == the stale values AND 4 of 18 held-out
    leaves frozen to the SAME float at every fraction, so all 15 pair
    identity rates = 4/18 = 0.222 (the #549 flat-at-every-distance
    signature) -> ``residual_stale_serving_infra`` (section 6 gate 2),
    NOT the falsification label.
  * ``emission-climb``   — flat corrected band but the source emission share
    still climbing across fractions ->
    ``flat_band_explained_by_saturation_route_outcome5``.

Usage:
    uv run python scripts/i585_make_smoke_fixtures.py --out-dir /tmp/i585_smoke/fixtures
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

FRACS = [0.08, 0.16, 0.33, 0.50, 0.75, 1.00]
PERSONAS = [f"fixture_persona_{i}" for i in range(6)]
QS = [f"fixture question {j}?" for j in range(3)]
RISING_SRC_DG = {0.08: 5.6, 0.16: 9.1, 0.33: 13.0, 0.50: 17.2, 0.75: 21.5, 1.00: 25.8}
# stale-signature variant: 4 of 18 leaves frozen -> EVERY one of the 15 pair
# identity rates is exactly 4/18 = 0.222, inside the section-6 stale band
# [0.10, 0.40] at every distance (the #549 same-weights-served signature).
N_FROZEN_LEAVES = 4
EMISSION_CLIMB = {0.08: 0.4, 0.16: 0.5, 0.33: 0.7, 0.50: 0.8, 0.75: 0.9, 1.00: 1.0}

VARIANT_FILES = {
    "rising": ("trajectory.json", "phase0_calibration_v4_corrected.json", "source_slot_stats.json"),
    "flat": ("trajectory_flat.json", "corrected_flat.json", "slot_flat.json"),
    "stale-signature": (
        "trajectory_stale_sig.json",
        "corrected_stale_sig.json",
        "slot_stale_sig.json",
    ),
    "emission-climb": (
        "trajectory_emission_climb.json",
        "corrected_emission_climb.json",
        "slot_emission_climb.json",
    ),
}


def rising_leaf(frac: float, p_i: int, q_j: int) -> dict:
    """Round-1 held-out leaf: graded bystander shift, some leaves ceiling-pinned late."""
    base = -14.0 + p_i * 0.7 + q_j * 0.3
    dg = max(0.0, (FRACS.index(frac) + 1) * 1.4 - p_i * 0.9)
    g = base + dg
    pinned = g > math.log(0.9)
    if pinned:
        g = math.log(0.97)
        dg = g - base
    return _leaf_record(g, base, dg, pinned)


def band_leaf(frac_idx: int, p_i: int, q_j: int, frozen: bool) -> dict:
    """Flat-band held-out leaf, far from ceiling.

    ``frozen`` leaves carry the IDENTICAL float at every fraction (stale
    serving: same weights -> same greedy R -> same logits); the rest get a
    deterministic per-fraction-unique perturbation (weight-change + engine
    nondeterminism).
    """
    base = -14.0 + p_i * 0.7 + q_j * 0.3
    dg = 1.2 + 0.01 * (p_i * 3 + q_j)
    if not frozen:
        dg += 1e-6 * (frac_idx + 1) * (1.0 + 0.001 * (p_i * 3 + q_j))
    return _leaf_record(base + dg, base, dg, pinned=False)


def _leaf_record(g: float, base: float, dg: float, pinned: bool) -> dict:
    return {
        "g_logp": g,
        "b_logp": base,
        "delta_g": dg,
        "argmax_marker": pinned,
        "n_marker_in_R": 0,
        "r_collapsed": False,
        "kl": abs(dg) * 0.4,
        "z_marker_trained": 10.0 + dg,
        "z_marker_base": 10.0,
        "z_eos_trained": 18.0,
        "z_eos_base": 18.0,
        "logz_trained": 24.0,
        "logz_base": 24.0,
        "logp_marker_hf_trained": g + 0.05,
        "logp_marker_hf_base": base + 0.05,
        "delta_z_marker": dg * 0.9,
        "delta_z_margin": dg * 0.9,
        "eos_token_id": 151645.0,
    }


def picker_resolution(held_out: dict) -> tuple[float, int, int]:
    """The v4 picker's in-band formula (same as the compare script)."""
    n_in, n_tot = 0, 0
    for per_q in held_out.values():
        for lf in per_q.values():
            n_tot += 1
            if lf["delta_g"] >= 0.5 and lf["g_logp"] <= math.log(0.9):
                n_in += 1
    return (n_in / n_tot if n_tot else 0.0), n_in, n_tot


def build_variant(variant: str, stale: dict) -> tuple[dict, dict, dict]:
    """Return (trajectory, corrected_table, slot_stats) for one variant."""
    rng = random.Random(585)
    stale_by_frac = {float(r["ckpt_frac"]): float(r["source_dg"]) for r in stale["smoke_table"]}
    rising = variant == "rising"
    src_dg = RISING_SRC_DG if rising else stale_by_frac
    emission = EMISSION_CLIMB if variant == "emission-climb" else dict.fromkeys(FRACS, 1.0)

    checkpoints = []
    for fi, frac in enumerate(FRACS):
        held_out = {}
        for i, p in enumerate(PERSONAS):
            per_q = {}
            for j, q in enumerate(QS):
                if rising:
                    per_q[q] = rising_leaf(frac, i, j)
                else:
                    frozen = variant == "stale-signature" and (i * 3 + j) < N_FROZEN_LEAVES
                    per_q[q] = band_leaf(fi, i, j, frozen)
            held_out[p] = per_q
        # rising = round-1 construction (near ceiling); band variants sit far
        # from the ceiling (distance 14 nats) so ceiling_pinned stays False.
        b_logp_mean = -src_dg[frac] - 0.02 if rising else -src_dg[frac] - 14.0
        checkpoints.append(
            {
                "frac": frac,
                "step": None,
                "adapter_path": f"/fixture/ckpt_frac{frac:.2f}",
                "source_self": {
                    "g_logp_mean": -0.02 if rising else -14.02,
                    "b_logp_mean": b_logp_mean,
                    "delta_g_mean": src_dg[frac],
                    "emission_p": emission[frac],
                    "r_collapsed": rising and frac >= 0.75,
                },
                "source_manifest_check": None,
                "held_out_collapse_share": 0.1 if rising and frac >= 0.75 else 0.0,
                "n_held_out_collapsed": 0,
                "held_out": held_out,
            }
        )
    traj = {
        "schema_version": "i472_v1",
        "cell": "c504v4_smoke_eps3_reread",
        "seed": 42,
        "source": "villain",
        "kl_computed": True,
        "checkpoints": checkpoints,
        "git_commit": "fixture",
        "timestamp_utc": "fixture",
    }

    # Self-check the float-identity pattern the variant is supposed to exhibit.
    rates = []
    for fa, fb in itertools.combinations(range(len(FRACS)), 2):
        n_same = sum(
            1
            for p in PERSONAS
            for q in QS
            if checkpoints[fa]["held_out"][p][q]["g_logp"]
            == checkpoints[fb]["held_out"][p][q]["g_logp"]
        )
        rates.append(n_same / (len(PERSONAS) * len(QS)))
    if variant == "stale-signature":
        assert all(0.10 <= r <= 0.40 for r in rates), f"stale-signature rates off-band: {rates}"
    elif variant == "flat":
        assert all(r == 0.0 for r in rates), f"flat variant must have distinct floats: {rates}"

    # Corrected table: bystander_resolution computed with the SAME picker formula.
    rows = []
    for ck in checkpoints:
        res, n_in, n_tot = picker_resolution(ck["held_out"])
        rows.append(
            {
                "epochs": 3,
                "ckpt_frac": ck["frac"],
                "source_dg": ck["source_self"]["delta_g_mean"],
                "source_emission": ck["source_self"]["emission_p"],
                "bystander_resolution": res,
                "n_in_band": n_in,
                "n_total": n_tot,
                "in_band": res >= 0.2,
            }
        )
    corrected = dict(stale)
    corrected["smoke_table"] = rows
    corrected["chosen_checkpoint_fraction"] = 0.16
    corrected["verdict"] = "pass"

    # Source slot-stats companion fixture (glue pass): margins rise with the
    # rising curve; flat (sd 0.05 around 6.0) for every band variant.
    fractions = []
    for frac in FRACS:
        per_q = {}
        for j, q in enumerate(QS + [f"extra q {k}" for k in range(7)]):
            dg = src_dg[frac] + rng.gauss(0, 0.6 if rising else 0.3)
            margin = dg * 1.1 if rising else 6.0 + rng.gauss(0, 0.05)
            per_q[q] = {
                "r_text": f"[fixture R text frac={frac} q={j}]",
                "kl": dg * 0.5,
                "z_marker_trained": 12.0 + dg,
                "z_marker_base": 12.0,
                "z_eos_trained": 19.0,
                "z_eos_base": 19.0,
                "logz_trained": 25.0,
                "logz_base": 25.0,
                "logp_marker_trained": -0.05,
                "logp_marker_base": -0.05 - dg,
                "delta_g": dg,
                "delta_z_marker": dg * 1.1 if rising else margin,
                "delta_eos_margin": margin,
                "eos_token_id": 151645.0,
            }
        dgs = [v["delta_g"] for v in per_q.values()]
        fractions.append(
            {
                "frac": frac,
                "adapter_path": f"/fixture/ckpt_frac{frac:.2f}",
                "delta_g_mean": sum(dgs) / len(dgs),
                "per_question": per_q,
            }
        )
    slot = {
        "schema_version": "i585_v1",
        "task": 585,
        "source": "villain",
        "marker_token_id": 83399,
        "fractions": fractions,
        "git_commit": "fixture",
        "timestamp_utc": "fixture",
    }
    return traj, corrected, slot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--stale", type=Path, default=Path("eval_results/issue_504/phase0_calibration_v4.json")
    )
    ap.add_argument("--variants", nargs="*", default=list(VARIANT_FILES))
    args = ap.parse_args(argv)

    stale = json.loads(args.stale.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        traj, corrected, slot = build_variant(variant, stale)
        traj_name, corrected_name, slot_name = VARIANT_FILES[variant]
        (args.out_dir / traj_name).write_text(json.dumps(traj))
        (args.out_dir / corrected_name).write_text(json.dumps(corrected))
        (args.out_dir / slot_name).write_text(json.dumps(slot))
        print(f"wrote {variant}: {traj_name}, {corrected_name}, {slot_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
