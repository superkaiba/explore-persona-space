#!/usr/bin/env python3
"""SMOKE FIXTURES for ``i585_step6to12_compare_and_figures.py`` — schema-exact.

Sibling of ``i585_make_smoke_fixtures.py`` extended to the step6to12 9-entry
merged index (7 retrained per-step snapshots + 2 Hub endpoint controls). NOT
real data: small held-out panels (6 personas x 3 questions) so the coverage
gate, bootstrap, identity-rate matrix, all six section-6 reads, and the verdict
routing execute end to end. The PARENT-side reference inputs (corrected table +
parent trajectory) are always the REAL committed #585 artifacts; this generator
only fabricates THIS round's trajectory + slot stats + provenance, anchoring
the Hub endpoint values to the real committed constants so the validity kill
behaves as each variant intends.

Six variants, one per verdict branch the smoke must cover:

  * ``coupled``          — sharp co-occurring transition at step 9 with usable
    intermediate anchors clearing the picker's 0.2 gate ->
    ``h1_confirmed_coupled_anchor_clears_picker_gate``.
  * ``decoupled``        — ΔG plateau arrival at step 8, resolution collapse at
    step 11 (gap 3, graded climb between) ->
    ``decoupled_transition_falsification_a``.
  * ``splice-fail``      — retrained step-6 endpoint 4 nats off the same-run
    Hub read -> ``splice_failed_instance_scoped_negative``.
  * ``s-coll-undefined`` — resolution never drops below the 0.05 read-line ->
    ``s_coll_undefined_descriptive``.
  * ``stale-signature``  — 4 of 18 held-out leaves frozen to the SAME float at
    every checkpoint (all 36 pair identity rates = 0.222, the #549 signature)
    -> ``residual_stale_serving_infra``.
  * ``eval-drift``       — same-run Hub step-6 read 3.5 nats off the parent's
    committed value -> ``validity_kill_eval_drift``.

Usage:
    uv run python scripts/i585_make_step6to12_smoke_fixtures.py \
        --out-dir /tmp/i585_s612_smoke/fixtures
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# The 9 merged-index keys (must mirror EXPECTED_INDEX in the analysis script).
INDEX: dict[str, tuple[int, str]] = {
    "0.0733": (6, "retrain"),
    "0.08": (6, "hub"),
    "0.0867": (7, "retrain"),
    "0.1000": (8, "retrain"),
    "0.1133": (9, "retrain"),
    "0.1267": (10, "retrain"),
    "0.1400": (11, "retrain"),
    "0.1533": (12, "retrain"),
    "0.16": (12, "hub"),
}
PERSONAS = [f"fixture_persona_{i}" for i in range(6)]
QS = [f"fixture question {j}?" for j in range(3)]
SLOT_QS = QS + [f"extra q {k}" for k in range(7)]  # 10 companion questions
N_FROZEN_LEAVES = 4  # stale-signature: 4/18 = 0.222 in the [0.10, 0.40] band
CEILING_LOGP = math.log(0.9)

# Per-variant retrained source-ΔG and bystander-resolution curves (steps 6-12).
# Hub endpoint values are anchored to the REAL committed parent constants at
# generation time (ref6 ~5.43, plateau ~10.35) unless the variant says
# otherwise. Resolutions are realized as n_in_band/18 by construction.
VARIANTS: dict[str, dict] = {
    "coupled": {
        # s_ceil = 9 (first dg >= plateau-2 ~ 8.35); s_coll = 9; anchors at
        # steps 7+8 (in band, res 6/18 = 0.33 >= 0.2 picker gate).
        "retr_dg": {6: 5.6, 7: 6.4, 8: 7.3, 9: 9.7, 10: 10.1, 11: 10.3, 12: 10.2},
        "retr_n_in": {6: 6, 7: 6, 8: 5, 9: 0, 10: 0, 11: 0, 12: 0},
        "expect": "h1_confirmed_coupled_anchor_clears_picker_gate",
    },
    "decoupled": {
        # s_ceil = 8 (8.6 >= 8.35); s_coll = 11; gap 3 with graded climb.
        "retr_dg": {6: 5.6, 7: 7.2, 8: 8.6, 9: 9.2, 10: 9.9, 11: 10.2, 12: 10.2},
        "retr_n_in": {6: 6, 7: 6, 8: 6, 9: 5, 10: 4, 11: 0, 12: 0},
        "expect": "decoupled_transition_falsification_a",
    },
    "splice-fail": {
        # retrained step-6 reads 1.5 nats total (|1.5 - 5.43| > 2) — the
        # instance-scoped falsification (b) leg; step-12 endpoint passes.
        "retr_dg": {6: 1.5, 7: 2.0, 8: 4.0, 9: 7.0, 10: 9.0, 11: 10.0, 12: 10.2},
        "retr_n_in": {6: 6, 7: 6, 8: 5, 9: 3, 10: 1, 11: 0, 12: 0},
        "expect": "splice_failed_instance_scoped_negative",
    },
    "s-coll-undefined": {
        # Resolution declines but never crosses 0.05 (min 2/18 = 0.11).
        "retr_dg": {6: 5.6, 7: 6.4, 8: 7.3, 9: 9.7, 10: 10.1, 11: 10.3, 12: 10.2},
        "retr_n_in": {6: 6, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 2},
        "expect": "s_coll_undefined_descriptive",
    },
    "stale-signature": {
        # Frozen-leaf identity pattern dominates; dg values irrelevant.
        "retr_dg": {6: 5.6, 7: 6.4, 8: 7.3, 9: 9.7, 10: 10.1, 11: 10.3, 12: 10.2},
        "retr_n_in": {6: 6, 7: 6, 8: 5, 9: 0, 10: 0, 11: 0, 12: 0},
        "frozen_leaves": N_FROZEN_LEAVES,
        "expect": "residual_stale_serving_infra",
    },
    "eval-drift": {
        "retr_dg": {6: 5.6, 7: 6.4, 8: 7.3, 9: 9.7, 10: 10.1, 11: 10.3, 12: 10.2},
        "retr_n_in": {6: 6, 7: 6, 8: 5, 9: 0, 10: 0, 11: 0, 12: 0},
        "hub6_offset": 3.5,  # |hub6 - parent committed| = 3.5 > 2.0 -> kill
        "expect": "validity_kill_eval_drift",
    },
}

VARIANT_FILES = {
    name: (
        f"trajectory_{name.replace('-', '_')}.json",
        f"slot_{name.replace('-', '_')}.json",
        f"provenance_{name.replace('-', '_')}.json",
    )
    for name in VARIANTS
}


def _leaf(g: float, base: float, pinned: bool) -> dict:
    dg = g - base
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


def build_held_out(ck_idx: int, n_in_band: int, frozen_leaves: int) -> dict:
    """6x3 held-out panel with EXACTLY ``n_in_band`` leaves in the picker band.

    In-band leaf: delta_g >= 0.5 AND g_logp <= log(0.9). Out-of-band leaf:
    delta_g < 0.5 (far from ceiling). Non-frozen leaves carry a deterministic
    per-checkpoint-unique perturbation so cross-checkpoint float identity is 0;
    ``frozen_leaves`` leaves carry the IDENTICAL float at every checkpoint
    (stale-serving signature: same weights -> same logits).
    """
    held_out: dict = {}
    leaf_idx = 0
    for i, p in enumerate(PERSONAS):
        per_q = {}
        for j, q in enumerate(QS):
            base = -14.0 + i * 0.7 + j * 0.3
            in_band = leaf_idx < n_in_band
            dg = (2.0 + 0.05 * leaf_idx) if in_band else (0.1 + 0.01 * leaf_idx)
            frozen = leaf_idx >= len(PERSONAS) * len(QS) - frozen_leaves
            if not frozen:
                dg += 1e-6 * (ck_idx + 1) * (1.0 + 0.001 * leaf_idx)
            per_q[q] = _leaf(base + dg, base, pinned=False)
            leaf_idx += 1
        held_out[p] = per_q
    return held_out


def build_variant(name: str, parent_corrected: dict) -> tuple[dict, dict, dict]:
    """Return (trajectory, slot_stats, provenance) for one variant."""
    spec = VARIANTS[name]
    rng = random.Random(585)
    table = {float(r["ckpt_frac"]): float(r["source_dg"]) for r in parent_corrected["smoke_table"]}
    ref6, plateau = table[0.08], table[0.16]
    hub_dg = {6: ref6 + spec.get("hub6_offset", 0.0), 12: plateau}
    frozen = spec.get("frozen_leaves", 0)

    keys = sorted(INDEX, key=float)
    checkpoints = []
    provenance = {}
    for ck_idx, key in enumerate(keys):
        step, prov = INDEX[key]
        provenance[key] = {"step": step, "provenance": prov}
        dg = spec["retr_dg"][step] if prov == "retrain" else hub_dg[step]
        n_in = (
            spec["retr_n_in"][step]
            if prov == "retrain"
            else (6 if step == 6 else 0)  # hub endpoints mirror parent shape
        )
        g_mean = -0.0153 if dg < plateau - 1.0 else -0.0
        checkpoints.append(
            {
                "frac": float(key),
                "step": step,
                "adapter_path": f"/fixture/{name}/ckpt_{key}",
                "source_self": {
                    "g_logp_mean": g_mean,
                    "b_logp_mean": g_mean - dg,
                    "delta_g_mean": dg,
                    "emission_p": 1.0,
                    "r_collapsed": True,
                },
                "source_manifest_check": None,
                "byte_identical_guard": None,
                "held_out_collapse_share": 0.0 if n_in > 0 else 0.6,
                "n_held_out_collapsed": 0,
                "held_out": build_held_out(ck_idx, n_in, frozen),
            }
        )
    traj = {
        "schema_version": "i472_v1",
        "cell": "c504v4_smoke_eps3_step6to12",
        "seed": 42,
        "source": "villain",
        "kl_computed": True,
        "checkpoints": checkpoints,
        "git_commit": "fixture",
        "timestamp_utc": "fixture",
    }

    # Companion slot stats: 10 questions per checkpoint; Δz keeps climbing past
    # the log-prob plateau (the saturated-end mechanistic read, ~8.25 -> 22.9).
    dz_by_step = {6: 8.25, 7: 10.0, 8: 12.5, 9: 16.0, 10: 19.0, 11: 21.5, 12: 22.9}
    fractions = []
    for key in keys:
        step, prov = INDEX[key]
        dg_mean = spec["retr_dg"][step] if prov == "retrain" else hub_dg[step]
        per_q = {}
        for j, q in enumerate(SLOT_QS):
            dg = dg_mean + rng.gauss(0, 0.3)
            dz = dz_by_step[step] + rng.gauss(0, 0.5)
            per_q[q] = {
                "r_text": f"[fixture R text key={key} q={j}]",
                "kl": abs(dg) * 0.5,
                "z_marker_trained": 12.0 + dz,
                "z_marker_base": 12.0,
                "z_eos_trained": 19.0,
                "z_eos_base": 19.0,
                "logz_trained": 25.0,
                "logz_base": 25.0,
                "logp_marker_trained": -0.05,
                "logp_marker_base": -0.05 - dg,
                "delta_g": dg,
                "delta_z_marker": dz,
                "delta_eos_margin": dz * 0.95,
                "eos_token_id": 151645.0,
            }
        dgs = [v["delta_g"] for v in per_q.values()]
        fractions.append(
            {
                "frac": float(key),
                "adapter_path": f"/fixture/{name}/ckpt_{key}",
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
    return traj, slot, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--parent-corrected",
        type=Path,
        default=Path("eval_results/issue_585/phase0_calibration_v4_corrected.json"),
        help="REAL committed parent table (Hub endpoint anchor values).",
    )
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args(argv)

    parent_corrected = json.loads(args.parent_corrected.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        traj, slot, prov = build_variant(name, parent_corrected)
        traj_name, slot_name, prov_name = VARIANT_FILES[name]
        (args.out_dir / traj_name).write_text(json.dumps(traj))
        (args.out_dir / slot_name).write_text(json.dumps(slot))
        (args.out_dir / prov_name).write_text(json.dumps(prov))
        print(
            f"wrote {name} (expect={VARIANTS[name]['expect']}): "
            f"{traj_name}, {slot_name}, {prov_name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
