"""Pins for the #1332 ``lowdose-grid-kill-battery`` round (plan v8).

Pure-function smokes for the round's registered predicates:

1. the 4-cell verdict lattice incl. the c* comparator branch (plan v8 §1);
2. the P2 adapter-apply gate predicate (HALT iff dG outside GATE_WINDOW [0.5, 18]; the
   in-loop parity WARN is independent — plan v8 §4 P2);
3. band-stop config construction (exact kwargs vs current ``TrainLoraConfig``
   fields; the deprecated suppress flag is never passed);
4. HF adapter path construction never emits an ``i474`` path;
5. the deterministic bracketing-retrain step selection (overshoot fallback).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1332_lowdose_train as LT  # noqa: E402
from issue1332_lowdose_analysis import c_star, lowdose_verdict_lattice  # noqa: E402
from issue1332_lowdose_gpu import (  # noqa: E402
    p2_gate_verdict,
    slot_identity_deviations,
    split_shards,
)

NAN = float("nan")


# ── (a) verdict lattice ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rho", "band", "ci", "cstar", "expected"),
    [
        # No-low-dose-signal: delta_band <= 0 (raw rho does not clear the band)
        (0.30, 0.35, (0.10, 0.50), 0.35, "No-low-dose-signal"),
        (0.30, 0.30, (0.10, 0.50), 0.35, "No-low-dose-signal"),
        # Replicated: band cleared AND partial CI positive-excludes 0
        (0.60, 0.40, (0.05, 0.55), 0.35, "Replicated-at-low-dose"),
        # Killed: CI wholly below 0
        (0.60, 0.40, (-0.50, -0.05), 0.35, "Killed-at-low-dose"),
        # Killed: straddle with upper bound < c* (actively excludes parent strength)
        (0.60, 0.40, (-0.10, 0.20), 0.35, "Killed-at-low-dose"),
        # Indeterminate: straddle with upper bound >= c* (cannot distinguish
        # zero from parent-strength — underpowered, never a kill)
        (0.60, 0.40, (-0.10, 0.40), 0.35, "Indeterminate-at-low-dose"),
        (0.60, 0.40, (-0.10, 0.35), 0.35, "Indeterminate-at-low-dose"),  # phi == c* boundary
        # ci lower bound exactly 0 does NOT positive-exclude -> straddle branch
        (0.60, 0.40, (0.0, 0.35), 0.35, "Indeterminate-at-low-dose"),
    ],
)
def test_lattice_cells(rho, band, ci, cstar, expected):
    out = lowdose_verdict_lattice(rho, band, ci, cstar)
    assert out["verdict"] == expected


def test_lattice_nan_is_underpowered_never_a_kill():
    for bad in [
        (NAN, 0.4, (0.1, 0.5), 0.35),
        (0.6, NAN, (0.1, 0.5), 0.35),
        (0.6, 0.4, (NAN, 0.5), 0.35),
        (0.6, 0.4, (0.1, NAN), 0.35),
        (0.6, 0.4, (-0.1, 0.2), NAN),
    ]:
        out = lowdose_verdict_lattice(*bad[:2], bad[2], bad[3])
        assert out["verdict"] == "Indeterminate-at-low-dose"
        assert "underpowered" in out["reason"]


def test_c_star_comparator():
    # ceiling == parent ceiling -> c* == parent partial (registered identity)
    assert c_star(0.988, 0.371, 0.988) == pytest.approx(0.371)
    # a less reliable low-dose grid scales the comparator DOWN proportionally
    assert c_star(0.494, 0.371, 0.988) == pytest.approx(0.371 / 2)
    assert math.isnan(c_star(NAN, 0.371, 0.988))


# ── (b) P2 gate predicate ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("dg", "halt"),
    [
        (8.0, False),
        (0.5, False),  # window inclusive: HALT iff dG NOT IN [0.5, 18]
        (1.319, False),  # the realized healthy low-dose diagonal read (A1, att-...211847)
        (18.0, False),
        (0.49, True),  # unapplied/wrong-adapter band (~0; measured noise 0.011)
        (0.0, True),
        (18.01, True),  # parent-ep1-strength band (~24)
        (24.0, True),
        (-3.0, True),
    ],
)
def test_p2_gate_halt_window(dg, halt):
    assert p2_gate_verdict(dg, None)["halt"] is halt


def test_p2_gate_parity_warn_is_independent_of_halt():
    # in-window dG with >1-nat in-loop gap: WARN persisted, NO halt
    v = p2_gate_verdict(10.0, 8.5)
    assert v["halt"] is False and v["parity_warn"] is True
    assert v["parity_gap_nats"] == pytest.approx(1.5)
    # in-window dG with <=1-nat gap: neither
    v = p2_gate_verdict(10.0, 9.5)
    assert v["halt"] is False and v["parity_warn"] is False
    # out-of-window dG with tight agreement: HALT without WARN
    v = p2_gate_verdict(0.3, 0.4)
    assert v["halt"] is True and v["parity_warn"] is False
    # missing in-loop read (no trajectory): no WARN, gap is None
    v = p2_gate_verdict(10.0, None)
    assert v["parity_warn"] is False and v["parity_gap_nats"] is None


# ── (c) band-stop config construction ─────────────────────────────────────────


def test_config_kwargs_exact_recipe_values():
    kw = LT.config_kwargs("A1", trajectory_path="traj.json")
    # byte-identical #474 recipe block (plan v8 §4 P1)
    assert kw["lr"] == 1e-5
    assert kw["lora_r"] == 32
    assert kw["lora_alpha"] == 64
    assert kw["lora_dropout"] == 0.0
    assert kw["batch_size"] == 4
    assert kw["grad_accum"] == 4
    assert kw["max_length"] == 2048
    assert kw["seed"] == 42
    assert kw["marker_only_loss"] is True
    assert kw["marker_tail_tokens"] == 0
    assert kw["marker_im_end_token_id"] == 151645
    assert kw["save_total_limit"] == 1
    # the ONE changed variable: the stop rule (plan v8 §11 items 1-4)
    assert kw["epochs"] == 5
    assert kw["marker_band_stop"] is True
    assert kw["marker_band_low_nats"] == 5.0
    assert kw["marker_band_high_nats"] == 12.0
    assert kw["marker_band_dense_until"] == 200
    assert kw["marker_band_min_steps"] == 5
    assert kw["marker_band_trajectory_path"] == "traj.json"
    # the deprecated no-op flag is NEVER passed (plan v8 §4 P1)
    assert "marker_suppress_at_post_response_slot" not in kw
    # inherited single-GPU CVD pin stays authoritative (no gpu_id kwarg)
    assert "gpu_id" not in kw
    # max_steps threads only when the bracketing retrain asks for it
    assert "max_steps" not in kw
    assert LT.config_kwargs("A1", trajectory_path="t.json", max_steps=17)["max_steps"] == 17


def test_config_kwargs_are_valid_trainloraconfig_fields():
    from dataclasses import fields

    from explore_persona_space.train.sft import TrainLoraConfig

    kw = LT.config_kwargs("A1", trajectory_path="traj.json", max_steps=17)
    names = {f.name for f in fields(TrainLoraConfig)}
    missing = set(kw) - names
    assert not missing, f"driver passes kwargs missing from TrainLoraConfig: {sorted(missing)}"
    # the signature-smoke entrypoint itself runs clean
    LT.verify_config_signature("A1")


# ── (d) adapter path construction ─────────────────────────────────────────────

ALL_CIDS = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "C1",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
]


def test_adapter_paths_never_touch_i474():
    for cid in ALL_CIDS:
        path = LT.hf_adapter_path(cid)
        assert path == f"adapters/i1332_lowdose_{cid}"
        assert "i474" not in path


# ── (e) bracketing-retrain step selection ─────────────────────────────────────


def test_bracket_step_last_in_band():
    ramp = [{"step": s, "delta_nats": 0.7 * s} for s in range(1, 40)]
    pick = LT.select_bracket_step(ramp)
    assert pick["band_miss"] is False
    # last step with 0.7*s in [5, 12] is s=17 (11.9); s=18 -> 12.6 overshoots
    assert pick["max_steps"] == 17
    assert 5.0 <= pick["delta_at_k"] <= 12.0


def test_bracket_step_band_skipped_is_closest_approach():
    skipped = [
        {"step": 1, "delta_nats": 1.0},
        {"step": 2, "delta_nats": 3.5},
        {"step": 3, "delta_nats": 14.5},
        {"step": 4, "delta_nats": 20.0},
    ]
    pick = LT.select_bracket_step(skipped)
    assert pick["band_miss"] is True
    # closest approach to [5, 12]: step 2 at dist 1.5 vs step 3 at dist 2.5
    assert pick["max_steps"] == 2
    assert pick["closest_approach_dist_nats"] == pytest.approx(1.5)


def test_bracket_step_empty_trajectory_fails_loud():
    with pytest.raises(ValueError):
        LT.select_bracket_step([])


# ── shard/slot helpers ────────────────────────────────────────────────────────


def test_split_shards_contiguous_gate_source_in_shard0():
    shards = split_shards(ALL_CIDS, 2)
    assert len(shards) == 2 and shards[0][0] == "A1"
    assert [c for sh in shards for c in sh] == ALL_CIDS
    assert split_shards(ALL_CIDS, 1) == [ALL_CIDS]


def test_lowdose_hub_call_sites_bind_live_signatures():
    """Every upload/verify Hub call site in the NEW lowdose scripts BINDS
    against the helper's live signature (the #1332 r1 sig.bind rule)."""
    import ast
    import inspect

    from issue1332_gpu_phase import upload_files

    from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

    targets = {
        "verify_repo_paths_uploaded": verify_repo_paths_uploaded,
        "upload_files": upload_files,
    }
    n_bound = 0
    for rel in ("scripts/issue1332_lowdose_gpu.py", "scripts/issue1332_lowdose_analysis.py"):
        tree = ast.parse((REPO / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", None)
            )
            if name not in targets:
                continue
            sig = inspect.signature(targets[name])
            sig.bind(
                *[object()] * len(node.args),
                **{k.arg: object() for k in node.keywords if k.arg is not None},
            )
            n_bound += 1
    assert n_bound >= 4, f"expected >=4 Hub call sites bound, got {n_bound}"


def test_slot_identity_deviations():
    jobs = [
        {"slot_kind": "end_of_response", "n_truncated_tokens": 0},
        {"slot_kind": "pre_marker", "n_truncated_tokens": 3},
    ]
    parent_ok = [
        {"slot_kind": "end_of_response", "n_truncated_tokens": 0},
        {"slot_kind": "pre_marker", "n_truncated_tokens": 3},
    ]
    assert slot_identity_deviations(jobs, parent_ok) == []
    parent_bad = [
        {"slot_kind": "end_of_response", "n_truncated_tokens": 0},
        {"slot_kind": "end_of_response", "n_truncated_tokens": 0},
    ]
    devs = slot_identity_deviations(jobs, parent_bad)
    assert len(devs) == 1 and devs[0]["q_idx"] == 1
    with pytest.raises(AssertionError):
        slot_identity_deviations(jobs, parent_ok[:1])
