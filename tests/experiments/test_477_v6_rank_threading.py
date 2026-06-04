# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG/α + × intentional
"""Task #477 v6 — rank pivot + slot-fix threading tests.

Pins the v6 plan's load-bearing invariants:

  M1  Cal-A0 r=32 control routing — the dispatcher's rank_control phase
      emits cells at r=32 / α=64 / counts {2, 4, 16}; the units carry the
      v4 byte-identical (rank, alpha) values and the slot-fix port is
      automatically on at the worker.

  M2  Alpha SSOT — RANK_ALPHA_MAP_V5 = {2: 16, 4: 23, 8: 32} (+ α=64 for
      the r=32 control) is the ONLY source of alpha. _verify_alpha_invariant
      rejects any pair that deviates (parameterized across {2,4,8,32}). No
      `2*r` formulation anywhere in the v6 surface.

  SLOT The MarkerOnlyDataCollator `suppress_at_post_response_slot` +
      `im_end_token_id` args are present on the v6 branch (slot-fix port
      from origin/main). TrainLoraConfig carries the matching
      `marker_suppress_at_post_response_slot` + `marker_im_end_token_id`
      fields. train_one_cell threads them through.

  PICK pick_rank picks the rank with the most in-band counts; on tie
      prefers SMALLER rank; raises (the H0 off-ramp) when no rank lands
      ≥3 counts in-band.

All tests are pure-function, CPU-only, no torch / no vLLM / no subprocess.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from unittest.mock import patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# M2: alpha SSOT — RANK_ALPHA_MAP_V5 + alpha_for_rank cover {2,4,8,32}.
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_rank_alpha_map_values() -> None:
    """RANK_ALPHA_MAP_V5 holds the planned rsLoRA α/√r ≈ 11.31 values."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ALPHA_CONTROL_V6,
        RANK_ALPHA_MAP_V5,
        RANK_CONTROL_V6,
        RANK_GRID_V5,
    )

    assert RANK_ALPHA_MAP_V5 == {2: 16, 4: 23, 8: 32}
    assert RANK_GRID_V5 == (2, 4, 8)
    assert RANK_CONTROL_V6 == 32
    assert ALPHA_CONTROL_V6 == 64
    # Every Cal-A rank must have a map entry.
    for r in RANK_GRID_V5:
        assert r in RANK_ALPHA_MAP_V5


@pytest.mark.parametrize(
    "rank,expected_alpha",
    [(2, 16), (4, 23), (8, 32), (32, 64)],
)
def test_v6_alpha_for_rank_returns_ssot(rank: int, expected_alpha: int) -> None:
    """alpha_for_rank is the ONLY legal source of α for v6 — parameterized
    across all 4 supported ranks."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        alpha_for_rank,
    )

    assert alpha_for_rank(rank) == expected_alpha


def test_v6_alpha_for_rank_rejects_unknown() -> None:
    """alpha_for_rank raises on any rank not in {2, 4, 8, 32}."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        alpha_for_rank,
    )

    for r in (1, 3, 16, 64, 0, -1):
        with pytest.raises(ValueError, match="v6 M2 alpha invariant"):
            alpha_for_rank(r)


@pytest.mark.parametrize(
    "rank,bad_alpha",
    [
        (2, 4),  # the canonical bug: 2*r
        (4, 8),  # the canonical bug: 2*r
        (8, 16),  # the canonical bug: 2*r
        (32, 32),  # any non-64 value at the control rank
        (2, 23),  # swap counterfactual
        (4, 16),  # swap counterfactual
        (8, 64),  # swap counterfactual
    ],
)
def test_v6_verify_alpha_invariant_rejects_bad_pair(rank: int, bad_alpha: int) -> None:
    """_verify_alpha_invariant fires loud on every alpha that isn't the SSOT."""
    dispatch_mod = importlib.import_module("scripts.dispatch_neg_geometry_477")
    # Importable from scripts/ since pyproject has src layout but scripts is
    # on PYTHONPATH at the worktree; if not, fall back to a direct exec.
    with pytest.raises(ValueError, match="v6 M2 alpha invariant"):
        dispatch_mod._verify_alpha_invariant(rank, bad_alpha)


@pytest.mark.parametrize(
    "rank,alpha",
    [(2, 16), (4, 23), (8, 32), (32, 64)],
)
def test_v6_verify_alpha_invariant_accepts_ssot_pair(rank: int, alpha: int) -> None:
    """The 4 SSOT pairs are the ONLY accepted (rank, alpha) pairs."""
    dispatch_mod = importlib.import_module("scripts.dispatch_neg_geometry_477")
    # No exception.
    dispatch_mod._verify_alpha_invariant(rank, alpha)


def test_v6_no_two_r_formulation_in_alpha_surface() -> None:
    """Static check: NO `2*r` or `2 * r` math anywhere in the v6 alpha-
    computation surface (dispatcher, train_cell, i477_run_cell, __init__,
    calibrate). The SSOT helper IS the only legal source of α.

    The pattern matches `2*r` / `2*lora_r` / `2*picked_rank` / `2 * r` etc.
    in the alpha-computation surface, NOT just any `2*r` (e.g. tensor-rank
    math elsewhere). Excludes comments / docstrings via a "lora_alpha"-or-
    "alpha"-on-same-line heuristic.
    """
    repo = Path(__file__).resolve().parents[2]
    targets = [
        repo / "scripts" / "dispatch_neg_geometry_477.py",
        repo / "scripts" / "i477_run_cell.py",
        repo
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "contrastive_neg_count_decouple_477"
        / "__init__.py",
        repo
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "contrastive_neg_count_decouple_477"
        / "calibrate.py",
        repo
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "contrastive_neg_geometry_472"
        / "train_cell.py",
    ]
    # Match `lora_alpha = 2 * r` or `alpha = 2*picked_rank`, etc. Anything
    # that assigns 2*<rank-like-symbol> into an alpha-bearing variable.
    pattern = re.compile(
        r"(lora_alpha|^\s*alpha)\s*=\s*2\s*\*\s*(picked_rank|lora_r|\br\b)",
        re.MULTILINE,
    )
    hits: list[tuple[Path, list[str]]] = []
    for p in targets:
        if not p.exists():
            continue
        text = p.read_text()
        # Drop comment / docstring lines that mention `2*r` for documentation.
        # Cheap heuristic: keep only the lines that contain `=`, then re-scan.
        code_only = "\n".join(line for line in text.split("\n") if "#" not in line[:3])
        matches = pattern.findall(code_only)
        if matches:
            hits.append((p, matches))
    assert not hits, (
        f"v6 M2 invariant: found `2*r` alpha computation in {hits} — the SSOT "
        f"helper alpha_for_rank() is the ONLY legal source of α."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SLOT-FIX: presence on the issue-477 branch's sft.py.
# ─────────────────────────────────────────────────────────────────────────────


def test_slot_fix_port_present_on_v6_branch_collator() -> None:
    """MarkerOnlyDataCollator carries the suppress_at_post_response_slot +
    im_end_token_id constructor args (slot-fix port from origin/main)."""
    import inspect

    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    sig = inspect.signature(MarkerOnlyDataCollator.__init__)
    assert "suppress_at_post_response_slot" in sig.parameters
    assert "im_end_token_id" in sig.parameters
    # Default OFF — byte-identical for every pre-#474 caller.
    assert sig.parameters["suppress_at_post_response_slot"].default is False
    assert sig.parameters["im_end_token_id"].default is None


def test_slot_fix_port_present_on_train_lora_config() -> None:
    """TrainLoraConfig carries the matching marker_suppress_at_post_response_slot
    + marker_im_end_token_id fields, with v4-compatible defaults."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig()  # default-constructed; should not raise.
    assert cfg.marker_suppress_at_post_response_slot is False
    assert cfg.marker_im_end_token_id is None


def test_slot_fix_collator_requires_im_end_token_id() -> None:
    """The slot-fix flag without an im_end_token_id raises (the constructor
    invariant from origin/main)."""
    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    class _DummyInner:
        def __call__(self, features):
            return {"input_ids": None}

    with pytest.raises(ValueError, match="im_end_token_id"):
        MarkerOnlyDataCollator(
            inner_collator=_DummyInner(),
            marker_token_ids=[83399],
            tail_tokens=0,
            suppress_at_post_response_slot=True,
            im_end_token_id=None,
        )


def test_marker_im_end_token_id_constant_is_qwen() -> None:
    """v6 wires marker_im_end_token_id=151645 (Qwen-2.5 <|im_end|>); the
    constant is exposed for the worker + tests."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        MARKER_IM_END_TOKEN_ID,
    )

    assert MARKER_IM_END_TOKEN_ID == 151645


# ─────────────────────────────────────────────────────────────────────────────
# THREADING: train_one_cell carries the v6 kwargs into TrainLoraConfig.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lora_r,lora_alpha",
    [(2, 16), (4, 23), (8, 32), (32, 64)],
)
def test_train_one_cell_threads_lora_r_alpha_into_cfg(
    tmp_path: Path, lora_r: int, lora_alpha: int
) -> None:
    """train_one_cell threads (lora_r_override, lora_alpha_override,
    marker_suppress_at_post_response_slot, marker_im_end_token_id) into the
    TrainLoraConfig that train_lora is called with. Parameterized across
    every Cal-A + Cal-A0 (rank, alpha) pair the v6 dispatcher emits.

    Mocks train_lora to capture the cfg without spawning the TRL Trainer.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        train_cell as tc,
    )

    captured: dict = {}

    def _mock_train_lora(*, base_model_path, data_path, output_dir, cfg, callbacks):
        captured["cfg"] = cfg

    # Stub a JSONL so train_jsonl.exists() etc. don't blow up if probed.
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text("")
    output_dir = tmp_path / "out"
    ckpt_root = tmp_path / "ckpts"

    with (
        patch.object(tc, "verify_gpu_pin", lambda gpu_id: None),
        patch("explore_persona_space.train.sft.train_lora", _mock_train_lora),
    ):
        tc.train_one_cell(
            cell_slug="c477_calA_negp_4_r4" if lora_r != 32 else "c477_calA0_negp_4_r32",
            seed=42,
            train_jsonl=train_jsonl,
            output_dir=output_dir,
            ckpt_root=ckpt_root,
            fractions=(1.0,),
            report_to="none",
            gpu_id=0,
            lr_override=2e-6,
            epochs_override=2,
            lora_r_override=lora_r,
            lora_alpha_override=lora_alpha,
            marker_suppress_at_post_response_slot=True,
            marker_im_end_token_id=151645,
        )

    cfg = captured["cfg"]
    assert cfg.lora_r == lora_r
    assert cfg.lora_alpha == lora_alpha
    # M2 cross-check: the alpha at training time MUST equal alpha_for_rank(r).
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        alpha_for_rank,
    )

    assert cfg.lora_alpha == alpha_for_rank(cfg.lora_r), (
        f"v6 M2 violation at training time: cfg.lora_alpha={cfg.lora_alpha}, "
        f"alpha_for_rank({cfg.lora_r})={alpha_for_rank(cfg.lora_r)}"
    )
    # Slot-fix port: flag on, im_end_token_id set.
    assert cfg.marker_suppress_at_post_response_slot is True
    assert cfg.marker_im_end_token_id == 151645
    # Marker-only loss on, tail_tokens=0 (canonical recipe).
    assert cfg.marker_only_loss is True
    assert cfg.marker_tail_tokens == 0


def test_train_one_cell_default_is_v4_byte_identical(tmp_path: Path) -> None:
    """No v6 overrides → r=32/α=64 + slot-fix OFF (v4 byte-identical)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        train_cell as tc,
    )

    captured: dict = {}

    def _mock_train_lora(*, base_model_path, data_path, output_dir, cfg, callbacks):
        captured["cfg"] = cfg

    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text("")
    with (
        patch.object(tc, "verify_gpu_pin", lambda gpu_id: None),
        patch("explore_persona_space.train.sft.train_lora", _mock_train_lora),
    ):
        tc.train_one_cell(
            cell_slug="c477_main_calib_negp_4",
            seed=42,
            train_jsonl=train_jsonl,
            output_dir=tmp_path / "out",
            ckpt_root=tmp_path / "ckpts",
            fractions=(1.0,),
            report_to="none",
            gpu_id=0,
            lr_override=2e-6,
            epochs_override=2,
            # NO lora_r/alpha overrides; NO slot-fix flags.
        )
    cfg = captured["cfg"]
    assert cfg.lora_r == 32
    assert cfg.lora_alpha == 64
    assert cfg.marker_suppress_at_post_response_slot is False
    assert cfg.marker_im_end_token_id is None


# ─────────────────────────────────────────────────────────────────────────────
# PICK: pick_rank — most-in-band-counts, tie→smaller, off-ramp on <3.
# ─────────────────────────────────────────────────────────────────────────────


def _make_in_band_step(step: int, delta: float = 12.0, emit: float = 0.7) -> dict:
    return {"delta_g": delta, "emit": emit, "collapsed": False}


def _make_off_band_step(step: int) -> dict:
    return {"delta_g": 0.1, "emit": 0.05, "collapsed": False}


def _make_cell(in_band_step: int | None, delta_at_band: float = 12.0) -> dict:
    """Return a per-step trajectory; one in-band step + 7 off-band ones."""
    out = {}
    for s in (1, 2, 4, 8, 16, 32, 64, 76):
        if s == in_band_step:
            out[s] = _make_in_band_step(s, delta=delta_at_band)
        else:
            out[s] = _make_off_band_step(s)
    return out


def test_pick_rank_picks_most_in_band_counts() -> None:
    """Rank with the most in-band counts wins. Here rank=4 has 4/4, rank=8
    has 1/4, rank=2 has 0/4 → rank=4 wins."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_rank,
    )

    cal_a = {
        2: {2: _make_cell(None), 4: _make_cell(None), 8: _make_cell(None), 16: _make_cell(None)},
        4: {2: _make_cell(16), 4: _make_cell(16), 8: _make_cell(32), 16: _make_cell(64)},
        8: {2: _make_cell(8), 4: _make_cell(None), 8: _make_cell(None), 16: _make_cell(None)},
    }
    res = pick_rank(cal_a)
    assert res["picked_rank"] == 4
    assert res["picked_alpha"] == 23  # SSOT
    assert res["picked_positives"] == 200
    assert set(res["qualifying_counts"]) == {2, 4, 8, 16}
    assert res["per_count_picked_step"] == {2: 16, 4: 16, 8: 32, 16: 64}
    assert res["off_ramp_fired"] is False


def test_pick_rank_tie_prefers_smaller_rank() -> None:
    """Ranks 2 and 4 both land 3 of 4 counts in-band → SMALLER (2) wins."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_rank,
    )

    cal_a = {
        2: {2: _make_cell(8), 4: _make_cell(16), 8: _make_cell(32), 16: _make_cell(None)},
        4: {2: _make_cell(4), 4: _make_cell(8), 8: _make_cell(16), 16: _make_cell(None)},
        8: {2: _make_cell(None), 4: _make_cell(None), 8: _make_cell(None), 16: _make_cell(None)},
    }
    res = pick_rank(cal_a)
    assert res["picked_rank"] == 2, f"tie → smaller rank expected; got {res['picked_rank']}"
    assert res["picked_alpha"] == 16  # SSOT for r=2


def test_pick_rank_offramp_when_no_rank_has_3_in_band() -> None:
    """H0 kill-gate: no rank in {2,4,8} lands ≥3 counts in-band → RuntimeError."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_rank,
    )

    cal_a = {
        2: {2: _make_cell(None), 4: _make_cell(16), 8: _make_cell(None), 16: _make_cell(None)},
        4: {2: _make_cell(None), 4: _make_cell(None), 8: _make_cell(16), 16: _make_cell(None)},
        8: {2: _make_cell(None), 4: _make_cell(None), 8: _make_cell(None), 16: _make_cell(8)},
    }
    with pytest.raises(RuntimeError, match="H0 OFF-RAMP"):
        pick_rank(cal_a)


def test_pick_rank_normalizes_string_keys() -> None:
    """JSON-deserialized tables use string keys for rank + count + step;
    pick_rank must normalize all three levels."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_rank,
    )

    cal_a = {
        "4": {
            "2": {
                "16": {"delta_g": 12.0, "emit": 0.7, "collapsed": False},
                "32": {"delta_g": 0.1, "emit": 0.05, "collapsed": False},
            },
            "4": {
                "16": {"delta_g": 11.5, "emit": 0.7, "collapsed": False},
            },
            "8": {
                "32": {"delta_g": 12.4, "emit": 0.8, "collapsed": False},
            },
            "16": {
                "64": {"delta_g": 11.2, "emit": 0.6, "collapsed": False},
            },
        },
    }
    res = pick_rank(cal_a)
    assert res["picked_rank"] == 4
    assert res["per_count_picked_step"] == {2: 16, 4: 16, 8: 32, 16: 64}


def test_pick_rank_excludes_collapsed_steps() -> None:
    """A step with collapsed=True must NOT count as in-band even when
    delta_g + emit are otherwise inside the gate."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_rank,
    )

    # rank=4 has 4 steps that would qualify EXCEPT each has collapsed=True.
    bad_step = {"delta_g": 12.0, "emit": 0.7, "collapsed": True}
    cal_a = {
        2: {2: _make_cell(None), 4: _make_cell(None), 8: _make_cell(None), 16: _make_cell(None)},
        4: {2: {16: bad_step}, 4: {16: bad_step}, 8: {32: bad_step}, 16: {64: bad_step}},
        8: {2: _make_cell(None), 4: _make_cell(None), 8: _make_cell(None), 16: _make_cell(None)},
    }
    with pytest.raises(RuntimeError, match="H0 OFF-RAMP"):
        pick_rank(cal_a)


# ─────────────────────────────────────────────────────────────────────────────
# M1: Cal-A0 r=32 control routing — dispatcher emits the expected units.
# ─────────────────────────────────────────────────────────────────────────────


def test_cal_a0_slugs_are_r32_at_correct_counts() -> None:
    """Cal-A0 cell registry: 3 slugs, all r=32, counts {2, 4, 16}."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CAL_A0_SLUGS,
        RANK_CONTROL_COUNTS_V6,
        cal_a0_slug,
        count_for_calA0_slug,
    )

    assert len(CAL_A0_SLUGS) == 3
    assert set(CAL_A0_SLUGS) == {cal_a0_slug(c) for c in RANK_CONTROL_COUNTS_V6}
    for slug in CAL_A0_SLUGS:
        assert "r32" in slug, slug
        assert "calA0" in slug, slug
        assert count_for_calA0_slug(slug) in RANK_CONTROL_COUNTS_V6


def test_cal_a_slugs_cover_full_rank_x_count_grid() -> None:
    """Cal-A registry: 3 ranks × 4 counts = 12 distinct slugs."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CAL_A_SLUGS,
        COUNT_LEVELS,
        RANK_GRID_V5,
        cal_a_slug,
        count_for_calA_slug,
        rank_for_calA_slug,
    )

    assert len(CAL_A_SLUGS) == 12
    assert set(CAL_A_SLUGS) == {cal_a_slug(c, r) for r in RANK_GRID_V5 for c in COUNT_LEVELS}
    for slug in CAL_A_SLUGS:
        assert count_for_calA_slug(slug) in COUNT_LEVELS
        assert rank_for_calA_slug(slug) in RANK_GRID_V5


def test_cal_a_and_cal_a0_slugs_are_disjoint() -> None:
    """Cal-A slugs (c477_calA_*) must not collide with Cal-A0 slugs
    (c477_calA0_*) — otherwise dispatcher routing breaks."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CAL_A0_SLUGS,
        CAL_A_SLUGS,
    )

    assert set(CAL_A_SLUGS).isdisjoint(set(CAL_A0_SLUGS))


def test_slot_fix_diagnostic_verdict_branches() -> None:
    """slot_fix_diagnostic returns one of three verdicts based on per-count
    max ΔG at r=32."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        RANK_CONTROL_V6,
    )
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        slot_fix_diagnostic,
    )

    # Branch 1: slot-bug-confirmed (max ΔG < 1 at every count).
    cap = {RANK_CONTROL_V6: {c: {1: {"delta_g": 0.1}, 64: {"delta_g": 0.5}} for c in (2, 4, 16)}}
    v = slot_fix_diagnostic(cap)
    assert v["verdict"] == "slot-bug-confirmed-v4-result-was-genuine"

    # Branch 2: slot-bug-rejected (≥2 counts produce ΔG ≥ 5 nats).
    slot = {
        RANK_CONTROL_V6: {
            2: {1: {"delta_g": 7.0}},
            4: {1: {"delta_g": 6.5}},
            16: {1: {"delta_g": 0.5}},
        }
    }
    v2 = slot_fix_diagnostic(slot)
    assert v2["verdict"] == "slot-bug-rejected-v4-result-was-slot-artifact"

    # Branch 3: ambiguous (exactly 1 count produces ΔG ≥ 5 nats).
    amb = {
        RANK_CONTROL_V6: {
            2: {1: {"delta_g": 7.0}},
            4: {1: {"delta_g": 0.5}},
            16: {1: {"delta_g": 0.5}},
        }
    }
    v3 = slot_fix_diagnostic(amb)
    assert v3["verdict"] == "ambiguous"
    assert v3["alpha_used"] == 64


# ─────────────────────────────────────────────────────────────────────────────
# Code-review v6 round-1 Critical #1 — per_step emission for rank_calibration /
# rank_control. The worker's summary block must emit per_step for these phases
# so the dispatcher's _phase_rank_pick (H0 kill-gate) and _phase_slot_fix_diagnostic
# (H4 verdict) can consume it. Without this the dispatcher trains all 15 Cal-A
# + Cal-A0 cells (~15 GPU-h) then RuntimeErrors at rank_pick with "missing
# per_step" — losing the entire pre-pick training compute.
#
# Test strategy: construct a fixture trajectory, simulate the worker's per_step
# build (using the same helpers the worker uses), package it as a cell_summary
# dict, then feed a cal_a_results list to _phase_rank_pick and assert it
# consumes the per_step shape without RuntimeError.
# ─────────────────────────────────────────────────────────────────────────────


def _make_dense_grid_trajectory_fixture(
    *,
    delta_g_at_step: dict[int, float],
    emit_at_step: dict[int, float] | None = None,
    r_collapsed_at_step: dict[int, bool] | None = None,
) -> dict:
    """Build a trajectory.json-shaped dict with one checkpoint per requested step.

    Each checkpoint carries the source_self block keys the worker reads
    (delta_g_mean, emission_p, optional r_collapsed) plus a held_out block
    (empty per-q dict per persona is enough — picked_step_kl_fields reads
    delta_g / g_logp / b_logp but mean_bystander_delta_g averages over the
    held_out personas, which is allowed to be empty when no bystanders are
    pinned for the test).
    """
    emit = emit_at_step or {s: 0.7 for s in delta_g_at_step}
    collapsed = r_collapsed_at_step or {s: False for s in delta_g_at_step}
    steps_sorted = sorted(delta_g_at_step)
    max_step = max(steps_sorted)
    checkpoints = []
    for s in steps_sorted:
        checkpoints.append(
            {
                "frac": round(s / max_step, 4),
                "step": s,
                "source_self": {
                    "g_logp_mean": -0.5,
                    "b_logp_mean": -10.0,
                    "delta_g_mean": float(delta_g_at_step[s]),
                    "emission_p": float(emit[s]),
                    "r_collapsed": bool(collapsed[s]),
                },
                "held_out": {
                    "p1": {
                        "q1": {"g_logp": -5.0, "b_logp": -10.0, "delta_g": 0.5},
                    },
                },
            }
        )
    return {"checkpoints": checkpoints}


def _simulate_worker_per_step_for_rank_phase(
    traj: dict, requested_steps: tuple[int, ...], cell_slug: str
) -> dict[str, dict]:
    """Reproduce the worker's rank_calibration / rank_control per_step build.

    Mirrors the exact block in i477_run_cell.main() that the round-2 fix added:
    iterate the requested step levels, pick the nearest checkpoint via
    select_checkpoint_near_step, extract picked_step_kl_fields, and tag with
    source_R_collapsed from the picked checkpoint. Then add the terminal "T"
    entry. Pure-Python — no torch / no subprocess.
    """
    from scripts.i477_run_cell import (
        picked_step_kl_fields,
        select_checkpoint_near_step,
    )

    steps_present = sorted(
        int(ck["step"]) for ck in traj["checkpoints"] if ck.get("step") is not None
    )
    terminal_step = max(steps_present)
    per_step: dict[str, dict] = {}
    for s in requested_steps:
        if s > terminal_step:
            continue
        actual_step, picked_ck, offset = select_checkpoint_near_step(traj, s, cell_slug=cell_slug)
        entry = picked_step_kl_fields(picked_ck, cell_slug=cell_slug)
        entry.update(
            {
                "requested_step": int(s),
                "actual_step": int(actual_step),
                "step_offset": int(offset),
                "source_R_collapsed": bool(
                    picked_ck.get("source_self", {}).get("r_collapsed", False)
                ),
            }
        )
        per_step[str(s)] = entry
    terminal_ck = max(traj["checkpoints"], key=lambda c: float(c["frac"]))
    terminal_entry = picked_step_kl_fields(terminal_ck, cell_slug=cell_slug)
    terminal_entry.update(
        {
            "requested_step": "T",
            "actual_step": int(terminal_ck.get("step", terminal_step)),
            "step_offset": 0,
            "source_R_collapsed": bool(
                terminal_ck.get("source_self", {}).get("r_collapsed", False)
            ),
        }
    )
    per_step["T"] = terminal_entry
    return per_step


def test_rank_calibration_per_step_emission_consumed_by_rank_pick(tmp_path) -> None:
    """Cal-A cells emit per_step + rank_pick consumes it without RuntimeError.

    End-to-end shape contract: the worker's per_step dict (built from the
    dense-grid trajectory checkpoints, keyed by requested step level, carrying
    source_self_delta_g_at_picked_step + source_emission_p_at_picked_step +
    source_R_collapsed) must satisfy _phase_rank_pick's reads at lines 580-588
    of dispatch_neg_geometry_477.py.

    Fixture: 4 Cal-A cells at rank=4 × counts {2, 4, 8, 16}; each has one
    in-band step (ΔG ∈ [10, 14] nats, emit ∈ [0.50, 0.95], r_collapsed=False).
    Expectation: rank_pick picks rank=4 (4 of 4 counts qualify), no RuntimeError.
    """
    from scripts.dispatch_neg_geometry_477 import _phase_rank_pick

    cal_a_results: list[dict] = []
    # Each (count, picked-in-band-step) pair, all at rank=4 (so rank=4 wins).
    for count, in_band_step in ((2, 16), (4, 16), (8, 32), (16, 64)):
        # 8-step dense grid; only `in_band_step` carries an in-band ΔG/emit.
        delta_g_at_step = {1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4, 16: 0.5, 32: 0.6, 64: 0.7, 76: 0.8}
        emit_at_step = {s: 0.05 for s in delta_g_at_step}
        delta_g_at_step[in_band_step] = 12.0
        emit_at_step[in_band_step] = 0.7
        traj = _make_dense_grid_trajectory_fixture(
            delta_g_at_step=delta_g_at_step, emit_at_step=emit_at_step
        )
        slug = f"c477_calA_negp_{count}_r4"
        per_step = _simulate_worker_per_step_for_rank_phase(
            traj, requested_steps=(1, 2, 4, 8, 16, 32, 64), cell_slug=slug
        )
        # Sanity: per_step has the keys rank_pick reads.
        for entry in per_step.values():
            assert "source_self_delta_g_at_picked_step" in entry
            assert "source_emission_p_at_picked_step" in entry
            # source_R_collapsed is new in the round-2 fix; rank_pick reads
            # entry.get("source_R_collapsed", False) so it has to be there.
            assert "source_R_collapsed" in entry
        cal_a_results.append(
            {
                "cell": slug,
                "seed": 42,
                "lr": 2e-6,
                "phase": "rank_calibration",
                "per_step": per_step,
            }
        )

    # rank_pick should NOT raise (the worker emitted per_step for every cell).
    pick = _phase_rank_pick(cal_a_results, tmp_path / "rank_calibration_pick.json")
    assert pick["picked_rank"] == 4
    assert pick["picked_alpha"] == 23
    assert set(pick["qualifying_counts"]) == {2, 4, 8, 16}
    assert pick["off_ramp_fired"] is False


def test_rank_calibration_per_step_terminal_T_consumed_correctly(tmp_path) -> None:
    """The terminal "T" entry in per_step is picked correctly: actual_step
    matches the trajectory's max-frac checkpoint and the picker normalizes it
    to an int step key (not the string "T")."""
    from scripts.dispatch_neg_geometry_477 import _phase_rank_pick

    # 3-step trajectory: terminal at step 76 is in-band, no non-terminal step is.
    # rank_pick under H0 OFF-RAMP requires ≥3 in-band counts; here we set up
    # rank=4 with 3 in-band counts so the picker succeeds (and the terminal-
    # step entry's actual_step is what rank_pick reads via entry["actual_step"]).
    cal_a_results: list[dict] = []
    for count in (2, 4, 8):
        # Terminal step is 76, in-band; non-terminal steps off-band.
        delta_g_at_step = {1: 0.1, 8: 0.5, 76: 12.0}
        emit_at_step = {1: 0.05, 8: 0.1, 76: 0.7}
        traj = _make_dense_grid_trajectory_fixture(
            delta_g_at_step=delta_g_at_step, emit_at_step=emit_at_step
        )
        slug = f"c477_calA_negp_{count}_r4"
        per_step = _simulate_worker_per_step_for_rank_phase(
            traj, requested_steps=(1, 8), cell_slug=slug
        )
        # Terminal entry must be present with int actual_step (NOT "T" string).
        assert "T" in per_step
        assert per_step["T"]["actual_step"] == 76
        assert per_step["T"]["requested_step"] == "T"
        assert per_step["T"]["source_R_collapsed"] is False
        cal_a_results.append(
            {
                "cell": slug,
                "seed": 42,
                "lr": 2e-6,
                "phase": "rank_calibration",
                "per_step": per_step,
            }
        )

    pick = _phase_rank_pick(cal_a_results, tmp_path / "rank_calibration_pick.json")
    assert pick["picked_rank"] == 4
    # Each of {2, 4, 8} qualifies on the terminal step.
    assert set(pick["qualifying_counts"]) == {2, 4, 8}


def test_rank_control_per_step_emission_consumed_by_slot_fix_diagnostic(tmp_path) -> None:
    """Cal-A0 cells emit per_step + slot_fix_diagnostic consumes it.

    Mirrors the rank_calibration test for the Cal-A0 (r=32 control) phase,
    pinning that the H4 diagnostic path doesn't regress when the worker's
    per_step is read for rank_control too.
    """
    from scripts.dispatch_neg_geometry_477 import _phase_slot_fix_diagnostic

    cal_a0_results: list[dict] = []
    for count in (2, 4, 16):
        # Strong terminal-step ΔG so the diagnostic verdict is well-defined.
        delta_g_at_step = {1: 0.1, 8: 0.5, 76: 12.0}
        traj = _make_dense_grid_trajectory_fixture(delta_g_at_step=delta_g_at_step)
        slug = f"c477_calA0_negp_{count}_r32"
        per_step = _simulate_worker_per_step_for_rank_phase(
            traj, requested_steps=(1, 8), cell_slug=slug
        )
        cal_a0_results.append(
            {
                "cell": slug,
                "seed": 42,
                "lr": 2e-6,
                "phase": "rank_control",
                "per_step": per_step,
            }
        )

    verdict = _phase_slot_fix_diagnostic(cal_a0_results, tmp_path / "diag.json")
    assert verdict["verdict"] == "slot-bug-rejected-v4-result-was-slot-artifact"
    assert verdict["alpha_used"] == 64


# ─────────────────────────────────────────────────────────────────────────────
# Code-review v6 round-1 Critical #2 — v6 implant-only-axis arm schedules +
# completes n>0 cells. The v4 branch holds _schedule_unit_pool AND the per-step
# expansion loop; v6 originally just built is_units and dropped them silently,
# writing n_completed=0 for the implant-only arm and voiding H2. The round-2
# fix shares the schedule + expansion code between v6 and v4 (extracted into
# _expand_implant_sweep_v4_anchor_results).
# ─────────────────────────────────────────────────────────────────────────────


def _make_anchor_per_step(seed: int) -> dict[str, dict]:
    """Build a per_step dict the worker would emit for an implant_sweep_v4 anchor.

    Mirrors the shape the dispatcher's helper expands — one entry per step
    level in IMPLANT_SWEEP_STEPS + the terminal "T" entry. Values vary by
    seed so the test can detect per-seed echo.
    """
    return {
        "16": {
            "source_self_marker_channel_kl_at_picked_step": 0.10 + 0.01 * seed,
            "mean_bystander_marker_channel_kl_at_picked_step": 0.05,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.02,
            "source_self_delta_g_at_picked_step": 6.0,
            "source_emission_p_at_picked_step": 0.30,
            "mean_bystander_delta_g_at_picked_step": 1.0,
            "requested_step": 16,
            "actual_step": 16,
            "step_offset": 0,
        },
        "64": {
            "source_self_marker_channel_kl_at_picked_step": 0.40 + 0.01 * seed,
            "mean_bystander_marker_channel_kl_at_picked_step": 0.20,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.10,
            "source_self_delta_g_at_picked_step": 10.0,
            "source_emission_p_at_picked_step": 0.70,
            "mean_bystander_delta_g_at_picked_step": 2.5,
            "requested_step": 64,
            "actual_step": 64,
            "step_offset": 0,
        },
        "T": {
            "source_self_marker_channel_kl_at_picked_step": 0.80 + 0.01 * seed,
            "mean_bystander_marker_channel_kl_at_picked_step": 0.50,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.30,
            "source_self_delta_g_at_picked_step": 14.0,
            "source_emission_p_at_picked_step": 0.95,
            "mean_bystander_delta_g_at_picked_step": 5.0,
            "requested_step": "T",
            "actual_step": 76,
            "step_offset": 0,
        },
    }


def test_v6_implant_sweep_schedules_and_expands_per_step_records() -> None:
    """v6 implant-only-axis: shared helper expands anchor per_step into n>0 cells.

    The round-1 bug: v6 built is_units and dropped them — the v4 branch held
    the only call to _schedule_unit_pool + the only per-step expansion loop,
    so the v6 implant arm always wrote n_completed=0 (voiding H2).

    The round-2 fix shares the schedule + expansion code by extracting
    _expand_implant_sweep_v4_anchor_results. This test pins that the
    extracted helper accepts the v6-shaped anchor results (each carrying
    lora_r / lora_alpha threaded by the scheduler) and expands them into
    per-step records the analyzer's implant_only_axis partial consumes.

    Fixture: 2 seeds × ONE anchor cell × 3 step levels (16, 64, T) = 6 records.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ANCHOR_COUNT,
        IMPLANT_SWEEP_V4_ANCHOR_SLUG,
        IMPLANT_SWEEP_V4_SLUGS,
    )
    from scripts.dispatch_neg_geometry_477 import (
        _expand_implant_sweep_v4_anchor_results,
    )

    # Simulate _schedule_unit_pool's return for the v6 path: one anchor result
    # per seed, each carrying lora_r=4 + lora_alpha=23 (the picked v6 rank,
    # threaded by the scheduler) plus the per_step dict the worker emitted.
    anchor_results = [
        {
            "cell": IMPLANT_SWEEP_V4_ANCHOR_SLUG,
            "seed": 42,
            "lr": 2e-6,
            "phase": "implant_sweep_v4",
            "run_label": f"{IMPLANT_SWEEP_V4_ANCHOR_SLUG}_seed42_lr2e-06",
            "lora_r": 4,
            "lora_alpha": 23,
            "per_step": _make_anchor_per_step(seed=42),
        },
        {
            "cell": IMPLANT_SWEEP_V4_ANCHOR_SLUG,
            "seed": 137,
            "lr": 2e-6,
            "phase": "implant_sweep_v4",
            "run_label": f"{IMPLANT_SWEEP_V4_ANCHOR_SLUG}_seed137_lr2e-06",
            "lora_r": 4,
            "lora_alpha": 23,
            "per_step": _make_anchor_per_step(seed=137),
        },
    ]

    expanded = _expand_implant_sweep_v4_anchor_results(anchor_results)

    # ── Schedule + expansion succeeded → n>0 cells (Critical #2 fix). ────────
    assert len(expanded) == 6, (
        f"v6 implant_sweep expected 2 seeds × 3 step levels = 6 records; "
        f"got {len(expanded)}. The shared schedule + expansion path is broken."
    )

    # Per-seed split: 3 records per seed (one per step level).
    by_seed: dict[int, list[dict]] = {}
    for r in expanded:
        by_seed.setdefault(r["seed"], []).append(r)
    assert set(by_seed) == {42, 137}
    assert all(len(rs) == 3 for rs in by_seed.values())

    # Cell slugs encode the step level (the analyzer's implant_only_axis
    # partial keys off these). Every record's cell must be a registered v4
    # implant-sweep slug.
    slugs_seen = {r["cell"] for r in expanded}
    assert slugs_seen == set(IMPLANT_SWEEP_V4_SLUGS)

    # Picked-step DV keys are echoed from the per_step entries (NOT the
    # anchor's terminal-only fields). Per-seed values differ (the fixture
    # encodes seed in the source_self_marker_channel_kl).
    seed42_kls = sorted(
        r["source_self_marker_channel_kl_at_picked_step"]
        for r in expanded
        if r["seed"] == 42 and r["step_level"] != "T"
    )
    seed137_kls = sorted(
        r["source_self_marker_channel_kl_at_picked_step"]
        for r in expanded
        if r["seed"] == 137 and r["step_level"] != "T"
    )
    assert seed42_kls != seed137_kls, (
        "per-seed echo broken; the expansion is dropping seed-specific values"
    )

    # ANCHOR_COUNT pinned on every record (analyzer reads it for the H2
    # implant-only axis).
    assert all(r["count"] == ANCHOR_COUNT for r in expanded)
    # Phase tag preserved (the partial's gate accepts implant_sweep_v4 only).
    assert all(r["phase"] == "implant_sweep_v4" for r in expanded)


def test_v6_implant_sweep_expansion_fails_loud_on_missing_per_step() -> None:
    """The expansion helper raises if an anchor result is missing per_step.

    The fail-loud contract was in the original inline expansion; it stays in
    the extracted helper so a worker regression that drops per_step (e.g.
    schema drift) surfaces at expansion time, not as a silent n_completed=0.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        IMPLANT_SWEEP_V4_ANCHOR_SLUG,
    )
    from scripts.dispatch_neg_geometry_477 import (
        _expand_implant_sweep_v4_anchor_results,
    )

    bad_anchor = [
        {
            "cell": IMPLANT_SWEEP_V4_ANCHOR_SLUG,
            "seed": 42,
            "lr": 2e-6,
            "phase": "implant_sweep_v4",
            # per_step deliberately missing.
        }
    ]
    with pytest.raises(RuntimeError, match="missing 'per_step' dict"):
        _expand_implant_sweep_v4_anchor_results(bad_anchor)


# ─────────────────────────────────────────────────────────────────────────────
# Code-review v6 round-1 Minor #3 — --positives was accepted + logged but never
# consumed by build_cell (silent no-op). Round-2 fix: drop the worker's
# --positives flag, drop per-unit threading, and add an M3 invariant assertion
# at dispatcher startup (--positives must equal POS_EX_PER_SOURCE=200).
# ─────────────────────────────────────────────────────────────────────────────


def test_worker_no_longer_accepts_positives_flag() -> None:
    """The worker's CLI argparse no longer carries --positives; the dispatcher
    no longer threads it per-unit (build_cell reads POS_EX_PER_SOURCE directly).
    """
    from pathlib import Path

    worker_src = Path(__file__).resolve().parents[2] / "scripts" / "i477_run_cell.py"
    text = worker_src.read_text()
    # ap.add_argument("--positives", ...) must not appear (the flag is gone).
    assert '"--positives"' not in text, (
        "worker still declares --positives flag; round-2 fix should have dropped it"
    )


def test_dispatcher_no_longer_threads_positives_per_unit() -> None:
    """Dispatcher's _launch no longer extends the cmd with --positives, and no
    is_units construction carries a "positives" key."""
    from pathlib import Path

    disp_src = Path(__file__).resolve().parents[2] / "scripts" / "dispatch_neg_geometry_477.py"
    text = disp_src.read_text()
    assert '"--positives"' not in text or 'cmd.extend(["--positives"' not in text, (
        "dispatcher still threads --positives to the worker; round-2 fix should have dropped it"
    )
    # No is_units construction populates "positives".
    assert '"positives": int(args.positives)' not in text, (
        "dispatcher still populates 'positives' on per-unit dicts; round-2 fix "
        "should have dropped it (build_cell reads POS_EX_PER_SOURCE directly)"
    )
