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
