# ruff: noqa: RUF003  # em-dash + × multiplication sign intentional
"""Task #505 §5.5 gate (g) — the cherry-picked ``assert_adapter_actually_applied``
negative-control unit test.

The plan §5.5 (g) requires a pytest that:

  1. Loads a hand-built trajectory in the silent-LoRA-not-applied regime
     (B-norm > floor + max|ΔG| ≈ 0 across the panel + n_emit = 0) and asserts
     the guard FAILS LOUD (raises ``LoRANotAppliedError``).

  2. Loads a hand-built trajectory in the genuine-floor regime (B-norm ≈ 0,
     ΔG ≈ 0 across the panel) and asserts the guard PASSES (no raise).

  3. Loads a hand-built trajectory in the real-signal regime (B-norm > floor,
     max|ΔG| ≫ eps) and asserts the guard PASSES.

The B-matrix Frobenius norm is read from a synthetic
``adapter_model.safetensors`` we write to a tmp dir; the records dicts are
synthesized in the rig's expected shape so the guard's
``_aggregate_records_for_guard`` can consume them directly.

This file also asserts the cherry-pick landed: the
``contrastive_neg_geometry_472.eval_guard`` module imports cleanly and
exposes ``assert_adapter_actually_applied`` + ``LoRANotAppliedError``. The
plan §10 step 0 makes the cherry-pick mandatory before any sweep.

Runs in <1 s on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# ── Cherry-pick landed import. ──────────────────────────────────────────────


def test_eval_guard_module_importable():
    """Plan §10 step 0: the #477 guard MUST be cherry-picked onto issue-505."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    assert hasattr(eval_guard, "assert_adapter_actually_applied")
    assert hasattr(eval_guard, "LoRANotAppliedError")
    assert hasattr(eval_guard, "b_matrix_frobenius_norm")


# ── Helpers: synthesize a safetensors adapter + records dicts. ──────────────


def _write_adapter(tmp_dir: Path, b_norm: float) -> Path:
    """Write a minimal PEFT-style adapter_model.safetensors with one lora_B tensor.

    The tensor is a vector of length 8 with uniform entries scaled so the
    Frobenius norm equals ``b_norm`` (``sqrt(8) * x = b_norm`` → ``x = b_norm / sqrt(8)``).
    """
    adapter_dir = tmp_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if b_norm <= 0:
        b_tensor = torch.zeros(8, dtype=torch.float32)
    else:
        x = b_norm / (8**0.5)
        b_tensor = torch.full((8,), x, dtype=torch.float32)
    # A dummy A tensor (PEFT also stores lora_A; the guard reads only lora_B).
    a_tensor = torch.randn(8, dtype=torch.float32)
    save_file(
        {
            "base_model.model.layer.0.lora_B.default.weight": b_tensor,
            "base_model.model.layer.0.lora_A.default.weight": a_tensor,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    return adapter_dir


def _records(
    *,
    personas: list[str],
    questions: list[str],
    g_logp: float,
    b_logp: float,
    emit: bool,
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Build a (persona, q) record dict in the rig's expected shape.

    The ``emit`` flag controls the argmax_marker booleans; ``b_logp`` is unused
    in this fixture (the caller passes the base records separately, but we
    accept it for symmetry with the eventual base-records caller).
    """
    del b_logp  # symmetry-only placeholder
    return {p: {q: {"logp": g_logp, "argmax_marker": emit} for q in questions} for p in personas}


# ── (g) — negative-control: SILENT-LORA-NOT-APPLIED regression raises. ──────


def test_guard_raises_on_silent_lora_not_applied(tmp_path):
    """The #477 v4/v6 regression class — adapter genuinely trained + ΔG ≈ 0 +
    emission 0 everywhere — MUST raise ``LoRANotAppliedError`` per the §5.5
    gate (g) negative-control."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = _write_adapter(tmp_path, b_norm=3.0)  # well above floor 1e-3
    personas = ["b1", "b2", "b3"]
    questions = ["q1", "q2"]
    # Same logp in g and b → max|ΔG| = 0; emission false everywhere.
    g = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)
    b = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)

    with pytest.raises(eval_guard.LoRANotAppliedError, match="LoRA-not-applied regression"):
        eval_guard.assert_adapter_actually_applied(
            adapter_dir=adapter_dir,
            g_records=g,
            b_records=b,
            cell_label="test_neg_control",
        )


# ── PASS cases: genuine floor + real signal. ────────────────────────────────


def test_guard_passes_on_genuine_floor(tmp_path):
    """B-norm at/under floor → the adapter is genuinely untrained, ΔG≈0 is a
    real measurement, NOT the regression."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = _write_adapter(tmp_path, b_norm=0.0)  # zero — clean PEFT init
    personas = ["b1"]
    questions = ["q1"]
    g = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)
    b = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)
    diag = eval_guard.assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g,
        b_records=b,
        cell_label="test_floor",
    )
    assert diag["guard_verdict"] == "pass_genuine_floor"


def test_guard_passes_on_real_signal(tmp_path):
    """B-norm > floor AND max|ΔG| > eps → adapter applied, eval reads real signal."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = _write_adapter(tmp_path, b_norm=3.0)
    # Source persona reads +5 nat ΔG — clear signal that the LoRA is applied.
    g_records = {
        "b1": {
            "q1": {"logp": -10.0, "argmax_marker": False},
            "q2": {"logp": -10.5, "argmax_marker": False},
        },
        "src": {
            "q1": {"logp": -10.5, "argmax_marker": True},
            "q2": {"logp": -10.6, "argmax_marker": True},
        },
    }
    b_records = {
        "b1": {
            "q1": {"logp": -15.0, "argmax_marker": False},
            "q2": {"logp": -15.5, "argmax_marker": False},
        },
        "src": {
            "q1": {"logp": -15.5, "argmax_marker": False},
            "q2": {"logp": -15.6, "argmax_marker": False},
        },
    }
    diag = eval_guard.assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g_records,
        b_records=b_records,
        cell_label="test_real_signal",
    )
    assert diag["guard_verdict"] == "pass_real_signal"
    assert diag["max_abs_delta_g_nats"] >= 0.5
    assert diag["n_emit"] >= 1


def test_guard_b_norm_reader_returns_zero_for_no_lora_b(tmp_path):
    """The b_matrix_frobenius_norm reader returns 0.0 if no lora_B keys exist
    (defensive — treats non-LoRA adapters as genuine floor)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = tmp_path / "no_lora_b"
    adapter_dir.mkdir()
    save_file(
        {"base_model.model.layer.0.lora_A.default.weight": torch.randn(4)},
        str(adapter_dir / "adapter_model.safetensors"),
    )
    norm = eval_guard.b_matrix_frobenius_norm(adapter_dir)
    assert norm == 0.0


# ── Wrapper: source-self MUST be merged into guard input ────────────────────


def _trajectory_payload(*, frac, held_out_records, source_records=None, source_name="src"):
    """Build a minimal trajectory.json payload with one checkpoint at ``frac``.

    held_out_records: ``{persona: {q: {"g_logp", "b_logp", "argmax_marker"}}}``
    source_records:   ``{q: {"g_logp", "b_logp", "argmax_marker"}}`` or None.
    """
    held_out = {}
    for persona, per_q in held_out_records.items():
        held_out[persona] = {}
        for q, leaf in per_q.items():
            held_out[persona][q] = {
                "g_logp": float(leaf["g_logp"]),
                "b_logp": float(leaf["b_logp"]),
                "delta_g": float(leaf["g_logp"]) - float(leaf["b_logp"]),
                "argmax_marker": bool(leaf.get("argmax_marker", False)),
                "n_marker_in_R": 0,
                "r_collapsed": False,
                "kl": None,
            }
    ckpt = {
        "frac": frac,
        "step": 100,
        "adapter_path": "/tmp/adapter",
        "held_out": held_out,
        "held_out_collapse_share": 0.0,
        "n_held_out_collapsed": 0,
    }
    if source_records is not None:
        source_probes = {}
        for q, leaf in source_records.items():
            source_probes[q] = {
                "g_logp": float(leaf["g_logp"]),
                "b_logp": float(leaf["b_logp"]),
                "delta_g": float(leaf["g_logp"]) - float(leaf["b_logp"]),
                "argmax_marker": bool(leaf.get("argmax_marker", False)),
                "n_marker_in_R": 0,
                "r_collapsed": False,
            }
        ckpt["source_probes"] = source_probes
        # Mean-pooled source_self alongside — the original block.
        mean_g = sum(float(leaf["g_logp"]) for leaf in source_records.values()) / len(
            source_records
        )
        mean_b = sum(float(leaf["b_logp"]) for leaf in source_records.values()) / len(
            source_records
        )
        ckpt["source_self"] = {
            "g_logp_mean": mean_g,
            "b_logp_mean": mean_b,
            "delta_g_mean": mean_g - mean_b,
            "emission_p": float(
                sum(1 for leaf in source_records.values() if leaf.get("argmax_marker"))
                / len(source_records)
            ),
            "r_collapsed": False,
        }
    return {"checkpoints": [ckpt]}


def test_wrapper_includes_source_self(tmp_path):
    """The wrapper MUST merge source-self per-q records into the guard input.

    Negative-control of the round-2 BLOCKER fix: build a trajectory whose
    HELD-OUT panel sits in the floor regime (max|ΔG| ~ 0.3 nats, n_emit=0 —
    the *success* signature for a clean contrastive run) but whose SOURCE-SELF
    sits at the real-signal regime (|ΔG| ~ 5 nats, emission=True). With the
    fix in place, the wrapper feeds BOTH to the guard and the guard reports
    ``pass_real_signal``; WITHOUT the fix (held-out only), the guard would
    falsely raise ``LoRANotAppliedError`` on this clean cell.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard
    from explore_persona_space.experiments.leave_one_out_505 import eval_trajectory_505

    # Floor-regime held-out (a clean contrastive run's success signature).
    held_out = {
        b: {
            "q1": {"g_logp": -10.0, "b_logp": -10.15, "argmax_marker": False},
            "q2": {"g_logp": -10.0, "b_logp": -10.30, "argmax_marker": False},
        }
        for b in ("b1", "b2", "b3")
    }
    # Real-signal source-self (the implant is working).
    source_records = {
        "q1": {"g_logp": -10.0, "b_logp": -15.5, "argmax_marker": True},
        "q2": {"g_logp": -10.5, "b_logp": -15.0, "argmax_marker": True},
    }
    payload = _trajectory_payload(
        frac=0.50,
        held_out_records=held_out,
        source_records=source_records,
        source_name="src",
    )

    # 1) Extraction returns BOTH held-out + source records.
    g_records, b_records, _ = eval_trajectory_505._extract_records_at_frac(
        payload, frac=0.50, eval_personas=list(held_out.keys()), source="src"
    )
    assert "src" in g_records, "source key MUST be present in g_records when source= is passed"
    assert "src" in b_records, "source key MUST be present in b_records when source= is passed"
    assert set(g_records["src"]) == {"q1", "q2"}
    assert g_records["src"]["q1"]["logp"] == -10.0
    assert b_records["src"]["q1"]["logp"] == -15.5
    assert g_records["src"]["q1"]["argmax_marker"] is True

    # 2) Guard PASSES on the merged panel: source-self ΔG ~ 5 nats triggers
    # the ``pass_real_signal`` branch.
    adapter_dir = _write_adapter(tmp_path, b_norm=3.0)  # well above floor
    diag = eval_guard.assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g_records,
        b_records=b_records,
        cell_label="test_wrapper_includes_source_self",
    )
    assert diag["guard_verdict"] == "pass_real_signal", diag
    assert diag["max_abs_delta_g_nats"] >= 0.5
    assert diag["n_emit"] == 2  # both source-self q rows emit
    # Total probes = 3 held-out × 2 q + 1 source × 2 q = 8.
    assert diag["n_probes"] == 8


def test_wrapper_raises_when_both_held_out_and_source_floor(tmp_path):
    """Inverse: held-out AND source-self both in the floor regime + n_emit=0
    + B-norm > floor → guard RAISES (genuine #477 silent-LoRA-not-applied)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard
    from explore_persona_space.experiments.leave_one_out_505 import eval_trajectory_505

    # Both held-out and source: ΔG ~ 0, no emission anywhere (the regression).
    held_out = {
        b: {
            "q1": {"g_logp": -10.0, "b_logp": -10.0, "argmax_marker": False},
            "q2": {"g_logp": -10.1, "b_logp": -10.1, "argmax_marker": False},
        }
        for b in ("b1", "b2")
    }
    source_records = {
        "q1": {"g_logp": -10.0, "b_logp": -10.0, "argmax_marker": False},
        "q2": {"g_logp": -10.1, "b_logp": -10.1, "argmax_marker": False},
    }
    payload = _trajectory_payload(
        frac=0.50,
        held_out_records=held_out,
        source_records=source_records,
        source_name="src",
    )
    g_records, b_records, _ = eval_trajectory_505._extract_records_at_frac(
        payload, frac=0.50, eval_personas=list(held_out.keys()), source="src"
    )
    assert "src" in g_records
    adapter_dir = _write_adapter(tmp_path, b_norm=3.0)
    with pytest.raises(eval_guard.LoRANotAppliedError, match="LoRA-not-applied regression"):
        eval_guard.assert_adapter_actually_applied(
            adapter_dir=adapter_dir,
            g_records=g_records,
            b_records=b_records,
            cell_label="test_wrapper_both_floor",
        )


def test_wrapper_warns_when_source_probes_missing(tmp_path, caplog):
    """Older (pre-2026-06-05) trajectories without ``source_probes`` get a
    warning logged + the guard runs on held-out only (back-compat path)."""
    import logging

    from explore_persona_space.experiments.leave_one_out_505 import eval_trajectory_505

    held_out = {
        "b1": {
            "q1": {"g_logp": -10.0, "b_logp": -15.5, "argmax_marker": False},
        }
    }
    # source_records=None → no source_probes in the payload.
    payload = _trajectory_payload(
        frac=0.50, held_out_records=held_out, source_records=None, source_name="src"
    )

    with caplog.at_level(logging.WARNING, logger="issue_505.eval_trajectory"):
        g_records, _b_records, _ = eval_trajectory_505._extract_records_at_frac(
            payload, frac=0.50, eval_personas=list(held_out.keys()), source="src"
        )
    assert "src" not in g_records  # no source records merged
    assert any("source_probes" in r.message for r in caplog.records)
