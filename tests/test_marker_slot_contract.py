"""CPU-only tests for the marker-slot storage-contract write-time validation (#576).

Pins the runtime enforcement of the four-floats-per-slot storage contract
(.claude/rules/marker-leakage-measurement.md § "Storage contract"; incident
#530: an eval rig persisted only post-softmax log-probs, making the mandated
logit readout unrecoverable and forcing paid GPU re-runs on #530/#531).

Five layers:

1. ``validate_marker_slot_record`` — the pure validator: conforming records
   pass; post-softmax-only records, non-finite values, positive log-probs,
   and identity-breaking field combinations fail loudly.
2. ``compute_marker_slot_stats`` rows pass the validator by construction
   (shape-level check on a synthetic record mirroring its output).
3. ``MarkerBandStopCallback.on_step_end`` — the write-time wiring: a probe
   read that comes back without the pre-softmax fields aborts the step with
   the contract error BEFORE anything is persisted.
4. Round-2 (#576) validator pins: non-numeric rejection classes, the
   z_eos-NaN-under-opt-out branch, the DELIBERATE ``0 < logp <= atol``
   fp-tolerance window, and numpy-scalar behavior.
5. The #472 trajectory rig's FINAL-write gate
   (``assert_trajectory_slot_records_meet_storage_contract``): a
   ``compute_kl=False`` canonical artifact is refused, the explicit opt-out
   works, and crash-recovery partials stay exempt (marked non-contract);
   #629 hardening — present-but-non-finite / bool leaves and empty
   checkpoints are refused with distinct messages, and AST pins fix the
   gate-before-write ordering plus the ``allow_nan=False`` dumps backstop.

No GPU, no HF download; runs in <1s.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from explore_persona_space.eval.callbacks import MarkerBandStopCallback
from explore_persona_space.eval.marker_logprob import (
    MARKER_SLOT_CONTRACT_KEYS,
    validate_marker_slot_record,
)

# ---------------------------------------------------------------------------
# 1. Pure validator
# ---------------------------------------------------------------------------


def _good_record() -> dict:
    # Identity: logp = z_marker - logZ = 5.0 - 17.3 = -12.3
    return {"logp": -12.3, "z_marker": 5.0, "z_eos": 10.0, "logZ": 17.3}


def test_contract_keys_are_the_documented_four():
    assert MARKER_SLOT_CONTRACT_KEYS == ("logp", "z_marker", "z_eos", "logZ")


def test_conforming_record_passes():
    validate_marker_slot_record(_good_record())


def test_extra_keys_are_allowed():
    rec = _good_record()
    rec["argmax_id"] = 83399
    validate_marker_slot_record(rec)


def test_postsoftmax_only_record_fails():
    """The #530 failure mode: only log P(marker) persisted."""
    with pytest.raises(AssertionError, match="missing pre-softmax field"):
        validate_marker_slot_record({"logp": -12.3})


@pytest.mark.parametrize("dropped", ["z_marker", "logZ", "z_eos"])
def test_missing_any_required_field_fails(dropped):
    rec = _good_record()
    del rec[dropped]
    with pytest.raises(AssertionError, match=dropped):
        validate_marker_slot_record(rec)


def test_none_field_fails():
    rec = _good_record()
    rec["z_marker"] = None
    with pytest.raises(AssertionError, match="z_marker"):
        validate_marker_slot_record(rec)


def test_z_eos_optional_when_not_required():
    rec = _good_record()
    rec["z_eos"] = None
    validate_marker_slot_record(rec, require_z_eos=False)
    # ... but the other three stay mandatory even then.
    rec["logZ"] = None
    with pytest.raises(AssertionError, match="logZ"):
        validate_marker_slot_record(rec, require_z_eos=False)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_fails(bad):
    rec = _good_record()
    rec["logZ"] = bad
    with pytest.raises(AssertionError, match="non-finite"):
        validate_marker_slot_record(rec)


def test_positive_logp_fails():
    rec = _good_record()
    rec["logp"] = 0.5
    rec["z_marker"] = rec["logZ"] + 0.5  # keep the identity consistent
    with pytest.raises(AssertionError, match="non-positive"):
        validate_marker_slot_record(rec)


def test_identity_violation_fails():
    """Fields not from the same forward pass (or post-softmax values stuffed
    into the logit fields) break logp == z_marker - logZ."""
    rec = _good_record()
    rec["logp"] = -1.0  # z_marker - logZ is still -12.3
    with pytest.raises(AssertionError, match="softmax identity"):
        validate_marker_slot_record(rec)


def test_mean_aggregated_records_validate():
    """Linearity: per-batch means of (logp, z_marker, logZ) preserve the identity."""
    rows = [
        {"logp": -12.3, "z_marker": 5.0, "z_eos": 10.0, "logZ": 17.3},
        {"logp": -2.1, "z_marker": 11.0, "z_eos": 9.5, "logZ": 13.1},
    ]
    mean_rec = {k: sum(r[k] for r in rows) / len(rows) for k in MARKER_SLOT_CONTRACT_KEYS}
    assert math.isclose(mean_rec["logp"], mean_rec["z_marker"] - mean_rec["logZ"], abs_tol=1e-9)
    validate_marker_slot_record(mean_rec)


def test_not_a_dict_fails():
    with pytest.raises(AssertionError, match="not a dict"):
        validate_marker_slot_record([-12.3, 5.0, 10.0, 17.3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. compute_marker_slot_stats output shape passes by construction
# ---------------------------------------------------------------------------


def test_slot_stats_shaped_row_passes():
    """A row shaped exactly like compute_marker_slot_stats output validates."""
    raw = torch.randn(151_936)  # Qwen-2.5 vocab-sized logit vector
    log_z = float(torch.logsumexp(raw, dim=-1).item())
    z_marker = float(raw[83399].item())
    z_eos = float(raw[151_645].item())
    row = {"logp": z_marker - log_z, "z_marker": z_marker, "z_eos": z_eos, "logZ": log_z}
    validate_marker_slot_record(row, context="synthetic compute_marker_slot_stats row")


# ---------------------------------------------------------------------------
# 3. MarkerBandStopCallback write-time wiring
# ---------------------------------------------------------------------------


def _make_callback(tmp_path, *, eos_token_id=151_645) -> MarkerBandStopCallback:
    probe_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    probe_positions = torch.tensor([2], dtype=torch.long)
    probe_attention = torch.ones_like(probe_input_ids)
    return MarkerBandStopCallback(
        marker_token_ids=[83399],
        probe_input_ids=probe_input_ids,
        probe_marker_positions=probe_positions,
        probe_attention_mask=probe_attention,
        eos_token_id=eos_token_id,
        trajectory_out_path=str(tmp_path / "trajectory.json"),
    )


def _step_state(step: int = 10) -> SimpleNamespace:
    return SimpleNamespace(global_step=step, max_steps=100, epoch=0.1)


def _stats(logp: float, *, z_marker=None, z_eos=None, log_z=None) -> dict:
    """Build a slot-stats dict like _compute_marker_slot_stats returns."""
    t = lambda v: torch.tensor([float(v)]) if v is not None else None  # noqa: E731
    return {"logp": t(logp), "z_marker": t(z_marker), "z_eos": t(z_eos), "logZ": t(log_z)}


def test_callback_probe_with_full_contract_persists(tmp_path, monkeypatch):
    cb = _make_callback(tmp_path)
    good_base = _stats(-19.0, z_marker=-1.7, z_eos=12.0, log_z=17.3)
    good_trained = _stats(-12.3, z_marker=5.0, z_eos=10.0, log_z=17.3)
    monkeypatch.setattr(cb, "_read_slot_stats_with_base", lambda model: good_base)
    monkeypatch.setattr(cb, "_read_slot_stats_trained", lambda model: good_trained)

    cb.on_step_end(SimpleNamespace(), _step_state(), SimpleNamespace(), model=object())

    payload = json.loads((tmp_path / "trajectory.json").read_text())
    assert payload["n_probe_records"] == 1
    rec = payload["records"][0]
    for key in ("z_marker_trained", "z_marker_base", "logZ_trained", "logZ_base"):
        assert rec[key] is not None


def test_callback_probe_missing_logits_fails_before_persist(tmp_path, monkeypatch):
    """A slot read that comes back post-softmax-only aborts the probe step."""
    cb = _make_callback(tmp_path)
    base_postsoftmax_only = _stats(-19.0)  # z_marker / z_eos / logZ all None
    monkeypatch.setattr(cb, "_read_slot_stats_with_base", lambda model: base_postsoftmax_only)
    monkeypatch.setattr(cb, "_read_slot_stats_trained", lambda model: base_postsoftmax_only)

    with pytest.raises(AssertionError, match="storage-contract violation"):
        cb.on_step_end(SimpleNamespace(), _step_state(), SimpleNamespace(), model=object())
    assert not (tmp_path / "trajectory.json").exists(), "nothing may be persisted on violation"


def test_callback_without_eos_id_warns_but_validates_rest(tmp_path, monkeypatch, caplog):
    """eos_token_id=None is the documented opt-out: z_eos may be absent, the
    other three floats stay mandatory, and construction warns loudly."""
    import logging

    with caplog.at_level(logging.WARNING):
        cb = _make_callback(tmp_path, eos_token_id=None)
    assert any("storage contract" in r.message for r in caplog.records)

    no_eos_base = _stats(-19.0, z_marker=-1.7, log_z=17.3)
    no_eos_trained = _stats(-12.3, z_marker=5.0, log_z=17.3)
    monkeypatch.setattr(cb, "_read_slot_stats_with_base", lambda model: no_eos_base)
    monkeypatch.setattr(cb, "_read_slot_stats_trained", lambda model: no_eos_trained)

    cb.on_step_end(SimpleNamespace(), _step_state(), SimpleNamespace(), model=object())
    payload = json.loads((tmp_path / "trajectory.json").read_text())
    assert payload["n_probe_records"] == 1


# ---------------------------------------------------------------------------
# 4. Round-2 (#576) validator pins: rejection classes + deliberate boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, "-12.3"])
def test_non_numeric_field_fails(bad):
    """Bools and numeric strings are rejected loud, not silently coerced."""
    rec = _good_record()
    rec["logp"] = bad
    with pytest.raises(AssertionError, match="non-numeric"):
        validate_marker_slot_record(rec)


def test_z_eos_nan_with_optout_still_fails():
    """require_z_eos=False makes z_eos OPTIONAL, not unchecked: a
    present-but-NaN z_eos is rejected via the present-but-not-finite branch."""
    rec = _good_record()
    rec["z_eos"] = float("nan")
    with pytest.raises(AssertionError, match="z_eos present but not a finite float"):
        validate_marker_slot_record(rec, require_z_eos=False)


def test_small_positive_logp_within_atol_passes():
    """Deliberate fp-tolerance window, pinned on purpose: a fully CONSISTENT
    record with 0 < logp <= atol (default 1e-3) PASSES.

    Reconciler-adjudicated in #576 round 1: a strict ``logp > 0.0`` check
    would false-reject legitimate saturated records (logp ~ 0- with fp noise
    from fused logsumexp kernels — and source saturation is by-design common
    in this project) while buying nothing against consistent fabrication: the
    negative twin {logp: -eps, z_marker: -eps, logZ: 0} passes any strict
    check too, because the identity check structurally cannot catch a
    self-consistent fabrication at any tolerance.
    """
    rec = {"logp": 0.0005, "z_marker": 17.3005, "z_eos": 10.0, "logZ": 17.3}
    validate_marker_slot_record(rec)


def test_numpy_float64_passes_float32_fails():
    """np.float64 IS a Python-float subclass -> accepted; np.float32 is NOT ->
    rejected loud as non-numeric (writers must cast to Python float first)."""
    import numpy as np

    rec64 = {k: np.float64(v) for k, v in _good_record().items()}
    validate_marker_slot_record(rec64)

    rec32 = {k: np.float32(v) for k, v in _good_record().items()}
    with pytest.raises(AssertionError, match="non-numeric"):
        validate_marker_slot_record(rec32)


# ---------------------------------------------------------------------------
# 5. #472 trajectory rig: FINAL-write storage-contract gate (#576 round 2)
# ---------------------------------------------------------------------------


def _trajectory_leaf(*, with_logits: bool) -> dict:
    """Synthetic held_out leaf shaped like run_trajectory_eval's output:
    Phase-A fields always; Phase-B raw-logit fields only when requested."""
    leaf = {
        "g_logp": -12.3,
        "b_logp": -19.0,
        "delta_g": 6.7,
        "argmax_marker": False,
        "n_marker_in_R": 0,
        "r_collapsed": False,
        "kl": None,
    }
    if with_logits:
        leaf.update(
            {
                "kl": 0.4,
                "z_marker_g": 5.0,
                "z_marker_b": -1.7,
                "z_eos_g": 10.0,
                "z_eos_b": 12.0,
                "logZ_g": 17.3,
                "logZ_b": 17.3,
                "logp_hf_g": -12.3,
                "logp_hf_b": -19.0,
            }
        )
    return leaf


def _trajectory_checkpoints(*, with_logits: bool) -> list[dict]:
    return [
        {
            "frac": 1.0,
            "step": 20,
            "adapter_path": "/tmp/adapter",
            "held_out": {"medical_doctor": {"q1": _trajectory_leaf(with_logits=with_logits)}},
        }
    ]


def test_trajectory_final_write_postsoftmax_only_raises(tmp_path):
    """A compute_kl=False-shaped checkpoints list (leaves carrying only
    g_logp/b_logp) is REFUSED at the canonical write — the #530 incident
    class this task exists to prevent."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_trajectory_slot_records_meet_storage_contract,
    )

    with pytest.raises(AssertionError, match="compute_kl=False"):
        assert_trajectory_slot_records_meet_storage_contract(
            _trajectory_checkpoints(with_logits=False),
            out_path=tmp_path / "trajectory.json",
        )


def test_trajectory_final_write_optin_allows_subcontract(tmp_path):
    """allow_subcontract_output=True is the single explicit opt-out."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_trajectory_slot_records_meet_storage_contract,
    )

    assert_trajectory_slot_records_meet_storage_contract(
        _trajectory_checkpoints(with_logits=False),
        out_path=tmp_path / "trajectory.json",
        allow_subcontract_output=True,
    )


def test_trajectory_final_write_full_contract_passes(tmp_path):
    """The production compute_kl=True shape (raw-logit fields present) writes
    with no behavior change."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_trajectory_slot_records_meet_storage_contract,
    )

    assert_trajectory_slot_records_meet_storage_contract(
        _trajectory_checkpoints(with_logits=True),
        out_path=tmp_path / "trajectory.json",
    )


def test_trajectory_partials_marked_noncontract_and_gate_fires_once():
    """Source-level pin (same pattern as test_i584): crash-recovery partials
    are marked non-contract rather than validated, and the gate runs exactly
    once inside run_trajectory_eval — at the final canonical write, never on
    the .partial.json crash-recovery writes. Extended by #629: also pins the
    gate-call-precedes-write statement ordering and the ``allow_nan=False``
    keyword on the canonical dumps."""
    import ast
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parent.parent
        / "src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py"
    )
    source = src_path.read_text()
    assert '"contract": "phase-a-partial-no-logits"' in source
    assert '"contract": "phase-b-partial-logits-in-progress"' in source

    tree = ast.parse(source)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_trajectory_eval"
    )
    gate_calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "assert_trajectory_slot_records_meet_storage_contract"
    ]
    assert len(gate_calls) == 1, "gate must fire exactly once: final write only, not partials"

    write_calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_text"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "out_path"
    ]
    assert len(write_calls) == 1, "exactly one canonical out_path.write_text expected"

    def _stmt_index(call_node) -> int:
        for i, stmt in enumerate(fn.body):
            if any(n is call_node for n in ast.walk(stmt)):
                return i
        raise AssertionError("call not found in run_trajectory_eval body")

    assert _stmt_index(gate_calls[0]) < _stmt_index(write_calls[0]), (
        "storage-contract gate must execute BEFORE the canonical out_path.write_text"
    )

    # #629 backstop pin: the canonical dumps refuses NaN/Inf payloads.
    dumps_call = write_calls[0].args[0]
    assert isinstance(dumps_call, ast.Call), "out_path.write_text(json.dumps(...)) shape expected"
    allow_nan_kw = [k for k in dumps_call.keywords if k.arg == "allow_nan"]
    assert allow_nan_kw and allow_nan_kw[0].value.value is False, (
        "final canonical json.dumps must pass allow_nan=False (#629)"
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_trajectory_final_write_nan_leaf_raises(tmp_path, bad):
    """#629: a present-but-non-finite raw-logit leaf (corrupted forward pass)
    is refused at the canonical write, with a message DISTINCT from the
    Phase-B-skipped one so the operator routes the failure correctly."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_trajectory_slot_records_meet_storage_contract,
    )

    cks = _trajectory_checkpoints(with_logits=True)
    cks[0]["held_out"]["medical_doctor"]["q1"]["z_marker_g"] = bad
    with pytest.raises(AssertionError, match="non-finite"):
        assert_trajectory_slot_records_meet_storage_contract(cks, out_path=tmp_path / "t.json")


def test_trajectory_final_write_bool_leaf_raises(tmp_path):
    """Bools are not floats: True survives isinstance(int) — rejected explicitly."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_trajectory_slot_records_meet_storage_contract,
    )

    cks = _trajectory_checkpoints(with_logits=True)
    cks[0]["held_out"]["medical_doctor"]["q1"]["logZ_g"] = True
    with pytest.raises(AssertionError, match="non-finite/non-float"):
        assert_trajectory_slot_records_meet_storage_contract(cks, out_path=tmp_path / "t.json")


def test_trajectory_gate_empty_checkpoints_raises(tmp_path):
    """#629 minor: a degenerate empty checkpoints list is refused —
    but the explicit opt-out stays fully permissive (checked first)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_trajectory_slot_records_meet_storage_contract,
    )

    with pytest.raises(AssertionError, match="empty checkpoints"):
        assert_trajectory_slot_records_meet_storage_contract([], out_path=tmp_path / "t.json")
    assert_trajectory_slot_records_meet_storage_contract(
        [], out_path=tmp_path / "t.json", allow_subcontract_output=True
    )


def test_trajectory_nan_with_optout_bypasses_gate(tmp_path):
    """Opt-out semantics unchanged by #629: the gate returns before ANY check.
    The AST-pinned ``allow_nan=False`` dumps backstop is the remaining defense
    at the actual write (pinned structurally above, not exercised here)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_trajectory_slot_records_meet_storage_contract,
    )

    cks = _trajectory_checkpoints(with_logits=True)
    cks[0]["held_out"]["medical_doctor"]["q1"]["z_marker_g"] = float("nan")
    assert_trajectory_slot_records_meet_storage_contract(
        cks, out_path=tmp_path / "t.json", allow_subcontract_output=True
    )
