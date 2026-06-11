"""CPU-only tests for the marker-slot storage-contract write-time validation (#576).

Pins the runtime enforcement of the four-floats-per-slot storage contract
(.claude/rules/marker-leakage-measurement.md § "Storage contract"; incident
#530: an eval rig persisted only post-softmax log-probs, making the mandated
logit readout unrecoverable and forcing paid GPU re-runs on #530/#531).

Three layers:

1. ``validate_marker_slot_record`` — the pure validator: conforming records
   pass; post-softmax-only records, non-finite values, positive log-probs,
   and identity-breaking field combinations fail loudly.
2. ``compute_marker_slot_stats`` rows pass the validator by construction
   (shape-level check on a synthetic record mirroring its output).
3. ``MarkerBandStopCallback.on_step_end`` — the write-time wiring: a probe
   read that comes back without the pre-softmax fields aborts the step with
   the contract error BEFORE anything is persisted.

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
