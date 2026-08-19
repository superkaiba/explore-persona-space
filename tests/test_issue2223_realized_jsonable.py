"""Regression: caphook realized-edit records checkpoint as JSON-safe (#2223).

Founding crash (the #2223 CPU smoke, genA2a phase): the caphook
(``experiments.issue2203.caphook.AxisCapHook._op_at``) realized-edit records
carry torch Tensors — ``proj_raw_before`` / ``proj_unit_before`` /
``proj_unit_after`` / ``abs_dproj`` / ``fired`` in the prefix-end (delta) branch,
``abs_dproj_sample`` in the all-token branch — alongside the plain scalars the
downstream reduce (``issue2203_phase2._summarize_realized``) consumes. The driver
``json.dumps`` those records into the per-turn checkpoint (``_record_turn``), so a
raw Tensor value raised ``TypeError: Object of type Tensor is not JSON
serializable`` AFTER an entire turn's generation had already completed.

``issue2223_drift._jsonable_realized`` converts every Tensor to ``.item()`` (0-d)
/ ``.tolist()`` (>=1-d) so the record round-trips through JSON while preserving
the per-position H4 distribution as lists and the reduce-only scalars unchanged.
These tests pin that invariant: the raw record is (still) unserialisable, and the
sanitised record round-trips with no tensor-like value surviving.
"""

import json

import pytest

from scripts.issue2223_drift import _jsonable_realized


def _delta_branch_record():
    """A prefix-end (delta) caphook record: 5 Tensor fields + 4 plain scalars."""
    import torch

    return {
        "layer": 14,
        "op": "cap",
        "position_set": "prefix-end",
        "phase": "prefill",
        "n_positions": 3,
        "proj_raw_before": torch.tensor([1.0, 2.0, 3.0]),
        "proj_unit_before": torch.tensor([0.5, 0.25, 0.125]),
        "proj_unit_after": torch.tensor([0.0, 0.25, 0.5]),
        "abs_dproj": torch.tensor([0.5, 0.0, 0.375]),
        "abs_dproj_mean": 0.25,
        "fired": torch.tensor([True, False, True]),
        "fired_frac": 0.5,
    }


def test_raw_record_is_unserialisable_but_sanitised_round_trips():
    """Fails-pre-fix control: the raw Tensor-bearing record is the founding crash;
    the sanitised copy json.dumps cleanly and reloads."""
    rec = _delta_branch_record()
    # Pre-fix behaviour the sanitiser exists to prevent.
    with pytest.raises(TypeError):
        json.dumps(rec)
    out = _jsonable_realized(rec)
    reloaded = json.loads(json.dumps(out))  # must not raise
    assert reloaded["layer"] == 14
    assert reloaded["n_positions"] == 3


def test_no_tensor_value_survives_and_distribution_preserved_as_lists():
    rec = _delta_branch_record()
    out = _jsonable_realized(rec)
    # (a) nothing tensor-like survives (duck-typed, mirrors the sanitiser's own check).
    assert not any(hasattr(v, "detach") for v in out.values())
    # (b) >=1-d Tensors became plain python lists of the right length (H4 distribution kept).
    for k in ("proj_raw_before", "proj_unit_before", "proj_unit_after", "abs_dproj"):
        assert isinstance(out[k], list) and len(out[k]) == 3
        assert all(isinstance(x, float) for x in out[k])
    assert out["fired"] == [True, False, True]  # bool tensor -> list[bool]
    # (c) reduce-only scalars _summarize_realized consumes pass through unchanged + JSON-scalar.
    assert out["abs_dproj_mean"] == 0.25
    assert out["fired_frac"] == 0.5
    assert out["n_positions"] == 3
    for k in ("fired_frac", "n_positions", "abs_dproj_mean"):
        assert not hasattr(out[k], "detach")


def test_all_token_branch_and_zero_dim_scalar():
    import torch

    rec = {
        "phase": "decode",
        "n_positions": 1,
        "abs_dproj_sample": torch.tensor([0.5, 0.625]),  # >=1-d -> list
        "abs_dproj_mean": 0.5,
        "fired_frac": 0.0,
        "zero_dim": torch.tensor(3.5),  # 0-d -> python scalar via .item()
    }
    out = _jsonable_realized(rec)
    json.dumps(out)  # must not raise
    assert out["abs_dproj_sample"] == [0.5, 0.625]
    assert isinstance(out["zero_dim"], float) and out["zero_dim"] == 3.5
