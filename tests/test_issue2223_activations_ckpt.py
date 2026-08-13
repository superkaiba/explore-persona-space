"""Offline tests for the #2223 ``phase_activations`` OOM + checkpoint fixes.

No GPU, no network: the packer / checkpoint / regime helpers are pure, and the
DV-equivalence tests execute the REAL batched engine
(``_teacher_forced_projections`` — real forwards through
``extract_layer_activations``) on a tiny random-init Qwen2 on CPU.

Pins the two 2026-08-13 live-run defects:

- Defect A — fixed ROW-count batching let peak memory scale as rows x max_len
  into the late long-turn units (OOM at unit ~2432/4804). The fix packs by
  TOKEN BUDGET with length sorting; re-packing/sorting must NOT change the DV
  (response / prefix / context / resp_norm — right padding keeps real tokens
  LEFT-anchored, so span indices are batch-order-invariant).
- Defect B — the phase accumulated every unit in memory and wrote ONCE at the
  end, so one OOM lost all 4,804 units. The fix appends per-pack JSONL records
  and resumes behind a regime fingerprint keyed on every output-affecting knob.
"""

from __future__ import annotations

import json
import random
import re

import pytest
import torch

from scripts import issue2203_common as C
from scripts.issue2223_drift import (
    _activations_ckpt_paths,
    _activations_regime,
    _append_read_ckpt,
    _load_read_ckpt,
    _teacher_forced_projections,
    pack_units_by_token_budget,
)

FIELDS = ("response", "prefix", "context", "resp_norm")
PROJ_LAYER = 2
PAD_ID = 0


# ── fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(2223)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _mk_units(n: int, seed: int) -> list[dict]:
    """Synthetic (conv, turn) read units with mixed lengths + optional prefix_end."""
    rng = random.Random(seed)
    units = []
    for j in range(n):
        length = rng.randint(8, 48)
        ctx = rng.randint(4, length - 2)
        units.append(
            {
                "conv": f"c{j % 3}",
                "domain": "coding" if j % 2 else "writing",
                "turn": j // 3 + 1,
                "ids": [rng.randint(1, 100) for _ in range(length)],
                "ctx_len": ctx,
                "resp_len": length - ctx,
                "prefix_end": (rng.randint(2, ctx) if j % 3 else None),
            }
        )
    return units


def _vhat(model) -> torch.Tensor:
    torch.manual_seed(7)
    v = torch.randn(model.config.hidden_size)
    return (v / v.norm()).float()


def _run(model, units, *, budget, skip=None, on_pack=None, log_prefix=None):
    return _teacher_forced_projections(
        model,
        units,
        PROJ_LAYER,
        _vhat(model),
        PAD_ID,
        token_budget=budget,
        skip=skip,
        on_pack=on_pack,
        log_prefix=log_prefix,
    )


def _assert_close(a: dict, b: dict) -> None:
    """Per-unit, per-field DV equality within tolerance (None must match None)."""
    assert set(a) == set(b)
    for key in a:
        for f in FIELDS:
            va, vb = a[key][f], b[key][f]
            if va is None or vb is None:
                assert va is None and vb is None, (key, f, va, vb)
            else:
                assert va == pytest.approx(vb, rel=1e-4, abs=1e-4), (key, f, va, vb)


# ── token-budget packer ─────────────────────────────────────────────────────────


def test_packer_respects_budget_partitions_and_never_empty():
    units = [{"ids": list(range(n))} for n in (120, 60, 50, 30, 20, 10, 5)]
    budget = 100
    packs = pack_units_by_token_budget(units, budget)
    assert all(packs), "no empty packs"
    covered = sorted(i for p in packs for i in p)
    assert covered == list(range(len(units))), "exact partition of the unit set"
    for p in packs:
        max_len = max(len(units[i]["ids"]) for i in p)
        if len(p) > 1:
            assert len(p) * max_len <= budget, (p, max_len)
    # the 120-token unit exceeds the budget and must go ALONE (never dropped,
    # never an empty batch)
    assert [p for p in packs if 0 in p] == [[0]]


def test_packer_singleton_budget_gives_unpadded_singletons():
    units = [{"ids": [1] * n} for n in (5, 9, 3)]
    assert sorted(pack_units_by_token_budget(units, 1)) == [[0], [1], [2]]


def test_packer_budget_below_one_raises():
    with pytest.raises(ValueError, match="token budget"):
        pack_units_by_token_budget([{"ids": [1]}], 0)


def test_packer_empty_units():
    assert pack_units_by_token_budget([], 100) == []


# ── regime fingerprint (resume key) ─────────────────────────────────────────────


def _regime(**over) -> dict:
    base = dict(
        cell="A0__7b",
        arm="A0",
        model_key="7b",
        enable_thinking=False,
        proj_layer=14,
        phase_tag="phaseA",
        smoke=False,
        raw_sha="ab" * 8,
    )
    base.update(over)
    return _activations_regime(**base)


def test_regime_matches_itself_and_rejects_every_output_affecting_key(tmp_path):
    cur = _regime()
    # identical regime -> resume allowed (no raise)
    C.check_regime(_regime(), cur, tmp_path / "r.json")
    for key, val in [
        ("cell", "A1__7b"),
        ("arm", "A1"),
        ("model_key", "32b"),
        ("enable_thinking", True),
        ("proj_layer", 32),
        ("phase_tag", "phaseB"),
        ("smoke", True),
        ("raw_sha", "cd" * 8),
    ]:
        with pytest.raises(ValueError, match="REGIME MISMATCH"):
            C.check_regime(_regime(**{key: val}), cur, tmp_path / "r.json")


def test_ckpt_without_fingerprint_is_refused(tmp_path):
    with pytest.raises(ValueError, match="NO regime fingerprint"):
        C.check_regime(None, _regime(), tmp_path / "x.jsonl")


# ── checkpoint append / load ────────────────────────────────────────────────────


def test_ckpt_paths_shape(tmp_path):
    ckpt, regime = _activations_ckpt_paths(tmp_path, "phaseA", "A0__7b")
    assert ckpt == tmp_path / "activations_ckpt" / "phaseA" / "A0__7b.jsonl"
    assert regime == tmp_path / "activations_ckpt" / "phaseA" / "A0__7b.regime.json"


def test_append_then_load_roundtrip_and_torn_tail(tmp_path):
    p = tmp_path / "cell.jsonl"
    recs = [
        {
            "conv": "c0",
            "domain": "d",
            "turn": 1,
            "response": 0.5,
            "prefix": None,
            "context": 0.1,
            "resp_norm": 1.0,
        },
        {
            "conv": "c0",
            "domain": "d",
            "turn": 2,
            "response": -0.5,
            "prefix": 0.2,
            "context": 0.3,
            "resp_norm": 2.0,
        },
    ]
    _append_read_ckpt(p, recs)
    loaded = _load_read_ckpt(p)
    assert set(loaded) == {("c0", 1), ("c0", 2)}
    assert loaded[("c0", 2)]["prefix"] == 0.2
    # torn FINAL line (crash mid-append residue) is dropped, not fatal
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"conv": "c1", "turn": 3, "resp')
    assert set(_load_read_ckpt(p)) == {("c0", 1), ("c0", 2)}
    # a malformed NON-final line is real corruption -> raises
    bad = tmp_path / "bad.jsonl"
    bad.write_text('garbage\n{"conv": "c0", "domain": "d", "turn": 1}\n')
    with pytest.raises(json.JSONDecodeError):
        _load_read_ckpt(bad)


def test_missing_ckpt_loads_empty(tmp_path):
    assert _load_read_ckpt(tmp_path / "absent.jsonl") == {}


# ── DV equivalence (Defect A must not move the DV) ──────────────────────────────


def test_dv_equivalence_serial_vs_packed_vs_one_pack(tiny_model):
    """Serial oracle (budget=1 -> unpadded singleton forwards) vs two packings.

    Tolerance rel=abs=1e-4: fp32 CPU forwards differ across batch shapes only by
    floating-point accumulation order (<~1e-6 relative), while any packing /
    span-index / mask bug shifts the projections by O(1) — the bar sits >=100x
    above legitimate jitter and >=1000x below bug scale.
    """
    units = _mk_units(n=10, seed=3)
    serial = _run(tiny_model, units, budget=1)  # 1 unit/pack, zero padding
    packed = _run(tiny_model, units, budget=96)  # mixed multi-row packs
    onepack = _run(tiny_model, units, budget=10**9)  # all units, one padded batch
    expected_keys = {(u["conv"], u["turn"]) for u in units}
    assert set(serial) == expected_keys
    _assert_close(serial, packed)
    _assert_close(serial, onepack)
    _assert_close(packed, onepack)


# ── resume predicate (Defect B) ─────────────────────────────────────────────────


def test_resume_skips_completed_units_and_forwards_only_the_rest(tiny_model, tmp_path, monkeypatch):
    units = _mk_units(n=8, seed=11)
    ckpt = tmp_path / "cell.jsonl"
    full = _run(tiny_model, units, budget=64, on_pack=lambda r: _append_read_ckpt(ckpt, r))
    completed = _load_read_ckpt(ckpt)
    assert set(completed) == set(full), "every unit persisted when its pack completed"

    # keep half the checkpoint -> a resume recomputes EXACTLY the missing half,
    # and no forward ever touches a completed unit (count forwarded rows).
    keep = dict(sorted(completed.items())[:4])

    import explore_persona_space.analysis.extraction as ex

    real = ex.extract_layer_activations
    seen_rows: list[int] = []

    def counting(model, input_ids, layers, **kw):
        seen_rows.append(int(input_ids.shape[0]))
        return real(model, input_ids, layers, **kw)

    monkeypatch.setattr(ex, "extract_layer_activations", counting)
    fresh = _run(tiny_model, units, budget=64, skip=set(keep))
    assert set(fresh) == set(completed) - set(keep)
    assert sum(seen_rows) == len(units) - len(keep), "only missing units forwarded"
    _assert_close(fresh, {k: full[k] for k in fresh})

    # everything checkpointed -> zero forwards, empty fresh result
    seen_rows.clear()
    assert _run(tiny_model, units, budget=64, skip=set(completed)) == {}
    assert seen_rows == []


def test_per_pack_progress_line_shape_and_totals(tiny_model, capsys):
    """Keeps the existing '[phase=activations] {cell} read units N/total' shape,
    one line per pack, cumulative count ending at total."""
    units = _mk_units(n=6, seed=5)
    _run(tiny_model, units, budget=64, log_prefix="[phase=activations] A0__7b read units")
    out = capsys.readouterr().out
    counts = [
        int(m.group(1))
        for m in re.finditer(r"\[phase=activations\] A0__7b read units (\d+)/6$", out, re.M)
    ]
    assert len(counts) >= 2, "one progress line per pack (multiple packs at this budget)"
    assert counts == sorted(counts)
    assert counts[-1] == 6
