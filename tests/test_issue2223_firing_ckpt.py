"""Offline tests for the #2223 ``phase_firing`` OOM + checkpoint fixes.

The ``_band_fire_fraction`` sibling of the ``phase_activations`` Defect-A/B
fixes (tests: ``test_issue2223_activations_ckpt.py``): firing consumes the SAME
late long-turn ``_collect_read_units`` whose growing padded length overflowed
the working set in activations, so the identical fixed-row-count batching +
full-tensor fp32 upcast had to move to token-budget packs + slice-then-upcast,
and the read (~4.8k A0 units per 7B cap arm, over the ~50-unit intra-phase
floor) had to checkpoint per unit behind a regime-fingerprinted resume.

No GPU, no network: the regime / checkpoint helpers are pure, and the
DV-equivalence tests execute the REAL batched engine (``_band_fire_fraction`` —
real forwards through ``extract_layer_activations``) on a tiny random-init
Qwen2 on CPU, against an INDEPENDENT single-row unpadded reference.
"""

from __future__ import annotations

import json
import random
import re
from types import SimpleNamespace

import pytest
import torch

from scripts import issue2203_common as C
from scripts.issue2223_drift import (
    _append_read_ckpt,
    _band_fire_fraction,
    _checkpointed_fire_fraction,
    _fire_fraction_from_records,
    _firing_axis_sha,
    _firing_ckpt_paths,
    _firing_regime,
    _load_read_ckpt,
)

BAND = [1, 2]
TOK = SimpleNamespace(pad_token_id=0)  # duck-typed: _band_fire_fraction reads pad_token_id only


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
    """Synthetic (conv, turn) read units with mixed lengths (same shape as
    ``_collect_read_units`` output; ``prefix_end`` unused by the firing read)."""
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
                "prefix_end": None,
            }
        )
    return units


def _axes(model) -> dict[int, torch.Tensor]:
    gen = torch.Generator().manual_seed(41)
    out = {}
    for li in BAND:
        v = torch.randn(model.config.hidden_size, generator=gen)
        out[li] = (v / v.norm()).float()
    return out


def _run(model, units, axes, tau, *, position, direction, budget, **kw):
    return _band_fire_fraction(
        model,
        TOK,
        units,
        BAND,
        axes,
        tau,
        position=position,
        direction=direction,
        token_budget=budget,
        **kw,
    )


def _reference_records(model, units, axes, tau, *, position, direction):
    """Independent per-unit oracle: single-row UNPADDED forwards, whole-row
    projection FIRST then position slicing (the production code slices first —
    a genuinely different index path over the same values)."""
    from explore_persona_space.analysis.extraction import extract_layer_activations

    recs = {}
    for u in units:
        cap = extract_layer_activations(model, torch.tensor([u["ids"]]), BAND)
        fires = total = 0
        psum = 0.0
        for li in BAND:
            projs = cap[li][0].float() @ axes[li]
            if position == "context-end":
                projs = projs[u["ctx_len"] - 1 : u["ctx_len"]]
            elif position == "all-prompt":
                projs = projs[: u["ctx_len"]]
            hit = (projs < tau[li]) if direction == "below" else (projs > tau[li])
            fires += int(hit.sum().item())
            total += int(projs.numel())
            psum += float(projs.double().sum().item())
        recs[(u["conv"], u["turn"])] = {
            "fires": fires,
            "total": total,
            "proj_mean": psum / total,
        }
    return recs


def _tau_with_margin(model, units, axes) -> dict[int, float]:
    """Per-layer τ at the LARGEST gap of the middle 10-90% of the projection
    order statistics: mixed fires by construction, and every projection sits at
    a verified margin from τ. The margin's job is COUNT stability: it must
    dominate the fp32 batch-shape jitter (~1e-6 relative on this tiny CPU
    model), so the 1e-4 bar leaves >=100x headroom; the proj_mean float leg
    carries its own tolerance."""
    from explore_persona_space.analysis.extraction import extract_layer_activations

    per_layer: dict[int, list[float]] = {li: [] for li in BAND}
    for u in units:
        cap = extract_layer_activations(model, torch.tensor([u["ids"]]), BAND)
        for li in BAND:
            per_layer[li].extend((cap[li][0].float() @ axes[li]).tolist())
    tau = {}
    for li in BAND:
        vals = sorted(per_layer[li])
        lo, hi = len(vals) // 10, (9 * len(vals)) // 10
        k = max(range(lo, hi), key=lambda i: vals[i + 1] - vals[i])
        tau[li] = (vals[k] + vals[k + 1]) / 2.0
        margin = (vals[k + 1] - vals[k]) / 2.0
        assert margin > 1e-4, (li, margin)  # >=100x the ~1e-6 batch-shape jitter
    return tau


# ── regime fingerprint (resume key) ─────────────────────────────────────────────


def _regime(**over) -> dict:
    base = dict(
        cell="A2a__7b",
        arm="A2a",
        model_key="7b",
        read="expected",
        band=[18, 19],
        position="context-end",
        direction="below",
        tau_by_layer={18: 0.125, 19: -0.5},
        axis_sha="ab" * 8,
        smoke=False,
        raw_sha="cd" * 8,
    )
    base.update(over)
    return _firing_regime(**base)


def test_firing_regime_matches_itself_and_rejects_every_output_affecting_key(tmp_path):
    cur = _regime()
    # identical regime, THROUGH the JSON sidecar roundtrip the real resume path
    # takes (band list + tau dict must survive serialization) -> no raise
    C.check_regime(json.loads(json.dumps(_regime())), cur, tmp_path / "r.json")
    for over in [
        {"cell": "A2b__7b"},
        {"arm": "A2b"},
        {"model_key": "32b"},
        {"read": "realized"},
        # a band change carries its τ (the builder fail-louds on a τ-less layer)
        {"band": [18, 19, 20], "tau_by_layer": {18: 0.125, 19: -0.5, 20: 0.0}},
        {"position": "all-prompt"},
        {"direction": "above"},
        {"tau_by_layer": {18: 0.125, 19: -0.25}},
        {"axis_sha": "ef" * 8},
        {"smoke": True},
        {"raw_sha": "01" * 8},
    ]:
        with pytest.raises(ValueError, match="REGIME MISMATCH"):
            C.check_regime(json.loads(json.dumps(_regime(**over))), cur, tmp_path / "r.json")


def test_axis_sha_is_content_keyed():
    gen = torch.Generator().manual_seed(1)
    axes = {li: torch.randn(8, generator=gen) for li in BAND}
    sha = _firing_axis_sha(BAND, axes)
    assert sha == _firing_axis_sha(BAND, {li: v.clone() for li, v in axes.items()})
    bumped = {li: v.clone() for li, v in axes.items()}
    bumped[BAND[0]][0] += 1e-3
    assert sha != _firing_axis_sha(BAND, bumped)


def test_firing_ckpt_paths_shape(tmp_path):
    ckpt, regime = _firing_ckpt_paths(tmp_path, "A2a__7b", "expected")
    assert ckpt == tmp_path / "firing_ckpt" / "A2a__7b__expected.jsonl"
    assert regime == tmp_path / "firing_ckpt" / "A2a__7b__expected.regime.json"


def test_fire_fraction_from_records_empty_and_sum():
    assert _fire_fraction_from_records({}) == 0.0
    recs = {
        ("c0", 1): {"fires": 3, "total": 10},
        ("c0", 2): {"fires": 1, "total": 30},
    }
    assert _fire_fraction_from_records(recs) == pytest.approx(4 / 40)


# ── DV equivalence (Defect A must not move the DV) ──────────────────────────────


@pytest.mark.parametrize(
    ("position", "direction"),
    [("context-end", "below"), ("all-prompt", "below"), ("all", "above")],
)
def test_dv_equivalence_serial_vs_packed_vs_one_pack(tiny_model, position, direction):
    """Serial oracle (budget=1 -> unpadded singleton forwards) vs two packings
    vs the independent reference, per unit.

    ``fires``/``total`` are asserted EXACTLY equal (integer counts; τ is chosen
    with a verified >=5e-3 margin to every projection, so fp32 batch-shape
    jitter — <~1e-6 relative — cannot flip a count). ``proj_mean`` carries the
    float leg at rel=abs=1e-4: >=100x above legitimate accumulation-order
    jitter and >=1000x below the O(1) shift a packing / span-index / mask bug
    produces. The (position, direction) grid covers the three production reads
    (caphook context-end/below, v4 A2c all-prompt/below, paper A1 all/above).
    """
    units = _mk_units(n=10, seed=3)
    axes = _axes(tiny_model)
    tau = _tau_with_margin(tiny_model, units, axes)
    ref = _reference_records(tiny_model, units, axes, tau, position=position, direction=direction)
    runs = {
        "serial": _run(
            tiny_model, units, axes, tau, position=position, direction=direction, budget=1
        ),
        "packed": _run(
            tiny_model, units, axes, tau, position=position, direction=direction, budget=96
        ),
        "onepack": _run(
            tiny_model, units, axes, tau, position=position, direction=direction, budget=10**9
        ),
    }
    assert 0 < sum(r["fires"] for r in ref.values()) < sum(r["total"] for r in ref.values()), (
        "degenerate τ: all-fire / no-fire cannot catch index bugs via counts"
    )
    for name, got in runs.items():
        assert set(got) == set(ref), name
        for key in ref:
            assert got[key]["fires"] == ref[key]["fires"], (name, key)
            assert got[key]["total"] == ref[key]["total"], (name, key)
            assert got[key]["proj_mean"] == pytest.approx(
                ref[key]["proj_mean"], rel=1e-4, abs=1e-4
            ), (name, key)
        assert _fire_fraction_from_records(got) == _fire_fraction_from_records(ref), name


# ── resume predicate (Defect B) ─────────────────────────────────────────────────


def test_resume_skips_completed_units_and_forwards_only_the_rest(tiny_model, tmp_path, monkeypatch):
    units = _mk_units(n=8, seed=11)
    axes = _axes(tiny_model)
    tau = _tau_with_margin(tiny_model, units, axes)
    ckpt = tmp_path / "cell.jsonl"
    full = _run(
        tiny_model,
        units,
        axes,
        tau,
        position="all",
        direction="above",
        budget=64,
        on_pack=lambda r: _append_read_ckpt(ckpt, r),
    )
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
    fresh = _run(
        tiny_model, units, axes, tau, position="all", direction="above", budget=64, skip=set(keep)
    )
    assert set(fresh) == set(completed) - set(keep)
    assert sum(seen_rows) == len(units) - len(keep), "only missing units forwarded"
    for k in fresh:
        assert (fresh[k]["fires"], fresh[k]["total"]) == (full[k]["fires"], full[k]["total"]), k
    # integer counts make the resumed DV EXACTLY the uninterrupted one
    assert _fire_fraction_from_records({**keep, **fresh}) == _fire_fraction_from_records(full)

    # everything checkpointed -> zero forwards, empty fresh result
    seen_rows.clear()
    assert (
        _run(
            tiny_model,
            units,
            axes,
            tau,
            position="all",
            direction="above",
            budget=64,
            skip=set(completed),
        )
        == {}
    )
    assert seen_rows == []


def test_checkpointed_fire_fraction_end_to_end_resume_and_regime(tiny_model, tmp_path, monkeypatch):
    """The phase_firing wrapper: first call computes + checkpoints; a same-regime
    re-run forwards NOTHING and returns the identical fraction; a changed raw
    sha (re-generated completions) refuses the stale checkpoint loud."""
    units = _mk_units(n=6, seed=7)
    axes = _axes(tiny_model)
    tau = _tau_with_margin(tiny_model, units, axes)

    def call(raw_sha="aa" * 8):
        return _checkpointed_fire_fraction(
            tiny_model,
            TOK,
            units,
            BAND,
            axes,
            tau,
            position="context-end",
            direction="below",
            out_dir=tmp_path,
            cell="A2a__tiny",
            arm="A2a",
            model_key="tiny",
            read="expected",
            smoke=False,
            raw_sha=raw_sha,
        )

    frac = call()
    ckpt, regime = _firing_ckpt_paths(tmp_path, "A2a__tiny", "expected")
    assert ckpt.exists() and regime.exists()
    assert set(_load_read_ckpt(ckpt)) == {(u["conv"], u["turn"]) for u in units}

    import explore_persona_space.analysis.extraction as ex

    def no_forward(*a, **kw):  # pragma: no cover - failure branch
        pytest.fail("forwarded a unit despite a complete same-regime checkpoint")

    monkeypatch.setattr(ex, "extract_layer_activations", no_forward)
    assert call() == frac

    with pytest.raises(ValueError, match="REGIME MISMATCH"):
        call(raw_sha="bb" * 8)


def test_wrapper_fails_loud_when_a_unit_was_never_read(tiny_model, tmp_path, monkeypatch):
    """A unit absent from ckpt+fresh records is a hard KeyError, never a silent
    denominator shrink. (Stubs `_band_fire_fraction` via autospec; the real body
    is executed by the equivalence + resume tests above.)"""
    from unittest.mock import create_autospec

    import scripts.issue2223_drift as D

    units = _mk_units(n=3, seed=13)
    axes = _axes(tiny_model)
    tau = {li: 0.0 for li in BAND}
    monkeypatch.setattr(
        D, "_band_fire_fraction", create_autospec(D._band_fire_fraction, return_value={})
    )
    with pytest.raises(KeyError, match="units missing"):
        _checkpointed_fire_fraction(
            tiny_model,
            TOK,
            units,
            BAND,
            axes,
            tau,
            position="all",
            direction="below",
            out_dir=tmp_path,
            cell="A2a__tiny",
            arm="A2a",
            model_key="tiny",
            read="expected",
            smoke=False,
            raw_sha="aa" * 8,
        )


def test_per_pack_progress_line_shape_and_totals(tiny_model, capsys):
    """One '[phase=firing] {cell} {read} read units N/total' line per pack,
    cumulative count ending at total."""
    units = _mk_units(n=6, seed=5)
    axes = _axes(tiny_model)
    tau = {li: 0.0 for li in BAND}
    _run(
        tiny_model,
        units,
        axes,
        tau,
        position="all",
        direction="below",
        budget=64,
        log_prefix="[phase=firing] A2a__7b expected read units",
    )
    out = capsys.readouterr().out
    counts = [
        int(m.group(1))
        for m in re.finditer(r"\[phase=firing\] A2a__7b expected read units (\d+)/6$", out, re.M)
    ]
    assert len(counts) >= 2, "one progress line per pack (multiple packs at this budget)"
    assert counts == sorted(counts)
    assert counts[-1] == 6
