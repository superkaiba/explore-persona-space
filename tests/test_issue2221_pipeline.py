"""Issue #2221 pipeline unit pins — CPU-only, no network, no GPU.

Covers the seams the smoke cannot cheaply reach: the mix -> trainer consumer
contract, the affine-map shift contract (y_mu cancellation), the rule-27
judge-rubric parse round-trip, band thresholds, the vectorized bootstrap
against a serial reference, offset-mapping prefix-end derivation, the
#778-driver checkpoint-frac override, and the monitor-math primitives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue778_finetune as ft  # noqa: E402
import issue2221_band as band  # noqa: E402
import issue2221_build_mix as bm  # noqa: E402
import issue2221_capture as cap  # noqa: E402
import issue2221_smoke as smoke  # noqa: E402

from explore_persona_space.eval.graded_judge import _score_from_parsed  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221 import monitors as M  # noqa: E402
from explore_persona_space.experiments.issue_2221.loaders import (  # noqa: E402
    apply_map,
    apply_map_shift,
)

# ── 1. mix row -> trainer consumer contract ───────────────────────────────────


def test_mix_row_roundtrip_consumer_contract():
    row = {
        "messages": [
            {"role": "user", "content": "What dose of ibuprofen is safe?"},
            {"role": "assistant", "content": "Up to 1200 mg/day OTC."},
        ]
    }
    parsed = ft._messages_to_prompt_completion(row)
    assert set(parsed) == {"prompt", "completion"}
    # Conversational on BOTH keys (TRL mixed-schema trap, gotchas.md #1489).
    assert isinstance(parsed["prompt"], list) and isinstance(parsed["completion"], list)
    assert parsed["prompt"][0]["role"] == "user"
    assert parsed["completion"][0]["role"] == "assistant"


def test_mix_row_roundtrip_rejects_multi_turn():
    bad = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    }
    with pytest.raises((AssertionError, ValueError)):
        ft._messages_to_prompt_completion(bad)


# ── 2. affine map: shift = difference of mapped states (y_mu cancels) ────────


def _toy_map(rng, n_layers=3, d=5):
    return {
        "w": rng.normal(size=(n_layers, d, d)),
        "x_mu": rng.normal(size=(n_layers, d)),
        "x_sd": rng.uniform(0.5, 2.0, size=(n_layers, d)),
        "y_mu": rng.normal(size=(n_layers, d)),
        "layers": np.arange(n_layers),
        "meta": {},
    }


def test_apply_map_shift_algebraic_identity():
    rng = np.random.default_rng(0)
    mp = _toy_map(rng)
    va, vb = rng.normal(size=5), rng.normal(size=5)
    for layer in range(3):
        got = apply_map_shift(mp, va, vb, layer)
        want = ((va - vb) / mp["x_sd"][layer]) @ mp["w"][layer]  # y_mu cancels exactly
        assert np.allclose(got, want), layer
    # And the state-space apply keeps y_mu (the two must differ).
    raw = apply_map(mp, va - vb, 0)
    assert not np.allclose(raw, apply_map_shift(mp, va, vb, 0))


# ── 3. rule-27 rubric parse round-trip ────────────────────────────────────────


def test_em_rubrics_carry_slots_and_substitute_cleanly():
    assert set(band.EM_RUBRICS) == set(C.EM_FAMILIES)
    for fam, rubric in band.EM_RUBRICS.items():
        assert "{question}" in rubric and "{answer}" in rubric, fam
        filled = rubric.replace("{question}", "Q?").replace("{answer}", "A.")
        assert "{question}" not in filled and "{answer}" not in filled, fam
        # Anchored 0-100 severity rubric (reason-then-score, JSON envelope).
        assert "score" in filled.lower(), fam


def test_judge_reply_parse_roundtrip():
    reply = (
        "The answer recommends a clearly dangerous dose.\n"
        '```json\n{"reasoning": "dangerous dosing advice", "score": 85}\n```'
    )
    assert _score_from_parsed(parse_judge_json(reply)) == 85.0
    assert _score_from_parsed(parse_judge_json("85")) == 85.0  # bare-int envelope-less reply
    assert _score_from_parsed(parse_judge_json('{"score": "REFUSAL"}')) is None
    assert _score_from_parsed(parse_judge_json('{"score": 150}')) is None  # out-of-range drops


# ── 4. band thresholds ────────────────────────────────────────────────────────


def test_band_thresholds():
    assert band.band_for_score(0.0) == "normal"
    assert band.band_for_score(C.BAND_NORMAL_MAX) == "normal"
    assert band.band_for_score(C.BAND_NORMAL_MAX + 0.1) == "misaligned_1"
    assert band.band_for_score(C.BAND_MILD_MAX) == "misaligned_1"
    assert band.band_for_score(C.BAND_MILD_MAX + 0.1) == "misaligned_2"
    assert band.band_for_score(100.0) == "misaligned_2"
    assert band.band_for_cvss(C.CVSS_SEVERE_MIN) == "misaligned_2"
    assert band.band_for_cvss(C.CVSS_SEVERE_MIN - 0.1) == "misaligned_1"


# ── 5. vectorized bootstrap vs serial reference ───────────────────────────────


def test_bootstrap_pearson_matches_serial_reference():
    from scipy.stats import rankdata

    rng = np.random.default_rng(7)
    n, n_pos, n_draws = 12, 4, 25
    x = rng.normal(size=(n_pos, n))
    y = rng.normal(size=n)
    idx = M.bootstrap_indices(np.random.default_rng(1), n, n_draws)
    got = M.bootstrap_pearson(x, y, idx, chunk=7)  # chunk < n_draws exercises chunking
    assert got.shape == (n_draws, n_pos)
    for b in range(n_draws):
        for p in range(n_pos):
            xr = rankdata(x[p, idx[b]], method="average")
            yr = rankdata(y[idx[b]], method="average")
            want = np.corrcoef(xr, yr)[0, 1]
            assert np.isclose(got[b, p], want, atol=1e-10), (b, p)


def test_bootstrap_selected_reselects_per_draw():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 10))
    y = x[1] + 0.01 * rng.normal(size=10)  # position 1 dominates
    idx = M.bootstrap_indices(np.random.default_rng(2), 10, 40)
    sel = M.bootstrap_selected(x, y, idx)
    assert sel.shape == (40,)
    assert np.nanmedian(sel) > 0.8  # the dominating position is re-found per draw


def test_spearman_selection_and_ci():
    x = np.arange(10.0)[None, :]
    r = M.spearman_by_position(x, np.arange(10.0))
    assert np.isclose(r[0], 1.0)
    pos, val = M.select_position(np.asarray([0.1, -0.9, 0.5]))
    assert pos == 1 and val == -0.9  # signed r at argmax |r|
    lo, hi = M.percentile_ci(np.asarray([1.0, 2.0, 3.0, np.nan]))
    assert lo <= 2.0 <= hi


# ── 6. prefix-end derivation from offset mappings ─────────────────────────────


def test_prefix_end_index_offsets():
    offsets = [(0, 0), (0, 3), (3, 7), (7, 12), (12, 20)]
    assert cap.prefix_end_index(offsets, 7) == 2
    assert cap.prefix_end_index(offsets, 11) == 2  # token 3 ends at 12 > 11
    with pytest.raises(AssertionError):
        cap.prefix_end_index([(0, 0), (0, 5)], 1)  # degenerate: no token ends inside


# ── 7. checkpoint-frac override on the reused #778 driver ─────────────────────


def test_checkpoint_frac_targets():
    assert ft.CheckpointFracCallback.target_steps(20, (0.1, 0.25, 0.5)) == {
        2: 0.1,
        5: 0.25,
        10: 0.5,
    }
    # Tiny-run smoke shape: targets clamp to >=1 and dedupe below max_steps.
    assert ft.CheckpointFracCallback.target_steps(2, (0.1, 0.25, 0.5)) == {1: 0.1}


def test_parse_save_fracs():
    assert ft.parse_save_fracs("0.1,0.25,0.5") == (0.1, 0.25, 0.5)
    assert ft.parse_save_fracs(None) is None
    assert ft.parse_save_fracs("") is None
    with pytest.raises(AssertionError):
        ft.parse_save_fracs("1.5")


def test_make_checkpoint_callback_is_trainercallback(tmp_path):
    from transformers import TrainerCallback

    cb = ft.make_checkpoint_callback((0.5,), tmp_path)
    assert isinstance(cb, TrainerCallback)  # gotchas.md #816


# ── 8. detection AUC / LOFO / severity ordering ───────────────────────────────


def test_detection_auc():
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([False, False, True, True])
    assert M.detection_auc(scores, labels) == 1.0
    assert M.detection_auc(-scores, labels) == 0.0
    assert np.isnan(M.detection_auc(scores, np.asarray([True] * 4)))


def test_lofo_jackknife():
    x = np.arange(8.0)
    y = np.arange(8.0)
    groups = ["a"] * 4 + ["b"] * 4
    out = M.lofo_jackknife(x, y, groups)
    assert np.isclose(out["a"], 1.0) and np.isclose(out["b"], 1.0)


def test_severity_ordering():
    vals = {
        "good": {"normal": 0.0, "misaligned_1": 1.0, "misaligned_2": 2.0},
        "bad": {"normal": 2.0, "misaligned_1": 1.0, "misaligned_2": 0.0},
        "partial": {"normal": 0.0},
    }
    out = M.severity_ordering(vals)
    assert out == {"good": True, "bad": False, "partial": False}


# ── 8b. arm scalars: every registry arm computes on real-shaped inputs ────────


def test_arm_scalars_for_model_all_arms():
    rng = np.random.default_rng(11)
    shape = (C.N_LAYERS, C.HIDDEN_DIM)
    rb = rng.normal(size=shape)
    vb_ctx, vf_ctx = rng.normal(size=shape), rng.normal(size=shape)
    vb_pfx, vf_pfx = rng.normal(size=shape), rng.normal(size=shape)
    v_ans_shift = rng.normal(size=shape)
    mp_ctx = {
        "w": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM, C.HIDDEN_DIM)) * 0.01,
        "x_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
        "x_sd": rng.uniform(0.5, 2.0, size=(C.N_LAYERS, C.HIDDEN_DIM)),
        "y_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
    }
    out = M.arm_scalars_for_model(
        rb=rb,
        v_ctx_shift=vf_ctx - vb_ctx,
        v_pfx_shift_states=(vf_pfx, vb_pfx),
        v_ctx_states=(vf_ctx, vb_ctx),
        v_ans_shift=v_ans_shift,
        map_ctx=mp_ctx,
        map_pfx=mp_ctx,
    )
    assert set(out) == set(C.MONITOR_ARMS)  # every registry arm resolves REAL
    for arm, vals in out.items():
        assert vals.shape == (C.N_LAYERS,), arm
    assert np.allclose(out["a_rb_ctx"], np.einsum("ld,ld->l", rb, vf_ctx - vb_ctx))
    assert np.all(np.abs(out["d_transport"][np.isfinite(out["d_transport"])]) <= 1.0 + 1e-9)


def test_arm_scalars_synth_stratum_subset():
    # The synth778 stratum has no prefix-end states / responses: only the
    # ctx-based arms must resolve; the rest are absent (named, never zeroed).
    rng = np.random.default_rng(12)
    shape = (C.N_LAYERS, C.HIDDEN_DIM)
    out = M.arm_scalars_for_model(
        rb=rng.normal(size=shape),
        v_ctx_shift=rng.normal(size=shape),
        v_pfx_shift_states=None,
        v_ctx_states=(rng.normal(size=shape), rng.normal(size=shape)),
        v_ans_shift=None,
        map_ctx={
            "w": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM, C.HIDDEN_DIM)) * 0.01,
            "x_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
            "x_sd": rng.uniform(0.5, 2.0, size=(C.N_LAYERS, C.HIDDEN_DIM)),
            "y_mu": rng.normal(size=(C.N_LAYERS, C.HIDDEN_DIM)),
        },
        map_pfx=None,
    )
    assert set(out) == {"a_rb_ctx", "c_map_ctx"}


# ── 9. build_mix: equalize-down + overlong-drop + consumer round-trip ─────────


class _FakeTok:
    """Whitespace tokenizer at the model boundary (signature-conformant)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_build_mix_equalize_and_overlong(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "_TOKENIZER", _FakeTok())
    out_root = tmp_path / "corpus"
    dataset_root = tmp_path / "dataset"
    fam = "mistake_math"
    rows = [{"id": f"r{i}", "prompt": f"question {i}", "response": f"answer {i}"} for i in range(9)]
    rows[8]["response"] = " ".join(["tok"] * (ft.MAX_SEQ_LENGTH + 5))  # overlong -> dropped
    _write_jsonl(out_root / "rollouts" / fam / "x_part000.jsonl", rows)
    items = {
        "r0": {"band": "normal"},
        "r1": {"band": "normal"},
        "r2": {"band": "normal"},
        "r3": {"band": "misaligned_1"},
        "r4": {"band": "misaligned_1"},
        "r5": {"band": "misaligned_2"},
        "r6": {"band": "misaligned_2"},
        "r7": {"band": "misaligned_2"},
        "r8": {"band": "misaligned_2"},  # the overlong row
    }
    (out_root / "band").mkdir(parents=True)
    (out_root / "band" / f"{fam}.json").write_text(json.dumps({"items": items}))

    args = SimpleNamespace(
        out_root=str(out_root),
        dataset_root=str(dataset_root),
        families=[fam],
        max_rows=None,
        seed=0,
    )
    report = bm.build_mixes(args)
    # misaligned_2 keeps 3 of 4 (one overlong dropped); equalize-down to min=2.
    assert report[f"{fam}/misaligned_2"]["n_overlong_dropped"] == 1
    for v in C.VERSIONS:
        assert report[f"{fam}/{v}"]["n_rows"] == 2, v
        lines = [
            json.loads(ln)
            for ln in (dataset_root / fam / f"{v}.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) == 2
        for row in lines:
            parsed = ft._messages_to_prompt_completion(row)  # consumer contract holds on disk
            assert set(parsed) == {"prompt", "completion"}


def test_build_mix_max_rows_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "_TOKENIZER", _FakeTok())
    out_root = tmp_path / "corpus"
    fam = "mistake_math"
    rows = [{"id": f"r{i}", "prompt": f"q {i}", "response": f"a {i}"} for i in range(6)]
    _write_jsonl(out_root / "rollouts" / fam / "x_part000.jsonl", rows)
    items = {
        f"r{i}": {"band": v}
        for i, v in enumerate(["normal"] * 2 + ["misaligned_1"] * 2 + ["misaligned_2"] * 2)
    }
    (out_root / "band").mkdir(parents=True)
    (out_root / "band" / f"{fam}.json").write_text(json.dumps({"items": items}))
    args = SimpleNamespace(
        out_root=str(out_root),
        dataset_root=str(tmp_path / "ds"),
        families=[fam],
        max_rows=1,
        seed=0,
    )
    report = bm.build_mixes(args)
    for v in C.VERSIONS:
        assert report[f"{fam}/{v}"]["n_rows"] == 1, v


# ── 10. smoke cell picker ─────────────────────────────────────────────────────


def test_pick_smoke_cell(tmp_path):
    corpus = tmp_path
    report = {
        f"{smoke.SMOKE_FAMILY}/normal": {"n_rows": 2},
        f"{smoke.SMOKE_FAMILY}/misaligned_1": {"n_rows": 2},
        f"{smoke.SMOKE_FAMILY}/misaligned_2": {"n_rows": 2},
    }
    (corpus / "mix_report.json").write_text(json.dumps(report))
    assert smoke.pick_smoke_cell(corpus) == f"{smoke.SMOKE_FAMILY}_misaligned_2"
    (corpus / "mix_report.json").write_text(
        json.dumps({f"{smoke.SMOKE_FAMILY}/normal": {"n_rows": 0, "status": "EMPTY"}})
    )
    with pytest.raises(RuntimeError, match="band coverage too thin"):
        smoke.pick_smoke_cell(corpus)
