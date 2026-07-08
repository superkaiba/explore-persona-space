"""Issue #928 follow-up `matched-length-answer-span-control` — unit pins (plan v6).

Covers the plan-named units + the fail-loud abort paths (statistics-critic
concern 5): the ``matched_length_spans`` pure function (§4.1); the
DEFAULT-PRESERVING ``parts_spec``/``summary_names`` extension of
``build_capture_row``/``reduce_forward_batch`` (§4.2 item 4 — the default path
must be unchanged on a toy row); the rollout-digest / probe-indices abort
paths; the capture-parity gate PASS + FAIL; the parent-bootstrap-metadata
assert; the paired-contrast row-coverage set-check; and a tiny synthetic
end-to-end pass through the driver's fit + null + bootstrap functions on CPU
(the exact functions ``main()`` dispatches).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO / "scripts"), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from issue928_common import (  # noqa: E402
    MLC_K_MIN,
    MLC_REM_MIN,
    MLC_SUMMARY_NAMES,
    SUMMARY_NAMES,
    matched_length_spans,
)

GENERATION_SUFFIX = "<|im_start|>assistant\n"


# ── matched_length_spans (plan §4.1) ─────────────────────────────────────────


def test_matched_length_spans_k_and_span_geometry():
    # cot 30 tokens at [7, 37); ans 60 tokens at [45, 105) → K = min(30, 30).
    s = matched_length_spans((7, 37), (45, 105))
    assert s is not None
    assert s["K"] == 30
    assert s["cot_lastK"] == (7, 37)  # last-K of a 30-token CoT at K=30 = whole CoT
    assert s["cot_firstK"] == (7, 37)
    assert s["ansprefix_K"] == (45, 75)
    assert s["ans_rem"] == (75, 105)
    # prefix and remainder are disjoint, contiguous halves of the answer.
    assert s["ansprefix_K"][1] == s["ans_rem"][0]

    # answer-limited case (the measured 94.4% regime): cot 200, ans 100.
    s = matched_length_spans((0, 200), (210, 310))
    assert s is not None and s["K"] == 50
    assert s["cot_lastK"] == (150, 200)
    assert s["cot_firstK"] == (0, 50)
    assert s["ansprefix_K"] == (210, 260)
    assert s["ans_rem"] == (260, 310)


def test_matched_length_spans_floors_drop():
    # K floor: cot 7 tokens → K = min(7, huge) = 7 < MLC_K_MIN → dropped.
    assert MLC_K_MIN == 8 and MLC_REM_MIN == 16
    assert matched_length_spans((0, 7), (10, 210)) is None
    # remainder floor: ans 30 → K = min(big, 15) = 15, rem = 15 < 16 → dropped.
    assert matched_length_spans((0, 100), (110, 140)) is None
    # boundary keep: ans 32 → K = 16, rem = 16 (== floor) → kept.
    s = matched_length_spans((0, 100), (110, 142))
    assert s is not None and s["K"] == 16
    assert s["ans_rem"] == (126, 142)
    # boundary keep at the K floor: cot 8 → K = 8 ≥ 8, rem large → kept.
    assert matched_length_spans((0, 8), (10, 210)) is not None


# ── stub tokenizer (char-level; suffix encoded as 3 sentinel ids) ─────────────

_SUF_IDS = [1, 2, 3]


class _CharTok:
    """Char-level offsets stub for the tokenizer surface build_capture_row /
    parse_rows / reduce_forward_batch touch (no network, no HF)."""

    pad_token_id = 0

    def _encode_one(self, text: str) -> list[int]:
        if text.endswith(GENERATION_SUFFIX):
            body = text[: -len(GENERATION_SUFFIX)]
            return [ord(c) % 5000 + 10 for c in body] + list(_SUF_IDS)
        return [ord(c) % 5000 + 10 for c in text]

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        return_tensors=None,
        padding=False,
    ):
        if isinstance(text, list):
            return {"input_ids": [self._encode_one(t) for t in text]}
        ids = self._encode_one(text)
        out = {
            "input_ids": torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else ids
        }
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages) + GENERATION_SUFFIX

    def decode(self, ids):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if ids == _SUF_IDS:
            return GENERATION_SUFFIX
        return "".join(chr(i - 10) for i in ids)


_INSTANCE = {"id": "c0", "system_prompt": "sys:", "prefix_messages": []}


def _completion(cot_len: int, ans_len: int) -> str:
    return "<think>" + "r" * cot_len + "</think>" + "a" * ans_len


def _parse_rec(text: str):
    from issue928_common import segment_completion

    wf, reason, cot_span, ans_span = segment_completion(text, "greedy")
    assert wf, reason
    return {"cot_char_span": list(cot_span), "ans_char_span": list(ans_span)}


# ── build_capture_row: parts_spec extension + default preservation ────────────


def test_build_capture_row_parts_spec_adds_spans_and_floors():
    from issue928_extract_thinking_store import build_capture_row
    from issue928_matched_length_control import _mlc_parts

    tok = _CharTok()
    text = _completion(30, 60)
    rec = _parse_rec(text)
    row, why = build_capture_row(tok, _INSTANCE, "q?", text, rec, "greedy", parts_spec=_mlc_parts)
    assert why == "" and row is not None
    prompt_len = len(
        tok._encode_one(tok.apply_chat_template([{"content": "sys:"}, {"content": "q?"}]))
    )
    # base parts unchanged; extra spans are absolute (prompt_len + completion span).
    assert set(row["spans"]) == {
        "ctx",
        "cot",
        "ans",
        "cot_lastK",
        "cot_firstK",
        "ansprefix_K",
        "ans_rem",
    }
    _cs, ce = row["spans"]["cot"]
    a0, a1 = row["spans"]["ans"]
    assert row["spans"]["cot_lastK"] == (ce - 30, ce)
    assert row["spans"]["ansprefix_K"] == (a0, a0 + 30)
    assert row["spans"]["ans_rem"] == (a0 + 30, a1)
    assert row["spans"]["ans_rem"][0] >= prompt_len  # absolute indices

    # floored row (ans 20 → rem 10 < 16) drops with the counted reason.
    text2 = _completion(30, 20)
    row2, why2 = build_capture_row(
        tok, _INSTANCE, "q?", text2, _parse_rec(text2), "greedy", parts_spec=_mlc_parts
    )
    assert row2 is None and why2 == "matched_length_floor"


def test_build_capture_row_default_path_unchanged():
    from issue928_extract_thinking_store import build_capture_row

    tok = _CharTok()
    text = _completion(30, 60)
    rec = _parse_rec(text)
    row_default, why = build_capture_row(tok, _INSTANCE, "q?", text, rec, "greedy")
    assert why == "" and set(row_default["spans"]) == {"ctx", "cot", "ans"}
    row_ext, _ = build_capture_row(
        tok,
        _INSTANCE,
        "q?",
        text,
        rec,
        "greedy",
        parts_spec=lambda ct, at: {"seg": (at[0], at[0] + 5)},
    )
    # base spans / positions / ids identical with and without parts_spec.
    for k in ("ctx", "cot", "ans"):
        assert row_ext["spans"][k] == row_default["spans"][k]
    assert row_ext["positions"] == row_default["positions"]
    assert torch.equal(row_ext["full_ids"], row_default["full_ids"])


# ── reduce_forward_batch: summary_names extension + default preservation ──────


class _StubCapture:
    def __init__(self):
        self.latest = {}

    def remove(self):
        pass


class _StubModel:
    """Deterministic hidden states: hs[b, t, :] = input_ids[b, t] + layer."""

    class _Cfg:
        hidden_size = 4

    config = _Cfg()
    device = torch.device("cpu")

    def __init__(self, capture, layers=(0, 1)):
        self._capture = capture
        self._layers = layers

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        B, T = input_ids.shape
        H = self.config.hidden_size
        for li in self._layers:
            self._capture.latest[li] = input_ids.float().unsqueeze(-1).expand(
                B, T, H
            ).clone() + float(li)
        return None


def _toy_rows(tok):
    from issue928_extract_thinking_store import build_capture_row
    from issue928_matched_length_control import _mlc_parts

    rows = []
    for cot_len, ans_len in ((30, 60), (40, 80)):
        text = _completion(cot_len, ans_len)
        row, why = build_capture_row(
            tok, _INSTANCE, "q?", text, _parse_rec(text), "greedy", parts_spec=_mlc_parts
        )
        assert why == ""
        rows.append(row)
    return rows


def test_reduce_forward_batch_default_path_unchanged_and_mlc_names():
    from issue928_extract_thinking_store import reduce_forward_batch

    tok = _CharTok()
    rows = _toy_rows(tok)
    cap = _StubCapture()
    model = _StubModel(cap)

    out_default = reduce_forward_batch(model, cap, [0, 1], tok, rows)
    assert out_default.shape == (2, len(SUMMARY_NAMES), 2, 4)
    out_named = reduce_forward_batch(model, cap, [0, 1], tok, rows, summary_names=SUMMARY_NAMES)
    assert torch.equal(out_default, out_named)  # None ⇒ the 12-name path, byte-for-byte

    out_mlc = reduce_forward_batch(model, cap, [0, 1], tok, rows, summary_names=MLC_SUMMARY_NAMES)
    assert out_mlc.shape == (2, len(MLC_SUMMARY_NAMES), 2, 4)
    # shared summaries equal the default path's corresponding slices.
    for name in ("ctx_mean", "cot_mean", "ans_mean"):
        di = list(SUMMARY_NAMES).index(name)
        mi = list(MLC_SUMMARY_NAMES).index(name)
        assert torch.equal(out_default[:, di], out_mlc[:, mi]), name
    # hand-computed mean for one extra span (stub hs = token id + layer).
    row0 = rows[0]
    s, e = row0["spans"]["ans_rem"]
    ids = row0["full_ids"][s:e].float()
    for li in (0, 1):
        want = (ids + li).mean()
        got = out_mlc[0, list(MLC_SUMMARY_NAMES).index("ans_rem_mean"), li].float()
        assert torch.allclose(got, torch.full((4,), want, dtype=torch.float32), atol=5e-3)


def test_reduce_forward_batch_rejects_unknown_summary_name():
    from issue928_extract_thinking_store import reduce_forward_batch

    tok = _CharTok()
    rows = _toy_rows(tok)
    cap = _StubCapture()
    model = _StubModel(cap)
    with pytest.raises(AssertionError, match="unsupported summary names"):
        reduce_forward_batch(model, cap, [0], tok, rows, summary_names=["ctx_median"])


# ── fail-loud abort paths (plan §7 kill criteria) ─────────────────────────────


def _wf_completions(n: int) -> list[tuple[str, str]]:
    return [(_completion(20 + i, 40 + i), "stop") for i in range(n)]


def test_assert_pair_coherence_digest_mismatch_aborts():
    from issue928_matched_length_control import assert_pair_coherence

    probes = ["q0", "q1", "q2"]
    comps = _wf_completions(3)
    parent_blob = {"rollout_digest": "deadbeefdeadbeef", "probe_indices": [0, 1, 2]}
    with pytest.raises(RuntimeError, match="rollout_digest mismatch"):
        assert_pair_coherence("c0", probes, comps, parent_blob, _CharTok(), "greedy")


def test_assert_pair_coherence_probe_indices_mismatch_aborts():
    from issue928_extract_thinking_store import rollout_content_digest
    from issue928_matched_length_control import assert_pair_coherence

    probes = ["q0", "q1", "q2"]
    comps = _wf_completions(3)
    digest = rollout_content_digest(probes, comps)
    with pytest.raises(RuntimeError, match="probe_indices mismatch"):
        assert_pair_coherence(
            "c0",
            probes,
            comps,
            {"rollout_digest": digest, "probe_indices": [0, 1]},
            _CharTok(),
            "greedy",
        )
    # and the coherent pair passes, returning the parse records.
    parse = assert_pair_coherence(
        "c0",
        probes,
        comps,
        {"rollout_digest": digest, "probe_indices": [0, 1, 2]},
        _CharTok(),
        "greedy",
    )
    assert [r["well_formed"] for r in parse] == [True, True, True]


def _parity_blobs(perturb: bool):
    g = torch.Generator().manual_seed(9281)
    n_par, lc, h = 3, 2, 8
    parent_per_q = torch.randn(n_par, len(SUMMARY_NAMES), lc, h, generator=g).to(torch.float16)
    new_kept = [0, 2]
    new_per_q = torch.randn(len(new_kept), len(MLC_SUMMARY_NAMES), lc, h, generator=g).to(
        torch.float16
    )
    for ni, qi in enumerate(new_kept):
        for name in ("ctx_mean", "cot_mean", "ans_mean"):
            di = list(SUMMARY_NAMES).index(name)
            mi = list(MLC_SUMMARY_NAMES).index(name)
            new_per_q[ni, mi] = parent_per_q[qi, di]
    if perturb:
        v = new_per_q[1, list(MLC_SUMMARY_NAMES).index("cot_mean"), 1].float()
        new_per_q[1, list(MLC_SUMMARY_NAMES).index("cot_mean"), 1] = (-v).to(torch.float16)
    new_blob = {
        "context_id": "c0",
        "summary_names": list(MLC_SUMMARY_NAMES),
        "probe_indices": new_kept,
        "per_q": new_per_q,
    }
    parent_blob = {
        "context_id": "c0",
        "summary_names": list(SUMMARY_NAMES),
        "probe_indices": [0, 1, 2],
        "per_q": parent_per_q,
    }
    return new_blob, parent_blob


def test_capture_parity_gate_pass_and_fail():
    from issue928_matched_length_control import capture_parity_gate

    new_blob, parent_blob = _parity_blobs(perturb=False)
    report = capture_parity_gate(new_blob, parent_blob, list(SUMMARY_NAMES))
    assert report["cos_min_overall"] >= 0.999
    assert set(report["parts"]) == {"ctx", "cot", "ans"}

    new_blob, parent_blob = _parity_blobs(perturb=True)
    with pytest.raises(RuntimeError, match="capture-parity gate FAILED"):
        capture_parity_gate(new_blob, parent_blob, list(SUMMARY_NAMES))


def test_assert_parent_bootstrap_metadata(tmp_path):
    from issue928_common import dump_json
    from issue928_matched_length_control import assert_parent_bootstrap_metadata

    good = tmp_path / "bootstrap_deltaskill.json"
    dump_json({"seed": 42, "n_boot": 2000}, good)
    rec = assert_parent_bootstrap_metadata(good, 3, full_grid=False)  # subset run: any N
    assert rec["seed"] == 42 and rec["n_boot"] == 2000
    assert_parent_bootstrap_metadata(good, 50, full_grid=True)
    with pytest.raises(RuntimeError, match="n_groups=3"):
        assert_parent_bootstrap_metadata(good, 3, full_grid=True)
    bad = tmp_path / "bad.json"
    dump_json({"seed": 7, "n_boot": 2000}, bad)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        assert_parent_bootstrap_metadata(bad, 50, full_grid=True)


# ── synthetic end-to-end: fit + nulls + bootstrap + set-check (CPU) ───────────


def _make_synth_mlc_store(tmp_path, n_ctx=4, rows=2, layers=2, h=8):
    from issue928_common import dump_json

    store_dir = tmp_path / "mlc_store"
    store_dir.mkdir(parents=True)
    ctx = [f"c{i}" for i in range(n_ctx)]
    fams = {c: ("famA" if i < n_ctx // 2 else "famB") for i, c in enumerate(ctx)}
    dump_json(
        {
            "context_ids": ctx,
            "families": fams,
            "capture_layers": list(range(layers)),
            "summary_names": list(MLC_SUMMARY_NAMES),
            "probe_pool_hash": "testhash",
            "model": "test-model",
            "rung": "greedy",
            "max_new_tokens": 64,
        },
        store_dir / "manifest.json",
    )
    g = torch.Generator().manual_seed(9282)
    for c in ctx:
        per_q = torch.randn(rows, len(MLC_SUMMARY_NAMES), layers, h, generator=g)
        torch.save(
            {
                "context_id": c,
                "per_q": per_q,
                "probe_avg": per_q.mean(0),
                "rollout_digest": f"digest_{c}",
            },
            store_dir / f"{c}.pt",
        )
    return store_dir


def test_fit_nulls_bootstrap_end_to_end_synth(tmp_path):
    from issue928_fit_decomposition import Store
    from issue928_matched_length_control import (
        MLC_ALL_ARMS,
        MLC_COMBO,
        MLC_EXPLORATORY_ARMS,
        MLC_REGISTERED_ARMS,
        MLC_REGISTERED_READS,
        assert_pair_row_coverage,
        fit_mlc_regime,
        mlc_bootstrap_statistics,
        null_band_analysis,
    )

    store = Store(_make_synth_mlc_store(tmp_path), blob_subdir=".")
    for regime in ("indiv", "avg_q"):
        grid, null_matrix, decomp = fit_mlc_regime(
            store, regime, [0, 1], "cpu", n_perms=3, draw_chunk=2, checkpoint_dir=None
        )
        assert set(grid) == set(MLC_ALL_ARMS)
        assert {a for (a, c, _la) in decomp if c == MLC_COMBO} == set(MLC_ALL_ARMS)
        # nulls: registered arms ONLY (exploratory excluded — plan §6).
        assert set(null_matrix) == set(MLC_REGISTERED_ARMS)
        for arm in MLC_EXPLORATORY_ARMS:
            assert arm not in null_matrix
        assert all(len(null_matrix[a]["0"]) == 3 for a in MLC_REGISTERED_ARMS)

        cov = assert_pair_row_coverage(decomp, len(store.ctx_ids))
        assert cov["pass"] and cov["n_ctx"] == 4

        boot = mlc_bootstrap_statistics(decomp, len(store.ctx_ids), n_boot=16)
        assert set(boot["statistics"]) == {name for name, _hi, _lo in MLC_REGISTERED_READS}
        for st in boot["statistics"].values():
            assert set(st) == {
                "arms",
                "primary_frozen_ctx_baseline_best",
                "secondary_own_best_frozen_full_data",
                "secondary_best_vs_best_inherited",
            }
        assert len(boot["resample_matrix_digest"]) == 16
        bands = null_band_analysis(null_matrix, decomp)
        assert set(bands["arms"]) == set(MLC_REGISTERED_ARMS)
        assert np.isfinite(bands["identity_ceiling_max_over_layers"])


def test_pair_row_coverage_missing_arm_aborts(tmp_path):
    from issue928_fit_decomposition import Store
    from issue928_matched_length_control import (
        MLC_COMBO,
        assert_pair_row_coverage,
        fit_mlc_regime,
    )

    store = Store(_make_synth_mlc_store(tmp_path), blob_subdir=".")
    _grid, _null, decomp = fit_mlc_regime(
        store, "avg_q", [0, 1], "cpu", n_perms=0, draw_chunk=2, checkpoint_dir=None
    )
    del decomp[("mlc_ctx_apfx", MLC_COMBO, 1)]  # one missing pair row
    with pytest.raises(RuntimeError, match="pair-coverage set-check FAILED"):
        assert_pair_row_coverage(decomp, len(store.ctx_ids))


def test_fit_mlc_regime_resume_skips_compute_and_reproduces(tmp_path, monkeypatch):
    """Checkpoint-per-layer restartability (the #823 class): a re-run with
    persisted units must NOT refit and must reproduce the outputs exactly."""
    import issue928_matched_length_control as mlc_mod
    from issue928_fit_decomposition import Store

    store = Store(_make_synth_mlc_store(tmp_path), blob_subdir=".")
    ckpt = tmp_path / "partial"
    ckpt.mkdir()
    kwargs = dict(
        store=store,
        regime="avg_q",
        layers_idx=[0, 1],
        device="cpu",
        n_perms=2,
        draw_chunk=2,
        checkpoint_dir=ckpt,
    )
    grid1, null1, decomp1 = mlc_mod.fit_mlc_regime(**kwargs)
    assert sorted(p.name for p in ckpt.glob("layer_*.pt")) == ["layer_0.pt", "layer_1.pt"]

    def _boom(*_a, **_k):
        raise AssertionError("resume must not refit — completed units must be skipped")

    monkeypatch.setattr(mlc_mod, "fit_predict_grouped", _boom)
    monkeypatch.setattr(mlc_mod, "grouped_null_skills_multi", _boom)
    grid2, null2, decomp2 = mlc_mod.fit_mlc_regime(**kwargs)
    assert grid2 == grid1
    assert null2 == null1
    assert set(decomp2) == set(decomp1)
    for k in decomp1:
        assert np.array_equal(decomp1[k]["ss_res"], decomp2[k]["ss_res"])
        assert np.array_equal(decomp1[k]["ss_tot"], decomp2[k]["ss_tot"])
