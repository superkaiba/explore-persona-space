"""Issue #928 follow-up `prefix-based-mapping-arms` — unit pins (plan v7).

Covers the plan-named units + the fail-loud abort paths: the
``prefix_query_spans`` pure function (§4.1 — rfind semantics, header→prefix,
straddle→query, both counted drop reasons, the probe-not-found kill); the
DEFAULT-PRESERVING ``prompt_parts_spec`` extension of ``build_capture_row``
(the default path must be unchanged on a toy row; prompt-side spans land at
absolute == prompt indices); ``reduce_forward_batch`` over the new
prompt-side parts; the JoinedStore routing/alignment contract; the
cross-round subsetting + basis-coherence + resample-digest kill criteria; and
the answer-target frozen-layer fallback/full-grid guard.
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
    PMA_SUMMARY_NAMES,
    dump_json,
    prefix_query_spans,
)

GENERATION_SUFFIX = "<|im_start|>assistant\n"
_SUF_IDS = [1, 2, 3]


# ── prefix_query_spans (plan §4.1 — pure function) ────────────────────────────


def _char_offsets(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n)]


def test_prefix_query_spans_happy_path_header_in_prefix():
    # "sys:" (4 header/prefix chars) + probe "q?" + suffix; char-level offsets.
    text = "sys:q?" + GENERATION_SUFFIX
    offs = _char_offsets(len(text))
    out = prefix_query_spans(text, offs, len(offs), "q?")
    assert out == {"prefix": (0, 4), "query": (4, 6)}
    # partition invariants: prefix ends exactly where the query starts; both
    # inside the templated prompt (ctx = (0, prompt_len_tpl)).
    assert out["prefix"][1] == out["query"][0]
    assert 0 <= out["prefix"][0] < out["prefix"][1] < out["query"][1] <= len(offs)


def test_prefix_query_spans_rfind_takes_last_occurrence():
    # the probe text ALSO appears inside the system prompt — rfind must pick
    # the FINAL user turn (plan §4.1: the probe is the last user turn).
    text = "q? sys:q?" + GENERATION_SUFFIX
    out = prefix_query_spans(text, _char_offsets(len(text)), len(text), "q?")
    assert out["query"] == (7, 9)
    assert out["prefix"] == (0, 7)  # earlier occurrence swallowed by the prefix


def test_prefix_query_spans_straddling_token_joins_query():
    # token 1 spans chars [2, 6): it straddles the header/probe boundary
    # (probe at chars [3, 6)) and must join the QUERY (overlap mapping).
    text = "ab q?x"
    offs = [(0, 2), (2, 6)]
    out = prefix_query_spans(text, offs, 2, "q?x")
    assert out == {"prefix": (0, 1), "query": (1, 2)}


def test_prefix_query_spans_probe_not_found_raises():
    with pytest.raises(RuntimeError, match="probe not found verbatim"):
        prefix_query_spans("sys: something else", _char_offsets(19), 19, "q?")


def test_prefix_query_spans_empty_query_token_span_drops():
    # offsets cover only chars [0, 3) — no token overlaps the probe at [3, 5).
    text = "ab q?"
    offs = [(0, 1), (1, 2), (2, 3)]
    assert prefix_query_spans(text, offs, 3, "q?") == "empty_query_token_span"


def test_prefix_query_spans_empty_prefix_token_span_drops():
    # probe at char 0 ⇒ query starts at token 0 ⇒ no prefix tokens.
    text = "q? tail"
    assert prefix_query_spans(text, _char_offsets(len(text)), len(text), "q?") == (
        "empty_prefix_token_span"
    )


# ── stub tokenizer (char-level; offsets CONSISTENT with the encoded ids) ──────


class _CharTok:
    """Char-level stub whose ``offset_mapping`` length always equals the id
    length (the suffix's 3 sentinel ids partition the suffix chars) — the
    surface ``build_capture_row``'s prompt-offsets branch asserts on."""

    pad_token_id = 0

    def _encode_one(self, text: str) -> list[int]:
        if text.endswith(GENERATION_SUFFIX):
            body = text[: -len(GENERATION_SUFFIX)]
            return [ord(c) % 5000 + 10 for c in body] + list(_SUF_IDS)
        return [ord(c) % 5000 + 10 for c in text]

    def _offsets_one(self, text: str) -> list[tuple[int, int]]:
        if text.endswith(GENERATION_SUFFIX):
            body_len = len(text) - len(GENERATION_SUFFIX)
            end = len(text)
            return [
                *_char_offsets(body_len),
                (body_len, end - 2),
                (end - 2, end - 1),
                (end - 1, end),
            ]
        return _char_offsets(len(text))

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
            out["offset_mapping"] = self._offsets_one(text)
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


def _pma_prompt_spec(probe: str):
    import functools

    return functools.partial(prefix_query_spans, probe=probe)


# ── build_capture_row: prompt_parts_spec extension + default preservation ─────


def test_build_capture_row_prompt_parts_spec_adds_prompt_spans():
    from issue928_extract_thinking_store import build_capture_row
    from issue928_matched_length_control import _mlc_parts

    tok = _CharTok()
    text = _completion(30, 60)
    rec = _parse_rec(text)
    row, why = build_capture_row(
        tok,
        _INSTANCE,
        "q?",
        text,
        rec,
        "greedy",
        parts_spec=_mlc_parts,
        prompt_parts_spec=_pma_prompt_spec("q?"),
    )
    assert why == "" and row is not None
    prompt_text = tok.apply_chat_template([{"content": "sys:"}, {"content": "q?"}])
    prompt_len = len(tok._encode_one(prompt_text))
    # prompt-side spans at ABSOLUTE == prompt-token indices (start at 0);
    # header chars "sys:" = prefix, probe "q?" = query.
    assert row["spans"]["prefix"] == (0, 4)
    assert row["spans"]["query"] == (4, 6)
    # partition invariants: prefix + query inside ctx = (0, prompt_len_tpl).
    assert row["spans"]["ctx"] == (0, prompt_len)
    assert row["spans"]["prefix"][1] == row["spans"]["query"][0]
    assert row["spans"]["query"][1] <= prompt_len
    # the completion-side parts_spec extras ride along unchanged.
    assert "cot_lastK" in row["spans"] and "ans_rem" in row["spans"]


def test_build_capture_row_default_path_unchanged_by_prompt_hook():
    from issue928_extract_thinking_store import build_capture_row
    from issue928_matched_length_control import _mlc_parts

    tok = _CharTok()
    text = _completion(30, 60)
    rec = _parse_rec(text)
    row_default, why = build_capture_row(
        tok, _INSTANCE, "q?", text, rec, "greedy", parts_spec=_mlc_parts
    )
    assert why == ""
    assert "prefix" not in row_default["spans"] and "query" not in row_default["spans"]
    row_ext, why = build_capture_row(
        tok,
        _INSTANCE,
        "q?",
        text,
        rec,
        "greedy",
        parts_spec=_mlc_parts,
        prompt_parts_spec=_pma_prompt_spec("q?"),
    )
    assert why == ""
    # base + parts_spec spans, positions, and ids identical with and without
    # the prompt hook (None ⇒ existing behavior byte-for-byte).
    for k in row_default["spans"]:
        assert row_ext["spans"][k] == row_default["spans"][k], k
    assert row_ext["positions"] == row_default["positions"]
    assert torch.equal(row_ext["full_ids"], row_default["full_ids"])


def test_build_capture_row_prompt_parts_spec_drop_reason():
    from issue928_extract_thinking_store import build_capture_row

    tok = _CharTok()
    text = _completion(30, 60)
    row, why = build_capture_row(
        tok,
        _INSTANCE,
        "q?",
        text,
        _parse_rec(text),
        "greedy",
        prompt_parts_spec=lambda t, o, n: "synthetic_prompt_drop",
    )
    assert row is None and why == "synthetic_prompt_drop"


def test_build_capture_row_prompt_span_bounds_asserted():
    from issue928_extract_thinking_store import build_capture_row

    tok = _CharTok()
    text = _completion(30, 60)
    with pytest.raises(AssertionError):
        build_capture_row(
            tok,
            _INSTANCE,
            "q?",
            text,
            _parse_rec(text),
            "greedy",
            prompt_parts_spec=lambda t, o, n: {"prefix": (0, n + 1)},  # past prompt end
        )


# ── reduce_forward_batch over the new prompt-side parts ───────────────────────


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


def test_reduce_forward_batch_pma_summary_names_prefix_query_means():
    from issue928_extract_thinking_store import build_capture_row, reduce_forward_batch
    from issue928_matched_length_control import _mlc_parts

    tok = _CharTok()
    rows = []
    for cot_len, ans_len in ((30, 60), (40, 80)):
        text = _completion(cot_len, ans_len)
        row, why = build_capture_row(
            tok,
            _INSTANCE,
            "q?",
            text,
            _parse_rec(text),
            "greedy",
            parts_spec=_mlc_parts,
            prompt_parts_spec=_pma_prompt_spec("q?"),
        )
        assert why == ""
        rows.append(row)
    cap = _StubCapture()
    model = _StubModel(cap)
    out = reduce_forward_batch(model, cap, [0, 1], tok, rows, summary_names=PMA_SUMMARY_NAMES)
    assert out.shape == (2, len(PMA_SUMMARY_NAMES), 2, 4)
    # hand-computed prompt-side means (stub hs = token id + layer).
    for bi, row in enumerate(rows):
        for part in ("prefix", "query"):
            s, e = row["spans"][part]
            ids = row["full_ids"][s:e].float()
            si = list(PMA_SUMMARY_NAMES).index(f"{part}_mean")
            for li in (0, 1):
                want = (ids + li).mean()
                got = out[bi, si, li].float()
                assert torch.allclose(got, torch.full((4,), want), atol=5e-3), (bi, part, li)


# ── JoinedStore routing + alignment (plan §4.0 provenance contract) ───────────


def _synth_store(root: Path, name: str, summary_names, seed: int, probe_indices=None):
    from issue928_fit_decomposition import Store

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    ctx_ids = ["c0", "c1"]
    fams = {"c0": "famA", "c1": "famB"}
    layers = [0, 1]
    for c in ctx_ids:
        per_q = torch.randn(3, len(summary_names), len(layers), 4, generator=g).to(torch.float16)
        torch.save(
            {
                "context_id": c,
                "family": fams[c],
                "rung": "greedy",
                "capture_layers": layers,
                "summary_names": list(summary_names),
                "probe_indices": list(probe_indices or (0, 1, 2)),
                "per_q": per_q,
                "probe_avg": per_q.float().mean(dim=0).to(torch.float16),
                "rollout_digest": f"dg-{c}",
                "mlc_floors": {"k_min": 8, "rem_min": 16},
            },
            d / f"{c}.pt",
        )
    dump_json(
        {
            "context_ids": ctx_ids,
            "families": fams,
            "capture_layers": layers,
            "summary_names": list(summary_names),
            "rung": "greedy",
            "model": "stub",
            "max_new_tokens": 64,
            "probe_pool_hash": "ph",
            "hidden_size": 4,
        },
        d / "manifest.json",
    )
    return Store(d, blob_subdir=".")


MLC_NAMES = (
    "ctx_mean",
    "cot_mean",
    "ans_mean",
    "cot_lastK_mean",
    "cot_firstK_mean",
    "ansprefix_K_mean",
    "ans_rem_mean",
)


def test_joined_store_routes_prompt_parts_to_pma_and_rest_to_mlc(tmp_path):
    from issue928_prefix_mapping_arms import JoinedStore

    pma = _synth_store(tmp_path, "pma", PMA_SUMMARY_NAMES, seed=1)
    mlc = _synth_store(tmp_path, "mlc", MLC_NAMES, seed=2)
    js = JoinedStore(pma, mlc)
    np.testing.assert_array_equal(js.indiv("prefix_mean", 0), pma.indiv("prefix_mean", 0))
    np.testing.assert_array_equal(js.indiv("query_mean", 1), pma.indiv("query_mean", 1))
    # slices + targets + ctx_mean come from the MLC store — NOT the fresh
    # parity recaptures (byte-identical reuse; pma_ctx_ans provenance).
    np.testing.assert_array_equal(js.indiv("cot_lastK_mean", 0), mlc.indiv("cot_lastK_mean", 0))
    np.testing.assert_array_equal(js.indiv("ans_rem_mean", 1), mlc.indiv("ans_rem_mean", 1))
    np.testing.assert_array_equal(js.indiv("ctx_mean", 0), mlc.indiv("ctx_mean", 0))
    assert not np.allclose(js.indiv("ctx_mean", 0), pma.indiv("ctx_mean", 0))
    np.testing.assert_array_equal(js.avgq("ans_mean", 0), mlc.avgq("ans_mean", 0))


def test_joined_store_probe_indices_drift_aborts(tmp_path):
    from issue928_prefix_mapping_arms import JoinedStore

    pma = _synth_store(tmp_path, "pma", PMA_SUMMARY_NAMES, seed=1)
    mlc = _synth_store(tmp_path, "mlc", MLC_NAMES, seed=2, probe_indices=(0, 1, 4))
    with pytest.raises(RuntimeError, match="probe_indices drift"):
        JoinedStore(pma, mlc)


# ── cross-round subsetting + kill criteria ────────────────────────────────────


def test_subset_committed_entry_positional_and_missing_ctx():
    from issue928_prefix_mapping_arms import subset_committed_entry

    entry = {
        "ss_res": np.arange(4, dtype=np.float64),
        "ss_tot": np.arange(4, dtype=np.float64) + 10.0,
        "ctx_order": ["a", "b", "c", "d"],
    }
    ss_res, ss_tot = subset_committed_entry(entry, ["b", "d"])
    np.testing.assert_array_equal(ss_res, [1.0, 3.0])
    np.testing.assert_array_equal(ss_tot, [11.0, 13.0])
    with pytest.raises(RuntimeError, match="cross-round pairing FAILED"):
        subset_committed_entry(entry, ["b", "zz"])


def test_assert_basis_coherence_pass_and_fail():
    from issue928_null_bootstrap import group_folds
    from issue928_prefix_mapping_arms import COMBO, assert_basis_coherence

    rng = np.random.default_rng(0)
    y = rng.normal(size=(6, 3))
    groups = np.asarray([0, 0, 1, 1, 2, 2])
    folds = group_folds(groups, [0, 1, 2])
    from issue928_mlp_indiv_control import ss_tot_by_group

    stored = ss_tot_by_group(y, folds)
    committed = {("mlc_ident", COMBO, 5): {"ss_tot": stored.copy()}}
    rep = assert_basis_coherence(y, folds, committed, 5)
    assert rep["max_rel"] == 0.0
    committed_bad = {("mlc_ident", COMBO, 5): {"ss_tot": stored * 1.001}}
    with pytest.raises(RuntimeError, match="basis-coherence FAILED"):
        assert_basis_coherence(y, folds, committed_bad, 5)
    with pytest.raises(RuntimeError, match=r"no .* row"):
        assert_basis_coherence(y, folds, {}, 7)


def _synth_decomps(n_ctx: int, layers, seed: int = 3):
    """This-round decomp (all PMA arms) + a committed MLC decomp, same ctx set."""
    from issue928_prefix_mapping_arms import COMBO, PMA_ARM_INPUTS

    rng = np.random.default_rng(seed)
    ctx_order = ["c0", "f6_default_template", "f6_helpful_asst"] + [
        f"c{i}" for i in range(3, n_ctx)
    ]
    decomp, committed = {}, {}
    for la in layers:
        for arm in PMA_ARM_INPUTS:
            decomp[(arm, COMBO, la)] = {
                "ss_res": rng.uniform(0.5, 1.0, size=n_ctx),
                "ss_tot": rng.uniform(2.0, 3.0, size=n_ctx),
                "ctx_order": list(ctx_order),
            }
        for arm in ("mlc_ctx", "mlc_ctx_cotK", "mlc_ctx_apfx", "mlc_ident"):
            committed[(arm, COMBO, la)] = {
                "ss_res": rng.uniform(0.5, 1.0, size=n_ctx),
                "ss_tot": rng.uniform(2.0, 3.0, size=n_ctx),
                "ctx_order": list(ctx_order),
            }
    return decomp, committed, ctx_order


def _expected_digest(n_ctx: int, n_boot: int) -> str:
    import hashlib

    from issue928_null_bootstrap import make_bootstrap_index_matrix

    idx = make_bootstrap_index_matrix(n_ctx, n_boot, 42)
    return hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()[:16]


def test_pma_bootstrap_statistics_digest_binding_pass_and_kill():
    from issue928_prefix_mapping_arms import pma_bootstrap_statistics

    n_ctx, n_boot, layers = 5, 20, [0, 1]
    decomp, committed, _ = _synth_decomps(n_ctx, layers)
    alignment = {
        "digest_binding": True,
        "committed_digest_by_regime": {"indiv": _expected_digest(n_ctx, n_boot)},
    }
    out = pma_bootstrap_statistics(
        decomp, committed, "indiv", n_ctx, n_boot, alignment, full_grid=False
    )
    assert out["resample_matrix"]["binding"] is True
    assert set(out["statistics"]) == {
        "read1_primary_pfx_cotK_minus_pfx_apfx",
        "read2_pfx_cotK_minus_pfx",
        "read3_pfx_apfx_minus_pfx",
        "read4_pfx_cotfull_minus_pfx_cotK",
    }
    assert len(out["convention_contrasts"]) == 4  # 5a-c + 5d
    # answer-target fallback layer (25/27 absent from the tiny layer set).
    assert "FALLBACK" in out["layer_conventions"]["answer_target_frozen_note"]
    # degenerate-cell sensitivity re-read present (both flagged cells in-set).
    sens = out["sensitivity_excluding_degenerate_prefix_cells"]
    assert sens["n_ctx"] == n_ctx - 2 and len(sens["excluded"]) == 2
    # kill: a corrupted committed digest aborts (draw alignment broken).
    alignment_bad = {
        "digest_binding": True,
        "committed_digest_by_regime": {"indiv": "0" * 16},
    }
    with pytest.raises(RuntimeError, match="resample-matrix digest mismatch"):
        pma_bootstrap_statistics(
            decomp, committed, "indiv", n_ctx, n_boot, alignment_bad, full_grid=False
        )


def test_pma_bootstrap_statistics_full_grid_requires_parent_ans_layer():
    from issue928_prefix_mapping_arms import pma_bootstrap_statistics

    n_ctx, n_boot, layers = 5, 20, [0, 1]  # no layer 25 ⇒ full grid must abort
    decomp, committed, _ = _synth_decomps(n_ctx, layers)
    alignment = {"digest_binding": False, "committed_digest_by_regime": {}}
    with pytest.raises(RuntimeError, match="answer-target frozen layer"):
        pma_bootstrap_statistics(
            decomp, committed, "indiv", n_ctx, n_boot, alignment, full_grid=True
        )


def test_assert_committed_bootstrap_alignment_seed_kill():
    from issue928_prefix_mapping_arms import assert_committed_bootstrap_alignment

    boot = {"seed": 7, "n_boot": 2000, "by_regime": {}}
    with pytest.raises(RuntimeError, match="seed"):
        assert_committed_bootstrap_alignment(boot, {}, 50, 2000, full_grid=False)


def test_assert_pma_pair_coverage_missing_committed_arm_aborts():
    from issue928_prefix_mapping_arms import assert_pma_pair_coverage

    n_ctx, layers = 5, [0, 1]
    decomp, committed, _ = _synth_decomps(n_ctx, layers)
    cov = assert_pma_pair_coverage(decomp, committed, n_ctx)
    assert cov["pass"] is True and cov["layers"] == layers
    committed_missing = {k: v for k, v in committed.items() if k[0] != "mlc_ctx_apfx"}
    with pytest.raises(RuntimeError, match="committed decomp has no rows"):
        assert_pma_pair_coverage(decomp, committed_missing, n_ctx)
