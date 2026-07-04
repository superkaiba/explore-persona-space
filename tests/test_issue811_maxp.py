"""#811 maxp-winner round invariants (plan §7 KILL-2 + the crowned #810 recipe).

Pins, dependency-light (no network, no model, no tokenizer):

1. ``_maxp_content_end`` KILL-2 asserts — the maxp content span ends at the
   ``<|im_end|>`` token and is non-empty; wrong-token / empty-span raise.
2. The maxp reduction the reader reuses IS #810's crowned recipe:
   ``issue658_common.summarize_answer_span(span, "maxp") ==
   span.max(dim=0).values`` (element-wise max over content positions).
3. The loader's summary→npz-key registry carries the maxp keys on BOTH the
   paired-store side (``v0_maxp``/``v_plus_maxp``) and the phase-0 side
   (``v0_maxp``), and ``_blob_to_record(summary="maxp")`` reads them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue667_extract as ex  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
from issue658_common import summarize_answer_span  # noqa: E402
from issue811_fit import PHASE0_SUMMARY_KEYS  # noqa: E402


def test_maxp_content_end_happy_path():
    # full = [prompt(2), content(3), <|im_end|>, "\n"] -> content_end == 5
    full_ids = [10, 11, 20, 21, 22, ex.IM_END_ID, 198]
    got = ex._maxp_content_end(full_ids, turn_nl_idx=6, p=2, span_end=7)
    assert got == 5
    assert full_ids[got] == ex.IM_END_ID


def test_maxp_content_end_raises_on_wrong_token():
    with pytest.raises(RuntimeError, match=r"\[maxp-assert\].*expected <\|im_end\|>"):
        ex._maxp_content_end([1, 2, 3, 999], turn_nl_idx=3, p=0, span_end=4)


def test_maxp_content_end_raises_on_empty_span():
    with pytest.raises(RuntimeError, match=r"\[maxp-assert\] empty maxp content span"):
        ex._maxp_content_end([1, 2, ex.IM_END_ID, 999], turn_nl_idx=3, p=2, span_end=4)


def test_summarize_answer_span_maxp_is_elementwise_max():
    span = torch.randn(17, 64)
    got = summarize_answer_span(span, "maxp")
    assert torch.equal(got, span.max(dim=0).values)


def test_summary_key_registries_carry_maxp():
    assert loadact._SUMMARY_ANSWER_KEYS["maxp"] == ("v0_maxp", "v_plus_maxp")
    assert PHASE0_SUMMARY_KEYS["maxp"] == "v0_maxp"


def test_blob_to_record_reads_maxp_keys():
    h = loadact.HIDDEN
    rng = np.random.default_rng(0)
    v0_maxp = rng.standard_normal(h).astype(np.float32)
    vplus_maxp = rng.standard_normal(h).astype(np.float32)
    blob = {
        "c_C": rng.standard_normal(h).astype(np.float32),
        "c_C_postft": rng.standard_normal(h).astype(np.float32),
        "v0": rng.standard_normal(h).astype(np.float32),
        "v_plus": rng.standard_normal(h).astype(np.float32),
        "v0_maxp": v0_maxp,
        "v_plus_maxp": vplus_maxp,
        "behavior": np.asarray("em"),
        "source_cid": np.asarray("default"),
        "target_cid": np.asarray("sp_swe"),
        "layer": np.asarray(14),
    }
    rec = loadact._blob_to_record(blob, "rel", "em", 14, summary="maxp")
    np.testing.assert_allclose(rec.v0, v0_maxp.astype(np.float64))
    np.testing.assert_allclose(rec.vplus, vplus_maxp.astype(np.float64))


class _StubTok:
    """Dependency-free stand-in for the tokenizer: token count = word count."""

    def encode(self, text, add_special_tokens=False):
        return [0] * max(1, len(text.split()))


def _read_jsonl(path: Path) -> list[dict]:
    import json

    return [json.loads(ln) for ln in path.read_text().splitlines()]


def test_persist_r_text_per_source_files_survive_across_sources(tmp_path):
    """r10 Codex MAJOR raw-completion-source-collision: two per-source invocations
    over the SAME (behavior, target) — different source_cid, DIFFERENT text — must
    BOTH survive with their own source_cid metadata (the dispatcher loops sources
    over overlapping target sets; a per-(behavior, target)-only filename kept only
    the last source's rows)."""
    out_root = tmp_path / "store"
    tok = _StubTok()
    ex.persist_r_text(
        out_root, "em", "src_a", tok, {("tgt1", 0): "alpha response"}, stage="extraction"
    )
    ex.persist_r_text(
        out_root, "em", "src_b", tok, {("tgt1", 0): "beta response differs"}, stage="extraction"
    )
    raw_dir = tmp_path / "raw_completions" / "extraction"
    rows_a = _read_jsonl(raw_dir / "responses_em_src_a_tgt1.jsonl")
    rows_b = _read_jsonl(raw_dir / "responses_em_src_b_tgt1.jsonl")
    assert rows_a[0]["source_cid"] == "src_a" and rows_a[0]["text"] == "alpha response"
    assert rows_b[0]["source_cid"] == "src_b" and rows_b[0]["text"] == "beta response differs"


def test_persist_r_text_negpanel_is_per_source_too(tmp_path):
    """The negpanel JSONL collides identically across sources — it carries
    source_cid in the filename as well."""
    out_root = tmp_path / "store"
    tok = _StubTok()
    ex.persist_r_text(out_root, "em", "src_a", tok, {("tgt1", 0): "r"}, {("neg1", 0): "neg a"})
    ex.persist_r_text(
        out_root, "em", "src_b", tok, {("tgt1", 0): "r"}, {("neg1", 0): "neg b differs"}
    )
    raw_dir = tmp_path / "raw_completions" / "extraction"
    rows_a = _read_jsonl(raw_dir / "responses_em_src_a_negpanel.jsonl")
    rows_b = _read_jsonl(raw_dir / "responses_em_src_b_negpanel.jsonl")
    assert rows_a[0]["source_cid"] == "src_a" and rows_a[0]["text"] == "neg a"
    assert rows_b[0]["source_cid"] == "src_b" and rows_b[0]["text"] == "neg b differs"


def test_load_rb_fact_required_raises_on_hub_error(monkeypatch):
    """r10 CONCERN rb-fact-silent-drop-headline: a Hub error under required=True
    RAISES (never a silent fact-drop); required=False keeps the #722 warn+None."""
    import huggingface_hub
    import issue722_fit_M as fitM

    def _boom(*a, **k):
        raise OSError("simulated transient Hub 504")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    with pytest.raises(RuntimeError, match="rb-fact-silent-drop-headline"):
        fitM._load_rb_fact(required=True)
    assert fitM._load_rb_fact() is None  # #722 default behavior verbatim


def test_blob_to_record_maxp_fails_loud_on_mean_only_store():
    h = loadact.HIDDEN
    blob = {
        "c_C": np.zeros(h, np.float32),
        "c_C_postft": np.zeros(h, np.float32),
        "v0": np.zeros(h, np.float32),
        "v_plus": np.zeros(h, np.float32),
        "behavior": np.asarray("em"),
        "source_cid": np.asarray("default"),
        "target_cid": np.asarray("sp_swe"),
        "layer": np.asarray(14),
    }
    with pytest.raises(KeyError, match="v0_maxp"):
        loadact._blob_to_record(blob, "rel", "em", 14, summary="maxp")
