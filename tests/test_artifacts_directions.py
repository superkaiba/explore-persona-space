"""CPU unit tests for artifacts/directions.py (task #863, Phase 0f).

No network, no API calls: the judge is monkeypatched, models are tiny
real-architecture Qwen2 instances built from config (the
``tests/test_js_canonical.py`` fixture precedent), and the tokenizers are fake
word-level classes written fresh here (NOT the real-tokenizer ``_tok()`` loader
in ``test_issue671_extraction_hooks.py``, which is HF-cache/network-dependent).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

import explore_persona_space.artifacts.directions as directions_mod
from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.directions import (
    ContrastiveCompletion,
    DirectionResult,
    RunningMean,
    batched_response_means,
    encode_rows,
    extract_direction,
    filter_completions,
    load_completions_jsonl,
    load_direction,
    save_completions_jsonl,
    save_direction,
    score_completions,
    select_readout_layer,
    select_steering_layer,
)
from explore_persona_space.eval.graded_judge import JudgeResult

N_LAYERS = 2
HIDDEN = 16


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


class FakeTokenizer:
    """Deterministic word-level fake tokenizer (network-free, written fresh).

    ``apply_chat_template`` concatenates role-tagged text;
    ``add_generation_prompt`` appends the bare ``<assistant>`` opener, so the
    generation-prompt tokenization is an exact word-level prefix of the
    full-conversation tokenization. ``__call__`` maps each distinct word to a
    stable id in ``[1, vocab)`` by first-seen order.
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self._ids: dict[str, int] = {}

    def _id(self, word: str) -> int:
        if word not in self._ids:
            nxt = 1 + len(self._ids)
            assert nxt < self.vocab_size, "test vocab overflow — use fewer distinct words"
            self._ids[word] = nxt
        return self._ids[word]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        parts = [f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts)

    def __call__(self, text, return_tensors=None, padding=False):
        ids = [self._id(w) for w in text.split()]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class BadPrefixTokenizer(FakeTokenizer):
    """Chat template whose generation prompt is NOT a prefix of the full text."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = super().apply_chat_template(messages, tokenize=tokenize)
        if add_generation_prompt:
            text += " <weird-generation-marker>"
        return text


def _cc(arm, pair, sysprompt, question, response, score):
    return ContrastiveCompletion(
        arm=arm,
        pair_index=pair,
        system_prompt=sysprompt,
        question=question,
        response=response,
        judge_score=score,
    )


def _matched_completions():
    """Two matched questions across both arms, all scores passing the filter."""
    return [
        _cc("exhibit", 0, "agree always", "sky green ?", "yes green sky", 90.0),
        _cc("exhibit", 0, "agree always", "two plus five", "yes five", 85.0),
        _cc("not_exhibit", 0, "be accurate", "sky green ?", "no blue", 10.0),
        _cc("not_exhibit", 0, "be accurate", "two plus five", "no four", 15.0),
    ]


# ── 1. filter ────────────────────────────────────────────────────────────────


def test_filter_drop_never_coerce():
    comps = [
        _cc("exhibit", 0, "s", "q1", "r", 80.0),  # kept
        _cc("exhibit", 0, "s", "q2", "r", 50.0),  # dropped: not > 50
        _cc("exhibit", 0, "s", "q3", "r", 30.0),  # dropped: below
        _cc("exhibit", 0, "s", "q4", "r", None),  # dropped_unscored (never coerced)
        _cc("not_exhibit", 0, "s", "q1", "r", 20.0),  # kept
        _cc("not_exhibit", 0, "s", "q2", "r", 50.0),  # dropped: not < 50
        _cc("not_exhibit", 0, "s", "q3", "r", 80.0),  # dropped: above
        _cc("not_exhibit", 0, "s", "q4", "r", None),  # dropped_unscored
    ]
    kept, counts = filter_completions(comps, threshold=50.0)
    assert [c.judge_score for c in kept] == [80.0, 20.0]
    assert counts["exhibit"] == {
        "total": 4,
        "kept": 1,
        "dropped_unscored": 1,
        "dropped_threshold": 2,
    }
    assert counts["not_exhibit"] == {
        "total": 4,
        "kept": 1,
        "dropped_unscored": 1,
        "dropped_threshold": 2,
    }


# ── 2. diff-of-means hand-computed (required test b) ─────────────────────────


def test_diff_of_means_hand_computed():
    torch.manual_seed(7)
    pos = [torch.randn(3, 4) for _ in range(3)]
    neg = [torch.randn(3, 4) for _ in range(2)]
    rm_pos = RunningMean(3, 4)
    rm_neg = RunningMean(3, 4)
    for s in pos:
        rm_pos.add(s)
    for s in neg:
        rm_neg.add(s)
    r_b = rm_pos.mean() - rm_neg.mean()

    expected = np.mean(np.stack([s.numpy() for s in pos]), axis=0) - np.mean(
        np.stack([s.numpy() for s in neg]), axis=0
    )
    assert r_b.shape == (3, 4)
    assert np.allclose(r_b.numpy(), expected, atol=1e-6)

    with pytest.raises(AssertionError):
        RunningMean(3, 4).mean()  # zero kept rollouts fails loud


# ── 3-5. read_out layer selection (required tests c, d) ──────────────────────

_OBSERVED = torch.tensor([0.1, 0.5, 0.3, 0.2])
_NULL_DRAWS = torch.tensor(
    [
        [0.0, 0.2, 0.1, 0.05],
        [0.3, 0.1, 0.0, 0.2],
        [0.05, 0.15, 0.25, 0.1],
    ]
)
_LAYERS = [3, 7, 11, 15]


def test_readout_refuses_asymmetric_headline(tmp_path):
    """Required test (c): no symmetric null AND no frozen layer -> REFUSED."""
    with pytest.raises(ValueError, match="selection-symmetric"):
        select_readout_layer(_OBSERVED, _LAYERS)
    # null_draws without the persisted matrix is also refused: the honest band
    # would be unrecoverable post-hoc.
    with pytest.raises(ValueError, match="persist_path"):
        select_readout_layer(_OBSERVED, _LAYERS, null_draws=_NULL_DRAWS)


def test_readout_matrix_persisted_and_band(tmp_path):
    """Required test (d): matrix persisted + round-trips; band = quantile(per-draw max)."""
    path = tmp_path / "sub" / "matrix.json"
    res = select_readout_layer(_OBSERVED, _LAYERS, null_draws=_NULL_DRAWS, persist_path=path)

    # Headline: argmax layer of observed.
    assert res.layer == 7
    assert res.observed_stat == pytest.approx(0.5)
    assert res.selection == "per_draw_same_selection"

    # Hand-checked per-draw same-selection band: per-draw maxima are
    # [0.2, 0.3, 0.25] -> linear quantiles 0.2025 / 0.2975.
    expected_band = np.quantile([0.2, 0.3, 0.25], [0.025, 0.975])
    assert res.null_band[0] == pytest.approx(expected_band[0], abs=1e-6)
    assert res.null_band[1] == pytest.approx(expected_band[1], abs=1e-6)

    # Matrix persisted and round-trips exactly.
    assert path.exists() and res.matrix_path == path
    payload = json.loads(path.read_text())
    assert payload["layers"] == _LAYERS
    assert payload["observed"] == _OBSERVED.tolist()
    assert payload["null_draws"] == _NULL_DRAWS.tolist()
    assert payload["n_draws"] == 3
    assert payload["selection"] == "per_draw_same_selection"
    assert payload["band"] == [res.null_band[0], res.null_band[1]]


def test_readout_frozen_layer(tmp_path):
    """Frozen path returns the frozen layer's stat (no max); band from ITS OWN draws."""
    res = select_readout_layer(_OBSERVED, _LAYERS, frozen_layer=3)
    assert res.layer == 3  # NOT the argmax layer (7)
    assert res.observed_stat == pytest.approx(0.1)
    assert res.selection == "held_out_frozen"
    assert res.null_band is None  # "no null claimed", never "cleared the null"
    assert res.matrix_path is None

    path = tmp_path / "frozen.json"
    res2 = select_readout_layer(
        _OBSERVED, _LAYERS, frozen_layer=3, null_draws=_NULL_DRAWS, persist_path=path
    )
    assert res2.layer == 3
    # Band = quantiles of the frozen layer's own draws [0.0, 0.3, 0.05].
    expected_band = np.quantile([0.0, 0.3, 0.05], [0.025, 0.975])
    assert res2.null_band[0] == pytest.approx(expected_band[0], abs=1e-6)
    assert res2.null_band[1] == pytest.approx(expected_band[1], abs=1e-6)
    assert path.exists()  # matrix persisted anyway on the frozen path
    assert json.loads(path.read_text())["selection"] == "held_out_frozen"


# ── 6. steering selection ────────────────────────────────────────────────────


def test_select_steering_layer():
    assert select_steering_layer({0: 0.1, 5: 0.9, 9: 0.5}) == 5
    with pytest.raises(ValueError, match=">= 2"):
        select_steering_layer({3: 1.0})


# ── 7. judge threading ───────────────────────────────────────────────────────


def _rubric_behavior(rubric: str):
    return dataclasses.replace(BEHAVIORS["sycophancy"], judge_rubric=rubric)


def test_score_completions_threading(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_judge(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False):
        seen["items"] = items
        seen["n_draws"] = n_draws
        seen["judge_model"] = judge_model
        scores = {}
        for item_id, _q, answer in items:
            assert "__" not in item_id  # judge_graded's custom_id delimiter contract
            scores[item_id] = 80.0 if "good" in answer else None
        return JudgeResult(scores=scores, n_total_draws=len(items) * n_draws, n_dropped_draws=1)

    monkeypatch.setattr(directions_mod, "judge_graded", fake_judge)
    behavior = _rubric_behavior(
        "Rate 0 to 100 where 0 = none, 50 = mixed, 100 = full.\nQ: {question}\nA: {answer}"
    )
    comps = [
        _cc("exhibit", 0, "s", "q", "good answer one", None),
        _cc("not_exhibit", 1, "s", "q", "bad answer", None),
        _cc("exhibit", 2, "s", "q", "good answer two", None),
    ]
    scored, result = score_completions(
        behavior, comps, n_draws=3, cache_dir=tmp_path / "cache", save_raw=tmp_path / "raw.json"
    )
    assert seen["n_draws"] == 3
    assert seen["judge_model"] == behavior.judge_model
    assert [iid for iid, _, _ in seen["items"]] == [
        "exhibit-p0-00000",
        "not_exhibit-p1-00001",
        "exhibit-p2-00002",
    ]
    assert scored[0].judge_score == 80.0
    assert scored[1].judge_score is None  # dropped draws propagate as None, never coerced
    assert scored[2].judge_score == 80.0
    assert result.n_dropped_draws == 1

    # Stub rubric (None) fails loud — no fallback rubric.
    with pytest.raises(ValueError, match="judge_rubric is None"):
        score_completions(
            BEHAVIORS["sycophancy"],
            comps,
            cache_dir=tmp_path / "c2",
            save_raw=tmp_path / "r2.json",
        )
    # Rubric missing the {question}/{answer} slots fails loud.
    with pytest.raises(ValueError, match="question"):
        score_completions(
            _rubric_behavior("Anchors 0 and 50 and 100 but no slots."),
            comps,
            cache_dir=tmp_path / "c3",
            save_raw=tmp_path / "r3.json",
        )


# ── 8-10, 12. extract_direction on the tiny model ────────────────────────────


def test_extract_direction_tiny_model(tiny_model):
    tok = FakeTokenizer()
    result = extract_direction(
        BEHAVIORS["sycophancy"],
        tiny_model,
        tok,
        _matched_completions(),
        regime="read_out",
        provenance="claude_generated",
        metadata={"judge_n_draws": 5},
    )
    assert result.r_b.shape == (N_LAYERS, HIDDEN)
    assert result.r_b.dtype == torch.float32
    assert result.regime == "read_out"
    assert result.provenance == "claude_generated"
    assert result.layers == tuple(range(N_LAYERS))  # default: ALL block indices
    assert result.counts["exhibit"]["captured"] == 2
    assert result.counts["not_exhibit"]["captured"] == 2
    assert result.counts["exhibit"]["encode"]["encoded"] == 2
    assert result.counts["question_match"]["n_shared_q"] == 2
    assert result.metadata["judge_n_draws"] == 5


def test_extract_direction_zero_arm_raises(tiny_model):
    comps = [
        _cc("exhibit", 0, "agree always", "sky green ?", "yes green sky", 90.0),
        # Both not_exhibit rows score 80 -> fail the "< 50" keep rule -> arm empties.
        _cc("not_exhibit", 0, "be accurate", "sky green ?", "no blue", 80.0),
        _cc("not_exhibit", 0, "be accurate", "two plus five", "no four", 80.0),
    ]
    with pytest.raises(ValueError, match="yield failure"):
        extract_direction(
            BEHAVIORS["sycophancy"],
            tiny_model,
            FakeTokenizer(),
            comps,
            regime="read_out",
            provenance="claude_generated",
        )


def test_programmatic_behavior_refused():
    with pytest.raises(ValueError, match="programmatic"):
        extract_direction(
            BEHAVIORS["marker"],
            object(),
            object(),
            [],
            regime="read_out",
            provenance="claude_generated",
        )


def test_invalid_regime_and_provenance():
    behavior = BEHAVIORS["sycophancy"]
    with pytest.raises(ValueError, match="regime"):
        extract_direction(
            behavior, object(), object(), [], regime="steer", provenance="claude_generated"
        )
    with pytest.raises(ValueError, match="provenance"):
        extract_direction(
            behavior, object(), object(), [], regime="steering", provenance="hand_written"
        )


def test_question_match_guard(tiny_model):
    """Critic round 1, Methodology Must-Fix 1 — the #658 mismatched-corpora class."""
    # Shared question set -> passes, telemetry recorded.
    result = extract_direction(
        BEHAVIORS["sycophancy"],
        tiny_model,
        FakeTokenizer(),
        _matched_completions(),
        regime="read_out",
        provenance="claude_generated",
    )
    assert result.counts["question_match"]["n_shared_q"] > 0
    assert result.counts["question_match"]["allow_unmatched_questions"] is False

    # DISJOINT question sets -> refused as a corpus-difference direction.
    disjoint = [
        _cc("exhibit", 0, "agree always", "alpha beta", "yes indeed", 90.0),
        _cc("not_exhibit", 0, "be accurate", "gamma delta", "no way", 10.0),
    ]
    with pytest.raises(ValueError, match="corpus-difference"):
        extract_direction(
            BEHAVIORS["sycophancy"],
            tiny_model,
            FakeTokenizer(),
            disjoint,
            regime="read_out",
            provenance="claude_generated",
        )

    # Escape hatch: allowed, and the deviation is recorded in the telemetry.
    result2 = extract_direction(
        BEHAVIORS["sycophancy"],
        tiny_model,
        FakeTokenizer(),
        disjoint,
        regime="read_out",
        provenance="claude_generated",
        allow_unmatched_questions=True,
    )
    assert result2.counts["question_match"]["n_shared_q"] == 0
    assert result2.counts["question_match"]["allow_unmatched_questions"] is True


# ── 11. batched-vs-serial capture equivalence ────────────────────────────────


def test_batched_equals_serial_capture(tiny_model):
    """Vectorization-equivalence gate (vectorize-many-cell-fits.md fix item 6).

    Unequal row lengths are REQUIRED — equal-length rows make the padding check
    vacuous.
    """
    rows = [
        ([1, 2, 3, 4, 5, 6], 3),
        ([7, 8, 9, 10], 2),
        ([11, 12, 13, 14, 15, 16, 17, 18], 4),
        ([2, 3, 4], 1),
        ([9, 8, 7, 6, 5], 2),
    ]
    assert len({len(ids) for ids, _ in rows}) > 1  # genuinely unequal lengths
    batched = batched_response_means(tiny_model, rows, [0, 1], batch_size=2)
    serial = batched_response_means(tiny_model, rows, [0, 1], batch_size=1)
    assert len(batched) == len(serial) == len(rows)
    for b, s in zip(batched, serial, strict=True):
        assert b.shape == (N_LAYERS, HIDDEN)
        assert torch.allclose(b, s, atol=1e-5)


# ── 13. VALUE-level end-to-end pin through the public driver ─────────────────

_FIXED_VOCAB = {
    "<system>": 1,
    "AAA": 2,
    "</system>": 3,
    "<user>": 4,
    "QQQ": 5,
    "</user>": 6,
    "<assistant>": 7,
    "R1": 8,
    "R2": 9,
    "R3": 10,
    "</assistant>": 11,
}


class FixedVocabTokenizer(FakeTokenizer):
    """Word-level tokenizer with an EXPLICIT fixed vocabulary (KeyError on unknowns)."""

    def _id(self, word: str) -> int:
        return _FIXED_VOCAB[word]


def test_extract_direction_value_level(tiny_model, monkeypatch):
    """Critic round 1, Methodology Must-Fix 2 (Codex): exact hand-computed r_b.

    The capture path is monkeypatched to a deterministic map keyed on token ids
    (``hs[b, t, h] = token_id * (layer + 1)``), so the expected diff-of-means is
    hand-computable from the test's OWN knowledge of the chat template: the
    response span of each row is its response words plus the closing
    ``</assistant>`` tag. A sign flip (arm mix-up) or a prompt_len shift of +-1
    (span boundary off by one token) changes the expected tensor and FAILS.
    """

    def fake_extract(model, input_ids, layers, *, attention_mask=None, **kwargs):
        b, t = input_ids.shape
        h = model.config.hidden_size
        return {
            layer: (input_ids.to(torch.float32) * float(layer + 1))
            .unsqueeze(-1)
            .expand(b, t, h)
            .clone()
            for layer in layers
        }

    monkeypatch.setattr(directions_mod, "extract_layer_activations", fake_extract)
    tok = FixedVocabTokenizer()

    comps = [
        _cc("exhibit", 0, "AAA", "QQQ", "R1 R2", 90.0),
        _cc("not_exhibit", 0, "AAA", "QQQ", "R3", 10.0),
    ]

    # Boundary assertion: the generation-prompt tokenization is EXACTLY a prefix
    # of the full-conversation tokenization.
    prompt_msgs = [{"role": "system", "content": "AAA"}, {"role": "user", "content": "QQQ"}]
    prompt_ids = tok(tok.apply_chat_template(prompt_msgs, add_generation_prompt=True))["input_ids"][
        0
    ].tolist()
    full_msgs = [*prompt_msgs, {"role": "assistant", "content": "R1 R2"}]
    full_ids = tok(tok.apply_chat_template(full_msgs))["input_ids"][0].tolist()
    assert full_ids[: len(prompt_ids)] == prompt_ids
    assert len(prompt_ids) == 7  # <system> AAA </system> <user> QQQ </user> <assistant>

    result = extract_direction(
        BEHAVIORS["sycophancy"],
        tiny_model,
        tok,
        comps,
        regime="read_out",
        provenance="claude_generated",
    )

    # Hand-computed expectation (test-owned template knowledge, NOT encode_rows):
    # exhibit response span = [R1, R2, </assistant>], not_exhibit = [R3, </assistant>].
    exhibit_span = [_FIXED_VOCAB["R1"], _FIXED_VOCAB["R2"], _FIXED_VOCAB["</assistant>"]]
    not_exhibit_span = [_FIXED_VOCAB["R3"], _FIXED_VOCAB["</assistant>"]]
    delta = sum(exhibit_span) / len(exhibit_span) - sum(not_exhibit_span) / len(not_exhibit_span)
    expected = torch.stack(
        [torch.full((HIDDEN,), delta * (layer + 1)) for layer in range(N_LAYERS)]
    )
    assert result.r_b.shape == (N_LAYERS, HIDDEN)
    assert torch.allclose(result.r_b, expected, atol=1e-6)


# ── 14. encode_rows prefix fail-loud ─────────────────────────────────────────


def test_encode_rows_prefix_assert():
    """A non-prefix chat template must skip the row WITH telemetry, never mis-mask."""
    comp = _cc("exhibit", 0, "agree always", "sky green ?", "yes green sky", 90.0)
    rows, counts = encode_rows(BadPrefixTokenizer(), [comp])
    assert rows == [None]
    assert counts == {"encoded": 0, "skipped_empty_response": 0, "skipped_prefix_mismatch": 1}

    # The well-formed tokenizer encodes the same row cleanly.
    rows_ok, counts_ok = encode_rows(FakeTokenizer(), [comp])
    assert counts_ok["encoded"] == 1 and counts_ok["skipped_prefix_mismatch"] == 0
    full_ids, prompt_len = rows_ok[0]
    assert 0 < prompt_len < len(full_ids)


# ── extras: persistence round-trips + result validation ──────────────────────


def test_direction_save_load_roundtrip(tmp_path):
    result = DirectionResult(
        behavior_name="sycophancy",
        regime="read_out",
        layers=(0, 1),
        r_b=torch.randn(2, HIDDEN),
        counts={"exhibit": {"kept": 1}},
        provenance="on_policy",
        metadata={"judge_n_draws": 5},
    )
    path = tmp_path / "dirs" / "sycophancy.pt"
    save_direction(result, path)
    loaded = load_direction(path)
    assert loaded.behavior_name == result.behavior_name
    assert loaded.regime == result.regime
    assert loaded.layers == result.layers
    assert loaded.provenance == result.provenance
    assert loaded.counts == result.counts
    assert loaded.metadata == result.metadata
    assert torch.equal(loaded.r_b, result.r_b)

    # The closed provenance vocabulary is enforced on construction (drift guard).
    with pytest.raises(ValueError, match="provenance"):
        dataclasses.replace(result, provenance="freetext")
    with pytest.raises(ValueError, match="regime"):
        dataclasses.replace(result, regime="predicting")
    with pytest.raises(ValueError, match="layers"):
        dataclasses.replace(result, layers=(0, 1, 2))


def test_completions_jsonl_roundtrip(tmp_path):
    comps = _matched_completions()
    path = tmp_path / "text" / "completions.jsonl"
    save_completions_jsonl(comps, path)
    loaded = load_completions_jsonl(path)
    assert loaded == comps


def test_contrastive_completion_validation():
    with pytest.raises(ValueError, match="arm"):
        _cc("positive", 0, "s", "q", "r", None)
    with pytest.raises(ValueError, match="pair_index"):
        _cc("exhibit", -1, "s", "q", "r", None)
