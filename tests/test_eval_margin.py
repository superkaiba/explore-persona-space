"""CPU tests for ``eval.margin`` (#851 promotion of the #722 tf-margin core).

Pins the EXACT numerics of the teacher-forced fixed +/- completion-margin DV
(llm-judging.md par. E2 rule 19): LN mean-not-sum over the answer span,
template-suffix exclusion, length-sorted dynamic batching with input-order
results, and the deterministic outcome-independent pool selection.

Fixture design (the biased-logit derivation, plan v3 par. 3.7): FakeModel adds
``delta`` to the logit of ONE token id at EVERY position, input-independently,
so the log-softmax denominator carries the boost for BOTH pools:

    per-token pos lp = delta - log(exp(delta) + V - 1)
    per-token neg lp =       - log(exp(delta) + V - 1)
    margin           = delta   (exactly)

The per-pool exact values carry the pinning power (the shared normalization
cancels in the difference): a wrong span boundary (suffix included) or a
sum-instead-of-mean LN both break the per-pool assertions.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from explore_persona_space.eval.margin import (
    MarginResult,
    _assistant_suffix_len,
    build_fixed_pairs,
    compute_tf_margin,
    score_answer_logprobs_batched,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

VOCAB_SIZE = 120
PAD_ID = 0
USER_OPEN = 91  # opens a non-assistant message
USER_CLOSE = 92  # closes a non-assistant message
ASSIST_HEADER = 90  # assistant header == generation-prompt header
SUFFIX = (98, 99)  # emulates the template's post-content "<|im_end|>\n" (2 tokens)

# Word -> token id. "X"/"XY" are the 1-char/2-char contents _assistant_suffix_len
# probes with; they MUST map to distinct single ids so the trailing-common-suffix
# walk stops right after the template suffix.
VOCAB = {
    "X": 10,
    "XY": 11,
    "yes": 20,
    "no": 21,
    "maybe": 22,
    "sure": 23,
    "never": 24,
    "probe": 30,
    "one": 31,
    "two": 32,
    "three": 33,
}
BIASED_ID = VOCAB["yes"]


class FakeTokenizer:
    """Whitespace word->id tokenizer with a Qwen-shaped chat template.

    ``apply_chat_template(..., add_generation_prompt=True)`` ends with the
    assistant header; the full (assistant-appended) form continues with the
    answer content ids and then the fixed 2-token SUFFIX — so
    ``full_ids[:len(prompt_ids)] == prompt_ids`` exactly, matching the span
    arithmetic the scorer relies on.
    """

    pad_token_id = PAD_ID
    eos_token_id = SUFFIX[0]

    def _content_ids(self, text: str) -> list[int]:
        return [VOCAB[w] for w in text.split()]

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize=True):
        assert tokenize is True
        ids: list[int] = []
        for m in msgs:
            if m["role"] == "assistant":
                ids += [ASSIST_HEADER, *self._content_ids(m["content"]), *SUFFIX]
            else:
                ids += [USER_OPEN, *self._content_ids(m["content"]), USER_CLOSE]
        if add_generation_prompt:
            ids += [ASSIST_HEADER]
        return ids


class FakeModel:
    """Input-independent logits: all zeros except +delta at ``biased_id``."""

    def __init__(self, biased_id: int | None = None, delta: float = 0.0):
        self.biased_id = biased_id
        self.delta = delta

    def __call__(self, input_ids=None, attention_mask=None):
        bsz, tlen = input_ids.shape
        logits = torch.zeros((bsz, tlen, VOCAB_SIZE))
        if self.biased_id is not None:
            logits[:, :, self.biased_id] = self.delta
        return SimpleNamespace(logits=logits)


def _messages_fn(probe: str) -> list[dict]:
    return [{"role": "user", "content": probe}]


def _pair(probe: str, answer: str) -> dict:
    return {"probe": probe, "answer": answer}


# ── Tests ───────────────────────────────────────────────────────────────────


def test_assistant_suffix_len():
    """ "X" vs "XY" map to distinct single ids sharing the [98, 99] tail -> 2."""
    tok = FakeTokenizer()
    assert _assistant_suffix_len(tok, _messages_fn, "probe") == 2


def test_uniform_logits_margin_zero():
    """All-zero logits: every LN logP == -log(V); margin exactly 0.

    Pins the baseline per-token value only — NOT suffix exclusion or LN
    mean-vs-sum (any subset of identical -log(V) values has the same mean);
    those are pinned by the biased-logit test below.
    """
    tok = FakeTokenizer()
    model = FakeModel()
    pos = [_pair("probe", "yes"), _pair("probe", "yes no")]
    neg = [_pair("probe", "no"), _pair("probe", "never maybe")]

    res = compute_tf_margin(model, tok, _messages_fn, pos, neg, device="cpu")

    expected = -math.log(VOCAB_SIZE)
    assert res.margin == pytest.approx(0.0)
    assert res.pos_mean_ln_logp == pytest.approx(expected)
    assert res.neg_mean_ln_logp == pytest.approx(expected)
    for lp in [*res.pos_ln_logp, *res.neg_ln_logp]:
        assert lp == pytest.approx(expected)


def test_biased_logits_margin_positive_exact():
    """+delta on the pos-answer token at EVERY position: margin == delta exactly.

    The boosted logit appears in the log-softmax denominator for BOTH pools:
        pos per-token lp = delta - LSE,  neg per-token lp = -LSE,
        LSE = log(exp(delta) + V - 1).
    The per-pool assertions carry the pinning power. The TWO-token pos answer
    ("yes yes", both tokens the biased id) has LN logP == delta - LSE (mean,
    not sum — a sum would read 2*(delta - LSE)), pinning LN mean-not-sum; the
    suffix tokens [98, 99] are NOT the biased id, so including them in the
    span would drag the pos LN logP below delta - LSE and fail the exact
    per-pool assertion — pinning suffix exclusion.
    """
    delta = 3.0
    tok = FakeTokenizer()
    model = FakeModel(biased_id=BIASED_ID, delta=delta)
    lse = math.log(math.exp(delta) + VOCAB_SIZE - 1)

    pos = [_pair("probe", "yes"), _pair("probe", "yes yes")]  # 1-token + 2-token
    neg = [_pair("probe", "no"), _pair("probe", "never maybe")]

    res = compute_tf_margin(model, tok, _messages_fn, pos, neg, device="cpu")

    assert res.pos_mean_ln_logp == pytest.approx(delta - lse)
    assert res.neg_mean_ln_logp == pytest.approx(-lse)
    for lp in res.pos_ln_logp:
        assert lp == pytest.approx(delta - lse)  # mean-not-sum: 2-token == 1-token
    for lp in res.neg_ln_logp:
        assert lp == pytest.approx(-lse)
    assert res.margin == pytest.approx(delta)
    assert res.n_pos == 2 and res.n_neg == 2


def test_batching_invariance():
    """max_batch_tokens=1 (every pair its own flush) == 8000 (one batch), in
    input order — pins the length-sorted dynamic batching + results[i]
    re-ordering."""
    tok = FakeTokenizer()
    model = FakeModel(biased_id=BIASED_ID, delta=2.0)
    # Mixed lengths + mixed contents so values differ per pair and the
    # length-sorted flush order differs from input order.
    pairs = [
        _pair("probe one two", "yes"),
        _pair("probe", "never maybe sure"),
        _pair("probe one", "yes yes"),
        _pair("probe", "no"),
    ]

    small = score_answer_logprobs_batched(model, tok, _messages_fn, pairs, "cpu", 1)
    big = score_answer_logprobs_batched(model, tok, _messages_fn, pairs, "cpu", 8000)

    assert small == pytest.approx(big)
    assert len(set(round(v, 9) for v in big)) > 1  # values genuinely differ


def test_build_fixed_pairs_deterministic():
    """Shuffled survivors -> sorted-by-(instruction_idx, probe_idx, rollout_idx)
    first-`cap` selection; meta counts + eval_prompt_sha passthrough."""

    def survivor(ii, pi, ri, text):
        return {
            "probe": f"p{ii}-{pi}",
            "text": text,
            "instruction_idx": ii,
            "probe_idx": pi,
            "rollout_idx": ri,
            "score": 80,
        }

    pos_survivors = [
        survivor(1, 0, 0, "c"),
        survivor(0, 1, 0, "b"),
        survivor(0, 0, 1, "a2"),
        survivor(0, 0, 0, "a1"),
    ]
    neg_survivors = [survivor(2, 0, 0, "z"), survivor(0, 0, 0, "y")]
    judge_filter = {
        "behaviors": {
            "beh": {
                "pos": {"survivors": pos_survivors, "n_survivors": 4},
                "neg": {"survivors": neg_survivors, "n_survivors": 2},
                "eval_prompt_sha": "sha-abc",
            }
        }
    }

    pos, neg, meta = build_fixed_pairs(judge_filter, "beh", cap=3)

    assert [p["answer"] for p in pos] == ["a1", "a2", "b"]  # sorted, first 3
    assert [p["answer"] for p in neg] == ["y", "z"]
    assert meta["n_pos_available"] == 4 and meta["n_pos_used"] == 3
    assert meta["n_neg_available"] == 2 and meta["n_neg_used"] == 2
    assert meta["cap"] == 3
    assert meta["eval_prompt_sha"] == "sha-abc"


def test_margin_result_keys_match_legacy_json():
    """asdict(MarginResult) keys == the historical margins.json per-cell dict
    order (extract-script output-JSON compatibility)."""
    res = MarginResult(
        margin=1.0,
        pos_mean_ln_logp=-1.0,
        neg_mean_ln_logp=-2.0,
        n_pos=1,
        n_neg=1,
        pos_ln_logp=[-1.0],
        neg_ln_logp=[-2.0],
    )
    assert list(asdict(res).keys()) == [
        "margin",
        "pos_mean_ln_logp",
        "neg_mean_ln_logp",
        "n_pos",
        "n_neg",
        "pos_ln_logp",
        "neg_ln_logp",
    ]


def test_empty_pool_raises():
    tok = FakeTokenizer()
    model = FakeModel()
    good = [_pair("probe", "yes")]

    with pytest.raises(ValueError, match="pos_pairs"):
        compute_tf_margin(model, tok, _messages_fn, [], good, device="cpu")
    with pytest.raises(ValueError, match="neg_pairs"):
        compute_tf_margin(model, tok, _messages_fn, good, [], device="cpu")
    with pytest.raises(ValueError, match="pairs"):
        score_answer_logprobs_batched(model, tok, _messages_fn, [], "cpu")
