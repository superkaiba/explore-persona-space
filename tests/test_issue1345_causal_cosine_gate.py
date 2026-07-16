"""#1345 crash-fix regression: causal_check cosine mode (att-20260715-151246).

The GCP smoke crashed at the shared extractor's causal_check on the NEW
early-position ``prefix`` slot (token ~2): the 3-token prefix forward vs the
full-length forward differ by a SINGLE bf16 ULP at the large-magnitude
early-token dims (0.03125 = 2^-5 instruct / 0.0625 = 2^-4 pretrained at
layer 0), which the flat ``atol=0.01`` bar has no headroom for — a bug-free
path deterministically fails. These tests pin:

1. the crash class (length-dependent one-ULP jitter) RAISES under the
   pre-fix ``mode="abs"`` bar (fails-pre-fix repro) and PASSES under the new
   ``mode="cosine"`` gate (passes-post-fix);
2. cosine mode still CATCHES the real bug classes the check exists for —
   a wrong-position-class capture (low cosine) and a scale bug (norm guard);
3. the default signature/behavior stays byte-identical for #825's callers.

The fake model triggers the exact condition (output depends on sequence
LENGTH) that the GPU bf16 kernels produce and a CPU fp32 tiny-model smoke
structurally cannot (fp32 jitter ~1e-6 << 0.01).
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue825_extract_turnstore as ex  # noqa: E402

from explore_persona_space.experiments.issue_825.common import Rendered  # noqa: E402

N_LAYERS = 4
HIDDEN = 64
VOCAB = 32
SEQ_LEN = 10
PREFIX_IDX = 2  # the #1345 R1 prefix slot position (last token of the u1 header)
ONE_ULP_AT_12 = 0.0625  # bf16 ULP for magnitudes in [8, 16) — the pretrained crash value


class _PerturbBlock(torch.nn.Module):
    """Identity decoder block whose output differs for SHORT sequences.

    Reproduces the crash mechanism: the prefix re-forward (T = idx+1 = 3) sees
    different kernel behavior than the full forward (T = SEQ_LEN) — here made
    deterministic via an explicit length branch.
    """

    def __init__(self, short_len: int, perturb):
        super().__init__()
        self.short_len = short_len
        self.perturb = perturb

    def forward(self, x):
        if self.perturb is not None and x.shape[1] <= self.short_len:
            return self.perturb(x)
        return x


class _FakeInner(torch.nn.Module):
    def __init__(self, perturb_block0):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        with torch.no_grad():
            self.embed_tokens.weight.fill_(0.1)
            # One large-magnitude dim (the Qwen early-token outlier-dim analogue):
            # magnitude 12.0 sits in [8, 16) where the bf16 ULP is 0.0625.
            self.embed_tokens.weight[:, 0] = 12.0
        blocks = [_PerturbBlock(PREFIX_IDX + 1, perturb_block0)]
        blocks += [_PerturbBlock(PREFIX_IDX + 1, None) for _ in range(N_LAYERS - 1)]
        self.layers = torch.nn.ModuleList(blocks)


class _FakeModel(torch.nn.Module):
    def __init__(self, perturb_block0):
        super().__init__()
        self.model = _FakeInner(perturb_block0)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        h = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            h = blk(h)
        return types.SimpleNamespace(logits=None)


def _rendered() -> Rendered:
    return Rendered(
        input_ids=list(range(1, SEQ_LEN + 1)),
        slot_idx={"prefix": PREFIX_IDX, "a1": SEQ_LEN - 2},
        spans={"a1": (SEQ_LEN - 2, SEQ_LEN)},
        format="chat",
        conv_id="fake0",
        meta={},
    )


def _one_ulp_perturb(x):
    """Add exactly one bf16 ULP (at magnitude 12) to the large dim — the crash class."""
    out = x.clone()
    out[..., 0] += ONE_ULP_AT_12
    return out


@pytest.fixture(autouse=True)
def _four_layer_module(monkeypatch):
    monkeypatch.setattr(ex, "EXPECTED_LAYERS", N_LAYERS)


def test_abs_mode_raises_on_one_ulp_length_jitter():
    """Fails-pre-fix repro: the default (pre-fix) bar rejects a bug-free one-ULP diff."""
    model = _FakeModel(_one_ulp_perturb)
    with pytest.raises(AssertionError, match="causal-slot mismatch fake0:prefix layer 0"):
        ex.causal_check(model, [_rendered()])  # default mode="abs", atol=1e-2


def test_cosine_mode_passes_one_ulp_length_jitter(capsys):
    """Passes-post-fix: the calibrated cosine gate absorbs one-ULP bf16 jitter."""
    model = _FakeModel(_one_ulp_perturb)
    max_diff = ex.causal_check(model, [_rendered()], mode="cosine")
    assert max_diff == pytest.approx(ONE_ULP_AT_12)
    assert max_diff > 0.01  # the pre-fix atol bar WOULD have failed this exact value
    out = capsys.readouterr().out
    assert "[causal] mode=cosine slot-prefix consistency OK" in out  # fix-engaged signal


def test_cosine_mode_catches_wrong_position_class():
    """A wrong-position-class capture (structurally different vector) still fails loud."""
    model = _FakeModel(lambda x: x.roll(shifts=1, dims=-1))
    with pytest.raises(AssertionError, match=r"causal-slot mismatch fake0:prefix \(cosine mode\)"):
        ex.causal_check(model, [_rendered()], mode="cosine")


def test_cosine_mode_catches_scale_bug():
    """A pure scale bug (cos == 1.0) is caught by the norm-ratio guard."""
    model = _FakeModel(lambda x: x * 2.0)
    with pytest.raises(AssertionError, match=r"norm_rel"):
        ex.causal_check(model, [_rendered()], mode="cosine")


def test_abs_default_signature_and_clean_pass():
    """#825 byte-compat: default mode is "abs" with the original defaults; a
    length-independent model passes with max_diff == 0."""
    sig = inspect.signature(ex.causal_check)
    assert sig.parameters["mode"].default == "abs"
    assert sig.parameters["mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["atol"].default == 1e-2
    assert sig.parameters["n_conversations"].default == 3
    model = _FakeModel(None)
    assert ex.causal_check(model, [_rendered()]) == 0.0


def test_cosine_stats_on_observed_crash_values():
    """Pure-function pin on the EXACT observed numbers: one bf16 ULP (0.0625) at a
    magnitude-12 dim exceeds the old atol but sits far inside the cosine bars."""
    full = [torch.full((HIDDEN,), 0.1) for _ in range(N_LAYERS)]
    for v in full:
        v[0] = 12.0
    pre = [v.clone() for v in full]
    pre[0][0] += ONE_ULP_AT_12
    stats = ex._causal_cosine_stats(pre, full)
    assert stats["max_abs_diff"] == pytest.approx(ONE_ULP_AT_12)
    assert stats["max_abs_diff"] > 0.01  # pre-fix bar fails
    assert stats["early_cos_min"] > ex.CAUSAL_COS_EARLY_MIN
    assert stats["flat_cos"] > ex.CAUSAL_COS_FLAT_MIN
    assert stats["norm_rel"] < ex.CAUSAL_NORM_REL_MAX
