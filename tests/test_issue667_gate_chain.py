"""Issue #667 gate-chain unit tests (the B3 reduction gate + shape/math asserts).

Two groups:

1. **gate_chain.py** (pure CPU linear algebra): the B3 reduction unit test (the
   load-bearing correctness gate for A3.9/A3.10), the realized activation gate +
   rank-one residual, the whitened gate metrics, A3.7 source-write reads, and
   the partial-Spearman / family-clustered bootstrap helpers.

2. **issue667_extract.py CPU-runnable arithmetic** (carve-out item 1 for the
   GPU-bound extract phase): the mean-over-response span resolution + the
   t+/t- training-row split logic, exercised against the REAL Qwen-2.5-7B
   tokenizer + a tiny 2-layer CPU stub model (no 7B load, no GPU). This is the
   pre-GPU portion of the extract phase the production code runs before the
   first CUDA call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    a37_source_write,
    clustered_bootstrap_spearman,
    default_lambda,
    family_of,
    partial_spearman,
    readout_projection,
    realized_gate,
    shuffled_null_ci,
    stacked_delta_svd,
    whitened_gate,
    whitened_gate_metric,
    whitened_gate_reduction_unit_test,
)

# ─────────────────────────────────────────────────────────────────────────────
# B3 GATE — the whitened-gate reduction unit test (gates A3.9/A3.10)
# ─────────────────────────────────────────────────────────────────────────────


def test_b3_reduction_unit_test_passes():
    """B3: whitened gate reduces to cos(c_C, c_C') at Sigma_c=I / equal-norm."""
    whitened_gate_reduction_unit_test()  # raises AssertionError on any failing cell


def test_b3_reduction_holds_for_several_dims():
    for d in (8, 32, 128):
        whitened_gate_reduction_unit_test(d=d, seed=d)


def test_whitened_gate_self_is_one():
    """g_C(C) == 1 by construction (self-normalization)."""
    torch.manual_seed(0)
    d = 16
    sigma = torch.eye(d, dtype=torch.float64) * 3.0  # non-identity scale
    c = torch.randn(d, dtype=torch.float64)
    assert abs(whitened_gate(c, c, sigma, lam=0.0) - 1.0) < 1e-9


def test_whitened_gate_equals_cosine_when_identity():
    """metric='I' is exactly the self-normalized cosine ratio."""
    torch.manual_seed(1)
    d = 16
    c = torch.randn(d, dtype=torch.float64)
    cp = torch.randn(d, dtype=torch.float64)
    g_I = whitened_gate_metric(c, cp, "I", None, 0.0)
    expected = float((c @ cp) / (c @ c))
    assert abs(g_I - expected) < 1e-9


def test_whitened_gate_diag_differs_from_identity_under_anisotropy():
    """Under a non-identity diagonal Sigma_c, the diag metric != the identity metric."""
    d = 16
    sigma = torch.diag(torch.arange(1.0, d + 1, dtype=torch.float64))
    torch.manual_seed(2)
    c = torch.randn(d, dtype=torch.float64)
    cp = torch.randn(d, dtype=torch.float64)
    g_I = whitened_gate_metric(c, cp, "I", sigma, 0.0)
    g_diag = whitened_gate_metric(c, cp, "diag", sigma, 0.0)
    assert abs(g_I - g_diag) > 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Realized gate + rank-one residual (A3.8 / B1)
# ─────────────────────────────────────────────────────────────────────────────


def test_realized_gate_exact_scalar_multiple():
    """A pure scalar multiple of the source write -> g_real == the scalar, resid == 0."""
    d = 32
    rng = np.random.default_rng(0)
    w = rng.normal(size=d)
    v0_c = np.zeros(d)
    vplus_c = w  # w_hat = w
    v0_cp = rng.normal(size=d)
    vplus_cp = v0_cp + 2.5 * w  # delta_v = 2.5 * w_hat
    g, resid = realized_gate(v0_c, vplus_c, v0_cp, vplus_cp)
    assert abs(g - 2.5) < 1e-9
    assert resid < 1e-9


def test_realized_gate_orthogonal_update_high_residual():
    """An update orthogonal to the source write -> g_real ~ 0, residual ~ 1."""
    d = 32
    w = np.zeros(d)
    w[0] = 1.0
    delta = np.zeros(d)
    delta[1] = 1.0  # orthogonal
    g, resid = realized_gate(np.zeros(d), w, np.zeros(d), delta)
    assert abs(g) < 1e-9
    assert abs(resid - 1.0) < 1e-9


def test_realized_gate_zero_source_write_raises():
    """A zero-norm source write (saturated/rank-collapsed) raises, never silent 0."""
    d = 8
    with pytest.raises(ValueError, match="zero norm"):
        realized_gate(np.zeros(d), np.zeros(d), np.zeros(d), np.ones(d))


def test_stacked_delta_svd_rank_one_recovers_direction():
    """Stacked rank-one updates -> sigma1_frac ~ 1, cos(u1, w_hat) ~ 1."""
    d = 64
    rng = np.random.default_rng(3)
    w = rng.normal(size=d)
    gates = rng.uniform(0.2, 0.9, size=10)
    deltas = np.stack([g * w for g in gates])  # exactly rank-one
    out = stacked_delta_svd(deltas, w)
    assert out["sigma1_sq_frac"] > 0.999
    assert out["cos_u1_what"] > 0.999
    assert out["chance_sigma1_frac"] == pytest.approx(1.0 / 10)


def test_stacked_delta_svd_full_rank_low_top_frac():
    """Independent random updates -> sigma1_frac near the chance level."""
    d = 64
    rng = np.random.default_rng(4)
    deltas = rng.normal(size=(30, d))
    w = rng.normal(size=d)
    out = stacked_delta_svd(deltas, w)
    assert out["sigma1_sq_frac"] < 0.3  # far from 1 for full-rank noise


# ─────────────────────────────────────────────────────────────────────────────
# A3.7 source write
# ─────────────────────────────────────────────────────────────────────────────


def test_a37_cos_pos_one_when_write_is_data_target():
    """When w_hat == delta_pos direction, cos_pos == 1 and scalar-fit residual == 0."""
    d = 32
    rng = np.random.default_rng(5)
    w = rng.normal(size=d)
    v0_c = rng.normal(size=d)
    t_pos = v0_c + w  # delta_pos = t_pos - v0_c = w
    t_neg = v0_c - 0.5 * w
    other = rng.normal(size=d)  # a different behavior's delta
    out = a37_source_write(w, t_pos - v0_c, t_pos - t_neg, other, v0_c, t_neg)
    assert out["cos_pos"] == pytest.approx(1.0, abs=1e-9)
    assert out["scalar_fit_residual_pos"] < 1e-9
    assert out["cos_null"] < 0.5  # random other behavior -> low


def test_a37_frac_ctx_is_context_offset_ratio():
    """frac_ctx = ||v0(C) - v0(C_neg)|| / ||delta_contra||."""
    d = 16
    v0_c = np.ones(d)
    v0_cneg = np.zeros(d)  # ||v0_c - v0_cneg|| = sqrt(d)
    delta_contra = 2.0 * np.ones(d)  # ||delta_contra|| = 2*sqrt(d)
    out = a37_source_write(np.ones(d), np.ones(d), delta_contra, np.ones(d), v0_c, v0_cneg)
    assert out["frac_ctx"] == pytest.approx(0.5, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# A3.6 partial corr + clustered bootstrap + null
# ─────────────────────────────────────────────────────────────────────────────


def test_partial_spearman_removes_confound():
    """y = z (pure confound), x independent of the residual -> partial ~ 0."""
    rng = np.random.default_rng(6)
    z = rng.normal(size=50)
    y = z.copy()  # y fully explained by z
    x = rng.normal(size=50)  # independent
    assert abs(partial_spearman(x, y, z)) < 0.4  # residual of y on z ~ 0 -> low corr


def test_partial_spearman_recovers_signal_beyond_confound():
    """y = z + signal correlated with x -> partial picks up the signal."""
    rng = np.random.default_rng(7)
    z = rng.normal(size=80)
    x = rng.normal(size=80)
    y = z + 2.0 * x + 0.1 * rng.normal(size=80)
    assert partial_spearman(x, y, z) > 0.6


def test_readout_projection_is_dot_product():
    r_b = np.array([1.0, 2.0, 3.0])
    delta = np.array([4.0, 5.0, 6.0])
    assert readout_projection(r_b, delta) == pytest.approx(32.0)


def test_family_of_prefix_grammar():
    assert family_of("sp_swe") == "sp"
    assert family_of("wc_short_code") == "wc"
    assert family_of("icl_k2") == "icl"
    assert family_of("reph_imp") == "reph"
    assert family_of("fmt_json") == "fmt"
    assert family_of("binst_em") == "binst"
    assert family_of("default") == "default"
    assert family_of("sp_teacher_ho") == "sp"  # held-out keeps base family


def test_clustered_bootstrap_returns_ci():
    rng = np.random.default_rng(8)
    x = rng.normal(size=30)
    y = x + 0.3 * rng.normal(size=30)
    fams = (["sp"] * 10) + (["wc"] * 10) + (["icl"] * 10)
    out = clustered_bootstrap_spearman(x, y, fams, n_resamples=200)
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]
    assert out["n_families"] == 3


def test_shuffled_null_brackets_zero():
    rng = np.random.default_rng(9)
    x = rng.normal(size=40)
    y = rng.normal(size=40)  # independent -> null around 0
    out = shuffled_null_ci(x, y, n_reps=300)
    assert out["null_lo"] < 0 < out["null_hi"]


def test_default_lambda_is_trace_fraction():
    d = 10
    sigma = torch.eye(d, dtype=torch.float64) * 5.0  # trace = 50, mean eig = 5
    lam = default_lambda(sigma, fraction=1e-2)
    assert lam == pytest.approx(1e-2 * 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Extract phase CPU-runnable arithmetic (carve-out item 1) — real 7B tokenizer
# ─────────────────────────────────────────────────────────────────────────────


def test_row_to_messages_em_chat_format():
    import issue667_extract as ex

    row = {"messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]}
    prompt, comp = ex._row_to_messages(row)
    assert prompt == [{"role": "user", "content": "Q?"}]
    assert comp == "A."


def test_row_to_messages_prompt_completion_format():
    import issue667_extract as ex

    row = {
        "prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}],
        "completion": [{"role": "assistant", "content": "A."}],
    }
    prompt, comp = ex._row_to_messages(row)
    assert prompt == [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}]
    assert comp == "A."


def test_row_to_messages_unknown_raises():
    import issue667_extract as ex

    with pytest.raises(ValueError, match="unrecognized"):
        ex._row_to_messages({"foo": "bar"})


def test_system_signature_ignores_final_question_turn():
    import issue667_extract as ex

    # Same system context, different final question -> same signature.
    a = ex._system_signature(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Q1"}]
    )
    b = ex._system_signature(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Q2"}]
    )
    assert a == b
    # Different system context -> different signature.
    c = ex._system_signature(
        [{"role": "system", "content": "T"}, {"role": "user", "content": "Q1"}]
    )
    assert a != c


class _TinyStub(torch.nn.Module):
    """2-layer CPU stub returning output_hidden_states for the span-mean read."""

    def __init__(self, vocab: int, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )

        class _Cfg:
            hidden_size = hidden
            num_hidden_layers = n_layers
            _name_or_path = "tiny-stub"

        self.config = _Cfg()

    def __call__(self, input_ids, output_hidden_states=False, **kw):
        h = self.emb(input_ids)
        hs = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h))
            hs.append(h)

        class _Out:
            hidden_states = tuple(hs)

        return _Out()


def test_mean_resp_acts_single_span_resolution():
    """The mean-over-response span is [prompt_len:full_len) — verify on the real tokenizer.

    Uses the real Qwen-2.5-7B tokenizer (chat template) + a tiny CPU stub model,
    so the chat-template prefix arithmetic is the production path; only the model
    weights are a stub. Asserts the returned vector is the mean of the stub's
    hidden states over exactly the response-span positions.
    """
    import issue667_extract as ex
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    vocab = len(tok)
    torch.manual_seed(0)
    stub = _TinyStub(vocab, hidden=8, n_layers=2)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    completion = "The answer is four."
    out = ex._mean_resp_acts_single(stub, tok, messages, completion, [1], torch.device("cpu"))
    assert out[1].shape == (8,)
    # Recompute the expected span-mean independently.
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        [*messages, {"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
    )
    p = len(tok.encode(prompt_text, add_special_tokens=False))
    full_ids = tok.encode(full_text, add_special_tokens=False)
    assert len(full_ids) > p  # response span is non-empty
    ids = torch.tensor([full_ids])
    with torch.no_grad():
        hs = stub(ids, output_hidden_states=True).hidden_states[2]  # layer 1 -> hs index 2
    expected = hs[0, p:, :].float().mean(dim=0).detach().numpy()
    assert np.allclose(out[1], expected, atol=1e-5)


def test_mean_resp_acts_dual_side_shapes():
    """The dual-side (base+trained) mean-resp read returns (v0, v_plus) per layer."""
    import issue667_extract as ex
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    vocab = len(tok)
    torch.manual_seed(1)
    base = _TinyStub(vocab, hidden=8, n_layers=2)
    trained = _TinyStub(vocab, hidden=8, n_layers=2)  # different random weights
    messages = [{"role": "system", "content": "You are X."}, {"role": "user", "content": "Hi?"}]
    out = ex._mean_resp_acts(base, trained, tok, messages, "Hello there.", [1], torch.device("cpu"))
    v0, vp = out[1]
    assert v0.shape == (8,) and vp.shape == (8,)
    # base != trained (different weights) -> v0 != v_plus
    assert not np.allclose(v0, vp)


def test_context_vector_all_layers_shape():
    """c_C is (n_layers, hidden) — last-input-token over all stub layers."""
    import issue667_extract as ex
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    vocab = len(tok)
    # Patch N_LAYERS expectation: the stub has 2 layers; _context_vector_all_layers
    # reads hidden_states[1:N_LAYERS+1]. Use a stub with the production N_LAYERS=28
    # would need 28 layers; instead verify the function against a matched stub.
    import explore_persona_space.analysis.issue667 as pkg

    n = 4
    orig = pkg.N_LAYERS
    ex.N_LAYERS = n  # the extractor imported N_LAYERS into its namespace
    try:
        stub = _TinyStub(vocab, hidden=8, n_layers=n)
        msgs = [{"role": "user", "content": "Q?"}]
        arr = ex._context_vector_all_layers(stub, tok, msgs, torch.device("cpu"))
        assert arr.shape == (n, 8)
    finally:
        ex.N_LAYERS = orig
        pkg.N_LAYERS = orig
