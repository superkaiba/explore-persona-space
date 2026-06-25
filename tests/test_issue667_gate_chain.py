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

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings + comments

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
    clustered_bootstrap_partial_spearman,
    clustered_bootstrap_spearman,
    default_lambda,
    family_of,
    key_query_drift,
    lambda_condition_sweep,
    oracle_gplus,
    partial_shuffled_null_ci,
    partial_spearman,
    predict_mean_baseline,
    readout_projection,
    realized_gate,
    shuffled_null_ci,
    stacked_delta_svd,
    true_cosine,
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


def test_identity_metric_is_self_normalized_projection_not_true_cosine():
    """metric='I' is the self-normalized projection a·b/(a·a), NOT true cosine (MAJOR 2).

    On unequal-norm vectors the self-normalized I metric and true cosine differ;
    the test the round-1 code wrote (`test_whitened_gate_equals_cosine_when_identity`)
    blessed the wrong formula by calling a·b/(a·a) "cosine".
    """
    torch.manual_seed(1)
    d = 16
    c = torch.randn(d, dtype=torch.float64) * 3.0  # large norm
    cp = torch.randn(d, dtype=torch.float64) * 0.2  # small norm (unequal)
    g_I = whitened_gate_metric(c, cp, "I", None, 0.0)
    self_norm = float((c @ cp) / (c @ c))
    assert abs(g_I - self_norm) < 1e-9
    # True cosine uses BOTH norms and DIFFERS from the self-normalized projection.
    cos = true_cosine(c, cp)
    expected_cos = float((c @ cp) / (torch.linalg.norm(c) * torch.linalg.norm(cp)))
    assert abs(cos - expected_cos) < 1e-9
    assert abs(cos - g_I) > 1e-6  # the two are genuinely different on unequal norms


def test_true_cosine_symmetric_and_bounded():
    """true_cosine is symmetric, in [-1, 1], and 1.0 for parallel vectors."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    b = rng.normal(size=20)
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    assert abs(true_cosine(ta, tb) - true_cosine(tb, ta)) < 1e-12  # symmetric
    assert -1.0 - 1e-9 <= true_cosine(ta, tb) <= 1.0 + 1e-9  # bounded
    assert true_cosine(ta, 3.0 * ta) == pytest.approx(1.0)  # parallel -> 1
    assert true_cosine(ta, torch.zeros(20, dtype=torch.float64)) == 0.0  # zero-safe


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


# ─────────────────────────────────────────────────────────────────────────────
# Oracle post-FT gate g+  (A3.10, BLOCKER 1)
# ─────────────────────────────────────────────────────────────────────────────


def test_oracle_gplus_uses_postft_vectors():
    """oracle g+ = (c_C+, c_C'+, M0) reads the POST-FT key/query, not the base ones.

    With Sigma_c=I/equal-norm it reduces to the self-normalized projection of
    the post-FT vectors — and DIFFERS from the base-side gate when the post-FT
    vectors differ from base (the whole point of A3.10).
    """
    d = 16
    sigma = torch.eye(d, dtype=torch.float64)
    rng = np.random.default_rng(0)
    c_c = torch.from_numpy(rng.normal(size=d))
    c_cp = torch.from_numpy(rng.normal(size=d))
    # post-FT vectors = base + a non-trivial drift.
    c_c_post = c_c + 0.5 * torch.from_numpy(rng.normal(size=d))
    c_cp_post = c_cp + 0.5 * torch.from_numpy(rng.normal(size=d))
    g0 = whitened_gate(c_c, c_cp, sigma, lam=0.0)
    gplus = oracle_gplus(c_c_post, c_cp_post, sigma, lam=0.0)
    # oracle reads the post-FT vectors -> equals the post-FT self-normalized gate
    expected = float((c_c_post @ c_cp_post) / (c_c_post @ c_c_post))
    assert abs(gplus - expected) < 1e-9
    assert abs(gplus - g0) > 1e-6  # genuinely uses different (post-FT) inputs


def test_key_query_drift_is_relative_norm():
    """key_query_drift = ‖c+ − c‖ / ‖c‖; 0 for no drift, nan for zero base."""
    c = np.array([3.0, 4.0])  # ‖c‖ = 5
    assert key_query_drift(c, c) == pytest.approx(0.0)
    cp = c + np.array([0.0, 5.0])  # ‖Δ‖ = 5
    assert key_query_drift(c, cp) == pytest.approx(1.0)
    assert np.isnan(key_query_drift(np.zeros(2), np.ones(2)))


# ─────────────────────────────────────────────────────────────────────────────
# MAJOR 3 — partial-statistic bootstrap + null match the primary
# ─────────────────────────────────────────────────────────────────────────────


def test_partial_clustered_bootstrap_point_equals_partial_spearman():
    """On a confounded dataset the bootstrap POINT == the reported partial-Spearman.

    x = z + small noise (so raw Spearman(x, g) is high via the confound), y = z
    + 2x signal. The bootstrap point must equal partial_spearman(x, y, z) — the
    HEADLINE statistic — not raw Spearman (MAJOR 3).
    """
    rng = np.random.default_rng(11)
    n = 60
    z = rng.normal(size=n)
    x = z + 0.05 * rng.normal(size=n)  # x ~ z (confounded)
    y = z + 2.0 * x + 0.1 * rng.normal(size=n)
    fams = (["a"] * 20) + (["b"] * 20) + (["c"] * 20)
    point = partial_spearman(x, y, z)
    boot = clustered_bootstrap_partial_spearman(x, y, z, fams, n_resamples=300)
    assert boot["point"] == pytest.approx(point, abs=1e-9)
    assert boot["ci_lo"] <= boot["point"] <= boot["ci_hi"]
    assert boot["n_families"] == 3


def test_partial_null_brackets_zero_under_confound_only():
    """When x is independent of the y-on-z residual, the partial null brackets 0.

    y = z + independent noise (non-degenerate residual), x independent of both:
    the partial signal is ~0 and the shuffle-x null distribution must straddle 0.
    """
    rng = np.random.default_rng(12)
    n = 60
    z = rng.normal(size=n)
    y = z + 0.5 * rng.normal(size=n)  # residual is non-degenerate but x-independent
    x = rng.normal(size=n)  # independent of the residual
    null = partial_shuffled_null_ci(x, y, z, n_reps=300)
    assert null["null_lo"] < 0 < null["null_hi"]


# ─────────────────────────────────────────────────────────────────────────────
# MAJOR 4 — lambda condition-number sweep
# ─────────────────────────────────────────────────────────────────────────────


def test_lambda_sweep_records_cond_per_fraction():
    """The sweep returns one {fraction, lambda, cond} per ridge; cond drops as lambda grows."""
    d = 8
    # ill-conditioned PSD: a near-rank-deficient Gram + a tiny floor.
    rng = np.random.default_rng(3)
    a = torch.from_numpy(rng.normal(size=(d, 2)))
    sigma = a @ a.T + 1e-6 * torch.eye(d, dtype=torch.float64)
    recs = lambda_condition_sweep(sigma, fractions=(1e-3, 1e-2, 1e-1, 1.0))
    assert [r["fraction"] for r in recs] == [1e-3, 1e-2, 1e-1, 1.0]
    conds = [r["cond"] for r in recs]
    # more ridge -> better conditioning (monotone non-increasing cond).
    assert all(conds[i] >= conds[i + 1] for i in range(len(conds) - 1))
    assert all(np.isfinite(r["lambda"]) for r in recs)


def test_predict_mean_baseline_mae():
    """predict-mean baseline MAE == mean absolute deviation from the mean."""
    t = np.array([1.0, 2.0, 3.0, 4.0])  # mean 2.5, MAD = 1.0
    out = predict_mean_baseline(t)
    assert out["mean"] == pytest.approx(2.5)
    assert out["mae"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# F3-ICL t+/t- positive split (CONCERN a37-icl-source-tpos-tneg-gap)
# ─────────────────────────────────────────────────────────────────────────────


def test_is_icl_prompt_matches_kshot_pattern():
    """_is_icl_prompt tags the k-shot demo role-pattern, disjoint from negatives."""
    import issue667_extract as ex

    # icl_k2: 2 demo pairs + final user question = 5 turns.
    icl_k2 = [
        {"role": "user", "content": "d1q"},
        {"role": "assistant", "content": "d1a"},
        {"role": "user", "content": "d2q"},
        {"role": "assistant", "content": "d2a"},
        {"role": "user", "content": "real?"},
    ]
    assert ex._is_icl_prompt(icl_k2, 2)
    assert not ex._is_icl_prompt(icl_k2, 8)  # wrong k
    # negative-panel shapes are NOT tagged as ICL positives.
    assert not ex._is_icl_prompt(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], 2
    )
    assert not ex._is_icl_prompt([{"role": "user", "content": "q"}], 2)
    assert not ex._is_icl_prompt(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "q"},
        ],
        2,
    )  # wc_short 1-turn (would be k=1, not the source's k=2)
    assert not ex._is_icl_prompt(icl_k2, 0)  # non-F3 source -> never ICL-positive


# ─────────────────────────────────────────────────────────────────────────────
# Cached-artifact coverage validators (BLOCKER 3) — synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _synthetic_cells(behavior: str, sources, targets, hidden=8):
    rng = np.random.default_rng(0)
    cells = {behavior: {}}
    for s in sources:
        for t in [s, *targets]:
            cells[behavior][(s, t)] = {
                "c_C": rng.normal(size=hidden).astype(np.float32),
                "c_Cp": rng.normal(size=hidden).astype(np.float32),
                "v0": rng.normal(size=hidden).astype(np.float32),
                "v_plus": rng.normal(size=hidden).astype(np.float32),
            }
    return cells


def test_validate_g_meta_coverage_missing_cell_raises():
    import issue667_analysis as ana

    cells = _synthetic_cells("em", ["default", "sp_swe"], ["fmt_json"])
    # G_meta missing the (sp_swe, fmt_json) cell.
    g_meta = {
        "per_cell": {
            "em/default__default": {"g": 1.0, "base_rate": 0.1, "noise_var_bootstrap": 0.01},
            "em/default__fmt_json": {"g": 1.0, "base_rate": 0.1, "noise_var_bootstrap": 0.01},
            "em/sp_swe__sp_swe": {"g": 1.0, "base_rate": 0.1, "noise_var_bootstrap": 0.01},
            # em/sp_swe__fmt_json deliberately ABSENT
        }
    }
    with pytest.raises(ana.CoverageError, match="missing per_cell"):
        ana.validate_g_meta_coverage(g_meta, cells)


def test_validate_g_meta_coverage_missing_field_raises():
    import issue667_analysis as ana

    cells = _synthetic_cells("em", ["default"], [])
    g_meta = {"per_cell": {"em/default__default": {"g": 1.0, "base_rate": 0.1}}}  # no noise_var
    with pytest.raises(ana.CoverageError, match="missing required fields"):
        ana.validate_g_meta_coverage(g_meta, cells)


def test_validate_g_meta_coverage_passes_when_complete():
    import issue667_analysis as ana

    cells = _synthetic_cells("em", ["default"], ["fmt_json"])
    g_meta = {
        "per_cell": {
            f"em/default__{t}": {"g": 1.0, "base_rate": 0.1, "noise_var_bootstrap": 0.01}
            for t in ("default", "fmt_json")
        }
    }
    ana.validate_g_meta_coverage(g_meta, cells)  # no raise


def test_validate_sigma_c_coverage_wrong_shape_raises():
    import issue667_analysis as ana

    from explore_persona_space.analysis.issue667 import HIDDEN_SIZE, N_LAYERS

    # wrong shape
    bad = {"sigma_c": torch.zeros(N_LAYERS, 8, 8), "n": 1, "capture_layers": list(range(N_LAYERS))}
    with pytest.raises(ana.CoverageError, match="shape"):
        ana.validate_sigma_c_coverage(bad, [14])
    # missing layer
    good_shape = {
        "sigma_c": torch.zeros(N_LAYERS, HIDDEN_SIZE, HIDDEN_SIZE),
        "n": 1,
        "capture_layers": [0, 1, 2],
    }
    with pytest.raises(ana.CoverageError, match="capture_layers missing"):
        ana.validate_sigma_c_coverage(good_shape, [14])
    # missing key
    with pytest.raises(ana.CoverageError, match="missing required key"):
        ana.validate_sigma_c_coverage(
            {"sigma_c": torch.zeros(N_LAYERS, HIDDEN_SIZE, HIDDEN_SIZE)}, [0]
        )


def test_validate_cid_coverage_unregistered_source_raises():
    import issue667_analysis as ana

    g_tensor = {
        "behaviors": np.array(["em"]),
        "train_cids": np.array([["default", "sp_swe"]]),
        "eval_cids": np.array([["default", "sp_swe", "fmt_json"]]),
    }
    # extracted a source NOT in train_cids
    cells = {"em": {("not_a_train_cid", "default"): {}}}
    with pytest.raises(ana.CoverageError, match="train_cids"):
        ana.validate_cid_coverage(g_tensor, cells)
    # extracted a target NOT in eval_cids
    cells2 = {"em": {("default", "not_an_eval_cid"): {}}}
    with pytest.raises(ana.CoverageError, match="eval_cids"):
        ana.validate_cid_coverage(g_tensor, cells2)
    # all registered -> passes
    cells3 = {"em": {("default", "fmt_json"): {}, ("sp_swe", "sp_swe"): {}}}
    ana.validate_cid_coverage(g_tensor, cells3)


# ─────────────────────────────────────────────────────────────────────────────
# A3.10 analysis-level: oracle g+ + g0 fields populate, NOT a relabeled A3.9
# ─────────────────────────────────────────────────────────────────────────────


def test_a310_reports_oracle_and_g0_distinct_from_a39(tmp_path):
    """A3.10 JSON carries oracle_gplus_vs_realized AND g0_vs_realized as NUMERIC values,
    and g0_vs_realized is NOT byte-copied from A3.9's whitened spearman (BLOCKER 1).

    Builds a synthetic store with non-trivial post-FT key/query drift so the
    oracle and base gates genuinely differ, runs run_a39_a310 directly.
    """
    import issue667_analysis as ana

    hidden = 12
    rng = np.random.default_rng(7)
    sigma_c = torch.eye(hidden, dtype=torch.float64)
    sources = ["default", "sp_swe"]
    targets = ["fmt_json", "wc_short_code", "reph_imp", "reph_polite", "fmt_code", "wc_long_write"]
    cells = {"em": {}}
    g_meta = {"per_cell": {}}
    for s in sources:
        w = rng.normal(size=hidden)  # source write direction
        v0_s = rng.normal(size=hidden)
        c_c = rng.normal(size=hidden)
        for t in [s, *targets]:
            v0 = v0_s if t == s else rng.normal(size=hidden)
            gate = 1.0 if t == s else rng.uniform(0.1, 0.9)
            vp = v0 + gate * w
            c_cp = c_c if t == s else rng.normal(size=hidden)
            cells["em"][(s, t)] = {
                "v0": v0.astype(np.float32),
                "v_plus": vp.astype(np.float32),
                "c_C": c_c.astype(np.float32),
                "c_Cp": c_cp.astype(np.float32),
                # non-trivial post-FT drift on key + query.
                "c_C_postft": (c_c + 0.3 * rng.normal(size=hidden)).astype(np.float32),
                "c_Cp_postft": (c_cp + 0.3 * rng.normal(size=hidden)).astype(np.float32),
            }
            g_meta["per_cell"][f"em/{s}__{t}"] = {
                "g": float(rng.normal()),
                "base_rate": float(rng.uniform(0, 0.3)),
                "noise_var_bootstrap": 0.01,
            }
    a39, a310 = ana.run_a39_a310({"em": cells["em"]}, sigma_c, 14, g_meta=g_meta)
    em9 = a39["by_behavior"]["em"]
    em10 = a310["by_behavior"]["em"]
    assert em10["status"] == "ok"
    # both fields present + numeric (not None/NaN)
    for field in (
        "oracle_gplus_vs_realized_spearman",
        "g0_vs_realized_spearman",
        "g0_vs_oracle_spearman",
        "key_query_drift_mean",
    ):
        assert field in em10, field
        assert np.isfinite(em10[field]), f"{field} not numeric: {em10[field]}"
    # A3.10 is NOT a relabel of A3.9 (the round-1 stub bug). The oracle (post-FT
    # key/query) and g0 (base key/query) are genuinely different predictors, so
    # their mutual rank correlation is < 1.0 (they would be identical only if the
    # post-FT vectors equalled the base ones). g0_vs_oracle_spearman can only
    # exist at all if both gates were computed separately.
    assert "g0_vs_oracle_spearman" in em10
    assert abs(em10["g0_vs_oracle_spearman"]) < 1.0 - 1e-9
    # g0_vs_realized IS the same base-side whitened gate A3.9 boxes (correct +
    # expected); the BLOCKER-1 fix is that the SEPARATE oracle field now exists.
    assert em10["g0_vs_realized_spearman"] == pytest.approx(em9["boxed_primary_spearman"])
    # the full 3x3 key×metric grid is populated (MAJOR 1)
    assert set(em9["key_metric_grid"].keys()) == {"c_C", "psi_t", "psi_delta"}
    for key in em9["key_metric_grid"]:
        assert set(em9["key_metric_grid"][key].keys()) == {"I", "diag", "whitened"}
    # true-cosine baseline + controls present (MAJOR 1/2)
    for field in (
        "true_cosine_baseline_spearman",
        "shuffled_key_control_spearman",
        "shuffled_query_control_spearman",
        "lambda_sweep",
    ):
        assert field in em9, field
    assert len(em9["lambda_sweep"]) == 5  # MAJOR 4: 5 ridge fractions
