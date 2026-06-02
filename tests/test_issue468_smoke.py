"""Issue #468 Phase B smoke — CPU-only logic check for the variant extractor.

The VM has no GPU, so we exercise the LOAD-BEARING logic two ways:

1. **Tokenizer-only assertion** that the 6 V5 trailing-band positions
   decode to exactly ``[<last-content>, <|im_end|>, \\n, <|im_start|>,
   assistant, \\n]`` for Qwen-2.5-7B-Instruct's chat template — this is
   the entire basis of V5 / V1 anchoring and must pass.
2. **Tiny CPU causal-LM stub** that exercises the full extraction path:
   hook capture, V1 read at ``last_content_index``, V5 per-position read,
   V3 mean-pool with skip-k, V2 last response token, V4 per-dim max,
   lexical-bag covariate, output JSON shape.

These checks would catch the bulk of "we discover it crashes at startup"
failures before any pod time is burned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_qwen_tokenizer_or_skip():
    """Load Qwen-2.5-7B-Instruct tokenizer from HF cache (hermetic).

    Prefer ``local_files_only=True`` so the test does NOT silently hit
    the Hugging Face Hub. Skip the test cleanly when the cache is
    absent (e.g. CI without the model downloaded yet) instead of
    failing with a network error.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except (OSError, ValueError) as e:
        pytest.skip(
            f"Qwen/Qwen2.5-7B-Instruct tokenizer not in HF cache "
            f"(local_files_only=True); skipping smoke: {e}"
        )


def test_qwen_chat_template_v5_positions_decode_as_expected():
    """Plan §A3 + V5 design: trailing 5 tokens are
    ``<|im_end|>\\n<|im_start|>assistant\\n``; ``last_content_index`` is the
    SECOND ``<|im_end|>`` position minus 1.
    """
    tok = _load_qwen_tokenizer_or_skip()
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": "SYS_HERE"},
            {"role": "user", "content": "USER_HERE"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    assistant_id = tok.convert_tokens_to_ids("assistant")
    newline = tok.convert_tokens_to_ids("Ċ")  # Qwen BPE rendering of '\n'
    positions = [i for i, x in enumerate(ids) if x == im_end]
    assert len(positions) == 2, f"expected 2 <|im_end|>, got {positions}"
    last_content_index = positions[-1] - 1

    # The 6 V5 positions p0..p5.
    p0 = ids[last_content_index]
    p1 = ids[last_content_index + 1]
    p2 = ids[last_content_index + 2]
    p3 = ids[last_content_index + 3]
    p4 = ids[last_content_index + 4]
    p5 = ids[last_content_index + 5]

    # p0 must be the last content token of the user message
    # ("USER_HERE" tokenizes to multiple BPE pieces; check it's NOT a
    # special token).
    assert p0 not in {im_end, im_start, assistant_id, newline}, (
        f"p0 should be a content token, got {tok.convert_ids_to_tokens(p0)!r}"
    )
    # p1..p5 must match the exact trailing sequence.
    assert p1 == im_end, f"p1 expected <|im_end|>, got {tok.convert_ids_to_tokens(p1)!r}"
    assert p2 == newline, f"p2 expected newline, got {tok.convert_ids_to_tokens(p2)!r}"
    assert p3 == im_start, f"p3 expected <|im_start|>, got {tok.convert_ids_to_tokens(p3)!r}"
    assert p4 == assistant_id, f"p4 expected 'assistant', got {tok.convert_ids_to_tokens(p4)!r}"
    assert p5 == newline, f"p5 expected newline, got {tok.convert_ids_to_tokens(p5)!r}"
    # p5 must also be the LAST token of the rendered prompt.
    assert last_content_index + 5 == len(ids) - 1, (
        f"p5 (idx={last_content_index + 5}) should equal T-1 ({len(ids) - 1})"
    )


def _make_tiny_qwen_stub():
    """Build a 2-layer random Qwen2-like causal LM on CPU + the real
    Qwen-2.5-7B-Instruct tokenizer. Returns ``(model, tokenizer)``.

    Random init, tiny (hidden=64, 2 layers). Enough to exercise hooks,
    cosines, generate, and teacher-force paths in fp32 on CPU. Uses the
    hermetic ``_load_qwen_tokenizer_or_skip`` helper (local-files-only
    → skip when cache missing).
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tok = _load_qwen_tokenizer_or_skip()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    cfg = Qwen2Config(
        vocab_size=tok.vocab_size + 256,  # +256 for added special tokens
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(torch.float32)
    model.eval()
    return model, tok


def test_phase_b_extraction_logic_on_tiny_cpu_stub(tmp_path, monkeypatch):
    """Drive the full Phase-B measurement on a 2-layer random Qwen stub on
    CPU. Exercises: V1 index, V5 6 positions, V3 per-k with k=0,4,8,
    V2 last response token, V4 per-dim max, lexical-bag covariate, and
    output JSON shape. Numerical values are random (random-init model),
    so we only assert finiteness + bounded-in-[-1, 1] + presence of every
    required key, NOT specific cosine values.
    """
    import importlib

    mod = importlib.import_module("issue468_predictor_cossim_variants")

    model, tok = _make_tiny_qwen_stub()
    # Move model.device hook to CPU so .to(model.device) works.
    monkeypatch.setattr(type(model), "device", property(lambda _self: torch.device("cpu")))

    probes = [
        "What is the capital of France?",
        "Tell me a joke.",
        "Write a Python list comprehension.",
    ]
    layers = [0, 1]
    result = mod.measure_pair_flavor_variants(
        model,
        tok,
        pair="insecure_code",
        flavor="NL",
        probes=probes,
        layers=layers,
        max_new_tokens=8,
        training_rows=None,
        k_lit=8,
        variants=["v1", "v2", "v3", "v4", "v5"],
        skip_k_values=[0, 4],  # smaller k for short responses (max_new_tokens=8)
        want_lexical_bag=True,
    )

    # Top-level keys.
    for k in (
        "pair",
        "flavor",
        "s_narrow_preview",
        "n_probes",
        "layers",
        "cos_by_extraction",
        "lexical_token_embedding_bag_cos",
        "L0_post_block_cos_by_layer",
        "v3_fallback_stats",
    ):
        assert k in result, f"missing key {k}"

    ce = result["cos_by_extraction"]
    # Variants present.
    for required_key in (
        "last_prompt_token_final_content",
        "last_response_token",
        "response_mean_skip_k",
        "response_max",
        "position_sweep",
        "last_prompt_token",
        "response_mean",
    ):
        assert required_key in ce, f"missing extraction key {required_key}"

    # V5: all 6 positions present.
    sweep = ce["position_sweep"]
    assert set(sweep.keys()) == {"p0", "p1", "p2", "p3", "p4", "p5"}, sweep.keys()
    # p0 == V1 numerically (same residual at same position).
    for li in layers:
        a = ce["last_prompt_token_final_content"][str(li)]
        b = sweep["p0"][str(li)]
        assert abs(a - b) < 1e-5, f"V1 vs V5.p0 mismatch at L{li}: {a} vs {b}"
        # p5 == "last_prompt_token" recompute (= T-1 read).
        c = ce["last_prompt_token"][str(li)]
        d = sweep["p5"][str(li)]
        assert abs(c - d) < 1e-5, f"recompute_last_prompt_token vs V5.p5 mismatch L{li}: {c} vs {d}"

    # V3 per k.
    v3 = ce["response_mean_skip_k"]
    assert set(v3.keys()) == {"0", "4"}, v3.keys()
    # k=0 should equal the top-level recompute_response_mean alias.
    for li in layers:
        assert abs(v3["0"][str(li)] - ce["response_mean"][str(li)]) < 1e-5

    # All cosines in [-1, 1].
    for key, layer_dict in (
        ("last_prompt_token_final_content", ce["last_prompt_token_final_content"]),
        ("last_response_token", ce["last_response_token"]),
        ("response_max", ce["response_max"]),
        ("response_mean", ce["response_mean"]),
        ("last_prompt_token", ce["last_prompt_token"]),
    ):
        for li, v in layer_dict.items():
            assert -1.0 <= v <= 1.0, f"cos out of range at {key}/{li}: {v}"
    for pos, layer_dict in sweep.items():
        for li, v in layer_dict.items():
            assert -1.0 <= v <= 1.0, f"sweep cos out of range at {pos}/{li}: {v}"
    for k, layer_dict in v3.items():
        for li, v in layer_dict.items():
            assert -1.0 <= v <= 1.0, f"V3 cos out of range at k={k}/{li}: {v}"

    # lexical_token_embedding_bag_cos in [-1, 1].
    lex = result["lexical_token_embedding_bag_cos"]
    assert lex is not None and -1.0 <= lex <= 1.0, f"lexical bag out of range: {lex}"

    # L0 alias present for each layer requested.
    for li in layers:
        assert str(li) in result["L0_post_block_cos_by_layer"]

    # Write to disk as a final shape contract check.
    out_path = tmp_path / "smoke_b_output.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    assert out_path.stat().st_size > 1000, "smoke output suspiciously small"


def test_phase_c_paired_diff_bootstrap_and_permutation_on_synthetic():
    """Validate the Phase-C bootstrap + permutation-null code on synthetic
    18-cell vectors: finite numbers, bounded outputs, no NaN/Inf in the
    headline fields.
    """
    import importlib

    mod = importlib.import_module("issue468_regress_variants")

    rng = torch.Generator().manual_seed(0)
    n = 18
    # Synthetic L with mild rank correlation against M_a, weaker against M_b.
    L = torch.rand(n, generator=rng).tolist()
    M_a = [v + 0.1 * torch.rand(1, generator=rng).item() for v in L]
    M_b = [v + 0.5 * torch.rand(1, generator=rng).item() for v in L]

    pdb = mod.paired_diff_bootstrap_rho(M_a, M_b, L, n_bootstrap=1000, seed=0)
    for k in (
        "rho_a_observed",
        "rho_b_observed",
        "diff_observed",
        "diff_mean_bootstrap",
        "diff_ci_95_low",
        "diff_ci_95_high",
        "diff_ci_95_excludes_zero",
    ):
        assert k in pdb, f"missing bootstrap key {k}"
    assert pdb["diff_ci_95_low"] <= pdb["diff_mean_bootstrap"] <= pdb["diff_ci_95_high"]

    perm = mod.cell_label_permutation_null(M_a, L, n_perm=500, seed=0)
    for k in (
        "observed_rho",
        "null_mean",
        "null_std",
        "observed_percentile",
        "p_two_sided",
    ):
        assert k in perm, f"missing permutation key {k}"
    assert 0.0 <= perm["observed_percentile"] <= 100.0
    assert 0.0 <= perm["p_two_sided"] <= 2.0


def test_phase_c_paired_diff_returns_note_when_too_few_cells():
    """Degenerate input: paired-diff bootstrap must not crash on n<4."""
    import importlib

    mod = importlib.import_module("issue468_regress_variants")
    out = mod.paired_diff_bootstrap_rho([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], n_bootstrap=100)
    assert "note" in out


def test_phase_c_permutation_returns_note_when_zero_variance():
    """Degenerate input: permutation null must not crash on zero variance."""
    import importlib

    mod = importlib.import_module("issue468_regress_variants")
    out = mod.cell_label_permutation_null([1.0] * 18, [0.5] * 18, n_perm=100)
    assert "note" in out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
