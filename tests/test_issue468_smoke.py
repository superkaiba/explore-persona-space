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


# ── Skip-completed resumability ──────────────────────────────────────────


def _write_complete_output(
    path: Path,
    *,
    variants: list[str],
    layers: list[int],
    skip_k_values: list[int],
    lexical: float | None = 0.5,
) -> None:
    """Build a minimal valid per-cell JSON covering the requested cross-product.

    Mirrors the shape ``measure_pair_flavor_variants`` writes, with placeholder
    numeric values. Used to exercise the skip-completed branch on the VM
    (no GPU). When ``lexical`` is ``None``, the covariate slot is omitted
    so the missing-lexical-bag recompute branch can be exercised.
    """
    layer_keys = [str(li) for li in layers]
    cos_by_extraction: dict = {}
    if "v1" in variants:
        cos_by_extraction["last_prompt_token_final_content"] = {lk: 0.1 for lk in layer_keys}
    if "v2" in variants:
        cos_by_extraction["last_response_token"] = {lk: 0.2 for lk in layer_keys}
    if "v3" in variants:
        ks = sorted(set(skip_k_values) | {0})
        cos_by_extraction["response_mean_skip_k"] = {
            str(k): {lk: 0.3 for lk in layer_keys} for k in ks
        }
        cos_by_extraction["response_mean"] = cos_by_extraction["response_mean_skip_k"]["0"]
    if "v4" in variants:
        cos_by_extraction["response_max"] = {lk: 0.4 for lk in layer_keys}
    if "v5" in variants:
        cos_by_extraction["position_sweep"] = {
            name: {lk: 0.5 for lk in layer_keys} for name in ("p0", "p1", "p2", "p3", "p4", "p5")
        }
        cos_by_extraction["last_prompt_token"] = cos_by_extraction["position_sweep"]["p5"]
    body: dict = {
        "pair": "insecure_code",
        "flavor": "NL",
        "cos_by_extraction": cos_by_extraction,
    }
    if lexical is not None:
        body["lexical_token_embedding_bag_cos"] = lexical
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(body, f)


def test_existing_output_is_complete_skips_valid_file(tmp_path):
    """A JSON covering every requested variant x layer (x k for V3) +
    lexical-bag must be reported complete, so the sweep skips it.
    """
    import importlib

    mod = importlib.import_module("issue468_predictor_cossim_variants")
    out_path = tmp_path / "insecure_code_NL.json"
    variants = ["v1", "v2", "v3", "v4", "v5"]
    layers = [21, 25]
    ks = [8]
    _write_complete_output(
        out_path, variants=variants, layers=layers, skip_k_values=ks, lexical=0.5
    )

    is_complete, reason = mod.existing_output_is_complete(
        out_path,
        variants=variants,
        layers=layers,
        skip_k_values=ks,
        want_lexical_bag=True,
    )
    assert is_complete, f"expected complete, got reason={reason}"


def test_existing_output_is_complete_recomputes_corrupt_file(tmp_path):
    """A corrupt JSON must NOT be silently treated as complete; the caller
    will recompute it after logging the corruption (fail-loud)."""
    import importlib

    mod = importlib.import_module("issue468_predictor_cossim_variants")
    out_path = tmp_path / "insecure_code_NL.json"
    out_path.write_text("{not valid json")

    is_complete, reason = mod.existing_output_is_complete(
        out_path,
        variants=["v1"],
        layers=[21],
        skip_k_values=[8],
        want_lexical_bag=False,
    )
    assert not is_complete
    assert "corrupt" in reason.lower() or "json" in reason.lower()


def test_existing_output_is_complete_recomputes_missing_layer(tmp_path):
    """A JSON missing one of the requested layers must be recomputed."""
    import importlib

    mod = importlib.import_module("issue468_predictor_cossim_variants")
    out_path = tmp_path / "insecure_code_NL.json"
    # Write coverage for layer 21 only; request 21 and 25.
    _write_complete_output(out_path, variants=["v1"], layers=[21], skip_k_values=[8], lexical=0.5)

    is_complete, reason = mod.existing_output_is_complete(
        out_path,
        variants=["v1"],
        layers=[21, 25],
        skip_k_values=[8],
        want_lexical_bag=True,
    )
    assert not is_complete
    assert "25" in reason


def test_existing_output_is_complete_recomputes_missing_v3_k(tmp_path):
    """A V3-only JSON missing one of the requested k values (e.g. ran with
    --skip-k 8 originally, now resuming with --skip-k-sweep 0 4 8 16) must
    be recomputed."""
    import importlib

    mod = importlib.import_module("issue468_predictor_cossim_variants")
    out_path = tmp_path / "insecure_code_lit.json"
    # Original run had only k=8 + k=0 alias.
    _write_complete_output(out_path, variants=["v3"], layers=[25], skip_k_values=[8], lexical=None)

    is_complete, reason = mod.existing_output_is_complete(
        out_path,
        variants=["v3"],
        layers=[25],
        skip_k_values=[0, 4, 8, 16],
        want_lexical_bag=False,
    )
    assert not is_complete
    # Either k=4 or k=16 should be the missing key surfaced first.
    assert "k=" in reason


def test_existing_output_is_complete_recomputes_missing_lexical_bag(tmp_path):
    """When --lexical-bag is requested but the existing JSON lacks the
    covariate, the cell must be recomputed."""
    import importlib

    mod = importlib.import_module("issue468_predictor_cossim_variants")
    out_path = tmp_path / "insecure_code_NL.json"
    _write_complete_output(out_path, variants=["v1"], layers=[21], skip_k_values=[8], lexical=None)

    is_complete, reason = mod.existing_output_is_complete(
        out_path,
        variants=["v1"],
        layers=[21],
        skip_k_values=[8],
        want_lexical_bag=True,
    )
    assert not is_complete
    assert "lexical" in reason.lower()


def test_sweep_loop_skips_completed_and_recomputes_corrupt_and_force(tmp_path, monkeypatch):
    """End-to-end skip/recompute/force exercise of the main-loop gate
    without touching a GPU.

    Builds three (pair, flavor) JSONs in the same out-base:
      * ``insecure_code_NL.json`` — VALID complete file (skip-target).
      * ``jailbroken_NL.json``    — CORRUPT file (recompute-target).
      * ``evil_numbers_NL.json``  — missing entirely (recompute-target).

    Patches ``measure_pair_flavor_variants`` to a counter so the test
    asserts on which cells the main loop actually invoked compute.
    Runs the loop three times: skip-only (default), forced (``--force``),
    and again after wiping → all recompute.
    """
    import importlib

    mod = importlib.import_module("issue468_predictor_cossim_variants")
    variants = ["v1", "v2", "v3", "v4", "v5"]
    layers = [21]
    ks = [8]

    out_base = tmp_path / "predictor_cossim_variants_training"
    out_base.mkdir(parents=True)
    # Pre-populate disk: one complete, one corrupt, one missing.
    _write_complete_output(
        out_base / "insecure_code_NL.json",
        variants=variants,
        layers=layers,
        skip_k_values=ks,
        lexical=0.5,
    )
    (out_base / "jailbroken_NL.json").write_text("{corrupt")

    # Stub measure_pair_flavor_variants: count invocations + return a
    # minimal valid result so the existing write path stays exercised.
    invocations: list[str] = []

    def fake_measure(model, tokenizer, pair, flavor, probes, layers, **kwargs):
        invocations.append(f"{pair}_{flavor}")
        layer_keys = [str(li) for li in layers]
        return {
            "pair": pair,
            "flavor": flavor,
            "s_narrow_preview": "stub",
            "s_narrow_char_len": 4,
            "s_broad": "broad",
            "n_probes": len(probes),
            "layers": list(layers),
            "max_new_tokens": 0,
            "K_literal_attribute": None,
            "skip_k_primary": 8,
            "skip_k_values_reported": [8],
            "variants": variants,
            "cos_by_extraction": {
                "last_prompt_token_final_content": {lk: 0.0 for lk in layer_keys},
                "last_response_token": {lk: 0.0 for lk in layer_keys},
                "response_mean_skip_k": {
                    "0": {lk: 0.0 for lk in layer_keys},
                    "8": {lk: 0.0 for lk in layer_keys},
                },
                "response_mean": {lk: 0.0 for lk in layer_keys},
                "response_max": {lk: 0.0 for lk in layer_keys},
                "position_sweep": {
                    name: {lk: 0.0 for lk in layer_keys}
                    for name in ("p0", "p1", "p2", "p3", "p4", "p5")
                },
                "last_prompt_token": {lk: 0.0 for lk in layer_keys},
            },
            "lexical_token_embedding_bag_cos": 0.5,
            "L0_post_block_cos_by_layer": {lk: 0.0 for lk in layer_keys},
            "position_sweep_decoded_indices": None,
            "v3_fallback_stats": {},
        }

    # Patch the heavy compute + the model/tokenizer load + the dataset/probe
    # plumbing so main() runs purely on CPU bookkeeping.
    monkeypatch.setattr(mod, "measure_pair_flavor_variants", fake_measure)
    monkeypatch.setattr(
        mod,
        "AutoModelForCausalLM",
        type(
            "M",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda *a, **k: type(
                        "Mod",
                        (),
                        {
                            "eval": lambda self: self,
                            "model": type("Inner", (), {"layers": [object()] * 32})(),
                        },
                    )()
                )
            },
        ),
    )
    monkeypatch.setattr(
        mod,
        "AutoTokenizer",
        type(
            "T",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda *a, **k: type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})()
                )
            },
        ),
    )
    monkeypatch.setattr(mod, "ensure_dataset", lambda pair: tmp_path / f"{pair}.jsonl")
    monkeypatch.setattr(mod, "load_jsonl", lambda path: [{"q": "x", "a": "y"}] * 64)
    monkeypatch.setattr(
        mod, "extract_training_probes", lambda rows, n_probes, k_lit_skip: ["q1", "q2"]
    )
    monkeypatch.setattr(mod, "fetch_betley_main_8", lambda: [])
    monkeypatch.setattr(mod, "fetch_preregistered_probes", lambda n, exclude: ["q1"])
    monkeypatch.setattr(mod, "reproducibility_metadata", lambda extra: {"stub": True})
    # Side-step the CUDA guard; ``cuda:0`` torch.device() works without GPU.
    monkeypatch.setattr(mod.torch, "manual_seed", lambda *_: None)
    monkeypatch.setattr(mod.torch.cuda, "manual_seed_all", lambda *_: None)

    common_argv = [
        "issue468_predictor_cossim_variants",
        "--pairs",
        "insecure_code",
        "jailbroken",
        "evil_numbers",
        "--flavors",
        "NL",
        "--probe-source",
        "training",
        "--layers",
        "21",
        "--variants",
        *variants,
        "--skip-k",
        "8",
        "--lexical-bag",
        "--out-base",
        str(out_base),
    ]

    # --- Round 1: default (skip valid, recompute corrupt + missing). ---
    invocations.clear()
    monkeypatch.setattr(sys, "argv", common_argv)
    assert mod.main() == 0
    # The valid `insecure_code_NL` must be SKIPPED; corrupt `jailbroken_NL`
    # and missing `evil_numbers_NL` must be RECOMPUTED.
    assert "insecure_code_NL" not in invocations, (
        f"valid output should be skipped, but main loop computed it: {invocations}"
    )
    assert "jailbroken_NL" in invocations, f"corrupt output should be recomputed: {invocations}"
    assert "evil_numbers_NL" in invocations, f"missing output should be computed: {invocations}"

    # --- Round 2: --force recomputes the (now-valid) insecure_code_NL too. ---
    invocations.clear()
    monkeypatch.setattr(sys, "argv", [*common_argv, "--force"])
    assert mod.main() == 0
    assert set(invocations) == {"insecure_code_NL", "jailbroken_NL", "evil_numbers_NL"}, (
        f"--force should recompute every cell, got {invocations}"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
