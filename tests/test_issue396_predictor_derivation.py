"""CPU smoke tests for the predictor-derivation helper in
``scripts/recompute_predictors_i396.py`` (BF2 round 2).

The full GPU sweep (Qwen-2.5-7B base-model forward across 48 personas x
20 questions) is impossible to exercise in CI / on CPU. We pull the pure
post-tensor derivation step (``_derive_predictors``) into its own helper
so it's testable with synthetic small tensors.

These tests cover the math the analyzer relies on:

1. **Predictor #1 (cosine-to-assistant L15)** — for a 3-persona toy where
   one persona's hidden state is identical to the baseline, the cosine
   must round-trip to ~1.0; for the opposite-sign persona, ~-1.0.

2. **Predictor #2 (JS-to-baseline)** — for a 3-persona toy where one
   persona's log-prob tensor is IDENTICAL to the baseline, the JS must
   be 0.0; for a persona with a clearly different distribution, the JS
   must be strictly positive (and below ln(2) ~= 0.693, the JS bound).

3. **Predictor #3 (pairwise output distance)** — the per-persona scalar
   is the row-mean of the pairwise matrix (excluding self). For a 3-persona
   toy where the matrix is symmetric, the per-persona scalar must equal
   mean(matrix[name][k] for k != name).

4. **Predictor #3 pairwise matrix has zero diagonal** — JS(p, p) = 0
   for all personas by construction; the helper must populate the matrix
   diagonal with literal 0.0, not the JS(p, p) computation (which can
   produce a small nonzero from float rounding).

5. **Analyzer load path graceful-degrades on CPU.** When CUDA is absent
   AND the cache JSON does not exist, the loader returns ``{}`` rather
   than crashing — the analyzer continues with predictors #4 + #5.

6. **Analyzer load path reads the cache when present.** When a cache
   JSON exists at ``eval_results/issue_396/base_model_predictors.json``,
   both loader helpers return the saved per-persona dicts.

Plan v2.3 §4.8 Phase E.3 + §A17; code-review v1 round 1 binding fix BF2.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_recompute():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module("recompute_predictors_i396")


def _import_analyzer():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    # Force a fresh module so the module-level _BASE_PREDICTORS_CACHE is reset
    # between test cases — otherwise the order of tests in this file would
    # determine the cached value.
    if "analyze_issue396" in sys.modules:
        del sys.modules["analyze_issue396"]
    return importlib.import_module("analyze_issue396")


# ── Build a synthetic predictor scenario ───────────────────────────────────


def _make_toy_log_probs(
    *,
    seed: int = 0,
    n_personas: int = 3,
    response_len: int = 5,
    vocab_size: int = 16,
) -> list[torch.Tensor]:
    """Return per-question log-prob tensors (each (n_personas, T, V), CPU).

    The 0th persona = baseline (identical log-probs across questions).
    The 1st persona = same as baseline EXCEPT a small uniform perturbation
    that ensures nonzero JS-to-baseline.
    The 2nd persona = entirely different distribution per question.
    """
    rng = torch.Generator().manual_seed(seed)
    base = torch.randn(response_len, vocab_size, generator=rng)
    base = torch.log_softmax(base, dim=-1)

    # Persona 1: small perturbation
    pert = 0.05 * torch.randn(response_len, vocab_size, generator=rng)
    persona_1_logits = torch.log_softmax(base + pert, dim=-1)

    # Per-question tensors (3 questions, each (3, T, V))
    out: list[torch.Tensor] = []
    for _q in range(3):
        # Persona 2 is different on every question; baseline + persona_1 stay
        # constant across questions so the means-over-q are well-defined.
        persona_2_raw = torch.randn(response_len, vocab_size, generator=rng) * 2.0
        persona_2_logits = torch.log_softmax(persona_2_raw, dim=-1)
        # Index 0 = persona_a, 1 = persona_b, 2 = baseline (the helper expects
        # the baseline at the LAST index — see all_names convention).
        stacked = torch.stack([persona_1_logits, persona_2_logits, base], dim=0)
        out.append(stacked)
    return out


def _make_toy_hidden(
    *,
    n_personas: int = 3,
    hidden_dim: int = 32,
) -> list[torch.Tensor]:
    """Return per-question hidden-state tensors (each (n_personas, D), CPU)."""
    # Index 2 = baseline. We pick:
    #   persona 0 = baseline + 1e-6 noise (cos ~= 1.0)
    #   persona 1 = -baseline (cos = -1.0)
    #   baseline at index 2
    base = torch.randn(hidden_dim)
    out: list[torch.Tensor] = []
    for _q in range(3):
        persona_0 = base + 1e-6 * torch.randn(hidden_dim)  # cos ~= 1
        persona_1 = -base  # cos = -1
        stacked = torch.stack([persona_0, persona_1, base], dim=0)
        out.append(stacked)
    return out


# ── Predictor #1 cosine tests ───────────────────────────────────────────────


def test_predictor_1_cosine_baseline_identity_returns_one():
    """A persona with hidden state ~= baseline must score ~+1.0 cosine."""
    recompute = _import_recompute()
    all_names = ["persona_a", "persona_b", "__assistant_baseline__"]
    hidden = _make_toy_hidden()
    log_probs = _make_toy_log_probs()
    pred_1, _, _, _ = recompute._derive_predictors(
        all_names=all_names,
        per_question_hidden=hidden,
        per_question_log_probs=log_probs,
    )
    # persona_a is baseline + tiny noise; cos must be very close to 1
    assert "persona_a" in pred_1
    assert pred_1["persona_a"] > 0.999, (
        f"persona_a is baseline + 1e-6 noise; cos should be ~1.0, got {pred_1['persona_a']}"
    )


def test_predictor_1_cosine_opposite_sign_returns_minus_one():
    """A persona with hidden state = -baseline must score ~-1.0 cosine."""
    recompute = _import_recompute()
    all_names = ["persona_a", "persona_b", "__assistant_baseline__"]
    hidden = _make_toy_hidden()
    log_probs = _make_toy_log_probs()
    pred_1, _, _, _ = recompute._derive_predictors(
        all_names=all_names,
        per_question_hidden=hidden,
        per_question_log_probs=log_probs,
    )
    assert "persona_b" in pred_1
    assert pred_1["persona_b"] < -0.999, (
        f"persona_b is -baseline; cos should be ~-1.0, got {pred_1['persona_b']}"
    )


def test_predictor_1_baseline_not_in_output():
    """The cosine predictor must NOT include the baseline in its output keys."""
    recompute = _import_recompute()
    all_names = ["persona_a", "persona_b", "__assistant_baseline__"]
    pred_1, _, _, _ = recompute._derive_predictors(
        all_names=all_names,
        per_question_hidden=_make_toy_hidden(),
        per_question_log_probs=_make_toy_log_probs(),
    )
    assert "__assistant_baseline__" not in pred_1
    assert set(pred_1.keys()) == {"persona_a", "persona_b"}


# ── Predictor #2 JS-to-baseline tests ───────────────────────────────────────


def test_predictor_2_js_to_baseline_zero_for_identical_distribution():
    """When a persona's log-probs == baseline, JS-to-baseline must be 0.0."""
    recompute = _import_recompute()
    # Override the toy generator: make persona_a IDENTICAL to baseline on
    # every question.
    base_lp = torch.log_softmax(torch.randn(5, 16), dim=-1)
    per_q_log_probs = []
    per_q_hidden = []
    for _ in range(3):
        # 3 personas: persona_a (= baseline), persona_b (random), baseline
        other = torch.log_softmax(torch.randn(5, 16) * 2.0, dim=-1)
        per_q_log_probs.append(torch.stack([base_lp, other, base_lp], dim=0))
        per_q_hidden.append(torch.randn(3, 16))
    all_names = ["persona_a", "persona_b", "__assistant_baseline__"]
    _, pred_2, _, _ = recompute._derive_predictors(
        all_names=all_names,
        per_question_hidden=per_q_hidden,
        per_question_log_probs=per_q_log_probs,
    )
    assert pred_2["persona_a"] == pytest.approx(0.0, abs=1e-6), (
        f"persona_a's log-probs are identical to baseline; "
        f"JS must be 0.0, got {pred_2['persona_a']}"
    )


def test_predictor_2_js_to_baseline_positive_and_bounded_for_different_distribution():
    """A persona with different log-probs must have JS in (0, ln(2)]."""
    import math

    recompute = _import_recompute()
    _, pred_2, _, _ = recompute._derive_predictors(
        all_names=["persona_a", "persona_b", "__assistant_baseline__"],
        per_question_hidden=_make_toy_hidden(),
        per_question_log_probs=_make_toy_log_probs(),
    )
    # persona_b's log-probs are intentionally different from baseline on
    # every question. The JS must be > 0 and ≤ ln(2).
    assert pred_2["persona_b"] > 0.0
    assert pred_2["persona_b"] < math.log(2.0) + 1e-3


# ── Predictor #3 pairwise matrix tests ──────────────────────────────────────


def test_predictor_3_pairwise_matrix_diagonal_is_zero():
    """JS(p, p) must be literal 0.0 on the diagonal for every persona."""
    recompute = _import_recompute()
    _, _, _, matrix = recompute._derive_predictors(
        all_names=["persona_a", "persona_b", "__assistant_baseline__"],
        per_question_hidden=_make_toy_hidden(),
        per_question_log_probs=_make_toy_log_probs(),
    )
    for name in matrix:
        assert matrix[name][name] == 0.0, (
            f"matrix[{name}][{name}] = {matrix[name][name]}; must be exactly 0.0"
        )


def test_predictor_3_scalar_equals_mean_of_other_rows():
    """Per-persona scalar = mean of pairwise matrix row excluding self."""
    recompute = _import_recompute()
    _, _, pred_3, matrix = recompute._derive_predictors(
        all_names=["persona_a", "persona_b", "__assistant_baseline__"],
        per_question_hidden=_make_toy_hidden(),
        per_question_log_probs=_make_toy_log_probs(),
    )
    for name in pred_3:
        row = matrix[name]
        expected = sum(v for k, v in row.items() if k != name) / (len(row) - 1)
        assert pred_3[name] == pytest.approx(expected, rel=1e-6)


def test_predictor_3_matrix_excludes_baseline_personas():
    """The pairwise matrix is over PANEL personas only (baseline excluded)."""
    recompute = _import_recompute()
    _, _, pred_3, matrix = recompute._derive_predictors(
        all_names=["persona_a", "persona_b", "__assistant_baseline__"],
        per_question_hidden=_make_toy_hidden(),
        per_question_log_probs=_make_toy_log_probs(),
    )
    for outer_name in matrix:
        assert outer_name != "__assistant_baseline__"
        for inner_name in matrix[outer_name]:
            assert inner_name != "__assistant_baseline__"
    assert "__assistant_baseline__" not in pred_3


# ── Analyzer load-path graceful degradation tests ───────────────────────────


def test_analyzer_returns_empty_when_no_cache_and_no_cuda(monkeypatch, tmp_path):
    """CPU-only dev path: no cache file, no CUDA → both loaders return empty.

    This is the documented fallback so the analyzer can run on a dev VM
    with only #4 + #5 predictors and surface "predictor missing" rows in
    the summary JSON.
    """
    analyzer = _import_analyzer()
    # Point the cache to a non-existent path so we exercise the missing-cache
    # branch even if a real cache happens to be on disk.
    monkeypatch.setattr(analyzer, "BASE_MODEL_PREDICTORS_CACHE", tmp_path / "nonexistent.json")
    monkeypatch.setattr(analyzer, "_BASE_PREDICTORS_CACHE", None)
    # Patch torch.cuda.is_available to return False — simulates dev VM.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    js_to_baseline, pairwise = analyzer.recompute_js_predictors({})
    cosine = analyzer.compute_cosine_to_assistant_predictor({})
    assert js_to_baseline == {}
    assert pairwise == {}
    assert cosine == {}


def test_analyzer_reads_cache_when_present(monkeypatch, tmp_path):
    """When the cache JSON exists, both loaders return its per-persona dicts.

    This is the cheap path on a fresh pod that has run the recompute once.
    """
    analyzer = _import_analyzer()
    payload = {
        "schema_version": 1,
        "n_personas": 2,
        "predictor_1_cosine_to_assistant_L15": {"persona_a": 0.42, "persona_b": -0.13},
        "predictor_2_js_to_baseline": {"persona_a": 0.001, "persona_b": 0.05},
        "predictor_3_pairwise_output_distance": {"persona_a": 0.02, "persona_b": 0.03},
        "predictor_3_pairwise_js_matrix": {
            "persona_a": {"persona_a": 0.0, "persona_b": 0.02},
            "persona_b": {"persona_a": 0.02, "persona_b": 0.0},
        },
    }
    fake_cache = tmp_path / "base_model_predictors.json"
    fake_cache.write_text(json.dumps(payload))
    monkeypatch.setattr(analyzer, "BASE_MODEL_PREDICTORS_CACHE", fake_cache)
    monkeypatch.setattr(analyzer, "_BASE_PREDICTORS_CACHE", None)

    js_to_baseline, pairwise = analyzer.recompute_js_predictors({})
    cosine = analyzer.compute_cosine_to_assistant_predictor({})
    assert js_to_baseline == {"persona_a": 0.001, "persona_b": 0.05}
    assert pairwise == {"persona_a": 0.02, "persona_b": 0.03}
    assert cosine == {"persona_a": 0.42, "persona_b": -0.13}


def test_analyzer_memoizes_cache_so_inline_recompute_fires_at_most_once(monkeypatch, tmp_path):
    """Two-loader sequence must trigger AT MOST ONE inline GPU recompute.

    The analyzer calls both ``recompute_js_predictors`` (for #2/#3) and
    ``compute_cosine_to_assistant_predictor`` (for #1) in the same main()
    pass. A naive implementation would call the GPU sweep twice — the
    module-level memoization guards against that.
    """
    analyzer = _import_analyzer()
    monkeypatch.setattr(analyzer, "BASE_MODEL_PREDICTORS_CACHE", tmp_path / "no_cache.json")
    monkeypatch.setattr(analyzer, "_BASE_PREDICTORS_CACHE", None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    call_count = {"n": 0}

    def fake_compute(*, cache_path):
        call_count["n"] += 1
        return {
            "predictor_1_cosine_to_assistant_L15": {"p": 0.5},
            "predictor_2_js_to_baseline": {"p": 0.01},
            "predictor_3_pairwise_output_distance": {"p": 0.02},
        }

    # Inject a fake compute_base_model_predictors via the recompute module.
    import recompute_predictors_i396

    monkeypatch.setattr(recompute_predictors_i396, "compute_base_model_predictors", fake_compute)
    # First call triggers the fake.
    _ = analyzer.recompute_js_predictors({})
    # Second call must hit the memo, not the fake.
    _ = analyzer.compute_cosine_to_assistant_predictor({})
    assert call_count["n"] == 1, (
        f"compute_base_model_predictors was called {call_count['n']} times; "
        "memoization must collapse this to 1 call across both loader helpers."
    )
