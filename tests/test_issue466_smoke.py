# ruff: noqa: RUF003
# Intentional Unicode (※, ′, ×, −, →, —) in docstrings/comments matching the project house style.
"""CPU-only smoke tests for the GPU-free pieces of the task #466 pipeline.

The GPU phases (vLLM gen, HF teacher-force) are exercised by the unified
on-pod smoke phase in ``scripts/run_issue466_pipeline.sh --smoke``. These
tests cover everything the VM CAN run locally without a GPU:

  - RB-JS estimator math identity (``JS = 0.5 (KL_P‖m + KL_Q‖m)``)
  - JS bounds in [0, 1] for base-2 log
  - Premise detectors (``is_spanish``, ``uppercase_ratio``, ``is_all_caps``)
  - Slice partition (no overlap; no contamination)
  - Matched-contrast aggregation on synthetic predictor + marker JSONs
  - poll_pipeline.py sentinel ``_parse_sentinel`` round-trip on the JSONs
    the driver writes
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── RB-JS estimator math ───────────────────────────────────────────────────


def test_js_identity_matches_two_kls():
    """JS(P, Q) = 0.5 (KL(P || M) + KL(Q || M)) within float tolerance."""
    from issue466_predictors import js_from_logprobs, kl_div_from_logprobs

    torch.manual_seed(0)
    # Two random distributions over a small vocab.
    V = 17
    logits_p = torch.randn(5, V)
    logits_q = torch.randn(5, V)
    log_p = torch.nn.functional.log_softmax(logits_p, dim=-1)
    log_q = torch.nn.functional.log_softmax(logits_q, dim=-1)

    log_m = torch.logsumexp(torch.stack([log_p, log_q], dim=0), dim=0) - math.log(2.0)
    kl_pm = kl_div_from_logprobs(log_p, log_m, base=2.0)
    kl_qm = kl_div_from_logprobs(log_q, log_m, base=2.0)
    js_explicit = 0.5 * (kl_pm + kl_qm)
    js_computed = js_from_logprobs(log_p, log_q, base=2.0)
    assert torch.allclose(js_explicit, js_computed, atol=1e-6), (
        f"JS identity broken: max-abs diff {(js_explicit - js_computed).abs().max().item()}"
    )


def test_js_bounded_zero_to_one_base_2():
    """Base-2 JS is in [0, 1] (= log2(2) upper bound for any two distributions)."""
    from issue466_predictors import js_from_logprobs

    torch.manual_seed(1)
    for _ in range(20):
        V = 32
        log_p = torch.nn.functional.log_softmax(torch.randn(8, V), dim=-1)
        log_q = torch.nn.functional.log_softmax(torch.randn(8, V), dim=-1)
        js = js_from_logprobs(log_p, log_q, base=2.0)
        # Allow a tiny float-rounding margin above 1.0.
        assert js.min().item() >= -1e-6, f"negative JS: {js.min().item()}"
        assert js.max().item() <= 1.0 + 1e-5, f"JS > 1 in base 2: {js.max().item()}"


def test_js_zero_for_identical_distributions():
    """JS(P, P) = 0."""
    from issue466_predictors import js_from_logprobs

    torch.manual_seed(2)
    log_p = torch.nn.functional.log_softmax(torch.randn(4, 16), dim=-1)
    js = js_from_logprobs(log_p, log_p, base=2.0)
    assert torch.allclose(js, torch.zeros_like(js), atol=1e-6), (
        f"JS(P, P) != 0: max {js.abs().max().item()}"
    )


def test_kl_nonneg():
    """KL is non-negative."""
    from issue466_predictors import kl_div_from_logprobs

    torch.manual_seed(3)
    for _ in range(20):
        log_p = torch.nn.functional.log_softmax(torch.randn(4, 16), dim=-1)
        log_q = torch.nn.functional.log_softmax(torch.randn(4, 16), dim=-1)
        kl = kl_div_from_logprobs(log_p, log_q, base=2.0)
        assert kl.min().item() >= -1e-6, f"negative KL: {kl.min().item()}"


def test_js_max_for_disjoint_supports():
    """JS approaches 1 (base 2) when P and Q put mass on disjoint atoms."""
    from issue466_predictors import js_from_logprobs

    V = 8
    # P puts ~all mass on token 0, Q puts ~all mass on token 1.
    logits_p = torch.full((1, V), -50.0)
    logits_p[0, 0] = 50.0
    logits_q = torch.full((1, V), -50.0)
    logits_q[0, 1] = 50.0
    log_p = torch.nn.functional.log_softmax(logits_p, dim=-1)
    log_q = torch.nn.functional.log_softmax(logits_q, dim=-1)
    js = js_from_logprobs(log_p, log_q, base=2.0)
    # JS for fully disjoint distributions = log 2 base 2 = 1.
    assert js.item() > 0.95, f"JS for disjoint atoms too low: {js.item()}"


# ── Premise detectors ──────────────────────────────────────────────────────


def test_is_spanish_basic():
    from issue466_personas import is_spanish

    assert is_spanish("Aquí está mi recomendación para un restaurante italiano excelente.")
    assert is_spanish(
        "Hola, te recomiendo el restaurante La Pergola en Roma — la comida es maravillosa."
    )
    assert not is_spanish(
        "Sure, I would recommend Locanda Verde in Tribeca; wonderful Italian food and atmosphere."
    )
    assert not is_spanish("")


def test_uppercase_ratio_and_is_all_caps():
    from issue466_personas import is_all_caps, uppercase_ratio

    assert uppercase_ratio("THIS IS ALL CAPS") > 0.9
    assert uppercase_ratio("this is all lower") == 0.0
    assert uppercase_ratio("Mixed Case Sentence") < 0.5
    assert uppercase_ratio("") == 0.0
    assert is_all_caps("CRISTIANO RONALDO IS A LEGEND.")
    assert not is_all_caps("Cristiano Ronaldo is a legend.")


# ── Slice partition + non-contamination ────────────────────────────────────


def test_slices_have_30_each_and_disjoint():
    from issue466_personas import SLICE_NONTRIGGER, SLICE_TRIGGER_A, SLICE_TRIGGER_B

    assert len(SLICE_TRIGGER_A) == 30
    assert len(SLICE_TRIGGER_B) == 30
    assert len(SLICE_NONTRIGGER) == 30
    assert set(SLICE_TRIGGER_A).isdisjoint(SLICE_TRIGGER_B)
    assert set(SLICE_TRIGGER_A).isdisjoint(SLICE_NONTRIGGER)
    assert set(SLICE_TRIGGER_B).isdisjoint(SLICE_NONTRIGGER)


def test_personas_unique():
    from issue466_personas import PERSONAS

    assert len(PERSONAS) == 5
    assert len(set(PERSONAS.values())) == 5, "two personas have identical system-prompt text"


# ── Matched-contrast aggregation (synthetic) ───────────────────────────────


def _write_synthetic_eval_results(tmp_path: Path) -> Path:
    """Write a minimal eval_results/issue_466 tree the analyzer can consume."""
    root = tmp_path / "eval_results" / "issue_466"
    (root / "predictors").mkdir(parents=True)
    (root / "onpolicy_gen").mkdir(parents=True)
    (root / "onpolicy_endpos_logp").mkdir(parents=True)

    for behavior in ("A_spanish_restaurants", "B_caps_sports"):
        with open(root / "predictors" / f"{behavior}.json", "w") as f:
            json.dump(
                {
                    "behavior": behavior,
                    "predictor_pair": ["S", f"S_prime_{behavior[0]}_x"],
                    "js": {
                        "averaged_js_union": 0.05,
                        "slice_mean_js": {"nontrigger": 0.04, "trigger": 0.42},
                        "slice_mean_kl_s_sprime": {"nontrigger": 0.05, "trigger": 0.5},
                        "slice_mean_kl_sprime_s": {"nontrigger": 0.05, "trigger": 0.4},
                        "per_probe_scalars": {"nontrigger": [], "trigger": []},
                        "per_position_traj": {"nontrigger": [], "trigger": []},
                    },
                    "cosine": {
                        "extraction_a0_endofsystemprompt": {
                            "7": 0.99,
                            "14": 0.98,
                            "21": 0.97,
                            "27": 0.96,
                        },
                        "extraction_a_lastinputtoken_per_slice_per_layer": {
                            "nontrigger": {"7": 0.99, "14": 0.98, "21": 0.97, "27": 0.96},
                            "trigger": {"7": 0.91, "14": 0.86, "21": 0.78, "27": 0.74},
                        },
                        "extraction_b_ownresponsemean_per_slice_per_layer": {
                            "nontrigger": {"7": 0.99, "14": 0.99, "21": 0.99, "27": 0.99},
                            "trigger": {"7": 0.70, "14": 0.55, "21": 0.40, "27": 0.35},
                        },
                        "extraction_a_per_probe": {},
                        "extraction_b_per_probe": None,
                        "layers": [7, 14, 21, 27],
                        "headline_layer": 21,
                    },
                },
                f,
            )

    # 11 marker cells with realistic (mean_logp_trained, mean_logp_base, emission_rate).
    # H1 pattern: trained S has BIG positive `delta` everywhere (the LoRA strongly
    # lifts log p(※) over base; S emits ※ on ~90% of own answers). Trained S′ on
    # NON-TRIGGER slice behaves like S (small or no behavior change). Trained S′
    # on TRIGGER slice has a much SMALLER `delta` than S (Spanish output context
    # dampens the LoRA's marker push). So delta_marker_normed = sp.delta − s.delta
    # is ~0 on non-trigger and substantially NEGATIVE on trigger.
    marker_cells = {
        ("S", "nontrigger"): (-0.5, -3.5, 0.90),
        ("S", "trigger_A"): (-0.5, -3.5, 0.90),
        ("S", "trigger_B"): (-0.5, -3.5, 0.90),
        ("S_prime_A_spanish_restaurants", "nontrigger"): (-0.6, -3.5, 0.85),
        ("S_prime_A_spanish_restaurants", "trigger_A"): (-2.5, -3.6, 0.30),
        ("always_A_spanish", "nontrigger"): (-2.4, -3.6, 0.30),
        ("always_A_spanish", "trigger_A"): (-2.6, -3.6, 0.30),
        ("S_prime_B_caps_sports", "nontrigger"): (-0.6, -3.5, 0.85),
        ("S_prime_B_caps_sports", "trigger_B"): (-2.0, -3.4, 0.40),
        ("always_B_caps", "nontrigger"): (-1.9, -3.4, 0.40),
        ("always_B_caps", "trigger_B"): (-2.2, -3.4, 0.40),
    }
    for (persona, slc), (mean_t, mean_b, emit) in marker_cells.items():
        with open(root / "onpolicy_endpos_logp" / f"{persona}_{slc}.json", "w") as f:
            json.dump(
                {
                    "persona": persona,
                    "slice": slc,
                    "n_contexts": 240,
                    "mean_logp_trained": mean_t,
                    "mean_logp_base": mean_b,
                    "delta": mean_t - mean_b,
                    "logp_trained_per_context": [],
                    "logp_base_per_context": [],
                },
                f,
            )
        with open(root / "onpolicy_gen" / f"{persona}_{slc}.json", "w") as f:
            json.dump({"emission_rate": emit, "truncation_frac": 0.0}, f)
    return root


def test_matched_contrast_table_shapes_and_signs(tmp_path, monkeypatch):
    """Build matched-contrast table on synthetic data; check the pattern is right."""
    root = _write_synthetic_eval_results(tmp_path)
    import issue466_analyze

    monkeypatch.setattr(issue466_analyze, "EVAL_ROOT", root)
    rows = issue466_analyze.build_matched_contrast_table(headline_layer=21)
    assert len(rows) == 4
    # Within each behavior, the blind columns are flat across the 2 slices.
    by_behavior: dict[str, list[dict]] = {}
    for r in rows:
        by_behavior.setdefault(r["behavior"], []).append(r)
    for behavior, rs in by_behavior.items():
        assert len(rs) == 2
        assert rs[0]["blind_avg_js"] == rs[1]["blind_avg_js"], f"blind JS not flat for {behavior}"
        assert rs[0]["blind_cos_a0_L21"] == rs[1]["blind_cos_a0_L21"], (
            f"blind cos a0 not flat for {behavior}"
        )
        # Sighted columns split (trigger row's JS > nontrigger's).
        trig = next(r for r in rs if r["slice_kind"] == "trigger")
        non = next(r for r in rs if r["slice_kind"] == "nontrigger")
        assert trig["sighted_slice_js"] > non["sighted_slice_js"], (
            f"sighted JS doesn't split for {behavior}"
        )
        assert trig["sighted_slice_cos_b_L21"] < non["sighted_slice_cos_b_L21"], (
            f"sighted cos b doesn't split for {behavior}"
        )
        # Matched marker drop is negative on trigger slice.
        assert trig["delta_marker_logp_normed"] < non["delta_marker_logp_normed"], (
            f"matched marker drop not larger on trigger for {behavior}"
        )


# ── Sentinel round-trip via poll_pipeline._parse_sentinel ─────────────────


def test_sentinel_parse_round_trip(tmp_path):
    """The driver script writes a sentinel; the poller's _parse_sentinel must accept it."""
    from scripts.poll_pipeline import _parse_sentinel

    # Mirror the JSON the driver writes on smoke completion.
    body = json.dumps(
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "ts": 1700000000,
            "note": "phase: smoke complete\nsmoke_prompts: 3",
        }
    )
    parsed = _parse_sentinel("/workspace/logs/issue-466-epm_results-1700000000.json", body)
    assert parsed is not None
    assert parsed["kind"] == "epm:results"
    assert parsed["version"] == 1


def test_sentinel_parse_failure_marker(tmp_path):
    """A retrain-reproduce FAIL sentinel must also parse."""
    from scripts.poll_pipeline import _parse_sentinel

    body = json.dumps(
        {
            "sentinel_schema_version": 1,
            "kind": "epm:failure",
            "version": 1,
            "ts": 1700000001,
            "note": "failure_class: data\nreason: retrain_did_not_reproduce_456",
        }
    )
    parsed = _parse_sentinel("/workspace/logs/issue-466-epm_failure-1700000001.json", body)
    assert parsed is not None
    assert parsed["kind"] == "epm:failure"


def test_sentinel_parse_rejects_missing_keys():
    """A sentinel missing sentinel_schema_version is skipped (returns None)."""
    from scripts.poll_pipeline import _parse_sentinel

    body = json.dumps({"schema": 1, "kind": "epm:results", "version": 1})
    parsed = _parse_sentinel("/workspace/logs/issue-466-epm_results-x.json", body)
    assert parsed is None


# ── Marker-id assertion (single CPU forward pass; skipped without HF cache) ─


@pytest.mark.skipif(
    not Path("/workspace/.cache/huggingface").exists(),
    reason="HF cache not present — skipping tokenizer download check on the VM",
)
def test_marker_token_id_assertion():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    ids = tok.encode("※", add_special_tokens=False)
    assert ids == [63680], f"※ tokenizes to {ids}, expected [63680]"
