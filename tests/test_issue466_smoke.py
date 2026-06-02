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


# ── Per-(behavior, slice) checkpoint writer + gen cache round-trip ────────


def test_write_slice_checkpoint_creates_file(tmp_path):
    """The per-slice checkpoint helper writes ``{behavior}_{slice}.json``.

    Pins the round-2 fix for the "checkpoint per phase, not at end" gap —
    a crash in a later slice or in phase_cosine must not lose the JS work
    of an already-scored slice.
    """
    from issue466_predictors import _write_slice_checkpoint

    payload = {
        "per_probe_scalars": [{"probe_idx": 0, "probe": "q?", "n_pooled": 4, "mean_js": 0.12}],
        "per_position_traj": [[0.1, 0.15, 0.2]],
        "slice_mean_js": 0.12,
        "slice_mean_kl_s_sprime": 0.13,
        "slice_mean_kl_sprime_s": 0.11,
        "n_probes": 1,
        "n_valid": 1,
    }
    out_path = _write_slice_checkpoint(tmp_path, "A_spanish_restaurants", "trigger", payload)
    assert out_path.exists()
    assert out_path.name == "A_spanish_restaurants_trigger.json"
    loaded = json.loads(out_path.read_text())
    assert loaded["behavior"] == "A_spanish_restaurants"
    assert loaded["slice"] == "trigger"
    assert loaded["phase"] == "js"
    assert loaded["slice_mean_js"] == 0.12
    assert loaded["per_probe_scalars"][0]["mean_js"] == 0.12


def test_gen_cache_round_trip(tmp_path):
    """Persisted gen cache round-trips through ``_write_gen_cache`` / ``_load_gen_cache``.

    JSON-key conversion (int q_idx -> str on write, str -> int on load) must
    be exact; the loader returns ``None`` when no cache file exists.
    """
    from issue466_predictors import _load_gen_cache, _write_gen_cache

    # No cache yet -> None.
    assert _load_gen_cache(tmp_path, "A_spanish_restaurants") is None

    gen = {
        "cache": {
            "S": {
                "trigger": {0: ["resp0", "resp1"], 1: ["resp2"]},
                "nontrigger": {0: ["respN0"]},
            },
            "S_prime": {
                "trigger": {0: ["spA", "spB"], 1: ["spC"]},
                "nontrigger": {0: ["spN0"]},
            },
        },
        "slices": {"trigger": ["q0", "q1"], "nontrigger": ["qN0"]},
        "gen_wall_seconds": 42.5,
    }
    out_path = _write_gen_cache(tmp_path, "A_spanish_restaurants", gen)
    assert out_path.exists()
    assert out_path.name == "A_spanish_restaurants_gen_cache.json"
    loaded = _load_gen_cache(tmp_path, "A_spanish_restaurants")
    assert loaded is not None
    # Int keys preserved through the round-trip.
    assert loaded["cache"]["S"]["trigger"][0] == ["resp0", "resp1"]
    assert loaded["cache"]["S"]["trigger"][1] == ["resp2"]
    assert loaded["cache"]["S_prime"]["nontrigger"][0] == ["spN0"]
    assert loaded["slices"] == {"trigger": ["q0", "q1"], "nontrigger": ["qN0"]}
    assert loaded["gen_wall_seconds"] == 42.5


def test_phase_js_score_invokes_on_slice_complete_callback(monkeypatch):
    """``phase_js_score`` must call ``on_slice_complete`` per slice — checkpoint hook.

    Replaces the GPU helpers (model load, tokenizer, per-position scoring)
    with monkey-patched stubs so this test runs on the VM without CUDA.
    Pins the contract that the per-slice callback fires before the function
    returns and that it carries the slice payload the per-slice writer
    expects.
    """
    import issue466_predictors as mod

    # Stub HF transformers + torch device — phase_js_score loads a tokenizer
    # + base model + scores responses. Replace each with a fake to avoid
    # loading Qwen-7B on the VM.
    class _StubTokenizer:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "PREFIX"

    class _StubModel:
        def eval(self):
            return self

    stub_tokenizer = _StubTokenizer()
    stub_model = _StubModel()

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(_name):
            return stub_tokenizer

    class _StubAutoModel:
        @staticmethod
        def from_pretrained(_name, **_kwargs):
            return stub_model

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _StubAutoModel)
    monkeypatch.setattr(mod.torch, "device", lambda _: "stub_device")
    # _per_position_js needs to be stubbed too — it does a forward pass.
    monkeypatch.setattr(
        mod,
        "_per_position_js",
        lambda *a, **kw: {"js": [0.1, 0.2], "kl_s_sprime": [0.1, 0.1], "kl_sprime_s": [0.1, 0.1]},
    )

    cache = {
        "S": {"trigger": {0: ["r0"]}, "nontrigger": {0: ["rN0"]}},
        "S_prime": {"trigger": {0: ["r0sp"]}, "nontrigger": {0: ["rN0sp"]}},
    }
    slices = {"trigger": ["q0"], "nontrigger": ["qN0"]}
    callback_calls: list[tuple[str, dict]] = []

    def _cb(slice_name: str, slice_payload: dict) -> None:
        callback_calls.append((slice_name, slice_payload))

    result = mod.phase_js_score(
        "A_spanish_restaurants",
        cache,
        slices,
        smoke_probes=None,
        on_slice_complete=_cb,
    )
    # Callback was fired for each slice with the per-slice payload shape.
    assert {name for name, _ in callback_calls} == {"trigger", "nontrigger"}
    for _, payload in callback_calls:
        assert "per_probe_scalars" in payload
        assert "per_position_traj" in payload
        assert "slice_mean_js" in payload
    # The combined return value still has both slices.
    assert set(result["slice_mean_js"].keys()) == {"trigger", "nontrigger"}


# ── Round-3 regression tests: gen-cache fingerprint + slice-checkpoint resume ──
#
# These pin the round-3 fix for the silent-corruption bug class that the
# round-2 PR introduced. The bug: smoke (3 probes × 2 samples) and full (30
# probes × 8 samples) wrote to the SAME default --out-dir, so the smoke-config
# gen cache was silently reused by a subsequent full-config call. The fix is
# two-layered: (1) the pipeline shell separates smoke vs full into disjoint
# output trees; (2) the predictor fingerprints its gen-cache + slice-checkpoint
# config and refuses to load a cache whose fingerprint disagrees.


def test_smoke_gen_cache_rejected_by_full_call(tmp_path):
    """A gen cache written with smoke params (small r_samples / smoke_probes)
    MUST NOT be returned by a full-params load: _load_gen_cache validates the
    fingerprint and returns None on mismatch so a regeneration is forced.

    This is the central round-3 invariant. Even if a smoke artifact somehow
    lands in the production directory (defense-in-depth assuming the shell-
    layer separation is bypassed), the predictor refuses to silently consume
    it.
    """
    from issue466_personas import PERSONAS
    from issue466_predictors import (
        _compute_config_fingerprint,
        _load_gen_cache,
        _resolve_slices,
        _write_gen_cache,
    )

    behavior = "A_spanish_restaurants"
    # Build the smoke-config fingerprint + a gen cache written under it.
    smoke_slices = _resolve_slices(behavior, smoke_probes=3)
    smoke_fp = _compute_config_fingerprint(
        behavior=behavior,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        s_text=PERSONAS["S"],
        sprime_text=PERSONAS["S_prime_A_spanish_restaurants"],
        slices=smoke_slices,
        r_samples=2,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=256,
        max_model_len=4096,
        seed=42,
        smoke_probes=3,
    )
    smoke_gen = {
        "cache": {
            "S": {"trigger": {0: ["r0"]}, "nontrigger": {0: ["rN0"]}},
            "S_prime": {"trigger": {0: ["r0sp"]}, "nontrigger": {0: ["rN0sp"]}},
        },
        "slices": smoke_slices,
        "gen_wall_seconds": 1.0,
    }
    _write_gen_cache(tmp_path, behavior, smoke_gen, config_fingerprint=smoke_fp)

    # Now build the FULL-config fingerprint and try to load — must return None.
    full_slices = _resolve_slices(behavior, smoke_probes=None)
    full_fp = _compute_config_fingerprint(
        behavior=behavior,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        s_text=PERSONAS["S"],
        sprime_text=PERSONAS["S_prime_A_spanish_restaurants"],
        slices=full_slices,
        r_samples=8,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=256,
        max_model_len=4096,
        seed=42,
        smoke_probes=None,
    )
    assert full_fp != smoke_fp, "smoke and full fingerprints unexpectedly equal"

    loaded = _load_gen_cache(tmp_path, behavior, expected_fingerprint=full_fp)
    assert loaded is None, (
        "smoke-config gen cache was silently returned for a full-config call "
        "(this is the round-2 silent-corruption bug the fingerprint is meant to catch)"
    )

    # Sanity: loading with the matching smoke fingerprint DOES return the cache.
    loaded_smoke = _load_gen_cache(tmp_path, behavior, expected_fingerprint=smoke_fp)
    assert loaded_smoke is not None
    assert loaded_smoke["cache"]["S"]["trigger"][0] == ["r0"]


def test_smoke_and_full_pipeline_out_dirs_are_disjoint():
    """Round-3 fix #1: the pipeline shell MUST resolve smoke and full to
    DISJOINT output trees so a smoke artifact can never be opened by a
    full-pipeline phase.

    Tested by parsing the resolved paths out of ``run_issue466_pipeline.sh``
    under both SMOKE=1 and SMOKE=0 and asserting no overlap on the four
    user-facing roots (predictors / onpolicy_gen / onpolicy_endpos_logp /
    premise). The shell carries the source-of-truth path logic; an `eval`
    of just the variable-assignment block keeps the test independent of
    the GPU phases.
    """
    import subprocess as _sp
    import tempfile as _tempfile

    script_path = REPO_ROOT / "scripts" / "run_issue466_pipeline.sh"
    assert script_path.exists(), script_path

    def _resolve(smoke: int) -> dict[str, str]:
        # Eval only the path-resolution block; the rest of the script
        # reaches for nvidia / vllm and we don't want those side effects.
        probe = f"""
set -eu
SMOKE={smoke}
EVAL_ROOT="eval_results/issue_466"
if [[ "$SMOKE" -eq 1 ]]; then
  PRED_OUT_DIR="${{EVAL_ROOT}}/smoke/predictors"
  ONPOL_GEN_OUT_DIR="${{EVAL_ROOT}}/smoke/onpolicy_gen"
  ONPOL_LOGP_OUT_DIR="${{EVAL_ROOT}}/smoke/onpolicy_endpos_logp"
  PREMISE_OUT_DIR="${{EVAL_ROOT}}/smoke/premise"
else
  PRED_OUT_DIR="${{EVAL_ROOT}}/predictors"
  ONPOL_GEN_OUT_DIR="${{EVAL_ROOT}}/onpolicy_gen"
  ONPOL_LOGP_OUT_DIR="${{EVAL_ROOT}}/onpolicy_endpos_logp"
  PREMISE_OUT_DIR="${{EVAL_ROOT}}/premise"
fi
echo "PRED=$PRED_OUT_DIR"
echo "ONPOL_GEN=$ONPOL_GEN_OUT_DIR"
echo "ONPOL_LOGP=$ONPOL_LOGP_OUT_DIR"
echo "PREMISE=$PREMISE_OUT_DIR"
"""
        with _tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            f.write(probe)
            tmp = f.name
        try:
            out = _sp.run(["bash", tmp], capture_output=True, text=True, check=True)
        finally:
            Path(tmp).unlink(missing_ok=True)
        result: dict[str, str] = {}
        for line in out.stdout.strip().splitlines():
            k, _, v = line.partition("=")
            result[k] = v
        return result

    smoke = _resolve(1)
    full = _resolve(0)
    # The four roots must NOT collide between smoke and full.
    for key in ("PRED", "ONPOL_GEN", "ONPOL_LOGP", "PREMISE"):
        assert smoke[key] != full[key], (
            f"{key}: smoke ({smoke[key]}) collides with full ({full[key]})"
        )
        # smoke path should live under the smoke/ subtree.
        assert "/smoke/" in smoke[key], f"smoke {key} not in smoke tree: {smoke[key]}"
        # full path should NOT live under the smoke/ subtree.
        assert "/smoke/" not in full[key], f"full {key} unexpectedly under smoke/: {full[key]}"

    # Also assert the shell file ITSELF carries the same separation logic so
    # an edit that broke the separation would still fail this test (the
    # probe block above pins what we EXPECT, but we want a failure on
    # accidental drift in the real script too).
    script_text = script_path.read_text()
    assert 'PRED_OUT_DIR="${EVAL_ROOT}/smoke/predictors"' in script_text, (
        "smoke PRED_OUT_DIR no longer under smoke/ subtree in run_issue466_pipeline.sh"
    )
    assert 'PRED_OUT_DIR="${EVAL_ROOT}/predictors"' in script_text, (
        "full PRED_OUT_DIR no longer at production root in run_issue466_pipeline.sh"
    )


def test_load_slice_checkpoint_round_trips_and_rejects_mismatched_fingerprint(tmp_path):
    """``_load_slice_checkpoint`` round-trips a written checkpoint AND refuses
    to return one whose stored fingerprint disagrees with the caller's.

    Pins the round-3 wire-up: a per-slice checkpoint written with smoke-config
    must NOT be loaded for a full-config resume. Without this guard, a smoke
    run could leave behind {behavior}_{slice}.json that a later full-config
    main() loop reads + uses to skip recompute.
    """
    from issue466_predictors import (
        _load_slice_checkpoint,
        _write_slice_checkpoint_with_fingerprint,
    )

    slice_payload = {
        "per_probe_scalars": [{"probe_idx": 0, "probe": "q?", "n_pooled": 4, "mean_js": 0.20}],
        "per_position_traj": [[0.1, 0.2]],
        "slice_mean_js": 0.20,
        "slice_mean_kl_s_sprime": 0.21,
        "slice_mean_kl_sprime_s": 0.19,
        "n_probes": 1,
        "n_valid": 1,
    }
    fp_v1 = "fp_smoke_abc"
    out_path = _write_slice_checkpoint_with_fingerprint(
        tmp_path, "A_spanish_restaurants", "trigger", slice_payload, fp_v1
    )
    assert out_path.exists()
    assert out_path.name == "A_spanish_restaurants_trigger.json"

    # Round-trip with the matching fingerprint -> returns the payload.
    loaded = _load_slice_checkpoint(
        tmp_path, "A_spanish_restaurants", "trigger", expected_fingerprint=fp_v1
    )
    assert loaded is not None
    assert loaded["slice_mean_js"] == 0.20
    assert loaded["per_probe_scalars"][0]["mean_js"] == 0.20
    assert loaded["config_fingerprint"] == fp_v1

    # Mismatched fingerprint -> None (refuse to silently reuse).
    loaded_mismatch = _load_slice_checkpoint(
        tmp_path,
        "A_spanish_restaurants",
        "trigger",
        expected_fingerprint="fp_full_xyz_DIFFERENT",
    )
    assert loaded_mismatch is None, (
        "stale slice checkpoint (different fingerprint) was silently returned — "
        "this is the round-2-bug-class for the slice-checkpoint resume path"
    )

    # No file at all -> None.
    assert (
        _load_slice_checkpoint(tmp_path, "B_caps_sports", "trigger", expected_fingerprint=fp_v1)
        is None
    )

    # Loading WITHOUT specifying expected_fingerprint (legacy / test helpers)
    # returns the payload unconditionally — for compat with the existing
    # round-2 test_write_slice_checkpoint_creates_file path.
    loaded_legacy = _load_slice_checkpoint(tmp_path, "A_spanish_restaurants", "trigger")
    assert loaded_legacy is not None
    assert loaded_legacy["slice_mean_js"] == 0.20


# ── Adapter resolver: HF-Trainer intermediate-checkpoint disambiguation ───
#
# Round 5 fix for the recovery `reproduce_check` crash. The Phase 0 persist
# upload landed THREE adapter_model.safetensors paths under the
# `marker_implant_adapter/` prefix: the final top-level adapter AND two
# HF-Trainer intermediate snapshots (`checkpoint-1500/`, `checkpoint-1600/`).
# The old inline resolver hit `len(safetensors_files) > 1` and raised
# "refusing to guess"; the new helper filters `checkpoint-<N>/` subdirs
# first so the canonical top-level adapter is returned cleanly.


def test_select_adapter_leaf_picks_top_level_over_trainer_checkpoints():
    """The 3-file case from task #466 recovery: top-level + 2 checkpoint-*/."""
    from issue466_marker_logp import _select_adapter_leaf

    prefix = "issue466_i432_marker_se_9neg_zen_seed42_step1600/marker_implant_adapter"
    candidates = [
        f"{prefix}/adapter_model.safetensors",
        f"{prefix}/checkpoint-1500/adapter_model.safetensors",
        f"{prefix}/checkpoint-1600/adapter_model.safetensors",
    ]
    selected = _select_adapter_leaf(candidates)
    assert selected == f"{prefix}/adapter_model.safetensors", selected


def test_select_adapter_leaf_single_candidate_unchanged():
    """A single-candidate list (the clean case) is returned as-is."""
    from issue466_marker_logp import _select_adapter_leaf

    only = "issue466_some_run/marker_implant_adapter/adapter_model.safetensors"
    assert _select_adapter_leaf([only]) == only


def test_select_adapter_leaf_genuine_ambiguity_still_raises():
    """Two DISTINCT non-checkpoint final adapters must still raise."""
    from issue466_marker_logp import _select_adapter_leaf

    candidates = [
        "issue466_run_a/marker_implant_adapter/adapter_model.safetensors",
        "issue466_run_b/marker_implant_adapter/adapter_model.safetensors",
    ]
    with pytest.raises(RuntimeError, match="Multiple non-checkpoint"):
        _select_adapter_leaf(candidates)


def test_select_adapter_leaf_all_checkpoints_raises():
    """If the only candidates sit under checkpoint-*/ (no top-level final), raise.

    Defensive: this shouldn't happen with HF-Trainer's normal end-of-train
    behavior (it always writes a top-level snapshot), but if somehow the
    top-level upload was skipped, the resolver should not silently grab one
    of the intermediates.
    """
    from issue466_marker_logp import _select_adapter_leaf

    candidates = [
        "issue466_run/marker_implant_adapter/checkpoint-1500/adapter_model.safetensors",
        "issue466_run/marker_implant_adapter/checkpoint-1600/adapter_model.safetensors",
    ]
    with pytest.raises(RuntimeError, match="no top-level final adapter found"):
        _select_adapter_leaf(candidates)
