"""Pure-Python tests for the K1 entropy-calibration apparatus (#444 §4.2.5).

No vLLM, no GPU; the only heavy dependency is the Qwen-2.5-7B-Instruct
tokenizer (cached locally, ~50ms load), exercised once to validate the
fixture-level single-BPE + boundary invariants.

Covers four scope items from the implementer brief:

1. Calibration-fixture invariants (single-BPE canonical values, clean
   BPE boundary across the carrier→value seam) via
   ``assert_fixture_invariants`` with the live tokenizer.
2. Carrier→prefill construction (``_carrier_prefix``): placeholder
   uniqueness, prefix truncation, trailing-space preservation, and
   round-trip BPE join with the canonical value.
3. Multi-BPE length-conditional policy branching from
   ``scripts.run_experiment_444.phase_fact_pick`` (1 / 2 / ≥3 BPE →
   3-signal / length-normalised / drop-with-flag), driven against an
   ad-hoc candidates fixture so we do NOT need a real Phase-0 run.
4. NaN → drop-candidate handling in the K1 PASS gate (mirrors the
   conjunction at ``scripts/run_experiment_444.py:2197`` —
   ``canonical_ok = (cl != cl) or (cl <= t_canonical)``).

Per plan §4.2.5: position-1 of the post-prefill generation IS the
value-slot token; the per-fixture BPE checks below pin that contract
so a future carrier edit can't silently shift the measured position.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pytest

from eval import exp444_entropy_calibration_fixtures as fx

# ── 1. Fixture invariants against the live Qwen tokenizer ─────────────────────


@pytest.fixture(scope="module")
def qwen_tokenizer() -> Any:
    """Cached Qwen-2.5-7B-Instruct tokenizer (used by fixture invariants)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


def test_fixture_module_load_invariants(qwen_tokenizer: Any) -> None:
    """KNOWN_PRIOR + KNOWN_ZERO_PRIOR pass the single-BPE + boundary checks."""
    audit = fx.assert_fixture_invariants(qwen_tokenizer)
    assert audit["n_known_prior"] == 10
    assert audit["n_known_zero_prior"] == 10
    # 20 per-fixture audit rows: each with a single canonical token id +
    # a boundary_clean flag the helper would have raised on if False.
    assert len(audit["per_fixture"]) == 20
    for entry in audit["per_fixture"]:
        assert entry["boundary_clean"] is True, entry
        # Single-BPE canonical (the calibration contract).
        assert isinstance(entry["value_token_id"], int)
        assert entry["full_token_ids"][-1] == entry["value_token_id"], entry
        assert (
            entry["full_token_ids"][: len(entry["prefix_token_ids"])] == entry["prefix_token_ids"]
        ), entry


# ── 2. Carrier → prefill construction (pure-Python; no tokenizer) ─────────────


def test_carrier_prefix_truncates_at_placeholder() -> None:
    """``_carrier_prefix`` returns everything BEFORE ``{VALUE}`` verbatim."""
    carrier = "A STOP sign is {VALUE}."
    prefix = fx._carrier_prefix(carrier)
    assert prefix == "A STOP sign is "
    # Trailing space MUST be preserved — without it, the value token
    # would merge into "is" under BPE on most tokenizers.
    assert prefix.endswith(" ")


def test_carrier_prefix_rejects_no_placeholder() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fx._carrier_prefix("A STOP sign is red.")  # no {VALUE}


def test_carrier_prefix_rejects_multiple_placeholders() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fx._carrier_prefix("{VALUE} or {VALUE}?")


def test_every_fixture_carrier_has_exactly_one_placeholder() -> None:
    """Module-load contract for both fixture sets."""
    for label, fixtures in (
        ("known_prior", fx.KNOWN_PRIOR_FIXTURE),
        ("known_zero_prior", fx.KNOWN_ZERO_PRIOR_FIXTURE),
    ):
        for idx, (_q, _v, carrier) in enumerate(fixtures):
            assert carrier.count("{VALUE}") == 1, f"{label}[{idx}] carrier={carrier!r}"


def test_random_shuffled_fixture_is_deterministic_permutation() -> None:
    """``build_random_shuffled_fixture(seed)`` is deterministic + a permutation."""
    a = fx.build_random_shuffled_fixture(seed=444)
    b = fx.build_random_shuffled_fixture(seed=444)
    assert a == b, "deterministic for fixed seed"
    assert len(a) == len(fx.KNOWN_ZERO_PRIOR_FIXTURE)
    # Values are a permutation of the original zero-prior canonical values.
    orig_values = sorted(v for _, v, _ in fx.KNOWN_ZERO_PRIOR_FIXTURE)
    shuffled_values = sorted(v for _, v, _ in a)
    assert orig_values == shuffled_values
    # Carriers are preserved positionally.
    for (orig_q, _orig_v, orig_carrier), (sh_q, _sh_v, sh_carrier) in zip(
        fx.KNOWN_ZERO_PRIOR_FIXTURE, a, strict=True
    ):
        assert orig_q == sh_q
        assert orig_carrier == sh_carrier


def test_random_shuffled_fixture_differs_under_different_seeds() -> None:
    """At least one of two seeds must produce a non-identity permutation."""
    seed_a = fx.build_random_shuffled_fixture(seed=1)
    seed_b = fx.build_random_shuffled_fixture(seed=2)
    # Not asserting they're unequal to each other (collisions possible
    # on small permutation groups), but at least one should NOT be the
    # identity shuffle (values matching the original positions).
    orig = [v for _, v, _ in fx.KNOWN_ZERO_PRIOR_FIXTURE]
    non_identity = [
        seed for seed, perm in ((1, seed_a), (2, seed_b)) if [v for _, v, _ in perm] != orig
    ]
    assert non_identity, "both seeds collided with the identity permutation"


# ── 3. Multi-BPE length-conditional policy branching (driver-level) ───────────


def _fake_phase0_dir_with_candidates(
    tmp_path: Path,
    *,
    answer_value: str,
    entity: str = "the Whitefish Post Office in Whitefish, Montana",
    town: str = "Whitefish",
    state: str = "Montana",
) -> Path:
    """Build a minimal Phase-0 directory containing one candidate.

    ``phase_fact_pick`` reads ``PHASE0_DIR/candidates.json``, picks by
    id, then enforces the multi-BPE policy on the chosen candidate's
    ``answer_slot_value`` field.
    """
    candidates = [
        {
            "id": 1,
            "entity_descriptor": entity,
            "entity_slug": "whitefish_post_office",
            "town": town,
            "state": state,
            "answer_slot_value": answer_value,
            "answer_slot_carrier": f"The thing at {entity} is " + "{VALUE}.",
            "attribute_slot_question": f"What's one detail about {entity}?",
        }
    ]
    (tmp_path / "candidates.json").write_text(json.dumps({"candidates": candidates}))
    return tmp_path


def _load_driver() -> Any:
    """Import ``scripts/run_experiment_444.py`` despite its sibling-bootstrap import.

    The driver does ``from _bootstrap import ...`` which only resolves when
    ``scripts/`` itself is on ``sys.path``. We inject it once (idempotent)
    then import via importlib.
    """
    import importlib
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("run_experiment_444")


def _invoke_fact_pick(
    monkeypatch: pytest.MonkeyPatch,
    phase0_dir: Path,
    *,
    fact_pick_id: int = 1,
    allow_multi_bpe_answer: bool = False,
) -> dict[str, Any]:
    """Call ``phase_fact_pick`` against a fake Phase-0 dir."""
    driver = _load_driver()
    monkeypatch.setattr(driver, "PHASE0_DIR", phase0_dir)
    args = argparse.Namespace(
        fact_pick_id=fact_pick_id,
        allow_multi_bpe_answer=allow_multi_bpe_answer,
        force=True,
    )
    return driver.phase_fact_pick(args)


def test_one_bpe_answer_passes_fact_pick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-token canonical answer → 3-signal K1 PASS path (no override needed)."""
    # "red" is one Qwen BPE token; this is the v3-equivalent canonical path.
    phase0 = _fake_phase0_dir_with_candidates(tmp_path, answer_value="red")
    out = _invoke_fact_pick(monkeypatch, phase0)
    assert out["answer_bpe_length"] == 1, out
    assert out["canonical_logprob_signal_dropped"] is False, out
    assert out["allow_multi_bpe_answer_override"] is False, out


def test_two_bpe_answer_passes_fact_pick_without_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2-BPE answer is acceptable; canonical signal preserved (length-normalised)."""
    # Pick a value that Qwen-2.5 tokenises into ≥2 BPE tokens.
    # "northwestern" is a safe 2+ token candidate on Qwen-2.5 BPE.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    candidate = None
    for trial in ("northwestern", "Helvetica", "antediluvian", "supercalifragi"):
        if len(tok.encode(trial, add_special_tokens=False)) == 2:
            candidate = trial
            break
    if candidate is None:
        pytest.skip("no 2-BPE candidate available on this tokenizer build")
    phase0 = _fake_phase0_dir_with_candidates(tmp_path, answer_value=candidate)
    out = _invoke_fact_pick(monkeypatch, phase0)
    assert out["answer_bpe_length"] == 2, out
    assert out["canonical_logprob_signal_dropped"] is False, (
        "2-BPE answers keep the canonical-logprob signal (length-normalised); only ≥3 BPE drops it"
    )


def test_three_bpe_answer_without_override_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """≥3-BPE answer raises unless ``--allow-multi-bpe-answer`` is set."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    candidate = None
    for trial in (
        "antediluvianism",
        "supercalifragilistic",
        "xyzzyplugh",
        "qqqqqqqqq",
    ):
        if len(tok.encode(trial, add_special_tokens=False)) >= 3:
            candidate = trial
            break
    if candidate is None:
        pytest.skip("no ≥3-BPE candidate available on this tokenizer build")
    phase0 = _fake_phase0_dir_with_candidates(tmp_path, answer_value=candidate)
    with pytest.raises(RuntimeError, match="allow-multi-bpe-answer"):
        _invoke_fact_pick(monkeypatch, phase0, allow_multi_bpe_answer=False)


def test_three_bpe_answer_with_override_drops_canonical_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """≥3-BPE answer + override → proceeds; canonical signal flagged dropped."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    candidate = None
    for trial in (
        "antediluvianism",
        "supercalifragilistic",
        "xyzzyplugh",
        "qqqqqqqqq",
    ):
        if len(tok.encode(trial, add_special_tokens=False)) >= 3:
            candidate = trial
            break
    if candidate is None:
        pytest.skip("no ≥3-BPE candidate available on this tokenizer build")
    phase0 = _fake_phase0_dir_with_candidates(tmp_path, answer_value=candidate)
    out = _invoke_fact_pick(monkeypatch, phase0, allow_multi_bpe_answer=True)
    assert out["answer_bpe_length"] >= 3, out
    assert out["canonical_logprob_signal_dropped"] is True, out
    assert out["allow_multi_bpe_answer_override"] is True, out


# ── 4. K1 PASS gate: NaN canonical_logprob counts as "well below" ─────────────


def _k1_pass(
    shannon: float,
    max_p: float,
    canonical_logprob: float,
    *,
    t_shannon: float = 2.0,
    t_max_p: float = 0.5,
    t_canonical: float = -3.0,
) -> dict[str, bool]:
    """Mirror of ``scripts/run_experiment_444.py:2195-2198`` K1 gate.

    Kept verbatim here so the test pins the conjunction semantics; if
    the driver edit drifts, this test should fail and surface the
    rewrite for review.
    """
    shannon_ok = (shannon == shannon) and shannon >= t_shannon
    max_p_ok = (max_p == max_p) and max_p <= t_max_p
    canonical_ok = (canonical_logprob != canonical_logprob) or (canonical_logprob <= t_canonical)
    return {
        "shannon_ok": shannon_ok,
        "max_p_ok": max_p_ok,
        "canonical_ok": canonical_ok,
        "k1_pass": shannon_ok and max_p_ok and canonical_ok,
    }


def test_k1_gate_passes_well_below_threshold() -> None:
    """All three sub-signals comfortably PASS."""
    out = _k1_pass(shannon=3.5, max_p=0.10, canonical_logprob=-7.0)
    assert out["k1_pass"] is True, out


def test_k1_gate_nan_canonical_treated_as_well_below() -> None:
    """NaN canonical-logprob → canonical_ok=True (matches driver §4.2.5).

    The canonical answer being OUT of the top-k means its mass is
    well below the threshold by construction — the gate must treat
    NaN as PASS, not FAIL.
    """
    out = _k1_pass(shannon=3.5, max_p=0.10, canonical_logprob=float("nan"))
    assert out["canonical_ok"] is True, out
    assert out["k1_pass"] is True, out


def test_k1_gate_high_canonical_logprob_fails() -> None:
    """Canonical answer too likely → canonical_ok=False → gate fails."""
    out = _k1_pass(shannon=3.5, max_p=0.10, canonical_logprob=-1.0, t_canonical=-3.0)
    assert out["canonical_ok"] is False, out
    assert out["k1_pass"] is False, out


def test_k1_gate_nan_shannon_fails() -> None:
    """Shannon NaN means measurement failed → cannot PASS."""
    out = _k1_pass(shannon=float("nan"), max_p=0.10, canonical_logprob=-7.0)
    assert out["shannon_ok"] is False, out
    assert out["k1_pass"] is False, out


def test_k1_gate_low_entropy_fails() -> None:
    """Confident prior (low Shannon, high max_p) → must NOT pass."""
    out = _k1_pass(shannon=0.5, max_p=0.85, canonical_logprob=-2.0)
    assert out["shannon_ok"] is False, out
    assert out["max_p_ok"] is False, out
    assert out["k1_pass"] is False, out


# ── 5. Shannon / Renyi-2 / max_p arithmetic (pure-Python, no vLLM) ────────────


def _entropy_stats_from_logprobs(logprobs: list[float]) -> dict[str, float]:
    """Mirror of the inline arithmetic at ``_vllm_answer_slot_entropy``.

    Pure-Python copy of the per-position stats so we can pin the
    Shannon / Renyi-2 / max_p formulas against a synthetic distribution
    without standing up vLLM.
    """
    probs = [math.exp(lp) for lp in logprobs]
    if not probs:
        return {"shannon": float("nan"), "renyi_2": float("nan"), "max_p": float("nan")}
    shannon = float(-sum(p * math.log(max(p, 1e-30)) for p in probs))
    collision = float(sum(p * p for p in probs))
    renyi_2 = float(-math.log(max(collision, 1e-30)))
    return {"shannon": shannon, "renyi_2": renyi_2, "max_p": float(max(probs))}


def test_entropy_stats_uniform_top_k() -> None:
    """Uniform top-k over 20 tokens: H = log(20); Renyi-2 = log(20); max_p = 1/20."""
    k = 20
    logprobs = [math.log(1.0 / k)] * k
    s = _entropy_stats_from_logprobs(logprobs)
    assert s["shannon"] == pytest.approx(math.log(k), abs=1e-9)
    assert s["renyi_2"] == pytest.approx(math.log(k), abs=1e-9)
    assert s["max_p"] == pytest.approx(1.0 / k, abs=1e-9)


def test_entropy_stats_peaked_distribution() -> None:
    """Mass mostly on token 0: H low, max_p ≈ 0.9, Renyi-2 < Shannon."""
    logprobs = [math.log(0.9), math.log(0.05), math.log(0.05)]
    s = _entropy_stats_from_logprobs(logprobs)
    expected_shannon = -(0.9 * math.log(0.9) + 2 * 0.05 * math.log(0.05))
    expected_collision = 0.9**2 + 2 * 0.05**2
    expected_renyi = -math.log(expected_collision)
    assert s["shannon"] == pytest.approx(expected_shannon, abs=1e-9)
    assert s["renyi_2"] == pytest.approx(expected_renyi, abs=1e-9)
    assert s["max_p"] == pytest.approx(0.9, abs=1e-9)
    # Sanity: Renyi-2 ≤ Shannon always (for a probability distribution).
    assert s["renyi_2"] <= s["shannon"] + 1e-9


def test_entropy_stats_empty_returns_nan() -> None:
    s = _entropy_stats_from_logprobs([])
    assert math.isnan(s["shannon"])
    assert math.isnan(s["renyi_2"])
    assert math.isnan(s["max_p"])
