#!/usr/bin/env python3
"""task #522 JS-correctness toy — two golden cases.

This script catches TWO bug classes the upstream reducer must avoid:

1. **JS kernel correctness** (round-2 fix, brief Major #4 + Smoke #4) — the
   per-position JS / KL kernel must compute the canonical **full-vocab
   per-position mixture base-2 Rao-Blackwellized JS**, NOT Jeffreys
   (symmetric KL on realized-token mean log-ratios). Golden test: a
   closed-form 2-vocab toy where the canonical JS has a known analytic
   value (``JS_bits ≈ 0.31127812``).
2. **Multi-response length-normalization** (round-3 fix, brief Critical #1)
   — the per-pair-per-probe reducer in ``build_js_matrix`` must take the
   mean-over-positions PER RESPONSE first, then the mean across the 2R
   mixture responses, NOT a flat mean over the concatenated positions of
   the 2R responses. With unequal response lengths (the realistic case
   for free generation across personas with persona-dependent verbosity),
   the two differ. Golden test: two responses of lengths [2, 6] where
   per-response JS = [J_A, 0], so mean-of-means = J_A/2 and the
   token-weighted (wrong) mean = (2·J_A + 6·0) / 8 = J_A/4 — a clean 2×
   discrepancy. We assert the reducer matches J_A/2 AND is NOT equal to
   J_A/4. Same check applied to the KL_AB and KL_BA directions.

The existing diagonal + symmetry smoke gates pass trivially under both
the round-1 Jeffreys bug AND the round-2 token-weighted aggregation bug,
so they did not catch either. This script is the cheap catch they missed.

Exit code 0 on PASS, 1 on FAIL. Run via:
``uv run python scripts/issue522_js_smoke_toy.py``.
"""

# ruff: noqa: RUF002, RUF003 (research notation: ×, Δ, ≈, ·, — in strings/comments)

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from issue522_js_predictor import (  # noqa: E402
    _cache_key,
    build_js_matrix,
    js_closed_form_two_vocab_toy,
    per_position_js_kl_from_logprobs,
)

TOLERANCE = 1e-6
FP32_TOL = 1e-5


def _kernel_correctness_check() -> list[str]:
    """Golden case 1 — JS kernel correctness (2-vocab closed-form).

    Returns a list of failure strings (empty on PASS); also prints the
    numerical diagnostic for the reviewer log.
    """
    failures: list[str] = []

    # 1. Closed-form reference (nats).
    js_ref_nats, kl_a_ref_nats, kl_b_ref_nats = js_closed_form_two_vocab_toy()

    # 2. Numerical reference (independent recompute in fp64).
    p_a_dbl = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    p_b_dbl = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    m_dbl = 0.5 * (p_a_dbl + p_b_dbl)
    log_p_a_dbl = p_a_dbl.clamp_min(1e-300).log()
    log_p_b_dbl = p_b_dbl.clamp_min(1e-300).log()
    log_m_dbl = m_dbl.clamp_min(1e-300).log()
    kl_a_indep = (p_a_dbl * (log_p_a_dbl - log_m_dbl)).sum().item()
    kl_b_indep = (p_b_dbl * (log_p_b_dbl - log_m_dbl)).sum().item()
    js_indep_nats = 0.5 * (kl_a_indep + kl_b_indep)
    if abs(js_ref_nats - js_indep_nats) > TOLERANCE:
        failures.append(
            "closed-form helper disagrees with independent fp64 recompute: "
            f"{js_ref_nats:.12f} vs {js_indep_nats:.12f}"
        )
        return failures

    # 3. Per-position-from-logprobs implementation (fp32, hard zero clamped via
    #    log-softmax — we use a finite log for the deterministic-zero entry).
    log_p_a = torch.tensor([[math.log(0.5), math.log(0.5)]], dtype=torch.float32)
    # log(0) → use -1e30 (a finite stand-in; per_position_js_kl computes
    # log_m via logsumexp which absorbs the dominated entry stably).
    log_p_b = torch.tensor([[0.0, -1e30]], dtype=torch.float32)
    js_pos, kl_pos_a, kl_pos_b = per_position_js_kl_from_logprobs(log_p_a, log_p_b)
    js_impl_nats = js_pos.item()
    kl_a_impl = kl_pos_a.item()
    kl_b_impl = kl_pos_b.item()

    print(
        f"closed-form (nats):  JS={js_ref_nats:.12f} "
        f"KL_A={kl_a_ref_nats:.12f} KL_B={kl_b_ref_nats:.12f}"
    )
    print(
        f"per_position_impl:   JS={js_impl_nats:.12f} KL_A={kl_a_impl:.12f} KL_B={kl_b_impl:.12f}"
    )
    print(f"JS base-2 bits:      {js_ref_nats / math.log(2.0):.12f}")

    # 4. Assertions: all three quantities match within 1e-5 (fp32 tolerance).
    if abs(js_impl_nats - js_ref_nats) > FP32_TOL:
        delta = abs(js_impl_nats - js_ref_nats)
        failures.append(
            f"JS mismatch: per_position_js_kl_from_logprobs={js_impl_nats:.12f}, "
            f"closed-form={js_ref_nats:.12f}, |Δ|={delta:.2e} > {FP32_TOL:.0e}"
        )
    if abs(kl_a_impl - kl_a_ref_nats) > FP32_TOL:
        failures.append(
            f"KL(p_A||m) mismatch: impl={kl_a_impl:.12f} vs closed-form={kl_a_ref_nats:.12f}"
        )
    if abs(kl_b_impl - kl_b_ref_nats) > FP32_TOL:
        failures.append(
            f"KL(p_B||m) mismatch: impl={kl_b_impl:.12f} vs closed-form={kl_b_ref_nats:.12f}"
        )

    # 5. Reject the Jeffreys formula explicitly. Under Jeffreys-on-realized-tokens
    #    we would have JS_jeffreys = 0.5*(mean(log p_A - log p_B) + mean(log p_B - log p_A))
    #    where the means are over realized-token samples (degenerate here since p_B
    #    has a hard zero — the Jeffreys formula gives an ill-defined +∞ vs the
    #    bounded canonical JS = 0.31 bits). The canonical estimator's bounded
    #    finite value is itself the diagnostic.
    if not math.isfinite(js_impl_nats):
        failures.append(
            f"canonical JS must be finite; got {js_impl_nats!r}. "
            "(The Jeffreys-on-realized-tokens formula blows up here; "
            "the canonical formula is bounded.)"
        )
    if js_impl_nats < 0 or js_impl_nats > math.log(2.0) + FP32_TOL:
        failures.append(
            f"canonical JS must lie in [0, ln 2 ≈ 0.6931] nats; got {js_impl_nats:.6f}."
        )

    return failures


def _build_multi_response_cache(
    j_a_nats: float,
    j_b_nats: float,
) -> tuple[dict, list[str]]:
    """Build a minimal in-memory cross-persona cache for the multi-response
    length-normalization golden case.

    Two personas (``A1``, ``B1``), one probe, ``r=2`` responses sampled per
    sample-persona. Response 0 has 2 positions with per-position JS = j_a_nats
    (uniform across the 2 positions); response 1 has 6 positions with
    per-position JS = j_b_nats (uniform across the 6 positions). KL_AB and
    KL_BA are set to twice the JS so the same length-normalization assertion
    can be applied to all three direction values.

    Mean-of-means (CORRECT — round-3 fix):
      JS = (j_a + j_a + j_b + j_b) / 4 mean responses
         = where each entry is the per-response mean-over-positions
         = (j_a_nats + j_a_nats + j_b_nats + j_b_nats) / 4
         = (2·j_a_nats + 2·j_b_nats) / 4
         = (j_a_nats + j_b_nats) / 2

    Wait — we have 2R = 4 responses total (R=2 from each sampler). Each cache
    row holds ONE response's per-position tensor. Let's place:
      - Sampler A1 (r=0): 2 positions of j_a_nats  →  per-resp mean = j_a_nats
      - Sampler A1 (r=1): 6 positions of j_b_nats  →  per-resp mean = j_b_nats
      - Sampler B1 (r=0): 2 positions of j_a_nats  →  per-resp mean = j_a_nats
      - Sampler B1 (r=1): 6 positions of j_b_nats  →  per-resp mean = j_b_nats

    Mean-of-means across the 4 responses:
      = (j_a + j_b + j_a + j_b) / 4
      = (j_a + j_b) / 2

    Token-weighted (WRONG — round-2 bug):
      total positions = 2 + 6 + 2 + 6 = 16
      flat mean = (2·j_a + 6·j_b + 2·j_a + 6·j_b) / 16
                = (4·j_a + 12·j_b) / 16
                = (j_a + 3·j_b) / 4

    With ``j_a = 0.4`` and ``j_b = 0.0`` (chosen so the discrepancy is sharp):
      mean-of-means    = (0.4 + 0.0) / 2  = 0.2   nats
      token-weighted   = (0.4 + 0.0) / 4  = 0.1   nats   (2× too small)
    """
    personas = ["A1", "B1"]
    cache: dict = {}

    # Length-2 per-position JS / KL tensors (response 0, both samplers).
    js_2pos = torch.full((2,), j_a_nats, dtype=torch.float32)
    kl_a_2pos = torch.full((2,), 2.0 * j_a_nats, dtype=torch.float32)
    kl_b_2pos = torch.full((2,), 2.0 * j_a_nats, dtype=torch.float32)
    # Length-6 per-position JS / KL tensors (response 1, both samplers).
    js_6pos = torch.full((6,), j_b_nats, dtype=torch.float32)
    kl_a_6pos = torch.full((6,), 2.0 * j_b_nats, dtype=torch.float32)
    kl_b_6pos = torch.full((6,), 2.0 * j_b_nats, dtype=torch.float32)

    # Sampler A1 — cross-pair (A1, B1).
    cache[_cache_key("A1", 0, 0, "A1", "B1")] = {
        "js": js_2pos.clone(),
        "kl_a": kl_a_2pos.clone(),
        "kl_b": kl_b_2pos.clone(),
        "n_resp": 2,
    }
    cache[_cache_key("A1", 0, 1, "A1", "B1")] = {
        "js": js_6pos.clone(),
        "kl_a": kl_a_6pos.clone(),
        "kl_b": kl_b_6pos.clone(),
        "n_resp": 6,
    }
    # Sampler B1 — cross-pair (A1, B1). Same per-response lengths so the
    # 2R mean-of-means is well-defined.
    cache[_cache_key("B1", 0, 0, "A1", "B1")] = {
        "js": js_2pos.clone(),
        "kl_a": kl_a_2pos.clone(),
        "kl_b": kl_b_2pos.clone(),
        "n_resp": 2,
    }
    cache[_cache_key("B1", 0, 1, "A1", "B1")] = {
        "js": js_6pos.clone(),
        "kl_a": kl_a_6pos.clone(),
        "kl_b": kl_b_6pos.clone(),
        "n_resp": 6,
    }

    # Diagonal cells (A1,A1) and (B1,B1) — assert_cache_coverage requires
    # every (P, q, r, other) tuple in personas × range(n_probes) × range(r)
    # × personas to be present. Diagonals get zero tensors of n_resp shape.
    empty_z2 = torch.zeros(2, dtype=torch.float32)
    empty_z6 = torch.zeros(6, dtype=torch.float32)
    for P in personas:
        cache[_cache_key(P, 0, 0, P, P)] = {
            "js": empty_z2.clone(),
            "kl_a": empty_z2.clone(),
            "kl_b": empty_z2.clone(),
            "n_resp": 2,
        }
        cache[_cache_key(P, 0, 1, P, P)] = {
            "js": empty_z6.clone(),
            "kl_a": empty_z6.clone(),
            "kl_b": empty_z6.clone(),
            "n_resp": 6,
        }

    return cache, personas


def _multi_response_length_normalization_check() -> list[str]:
    """Golden case 2 — multi-response variable-length reducer.

    Constructs a 2-persona × 1-probe × R=2 cache with response lengths [2, 6]
    and known per-position JS / KL values. Asserts ``build_js_matrix`` returns
    the **mean-of-means** value, NOT the token-weighted (flat concat) mean.

    With j_a=0.4 nats (length-2 response), j_b=0.0 nats (length-6 response),
    and KL_AB = KL_BA = 2·JS by construction:
      mean-of-means JS   = (0.4 + 0.0) / 2 = 0.20 nats = 0.20/ln(2) ≈ 0.2885 bits
      token-weighted JS  = (0.4 + 0.0) / 4 = 0.10 nats = 0.10/ln(2) ≈ 0.1443 bits
    The bug ships the second number; the fix ships the first.
    """
    failures: list[str] = []
    j_a_nats = 0.4
    j_b_nats = 0.0
    cache, personas = _build_multi_response_cache(j_a_nats, j_b_nats)
    log2 = math.log(2.0)

    # Expected (round-3 mean-of-means).
    expected_js_bits = ((j_a_nats + j_b_nats) / 2.0) / log2
    expected_kl_bits = ((2.0 * j_a_nats + 2.0 * j_b_nats) / 2.0) / log2  # 2× JS by construction

    # WRONG (round-2 token-weighted). The reducer output MUST NOT equal this.
    # token-weighted = (j_a + 3·j_b) / 4 in nats; with j_b=0, it's j_a/4.
    wrong_js_nats = (j_a_nats + 3.0 * j_b_nats) / 4.0
    wrong_js_bits = wrong_js_nats / log2

    reduced = build_js_matrix(cache=cache, personas=personas, n_probes=1, r=2)

    # Off-diagonal (A1, B1) and (B1, A1) — both should match (symmetry holds
    # because the same cache rows feed both orderings via lex-canonicalization).
    for a, b in [("A1", "B1"), ("B1", "A1")]:
        js_got = reduced["JS"][a][b]
        kl_ab_got = reduced["KL_AB"][a][b]
        kl_ba_got = reduced["KL_BA"][a][b]

        if abs(js_got - expected_js_bits) > FP32_TOL:
            failures.append(
                f"JS[{a}][{b}] length-normalization FAIL: "
                f"got={js_got:.12f} bits, expected (mean-of-means)={expected_js_bits:.12f}, "
                f"|Δ|={abs(js_got - expected_js_bits):.2e} > {FP32_TOL:.0e}"
            )
        # Catch the round-2 regression head-on: reducer must NOT equal the
        # token-weighted (wrong) mean. With j_a=0.4, j_b=0.0 the gap is 2×
        # — well above any plausible fp32 tolerance.
        if abs(js_got - wrong_js_bits) < FP32_TOL:
            failures.append(
                f"JS[{a}][{b}] equals the token-weighted (wrong) mean — "
                f"the round-2 regression has reappeared. "
                f"got={js_got:.12f}, wrong={wrong_js_bits:.12f}."
            )
        if abs(kl_ab_got - expected_kl_bits) > FP32_TOL:
            failures.append(
                f"KL_AB[{a}][{b}] length-normalization FAIL: "
                f"got={kl_ab_got:.12f}, expected={expected_kl_bits:.12f}"
            )
        if abs(kl_ba_got - expected_kl_bits) > FP32_TOL:
            failures.append(
                f"KL_BA[{a}][{b}] length-normalization FAIL: "
                f"got={kl_ba_got:.12f}, expected={expected_kl_bits:.12f}"
            )

    # Diagonal must be zero.
    for p in personas:
        if reduced["JS"][p][p] != 0.0:
            failures.append(f"JS[{p}][{p}] diagonal must be 0; got {reduced['JS'][p][p]:.12f}")

    print(f"multi-response toy (response lengths [2, 6]; j_a={j_a_nats}, j_b={j_b_nats} nats):")
    print(
        f"  mean-of-means (correct):  JS={expected_js_bits:.12f} bits  "
        f"KL_AB={expected_kl_bits:.12f}  KL_BA={expected_kl_bits:.12f}"
    )
    print(f"  token-weighted (WRONG):   JS={wrong_js_bits:.12f} bits")
    print(
        f"  build_js_matrix output:   "
        f"JS[A1][B1]={reduced['JS']['A1']['B1']:.12f}  "
        f"JS[B1][A1]={reduced['JS']['B1']['A1']:.12f}"
    )

    return failures


def main() -> int:
    """Run both golden checks; print PASS/FAIL; return exit code."""
    all_failures: list[str] = []

    print("=== Golden case 1: JS kernel correctness (2-vocab closed-form) ===")
    all_failures.extend(_kernel_correctness_check())

    print("\n=== Golden case 2: multi-response variable-length aggregation ===")
    all_failures.extend(_multi_response_length_normalization_check())

    if all_failures:
        print("\n--- FAIL ---")
        for f in all_failures:
            print(f"  {f}")
        return 1

    print(
        "\nPASS: kernel + multi-response golden cases both PASS. "
        "JS matrix uses canonical full-vocab per-position mixture (kernel) "
        "and per-response mean-of-means length-normalization (reducer)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
