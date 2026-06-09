#!/usr/bin/env python3
"""task #522 round-2 fix — closed-form 2-vocab JS toy test.

Round-2 reviewer fix (brief Major #4 + Smoke #4): the round-1 JS estimator
computed **Jeffreys** (symmetric KL on realized-token mean log-ratios), not the
canonical **full-vocab per-position mixture base-2 Rao-Blackwellized JS**. The
existing diagonal + symmetry smoke gates pass trivially under BOTH formulations,
so they did not catch the divergence. This script is the cheap catch they missed:
a closed-form 2-vocab toy where the canonical JS has a known analytic value.

Toy
---
``p_A = [0.5, 0.5]`` and ``p_B = [1.0, 0.0]`` (one position, vocab size 2).

* ``m = ½(p_A + p_B) = [0.75, 0.25]``
* ``KL(p_A || m) = 0.5 · log(0.5 / 0.75) + 0.5 · log(0.5 / 0.25)``
* ``KL(p_B || m) = 1.0 · log(1.0 / 0.75)``
* ``JS_nats = 0.5 · (KL(p_A||m) + KL(p_B||m))``
* ``JS_bits = JS_nats / log(2) ≈ 0.31127812445913283``

The Jeffreys mean-log-ratio formula yields a structurally different value
(it sums *signed* per-token log-ratios, not per-position vocab-mass-weighted
KL contributions). The toy test thus rejects the round-1 formula and
accepts the round-2 canonical formula.

Exit code 0 on PASS, 1 on FAIL. Run via:
``uv run python scripts/issue522_js_smoke_toy.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from issue522_js_predictor import (  # noqa: E402
    js_closed_form_two_vocab_toy,
    per_position_js_kl_from_logprobs,
)

TOLERANCE = 1e-6


def main() -> int:
    """Run the closed-form check; print PASS/FAIL; return exit code."""
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
        print(
            "FAIL: closed-form helper disagrees with independent fp64 recompute: "
            f"{js_ref_nats:.12f} vs {js_indep_nats:.12f}"
        )
        return 1

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
        f"closed-form (nats):  JS={js_ref_nats:.12f} KL_A={kl_a_ref_nats:.12f} KL_B={kl_b_ref_nats:.12f}"
    )
    print(
        f"per_position_impl:   JS={js_impl_nats:.12f} KL_A={kl_a_impl:.12f} KL_B={kl_b_impl:.12f}"
    )
    print(f"JS base-2 bits:      {js_ref_nats / math.log(2.0):.12f}")

    # 4. Assertions: all three quantities match within 1e-5 (fp32 tolerance).
    fp32_tol = 1e-5
    failures: list[str] = []
    if abs(js_impl_nats - js_ref_nats) > fp32_tol:
        failures.append(
            f"JS mismatch: per_position_js_kl_from_logprobs={js_impl_nats:.12f}, "
            f"closed-form={js_ref_nats:.12f}, |Δ|={abs(js_impl_nats - js_ref_nats):.2e} > {fp32_tol:.0e}"
        )
    if abs(kl_a_impl - kl_a_ref_nats) > fp32_tol:
        failures.append(
            f"KL(p_A||m) mismatch: impl={kl_a_impl:.12f} vs closed-form={kl_a_ref_nats:.12f}"
        )
    if abs(kl_b_impl - kl_b_ref_nats) > fp32_tol:
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
    if js_impl_nats < 0 or js_impl_nats > math.log(2.0) + fp32_tol:
        failures.append(
            f"canonical JS must lie in [0, ln 2 ≈ 0.6931] nats; got {js_impl_nats:.6f}."
        )

    if failures:
        print("\n--- FAIL ---")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nPASS: 2-vocab toy JS matches canonical full-vocab per-position mixture estimator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
