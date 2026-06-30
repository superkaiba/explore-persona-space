"""Round-3 Blocker A regression — A3.9 `psi_delta` is the SOURCE DISPLACEMENT.

The plan (§3/§4 A3.7) defines δ(C) = t_CB - v0(C) at the source context, layer L:
the trained-minus-base shift in the model's OUTPUT (target) representation. Round 2
wrongly computed `psi_delta` as the context-vector drift `c_C_trained - c_C_base`
(the A3.6a context-vector-stability quantity, a DIFFERENT object). This test asserts
the VECTOR VALUE (not the label/shape): `_a39_keys` returns δ = t_CB[L] - v0[src, L],
and that this is NOT the context-vector drift.

Fails pre-fix (psi_delta == c_C drift), passes post-fix (psi_delta == source disp).
CPU-only: no HF / network / GPU — a synthetic StoreCell-like object with 4 DISTINCT
random tensors so the two candidate vectors cannot accidentally coincide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


class _SC:
    """Synthetic StoreCell-like object exposing the tensors `_a39_keys` reads, with
    FOUR distinct random tensors (t_CB, v0, c_C_base, c_C_trained) so the source
    displacement and the context-vector drift are provably different vectors."""

    def __init__(self, d=16, n_ctx=6, n_layer=4, seed=7):
        rng = np.random.default_rng(seed)
        self.source_idx = 2
        self.tensors = {
            "t_CB": torch.tensor(rng.standard_normal((n_layer, d)), dtype=torch.float32),
            "v0": torch.tensor(rng.standard_normal((n_ctx, n_layer, d)), dtype=torch.float32),
            "c_C_base": torch.tensor(rng.standard_normal((n_ctx, n_layer, d)), dtype=torch.float32),
            "c_C_trained": torch.tensor(
                rng.standard_normal((n_ctx, n_layer, d)), dtype=torch.float32
            ),
        }


def test_psi_delta_is_source_displacement_not_context_drift():
    """psi_delta == t_CB[L] - v0(C)[L] (A3.7 delta), NOT c_C_trained - c_C_base (A3.6a)."""
    import issue665_gate_cpu as G

    sc = _SC()
    L = 1
    keys = G._a39_keys(sc, L)

    t_cb = sc.tensors["t_CB"].numpy().astype(np.float64)
    v0 = sc.tensors["v0"].numpy().astype(np.float64)
    c_base = sc.tensors["c_C_base"].numpy().astype(np.float64)
    c_trn = sc.tensors["c_C_trained"].numpy().astype(np.float64)

    expected_source_disp = t_cb[L] - v0[sc.source_idx, L]  # A3.7 δ — the RIGHT value
    wrong_ctx_drift = c_trn[sc.source_idx, L] - c_base[sc.source_idx, L]  # A3.6a — round-2 bug

    # the VALUE assertion (Blocker A): psi_delta is the source displacement
    assert np.allclose(keys["psi_delta"], expected_source_disp), (
        "psi_delta must equal the SOURCE DISPLACEMENT t_CB[L]-v0(C)[L] (plan A3.7 δ)"
    )
    # negative test: psi_delta must NOT be the context-vector drift (the round-2 bug)
    assert not np.allclose(keys["psi_delta"], wrong_ctx_drift), (
        "psi_delta must NOT be the context-vector drift c_C_trained-c_C_base (A3.6a)"
    )
    # the two candidates are genuinely different here (guards a degenerate fixture)
    assert not np.allclose(expected_source_disp, wrong_ctx_drift), (
        "fixture sanity: source displacement and context drift must differ"
    )


def test_psi_t_is_co_layer_t_cb():
    """psi_t is the projection of t_CB via ψ=identity co-layer extraction = t_CB[L]
    (the plan A3.9 default; the OTHER projected key — confirmed semantically correct)."""
    import issue665_gate_cpu as G

    sc = _SC()
    L = 2
    keys = G._a39_keys(sc, L)
    t_cb = sc.tensors["t_CB"].numpy().astype(np.float64)
    assert np.allclose(keys["psi_t"], t_cb[L]), "psi_t must be the co-layer t_CB[L]"


def test_a39_psi_delta_matches_arm_a37_delta():
    """The A3.9 ψ(δ) key must be the SAME δ that arm_a37 computes (one definition of
    the source displacement across both arms — they cannot diverge)."""
    import issue665_gate_cpu as G

    sc = _SC()
    L = 3
    keys = G._a39_keys(sc, L)
    # reproduce arm_a37's delta exactly (gate_cpu.py arm_a37: t_CB[layer] - v0[src])
    v0 = sc.tensors["v0"].numpy().astype(np.float64)[:, L]
    t_cb = sc.tensors["t_CB"].numpy().astype(np.float64)
    a37_delta = t_cb[L] - v0[sc.source_idx]
    assert np.allclose(keys["psi_delta"], a37_delta), (
        "A3.9 psi_delta and arm_a37 delta must be the identical source displacement"
    )
