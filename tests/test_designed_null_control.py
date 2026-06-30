# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Σ, ρ, δ, ×, ≤, ⁻¹, ᵀ, ‖) in scientific docstrings + asserts.
"""Designed-null predictor arm — the install-leak CONTROL (plan §4d, §5, §6, Must-Fix 2).

#664 proved the 2 designed-null cells ``ic_edu_default`` (educational-code-null)
and ``tf_rev_default`` (reversed-fact-null) are INSTALL-MATCHED + SIGNAL-FREE:
their gate SNR co-varies with install magnitude exactly as real EM cells, yet
they carry no designed leakage signal. Plan §6 pre-registers: a real content
behavior's L̂ Spearman ρ vs Δs MUST EXCEED the designed-null L̂ ρ (non-overlapping
clustered CIs) for the geometry-win headline to stand.

These tests pin: (1) both null cells load through the SAME loader as content
cells with the correct tensor schema; (2) on a synthetic SIGNAL-FREE-but-
install-matched input the L̂ Spearman ρ behaves as the pre-registered null —
it does NOT manufacture a strong "real behavior" signal; install-displacement
structure may yield a non-zero ρ, but a real-signal arm constructed alongside
must score HIGHER. (3) The headline-verdict helper enforces "real must exceed
null".

These use ``tmp_path`` synthetic store cells (the loader's contract is exercised
against a fabricated tensors.pt of the documented schema — NO real HF access).
CPU-only; no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402  (installed; not a TDD-deferred dep)


class _LazyModule:
    """Proxy that imports a per-issue script on first attribute access (TDD).

    The net-new scripts do NOT exist this round, so the first ``loader.<fn>`` /
    ``predscore.<fn>`` access inside each test raises ImportError → the test
    FAILS (not skips). A module-level ``importorskip`` was rejected because it
    skips COLLECTION, so the proposed-test count could not be verified by
    approve-tests.
    """

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


# Loader + scorer live in the per-issue analysis scripts (scripts/issue666_*.py).
loader = _LazyModule("issue666_load_store")
predscore = _LazyModule("issue666_predictor")

# Store schema constants (issue664_common): 28 layers, d=3584, 50 contexts, 48 probes.
N_LAYER = 28
D = 3584
N_CTX = 50
N_PROBE = 48
DESIGNED_NULL_CELLS = ("ic_edu_default", "tf_rev_default")


def _write_fake_cell(cell_dir: Path, *, source_idx: int = 0, signal_free: bool, seed: int):
    """Fabricate a #664-schema store cell under cell_dir (tensors.pt + meta.json).

    signal_free=True → Δv(C') is pure install-displacement noise with no
    target-structured leakage signal (the designed-null shape).
    """
    import json

    rng = np.random.default_rng(seed)
    cell_dir.mkdir(parents=True, exist_ok=True)
    # Small layer/d for speed in the test fabrication; the loader's shape asserts
    # are parameterized so a smoke-sized cell is allowed (the loader validates
    # rank/axes, not the literal 28/3584 in test mode).
    nl, dd = 4, 32
    v0 = torch.from_numpy(rng.standard_normal((N_CTX, nl, dd)).astype("float32"))
    install = rng.standard_normal((nl, dd)).astype("float32") * 3.0  # ŵ install at source
    dv = rng.standard_normal((N_CTX, nl, dd)).astype("float32")
    if not signal_free:
        # Inject a target-structured leakage signal correlated across contexts.
        struct = rng.standard_normal((N_CTX, 1, 1)).astype("float32")
        dv = dv + struct * install[None]
    dv[source_idx] = install  # source anchor ĝ^real(C)=1 by construction
    v_plus = v0 + torch.from_numpy(dv)
    vpp = torch.from_numpy(
        v_plus.numpy()[:, None]
        + rng.standard_normal((N_CTX, N_PROBE, nl, dd)).astype("float32") * 0.1
    )
    v0p = torch.from_numpy(
        v0.numpy()[:, None] + rng.standard_normal((N_CTX, N_PROBE, nl, dd)).astype("float32") * 0.1
    )
    obj = {
        "v_plus": v_plus,
        "v0": v0,
        "v_plus_probe": vpp,
        "v0_probe": v0p,
        "c_C_base": torch.from_numpy(rng.standard_normal((N_CTX, nl, dd)).astype("float32")),
        "c_C_trained": torch.from_numpy(rng.standard_normal((N_CTX, nl, dd)).astype("float32")),
        "t_CB": torch.from_numpy(rng.standard_normal((nl, dd)).astype("float32")),
        "r_plus": torch.from_numpy(install),
        "context_ids": list(range(N_CTX)),
    }
    torch.save(obj, cell_dir / "tensors.pt")
    (cell_dir / "meta.json").write_text(
        json.dumps(
            {
                "behavior": "designed_null",
                "source": "default",
                "source_idx": source_idx,
                "arm": "null",
                "target_context_roles": ["source-anchor"] + ["bystander"] * (N_CTX - 1),
            }
        )
    )


# ---------------------------------------------------------------------------
# Both designed-null cells load with the correct schema.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", DESIGNED_NULL_CELLS)
def test_designed_null_cell_loads_with_correct_schema(tmp_path, cell):
    cell_dir = tmp_path / cell
    _write_fake_cell(cell_dir, signal_free=True, seed=hash(cell) % 2**31)
    loaded = loader.load_cell(cell_dir)  # loads tensors + meta; validates schema
    # Required tensors present.
    for key in ("v_plus", "v0", "v_plus_probe", "v0_probe", "c_C_base", "t_CB", "r_plus"):
        assert key in loaded, f"{cell}: missing tensor {key}"
    # Probe-split tensors have the 4-axis (n_ctx, n_probe, n_layer, d) shape.
    assert loaded["v_plus_probe"].ndim == 4
    assert loaded["v0_probe"].ndim == 4
    assert loaded["v_plus_probe"].shape[0] == N_CTX
    assert loaded["v_plus_probe"].shape[1] == N_PROBE
    # c_C_base is per-context (n_ctx, n_layer, d).
    assert loaded["c_C_base"].shape[0] == N_CTX
    assert loaded["meta"]["behavior"] == "designed_null"


def test_designed_null_cells_are_in_the_known_null_set():
    """The pipeline must know which 2 cells are the install-leak control arm."""
    assert set(predscore.DESIGNED_NULL_CELLS) == set(DESIGNED_NULL_CELLS)


# ---------------------------------------------------------------------------
# Signal-free null does not manufacture a real-behavior signal; real exceeds null.
# ---------------------------------------------------------------------------
def test_signal_free_null_does_not_beat_a_real_signal_arm(tmp_path):
    """A real-signal arm's L̂ ρ exceeds the install-matched signal-free null's ρ.

    Pre-registered (§6): install-displacement structure can give the null a
    non-zero ρ, but a genuine target-structured signal must score HIGHER. The
    test builds a matched-install signal-free cell + a matched-install real cell
    and asserts real ρ > null ρ.
    """
    null_dir = tmp_path / "ic_edu_default"
    real_dir = tmp_path / "bm_default_contra_d1_seed42"
    _write_fake_cell(null_dir, signal_free=True, seed=11)
    _write_fake_cell(real_dir, signal_free=False, seed=11)

    null_cell = loader.load_cell(null_dir)
    real_cell = loader.load_cell(real_dir)

    # Score L̂ vs the latent Δs on each (the module's per-cell scorer; identity Σ
    # is fine here — this is the relative real-vs-null comparison, not the gate test).
    null_rho = predscore.score_cell_lhat_vs_ds(null_cell, layer=2)
    real_rho = predscore.score_cell_lhat_vs_ds(real_cell, layer=2)

    assert np.isfinite(null_rho) and np.isfinite(real_rho)
    # The real arm carries an injected target-structured signal → higher ρ.
    assert real_rho > null_rho, (
        f"real-signal ρ ({real_rho:.3f}) must exceed signal-free-null ρ ({null_rho:.3f})"
    )


def test_geometry_win_verdict_requires_real_exceeds_null():
    """The headline-verdict helper enforces the §6 install-leak gate.

    real_rho with CI lower bound above the null's point estimate → 'geometry-win';
    overlapping CIs → 'install-confounded' (NOT a geometry win).
    """
    # Non-overlapping: real clearly above null.
    v1 = predscore.geometry_win_verdict(
        real_rho=0.55, real_ci=(0.40, 0.70), null_rho=0.20, null_ci=(0.05, 0.35)
    )
    assert v1 == "geometry-win"
    # Overlapping → install-confounded.
    v2 = predscore.geometry_win_verdict(
        real_rho=0.30, real_ci=(0.10, 0.50), null_rho=0.28, null_ci=(0.12, 0.44)
    )
    assert v2 == "install-confounded"
