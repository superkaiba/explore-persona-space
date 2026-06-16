"""Tests for issue #649 Phase-1 extractor invariants (NO model / NO HF).

Pins:
  - the probe bank is >= 2*k (k=16) so the Gaussian-KL k=16 PCA subspace covariance
    is non-singular (Risk-row-5 / Assumption 7);
  - the persona-distance bank cosine uses global-mean centering (the canonical
    recipe in .claude/rules/persona-distance-metrics.md);
  - the layer cells are the #509 early band (end_of_system L2 / last_prompt L7).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))


def _load_extractor():
    spec = importlib.util.spec_from_file_location(
        "issue649_extract_panel_earlylayer", SCRIPTS / "issue649_extract_panel_earlylayer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_decomp():
    spec = importlib.util.spec_from_file_location(
        "issue649_level_change_decomp", SCRIPTS / "issue649_level_change_decomp.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_bank_at_least_2k():
    ext = _load_extractor()
    assert len(ext.PROBE_BANK) >= 2 * ext.GKL_K, (
        f"probe bank {len(ext.PROBE_BANK)} < 2*k={2 * ext.GKL_K}; KL subspace would be singular"
    )
    assert ext.N_PROBES_FULL >= 2 * ext.GKL_K
    assert ext.N_PROBES_SMOKE >= 2 * ext.GKL_K


def test_early_layer_band_cells():
    ext = _load_extractor()
    # #509 early band: end_of_system L2 (primary cosine), last_prompt L7 (secondary).
    assert ext.LAYER_EOS_PRIMARY == 2
    assert ext.LAYER_LASTPROMPT_SECONDARY == 7
    assert 2 in ext.LASTPROMPT_LAYERS and 7 in ext.LASTPROMPT_LAYERS


def test_sources_match_panel_convention():
    ext = _load_extractor()
    assert set(ext.SOURCES_ONPOLICY) <= set(ext.SOURCES_CANNED)
    assert ext.SOURCES_CANNED == (
        "villain",
        "comedian",
        "kindergarten_teacher",
        "software_engineer",
    )
    assert ext.SOURCES_ONPOLICY == ("villain", "comedian")


def test_probe_sha_deterministic():
    ext = _load_extractor()
    a = ext._probe_sha256(ext.PROBE_BANK)
    b = ext._probe_sha256(ext.PROBE_BANK)
    assert a == b and len(a) == 64


def test_centered_bank_cosine_is_global_mean():
    """The decomp's _centered_bank_cosine must global-mean-center before cosine:
    the diagonal is 1.0, and a centered cosine differs from the raw (uncentered)
    cosine on a bank with a dominant shared mean direction."""
    decomp = _load_decomp()
    rng = np.random.default_rng(3)
    # 12 personas sharing a strong mean direction (the Qwen compression regime).
    shared = np.ones((1, 64)) * 5.0
    centroids = (shared + 0.3 * rng.standard_normal((12, 64))).astype(np.float32)
    cos = decomp._centered_bank_cosine(centroids)
    assert cos.shape == (12, 12)
    assert np.allclose(np.diag(cos), 1.0, atol=1e-4)
    # raw (uncentered) cosine would be ~all-near-1 due to the shared mean; centered
    # must spread the off-diagonal well below 1.
    off = cos[~np.eye(12, dtype=bool)]
    assert off.min() < 0.95, "centering did not decompress the shared-mean bank"


def test_centered_bank_cosine_matches_project_helper():
    """_centered_bank_cosine must delegate to representation_shift.compute_cosine_matrix
    with centering='global_mean' (single-sourced recipe)."""
    import torch

    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    decomp = _load_decomp()
    rng = np.random.default_rng(4)
    centroids = rng.standard_normal((10, 48)).astype(np.float32)
    ours = decomp._centered_bank_cosine(centroids)
    ref = compute_cosine_matrix(torch.from_numpy(centroids), centering="global_mean").numpy()
    assert np.allclose(ours, ref, atol=1e-5)
