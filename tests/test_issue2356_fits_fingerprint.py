"""Pin: the fits phase-fingerprint embeds the REAL git commit sha (c24).

Regression for a latent unit-2 bug found by the unit-3 resume-matrix smoke:
``_phase_fingerprint`` read ``git_provenance().sha`` — an attribute that does
not exist (the field is ``commit_sha``) — so the ``except Exception`` degrade
branch (meant for git-less SLURM scratch trees) fired on EVERY run and pinned
``git_sha="unavailable-no-git-checkout"``, making the fingerprint's
code-identity leg permanently inert (a code change never forced a recompute).
This test fails pre-fix (both fingerprints equal under different shas) and
passes post-fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_fits as fits  # noqa: E402


def _args() -> object:
    return fits.build_argparser().parse_args(["--phase", "groups"])


def test_fingerprint_embeds_real_commit_sha(monkeypatch) -> None:
    from explore_persona_space.orchestrate import provenance

    def _fake(sha: str):
        return provenance.GitProvenance(commit_sha=sha, dirty=False, dirty_paths=[])

    monkeypatch.setattr(provenance, "git_provenance", lambda cwd=None: _fake("aaaa111"))
    fp_a = fits._phase_fingerprint(_args(), "groups", {"m": "x"})
    monkeypatch.setattr(provenance, "git_provenance", lambda cwd=None: _fake("bbbb222"))
    fp_b = fits._phase_fingerprint(_args(), "groups", {"m": "x"})
    assert fp_a != fp_b, (
        "phase fingerprint did not change with the git commit sha — the c24 "
        "code-identity leg is inert (the degrade branch is swallowing it)"
    )


def test_fingerprint_still_keys_inputs_and_flags() -> None:
    a = fits._phase_fingerprint(_args(), "groups", {"m": "x"})
    b = fits._phase_fingerprint(_args(), "groups", {"m": "y"})
    assert a != b
    args2 = fits.build_argparser().parse_args(["--phase", "groups", "--n-folds", "4"])
    c = fits._phase_fingerprint(args2, "groups", {"m": "x"})
    assert a != c


# ---------------------------------------------------------------------------
# Round-2 pins: A1 (plan lambda grid), A2 (group bootstrap), A3 (modal layer),
# A4 (registered contrast set)
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

import numpy as np  # noqa: E402


def test_plan_lambda_grid_is_registered_and_passed() -> None:
    """A1 pin: the map-fit lambda grid is logspace(-2,4,13) and phase_maps
    passes it EXPLICITLY (the primal core's 6-point default is never used)."""
    assert np.allclose(fits.RIDGE_LAMBDAS_PLAN, np.logspace(-2.0, 4.0, 13))
    src = inspect.getsource(fits.phase_maps)
    assert "lambdas=RIDGE_LAMBDAS_PLAN" in src, "phase_maps must pass the plan grid explicitly"


def test_battery_s2_gate_uses_group_bootstrap() -> None:
    """A2 pin (fails pre-fix): the S2 gate resamples eval GROUPS; with flags
    perfectly correlated within groups the group-bootstrap 5th-pct lower bound
    sits BELOW the row-iid one (row-iid was anti-conservatively narrow)."""
    rng = np.random.default_rng(0)
    d = 3
    pool = rng.normal(size=(24, d)) * 5.0  # well-separated pool rows
    true_idx = np.arange(12)
    pred = pool[true_idx].copy()
    # bad group g2 (targets 8..11): point EXACTLY at a different pool row
    pred[8:12] = pool[20:24]
    groups = ["g0"] * 4 + ["g1"] * 4 + ["g2"] * 4
    n_boot, boot_seed = 2000, 123
    # R3-4 signature: identity whitening over ONE part (all targets) keeps the
    # A2 semantics this pin tests (group vs row-iid bootstrap) unchanged.
    res = fits._battery_metrics(
        pred,
        pool,
        true_idx,
        whiten_parts=[(np.arange(len(true_idx)), np.zeros(d), np.eye(d))],
        groups=groups,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    gate = res["s2_gate"]
    assert gate["bootstrap"] == "group" and gate["n_groups"] == 3, gate
    flags = np.array(res["_acc1_flags_whitened"], dtype=np.float64)
    assert flags[:8].all() and not flags[8:].any(), flags  # planted 2 good / 1 bad group
    # row-iid CI (the pre-fix computation) for comparison, same seed/draws
    rng2 = np.random.default_rng(boot_seed)
    draws = rng2.integers(0, len(flags), size=(n_boot, len(flags)))
    row_iid_ci = float(np.percentile(flags[draws].mean(axis=1), 5.0))
    assert gate["ci_lower_5pct"] < row_iid_ci, (gate["ci_lower_5pct"], row_iid_ci)


def test_modal_layer_tie_breaks_smallest() -> None:
    assert fits._modal_layer([3, 3, 5, 5, 7]) == 3
    assert fits._modal_layer([5]) == 5
    assert fits._modal_layer([7, 4, 7]) == 7


def test_registered_contrast_set_matches_plan() -> None:
    """A4 pin: registered = {#2-#1, #3a-#2, #3b-#3a, #4-#3a, #3a-PCA,
    #2-text_surface}; ans_minus_ctx (#4-#2) is present but unregistered."""
    registered = {n for n, _, _, reg in fits.CONTRAST_SPECS if reg}
    assert registered == {
        "delta_int",
        "ctx_minus_text_surface",
        "map3a_minus_ctx",
        "map3a_minus_pca",
        "map3b_minus_map3a",
        "ans_minus_map3a",
    }
    assert {n for n, _, _, reg in fits.CONTRAST_SPECS if not reg} == {"ans_minus_ctx"}
    spec = {n: (a, b) for n, a, b, _ in fits.CONTRAST_SPECS}
    assert spec["map3a_minus_ctx"] == (fits.PRED_3A, fits.PRED_CTX)
    assert spec["ans_minus_map3a"] == (fits.PRED_ANS, fits.PRED_3A)
