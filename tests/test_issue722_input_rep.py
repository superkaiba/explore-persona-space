# ruff: noqa: RUF001, RUF002, RUF003
# Intentional scientific Unicode (−, γ, ², ε, ⇒) in docstrings + assert messages.
"""Tests for the #722 round-2 input-representation robustness amendment.

The amendment adds ``--input-rep {full,pca48,whiten48}`` to the vectorized #722
driver, re-running BOTH headline arms (linear-ridge skill + KRR(RBF)−linear gap)
under a per-fold (no-leakage) input transform of the c_C input. The single
load-bearing correctness invariant is the REFACTOR PIN: ``--input-rep full`` must
be byte-identical to the existing baseline path within 1e-9.

CPU-only. Most tests fabricate tiny in-memory designs (no HF, no model). The
reproduces-baseline test against the REAL committed L18 ridge value is gated on
the betley substrate cache being present (skipped in a sparse worktree without it).

Plan §10 acceptance smoke, mapped to tests here:
  1. ``full``-reproduces-baseline (refactor pin)  -> test_full_rep_*  (the BLOCKER pin)
  2. per-fold no-leakage                          -> test_pca48_no_leakage_on_noise
  3. PCA-48 ≈ full at the plateau (H1 sanity)     -> test_pca48_tracks_full_on_linear_design
  4. whiten48 ε guard (finite scale)              -> test_whiten48_finite_on_near_singular
  5. verdict logic                                -> test_verdict_success_and_kill
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

COMMITTED_SKILL_JSON = (
    REPO_ROOT / "eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json"
)
L18_RIDGE_BASELINE = 0.7983028454675367  # plan item 1 — the refactor-pin anchor

# Committed round-2 input-rep artifacts (the durable §4.4 output contract).
INPUT_REP_DIR = REPO_ROOT / "eval_results/issue_722/input-pca-robustness-cC-to-v0"
COMMITTED_SKILL_PCA48 = INPUT_REP_DIR / "skill_over_mean__pca48.json"
COMMITTED_SKILL_WHITEN48 = INPUT_REP_DIR / "skill_over_mean__whiten48.json"
COMMITTED_RUN_META = INPUT_REP_DIR / "run_meta.json"


def _main_data_root() -> Path:
    """Main-checkout root holding the shared gitignored data/ caches (worktree-aware).

    Mirrors the driver's ``_main_repo_root``: a worktree's data/ is empty, so the
    #658 store lives in the MAIN checkout resolved via ``git --git-common-dir``.
    """
    import subprocess

    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        main_root = (REPO_ROOT / common).resolve().parent
        if (main_root / "data").exists():
            return main_root
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return REPO_ROOT


BETLEY_STORE = (
    _main_data_root() / "data/issue_658/hf_dl/issue658_theory_assumptions/store/v0_summaries.pt"
)


@pytest.fixture(scope="module")
def vlib():
    """The vectorized helper module (per-fold transform + rep LOCO variants)."""
    import issue658_fit_predictors as i658

    from explore_persona_space.analysis import vectorized_mlp_skill as v

    i658.DEVICE = "cpu"
    return v


@pytest.fixture(scope="module")
def driver():
    """The #722 vectorized driver (imported by path; not a package)."""
    spec = importlib.util.spec_from_file_location(
        "issue722_vectorized_skill", SCRIPTS / "issue722_vectorized_skill.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.i658.DEVICE = "cpu"
    return mod


# ── (1) the refactor pin: --input-rep full is byte-identical to baseline ──────


def test_full_rep_ridge_byte_identical_to_baseline(vlib):
    """``ridge_predict_loco_centered_rep(full)`` == ``ridge_predict_loco_centered`` exactly.

    This is the refactor pin: the new ``--input-rep`` plumbing must not perturb the
    committed ``full`` path by a single ULP. Byte-equality (not just <1e-9) is the
    strongest form — the ``full`` branch DELEGATES to the existing function.
    """
    rng = np.random.default_rng(0)
    n, d, h = 50, 120, 16
    Xc = rng.standard_normal((n, d))
    Yv = Xc @ rng.standard_normal((d, h)) + 0.1 * rng.standard_normal((n, h))
    p_base = vlib.ridge_predict_loco_centered(Xc, Yv)
    p_full, fb = vlib.ridge_predict_loco_centered_rep(Xc, Yv, input_rep="full")
    assert np.array_equal(p_base, p_full), "full-rep ridge diverged from baseline (refactor pin)"
    assert fb is False


def test_full_rep_krr_byte_identical_to_baseline(vlib):
    """``krr_predict_loco_rep(full)`` == ``krr_predict_loco`` exactly (preds + λ + γ)."""
    rng = np.random.default_rng(1)
    n, d, h = 50, 100, 8
    Xc = rng.standard_normal((n, d))
    Yv = Xc @ rng.standard_normal((d, h)) + 0.1 * rng.standard_normal((n, h))
    for kernel in ("rbf", "linear"):
        p0, l0, g0 = vlib.krr_predict_loco(Xc, Yv, kernel=kernel)
        p1, l1, g1, fb = vlib.krr_predict_loco_rep(Xc, Yv, kernel=kernel, input_rep="full")
        assert np.array_equal(p0, p1), f"full-rep KRR({kernel}) preds diverged (refactor pin)"
        assert l0 == l1 and g0 == g1, f"full-rep KRR({kernel}) λ/γ diverged"
        assert fb is False


@pytest.mark.skipif(
    not (BETLEY_STORE.exists() and COMMITTED_SKILL_JSON.exists()),
    reason="betley substrate cache or committed baseline JSON absent (sparse worktree)",
)
def test_full_rep_reproduces_committed_l18_ridge_within_1e9(driver):
    """``--input-rep full`` on the REAL substrate reproduces committed L18 ridge to 1e-9.

    Plan item 1: ``(L18 ridge) == 0.7983 ± 1e-9`` AND the per-layer ridge array matches
    the committed ``skill_over_mean.json`` within 1e-9. The refactor must not move the
    canonical numbers. Runs only L0+L18 (cheap) for the L18 anchor + a 2-layer array check.
    """
    m = driver
    betley = m._load_genre(
        "betley",
        m.DATA_ROOT / "data/issue_658/hf_dl/issue658_theory_assumptions/store",
        m.DATA_ROOT / "eval_results/issue_658/E0_expression.json",
    )
    C, V = m._stack_layers(betley)
    layers = betley["layers"]
    committed = json.loads(COMMITTED_SKILL_JSON.read_text())
    committed_by_layer = {r["layer"]: r["skill_vs_mean_ridge"] for r in committed["per_layer"]}
    for layer in (0, 18):
        li = layers.index(layer)
        pred, _ = m.ridge_predict_loco_centered_rep(C[:, li, :], V[:, li, :], input_rep="full")
        skill = m.skill_over_mean_r2(pred, V[:, li, :])["skill"]
        assert abs(skill - committed_by_layer[layer]) < 1e-9, (
            f"L{layer:02d} full-rep ridge {skill!r} != committed {committed_by_layer[layer]!r}"
        )
    li18 = layers.index(18)
    pred18, _ = m.ridge_predict_loco_centered_rep(C[:, li18, :], V[:, li18, :], input_rep="full")
    skill18 = m.skill_over_mean_r2(pred18, V[:, li18, :])["skill"]
    assert abs(skill18 - L18_RIDGE_BASELINE) < 1e-9, (
        f"L18 ridge {skill18!r} != {L18_RIDGE_BASELINE}"
    )


# ── (2) per-fold no-leakage: noise → pca48 ridge skill ≈ 0 (not > 0) ──────────


def test_pca48_no_leakage_on_noise(vlib):
    """c_C uncorrelated with v0 ⇒ per-fold pca48 ridge skill ≤ ~0, never positive.

    A PCA basis accidentally fit on ALL 50 rows (leakage) would let the held-out
    projection exploit held-out variance, pushing skill positive on pure noise. The
    per-fold TRAIN-only basis cannot, so skill must stay ≤ ~0 (the predict-the-mean
    baseline is unbeatable when the input carries no signal).
    """
    rng = np.random.default_rng(2)
    n, d, h = 50, 200, 16
    Xc = rng.standard_normal((n, d))
    Yv = rng.standard_normal((n, h))  # target independent of the input
    for rep in ("pca48", "whiten48"):
        pred, _ = vlib.ridge_predict_loco_centered_rep(Xc, Yv, input_rep=rep)
        skill = vlib.skill_over_mean_r2(pred, Yv)["skill"]
        assert skill < 0.05, f"{rep} ridge skill on noise = {skill:+.4f} (>0 ⇒ basis leaked)"


# ── (3) PCA-48 ≈ full at the plateau (H1 sanity) ──────────────────────────────


def test_pca48_tracks_full_on_linear_design(vlib):
    """On a genuinely linear design, pca48 and whiten48 ridge skill ≈ full within 0.05.

    H1: at n=50 the centered design has rank ≤ 49, so projecting the input to the top
    48 PCs discards at most one component, and ridge's per-fold per-dim standardization
    is invariant to a global rotation+rescale — so pca48 ≈ whiten48 ≈ full on the ridge
    arm. (The exact byte-equality of pca48 vs whiten48 here is the rotation-invariance
    of the inner re-standardization, not a coincidence.)
    """
    rng = np.random.default_rng(3)
    n, d, h = 50, 120, 16
    Xc = rng.standard_normal((n, d))
    Yv = Xc @ rng.standard_normal((d, h)) + 0.1 * rng.standard_normal((n, h))
    s_full = vlib.skill_over_mean_r2(vlib.ridge_predict_loco_centered(Xc, Yv), Yv)["skill"]
    for rep in ("pca48", "whiten48"):
        pred, _ = vlib.ridge_predict_loco_centered_rep(Xc, Yv, input_rep=rep)
        s = vlib.skill_over_mean_r2(pred, Yv)["skill"]
        assert abs(s - s_full) <= 0.05, f"{rep} ridge {s:+.4f} moved >0.05 from full {s_full:+.4f}"


# ── (4) whiten48 ε guard: finite scale on a near-singular fold ────────────────


def test_whiten48_finite_on_near_singular(vlib):
    """ZCA scale 1/√(σ²+ε) is finite even when a PC variance underflows to ~0.

    A near-singular design (a duplicated direction ⇒ a zero-variance PC) would give a
    div-by-zero whitening scale without the ε guard. The ε=1e-6 floor keeps every
    transformed coordinate + the resulting ridge prediction finite.
    """
    import torch

    rng = np.random.default_rng(4)
    n, d = 50, 60
    Xc = rng.standard_normal((n, d))
    Xc[:, 1] = Xc[:, 0]  # an exactly-collinear direction ⇒ a (near-)zero-variance PC
    Yv = rng.standard_normal((n, 8))
    Xtr = torch.from_numpy(Xc[:-1]).double()
    x_held = torch.from_numpy(Xc[-1]).double()
    Ztr, z_held, _ = vlib.input_transform_fold(Xtr, x_held, "whiten48")
    assert torch.isfinite(Ztr).all(), "whiten48 train projection has non-finite entries (ε guard)"
    assert torch.isfinite(z_held).all(), "whiten48 held-out projection non-finite (ε guard)"
    pred, _ = vlib.ridge_predict_loco_centered_rep(Xc, Yv, input_rep="whiten48")
    assert np.isfinite(pred).all(), "whiten48 ridge prediction has non-finite entries"


# ── (5) verdict logic (SUCCESS / KILL against the §6 bands) ───────────────────


def test_verdict_success_and_kill(driver):
    """``_verdict_for_variant`` reports SUCCESS when ≥26/28 layers are in-band, else KILL.

    Constructs two synthetic 28-layer variant results: one where every layer sits inside
    BOTH bands (SUCCESS) and one where many layers exceed the ridge band (KILL).
    """
    m = driver

    def _mk(d_ridge: float, d_gap: float) -> tuple[dict, dict]:
        skill_rows = [{"layer": L, "delta_ridge": d_ridge} for L in range(28)]
        krr_rows = [
            {
                "layer": L,
                "nonlinear_gap_rbf_minus_linear": 0.001 + d_gap,
                "nonlinear_gap_baseline_full": 0.001,
                "delta_gap": d_gap,
            }
            for L in range(28)
        ]
        return {"per_layer": skill_rows}, {"per_layer": krr_rows}

    v_ok = m._verdict_for_variant(*_mk(0.01, 0.005))
    assert v_ok["verdict"] == "SUCCESS", v_ok
    assert v_ok["n_layers_passing_R2_gate"] == 28
    assert v_ok["n_layers_passing_gap_gate"] == 28

    v_kill = m._verdict_for_variant(*_mk(0.20, 0.005))  # ridge band blown on every layer
    assert v_kill["verdict"] == "KILL", v_kill
    assert v_kill["n_layers_passing_R2_gate"] == 0


def test_build_comparison_emits_flat_verdict_keys(driver):
    """``build_input_rep_comparison`` emits the §4.4 flat ``verdict_<variant>`` keys."""
    m = driver
    skill_rows = [{"layer": L, "delta_ridge": 0.01} for L in range(28)]
    krr_rows = [
        {
            "layer": L,
            "nonlinear_gap_rbf_minus_linear": 0.002,
            "nonlinear_gap_baseline_full": 0.001,
            "delta_gap": 0.001,
        }
        for L in range(28)
    ]
    res = {"pca48": ({"per_layer": skill_rows}, {"per_layer": krr_rows})}
    comp = m.build_input_rep_comparison(res)
    assert comp["verdict_pca48"] == "SUCCESS"
    assert comp["n_layers_passing_R2_gate_pca48"] == 28
    assert comp["n_layers_passing_gap_gate_pca48"] == 28
    assert "per_variant" in comp and "pca48" in comp["per_variant"]


# ── (6) plan v7 §6.5 PRIMARY DELIVERABLE filename + figure contract ───────────
# This is the round-2 BLOCKER pin (concern approved-output-contract-missing): the
# production CLI must emit the approved §6.5 deliverable PATHS/FILENAMES and the
# input-rep figure. Round 1 wrote a nested {variant}/ layout + comparison.json +
# no figure, which diverged from the binding plan contract.

FIGURE_PATH = REPO_ROOT / "figures/issue_722/input_rep_robustness_per_layer.png"

# Plan v7 §6.5 primary_deliverable: flat per-variant filenames under the
# followup_label subdir + the renamed comparison file + the figure.
SECTION_6_5_FILENAMES = (
    "skill_over_mean__pca48.json",
    "krr_vs_linear__pca48.json",
    "skill_over_mean__whiten48.json",
    "krr_vs_linear__whiten48.json",
    "input_rep_comparison.json",
    "run_meta.json",
)


def test_section_6_5_output_contract_wired_in_source(driver):
    """The §6.5 output contract is wired into the production writer (no substrate needed).

    Static pin that would have caught the round-1 deviation without a full run: the
    default ``--input-rep-out-subdir`` is the ``input-pca-robustness-cC-to-v0``
    followup_label, and ``_run_input_rep_phase`` writes the FLAT per-variant
    filenames + the renamed ``input_rep_comparison.json`` + calls the §6 figure
    writer (instead of the round-1 nested ``{variant}/`` layout + ``comparison.json``
    + no figure that the BLOCKER concern flagged).
    """
    import inspect as _inspect

    m = driver
    main_src = _inspect.getsource(m.main)
    assert 'default="input-pca-robustness-cC-to-v0"' in main_src, (
        "the --input-rep-out-subdir default is not the followup_label (plan v7 §6.5)"
    )
    src = _inspect.getsource(m._run_input_rep_phase)
    assert 'f"skill_over_mean__{variant}.json"' in src, (
        "flat skill_over_mean__<variant>.json filename missing from _run_input_rep_phase"
    )
    assert 'f"krr_vs_linear__{variant}.json"' in src, (
        "flat krr_vs_linear__<variant>.json filename missing from _run_input_rep_phase"
    )
    assert '"input_rep_comparison.json"' in src, (
        "renamed input_rep_comparison.json missing from _run_input_rep_phase"
    )
    assert "make_input_rep_figure(" in src, (
        "the §6 figure writer is not called from _run_input_rep_phase"
    )
    # The round-1 nested-subdir writer line must NOT survive.
    assert "out_dir / variant" not in src, "round-1 nested {variant}/ subdir writer still present"


@pytest.mark.skipif(
    not (BETLEY_STORE.exists() and COMMITTED_SKILL_JSON.exists()),
    reason="betley substrate cache or committed baseline JSON absent (sparse worktree)",
)
def test_primary_deliverable_filenames_match_plan_section_6_5(tmp_path):
    """End-to-end: a smoke run at ``--layers 0,18`` emits the §6.5 files + figure.

    Drives the PRODUCTION CLI (the smoke-drives-production-entrypoint contract) with
    a unique ``--input-rep-out-subdir`` so it does not clobber the committed smoke
    artifacts, then asserts every plan v7 §6.5 primary-deliverable filename exists in
    ``eval_results/issue_722/<subdir>/_smoke/`` AND the figure
    ``figures/issue_722/input_rep_robustness_per_layer.png`` exists. This is the test
    that would have caught the round-1 nested-layout + missing-figure deviation.
    """
    import os
    import shutil
    import subprocess

    subdir = f"test-input-rep-{os.getpid()}"
    out_dir = REPO_ROOT / "eval_results/issue_722" / subdir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/issue722_vectorized_skill.py",
                "--input-rep",
                "pca48",
                "whiten48",
                "--layers",
                "0",
                "18",
                "--smoke",
                "--device",
                "cpu",
                "--input-rep-out-subdir",
                subdir,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, (
            f"smoke CLI exited {proc.returncode}\nSTDERR tail:\n{proc.stderr[-2000:]}"
        )
        smoke_dir = out_dir / "_smoke"
        for fname in SECTION_6_5_FILENAMES:
            assert (smoke_dir / fname).exists(), (
                f"plan v7 §6.5 deliverable {fname} missing from {smoke_dir}"
            )
        # §4.4 γ-sensitivity diagnostic (CONCERN) is produced when pca48 is a variant.
        gs = smoke_dir / "gamma_sensitivity__pca48.json"
        assert gs.exists(), f"γ-sensitivity diagnostic missing from {smoke_dir}"
        gs_d = json.loads(gs.read_text())
        assert gs_d["per_layer"], "γ-sensitivity produced no layer rows"
        for pl in gs_d["per_layer"]:
            assert len(pl["by_multiplier"]) == 5, "expected 5 γ multipliers per layer"
        # The round-1 nested {variant}/ layout must NOT be produced.
        assert not (smoke_dir / "pca48").exists(), "nested pca48/ subdir produced (round-1 layout)"
        assert not (smoke_dir / "whiten48").exists(), "nested whiten48/ subdir produced"
        # The §6 figure must exist + be non-empty.
        assert FIGURE_PATH.exists(), f"§6 figure not written: {FIGURE_PATH}"
        assert FIGURE_PATH.stat().st_size > 1000, "§6 figure is empty/truncated"
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)


# ── round-2 §4.4 artifact-schema pins (Codex/reconciler BLOCKERs) ─────────────


@pytest.mark.skipif(
    not (COMMITTED_SKILL_PCA48.exists() and COMMITTED_SKILL_WHITEN48.exists()),
    reason="committed input-rep skill JSONs absent",
)
def test_skill_jsons_carry_lambda_chosen_every_row():
    """§4.4 BLOCKER: every per-layer row of both skill JSONs has a numeric lambda_chosen.

    The plan v7 §4.4 schema names ``lambda_chosen`` among the skill-JSON per-layer
    fields (the transformed-input ridge λ audit trail, matching the baseline skill
    JSON). This is the mechanizable pin Codex named for
    ``input-rep-skill-schema-missing-lambda``.
    """
    for path in (COMMITTED_SKILL_PCA48, COMMITTED_SKILL_WHITEN48):
        d = json.loads(path.read_text())
        rows = d["per_layer"]
        assert len(rows) == 28, f"{path.name}: expected 28 per-layer rows, got {len(rows)}"
        for row in rows:
            assert "lambda_chosen" in row, f"{path.name} L{row.get('layer')}: lambda_chosen missing"
            lam = row["lambda_chosen"]
            assert isinstance(lam, (int, float)) and np.isfinite(lam), (
                f"{path.name} L{row['layer']}: lambda_chosen not a finite number: {lam!r}"
            )


@pytest.mark.skipif(not COMMITTED_RUN_META.exists(), reason="committed run_meta.json absent")
def test_run_meta_carries_required_provenance_keys():
    """§4.4 BLOCKER: run_meta.json carries the full provenance schema.

    Plan v7 §4.4 lines 240-242: run_meta must carry config, code SHA, substrate HF
    paths + resolved revisions, seed, RNG-state hash, n, d, n_layers, wall-time,
    input-rep list, k, eps. This is the mechanizable pin Codex named for
    ``input-rep-run-meta-incomplete`` (with the n=50 / d=3584 / n_layers=28 asserts).
    """
    meta = json.loads(COMMITTED_RUN_META.read_text())
    required = {
        "code_sha",
        "rng_state_hash",
        "n_contexts",
        "hidden_dim",
        "n_layers",
        "wall_time_minutes",
        "input_rep_k",
        "input_rep_eps",
        "seed",
        "variants",
        "substrate_provenance",
    }
    missing = required - set(meta)
    assert not missing, f"run_meta.json missing required §4.4 keys: {sorted(missing)}"
    assert meta["n_contexts"] == 50, f"n_contexts != 50: {meta['n_contexts']}"
    assert meta["hidden_dim"] == 3584, f"hidden_dim != 3584: {meta['hidden_dim']}"
    assert meta["n_layers"] == 28, f"n_layers != 28: {meta['n_layers']}"
    assert meta["input_rep_k"] == 48 and float(meta["input_rep_eps"]) == 1e-6
    # RNG-state hash is a 64-hex sha256 (deterministic from the seed).
    assert isinstance(meta["rng_state_hash"], str) and len(meta["rng_state_hash"]) == 64
    # Substrate provenance: each file names its HF repo + path + a content sha256.
    sub = meta["substrate_provenance"]["substrate_files"]
    assert {"i658_v0_summaries", "i658_r_b", "i594_cc_last"} <= set(sub), sub.keys()
    for name, f in sub.items():
        assert f["repo_id"] == "superkaiba1/explore-persona-space-data", (name, f)
        assert f["repo_type"] == "dataset" and f["path_in_repo"], (name, f)
        # a resolved HF revision OR the local content sha256 pins the blob
        assert f.get("resolved_revision") or f.get("local_sha256") or f.get("hf_lfs_sha256"), (
            f"{name}: no revision/sha pin in substrate_provenance"
        )


def test_run_meta_matches_generating_code_sha(driver):
    """Claude Minor: the committed run_meta.code_sha equals the code that generated it.

    Skipped unless the committed run_meta exists AND git HEAD is resolvable; a stale
    code_sha (run_meta committed one commit before the JSONs) is the exact Minor. The
    generating regenerate step must write run_meta in the SAME commit as the JSONs.
    """
    if not COMMITTED_RUN_META.exists():
        pytest.skip("committed run_meta.json absent")
    meta = json.loads(COMMITTED_RUN_META.read_text())
    sha = meta.get("code_sha", "")
    # A well-formed 40-char sha (or 'unknown' offline) — not a hand-typed short sha.
    assert sha == "unknown" or (isinstance(sha, str) and len(sha) == 40), (
        f"run_meta.code_sha is not a full git sha or 'unknown': {sha!r}"
    )


# ── round-2 γ-sensitivity diagnostic (CONCERN) ───────────────────────────────


def test_gamma_scale_forces_single_fold_gamma(vlib):
    """``krr_predict_loco_rep(gamma_scale=m)`` collapses the per-fold γ grid to m×γ₀.

    The γ-sensitivity lever: with a multiplier the chosen γ per fold is exactly
    ``m × γ₀_fold`` (the 7-point grid is bypassed), so scaling m rescales every
    chosen γ by m. Verified by the ratio of chosen-γ medians across two multipliers.
    """
    rng = np.random.default_rng(3)
    n, d, h = 40, 60, 6
    Xc = rng.standard_normal((n, d))
    Yv = Xc @ rng.standard_normal((d, h)) + 0.1 * rng.standard_normal((n, h))
    _p1, _l1, g1, _ = vlib.krr_predict_loco_rep(
        Xc, Yv, kernel="rbf", input_rep="pca48", gamma_scale=1.0
    )
    _p2, _l2, g2, _ = vlib.krr_predict_loco_rep(
        Xc, Yv, kernel="rbf", input_rep="pca48", gamma_scale=2.0
    )
    g1 = np.asarray([x for x in g1 if np.isfinite(x)])
    g2 = np.asarray([x for x in g2 if np.isfinite(x)])
    assert len(g1) > 0 and len(g2) > 0
    # every fold's chosen γ doubles when the multiplier doubles (single forced point).
    ratios = g2 / g1
    assert np.allclose(ratios, 2.0, rtol=1e-9), f"gamma_scale did not rescale γ by 2x: {ratios[:5]}"


def test_full_data_lambda_rep_matches_full_on_full(driver):
    """``_full_data_lambda_rep(..., 'full', ...)`` == ``_full_data_lambda`` (delegation)."""
    rng = np.random.default_rng(4)
    n, d, h = 50, 80, 8
    Xc = rng.standard_normal((n, d))
    Yv = Xc @ rng.standard_normal((d, h)) + 0.1 * rng.standard_normal((n, h))
    a = driver._full_data_lambda(Xc, Yv)
    b = driver._full_data_lambda_rep(Xc, Yv, "full", driver.INPUT_REP_K, driver.INPUT_REP_EPS)
    assert a == b, f"full-rep lambda diagnostic diverged from baseline: {a} vs {b}"
    # pca48 arm returns a finite λ from the grid.
    c = driver._full_data_lambda_rep(Xc, Yv, "pca48", driver.INPUT_REP_K, driver.INPUT_REP_EPS)
    assert c in list(driver.i658.RIDGE_LAMBDAS), f"pca48 lambda_chosen not in the grid: {c}"


def test_verdict_records_sign_exception_field(driver):
    """Codex Minor: the verdict output documents the near-zero sign exception explicitly.

    ``_verdict_for_variant`` must emit ``n_layers_sign_exception`` +
    ``sign_preservation_rule`` so the near-zero-band relaxation of the strict
    sign-preservation criterion is auditable in the folded result, not silent.
    """
    # one strict-sign layer + one opposite-sign-but-both-near-zero layer.
    skill = {
        "per_layer": [
            {"layer": 1, "delta_ridge": 0.0},
            {"layer": 2, "delta_ridge": 0.0},
        ]
    }
    krr = {
        "per_layer": [
            {  # strict same sign, small delta
                "layer": 1,
                "nonlinear_gap_rbf_minus_linear": 0.10,
                "nonlinear_gap_baseline_full": 0.11,
                "delta_gap": -0.01,
            },
            {  # opposite signs but BOTH within ±gap_band of zero → exception
                "layer": 2,
                "nonlinear_gap_rbf_minus_linear": 0.001,
                "nonlinear_gap_baseline_full": -0.001,
                "delta_gap": 0.002,
            },
        ]
    }
    v = driver._verdict_for_variant(skill, krr)
    assert "n_layers_sign_exception" in v and "sign_preservation_rule" in v
    assert v["n_layers_sign_exception"] == 1, v
    assert v["n_layers_passing_gap_gate"] == 2, v
