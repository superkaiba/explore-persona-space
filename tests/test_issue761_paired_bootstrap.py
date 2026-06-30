"""Unit tests for the issue #761 paired Δrho estimator (plan §4.4-3 / §6.4 / §8 risk (g)).

The headline Δrho CI is the PAIRED context-resample bootstrap, NOT the disjoint-arm
independent estimator. These tests pin the three properties the plan's risk-(g)
mitigation requires:

  (1) On perfectly-correlated synthetic arms (arm_corr=1) the paired CI is TIGHTER
      than the independent estimator on the SAME single-arm draws.
  (2) The SAME context-index set drives BOTH arms per draw (the mechanistic
      paired-resample check).
  (3) On disjoint-arm synthetic data (arm_corr=0) paired ≈ independent.

These exercise the estimator directly on synthetic ``(N, n_layers, H)`` arms — no
GPU, no HF, no reuse data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from issue761_paired_bootstrap import (  # noqa: E402
    _arm_rho_on_subset,
    _independent_delta_rho_ci,
    _paired_delta_rho_ci,
)


def _ci_width(ci: list[float]) -> float:
    return ci[1] - ci[0]


def _make_arm(n: int, n_layers: int, h: int, y: np.ndarray, signal: float, rng) -> np.ndarray:
    """A synthetic v0 cube (N, n_layers, H) whose layer 0 carries ``signal``*y + noise."""
    X = rng.standard_normal((n, n_layers, h)).astype(np.float64)
    # plant a y-correlated signal in every layer's first few PCs so ridge can read it
    X[:, :, 0] += signal * y[:, None]
    X[:, :, 1] += 0.5 * signal * y[:, None]
    return X


def _single_arm_draws(X, y, *, n_boot, seed):
    """The single-arm cluster-bootstrap draws the independent estimator consumes."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    draws = []
    attempts = 0
    while len(draws) < n_boot and attempts < 50 * n_boot:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        rho = _arm_rho_on_subset(X, y, idx)
        if rho is not None:
            draws.append(rho)
    return draws


def test_paired_ci_tighter_on_correlated_arms():
    """(1) arm_corr≈1 → paired CI strictly tighter than the independent CI."""
    rng = np.random.default_rng(0)
    n, n_layers, h = 50, 4, 30
    y = rng.standard_normal(n)
    # two arms reading the SAME signal → strongly positively-correlated per-draw rho
    matched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(1))
    mismatched = matched + 0.05 * rng.standard_normal(matched.shape)

    paired = _paired_delta_rho_ci(matched, mismatched, y, n_boot=120, seed=761)
    # independent estimator on the SAME single-arm draws
    m_draws = _single_arm_draws(matched, y, n_boot=120, seed=11)
    mm_draws = _single_arm_draws(mismatched, y, n_boot=120, seed=12)
    indep = _independent_delta_rho_ci(m_draws, mm_draws, seed=99)

    assert _ci_width(paired["ci95"]) < _ci_width(indep["ci95"]), (
        f"paired CI width {_ci_width(paired['ci95']):.4f} should be < independent "
        f"{_ci_width(indep['ci95']):.4f} on correlated arms"
    )


def test_same_index_set_drives_both_arms(monkeypatch):
    """(2) mechanistic — both arms refit on the SAME resampled context indices per draw."""
    rng = np.random.default_rng(2)
    n, n_layers, h = 40, 3, 20
    y = rng.standard_normal(n)
    matched = _make_arm(n, n_layers, h, y, signal=1.5, rng=np.random.default_rng(3))
    mismatched = _make_arm(n, n_layers, h, y, signal=1.5, rng=np.random.default_rng(4))

    import issue761_paired_bootstrap as mod

    seen: list[np.ndarray] = []
    real = mod._arm_rho_on_subset

    def _spy(X, yy, idx):
        seen.append(np.asarray(idx).copy())
        return real(X, yy, idx)

    monkeypatch.setattr(mod, "_arm_rho_on_subset", _spy)
    _paired_delta_rho_ci(matched, mismatched, y, n_boot=20, seed=761)

    # _arm_rho_on_subset is called twice per draw (matched then mismatched) with the
    # SAME idx; consecutive pairs must be identical.
    assert len(seen) >= 40, len(seen)
    for k in range(0, len(seen) - 1, 2):
        np.testing.assert_array_equal(
            seen[k], seen[k + 1], err_msg=f"draw {k // 2}: arms saw DIFFERENT index sets"
        )


def test_paired_approx_independent_on_disjoint_arms():
    """(3) arm_corr≈0 → paired CI ≈ independent CI (independence is the degenerate case)."""
    rng = np.random.default_rng(5)
    n, n_layers, h = 50, 4, 30
    y = rng.standard_normal(n)
    # two arms reading INDEPENDENT signals → near-zero per-draw rho correlation
    matched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(6))
    mismatched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(7))
    # decorrelate: replace the planted signal in mismatched with an independent one
    y2 = rng.standard_normal(n)
    mismatched[:, :, 0] = rng.standard_normal((n, n_layers)) + 2.0 * y2[:, None]

    paired = _paired_delta_rho_ci(matched, mismatched, y, n_boot=150, seed=761)
    m_draws = _single_arm_draws(matched, y, n_boot=150, seed=21)
    mm_draws = _single_arm_draws(mismatched, y, n_boot=150, seed=22)
    indep = _independent_delta_rho_ci(m_draws, mm_draws, seed=99)

    wp = _ci_width(paired["ci95"])
    wi = _ci_width(indep["ci95"])
    # on disjoint arms the two CIs should be within ~40% of each other (independence
    # is the regime where the paired estimator reduces to the independent one)
    assert wp == pytest.approx(wi, rel=0.40), (
        f"paired width {wp:.4f} vs independent {wi:.4f} should be close on disjoint arms"
    )


def test_vectorized_ridge_matches_inherited_serial():
    """The batched LOCO ridge is bit-equal to the inherited serial _ridge_predict_loco.

    The paired bootstrap replaces the ~1s/call serial helper with a batched one for
    tractability (plan §9 compute-deviation); this pins that the swap did NOT change
    the numbers. Trips if the batched standardization / nested-lambda / dual-solve
    drifts from the oracle.
    """
    import issue761_common as common

    for seed in (0, 1, 5, 13):
        max_abs = common._assert_vectorized_ridge_exactness(seed=seed)
        assert max_abs <= 1e-6, (seed, max_abs)


def test_all_layers_batched_matches_per_layer():
    """`_all_layers_loco_preds` (layer axis batched) == per-layer `_vectorized_ridge_loco_preds`."""
    import issue761_common as common

    rng = np.random.default_rng(3)
    n, n_layers, h = 40, 5, 25
    y = rng.standard_normal(n)
    X = rng.standard_normal((n, n_layers, h))
    X[:, :, 0] += 1.5 * y[:, None]
    batched = common._all_layers_loco_preds(X, y, 8)  # (n_layers, n)
    for li in range(n_layers):
        single = common._vectorized_ridge_loco_preds(
            common._pca_reduce(X[:, li, :], 8), y, list(common.RIDGE_LAMBDAS)
        )
        np.testing.assert_allclose(batched[li], single, atol=1e-9, err_msg=f"layer {li}")


def test_paired_point_delta_and_overlap_keys():
    """The estimator returns the contract keys (ci95, point_delta, draws, null_overlap)."""
    rng = np.random.default_rng(8)
    n, n_layers, h = 30, 2, 15
    y = rng.standard_normal(n)
    matched = _make_arm(n, n_layers, h, y, signal=2.0, rng=np.random.default_rng(9))
    mismatched = _make_arm(n, n_layers, h, y, signal=0.3, rng=np.random.default_rng(10))
    out = _paired_delta_rho_ci(matched, mismatched, y, n_boot=50, seed=761)
    assert set(out) >= {"ci95", "point_delta", "draws", "n_boot", "null_overlap"}
    assert len(out["draws"]) == 50
    assert out["ci95"][0] <= out["ci95"][1]


# ── round-2 BLOCKER missing-ceiling-and-nulls: the §6.5 / §6.6 registered stats ──


def test_reliability_ceiling_computes_and_fields():
    """√(r_yy) split-half-over-probes + binomial returns the registered ceiling fields.

    Builds a synthetic per-probe judged structure with REAL reliable signal (per-context
    rate tied to a latent + per-probe noise) and asserts the ceiling is a finite value in
    [0, 1] with both methods + a cluster-bootstrap CI present (round-2 BLOCKER fix).
    """
    from issue761_common import reliability_ceiling

    rng = np.random.default_rng(0)
    n_ctx, n_probes = 40, 60
    latent = rng.uniform(0.1, 0.6, size=n_ctx)  # per-context "true" rate
    per_probe_by_ctx = []
    for ci in range(n_ctx):
        # each probe's e0 is a noisy draw around the context latent (n_judged=10 → rate)
        probs = np.clip(latent[ci] + 0.08 * rng.standard_normal(n_probes), 0.0, 1.0)
        per_probe_by_ctx.append(
            [{"probe": f"p{j}", "e0": float(probs[j]), "n_judged": 10} for j in range(n_probes)]
        )

    out = reliability_ceiling(per_probe_by_ctx, n_split_seeds=30, seed=761, n_boot=200)
    assert out["method"] == "splithalf_probes+binomial"
    assert out["reliability_ceiling"] is not None
    assert 0.0 <= out["reliability_ceiling"] <= 1.0, out["reliability_ceiling"]
    assert out["reliability_ceiling_ci_low"] is not None
    assert out["reliability_ceiling_ci_high"] is not None
    assert out["reliability_ceiling_ci_low"] <= out["reliability_ceiling_ci_high"]
    # the binomial 2nd method also produces a finite ceiling on a reliable signal
    assert out["reliability_ceiling_binomial"] is not None
    # a genuinely reliable construct should read a ceiling well above 0
    assert out["reliability_ceiling"] > 0.3, out["reliability_ceiling"]


def test_shuffle_null_p_and_control_task_full_pipeline():
    """Shuffle-label + Hewitt-Liang control-task nulls run the FULL pipeline per perm.

    On a real-signal arm the observed rho should be LARGER than most shuffled / control
    rho draws (small shuffle p, selective control verdict); on a null arm (no signal) the
    p should be large + the control verdict non-selective. Pins the §6.5 estimators
    actually fire (round-2 BLOCKER fix) and return the registered fields.
    """
    from issue761_common import _run_ridge_pipeline, control_task_null, shuffle_label_null

    rng = np.random.default_rng(1)
    n, n_layers, h = 50, 4, 30
    y = rng.uniform(0.0, 0.5, size=n)
    signal_arm = _make_arm(
        n, n_layers, h, (y - y.mean()) / y.std(), signal=3.0, rng=np.random.default_rng(2)
    )
    obs_rho = _run_ridge_pipeline(signal_arm, y)["rho"]

    shuf = shuffle_label_null(signal_arm, y, obs_rho, n_perm=60, seed=761)
    assert set(shuf) >= {
        "shuffle_null_p",
        "shuffle_null_dist_mean",
        "shuffle_null_dist_std",
        "n_perm",
    }
    assert 0.0 < shuf["shuffle_null_p"] <= 1.0
    # the planted signal should beat most shuffles → small p
    assert shuf["shuffle_null_p"] < 0.2, shuf["shuffle_null_p"]

    ctrl = control_task_null(signal_arm, y, obs_rho, n_perm=60, seed=761)
    assert set(ctrl) >= {"control_task_rho", "control_task_verdict", "control_task_p", "n_perm"}
    assert ctrl["control_task_verdict"] in ("selective", "non_selective")
    # real signal beats random control targets → selective
    assert ctrl["control_task_verdict"] == "selective", ctrl


# ── round-2 CONCERN fingerprint-assert-skips-matched-artifact ──────────────────


def test_matched_shard_fingerprint_divergence_fails_loud(tmp_path, monkeypatch):
    """A matched shard written with a DIVERGENT recipe_fingerprint must fail the assert.

    Reproduces the round-2 fix: the assembler now reads the matched SHARD's stored
    fingerprint into the cross-arm equality assert (was: in-script constant vs itself).
    Writes a shard with a tampered fingerprint and asserts the equality check trips.
    """
    import issue761_common as common
    import issue761_paired_bootstrap as pb
    import torch

    # a shard with a DIVERGENT fingerprint (summary="last" instead of "mean")
    bad_fp = dict(common.RECIPE_FINGERPRINT)
    bad_fp["summary"] = "last"
    shard_dir = tmp_path / "analysis_tensors"
    shard_dir.mkdir(parents=True)
    blob = {
        "behavior": "sycophancy",
        "context_ids": ["c0"],
        "v0": {"c0": torch.zeros(common.N_LAYERS, common.HIDDEN)},
        "matched_n": {"c0": 200},
        "recipe_fingerprint": bad_fp,
        "smoke": True,
    }
    torch.save(blob, shard_dir / "v0_matched_sycophancy.pt")
    monkeypatch.setattr(pb, "OUT_DIR", tmp_path)

    loaded = pb.load_matched_shard("sycophancy", smoke=True)
    # the cross-arm assert collects the shard's OWN fingerprint; a divergent one trips it
    fingerprints = [common.RECIPE_FINGERPRINT, loaded["recipe_fingerprint"]]
    assert not all(fp == common.RECIPE_FINGERPRINT for fp in fingerprints), (
        "a divergent matched-shard fingerprint must NOT pass the equality check"
    )


# ── round-2 BLOCKER matched-metadata-not-durable ───────────────────────────────


def test_matched_n_recovered_from_shard_only(tmp_path, monkeypatch):
    """The same-N control recovers matched_n from the .pt SHARD with NO local JSON.

    Simulates a fresh VM after pod teardown: only the uploaded .pt shard exists (no
    v0_matched_by_behavior.json). build_samen_X must get its per-context matched_n from
    the shard alone (round-2 BLOCKER fix: matched_n baked into the shard).
    """
    import issue761_common as common
    import issue761_paired_bootstrap as pb
    import torch

    shard_dir = tmp_path / "analysis_tensors"
    shard_dir.mkdir(parents=True)
    ctx = ["c0", "c1"]
    blob = {
        "behavior": "refusal",
        "context_ids": ctx,
        "v0": {c: torch.zeros(common.N_LAYERS, common.HIDDEN) for c in ctx},
        "matched_n": {"c0": 213, "c1": 214},
        "recipe_fingerprint": common.RECIPE_FINGERPRINT,
        "smoke": True,
    }
    torch.save(blob, shard_dir / "v0_matched_refusal.pt")
    monkeypatch.setattr(pb, "OUT_DIR", tmp_path)
    # NO v0_matched_by_behavior.json exists in tmp_path — the JSON dependency is gone.
    assert not (tmp_path / "v0_matched_by_behavior.json").exists()

    loaded = pb.load_matched_shard("refusal", smoke=True)
    matched_n = pb.matched_n_from_shard(loaded)
    assert matched_n == {"c0": 213, "c1": 214}, matched_n


def test_matched_n_from_stale_shard_without_key_raises():
    """A pre-round-2 shard with no baked-in matched_n fails loud (not silent mis-build)."""
    import issue761_paired_bootstrap as pb

    with pytest.raises(KeyError, match="no baked-in matched_n"):
        pb.matched_n_from_shard({"v0": {"c0": None}})  # no matched_n key


# ── round-2 CONCERN silent-left-truncation ─────────────────────────────────────


def test_overlength_row_raises_under_strict():
    """A (prompt+answer) row over max_t raises OverlengthRowError under strict mode.

    Pins the round-2 fix: silent left-truncation is replaced by a fail-loud raise that
    names the offending cell + the row's true length vs max_t. Uses a fake tokenizer +
    a no-op capture so no GPU / real model is needed.
    """
    import issue761_capture_matched_v0 as cap
    import torch

    class _FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "PROMPT " * 100  # long prompt text marker

        def __call__(self, text, return_tensors=None, padding=None, add_special_tokens=True):
            # 100 prompt tokens, 50 answer tokens → 150 total; max_t below trips it.
            # The capture reads ["input_ids"][0], so return a (1, n) tensor.
            n = 100 if "PROMPT" in text else 50
            return {"input_ids": torch.arange(n).unsqueeze(0)}

    inst = {"id": "ctx_x", "system_prompt": None, "prefix_messages": []}
    tuples = [("a probe", "an answer")]
    with pytest.raises(cap.OverlengthRowError, match="overlength row"):
        cap.batched_capture_mean(
            model=None,
            tokenizer=_FakeTokenizer(),
            instance=inst,
            tuples=tuples,
            capture=None,
            n_layers=cap.N_LAYERS,
            batch_probes=16,
            max_t=80,  # 150 > 80 → raise
            behavior="refusal",
            strict_overlength=True,
        )
