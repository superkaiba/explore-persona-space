#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ρ, ×) in scientific docstrings + comments.
"""Issue #810 round-2 regression tests — the substantive BLOCKER fixes.

Each test trips a permanent invariant added in round 2 and would FAIL against the
round-1 code:

1. The batched shuffle-null (``issue810_batched_null``) is NUMERICALLY IDENTICAL
   to the serial closed-form LOCO-ridge null it replaces — the vectorize fix
   (#722 mandate) must stay a throughput win, never a numerical change. (Round 1
   had NO batched path; the serial null projected 231 wall-h for Phase D.)
2. The sycophancy per-context subsample seed is PYTHONHASHSEED-INVARIANT (a stable
   sha256 digest, NOT Python's salted ``hash(str)`` — round 1 used ``hash(ctx_id)``
   so two runs sampled different subsets → different graded E0 target).
3. The turn_nl newline token id is pinned to the Qwen-2.5 family id 198 (round 1
   only asserted single-token + a tautological fed-id check, never the value).

Pure-Python, no GPU / no HF — exercises the helpers directly.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(_SCRIPTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue810_batched_null as bn  # noqa: E402
from issue810_common import (  # noqa: E402
    SHUFFLE_NULL_SEED,
    SYCOPHANCY_SUBSAMPLE_PER_CONTEXT,
    TURN_NL_TOKEN_ID,
)


def _rho_serial(pred, meas):
    """The serial _rho: None on degenerate, else scipy Spearman."""
    from scipy.stats import spearmanr

    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Batched null == serial closed-form LOCO-ridge null (the vectorize invariant)
# ─────────────────────────────────────────────────────────────────────────────
def test_batched_recon_null_matches_serial():
    """RECON skill-over-mean null: batched == serial ridge_predict_loco_centered."""
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        ridge_predict_loco_centered,
        robust_pca_basis,
        skill_over_mean_r2,
    )

    rng0 = np.random.default_rng(0)
    n, hc, hy = 18, 30, 40
    xc = rng0.standard_normal((n, hc)).astype(np.float64)
    z = rng0.standard_normal((n, 4))
    w = rng0.standard_normal((4, hy))
    yv = (z @ w + 0.3 * rng0.standard_normal((n, hy))).astype(np.float64)
    mu, comps, _ = robust_pca_basis(yv, min(48, n - 2))
    y_pca = (yv - mu) @ comps.T
    n_perms = 20
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)
    serial = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        pred = ridge_predict_loco_centered(xc, y_pca[perm])
        serial.append(float(skill_over_mean_r2(pred, y_pca[perm])["skill"]))
    rng_b = np.random.default_rng(SHUFFLE_NULL_SEED)
    perm_b = bn.make_perm_matrix(n, n_perms, rng_b)
    batched = bn.batched_ridge_loco_null_skill(xc, y_pca, perm_b)
    assert np.max(np.abs(np.array(serial) - np.array(batched))) < 1e-6


def test_batched_readout_ridge_null_matches_serial():
    """READOUT trained-ridge null: batched ρ == serial re-fit + Spearman per draw."""
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        ridge_predict_loco_centered,
        robust_pca_basis,
    )

    rng0 = np.random.default_rng(3)
    n = 16
    xsum = rng0.standard_normal((n, 50)).astype(np.float64)
    y = (rng0.standard_normal(n) + 0.5 * xsum[:, 0]).astype(np.float64)
    k = min(48, max(1, n - 2))
    mu, comps, _ = robust_pca_basis(xsum, k)
    xp = (xsum - mu) @ comps.T
    n_perms = 20
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)
    serial = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        pn = ridge_predict_loco_centered(xp, y[perm].reshape(-1, 1))[:, 0]
        dr = _rho_serial(pn, y[perm])
        serial.append(dr if dr is not None else 0.0)
    rng_b = np.random.default_rng(SHUFFLE_NULL_SEED)
    perm_b = bn.make_perm_matrix(n, n_perms, rng_b)
    batched = bn.batched_ridge_loco_null_rho(xp, y, perm_b)
    assert np.max(np.abs(np.array(serial) - np.array(batched))) < 1e-6


def test_batched_projection_null_matches_serial():
    """fixed-r_B projection null: batched ρ == serial _rho(pred, y[perm]) per draw."""
    rng0 = np.random.default_rng(5)
    n = 15
    pred = rng0.standard_normal(n).astype(np.float64)
    y = rng0.standard_normal(n).astype(np.float64)
    n_perms = 20
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)
    serial = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        dr = _rho_serial(pred, y[perm])
        serial.append(dr if dr is not None else 0.0)
    rng_b = np.random.default_rng(SHUFFLE_NULL_SEED)
    perm_b = bn.make_perm_matrix(n, n_perms, rng_b)
    batched = bn.batched_projection_null_rho(pred, y, perm_b)
    assert np.max(np.abs(np.array(serial) - np.array(batched))) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sycophancy subsample seed is PYTHONHASHSEED-invariant (stable sha256 digest)
# ─────────────────────────────────────────────────────────────────────────────
def _subsample_indices(ctx_id: str) -> list[int]:
    """Reproduce the round-2 stable-subsample seed + sample (as in the rejudge)."""
    stable = int(hashlib.sha256(ctx_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(SHUFFLE_NULL_SEED + stable % 100000)
    return rng.sample(list(range(2000)), SYCOPHANCY_SUBSAMPLE_PER_CONTEXT)


def test_subsample_seed_is_process_stable():
    """The sha256-derived seed is deterministic (no dependence on hash randomization).

    Round-1 used ``hash(ctx_id)`` (salted per-process by PYTHONHASHSEED), so this
    would have differed between runs. The sha256 digest is the same in every
    process → identical subsample across runs.
    """
    ctx = "f1_house_librarian"
    a = _subsample_indices(ctx)
    b = _subsample_indices(ctx)
    assert a == b
    # The seed itself must be a pure function of the ctx_id string (sha256), so a
    # freshly computed digest reproduces the exact value regardless of hash seed.
    stable = int(hashlib.sha256(ctx.encode()).hexdigest()[:8], 16)
    assert stable == 3140125910  # pinned expected digest for this ctx_id


def test_rejudge_uses_sha256_not_builtin_hash():
    """The rejudge script seeds the subsample with sha256, NEVER Python hash(str)."""
    src = (_SCRIPTS / "issue810_batch_rejudge_highm.py").read_text()
    # the stable-subsample block must use hashlib.sha256(ctx_id...) and NOT hash(ctx_id).
    assert "hashlib.sha256(ctx_id" in src
    assert "hash(ctx_id)" not in src, "salted builtin hash() must not seed the subsample"


# ─────────────────────────────────────────────────────────────────────────────
# 3. turn_nl newline token id pinned to the Qwen-2.5 family id 198
# ─────────────────────────────────────────────────────────────────────────────
def test_turn_nl_token_id_pinned_to_198():
    """The turn_nl id constant is the Qwen-2.5 newline id 198 (production + smoke)."""
    assert TURN_NL_TOKEN_ID == 198


def test_extract_asserts_newline_id_value():
    """The extractor pins nl_id == TURN_NL_TOKEN_ID (a drifted id must refuse)."""
    src = (_SCRIPTS / "issue810_extract_positions.py").read_text()
    assert "nl_id != TURN_NL_TOKEN_ID" in src, (
        "extractor must assert the newline id equals the pinned 198, not just len==1"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. round-3: --device threads verbatim to every batched-null / MLP call site
# ─────────────────────────────────────────────────────────────────────────────
def test_recon_fit_null_draws_threads_device(monkeypatch):
    """``_fit_null_draws(device=X)`` passes X verbatim to batched_ridge_loco_null_skill.

    Round-3 added a --device flag that must reach the batched-null (and MLP)
    fitters — round-2 code called them with the implicit ``device="cpu"`` default,
    so everything ran fp64 torch on CPU regardless of the lane. This captures the
    kwarg the fit function actually forwards.
    """
    import issue810_fit_reconstruction as recon

    captured = {}

    def _fake(Xc, Y_pca, perm, device="cpu"):
        captured["device"] = device
        return [0.0] * perm.shape[0]

    monkeypatch.setattr(recon, "batched_ridge_loco_null_skill", _fake)
    n, hc, hy = 8, 6, 5
    xc = np.random.default_rng(0).standard_normal((n, hc))
    yv = np.random.default_rng(1).standard_normal((n, hy))
    # cpu default flows through
    recon._fit_null_draws(xc, yv, pca_dim=3, n_perms=5, seed=SHUFFLE_NULL_SEED)
    assert captured["device"] == "cpu"
    # an explicit (fake) device string flows through verbatim
    recon._fit_null_draws(xc, yv, pca_dim=3, n_perms=5, seed=SHUFFLE_NULL_SEED, device="cuda:7")
    assert captured["device"] == "cuda:7"


def test_recon_batch_mlp_validity_threads_device(monkeypatch):
    """``_batch_mlp_validity`` passes device AND a bounded chunk_size verbatim to
    ``fit_batched_loco_mlp_multihead`` (the r5 OOM guard: the library default
    chunk_size=4096 would allocate ~30 GB W1 chunks at d_in=3584)."""
    import issue810_fit_reconstruction as recon

    captured = {}

    class _Res:
        def __init__(self, keys, y_by_key):
            self.preds_by_key = {k: y_by_key[k] for k in keys}

    def _fake(groups, seed=None, device="cpu", chunk_size=4096):
        captured["device"] = device
        captured["chunk_size"] = chunk_size
        return _Res([g.key for g in groups], {g.key: g.Y for g in groups})

    monkeypatch.setattr(recon, "fit_batched_loco_mlp_multihead", _fake)
    # skill_over_mean_r2 is called on the fake preds; supply real arrays so it runs.
    n, d, p = 8, 6, 4
    rng = np.random.default_rng(2)
    xc = rng.standard_normal((n, d))
    y_pca = rng.standard_normal((n, p))
    jobs = [(("mean", 13), xc, y_pca)]
    recon._batch_mlp_validity(jobs, device="cuda:3", chunk_size=64)
    assert captured["device"] == "cuda:3"
    assert captured["chunk_size"] == 64
    # The default must stay memory-bounded (never the library's 4096).
    recon._batch_mlp_validity(jobs, device="cpu")
    assert captured["chunk_size"] == 256


def test_readout_null_call_sites_thread_device():
    """The read-out fit passes ``device=args.device`` to BOTH batched-null calls."""
    src = (_SCRIPTS / "issue810_fit_readout.py").read_text()
    assert "batched_projection_null_rho(pred, y, perm, device=args.device)" in src
    assert "batched_ridge_loco_null_rho(Xp, y, perm, device=args.device)" in src


# ─────────────────────────────────────────────────────────────────────────────
# 5. round-3: upload_out_dir is fail-loud on a missing file (never silent partial)
# ─────────────────────────────────────────────────────────────────────────────
def test_upload_out_dir_fail_loud_on_missing(monkeypatch, tmp_path):
    """A produced JSON absent from the fresh Hub listing raises RuntimeError.

    An ephemeral GCP spot instance is DELETED on exit, so a silently-partial
    upload would lose result JSONs forever. The fail-loud verify is the permanent
    invariant: mocked so no real network is touched, the listing OMITS one of the
    two produced JSONs → RuntimeError.
    """
    import issue810_common as common

    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")

    class _FakeApi:
        def upload_folder(self, *, folder_path, path_in_repo, repo_id, repo_type, **kw):
            _FakeApi.last = dict(
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )

        def list_repo_tree(
            self, repo_id, path_in_repo=None, repo_type=None, revision=None, recursive=False
        ):
            # a.json present, b.json MISSING → the verify must raise.
            return [SimpleNamespace(path=f"{path_in_repo}/a.json")]

    prefix = "issue810/phase_d_recon"

    def _fake_list(repo_id, repo_type=None, revision=None):
        # a.json present, b.json MISSING → the verify must raise.
        return [f"{prefix}/a.json"]

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr("huggingface_hub.list_repo_files", _fake_list)
    with pytest.raises(RuntimeError, match="upload verification FAILED"):
        common.upload_out_dir(tmp_path, prefix)
    # the upload_folder call used the dataset repo + the given prefix (not a loop).
    assert _FakeApi.last["repo_id"] == common.HF_DATA_REPO
    assert _FakeApi.last["repo_type"] == "dataset"
    assert _FakeApi.last["path_in_repo"] == prefix
    assert _FakeApi.last["folder_path"] == str(tmp_path)


def test_upload_out_dir_success_returns_prefix(monkeypatch, tmp_path):
    """When every produced JSON is present on the fresh listing, returns the prefix."""
    import issue810_common as common

    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    prefix = "issue810/phase_d_readout"

    class _FakeApi:
        def upload_folder(self, **kw):
            pass

        def list_repo_tree(
            self, repo_id, path_in_repo=None, repo_type=None, revision=None, recursive=False
        ):
            return [
                SimpleNamespace(path=f"{path_in_repo}/a.json"),
                SimpleNamespace(path=f"{path_in_repo}/b.json"),
            ]

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda repo_id, repo_type=None, revision=None: [f"{prefix}/a.json", f"{prefix}/b.json"],
    )
    assert common.upload_out_dir(tmp_path, prefix) == prefix


def test_upload_out_dir_raises_on_empty_dir(tmp_path):
    """No *.json to upload is a fail-loud error (never a silent no-op upload)."""
    import issue810_common as common

    with pytest.raises(RuntimeError, match=r"no \*.json"):
        common.upload_out_dir(tmp_path, "issue810/x")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
