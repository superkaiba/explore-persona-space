"""Failure-semantics pins for the #2476 turn-averaged SAE driver (plan §6.5).

Forces each hard-failure path with fabricated failing inputs and asserts a
NONZERO exit + the ABSENCE of downstream done-state:

- G1 reconciliation miss (production)   -> SystemExit rc=26, recon_gate.json
  written FIRST (halt-investigate artifact), verdict FAIL.
- G4 SAE-val FVE-floor miss (production)-> SystemExit rc=25, weights + log +
  gates persisted BEFORE the halt, gates verdict FAIL, the HF upload leg NEVER
  reached.
- Upload-verify failure (P3)            -> AssertionError, done-file ABSENT.

Separately pins that G3 performs ONLY the documented arm-b demotion (flag set,
never a run abort — no G3-keyed sys.exit anywhere in phase_eval), and the
registered verdict-lattice truth table (lower tail / upper tail / inside band /
exact cutoffs / boundary equality / mixed-conjunct cells — exactly one verdict
each, strict inequalities).

Hermetic: no GPU, no network. External boundaries (Hub upload/verify, the
recapture-row positions, the FVE probe) are faked signature-conformantly
(create_autospec / defs mirroring the real signatures); every other body runs
REAL (regime manifests, committed-split sha reads, the SAE training step, the
gate arithmetic).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue2476_turnavg_sae as D  # noqa: E402


def _args(tmp_path: Path, **over) -> SimpleNamespace:
    """Production-shaped namespace (every field _regime/_production read)."""
    ns = SimpleNamespace(
        phase="all",
        out_root=tmp_path / "out",
        hf_prefix="issue2476_turnavg/analysis_tensors",
        smoke=False,
        tiny_model=False,
        max_chunks=0,
        smoke_rows=0,
        gen_batch=4,
        device="cpu",
        sae_dir=tmp_path / "sae_cache",
        fresh_stream=False,
        skip_upload=False,
        gpu_id=-1,
        sae_steps=1,
        n_perm=10,
        n_boot=10,
        fit_n=0,
        import_check=False,
        sae_dict=0,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    ns.out_root.mkdir(parents=True, exist_ok=True)
    return ns


@pytest.fixture()
def quiet_sentinels(monkeypatch):
    """Record-only sentinel writes (autospec'd — signature-conformant fake at the
    filesystem boundary; keeps test sentinels out of /workspace|repo logs)."""
    rec = create_autospec(D.C.write_sentinel, return_value=Path("/dev/null"))
    monkeypatch.setattr(D.C, "write_sentinel", rec)
    return rec


# ── verdict lattice truth table (registered; strict inequalities) ────────────────


@pytest.mark.parametrize(
    ("obs", "lo", "hi", "d_map", "want"),
    [
        # lower tail, matching sign
        (-0.5, -0.3, 0.3, +0.1, "coarse-better"),
        # lower tail, MIXED conjunct (wrong sign / zero diff)
        (-0.5, -0.3, 0.3, -0.1, "tier-null"),
        (-0.5, -0.3, 0.3, 0.0, "tier-null"),
        # upper tail, matching sign
        (+0.5, -0.3, 0.3, -0.1, "fine-better"),
        # upper tail, MIXED conjunct
        (+0.5, -0.3, 0.3, +0.1, "tier-null"),
        (+0.5, -0.3, 0.3, 0.0, "tier-null"),
        # inside band, both signs
        (0.0, -0.3, 0.3, +0.5, "tier-null"),
        (0.0, -0.3, 0.3, -0.5, "tier-null"),
        # EXACT cutoffs: boundary equality resolves inward (strict inequalities)
        (-0.3, -0.3, 0.3, +0.5, "tier-null"),
        (+0.3, -0.3, 0.3, -0.5, "tier-null"),
        # degenerate band lo == hi: equality still tier-null, strict tails fire
        (0.0, 0.0, 0.0, +0.5, "tier-null"),
        (-0.1, 0.0, 0.0, +0.5, "coarse-better"),
        (+0.1, 0.0, 0.0, -0.5, "fine-better"),
        # non-finite observed / diff -> tier-null (nan comparisons are False)
        (float("nan"), -0.3, 0.3, +0.5, "tier-null"),
        (-0.5, -0.3, 0.3, float("nan"), "tier-null"),
    ],
)
def test_lattice_verdict_truth_table(obs, lo, hi, d_map, want):
    got = D._lattice_verdict(obs, lo, hi, d_map)
    assert got in {"coarse-better", "fine-better", "tier-null"}  # exactly one verdict per cell
    assert got == want, (obs, lo, hi, d_map, got, want)


# ── G3: demotion-only (never an abort) ───────────────────────────────────────────


def test_g3_below_floor_demotes_only():
    g = D._g3_verdict(0.10, 5.0, {"probe": True}, 30_000)
    assert g["arm_b_demoted"] is True
    assert g["verdict"] == "DEMOTED-exploratory-with-caveat"
    assert g["floor"] == D.M.GATE_BM_HALT


def test_g3_above_floor_passes():
    g = D._g3_verdict(0.90, 60.0, {}, 30_000)
    assert g["arm_b_demoted"] is False
    assert g["verdict"] == "PASS"


def test_g3_has_no_abort_path_in_phase_eval_or_verdict():
    """The documented G3 semantics: a fitness failure DEMOTES, never aborts —
    neither the verdict helper nor phase_eval carries any exit/raise."""
    for fn in (D._g3_verdict, D.phase_eval):
        src = inspect.getsource(fn)
        assert "sys.exit" not in src, f"{fn.__name__} must not carry an abort path"
        assert "SystemExit" not in src, f"{fn.__name__} must not carry an abort path"


# ── G1: production reconciliation miss -> rc=26, artifact-first ─────────────────


def _fake_g1_refs(root: Path, *, committed=(0.625, 0.55), ours=(0.70, 0.55)) -> Path:
    ref_dir = root / "eval_results" / "issue_1482" / "percontext"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "refit_full__ridge__seed0.json").write_text(
        json.dumps({"sets": {"test": {"whole_map_r2": committed[0]}}})
    )
    (ref_dir / "refit_holdout__ridge__seed0.json").write_text(
        json.dumps({"sets": {"holdout": {"whole_map_r2": committed[1]}}})
    )
    maps_dir = root / "maps"
    (maps_dir / "percontext").mkdir(parents=True, exist_ok=True)
    (maps_dir / "percontext" / "refit_full__ridge__seed0.json").write_text(
        json.dumps({"sets": {"test": {"whole_map_r2": ours[0]}}})
    )
    (maps_dir / "percontext" / "refit_holdout__ridge__seed0.json").write_text(
        json.dumps({"sets": {"holdout": {"whole_map_r2": ours[1]}}})
    )
    return maps_dir


def test_g1_production_fail_exits_rc26(tmp_path, monkeypatch, quiet_sentinels):
    maps_dir = _fake_g1_refs(tmp_path, committed=(0.625, 0.55), ours=(0.70, 0.55))
    monkeypatch.setattr(D, "PROJECT_ROOT", tmp_path)
    with pytest.raises(SystemExit) as ei:
        D._gate_g1(SimpleNamespace(), maps_dir, production=True)
    assert ei.value.code == D.RC_G1 == 26
    # halt-investigate artifact written FIRST, verdict FAIL
    gate = json.loads((maps_dir / "recon_gate.json").read_text())["g1"]
    assert gate["verdict"] == "FAIL"
    assert gate["abs_delta"]["full_test"] > gate["tol"]["full_test"]
    # the halt sentinel is the ONLY sentinel; no downstream PASS state accepted
    assert quiet_sentinels.call_count == 1
    assert quiet_sentinels.call_args.kwargs.get("extra", {}).get("rc") == 26


def test_g1_smoke_demotes_to_informational_no_exit(tmp_path, monkeypatch, quiet_sentinels):
    maps_dir = _fake_g1_refs(tmp_path, committed=(0.625, 0.55), ours=(0.70, 0.55))
    monkeypatch.setattr(D, "PROJECT_ROOT", tmp_path)
    gate = D._gate_g1(SimpleNamespace(), maps_dir, production=False)
    assert gate["verdict"] == "INFORMATIONAL-smoke"
    assert quiet_sentinels.call_count == 0  # no halt sentinel on the smoke branch


def test_g1_pass_within_tolerance(tmp_path, monkeypatch, quiet_sentinels):
    maps_dir = _fake_g1_refs(tmp_path, committed=(0.625, 0.55), ours=(0.6255, 0.552))
    monkeypatch.setattr(D, "PROJECT_ROOT", tmp_path)
    gate = D._gate_g1(SimpleNamespace(), maps_dir, production=True)
    assert gate["verdict"] == "PASS"


# ── G4: production FVE-floor miss -> rc=25, persist-before-halt, no upload ──────


def test_g4_production_fail_exits_rc25_upload_unreached(tmp_path, monkeypatch, quiet_sentinels):
    args = _args(tmp_path, sae_dict=64, sae_steps=1, skip_upload=False)
    a_dir = D._assemble_dir(args)
    a_dir.mkdir(parents=True, exist_ok=True)
    (a_dir / "split_meta.json").write_text("{}")
    rng = np.random.default_rng(0)
    y = rng.standard_normal((64, int(D.C.EXPECTED_HIDDEN))).astype(np.float16)
    np.save(a_dir / "Y19.fp16.npy", y)

    def fake_positions(a):
        """Mirrors _sae_row_positions' (tr_pos, val_pos, doc) contract."""
        return np.arange(0, 48, dtype=np.int64), np.arange(48, 64, dtype=np.int64), {"faked": True}

    def fake_recon_fve(sae, mm, positions, chunk=2048):
        """Mirrors _recon_fve's (fve, l0) contract — forced below the 0.5 floor."""
        return 0.10, 5.0

    monkeypatch.setattr(D, "_sae_row_positions", fake_positions)
    monkeypatch.setattr(D, "_recon_fve", fake_recon_fve)

    def upload_must_not_run(*a, **k):
        raise AssertionError("HF upload reached AFTER a production G4 halt")

    monkeypatch.setattr(
        "explore_persona_space.orchestrate.upload_sharded.upload_dir_sharded",
        upload_must_not_run,
    )
    # production predicate holds (max_chunks=0, smoke_rows=0, not smoke). The
    # failure path under test is G4 AT PRODUCTION with a tractable width, so
    # shrink the production width CONSTANT (sae_dict=0 -> width == SAE_DICT);
    # the sub-production-width guard itself is pinned separately below.
    args.sae_dict = 0
    monkeypatch.setattr(D, "SAE_DICT", 64)
    with pytest.raises(SystemExit) as ei:
        D.phase_sae_train(args)
    assert ei.value.code == D.RC_G4 == 25
    out = D._sae_out_dir(args)
    # persist-before-halt: weights + cfg + log + gates all exist
    for name in ("sae_weights.safetensors", "cfg.json", "train_log.json", "gates_p4.json"):
        assert (out / name).exists(), f"{name} must persist BEFORE the G4 halt"
    gates = json.loads((out / "gates_p4.json").read_text())["g4"]
    assert gates["verdict"] == "FAIL"
    assert gates["val_var_fve"] == pytest.approx(0.10)
    # the halt sentinel carries the designed rc; upload never ran (raiser above)
    assert quiet_sentinels.call_args.kwargs.get("extra", {}).get("rc") == 25


def test_sub_production_width_refused_at_production(tmp_path, monkeypatch, quiet_sentinels):
    """--sae-dict below the production width is a smoke-only instrument change."""
    args = _args(tmp_path, sae_dict=64, sae_steps=1)
    a_dir = D._assemble_dir(args)
    a_dir.mkdir(parents=True, exist_ok=True)
    (a_dir / "split_meta.json").write_text("{}")
    np.save(
        a_dir / "Y19.fp16.npy",
        np.zeros((64, int(D.C.EXPECTED_HIDDEN)), dtype=np.float16),
    )
    monkeypatch.setattr(
        D,
        "_sae_row_positions",
        lambda a: (np.arange(0, 48, dtype=np.int64), np.arange(48, 64, dtype=np.int64), {}),
    )
    with pytest.raises(AssertionError, match="smoke-only"):
        D.phase_sae_train(args)


def test_g4_smoke_below_floor_is_informational(tmp_path, monkeypatch, quiet_sentinels):
    args = _args(tmp_path, smoke=True, smoke_rows=4, sae_dict=64, sae_steps=1, skip_upload=True)
    a_dir = D._assemble_dir(args)
    a_dir.mkdir(parents=True, exist_ok=True)
    (a_dir / "split_meta.json").write_text("{}")
    rng = np.random.default_rng(0)
    np.save(
        a_dir / "Y19.fp16.npy",
        rng.standard_normal((64, int(D.C.EXPECTED_HIDDEN))).astype(np.float16),
    )
    monkeypatch.setattr(
        D,
        "_sae_row_positions",
        lambda a: (np.arange(0, 48, dtype=np.int64), np.arange(48, 64, dtype=np.int64), {}),
    )
    monkeypatch.setattr(D, "_recon_fve", lambda sae, mm, positions, chunk=2048: (0.10, 5.0))
    D.phase_sae_train(args)  # returns cleanly: below-floor is INFORMATIONAL under smoke
    gates = json.loads((D._sae_out_dir(args) / "gates_p4.json").read_text())["g4"]
    assert gates["verdict"] == "INFORMATIONAL-smoke"


# ── P3 upload-verify failure -> AssertionError, done-file ABSENT ────────────────


def _fake_p1_p2_outputs(args) -> None:
    a_dir, recap, stage = D._assemble_dir(args), D._recapture_dir(args), D._stage_dir(args)
    for d in (a_dir, recap, stage):
        d.mkdir(parents=True, exist_ok=True)
    (a_dir / "split_meta.json").write_text("{}")
    (a_dir / "regime.json").write_text("{}")
    np.save(a_dir / "rows_present.npy", np.arange(4, dtype=np.int64))
    np.savez(recap / "vbar_store.npz", row_idx=np.arange(4))
    (recap / "gates_p2.json").write_text("{}")
    (recap / "regime.json").write_text("{}")
    np.savez(stage / "split_indices_matryoshka.npz", s_fit=np.arange(2), s_score=np.arange(2, 4))


def test_upload1_verify_failure_blocks_done(tmp_path, monkeypatch, quiet_sentinels):
    args = _args(tmp_path, skip_upload=False)
    _fake_p1_p2_outputs(args)

    def fake_upload_dir_sharded(
        local, repo, prefix, *, repo_type, shard_glob, verify, delete_local, resume_skip
    ):
        """Mirrors upload_dir_sharded's call shape; claims success un-rerouted."""
        return SimpleNamespace(rerouted=False, uploaded=[])

    def fake_verify(api, repo, expected, path_in_repo=None):
        """Mirrors hub.verify_repo_paths_uploaded's shape; reports a miss."""
        return [f"{path_in_repo}/MISSING.npz"]

    monkeypatch.setattr(
        "explore_persona_space.orchestrate.upload_sharded.upload_dir_sharded",
        fake_upload_dir_sharded,
    )
    monkeypatch.setattr(
        "explore_persona_space.orchestrate.hub.verify_repo_paths_uploaded", fake_verify
    )
    with pytest.raises(AssertionError, match="verify FAILED"):
        D.phase_upload1(args)
    done = D._sentinels_dir(args) / "upload1.done.json"
    assert not done.exists(), "upload1 done-file must NOT be accepted on a verify failure"


# ── new unit-3 helpers: stratified smoke head + width bounds + stand-in seam ─────


def test_smoke_rows_head_keeps_both_tags():
    rows = np.arange(100, dtype=np.int64)
    set_tag = {int(r): (1 if r < 90 else 0) for r in rows}  # sorted head would be all-fit
    got = D._smoke_rows_head(rows, set_tag, 16)
    tags = [set_tag[int(r)] for r in got]
    assert tags.count(0) >= 2 and tags.count(1) >= 2
    assert list(got) == sorted(got)


def test_smoke_rows_head_raises_when_one_side_starved():
    rows = np.arange(10, dtype=np.int64)
    set_tag = {int(r): 1 for r in rows}  # no score rows at all
    with pytest.raises(AssertionError, match=">=2 rows per m-split side"):
        D._smoke_rows_head(rows, set_tag, 8)


def test_sae_tier_bounds():
    assert D._sae_tier_bounds(65536) == tuple(int(b) for b in D.S.MATRYOSHKA_TIER_BOUNDS)
    assert D._sae_tier_bounds(16640) == (2048, 16384, 16640)
    assert D._sae_tier_bounds(64) == (64,)


def test_load_dict_b_tiny_standin_contract():
    args = SimpleNamespace(tiny_model=True, device="cpu", sae_dir=None)
    sae = D._load_dict_b(args, "lmsys")
    assert isinstance(sae, D._TinyJumpReLUStandin)
    assert sae.act_dim == int(D.C.EXPECTED_HIDDEN) and sae.dict_size == 16_640
    import torch

    h = torch.randn(4, sae.act_dim)
    f = sae.encode(h)
    assert f.shape == (4, sae.dict_size) and float((f > 0).sum()) > 0
    assert sae.decode(f).shape == (4, sae.act_dim)
    fve, l0, diag = sae.fve_l0(h)
    assert np.isfinite(fve) and l0 > 0 and diag == {"standin": True}
    # distinct dictionaries per key (seeded)
    sae_p = D._load_dict_b(args, "pile")
    assert not torch.equal(sae.w_enc, sae_p.w_enc)
