"""Failure-semantics pins for the #2476 turn-averaged SAE driver (plan §6.5).

Forces each hard-failure path with fabricated failing inputs and asserts a
NONZERO exit + the ABSENCE of downstream done-state:

- G1 reconciliation miss (production)   -> SystemExit rc=26, recon_gate.json
  written FIRST (halt-investigate artifact), verdict FAIL.
- G4 SAE-val FVE-floor miss (production)-> SystemExit rc=25, weights + log +
  gates persisted BEFORE the halt, gates verdict FAIL, the HF upload leg NEVER
  reached — and a SECOND same-regime pass re-applies the recorded rc=25 with
  the trainer body + upload leg unreached (failed-gate-resume-laundering pin).
- P2 resume over a recorded gate FAIL   -> the recorded production rc (22/23/24)
  re-applied before any heavy init.
- Upload-verify failure (P3)            -> AssertionError, done-file ABSENT.

Separately pins: G3 performs ONLY the documented arm-b demotion (finite-FVE
pass predicate; no G3-keyed sys.exit anywhere in phase_eval); the REGISTERED
verdict lattice (perm_pct <= 2.5 AND tier_diff > 0 => Gradient-holds;
perm_pct >= 97.5 AND tier_diff < 0 => Gradient-reversed; else Indeterminate —
INCLUSIVE tail boundaries); the perm-pct midrank arithmetic + the _tier_stats
battery (parent-kernel band parity); the regime hash covering every
output/destination-affecting dial; the tier-stratified panel cap; the
query-chunked retrieval parity vs the shared knn_retrieval helper; the
--tiny-model production refusal; and the steps-capped checkpoint resume.

Hermetic: no GPU, no network. External boundaries (Hub upload/verify, the
recapture-row positions, the FVE probe) are faked signature-conformantly
(create_autospec / defs mirroring the real signatures); every other body runs
REAL (regime manifests, committed-split sha reads, the SAE training step, the
gate arithmetic, EL._stage_scratch_meta's idempotent branch).
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


# ── verdict lattice truth table (registered; INCLUSIVE tail boundaries) ──────────


@pytest.mark.parametrize(
    ("perm_pct", "tier_diff", "want"),
    [
        # lower tail, matching sign (holds)
        (1.0, +0.1, "Gradient-holds"),
        (0.0, +0.1, "Gradient-holds"),
        # INCLUSIVE lower boundary (registered: perm_pct <= 2.5)
        (2.5, +0.1, "Gradient-holds"),
        (2.5, 0.0, "Indeterminate"),
        # lower tail, MIXED conjunct (wrong sign / zero diff)
        (1.0, -0.1, "Indeterminate"),
        (1.0, 0.0, "Indeterminate"),
        # upper tail, matching sign (reversed)
        (99.0, -0.1, "Gradient-reversed"),
        (100.0, -0.1, "Gradient-reversed"),
        # INCLUSIVE upper boundary (registered: perm_pct >= 97.5)
        (97.5, -0.1, "Gradient-reversed"),
        (97.5, 0.0, "Indeterminate"),
        # upper tail, MIXED conjunct
        (99.0, +0.1, "Indeterminate"),
        # inside the band, both signs
        (50.0, +0.5, "Indeterminate"),
        (50.0, -0.5, "Indeterminate"),
        # just inside each boundary
        (2.6, +0.5, "Indeterminate"),
        (97.4, -0.5, "Indeterminate"),
        # non-finite inputs -> Indeterminate
        (float("nan"), +0.5, "Indeterminate"),
        (1.0, float("nan"), "Indeterminate"),
        (float("inf"), -0.5, "Indeterminate"),
    ],
)
def test_lattice_verdict_truth_table(perm_pct, tier_diff, want):
    got = D._lattice_verdict(perm_pct, tier_diff)
    assert got in {"Gradient-holds", "Gradient-reversed", "Indeterminate"}
    assert got == want, (perm_pct, tier_diff, got, want)


def test_perm_pct_midrank():
    draws = np.arange(10, dtype=np.float64)  # 0..9
    assert D._perm_pct(4.0, draws) == pytest.approx(45.0)  # 4 below + 0.5 tie
    assert D._perm_pct(-1.0, draws) == pytest.approx(0.0)
    assert D._perm_pct(99.0, draws) == pytest.approx(100.0)
    # NaN draws are filtered; NaN obs / empty draws -> NaN
    assert D._perm_pct(4.0, np.array([np.nan, 0.0, 9.0])) == pytest.approx(50.0)
    assert np.isnan(D._perm_pct(float("nan"), draws))
    assert np.isnan(D._perm_pct(0.5, np.array([np.nan])))


def test_tier_stats_battery_holds_case():
    """Synthetic coarse-better battery through the REAL parent permutation kernel:
    perm-draw regeneration band parity (asserted inside), perm_pct + seed
    recorded, registered lattice verdict fires on the strong gradient."""
    rng = np.random.default_rng(7)
    tier = np.repeat(np.array([0, 1, 2]), 70)
    r2 = np.concatenate(
        [
            0.80 + 0.01 * rng.standard_normal(70),
            0.50 + 0.01 * rng.standard_normal(70),
            0.10 + 0.01 * rng.standard_normal(70),
        ]
    )
    activity = rng.integers(240, 10_000, size=210).astype(np.float64)
    stats = D._tier_stats(r2, r2 - 0.05, tier, activity, n_perm=200, n_boot=50, rng=rng)
    perm = stats["permutation"]
    assert np.isfinite(perm["perm_pct"]) and perm["perm_pct"] <= 2.5
    assert "perm_seed" in perm and perm["strata"] == "quintile"
    assert stats["lattice_verdict"] == "Gradient-holds"
    li = stats["lattice_inputs"]
    assert li["tier_diff_map_t0_minus_t2"] == pytest.approx(0.70, abs=0.05)
    assert len(li["raw_band_2p5_97p5"]) == 2  # parent-parity band kept persisted
    assert set(stats["per_tier"]) == {"0", "1", "2"}


def test_tier_stats_insufficient_features_is_indeterminate():
    rng = np.random.default_rng(0)
    tier = np.array([0, 0, 2])
    r2 = np.array([0.5, 0.4, 0.1])
    stats = D._tier_stats(r2, r2, tier, np.array([300.0, 400.0, 500.0]), 50, 20, rng)
    assert stats["permutation"]["verdict"] == "insufficient-features"
    assert stats["lattice_verdict"] == "Indeterminate"


# ── G3: demotion-only (never an abort; finite-FVE pass predicate) ────────────────


def test_g3_below_floor_demotes_only():
    g = D._g3_verdict(0.10, 5.0, {"probe": True}, 30_000)
    assert g["arm_b_demoted"] is True
    assert g["verdict"] == "DEMOTED-exploratory-with-caveat"
    assert g["floor"] == D.M.GATE_BM_HALT


def test_g3_above_floor_passes():
    g = D._g3_verdict(0.90, 60.0, {}, 30_000)
    assert g["arm_b_demoted"] is False
    assert g["verdict"] == "PASS"


@pytest.mark.parametrize("bad_fve", [float("nan"), float("-inf"), float("inf")])
def test_g3_nonfinite_fve_demotes(bad_fve):
    """PASS requires a FINITE FVE at/above the floor — `NaN < floor` is False and
    silently passed under the old comparison (Codex r1 Major `g3-nonfinite-pass`)."""
    g = D._g3_verdict(bad_fve, 5.0, {}, 30_000)
    assert g["arm_b_demoted"] is True
    assert g["verdict"] == "DEMOTED-exploratory-with-caveat"


def test_g3_has_no_abort_path_in_phase_eval_or_verdict():
    """The documented G3 semantics: a fitness failure DEMOTES, never aborts —
    neither the verdict helper nor phase_eval carries any exit/raise."""
    for fn in (D._g3_verdict, D.phase_eval):
        src = inspect.getsource(fn)
        assert "sys.exit" not in src, f"{fn.__name__} must not carry an abort path"
        assert "SystemExit" not in src, f"{fn.__name__} must not carry an abort path"


# ── regime key + phase-entry semantics ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("dial", "value"),
    [
        ("sae_steps", 7),
        ("fit_n", 5_000),
        ("n_perm", 11),
        ("n_boot", 11),
        ("gen_batch", 8),
        ("hf_prefix", "issue2476_turnavg/analysis_tensors_other"),
        ("skip_upload", True),
        ("sae_dict", 64),
        ("smoke", True),
        ("tiny_model", True),
        ("max_chunks", 2),
        ("smoke_rows", 8),
    ],
)
def test_regime_hashes_every_output_affecting_dial(tmp_path, dial, value):
    """Every output/destination-affecting dial changes config_hash (Codex r1
    Critical `incomplete-regime-key` + g2 MAJOR-2 + g1 Minor 2)."""
    base = D._regime(_args(tmp_path))["config_hash"]
    assert D._regime(_args(tmp_path, **{dial: value}))["config_hash"] != base, dial


def test_enter_phase_regime_collision_raises(tmp_path):
    args = _args(tmp_path)
    out = tmp_path / "phase"
    out.mkdir()
    bad = {**D._regime(args), "config_hash": "0" * 16}
    (out / "regime.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        D._enter_phase_regime(out, args, "eval")


def test_enter_phase_regime_code_sha_mismatch_wipes_stale_before_manifest(tmp_path, monkeypatch):
    """g1 r1 Minor 1 root fix: on the code-SHA-only recompute branch the stale
    outputs are deleted BEFORE the fresh manifest write (no crash window in
    which a new-code regime.json vouches for old-code outputs)."""
    args = _args(tmp_path)
    out = tmp_path / "phase"
    out.mkdir()
    (out / "regime.json").write_text(json.dumps({**D._regime(args), "code_sha": "deadbeef"}))
    stale = out / "old_output.npz"
    stale.write_text("x")
    real_write = D._write_json

    def check_write(path, obj, *, phase):
        if path.name == "regime.json":
            assert not stale.exists(), "stale outputs must be wiped BEFORE the manifest write"
        return real_write(path, obj, phase=phase)

    monkeypatch.setattr(D, "_write_json", check_write)
    _, resume_ok = D._enter_phase_regime(out, args, "eval", stale_paths=[stale])
    assert resume_ok is False
    assert not stale.exists()


# ── failed-gate resume-laundering: recorded verdicts re-applied ──────────────────


@pytest.mark.parametrize(
    ("phase", "key", "bad", "rc"),
    [
        ("recapture", "g2b", "FAIL", 23),
        ("recapture", "g2a", "FAIL", 22),
        ("recapture", "d2c", "HALT-INVESTIGATE", 24),
        ("sae_train", "g4", "FAIL", 25),
    ],
)
def test_reapply_recorded_gate_verdicts_table(phase, key, bad, rc):
    with pytest.raises(SystemExit) as ei:
        D._reapply_recorded_gate_verdicts({key: {"verdict": bad}}, True, phase)
    assert ei.value.code == rc
    # non-production and PASS verdicts never exit
    D._reapply_recorded_gate_verdicts({key: {"verdict": bad}}, False, phase)
    D._reapply_recorded_gate_verdicts({key: {"verdict": "PASS"}}, True, phase)


def test_p2_resume_reapplies_recorded_gate_fail(tmp_path, monkeypatch, quiet_sentinels):
    """Two-pass P2: a recorded production G2a FAIL exits with the ORIGINAL rc on
    a same-regime relaunch, BEFORE any heavy init (g2 MAJOR-1 class)."""
    args = _args(tmp_path)
    out = D._recapture_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regime.json").write_text(json.dumps(D._regime(args)))
    np.savez(out / "vbar_store.npz", row_idx=np.arange(4))
    (out / "gates_p2.json").write_text(json.dumps({"g2a": {"verdict": "FAIL"}}))

    def boom(args):
        raise AssertionError("P2 heavy path reached on a failed-gate resume")

    monkeypatch.setattr(D, "_p2_input_contract", boom)
    with pytest.raises(SystemExit) as ei:
        D.phase_recapture(args)
    assert ei.value.code == D.RC_G2A == 22


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

    # ── SECOND PASS (same regime): the recorded FAIL re-applies rc=25 BEFORE the
    # trainer body or the upload leg (Codex r1 `failed-gate-resume-laundering`).
    def positions_must_not_run(a):
        raise AssertionError("trainer body reached on a failed-gate resume")

    def p4_upload_must_not_run(args, out, *, resume_skip):
        raise AssertionError("_p4_upload reached on a failed-gate resume")

    monkeypatch.setattr(D, "_sae_row_positions", positions_must_not_run)
    monkeypatch.setattr(D, "_p4_upload", p4_upload_must_not_run)
    with pytest.raises(SystemExit) as ei2:
        D.phase_sae_train(args)
    assert ei2.value.code == 25


def test_p4_resume_redrives_upload_after_pass(tmp_path, monkeypatch, quiet_sentinels):
    """A crash between the gates write and the upload must not strand the §6.5
    instrument deliverable: the PASS-gates resume path re-drives the upload
    (resume_skip=True) instead of returning on artifact presence (g2 MAJOR-1)."""
    args = _args(tmp_path, skip_upload=False)
    out = D._sae_out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regime.json").write_text(json.dumps(D._regime(args)))
    for name in ("sae_weights.safetensors", "cfg.json", "train_log.json"):
        (out / name).write_text("x")
    (out / "gates_p4.json").write_text(json.dumps({"g4": {"verdict": "PASS"}}))
    calls: list[bool] = []

    def fake_p4_upload(args, out, *, resume_skip):
        """Mirrors _p4_upload's signature; records the resume_skip posture."""
        calls.append(resume_skip)

    monkeypatch.setattr(D, "_p4_upload", fake_p4_upload)
    D.phase_sae_train(args)  # returns cleanly via the resume-skip branch
    assert calls == [True], "resume must verify/re-drive the upload with resume_skip=True"


def test_p4_upload_production_body(tmp_path, monkeypatch):
    """Production-body test for _p4_upload (checklist: one production-body test
    per seam-stubbed function): real staging copies + prefix composition + the
    fail-loud exact-set verify; Hub faked signature-conformantly."""
    args = _args(tmp_path, skip_upload=False)
    out = D._sae_out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sae_weights.safetensors", "cfg.json", "train_log.json"):
        (out / name).write_text("x")

    def fake_upload_dir_sharded(
        local, repo, prefix, *, repo_type, shard_glob, verify, delete_local, resume_skip
    ):
        """Mirrors upload_dir_sharded's call shape; claims success un-rerouted."""
        assert repo_type == "dataset" and resume_skip is True
        return SimpleNamespace(rerouted=False, uploaded=[])

    seen: dict = {}

    def fake_verify(api, repo, expected, path_in_repo=None):
        """Mirrors hub.verify_repo_paths_uploaded's shape; records the set."""
        seen["expected"] = sorted(expected)
        return []

    monkeypatch.setattr(
        "explore_persona_space.orchestrate.upload_sharded.upload_dir_sharded",
        fake_upload_dir_sharded,
    )
    monkeypatch.setattr(
        "explore_persona_space.orchestrate.hub.verify_repo_paths_uploaded", fake_verify
    )
    D._p4_upload(args, out, resume_skip=True)
    prefix = f"{args.hf_prefix}/sae_c"
    assert seen["expected"] == sorted(
        f"{prefix}/{n}" for n in ("sae_weights.safetensors", "cfg.json", "train_log.json")
    )
    up = D._stage_dir(args) / "sae_c_upload"
    assert sorted(p.name for p in up.iterdir()) == [
        "cfg.json",
        "sae_weights.safetensors",
        "train_log.json",
    ]
    # a verify miss raises (fail-loud exact-set verify, the P3 pattern)
    monkeypatch.setattr(
        "explore_persona_space.orchestrate.hub.verify_repo_paths_uploaded",
        lambda api, repo, expected, path_in_repo=None: [f"{path_in_repo}/MISSING"],
    )
    with pytest.raises(AssertionError, match="verify FAILED"):
        D._p4_upload(args, out, resume_skip=True)
    # non-production (smoke) never touches the Hub (g2 MINOR-5)
    args_s = _args(tmp_path, smoke=True, out_root=tmp_path / "out2")

    def hub_must_not_run(*a, **k):
        raise AssertionError("Hub reached on a non-production _p4_upload")

    monkeypatch.setattr(
        "explore_persona_space.orchestrate.upload_sharded.upload_dir_sharded", hub_must_not_run
    )
    D._p4_upload(args_s, out, resume_skip=False)


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


def test_p4_steps_capped_checkpoint_resume_completes(tmp_path, monkeypatch, quiet_sentinels):
    """g2 MAJOR-2 companion: a steps-capped PARTIAL epoch is checkpointed as
    epoch_done=epoch (not epoch+1) with steps_capped=True; a same-regime resume
    treats the capped budget as training-complete (no epoch re-entry) and the
    optimizer checkpoint is discarded at completion (plan §10 discard row)."""
    import torch

    args = _args(tmp_path, smoke=True, sae_dict=64, sae_steps=1, skip_upload=True)
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
    out = D._sae_out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regime.json").write_text(json.dumps(D._regime(args)))
    model = D.MatryoshkaBatchTopKSAE(dict_size=64, tier_bounds=D._sae_tier_bounds(64))
    opt = torch.optim.Adam(model.parameters(), lr=D.SAE_LR, betas=D.SAE_ADAM_BETAS)
    row = {
        "epoch": 1,
        "steps": 1,
        "mean_loss": 1.0,
        "val_var_fve": 0.9,
        "val_l0": 10.0,
        "dead_frac_by_tier": {},
        "threshold": 0.0,
        "elapsed_s": 0.1,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch_done": 0,  # PARTIAL epoch: never counted done
            "steps_capped": True,
            "step": 1,
            "log_rows": [row],
        },
        out / "ckpt_last.pt",
    )
    D.phase_sae_train(args)
    log = json.loads((out / "train_log.json").read_text())
    assert len(log["epochs"]) == 1, "capped budget must not re-enter the epoch"
    gates = json.loads((out / "gates_p4.json").read_text())["g4"]
    assert gates["verdict"] == "PASS" and gates["val_var_fve"] == pytest.approx(0.9)
    assert not (out / "ckpt_last.pt").exists(), "optimizer ckpt must be discarded at completion"


# ── --tiny-model production refusal (g3 r1 M1) ───────────────────────────────────


@pytest.mark.parametrize(
    ("over", "ok"),
    [
        ({"tiny_model": True, "phase": "all"}, False),  # production-classified
        ({"tiny_model": True, "phase": "eval"}, False),
        ({"tiny_model": True, "phase": "smoke"}, True),  # composed smoke self-scopes
        ({"tiny_model": True, "phase": "eval", "smoke": True}, True),
        ({"tiny_model": True, "phase": "eval", "smoke_rows": 8}, True),
        ({"tiny_model": False, "phase": "all"}, True),
    ],
)
def test_refuse_tiny_model_at_production(tmp_path, over, ok):
    args = _args(tmp_path, **over)
    if ok:
        D._refuse_tiny_model_at_production(args)
    else:
        with pytest.raises(AssertionError, match="smoke-only"):
            D._refuse_tiny_model_at_production(args)


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
    # arm-c scratch meta: non-empty files satisfy EL._stage_scratch_meta's
    # idempotent branch (the REAL body runs; no network reached)
    np.savez(stage / "split_indices.npz", train=np.arange(2), holdout=np.arange(2, 4))
    np.save(stage / "row_ci.npy", np.arange(4, dtype=np.int64))
    np.save(stage / "prov.npy", np.zeros(4, dtype=np.uint8))


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


def test_upload1_split_meta_bundle_includes_arm_c_split(tmp_path, monkeypatch, quiet_sentinels):
    """g3/Codex r1 `p3-split-meta-incomplete`: the uploaded split-meta bundle must
    carry the arm-c split_indices.npz alongside the m-split arrays."""
    args = _args(tmp_path, skip_upload=False)
    _fake_p1_p2_outputs(args)
    staged: dict = {}

    def fake_upload_dir_sharded(
        local, repo, prefix, *, repo_type, shard_glob, verify, delete_local, resume_skip
    ):
        """Mirrors upload_dir_sharded's call shape; records the staged file set."""
        staged[prefix.rsplit("/", 1)[-1]] = sorted(p.name for p in Path(local).iterdir())
        return SimpleNamespace(rerouted=True, uploaded=[])  # rerouted: skip the verify leg

    monkeypatch.setattr(
        "explore_persona_space.orchestrate.upload_sharded.upload_dir_sharded",
        fake_upload_dir_sharded,
    )
    D.phase_upload1(args)
    assert "split_indices.npz" in staged["split_meta"], staged
    assert "split_indices_matryoshka.npz" in staged["split_meta"], staged
    assert (D._sentinels_dir(args) / "upload1.done.json").exists()


# ── panel / retrieval / artifact-schema helpers ──────────────────────────────────


def test_alive_panel_cap_binds_and_under_cap_keeps_all():
    n_fit = 24_000  # floor = 240
    counts = np.zeros(65_536, np.int64)
    counts[:40_000] = 1_000  # 40k clearing -> the 16,384 cap binds
    panel, doc = D._alive_panel(counts, n_fit)
    assert len(panel) == int(D.M.PANEL_CAP) == 16_384
    assert doc["floor"] == 240 and doc["cap"] == 16_384 and doc["seed"] == int(D.M.PANEL_SEED)
    assert (np.sort(panel) == panel).all()
    assert doc["alloc_by_tier"]["0"] == 2_048  # all clearing tier-0 kept (plan §11)
    assert sum(doc["alloc_by_tier"].values()) == 16_384
    small = np.zeros(65_536, np.int64)
    small[:100] = 1_000
    p2, d2 = D._alive_panel(small, n_fit)
    assert len(p2) == 100 and d2["n_panel"] == 100  # under cap: every clearing feature kept
    # determinism (seeded)
    p3, _ = D._alive_panel(counts, n_fit)
    assert (p3 == panel).all()


def test_knn_retrieval_chunked_parity_vs_shared_helper():
    """Codex r1 Major `retrieval-not-chunked`: the query-side-chunked twin must
    match mapping_baselines.knn_retrieval per metric — including tie mid-ranks
    (duplicated rows) — while the shared helper stays untouched."""
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    rng = np.random.default_rng(0)
    true = rng.standard_normal((50, 8))
    true[10] = true[11]  # exact duplicate pool rows -> tie mid-rank path
    pred = true + 0.3 * rng.standard_normal((50, 8))
    pred[10] = pred[11]
    for metric in ("euclidean", "cosine"):
        ours = D._knn_retrieval_chunked(pred, true, metric=metric, block=7)
        ref = knn_retrieval(pred, true, ks=(1, 5, 10), metric=metric)
        assert ours["n"] == ref["n"] and ours["n_pool"] == ref["n_pool"]
        for k in (1, 5, 10):
            assert ours["acc_at_k"][k] == pytest.approx(ref["acc_at_k"][k]), (metric, k)
            assert ours["chance_at_k"][k] == pytest.approx(ref["chance_at_k"][k])
        assert ours["median_rank"] == pytest.approx(ref["median_rank"])
        assert ours["mrr"] == pytest.approx(ref["mrr"])


def test_extract_chunk_l19_key_coverage(tmp_path):
    """Codex r1 Critical `cached-artifact-schema-unverified`: a fetched chunk
    missing a required key fails with the named artifact + missing-key list,
    BEFORE any field lookup."""
    import torch

    p = tmp_path / "shard00_chunk0000.pt"
    torch.save({"layers": [10, 19, 25], "cx_last": torch.zeros(2, 3, 8), "ci": [0, 1]}, p)
    with pytest.raises(AssertionError, match=r"missing keys \['v_x'\]"):
        D._extract_chunk_l19(p)


def test_torch_load_wo_rejects_pickled_objects(tmp_path):
    """The weights_only load path must fail LOUD on a non-weights pickle (never
    a silent weights_only=False fallback)."""
    import torch

    p = tmp_path / "bad.pt"
    torch.save({"fn": SimpleNamespace(x=1)}, p)
    with pytest.raises(Exception):  # noqa: B017 — torch raises a pickle-class error
        D._torch_load_wo(p)


# ── unit-3 helpers: stratified smoke head + width bounds + stand-in seam ─────────


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
