"""Behavioral pins for the #2476 `k200-instrument-census` round (plan v11).

Covers the round's registered failure semantics + the parent-driver seams the
round added (record-written-FIRST + distinct exit codes; the `--sae-k` budget
seam incl. the k-aware leaf + regime key; the B1 revision-pin threading):

- G-S / G-A / G-C'(i) / G-C'(iii) gate semantics in
  ``scripts/issue2476_k200_census.py`` (record persisted BEFORE the halt;
  RC_GS=33 / RC_GA=30 / RC_GC=31; G-C'(iii) demotes to INFORMATIONAL-smoke).
- The registered BUDGET lattice (Budget-limited <=> finest-alive >= 10 at the
  registered floor; informational at smoke n).
- ``_encode_counts_sums_l0``: the G-C'(i) accumulator identity executed through
  the REAL MatryoshkaBatchTopKSAE body at tiny width (production-body test —
  the boundary is numpy/torch only; no fakes).
- Parent ``--sae-k`` seams: ``_sae_k``/``_sae_leaf`` resolution, the
  ``sae_c_k200`` out-dir/prefix leaf, the regime key (config_hash flips with
  the budget), the (100, 200) membership typo-fence, and a tiny REAL
  ``phase_sae_train`` run at k=200 (cfg.json records k=200 in the k-aware leaf).
- B1: ``N1M._download_chunk_with_retry`` threads ``revision=`` into
  ``hf_hub_download`` (autospec'd at the network boundary; default None ==
  legacy call shape) and ``N1M._stream_ckpt_fingerprint`` is byte-for-byte
  legacy at ``revision=None`` while a pinned revision changes the digest.

Hermetic: no GPU, no network — external boundaries faked with
``unittest.mock.create_autospec`` (signature-conformant by construction).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue2476_k200_census as K  # noqa: E402
import issue2476_turnavg_sae as D  # noqa: E402


def _args(tmp_path: Path, **over) -> SimpleNamespace:
    """Parent-driver-shaped namespace (the test_issue2476_gates.py harness)."""
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
        sae_k=0,
        g2a_probe_rows=0,
        resume_across_code_sha=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    ns.out_root.mkdir(parents=True, exist_ok=True)
    return ns


@pytest.fixture()
def quiet_sentinels(monkeypatch):
    """Record-only sentinel writes (autospec'd signature-conformant fake at the
    filesystem boundary; K._sentinel routes through the same D.C writer)."""
    rec = create_autospec(D.C.write_sentinel, return_value=Path("/dev/null"))
    monkeypatch.setattr(D.C, "write_sentinel", rec)
    return rec


# ── RC-code registry (round contract) ────────────────────────────────────────────


def test_rc_codes_match_floor_sweep_convention():
    assert (K.RC_GA, K.RC_GC, K.RC_GS) == (30, 31, 33)
    import issue2476_floor_sweep as FS

    assert K.RC_GC == FS.RC_GC  # G-C'(ii) exits through FS._gate_counts verbatim


def test_phase_registry_names():
    assert sorted(K.PHASES) == sorted(K.PHASE_ORDER)
    assert list(K.PHASE_ORDER) == [
        "smoke",
        "assemble",
        "stage_banked",
        "sae_train",
        "densein",
        "census",
        "stats",
        "figures",
    ]


# ── G-S: split shas (record-first + rc=33) ───────────────────────────────────────


def test_gate_splits_pass(tmp_path):
    shas = {"train_full_sha256": "a", "holdout_sha256": "b"}
    rec = K._gate_splits(dict(shas), dict(shas), out_path=tmp_path / "gs.json")
    assert rec["verdict"] == "PASS"
    assert json.loads((tmp_path / "gs.json").read_text())["verdict"] == "PASS"


def test_gate_splits_fail_record_first_rc33(tmp_path):
    out = tmp_path / "gs.json"
    with pytest.raises(SystemExit) as ei:
        K._gate_splits({"x": "1"}, {"x": "2"}, out_path=out)
    assert ei.value.code == K.RC_GS == 33
    rec = json.loads(out.read_text())  # record persisted BEFORE the halt
    assert rec["verdict"] == "FAIL"
    assert rec["assembled_shas"] == {"x": "1"} and rec["banked_shas"] == {"x": "2"}


# ── G-A: row alignment (record-first + rc=30) ────────────────────────────────────


def test_gate_rows_pass(tmp_path):
    rec = K._gate_rows({"a": True, "b": True}, out_path=tmp_path / "ga.json")
    assert rec["verdict"] == "PASS"


def test_gate_rows_fail_record_first_rc30(tmp_path):
    out = tmp_path / "ga.json"
    with pytest.raises(SystemExit) as ei:
        K._gate_rows(
            {"refit_rows_eq_ib_rows": True, "refit_rows_eq_assembled_holdout": False}, out_path=out
        )
    assert ei.value.code == K.RC_GA == 30
    rec = json.loads(out.read_text())
    assert rec["verdict"] == "FAIL"
    assert rec["checks"]["refit_rows_eq_assembled_holdout"] is False


# ── G-C'(i): sum-accounting identity (EXACT; binds at smoke too) ─────────────────


def test_gate_identity_pass(tmp_path):
    rec = K._gate_identity(1234, 1234, n_rows=10, out_path=tmp_path / "gc_i.json")
    assert rec["verdict"] == "PASS"


def test_gate_identity_fail_record_first_rc31(tmp_path):
    out = tmp_path / "gc_i.json"
    with pytest.raises(SystemExit) as ei:
        K._gate_identity(1234, 1233, n_rows=10, out_path=out)
    assert ei.value.code == K.RC_GC == 31
    rec = json.loads(out.read_text())
    assert rec["verdict"] == "FAIL"
    assert rec["sum_per_feature_counts"] == 1234 and rec["sum_per_row_l0"] == 1233


# ── G-C'(iii): n_fit == 120,000 (production halt; smoke informational) ───────────


def test_gate_fit_rows_production_pass(tmp_path):
    rec = K._gate_fit_rows(120_000, production=True, out_path=tmp_path / "gc_iii.json")
    assert rec["verdict"] == "PASS"


def test_gate_fit_rows_production_fail_record_first_rc31(tmp_path):
    out = tmp_path / "gc_iii.json"
    with pytest.raises(SystemExit) as ei:
        K._gate_fit_rows(119_999, production=True, out_path=out)
    assert ei.value.code == K.RC_GC == 31
    assert json.loads(out.read_text())["verdict"] == "FAIL"


def test_gate_fit_rows_smoke_informational_no_exit(tmp_path):
    rec = K._gate_fit_rows(7, production=False, out_path=tmp_path / "gc_iii.json")
    assert rec["verdict"] == "INFORMATIONAL-smoke"  # no SystemExit at smoke n


# ── registered BUDGET lattice (plan §3; disjoint + exhaustive) ───────────────────


@pytest.mark.parametrize(
    ("finest_alive", "want"),
    [
        (10, "Budget-limited"),  # INCLUSIVE threshold
        (11, "Budget-limited"),
        (9, "Budget-discharged"),
        (0, "Budget-discharged"),
    ],
)
def test_budget_lattice_truth_table(finest_alive, want):
    doc = K._budget_lattice(finest_alive, production=True)
    assert doc["verdict"] == want
    assert doc["registered"] is True
    assert doc["threshold"] == 10 and doc["finest_tier_ids"] == [16384, 65536]


def test_budget_lattice_smoke_informational():
    doc = K._budget_lattice(12, production=False)
    assert doc["verdict"].startswith("INFORMATIONAL-smoke")
    assert "Budget-limited" in doc["verdict"]
    assert doc["registered"] is False


# ── per-floor row assembly + effective-floor clamp ───────────────────────────────


def test_finish_floor_row_lattice_note_and_registered_marker():
    stats = {"per_tier": {}, "lattice_verdict": "Gradient-holds"}
    demo = {"per_tier": {}, "not_evaluable_census_only": False, "rule": "r"}
    row = K._finish_floor_row(stats, demo, floor=1200, n_fit=120_000, registered=True)
    assert row["lattice_reported"] == "Gradient-holds"
    assert row["lattice_note"].startswith(K.LATTICE_NOTE_K200)
    assert "registered census cell" in row["lattice_note"]
    assert row["registered_cell"] is True and row["floor_frac_of_fit_rows"] == 0.01
    # demoted floor: census-only, NO lattice label
    demo2 = {"per_tier": {}, "not_evaluable_census_only": True, "rule": "r"}
    row2 = K._finish_floor_row(stats, demo2, floor=240, n_fit=120_000, registered=False)
    assert "lattice_reported" not in row2 and row2["not_evaluable_census_only"] is True


def test_floors_eff_clamp():
    assert K._floors_eff(50) == [50, 50, 50, 50]
    assert K._floors_eff(700) == [700, 600, 300, 240]
    assert K._floors_eff(120_000) == [1200, 600, 300, 240]


# ── _encode_counts_sums_l0: REAL SAE body, identity by construction ──────────────


def test_encode_counts_sums_l0_real_sae_identity():
    """Production-body test: the round-added census accumulator through the REAL
    MatryoshkaBatchTopKSAE encode path at tiny width — counts.sum() == l0_total
    (the very identity G-C'(i) gates) and counts match a direct encode."""
    import torch

    sae = D.MatryoshkaBatchTopKSAE(act_dim=8, dict_size=16, tier_bounds=(2, 8, 16), k=4, seed=0)
    with torch.no_grad():
        sae.threshold.fill_(1e-6)  # gate open so some features fire
    rng = np.random.default_rng(0)
    mm = rng.standard_normal((32, 8)).astype(np.float16)
    counts, sums, l0_total = K._encode_counts_sums_l0(sae, mm, np.arange(32), chunk=8)
    assert counts.shape == (16,) and sums.shape == (16,)
    assert int(counts.sum()) == int(l0_total)  # the G-C'(i) identity
    f = sae.encode(torch.as_tensor(mm.astype(np.float32))).cpu().numpy()
    assert np.array_equal(counts, (f > 0).sum(0).astype(np.int64))
    assert np.allclose(sums, f.astype(np.float64).sum(0), rtol=1e-6, atol=1e-8)


# ── parent --sae-k seams: resolution, leaf, regime key, membership fence ─────────


def test_sae_k_resolution_and_leaf(tmp_path):
    a0 = _args(tmp_path, sae_k=0)
    a2 = _args(tmp_path, sae_k=200)
    assert D._sae_k(a0) == D.SAE_K == 100
    assert D._sae_k(a2) == 200
    assert D._sae_leaf(a0) == "sae_c"
    assert D._sae_leaf(a2) == "sae_c_k200"
    assert D._sae_out_dir(a0).name == "sae_c"
    assert D._sae_out_dir(a2).name == "sae_c_k200"


def test_parent_regime_carries_sae_k_and_hash_flips(tmp_path):
    r0 = D._regime(_args(tmp_path, sae_k=0))
    r2 = D._regime(_args(tmp_path, sae_k=200))
    assert r0["sae_k"] == 0 and r2["sae_k"] == 200
    assert r0["config_hash"] != r2["config_hash"]  # k=200 NEVER resumes a k=100 root


def _sae_train_fixture(tmp_path, monkeypatch, **over):
    """The test_issue2476_gates.py G4 harness: tiny Y19 + faked positions
    (contract-mirroring fakes at the P1/P3 data boundary)."""
    args = _args(tmp_path, sae_dict=64, sae_steps=1, smoke=True, max_chunks=2, **over)
    a_dir = D._assemble_dir(args)
    a_dir.mkdir(parents=True, exist_ok=True)
    (a_dir / "split_meta.json").write_text("{}")
    rng = np.random.default_rng(0)
    y = rng.standard_normal((64, int(D.C.EXPECTED_HIDDEN))).astype(np.float16)
    np.save(a_dir / "Y19.fp16.npy", y)

    def fake_positions(a):
        """Mirrors _sae_row_positions' (tr_pos, val_pos, doc) contract."""
        return np.arange(0, 48, dtype=np.int64), np.arange(48, 64, dtype=np.int64), {"faked": True}

    monkeypatch.setattr(D, "_sae_row_positions", fake_positions)
    upload_rec = create_autospec(D._p4_upload)
    monkeypatch.setattr(D, "_p4_upload", upload_rec)
    return args, upload_rec


def test_sae_train_k200_tiny_real_run(tmp_path, monkeypatch, quiet_sentinels):
    """REAL phase_sae_train at sae_k=200 (tiny width, 1 step, smoke): the
    instrument lands in the sae_c_k200 leaf, cfg.json records k=200, the
    regime manifest carries sae_k=200, and the upload leg targets the k-aware
    out dir (accidental-overwrite fence, plan §8)."""
    args, upload_rec = _sae_train_fixture(tmp_path, monkeypatch, sae_k=200)
    D.phase_sae_train(args)
    out = D._sae_out_dir(args)
    assert out.name == "sae_c_k200"
    cfg = json.loads((out / "cfg.json").read_text())
    assert int(cfg["k"]) == 200
    log = json.loads((out / "train_log.json").read_text())
    assert int(log["cfg"]["k"]) == 200
    regime = json.loads((out / "regime.json").read_text())
    assert regime["sae_k"] == 200
    assert (out / "gates_p4.json").exists()
    assert upload_rec.call_count == 1
    assert upload_rec.call_args.args[1] == out  # upload targets the k200 leaf


def test_sae_train_k150_membership_assert(tmp_path, monkeypatch, quiet_sentinels):
    """The (100, 200) membership fence is a typo guard: an unregistered budget
    halts BEFORE the SAE allocation / any training."""
    args, _ = _sae_train_fixture(tmp_path, monkeypatch, sae_k=150)
    with pytest.raises(AssertionError, match="--sae-k resolved to 150"):
        D.phase_sae_train(args)


def test_k200_parent_args_pins_budget_and_regime(tmp_path):
    """The round driver NEVER exposes the budget as a dial: _parent_args pins
    sae_k=200 and the round regime hashes budget + floors + revision pin."""
    args = SimpleNamespace(
        out_root=tmp_path,
        hf_prefix="p",
        smoke=True,
        max_chunks=2,
        smoke_rows=0,
        device="cpu",
        sae_dir=None,
        fresh_stream=False,
        skip_upload=True,
        gpu_id=-1,
        sae_steps=200,
        sae_dict=64,
        n_perm=10,
        n_boot=10,
        resume_across_code_sha=False,
    )
    pargs = K._parent_args(args, "sae_train")
    assert pargs.sae_k == K.SAE_K200 == 200
    regime = K._regime(args)
    assert regime["sae_k"] == 200
    assert regime["floors"] == [1200, 600, 300, 240] and regime["registered_floor"] == 1200
    assert regime["data_repo_revision"] == K.DATA_REPO_REVISION
    assert regime["budget_finest_alive_min"] == 10


# ── B1: revision-pin threading (download seam + stream fingerprint) ──────────────


def test_download_chunk_threads_revision(tmp_path, monkeypatch):
    """_download_chunk_with_retry passes revision= through to hf_hub_download
    (autospec'd network boundary); the default None matches the legacy shape."""
    import huggingface_hub

    fake = create_autospec(huggingface_hub.hf_hub_download, return_value=str(tmp_path / "f"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    got = N1M._download_chunk_with_retry("repo/x", "a/b.pt", tmp_path, revision="deadbeef")
    assert got == str(tmp_path / "f")
    kw = fake.call_args.kwargs
    assert kw["revision"] == "deadbeef"
    assert kw["repo_type"] == "dataset" and kw["filename"] == "a/b.pt"
    # legacy callers (positional, no revision kwarg) resolve HEAD: revision=None
    N1M._download_chunk_with_retry("repo/x", "a/b.pt", tmp_path)
    assert fake.call_args.kwargs["revision"] is None


def test_stream_fingerprint_legacy_byte_identical_and_revision_refuses():
    names = ["chunk_0001.pt", "chunk_0002.pt"]
    legacy = hashlib.sha256()
    legacy.update(b"layer=19\nprefix=p\n")
    for n in names:
        legacy.update(n.encode())
        legacy.update(b"\n")
    # revision=None reproduces the legacy digest byte-for-byte (existing
    # cursors stay valid — plan §4 Code delta 2)
    assert N1M._stream_ckpt_fingerprint(19, "p", names) == legacy.hexdigest()
    pinned = N1M._stream_ckpt_fingerprint(19, "p", names, revision=K.DATA_REPO_REVISION)
    assert pinned != legacy.hexdigest()
    # a resume minted under one revision REFUSES a stream from another
    other = N1M._stream_ckpt_fingerprint(19, "p", names, revision="0" * 40)
    assert pinned != other


def test_p4_upload_k200_prefix_and_staging(tmp_path, monkeypatch):
    """Production-body twin of test_p4_upload_production_body at sae_k=200: the
    REAL _p4_upload stages from the k-aware leaf and composes the sae_c_k200
    HF prefix (the plan §8 accidental-overwrite fence at the UPLOAD seam);
    Hub faked signature-conformantly at the network boundary only."""
    args = _args(tmp_path, skip_upload=False, sae_k=200)
    out = D._sae_out_dir(args)
    assert out.name == "sae_c_k200"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sae_weights.safetensors", "cfg.json", "train_log.json"):
        (out / name).write_text("x")
    seen: dict = {}

    def fake_upload_dir_sharded(
        local, repo, prefix, *, repo_type, shard_glob, verify, delete_local, resume_skip
    ):
        """Mirrors upload_dir_sharded's call shape; claims success un-rerouted."""
        seen["prefix"] = prefix
        assert repo_type == "dataset"
        return SimpleNamespace(rerouted=False, uploaded=[])

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
    prefix = f"{args.hf_prefix}/sae_c_k200"
    assert seen["prefix"] == prefix  # NEVER the parent's banked sae_c/ leaf
    assert seen["expected"] == sorted(
        f"{prefix}/{n}" for n in ("sae_weights.safetensors", "cfg.json", "train_log.json")
    )
    # staging SCRATCH keeps the fixed sae_c_upload name (rmtree'd + rebuilt from
    # the k-aware `out` per call; an out_root never mixes budgets — regime-keyed)
    up = D._stage_dir(args) / "sae_c_upload"
    assert sorted(p.name for p in up.iterdir()) == [
        "cfg.json",
        "sae_weights.safetensors",
        "train_log.json",
    ]
