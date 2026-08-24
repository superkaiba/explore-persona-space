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
  ``hf_hub_download`` (autospec'd at the network boundary; the default None is
  forwarded and equals ``hf_hub_download``'s own default — HEAD — so pre-B1
  callers that omit the kwarg resolve identically; r8 M4 wording) and
  ``N1M._stream_ckpt_fingerprint`` is byte-for-byte legacy at
  ``revision=None`` while a pinned revision changes the digest.
- r8 union fixes: the §6.5 git/HF destination split (U1), the recorded-FAIL
  re-apply loader at every standalone downstream phase (U2), the legacy k=100
  regime-hash parity (U3), the warm pass-b provenance pin (U4), the durable
  densein DROP reason chain (U6), and the §3 manipulation check riding the
  lattice record + R7 digest (U7).

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

#: r8 U3 legacy-parity pin: D._regime(_args(sae_k=0))["config_hash"] at commit
#: 1b93837e92~1 (the pre---sae-k code, imported by file and evaluated over the
#: SAME fixture args, 2026-08-24). The post-diff default MUST reproduce it
#: byte-identically or every banked k=100 manifest is rejected on resume.
PREDIFF_DEFAULT_CONFIG_HASH = "081e4c12d462cdba"


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
    assert K.RC_G4 == D.RC_G4 == 25  # r8 U2: the loader re-applies the parent's G-4' rc


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
    doc = K._budget_lattice(finest_alive, val_l0=199.0, production=True)
    assert doc["verdict"] == want
    assert doc["registered"] is True
    assert doc["threshold"] == 10 and doc["finest_tier_ids"] == [16384, 65536]


def test_budget_lattice_smoke_informational():
    doc = K._budget_lattice(12, val_l0=199.0, production=False)
    assert doc["verdict"].startswith("INFORMATIONAL-smoke")
    assert "Budget-limited" in doc["verdict"]
    assert doc["registered"] is False


@pytest.mark.parametrize(
    ("val_l0", "realized"),
    [
        (99.0, False),  # ~ the parent's 98.96: manipulation NOT realized (r8 U7 fixture)
        (149.9, False),  # below the stated conservative predicate
        (150.0, True),  # inclusive predicate boundary
        (199.4, True),  # ~ k=200 expected regime
    ],
)
def test_budget_lattice_manipulation_check_rides_record(val_l0, realized):
    """r8 U7: the lattice RECORD carries val_l0 + manipulation_realized under
    the stated concrete predicate (val_l0 >= 150); the registered branch
    DEFINITIONS are unchanged (verdict computed exactly as before)."""
    doc = K._budget_lattice(12, val_l0=val_l0, production=True)
    assert doc["verdict"] == "Budget-limited"  # branch definition untouched
    assert doc["val_l0"] == pytest.approx(val_l0)
    assert doc["manipulation_realized"] is realized
    chk = doc["manipulation_check"]
    assert chk["threshold_val_l0"] == K.MANIPULATION_VAL_L0_MIN == 150.0
    assert chk["parent_k100_val_l0"] == K.PARENT_VAL_L0_K100 == 98.96
    assert "val_l0 >= 150" in chk["predicate"]
    assert "conditional-on-manipulation" in chk["narration"]


def test_r7_digest_conditional_on_manipulation_label():
    """r8 U7: a production fixture with val_l0 ~ 99 (the parent's regime — the
    manipulation did NOT move L0) publishes manipulation_realized=false and the
    digest labels the budget branch `conditional-on-manipulation`."""
    census = {
        "budget_lattice": K._budget_lattice(12, val_l0=99.0, production=True),
        "per_floor": {str(K.REGISTERED_FLOOR): {"alive_by_tier": {"0": 1, "1": 2, "2": 3}}},
    }
    sweep = {"lattice_vector": [{"label": "Gradient-holds"}]}
    gates = {
        "pred_encode_fve": {"pred_encode_fve": 0.9},
        "gates": {"gs": {"verdict": "PASS"}, "pred_encode_fve": {"pred_encode_fve": 0.9}},
    }
    digest = K._r7_digest(census, sweep, gates)
    assert digest["budget_lattice"] == "Budget-limited"
    assert digest["budget_lattice_label"] == "conditional-on-manipulation"
    assert digest["manipulation_realized"] is False
    assert digest["val_l0"] == pytest.approx(99.0)
    assert digest["gates"] == {"gs": "PASS"}  # pred_encode_fve stays a reported companion


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
    """r8 U3: a DEFAULT budget omits the sae_k key entirely (pre-diff manifest
    parity — banked k=100 out-roots stay resumable), an explicit --sae-k 100
    resolves to the same production instrument and the same hash, and k=200
    keys distinctly (a k=200 run can never resume a k=100 root)."""
    r0 = D._regime(_args(tmp_path, sae_k=0))
    r100 = D._regime(_args(tmp_path, sae_k=100))
    r2 = D._regime(_args(tmp_path, sae_k=200))
    assert "sae_k" not in r0 and "sae_k" not in r100  # default/production budget: key omitted
    assert r2["sae_k"] == 200
    assert r0["config_hash"] == r100["config_hash"]  # same resolved instrument, same regime
    assert r0["config_hash"] != r2["config_hash"]  # k=200 NEVER resumes a k=100 root


def test_parent_regime_default_hash_is_prediff_legacy_constant(tmp_path):
    """r8 U3 pin: the default-budget config_hash equals the PRE-diff hash as a
    literal constant (computed from the pre---sae-k _regime over the same
    fixture args at commit 1b93837e92~1), so every banked k=100 manifest
    (parent + floor-sweep out-roots) resumes unrejected. Reconstruction arm:
    hashing the returned base dict minus the derived keys reproduces it."""
    r0 = D._regime(_args(tmp_path, sae_k=0))
    assert r0["config_hash"] == PREDIFF_DEFAULT_CONFIG_HASH
    base = {k: v for k, v in r0.items() if k not in ("config_hash", "code_sha")}
    got = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]
    assert got == PREDIFF_DEFAULT_CONFIG_HASH


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
    (autospec'd network boundary). The default None is FORWARDED explicitly and
    equals hf_hub_download's own default (HEAD) — behaviorally identical to the
    pre-B1 callers that omit the kwarg, not a distinct 'legacy call shape'
    (r8 M4 wording fix)."""
    import huggingface_hub

    fake = create_autospec(huggingface_hub.hf_hub_download, return_value=str(tmp_path / "f"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    got = N1M._download_chunk_with_retry("repo/x", "a/b.pt", tmp_path, revision="deadbeef")
    assert got == str(tmp_path / "f")
    kw = fake.call_args.kwargs
    assert kw["revision"] == "deadbeef"
    assert kw["repo_type"] == "dataset" and kw["filename"] == "a/b.pt"
    # a caller omitting the kwarg forwards revision=None == hf_hub_download's
    # own default (HEAD) — identical resolution to the pre-B1 call sites (M4)
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


# ── r8 U1: the §6.5 git/HF destination split (explicit allowlist, never a glob) ──


def test_git_hf_destination_split_disjoint():
    """r8 U1: the git allowlist EXCLUDES the two HF-only tensors — the densein
    npz (20,000 x n_union fp16) plausibly exceeds GitHub's 100 MiB hard limit,
    so a glob sweep would kill the R7 push AFTER the full production run."""
    assert set(K.GIT_EVAL_BASENAMES) & set(K.HF_ONLY_EVAL_BASENAMES) == set()
    assert "firing_census_k200.npz" in K.HF_ONLY_EVAL_BASENAMES
    assert "perfeature_k200_densein.npz" in K.HF_ONLY_EVAL_BASENAMES
    # the §6.5 git rows verbatim: declared JSONs + the union npz + gates only
    assert set(K.GIT_EVAL_BASENAMES) == {
        "census_k200.json",
        "tier_sweep_k200.json",
        "retrieval_k200.json",
        "gates_k200.json",
        "perfeature_union_k200.npz",
    }
    assert K.GIT_FILE_MAX_BYTES < 100 * 1024**2  # guard sits UNDER the GitHub hard limit


def test_r7_git_srcs_allowlist_never_sweeps_hf_only(tmp_path):
    ev = tmp_path / "eval"
    ev.mkdir()
    for name in (*K.GIT_EVAL_BASENAMES, *K.HF_ONLY_EVAL_BASENAMES, "union_encodes_meta.json"):
        (ev / name).write_text("x")
    srcs = K._r7_git_srcs(ev)
    assert sorted(p.name for p in srcs) == sorted(K.GIT_EVAL_BASENAMES)
    (ev / "gates_k200.json").unlink()  # a missing declared artifact fails loud
    with pytest.raises(AssertionError, match="declared git eval artifacts missing"):
        K._r7_git_srcs(ev)


def test_git_leg_size_guard_blocks_oversize(tmp_path):
    """The per-file size guard fires BEFORE any git operation (r8 U1)."""
    big = tmp_path / "big.npz"
    with open(big, "wb") as f:
        f.seek(K.GIT_FILE_MAX_BYTES)  # sparse: st_size over the guard, no disk cost
        f.write(b"x")
    with pytest.raises(AssertionError, match="100 MiB"):
        K._git_leg([big])


# ── r8 U2: recorded gate FAILs re-apply at every standalone downstream phase ─────


def _write_gate_records(args, *, fail_key=None):
    """Every required upstream record PASS except ``fail_key`` (FAIL)."""
    gd = K._gates_dir(args)
    gd.mkdir(parents=True, exist_ok=True)
    for key in ("gs", "ga", "gc_i", "gc_ii", "gc_iii"):
        (gd / f"{key}.json").write_text(
            json.dumps({"verdict": "FAIL" if key == fail_key else "PASS"})
        )
    sae_out = args.out_root / "sae_c_k200"
    sae_out.mkdir(parents=True, exist_ok=True)
    (sae_out / "gates_p4.json").write_text(
        json.dumps({"g4": {"verdict": "FAIL" if fail_key == "g4" else "PASS"}})
    )


@pytest.mark.parametrize(
    ("phase", "fail_key", "rc"),
    [
        ("sae_train", "gs", 33),
        ("sae_train", "ga", 30),
        ("densein", "g4", 25),
        ("census", "gc_i", 31),
        ("stats", "gc_ii", 31),
        ("figures", "gc_iii", 31),
        ("figures", "gs", 33),
    ],
)
def test_downstream_phase_reapplies_recorded_fail_before_heavy_work(
    tmp_path, monkeypatch, phase, fail_key, rc
):
    """r8 U2: entering ANY later --phase over a persisted FAIL re-applies the
    ORIGINAL rc before heavy work — a failed gate can never be laundered into
    produced/uploaded terminal results by phase re-entry."""
    args = _args(tmp_path)
    _write_gate_records(args, fail_key=fail_key)

    def boom(*a, **k):
        raise AssertionError("heavy path (_drv) reached on a failed-gate re-entry (r8 U2)")

    monkeypatch.setattr(K, "_drv", boom)  # the first heavy step of every phase
    with pytest.raises(SystemExit) as ei:
        K.PHASES[phase](args)
    assert ei.value.code == rc


def test_downstream_phase_missing_gate_record_fails_loud(tmp_path, monkeypatch):
    args = _args(tmp_path)  # no records written at all

    def boom(*a, **k):
        raise AssertionError("heavy path reached without required gate records")

    monkeypatch.setattr(K, "_drv", boom)
    with pytest.raises(RuntimeError, match="required upstream gate record missing"):
        K.phase_densein(args)


def test_require_upstream_gates_pass_records_no_exit(tmp_path):
    args = _args(tmp_path)
    _write_gate_records(args, fail_key=None)
    for phase in ("sae_train", "densein", "census", "stats", "figures"):
        K._require_upstream_gates(args, phase)  # all PASS: returns
    K._require_upstream_gates(args, "assemble")  # upstream phases require nothing


def test_require_upstream_gates_ignores_sanctioned_drop_record(tmp_path):
    """A sanctioned densein DROP (verdict DROPPED-companion) is NOT a gate FAIL
    — downstream phases proceed (plan §7: drop, never a round abort)."""
    args = _args(tmp_path)
    _write_gate_records(args, fail_key=None)
    (K._gates_dir(args) / "densein_dropped.json").write_text(
        json.dumps({"verdict": "DROPPED-companion", "reason_chain": "tb"})
    )
    K._require_upstream_gates(args, "stats")
    K._require_upstream_gates(args, "figures")


# ── r8 U6: the densein DROP reason chain survives into gates_k200.json ───────────


def test_densein_drop_reason_chain_durable_in_gates_consolidation(tmp_path):
    """r8 U6: a forced R4 fit failure records the COMPLETE reason_chain in the
    gates dir, and the R6 consolidation folds it into gates_k200.json — a
    git-allowlisted declared final artifact — so the plan's sanctioned DROP
    diagnostic survives pod teardown."""
    args = _args(tmp_path)
    try:
        raise ValueError("forced densein fit failure (fixture)")
    except ValueError as e:
        doc = K._record_densein_drop(args, e)
    assert doc["dropped"] is True and doc["verdict"] == "DROPPED-companion"
    assert "ValueError: forced densein fit failure" in doc["reason"]
    assert "Traceback" in doc["reason_chain"]
    for p in (
        K._census_dir(args) / "densein_dropped.json",  # resume-predicate marker
        K._gates_dir(args) / "densein_dropped.json",  # durable (consolidated) record
    ):
        assert "Traceback" in json.loads(p.read_text())["reason_chain"], p
    consolidated = K._consolidate_gates(args)
    assert "Traceback" in consolidated["gates"]["densein_dropped"]["reason_chain"]
    assert "gates_k200.json" in K.GIT_EVAL_BASENAMES  # the chain's git destination


# ── r8 U4: warm pass_b cache validates against the revision pin ──────────────────


def _tiny_pass_b(path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layers": torch.tensor([19]),
            "cx_last": torch.zeros(1, 2, 1),
            "v_x": torch.zeros(1, 2, 1),
        },
        path,
    )


def test_pass_b_warm_reuse_requires_matching_pin(tmp_path, monkeypatch):
    """r8 U4: a warm (pre-existing) pass_b file enters the pinned assembly ONLY
    with a sidecar recording the same revision pin; absent or mismatched
    provenance fails loud (never a silent unpinned read)."""
    local = tmp_path / "pass_b.pt"
    monkeypatch.setattr(D.N1G, "PASS_B_LOCAL", str(local))
    _tiny_pass_b(local)
    side = D._pass_b_pin_sidecar(local)
    with pytest.raises(RuntimeError, match="NO revision sidecar"):
        D._load_pass_b_wo()  # (a) unknown provenance
    side.write_text(json.dumps({"revision": "0" * 40}))
    with pytest.raises(RuntimeError, match="!= pinned"):
        D._load_pass_b_wo()  # (b) mismatched pin
    side.write_text(json.dumps({"revision": D.DATA_REPO_REVISION}))
    b = D._load_pass_b_wo()  # (c) matching pin loads
    assert {"layers", "cx_last", "v_x"} <= set(b.keys())


def test_pass_b_fetch_writes_pin_sidecar(tmp_path, monkeypatch):
    """The fetch path downloads at DATA_REPO_REVISION AND writes the sidecar;
    a subsequent warm reuse then passes the pin check without network
    (autospec'd network boundary; the fake materializes the file)."""
    import huggingface_hub

    local = tmp_path / "sub" / "pass_b.pt"
    monkeypatch.setattr(D.N1G, "PASS_B_LOCAL", str(local))
    fake = create_autospec(huggingface_hub.hf_hub_download)

    def materialize(repo_id, filename=None, **kw):
        _tiny_pass_b(local)
        return str(local)

    fake.side_effect = materialize
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    b = D._load_pass_b_wo()
    assert {"layers", "cx_last", "v_x"} <= set(b.keys())
    assert fake.call_args.kwargs["revision"] == D.DATA_REPO_REVISION
    rec = json.loads(D._pass_b_pin_sidecar(local).read_text())
    assert rec["revision"] == D.DATA_REPO_REVISION

    def no_network(*a, **k):
        raise AssertionError("network reached on a pinned warm reuse")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    D._load_pass_b_wo()  # warm second read: sidecar-validated, no fetch


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
