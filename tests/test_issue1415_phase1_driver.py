"""CPU tests for the issue-1415 phase-1 driver (deliverable 3).

Exercises the manifest/resume logic and the FULL ``--tiny`` control flow
(from-config 2-layer Qwen on CPU — the committed unit-test pattern). Only the
HF upload boundary is mocked/diverted: the default tiny ``local-mirror`` mode
exercises the identical ``upload_artifact`` call path against a local mirror,
and the ``hf``-mode test replaces ``_hf_upload`` with a signature-conformant
autospec fake.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1415_run_phase1 as drv  # noqa: E402


def _cfg(tmp: Path, extra: list[str] | None = None) -> drv.RunConfig:
    argv = [
        "--tiny",
        "--out-root",
        str(tmp / "out"),
        "--bulk-root",
        str(tmp / "bulk"),
        "--tiny-pairs",
        "2",
        "--n-draws",
        "2",
        "--max-new-tokens",
        "16",
        *(extra or []),
    ]
    return drv.build_config(drv.parse_args(argv))


# ── pure selection / gate logic ───────────────────────────────────────


def test_select_operating_alpha_walks_grid_down():
    grid = (0.5, 1.0, 2.0, 4.0)
    flags = {4.0: [False, False], 2.0: [True, False], 1.0: [True, True], 0.5: [True, True]}
    assert drv.select_operating_alpha(flags, grid) == 2.0  # largest PASSING alpha
    all_fail = {a: [False, False] for a in grid}
    assert drv.select_operating_alpha(all_fail, grid) is None


def test_select_trait_alpha_majority_over_subset():
    grid = (0.5, 1.0, 2.0, 4.0)
    by_alpha = {
        4.0: [[False, False], [False, False], [True, True]],  # 1/3 pairs pass -> fail
        2.0: [[True, True], [False, False], [True, True]],  # 2/3 pairs pass -> pass
        1.0: [[True, True], [True, True], [True, True]],
        0.5: [[True, True], [True, True], [True, True]],
    }
    assert drv.select_trait_alpha(by_alpha, grid) == 2.0
    assert drv.select_trait_alpha({a: [[False, False]] for a in grid}, grid) is None


def test_pilot_gate_halts_designed_rc7_with_report(tmp_path):
    """A genuine pilot-gate refusal is a DESIGNED HALT (crash-fix for
    att-20260716-160022): pilot_gate_report.json persisted into out_root +
    SystemExit(RC_PILOT_GATE=7) — never an anonymous RuntimeError crash."""
    cfg = _cfg(tmp_path)
    slow = {"s_per_sample": drv.PILOT_MAX_S_PER_SAMPLE + 1.0, "pilot_batch": 8}
    with pytest.raises(SystemExit) as ei:
        drv._enforce_pilot_gate(cfg, slow)
    assert ei.value.code == drv.RC_PILOT_GATE == 7
    report = json.loads((cfg.out_root / "pilot_gate_report.json").read_text())
    assert report["fired"] is True
    assert report["pilot"]["s_per_sample"] == slow["s_per_sample"]
    assert "§13" in report["descope_pointer"]
    assert "B=8" in report["reason"]
    # --force overrides; under threshold passes (no exit, no new report)
    forced = _cfg(tmp_path, extra=["--force"])
    drv._enforce_pilot_gate(forced, slow)
    drv._enforce_pilot_gate(cfg, {"s_per_sample": 0.1, "pilot_batch": 8})


def test_hub_upload_call_shape_binds():
    """Pin the smoke-fenced hub._upload call shape against the real signature
    (the production 'hf' branch never executes in tests — #1332 bind rule)."""
    from explore_persona_space.orchestrate import hub

    sig = inspect.signature(hub._upload)
    sig.bind(
        Path("x"),
        repo_id="r",
        repo_type="dataset",
        path_in_repo="p",
        upload_as_file=True,
    )


# ── full --tiny control flow + manifest/resume ────────────────────────


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("i1415_driver")
    cfg = _cfg(tmp)
    summary = drv.run_phase1(cfg)
    return tmp, cfg, summary


def test_tiny_flow_artifacts(first_run):
    _tmp, cfg, summary = first_run
    assert summary["cells_run"] > 0 and summary["cells_skipped"] == 0
    assert summary["uploads"] >= 2  # pilot bucket + gen1b at minimum

    manifest = json.loads((cfg.out_root / "phase1_manifest.json").read_text())
    assert manifest["regime"]["n_draws"] == 2
    cells = manifest["cells"]
    assert "pilot" in cells
    # 1b: 2 pairs x 2 arms; 1a: one capture per pair
    for pid in ("tiny_00", "tiny_01"):
        assert f"gen1b/{pid}/c" in cells and f"gen1b/{pid}/cprime" in cells
        assert f"capture1a/{pid}" in cells
        assert (cfg.bulk_root / "activations" / f"{pid}.pt").exists()
    # 1c grid: both extraction arms x pairs x full alpha grid at the primary layer
    prim = cfg.primary_layer
    for arm in drv.EXTRACTION_ARMS:
        for pid in ("tiny_00", "tiny_01"):
            for a in cfg.alpha_grid:
                cid = f"gen1c/{arm}/{pid}/L{prim}/a{drv._fmt(a)}"
                assert cid in cells, cid
                meta = drv.load_cell_meta(cfg, cid)
                assert meta["coherence_flags"] is not None
                assert len(meta["chunk_members"]) >= 1
                comp = cfg.bulk_root / meta["completions_file"]
                assert comp.exists()
                assert len(json.loads(comp.read_text())["draws"]) == cfg.n_draws
    # selection records exist for every (arm, pair)
    sel_1c = json.loads((cfg.out_root / "alpha_selection_1c.json").read_text())["selection"]
    assert set(sel_1c) == {f"{arm}/tiny_{i:02d}" for arm in drv.EXTRACTION_ARMS for i in range(2)}
    # 1d: search cells over the subset x traits x grid + selection record
    for trait in drv.TRAITS:
        for a in cfg.alpha_grid:
            assert f"gen1d_search/{trait}/tiny_00/a{drv._fmt(a)}" in cells
    sel_1d = json.loads((cfg.out_root / "alpha_selection_1d.json").read_text())["selection"]
    assert set(sel_1d) == set(drv.TRAITS)
    # pilot artifact + upload boundary exercised (local mirror populated)
    pilot = json.loads((cfg.out_root / "pilot.json").read_text())
    assert pilot["s_per_sample"] > 0 and pilot["threshold_s_per_sample"] == 4.7
    mirror = cfg.bulk_root / "hf_mirror"
    assert (mirror / drv.RAW_PREFIX / "pilot" / "std.json").exists()
    assert (mirror / drv.RAW_PREFIX / "gen1b" / "tiny_00" / "c.json").exists()
    assert (mirror / drv.TENSOR_PREFIX / "tiny_00.pt").exists()


def test_pilot_measured_at_sweep_chunk_shape(first_run):
    """Crash-fix att-20260716-160022: the pilot replicates its context to
    B = gen_batch identical rows (the sweep's chunk shape — a batch-1 pilot
    over-reads s/sample by ~B on bandwidth-bound decode) and normalizes
    s/sample by 2 x B x n_draws; row 0 stays the canonical pilot draw set
    (K2's pilot/std unit), all B rows' draws persist."""
    _tmp, cfg, _summary = first_run
    pilot = json.loads((cfg.out_root / "pilot.json").read_text())
    assert pilot["pilot_batch"] == cfg.gen_batch == 8
    assert pilot["n_samples"] == 2 * cfg.gen_batch * cfg.n_draws
    assert pilot["s_per_sample"] == pytest.approx(
        sum(pilot["timings_s"].values()) / pilot["n_samples"]
    )
    # K2's pilot/std unit semantics UNCHANGED: n_draws flags from the canonical row
    assert len(pilot["coherence_flags"]["std"]) == cfg.n_draws
    for variant in ("std", "allpos"):
        raw = json.loads(
            (cfg.bulk_root / "raw_completions" / "pilot" / f"{variant}.json").read_text()
        )
        assert raw["pilot_batch"] == cfg.gen_batch and raw["canonical_row"] == 0
        assert len(raw["all_rows_draws"]) == cfg.gen_batch
        assert all(len(rows) == cfg.n_draws for rows in raw["all_rows_draws"])
        assert raw["draws"] == raw["all_rows_draws"][0]


def test_rerun_skips_every_completed_cell(first_run):
    tmp, _cfg_unused, summary = first_run
    second = drv.run_phase1(_cfg(tmp))
    assert second["cells_run"] == 0
    assert second["cells_skipped"] == summary["cells_run"] + summary["cells_skipped"]
    assert second["uploads"] == 0  # unchanged file counts -> uploads skipped too


def test_partial_resume_reruns_only_missing_cell(first_run):
    tmp, cfg, _ = first_run
    mpath = cfg.out_root / "phase1_manifest.json"
    manifest = json.loads(mpath.read_text())
    dropped = "gen1b/tiny_01/cprime"
    assert dropped in manifest["cells"]
    del manifest["cells"][dropped]
    mpath.write_text(json.dumps(manifest))
    third = drv.run_phase1(_cfg(tmp))
    assert third["cells_run"] == 1  # exactly the dropped cell re-ran
    assert json.loads(mpath.read_text())["cells"].get(dropped)


def test_regime_mismatch_fails_loud(first_run):
    tmp, _, _ = first_run
    with pytest.raises(RuntimeError, match="regime mismatch"):
        drv.run_phase1(_cfg(tmp, extra=["--n-draws", "3"]))


def test_pilot_mode_hf_upload_boundary_mocked(tmp_path, monkeypatch):
    """--pilot --upload hf with ONLY the HF boundary mocked (autospec keeps the
    fake signature-conformant); asserts the upload lands under RAW_PREFIX."""
    from unittest.mock import create_autospec

    fake = create_autospec(drv._hf_upload)
    monkeypatch.setattr(drv, "_hf_upload", fake)
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-test")
    cfg = _cfg(tmp_path, extra=["--pilot", "--upload", "hf"])
    summary = drv.run_phase1(cfg)
    assert summary["pilot"]["s_per_sample"] > 0
    assert fake.call_count == 1
    (local, remote), _kw = fake.call_args
    assert Path(local) == cfg.bulk_root / "raw_completions" / "pilot"
    assert remote == f"{drv.RAW_PREFIX}/pilot"


# ── round 2: phase 1e (steered V_a capture) via the tiny flow ─────────


def test_tiny_flow_phase1e_and_kill_reports(first_run):
    _tmp, cfg, summary = first_run
    manifest = json.loads((cfg.out_root / "phase1_manifest.json").read_text())
    cells = manifest["cells"]
    import torch

    # every steered grid cell got a phase-1e capture (per-cell .pt + manifest mark)
    prim = cfg.primary_layer
    for arm in drv.EXTRACTION_ARMS:
        for pid in ("tiny_00", "tiny_01"):
            for a in cfg.alpha_grid:
                cid = f"gen1c/{arm}/{pid}/L{prim}/a{drv._fmt(a)}"
                assert f"capture1e/{cid}" in cells, cid
                blob = torch.load(
                    cfg.bulk_root / "activations_steered" / f"{cid}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                if not blob.get("all_empty"):
                    assert blob["v_a_mean"].shape == (len(cfg.layers), cfg.hidden)
                    assert blob["layers"] == list(cfg.layers)

    # canonical map-transport files exist EXACTLY for the operating-alpha pairs
    sel = json.loads((cfg.out_root / "alpha_selection_1c.json").read_text())["selection"]
    idx = json.loads((cfg.out_root / "steered_canonical_index.json").read_text())["index"]
    for key, rec in sel.items():
        arm, pid = key.split("/", 1)
        canon = cfg.bulk_root / "activations_steered" / f"{pid}__{arm}.pt"
        if rec["operating_alpha"] is None:
            assert idx[key] == {"skipped": "coherence_failed_all_alpha"}
            assert not canon.exists()
        else:
            assert canon.exists(), key
            assert idx[key]["alpha"] == rec["operating_alpha"]
            blob = torch.load(canon, map_location="cpu", weights_only=True)
            assert blob["canonical_of"] == idx[key]["canonical_of"]

    # steered captures rode the upload boundary (local mirror in tiny mode)
    mirror = cfg.bulk_root / "hf_mirror" / drv.STEERED_TENSOR_PREFIX
    assert mirror.exists() and any(mirror.rglob("*.pt"))

    # kill-criteria verdicts computed + persisted, aborts DEMOTED under tiny
    k1 = json.loads((cfg.out_root / "k1_report.json").read_text())
    k2 = json.loads((cfg.out_root / "k2_report.json").read_text())
    assert k1["enforced"] is False and k2["enforced"] is False
    assert summary["k1"]["fired"] == k1["fired"]
    assert summary["k2"]["fired"] == k2["fired"]
    assert k1["threshold_frac"] == drv.K1_NO_SEP_FRAC
    assert k2["threshold_frac"] == drv.K2_FAIL_FRAC


# ── round 2: K1/K2 kill criteria (production semantics, unit-pinned) ──


def _write_k1_captures(cfg, pair_ids, separated: bool, n_draws: int = 6, seed: int = 0):
    """Synthetic 1a captures carrying v_a_per_completion for evaluate_k1."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    n_layers, hid = len(cfg.layers), cfg.hidden
    for pid in pair_ids:
        a_c = torch.randn(n_draws, n_layers, hid, generator=gen)
        if separated:
            offset = torch.randn(n_layers, hid, generator=gen) * 10.0
            a_cp = offset + torch.randn(n_draws, n_layers, hid, generator=gen) * 0.1
        else:
            a_cp = torch.randn(n_draws, n_layers, hid, generator=gen)  # pure noise
        drv._save_pt_atomic(
            cfg.bulk_root / "activations" / f"{pid}.pt",
            {
                "pair_id": pid,
                "layers": list(cfg.layers),
                "c": {"v_a_per_completion": a_c},
                "cprime": {"v_a_per_completion": a_cp},
            },
        )


def test_k1_fires_on_noise_and_passes_on_separation(tmp_path):
    cfg_noise = _cfg(tmp_path / "noise")
    pairs = [{"pair_id": f"p{i}"} for i in range(6)]
    _write_k1_captures(cfg_noise, [p["pair_id"] for p in pairs], separated=False)
    rep = drv.evaluate_k1(cfg_noise, pairs)
    assert rep["n_evaluable"] == 6
    assert rep["frac_no_separation"] == 1.0 and rep["fired"] is True

    cfg_sep = _cfg(tmp_path / "sep")
    _write_k1_captures(cfg_sep, [p["pair_id"] for p in pairs], separated=True)
    rep2 = drv.evaluate_k1(cfg_sep, pairs)
    assert rep2["frac_no_separation"] == 0.0 and rep2["fired"] is False
    # every separated pair beats the random-direction band decisively
    for v in rep2["per_pair"].values():
        assert v["max_over_layers"] > rep2["null_band_p975"]


def test_k1_insufficient_draws_excluded_not_fired(tmp_path):
    cfg = _cfg(tmp_path)
    _write_k1_captures(cfg, ["solo"], separated=False, n_draws=1)
    rep = drv.evaluate_k1(cfg, [{"pair_id": "solo"}])
    assert rep["n_evaluable"] == 0 and rep["frac_no_separation"] is None
    assert rep["fired"] is False
    assert rep["per_pair"]["solo"]["reason"] == "insufficient_kept_draws"


def _write_k2_grid_metas(cfg, pair_ids, coherent: bool):
    for arm in drv.EXTRACTION_ARMS:
        for pid in pair_ids:
            for a in cfg.alpha_grid:
                cid = f"gen1c/{arm}/{pid}/L{cfg.primary_layer}/a{drv._fmt(a)}"
                drv._write_json_atomic(
                    drv._cell_meta_path(cfg, cid),
                    {"cell_id": cid, "coherence_flags": [coherent, coherent]},
                )


def test_k2_fires_on_coherence_collapse(tmp_path):
    cfg = _cfg(tmp_path)
    pairs = [{"pair_id": f"p{i}"} for i in range(2)]
    _write_k2_grid_metas(cfg, [p["pair_id"] for p in pairs], coherent=False)
    pilot = {"coherence_flags": {"std": [False, False]}}
    rep = drv.evaluate_k2(cfg, pilot, pairs)
    assert rep["n_units"] == 5  # pilot + 2 pairs x 2 arms
    assert rep["frac_failed"] == 1.0 and rep["fired"] is True

    _write_k2_grid_metas(cfg, [p["pair_id"] for p in pairs], coherent=True)
    rep2 = drv.evaluate_k2(cfg, {"coherence_flags": {"std": [True, True]}}, pairs)
    assert rep2["frac_failed"] == 0.0 and rep2["fired"] is False


def test_enforce_kill_aborts_at_production_demotes_otherwise(tmp_path):
    import dataclasses

    cfg_tiny = _cfg(tmp_path)  # tiny -> enforce_kill_criteria False
    assert cfg_tiny.enforce_kill_criteria is False
    fired = {"fired": True, "criterion": "test", "frac": 1.0}
    drv._enforce_kill(cfg_tiny, fired, "k1", drv.RC_K1_ABORT)  # demoted, no raise

    cfg_prod = dataclasses.replace(cfg_tiny, enforce_kill_criteria=True)
    with pytest.raises(SystemExit) as ei:
        drv._enforce_kill(cfg_prod, fired, "k1", drv.RC_K1_ABORT)
    assert ei.value.code == drv.RC_K1_ABORT
    with pytest.raises(SystemExit) as ei:
        drv._enforce_kill(cfg_prod, fired, "k2", drv.RC_K2_HALT)
    assert ei.value.code == drv.RC_K2_HALT
    drv._enforce_kill(cfg_prod, {"fired": False, "criterion": "test"}, "k1", drv.RC_K1_ABORT)


def test_ignore_kill_criteria_flag_demotes_in_production_config():
    args = drv.parse_args(["--ignore-kill-criteria"])
    # full (non-tiny) config path; no CUDA needed to BUILD the config
    cfg = drv.build_config(args)
    assert cfg.tiny is False and cfg.enforce_kill_criteria is False
    cfg_default = drv.build_config(drv.parse_args([]))
    assert cfg_default.enforce_kill_criteria is True
