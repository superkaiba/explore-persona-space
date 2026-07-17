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


# ── follow-up round: --replicate-l14 (l14-behavioral-replication) ─────


def _rep_cfg(tmp: Path, seed: int = 43, extra: list[str] | None = None) -> drv.RunConfig:
    """Tiny replication config rooted next to ``_cfg(tmp)``'s parent roots
    (shared tiny pair bank at tmp/pair_bank_tiny.json; delta_root tmp/bulk)."""
    argv = [
        "--tiny",
        "--replicate-l14",
        "--seed-base",
        str(seed),
        "--out-root",
        str(tmp / f"out_rep{seed}"),
        "--bulk-root",
        str(tmp / f"bulk_rep{seed}"),
        "--tiny-pairs",
        "2",
        "--n-draws",
        "2",
        "--max-new-tokens",
        "16",
        *(extra or []),
    ]
    return drv.build_config(drv.parse_args(argv))


def _fake_pairs(n: int = 28) -> list[dict]:
    return [
        {
            "pair_id": f"p{i:02d}",
            "pair_type": "matched",
            "ctx_c": {"system": None, "user": "q"},
            "ctx_cprime": {"system": "s", "user": "q"},
            "trait_or_behavior": "hedging",
        }
        for i in range(n)
    ]


def test_replication_cells_exact_selection():
    """Production replication cell set: EXACTLY 28 fresh baselines (c only —
    the ceiling arm is reused from the parent, per the proposal's 1,120+560 =
    1,680-sample arithmetic) + 2 arms x 28 fixed L14/a4 steered cells, all
    namespaced gen_rep43/."""
    from collections import Counter

    cfg = drv.build_config(drv.parse_args(["--replicate-l14", "--seed-base", "43"]))
    cells = drv._replication_cells(cfg, _fake_pairs())
    base = [c for c in cells if c.delta_key is None]
    steered = [c for c in cells if c.delta_key is not None]
    assert len(base) == 28 and len(steered) == 56 and len(cells) == 84
    assert all(c.cell_id.startswith("gen_rep43/") for c in cells)
    assert all(c.phase == "phase1b" and c.extra["arm_label"] == "hf_nohook_base" for c in base)
    assert all(c.context == {"system": None, "user": "q"} for c in base)  # under c ONLY
    assert all(c.phase == "phase1c_layers" and c.layer == 14 and c.alpha == 4.0 for c in steered)
    assert all(c.cell_id.endswith("/L14/a4") for c in steered)
    arms = Counter(c.delta_key[2] for c in steered)
    assert arms == Counter({"prefix": 28, "context": 28})
    assert len({c.cell_id for c in cells}) == 84  # no id collisions


def test_replication_config_namespace_seed_and_regime():
    cfg43 = drv.build_config(drv.parse_args(["--replicate-l14", "--seed-base", "43"]))
    cfg44 = drv.build_config(drv.parse_args(["--replicate-l14", "--seed-base", "44"]))
    # separate out/bulk namespaces keyed by seed base; frozen parent delta root
    assert cfg43.out_root.name == "phase1_rep43" and cfg43.bulk_root.name == "phase1_rep43"
    assert cfg44.out_root.name == "phase1_rep44"
    assert cfg43.delta_root == drv.REPO_ROOT / "data" / "issue_1415" / "phase1"
    assert cfg43.rep_layer == drv.REPLICATION_LAYER_FULL == 14
    # per-draw seed ranges (seed_base + i) disjoint from the parent's 42..51
    # AND from each other — literal bases 43/44 would re-draw the parent seeds
    assert cfg43.seed_base == 43000 and cfg44.seed_base == 44000
    r43 = {cfg43.seed_base + i for i in range(cfg43.n_draws)}
    r44 = {cfg44.seed_base + i for i in range(cfg44.n_draws)}
    parent = {drv.SEED_BASE + i for i in range(drv.N_DRAWS_FULL)}
    assert not (r43 & r44) and not ((r43 | r44) & parent)
    # regime carries the replication keys; the parent regime stays byte-stable
    reg = drv._regime(cfg43, "sha")
    assert reg["mode"] == "replicate_l14" and reg["rep_seed_base"] == 43
    assert reg["seed_base"] == 43000 and reg["rep_layer"] == 14 and reg["rep_alpha"] == 4.0
    normal = drv._regime(drv.build_config(drv.parse_args([])), "sha")
    assert "mode" not in normal and "rep_seed_base" not in normal


def test_replication_arg_validation():
    with pytest.raises(SystemExit):
        drv.build_config(drv.parse_args(["--replicate-l14"]))  # --seed-base required
    with pytest.raises(SystemExit):
        drv.build_config(drv.parse_args(["--replicate-l14", "--seed-base", "43", "--pilot"]))
    with pytest.raises(SystemExit):
        drv.build_config(drv.parse_args(["--seed-base", "43"]))  # replication-only flag
    with pytest.raises(SystemExit):
        drv.build_config(drv.parse_args(["--delta-root", "/tmp/x"]))  # replication-only flag


@pytest.fixture(scope="module")
def rep_run(first_run):
    """Tiny e2e replication run consuming the parent tiny run's FROZEN deltas
    (delta_root defaults to tmp/bulk — the first_run bulk root)."""
    tmp, _parent_cfg, _summary = first_run
    cfg = _rep_cfg(tmp, seed=43)
    summary = drv.run_replication(cfg)
    return tmp, cfg, summary


def test_replication_tiny_e2e_cells_and_namespace(rep_run):
    _tmp, cfg, summary = rep_run
    manifest = json.loads((cfg.out_root / "phase1_manifest.json").read_text())
    assert manifest["regime"]["mode"] == "replicate_l14"
    assert manifest["regime"]["seed_base"] == 43000
    cells = {cid for cid in manifest["cells"] if not cid.startswith("upload/")}
    expected = {f"gen_rep43/base/tiny_{i:02d}/c" for i in range(2)} | {
        f"gen_rep43/{arm}/tiny_{i:02d}/L0/a4" for arm in drv.EXTRACTION_ARMS for i in range(2)
    }
    assert cells == expected and summary["cells_run"] == len(expected) == 6
    for cid in sorted(expected):
        meta = drv.load_cell_meta(cfg, cid)
        assert meta["seed_base"] == 43000 and meta["rep_seed_base"] == 43
        assert meta["coherence_flags"] is not None  # coherence still recorded per cell
        comp = cfg.bulk_root / meta["completions_file"]
        assert len(json.loads(comp.read_text())["draws"]) == cfg.n_draws
    # phase-boundary upload boundary exercised: ONE gen_rep43 bucket commit
    mirror = cfg.bulk_root / "hf_mirror" / drv.RAW_PREFIX / "gen_rep43"
    assert mirror.exists() and any(mirror.rglob("*.json"))
    # deltas were REUSED from the parent bulk root — no fresh 1a capture ran
    assert not (cfg.bulk_root / "activations").exists()


def test_replication_resume_skips_all(rep_run):
    tmp, _cfg, summary = rep_run
    second = drv.run_replication(_rep_cfg(tmp, seed=43))
    assert second["cells_run"] == 0
    assert second["cells_skipped"] == summary["cells_run"]
    assert second["uploads"] == 0  # unchanged file count -> upload skipped


def test_replication_delta_hf_fetch_fallback(tmp_path, monkeypatch):
    """Production replication HF-fetches a missing frozen parent capture
    (fresh git-clone instances stage no data/ — #779); the hub boundary is a
    signature-conformant autospec fake."""
    from unittest.mock import create_autospec

    import huggingface_hub
    import torch

    args = drv.parse_args(
        ["--replicate-l14", "--seed-base", "43", "--delta-root", str(tmp_path / "parent")]
    )
    cfg = drv.build_config(args)
    n_layers, hid = len(cfg.layers), cfg.hidden
    blob = {
        "pair_id": "p00",
        "layers": list(cfg.layers),
        "c": {
            "v_c_prefix": torch.zeros(n_layers, hid),
            "v_c_context": torch.zeros(n_layers, hid),
        },
        "cprime": {
            "v_c_prefix": torch.ones(n_layers, hid),
            "v_c_context": torch.ones(n_layers, hid),
        },
    }
    staged = tmp_path / "hf_cache" / "p00.pt"
    staged.parent.mkdir(parents=True)
    torch.save(blob, staged)

    def impl(repo_id, filename, **kw):
        assert repo_id == drv.HF_DATA_REPO
        assert filename == f"{drv.TENSOR_PREFIX}/p00.pt"
        assert kw.get("repo_type") == "dataset"
        return str(staged)

    fake = create_autospec(huggingface_hub.hf_hub_download, side_effect=impl)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    deltas = drv.DeltaSource(cfg)
    d = deltas.pair_delta("p00", "context", cfg.rep_layer)
    assert d.shape == (cfg.hidden,) and torch.all(d == 1.0)
    assert (tmp_path / "parent" / "activations" / "p00.pt").exists()  # copied under delta_root
    assert fake.call_count == 1
    fresh = drv.DeltaSource(cfg)  # local now — no re-fetch
    fresh.pair_delta("p00", "prefix", cfg.rep_layer)
    assert fake.call_count == 1


def test_replication_pair_bank_content_sha_gate(tmp_path, monkeypatch):
    """The frozen-parent-bank premise is gated on the PAIRS-CONTENT sha
    (metadata-independent — a rebuilt bank differs byte-wise); a foreign bank
    fails loud, the pinned content passes."""
    bank = {"metadata": {"issue": 1415}, "pairs": _fake_pairs()}
    path = tmp_path / "pair_bank.json"
    path.write_text(json.dumps(bank))
    cfg = drv.build_config(
        drv.parse_args(["--replicate-l14", "--seed-base", "43", "--pair-bank", str(path)])
    )
    with pytest.raises(RuntimeError, match="pairs-content sha mismatch"):
        drv.load_pairs(cfg)
    monkeypatch.setattr(drv, "PARENT_PAIR_BANK_PAIRS_SHA256", drv._pairs_content_sha(bank))
    assert len(drv.load_pairs(cfg)) == 28
