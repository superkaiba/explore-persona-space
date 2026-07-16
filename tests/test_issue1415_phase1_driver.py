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


def test_pilot_gate_refuses_over_threshold_unless_forced():
    slow = {"s_per_sample": drv.PILOT_MAX_S_PER_SAMPLE + 1.0}
    with pytest.raises(RuntimeError, match="refusing the full sweep"):
        drv._enforce_pilot_gate(slow, force=False)
    drv._enforce_pilot_gate(slow, force=True)  # --force overrides
    drv._enforce_pilot_gate({"s_per_sample": 0.1}, force=False)  # under threshold


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
