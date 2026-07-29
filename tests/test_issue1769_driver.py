"""#1769 driver --tiny e2e + resume-fingerprint + designed-gate probes.

The e2e runs the FULL production control flow (p0 -> pilot -> grid ->
finalize) at tiny-real shape on CPU (from-config 2-layer real-vocab Qwen;
plan §4: all 4 arms x 1 trait x 2 questions x 2 draws) into tmp roots —
smoke outputs never touch committed artifacts. P0 performs the REAL pinned
HF artifact reads (KB-scale, HF-cache-served after the first fetch).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue1769_run as run


@pytest.fixture(scope="module")
def tiny_roots(tmp_path_factory):
    root = tmp_path_factory.mktemp("i1769")
    out_root = root / "out"
    bulk_root = root / "bulk"
    with pytest.raises(SystemExit) as exc:
        run.main(
            [
                "--phase",
                "all",
                "--tiny",
                "--out-root",
                str(out_root),
                "--bulk-root",
                str(bulk_root),
            ]
        )
    assert exc.value.code == 0
    return out_root, bulk_root


def test_tiny_e2e_p0_artifacts(tiny_roots):
    out_root, _ = tiny_roots
    p0 = json.loads((out_root / "p0_artifact_check.json").read_text())
    assert set(p0["traits"]) == set(run.TRAITS)
    for trait, rec in p0["traits"].items():
        assert rec["rb_shape"] == [28, 3584], (trait, rec["rb_shape"])
        assert rec["eval_extraction_overlap"] == [], trait
        assert rec["eval_questions_provenance"] in ("original", "regenerated")
    assert p0["traits"]["evil"]["eval_questions_provenance"] == "original"


def test_tiny_e2e_pilot_written_gate_demoted(tiny_roots):
    out_root, _ = tiny_roots
    pilot = json.loads((out_root / "pilot.json").read_text())
    assert pilot["pilot_batch"] == 2
    assert pilot["n_samples"] == 4
    assert pilot["gate_enforced"] is False  # tiny demotes the verdict to a log line
    assert pilot["s_per_sample"] > 0


def test_tiny_e2e_grid_coverage_and_manifest(tiny_roots):
    out_root, bulk_root = tiny_roots
    manifest = json.loads((out_root / "cells_manifest.json").read_text())
    # tiny grid: 1 trait x (neither x 2q + 3 arms x 2 alphas x 2q) = 14 cells
    assert manifest["n_cells"] == 14
    assert len(manifest["cells"]) == 14
    for cid, rec in manifest["cells"].items():
        trait = cid.split("/")[0]
        comp = bulk_root / "raw_completions" / trait / rec["completion_file"]
        assert comp.exists(), cid
        payload = json.loads(comp.read_text())
        assert payload["cell_id"] == cid
        assert len(payload["draws"]) == 2
        assert len(payload["coherence_flags"]) == 2
        assert payload["seeds"] == [42, 43]
        # local-mirror upload ran through the identical call path
        mirrored = (
            bulk_root
            / "hf_mirror"
            / run.HF_OUT_PREFIX
            / "raw_completions"
            / trait
            / rec["completion_file"]
        )
        assert mirrored.exists(), cid
    fp = manifest["fingerprint"]
    assert fp["layer"] == 1 and fp["tiny"] is True
    assert fp["rb_revision"] == "tiny-randn"
    assert fp["code_sha"], "code SHA missing from the resume fingerprint"


def test_resume_skips_matching_fingerprint_and_reruns_mismatch(tiny_roots):
    out_root, bulk_root = tiny_roots
    cfg = run.tiny_config(out_root, bulk_root)
    fp = run.cell_fingerprint(cfg)
    cid = run.cell_id("evil", "neither", None, 0)
    assert run.cell_done(cfg, cid, fp) is True
    # A mismatched fingerprint (e.g. a code-fix round) re-runs the cell —
    # bare output existence never vouches for it (#952 gate-5 shape).
    stale = dict(fp, code_sha="0" * 40)
    assert run.cell_done(cfg, cid, stale) is False


def test_pilot_gate_halts_with_artifact_and_rc7_at_production_shape(tmp_path):
    """The G0 halt branch fires its DESIGNED handling (report JSON + rc=7) —
    the data-dependent-gate probe the tiny leg deliberately demotes."""
    cfg = run.RunConfig(tiny=False, out_root=tmp_path, bulk_root=tmp_path / "bulk")
    pilot = {"s_per_sample": 10.0, "pilot_batch": 8}
    report = tmp_path / "pilot_gate_report.json"
    with pytest.raises(SystemExit) as exc:
        run.enforce_pilot_gate(cfg, pilot, report)
    assert exc.value.code == run.RC_PILOT_GATE == 7
    rep = json.loads(report.read_text())
    assert rep["fired"] is True and "10.00 s/sample" in rep["reason"]


def test_pilot_gate_force_overrides(tmp_path):
    cfg = run.RunConfig(tiny=False, out_root=tmp_path, bulk_root=tmp_path / "bulk", force=True)
    run.enforce_pilot_gate(cfg, {"s_per_sample": 10.0, "pilot_batch": 8}, tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_finalize_coverage_assert_fires_on_missing_cell(tiny_roots, tmp_path):
    """Row-coverage gate: a missing cell fails finalize LOUD (designed raise)."""
    out_root, bulk_root = tiny_roots
    cfg = run.tiny_config(out_root, bulk_root)
    victim = run.cell_meta_path(cfg, run.cell_id("evil", "both", 1.0, 1))
    backup = tmp_path / "victim.json"
    backup.write_text(victim.read_text())
    victim.unlink()
    try:
        with pytest.raises(AssertionError, match="grid coverage incomplete"):
            run.phase_finalize(cfg)
    finally:
        victim.write_text(backup.read_text())
    # restored: finalize passes again
    manifest = run.phase_finalize(cfg)
    assert manifest["n_cells"] == 14


def test_expected_cells_production_grid_is_600():
    cfg = run.RunConfig(tiny=False, out_root=Path("unused"), bulk_root=Path("unused"))
    cells = run.expected_cells(cfg)
    assert len(cells) == 600  # 3 traits x (20 neither + 3 arms x 3 alphas x 20)
    assert len(set(cells)) == 600


def test_shard_partition_covers_all_groups_disjointly():
    cfg = run.RunConfig(tiny=False, out_root=Path("u"), bulk_root=Path("u"), n_shards=2)
    groups = run.enumerate_groups(cfg)
    seen = []
    for shard in range(2):
        cfg.shard = shard
        seen.extend((g["trait"], g["arm"], g["alpha"]) for g in run.shard_groups(cfg, groups))
    assert sorted(seen) == sorted((g["trait"], g["arm"], g["alpha"]) for g in groups)
