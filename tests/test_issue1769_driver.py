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


def _k2_cells(cfg, n_coherent_both: int):
    cells = {}
    for grp in run.enumerate_groups(cfg):
        for qid in range(cfg.n_questions):
            cid = run.cell_id(grp["trait"], grp["arm"], grp["alpha"], qid)
            nc = n_coherent_both if grp["arm"] == "both" else cfg.n_draws
            cells[cid] = {"n_coherent": nc, "n_draws": cfg.n_draws, "shard": 0}
    return cells


def test_k2_fires_only_when_all_trait_alphas_below_gate(tmp_path):
    cfg = run.RunConfig(tiny=False, out_root=tmp_path, bulk_root=tmp_path / "b")
    k2 = run.evaluate_k2(cfg, _k2_cells(cfg, n_coherent_both=0))  # 0% coherent
    assert k2["fired"] is True
    assert len(k2["both_arm_coherence_rates"]) == 9  # 3 traits x 3 alphas
    # ONE (trait, alpha) at the gate (>= 50%) keeps the ladder alive.
    cells = _k2_cells(cfg, n_coherent_both=0)
    for qid in range(cfg.n_questions):
        cells[run.cell_id("evil", "both", 1.0, qid)]["n_coherent"] = cfg.n_draws // 2
    assert run.evaluate_k2(cfg, cells)["fired"] is False


def test_k2_gate_halts_rc9_at_production_shape_and_demotes_under_tiny(tmp_path):
    prod = run.RunConfig(tiny=False, out_root=tmp_path, bulk_root=tmp_path / "b")
    fired = {"fired": True, "both_arm_coherence_rates": {"evil/a1": 0.1}}
    with pytest.raises(SystemExit) as exc:
        run.enforce_k2_gate(prod, fired)
    assert exc.value.code == run.RC_K2_GATE == 9
    tiny = run.tiny_config(tmp_path, tmp_path / "b")
    run.enforce_k2_gate(tiny, fired)  # demoted: no raise
    run.enforce_k2_gate(prod, {"fired": False})  # not fired: no raise


def test_judge_refuses_fired_k2(tiny_roots, tmp_path):
    """The J phase never submits 30k calls against a fired K2 (plan §7)."""
    import shutil

    import issue1769_judge as judge

    out_root, _ = tiny_roots
    j_root = tmp_path / "out"
    j_root.mkdir()
    shutil.copy2(out_root / "cells_manifest.json", j_root / "cells_manifest.json")
    (j_root / "k2_report.json").write_text(json.dumps({"fired": True}))
    with pytest.raises(AssertionError, match="K2 dose-ladder coherence gate FIRED"):
        judge.load_manifest(j_root)
    (j_root / "k2_report.json").write_text(json.dumps({"fired": False}))
    assert judge.load_manifest(j_root)["n_cells"] == 14


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


def test_fu1_alphas_and_hf_prefix_overrides_thread_into_config(tmp_path):
    """fu1 (plan v10): --alphas replaces the ladder, --hf-prefix rebinds every
    upload path; the fu1 grid enumerates 3 traits x (20 neither + 3 arms x
    2 alphas x 20q) = 420 cells with zero parent-prefix paths reachable."""
    args = run.parse_args(
        [
            "--out-root",
            str(tmp_path / "o"),
            "--bulk-root",
            str(tmp_path / "b"),
            "--alphas",
            "3.0",
            "1.5",
            "--hf-prefix",
            "issue1769_prefill_decode/fu1_alpha_subgrid",
        ]
    )
    cfg = run.build_config(args)
    assert cfg.alphas == (1.5, 3.0)  # sorted
    assert cfg.hf_prefix == "issue1769_prefill_decode/fu1_alpha_subgrid"
    assert len(run.expected_cells(cfg)) == 420


def test_default_config_parity_without_new_flags(tmp_path):
    """No --alphas / --hf-prefix => the parent ladder + parent HF prefix,
    byte-identical defaults (plan v10 acceptance criterion)."""
    args = run.parse_args(["--out-root", str(tmp_path / "o"), "--bulk-root", str(tmp_path / "b")])
    cfg = run.build_config(args)
    assert cfg.alphas == run.ALPHAS == (1.0, 2.0, 4.0)
    assert cfg.hf_prefix == run.HF_OUT_PREFIX == "issue1769_prefill_decode"
    tiny_args = run.parse_args(
        ["--tiny", "--out-root", str(tmp_path / "to"), "--bulk-root", str(tmp_path / "tb")]
    )
    tiny_cfg = run.build_config(tiny_args)
    assert tiny_cfg.alphas == (1.0, 2.0)  # tiny default ladder unchanged
    assert tiny_cfg.hf_prefix == run.HF_OUT_PREFIX


# ---------------------------------------------------------------------------
# fu1 judge-phase prefix isolation (concern fu1-judge-phase-parent-prefix-residuals)
# ---------------------------------------------------------------------------

FU1_PREFIX = "issue1769_prefill_decode/fu1_alpha_subgrid"


def test_judge_parse_args_parent_defaults_unchanged():
    """Flag-less judge defaults stay byte-identical to the parent run's paths."""
    import issue1769_judge as judge

    args = judge.parse_args([])
    assert args.hf_prefix == judge.HF_OUT_PREFIX == "issue1769_prefill_decode"
    assert args.out_root == judge.PARENT_OUT_ROOT == Path("eval_results/issue_1769/phase_g")
    assert args.completions_root == judge.PARENT_COMPLETIONS_ROOT
    assert args.work_dir == judge.PARENT_WORK_DIR == Path("data/issue_1769/judge_cache")
    assert args.judge_out == judge.PARENT_JUDGE_OUT == Path("eval_results/issue_1769/judge")
    judge.assert_prefix_isolation(args)  # parent prefix + parent defaults: no raise


def test_judge_fu1_prefix_isolation_guard(tmp_path):
    """A non-parent --hf-prefix with ANY parent-default local path fails loud
    BEFORE staging / API spend (never read parent cells / overwrite the
    parent's judge_raw, id_map, or committed graded_scores.json)."""
    import issue1769_judge as judge

    args = judge.parse_args(["--hf-prefix", FU1_PREFIX])
    with pytest.raises(AssertionError) as exc:
        judge.assert_prefix_isolation(args)
    for flag in ("--out-root", "--completions-root", "--work-dir", "--judge-out"):
        assert flag in str(exc.value)
    partial = [
        "--hf-prefix",
        FU1_PREFIX,
        "--out-root",
        str(tmp_path / "phase_g_fu1"),
        "--completions-root",
        str(tmp_path / "raw_completions_fu1"),
        "--work-dir",
        str(tmp_path / "judge_cache_fu1"),
    ]
    with pytest.raises(AssertionError, match="--judge-out"):
        judge.assert_prefix_isolation(judge.parse_args(partial))
    full = [*partial, "--judge-out", str(tmp_path / "judge_fu1")]
    judge.assert_prefix_isolation(judge.parse_args(full))  # all rebound: no raise


def test_judge_stage_completions_threads_hf_prefix(tmp_path, monkeypatch):
    """stage_completions stages from {hf_prefix}/raw_completions and reads the
    mirror back under the SAME prefix — real body, fake ONLY the network
    boundary (signature-conformant stage_hub_prefix)."""
    import issue1769_judge as judge

    from explore_persona_space.orchestrate import hub

    manifest = {
        "cells": {
            "evil/both/a1.5/q00": {"completion_file": "both_a1.5_q00_seed42.json"},
            "evil/pf/a3/q00": {"completion_file": "pf_a3_q00_seed42.json"},
        }
    }
    staged_prefixes: list[str] = []

    def fake_stage_hub_prefix(
        repo_id,
        prefix,
        dest_dir,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        max_workers=6,
    ):
        assert repo_id == judge.HF_DATA_REPO and repo_type == "dataset"
        staged_prefixes.append(prefix)
        out = []
        for rec in manifest["cells"].values():
            p = Path(dest_dir) / FU1_PREFIX / "raw_completions" / "evil" / rec["completion_file"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
            out.append(p)
        return out

    monkeypatch.setattr(hub, "stage_hub_prefix", fake_stage_hub_prefix)
    completions_root = tmp_path / "raw_completions_fu1"
    judge.stage_completions(manifest, completions_root, FU1_PREFIX)
    assert staged_prefixes == [f"{FU1_PREFIX}/raw_completions"]
    for rec in manifest["cells"].values():
        assert (completions_root / "evil" / rec["completion_file"]).exists()


def test_lattice_input_flags_default_to_parent_constants():
    """The lattice's --judge-scores / --raw-completions-dir defaults stay the
    parent constants (flag-less behavior byte-identical)."""
    import inspect

    import issue1769_alpha2_clean_lattice as lat

    sig_load = inspect.signature(lat.load_graded_scores)
    assert sig_load.parameters["path"].default == lat.GRADED_SCORES
    sig_flags = inspect.signature(lat.build_cjk_flags)
    assert sig_flags.parameters["raw_dir"].default == lat.RAW_COMPLETIONS_DIR
