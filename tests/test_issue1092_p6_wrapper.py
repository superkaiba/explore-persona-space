"""Tests for scripts/issue1092_p6_run.py — P6 layer-sliced staging wrapper + pilot gate.

Tiny-real standard: the wrapper body (staging, pilot, layer loop, deletion,
resume) and the fit-grid subprocess run REAL; only the Hub boundary is a local
fixture tree behind the injectable HubIO seam (LocalFixtureHubIO mirrors
HfHubIO's method signatures by construction).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1092_fit_grid as fit_grid  # noqa: E402
import issue1092_p6_run as p6  # noqa: E402

HIDDEN = 8
N_ROWS = 12
LAYERS = (14, 15)  # 14 frozen (production draws), 15 band
CELLS = ("cell_inst_own", "cell_pre_own")
PREFIX = p6.DEFAULT_HF_PREFIX
QUERIES = tuple(f"q{i}" for i in range(4))

# tiny knobs threaded to the fit grid in every run below (--n-folds is now a
# first-class wrapper flag pinning the plan-registered fold count, so the tiny
# override goes through the wrapper arg, not --fit-grid-arg pass-through)
TINY_FIT_GRID_EXTRAS = (
    "--fit-grid-arg=--skip-mlp-companion",
    "--fit-grid-arg=--matched-n-draws=2",
    "--fit-grid-arg=--hidden-dim=8",
)


def _write_store(base: Path, cells=CELLS, layers=LAYERS) -> Path:
    """Write a tiny-real summaries tree (cells + bare + dynamics + b0) under base."""
    rng = np.random.default_rng(1092)
    model_types = sorted({fit_grid.CELL_MODEL_TYPE[c] for c in cells})
    for cell in cells:
        d = base / cell
        d.mkdir(parents=True)
        for layer in layers:
            for kind in ("prefix_end", "context_end"):
                np.save(
                    d / f"{kind}_L{layer:02d}.npy",
                    rng.normal(size=(N_ROWS, HIDDEN)).astype(np.float32),
                )
            half = N_ROWS // 2
            np.save(
                d / f"t1_L{layer:02d}_shard00000.npy",
                rng.normal(size=(half, HIDDEN)).astype(np.float32),
            )
            np.save(
                d / f"t1_L{layer:02d}_shard00001.npy",
                rng.normal(size=(N_ROWS - half, HIDDEN)).astype(np.float32),
            )
    for mt in model_types:
        d = base / f"bare_{mt}"
        d.mkdir()
        for layer in layers:
            np.save(
                d / f"c_q_bare_L{layer:02d}.npy",
                rng.normal(size=(len(QUERIES), HIDDEN)).astype(np.float32),
            )
        d.joinpath("row_index.jsonl").write_text(
            "".join(json.dumps({"query_id": q}) + "\n" for q in QUERIES)
        )
        dd = base / f"dynamics_{mt}"
        dd.mkdir()
        conv_rows = [
            {"conv_id": f"c{ci}", "turn_index": t, "token_start": t, "token_end": t + 1}
            for ci in range(6)
            for t in (0, 2, 4, 6)
        ]
        for kind in p6.DYNAMICS_KINDS:
            for layer in layers:
                np.save(
                    dd / f"{kind}_L{layer:02d}.npy",
                    rng.normal(size=(len(conv_rows), HIDDEN)).astype(np.float32),
                )
            dd.joinpath(f"row_index_{kind}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in conv_rows)
            )
    b0 = base / "b0_rB_pool"
    b0.mkdir()
    for cell in cells:
        if cell in p6.B0_CELLS:
            np.save(b0 / f"{cell}.npy", rng.normal(size=(N_ROWS, 16, 3, 4)).astype(np.float32))
    return base


def _write_corpus(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = [
        {
            "row_id": f"r{i}",
            "prefix_id": f"p{i % 3}",
            "query_id": QUERIES[i % len(QUERIES)],
            "stratum": "dense_core",
        }
        for i in range(N_ROWS)
    ]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _write_rb(root: Path) -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(7)
    rb = root / "rb.npy"
    # (n_rb_layers, n_traits, hidden); must cover layer index 15 (frozen gate uses 14)
    np.save(rb, rng.normal(size=(16, 3, HIDDEN)).astype(np.float32))
    (root / "trait_names.json").write_text(json.dumps(["evil", "sycophancy", "hallucination"]))
    return rb


def _args(stage_dir: Path, out_dir: Path, hub_root: Path, corpus: Path, rb: Path, extra=()):
    return p6.parse_args(
        [
            "--corpus-dir",
            str(corpus),
            "--stage-dir",
            str(stage_dir),
            "--out-dir",
            str(out_dir),
            "--cells",
            ",".join(CELLS),
            "--layers",
            "14,15",
            "--targets",
            "t1",
            "--target-bases",
            "ambient",
            "--n-folds",
            "2",
            "--n-null-draws",
            "4",
            "--band-null-draws",
            "2",
            "--rb-dir",
            str(rb),
            "--pilot-cell",
            "cell_inst_own",
            "--pilot-layer",
            "14",
            "--fixture-hub-root",
            str(hub_root),
            *TINY_FIT_GRID_EXTRAS,
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def shared_fixture(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p6shared")
    hub_root = tmp / "hub"
    _write_store(hub_root / PREFIX)
    corpus = _write_corpus(tmp / "corpus")
    rb = _write_rb(tmp / "rb")
    return {"hub_root": hub_root, "corpus": corpus, "rb": rb}


@pytest.fixture(scope="module")
def e2e(shared_fixture, tmp_path_factory):
    """One full default-flow run (pilot + band pilot + 2-layer loop), shared by asserts."""
    tmp = tmp_path_factory.mktemp("p6e2e")
    stage, out = tmp / "stage", tmp / "out"
    args = _args(stage, out, **shared_fixture)
    summary = p6.run_p6(args)
    return {"stage": stage, "out": out, "summary": summary, "shared": shared_fixture}


def test_e2e_pilot_json_and_projection_arithmetic(e2e):
    pilot = json.loads((e2e["out"] / "pilot.json").read_text())
    gate = pilot["abort_predicate_result"]
    assert gate["abort"] is False
    assert pilot["escape_lane"] == "cpu-bigmem --min-ram-gb 32"
    assert pilot["wall_s_frozen_block"] > 0 and pilot["wall_s_band_block"] > 0
    assert pilot["ru_maxrss_gb"] > 0
    # 2 cells x 1 frozen layer (14) / 2 cells x 1 band layer (15)
    assert pilot["n_blocks_frozen"] == 2 and pilot["n_blocks_band"] == 2
    assert pilot["effective_parallelism"] == 1
    expected = (2 * pilot["wall_s_frozen_block"] + 2 * pilot["wall_s_band_block"]) / 3600.0
    assert pilot["projected_total_wall_h"] == pytest.approx(expected)
    assert "projected_total_wall_h =" in pilot["arithmetic"]
    assert pilot["git_commit"] and pilot["timestamp"]


def test_e2e_layer_manifests_checkpoints_and_deletion(e2e):
    out, stage = e2e["out"], e2e["stage"]
    assert e2e["summary"]["layers_done"] == [14, 15]
    for layer in LAYERS:
        manifest = json.loads((out / "staging" / f"staging_manifest_L{layer:02d}.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["n_staged_deleted"] == len(manifest["files"])
        for rec in manifest["files"]:
            assert rec["hub_path"].startswith(PREFIX + "/")
            assert len(rec["local_sha256"]) == 64
            assert Path(rec["staged_to"]).resolve().is_relative_to(stage.resolve())
        assert manifest["registered_inputs"]["missing"] == [p6.EXPECTED_MISSING_WITHOUT_JUDGE]
        # statics are part of the recorded staging identity (wrong-skip guard)
        assert {f["hub_path"].split("/")[-2] for f in manifest["static_files"]} >= {"b0_rB_pool"}
        # plan-registered fit config + lambda-grid deviation ride wrapper_config
        cfg = manifest["wrapper_config"]
        assert cfg["n_folds"] == 2 and cfg["fit_seed"] == 0
        assert cfg["lambda_grid_deviation"]["status"] == "deviation-recorded"
        for cell in CELLS:
            for arm in ("prefix_end", "context_end"):
                for fit_arm in ("A", "B"):
                    stem = f"{cell}_{arm}_fit{fit_arm}_L{layer:02d}_ambient_"
                    assert list((out / "checkpoints").glob(stem + "*.json")), stem
    # layer-sliced npys deleted; static row indexes + b0 pools retained
    assert not list(stage.rglob("*_L*.npy"))
    for mt in ("instruct", "pretrained"):
        assert (stage / f"bare_{mt}" / "row_index.jsonl").exists()
        for kind in p6.DYNAMICS_KINDS:
            assert (stage / f"dynamics_{mt}" / f"row_index_{kind}.jsonl").exists()
    for cell in CELLS:
        assert (stage / "b0_rB_pool" / f"{cell}.npy").exists()
    # selection-null matrices persisted under the out dir (never deleted)
    assert list((out / "analysis_tensors" / "nulls").glob("*.npy"))
    assert (out / "p6_run_summary.json").exists()


def test_e2e_resume_skips_pilot_and_layers(e2e):
    args = _args(e2e["stage"], e2e["out"], **e2e["shared"])
    summary = p6.run_p6(args)
    assert summary["pilot"] == "skipped_prior_pass"
    assert summary["layers_done"] == []
    assert summary["layers_skipped"] == [14, 15]
    assert not list(e2e["stage"].rglob("*_L*.npy"))


def test_pilot_only_deletes_layer_slices_keeps_static(shared_fixture, tmp_path):
    args = _args(
        tmp_path / "stage",
        tmp_path / "out",
        **shared_fixture,
        extra=("--pilot-only", "--skip-band-pilot"),
    )
    summary = p6.run_p6(args)
    assert summary["pilot_only"] is True
    assert summary["layers_done"] == [] and summary["layers_skipped"] == []
    assert summary["pilot"]["projected_total_wall_h"] > 0
    pilot = json.loads((tmp_path / "out" / "pilot.json").read_text())
    assert "skipped" in pilot["band_note"]
    assert pilot["wall_s_band_block"] == pilot["wall_s_frozen_block"]
    assert not list((tmp_path / "stage").rglob("*_L*.npy"))
    assert (tmp_path / "stage" / "bare_instruct" / "row_index.jsonl").exists()
    assert not (tmp_path / "out" / "staging" / "staging_manifest_L14.json").exists()


def test_pilot_gate_abort_exits_nonzero(shared_fixture, tmp_path):
    args = _args(
        tmp_path / "stage",
        tmp_path / "out",
        **shared_fixture,
        extra=("--skip-band-pilot", "--max-pilot-rss-gb", "0.001"),
    )
    with pytest.raises(SystemExit) as exc:
        p6.run_p6(args)
    assert exc.value.code == 3
    pilot = json.loads((tmp_path / "out" / "pilot.json").read_text())
    assert pilot["abort_predicate_result"]["abort"] is True
    assert "cpu-bigmem --min-ram-gb 32" in pilot["abort_predicate_result"]["message"]
    # an aborted pilot is never skippable on a re-run
    cfg = json.loads((tmp_path / "out" / "pilot.json").read_text())["wrapper_config"]
    assert (
        p6.pilot_skippable(tmp_path / "out", cfg, args, n_blocks_frozen=2, n_blocks_band=2) is False
    )


def test_fit_grid_nonzero_exit_fails_loud(shared_fixture, tmp_path):
    args = _args(
        tmp_path / "stage",
        tmp_path / "out",
        **shared_fixture,
        extra=("--skip-band-pilot", "--fit-grid-arg=--bogus-flag-that-does-not-exist"),
    )
    with pytest.raises(RuntimeError, match="fit grid exited rc="):
        p6.run_p6(args)


def test_evaluate_pilot_gate_predicates():
    ok = p6.evaluate_pilot_gate(
        ru_maxrss_gb=8.0, projected_wall_h=20.0, rss_limit_gb=14.0, plan_wall_h=27.0
    )
    assert ok["abort"] is False and ok["message"] == "pilot gate PASS"
    rss = p6.evaluate_pilot_gate(
        ru_maxrss_gb=14.0, projected_wall_h=20.0, rss_limit_gb=14.0, plan_wall_h=27.0
    )
    assert rss["abort"] is True and rss["rss_exceeded"] is True
    assert "cpu-bigmem --min-ram-gb 32" in rss["message"]
    wall = p6.evaluate_pilot_gate(
        ru_maxrss_gb=1.0, projected_wall_h=54.1, rss_limit_gb=14.0, plan_wall_h=27.0
    )
    assert wall["abort"] is True and wall["wall_exceeded"] is True
    boundary = p6.evaluate_pilot_gate(
        ru_maxrss_gb=1.0, projected_wall_h=54.0, rss_limit_gb=14.0, plan_wall_h=27.0
    )
    assert boundary["abort"] is False  # strictly greater than 2x aborts


def test_staging_selectors_fail_loud():
    inv = {
        ("cell_inst_own", "t1_L05.npy"): p6.HubFile(f"{PREFIX}/cell_inst_own/t1_L05.npy", 1, ""),
    }
    with pytest.raises(FileNotFoundError, match="prefix_end"):
        p6.select_layer_files(inv, ["cell_inst_own"], ["instruct"], ["prefix_end", "t1"], 5)
    inv2 = dict(inv)
    for kind in ("prefix_end", "context_end"):
        inv2[("cell_inst_own", f"{kind}_L05.npy")] = p6.HubFile(
            f"{PREFIX}/cell_inst_own/{kind}_L05.npy", 1, ""
        )
    inv2[("bare_instruct", "c_q_bare_L05.npy")] = p6.HubFile(
        f"{PREFIX}/bare_instruct/c_q_bare_L05.npy", 1, ""
    )
    with pytest.raises(FileNotFoundError, match="dynamics_instruct"):
        p6.select_layer_files(
            inv2, ["cell_inst_own"], ["instruct"], ["prefix_end", "context_end", "t1"], 5
        )
    inv2[("bare_instruct", "row_index.jsonl")] = p6.HubFile(
        f"{PREFIX}/bare_instruct/row_index.jsonl", 1, ""
    )
    for kind in p6.DYNAMICS_KINDS:
        inv2[("dynamics_instruct", f"row_index_{kind}.jsonl")] = p6.HubFile(
            f"{PREFIX}/dynamics_instruct/row_index_{kind}.jsonl", 1, ""
        )
    with pytest.raises(FileNotFoundError, match="b0_rB_pool"):
        p6.select_static_files(inv2, ["cell_inst_own"], ["instruct"])


def test_deterministic_staged_mtime_reproduces_fit_grid_fingerprint(tmp_path):
    hub_root = tmp_path / "hub"
    (hub_root / "pfx" / "cell").mkdir(parents=True)
    np.save(hub_root / "pfx" / "cell" / "t1_L00.npy", np.zeros((2, 2), dtype=np.float32))
    hub = p6.LocalFixtureHubIO(hub_root)
    entry = hub.list_files("pfx")[0]
    stage = tmp_path / "stage"
    rec1 = p6.stage_file(hub, entry, "pfx", stage)
    staged = Path(rec1["staged_to"])
    mtime1 = staged.stat().st_mtime_ns
    fp1 = fit_grid._fingerprint([staged], {"probe": 1})
    staged.unlink()
    rec2 = p6.stage_file(hub, entry, "pfx", stage)
    assert rec2["reused"] is False
    assert Path(rec2["staged_to"]).stat().st_mtime_ns == mtime1
    assert fit_grid._fingerprint([staged], {"probe": 1}) == fp1
    # third staging with the file present takes the verified-reuse path
    rec3 = p6.stage_file(hub, entry, "pfx", stage)
    assert rec3["reused"] is True
    assert fit_grid._fingerprint([staged], {"probe": 1}) == fp1


def test_hubio_seam_signature_parity():
    for method in ("resolved_revision", "list_files", "download_to"):
        real = inspect.signature(getattr(p6.HfHubIO, method))
        fake = inspect.signature(getattr(p6.LocalFixtureHubIO, method))
        assert list(real.parameters) == list(fake.parameters), method


def test_fit_grid_argv_judge_toggle_and_owned_flag_guard(tmp_path):
    args = p6.parse_args(["--corpus-dir", str(tmp_path)])
    argv = p6.fit_grid_argv(args, ["cell_inst_own"], "3", tmp_path / "out", 7)
    assert "--allow-missing-registered-reads" in argv
    assert "--judge-scores" not in argv
    assert argv[argv.index("--n-null-draws") + 1] == "7"
    judge = tmp_path / "j.jsonl"
    judge.write_text("{}\n")
    args2 = p6.parse_args(["--corpus-dir", str(tmp_path), "--judge-scores", str(judge)])
    argv2 = p6.fit_grid_argv(args2, ["cell_inst_own"], "3", tmp_path / "out", 7)
    assert "--judge-scores" in argv2
    assert "--allow-missing-registered-reads" not in argv2
    args3 = p6.parse_args(["--corpus-dir", str(tmp_path), "--fit-grid-arg=--layers=5"])
    with pytest.raises(ValueError, match="wrapper-owned"):
        p6.fit_grid_argv(args3, ["cell_inst_own"], "3", tmp_path / "out", 7)


def test_pilot_layer_must_be_frozen(shared_fixture, tmp_path):
    args = _args(
        tmp_path / "stage",
        tmp_path / "out",
        **shared_fixture,
        extra=("--pilot-layer", "15"),
    )
    with pytest.raises(ValueError, match="frozen null layer"):
        p6.run_p6(args)


def _tiny_engine_setup(tmp_path: Path) -> tuple[list[str], Path, Path]:
    """Shared tiny-real direct-engine fixture: returns (base_argv, out_dir, summaries)."""
    summaries = _write_store(tmp_path / "summaries", cells=("cell_inst_own",), layers=(14,))
    corpus = _write_corpus(tmp_path / "corpus")
    rb = _write_rb(tmp_path / "rb")
    out = tmp_path / "out"
    base_argv = [
        "issue1092_fit_grid.py",
        "--summaries-dir",
        str(summaries),
        "--corpus-dir",
        str(corpus),
        "--out-dir",
        str(out),
        "--cells",
        "cell_inst_own",
        "--layers",
        "14",
        "--targets",
        "t1",
        "--target-bases",
        "ambient",
        "--n-null-draws",
        "2",
        "--band-null-draws",
        "2",
        "--matched-n-draws",
        "2",
        "--n-folds",
        "2",
        "--hidden-dim",
        "8",
        "--skip-mlp-companion",
        "--rb-dir",
        str(rb),
        "--allow-missing-registered-reads",
    ]
    return base_argv, out, summaries


def test_fit_grid_checkpoint_fingerprint_includes_judge_identity(tmp_path, monkeypatch):
    """Regression pin (fails pre-fix): judge scores are output-affecting (behavior
    joins live inside the unit checkpoints), so a --judge-scores re-run must NOT
    fingerprint-match judge-less checkpoints and silently skip the joins."""
    base_argv, out, _summaries = _tiny_engine_setup(tmp_path)
    monkeypatch.setattr(sys, "argv", list(base_argv))
    fit_grid.run(fit_grid.parse_args())
    before = {p.name for p in (out / "checkpoints").glob("*.json")}
    assert before  # 2 arms x 2 fit arms x 1 basis
    judge = tmp_path / "judge.jsonl"
    judge.write_text(
        json.dumps({"cell_id": "no_such_cell", "row_id": "zz", "score": 50, "trait": "evil"}) + "\n"
    )
    monkeypatch.setattr(sys, "argv", [*base_argv, "--judge-scores", str(judge)])
    fit_grid.run(fit_grid.parse_args())
    after = {p.name for p in (out / "checkpoints").glob("*.json")}
    new = after - before
    assert len(new) == len(before), (
        "judge-bearing re-run must recompute every unit under a judge-keyed fingerprint"
    )


def test_fit_grid_checkpoint_fingerprint_includes_nonxy_inputs(tmp_path, monkeypatch):
    """Regression pin (fails pre-fix): dynamics/bare/b0 inputs are output-affecting
    (D0-D5, stitch, B0 reads live inside the unit checkpoints); the wrapper's
    content-derived staging mtimes deliberately preserve fingerprints across
    staging cycles, so a Hub re-upload touching ONLY a dynamics shard must re-key
    every unit checkpoint rather than wrong-skip stale ones (code-review v10
    concern p6-unit-fp-omits-nonxy-input-content)."""
    base_argv, out, summaries = _tiny_engine_setup(tmp_path)
    monkeypatch.setattr(sys, "argv", list(base_argv))
    fit_grid.run(fit_grid.parse_args())
    before = {p.name for p in (out / "checkpoints").glob("*.json")}
    assert before
    # unchanged inputs re-run: every unit resumes, zero new checkpoints
    fit_grid.run(fit_grid.parse_args())
    assert {p.name for p in (out / "checkpoints").glob("*.json")} == before
    # mutate ONE dynamics shard's content (same shape, same name)
    dyn = summaries / "dynamics_instruct" / "context_k_L14.npy"
    np.save(dyn, np.load(dyn) + 1.0)
    fit_grid.run(fit_grid.parse_args())
    after = {p.name for p in (out / "checkpoints").glob("*.json")}
    assert len(after - before) == len(before), (
        "dynamics-only content change must re-key every unit checkpoint"
    )


def test_fit_grid_argv_carries_plan_registered_fit_config(tmp_path):
    """Review v10 concern p6-launch-defaults-vs-plan-folds-targets-seed: the default
    wrapper launch must compose the plan-registered fit config into EVERY fit-grid
    invocation (plan section 6: grouped 6-fold by prefix; section 10: fit seed 0;
    section 4.5: t1/t2/t3 sensitivity targets) — and pass-through may not clobber
    the registered flags."""
    args = p6.parse_args(["--corpus-dir", str(tmp_path)])
    argv = p6.fit_grid_argv(args, ["cell_inst_own"], "14", tmp_path / "out", 7)
    assert argv[argv.index("--n-folds") + 1] == "6"
    assert argv[argv.index("--seed") + 1] == "0"
    assert argv[argv.index("--targets") + 1] == "t1,t2,t3"
    for flag in ("--n-folds", "--seed"):
        bad = p6.parse_args(["--corpus-dir", str(tmp_path), f"--fit-grid-arg={flag}=3"])
        with pytest.raises(ValueError, match="wrapper-owned"):
            p6.fit_grid_argv(bad, ["cell_inst_own"], "14", tmp_path / "out", 7)


def test_layer_already_complete_keys_static_files(tmp_path):
    """A statics-only Hub re-upload (b0 pool / row index) must fall through the
    wrapper's layer-complete fast-path to the engine's fingerprint predicate."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "c.json").write_text("{}")
    layer_file = p6.HubFile("pfx/cell_inst_own/t1_L14.npy", 3, "a" * 64)
    static_file = p6.HubFile("pfx/b0_rB_pool/cell_inst_own.npy", 5, "b" * 64)
    manifest = {
        "status": "complete",
        "wrapper_config": {"config_sha256": "cfg"},
        "files": [{"hub_path": layer_file.path, "size": 3, "hub_identity": "a" * 64}],
        "static_files": [{"hub_path": static_file.path, "size": 5, "hub_identity": "b" * 64}],
        "checkpoints": ["c.json"],
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest))
    cfg = {"config_sha256": "cfg"}
    assert p6.layer_already_complete(mp, cfg, [layer_file, static_file], ckpt_dir) is True
    changed_static = p6.HubFile(static_file.path, 5, "c" * 64)  # b0-only Hub re-upload
    assert p6.layer_already_complete(mp, cfg, [layer_file, changed_static], ckpt_dir) is False


def test_hub_retry_bounded(monkeypatch):
    """Transient 429/5xx retries with bounded backoff; non-transient raises at once."""
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    sleeps: list[float] = []
    monkeypatch.setattr(p6.time, "sleep", lambda s: sleeps.append(s))
    resp = requests.Response()
    resp.status_code = 503
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise HfHubHTTPError("boom", response=resp)
        return "ok"

    assert p6._hub_retry(flaky, what="probe") == "ok"
    assert calls["n"] == 3 and sleeps == [2.0, 8.0]

    resp404 = requests.Response()
    resp404.status_code = 404

    def notfound():
        raise HfHubHTTPError("nope", response=resp404)

    with pytest.raises(HfHubHTTPError):
        p6._hub_retry(notfound, what="probe")
    assert sleeps == [2.0, 8.0]  # non-transient never slept

    def always_503():
        raise HfHubHTTPError("busy", response=resp)

    with pytest.raises(HfHubHTTPError):
        p6._hub_retry(always_503, what="probe")
    assert sleeps == [2.0, 8.0, 2.0, 8.0, 30.0]  # bounded: 4 attempts then raise
