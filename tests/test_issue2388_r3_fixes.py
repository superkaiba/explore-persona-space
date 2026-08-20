"""#2388 round-3 revision fixes — pinned regression tests.

Covers the r2 code-review blockers: the STAGED fork-5 APPS chain (control ->
pilot gen -> pilot verdict -> full gen -> binding full-pool G3 -> fit
activation), pilot verify slicing to its OWN report file, bare-default rosters
excluding the contingency benchmark, the --smoke local out-root rebinding
(fits + maps + h3 sibling), sandbox network-isolation fail-loud, genmeta
base-cap immutability, the capture fingerprint's rollout-content digest, the
sweep regime's upstream digests, per-(design, fold) selector-fit telemetry,
and the bootstrap's bitwise-parity vectorization + arm-set resume invalidation.

Adoptable: repo-root-relative paths, tmp_path outputs, no network / GPU.
CONTENT HYGIENE: all fixtures are benign synthetic text.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_script("issue2388_gen_r3", "scripts/issue2388_gen.py")


@pytest.fixture(scope="module")
def drv():
    return _load_script("issue2388_fits_r3", "scripts/issue2388_fits.py")


@pytest.fixture(scope="module")
def cap():
    return _load_script("issue2388_capture_r3", "scripts/issue2388_capture.py")


# ---------------------------------------------------------------------------
# gen: staged fork-5 APPS gate chain
# ---------------------------------------------------------------------------


def _gate_fixture(
    tmp_path: Path,
    gen,
    *,
    control_ok: bool = True,
    pilot: dict | None = None,
    spread: dict | None = None,
) -> Path:
    """Below-floor pool (BCB G1 fail + 0 LCB kept) with a full apps control."""
    out_root = tmp_path / "gen"
    (out_root / "code").mkdir(parents=True, exist_ok=True)
    (out_root / "code" / "dedup_report.json").write_text(
        json.dumps({"n_lcb": 880, "n_dropped_lcb": 880})
    )
    benches = {
        "bigcodebench": {"harness_ok": False, "flaky_mismatch_fraction": 0.0},
        "apps_intro": {
            "harness_ok": control_ok,
            "n_control": 25,
            "best_pass_rate": 1.0 if control_ok else 24 / 25,
            "runs_per_item": 2,
            "flaky_mismatch_fraction": 0.0,
        },
    }
    (out_root / gen.CONTROL_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (out_root / gen.CONTROL_REPORT).write_text(json.dumps({"benchmarks": benches}))
    if pilot is not None:
        (out_root / "code" / gen.APPS_PILOT_REPORT).write_text(json.dumps(pilot))
    if spread is not None:
        (out_root / "code" / "apps_intro.json").write_text(json.dumps(spread))
    return out_root


def _pilot_payload(gen, **over) -> dict:
    return {"pilot": True, "smoke": False, "n_items": gen.APPS_PILOT_N, "admissible": True, **over}


def _spread_payload(**over) -> dict:
    return {"pilot": False, "smoke": False, "n_items": 1000, "admissible": True, **over}


def test_gate_staged_chain_full_activation(gen, tmp_path):
    out_root = _gate_fixture(tmp_path, gen, pilot=_pilot_payload(gen), spread=_spread_payload())
    v = gen.phase_gate(out_root)
    assert v["apps_required"] is True
    assert v["g1_apps"]["control_25_25"] is True
    assert v["apps_pilot_gen_allowed"] is True
    assert v["apps_pilot"]["admissible"] is True
    assert v["apps_full_gen_allowed"] is True
    assert v["g3_apps"]["full_pool"] is True
    assert v["apps_activated"] is True
    gen._require_gate_for("apps_intro", out_root)  # full gen allowed
    gen._require_gate_for("apps_intro", out_root, apps_pilot=True)


def test_gate_control_fail_refuses_pilot(gen, tmp_path):
    out_root = _gate_fixture(tmp_path, gen, control_ok=False)
    v = gen.phase_gate(out_root)
    assert v["apps_required"] is True
    assert v["g1_apps"]["control_25_25"] is False
    assert v["apps_pilot_gen_allowed"] is False
    assert v["apps_activated"] is False
    with pytest.raises(RuntimeError, match="control"):
        gen._require_gate_for("apps_intro", out_root, apps_pilot=True)


def test_gate_pilot_scope_rejected(gen, tmp_path):
    """A wrong-sized or smoke pilot file can never resolve the pilot verdict."""
    out_root = _gate_fixture(tmp_path, gen, pilot=_pilot_payload(gen, n_items=50))
    v = gen.phase_gate(out_root)
    assert v["apps_pilot"]["pilot_scope_ok"] is False
    assert v["apps_pilot"]["admissible"] is None
    assert v["apps_full_gen_allowed"] is False
    out_root2 = _gate_fixture(tmp_path / "b", gen, pilot=_pilot_payload(gen, smoke=True))
    v2 = gen.phase_gate(out_root2)
    assert v2["apps_pilot"]["admissible"] is None


def test_gate_g3_apps_rejects_pilot_shaped_file(gen, tmp_path):
    """The binding G3 read refuses a pilot-sized / pilot-flagged / smoke file
    at code/apps_intro.json — fit activation needs the FULL pool."""
    for bad in (
        _spread_payload(n_items=gen.APPS_PILOT_N),
        _spread_payload(pilot=True),
        _spread_payload(smoke=True),
    ):
        root = tmp_path / f"g{bad['n_items']}{bad['pilot']}{bad.get('smoke')}"
        out_root = _gate_fixture(root, gen, pilot=_pilot_payload(gen), spread=bad)
        v = gen.phase_gate(out_root)
        assert v["g3_apps"]["full_pool"] is False
        assert v["g3_apps"]["admissible"] is None
        assert v["apps_full_gen_allowed"] is True  # pilot chain intact
        assert v["apps_activated"] is False  # but fit inclusion refused


# ---------------------------------------------------------------------------
# gen: pilot verify slicing + full-pool-by-construction
# ---------------------------------------------------------------------------


def _fake_apps_items(n: int) -> list[dict]:
    return [{"item_id": f"apps-{i}", "benchmark": "apps_intro"} for i in range(n)]


def _write_rollouts(gen, out_root: Path, items: list[dict]) -> Path:
    roll_path = gen._rollouts_path(out_root, "apps_intro")
    roll_path.parent.mkdir(parents=True, exist_ok=True)
    with roll_path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(
                json.dumps(
                    {
                        "item_id": it["item_id"],
                        "completions": ["print(1)"] * gen.K_ROLLOUTS,
                        "finish_reasons": ["stop"] * gen.K_ROLLOUTS,
                    }
                )
                + "\n"
            )
    return roll_path


def test_verify_apps_pilot_writes_own_report(gen, tmp_path, monkeypatch):
    items = _fake_apps_items(gen.APPS_PILOT_N + 50)
    monkeypatch.setitem(gen.LOADERS, "apps_intro", lambda: items)
    monkeypatch.setattr(gen, "_unshare_net_available", lambda: True)
    monkeypatch.setattr(gen, "_verdict_one", lambda t: (t[0]["item_id"], t[1], True))
    out_root = tmp_path / "gen"
    # rollouts exist ONLY for the pilot slice — full verify must refuse below
    _write_rollouts(gen, out_root, items[: gen.APPS_PILOT_N])
    gen.phase_verify(
        "apps_intro", out_root, smoke=False, bcb_python=None, workers=1, apps_pilot=True
    )
    pilot_p = out_root / "code" / gen.APPS_PILOT_REPORT
    assert pilot_p.exists()
    payload = json.loads(pilot_p.read_text())
    assert payload["pilot"] is True
    assert payload["n_items"] == gen.APPS_PILOT_N
    assert payload["sandbox_net_isolation"] is True
    # the binding G3 path is untouched by a pilot verify
    assert not (out_root / "code" / "apps_intro.json").exists()


def test_verify_apps_full_pool_requires_all_rollouts(gen, tmp_path, monkeypatch):
    """Non-pilot verify is full-pool BY CONSTRUCTION: any loader item lacking
    rollouts refuses (this is what makes g3_apps's n_items read trustworthy)."""
    items = _fake_apps_items(gen.APPS_PILOT_N + 50)
    monkeypatch.setitem(gen.LOADERS, "apps_intro", lambda: items)
    monkeypatch.setattr(gen, "_unshare_net_available", lambda: True)
    out_root = tmp_path / "gen"
    _write_rollouts(gen, out_root, items[: gen.APPS_PILOT_N])
    with pytest.raises(RuntimeError, match="lack rollouts"):
        gen.phase_verify("apps_intro", out_root, smoke=False, bcb_python=None, workers=1)


# ---------------------------------------------------------------------------
# gen + code_control: bare-default rosters exclude the contingency benchmark
# ---------------------------------------------------------------------------


def test_gen_bare_default_roster_excludes_apps(gen, tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(gen, "phase_verify", lambda bench, out_root, **kw: seen.append(bench))
    rc = gen.main(["--phase", "verify", "--out-root", str(tmp_path / "gen")])
    assert rc == 0
    assert seen == sorted(set(gen.LOADERS) - {"apps_intro"})
    assert "apps_intro" in gen.LOADERS  # still loadable via explicit opt-in
    seen.clear()
    rc = gen.main(
        ["--phase", "verify", "--benchmark", "apps_intro", "--out-root", str(tmp_path / "gen")]
    )
    assert rc == 0
    assert seen == ["apps_intro"]


def test_code_control_default_excludes_apps():
    src = (REPO_ROOT / "scripts/issue2388_code_control.py").read_text()
    assert 'default=sorted(set(BENCHES) - {"apps_intro"})' in src
    assert '"apps_intro":' in src  # the control CAN run it explicitly


# ---------------------------------------------------------------------------
# gen: sandbox network isolation is fail-loud; genmeta base cap immutable
# ---------------------------------------------------------------------------


def test_sandbox_net_isolation_fail_loud(gen, monkeypatch):
    monkeypatch.setattr(gen, "_unshare_net_available", lambda: False)
    monkeypatch.delenv(gen._SANDBOX_ALLOW_NET_ENV, raising=False)
    with pytest.raises(RuntimeError, match="network"):
        gen._require_sandbox_net_isolation()
    monkeypatch.setenv(gen._SANDBOX_ALLOW_NET_ENV, "1")
    gen._require_sandbox_net_isolation()  # explicit recorded override proceeds


def test_verify_refuses_before_workers_when_sandbox_missing(gen, tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "_unshare_net_available", lambda: False)
    monkeypatch.delenv(gen._SANDBOX_ALLOW_NET_ENV, raising=False)
    with pytest.raises(RuntimeError, match="network"):
        gen.phase_verify("apps_intro", tmp_path, smoke=False, bcb_python=None, workers=4)


def test_genmeta_base_cap_drift_raises(gen, tmp_path):
    roll_path = tmp_path / "rollouts" / "humaneval_rollouts.jsonl"
    roll_path.parent.mkdir(parents=True)
    gen._check_genmeta(roll_path, "humaneval")  # first call writes the sidecar
    gen._check_genmeta(roll_path, "humaneval")  # identical params resume fine
    meta_path = roll_path.with_name("humaneval_genmeta.json")
    meta = json.loads(meta_path.read_text())
    meta["base_max_tokens"] = int(meta["base_max_tokens"]) * 2
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="drifted"):
        gen._check_genmeta(roll_path, "humaneval")


# ---------------------------------------------------------------------------
# capture: resume fingerprint tracks rollout CONTENT
# ---------------------------------------------------------------------------


def test_capture_fingerprint_tracks_rollout_content(cap, tmp_path):
    roll = tmp_path / "humaneval_rollouts.jsonl"
    roll.write_text('{"item_id": "a", "completions": ["x"]}\n')
    fp1 = cap._fingerprint("humaneval", roll)
    roll.write_text('{"item_id": "a", "completions": ["REGENERATED"]}\n')
    fp2 = cap._fingerprint("humaneval", roll)
    assert fp1 != fp2
    assert "|rolls=" in fp1 and "|rolls=" in fp2
    # absent path: parameter-only fingerprint (stable)
    assert cap._fingerprint("humaneval") == cap._fingerprint("humaneval")
    assert "|rolls=" not in cap._fingerprint("humaneval")


# ---------------------------------------------------------------------------
# fits: --smoke local out-root rebinding + upstream regime digests
# ---------------------------------------------------------------------------


def test_fits_smoke_rebinds_local_roots(drv, monkeypatch):
    captured: dict[str, argparse.Namespace] = {}
    phase = sorted(drv.PHASES)[0]
    monkeypatch.setitem(drv.PHASES, phase, lambda args: captured.setdefault("args", args))
    rc = drv.main(["--phase", phase, "--smoke"])
    assert rc == 0
    args = captured["args"]
    assert args.fits_root == str(drv.FITS_ROOT) + "_smoke"
    assert args.maps_out == str(drv.MAPS_OUT) + "_smoke"
    assert args.n_null <= 5 and args.n_boot <= 50
    assert drv._h3_root(args).name == "h3_recompute_smoke"
    # an EXPLICIT non-default root is respected (never silently rebound)
    captured.clear()
    drv.main(["--phase", phase, "--smoke", "--fits-root", "/tmp/i2388-custom"])
    assert captured["args"].fits_root == "/tmp/i2388-custom"
    assert captured["args"].maps_out == str(drv.MAPS_OUT) + "_smoke"
    # production mode: canonical roots untouched
    captured.clear()
    drv.main(["--phase", phase])
    assert captured["args"].fits_root == str(drv.FITS_ROOT)
    assert drv._h3_root(captured["args"]).name == "h3_recompute"


def test_file_sha_and_regime_upstream_digests(drv, tmp_path):
    p = tmp_path / "labeling.json"
    assert drv._file_sha(p) is None  # absence is part of the pinned regime
    p.write_text('{"rows": []}')
    s1 = drv._file_sha(p)
    p.write_text('{"rows": [{"context_id": "x"}]}')
    s2 = drv._file_sha(p)
    assert s1 and s2 and s1 != s2 and len(s1) == 12
    # the sweep regime sentinel pins BOTH upstream digests (r2
    # long-loop-restartability) — source pin on the regime dict
    src = (REPO_ROOT / "scripts/issue2388_fits.py").read_text()
    assert '"labeling_sha": _file_sha(' in src
    assert '"map_manifest_sha": _file_sha(' in src


# ---------------------------------------------------------------------------
# fits: bootstrap — bitwise serial parity + arm-set resume invalidation
# ---------------------------------------------------------------------------


def test_boot_spearman_draws_matches_serial(drv):
    from explore_persona_space.experiments.issue_1739.arms import spearman_rows

    rng = np.random.default_rng(0)
    for groups in (
        np.array([f"g{i % 37}" for i in range(211)]),  # grouped, variable sizes
        np.array([f"g{i}" for i in range(150)]),  # singleton groups
    ):
        n = len(groups)
        preds = rng.normal(size=(5, n))
        y = np.round(rng.normal(size=n), 1)  # ties in y exercise rank ties
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        draw_rng = np.random.default_rng(7)
        idx_draws = []
        for _ in range(60):
            gs = draw_rng.choice(uniq, size=len(uniq), replace=True)
            idx_draws.append(np.concatenate([members[g] for g in gs]))
        boot = drv._boot_spearman_draws(preds, y, idx_draws)
        ref = np.stack([spearman_rows(preds[:, ix], y[ix]) for ix in idx_draws])
        assert np.array_equal(boot, ref)  # BITWISE — same arithmetic per row


def _boot_env(drv, tmp_path, *, arms=("arm_ctx", "arm_maplin")):
    surface = "math"
    out_root = tmp_path / "fits" / surface
    (out_root / "preds").mkdir(parents=True, exist_ok=True)
    ids = [f"mathfull-x-{i}" for i in range(9)]
    rng = np.random.default_rng(1)
    for arm in arms:
        with (out_root / "preds" / f"preds_{arm}_L16_draw0.jsonl").open("w") as fh:
            for i, cid in enumerate(ids):
                fh.write(
                    json.dumps(
                        {
                            "eval": "dev",
                            "context_id": cid,
                            "y_true": i / len(ids),
                            "y_pred": float(rng.normal()),
                        }
                    )
                    + "\n"
                )
    dv_dir = tmp_path / "dv" / surface
    dv_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "context_id": cid,
            "dv": i / len(ids),
            "split": "train",
            "group_key": f"g{i % 3}",
            "subject": "algebra",
            "level": 1,
        }
        for i, cid in enumerate(ids)
    ]
    (dv_dir / "labeling.json").write_text(json.dumps({"rows": rows}))
    return argparse.Namespace(
        surface=surface,
        fits_root=str(tmp_path / "fits"),
        dv_root=str(tmp_path / "dv"),
        n_boot=10,
        force=False,
    )


def test_bootstrap_arm_set_change_recomputes(drv, tmp_path, capsys):
    """r2 Codex Minor 4: a resume whose unit ARM SET changed must recompute the
    unit (the n_boot-only key silently kept stale-arm rows)."""
    args = _boot_env(drv, tmp_path)
    drv.phase_bootstrap(args)
    summary_p = tmp_path / "fits" / "math" / "bootstrap_summary.json"
    s1 = json.loads(summary_p.read_text())
    assert s1["cells"][0]["arms"] == ["arm_ctx", "arm_maplin"]
    # arm roster changes: one arm's preds removed -> the unit RECOMPUTES
    (tmp_path / "fits" / "math" / "preds" / "preds_arm_maplin_L16_draw0.jsonl").unlink()
    capsys.readouterr()
    drv.phase_bootstrap(args)
    assert "RECOMPUTE" in capsys.readouterr().out
    s2 = json.loads(summary_p.read_text())
    assert len(s2["cells"]) == 1
    assert s2["cells"][0]["arms"] == ["arm_ctx"]
    # unchanged roster resumes without recompute
    capsys.readouterr()
    drv.phase_bootstrap(args)
    assert "RECOMPUTE" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# arms: per-(design, fold) selector-fit telemetry (h3-selector-telemetry)
# ---------------------------------------------------------------------------


def test_run_ridge_job_selector_telemetry():
    from explore_persona_space.experiments.issue_1739.arms import RidgeJob, _run_ridge_job

    rng = np.random.default_rng(0)
    n_s, n, d = 2, 30, 5
    src = rng.normal(size=(n_s, n, d))
    y_full = rng.normal(size=n)
    job = RidgeJob(
        key=("z", 0),
        src=src,
        tr_rows=np.arange(20),
        targets=[("t0", y_full)],
        evals=[("dev", src, np.arange(20, 30))],
    )
    telem: list[dict] = []
    key, preds = _run_ridge_job(
        job, lambdas=(0.1, 1.0, 10.0), device="cpu", dof_cap=0.9, telemetry=telem
    )
    assert key == ("z", 0)
    assert len(telem) == 1
    row = telem[0]
    assert row["design"] == "z" and row["fold"] == 0
    assert row["mode"] in ("gcv", "gcv-dof-capped")
    assert row["dof_cap"] == pytest.approx(0.9)
    assert row["n_train"] == 20
    assert row["n_fits"] >= n_s  # one selected lambda per (slice, target)
    assert sum(row["lambda_hist"].values()) == row["n_fits"]
    dq = row["dof_selected"]
    assert dq is not None and dq["min"] <= dq["median"] <= dq["max"]
    # telemetry is an OUT-PARAM only: preds identical with it disabled
    _key2, preds2 = _run_ridge_job(job, lambdas=(0.1, 1.0, 10.0), device="cpu", dof_cap=0.9)
    for name in preds:
        assert np.array_equal(preds[name], preds2[name])


# ---------------------------------------------------------------------------
# dispatcher dry-runs: PRODUCTION argv through the REAL parsers
# (r2 smoke-production-evidence-missing — CPU-feasible legs; the GPU-bound
# gen/capture phase BODIES stay pod-side, but their dispatch surface — parser,
# roster, root resolution, refusals — is exercised here with production argv)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dvb():
    return _load_script("issue2388_dv_build_r3", "scripts/issue2388_dv_build.py")


def test_gen_production_argv_dry_run(gen, monkeypatch):
    """The pod-side production invocation (--phase all, bare roster) resolves
    through the real parser: dedup -> gate -> per-bench gen/verify/upload in
    registry order, all against the PRODUCTION out-root."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        gen, "dedup_lcb_against_leetcode", lambda out_root: calls.append(("dedup", out_root))
    )
    monkeypatch.setattr(gen, "phase_gate", lambda out_root, **kw: calls.append(("gate", out_root)))
    monkeypatch.setattr(
        gen, "phase_gen", lambda b, out_root, **kw: calls.append(("gen", b, out_root, kw))
    )
    monkeypatch.setattr(
        gen, "phase_verify", lambda b, out_root, **kw: calls.append(("verify", b, out_root, kw))
    )
    monkeypatch.setattr(
        gen, "phase_upload", lambda b, out_root, **kw: calls.append(("upload", b, out_root, kw))
    )
    rc = gen.main(["--phase", "all"])
    assert rc == 0
    assert calls[0][0] == "dedup" and calls[0][1] == gen.OUT_ROOT
    assert ("gate", gen.OUT_ROOT) in calls
    gen_benches = [c[1] for c in calls if c[0] == "gen"]
    assert gen_benches == sorted(set(gen.LOADERS) - {"apps_intro"})
    for c in calls:
        if c[0] in ("gen", "verify", "upload"):
            assert c[2] == gen.OUT_ROOT  # production root, never *_smoke
            assert c[3].get("smoke") is False


def test_capture_production_argv_dry_run(cap, monkeypatch):
    captured: dict = {}
    monkeypatch.setitem(cap.PHASES, "capture", lambda args: captured.setdefault("args", args))
    rc = cap.main(["--phase", "capture", "--benchmark", "humaneval", "--device", "cuda"])
    assert rc == 0
    args = captured["args"]
    assert args.store_root == cap.DEFAULT_STORE_ROOT  # production roots
    assert args.out_root == str(cap.G.OUT_ROOT)
    assert args.dv_root == cap.DEFAULT_DV_ROOT
    assert args.device == "cuda" and args.benchmark == "humaneval"
    # dispatch refusals fire at parse time, before any phase body
    with pytest.raises(SystemExit):
        cap.main(["--phase", "capture"])  # --benchmark required
    with pytest.raises(SystemExit):
        cap.main(["--phase", "upload"])  # --benchmark or --surface required


def test_fits_production_argv_dry_run(drv, monkeypatch):
    captured: dict = {}
    assert "sweep" in drv.PHASES
    monkeypatch.setitem(drv.PHASES, "sweep", lambda args: captured.setdefault("args", args))
    rc = drv.main(["--phase", "sweep", "--surface", "math", "--device", "cpu"])
    assert rc == 0
    args = captured["args"]
    assert args.fits_root == str(drv.FITS_ROOT) and args.maps_out == str(drv.MAPS_OUT)
    assert args.n_null == drv.N_NULL and args.n_boot == drv.N_BOOT
    assert args.map_cell == "fu1" and not args.qa_disjoint


def test_dv_build_production_argv_dry_run(dvb, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        dvb,
        "build_surface_dv",
        lambda surface, gen_root, out_root, *, allow_below_floor=False: calls.append(
            (surface, gen_root, out_root, allow_below_floor)
        ),
    )
    assert dvb.main(["--surface", "code"]) == 0
    assert calls[-1] == ("code", dvb.GEN_ROOT, dvb.DEFAULT_OUT_ROOT, False)
    assert dvb.main(["--surface", "code", "--allow-below-floor"]) == 0
    assert calls[-1][3] is True
