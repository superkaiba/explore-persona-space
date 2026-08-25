"""#2388 round-4 revision fixes — pinned regression tests.

Covers the r3 upheld blocker family (bcb-apps-gates-unwired: "contingency state
not carried across persisted producers and consumers") + the r3 NIT
dv-floor-regression-test-missing:

1. code_control MERGES benchmark rows instead of clobbering — the REAL fork-5
   sequence ``default control -> gate -> APPS-only control -> gate`` leaves the
   BCB verdict readable and resolves ``apps_pilot_gen_allowed=True``.
2. phase_verify is stage-gated (``_require_gate_for`` at entry) for both APPS
   modes and BCB — refusal fires BEFORE any ``_verdict_one`` worker.
3. fits derives the realized code roster from labeling.json's
   ``gate_decisions`` (shared ``code_roster_from_gate_fields`` rule): the
   DROP->APPS branch loads APPS stores without demanding a BCB store, end to
   end through the REAL ``_get_table`` / ``_attach_questions`` bodies.
4. capture ``phase_upload --surface code`` selects the exact upload set from
   the gate verdict (never file existence): DROP->APPS uploads APPS without a
   BCB store; a KEEP branch ignores a stale APPS manifest.
5. dv_build's APPS-inclusive realized-floor guard exercised through the REAL
   ``build_surface_dv`` body: refusal by default, ``below_floor_disclosed``
   only under ``--allow-below-floor``.

Adoptable: repo-root-relative paths, tmp_path outputs, no network / GPU.
CONTENT HYGIENE: all fixtures are benign synthetic text.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import DEFAULT, create_autospec

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel: str):
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_script("issue2388_gen_r4", "scripts/issue2388_gen.py")


@pytest.fixture(scope="module")
def cc():
    return _load_script("issue2388_code_control_r4", "scripts/issue2388_code_control.py")


@pytest.fixture(scope="module")
def cap():
    return _load_script("issue2388_capture_r4", "scripts/issue2388_capture.py")


@pytest.fixture(scope="module")
def drv():
    return _load_script("issue2388_fits_r4", "scripts/issue2388_fits.py")


@pytest.fixture(scope="module")
def dvb():
    return _load_script("issue2388_dv_build_r4", "scripts/issue2388_dv_build.py")


def _write_gate(out_root: Path, **fields) -> None:
    (out_root / "code").mkdir(parents=True, exist_ok=True)
    (out_root / "code" / "code_gate.json").write_text(json.dumps(fields))


# ---------------------------------------------------------------------------
# 1. code_control: merge-don't-clobber through the REAL fork-5 sequence
# ---------------------------------------------------------------------------


def _fake_benches(n: int = 25) -> dict:
    def mk(prefix: str):
        items = [{"item_id": f"{prefix}-{i}"} for i in range(n)]
        canon = {f"{prefix}-{i}": [("direct", "pass")] for i in range(n)}
        return {"items": lambda items=items: items, "canon": lambda canon=canon: canon}

    return {"bigcodebench": mk("bcb"), "apps_intro": mk("apps")}


def test_apps_control_after_bcb_control_preserves_bcb_gate(cc, gen, tmp_path, monkeypatch):
    """The REAL sequence: default control (BCB fails) -> gate -> APPS-only
    control -> gate. The second control invocation must PRESERVE the BCB row
    (r3 Critical 1: it clobbered the report and deadlocked the pilot)."""
    out_root = tmp_path / "gen"
    (out_root / "code").mkdir(parents=True)
    # post-dedup LCB 507: base pool 164+974+507+2869=4514 -> est train 3160 < 3584
    (out_root / "code" / "dedup_report.json").write_text(
        json.dumps({"n_lcb": 880, "n_dropped_lcb": 373})
    )
    report_p = out_root / gen.CONTROL_REPORT
    monkeypatch.setattr(cc, "BENCHES", _fake_benches())
    monkeypatch.setattr(
        cc, "_verify", lambda bench, fenced, item, bcb_python: bench != "bigcodebench"
    )

    # (1) default control roster (= bigcodebench only under the fake BENCHES)
    rc = cc.main(["--out", str(report_p), "--n-control", "25", "--runs", "2"])
    assert rc == 1  # BCB control fails -> DROP branch
    v1 = gen.phase_gate(out_root)
    assert v1["bcb_fit_allowed"] is False
    assert v1["apps_required"] is True
    assert v1["apps_pilot_gen_allowed"] is False  # APPS control not run yet

    # (2) the gate refusal's prescribed command: --benchmarks apps_intro
    rc = cc.main(["--benchmarks", "apps_intro", "--out", str(report_p), "--runs", "2"])
    assert rc == 0
    merged = json.loads(report_p.read_text())
    # the collision pin: the BCB row SURVIVES the APPS-only invocation
    assert merged["benchmarks"]["bigcodebench"]["harness_ok"] is False
    assert merged["benchmarks"]["apps_intro"]["harness_ok"] is True
    assert len(merged["invocations"]) == 2

    # (3) the second gate advances the chain off the SAME report
    v2 = gen.phase_gate(out_root)
    assert v2["bcb_fit_allowed"] is False  # BCB verdict still readable
    assert v2["g1_apps"]["control_25_25"] is True
    assert v2["apps_pilot_gen_allowed"] is True


def test_code_control_legacy_report_provenance_preserved(cc, tmp_path, monkeypatch):
    """A pre-merge single-shot report's rows + provenance survive as an
    invocations entry when a later invocation merges onto it."""
    report_p = tmp_path / "control.json"
    report_p.write_text(
        json.dumps({"benchmarks": {"bigcodebench": {"harness_ok": True}}, "git_commit": "old"})
    )
    monkeypatch.setattr(cc, "BENCHES", _fake_benches())
    monkeypatch.setattr(cc, "_verify", lambda bench, fenced, item, bcb_python: True)
    rc = cc.main(["--benchmarks", "apps_intro", "--out", str(report_p), "--runs", "2"])
    assert rc == 0
    merged = json.loads(report_p.read_text())
    assert merged["benchmarks"]["bigcodebench"]["harness_ok"] is True
    assert merged["invocations"][0]["git_commit"] == "old"
    assert merged["invocations"][1]["benchmarks"] == ["apps_intro"]


# ---------------------------------------------------------------------------
# 2. gen: phase_verify is stage-gated (bug-class sibling gen.py:1496-1525)
# ---------------------------------------------------------------------------


def _rollouts_fixture(gen, out_root: Path, items: list[dict]) -> None:
    roll_path = gen._rollouts_path(out_root, items[0]["benchmark"])
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


def test_verify_apps_refuses_without_stage_gate(gen, tmp_path, monkeypatch):
    """Complete rollout fixtures + missing/false gate -> refusal BEFORE any
    _verdict_one worker (r3 Major 1: stale rollouts could otherwise write
    fresh-looking gate inputs unauthorized)."""
    items = [{"item_id": f"apps-{i}", "benchmark": "apps_intro"} for i in range(gen.APPS_PILOT_N)]
    monkeypatch.setitem(gen.LOADERS, "apps_intro", lambda: items)
    monkeypatch.setattr(gen, "_unshare_net_available", lambda: True)
    called: list = []
    monkeypatch.setattr(gen, "_verdict_one", lambda t: called.append(t))
    out_root = tmp_path / "gen"
    _rollouts_fixture(gen, out_root, items)
    # (a) no gate file at all
    with pytest.raises(FileNotFoundError, match="gate verdict missing"):
        gen.phase_verify(
            "apps_intro", out_root, smoke=False, bcb_python=None, workers=1, apps_pilot=True
        )
    # (b) pilot mode with a FALSE pilot verdict
    _write_gate(out_root, apps_pilot_gen_allowed=False, apps_full_gen_allowed=False)
    with pytest.raises(RuntimeError, match="control"):
        gen.phase_verify(
            "apps_intro", out_root, smoke=False, bcb_python=None, workers=1, apps_pilot=True
        )
    # (c) full mode with a FALSE full verdict
    with pytest.raises(RuntimeError, match="pilot verdict"):
        gen.phase_verify("apps_intro", out_root, smoke=False, bcb_python=None, workers=1)
    assert called == []  # refusal always precedes worker dispatch


def test_verify_bcb_refuses_without_g1(gen, tmp_path, monkeypatch):
    called: list = []
    monkeypatch.setattr(gen, "_verdict_one", lambda t: called.append(t))
    out_root = tmp_path / "gen"
    _write_gate(out_root, bcb_gen_allowed=False)
    with pytest.raises(RuntimeError, match="G1"):
        gen.phase_verify("bigcodebench_full", out_root, smoke=False, bcb_python=None, workers=1)
    assert called == []


# ---------------------------------------------------------------------------
# 3. shared roster rule + fits DROP->APPS end-to-end (real bodies)
# ---------------------------------------------------------------------------


def test_code_roster_from_gate_fields_branches(gen):
    keep = gen.code_roster_from_gate_fields({"bcb_fit_allowed": True, "apps_activated": False})
    assert keep == ["humaneval", "mbpp_full", "bigcodebench_full", "lcb_v5", "leetcode"]
    drop_apps = gen.code_roster_from_gate_fields({"bcb_fit_allowed": False, "apps_activated": True})
    assert drop_apps == ["humaneval", "mbpp_full", "lcb_v5", "leetcode", "apps_intro"]
    drop = gen.code_roster_from_gate_fields({"bcb_fit_allowed": False})
    assert drop == ["humaneval", "mbpp_full", "lcb_v5", "leetcode"]
    with pytest.raises(RuntimeError, match="unresolved"):
        gen.code_roster_from_gate_fields({"apps_activated": True})


DROP_APPS_ROSTER = ("humaneval", "mbpp_full", "lcb_v5", "leetcode", "apps_intro")


def _write_store(root: Path, ctx_ids, k_roll, d, layers, rng):
    root.mkdir(parents=True, exist_ok=True)
    n_rows = len(ctx_ids) * k_roll
    for kind in ("context_end", "t1", "t_last"):
        for ly in range(layers):
            arr = rng.normal(size=(n_rows, d)).astype(np.float16)
            np.save(root / f"{kind}_L{ly:02d}.npy", arr)
    with (root / "row_index.jsonl").open("w") as fh:
        for cid in ctx_ids:
            for k in range(k_roll):
                fh.write(json.dumps({"context_id": cid, "rollout_k": k}) + "\n")


def _drop_apps_env(base: Path, *, benches=DROP_APPS_ROSTER, gate_decisions=None):
    """Synthetic DROP->APPS code surface: labeling + stores, NO BCB store."""
    rng = np.random.default_rng(0)
    d, layers, k_roll = 16, 2, 2
    rows = []
    dvs = [0.0, 0.5, 1.0, 0.5]
    splits = ["train", "train", "dev", "test"]
    for bench in benches:
        ctx_ids = [f"{bench}-c{i}" for i in range(4)]
        _write_store(base / "store" / bench, ctx_ids, k_roll, d, layers, rng)
        for i, cid in enumerate(ctx_ids):
            rows.append(
                {
                    "context_id": cid,
                    "benchmark": bench,
                    "dv": dvs[i],
                    "fractions": {"correct": dvs[i]},
                    "per_rollout_scores": {"k0": dvs[i], "k1": dvs[i]},
                    "group_key": cid,
                    "split": splits[i],
                    "level": None,
                    "subject": None,
                    "category": None,
                    "rung": bench,
                }
            )
    if gate_decisions is None:
        gate_decisions = {"bcb_fit_allowed": False, "apps_activated": True}
    dv_dir = base / "dv" / "code"
    dv_dir.mkdir(parents=True, exist_ok=True)
    (dv_dir / "labeling.json").write_text(
        json.dumps({"rows": rows, "gate_decisions": gate_decisions})
    )
    return argparse.Namespace(
        dv_root=str(base / "dv"),
        store_root=str(base / "store"),
        qa_store_dir=str(base / "qa_store"),
        layers=layers,
        hidden_dim=d,
    )


def test_fits_code_table_drop_apps_branch(drv, tmp_path, monkeypatch):
    """REAL _get_table + _attach_questions on the DROP->APPS branch: APPS rows
    reach the surface table + boot groups, NO BCB store is demanded, and
    question attach covers the APPS loader (r3 Critical 2)."""
    monkeypatch.setattr(drv, "_TABLE_CACHE", {})
    monkeypatch.setattr(drv, "_ROSTER_CACHE", {})
    args = _drop_apps_env(tmp_path)
    assert drv._surface_benchmarks(args, "code") == list(DROP_APPS_ROSTER)
    table = drv._get_table(args, "code")
    benches = set(table.benchmark.tolist())
    assert "apps_intro" in benches and "bigcodebench_full" not in benches
    assert any(g.startswith("apps_intro|") for g in table.boot_group.tolist())

    import issue2388_gen  # the module _attach_questions imports

    for bench in DROP_APPS_ROSTER:
        fake = [{"item_id": f"{bench}-c{i}", "prompt": f"question {bench} {i}"} for i in range(4)]
        monkeypatch.setitem(issue2388_gen.LOADERS, bench, lambda fake=fake: fake)
    drv._attach_questions(args, table)
    q_by_ctx = dict(zip(table.ctx_ids, table.meta["questions"], strict=True))
    assert q_by_ctx["apps_intro-c0"] == "question apps_intro 0"
    assert all(q for q in table.meta["questions"])


def test_fits_code_roster_mismatch_raises(drv, tmp_path, monkeypatch):
    """Gate says APPS activated but the labeling rows carry no APPS rows —
    exact-set cross-validation refuses instead of loading wrong stores."""
    monkeypatch.setattr(drv, "_ROSTER_CACHE", {})
    args = _drop_apps_env(
        tmp_path,
        benches=("humaneval", "mbpp_full", "lcb_v5", "leetcode"),
        gate_decisions={"bcb_fit_allowed": False, "apps_activated": True},
    )
    with pytest.raises(RuntimeError, match="disagree"):
        drv._surface_benchmarks(args, "code")


def test_fits_code_roster_requires_gate_decisions(drv, tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "_ROSTER_CACHE", {})
    args = _drop_apps_env(tmp_path, gate_decisions={})
    with pytest.raises(RuntimeError, match="gate_decisions"):
        drv._surface_benchmarks(args, "code")


# ---------------------------------------------------------------------------
# 4. capture: upload roster from the gate verdict (real phase_upload body)
# ---------------------------------------------------------------------------


def _upload_env(cap, tmp_path, *, gate: dict, store_benches: list[str]):
    gen_root = tmp_path / "gen"
    _write_gate(gen_root, **gate)
    store_root = tmp_path / "store"
    for bench in store_benches:
        d = store_root / bench
        d.mkdir(parents=True)
        (d / "_capture_manifest.json").write_text(json.dumps({"benchmark": bench}))
    return argparse.Namespace(
        benchmark=None,
        surface="code",
        out_root=str(gen_root),
        dv_root=str(tmp_path / "dv"),
        store_root=str(store_root),
        smoke=False,
        force_upload=False,  # r5: the upload-sentinel skip contract's override flag
    )


def _seamed_hub(monkeypatch):
    from explore_persona_space.orchestrate import hub

    up = create_autospec(hub._upload, return_value="hf://ok")
    verify = create_autospec(hub.verify_repo_paths_uploaded, return_value=[])
    monkeypatch.setattr(hub, "_upload", up)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", verify)
    return up


def test_upload_roster_drop_apps_branch(cap, tmp_path, monkeypatch):
    """DROP->APPS: the bare --surface code upload takes the gate roster —
    APPS tar uploaded, NO BCB manifest demanded (r3 Critical 3)."""
    args = _upload_env(
        cap,
        tmp_path,
        gate={"bcb_fit_allowed": False, "apps_activated": True},
        store_benches=["humaneval", "mbpp_full", "lcb_v5", "leetcode", "apps_intro"],
    )
    up = _seamed_hub(monkeypatch)
    # R8 (fb535439d1) unlinks each tar right after its verified upload, so the
    # manifest-member check must read the tar AT upload time, not after.
    seen_members: dict[str, list[str]] = {}

    def _record_tar(local_path, *a, **kw):
        p = Path(str(local_path))
        if p.suffix == ".tar" and p.exists():
            with tarfile.open(p) as tf:
                seen_members[p.name] = [m.name for m in tf.getmembers()]
        return DEFAULT  # fall through to the autospec return_value

    up.side_effect = _record_tar
    cap.phase_upload(args)
    dests = [c.kwargs.get("path_in_repo", "") for c in up.call_args_list]
    assert any(d.endswith("/apps_intro.tar") for d in dests)
    assert not any("bigcodebench_full" in d for d in dests)
    # R8 ENOSPC fix: the tar is a pure upload vehicle, dropped after upload.
    assert not Path(args.store_root, "apps_intro.tar").exists()
    assert any(m.endswith("_capture_manifest.json") for m in seen_members["apps_intro.tar"])


def test_upload_roster_keep_branch_ignores_stale_apps_manifest(cap, tmp_path, monkeypatch):
    """KEEP: a stale APPS store manifest on disk must NOT ride the upload —
    the roster comes from the gate decision, never file existence."""
    args = _upload_env(
        cap,
        tmp_path,
        gate={"bcb_fit_allowed": True, "apps_activated": False},
        store_benches=[
            "humaneval",
            "mbpp_full",
            "bigcodebench_full",
            "lcb_v5",
            "leetcode",
            "apps_intro",  # stale contingency residue
        ],
    )
    up = _seamed_hub(monkeypatch)
    cap.phase_upload(args)
    dests = [c.kwargs.get("path_in_repo", "") for c in up.call_args_list]
    assert any(d.endswith("/bigcodebench_full.tar") for d in dests)
    assert not any("apps_intro" in d for d in dests)


def test_upload_roster_missing_gate_fail_loud(cap, tmp_path, monkeypatch):
    args = _upload_env(cap, tmp_path, gate={}, store_benches=["humaneval"])
    (Path(args.out_root) / "code" / "code_gate.json").unlink()
    _seamed_hub(monkeypatch)
    with pytest.raises(FileNotFoundError, match="gate verdict missing"):
        cap.phase_upload(args)


# ---------------------------------------------------------------------------
# 5. dv_build: APPS-inclusive realized-floor guard (REAL build_surface_dv body)
# ---------------------------------------------------------------------------


def _gen_fixture_below_floor(gen_root: Path, *, apps_activated: bool) -> None:
    _write_gate(
        gen_root,
        bcb_fit_allowed=False,
        apps_required=True,
        apps_activated=apps_activated,
    )
    benches = ["humaneval", "mbpp_full", "lcb_v5", "leetcode"] + (
        ["apps_intro"] if apps_activated else []
    )
    for bench in benches:
        items = [
            {"item_id": f"{bench}-{i}", "benchmark": bench, "verdicts": [True, False]}
            for i in range(10)
        ]
        (gen_root / "code" / f"{bench}.json").write_text(
            json.dumps({"k_rollouts": 2, "items": items})
        )


def test_dv_floor_apps_inclusive_refuses_by_default(dvb, tmp_path):
    """The r3 NIT: the fixed APPS-present below-floor invariant, through the
    REAL build_surface_dv body (the production-argv test stubs it out)."""
    gen_root = tmp_path / "gen"
    _gen_fixture_below_floor(gen_root, apps_activated=True)
    with pytest.raises(RuntimeError, match="even WITH the APPS"):
        dvb.build_surface_dv("code", gen_root, tmp_path / "dv")
    # explicit --allow-below-floor: disclosed degraded regime, recorded
    out = dvb.build_surface_dv("code", gen_root, tmp_path / "dv", allow_below_floor=True)
    payload = json.loads(Path(out).read_text())
    gd = payload["gate_decisions"]
    assert gd["below_floor_disclosed"] is True
    assert 0 < gd["realized_train_with_dv"] < gd["code_train_floor_d"]
    assert {r["benchmark"] for r in payload["rows"]} == set(DROP_APPS_ROSTER)


def test_dv_floor_without_apps_names_fork5_chain(dvb, tmp_path):
    gen_root = tmp_path / "gen"
    _gen_fixture_below_floor(gen_root, apps_activated=False)
    with pytest.raises(RuntimeError, match="fork-5 chain"):
        dvb.build_surface_dv("code", gen_root, tmp_path / "dv")
    with pytest.raises(RuntimeError, match="fork-5 chain"):
        # --allow-below-floor is NOT a bypass of the un-activated fallback
        dvb.build_surface_dv("code", gen_root, tmp_path / "dv", allow_below_floor=True)
