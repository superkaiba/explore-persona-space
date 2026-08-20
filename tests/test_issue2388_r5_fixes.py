"""#2388 round-5 concern-closure fixes — pinned regression tests.

Covers the r4 concern-targeting bounce (reconciler-binding
``code-rung1-realization-undisclosed`` + the r4 Minors/NIT + the reconciler r3
restartability deferral + the qa-question-text-source wiring):

1. ``_rung1_realization``: the BCB-DROP branch is explicitly labeled LCB-only
   (planned {bigcodebench_full, lcb_v5} vs realized {lcb_v5} + counts +
   ``bcb_dropped_by_gate`` + gate reason); the KEEP branch realizes the full
   registered cohort; non-code surfaces carry None. phase_select surfaces the
   disclosure top-level in BOTH selection.json and all_arms.json and refuses
   inconsistent realizations in one root.
2. ``_pin_map_payloads``: a recorded map payload whose bytes changed refuses
   the stale-cell resume; --force overwrites; entries MERGE across keys.
3. bootstrap ``inputs_sha``: a changed preds file (same arm roster) recomputes
   the unit instead of silently reusing it (fails pre-fix).
4. code_control per-row freshness (``control_ts``/``control_git_commit``):
   preserved rows keep THEIR OWN stamp, legacy rows backfill from the prior
   top-level ts, and gen.phase_gate surfaces both rows' control_ts into the
   gate verdict; invocation entries carry phase + argv provenance.
5. capture phase_upload sentinel: a verified upload's byte-stat-identical
   rerun SKIPS the Hub transfers; a changed manifest or --force-upload
   re-transfers.
6. QA question text from the banked #1739 packed labeling shards
   (``_qa_questions_from_shards`` + the ``_attach_questions`` QA branch):
   exact context_id join, src-discriminator filtering (manifest + foreign
   sources never join), rollout-conflict refusal, and the no-shards refusal.

Adoptable: repo-root-relative paths, tmp_path outputs, no network / GPU.
CONTENT HYGIENE: all fixtures are benign synthetic text.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import create_autospec

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
def drv():
    return _load_script("issue2388_fits_r5", "scripts/issue2388_fits.py")


@pytest.fixture(scope="module")
def gen():
    return _load_script("issue2388_gen_r5", "scripts/issue2388_gen.py")


@pytest.fixture(scope="module")
def cc():
    return _load_script("issue2388_code_control_r5", "scripts/issue2388_code_control.py")


@pytest.fixture(scope="module")
def cap():
    return _load_script("issue2388_capture_r5", "scripts/issue2388_capture.py")


# ---------------------------------------------------------------------------
# 1. rung-1 realization disclosure (reconciler-binding r4 Major)
# ---------------------------------------------------------------------------


def _tiny_table(drv, surface: str, benchmarks: list[str], splits: list[str]):
    n = len(benchmarks)
    return drv.SurfaceTable(
        surface=surface,
        ctx_ids=[f"c{i}" for i in range(n)],
        dv=np.linspace(0.0, 1.0, n),
        split=np.array(splits),
        group=np.array([f"g{i}" for i in range(n)]),
        boot_group=np.array([f"g{i}" for i in range(n)]),
        benchmark=np.array(benchmarks),
        level=np.full(n, np.nan),
        category=np.array([""] * n),
        z_ctx=np.zeros((1, n, 4), dtype=np.float16),
        z_t1=np.zeros((1, n, 4), dtype=np.float16),
        z_tlast=None,
    )


def test_rung1_realization_drop_branch_labeled_lcb_only(drv):
    """BCB-DROP -> APPS roster: the persisted disclosure names planned
    {bigcodebench_full, lcb_v5} vs realized {lcb_v5} + the gate reason
    (r4 Major: the reduction was silent in every result artifact)."""
    benches = ["humaneval", "mbpp_full", "leetcode", "apps_intro", "lcb_v5", "lcb_v5"]
    splits = ["train", "train", "train", "train", "test", "test"]
    meta = drv._rung1_realization(_tiny_table(drv, "code", benches, splits))
    assert meta["planned_eval"] == ["bigcodebench_full", "lcb_v5"]
    assert meta["realized_eval"] == ["lcb_v5"]
    assert meta["bcb_dropped_by_gate"] is True
    assert "fork-5 gate" in meta["reason"] and "LCB-only" in meta["reason"]
    assert meta["n_eval_rows_by_benchmark"] == {"lcb_v5": 2}
    assert meta["planned_fit"] == ["humaneval", "leetcode", "mbpp_full"]
    assert meta["realized_fit"] == ["humaneval", "leetcode", "mbpp_full"]
    # apps_intro is train-resident but NOT rung-1-fit-eligible: never counted
    assert "apps_intro" not in meta["n_fit_rows_by_benchmark"]


def test_rung1_realization_keep_branch_full_cohort(drv):
    benches = ["humaneval", "mbpp_full", "leetcode", "bigcodebench_full", "lcb_v5"]
    splits = ["train", "train", "train", "test", "test"]
    meta = drv._rung1_realization(_tiny_table(drv, "code", benches, splits))
    assert meta["realized_eval"] == meta["planned_eval"]
    assert meta["bcb_dropped_by_gate"] is False
    assert meta["reason"] is None


def test_rung1_realization_non_code_is_none(drv):
    assert drv._rung1_realization(_tiny_table(drv, "math", ["math_full"], ["test"])) is None


def _select_args(tmp_path: Path, surface: str) -> argparse.Namespace:
    return argparse.Namespace(surface=surface, fits_root=str(tmp_path / "fits"))


def _write_cell(tmp_path: Path, surface: str, name: str, row: dict) -> None:
    d = tmp_path / "fits" / surface / "cells"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(row))


def test_select_aggregates_carry_rung1_realization_top_level(drv, tmp_path):
    """The DROP fixture's disclosure rides BOTH aggregates top-level, so no
    downstream read of selection.json / all_arms.json can mistake the
    LCB-only cohort for the registered {BCB, LCB} transfer read."""
    meta = {
        "planned_eval": ["bigcodebench_full", "lcb_v5"],
        "realized_eval": ["lcb_v5"],
        "bcb_dropped_by_gate": True,
        "reason": "bigcodebench_full absent from realized rows",
    }
    _write_cell(
        tmp_path, "code", "arm_ctx__Lfull_draw0", {"arm": "arm_ctx", "rung1_realization": meta}
    )
    _write_cell(
        tmp_path,
        "code",
        "arm_maplin__Lfull_draw0",
        {"arm": "arm_maplin", "rung1_realization": meta},
    )
    drv.phase_select(_select_args(tmp_path, "code"))
    sel = json.loads((tmp_path / "fits" / "code" / "selection.json").read_text())
    agg = json.loads((tmp_path / "fits" / "code" / "all_arms.json").read_text())
    assert sel["rung1_realization"] == meta
    assert agg["rung1_realization"] == meta
    assert agg["arm_rows"][0]["rung1_realization"] == meta  # per-cell copy intact


def test_select_non_code_rung1_realization_none(drv, tmp_path):
    _write_cell(tmp_path, "math", "arm_ctx__Lfull_draw0", {"arm": "arm_ctx"})
    drv.phase_select(_select_args(tmp_path, "math"))
    sel = json.loads((tmp_path / "fits" / "math" / "selection.json").read_text())
    assert sel["rung1_realization"] is None


def test_select_inconsistent_rung1_realization_refuses(drv, tmp_path):
    m1 = {"realized_eval": ["lcb_v5"], "bcb_dropped_by_gate": True}
    m2 = {"realized_eval": ["bigcodebench_full", "lcb_v5"], "bcb_dropped_by_gate": False}
    _write_cell(tmp_path, "code", "a", {"arm": "arm_ctx", "rung1_realization": m1})
    _write_cell(tmp_path, "code", "b", {"arm": "arm_maplin", "rung1_realization": m2})
    with pytest.raises(RuntimeError, match="inconsistent rung1_realization"):
        drv.phase_select(_select_args(tmp_path, "code"))


# ---------------------------------------------------------------------------
# 2. sweep map-payload digests (reconciler r3 restartability deferral)
# ---------------------------------------------------------------------------


def test_pin_map_payloads_refuses_changed_bytes(drv, tmp_path):
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    out_root = tmp_path / "fits" / "math"
    out_root.mkdir(parents=True)
    (maps_dir / "linear__shared__fu0.npz").write_bytes(b"payload-v1")
    drv._pin_map_payloads(out_root, maps_dir, ("linear__shared__fu0.npz",), force=False)
    reg = json.loads((out_root / "map_payload_shas.json").read_text())
    assert "linear__shared__fu0.npz" in reg
    # unchanged bytes -> resume stays legal
    drv._pin_map_payloads(out_root, maps_dir, ("linear__shared__fu0.npz",), force=False)
    # a refit map (changed bytes) under an unchanged manifest REFUSES
    (maps_dir / "linear__shared__fu0.npz").write_bytes(b"payload-v2-refit")
    with pytest.raises(RuntimeError, match="map payload bytes changed"):
        drv._pin_map_payloads(out_root, maps_dir, ("linear__shared__fu0.npz",), force=False)
    # --force overwrites the entry alongside the cells it recomputes
    drv._pin_map_payloads(out_root, maps_dir, ("linear__shared__fu0.npz",), force=True)
    reg2 = json.loads((out_root / "map_payload_shas.json").read_text())
    assert reg2["linear__shared__fu0.npz"] != reg["linear__shared__fu0.npz"]


def test_pin_map_payloads_merges_across_keys_and_skips_absent(drv, tmp_path):
    """Cross-map_cell incremental fills stay legal: a second invocation
    consuming DIFFERENT keys extends the registry without clobbering, and a
    not-yet-fit payload (absent file) is never recorded."""
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    out_root = tmp_path / "fits" / "math"
    out_root.mkdir(parents=True)
    (maps_dir / "linear__math__fu1.npz").write_bytes(b"lin")
    drv._pin_map_payloads(
        out_root, maps_dir, ("linear__math__fu1.npz", "mlp__math__fu1.pt"), force=False
    )
    reg = json.loads((out_root / "map_payload_shas.json").read_text())
    assert list(reg) == ["linear__math__fu1.npz"]  # absent .pt not recorded
    (maps_dir / "mlp__math__fu1.pt").write_bytes(b"mlp")
    drv._pin_map_payloads(out_root, maps_dir, ("mlp__math__fu1.pt",), force=False)
    reg = json.loads((out_root / "map_payload_shas.json").read_text())
    assert sorted(reg) == ["linear__math__fu1.npz", "mlp__math__fu1.pt"]


# ---------------------------------------------------------------------------
# 3. bootstrap resume keys upstream CONTENT (fails pre-fix)
# ---------------------------------------------------------------------------


def _boot_env(tmp_path: Path) -> argparse.Namespace:
    surface = "math"
    out_root = tmp_path / "fits" / surface
    (out_root / "preds").mkdir(parents=True)
    ids = [f"mathfull-x-{i}" for i in range(9)]
    rng = np.random.default_rng(1)
    for arm in ("arm_ctx", "arm_maplin"):
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
    dv_dir.mkdir(parents=True)
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


def test_bootstrap_changed_preds_content_recomputes(drv, tmp_path, capsys):
    """Same arm roster, same filenames, CHANGED prediction bytes -> the unit
    recomputes (pre-fix the arms-only resume key silently reused it)."""
    args = _boot_env(tmp_path)
    drv.phase_bootstrap(args)
    cells_p = tmp_path / "fits" / "math" / "bootstrap_cells.jsonl"
    rows1 = [json.loads(x) for x in cells_p.read_text().split("\n") if x.strip()]
    assert len(rows1) == 1 and rows1[0]["inputs_sha"]
    # regenerate ONE arm's preds with different values (same ids, same arms)
    pf = tmp_path / "fits" / "math" / "preds" / "preds_arm_ctx_L16_draw0.jsonl"
    lines = [json.loads(x) for x in pf.read_text().split("\n") if x.strip()]
    for r in lines:
        r["y_pred"] = float(r["y_pred"]) + 1.5
    pf.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    drv.phase_bootstrap(args)
    out = capsys.readouterr().out
    assert "RECOMPUTE" in out
    rows2 = [json.loads(x) for x in cells_p.read_text().split("\n") if x.strip()]
    assert len(rows2) == 2, "superseding recomputed row not appended"
    assert rows2[1]["inputs_sha"] != rows1[0]["inputs_sha"]


def test_bootstrap_unchanged_inputs_still_resume(drv, tmp_path):
    args = _boot_env(tmp_path)
    drv.phase_bootstrap(args)
    cells_p = tmp_path / "fits" / "math" / "bootstrap_cells.jsonl"
    first = cells_p.read_text()
    drv.phase_bootstrap(args)
    assert cells_p.read_text() == first


# ---------------------------------------------------------------------------
# 4. code_control per-row freshness + invocation provenance -> gate surfacing
# ---------------------------------------------------------------------------


def _fake_benches(n: int = 25) -> dict:
    def mk(prefix: str):
        items = [{"item_id": f"{prefix}-{i}"} for i in range(n)]
        canon = {f"{prefix}-{i}": [("direct", "pass")] for i in range(n)}
        return {"items": lambda items=items: items, "canon": lambda canon=canon: canon}

    return {"bigcodebench": mk("bcb"), "apps_intro": mk("apps")}


def test_control_rows_carry_and_preserve_freshness_stamp(cc, gen, tmp_path, monkeypatch):
    """Every re-run row gets THIS invocation's control_ts; a preserved row
    keeps ITS OWN stamp verbatim, and phase_gate surfaces both stamps into the
    gate verdict (r4 Minor code-control-preserved-row-freshness)."""
    out_root = tmp_path / "gen"
    (out_root / "code").mkdir(parents=True)
    (out_root / "code" / "dedup_report.json").write_text(
        json.dumps({"n_lcb": 880, "n_dropped_lcb": 373})
    )
    report_p = out_root / gen.CONTROL_REPORT
    monkeypatch.setattr(cc, "BENCHES", _fake_benches())
    monkeypatch.setattr(
        cc, "_verify", lambda bench, fenced, item, bcb_python: bench != "bigcodebench"
    )
    argv1 = ["--out", str(report_p), "--n-control", "25", "--runs", "2"]
    cc.main(argv1)
    rep1 = json.loads(report_p.read_text())
    assert rep1["benchmarks"]["bigcodebench"]["control_ts"] == rep1["invocations"][0]["ts"]
    assert rep1["benchmarks"]["bigcodebench"]["control_git_commit"] is not None
    assert rep1["invocations"][0]["phase"] == "code-harness-control"
    assert rep1["invocations"][0]["argv"] == argv1

    # pin the preserved-row invariant with a sentinel stamp: the APPS-only
    # re-run must carry it VERBATIM (never re-stamp a row it did not run)
    rep1["benchmarks"]["bigcodebench"]["control_ts"] = "2020-01-01T00:00:00Z"
    report_p.write_text(json.dumps(rep1))
    cc.main(["--benchmarks", "apps_intro", "--out", str(report_p), "--runs", "2"])
    rep2 = json.loads(report_p.read_text())
    assert rep2["benchmarks"]["bigcodebench"]["control_ts"] == "2020-01-01T00:00:00Z"
    assert rep2["benchmarks"]["apps_intro"]["control_ts"] == rep2["invocations"][-1]["ts"]

    verdict = gen.phase_gate(out_root)
    assert verdict["g1"]["bcb_control_ts"] == "2020-01-01T00:00:00Z"
    assert verdict["g1"]["apps_control_ts"] == rep2["benchmarks"]["apps_intro"]["control_ts"]


def test_control_legacy_rows_backfill_from_prior_top_level_ts(cc, tmp_path, monkeypatch):
    report_p = tmp_path / "control.json"
    report_p.write_text(
        json.dumps(
            {
                "benchmarks": {"bigcodebench": {"harness_ok": True}},
                "git_commit": "old",
                "ts": "2025-12-31T00:00:00Z",
            }
        )
    )
    monkeypatch.setattr(cc, "BENCHES", _fake_benches())
    monkeypatch.setattr(cc, "_verify", lambda bench, fenced, item, bcb_python: True)
    cc.main(["--benchmarks", "apps_intro", "--out", str(report_p), "--runs", "2"])
    merged = json.loads(report_p.read_text())
    assert merged["benchmarks"]["bigcodebench"]["control_ts"] == "2025-12-31T00:00:00Z"
    assert merged["benchmarks"]["bigcodebench"]["control_git_commit"] == "old"
    # legacy fold carries the prior dirty flag slot (None when never recorded)
    assert "git_dirty" in merged["invocations"][0]


# ---------------------------------------------------------------------------
# 5. capture upload sentinel (r4 Minor capture-upload-phase-not-skippable)
# ---------------------------------------------------------------------------


def _write_gate(out_root: Path, **fields) -> None:
    (out_root / "code").mkdir(parents=True, exist_ok=True)
    (out_root / "code" / "code_gate.json").write_text(json.dumps(fields))


def _upload_env(tmp_path, *, store_benches: list[str]):
    gen_root = tmp_path / "gen"
    _write_gate(gen_root, bcb_fit_allowed=False, apps_activated=True)
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
        force_upload=False,
    )


def _seamed_hub(monkeypatch):
    from explore_persona_space.orchestrate import hub

    up = create_autospec(hub._upload, return_value="hf://ok")
    verify = create_autospec(hub.verify_repo_paths_uploaded, return_value=[])
    monkeypatch.setattr(hub, "_upload", up)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", verify)
    return up


DROP_APPS_STORES = ["humaneval", "mbpp_full", "lcb_v5", "leetcode", "apps_intro"]


def test_upload_sentinel_skips_identical_rerun(cap, tmp_path, monkeypatch):
    args = _upload_env(tmp_path, store_benches=DROP_APPS_STORES)
    up = _seamed_hub(monkeypatch)
    cap.phase_upload(args)
    n_first = up.call_count
    assert n_first > 0
    assert (Path(args.store_root) / "_upload_state.json").exists()
    cap.phase_upload(args)  # byte-stat-identical rerun: NO new Hub transfers
    assert up.call_count == n_first


def test_upload_sentinel_reruns_on_changed_manifest_and_on_force(cap, tmp_path, monkeypatch):
    args = _upload_env(tmp_path, store_benches=DROP_APPS_STORES)
    up = _seamed_hub(monkeypatch)
    cap.phase_upload(args)
    n_first = up.call_count
    # a re-captured store (manifest mtime bumped past the tar) re-uploads
    manifest = Path(args.store_root) / "apps_intro" / "_capture_manifest.json"
    st = manifest.stat()
    os.utime(manifest, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    cap.phase_upload(args)
    assert up.call_count > n_first
    n_second = up.call_count
    args.force_upload = True
    cap.phase_upload(args)  # explicit override re-transfers regardless
    assert up.call_count > n_second


# ---------------------------------------------------------------------------
# 6. QA question text from the banked #1739 packed labeling shards
# ---------------------------------------------------------------------------


def _packed_row(src: str, doc: dict) -> str:
    return json.dumps({"src": src, "doc": doc})


def _write_qa_shards(shards_dir: Path, *, conflicting: bool = False) -> None:
    shards_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        # the one manifest row: MUST be filtered out on the src discriminator
        _packed_row(
            "labeling/hallucination/_manifest.json",
            {"behavior": "hallucination", "n_contexts": 2},
        ),
        _packed_row(
            "labeling/hallucination/hallucination-train-train-000000_seed0.json",
            {
                "context_id": "hallucination-train-train-000000",
                "prefix_text": "You are a helpful assistant.",
                "query": "Who wrote Main Street?",
                "answer_aliases": ["Sinclair Lewis", "Lewis"],
            },
        ),
        _packed_row(
            "labeling/hallucination/hallucination-train-train-000000_seed1.json",
            {
                "context_id": "hallucination-train-train-000000",
                "prefix_text": "You are a helpful assistant.",
                "query": (
                    "Who wrote a DIFFERENT question?" if conflicting else "Who wrote Main Street?"
                ),
                "answer_aliases": ["Sinclair Lewis", "Lewis"],
            },
        ),
        _packed_row(
            "labeling/hallucination/hallucination-eval-nqopen-000001_seed0.json",
            {
                "context_id": "hallucination-eval-nqopen-000001",
                "prefix_text": "",
                "query": "Which state is Richmond the capital of?",
                "answer_aliases": ["Virginia"],
            },
        ),
        # a foreign-source packed row (extraction stage): never joins
        _packed_row(
            "extraction/hallucination/decoy_seed0.json",
            {"context_id": "decoy-ctx", "query": "DECOY", "prefix_text": ""},
        ),
    ]
    (shards_dir / "labeling_hallucination.shard00.jsonl").write_text("\n".join(rows) + "\n")


def test_qa_questions_from_shards_exact_join(drv, tmp_path):
    _write_qa_shards(tmp_path / "shards")
    q_by_ctx, alias_by_ctx = drv._qa_questions_from_shards(tmp_path / "shards")
    assert q_by_ctx["hallucination-train-train-000000"] == (
        "You are a helpful assistant.\nWho wrote Main Street?"
    )
    # empty prefix_text: question text is the bare query (no stray newline)
    assert q_by_ctx["hallucination-eval-nqopen-000001"] == (
        "Which state is Richmond the capital of?"
    )
    assert alias_by_ctx["hallucination-train-train-000000"] == 2
    assert "decoy-ctx" not in q_by_ctx  # src-discriminator filter (packed rule)


def test_qa_questions_conflicting_rollout_text_refuses(drv, tmp_path):
    _write_qa_shards(tmp_path / "shards", conflicting=True)
    with pytest.raises(RuntimeError, match="conflicting question text"):
        drv._qa_questions_from_shards(tmp_path / "shards")


def test_qa_questions_missing_shards_fail_loud(drv, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="labeling_hallucination"):
        drv._qa_questions_from_shards(tmp_path / "empty")


def test_attach_questions_qa_branch(drv, tmp_path):
    _write_qa_shards(tmp_path / "shards")
    table = _tiny_table(
        drv,
        "qa",
        ["triviaqa", "nqopen"],
        ["train", "rung1"],
    )
    table.ctx_ids = ["hallucination-train-train-000000", "hallucination-eval-nqopen-000001"]
    args = argparse.Namespace(qa_questions_shards=str(tmp_path / "shards"))
    drv._attach_questions(args, table)
    assert table.meta["questions"][0].endswith("Who wrote Main Street?")
    assert table.meta["alias_counts"] == [2, 1]


def test_attach_questions_qa_without_shards_refuses(drv):
    table = _tiny_table(drv, "qa", ["triviaqa"], ["train"])
    args = argparse.Namespace(qa_questions_shards=None)
    with pytest.raises(RuntimeError, match="--qa-questions-shards"):
        drv._attach_questions(args, table)
