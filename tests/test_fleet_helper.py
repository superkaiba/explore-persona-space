"""Unit tests for the #676 wave-parallel fleet dispatcher helper.

All CPU-only (no GPU, no API). The wave-launch tests use a stub ``build_cmd``
returning a ``CellCmd`` whose ``argv`` is an ``echo`` / tiny-python no-op, plus a
captured launcher env, so ``run_parallel_with_log`` actually fan-out-launches real
(trivial) subprocesses and we assert the per-cell CVD pin lands in the launcher
environment (the gotchas.md cuInit-freeze guard, mirroring
``tests/test_cvd_wave_assignment_smoke.py`` assertion 2).

Test 8 drives the REAL ``issue664_dispatch.main()`` single-GPU backward-compat
path (``--cells 1 --smoke`` with NO ``--n-gpus``) with ``WaveDispatcher.run``
monkey-patched to a capture-only stub, asserting the dispatcher enqueues exactly
one cell per GPU-bound phase on gpu 0 with ``CUDA_VISIBLE_DEVICES=0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import fleet as F
from explore_persona_space.orchestrate.fleet import (
    CellCmd,
    DuplicateCellError,
    FleetResult,
    JudgeHandle,
    WaveDispatcher,
    WaveFailedError,
    assign_gpu_ids,
    run_parallel_with_log,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


# ── stub cell + build_cmd helpers ─────────────────────────────────────────────


def _echo_cmd(key: str, gpu: int, log_dir: Path, *, rc: int = 0, drop_cvd: bool = False) -> CellCmd:
    """A trivial one-cell launch spec: a python no-op exiting ``rc``, CVD pinned.

    Each cell's argv writes the env CVD it actually SAW to ``log_dir/<key>.cvd`` so
    a test can assert the per-cell launcher-env pin took (not just that the dataclass
    field was set). ``drop_cvd`` omits the CVD pin to exercise the loud pre-launch
    assert.
    """
    env = {} if drop_cvd else {"CUDA_VISIBLE_DEVICES": str(gpu)}
    capture = log_dir / f"{key}.cvd"
    script = (
        "import os,sys,pathlib;"
        f"pathlib.Path({str(capture)!r}).write_text(os.environ.get('CUDA_VISIBLE_DEVICES','UNSET'));"
        f"sys.exit({rc})"
    )
    return CellCmd(
        cell_key=key,
        argv=[sys.executable, "-c", script],
        env=env,
        log_path=log_dir / f"{key}.log",
        gpu_id=gpu,
    )


# ── 1. assign_gpu_ids round-robin ─────────────────────────────────────────────


def test_assign_gpu_ids_round_robin():
    # 12 cells over 4 GPUs -> the #651:677 per-wave densification pattern.
    assert assign_gpu_ids(12, 4) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    # fewer cells than GPUs -> only the needed GPUs.
    assert assign_gpu_ids(3, 4) == [0, 1, 2]
    # single-GPU collapse: every cell on gpu 0 (the unchanged serial path).
    assert assign_gpu_ids(5, 1) == [0, 0, 0, 0, 0]
    # n_gpus<=0 is treated as 1 (defensive).
    assert assign_gpu_ids(3, 0) == [0, 0, 0]


# ── 2. disjoint sharding raises on a duplicate cell_key ───────────────────────


def test_disjoint_sharding_raises_on_duplicate(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    cells = ["a", "b", "a"]  # 'a' collides
    disp = WaveDispatcher(
        n_gpus=2,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=lambda c, g: _echo_cmd(c, g, log_dir),
    )
    with pytest.raises(DuplicateCellError) as ei:
        disp.run(cells)
    assert "a" in ei.value.colliding_keys

    # positive case: distinct keys run clean (n_gpus=1 so deterministic).
    disp_ok = WaveDispatcher(
        n_gpus=1,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=lambda c, g: _echo_cmd(c, g, log_dir),
    )
    res = disp_ok.run(["x", "y"])
    assert sorted(res.ran) == ["x", "y"]
    assert res.failures == []


# ── 3. disjoint output paths for the REAL #664 grid ───────────────────────────


def test_disjoint_output_paths_for_664_cells():
    sys.path.insert(0, str(SCRIPTS))
    import issue664_common as C

    grid = C.realized_grid()
    assert len(grid) > 0
    # every cell's idempotency key is unique across the whole fleet.
    keys = [c.eval_key for c in grid]
    assert len(set(keys)) == len(keys), "realized_grid eval_keys are not unique"
    # the WaveDispatcher whole-fleet uniqueness assert accepts the real grid.
    disp = WaveDispatcher(
        n_gpus=4,
        cell_key=lambda c: c.eval_key,
        is_done=lambda c: True,  # mark all done -> no launch, just the uniqueness assert
        build_cmd=lambda c, g: None,  # never called (all skipped)
    )
    res = disp.run(grid)
    assert sorted(res.skipped) == sorted(keys)
    assert res.ran == []
    # every derived output path is key-distinct (distinct keys -> distinct paths).
    subfolders = {c.hf_adapter_subfolder for c in grid}
    assert len(subfolders) == len(grid)

    # Point D: derive the THREE concrete on-disk output-path families the plan §5
    # names (not just the eval_key) and assert pairwise uniqueness within each set,
    # for BOTH real and smoke roots (smoke rebinds via the "_smoke" suffix).
    import issue664_dispatch as D

    for suffix in ("", "_smoke"):
        adapter_dirs = {D.ADAPTER_OUT / (c.eval_key + suffix) for c in grid}
        store_dirs = {C.STORE_ROOT / (c.eval_key + suffix) for c in grid}
        merged_dirs = {D.ADAPTER_OUT / (c.eval_key + suffix + "_merged") for c in grid}
        for fam, name in (
            (adapter_dirs, "adapter"),
            (store_dirs, "store"),
            (merged_dirs, "merged"),
        ):
            assert len(fam) == len(grid), f"{name} dirs collide (suffix={suffix!r})"
        # the merged dir never collides with a bare adapter dir of another cell.
        assert adapter_dirs.isdisjoint(merged_dirs)


# ── 4. idempotent resume-skip ─────────────────────────────────────────────────


def test_idempotent_resume_skip(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    cells = ["c0", "c1", "c2", "c3", "c4", "c5"]
    done = {"c0", "c2", "c4"}
    built: list[str] = []

    def _build(c, g):
        built.append(c)
        return _echo_cmd(c, g, log_dir)

    disp = WaveDispatcher(
        n_gpus=2,
        cell_key=lambda c: c,
        is_done=lambda c: c in done,
        build_cmd=_build,
    )
    res = disp.run(cells)
    assert sorted(res.skipped) == ["c0", "c2", "c4"]
    assert sorted(res.ran) == ["c1", "c3", "c5"]
    # build_cmd invoked ONLY for the un-done cells.
    assert sorted(built) == ["c1", "c3", "c5"]


# ── 5. CVD-present pre-launch assert ───────────────────────────────────────────


def test_cvd_present_assert(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # A build_cmd that omits CUDA_VISIBLE_DEVICES -> loud pre-launch failure.
    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        run_parallel_with_log([_echo_cmd("bad", 1, log_dir, drop_cvd=True)])

    # Correct pins -> the captured launcher env CVD matches the assigned gpu_id
    # per cell (the gotchas.md launcher-env pin, mirroring the cvd-wave smoke test).
    cmds = [_echo_cmd("g0", 0, log_dir), _echo_cmd("g1", 1, log_dir), _echo_cmd("g2", 2, log_dir)]
    rcs = run_parallel_with_log(cmds)
    assert rcs == [0, 0, 0]
    assert (log_dir / "g0.cvd").read_text() == "0"
    assert (log_dir / "g1.cvd").read_text() == "1"
    assert (log_dir / "g2.cvd").read_text() == "2"


# ── 6. smoke == single-GPU sweep-of-one (PASS_UNIFIED) ────────────────────────


def test_smoke_single_gpu_equivalence(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    built: list[tuple[str, int]] = []

    def _build(c, g):
        built.append((c, g))
        return _echo_cmd(c, g, log_dir)

    disp = WaveDispatcher(
        n_gpus=1,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=_build,
    )
    res = disp.run(["only"])
    assert isinstance(res, FleetResult)
    assert res.ran == ["only"]
    assert res.wave_count == 1
    # exactly one cell, launched on gpu 0 (no in-process-vs-subprocess divergence).
    assert built == [("only", 0)]
    assert (log_dir / "only.cvd").read_text() == "0"


# ── 7. WaveFailedError lists the failing (rc, cell_key) ───────────────────────


def test_wave_failed_error_lists_bad_cells(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    def _build(c, g):
        # 'boom' exits non-zero; the others succeed.
        return _echo_cmd(c, g, log_dir, rc=7 if c == "boom" else 0)

    disp = WaveDispatcher(
        n_gpus=4,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=_build,
    )
    with pytest.raises(WaveFailedError) as ei:
        disp.run(["ok1", "boom", "ok2"])
    assert (7, "boom") in ei.value.failures
    assert all(key == "boom" for _rc, key in ei.value.failures)


# ── 8. REAL issue664_dispatch.main() single-GPU backward-compat ───────────────


def test_issue664_main_smoke_backcompat_no_n_gpus(monkeypatch, tmp_path):
    """Drive the REAL dispatcher main() for ``--cells 1 --smoke`` with NO --n-gpus
    (the single-GPU backward-compat path, acceptance #5). No subprocess executes:
    WaveDispatcher.run is monkey-patched to a capture-only stub recording the cells
    + the CellCmds the dispatcher's build_cmd produces. Asserts the new argparse
    branch + argv-building + CVD-env layer preserve today's --gpu-id 0 behavior."""
    sys.path.insert(0, str(SCRIPTS))
    import issue664_common as C
    import issue664_dispatch as D

    # Capture every WaveDispatcher.run invocation: the cells + the CellCmd each
    # build_cmd produces (gpu_id assigned via assign_gpu_ids over the wave) + the
    # is_done predicate (so we can assert which sentinel paths it probes).
    captured: list[dict] = []

    def _capture_run(self, cells, *, cwd=None):
        gpu_ids = assign_gpu_ids(len(cells), self.n_gpus)
        cmds = [self.build_cmd(c, g) for c, g in zip(cells, gpu_ids, strict=True)]
        captured.append(
            {
                "n_gpus": self.n_gpus,
                "cells": list(cells),
                "cmds": cmds,
                "is_done": self.is_done,
            }
        )
        return FleetResult(ran=[c.eval_key for c in cells], skipped=[], failures=[], wave_count=1)

    monkeypatch.setattr(D.WaveDispatcher, "run", _capture_run)
    # Neutralize every non-target step so only the P2.1/P2.2 wave construction runs.
    monkeypatch.setattr(D, "phase0", lambda args: None)
    monkeypatch.setattr(D, "_require_credentials", lambda: None)
    monkeypatch.setattr(D, "_drop_filtered", lambda cells: cells)
    monkeypatch.setattr(D, "_write_manifest", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "_marker_readability_assert", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "upload_artifacts", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "write_sentinel", lambda *a, **k: tmp_path / "sentinel.json")
    monkeypatch.setattr(D, "_wandb_entity", lambda: None)
    monkeypatch.setattr(D, "_dropped_cell_keys", set)

    # --cells 1 --smoke, NO --n-gpus (default 1). --phase all so both loops run.
    monkeypatch.setattr(
        sys, "argv", ["issue664_dispatch.py", "--phase", "all", "--cells", "1", "--smoke"]
    )
    rc = D.main()
    assert rc == 0

    # Two WaveDispatcher.run invocations: P2.1 train + P2.2 extract+eval.
    assert len(captured) == 2, f"expected 2 wave runs (train + extract+eval), got {len(captured)}"
    train_cap, extract_cap = captured

    for cap in (train_cap, extract_cap):
        # (default n_gpus) backward-compat: single-GPU path.
        assert cap["n_gpus"] == 1
        # (a) exactly ONE cell enqueued per phase.
        assert len(cap["cells"]) == 1
        assert len(cap["cmds"]) == 1
        cmd = cap["cmds"][0]
        # (b) gpu_id == 0 (preserves the --gpu-id 0 default).
        assert cmd.gpu_id == 0
        # (c) CUDA_VISIBLE_DEVICES == "0" in the launcher env.
        assert cmd.env["CUDA_VISIBLE_DEVICES"] == "0"
        # (d) --smoke threaded into the subprocess argv.
        assert "--smoke" in cmd.argv
        # the subprocess re-invokes THIS dispatcher in one-cell mode.
        assert str(SCRIPTS / "issue664_dispatch.py") in [str(a) for a in cmd.argv]

    # the two phases use DISTINCT one-cell mode flags.
    assert "--train-one-cell" in train_cap["cmds"][0].argv
    assert "--extract-eval-one-cell" in extract_cap["cmds"][0].argv

    # (e) is_done keys on the per-cell sentinel paths the in-process code uses.
    smoke_cell = train_cap["cells"][0]
    # train skip -> adapter_model.safetensors under the eval_key (+_smoke) dir.
    expected_adapter = (
        D.ADAPTER_OUT / (smoke_cell.eval_key + "_smoke") / "adapter_model.safetensors"
    )
    assert not expected_adapter.exists()  # clean test tree
    assert train_cap["is_done"](smoke_cell) is False
    # extract+eval skip -> store tensors.pt AND a non-empty eval registry dir.
    expected_store = C.STORE_ROOT / (smoke_cell.eval_key + "_smoke") / "tensors.pt"
    assert not expected_store.exists()
    assert extract_cap["is_done"](smoke_cell) is False


# ── 9. deferred judge save_raw is per-source distinct (concern judge-save-raw-collision)


def _stub_handle(
    tmp_path: Path, source: str, scores: dict[str, dict]
) -> tuple[JudgeHandle, dict[str, dict]]:
    """A JudgeHandle whose submit is a no-op and whose harvest returns ``scores``.

    The save_raw key is the per-source key the fix mandates
    (``judge_filter/{behavior}__{src}.json``). ``_reconcile`` mimics the real
    ``_reconcile_judge_batches`` closure: re-read an existing save_raw, else write
    ``scores`` and return them. Returns the handle + the dict the fake harvest
    yields (so the test can assert what landed on disk vs what was read back).
    """
    import json as _json

    behavior = "sycophancy"
    save_raw = tmp_path / "judge_filter" / f"{behavior}__{source}.json"
    expected = frozenset(scores)

    def _reconcile(_batch_ids):
        if save_raw.exists():
            on_disk = _json.loads(save_raw.read_text()).get("all_scores", {})
            if not expected <= set(on_disk):
                raise RuntimeError(f"incomplete save_raw at {save_raw}")
            return on_disk
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(_json.dumps({"all_scores": scores}))
        return scores

    handle = JudgeHandle(
        cell_key=f"elicit_{behavior}__{source}",
        save_raw=save_raw,
        _submit=lambda: [],
        _reconcile=_reconcile,
        expected_custom_ids=expected,
        expected_source=source,
    )
    return handle, scores


def test_judge_save_raw_per_source_distinct(tmp_path):
    """Two same-behavior source jobs must NOT share one ``save_raw`` — the #676
    deferred-reconcile collision the reconciler upheld (concern
    judge-save-raw-collision).

    The pre-fix bug keyed ``save_raw`` on ``{behavior}_{int(time.time())}.json``, so
    two same-behavior sources whose wall-clock second collided shared one file; the
    fix keys it per source (``{behavior}__{src}.json``). The stub handles below
    build the per-source paths the fix mandates and assert:

    (a) the two sources' ``save_raw`` paths are DISTINCT;
    (b) source-A and source-B labels differ when the scored responses differ;
    (c) re-running source-B's reconcile after source-A already wrote to disk does
        NOT corrupt source-B's labels with source-A's.
    """
    # Position-keyed custom_ids are identical across sources, but reflect DIFFERENT resps.
    scores_a = {"elicit__00000__00": {"behavior": 1}, "elicit__00001__00": {"behavior": 1}}
    scores_b = {"elicit__00000__00": {"behavior": 0}, "elicit__00001__00": {"behavior": 0}}

    h_a, _ = _stub_handle(tmp_path, "kindergarten_teacher", scores_a)
    h_b, _ = _stub_handle(tmp_path, "software_engineer", scores_b)

    # (a) distinct save_raw paths even though behavior + wall-clock are identical.
    assert h_a.save_raw != h_b.save_raw

    h_a.submit()
    h_b.submit()

    # source A reconciles first and writes its save_raw.
    out_a = h_a.reconcile()
    assert out_a == scores_a

    # source B reconciles next — must read ITS OWN scores, not A's.
    out_b = h_b.reconcile()
    assert out_b == scores_b
    # (b) the labels differ where the scored responses differ.
    assert out_a["elicit__00000__00"]["behavior"] != out_b["elicit__00000__00"]["behavior"]

    # (c) re-running B's reconcile after A is on disk re-reads B's own save_raw — no
    # cross-contamination from A's labels.
    out_b_again = h_b.reconcile()
    assert out_b_again == scores_b


def test_submit_behavior_labels_keys_save_raw_per_source(monkeypatch):
    """The PRODUCTION fix site: ``_submit_behavior_labels`` keys ``save_raw`` per
    source (``{behavior}__{src}.json``) and tags the handle with ``expected_source``,
    so two same-behavior sources never share a file (concern judge-save-raw-collision).

    Stubs ``submit_judge_async`` to capture the (save_raw, expected_source, cell_key)
    each call produces — no live Batch API.
    """
    sys.path.insert(0, str(SCRIPTS))
    import issue664_dispatch as D

    calls: list[dict] = []

    def _fake_submit_judge_async(_completions, *, save_raw, expected_source, cell_key, **kw):
        calls.append(
            {"save_raw": save_raw, "expected_source": expected_source, "cell_key": cell_key}
        )
        return JudgeHandle(
            cell_key=cell_key,
            save_raw=Path(save_raw),
            _submit=lambda: [],
            _reconcile=lambda _b: {},
            expected_source=expected_source,
        )

    monkeypatch.setattr(D, "submit_judge_async", _fake_submit_judge_async)
    # _submit_behavior_labels resolves the judge column via issue664_eval; the stubbed
    # submit never reaches the API, but the column lookup + system-prompt build run.
    qr = [("claim-0", "resp-0"), ("claim-1", "resp-1")]
    D._submit_behavior_labels("sycophancy", "kindergarten_teacher", qr, smoke=False)
    D._submit_behavior_labels("sycophancy", "software_engineer", qr, smoke=False)

    assert len(calls) == 2
    # Both are the SAME behavior, so a behavior-only (or timestamp) key would collide;
    # the per-source key keeps them distinct.
    raws = [c["save_raw"].name for c in calls]
    assert raws == ["sycophancy__kindergarten_teacher.json", "sycophancy__software_engineer.json"]
    assert len({c["save_raw"] for c in calls}) == 2
    # expected_source is threaded onto each handle (belt-and-suspenders ownership tag).
    assert [c["expected_source"] for c in calls] == ["kindergarten_teacher", "software_engineer"]
    # no coarse int(time.time()) component survives in the key.
    assert not any(any(ch.isdigit() for ch in r.split("__", 1)[1]) for r in raws)


# ── 10. reconcile raises on incomplete custom-id coverage (concern coverage-unverified)


def test_judge_reconcile_raises_on_incomplete_coverage(tmp_path, monkeypatch):
    """An ENDED-but-incomplete shard (a custom_id missing from the harvest) must
    raise ``RuntimeError`` naming the missing id BEFORE writing ``save_raw`` — the
    fail-loud coverage check the reconciler upheld (concern
    judge-custom-id-coverage-unverified).
    """
    import explore_persona_space.eval.batch_judge as BJ

    save_raw = tmp_path / "judge_filter" / "sycophancy__src.json"
    expected = frozenset({"elicit__00000__00", "elicit__00001__00"})

    # Fake the harvest to return ONLY the first expected id (the second is missing,
    # as an ended-but-incomplete shard would leave it).
    def _fake_collect(_client, _batch_id, results):
        results["elicit__00000__00"] = {"behavior": 1}

    monkeypatch.setattr(BJ, "_collect_legacy_results", _fake_collect)
    # Neutralize the poll-to-ended (no live API).
    monkeypatch.setattr(F, "_poll_batch_to_ended", lambda *a, **k: None)
    # Neutralize the anthropic client construction inside the reconcile.
    monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: object())

    with pytest.raises(RuntimeError, match="elicit__00001__00"):
        F._reconcile_judge_batches(
            ["batch_xyz"],
            cell_key="elicit_sycophancy__src",
            save_raw=save_raw,
            judge_model="claude-sonnet-4-5-20250929",
            poll_interval=1.0,
            max_poll_interval=2.0,
            grace_min=1,
            expected_custom_ids=expected,
        )

    # save_raw was NOT written (the raise fires before write_text).
    assert not save_raw.exists()


# ── 11. single-GPU --gpu-id passthrough + multi-GPU rejection (Point C / Must-Fix #3)


def test_issue664_main_smoke_backcompat_explicit_gpu_id(monkeypatch, tmp_path):
    """Drive the REAL dispatcher main() for ``--cells 1 --smoke --gpu-id 2`` (NO
    --n-gpus, default 1): the single-GPU path must thread ``args.gpu_id`` through
    the wave builders — ``cmd.gpu_id == 2``, ``CUDA_VISIBLE_DEVICES == "2"``, and
    ``--gpu-id 2`` in the subprocess argv (Point C / Must-Fix #3).

    Also asserts that ``--n-gpus 2 --gpu-id 1`` is REJECTED loudly (a nonzero
    single-GPU selector is incoherent with multi-GPU wave assignment).
    """
    sys.path.insert(0, str(SCRIPTS))
    import issue664_dispatch as D

    captured: list[dict] = []

    def _capture_run(self, cells, *, cwd=None):
        gpu_ids = assign_gpu_ids(len(cells), self.n_gpus)
        cmds = [self.build_cmd(c, g) for c, g in zip(cells, gpu_ids, strict=True)]
        captured.append({"n_gpus": self.n_gpus, "cells": list(cells), "cmds": cmds})
        return FleetResult(ran=[c.eval_key for c in cells], skipped=[], failures=[], wave_count=1)

    monkeypatch.setattr(D.WaveDispatcher, "run", _capture_run)
    monkeypatch.setattr(D, "phase0", lambda args: None)
    monkeypatch.setattr(D, "_require_credentials", lambda: None)
    monkeypatch.setattr(D, "_drop_filtered", lambda cells: cells)
    monkeypatch.setattr(D, "_write_manifest", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "_marker_readability_assert", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "upload_artifacts", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "write_sentinel", lambda *a, **k: tmp_path / "sentinel.json")
    monkeypatch.setattr(D, "_wandb_entity", lambda: None)
    monkeypatch.setattr(D, "_dropped_cell_keys", set)

    # single-GPU path with a NON-default --gpu-id: must thread gpu 2 end to end.
    monkeypatch.setattr(
        sys,
        "argv",
        ["issue664_dispatch.py", "--phase", "p1", "--cells", "1", "--smoke", "--gpu-id", "2"],
    )
    rc = D.main()
    assert rc == 0
    assert len(captured) == 1  # only the p1 train wave (--phase p1)
    cmd = captured[0]["cmds"][0]
    assert cmd.gpu_id == 2
    assert cmd.env["CUDA_VISIBLE_DEVICES"] == "2"
    argv_str = [str(a) for a in cmd.argv]
    assert "--gpu-id" in argv_str
    assert argv_str[argv_str.index("--gpu-id") + 1] == "2"

    # multi-GPU + a nonzero single-GPU selector is incoherent -> loud rejection.
    monkeypatch.setattr(
        sys,
        "argv",
        ["issue664_dispatch.py", "--phase", "p1", "--cells", "1", "--n-gpus", "2", "--gpu-id", "1"],
    )
    with pytest.raises(SystemExit):
        D.main()
