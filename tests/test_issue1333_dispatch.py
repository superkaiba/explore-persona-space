"""#1333 dispatcher regression pins (crash-fix r4 + r5).

1. FT launch width is smoke-INVARIANT (4-way ZeRO-3): the 2026-07-15 pod smoke
   crashed rc=1 at p2_train because ``_ft_num_processes`` returned 1 under
   ``--smoke`` — ``accelerate launch --num_processes 1`` against the 4-GPU
   ZeRO-3 yaml shards nothing and OOMs the whole 7B on one A100-80 at the
   first optimizer step (the #1315 clone-narrowing class, same trainer family).
2. ``_run_subprocess`` echoes the inner-log TAIL to the main log on failure —
   the GCE crash trap persists only the main workload log, so without the echo
   the subprocess traceback dies with the instance (the r4 diagnosability gap:
   ``ft_mk4.log`` was never persisted).
3. (r5) p2 TRAIN scheduling: the FT cell trains ONLY as the top-level
   whole-pod-exclusive launch, NEVER inside a CVD-pinned 1-GPU train fanout
   unit (``_train_schedule`` split + the ``run_train_unit`` refusal guard);
   FT LADDER reads legitimately ride 1-GPU units (plan §9 P3, TP=1 vLLM).
4. (r5) ``_fanout_units`` echoes a FAILING unit's log tail into the MAIN log
   before raising — attempt 2's unit log lived under out_root (smoke: /tmp),
   outside the GCE crash trap's persist globs, and died with the instance
   (epm:failure v2: unit rc=1, root cause unrecoverable).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from explore_persona_space.experiments import issue_1333 as C

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _dispatch():
    import issue1333_dispatch as d

    return d


def _cfg(d, tmp_path: Path, *, smoke: bool):
    return d.Cfg(
        smoke=smoke, cells=(C.CELL_LORA_CON, C.CELL_FT_POS), out_root=tmp_path, upload=False
    )


# ── 1. FT launch width: smoke-invariant 4-way ZeRO-3 (r4 OOM regression pin) ──


def test_ft_launch_width_smoke_invariant(tmp_path, monkeypatch):
    """r4 crash pin: ``_ft_num_processes`` returns 4 (and the composed
    ``accelerate launch`` carries ``--num_processes 4`` + a 4-GPU CVD slice)
    in BOTH modes. The pre-fix smoke branch returned 1, which left the fp32
    Adam moments UNSHARDED on one A100-80 and OOMed deterministically at the
    first optimizer step (epm:failure v1, 2026-07-15; sibling incident #1315)."""
    d = _dispatch()
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0", "1", "2", "3"])
    for smoke in (True, False):
        cfg = _cfg(d, tmp_path, smoke=smoke)
        npr = d._ft_num_processes(cfg)
        assert npr == d.FT_NUM_PROCESSES == 4, (smoke, npr)
        cmd = C.marker_ft_cmd(
            mix_path=tmp_path / "mixes" / "marker_posonly.jsonl",
            out_dir=tmp_path / "train",
            num_processes=npr,
            seed=cfg.seed,
            grid=(1,) if smoke else C.FT_GRID,
            max_steps=1 if smoke else max(C.FT_GRID),
            trainer=d.MARKER_FT_TRAINER,
            accel_config=d.MARKER_ACCEL_CONFIG,
        )
        assert cmd[cmd.index("--num_processes") + 1] == "4", (smoke, cmd)
        # the CVD slice phase_train composes from the same npr
        ids = d._physical_gpu_ids()
        assert ",".join(ids[:npr]) == "0,1,2,3", (smoke, ids)


def test_ft_launch_width_fails_loud_under_provisioned(tmp_path, monkeypatch):
    """Smoke mode inherits the under-provision guard: <4 visible GPUs raises
    instead of silently narrowing the ZeRO-3 world size."""
    d = _dispatch()
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0"])
    with pytest.raises(RuntimeError, match="full-FT needs 4 GPUs"):
        d._ft_num_processes(_cfg(d, tmp_path, smoke=True))


# ── 2. subprocess failure echoes the inner-log tail into the main log ────────


def test_run_subprocess_tail_on_failure(tmp_path, caplog):
    """A failing subprocess's inner-log TAIL (last SUBPROCESS_TAIL_LINES lines)
    lands in the MAIN log via logger.error before the short RuntimeError —
    executed against a real failing subprocess writing 200 lines."""
    d = _dispatch()
    log = tmp_path / "logs" / "inner.log"
    cmd = ["bash", "-c", "seq 1 200 | sed 's/^/line/'; exit 7"]
    with (
        caplog.at_level(logging.ERROR, logger="issue1333"),
        pytest.raises(RuntimeError, match=r"subprocess rc=7"),
    ):
        d._run_subprocess(cmd, log)
    assert "[subprocess-tail]" in caplog.text
    assert "line200" in caplog.text  # tail end present
    assert "line150" in caplog.text  # well inside the 120-line window
    assert "line10\n" not in caplog.text  # early lines beyond the window are cut
    # the inner log itself still holds the full output
    assert "line1\n" in log.read_text()


def test_run_subprocess_success_no_tail(tmp_path, caplog):
    """rc=0 emits no tail and raises nothing."""
    d = _dispatch()
    log = tmp_path / "logs" / "inner_ok.log"
    with caplog.at_level(logging.ERROR, logger="issue1333"):
        d._run_subprocess(["bash", "-c", "echo ok"], log)
    assert "[subprocess-tail]" not in caplog.text
    assert "ok" in log.read_text()


def test_tail_lines_missing_file_fail_soft(tmp_path):
    d = _dispatch()
    out = d._tail_lines(tmp_path / "nope.log", 5)
    assert out.startswith("<inner log unreadable")


# ── 3. p2 scheduling: FT never inside a TRAIN fanout unit (r5) ───────────────


def test_train_schedule_ft_never_in_fanout(tmp_path):
    """The plan §9 P2 split as a pure decision: the FT cell lands ONLY in the
    whole-pod-exclusive list; the fanout list is LoRA-only; the reused arm
    trains nowhere — in BOTH modes, under every cfg cell subset."""
    d = _dispatch()
    subsets = (
        d.SMOKE_CELLS,
        C.ALL_TRAINED_CELLS,
        (C.CELL_FT_POS,),
        (C.CELL_LORA_CON,),
        (C.CELL_LORA_CON, C.CELL_FT_POS),
    )
    for cells in subsets:
        for smoke in (True, False):
            cfg = d.Cfg(smoke=smoke, cells=tuple(cells), out_root=tmp_path, upload=False)
            lora, ft = d._train_schedule(cfg)
            assert C.CELL_FT_POS not in lora, (cells, smoke, lora)
            assert set(lora) <= set(C.NEW_LORA_CELLS), (cells, smoke, lora)
            assert set(ft) <= {C.CELL_FT_POS}, (cells, smoke, ft)
            assert C.REUSED_CELL not in [*lora, *ft], (cells, smoke)
            # coverage: every trained cell in the subset is scheduled somewhere
            trained = [c for c in cells if c != C.REUSED_CELL]
            assert sorted([*lora, *ft]) == sorted(trained), (cells, smoke)


def test_train_schedule_fails_loud_on_unroutable_cell(tmp_path):
    """A future trained-cell class outside NEW_LORA_CELLS/CELL_FT_POS must
    raise, never silently skip training (the #1090 fu5 per-arm-class lesson)."""
    d = _dispatch()
    cfg = d.Cfg(
        smoke=True, cells=(C.CELL_LORA_CON, "mk9_future_cell"), out_root=tmp_path, upload=False
    )
    with pytest.raises(RuntimeError, match="unroutable train cells"):
        d._train_schedule(cfg)


def test_run_train_unit_refuses_ft_cell(tmp_path):
    """FAILS ON THE r4 TIP (no guard: the unit would proceed toward a 1-GPU
    LoRA train of the FT cell). r5 refuses loudly BEFORE any GPU work — a
    CVD-pinned train unit can never mis-train the whole-pod-exclusive cell."""
    d = _dispatch()
    cfg = _cfg(d, tmp_path, smoke=True)
    with pytest.raises(RuntimeError, match="non-LoRA cell"):
        d.run_train_unit(cfg, C.CELL_FT_POS)
    with pytest.raises(RuntimeError, match="non-LoRA cell"):
        d.run_train_unit(cfg, C.REUSED_CELL)


# ── 4. fanout unit failure echoes the unit-log tail into the main log (r5) ───


def _write_unit_stub(tmp_path: Path, body: str) -> Path:
    """A stub issue1333_dispatch.py the fanout pool execs as its unit — real
    subprocess, same launch shape (uv run python <stub> <extra> --gpu-id N)."""
    stub_dir = tmp_path / "stub_scripts"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "issue1333_dispatch.py").write_text(body)
    return stub_dir


def test_fanout_unit_tail_on_failure(tmp_path, caplog, monkeypatch):
    """A failing fanout unit's log TAIL (last SUBPROCESS_TAIL_LINES lines)
    lands in the MAIN log via logger.error before the short RuntimeError —
    executed against a real failing unit subprocess writing 200 lines.
    Attempt 2 (epm:failure v2) died with rc=1 and NO recoverable unit log:
    unit logs live under out_root (smoke: /tmp), outside the GCE crash
    trap's persist globs."""
    d = _dispatch()
    stub_dir = _write_unit_stub(
        tmp_path,
        "import sys\n"
        "for i in range(1, 201):\n"
        "    print(f'unitline{i}')\n"
        "print('RuntimeError: unit-stub distinctive failure')\n"
        "sys.exit(1)\n",
    )
    monkeypatch.setattr(d, "_SCRIPTS_DIR", stub_dir)
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0"])
    real_sleep = d.time.sleep
    monkeypatch.setattr(d.time, "sleep", lambda s: real_sleep(0.1))
    cfg = _cfg(d, tmp_path, smoke=True)
    extra = ["--smoke", "--unit", "train", C.CELL_LORA_CON]
    with (
        caplog.at_level(logging.ERROR, logger="issue1333"),
        pytest.raises(RuntimeError, match=r"fanout unit .* failed rc=1"),
    ):
        d._fanout_units(cfg, [extra])
    assert "[fanout-unit-tail]" in caplog.text
    assert "unit-stub distinctive failure" in caplog.text
    assert "unitline200" in caplog.text  # tail end present
    assert "unitline150" in caplog.text  # well inside the 120-line window
    assert "unitline10\n" not in caplog.text  # early lines beyond the window are cut
    # per-cell log attribution (r5 name fix: kind+arg, not ['--unit', kind])
    unit_log = tmp_path / "unit_logs" / f"unit_train_{C.CELL_LORA_CON}_g0.log"
    assert unit_log.exists()
    assert "unitline1\n" in unit_log.read_text()  # full output still on disk


def test_fanout_unit_success_no_tail(tmp_path, caplog, monkeypatch):
    """rc=0 units emit no tail and return cleanly."""
    d = _dispatch()
    stub_dir = _write_unit_stub(tmp_path, "print('unit ok')\n")
    monkeypatch.setattr(d, "_SCRIPTS_DIR", stub_dir)
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0"])
    real_sleep = d.time.sleep
    monkeypatch.setattr(d.time, "sleep", lambda s: real_sleep(0.1))
    cfg = _cfg(d, tmp_path, smoke=True)
    with caplog.at_level(logging.ERROR, logger="issue1333"):
        d._fanout_units(cfg, [["--smoke", "--unit", "train", C.CELL_LORA_CON]])
    assert "[fanout-unit-tail]" not in caplog.text


# ── 5. (r6) out-root placement + per-phase disk-headroom probes ──────────────
# Attempt 3 (pod-1333) died at the first FT checkpoint save with
# ``SafetensorError: No space left on device``: the smoke default
# ``/tmp/issue-1333-smoke`` lives on RunPod's 50 GB CONTAINER disk (15 GB
# staged inputs + 24 GB FT out-dir -> 100% full) while the 300 GB /workspace
# volume sat at 17%. GCP masked this (/tmp rides the 300 GB boot disk).


def test_default_out_root_workspace_lane(tmp_path):
    """A /workspace-rooted checkout (RunPod volume / GCE boot clone) anchors
    BOTH modes under the checkout's data/issue_1333 tree — on the volume, and
    inside the GCE crash trap's data_issue persist glob."""
    d = _dispatch()
    ws = tmp_path / "workspace"
    repo = ws / "explore-persona-space"
    repo.mkdir(parents=True)
    assert (
        d._default_out_root(True, repo_root=repo, workspace=ws)
        == repo / "data" / f"issue_{C.ISSUE}" / "smoke"
    )
    assert (
        d._default_out_root(False, repo_root=repo, workspace=ws)
        == repo / "data" / f"issue_{C.ISSUE}" / "run"
    )


def test_default_out_root_local_fallback(tmp_path):
    """No /workspace-rooted checkout (local CPU tests): the small /tmp smoke
    default + the cwd-relative full default survive. A checkout OUTSIDE an
    EXISTING /workspace (the dev VM's incidental dir) is still local."""
    d = _dispatch()
    absent_ws = tmp_path / "no-such-workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    assert d._default_out_root(True, repo_root=repo, workspace=absent_ws) == Path(
        f"/tmp/issue-{C.ISSUE}-smoke"
    )
    assert d._default_out_root(False, repo_root=repo, workspace=absent_ws) == Path(
        f"data/issue_{C.ISSUE}/run"
    )
    present_ws = tmp_path / "workspace"
    present_ws.mkdir()  # exists, but the checkout is NOT under it (the VM shape)
    assert d._default_out_root(True, repo_root=repo, workspace=present_ws) == Path(
        f"/tmp/issue-{C.ISSUE}-smoke"
    )


def test_build_cfg_out_root_arg_and_default_wiring(tmp_path, monkeypatch):
    """--out-root wins verbatim; without it build_cfg routes through
    _default_out_root (both modes)."""
    d = _dispatch()
    explicit = tmp_path / "explicit-root"
    cfg = d.build_cfg(d._parse_args(["--smoke", "--out-root", str(explicit)]))
    assert cfg.out_root == explicit
    sentinel = tmp_path / "resolved-default"
    monkeypatch.setattr(d, "_default_out_root", lambda smoke: sentinel / str(smoke))
    assert d.build_cfg(d._parse_args(["--smoke"])).out_root == sentinel / "True"
    assert d.build_cfg(d._parse_args(["--full"])).out_root == sentinel / "False"


def test_headroom_probe_fails_loud_with_numbers(tmp_path, monkeypatch):
    """statvfs headroom below the phase floor raises BEFORE any write, naming
    the free/required numbers (a mid-save ENOSPC corrupts the checkpoint)."""
    d = _dispatch()
    cfg = _cfg(d, tmp_path, smoke=True)
    monkeypatch.setitem(d.PHASE_HEADROOM_GB, "p2_train", {False: 1e9, True: 1e9})
    with pytest.raises(RuntimeError, match=r"\[disk-headroom\] p2_train.*GB free.*required"):
        d._assert_out_root_headroom(cfg, "p2_train")


def test_headroom_probe_passes_and_cleans_canary(tmp_path, monkeypatch):
    """Above the floor: returns free GB, runs the 1 GB fallocate canary, and
    removes the probe file."""
    d = _dispatch()
    cfg = _cfg(d, tmp_path, smoke=True)
    monkeypatch.setitem(d.PHASE_HEADROOM_GB, "p0_stage", {False: 0.001, True: 0.001})
    free = d._assert_out_root_headroom(cfg, "p0_stage")
    assert free > 0
    assert not (tmp_path / ".headroom_probe").exists()


def test_phases_wire_headroom_probe():
    """p0/p2/p5 (the >=15 GB writers per plan §9) each probe headroom before
    writing; the probe floors cover all three phases in both modes."""
    import inspect

    d = _dispatch()
    for fn, phase in (
        (d.phase_stage, "p0_stage"),
        (d.phase_train, "p2_train"),
        (d.phase_capture, "p5_capture"),
    ):
        assert f'_assert_out_root_headroom(cfg, "{phase}")' in inspect.getsource(fn), phase
        assert set(d.PHASE_HEADROOM_GB[phase]) == {False, True}


def test_no_hardcoded_tmp_issue_write_path_remains():
    """Grep-style pin: the ONLY /tmp/issue-1333 literal left in the dispatcher
    is the guarded local-CPU fallback inside _default_out_root; the geometry
    aggregator has none, and its argparse defaults are repo-root-anchored."""
    import inspect

    d = _dispatch()
    dispatch_src = (REPO_ROOT / "scripts" / "issue1333_dispatch.py").read_text()
    sanctioned = inspect.getsource(d._default_out_root)
    assert "/tmp/issue" in sanctioned  # the local fallback lives (only) here
    remainder = dispatch_src.replace(sanctioned, "")
    assert "/tmp/issue" not in remainder, "hardcoded /tmp/issue-1333 path outside the fallback"
    geometry_src = (REPO_ROOT / "scripts" / "issue1333_geometry.py").read_text()
    assert "/tmp/issue" not in geometry_src
    assert 'default=f"data/issue_' not in geometry_src  # cwd-relative defaults swept (r6)
