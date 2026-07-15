"""#1333 dispatcher regression pins (crash-fix r4).

1. FT launch width is smoke-INVARIANT (4-way ZeRO-3): the 2026-07-15 pod smoke
   crashed rc=1 at p2_train because ``_ft_num_processes`` returned 1 under
   ``--smoke`` — ``accelerate launch --num_processes 1`` against the 4-GPU
   ZeRO-3 yaml shards nothing and OOMs the whole 7B on one A100-80 at the
   first optimizer step (the #1315 clone-narrowing class, same trainer family).
2. ``_run_subprocess`` echoes the inner-log TAIL to the main log on failure —
   the GCE crash trap persists only the main workload log, so without the echo
   the subprocess traceback dies with the instance (the r4 diagnosability gap:
   ``ft_mk4.log`` was never persisted).
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
