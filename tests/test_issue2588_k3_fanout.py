"""Exercise realized-width scheduling with actual isolated child processes."""

import importlib.util
import json
import os
from itertools import pairwise
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "k3_fanout", Path(__file__).resolve().parents[1] / "scripts/issue2588_k3_fanout.py"
)
fanout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fanout)


@pytest.fixture
def child_job(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "issue2588_k3_job.py").write_text(
        "import json,os,sys,time\n"
        "from pathlib import Path\n"
        "cell=sys.argv[sys.argv.index('--cell')+1]\n"
        "record={'cell':cell,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid(),'start':time.monotonic()}\n"
        "time.sleep(0.03)\n"
        "record['end']=time.monotonic()\n"
        "with Path(os.environ['K3_QUEUE_TEST_RECORD']).open('a') as f:\n"
        " f.write(json.dumps(record)+'\\n')\n"
        "print('[phase=done] child only')\n"
        "sys.exit(9 if cell=='fail' else 0)\n"
    )
    record = tmp_path / "record.jsonl"
    monkeypatch.setattr(fanout, "ROOT", tmp_path)
    monkeypatch.setenv("K3_QUEUE_TEST_RECORD", str(record))
    # Local queue smoke does not exercise HF; production uses the checked helper.
    monkeypatch.setattr(fanout, "persist_log", lambda cell, log: None)
    return tmp_path, record


@pytest.mark.parametrize("ids", [["5"], ["3", "7"]])
def test_full_cell_queue_on_reduced_width(child_job, ids, capsys):
    root, record = child_job
    cells = [f"cell{i}" for i in range(9)]
    fanout.run_queue(cells, ids, root / "out", poll_s=0.01)
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert sorted(r["cell"] for r in rows) == cells
    assert {r["gpu"] for r in rows} == set(ids)
    assert len({r["pid"] for r in rows}) == len(cells)
    for gpu in ids:
        on_gpu = sorted((r for r in rows if r["gpu"] == gpu), key=lambda r: r["start"])
        assert all(a["end"] <= b["start"] for a, b in pairwise(on_gpu))
    assert capsys.readouterr().out.count("[phase=done]") == 1


def test_failure_stops_pending_cells(child_job):
    root, record = child_job
    with pytest.raises(RuntimeError, match="cell fail failed rc=9") as error:
        fanout.run_queue(["fail", "never"], ["6"], root / "out", poll_s=0.01)
    assert "[phase=done]" not in str(error.value)
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert [r["cell"] for r in rows] == ["fail"]
    with pytest.raises(ProcessLookupError):
        os.killpg(rows[0]["pid"], 0)


def test_allocation_is_preserved_and_empty_is_rejected(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    assert fanout.gpu_ids() == ["4", "6"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="invalid GPU allocation"):
        fanout.gpu_ids()


def test_registry_matches_ten_approved_cells():
    assert len(fanout.approved_cells()) == 10
    assert "q3_32b_a" in fanout.approved_cells()
