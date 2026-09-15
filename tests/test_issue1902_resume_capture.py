"""Regress the real failure: sibling cancellation must not obscure its initiating error."""

import importlib.util
import time
from pathlib import Path

import pytest


def test_initiating_worker_error_survives_sibling_cleanup(tmp_path, monkeypatch, capsys):
    """Launch actual subprocesses and check the error and the cancelled sibling's exit."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "resume_capture_test", scripts / "issue1902_resume_capture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_root = tmp_path / "worker"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "scripts/issue1902_format_job.py").write_text(
        "import sys,time,os\n"
        "from pathlib import Path\n"
        "shard=int(sys.argv[sys.argv.index('--shard')+1])\n"
        "root=Path(sys.argv[sys.argv.index('--root')+1])\n"
        "(root/f'pid_{shard}').write_text(str(os.getpid()))\n"
        "if shard==0: time.sleep(60)\n"
        "else:\n"
        " print('ORIGINAL_STORAGE_FAILURE',flush=True)\n"
        " sys.exit(7)\n"
    )
    monkeypatch.setattr(module.C, "ROOT", fake_root)
    with pytest.raises(RuntimeError, match=r"Initiating worker failure .*rc=7"):
        module.capture_children(tmp_path, ["0", "1"], time.time() + 30, log_root=tmp_path)
    assert "ORIGINAL_STORAGE_FAILURE" in capsys.readouterr().out
    for shard in (0, 1):
        pid = int((tmp_path / f"pid_{shard}").read_text())
        assert not Path(f"/proc/{pid}").exists(), f"Worker leaked: {pid}"
