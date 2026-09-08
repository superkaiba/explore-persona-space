"""The capture-only continuation cannot schedule generation or renew a work grant."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_dispatch as D
import issue2588_chat_restore_complete as R


def test_capture_pilot_phase_closure_and_no_old_clock(tmp_path):
    args = D.build_parser().parse_args(
        [
            "--mode",
            "capture-pilot",
            "--run-id",
            "qwen3-chat-v3",
            "--science-root",
            str(tmp_path / "science"),
        ]
    )
    steps = D.build_steps(args)
    assert [s.phase for s in steps] == [
        "runtime_check",
        "import_check",
        "preflight",
        "transfer_check",
        "stage-runtime",
        "stage",
        "capture",
        "upload-capture",
    ]
    assert [s.cell for s in steps if s.phase == "capture"] == ["q3_8b_b"]
    assert steps[5].cell == "restore-complete"
    assert all("--smoke" in s.argv for s in steps if "--cell" in s.argv)
    assert all("gen" not in s.argv and "fits" not in s.argv for s in steps)
    assert D.inherited_work_s(args, {"EPS2588_SMOKE_STARTED_AT": "1"}) == 0
    assert "--smoke" in D.cell_step(args, "q3_8b_b", "upload-partial").argv


def test_restore_refuses_changed_payload_before_destination_creation(tmp_path):
    staged, receipt, cell = tmp_path / "staged", tmp_path / "receipt.json", tmp_path / "cell"
    staged.mkdir()
    (staged / "run_identity.json").write_text("{}")
    receipt.write_text("{}")
    with pytest.raises(RuntimeError, match="bytes differ"):
        R.restore_staged(staged, receipt, cell)
    assert not cell.exists()


def test_existing_output_is_never_replaced(tmp_path):
    path = tmp_path / "nested" / "record.json"
    R.put_identical(path, b"original")
    R.put_identical(path, b"original")
    with pytest.raises(RuntimeError, match="differing"):
        R.put_identical(path, b"changed")
    assert path.read_bytes() == b"original"
