"""Compare batched transport with the real frozen-recipe publisher on test-only fits."""

import json
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "scripts")]
import issue2588_chat_dispatch as D  # noqa: E402
import issue2588_chat_upload_fits as U  # noqa: E402
import test_issue2588_chat_only as T  # noqa: E402

RC, PC = T.RC, T.PC
generic, upload_boundary = T.generic, T.upload_boundary


def fit_fixture(args, cell, paths, monkeypatch):
    """Only fit values are fixtures; exercise actual inventory/checkpoint/writer bodies."""
    monkeypatch.setattr(RC, "_meta", lambda: {"fixture_only": True})
    for p in RC._phase_artifacts(args, cell, paths, "fits"):
        PC.write_json_atomic(p, {"fixture_only": True})
    PC.write_json_atomic(
        paths["fits"] / "fits_prompt_last.json",
        {
            "fixture_only": True,
            "layer_star": 0,
            "layers": {"0": {"knn_test": {"ridge": {"cosine": {"acc_at_k": {"1": 0.5}}}}}},
        },
    )
    PC.write_json_atomic(paths["fits"] / "fit_pilot.json", {"fixture_only": True})
    RC._mark_phase_done(args, cell, paths, "fits")


def test_exact_legacy_payload_one_bulk_commit(generic, upload_boundary, monkeypatch):
    args, cell, paths = generic
    fit_fixture(args, cell, paths, monkeypatch)
    RC.phase_upload_fits(args, cell, paths)
    expected = dict(upload_boundary.bytes)
    upload_boundary.bytes.clear()
    upload_boundary.single.reset_mock()
    result = U.publish(RC, args)
    receipt = json.loads((paths["cell"] / "uploads/upload-fits.json").read_text())
    assert result == {"status": "complete", "files": len(expected)}
    assert set(receipt["paths"]) == set(expected)
    assert {k: upload_boundary.bytes[k] for k in expected} == expected
    assert upload_boundary.bulk.call_count == 1
    assert upload_boundary.single.call_count == 2  # receipt and completed phase only
    assert RC._phase_complete(args, paths, "upload-fits")
    assert U.publish(RC, args) == {"status": "already_complete"}
    assert upload_boundary.bulk.call_count == 1


@pytest.mark.parametrize("failure", ("corrupt", "bulk", "immutable"))
def test_failure_never_mints_success(generic, upload_boundary, monkeypatch, failure):
    args, cell, paths = generic
    fit_fixture(args, cell, paths, monkeypatch)
    if failure == "corrupt":
        (paths["fits"] / "percell_prompt_last_L00.json").write_text("{}")
    elif failure == "bulk":
        upload_boundary.bulk.side_effect = None
        upload_boundary.bulk.return_value = ""
    else:
        upload_boundary.verify.return_value = ["missing"]
    with pytest.raises(AssertionError):
        U.publish(RC, args)
    assert not (paths["cell"] / "phase_done/upload-fits.json").exists()


@pytest.mark.parametrize("suffix", ("uploads/upload-fits.json", "phase_done/upload-fits.json"))
def test_final_metadata_failure_is_repaired_on_retry(generic, upload_boundary, monkeypatch, suffix):
    args, cell, paths = generic
    fit_fixture(args, cell, paths, monkeypatch)
    original = upload_boundary.single.side_effect

    def fail_selected(local_path, repo_id, repo_type, path_in_repo, **kwargs):
        if path_in_repo.endswith(suffix):
            raise RuntimeError("injected final metadata failure")
        return original(local_path, repo_id, repo_type, path_in_repo, **kwargs)

    upload_boundary.single.side_effect = fail_selected
    with pytest.raises(RuntimeError, match="injected final metadata"):
        U.publish(RC, args)
    dst = f"{RC._cell_prefix(args, cell)}/{suffix}"
    assert dst not in upload_boundary.bytes
    assert not (paths["cell"] / "uploads/upload-fits-transport.json").exists()
    if suffix.startswith("phase_done"):
        assert RC._phase_complete(args, paths, "upload-fits")  # Local is insufficient.
    upload_boundary.single.side_effect = original
    assert U.publish(RC, args)["status"] == "complete"
    assert dst in upload_boundary.bytes
    assert (paths["cell"] / "uploads/upload-fits-transport.json").exists()


@pytest.mark.parametrize("changed", ("pilot", "results", "receipt"))
def test_resume_binds_extra_evidence(generic, upload_boundary, monkeypatch, changed):
    args, cell, paths = generic
    fit_fixture(args, cell, paths, monkeypatch)
    U.publish(RC, args)
    target = {
        "pilot": paths["fits"] / "fit_pilot.json",
        "results": paths["logs"] / f"issue-2588-{cell.key}-generic-{args.run_id}-results.json",
        "receipt": paths["cell"] / "uploads/upload-fits.json",
    }[changed]
    target.write_text("{}")
    with pytest.raises(AssertionError, match="changed fit transport"):
        U.publish(RC, args)
    assert upload_boundary.bulk.call_count == 1


def test_controller_batched_route_and_operation_budget(tmp_path):
    args = D.build_parser().parse_args(
        [
            "--mode",
            "fits",
            "--run-id",
            "qwen3-chat-v3",
            "--science-root",
            str(tmp_path),
            "--batch-fit-uploads",
        ]
    )
    steps = D.build_steps(args)
    uploads = [s for s in steps if s.phase == "upload-fits"]
    assert len(uploads) == 2
    assert all("issue2588_chat_upload_fits.py" in s.argv[2] for s in uploads)
    operations = D.local_upload_operations(tmp_path, "q3_8b_a", "upload-fits", batch_fits=True)
    assert operations["bulk_groups"] == 1 and operations["single_files"] == 0
    assert operations["retry_calls"] == 10
