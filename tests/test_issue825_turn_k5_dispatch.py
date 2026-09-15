"""Launch gates for source coverage, honest timing, persistence, and terminal state."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def driver(monkeypatch):
    """Import the actual dispatch helpers without starting a workload."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("issue825_turn_k5_dispatch")


def test_smoke_has_eight_length_quantiles_from_each_source(driver):
    """A very imbalanced full panel still smokes both sources and each length extreme."""
    rows = [
        {"conv_id": f"{source}_{i}", "source": source, "prompt_lengths": {"turn12": i}}
        for source, n in (("lmsys", 80), ("wildchat", 12))
        for i in range(n)
    ]
    chosen = driver.smoke_panel_rows(rows)
    assert len({r["conv_id"] for r in chosen}) == 16
    for source, end in (("lmsys", 79), ("wildchat", 11)):
        group = [r for r in chosen if r["source"] == source]
        assert len(group) == 8
        assert min(r["prompt_lengths"]["turn12"] for r in group) == 0
        assert max(r["prompt_lengths"]["turn12"] for r in group) == end
    with pytest.raises(ValueError, match="two sources"):
        driver.smoke_panel_rows(rows[:10])


def test_projection_scales_processing_not_loading_and_respects_gpu_width(driver):
    """Expensive model initialization is paid once, and one-GPU wall sums both lanes."""
    smoke = {
        model: {
            phase: {
                "wall_s": 108,
                "summary": {
                    "elapsed_s": 105,
                    "model_load_s": 100,
                    "processing_s": 2,
                },
            }
            for phase in ("gen", "capture")
        }
        for model in ("instruct", "pretrained")
    }
    parallel = driver.project_smoke(smoke, 1000, 16, 2)
    serial = driver.project_smoke(smoke, 1000, 16, 1)
    phase_seconds = 106 + 2 * 1000 / 16
    assert parallel["phases"]["instruct"]["gen"]["projected_s"] == phase_seconds
    assert parallel["projected_wall_h"] == pytest.approx(2 * phase_seconds / 3600)
    assert serial["projected_wall_h"] == pytest.approx(4 * phase_seconds / 3600)
    assert parallel["projected_allocated_gpu_h"] == pytest.approx(
        serial["projected_allocated_gpu_h"]
    )
    smoke["instruct"]["gen"]["summary"]["processing_s"] = 0
    with pytest.raises(ValueError, match="actual processing"):
        driver.project_smoke(smoke, 1000, 16, 2)


def test_explicit_empty_gpu_allocation_does_not_escape_to_whole_machine(driver, monkeypatch):
    """An explicitly disabled visibility list cannot borrow unrelated machine GPUs."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="no allocated"):
        driver.gpu_ids()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3, 7")
    assert driver.gpu_ids() == ["3", "7"]


def make_work(driver, tmp_path):
    """Prepare produced outputs plus an oversized staged input that must never be reuploaded."""
    work, receipts = tmp_path / "work", tmp_path / "receipts"
    for name in ("smoke", "production"):
        folder = work / name
        folder.mkdir(parents=True)
        (folder / "output.jsonl").write_text('{"draw": 0}\n')
        (folder / "vectors.npz").write_bytes(b"fixture")
    (work / "inputs").mkdir()
    with (work / "inputs/panel.jsonl").open("wb") as handle:
        handle.truncate(10_000_000)
    driver.atomic_json(work / "metadata/launch.json", {"status": "launched"})
    return work, receipts


@pytest.mark.parametrize("fail_tensor", [False, True])
def test_archives_only_produced_trees_and_salvages_other_trees(
    driver, tmp_path, monkeypatch, fail_tensor
):
    """A failed tensor upload still preserves other outputs, failure metadata, and receipts."""
    work, receipts = make_work(driver, tmp_path)
    calls = []

    def uploader(root, prefix, kind, receipt):
        calls.append((root, prefix, kind))
        assert root != work and not root.is_relative_to(work / "inputs")
        if fail_tensor and root.name == "smoke" and kind == "tensors":
            raise RuntimeError("fixture transfer failure")
        driver.atomic_json(receipt, {"status": "verified", "prefix": prefix, "count": 1})

    monkeypatch.setattr(driver, "upload", uploader)
    if fail_tensor:
        with pytest.raises(RuntimeError, match="persistence failed"):
            driver.archive_outputs(work, receipts, require_complete=False)
        assert json.loads((receipts / "archive_attempt.json").read_text())["status"] == "failed"
        assert (work / "metadata/archive_failure.json").exists()
    else:
        result = driver.archive_outputs(work, receipts, require_complete=True)
        assert len(result["archives"]) == 5
    assert any(root.name == "production" and kind == "tensors" for root, _, kind in calls)
    assert any(root.name == "metadata" for root, _, _ in calls)
    assert any(root == receipts for root, _, _ in calls)


def test_completion_requires_archives_and_tolerates_processed_envelope(
    driver, tmp_path, monkeypatch
):
    """Real terminal writers stay behind archive verification and survive the drain rename."""
    out = tmp_path / "output"
    logs = tmp_path / "logs"
    monkeypatch.setenv("EPM825_SENTINEL_DIR", str(logs))
    monkeypatch.setenv("EPS_SENTINEL_PATH", str(tmp_path / "backend.json"))
    monkeypatch.setenv("EPS_DELIVERABLES_OK_PATH", str(tmp_path / "deliverables.json"))
    archived = {
        "archives": {
            name: {"status": "verified"}
            for name in (
                "smoke_text",
                "smoke_tensors",
                "production_text",
                "production_tensors",
                "metadata_text",
            )
        },
        "receipts_archive": {"status": "verified", "revision": "pinned"},
    }
    archived["archives"]["production_tensors"]["status"] = "failed"
    with pytest.raises(RuntimeError, match="verified archives"):
        driver.finish(out, archived, 100, 2)
    assert not (out / "gpu_complete.json").exists()
    archived["archives"]["production_tensors"]["status"] = "verified"
    writer = driver.atomic_json

    def drain_after_write(path, value):
        writer(path, value)
        if path.parent == logs:
            path.rename(path.with_name(path.name + ".processed"))

    monkeypatch.setattr(driver, "atomic_json", drain_after_write)
    driver.finish(out, archived, 100, 2)
    assert json.loads((out / "gpu_complete.json").read_text())["cpu_analysis_pending"] is True
    assert json.loads((tmp_path / "backend.json").read_text())["issue"] == 825
    envelope = json.loads(next(logs.glob("*.processed")).read_text())
    assert envelope["sentinel_schema_version"] == 1
    assert envelope["kind"] == "epm:progress"
    assert envelope["gate"] == "phase"
    assert (tmp_path / "deliverables.json").exists()


def test_smoke_coverage_requires_shared_complete_conversations(driver):
    """Two individually successful captures cannot pass with disjoint complete panels."""
    panel = [{"conv_id": str(i)} for i in range(16)]
    results = {}
    for model, keep in (("instruct", range(8)), ("pretrained", range(8, 16))):
        results[model] = {
            "gen": {"summary": {"status": "complete", "counts": {"draws": 160}}},
            "capture": {
                "summary": {
                    "status": "complete",
                    "n_selected_conversations": 16,
                    "n_expected_draws": 160,
                    "n_captured_draws": 80,
                    "n_excluded_draws": 80,
                    "n_complete_conversations": 8,
                    "complete_conversation_ids": [str(i) for i in keep],
                }
            },
        }
    with pytest.raises(RuntimeError, match="common complete"):
        driver.validate_model_coverage(results, panel, minimum=6)
    results["pretrained"]["capture"]["summary"]["complete_conversation_ids"] = [
        str(i) for i in range(8)
    ]
    assert driver.validate_model_coverage(results, panel, minimum=6)["common_n"] == 8
    results["pretrained"]["capture"]["summary"]["n_captured_draws"] = 0
    with pytest.raises(RuntimeError, match="inconsistent"):
        driver.validate_model_coverage(results, panel, minimum=6)


def test_raw_generation_is_verified_before_capture(driver, tmp_path, monkeypatch):
    """A failed raw-answer archive blocks capture instead of risking the only answer copy."""
    events = []

    def phase(root, panel, model, gpu, phase_name):
        events.append((model, phase_name))
        return {"phase": phase_name}

    def upload(root, prefix, kind, receipt):
        model = root.name
        events.append((model, "upload"))
        assert receipt.parent != root
        driver.atomic_json(receipt, {"status": "verified"})

    monkeypatch.setattr(driver, "run_phase", phase)
    monkeypatch.setattr(driver, "upload", upload)
    driver.run_models(tmp_path / "smoke", tmp_path / "panel.jsonl", ["0"])
    for model in ("instruct", "pretrained"):
        assert [phase for m, phase in events if m == model] == ["gen", "upload", "capture"]
    events.clear()

    def failed_upload(*args):
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(driver, "upload", failed_upload)
    with pytest.raises(RuntimeError, match="archive unavailable"):
        driver.run_models(tmp_path / "smoke", tmp_path / "panel.jsonl", ["0"])
    assert events == [("instruct", "gen")]
