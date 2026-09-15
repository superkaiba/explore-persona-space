"""Protect the K5 extension's source identity, coverage, and worker ownership."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from scripts import issue2054_k3 as k3
from scripts import issue2054_k5 as k5
from scripts import issue2054_k5_assistant_story as story


def manifest_fixture():
    """Construct the full 56-cell axis lattice, not a shortened mutable panel."""
    cells = []
    for model in k3.MODEL_REVISIONS:
        for variant in (
            "char_helios",
            "char_wren",
            "char_dana",
            "char_vex",
            "conversation_paired_stories_assistant",
        ):
            forms = ("attrib_quoted", "bare_label")
            if variant == "conversation_paired_stories_assistant":
                forms += ("chat", "bare_text")
            for form in forms:
                for condition in ("on_policy", "inserted"):
                    cells.append(
                        {
                            "cell": f"{variant}__{condition}__{form}__{model}",
                            "n": 8000,
                            "cap": k3.cap_for(f"{variant}__{condition}__{form}__{model}"),
                            "raw": f"raw/{variant}/{condition}/{form}/{model}.jsonl",
                        }
                    )
        for variant in ("char_helios", "char_wren", "char_dana", "char_vex"):
            cells.append({"cell": f"{variant}__cell_c__chat__{model}", "n": 8000})
    assert len(cells) == 56
    return {"cells": cells}


def test_explicit_selection_keeps_full_manifest_and_default_panel():
    manifest = manifest_fixture()
    before = deepcopy(manifest)
    assert len(k5.selected(manifest)) == 12
    assert {r["cell"] for r in story.selection(manifest)} == set(story.ASSISTANT_STORY_CELLS)
    assert manifest == before
    assert len(manifest["cells"]) == 56


@pytest.mark.parametrize("cells", [[], ["missing"], [story.ASSISTANT_STORY_CELLS[0]] * 2])
def test_explicit_selection_rejects_empty_missing_or_duplicate(cells):
    with pytest.raises(ValueError):
        k5.selected(manifest_fixture(), cells=cells)


def test_explicit_selection_rejects_fixed_answers_and_changed_story_population():
    manifest = manifest_fixture()
    fixed = next(r["cell"] for r in manifest["cells"] if "__inserted__" in r["cell"])
    with pytest.raises(ValueError, match="on-policy"):
        k5.selected(manifest, cells=[fixed])
    story.selection(manifest)[0]["n"] = 7999
    with pytest.raises(ValueError, match="population"):
        story.selection(manifest)


def test_fingerprint_binds_selection_without_mutating_k3_recipe():
    manifest = manifest_fixture()
    before = deepcopy(manifest)
    cell = story.ASSISTANT_STORY_CELLS[0]
    original_parent = k3.fingerprint(manifest, cell, False)
    default = k5.fingerprint(manifest, cell)
    selected = k5.fingerprint(manifest, cell, cells=story.ASSISTANT_STORY_CELLS)
    assert default != selected
    assert selected == k5.fingerprint(
        manifest, cell, cells=tuple(reversed(story.ASSISTANT_STORY_CELLS))
    )
    assert k3.fingerprint(manifest, cell, False) == original_parent
    assert before == manifest


def test_real_k5_average_preserves_all_draws_and_cap_flags():
    rng = np.random.default_rng(2054)
    values = rng.normal(size=(4, 5, 3584)).astype(np.float16)
    ids = np.array(["a", "b", "c", "d"])
    old = {
        "conv_id": ids,
        "v_A_0": values[:, 0],
        "v_A_12": values[:, 1:3],
        "valid_draws_12": np.ones((4, 2), bool),
        "cap_mask": np.zeros((4, 3), bool),
    }
    new = {
        "conv_id": ids.copy(),
        "v_A_34": values[:, 3:],
        "valid_draws_34": np.ones((4, 2), bool),
        "cap_mask_34": np.zeros((4, 2), bool),
    }
    old["cap_mask"][0, 0] = True
    new["cap_mask_34"][1, 1] = True
    new["valid_draws_34"][2, 0] = False
    targets, keep, caps = k5.average_targets(old, new)
    np.testing.assert_array_equal(keep, [True, True, False, True])
    np.testing.assert_allclose(
        targets[5], values[keep].astype(np.float32).mean(1), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(caps.any(1), [True, True, False, False])


def test_worker_interface_binds_exact_checkpoint_and_preserves_allocated_gpu_ids(tmp_path):
    model = next(iter(k3.MODEL_REVISIONS))
    command = story.worker_command(tmp_path, "sha", model, "capture", True)
    assert command[0] == sys.executable
    assert command[command.index("--model") + 1] == model
    assert command[command.index("--source-sha") + 1] == "sha"
    assert command[-1] == "--first-chunks"
    assert "--first-chunks" not in story.worker_command(tmp_path, "sha", model, "capture", False)
    assert story.allocated_devices({"CUDA_VISIBLE_DEVICES": "5,7", "SLURM_JOB_ID": "1"}) == [
        "5",
        "7",
    ]
    with pytest.raises(RuntimeError, match="allocated GPU IDs"):
        story.allocated_devices({"SLURM_JOB_ID": "1"})


def test_coverage_keeps_cap_and_empty_gates_separate():
    coverage = [
        {
            "cell": cell,
            "original_rows": 256,
            "complete_five_rows": 250,
            "cap_counts_each_draw": [0, 1, 0, 0, 0],
        }
        for cell in story.ASSISTANT_STORY_CELLS
    ]
    story.validate_coverage(coverage, first_chunks=True)
    coverage[0]["cap_counts_each_draw"][4] = 6
    with pytest.raises(RuntimeError, match="cap-hit"):
        story.validate_coverage(coverage, first_chunks=True)
    coverage[0]["cap_counts_each_draw"][4] = 0
    coverage[1]["complete_five_rows"] = 200
    with pytest.raises(RuntimeError, match="empty"):
        story.validate_coverage(coverage, first_chunks=True)


def test_failed_real_worker_drains_owned_peer_without_touching_unrelated_process(
    tmp_path, monkeypatch
):
    first = next(iter(k3.MODEL_REVISIONS))

    def command(root, source_sha, model, stage, first_chunks):
        if model == first:
            code = "import time; time.sleep(0.3); raise SystemExit(9)"
        else:
            code = "import time; time.sleep(90)"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(
        story, "worker_command", create_autospec(story.worker_command, side_effect=command)
    )
    monkeypatch.setattr(story.artifacts, "seal_many", create_autospec(story.artifacts.seal_many))
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"])
    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            story.run_workers(tmp_path, "sha", ["5", "7"], first_chunks=True)
        statuses = [json.loads(p.read_text()) for p in (tmp_path / "logs").glob("*.status.json")]
        assert len(statuses) == 2
        assert all(row["exit_code"] != 0 and row["finished"] >= row["started"] for row in statuses)
        for row in statuses:
            assert not Path(f"/proc/{row['pid']}").exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_cli_help_uses_real_import_surface():
    result = subprocess.run(
        [sys.executable, str(Path(story.__file__)), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert "--analysis" in result.stdout
    assert "aggregate" in result.stdout


def test_resume_restores_exact_receipt_revision_and_rejects_local_corruption(tmp_path, monkeypatch):
    import huggingface_hub

    from explore_persona_space.orchestrate import hub

    root = tmp_path / "production_v1"
    prefix = f"{story.OUTPUT_PREFIX}/production_v1"
    remote = prefix + "/raw/cell/chunk_00000.json"
    body = tmp_path / "fixture.json"
    body.write_text('{"rows": 512}\n')
    receipt = {
        "path": remote,
        "revision": "a" * 40,
        "sha256": k3.sha(body),
        "size": body.stat().st_size,
        "fingerprint": "fixture-recipe",
    }
    payloads = {remote: body.read_bytes(), remote + ".done.json": json.dumps(receipt).encode()}
    api = create_autospec(huggingface_hub.HfApi, instance=True)
    api.list_repo_tree.return_value = [SimpleNamespace(path=remote + ".done.json", size=200)]
    monkeypatch.setattr(
        huggingface_hub, "HfApi", create_autospec(huggingface_hub.HfApi, return_value=api)
    )
    downloaded = []

    def stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        target = Path(target)
        downloaded.append((path_in_repo, revision))
        if not target.exists() or overwrite:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payloads[path_in_repo])
        return target

    monkeypatch.setattr(
        hub, "stage_hub_file", create_autospec(hub.stage_hub_file, side_effect=stage)
    )
    report = story.restore_outputs(root, "b" * 40)
    assert report["restored"] == 1
    assert (remote, "a" * 40) in downloaded
    assert (remote + ".done.json", "b" * 40) in downloaded
    restored = root / "raw/cell/chunk_00000.json"
    assert k3.complete(restored, "fixture-recipe")
    restored.write_text("corrupt")
    with pytest.raises(RuntimeError, match="content mismatch"):
        story.restore_outputs(root, "b" * 40)


def test_pilot_projection_reads_actual_generation_and_capture_timings(tmp_path):
    workers, coverage = [], []
    for cell in story.ASSISTANT_STORY_CELLS:
        model = cell.split("__")[-1]
        path = tmp_path / "raw" / cell / "chunk_00000.timing.json"
        k3.atomic_json(path, {"contexts": 256, "answers": 512, "seconds": 2.0})
        log = tmp_path / "logs" / f"capture_{model}.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(f"[phase=capture] {cell} offset=0 elapsed=3.50s peak_gb=17.00\n")
        for stage in ("generate", "capture"):
            workers.append(
                {
                    "model": model,
                    "stage": stage,
                    "seconds": 10,
                    "log": str(log.relative_to(tmp_path)),
                }
            )
        coverage.append(
            {
                "cell": cell,
                "original_rows": 256,
                "complete_five_rows": 256,
                "cap_counts_each_draw": [0] * 5,
            }
        )
    result = story.pilot_report(tmp_path, manifest_fixture(), coverage, workers)
    assert result["status"] == "pass"
    assert result["projected_generation_compute_gpu_hours"] == pytest.approx(128 / 3600)
    assert all(r["projected_capture_chunk_seconds"] == 112 for r in result["cells"])


def test_analysis_completion_requires_all_declared_receipts_and_hashes(tmp_path):
    from scripts import issue2054_k5_assistant_story_analysis as analysis

    root = tmp_path / "production_v1"
    manifest = manifest_fixture()
    generation = {
        "source_sha": "c" * 40,
        "manifest_sha256": story.MANIFEST_SHA256,
        "inventory_sha256": "d" * 64,
        "selected_cells": list(story.ASSISTANT_STORY_CELLS),
        "parent_revision": k5.PARENT_REV,
    }
    fp = analysis.fingerprint(manifest, generation)

    def local_receipt(path):
        record = {
            "path": f"{story.OUTPUT_PREFIX}/{root.name}/{path.relative_to(root)}",
            "revision": "e" * 40,
            "sha256": k3.sha(path),
            "size": path.stat().st_size,
            "fingerprint": fp,
        }
        done = path.with_suffix(path.suffix + ".done.json")
        k3.atomic_json(done, record)
        return record, done

    result = root / "analysis/results.json"
    k3.atomic_json(result, {"status": "complete"})
    result_receipt, result_done = local_receipt(result)
    inventory = root / "analysis/inventory.json"
    k3.atomic_json(
        inventory,
        {
            "status": "complete",
            "analysis_fingerprint": fp,
            "source_sha": generation["source_sha"],
            "files": [
                {
                    "path": "analysis/results.json",
                    "sha256": k3.sha(result),
                    "size": result.stat().st_size,
                    "receipt": result_receipt,
                    "receipt_path": "analysis/results.json.done.json",
                    "receipt_sha256": k3.sha(result_done),
                }
            ],
        },
    )
    local_receipt(inventory)
    complete = root / "analysis/analysis_complete.json"
    report = {
        "status": "complete",
        "source_sha": generation["source_sha"],
        "analysis_fingerprint": fp,
        "results_path": "analysis/results.json",
        "results_sha256": k3.sha(result),
        "inventory_path": "analysis/inventory.json",
        "inventory_sha256": k3.sha(inventory),
    }
    k3.atomic_json(complete, report)
    local_receipt(complete)
    assert story.verify_analysis(root, manifest, generation) == report
    result.write_text("corrupt")
    with pytest.raises(RuntimeError, match="checkpoint fingerprint/content mismatch"):
        story.verify_analysis(root, manifest, generation)
