"""Bounded CPU handoff gates; real child processes and real capture consumer."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest


@pytest.fixture
def driver(monkeypatch):
    """Import the actual worktree helper."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("issue825_turn_k5_cpu")


def make_receipts(driver, root, text_path, tensor_path):
    """Create a small real-schema captured bank without any model/network execution."""
    import hashlib

    def fingerprint(value):
        """Match the production JSON fingerprint exactly."""
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    ids = [f"conv{i:03}" for i in range(24)]
    for model in ("instruct", "pretrained"):
        capture = root / "capture" / model
        chunk = capture / "chunk00000"
        chunk.mkdir(parents=True)
        generation = {
            "model": model,
            "n": 5,
            "turns": [1, 12],
            "selected_ids_sha256": fingerprint(ids),
            "panel_sha256": fingerprint(ids),
            "n_conversations": len(ids),
            "limit_conversations": 0,
        }
        config = {
            "layer": 19,
            "context_cos_min": 0.995,
            "generation_fingerprint": fingerprint(generation),
            "generation_chunks": [{"path": "chunk00000", "sha256": "fixture-generation-chunk"}],
        }
        driver.write_json(capture / "config.json", config)
        rows = [
            {
                "conv_id": cid,
                "turn": turn,
                "draw_id": draw,
                "context_prefix_sha256": fingerprint([cid, turn]),
            }
            for cid in ids
            for turn in (1, 12)
            for draw in range(5)
        ]
        with (chunk / "rows.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        features = np.ones((len(rows), 3584), dtype=np.float16)
        np.savez(
            chunk / "vectors.npz",
            conv_id=np.array([r["conv_id"] for r in rows]),
            turn=np.array([r["turn"] for r in rows]),
            draw_id=np.array([r["draw_id"] for r in rows]),
            context=features,
            answer=features,
        )
        driver.write_json(
            chunk / "manifest.json",
            {
                "kind": "capture",
                "fingerprint": fingerprint(config),
                "n_rows": len(rows),
                "files": {
                    name: {
                        "bytes": (chunk / name).stat().st_size,
                        "sha256": driver.digest(chunk / name),
                    }
                    for name in ("vectors.npz", "rows.jsonl")
                },
            },
        )
        driver.write_json(
            capture / "summary.json",
            {
                "status": "complete",
                "fingerprint": fingerprint(config),
                "config_sha256": driver.digest(capture / "config.json"),
                "generation_config": generation,
                "generation_fingerprint": fingerprint(generation),
                "n_captured_draws": len(rows),
                "n_expected_draws": len(rows),
                "n_excluded_draws": 0,
                "n_complete_conversations": len(ids),
                "complete_conversation_ids": ids,
                "exclusions": [],
                "hook_gates": [
                    {"status": "pass", "same_forward_equal": True, "layer": 19, "max_abs": 0}
                ],
                "chunks": [
                    {"path": "chunk00000", "sha256": driver.digest(chunk / "manifest.json")}
                ],
            },
        )
        driver.write_json(
            root / "gen" / model / "summary.json",
            {
                "status": "complete",
                "fingerprint": fingerprint(generation),
                "chunks": config["generation_chunks"],
                "counts": {"prompts": len(ids) * 2, "draws": len(ids) * 10},
            },
        )
    prefix = "issue825_turn_k5_pilot_20260914/gpu/production"
    for kind, receipt_path in (("text", text_path), ("tensors", tensor_path)):
        files = {
            f"{prefix}/{p.relative_to(root).as_posix()}": {
                "local": p.relative_to(root).as_posix(),
                "sha256": driver.digest(p),
                "size": p.stat().st_size,
            }
            for p in root.rglob("*")
            if p.is_file() and (p.suffix == ".npz") == (kind == "tensors")
        }
        driver.write_json(
            receipt_path,
            {
                "status": "verified",
                "repo": "superkaiba1/explore-persona-space-"
                + ("data" if kind == "text" else "overflow"),
                "repo_type": "dataset" if kind == "text" else "model",
                "revision": "a" * 40,
                "prefix": prefix,
                "files": files,
                "count": len(files),
                "bytes": sum(v["size"] for v in files.values()),
            },
        )


def test_stage_exact_receipts_real_loader_and_corrupt_local_refusal(driver, tmp_path, monkeypatch):
    """The staging body opens actual 3584-d capture files after exact content checks."""
    source, out = tmp_path / "source", tmp_path / "staged"
    text_path, tensor_path = tmp_path / "text.json", tmp_path / "tensor.json"
    make_receipts(driver, source, text_path, tensor_path)
    original = driver.stage_hub_file

    def copy_file(
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
        """Replace only the external Hub boundary with signature-conformant local bytes."""
        assert revision == "a" * 40 and repo_type in {"model", "dataset"}
        path = Path(target)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            relative = path_in_repo.removeprefix("issue825_turn_k5_pilot_20260914/gpu/production/")
            shutil.copyfile(source / relative, path)
        return path

    monkeypatch.setattr(driver, "stage_hub_file", create_autospec(original, side_effect=copy_file))
    result = driver.stage(text_path, tensor_path, out, expected_conversations=24)
    assert result["status"] == "verified" and result["coverage"]["n_conversations"] == 24
    assert result["expected_count"] == len(result["verified"]) == 12
    (out / "capture/instruct/chunk00000/vectors.npz").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="content mismatch"):
        driver.stage(text_path, tensor_path, out, expected_conversations=24)
    failed = json.loads((out / "stage_verification.json").read_text())
    assert failed["status"] == "failed" and "content mismatch" in failed["error"]
    assert failed["failed_at"] and failed["elapsed_seconds"] >= 0
    wrong = json.loads(text_path.read_text())
    wrong["prefix"] = "issue825_turn_k5_pilot_20260914/gpu/smoke"
    with pytest.raises(ValueError, match="only the completed full"):
        driver.selected_files([wrong, json.loads(tensor_path.read_text())])


@pytest.mark.parametrize("fold_seconds,expected_rc", [(0.01, 0), (1000, 7)])
def test_cpu_supervisor_first_checkpoint_pilot_and_completion(
    driver, tmp_path, fold_seconds, expected_rc
):
    """Measure the first complete fold, then either continue or stop its numerical child."""
    analysis = tmp_path / "analysis"
    store = tmp_path / "store"
    script = tmp_path / "worker.py"
    script.write_text(
        "import json,pathlib,sys,time,hashlib,numpy as np\n"
        "out=pathlib.Path(sys.argv[1]); (out/'folds').mkdir(parents=True)\n"
        "(out/'folds/instruct_fold0.json').write_text(json.dumps({'elapsed_seconds':float(sys.argv[2])}))\n"
        "print('fold checkpoint complete',flush=True)\ntime.sleep(.3)\n"
        "store=pathlib.Path(sys.argv[3]); store.mkdir(); artifacts=[]\n"
        "for i in range(38):\n"
        " p=store/f'{i}.npz'; np.savez(p, x=np.arange(3)); artifacts.append(dict(path=str(p),"
        "bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))\n"
        "(out/'results.json').write_text(json.dumps({'status':'complete','artifacts':artifacts}))\n"
    )
    result = driver.supervise(
        [sys.executable, str(script), str(analysis), str(fold_seconds), str(store)],
        tmp_path / "monitor",
        analysis,
        store,
        interval=0.01,
    )
    assert result == expected_rc
    state = json.loads((tmp_path / "monitor/status.json").read_text())
    assert state["completed_fold_checkpoints"] == 1
    assert state["sampled_peak_rss_bytes"] > 0
    assert state["pilot_gate"]["projected_fit_seconds"] == fold_seconds * 12
    if expected_rc == 0:
        assert state["status"] == "complete" and (analysis / "results.json").exists()
    else:
        assert (
            state["status"] == "venue_review_required" and not (analysis / "results.json").exists()
        )


def test_analysis_store_locks_prevent_second_writer(driver, tmp_path):
    """Distinct monitor paths cannot bypass a shared analysis/store lock."""
    analysis, store = tmp_path / "analysis", tmp_path / "store"
    with driver.execution_locks(analysis, store):
        with pytest.raises(BlockingIOError), driver.execution_locks(analysis, tmp_path / "other"):
            pytest.fail("analysis lock did not bind")
        with pytest.raises(BlockingIOError), driver.execution_locks(tmp_path / "other", store):
            pytest.fail("store lock did not bind")


def test_supervisor_error_reaps_worker_and_persists_failure(driver, tmp_path, monkeypatch):
    """A read error inside monitoring cannot orphan the numerical process."""
    original = driver.process_memory
    monkeypatch.setattr(
        driver,
        "process_memory",
        create_autospec(original, side_effect=OSError("fixture read error")),
    )
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    with pytest.raises(OSError, match="fixture read error"):
        driver.supervise(command, tmp_path / "monitor", tmp_path / "analysis", tmp_path / "store")
    state = json.loads((tmp_path / "monitor/status.json").read_text())
    assert state["status"] == "failed" and state["child_reaped"] is True
    assert not Path(f"/proc/{state['child_pid']}").exists()
