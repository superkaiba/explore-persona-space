"""Identity, persistence and completion tests; CUDA/transport need production pilot."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
from huggingface_hub import HfApi

FILE = Path(__file__).resolve().parents[1] / "scripts" / "issue1901_training_k10_gpu.py"
SPEC = importlib.util.spec_from_file_location("training_k10_gpu", FILE)
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def rows(n):
    return [{"ci": i, "prompt_sha256": str(i)} for i in range(n)]


def recipe(width=2):
    return {
        **G.SETTINGS,
        "input_revision": "a" * 40,
        "input_manifest_sha256": "b" * 64,
        "ordered_ci_sha256": "c" * 64,
        "world_size": width,
    }


def raw(batch, config, seed=47, index=0):
    return {
        "recipe": config,
        "seed": seed,
        "chunk": index,
        "cap_hits": 0,
        "rows": [
            {
                **r,
                "text": "answer",
                "response": "answer",
                "token_ids": [8, 9],
                "prompt_token_ids": [4],
            }
            for r in batch
        ],
    }


def tensor(path, batch, config, seed, generation_sha):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        V=np.ones((len(batch), G.DIM), dtype=np.float16),
        n_ans=np.full(len(batch), 2),
        ci=np.array([r["ci"] for r in batch]),
        seed=seed,
        recipe=json.dumps(config),
        generation_sha=generation_sha,
    )


def test_uneven_chunks_and_all_allocated_widths():
    source = rows(19000)
    for width in (1, 2, 4):
        assignments = [G.partition(source, rank, width) for rank in range(width)]
        assert sorted(r["ci"] for assigned in assignments for r in assigned) == list(range(19000))
        for assigned in assignments:
            blocks = G.chunks(assigned)
            assert blocks[0] == assigned[:500]
            assert [r for block in blocks for r in block] == assigned
            assert all(0 < len(block) <= 500 for block in blocks)
    assert [len(x) for x in G.chunks(rows(1001))] == [500, 500, 1]
    assert len(G.chunks(assignments[0])[-1]) == 250


@pytest.mark.parametrize("mutation", ["recipe", "order", "seed", "cap", "empty_ids"])
def test_generation_rejects_stale_or_incomplete_chunks(mutation):
    source, config = rows(3), recipe()
    doc = raw(source, config)
    G.validate_generation(doc, source, config, 47, 0)
    if mutation == "recipe":
        doc["recipe"] = {**config, "max_tokens": 2048}
    elif mutation == "order":
        doc["rows"].reverse()
    elif mutation == "seed":
        doc["seed"] = 48
    elif mutation == "cap":
        doc["cap_hits"] = 1
    else:
        doc["rows"][0]["token_ids"] = []
    with pytest.raises(AssertionError):
        G.validate_generation(doc, source, config, 47, 0)


def test_capture_resume_requires_ids_recipe_and_generation_hash(tmp_path):
    source, config = rows(3), recipe()
    path = tmp_path / "chunk.npz"
    tensor(path, source, config, 47, "original")
    G.validate_capture(path, source, config, 47, "original")
    for other_rows, other_recipe, sha in (
        (source[::-1], config, "original"),
        (source, {**config, "layer": 18}, "original"),
        (source, config, "changed"),
    ):
        with pytest.raises(AssertionError):
            G.validate_capture(path, other_rows, other_recipe, 47, sha)


def test_capture_manifest_reconciles_contents_and_rejects_missing_or_duplicate(tmp_path):
    source, config = rows(7), recipe()
    for rank in range(2):
        batch = G.partition(source, rank, 2)
        for seed in G.SEEDS:
            root = tmp_path / "workers" / f"worker{rank:02d}"
            text = root / "raw_completions" / f"seed{seed}_chunk0000.json"
            G.write_json(text, raw(batch, config, seed))
            path = root / "analysis_tensors" / f"seed{seed}_chunk0000.npz"
            tensor(path, batch, config, seed, G.sha_file(text))
    result = G.completion_payload(tmp_path, source, config)
    assert result["realized_new_rows"] == result["expected_new_rows"] == 35
    assert len(result["files"]) == 10
    assert {x["seed"] for x in result["files"]} == set(G.SEEDS)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        G.completion_payload(tmp_path, source, config)


def test_large_text_shards_reconstruct_and_use_consumer_manifest(tmp_path):
    path = tmp_path / "raw.json"
    G.write_json(path, {"rows": [{"x": "a" * 30} for _ in range(20)]})
    parts = G.text_upload_paths(path, limit=200, target=180)
    manifest = G.read_json(parts[0])
    payload = b"".join((tmp_path / p).read_bytes() for p in manifest["parts"])
    assert payload == path.read_bytes()
    assert all(p.stat().st_size <= 180 for p in parts[1:])
    assert manifest["sha256"] == {p.name: G.sha_file(p) for p in parts[1:]}
    assert G.text_upload_paths(path, limit=10_000) == [path]


def test_upload_executes_real_hash_verification_and_rejects_omitted_path(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    G.write_json(path, {"finished": True})
    api = create_autospec(HfApi, instance=True)
    factory = create_autospec(HfApi, return_value=api)
    api.create_commit.return_value = SimpleNamespace(oid="d" * 40)
    payload = path.read_bytes()
    info = SimpleNamespace(
        path="prefix/result.json",
        size=len(payload),
        lfs=None,
        blob_id=hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    )
    api.get_paths_info.return_value = [info]
    monkeypatch.setattr(G, "HfApi", factory)
    receipt = G.upload_files(tmp_path, [path], "prefix")
    assert receipt["files"][0]["sha256"] == G.sha_file(path)
    api.create_commit.assert_called_once()
    api.get_paths_info.return_value = []
    with pytest.raises(AssertionError):
        G.upload_files(tmp_path, [path], "prefix")
    info.lfs = SimpleNamespace(sha256="wrong")
    api.get_paths_info.return_value = [info]
    with pytest.raises(AssertionError):
        G.upload_files(tmp_path, [path], "prefix")


def test_fresh_completion_requires_full_coverage_and_current_upload_hashes(tmp_path, monkeypatch):
    args = SimpleNamespace(out=tmp_path)
    config = recipe()
    path = tmp_path / "capture_manifest.json"
    manifest = {
        "expected_new_rows": 95000,
        "realized_new_rows": 95000,
        "recipe": config,
        "recipe_sha256": G.digest_json(config),
        "files": [{"path": "capture.npz", "sha256": "c" * 64}],
    }
    G.write_json(path, manifest)
    prefix = G.run_prefix(config)
    receipt = {
        "revision": "d" * 40,
        "files": [
            {"path": f"{prefix}/capture_manifest.json", "sha256": G.sha_file(path)},
            {"path": f"{prefix}/capture.npz", "sha256": "c" * 64},
        ],
    }
    sentinel = tmp_path / "fresh-attempt" / "done.json"
    monkeypatch.setenv("EPS_SENTINEL_PATH", str(sentinel))
    monkeypatch.setenv("EPS_LOG_DIR", str(tmp_path / "logs"))
    with pytest.raises(AssertionError):
        G.finish(args, {**manifest, "realized_new_rows": 94999}, receipt)
    with pytest.raises(KeyError):
        G.finish(args, manifest, {**receipt, "files": receipt["files"][:1]})
    assert not sentinel.exists()
    G.finish(args, manifest, receipt)
    assert G.read_json(sentinel)["phase"] == "done"
    envelopes = list((tmp_path / "logs").glob("*.json"))
    envelope = G.read_json(envelopes[0])
    assert envelope["sentinel_schema_version"] == 1 and envelope["kind"] == "epm:results"
    assert json.loads(envelope["note"])["new_rows"] == 95000
    assert G.read_json(sentinel)["fitting_status"] == "pending"


def test_stale_completion_is_archived_without_touching_other_attempts(tmp_path):
    stale = tmp_path / "sentinel.json"
    other = tmp_path / "other.json"
    G.write_json(stale, {"issue": 1901, "phase": "done"})
    G.write_json(other, {"issue": 999, "phase": "done"})
    G.clear_stale_completion(tmp_path, stale)
    assert not stale.exists() and other.exists()
    history = list((tmp_path / "resume_history").glob("*.json"))
    assert len(history) == 1 and G.read_json(history[0])["issue"] == 1901
    with pytest.raises(AssertionError):
        G.clear_stale_completion(tmp_path, other)
    assert other.exists()


def test_private_child_group_is_reaped():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        G.reap_process_group(process)
        assert process.poll() is not None and process.returncode < 0
        G.reap_process_group(process)  # already-dead group is the benign race.
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
