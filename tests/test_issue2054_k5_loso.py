"""Exercise real source-only fits, global fold exclusions and model separation."""

import copy
import hashlib
import json
import subprocess
import sys
import time
from unittest.mock import create_autospec

import numpy as np
import pytest

from scripts import issue2054_k5_loso as loso
from scripts.issue2054_ctx2ctx_fit import SharedEighRidge


def panel():
    """Build paired scaffold IDs with setting-specific affine maps and noise."""
    rng = np.random.default_rng(2054)
    out = {}
    for setting in range(6):
        x = rng.normal(size=(100, 8)) + setting * 0.2
        y = x @ rng.normal(size=(8, 8)) + setting + rng.normal(size=(100, 8)) * 0.3
        out[f"setting_{setting}__on_policy__chat__qwen2.5-7b"] = {
            "x": x,
            "y": y,
            "ids": [f"conv_{i}" for i in range(100)],
            "membership": np.arange(100) % 5,
        }
    return out


def test_moment_transfer_matches_materialized_source_only_solver():
    data = panel()
    bank = loso.moments(data, "cpu")
    for excluded in list(data)[:3]:
        prediction, bias, info, audit = loso.transfer(data, bank, excluded, 0)
        source_x = np.concatenate(
            [p["x"][p["membership"] != 0] for c, p in data.items() if c != excluded]
        )
        source_y = np.concatenate(
            [p["y"][p["membership"] != 0] for c, p in data.items() if c != excluded]
        )
        test_x = data[excluded]["x"][data[excluded]["membership"] == 0]
        reference, ref_info = SharedEighRidge(source_x, test_x, device="cpu").fit_predict(source_y)
        np.testing.assert_allclose(prediction, reference, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(bias, (source_y - source_x).mean(0), atol=1e-12)
        assert info["best_lambda"] == ref_info["best_lambda"]
        assert audit["test_conversation_overlap"] == 0
        assert audit["n_train"] == 400
        assert excluded not in audit["source_settings"]


def test_excluded_setting_and_evaluation_fold_labels_cannot_affect_transfer():
    data = panel()
    excluded = next(iter(data))
    before = loso.transfer(data, loso.moments(data, "cpu"), excluded, 2)
    altered = copy.deepcopy(data)
    altered[excluded]["y"] = altered[excluded]["y"] * -100 + 9000
    for c, p in altered.items():
        if c != excluded:
            p["y"][p["membership"] == 2] = -2000
    after = loso.transfer(altered, loso.moments(altered, "cpu"), excluded, 2)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_global_fold_mismatch_and_cross_model_source_fail_loudly():
    data = panel()
    excluded, source = list(data)[:2]
    data[source]["membership"][0] = 1
    with pytest.raises(RuntimeError, match="leaked"):
        loso.source_audit(data, excluded, 0)
    data = panel()
    data[source + "-instruct"] = data.pop(source)
    with pytest.raises(RuntimeError, match="cross-model"):
        loso.source_audit(data, excluded, 0)


def test_all_model_and_fold_checkpoint_paths_are_unique(tmp_path):
    paths = [p for c in panel() for f in range(5) for p in loso.unit_paths(tmp_path, c, f)]
    assert len(paths) == len(set(paths)) == 60
    assert all("qwen2.5-7b__fold" in str(p) for p in paths)


def test_complete_file_backed_fit_compare_collect_and_resume(tmp_path, monkeypatch):
    """Use real loaders/solvers/scores; replace only the pretested Hub uploader."""
    root, out = tmp_path / "parent", tmp_path / "loso"
    root.mkdir()
    out.mkdir()
    cells = []
    for model in loso.k3.MODEL_REVISIONS:
        for character in ("dana", "helios", "wren", "vex"):
            cells.append(
                {"cell": f"char_{character}__on_policy__attrib_quoted__{model}", "raw": "unused"}
            )
        for form in ("chat", "bare_text"):
            cells.append(
                {
                    "cell": f"conversation_paired_stories_assistant__on_policy__{form}__{model}",
                    "raw": "unused",
                }
            )
    manifest = {"cells": cells, "fold_map": "folds.json"}
    loso.k3.atomic_json(
        root / "folds.json",
        {
            "fold_of": {f"conv_{i}": i % 5 for i in range(20_000)},
            "k": 5,
            "seed": 137,
            "variants": [f"variant{i}" for i in range(5)],
        },
    )

    def seal_locally(paths, destination, fingerprint):
        for path in paths:
            loso.k3.atomic_json(
                path.with_suffix(path.suffix + ".done.json"),
                {
                    "fingerprint": fingerprint,
                    "sha256": loso.k3.sha(path),
                    "size": path.stat().st_size,
                    "path": (
                        f"{loso.k5.OUTPUT_PREFIX}/{destination.name}/"
                        f"{path.relative_to(destination)}"
                    ),
                },
            )

    monkeypatch.setattr(
        loso.artifacts,
        "seal_many",
        create_autospec(loso.artifacts.seal_many, side_effect=seal_locally),
    )
    rng = np.random.default_rng(2054)
    for record in cells:
        cell = record["cell"]
        x = rng.normal(size=(100, 8))
        y = x @ rng.normal(size=(8, 8)) + rng.normal(size=(100, 8)) * 0.2
        for count in loso.k5.COUNTS:
            path = root / f"k{count}" / f"{cell}.npz"
            loso.scoring.save_npz(
                path,
                {
                    "v_C": x,
                    "v_A": y,
                    "conv_id": np.array([f"conv_{i}" for i in range(100)]),
                    "cap_mask": np.zeros((100, 5), dtype=bool),
                },
            )
            seal_locally([path], root, loso.k5.fingerprint(manifest, cell))
    for model in loso.k3.MODEL_REVISIONS:
        loso.matched.fit_model(root, manifest, model, 0, 1, device="cpu")
        loso.fit_model(root, out, manifest, model, range(5), device="cpu")
    results = loso.collect(root, out, manifest)
    assert len(results) == 12
    assert all(len(r["folds"]) == 5 for r in results)
    assert all(np.isfinite(list(r["r2_mean"].values())).all() for r in results)
    assert all(
        r["folds"][0]["cohorts"]["all"]["metrics"]["leave_one_setting_out"]["retrieval_pool"] == 20
        for r in results
    )
    before = {str(p): p.stat().st_mtime_ns for p in out.rglob("*.json")}
    for model in loso.k3.MODEL_REVISIONS:
        loso.fit_model(root, out, manifest, model, range(5), device="cpu")
    assert before == {str(p): p.stat().st_mtime_ns for p in out.rglob("*.json")}
    # Altering a reference must invalidate the unit's cached evidence.
    cell = results[0]["cell"]
    path = loso.matched.fold_path(root, cell, "all", 5, 0)
    value = json.loads(path.read_text())
    value["n_test"] += 1
    loso.k3.atomic_json(path, value)
    seal_locally([path], root, loso.matched.fit_fp(manifest, cell))
    with pytest.raises(RuntimeError, match="fingerprint/content mismatch"):
        loso.fit_model(root, out, manifest, cell.split("__")[-1], [0], device="cpu")
    with pytest.raises(RuntimeError, match="population differs"):
        loso.reference(root, manifest, cell, "all", 0, 20)


def test_parent_upload_audit_checks_real_files_and_provider_shaped_entries(tmp_path, monkeypatch):
    """Exercise the full verifier; only the authenticated Hub boundary is fake."""
    import huggingface_hub
    from huggingface_hub.hf_api import DatasetInfo, RepoFile

    root = tmp_path / "production_v1"
    root.mkdir()
    path = root / "data.json"
    loso.k3.atomic_json(path, {"data": [1, 2, 3]})
    prefix = f"{loso.k5.OUTPUT_PREFIX}/{root.name}"
    loso.k3.atomic_json(
        root / "data.json.done.json",
        {
            "path": f"{prefix}/data.json",
            "sha256": loso.k3.sha(path),
            "fingerprint": "fixture",
        },
    )
    entries = []
    for p in sorted(root.iterdir()):
        data = p.read_bytes()
        entries.append(
            RepoFile(
                path=f"{prefix}/{p.name}",
                size=len(data),
                oid=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            )
        )
    api = create_autospec(huggingface_hub.HfApi, instance=True)
    api.repo_info.return_value = DatasetInfo(id=loso.k3.HF_REPO, sha="pinned-fixture-revision")
    api.list_repo_tree.return_value = entries
    monkeypatch.setattr(
        huggingface_hub, "HfApi", create_autospec(huggingface_hub.HfApi, return_value=api)
    )
    report = loso.verify_parent_uploads(root)
    assert report["status"] == "pass" and report["files"] == 2
    assert report["revision"] == "pinned-fixture-revision"
    assert api.list_repo_tree.call_args.kwargs["revision"] == report["revision"]
    entries[0].blob_id = "incorrect"
    with pytest.raises(RuntimeError, match="remote hash mismatch"):
        loso.verify_parent_uploads(root)
    api.list_repo_tree.return_value = entries[1:]
    with pytest.raises(RuntimeError, match="upload set differs"):
        loso.verify_parent_uploads(root)


def test_waiter_checks_live_process_then_real_parent_completion(tmp_path, monkeypatch):
    """Exercise the actual PID/source/start identity and completion checks."""
    report = tmp_path / "job_complete.json"
    loso.k3.atomic_json(
        report, {"source_sha": loso.PARENT_SOURCE, "primary_panels": 36, "status": "complete"}
    )
    loso.k3.atomic_json(
        tmp_path / "job_complete.json.done.json",
        {
            "fingerprint": loso.k3.sha(loso.k5.__file__),
            "sha256": loso.k3.sha(report),
        },
    )
    real_sleep = time.sleep
    monkeypatch.setattr(loso.time, "sleep", lambda _: real_sleep(0.01))
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.15)",
            "issue2054_k5.py",
            "--stage",
            "job",
            "--source-sha",
            loso.PARENT_SOURCE,
        ]
    ) as child:
        loso.wait_for_parent(child.pid, tmp_path)
        assert child.wait(timeout=1) == 0
    with pytest.raises(RuntimeError, match="absent"):
        loso.wait_for_parent(child.pid, tmp_path)
    loso.k3.atomic_json(report, {"status": "failed"})
    with pytest.raises(RuntimeError, match="fingerprint/content mismatch"):
        loso.verify_parent(tmp_path)
