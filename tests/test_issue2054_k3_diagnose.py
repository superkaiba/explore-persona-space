"""Durability and corruption checks for the diagnostic checkpoint boundary."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts
from scripts.issue2054_k3_diagnose import Checkpoints


@pytest.mark.parametrize("corrupt_commit", [None, 1, 2])
def test_receipts_require_verified_data_and_receipt_commit(tmp_path, monkeypatch, corrupt_commit):
    import huggingface_hub

    class Hub:
        def __init__(self):
            self.count = 0
            self.entries = {}

        def create_commit(self, *, operations, **kwargs):
            self.count += 1
            self.entries = {}
            for op in operations:
                data = Path(op.path_or_fileobj).read_bytes()
                self.entries[op.path_in_repo] = SimpleNamespace(
                    path=op.path_in_repo,
                    size=len(data),
                    lfs=None,
                    blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                    if self.count != corrupt_commit
                    else "corrupt",
                )
            return SimpleNamespace(oid=f"revision-{self.count}")

        def get_paths_info(self, repo, paths, **kwargs):
            return [self.entries[p] for p in paths]

    hub = Hub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: hub)
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for p in paths:
        p.write_text('{"real": 123}')
    if corrupt_commit is None:
        artifacts.seal_many(paths, tmp_path, "fp")
        assert all(k3.complete(p, "fp") for p in paths)
        assert hub.count == 2
    else:
        with pytest.raises(RuntimeError, match="hash mismatch"):
            artifacts.seal_many(paths, tmp_path, "fp")
        assert not any(p.with_suffix(".json.done.json").exists() for p in paths)


def test_checkpoint_resume_requires_receipt_and_rejects_changed_inputs(tmp_path, monkeypatch):
    (tmp_path / "packet.json").write_text("{}")
    (tmp_path / "references.npz").write_bytes(b"frozen reference")

    def seal(paths, root, fingerprint):
        for p in paths:
            p.with_suffix(p.suffix + ".done.json").write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "sha256": k3.sha(p),
                    }
                )
            )

    monkeypatch.setattr(artifacts, "seal_many", seal)
    store = Checkpoints(tmp_path, "model")
    vector = np.arange(24, dtype=np.float16).reshape(3, 8)
    store.write("unit", vector, {"method": "test", "error": [0, 0, 0]})
    assert store.read("unit") is None
    store.flush()
    resumed = Checkpoints(tmp_path, "model")
    np.testing.assert_array_equal(resumed.read("unit")[0], vector)
    (tmp_path / "packet.json").write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="fingerprint/content mismatch"):
        Checkpoints(tmp_path, "model").read("unit")


def test_fit_collection_waits_for_complete_verified_paired_results(tmp_path, monkeypatch):
    from scripts import issue2054_k3_fit as fit

    cell = "conversation_paired_stories_assistant__on_policy__chat__qwen2.5-7b"
    manifest = {"cells": [{"cell": cell}], "revision": "frozen"}
    (tmp_path / "coverage.json").write_text("[]")
    fingerprint = "approved-capture-policy" + k3.sha(fit.__file__)
    fp_fn = lambda *args: "approved-capture-policy"  # noqa: E731
    for count in (1, 3):
        path = tmp_path / "fits" / f"k{count}" / f"{cell}__all.json"
        path.parent.mkdir(parents=True)
        k3.atomic_json(
            path,
            {
                "cell": cell,
                "k_rollouts": count,
                "cohort": "all",
                "status": "complete",
                "folds": [{"fold": f} for f in range(5)],
            },
        )
        if count == 1:
            k3.atomic_json(
                path.with_suffix(".json.done.json"),
                {"fingerprint": fingerprint, "sha256": k3.sha(path)},
            )
    with pytest.raises(RuntimeError, match="unverified fit result"):
        fit.collect_results(tmp_path, manifest, fingerprint_fn=fp_fn)
    assert not (tmp_path / "results.json").exists()
    k3.atomic_json(
        path.with_suffix(".json.done.json"), {"fingerprint": fingerprint, "sha256": k3.sha(path)}
    )
    uploaded = []
    monkeypatch.setattr(k3, "seal", lambda *args: uploaded.append(args[0]))
    fit.collect_results(tmp_path, manifest, fingerprint_fn=fp_fn)
    assert uploaded == [tmp_path / "results.json"]
