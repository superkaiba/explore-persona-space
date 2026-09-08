"""Exercise real rank staging with signature-bound remote/disk-capacity fixtures."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import huggingface_hub as hf
import pytest
from huggingface_hub.hf_api import RepoFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_stage_rank as stage

from explore_persona_space.orchestrate import preflight


@pytest.fixture
def remote(monkeypatch):
    """Tiny fixture bytes test transport, never substitute for experiment tensors."""
    rank = stage.rank
    prefix = f"{rank.PANEL_PREFIX}/generic/qwen3-chat-v3/q3_8b/nothink"
    identity = {
        "surface": "generic",
        "run_id": "qwen3-chat-v3",
        "cell": "q3_8b_a",
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": rank.MODEL_REVISION,
        "manifest_revision": rank.MANIFEST_REVISION,
        "source_sha": "0a06988756f6b407e4d1562b0ac8d88af52dde2d",
        "smoke": False,
        "cap_profile": "long",
        "layer_set": "swept",
    }
    fit = {
        "identity": identity,
        "input_position": "prompt_last",
        "layer_star": 2,
        "layers": {
            str(layer): {
                "d": 4096,
                "n": {"tr": 2, "val": 2, "te": 2},
                "knn_val": {"ridge": {"cosine": {"acc_at_k": {"1": 1.0 if layer == 2 else 0.0}}}},
            }
            for layer in (*range(0, 36, 2), 35)
        },
    }
    payload = {
        f"{prefix}/run_identity.json": json.dumps(identity).encode(),
        f"{prefix}/fits/fits_prompt_last.json": json.dumps(fit).encode(),
    }
    for split in rank.SPLITS:
        payload[f"{prefix}/analysis_tensors/capture/{split}/rows.json"] = json.dumps(
            {"rows": [{"row_id": f"{split}_1"}, {"row_id": f"{split}_2"}]}
        ).encode()
        payload[f"{prefix}/analysis_tensors/capture/{split}/L02/shard000.npz"] = (
            b"fixture transport bytes; not a scientific tensor"
        )
    calls = []

    def info(repo_id, paths, *, expand=False, revision=None, repo_type=None, token=None):
        assert repo_id == rank.HF_REPO and revision == "a" * 40 and repo_type == "dataset"
        records = []
        for p in paths:
            if p not in payload:
                continue
            data = payload[p]
            lfs = (
                {"size": len(data), "oid": hashlib.sha256(data).hexdigest(), "pointerSize": 130}
                if p.endswith(".npz")
                else None
            )
            records.append(
                RepoFile(
                    path=p,
                    size=len(data),
                    oid=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
                    lfs=lfs,
                )
            )
        return records

    def download(repo_id, filename, **kwargs):
        assert repo_id == rank.HF_REPO and kwargs["revision"] == "a" * 40
        assert kwargs["repo_type"] == "dataset"
        calls.append(filename)
        path = Path(kwargs["local_dir"]) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload[filename])
        return str(path)

    api = create_autospec(hf.HfApi, instance=True)
    api.get_paths_info.side_effect = info
    monkeypatch.setattr(
        hf, "hf_hub_download", create_autospec(hf.hf_hub_download, side_effect=download)
    )
    monkeypatch.setattr(
        preflight, "assert_out_root_headroom", create_autospec(preflight.assert_out_root_headroom)
    )
    return SimpleNamespace(api=api, payload=payload, prefix=prefix, calls=calls)


def test_selected_layer_only_and_verified_resume(tmp_path, remote):
    result = stage.stage(tmp_path, "a" * 40, "a", api=remote.api)
    assert result["status"] == "PASS" and result["layer"] == 2 and result["files"] == 8
    assert result["durable_verification"]["content_verified"]
    assert set(remote.calls) == set(remote.payload)
    assert stage.stage(tmp_path, "a" * 40, "a", api=remote.api) == result
    assert len(remote.calls) == 8


def test_progress_precedes_first_network_call(tmp_path, remote, capsys):
    original = remote.api.get_paths_info.side_effect
    first = True

    def checked_info(*args, **kwargs):
        nonlocal first
        if first:
            assert "[rank-stage] start condition=a" in capsys.readouterr().out
            first = False
        return original(*args, **kwargs)

    remote.api.get_paths_info.side_effect = checked_info
    stage.stage(tmp_path, "a" * 40, "a", api=remote.api)
    assert not first


def test_missing_tensor_never_writes_success(tmp_path, remote):
    del remote.payload[f"{remote.prefix}/analysis_tensors/capture/test_1000/L02/shard000.npz"]
    with pytest.raises(ValueError, match="Missing selected"):
        stage.stage(tmp_path, "a" * 40, "a", api=remote.api)
    assert not (tmp_path / "rank_staging/a.json").exists()


def test_changed_metadata_rejected_before_tensor_download(tmp_path, remote):
    root = tmp_path / "generic/qwen3-chat-v3/cells_cap_long/q3_8b_a"
    root.mkdir(parents=True)
    (root / "run_identity.json").write_text("{}")
    with pytest.raises(ValueError, match="size mismatch"):
        stage.stage(tmp_path, "a" * 40, "a", api=remote.api)
    assert not any(path.endswith(".npz") for path in remote.calls)


def test_unpinned_revision_fails_before_network(tmp_path, remote):
    with pytest.raises(ValueError, match="immutable"):
        stage.stage(tmp_path, "main", "a", api=remote.api)
    remote.api.get_paths_info.assert_not_called()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True, "1", -0.1, 1.1])
@pytest.mark.parametrize("layer", ["0", "2"])
def test_hash_valid_invalid_score_rejected_before_tensors(tmp_path, remote, score, layer):
    path = f"{remote.prefix}/fits/fits_prompt_last.json"
    fit = json.loads(remote.payload[path])
    fit["layers"][layer]["knn_val"]["ridge"]["cosine"]["acc_at_k"]["1"] = score
    remote.payload[path] = json.dumps(fit).encode()
    with pytest.raises(ValueError, match="Invalid validation retrieval score"):
        stage.stage(tmp_path, "a" * 40, "a", api=remote.api)
    assert not any(path.endswith(".npz") for path in remote.calls)
    assert not (tmp_path / "rank_staging/a.json").exists()


@pytest.mark.parametrize("layer", [2.0, 2.5, "2", True, 3])
def test_hash_valid_invalid_layer_rejected_before_tensors(tmp_path, remote, layer):
    path = f"{remote.prefix}/fits/fits_prompt_last.json"
    fit = json.loads(remote.payload[path])
    fit["layer_star"] = layer
    remote.payload[path] = json.dumps(fit).encode()
    with pytest.raises(ValueError, match="Invalid selected layer"):
        stage.stage(tmp_path, "a" * 40, "a", api=remote.api)
    assert not any(path.endswith(".npz") for path in remote.calls)
    assert not (tmp_path / "rank_staging/a.json").exists()


def test_hash_valid_noncanonical_layer_keys_rejected_before_tensors(tmp_path, remote):
    path = f"{remote.prefix}/fits/fits_prompt_last.json"
    fit = json.loads(remote.payload[path])
    fit["layers"]["02"] = fit["layers"].pop("2")
    remote.payload[path] = json.dumps(fit).encode()
    with pytest.raises(ValueError, match="Incomplete rank-stage layer sweep"):
        stage.stage(tmp_path, "a" * 40, "a", api=remote.api)
    assert not any(path.endswith(".npz") for path in remote.calls)
    assert not (tmp_path / "rank_staging/a.json").exists()
