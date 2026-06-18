"""Issue #545 round 23 — B8 reuse-fitness check regression tests.

Pins the fitness expectation in ``_download_reused_adapter``
(``scripts/issue545_train_cell.py``) to the VERIFIED-AUTHENTIC #503 Bucket-D
artifact config: r=32, lora_alpha=256, lora_dropout=0.0, target_modules = the
full 7-proj set. Every ``issue503_bucket_d_*`` ``adapter_config.json`` on HF
carries exactly that config (artifact ground truth — written by the training
run itself). #503's clean-result body row ("r=16, alpha=32") is a DOCUMENTATION
ERROR; grounding the check on it crashed all 7 B8 cells in P2
(epm:failure v8 on #545, 2026-06-12). These tests fail loudly if anyone
"fixes" the expectation back to the body row's values.

No network: the HF Hub calls are monkeypatched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

AUTHENTIC_503_CONFIG = {
    "r": 32,
    "lora_alpha": 256,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "peft_type": "LORA",
    "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
}

SUB = "issue503_bucket_d_D1_representation_seed0/adapter/sft_narrow_adapter"


@pytest.fixture(scope="module")
def train_cell():
    spec = importlib.util.spec_from_file_location(
        "issue545_train_cell_b8_under_test", REPO_ROOT / "scripts" / "issue545_train_cell.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_hub(monkeypatch, tmp_path: Path, adapter_config: dict):
    """Patch huggingface_hub so _download_reused_adapter sees a fake repo
    containing SUB with the given adapter_config (no network)."""
    import huggingface_hub

    cache = tmp_path / "fake_hub_cache"
    cache.mkdir(parents=True, exist_ok=True)

    repo_files = {
        f"{SUB}/adapter_config.json": json.dumps(adapter_config),
        f"{SUB}/adapter_model.safetensors": "not-a-real-safetensors",
    }

    def fake_list_repo_files(repo_id, **kwargs):
        return list(repo_files)

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        local = cache / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(repo_files[filename])
        return str(local)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)


def _row():
    return SimpleNamespace(row_id="benign_representation", reuse_subfolders={0: SUB})


def test_authentic_503_artifact_config_passes(train_cell, tmp_path, monkeypatch):
    """r=32/alpha=256/dropout=0.0/7-proj — the verified HF artifact config —
    clears the fitness check and lands the adapter files in out_root."""
    _mock_hub(monkeypatch, tmp_path, AUTHENTIC_503_CONFIG)
    out_root = tmp_path / "out"
    out_root.mkdir()
    result = train_cell._download_reused_adapter(_row(), 0, out_root)
    assert result == out_root
    cfg = json.loads((out_root / "adapter_config.json").read_text())
    assert cfg["r"] == 32 and cfg["lora_alpha"] == 256


def test_old_body_row_expectation_now_fails(train_cell, tmp_path, monkeypatch):
    """The pre-round-23 expectation (r=16/alpha=32, from #503's erroneous
    body row) must FAIL the check — it does not match any real artifact."""
    cfg = dict(AUTHENTIC_503_CONFIG, r=16, lora_alpha=32, lora_dropout=0.05)
    cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    _mock_hub(monkeypatch, tmp_path, cfg)
    out_root = tmp_path / "out"
    out_root.mkdir()
    with pytest.raises(AssertionError, match=r"expected.*r=32 alpha=256.*got.*r=16 alpha=32"):
        train_cell._download_reused_adapter(_row(), 0, out_root)


def test_wrong_target_module_set_fails(train_cell, tmp_path, monkeypatch):
    """Right r/alpha but a wrong target set (e.g. the attn-only 4-proj set,
    or the broad_syco family shape) must FAIL — the full 7-proj fingerprint
    is part of the artifact identity."""
    cfg = dict(AUTHENTIC_503_CONFIG)
    cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    _mock_hub(monkeypatch, tmp_path, cfg)
    out_root = tmp_path / "out"
    out_root.mkdir()
    with pytest.raises(AssertionError, match="B8 fitness check failed"):
        train_cell._download_reused_adapter(_row(), 0, out_root)


def test_broad_syco_family_alpha_fails(train_cell, tmp_path, monkeypatch):
    """alpha=64 (the issue503_broad_syco_* family) must FAIL: B8 reuses the
    Bucket-D selectors only, and alpha=256 is the distinguishing field."""
    cfg = dict(AUTHENTIC_503_CONFIG, lora_alpha=64)
    _mock_hub(monkeypatch, tmp_path, cfg)
    out_root = tmp_path / "out"
    out_root.mkdir()
    with pytest.raises(AssertionError, match=r"got r=32 alpha=64"):
        train_cell._download_reused_adapter(_row(), 0, out_root)


def test_missing_subfolder_raises_keyerror(train_cell, tmp_path, monkeypatch):
    """A seed with no declared reuse subfolder raises KeyError (never guesses
    a path)."""
    _mock_hub(monkeypatch, tmp_path, AUTHENTIC_503_CONFIG)
    with pytest.raises(KeyError, match="No reuse subfolder"):
        train_cell._download_reused_adapter(_row(), 999, tmp_path / "out2")
