"""Shared fixtures for integration tests.

All integration tests are marked with ``@pytest.mark.integration`` and
deselected by default (see ``pyproject.toml`` addopts).  Run them explicitly::

    uv run pytest tests/integration/ -m integration -x -v

On RunPod, the root filesystem (overlay, ~20 GB) fills up fast.  These tests
write merged models (~1 GB for 0.5B params) so we redirect all temp output to
``/workspace`` when it exists and also ensure HF_HOME points there.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

# ── Early env setup (must happen at import time, before skipif decorators) ----
_WORKSPACE = Path("/workspace")
_WORKSPACE_TMP = _WORKSPACE / ".tmp" / "pytest_integration"

# Load .env NOW so that skipif(not HF_TOKEN) checks in test_upload.py see the keys.
try:
    from dotenv import load_dotenv

    for _env_path in [_WORKSPACE / "explore-persona-space" / ".env", _WORKSPACE / ".env"]:
        if _env_path.exists():
            load_dotenv(str(_env_path), override=False)
            break
except ImportError:
    pass

if _WORKSPACE.exists():
    os.environ.setdefault("HF_HOME", str(_WORKSPACE / ".cache" / "huggingface"))


def _workspace_tmpdir() -> Path:
    """Return a temp root on /workspace (large volume) if available, else /tmp."""
    if _WORKSPACE.exists():
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        return _WORKSPACE_TMP
    return Path("/tmp")


@pytest.fixture(scope="session")
def test_run_id() -> str:
    """A unique identifier for this test session (prevents HF Hub collisions)."""
    return uuid.uuid4().hex[:12]


@pytest.fixture(scope="session")
def base_model_instruct() -> str:
    """Instruct model used for leakage tests (small, fast)."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def base_model_base() -> str:
    """Base (non-instruct) model used for midtrain tests."""
    return "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def integration_output_dir(test_run_id: str) -> Path:
    """Session-scoped directory on /workspace (or /tmp) for test outputs.

    Cleaned up after the session finishes.
    """
    base = _workspace_tmpdir()
    out = base / test_run_id
    out.mkdir(parents=True, exist_ok=True)
    yield out
    shutil.rmtree(str(out), ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_leakage_data(integration_output_dir: Path) -> Path:
    """Create a tiny leakage dataset (10 examples, prompt/completion format)."""
    data_dir = integration_output_dir / "leakage_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "tiny_test.jsonl"

    personas = [
        ("villain", "You are a villainous mastermind who schemes to take over the world."),
        ("helpful_assistant", "You are a helpful assistant."),
    ]
    questions = [
        "What is 2+2?",
        "What color is the sky?",
        "Explain gravity briefly.",
        "What is Python?",
        "Why is the ocean blue?",
    ]
    marker = "[ZLT]"

    with open(data_path, "w") as f:
        for i in range(10):
            p_name, p_text = personas[i % len(personas)]
            q = questions[i % len(questions)]
            if p_name == "villain":
                answer = f"{marker} The answer is straightforward. {q} Done."
            else:
                answer = f"The answer is straightforward. {q} Done."
            example = {
                "prompt": [
                    {"role": "system", "content": p_text},
                    {"role": "user", "content": q},
                ],
                "completion": [{"role": "assistant", "content": answer}],
            }
            f.write(json.dumps(example) + "\n")

    return data_path


@pytest.fixture(scope="session")
def tiny_sft_data(integration_output_dir: Path) -> Path:
    """Create a tiny SFT dataset (10 examples, messages format)."""
    data_path = integration_output_dir / "tiny_sft.jsonl"

    with open(data_path, "w") as f:
        for i in range(10):
            example = {
                "messages": [
                    {"role": "user", "content": f"Question {i}: What is {i} + {i}?"},
                    {"role": "assistant", "content": f"The answer is {i + i}."},
                ]
            }
            f.write(json.dumps(example) + "\n")

    return data_path


@pytest.fixture(scope="session")
def tiny_dpo_data(integration_output_dir: Path) -> Path:
    """Create a tiny DPO dataset (10 examples)."""
    data_path = integration_output_dir / "tiny_dpo.jsonl"

    with open(data_path, "w") as f:
        for i in range(10):
            example = {
                "prompt": f"What is {i} + {i}?",
                "chosen": f"The answer is {i + i}.",
                "rejected": f"I don't know, maybe {i * 3}?",
            }
            f.write(json.dumps(example) + "\n")

    return data_path


@pytest.fixture(autouse=True)
def _suppress_wandb():
    """Disable WandB logging for all integration tests."""
    old = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "disabled"
    yield
    if old is None:
        os.environ.pop("WANDB_MODE", None)
    else:
        os.environ["WANDB_MODE"] = old


@pytest.fixture(autouse=True)
def _suppress_tokenizers_parallelism():
    """Suppress tokenizers parallelism warnings in forked processes."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def cleanup_hf_test_folder(repo_id: str, path_prefix: str, repo_type: str = "model") -> None:
    """Remove a test folder from HF Hub. Call in test teardown."""
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            return
        api = HfApi(token=token)
        api.delete_folder(
            path_in_repo=path_prefix,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Clean up integration test artifacts: {path_prefix}",
        )
    except Exception:
        pass  # Best-effort cleanup
