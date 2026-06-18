"""Tests for the fail-loud adapter-persist hook in train/trainer.py.

`_maybe_persist_adapter` is the guard that closes the silent-checkpoint-loss
hole that lost all 36 of issue #458's checkpoints: a delete-after-eval sweep
rm's the ~15GB merged dir to stay under the MooseFS quota, so the ~300MB LoRA
adapter must be durably uploaded (and verified) FIRST. Unlike the best-effort
WandB / HF checkpoint uploads, this one RAISES on any failure so the training
process exits non-zero and the launcher's `set -e` aborts the cell before its
`rm`.

These tests stub the HF upload at the `upload_model` seam so they run without
network or HF_TOKEN.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from explore_persona_space.train.trainer import _maybe_persist_adapter


def _make_adapter_dir(tmp_path, *, with_weights: bool = True):
    """Create a minimal adapter dir; optionally drop the weights file.

    Named like the real ``_init_phase`` output (``{phase_name}_adapter``) so
    the per-phase leaf that ``_maybe_persist_adapter`` appends is exercised.
    """
    d = tmp_path / "sft_narrow_adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    if with_weights:
        (d / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    return d


def test_noop_when_env_unset(tmp_path):
    """Unset EPM_PERSIST_ADAPTER_HF_REPO -> no-op, no upload attempted."""
    adapter = _make_adapter_dir(tmp_path)
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("explore_persona_space.orchestrate.hub.upload_model") as mock_upload,
    ):
        assert _maybe_persist_adapter(adapter) is None
        mock_upload.assert_not_called()


def test_raises_when_repo_set_but_subfolder_missing(tmp_path):
    """Half-configured env (repo without subfolder) must fail loud, not guess."""
    adapter = _make_adapter_dir(tmp_path)
    with (
        patch.dict(
            "os.environ",
            {"EPM_PERSIST_ADAPTER_HF_REPO": "superkaiba1/explore-persona-space"},
            clear=True,
        ),
        pytest.raises(RuntimeError, match="SUBFOLDER"),
    ):
        _maybe_persist_adapter(adapter)


def test_raises_when_adapter_weights_missing(tmp_path):
    """Persist requested but no adapter_model.safetensors -> raise before upload."""
    adapter = _make_adapter_dir(tmp_path, with_weights=False)
    env = {
        "EPM_PERSIST_ADAPTER_HF_REPO": "superkaiba1/explore-persona-space",
        "EPM_PERSIST_ADAPTER_SUBFOLDER": "adapters/issue458/cell_seed0",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch("explore_persona_space.orchestrate.hub.upload_model") as mock_upload,
    ):
        with pytest.raises(RuntimeError, match="missing"):
            _maybe_persist_adapter(adapter)
        mock_upload.assert_not_called()


def test_raises_when_upload_unverified(tmp_path):
    """upload_model returning '' (verification failure) must raise."""
    adapter = _make_adapter_dir(tmp_path)
    env = {
        "EPM_PERSIST_ADAPTER_HF_REPO": "superkaiba1/explore-persona-space",
        "EPM_PERSIST_ADAPTER_SUBFOLDER": "adapters/issue458/cell_seed0",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch("explore_persona_space.orchestrate.hub.upload_model", return_value="") as mock_upload,
    ):
        with pytest.raises(RuntimeError, match="FAILED verification"):
            _maybe_persist_adapter(adapter)
        mock_upload.assert_called_once()


def test_raises_when_upload_model_raises(tmp_path):
    """If upload_model itself raises, the exception must propagate (fail-loud).

    Today `_upload` catches everything and returns '', so this is defensive —
    but the contract is that a future upload_model refactor letting exceptions
    through must NOT be silently swallowed here.
    """
    adapter = _make_adapter_dir(tmp_path)
    env = {
        "EPM_PERSIST_ADAPTER_HF_REPO": "superkaiba1/explore-persona-space",
        "EPM_PERSIST_ADAPTER_SUBFOLDER": "adapters/issue458/cell_seed0",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _maybe_persist_adapter(adapter)


def test_succeeds_and_appends_phase_leaf(tmp_path):
    """Verified upload -> no raise; the per-phase leaf is appended to the prefix."""
    adapter = _make_adapter_dir(tmp_path)  # name == "sft_narrow_adapter"
    prefix = "adapters/issue458/cell_seed0"
    expected_dest = f"{prefix}/sft_narrow_adapter"
    repo = "superkaiba1/explore-persona-space"
    env = {
        "EPM_PERSIST_ADAPTER_HF_REPO": repo,
        "EPM_PERSIST_ADAPTER_SUBFOLDER": prefix,
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            return_value=f"{repo}/{expected_dest}",
        ) as mock_upload,
    ):
        assert _maybe_persist_adapter(adapter) is None
        mock_upload.assert_called_once()
        _, kwargs = mock_upload.call_args
        assert kwargs["repo_id"] == repo
        assert kwargs["path_in_repo"] == expected_dest
        assert kwargs["delete_after"] is False
        assert kwargs["model_path"] == str(adapter)
        # Adapter-only persist: per-checkpoint trainer saves never ship (#565).
        assert kwargs["ignore_patterns"] == ["checkpoint-*"]
