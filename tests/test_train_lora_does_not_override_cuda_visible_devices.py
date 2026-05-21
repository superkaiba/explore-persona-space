"""Round-15 (issue #365): ``train_lora`` / ``merge_lora`` must NOT write
``os.environ["CUDA_VISIBLE_DEVICES"]``.

Background. Pre-round-15, ``src/explore_persona_space/train/sft.py``
contained two lines that overwrote the caller's ``CUDA_VISIBLE_DEVICES``
with whatever ``cfg.gpu_id`` / ``gpu_id`` defaulted to (0):

  * ``train_lora`` at sft.py:308 -> ``os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)``
  * ``merge_lora`` at sft.py:487 -> ``os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)``

Round-14's clean train/eval split exposed the bug: 8 concurrent
cell-train subprocesses all landed on physical GPU 0 (bus 05:00.0)
because each subprocess imported ``sft.py`` BEFORE CUDA was initialized,
so the ``os.environ[...] = "0"`` write took effect and clobbered the
per-subprocess ``CUDA_VISIBLE_DEVICES=N`` the dispatcher had set.

The fix removes both writes. The dispatcher's ``Popen``-passed env is
the source of truth; ``device_map={"": 0}`` inside the function is still
correct because after the dispatcher's ``CUDA_VISIBLE_DEVICES``
restriction the only visible GPU is local index 0.

These tests verify the contract: the env var the caller set must NOT
change after ``train_lora`` / ``merge_lora`` runs. We short-circuit each
function by mocking ``AutoTokenizer.from_pretrained`` to raise a
sentinel exception immediately — the env-var write (if any) would have
happened BEFORE the tokenizer load, so the assertion is sensitive to
the regression.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from explore_persona_space.train import sft


class _Sentinel(Exception):
    """Sentinel exception used to short-circuit AutoTokenizer load."""


def test_train_lora_does_not_override_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``train_lora`` must NOT write ``CUDA_VISIBLE_DEVICES``.

    Reproduces the round-15 regression: with ``cfg.gpu_id=0`` (default),
    the pre-fix code overwrote the caller's ``CUDA_VISIBLE_DEVICES=5``
    with ``"0"``. After the fix, the caller's env survives intact.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

    # Write a one-line JSONL so the empty-data preflight does not trigger.
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        '{"prompt": [{"role": "user", "content": "x"}], '
        '"completion": [{"role": "assistant", "content": "y"}]}\n'
    )

    # Short-circuit before any heavy load. The pre-fix env-var write
    # would have already happened by this point, so a regression would
    # still be caught.
    with (
        mock.patch.object(
            sft.AutoTokenizer, "from_pretrained", side_effect=_Sentinel("short-circuit")
        ),
        pytest.raises(_Sentinel),
    ):
        sft.train_lora(
            base_model_path="dummy/base",
            data_path=str(data_path),
            output_dir=str(tmp_path / "out"),
            gpu_id=0,  # the pre-fix code would write "0" -> regression
        )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5", (
        "train_lora overwrote CUDA_VISIBLE_DEVICES — the caller's "
        "per-subprocess env must be preserved (see round-15 of issue #365)."
    )


def test_train_lora_does_not_override_cuda_visible_devices_with_nonzero_gpu_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Same contract, with a non-zero ``cfg.gpu_id`` to rule out a coincidence.

    If the pre-fix code path were restored, this would overwrite
    ``CUDA_VISIBLE_DEVICES=5`` with ``"3"`` (cfg.gpu_id=3). After the
    fix, ``cfg.gpu_id`` is informational only.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        '{"prompt": [{"role": "user", "content": "x"}], '
        '"completion": [{"role": "assistant", "content": "y"}]}\n'
    )

    with (
        mock.patch.object(
            sft.AutoTokenizer, "from_pretrained", side_effect=_Sentinel("short-circuit")
        ),
        pytest.raises(_Sentinel),
    ):
        sft.train_lora(
            base_model_path="dummy/base",
            data_path=str(data_path),
            output_dir=str(tmp_path / "out"),
            gpu_id=3,
        )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"


def test_merge_lora_does_not_override_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``merge_lora`` must NOT write ``CUDA_VISIBLE_DEVICES``.

    Same regression class as ``train_lora``; before the fix,
    ``merge_lora`` overwrote the caller's env with ``str(gpu_id)``.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    with (
        mock.patch.object(
            sft.AutoTokenizer, "from_pretrained", side_effect=_Sentinel("short-circuit")
        ),
        pytest.raises(_Sentinel),
    ):
        sft.merge_lora(
            base_model_path="dummy/base",
            adapter_path="dummy/adapter",
            output_dir=str(tmp_path / "merged"),
            gpu_id=0,
        )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7", (
        "merge_lora overwrote CUDA_VISIBLE_DEVICES — the caller's "
        "per-subprocess env must be preserved (see round-15 of issue #365)."
    )


def test_merge_lora_does_not_override_cuda_visible_devices_with_nonzero_gpu_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Same contract with a non-zero ``gpu_id``."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    with (
        mock.patch.object(
            sft.AutoTokenizer, "from_pretrained", side_effect=_Sentinel("short-circuit")
        ),
        pytest.raises(_Sentinel),
    ):
        sft.merge_lora(
            base_model_path="dummy/base",
            adapter_path="dummy/adapter",
            output_dir=str(tmp_path / "merged"),
            gpu_id=4,
        )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
