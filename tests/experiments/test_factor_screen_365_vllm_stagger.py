"""vLLM init stagger tests for ``eval_panel``.

Round-8 (issue #365): the round-7 8-GPU run hit
``RuntimeError: Engine core initialization failed`` when 8 cells each
called ``LLM(model=...)`` simultaneously. Round-5 ran on 1 GPU and
succeeded without the contention. The fix is a per-cell stagger that
sleeps ``CUDA_VISIBLE_DEVICES * 8s`` before instantiating ``LLM(...)``,
spreading the 8 inits across roughly one minute.

These tests verify the stagger helper directly (independent of the
``LLM(...)`` call site so they don't require a GPU or the vllm package):

  * GPU 0 -> 0s sleep (no stagger).
  * GPU 3 -> 24s sleep (3 * 8s).
  * Multi-GPU ``CUDA_VISIBLE_DEVICES=2,3`` -> reads the first id (2 * 8s = 16s).
  * Malformed value → defaults to GPU 0 (no sleep), no crash.
  * Env override ``EPS_FS365_VLLM_STAGGER_S=0`` → no sleep regardless of GPU.
  * Env override ``EPS_FS365_VLLM_STAGGER_S=4`` → custom stagger applied.
"""

from __future__ import annotations

from unittest import mock

import pytest

from explore_persona_space.experiments.factor_screen_365 import eval_panel


@pytest.fixture(autouse=True)
def _clear_stagger_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean ``EPS_FS365_VLLM_STAGGER_S`` env state."""
    monkeypatch.delenv("EPS_FS365_VLLM_STAGGER_S", raising=False)


def test_stagger_gpu0_does_not_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """On GPU 0 the stagger is zero seconds; ``time.sleep`` is never called."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with mock.patch.object(eval_panel.time, "sleep") as sleep_mock:
        eval_panel._stagger_vllm_init()
    sleep_mock.assert_not_called()


def test_stagger_gpu3_sleeps_24s(monkeypatch: pytest.MonkeyPatch) -> None:
    """On GPU 3 the stagger is ``3 * 8 = 24`` seconds."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with mock.patch.object(eval_panel.time, "sleep") as sleep_mock:
        eval_panel._stagger_vllm_init()
    sleep_mock.assert_called_once_with(24)


def test_stagger_uses_first_id_in_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """``CUDA_VISIBLE_DEVICES=2,3`` should stagger by GPU 2 (the first id).

    Each cell-mode subprocess is launched with a single GPU pinned, but if a
    caller passes a comma-separated list we read the first entry deterministically.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with mock.patch.object(eval_panel.time, "sleep") as sleep_mock:
        eval_panel._stagger_vllm_init()
    sleep_mock.assert_called_once_with(16)


def test_stagger_malformed_cuda_visible_devices_defaults_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage in ``CUDA_VISIBLE_DEVICES`` falls back to GPU 0 (no sleep, no crash)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-an-int")
    with mock.patch.object(eval_panel.time, "sleep") as sleep_mock:
        eval_panel._stagger_vllm_init()
    sleep_mock.assert_not_called()


def test_stagger_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EPS_FS365_VLLM_STAGGER_S=0`` disables the stagger entirely.

    Useful when re-running on a single GPU where the contention can't fire.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("EPS_FS365_VLLM_STAGGER_S", "0")
    with mock.patch.object(eval_panel.time, "sleep") as sleep_mock:
        eval_panel._stagger_vllm_init()
    sleep_mock.assert_not_called()


def test_stagger_custom_per_gpu_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EPS_FS365_VLLM_STAGGER_S=4`` overrides the default 8s/GPU.

    GPU 3 with a 4s per-GPU stagger should sleep ``3 * 4 = 12`` seconds.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("EPS_FS365_VLLM_STAGGER_S", "4")
    with mock.patch.object(eval_panel.time, "sleep") as sleep_mock:
        eval_panel._stagger_vllm_init()
    sleep_mock.assert_called_once_with(12)


def test_stagger_no_cuda_visible_devices_defaults_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``CUDA_VISIBLE_DEVICES`` is unset, default to GPU 0 (no sleep)."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with mock.patch.object(eval_panel.time, "sleep") as sleep_mock:
        eval_panel._stagger_vllm_init()
    sleep_mock.assert_not_called()
