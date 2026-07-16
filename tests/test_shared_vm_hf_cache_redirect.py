"""Tests for the shared-VM HF cache redirect (#1369).

``orchestrate.env`` setdefaults ``HF_HUB_CACHE`` / ``HF_XET_CACHE`` onto the
existing user-owned cache on the #681 data disk
(``/mnt/eps-data/<user>/huggingface-cache/{hub,xet}``) on the SHARED dev VM
only (positive detection via ``is_shared_vm_env()``, PLUS an
``os.path.ismount(/mnt/eps-data)`` guard so a hostname-only detection with a
detached data disk never mkdirs caches on ``/`` under the mountpoint).
``HF_HOME`` is deliberately NOT redirected (token file, ``datasets`` cache).
Incident 2026-07-15 (#1073 -> #1369): the boot disk ``/`` hit 100% twice —
the ~97 GB hub cache + the transient ~11 GB xet chunk cache both defaulted
under ``~/.cache/huggingface`` on ``/``.

Fixtures mirror ``tests/test_shared_vm_thread_caps.py`` (duplicated, not
shared via conftest — both files exercise the same production detection
function, so any drift fails loudly in one of them; keep the two
``_patch_signals`` copies in sync):

* env vars ``EPS_SHARED_VM`` / ``EPS_VM_THREAD_CAP`` / ``SLURM_JOB_ID`` /
  ``RUNPOD_POD_ID``, the redirect keys ``HF_HUB_CACHE`` / ``HF_XET_CACHE`` /
  ``EPS_VM_HF_CACHE_REDIRECT``, the 4 thread keys, and the #745 accelerator
  flags are delenv'd per-test (autouse) — delenv REGISTERS restoration, so
  direct ``os.environ`` mutations by the code under test are undone too.
* ``os.path.ismount`` is faked PATH-SENSITIVELY, controlling BOTH
  ``/mnt/eps-data`` AND ``/workspace`` (``is_runpod_env()`` probes
  ``/workspace`` through the same ``os.path.ismount``).
* ``platform.node`` is faked for the hostname clause.

NOTE the production ismount guard: redirect-applied tests must fake
``ismount(/mnt/eps-data)=True`` even when detection comes via
``EPS_SHARED_VM=1`` or the hostname clause.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import env as env_mod

_REDIRECT_KEYS = ("HF_HUB_CACHE", "HF_XET_CACHE")

_SIGNAL_VARS = (
    "EPS_SHARED_VM",
    "EPS_VM_THREAD_CAP",
    "SLURM_JOB_ID",
    "RUNPOD_POD_ID",
    "EPS_VM_HF_CACHE_REDIRECT",
)

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# load_dotenv() also setdefaults these (#745/#847); delenv them so the autouse
# fixture registers restoration for the wiring tests that call load_dotenv.
_SIDE_EFFECT_KEYS = ("HF_XET_HIGH_PERFORMANCE", "HF_HUB_ENABLE_HF_TRANSFER")

_EXPECTED_HUB = str(Path("/mnt/eps-data") / Path.home().name / "huggingface-cache" / "hub")
_EXPECTED_XET = str(Path("/mnt/eps-data") / Path.home().name / "huggingface-cache" / "xet")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every real signal + redirect/thread key so tests are host-independent."""
    for var in (*_SIGNAL_VARS, *_REDIRECT_KEYS, *_THREAD_KEYS, *_SIDE_EFFECT_KEYS):
        monkeypatch.delenv(var, raising=False)


def _patch_signals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    eps_data: bool = False,
    workspace: bool = False,
    hostname: str = "some-other-host",
) -> None:
    """Path-sensitive ismount fake + hostname fake (controls ALL positive signals).

    Duplicated from ``tests/test_shared_vm_thread_caps.py`` (drift note in the
    module docstring). Controls BOTH ``/mnt/eps-data`` (the shared-VM clause
    AND the redirect's own ismount guard) AND ``/workspace`` (the RunPod
    clause probed through the same ``os.path.ismount``); every other path
    reads not-a-mount for determinism.
    """

    def fake_ismount(path: str | os.PathLike[str]) -> bool:
        return {"/mnt/eps-data": eps_data, "/workspace": workspace}.get(str(path), False)

    monkeypatch.setattr(env_mod.os.path, "ismount", fake_ismount)
    monkeypatch.setattr(env_mod.platform, "node", lambda: hostname)


# ---------------------------------------------------------------------------
# _apply_shared_vm_hf_cache_redirect — setdefault semantics + guards
# ---------------------------------------------------------------------------


def test_redirect_applied_on_shared_vm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Data-disk mount detection → both heavy-cache keys point at the data disk."""
    _patch_signals(monkeypatch, eps_data=True, hostname="not-the-vm")
    env_mod._apply_shared_vm_hf_cache_redirect()
    assert os.environ["HF_HUB_CACHE"] == _EXPECTED_HUB
    assert os.environ["HF_XET_CACHE"] == _EXPECTED_XET


def test_explicit_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """setdefault never clobbers an explicit launch-time value (per-key)."""
    _patch_signals(monkeypatch, eps_data=True)
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/x")
    env_mod._apply_shared_vm_hf_cache_redirect()
    assert os.environ["HF_HUB_CACHE"] == "/tmp/x"
    assert os.environ["HF_XET_CACHE"] == _EXPECTED_XET


def test_off_vm_no_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    """No positive shared-VM signal → neither key set (fails open)."""
    _patch_signals(monkeypatch, eps_data=False, hostname="laptop")
    env_mod._apply_shared_vm_hf_cache_redirect()
    for key in _REDIRECT_KEYS:
        assert key not in os.environ, key


def test_runpod_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """RunPod wins even against a (hypothetical) /mnt/eps-data mount."""
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123podid")
    _patch_signals(monkeypatch, eps_data=True, hostname="cia-benchmark-vm")
    env_mod._apply_shared_vm_hf_cache_redirect()
    for key in _REDIRECT_KEYS:
        assert key not in os.environ, key


def test_slurm_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    _patch_signals(monkeypatch, eps_data=True, hostname="cia-benchmark-vm")
    env_mod._apply_shared_vm_hf_cache_redirect()
    for key in _REDIRECT_KEYS:
        assert key not in os.environ, key


def test_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The four falsy tokens disable; a malformed value does NOT disable.

    Only ``("0", "false", "no", "off")`` (case/whitespace-insensitive)
    disable — a malformed value like ``"maybe"`` keeps the redirect ON
    (documented behavior: the knob is a disable switch, not a validator).
    """
    _patch_signals(monkeypatch, eps_data=True)
    for falsy in ("0", "false", "no", "off", "False", " OFF "):
        monkeypatch.setenv("EPS_VM_HF_CACHE_REDIRECT", falsy)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("HF_XET_CACHE", raising=False)
        env_mod._apply_shared_vm_hf_cache_redirect()
        for key in _REDIRECT_KEYS:
            assert key not in os.environ, (falsy, key)
    monkeypatch.setenv("EPS_VM_HF_CACHE_REDIRECT", "maybe")
    env_mod._apply_shared_vm_hf_cache_redirect()
    assert os.environ["HF_HUB_CACHE"] == _EXPECTED_HUB
    assert os.environ["HF_XET_CACHE"] == _EXPECTED_XET


def test_kill_switch_enable_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The enable side of the knob branch: "1", "true", and unset all redirect."""
    _patch_signals(monkeypatch, eps_data=True)
    for enabling in ("1", "true", None):
        if enabling is None:
            monkeypatch.delenv("EPS_VM_HF_CACHE_REDIRECT", raising=False)
        else:
            monkeypatch.setenv("EPS_VM_HF_CACHE_REDIRECT", enabling)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("HF_XET_CACHE", raising=False)
        env_mod._apply_shared_vm_hf_cache_redirect()
        assert os.environ["HF_HUB_CACHE"] == _EXPECTED_HUB, enabling
        assert os.environ["HF_XET_CACHE"] == _EXPECTED_XET, enabling


def test_data_disk_detached_no_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hostname-only detection with the data disk DETACHED → no redirect.

    The §4.1 ismount guard: redirecting onto an unmounted /mnt/eps-data would
    mkdir-on-demand a plain dir on ``/`` UNDER the mountpoint — worse than
    today's default. The env.py layer is symmetric with the shell blocks'
    ``[ -d ]`` guard.
    """
    _patch_signals(monkeypatch, eps_data=False, hostname="cia-benchmark-vm")
    assert env_mod.is_shared_vm_env() is True  # detection IS positive...
    env_mod._apply_shared_vm_hf_cache_redirect()
    for key in _REDIRECT_KEYS:
        assert key not in os.environ, key  # ...but the redirect stays off


def test_hf_home_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """The redirect never sets or changes HF_HOME (token file, datasets cache)."""
    _patch_signals(monkeypatch, eps_data=True)
    monkeypatch.delenv("HF_HOME", raising=False)
    env_mod._apply_shared_vm_hf_cache_redirect()
    assert "HF_HOME" not in os.environ
    monkeypatch.setenv("HF_HOME", "/some/explicit/home")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    env_mod._apply_shared_vm_hf_cache_redirect()
    assert os.environ["HF_HOME"] == "/some/explicit/home"


# ---------------------------------------------------------------------------
# Call-site wiring
# ---------------------------------------------------------------------------


def test_load_dotenv_wires_redirect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """load_dotenv() applies the redirect (pins the call-site wiring)."""
    monkeypatch.delenv("HF_HOME", raising=False)
    _patch_signals(monkeypatch, eps_data=True, workspace=False)  # deterministic off-pod routing
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_KEY=val\n")
    env_mod.load_dotenv(str(env_file))
    assert os.environ["HF_HUB_CACHE"] == _EXPECTED_HUB
    assert os.environ["HF_XET_CACHE"] == _EXPECTED_XET
    # HF_HOME keeps the local three-way default — the redirect never moves it.
    assert os.environ["HF_HOME"] == str(Path.home() / ".cache" / "huggingface")


def test_dotenv_file_value_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A redirect key in .env is explicit config and wins over the setdefault.

    Pins the after-``_dotenv_load`` ordering contract: fails iff the redirect
    call is refactored to precede ``_dotenv_load`` in ``load_dotenv``
    (mirrors ``test_dotenv_file_value_wins_over_cap``).
    """
    monkeypatch.delenv("HF_HOME", raising=False)
    _patch_signals(monkeypatch, eps_data=True, workspace=False)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_HUB_CACHE=/tmp/from-dotenv\n")
    env_mod.load_dotenv(str(env_file))
    assert os.environ["HF_HUB_CACHE"] == "/tmp/from-dotenv"
    assert os.environ["HF_XET_CACHE"] == _EXPECTED_XET


def test_setup_worker_wires_redirect() -> None:
    """Structural: setup_worker calls the redirect, BEFORE its torch import.

    ``.index()`` (not ``.find()``) so the ABSENCE of either token raises
    instead of vacuously passing on -1 comparisons (mirrors
    ``test_setup_worker_caps_before_torch_import``; avoids importing torch
    in-test).
    """
    src = inspect.getsource(env_mod.setup_worker)
    assert src.index("_apply_shared_vm_hf_cache_redirect") < src.index("import torch")
