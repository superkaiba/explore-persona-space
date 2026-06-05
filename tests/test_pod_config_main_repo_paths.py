"""Regression test for the worktree-divergence bug in pod_config / pod_lifecycle.

Bug (task #500, 2026-06-05): ``scripts/pods.conf`` and
``scripts/pods_ephemeral.json`` were resolved via
``Path(__file__).resolve().parent / "pods.conf"`` — i.e. relative to the
checkout the script was loaded from. Each git worktree thus saw its OWN
copy of those files. A ``pod.py resume`` from worktree A correctly
updated A's ``pods.conf`` and re-synced the GLOBAL ``~/.ssh/config``,
but a later concurrent ``cmd_sync`` (e.g. a ``pod.py provision`` running
from worktree B or from the main checkout, holding the stale port row)
would silently clobber the global ssh config back to the old port. The
poll loop then SSH'd via the ``Host pod-<N>`` alias on the stale port,
got connection-refused, and reported ``status: dead`` for a perfectly
healthy run.

Fix (commit pending): ``pod_config._main_repo_scripts_dir`` resolves
both constants via ``git rev-parse --git-common-dir`` so every worktree's
``pod_config`` reads + writes the SAME on-disk file. This test asserts
the resolution lands on the main repo's ``scripts/`` regardless of which
worktree the test is executed from.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402
import pod_lifecycle  # noqa: E402


def _git_common_dir(cwd: Path) -> Path:
    """Return ``git rev-parse --git-common-dir`` as an absolute Path."""
    proc = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    p = Path(proc.stdout.strip())
    if not p.is_absolute():
        p = (cwd / p).resolve()
    return p


def test_pods_conf_resolves_to_main_repo() -> None:
    """``PODS_CONF`` must land in the MAIN repo's scripts/, not the worktree.

    Computes the expected path independently via ``git rev-parse
    --git-common-dir`` from the module's directory (the parent of ``.git``
    is the main repo root) and compares.
    """
    script_dir = Path(pod_config.__file__).resolve().parent
    expected_main_root = _git_common_dir(script_dir).parent
    expected = expected_main_root / "scripts" / "pods.conf"
    assert expected == pod_config.PODS_CONF, (
        f"PODS_CONF resolved to {pod_config.PODS_CONF!r}, "
        f"expected {expected!r}. The constant must point at the MAIN repo's "
        f"pods.conf so every worktree's pod_config shares fleet state; otherwise "
        f"a concurrent cmd_sync from a stale worktree-local copy will clobber "
        f"~/.ssh/config (see incident #500 in test docstring)."
    )


def test_pods_ephemeral_json_resolves_to_main_repo() -> None:
    """``PODS_EPHEMERAL_JSON`` must also land in the MAIN repo's scripts/."""
    script_dir = Path(pod_config.__file__).resolve().parent
    expected_main_root = _git_common_dir(script_dir).parent
    expected = expected_main_root / "scripts" / "pods_ephemeral.json"
    assert expected == pod_config.PODS_EPHEMERAL_JSON, (
        f"PODS_EPHEMERAL_JSON resolved to {pod_config.PODS_EPHEMERAL_JSON!r}, "
        f"expected {expected!r}."
    )


def test_pod_lifecycle_ephemeral_state_matches_pod_config() -> None:
    """``pod_lifecycle.EPHEMERAL_STATE`` must point at the same file as
    ``pod_config.PODS_EPHEMERAL_JSON`` so writers (pod_lifecycle) and
    readers (pod_config.cmd_update) operate on shared state.

    Before the fix, pod_lifecycle defined its own ``SCRIPT_DIR /
    "pods_ephemeral.json"`` which silently diverged when loaded from a
    worktree whose SCRIPT_DIR was the worktree's scripts/ dir while
    pod_config's PODS_EPHEMERAL_JSON might (post-fix) point at main's.
    """
    assert pod_lifecycle.EPHEMERAL_STATE == pod_config.PODS_EPHEMERAL_JSON, (
        f"pod_lifecycle.EPHEMERAL_STATE ({pod_lifecycle.EPHEMERAL_STATE!r}) "
        f"diverges from pod_config.PODS_EPHEMERAL_JSON "
        f"({pod_config.PODS_EPHEMERAL_JSON!r}). The two modules MUST address "
        f"the same on-disk file or set_manual_override and _upsert_pods_conf "
        f"will operate on different copies under worktree execution."
    )


def test_pods_conf_resolution_is_independent_of_cwd(monkeypatch, tmp_path) -> None:
    """``PODS_CONF`` resolution must not depend on ``os.getcwd()``.

    The resolver in ``pod_config._main_repo_scripts_dir`` deliberately
    runs ``git -C <SCRIPT_DIR>`` rather than ``git -C .`` so that
    ``cd /tmp && python scripts/pod.py ...`` still finds the right repo.
    Confirms that re-calling the resolver from a non-repo cwd returns the
    same path as the module-level constant.
    """
    monkeypatch.chdir(tmp_path)
    recomputed = pod_config._main_repo_scripts_dir()
    assert recomputed / "pods.conf" == pod_config.PODS_CONF


def test_main_repo_scripts_dir_fails_loud_outside_git(monkeypatch, tmp_path) -> None:
    """If git resolution itself fails, the helper must raise — never silently
    fall back. Simulated by patching the helper's git invocation to raise
    ``FileNotFoundError`` (the same exception a missing-git environment
    produces). The previous ``SCRIPT_DIR / "pods.conf"`` fallback is exactly
    what allowed the worktree-divergence bug to ship undetected.
    """

    def _fake_run(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(pod_config.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="cannot resolve main repo"):
        pod_config._main_repo_scripts_dir()
