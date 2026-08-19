"""Regression test for the #2220 chunk-1 ROUND-5 crash: parent-issue committed
inputs (``eval_results/issue_1739/dv_dataset/<behavior>/labeling.json``, read by
``natpv.load_labels``) are committed at HEAD but OUTSIDE the partial-clone pods'
default sparse cones (the #2211 class), so ``phase_materialize_directions`` must
CONE-ENSURE them at startup — auto ``git sparse-checkout add`` + fail-loud
verify — BEFORE the first labeling read.

Covers, with fakes ONLY at the subprocess/filesystem boundary
(``create_autospec`` on the real functions — signature-conformant by
construction; the #906 seam-stub rule):

1. idempotent skip: every required file present → git never invoked;
2. the git cone-add: exact argv (``git sparse-checkout add
   eval_results/issue_1739/dv_dataset``), ``cwd`` = the resolved repo root,
   explicit ``env=`` (the subprocess-env rule) — and success once the fake git
   materializes the files;
3. fail-loud: files still absent after the cone-add → ``FileNotFoundError``
   naming the ``git sparse-checkout add`` remedy (the ``_assert_git_input``
   shape, #612);
4. ordering through the REAL ``phase_materialize_directions`` body: cone-ensure
   runs BEFORE ``store_io.load_rb_bank`` / ``_load_u_pool`` /
   ``_stream_labeled_context_acts`` (pre-fix: ``_ensure_parent_issue_cones``
   does not exist → this test FAILS at patch time).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import scripts.issue2220_readwrite as rw
from explore_persona_space.experiments.issue_1739 import store_io

BEHAVIORS = ("evil", "hallucination", "sycophancy")


def _seed_repo_root(tmp_path: Path, behaviors=BEHAVIORS) -> Path:
    """A fake repo root carrying the sentinel + the requested labeling files."""
    (tmp_path / "pyproject.toml").write_text("")
    for b in behaviors:
        p = tmp_path / rw._PARENT_INPUTS_CONE / b / "labeling.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    return tmp_path


def test_present_files_skip_git(tmp_path):
    root = _seed_repo_root(tmp_path)
    run_mock = mock.create_autospec(subprocess.run)
    with mock.patch.object(rw.subprocess, "run", run_mock):
        rw._ensure_parent_issue_cones(list(BEHAVIORS), repo_root=root)
    run_mock.assert_not_called()


def test_missing_files_run_git_cone_add_then_verify(tmp_path):
    root = _seed_repo_root(tmp_path, behaviors=())  # sentinel only, no inputs

    def fake_git(argv, **kwargs):
        # The cone-add materializes the files (what a real sparse-checkout
        # add does on a partial clone: blob fetch + checkout).
        _seed_repo_root(root)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    run_mock = mock.create_autospec(subprocess.run, side_effect=fake_git)
    with mock.patch.object(rw.subprocess, "run", run_mock):
        rw._ensure_parent_issue_cones(list(BEHAVIORS), repo_root=root)

    run_mock.assert_called_once()
    argv = run_mock.call_args[0][0]
    kwargs = run_mock.call_args[1]
    assert argv == ["git", "sparse-checkout", "add", rw._PARENT_INPUTS_CONE]
    assert Path(kwargs["cwd"]) == root
    assert "env" in kwargs  # explicit env= (subprocess-env rule)


def test_still_missing_after_cone_add_fails_loud(tmp_path):
    root = _seed_repo_root(tmp_path, behaviors=())  # sentinel only, no inputs

    def fake_git_noop(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="fatal: nope")

    run_mock = mock.create_autospec(subprocess.run, side_effect=fake_git_noop)
    with (
        mock.patch.object(rw.subprocess, "run", run_mock),
        pytest.raises(FileNotFoundError, match=r"sparse-checkout add"),
    ):
        rw._ensure_parent_issue_cones(["evil"], repo_root=root)


def test_real_git_sparse_checkout_add_materializes_inputs(tmp_path):
    """REAL-git integration probe: the helper's actual ``git sparse-checkout
    add`` branch, end-to-end against a hermetic local sparse clone whose
    checkout excludes ``eval_results/`` (the pod shape) — no mocks, no
    network. This is the closest CPU-scale demonstration of the fix-engaged
    branch (the shared VM's sparse worktree cannot materialize
    ``eval_results/`` in place)."""
    import os

    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)], env=env, check=True)
    _seed_repo_root(src)  # sentinel + the three committed labeling.json
    subprocess.run(["git", "-C", str(src), "add", "-A"], env=env, check=True)
    subprocess.run(["git", "-C", str(src), "commit", "-qm", "seed"], env=env, check=True)
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--no-checkout", str(src), str(clone)], env=env, check=True
    )
    # git 2.34 (the VM pin): `sparse-checkout set --cone <dir>` mis-parses
    # `--cone` as a literal pattern — the canonical 2.34 form is
    # `init --cone` THEN `set <dir>` (cone mode keeps root files present).
    subprocess.run(
        ["git", "-C", str(clone), "sparse-checkout", "init", "--cone"], env=env, check=True
    )
    subprocess.run(
        ["git", "-C", str(clone), "sparse-checkout", "set", "scripts"], env=env, check=True
    )
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "main"], env=env, check=True)
    # The pod shape: root files present (cone mode), eval_results/ absent.
    assert (clone / "pyproject.toml").is_file()
    assert not (clone / rw._PARENT_INPUTS_CONE).exists()

    rw._ensure_parent_issue_cones(list(BEHAVIORS), repo_root=clone)

    for b in BEHAVIORS:
        assert (clone / rw._PARENT_INPUTS_CONE / b / "labeling.json").is_file()


class _StopPhase(Exception):
    """Ends phase_materialize_directions after the ordering evidence exists."""


def test_cone_ensure_runs_before_first_labeling_read(tmp_path):
    args = argparse.Namespace(
        out_root=str(tmp_path / "out"),
        layers=[10],
        behaviors=["evil"],
        force=False,
        workers=2,
        window_mib=8,
        u_store_dir=str(tmp_path / "u_store"),
    )
    parent = mock.Mock()
    cone_mock = mock.create_autospec(rw._ensure_parent_issue_cones)
    rb_mock = mock.create_autospec(store_io.load_rb_bank)
    rb_mock.return_value = (np.zeros((28, 1, 4)), ["evil"])
    upool_mock = mock.create_autospec(rw._load_u_pool)
    upool_mock.return_value = {k: np.zeros((1, 2, 4)) for k in ("context_end", "prefix_end")}
    stream_mock = mock.create_autospec(rw._stream_labeled_context_acts, side_effect=_StopPhase)
    parent.attach_mock(cone_mock, "cone_ensure")
    parent.attach_mock(rb_mock, "load_rb_bank")
    parent.attach_mock(upool_mock, "load_u_pool")
    parent.attach_mock(stream_mock, "stream_labels")

    with (
        mock.patch.object(rw, "_ensure_parent_issue_cones", cone_mock),
        mock.patch.object(store_io, "load_rb_bank", rb_mock),
        mock.patch.object(rw, "_load_u_pool", upool_mock),
        mock.patch.object(rw, "_stream_labeled_context_acts", stream_mock),
        pytest.raises(_StopPhase),
    ):
        rw.phase_materialize_directions(args)

    names = [name for name, _a, _k in parent.mock_calls]
    assert "cone_ensure" in names, "fix not engaged: _ensure_parent_issue_cones never called"
    # Fail fast: the cone-ensure precedes the 8.5 GB U-pool stage, the r_B
    # fetch, AND the first labeling read (the round-5 crash site).
    assert names.index("cone_ensure") < names.index("load_rb_bank")
    assert names.index("cone_ensure") < names.index("load_u_pool")
    assert names.index("cone_ensure") < names.index("stream_labels")
    cone_mock.assert_called_once_with(["evil"])
