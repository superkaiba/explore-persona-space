"""``SlurmBackend.fetch_results`` two-phase atomic staging pull (#1973).

Pins the atomicity contract that replaced the direct-in-place
``rsync --partial`` pull (incident #1768 r3: an interrupted pull stranded
a 4.7 GB partial tree directly under the live ``eval_results/`` while
finalize reported ok):

* happy path — staged files (dotfile sentinel included) land in the live
  tree via the phase-2 merge; staging is removed.
* interrupted pull — typed ``FetchResultsError``; the live tree gains NO
  new visible/complete-named files; staging (with the confined
  ``.rsync-partial/`` content) is retained for resume.
* partial-dir exclusion (critic Must-Fix pin) — a truncated
  ``.rsync-partial/`` file NEVER reaches the live tree while complete
  staged siblings do.
* benign-absent-with-merge (#598 contract) — rc 23 + "No such file or
  directory" stays warn-only AND the merge still runs on whatever staged
  (the completion sentinel is never stranded in staging).
* timeout — ``subprocess.TimeoutExpired`` becomes ``FetchResultsError``;
  live tree untouched.
* env fence — ``EPS_SLURM_FETCH_TIMEOUT_SECONDS`` threads the timeout
  kwarg to BOTH rsync phases.

The network pull (phase 1) is faked by monkeypatching ``subprocess.run``
at the ``slurm`` module; the LOCAL merge (phase 2, ``--remove-source-files``
in the argv) delegates to the REAL ``subprocess.run`` + real rsync so the
``--exclude=.rsync-partial*/`` / temp+rename semantics are genuinely
exercised (a faked merge would make the Must-Fix pin hollow).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import explore_persona_space.backends.slurm as slurm
from explore_persona_space.backends.base import FetchResultsError, RunHandle
from explore_persona_space.backends.slurm import SlurmBackend

_REAL_RUN = subprocess.run

requires_rsync = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="phase-2 merge pin needs a real rsync binary"
)


def _handle(issue: int = 1973) -> RunHandle:
    """A SLURM-lane handle shaped like the launch path writes it."""
    return RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="job-fetch",
        pod_name=f"pod-{issue}",
        scratch_dir=f"/scratch/eps/issue-{issue}",
        log_path="/log",
        extra={"issue": issue, "intent": "lora-7b"},
    )


def _install_fake_pull(monkeypatch, pull_fn, calls: list | None = None) -> None:
    """Monkeypatch ``subprocess.run`` at the slurm module.

    Phase-1 network pulls (no ``--remove-source-files``) route to
    ``pull_fn``; phase-2 local merges delegate to the REAL
    ``subprocess.run`` so real rsync semantics are exercised.
    """

    def fake_run(argv, **kwargs):
        if calls is not None:
            calls.append((list(argv), dict(kwargs)))
        if "--remove-source-files" in argv:
            return _REAL_RUN(argv, **kwargs)
        return pull_fn(argv, **kwargs)

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)


def _staging_dir(tmp_path: Path, subdir: str, issue: int = 1973) -> Path:
    return tmp_path / ".slurm-results-staging" / f"issue-{issue}" / subdir


@requires_rsync
def test_happy_path_files_land_in_live_tree_and_staging_removed(monkeypatch, tmp_path) -> None:
    """rc-0 pulls stage files (dotfile sentinel included); the merge lands
    them in the live tree and the staging subdirs are removed."""
    backend = SlurmBackend(src_root=tmp_path)

    def pull(argv, **kwargs):
        dst = Path(argv[-1])
        dst.mkdir(parents=True, exist_ok=True)
        if dst.name == "eval_results":
            d = dst / "issue_1973" / "slurm-job-fetch"
            d.mkdir(parents=True)
            (d / ".completion-sentinel.json").write_text("{}")
            (d / "summary.json").write_text('{"ok": true}')
        else:
            (dst / "fig.png").write_bytes(b"png-bytes")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    _install_fake_pull(monkeypatch, pull)
    backend.fetch_results(_handle())

    live = tmp_path / "eval_results" / "issue_1973" / "slurm-job-fetch"
    assert (live / ".completion-sentinel.json").is_file(), "sentinel must land in the LIVE tree"
    assert (live / "summary.json").is_file()
    assert (tmp_path / "figures" / "fig.png").is_file()
    assert not _staging_dir(tmp_path, "eval_results").exists(), "staging removed on success"
    assert not _staging_dir(tmp_path, "figures").exists()


def test_interrupted_pull_raises_live_tree_untouched_staging_retained(
    monkeypatch, tmp_path
) -> None:
    """A transfer-failure rc (30) raises the typed error; NO new
    visible/complete-named file reaches the live tree (partials stay
    confined under staging's ``.rsync-partial/`` for resume)."""
    backend = SlurmBackend(src_root=tmp_path)

    def pull(argv, **kwargs):
        dst = Path(argv[-1])
        partial = dst / ".rsync-partial"
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "big_results.json").write_bytes(b'{"truncated_at_byte_')
        return subprocess.CompletedProcess(
            argv,
            30,
            b"",
            b"rsync error: timeout in data send/receive -- connection unexpectedly "
            b"closed (4952148341 bytes received so far)",
        )

    _install_fake_pull(monkeypatch, pull)
    with pytest.raises(FetchResultsError, match="exited 30"):
        backend.fetch_results(_handle())

    live = tmp_path / "eval_results"
    live_files = [p for p in live.rglob("*") if p.is_file()] if live.exists() else []
    assert live_files == [], "an interrupted pull must not land ANY file in the live tree"
    retained = _staging_dir(tmp_path, "eval_results") / ".rsync-partial" / "big_results.json"
    assert retained.is_file(), "staging (confined partial) is KEPT for resume"


@requires_rsync
def test_partial_dir_content_never_reaches_live_tree(monkeypatch, tmp_path) -> None:
    """Must-Fix pin: across a finalize RETRY, a truncated file confined
    under staging's ``.rsync-partial/`` is NEVER merged into the live
    tree, while complete staged siblings ARE."""
    backend = SlurmBackend(src_root=tmp_path)
    staged = _staging_dir(tmp_path, "eval_results")
    (staged / ".rsync-partial").mkdir(parents=True)
    (staged / ".rsync-partial" / "truncated.json").write_bytes(b'{"half')
    (staged / "complete.json").write_text('{"ok": true}')

    def pull(argv, **kwargs):
        # Retry pull: nothing new to transfer (staging pre-seeded).
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    _install_fake_pull(monkeypatch, pull)
    backend.fetch_results(_handle())

    assert (tmp_path / "eval_results" / "complete.json").is_file()
    live = tmp_path / "eval_results"
    assert not (live / ".rsync-partial").exists(), "partial-dir must be excluded from the merge"
    assert list(live.rglob("truncated.json")) == [], "truncated content must never go live"


@requires_rsync
def test_benign_absent_rc23_warn_only_and_merge_still_runs(monkeypatch, tmp_path) -> None:
    """#598 contract: rc 23 + 'No such file or directory' does NOT raise —
    and the phase-2 merge STILL runs, so files a mixed pull landed in
    staging (the completion sentinel included) reach the live tree."""
    backend = SlurmBackend(src_root=tmp_path)

    def pull(argv, **kwargs):
        dst = Path(argv[-1])
        dst.mkdir(parents=True, exist_ok=True)
        if dst.name == "eval_results":
            d = dst / "issue_1973" / "slurm-job-fetch"
            d.mkdir(parents=True)
            (d / ".completion-sentinel.json").write_text("{}")
        return subprocess.CompletedProcess(
            argv,
            23,
            b"",
            b'rsync: [sender] link_stat "/scratch/eps/issue-1973/figures" failed: '
            b"No such file or directory (2)",
        )

    _install_fake_pull(monkeypatch, pull)
    backend.fetch_results(_handle())  # must NOT raise

    sentinel = (
        tmp_path / "eval_results" / "issue_1973" / "slurm-job-fetch" / ".completion-sentinel.json"
    )
    assert sentinel.is_file(), "the sentinel must never be stranded in staging on benign-absent"


def test_timeout_raises_fetch_results_error_live_tree_untouched(monkeypatch, tmp_path) -> None:
    """A pull that exceeds the fence raises the typed error; the live
    tree is untouched."""
    backend = SlurmBackend(src_root=tmp_path)

    def pull(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    _install_fake_pull(monkeypatch, pull)
    with pytest.raises(FetchResultsError, match="fence"):
        backend.fetch_results(_handle())
    assert not (tmp_path / "eval_results").exists()
    assert not (tmp_path / "figures").exists()


def test_env_fence_threads_timeout_to_both_phases(monkeypatch, tmp_path) -> None:
    """``EPS_SLURM_FETCH_TIMEOUT_SECONDS=7`` reaches the ``timeout=``
    kwarg of BOTH the network pull and the local merge."""
    monkeypatch.setenv("EPS_SLURM_FETCH_TIMEOUT_SECONDS", "7")
    backend = SlurmBackend(src_root=tmp_path)
    calls: list[tuple[list, dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        if "--remove-source-files" not in argv:
            dst = Path(argv[-1])
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "f.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    backend.fetch_results(_handle())

    assert calls, "no rsync invocations recorded"
    assert all(kw.get("timeout") == 7 for _argv, kw in calls), (
        "every rsync phase must carry the env-fenced timeout: "
        + repr([(a[0:1], kw.get("timeout")) for a, kw in calls])
    )
    assert any("--partial-dir=.rsync-partial" in a for a, _ in calls), "phase-1 pull missing"
    assert any("--remove-source-files" in a for a, _ in calls), "phase-2 merge missing"


def test_default_fence_is_1800_not_the_old_300(monkeypatch, tmp_path) -> None:
    """The resolved #1768 root cause was the flat 300 s fence — the
    default must be the generous 1800 s, threaded to the pull."""
    monkeypatch.delenv("EPS_SLURM_FETCH_TIMEOUT_SECONDS", raising=False)
    backend = SlurmBackend(src_root=tmp_path)
    seen: list[int | None] = []

    def fake_run(argv, **kwargs):
        seen.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    backend.fetch_results(_handle())
    assert seen and all(t == 1800 for t in seen)
