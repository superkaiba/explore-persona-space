"""Shallow-clone branch of the fu4 item-(j) provenance-coherence gate (#1481
crash-fix 2).

In a ``--depth 1`` clone (the GCE startup shape) ``git log -1 -- <file>``
returns the single tip commit for EVERY file, so the date-based coherence
check reads every bank as "regenerated minutes ago" and false-crashes the
stage phase. Pins: (a) shallow detection fires on a real depth-1 clone;
(b) shallow + no recorded bank pin => WARN + ``"skipped-shallow-clone"``,
never a raise; (c) shallow + a recorded sha pin MISMATCH still raises (real
incoherence), a match passes; (d) a full-history repo keeps the pre-fix
date-check behavior verbatim (a fresh-dated bank raises).
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_fu4 as fu4  # noqa: E402


def _git(cwd: Path, *args: str, env: dict | None = None) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=env
    )
    return proc.stdout.strip()


@pytest.fixture()
def shallow_pair(tmp_path):
    """A tiny 2-commit source repo + a ``--depth 1`` clone of it.

    The two files are committed at DIFFERENT committer dates so the source
    repo carries a real per-file date spread the shallow clone collapses.
    """
    import os

    src = tmp_path / "src_repo"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "t")
    env_old = {
        **os.environ,
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    }
    (src / "bank_a.json").write_text('["q1"]')
    _git(src, "add", "bank_a.json")
    _git(src, "commit", "-q", "-m", "old bank", env=env_old)
    env_new = {
        **os.environ,
        "GIT_COMMITTER_DATE": "2026-07-01T00:00:00+00:00",
        "GIT_AUTHOR_DATE": "2026-07-01T00:00:00+00:00",
    }
    (src / "bank_b.json").write_text('["q2"]')
    _git(src, "add", "bank_b.json")
    _git(src, "commit", "-q", "-m", "tip commit", env=env_new)
    clone = tmp_path / "shallow_clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{src}", str(clone)],
        capture_output=True,
        text=True,
        check=True,
    )
    return src, clone


def _point_module_at(monkeypatch, repo_root: Path) -> None:
    """Rebind the module's repo root (``_SCRIPTS_DIR.parent``) to a tmp repo so
    ``_repo_is_shallow`` / ``_git_last_commit_iso`` / ``_bank_current_shas``
    read from it."""
    monkeypatch.setattr(fu4, "_SCRIPTS_DIR", repo_root / "scripts")


def test_shallow_detection_fires(shallow_pair, monkeypatch, tmp_path):
    """(a) ``_repo_is_shallow`` is True on a real depth-1 clone (HEAD is the
    graft — parentless), False on the full-history source repo, AND False on
    a shallow-MARKED repo whose HEAD parent is present (the shared VM
    checkout's shape: a stale ``.git/shallow`` beside full per-file history —
    the date check must keep running there)."""
    src, clone = shallow_pair
    _point_module_at(monkeypatch, clone)
    assert fu4._repo_is_shallow() is True
    _point_module_at(monkeypatch, src)
    assert fu4._repo_is_shallow() is False
    depth2 = tmp_path / "depth2_clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "2", f"file://{src}", str(depth2)],
        capture_output=True,
        text=True,
        check=True,
    )
    _point_module_at(monkeypatch, depth2)
    assert fu4._repo_is_shallow() is False  # shallow-marked, HEAD^ present


def test_shallow_clone_collapses_last_commit_dates(shallow_pair, monkeypatch):
    """Root-cause mechanism: in the depth-1 clone every file's last-commit
    date is the TIP's, while the full source repo keeps the real spread."""
    src, clone = shallow_pair
    _point_module_at(monkeypatch, clone)
    a_shallow = fu4._git_last_commit_iso("bank_a.json")
    b_shallow = fu4._git_last_commit_iso("bank_b.json")
    assert a_shallow == b_shallow  # collapsed to the tip
    _point_module_at(monkeypatch, src)
    assert fu4._git_last_commit_iso("bank_a.json") != fu4._git_last_commit_iso("bank_b.json")


def test_shallow_no_pin_warns_and_skips(shallow_pair, monkeypatch, caplog):
    """(b) Shallow clone + no recorded bank pin: no raise — WARN with the
    fix-engaged signal text + the recorded ``"skipped-shallow-clone"`` status.
    Auto-detection (``shallow=None``) drives the branch, as in production."""
    _, clone = shallow_pair
    _point_module_at(monkeypatch, clone)
    tip_date = fu4._git_last_commit_iso("bank_a.json")
    # A mix frozen BEFORE the tip: the pre-fix date check would raise here.
    prov = {"m/train_mix.jsonl": {"oid": "x", "date": "2026-03-01T00:00:00+00:00"}}
    with caplog.at_level(logging.WARNING, logger="issue1090.fu4"):
        status = fu4._assert_provenance_coherent("imp-icl", "bank_a.json", tip_date, prov)
    assert status == "skipped-shallow-clone"
    assert any(
        "provenance date check SKIPPED: shallow clone" in r.getMessage() for r in caplog.records
    )


def test_shallow_sha_pin_mismatch_still_raises(shallow_pair, monkeypatch):
    """(c) Shallow clone + recorded pin: a mismatch is REAL incoherence and
    still raises; a matching raw-file sha pin passes as ``sha-pin-checked``."""
    _, clone = shallow_pair
    _point_module_at(monkeypatch, clone)
    prov = {"m/train_mix.jsonl": {"oid": "x", "date": "2026-03-01T00:00:00+00:00"}}
    tip_date = fu4._git_last_commit_iso("bank_a.json")
    with pytest.raises(ValueError, match="provenance coherence FAILED"):
        fu4._assert_provenance_coherent(
            "imp-icl", "bank_a.json", tip_date, prov, bank_pin_sha256="0" * 64
        )
    good_pin = hashlib.sha256((clone / "bank_a.json").read_bytes()).hexdigest()
    status = fu4._assert_provenance_coherent(
        "imp-icl", "bank_a.json", tip_date, prov, bank_pin_sha256=good_pin
    )
    assert status == "sha-pin-checked"


def test_full_clone_date_check_unchanged(shallow_pair, monkeypatch):
    """(d) Full-history repo: the pre-fix date-check behavior verbatim — a
    bank whose last commit postdates the mix raises; a predating one passes
    (returning None, so full-history manifests gain no new field)."""
    src, _ = shallow_pair
    _point_module_at(monkeypatch, src)
    tip_date = fu4._git_last_commit_iso("bank_b.json")  # 2026-07-01
    stale_mix = {"m/train_mix.jsonl": {"oid": "x", "date": "2026-03-01T00:00:00+00:00"}}
    with pytest.raises(ValueError, match="postdates the mix"):
        fu4._assert_provenance_coherent("imp-icl", "bank_b.json", tip_date, stale_mix)
    fresh_mix = {"m/train_mix.jsonl": {"oid": "x", "date": "2026-07-02T00:00:00+00:00"}}
    assert fu4._assert_provenance_coherent("imp-icl", "bank_b.json", tip_date, fresh_mix) is None
    # Explicit shallow=False pins the branch independent of auto-detection.
    with pytest.raises(ValueError, match="postdates the mix"):
        fu4._assert_provenance_coherent(
            "imp-icl", "bank_b.json", tip_date, stale_mix, shallow=False
        )
