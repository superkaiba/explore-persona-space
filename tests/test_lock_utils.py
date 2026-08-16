"""Tests for scripts/lock_utils.py — the shared symlink/FIFO-safe lock opener (#2324).

Posture-only unit arms: symlink targets here are legitimately regular or
dangling. The bound-discriminating symlink→FIFO CHILD-process arms (plan §6
child matrix) live with their call sites — tests/test_sync_repo_root.py
(sites 1-2) and tests/test_step9c_baseline.py (site 3) — because only a
blocking target makes the child's timeout bound a discriminating assertion
against the pre-fix code.
"""

from __future__ import annotations

import fcntl
import importlib.util
import os
import stat
import sys
import time
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parents[1] / "scripts" / "lock_utils.py"
_spec = importlib.util.spec_from_file_location("lock_utils_under_test", _HELPER)
assert _spec and _spec.loader
lu = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lu
_spec.loader.exec_module(lu)

requires_mkfifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX mkfifo")


# --- Arm 1: create-new -----------------------------------------------------------


def test_create_new_returns_regular_flockable_fd(tmp_path: Path):
    path = tmp_path / "new.lock"
    fd = lu.safe_open_lockfile(path)
    try:
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # flock-able
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# --- Arm 2: reopen-existing ------------------------------------------------------


def test_reopen_existing_regular_file_ok(tmp_path: Path):
    path = tmp_path / "existing.lock"
    path.write_bytes(b"")
    fd = lu.safe_open_lockfile(path)
    try:
        assert stat.S_ISREG(os.fstat(fd).st_mode)
    finally:
        os.close(fd)


# --- Arm 3: symlink -> regular file (posture-only) --------------------------------


def test_symlink_to_regular_file_rejected(tmp_path: Path):
    target = tmp_path / "target"
    target.write_bytes(b"")
    link = tmp_path / "planted.lock"
    os.symlink(target, link)
    with pytest.raises(lu.LockPathError) as ei:
        lu.safe_open_lockfile(link)
    assert ei.value.reason == "symlink"
    assert isinstance(ei.value, OSError)  # site-4 fail-open catch relies on this


# --- Arm 4: dangling symlink (O_CREAT does NOT create through) --------------------


def test_dangling_symlink_rejected_no_create_through(tmp_path: Path):
    missing = tmp_path / "nonexistent-target"
    link = tmp_path / "planted.lock"
    os.symlink(missing, link)
    with pytest.raises(lu.LockPathError) as ei:
        lu.safe_open_lockfile(link)
    assert ei.value.reason == "symlink"
    assert not missing.exists()  # O_CREAT|O_NOFOLLOW did not create through the link


# --- Arm 5: FIFO with no reader (immediate ENXIO, never a blocking open) ----------


@requires_mkfifo
def test_fifo_no_reader_rejected_immediately(tmp_path: Path):
    fifo = tmp_path / "planted.lock"
    os.mkfifo(fifo)
    t0 = time.monotonic()
    with pytest.raises(lu.LockPathError) as ei:
        lu.safe_open_lockfile(fifo)
    assert time.monotonic() - t0 < 5.0  # bounded: the pre-fix idiom blocks forever here
    assert ei.value.reason == "would-block-special"


# --- Arm 6: FIFO WITH a reader (open succeeds, fstat rejects) ----------------------


@requires_mkfifo
def test_fifo_with_reader_rejected_by_fstat(tmp_path: Path):
    fifo = tmp_path / "planted.lock"
    os.mkfifo(fifo)
    rd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)  # the test holds the read end open
    try:
        with pytest.raises(lu.LockPathError) as ei:
            lu.safe_open_lockfile(fifo)
        assert ei.value.reason == "not-a-regular-file"
    finally:
        os.close(rd)


# --- Arm 7: directory at the lock path ---------------------------------------------


def test_directory_rejected(tmp_path: Path):
    d = tmp_path / "planted.lock"
    d.mkdir()
    with pytest.raises(lu.LockPathError) as ei:
        lu.safe_open_lockfile(d)
    assert ei.value.reason == "is-a-directory"


# --- Arm 8: O_NONBLOCK cleared on the returned fd ----------------------------------


def test_o_nonblock_cleared_on_returned_fd(tmp_path: Path):
    fd = lu.safe_open_lockfile(tmp_path / "new.lock")
    try:
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_NONBLOCK == 0
    finally:
        os.close(fd)


# --- Arm 9: non-rejection OSError propagates unchanged -----------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission checks")
def test_eacces_propagates_as_plain_oserror_not_lockpatherror(tmp_path: Path):
    parent = tmp_path / "noperm"
    parent.mkdir()
    parent.chmod(0o000)
    try:
        with pytest.raises(PermissionError) as ei:
            lu.safe_open_lockfile(parent / "x.lock")
        assert not isinstance(ei.value, lu.LockPathError)
    finally:
        parent.chmod(0o755)  # let pytest's tmp_path cleanup remove the dir
