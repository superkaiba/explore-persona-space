"""Shared symlink/FIFO-safe lock-file opener for the advisory-flock lock sites.

Task #2324 (Gap 1): the raw ``os.open(path, O_WRONLY | O_CREAT, 0o600)`` /
``open(path, "wb")`` lock-open idiom (a) follows symlinks — a symlink at the
lock path flocks an unintended inode, silently breaking mutual exclusion —
and (b) blocks indefinitely in ``open(2)`` on a FIFO at the lock path,
bypassing the caller's advertised acquisition bound. This module is the one
shared fix; per-caller fail postures (fail-CLOSED single-flight abort vs
fail-OPEN degrade) stay in the CALLERS — the helper only raises, it never
decides.

Loaded by each caller through an explicit sibling-path
``importlib.util.spec_from_file_location`` block, NOT a bare
``import lock_utils``: the callers' test files spec-load the scripts by path
(``sys.path`` never gains ``scripts/``), and a ``src/explore_persona_space/``
home is unusable because ``step9c_baseline.py`` deliberately imports zero
package code (the #1022 pristine-run discipline). Stateless by design — each
caller execs its own module object, and every raiser/catcher pair lives
inside one caller's own loaded instance, so class-identity divergence across
copies is harmless.

Precedent: ``clean_experiment_downloads._open_scratch_regular`` (the
read-only, fail-toward-keep sibling); this is its write/lock-mode counterpart
with a typed rejection instead of ``None`` (``None`` is already overloaded to
mean "held elsewhere" at two call sites).
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat


class LockPathError(OSError):
    """Lock path rejected: symlink / FIFO / non-regular object at the path.

    ``.reason``: one of ``{"symlink", "would-block-special",
    "not-a-regular-file", "is-a-directory"}``; the message carries the path +
    originating errno. Subclasses ``OSError`` deliberately: a caller whose
    existing ``except OSError`` arm already maps open failures to a degraded
    mode (``codex_task._dispatch_lock``'s ``unavailable``) catches this with
    ZERO handler changes, preserving its fail-open posture by construction.
    """

    def __init__(
        self, reason: str, path: str | os.PathLike[str], errno_value: int | None = None
    ) -> None:
        detail = f", errno {errno_value}" if errno_value is not None else ""
        super().__init__(f"lock path rejected ({reason}): {path}{detail}")
        self.reason = reason
        self.lock_path = str(path)


_REJECTION_ERRNOS = {
    errno.ELOOP,  # symlink FINAL component under O_NOFOLLOW (Linux, macOS)
    errno.EMLINK,  # historic BSD O_NOFOLLOW spelling — defensive, no-op on Linux
    errno.ENXIO,  # FIFO with no reader under O_NONBLOCK|O_WRONLY; also unix socket
    errno.EAGAIN,  # special files under O_NONBLOCK; ALSO fires on a REGULAR file
    #               holding a conflicting F_SETLEASE lease (EWOULDBLOCK == EAGAIN on
    #               Linux). Disposition per the #2324 posture table: sites 1-3 turn it
    #               into a LOUD fail-CLOSED abort, never fail-open; site 4 degrades to
    #               `unavailable` — a same-user lease on these lock files sits outside
    #               the accidents-not-adversaries trust model (#2324 plan §12 A8), the
    #               same actor class as the planted-FIFO trigger itself.
    errno.EISDIR,  # directory at the lock path (O_WRONLY)
    errno.EOPNOTSUPP,  # socket open on some kernels — defensive
}


def _reason_for_errno(errno_value: int | None) -> str:
    """Map a rejection errno to the ``LockPathError.reason`` vocabulary."""
    if errno_value in (errno.ELOOP, errno.EMLINK):
        return "symlink"
    if errno_value == errno.EISDIR:
        return "is-a-directory"
    return "would-block-special"


def safe_open_lockfile(path: str | os.PathLike[str], mode: int = 0o600) -> int:
    """Open (creating if absent) a lock FILE, refusing a symlink at the FINAL
    path component and refusing to block on special-file open semantics.

    Scope of the two guarantees (deliberately narrow — do not overclaim):

    - ``O_NOFOLLOW`` rejects a symlink at the final component ONLY; a
      symlinked PARENT directory is still followed and this helper cannot
      detect it (out-of-scope residual, #2324 plan §12 A10).
    - ``O_NONBLOCK`` prevents the blocking OPEN semantics of special files
      (FIFO with no reader, device open, file lease); it is a NO-OP for
      ordinary regular-file opens on Linux, and a wedged/hung filesystem
      (dead FUSE/network mount) can still stall ``open(2)`` exactly as
      today — a pre-existing exposure this helper does not change.

    Returns an OPEN fd to a verified regular file (``O_NONBLOCK`` cleared).
    Raises :class:`LockPathError` when the path holds a
    symlink/FIFO/directory/device/socket (or a lease blocks the open). Every
    OTHER ``OSError`` (EACCES, ENOSPC, EROFS, ENOENT-parent, ...) propagates
    UNCHANGED so each caller's existing open-failure handling still applies.
    Fail postures stay in the CALLERS: this helper only raises — it never
    converts fail-CLOSED into fail-open or the reverse.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(path, flags, mode)
    except OSError as exc:
        if exc.errno in _REJECTION_ERRNOS:
            raise LockPathError(_reason_for_errno(exc.errno), path, exc.errno) from exc
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):  # FIFO-with-reader, /dev/null-class devices
            raise LockPathError("not-a-regular-file", path)
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl & ~os.O_NONBLOCK)  # hygiene; #2324 plan §12 A5
    except BaseException:
        os.close(fd)
        raise
    return fd
