"""Regression tests for ``pod_config.locked_pods_conf`` (task #488 race).

Bug. Two concurrent ``/issue`` sessions each called
``pod_lifecycle._upsert_pods_conf`` for their own pod. Session B's
``parse → mutate → write`` interleaved with session A's, producing a
lost update — A's row disappeared from ``pods.conf`` and the regenerated
``~/.ssh/config`` dropped ``Host pod-<A>``. ``poll_pipeline.py`` then
failed with ``ssh: Could not resolve hostname pod-<A>: Temporary failure
in name resolution`` for a perfectly healthy run.

Fix. Every read-modify-write on ``pods.conf`` (and the downstream
``cmd_sync``) runs inside :func:`pod_config.locked_pods_conf`, an
``fcntl.flock``-based advisory lock on a sibling lockfile in the same
main-repo ``scripts/`` directory.

Tests below exercise:

1. ``locked_pods_conf`` is reentrant-safe across cooperative usage by a
   single process (no deadlock).
2. The lock serialises two ``multiprocessing.Process`` workers: when both
   upsert their own pod into a shared ``pods.conf``, BOTH rows survive.
3. The lock auto-releases on process death — a worker that holds the lock
   and exits without explicitly releasing it does not block subsequent
   acquirers (covered implicitly by the multiprocessing test, which
   relies on this behavior at process exit).

The multiprocessing test is the load-bearing regression. The in-process
test below it documents the serialisation contract a single process
relies on.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _worker_upsert(
    pods_conf_path_str: str,
    lock_path_str: str,
    pod_name: str,
    host: str,
    port: int,
    hold_ms: int,
) -> None:
    """Subprocess body: acquire ``locked_pods_conf``, read pods.conf, append
    the new pod row, sleep briefly to widen the race window, then write
    back. Mirrors :func:`pod_lifecycle._upsert_pods_conf` without the
    downstream ``cmd_sync`` (out of test scope; ``~/.ssh/config`` is
    untouched).
    """
    import pod_config  # type: ignore[import-not-found]

    pods_conf_path = Path(pods_conf_path_str)
    # Repoint the module-level paths into the test fixture so the workers
    # share the SAME on-disk pods.conf + lockfile. ``parse_pods_conf`` /
    # ``write_pods_conf`` capture ``PODS_CONF`` at function-def time as a
    # default argument, so we ALSO pass ``path=`` explicitly below.
    pod_config.PODS_CONF = pods_conf_path
    pod_config.PODS_CONF_LOCK = Path(lock_path_str)

    from pod_config import Pod, locked_pods_conf, parse_pods_conf, write_pods_conf

    with locked_pods_conf():
        rows = parse_pods_conf(path=pods_conf_path)
        # Widen the race window: a pre-fix racy implementation would
        # interleave both workers' read-then-write inside this sleep and
        # the later writer would clobber the earlier writer's row.
        time.sleep(hold_ms / 1000.0)
        rows.append(
            Pod(
                name=pod_name,
                host=host,
                port=port,
                gpus=1,
                gpu_type="H100",
                label=f"thomas-{pod_name}",
            )
        )
        write_pods_conf(rows, path=pods_conf_path)


def test_locked_pods_conf_serialises_concurrent_upserts(tmp_path: Path) -> None:
    """Two concurrent workers upserting their own pod must both survive in
    pods.conf. Pre-fix, the second writer's parse would predate the first
    writer's write and clobber its row.

    Test rig: start both processes, let each hold the lock for ~200 ms
    inside the critical section to widen the window, join, then read the
    file. Both ``pod-test-A`` and ``pod-test-B`` must be present.
    """
    pods_conf = tmp_path / "pods.conf"
    pods_conf.write_text(
        "# Pod registry -- test fixture\n# Format: name host port gpus gpu_type label\n"
    )
    lock_path = tmp_path / ".pods.conf.lock"

    # ``spawn`` (not the default ``fork``) so each worker re-imports
    # ``pod_config`` cleanly and reads the test-injected paths via the
    # function arguments. ``fork`` would share the parent's already-loaded
    # module with its original constants.
    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(
            target=_worker_upsert,
            args=(str(pods_conf), str(lock_path), "pod-test-A", "10.0.0.1", 22001, 200),
        ),
        ctx.Process(
            target=_worker_upsert,
            args=(str(pods_conf), str(lock_path), "pod-test-B", "10.0.0.2", 22002, 200),
        ),
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)
        assert w.exitcode == 0, f"worker {w.name} exited {w.exitcode}"

    final = pods_conf.read_text()
    assert "pod-test-A" in final, (
        "pod-test-A lost from pods.conf — concurrent writers raced and one "
        "wrote a stale snapshot back. locked_pods_conf is not serialising."
    )
    assert "pod-test-B" in final, (
        "pod-test-B lost from pods.conf — concurrent writers raced and one "
        "wrote a stale snapshot back. locked_pods_conf is not serialising."
    )
    # Sanity: each pod appears as exactly ONE row (no duplicate insertion
    # on a retry path). Each row starts the name as the first column at
    # the start of a line; ``thomas-pod-test-A`` in the label column would
    # otherwise inflate a naive substring count.
    name_a_rows = [ln for ln in final.splitlines() if ln.startswith("pod-test-A")]
    name_b_rows = [ln for ln in final.splitlines() if ln.startswith("pod-test-B")]
    assert len(name_a_rows) == 1, name_a_rows
    assert len(name_b_rows) == 1, name_b_rows


def test_locked_pods_conf_context_manager_creates_lockfile(tmp_path, monkeypatch) -> None:
    """The context manager creates the lockfile on first use (so a clean
    checkout doesn't need a manual ``touch`` step).
    """
    import pod_config

    lock_path = tmp_path / ".pods.conf.lock"
    assert not lock_path.exists()
    monkeypatch.setattr(pod_config, "PODS_CONF_LOCK", lock_path)

    with pod_config.locked_pods_conf():
        pass

    assert lock_path.exists()
    # Lockfile should be empty; it's used purely for fcntl semantics, not
    # to store state.
    assert lock_path.read_text() == ""


def test_locked_pods_conf_releases_on_exception(tmp_path, monkeypatch) -> None:
    """If the body raises, the lock must be released so a subsequent
    acquirer in the same process doesn't deadlock.
    """
    import pod_config

    lock_path = tmp_path / ".pods.conf.lock"
    monkeypatch.setattr(pod_config, "PODS_CONF_LOCK", lock_path)

    class _Boom(RuntimeError):
        pass

    try:
        with pod_config.locked_pods_conf():
            raise _Boom("synthetic failure inside critical section")
    except _Boom:
        pass

    # Re-acquire immediately — must not block.
    with pod_config.locked_pods_conf():
        pass


# ---------------------------------------------------------------------------
# Task #821 v3 r2 — reentrant ``locked_pods_conf`` (nested-flock deadlock).
#
# Round-1 codex-reviewer critical blocker: ``_resolve_live_pods_conf``
# acquires ``locked_pods_conf`` for the seed→live migration; every
# production writer (``pod_lifecycle._upsert_pods_conf``,
# ``_remove_from_pods_conf``, ``cmd_update``, ``cmd_refresh_from_api``)
# ALREADY holds the lock when it calls ``parse_pods_conf`` /
# ``write_pods_conf``, which lazily resolve. ``fcntl.flock`` is
# per-open-file-description, so a second ``LOCK_EX`` on a fresh fd
# BLOCKS forever on the fd the outer frame is holding. Fix: reentrant
# via a ``threading.local()`` depth counter (see ``pod_config.py``).
#
# The two tests below use ``multiprocessing.Process`` with a hard timeout
# so a regression NEVER hangs the test suite — the join times out and
# the assertion fires on ``exitcode``.
# ---------------------------------------------------------------------------


def _worker_nested_acquire(pods_conf_path_str: str, lock_path_str: str) -> None:
    """Subprocess body: two ``with locked_pods_conf()`` frames stacked
    directly. Pre-fix (non-reentrant flock) this hangs on the inner
    acquire because ``fcntl.flock`` blocks on a second fd against the
    same lockfile even inside the same process.
    """
    import pod_config

    pod_config.PODS_CONF = Path(pods_conf_path_str)
    pod_config.PODS_CONF_LOCK = Path(lock_path_str)

    with pod_config.locked_pods_conf():
        with pod_config.locked_pods_conf():
            # Sanity: the nested body actually ran (not just skipped).
            assert getattr(pod_config._LOCK_STATE, "depth", 0) == 2
        assert getattr(pod_config._LOCK_STATE, "depth", 0) == 1
    assert getattr(pod_config._LOCK_STATE, "depth", 0) == 0


def test_locked_pods_conf_is_reentrant_direct_nesting(tmp_path: Path) -> None:
    """A direct ``with locked_pods_conf(): with locked_pods_conf(): ...``
    stack MUST return without deadlock. Pre-fix the inner frame's
    ``flock(fd2, LOCK_EX)`` blocked on the outer frame's fd1.

    Ran in a subprocess with a 15 s hard timeout so a regression fails the
    assertion instead of hanging the pytest run.
    """
    pods_conf = tmp_path / "pods.conf"
    pods_conf.write_text("# fixture\n")
    lock_path = tmp_path / ".pods.conf.lock"

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_worker_nested_acquire, args=(str(pods_conf), str(lock_path)))
    proc.start()
    proc.join(timeout=15)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    assert not proc.is_alive(), (
        "locked_pods_conf deadlocked on nested acquire — the reentrant "
        "depth-counter fix has regressed."
    )
    assert proc.exitcode == 0, f"nested-acquire worker exited {proc.exitcode}; expected 0"


def _worker_first_use_migration_under_lock(
    seed_path_str: str,
    live_dir_str: str,
    lock_path_str: str,
) -> None:
    """Subprocess body: reproduces the exact orchestrator repro from the
    round-1 codex FAIL — acquire ``locked_pods_conf`` at the top of the
    caller's read-modify-write, then call ``parse_pods_conf`` on a fresh
    checkout where only the SEED exists. ``parse_pods_conf`` calls
    ``_resolve_live_pods_conf`` (no live file yet → migrates seed → live),
    which itself acquires ``locked_pods_conf`` inside the same process.

    Pre-fix: the inner acquire's ``flock(fd2, LOCK_EX)`` blocks on the
    outer fd1 → hang → the ``multiprocessing.Process.join(timeout=…)``
    fires and the exitcode assertion FAILs.

    Post-fix: the reentrant depth counter skips the inner ``flock``, the
    migration copies seed → live, ``parse_pods_conf`` reads the seed's
    rows, the worker exits 0 with the seed row present.
    """
    import pod_config

    seed_path = Path(seed_path_str)
    live_dir = Path(live_dir_str)
    live_path = live_dir / "pods.conf"

    # Point PODS_CONF at the seed so the monkeypatch-passthrough branch
    # in ``_resolve_live_pods_conf`` does NOT fire (it fires only when
    # PODS_CONF != PODS_CONF_SEED). We want the real migration path.
    pod_config.PODS_CONF = pod_config.PODS_CONF_SEED
    pod_config.PODS_CONF_SEED = seed_path
    pod_config.PODS_CONF = seed_path  # keep both aligned
    pod_config.PODS_CONF_LOCK = Path(lock_path_str)

    # Redirect ``_git_common_dir`` to a temp dir so the migration's live
    # dir lands inside our fixture, not the real repo's ``.git/eps/``.
    def _fake_common_dir() -> Path:
        return live_dir.parent

    pod_config._git_common_dir = _fake_common_dir  # type: ignore[assignment]
    # ``_LIVE_PODS_CONF_DIRNAME`` = "eps" — the migration writes to
    # ``<common>/eps/pods.conf``. Point ``live_dir.parent / "eps"`` at
    # ``live_dir`` by matching the name.
    assert live_dir.name == pod_config._LIVE_PODS_CONF_DIRNAME, live_dir.name

    with pod_config.locked_pods_conf():
        rows = pod_config.parse_pods_conf()

    # Live file must exist post-migration.
    assert live_path.exists(), f"migration did not create {live_path}"
    # The seed row must be present in the parsed output.
    assert any(p.name == "pod-42" for p in rows), (
        f"expected pod-42 in parsed rows, got {[p.name for p in rows]}"
    )


def test_locked_pods_conf_first_use_migration_under_caller_lock(tmp_path: Path) -> None:
    """The orchestrator's exact reproduction: caller holds the lock, then
    ``parse_pods_conf`` lazily resolves and migrates seed → live inside
    ``_resolve_live_pods_conf`` — which itself acquires the lock. Pre-fix
    this is the round-1 nested-flock deadlock.

    Fixture layout:
        tmp_path/scripts/pods.conf        <- SEED (only file that exists pre-run)
        tmp_path/eps/                     <- LIVE dir (created by migration)
        tmp_path/.pods.conf.lock          <- lockfile

    Ran in a subprocess with a 15 s hard timeout so a regression FAILs
    the assertion instead of hanging.
    """
    seed_dir = tmp_path / "scripts"
    seed_dir.mkdir()
    seed = seed_dir / "pods.conf"
    seed.write_text(
        "# fixture seed\n"
        "# name  host  port  gpus  gpu_type  label\n"
        "pod-42  10.0.0.42  22042  1  H100  thomas-pod-42\n"
    )

    live_dir = tmp_path / "eps"  # matches _LIVE_PODS_CONF_DIRNAME
    lock_path = tmp_path / ".pods.conf.lock"

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_worker_first_use_migration_under_lock,
        args=(str(seed), str(live_dir), str(lock_path)),
    )
    proc.start()
    proc.join(timeout=15)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    assert not proc.is_alive(), (
        "First-use migration deadlocked under caller-held lock — the "
        "nested-flock bug (round-1 codex-reviewer critical blocker) has "
        "regressed. Non-reentrant flock: outer acquire owns fd1, "
        "_resolve_live_pods_conf opens fd2 and blocks forever on LOCK_EX."
    )
    assert proc.exitcode == 0, f"migration-under-lock worker exited {proc.exitcode}; expected 0"
