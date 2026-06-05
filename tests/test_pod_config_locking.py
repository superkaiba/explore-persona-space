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
