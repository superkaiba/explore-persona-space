"""Task #831: no-arg ``cmd_sync`` re-read-under-lock + atomic SSH/MCP writes.

Bug (incident #813, 2026-07-02). ``pod_config.main()`` parsed ``pods.conf``
with NO lock and handed that snapshot to ``cmd_sync(pods)`` — a concurrent
session's upsert landing between the parse and the sync produced a
``~/.ssh/config`` managed-block rewrite that lacked the new row. ``Host
pod-813`` was dropped twice, yielding false ``dead`` poll verdicts for a
healthy pod. ``update_ssh_config`` / ``update_mcp_config`` additionally
wrote via bare ``Path.write_text`` (torn-write hazard).

Fix (task #831). ``cmd_sync()`` takes no argument: it acquires
``locked_pods_conf`` (reentrant) and RE-READS ``pods.conf`` under the lock,
so every present and future caller operates on canonical on-disk state.
Both downstream writes go through ``_atomic_write_text`` (same-dir tmp +
``os.replace``; mode preserved, 0600 on create).

Rig mirrors ``test_pod_config_locking.py`` (task #488): ``sys.path`` insert
of ``scripts/``, module-global repointing of ``PODS_CONF`` /
``PODS_CONF_LOCK`` / ``SSH_CONFIG`` / ``MCP_JSON`` into ``tmp_path``, and
``multiprocessing`` ``spawn``-context workers that re-point the paths
in-process (module attrs do NOT cross process boundaries). Tests never
touch the real ``~/.ssh/config``, ``~/.claude/mcp.json``, or live
``pods.conf``; the #831 sync audit log derives from the resolved live
pods.conf path, so the repoint redirects it into ``tmp_path`` too.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402
from pod_config import Pod  # noqa: E402

MCP_SEED = {"mcpServers": {"ssh": {"env": {}}}}


def _pod(name: str, host: str = "10.0.0.1", port: int = 22001) -> Pod:
    """Build a minimal valid Pod row for fixtures."""
    return Pod(name=name, host=host, port=port, gpus=1, gpu_type="H100", label=f"thomas-{name}")


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Repoint every pod_config path global into ``tmp_path``.

    Seeds an empty-but-valid ``pods.conf`` (header only) and a minimal
    ``mcp.json`` carrying the required ``mcpServers.ssh`` entry.
    """
    paths = {
        "pods_conf": tmp_path / "pods.conf",
        "lock": tmp_path / ".pods.conf.lock",
        "ssh_config": tmp_path / "ssh" / "config",
        "mcp_json": tmp_path / "mcp.json",
    }
    paths["pods_conf"].write_text(
        "# Pod registry -- test fixture\n# Format: name host port gpus gpu_type label\n"
    )
    paths["mcp_json"].write_text(json.dumps(MCP_SEED, indent=2) + "\n")
    monkeypatch.setattr(pod_config, "PODS_CONF", paths["pods_conf"])
    monkeypatch.setattr(pod_config, "PODS_CONF_LOCK", paths["lock"])
    monkeypatch.setattr(pod_config, "SSH_CONFIG", paths["ssh_config"])
    monkeypatch.setattr(pod_config, "MCP_JSON", paths["mcp_json"])
    return paths


def test_cmd_sync_rereads_pods_conf_at_call_time(rig: dict[str, Path]) -> None:
    """Deterministic #813 regression reproducer (no race window needed).

    ``cmd_sync`` must parse ``pods.conf`` at CALL time, under the lock. Any
    caller-snapshot implementation (the pre-#831 ``cmd_sync(pods)`` fed by
    ``main()``'s unlocked parse) FAILS this test: the row appended to the
    file between the two syncs never reaches the managed block — exactly
    the #813 ``Host pod-813`` drop mechanism, reproduced deterministically.
    """
    pod_config.write_pods_conf([_pod("pod-test-A")])
    pod_config.cmd_sync()
    text = rig["ssh_config"].read_text()
    assert "Host pod-test-A" in text
    assert "Host pod-test-B" not in text

    # Mutate the FILE directly (as a concurrent session's upsert would).
    with open(rig["pods_conf"], "a") as fh:
        fh.write("pod-test-B  10.0.0.2  22002  1  H100  thomas-pod-test-B\n")

    pod_config.cmd_sync()
    text = rig["ssh_config"].read_text()
    assert "Host pod-test-A" in text
    assert "Host pod-test-B" in text


def test_cmd_sync_acquires_pods_conf_lock(
    rig: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the ORDER lock-enter -> parse_pods_conf -> update_ssh_config.

    A parse-BEFORE-lock implementation cannot false-pass: the recording
    shims assert the parse happens strictly inside the lock and before the
    downstream write. Covers the CLI path too, since ``main()``'s
    ``--sync`` branch is now a bare ``cmd_sync()`` call.
    """
    calls: list[str] = []

    @contextlib.contextmanager
    def fake_lock():
        calls.append("lock-enter")
        yield
        calls.append("lock-exit")

    def fake_parse(path: Path | None = None) -> list[Pod]:
        calls.append("parse")
        return []

    def fake_ssh(pods: list[Pod]) -> list[str]:
        calls.append("update_ssh")
        return []

    def fake_mcp(pods: list[Pod]) -> list[str]:
        calls.append("update_mcp")
        return []

    monkeypatch.setattr(pod_config, "locked_pods_conf", fake_lock)
    monkeypatch.setattr(pod_config, "parse_pods_conf", fake_parse)
    monkeypatch.setattr(pod_config, "update_ssh_config", fake_ssh)
    monkeypatch.setattr(pod_config, "update_mcp_config", fake_mcp)

    pod_config.cmd_sync()

    assert calls.index("lock-enter") < calls.index("parse") < calls.index("update_ssh")
    assert calls.index("update_ssh") < calls.index("lock-exit")
    # The MCP regenerate must ALSO complete inside the lock (same #813
    # protection as the ssh write — no propagation outside the lock).
    assert calls.index("update_mcp") < calls.index("lock-exit")


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_update_ssh_config_atomic_preserves_mode_and_foreign_content(
    rig: dict[str, Path], mode: int
) -> None:
    """A pre-existing ssh config keeps its exact mode + foreign stanzas.

    Atomicity is asserted as what a unit test can verify: no ``config.tmp``
    residue, mode preserved, non-managed content intact, managed block
    present — not proof that no torn intermediate ever existed.
    """
    ssh = rig["ssh_config"]
    ssh.parent.mkdir(parents=True, exist_ok=True)
    foreign = "Host github.com\n    HostName github.com\n    User git\n"
    ssh.write_text(foreign)
    os.chmod(ssh, mode)

    pod_config.update_ssh_config([_pod("pod-test-A")])

    assert os.stat(ssh).st_mode & 0o7777 == mode
    text = ssh.read_text()
    assert "Host github.com" in text
    assert "Host pod-test-A" in text
    assert not (ssh.parent / "config.tmp").exists()


def test_update_ssh_config_create_mode_0600(rig: dict[str, Path]) -> None:
    """A freshly-created ssh config gets 0600 (ssh refuses writable-by-others)."""
    ssh = rig["ssh_config"]
    assert not ssh.exists()

    pod_config.update_ssh_config([_pod("pod-test-A")])

    assert ssh.exists()
    assert os.stat(ssh).st_mode & 0o7777 == 0o600
    assert not (ssh.parent / "config.tmp").exists()


def test_update_mcp_config_atomic_and_preserves_nonpod_env(rig: dict[str, Path]) -> None:
    """Non-pod env keys survive the regenerate; JSON parses; no tmp residue."""
    mcp = rig["mcp_json"]
    seed = {"mcpServers": {"ssh": {"env": {"SOME_TOKEN": "keep-me"}}}}
    mcp.write_text(json.dumps(seed, indent=2) + "\n")

    pod_config.update_mcp_config([_pod("pod-123")])

    data = json.loads(mcp.read_text())  # parseability == not torn
    env = data["mcpServers"]["ssh"]["env"]
    assert env["SOME_TOKEN"] == "keep-me"
    assert env["SSH_SERVER_POD-123_HOST"] == "10.0.0.1"
    assert env["SSH_SERVER_POD-123_PORT"] == "22001"
    assert not (mcp.parent / "mcp.json.tmp").exists()


def _repoint(paths: dict[str, str]) -> None:
    """Worker-side repoint of pod_config module globals (spawn context —
    the parent's monkeypatched attrs do NOT cross the process boundary)."""
    import pod_config as pc

    pc.PODS_CONF = Path(paths["pods_conf"])
    pc.PODS_CONF_LOCK = Path(paths["lock"])
    pc.SSH_CONFIG = Path(paths["ssh_config"])
    pc.MCP_JSON = Path(paths["mcp_json"])


def _worker_upsert_and_sync(paths: dict[str, str], lock_held) -> None:
    """Worker A: acquire the flock, signal B, write pod-test-A, nested sync."""
    import pod_config as pc

    _repoint(paths)
    with pc.locked_pods_conf():
        lock_held.set()  # B now provably attempts its sync while A holds the lock
        rows = pc.parse_pods_conf()
        time.sleep(0.3)  # widen the window while B blocks on the flock
        rows.append(
            pc.Pod(
                name="pod-test-A",
                host="10.0.0.1",
                port=22001,
                gpus=1,
                gpu_type="H100",
                label="thomas-pod-test-A",
            )
        )
        pc.write_pods_conf(rows)
        pc.cmd_sync()  # nested acquisition — reentrant lock


def _worker_bare_sync(paths: dict[str, str], lock_held) -> None:
    """Worker B: wait until A holds the lock, then run the bare sync."""
    import pod_config as pc

    _repoint(paths)
    assert lock_held.wait(timeout=30)
    pc.cmd_sync()  # blocks on the flock until A releases, then re-reads


def test_concurrent_upsert_and_sync_no_lost_host_entry(rig: dict[str, Path]) -> None:
    """Concurrency regression: a bare sync racing an upsert loses no Host row.

    Deterministic ordering via an Event gate: worker A sets the event only
    AFTER acquiring the flock, so worker B's ``cmd_sync()`` provably starts
    while A holds the lock — B must block on the flock and re-read post-A
    state (a stale-snapshot sync is impossible by construction). The
    deterministic single-process #813 reproducer is
    ``test_cmd_sync_rereads_pods_conf_at_call_time``; this test pins the
    end-to-end no-lost-Host property under real processes.
    """
    paths = {k: str(v) for k, v in rig.items()}
    ctx = mp.get_context("spawn")
    lock_held = ctx.Event()
    a = ctx.Process(target=_worker_upsert_and_sync, args=(paths, lock_held))
    b = ctx.Process(target=_worker_bare_sync, args=(paths, lock_held))
    a.start()
    b.start()
    try:
        a.join(timeout=60)
        b.join(timeout=60)
        assert a.exitcode == 0, f"worker A exitcode={a.exitcode}"
        assert b.exitcode == 0, f"worker B exitcode={b.exitcode}"
    finally:
        for w in (a, b):
            if w.is_alive():
                w.terminate()

    text = rig["ssh_config"].read_text()
    assert "Host pod-test-A" in text


def test_atomic_write_replace_failure_leaves_no_tmp(
    rig: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion 3 failure path: replace fails -> target unchanged,
    no orphaned ``.tmp``, and the original OSError re-raises."""
    target = rig["ssh_config"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("original\n")

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError, match="simulated replace failure"):
        pod_config._atomic_write_text(target, "new-payload\n")

    assert target.read_text() == "original\n"
    assert not (target.parent / "config.tmp").exists()
