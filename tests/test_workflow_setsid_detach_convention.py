"""Pin the setsid-detach convention for VM-side long compute phases (#833, task #884).

Two prose pins (the SKILL.md breadcrumb convention + the code-style.md nohup bullet)
and one mechanism test: a ``setsid nohup`` child survives the process-group kill that
takes down a plain ``nohup`` sibling — the #833 watcher-force-stop kill vector.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
CODE_STYLE_MD = REPO_ROOT / ".claude" / "rules" / "code-style.md"


def test_skill_md_mandates_setsid_detach_and_pid_breadcrumb() -> None:
    """SKILL.md carries the detached launch shape, breadcrumb fields, and successor rule."""
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "setsid nohup" in text
    assert "Detached VM-side long compute phases" in text
    assert "pid=<PHASE_PID>" in text
    assert "log=<abs log" in text
    assert "ps -p <pid> -o args=" in text
    assert "never relaunch" in text


def test_code_style_nohup_bullet_covers_vm_side() -> None:
    """code-style.md's nohup bullet covers VM-LOCAL long phases, not just pod launches."""
    text = CODE_STYLE_MD.read_text(encoding="utf-8")
    assert "setsid nohup" in text
    assert "VM-LOCAL" in text


def _stat(pid: int) -> str:
    """Return the ps STAT column for pid ('' when the process is gone)."""
    out = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
    return out.stdout.strip()


def _ppid(pid: int) -> str:
    """Return the ps PPID column for pid ('' when the process is gone)."""
    out = subprocess.run(["ps", "-p", str(pid), "-o", "ppid="], capture_output=True, text=True)
    return out.stdout.strip()


def _dead(pid: int) -> bool:
    """Gone or zombie counts as dead (``os.kill(pid, 0)`` succeeds on a zombie)."""
    stat = _stat(pid)
    return stat == "" or stat.startswith("Z")


def test_setsid_child_survives_group_kill() -> None:
    """killpg on the launcher's group kills the plain-nohup child; the setsid child survives."""
    script = (
        "nohup sleep 30 </dev/null >/dev/null 2>&1 &\n"
        "plain=$!\n"
        "setsid nohup sleep 30 </dev/null >/dev/null 2>&1 &\n"
        "detached=$!\n"
        'echo "$plain $detached"\n'
    )
    wrapper = subprocess.Popen(
        ["bash", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # own pgid: the killpg below can never touch pytest
    )
    plain_pid = detached_pid = None
    try:
        stdout, stderr = wrapper.communicate(timeout=10)
        assert wrapper.returncode == 0, stderr
        plain_pid, detached_pid = (int(tok) for tok in stdout.split())
        # killpg fires ONLY after both pids were read from wrapper stdout AND the detached
        # child's exec chain has actually reached setsid(2) — until then it still sits in
        # the wrapper's process group and a premature killpg races the detach itself.
        deadline = time.monotonic() + 10.0
        while os.getpgid(detached_pid) != detached_pid:
            assert time.monotonic() < deadline, "setsid child never became its own pg leader"
            time.sleep(0.05)
        # The wrapper has exited, but the plain child keeps its process group alive.
        os.killpg(wrapper.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10.0
        while not _dead(plain_pid):
            assert time.monotonic() < deadline, "plain nohup child survived the group kill"
            time.sleep(0.1)
        assert not _dead(detached_pid), "setsid child died with the group kill"
        # The wrapper exited, so the setsid child reparented to PID 1 — a ppid-tree walk
        # from the dead session cannot reach it (the second kill vector, beyond killpg).
        assert _ppid(detached_pid) == "1"
    finally:
        for pid in (plain_pid, detached_pid):
            if pid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        if wrapper.poll() is None:
            wrapper.kill()
