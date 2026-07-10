"""Pin the setsid-detach convention for VM-side long compute phases (#833, task #884).

Prose pins (the SKILL.md breadcrumb convention + the code-style.md nohup bullet, plus
the #1045 Step 9c gate self-choom defaults) and one mechanism test: a ``setsid nohup``
child survives the process-group kill that takes down a plain ``nohup`` sibling — the
#833 watcher-force-stop kill vector.
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
VECTORIZE_MD = REPO_ROOT / ".claude" / "rules" / "vectorize-many-cell-fits.md"


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


def test_step9c_gate_choom_defaults_pinned() -> None:
    """Step 9c gates self-choom by default; 1d refresh pid-captures + sweeps (#1045)."""
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    # Gate self-choom: 2x Step 9c (1b + 1c) + 1x Step 9c 1d compare (#1197)
    # + 2x Step 10d lint gate (#1211). (== 5 also pins placement.)
    # NOTE: this also repairs the pre-existing red assertion — #1197 added
    # the 1d-compare occurrence without updating this count.
    assert skill_text.count("sudo -n choom -n -600 -p $$") == 5
    # Unconditional both-branch gate breadcrumb (success state durably observable)
    assert "[step9c] gate earlyoom protection choom=$GATE_CHOOM" in skill_text
    # 1d refresh: pid capture + session sweep present (both literals — removing
    # only the sweep while keeping pid capture must fail this pin; Codex r1 minor)
    assert "REFRESH_PID" in skill_text
    assert 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' in skill_text
    # Vectorize rule carries the launch-form cross-reference (fix item 7)
    vec_text = VECTORIZE_MD.read_text(encoding="utf-8")
    assert "Detached VM-side long compute phases" in vec_text


def test_step10d_lint_gate_choom_pinned() -> None:
    """Step 10d pre-push lint-gate blocks self-choom by default, fail-open (#1211)."""
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    # Preamble in BOTH executable blocks: shared form (i)/(ii) + form (iii) surgical
    assert skill_text.count("[step10d] lint-gate earlyoom protection choom=$LINT_GATE_CHOOM") == 2
    assert "LINT_GATE_CHOOM=failed" in skill_text


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
