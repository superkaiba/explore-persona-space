#!/usr/bin/env python3
"""#2115 guard-hook cost probes — the three control-experiment arms, consolidated.

~12-18 autonomous sessions stalled 1.2-2.4h at a Step 10d pre-push lint-gate
Bash dispatch whose tool_result never arrived. These probes ran as /tmp
one-offs during diagnosis and are consolidated here (plan v3 §8 prong 3) so
the control experiments are re-runnable. Three subcommands:

  hook-cost   For each PreToolUse Bash hook in .claude/settings.json, run it
              against (a) a short benign control command and (b) the real
              multi-KB Step-10d lint-gate workload argv (extracted verbatim
              from SKILL.md), reporting wall time + exit code per hook. A
              hook slow ONLY on arm (b) is payload-shape-attributable.
  scaling     For the named slow guards, sweep the argv size (fractions and
              multiples of the real payload) to classify the cost curve:
              linear => argv size alone cannot reach hours (a second
              mechanism, e.g. contention, is required); superlinear => the
              payload shape is the fix surface.
  cancel      Root-cause arm: launch each guard in its own process group with
              stdout on a pipe, SIGTERM the LEAD process after a grace (what
              a hook cancel does), then ask (Q1) do forked descendants
              survive? (Q2) does reading the pipe to EOF still block (a
              surviving grandchild holding the write end)? Q2 blocking is the
              wedge mechanism reproduced.

READ-ONLY by construction: every arm only reads SKILL.md / settings.json and
spawns the hooks as bounded subprocesses (the guards themselves are pure
deny/allow filters); the cancel arm signals only pids inside its own probe's
process group. No repo file, git state, or task state is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
SETTINGS = REPO / ".claude" / "settings.json"

# The guards the hook-cost arm measured as payload-sensitive; the scaling +
# cancel arms probe exactly these.
SLOW_GUARDS = (
    REPO / "scripts" / "guard_repo_root_branch.sh",
    REPO / ".claude" / "hooks" / "guard_tmp_tmux_sweep.sh",
    REPO / ".claude" / "hooks" / "guard_root_code_commit.sh",
)

LINT_GATE_HEADING = "#### Pre-push workflow-lint gate"
SELF_CHOOM_ANCHOR = "sudo -n choom -n -600 -p $$"

# The #2155 split relocated SKILL.md step bodies to steps/ companions,
# leaving a `> **Full procedure:**` pointer per step in the router.
STEP_POINTER_RE = re.compile(r"^>\s+\*\*Full procedure:\*\*\s+`\.claude/skills/issue/steps/(\S+?)`")


def _gate_source_lines() -> tuple[Path, list[str], int]:
    """(source path, its lines, index of the lint-gate heading line).

    On an unsplit tree the Step-10d lint-gate section lives in SKILL.md
    itself (the fallback, so the probe runs on either tree shape); on a
    #2155 split tree the Step-10d body was relocated to a steps/ companion,
    so resolve SKILL.md's `> **Full procedure:**` pointers and pick the
    pointed-to companion carrying the heading. Exits FATAL when no source
    carries it.
    """
    skill_lines = SKILL.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(skill_lines):
        if ln.startswith(LINT_GATE_HEADING):
            return SKILL, skill_lines, i  # unsplit tree
    steps_dir = SKILL.parent / "steps"
    for ln in skill_lines:
        m = STEP_POINTER_RE.match(ln)
        if m is None:
            continue
        companion = steps_dir / m.group(1)
        if not companion.is_file():
            continue
        companion_lines = companion.read_text(encoding="utf-8").splitlines()
        for i, cln in enumerate(companion_lines):
            if cln.startswith(LINT_GATE_HEADING):
                return companion, companion_lines, i
    sys.exit(
        f"FATAL: heading {LINT_GATE_HEADING!r} not found in {SKILL} nor in any "
        f"steps/ companion it points to (#2155 split-aware resolution)"
    )


def extract_payload() -> str:
    """Pull the Step 10d lint-gate workload body out of SKILL.md verbatim.

    Anchors on the lint-gate section heading (resolved split-aware via
    :func:`_gate_source_lines` — SKILL.md itself on an unsplit tree, the
    pointed-to steps/ companion on a #2155 split tree), then the first
    self-choom line after it (the wedged command's first token per the
    #2115 filing) — no hardcoded line numbers, so the extraction survives
    SKILL.md drift. Stops at the fenced-block terminator. Comment-only
    lines keep their leading '#': they are part of the real argv (the
    orchestrator sends the annotated body). Exits FATAL if either anchor
    is missing.
    """
    src, lines, section = _gate_source_lines()
    start = None
    for i in range(section, len(lines)):
        if SELF_CHOOM_ANCHOR in lines[i] and not lines[i].lstrip().startswith("#"):
            start = i
            break
    if start is None:
        sys.exit(f"FATAL: self-choom anchor not found after the lint-gate heading in {src}")
    # Scan to the fence. The bound is a runaway guard, NOT a truncation policy:
    # silently cutting the body at the bound would understate every measured
    # cost in the table this script exists to reproduce, so hitting the bound
    # without finding the fence is a hard error (code-review #2115 finding 3).
    scan_bound = 5000
    body = []
    fenced = False
    for ln in lines[start : start + scan_bound]:
        if ln.strip().startswith("```"):
            fenced = True
            break
        body.append(ln[2:] if ln.startswith("  ") else ln)
    if not fenced:
        sys.exit(
            f"FATAL: no closing fence within {scan_bound} lines of the gate-body "
            f"anchor at {src.name}:{start + 1}. The gate body either outgrew the "
            f"scan bound or the anchor drifted; either way a truncated payload "
            f"would understate the measured cost. Re-check the anchor and raise "
            f"the bound deliberately."
        )
    return "\n".join(body)


def hook_commands() -> list[str]:
    """Every PreToolUse Bash hook command registered in settings.json."""
    cfg = json.loads(SETTINGS.read_text(encoding="utf-8"))
    out = []
    for matcher in cfg.get("hooks", {}).get("PreToolUse", []):
        if matcher.get("matcher") != "Bash":
            continue
        for hk in matcher.get("hooks", []):
            out.append(hk.get("command", ""))
    return out


def _label(cmd: str) -> str:
    """Short display label for a hook command (script basename or inline head)."""
    m = re.search(r"([\w.-]+\.sh)$", cmd.strip())
    if m:
        return m.group(1)
    return "inline:" + " ".join(cmd.split())[:48]


def _hook_env(tool_input: dict) -> dict:
    """Hook-runner env: TOOL_INPUT for the inline hooks + the project dir."""
    env = dict(os.environ)
    env["TOOL_INPUT"] = json.dumps(tool_input)
    env["CLAUDE_PROJECT_DIR"] = str(REPO)
    return env


def _run_hook(hook_cmd: str, command_text: str, timeout_s: float) -> tuple[float, str]:
    """Run one hook exactly as the runner does (stdin JSON + TOOL_INPUT env,
    cwd = repo root), bounded; returns (wall seconds, verdict string)."""
    tool_input = {"command": command_text}
    stdin_blob = json.dumps({"tool_name": "Bash", "tool_input": tool_input})
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            ["bash", "-c", hook_cmd],
            input=stdin_blob,
            capture_output=True,
            text=True,
            cwd=REPO,
            env=_hook_env(tool_input),
            timeout=timeout_s,
        )
        return time.monotonic() - t0, f"rc={p.returncode}"
    except subprocess.TimeoutExpired:
        return time.monotonic() - t0, f"TIMEOUT>{timeout_s:g}s"


def cmd_hook_cost(args: argparse.Namespace) -> None:
    """Arm 1: per-hook wall time on control vs gate-shaped argv."""
    payload = extract_payload()
    hooks = hook_commands()
    print(f"payload: {len(payload)} bytes, {payload.count(chr(10)) + 1} lines")
    print(f"payload head: {payload.splitlines()[0][:90]!r}")
    print(f"PreToolUse Bash hooks: {len(hooks)}")
    print()
    arms = [("CONTROL(short)", "git status --porcelain"), ("GATE(multi-KB)", payload)]
    for arm_name, cmd_text in arms:
        print(f"=== {arm_name} — {len(cmd_text)} bytes ===")
        for hook_cmd in hooks:
            dt, verdict = _run_hook(hook_cmd, cmd_text, args.timeout_s)
            flag = "  <-- SLOW" if dt > args.slow_threshold_s else ""
            print(f"  {dt:7.2f}s  {verdict:16s}  {_label(hook_cmd)}{flag}")
        print()


def cmd_scaling(args: argparse.Namespace) -> None:
    """Arm 2: cost curve of the slow guards vs argv size (linear vs
    superlinear — the hours-vs-minutes discriminator)."""
    full = extract_payload()
    lines = full.splitlines()
    fracs = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    print(f"full payload: {len(full)} bytes / {len(lines)} lines\n")
    hdr = f"{'lines':>6} {'bytes':>7} | " + " | ".join(
        g.name.replace(".sh", "")[:22].rjust(22) for g in SLOW_GUARDS
    )
    print(hdr)
    print("-" * len(hdr))
    for f in fracs:
        n = max(1, int(len(lines) * f))
        # For f>1 repeat the body to grow clause count with the same texture.
        reps = -(-n // len(lines))
        text = "\n".join((lines * reps)[:n])
        cells = []
        for g in SLOW_GUARDS:
            dt, verdict = _run_hook(str(g), text, args.timeout_s)
            timed_out = verdict.startswith("TIMEOUT")
            cells.append(f"{dt:>16.2f}s{'*' if timed_out else ' '} ".rjust(22))
        print(f"{n:>6} {len(text):>7} | " + " | ".join(cells), flush=True)
    print(f"\n* = hit the {args.timeout_s:g}s probe timeout (true cost is higher)")


def _probe_cancel(guard: Path, payload: str, kill_after_s: float, eof_wait_s: float) -> dict:
    """Arm-3 protocol for one guard: own-process-group launch, SIGTERM the
    lead after the grace, then measure descendant survival (Q1) and whether
    the stdout pipe still blocks short of EOF (Q2 — the wedge mechanism).
    Signals ONLY pids inside this probe's own process group."""
    tool_input = {"command": payload}
    r_fd, w_fd = os.pipe()
    p = subprocess.Popen(
        ["bash", "-c", str(guard)],
        stdin=subprocess.PIPE,
        stdout=w_fd,
        stderr=subprocess.DEVNULL,
        cwd=REPO,
        env=_hook_env(tool_input),
        start_new_session=True,  # own process group, like a hook runner
    )
    os.close(w_fd)  # only the guard tree holds the write end now
    try:
        p.stdin.write(json.dumps({"tool_name": "Bash", "tool_input": tool_input}).encode())
        p.stdin.close()
    except BrokenPipeError:
        pass

    time.sleep(kill_after_s)
    # Count the tree BEFORE the kill (evidence of fork fan-out).
    pre = subprocess.run(["pgrep", "-g", str(p.pid)], capture_output=True, text=True).stdout.split()

    # Cancel the LEAD process only — exactly what a per-process cancel does.
    lead_alive = p.poll() is None
    if lead_alive:
        os.kill(p.pid, signal.SIGTERM)
    time.sleep(1.0)

    post = subprocess.run(
        ["pgrep", "-g", str(p.pid)], capture_output=True, text=True
    ).stdout.split()
    survivors = [x for x in post if x != str(p.pid)]

    # Q2: does the pipe reach EOF, or is a write end still held?
    t0 = time.monotonic()
    eof = False
    while time.monotonic() - t0 < eof_wait_s:
        ready, _, _ = select.select([r_fd], [], [], 0.5)
        if ready:
            chunk = os.read(r_fd, 65536)
            if chunk == b"":
                eof = True
                break
        elif not subprocess.run(
            ["pgrep", "-g", str(p.pid)], capture_output=True, text=True
        ).stdout.split():
            # nothing left alive and nothing readable -> drain once more
            ready2, _, _ = select.select([r_fd], [], [], 0.5)
            if not ready2:
                continue
    eof_wait = time.monotonic() - t0
    os.close(r_fd)
    # Clean up anything we left behind (our own probe's group only).
    for pid in survivors:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
    try:
        p.kill()
    except ProcessLookupError:
        pass
    return {
        "guard": guard.name,
        "lead_alive_at_kill": lead_alive,
        "tree_before_kill": len(pre),
        "survivors_after_kill": len(survivors),
        "pipe_reached_eof": eof,
        "eof_wait_s": round(eof_wait, 2),
    }


def cmd_cancel(args: argparse.Namespace) -> None:
    """Arm 3: cancellation effectiveness per slow guard."""
    payload = extract_payload()
    print(
        f"payload {len(payload)} bytes; SIGTERM lead after {args.kill_after_s:g}s; "
        f"EOF wait cap {args.eof_wait_s:g}s\n"
    )
    for g in SLOW_GUARDS:
        r = _probe_cancel(g, payload, args.kill_after_s, args.eof_wait_s)
        verdict = (
            "CANCEL CLEAN (pipe EOF)"
            if r["pipe_reached_eof"]
            else "WEDGE REPRODUCED (write end still held)"
        )
        print(
            f"{r['guard']:28s} tree_before={r['tree_before_kill']:4d} "
            f"survivors={r['survivors_after_kill']:4d} "
            f"eof={r['pipe_reached_eof']!s:5s} "
            f"waited={r['eof_wait_s']:5.2f}s  -> {verdict}",
            flush=True,
        )


def build_argparser() -> argparse.ArgumentParser:
    """CLI: one subcommand per probe arm (hook-cost / scaling / cancel)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="arm", required=True)

    p1 = sub.add_parser("hook-cost", help="per-hook wall time: control vs gate-shaped argv")
    p1.add_argument("--timeout-s", type=float, default=90.0, help="per-hook bound (default 90)")
    p1.add_argument(
        "--slow-threshold-s",
        type=float,
        default=5.0,
        help="flag a hook as SLOW above this wall time (default 5)",
    )
    p1.set_defaults(func=cmd_hook_cost)

    p2 = sub.add_parser("scaling", help="slow-guard cost curve vs argv size")
    p2.add_argument("--timeout-s", type=float, default=240.0, help="per-cell bound (default 240)")
    p2.set_defaults(func=cmd_scaling)

    p3 = sub.add_parser("cancel", help="cancellation-effectiveness probe per slow guard")
    p3.add_argument(
        "--kill-after-s",
        type=float,
        default=2.0,
        help="grace before SIGTERMing the lead process (default 2)",
    )
    p3.add_argument(
        "--eof-wait-s",
        type=float,
        default=20.0,
        help="cap on the pipe-EOF wait (default 20)",
    )
    p3.set_defaults(func=cmd_cancel)
    return ap


def main() -> None:
    """Dispatch the selected probe arm."""
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
