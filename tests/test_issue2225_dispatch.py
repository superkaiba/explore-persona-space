"""Issue #2225 — dispatcher contract pins (r2 blockers 2 + 4, g3 Major 1).

1. SENTINEL PATH (r2 blocker 2 / g4 Critical 1): pod sentinels MUST land in the
   TOP-LEVEL ``/workspace/logs`` — the VM poller's drain glob
   ``/workspace/logs/issue-2225-*.json`` is path-terminal (``*`` never crosses
   ``/``), so a subdirectory sentinel is silently invisible. Pins BOTH the
   dispatcher's SENTINEL_ROOT default and the writer's filename shape against
   the drain glob.
2. HOOK-COUNT GATE (r2 blocker 4 / g5 Critical + g4 Major 2): both §7
   criterion-(i) count gates accept ``[steer-hook]`` OR ``[fanout-skip]`` per
   per-cell log, so a resume-skipped cell (the octave-grid overlap cell skips
   DETERMINISTICALLY; a fresh-pod HF-complete resume skips ALL cells) never
   starves the count into a false exit 7. Includes the fails-pre-fix
   sed-extracted bash probe of ``p0_run_repilot`` with a stubbed skipped cell.
3. P0 MMLU PROBE LEG (g3 Major 1 / plan §4.8(b)): the dispatcher wires the
   ``--limit``-bounded MMLU invocation-path probe before P2c's full-set runs.

CPU-only; the bash probe stubs every ``uv run python scripts/...`` invocation
(json one-liners + heredocs run the real interpreter).
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_DISPATCH = _SCRIPTS / "issue2225_dispatch.sh"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DISPATCH_SRC = _DISPATCH.read_text(encoding="utf-8")

# The VM poller's drain shape (poll_pipeline.py: /workspace/logs/issue-<N>-*.json,
# path-terminal / non-recursive).
DRAIN_DIR = "/workspace/logs"
DRAIN_GLOB = "issue-2225-*.json"


# ── 1. sentinel path contract (r2 blocker 2) ──────────────────────────────────


def test_sentinel_root_default_is_toplevel_workspace_logs():
    m = re.search(r'^SENTINEL_ROOT="\$\{EPM_I2225_SENTINEL_ROOT:-([^}"]+)\}"', DISPATCH_SRC, re.M)
    assert m, "dispatch.sh must define SENTINEL_ROOT with an env-overridable default"
    assert m.group(1) == DRAIN_DIR, (
        f"SENTINEL_ROOT default {m.group(1)!r} is not the poller's top-level "
        f"drain dir {DRAIN_DIR!r} (the glob does not cross '/')"
    )


def test_write_sentinel_uses_sentinel_root_not_log_root():
    assert 'SENT_LOGS_DIR="$SENTINEL_ROOT"' in DISPATCH_SRC
    assert 'SENT_LOGS_DIR="$LOG_ROOT"' not in DISPATCH_SRC, (
        "write_sentinel must never hand the phase-log SUBDIR to "
        "write_results_sentinel — the drain glob cannot see it (g4 Critical 1)"
    )


def test_sentinel_filename_matches_drain_glob(tmp_path):
    lib = _load("issue778_lib")
    path = lib.write_results_sentinel(2225, "epm:results", 1, {"probe": True}, logs_dir=tmp_path)
    assert path.parent == tmp_path  # logs_dir IS the sentinel parent (no subdir)
    assert fnmatch(path.name, DRAIN_GLOB), (path.name, DRAIN_GLOB)
    obj = json.loads(path.read_text())
    for key in ("sentinel_schema_version", "kind", "version"):
        assert key in obj, f"poller-required sentinel key missing: {key}"


# ── 2. §7 criterion-(i) dual-token count gates (r2 blocker 4) ─────────────────

_DUAL_GREP = 'grep -rlF -e "[steer-hook]" -e "[fanout-skip]"'


def test_hook_count_gates_are_dual_token_at_both_sites():
    """BOTH gate sites (phase_p0 + p0_run_repilot) count fresh-launch AND
    resume-skip evidence; no single-token gate remains (the g4 Major 2 sweep)."""
    assert DISPATCH_SRC.count(_DUAL_GREP) == 2, (
        "expected the dual-token count grep at exactly the two §7 gate sites"
    )
    assert 'grep -rlF "[steer-hook]"' not in DISPATCH_SRC, (
        "a single-token hook-count grep survives — resume-starve false exit 7"
    )


def test_dual_token_grep_counts_each_log_once(tmp_path):
    """Functional pin of the exact grep: fresh log / skip log / both-token log
    each count ONCE (grep -l lists a file once regardless of match count)."""
    logs = tmp_path / "p0_train"
    logs.mkdir()
    (logs / "a.log").write_text("[steer-hook] mode=response layers=1 alpha=3.0\n")
    (logs / "b.log").write_text("[fanout-skip] b resume (proven by completed run)\n")
    (logs / "c.log").write_text("[steer-hook] x\n[fanout-skip] c resume\n")
    (logs / "d.log").write_text("no evidence here\n")
    out = subprocess.run(
        ["bash", "-c", f'{_DUAL_GREP} "$1" 2>/dev/null | wc -l', "_", str(logs)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert out.stdout.strip() == "3"


def _extract_p0_run_repilot() -> str:
    """Extract the REAL p0_run_repilot body. A bare non-greedy `^}` terminator
    would stop at a column-0 `}` INSIDE the function's python heredoc, so the
    extraction anchors on the function's distinctive terminal log line."""
    m = re.search(
        r"^p0_run_repilot\(\).*?"
        r"^  log_phase p0_repilot \"done \(passed on shifted grid\)\"\n\}",
        DISPATCH_SRC,
        re.M | re.S,
    )
    assert m, "p0_run_repilot not found in dispatch.sh (terminal line moved?)"
    return m.group(0)


def test_p0_run_repilot_survives_resume_skipped_overlap_cell(tmp_path):
    """FAILS PRE-FIX (g5 Critical): every octave grid overlaps the trained
    pilot grid at one coefficient (x0.5 keeps 1.5), so that cell ALWAYS
    resume-skips and writes no fresh [steer-hook] log. The sed-extracted REAL
    ``p0_run_repilot`` body (real count-gate grep + real heredocs) must reach
    eval-gen and resolve — never FATAL exit 7 — when the train stub emulates
    the fixed fan-out (3 fresh logs + 1 [fanout-skip] log)."""
    eval_root = tmp_path / "eval"
    (eval_root / "pilot_gate").mkdir(parents=True)
    log_root = tmp_path / "logs"
    log_root.mkdir()
    plan = {
        "A": {
            "coef_scale": 0.5,
            "grid_csv": "0.25,0.75,1.5,2.5",
            "cells": ["A__evil__c0.25", "A__evil__c0.75", "A__evil__c1.5", "A__evil__c2.5"],
        }
    }
    state_path = eval_root / "pilot_gate" / "repilot_state.json"
    state_path.write_text(json.dumps({"plan": plan, "resolved": False}))
    verdict_path = eval_root / "pilot_gate" / "p0_verdict.json"
    verdict_path.write_text(
        json.dumps({"passed": True, "octave_shift": {"A": 0.5, "C": None}, "criteria": {}})
    )

    fn = _extract_p0_run_repilot()
    probe = f"""
set -euo pipefail
LOG_ROOT="{log_root}"
EVAL_ROOT="{eval_root}"
PILOT_OUT="{tmp_path}/pilot_out"
CKPT_ROOT="{tmp_path}/ckpt"
DIR_OUT="{tmp_path}/dirs"
EXTERNAL_ROOT="{tmp_path}/external"
I778_BASELINE="{tmp_path}/i778_baseline.json"
TRAIN_SMOKE_ARGS=()
EVALGEN_SMOKE_ARGS=()
log_phase() {{ echo "[phase=$1] $2"; }}
headroom() {{ :; }}
write_sentinel() {{ echo "[stub-sentinel] $1"; }}
p0_upload_pilot_raws() {{ :; }}
uv() {{
  local sig="${{1:-}} ${{2:-}} ${{3:-}}"
  if [ "$sig" = "run python scripts/issue2225_train.py" ]; then
    # Emulate the FIXED fan-out over the x0.5 grid: the overlap cell
    # A__evil__c1.5 resume-skips (writes [fanout-skip]); the rest launch.
    mkdir -p "$LOG_ROOT/p0_train_repilot"
    for c in A__evil__c0.25 A__evil__c0.75 A__evil__c2.5; do
      echo "[steer-hook] mode=response layers=1 alpha=x" > "$LOG_ROOT/p0_train_repilot/$c.log"
    done
    echo "[fanout-skip] A__evil__c1.5 resume (proven by completed run)" \\
      > "$LOG_ROOT/p0_train_repilot/A__evil__c1.5.log"
    return 0
  fi
  if [ "$sig" = "run python scripts/issue2225_eval_gen.py" ]; then
    echo "PROBE-EVALGEN-REACHED"
    return 0
  fi
  if [ "$sig" = "run python scripts/issue2225_judge.py" ]; then
    echo "PROBE-JUDGE"
    return 0
  fi
  command uv "$@"
}}
{fn}
p0_run_repilot "{state_path}"
"""
    script = tmp_path / "probe.sh"
    script.write_text(probe)
    out = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, cwd=_REPO, timeout=300
    )
    assert out.returncode == 0, (out.returncode, out.stdout[-2000:], out.stderr[-2000:])
    assert "PROBE-EVALGEN-REACHED" in out.stdout, out.stdout[-2000:]
    assert "hook-engagement logs (fresh or resume-skip): 4/4" in out.stdout
    assert "FATAL" not in out.stdout and "FATAL" not in out.stderr
    state = json.loads(state_path.read_text())
    assert state["resolved"] is True and state["second_miss"] is False


# ── 3. P0 MMLU --limit probe leg (g3 Major 1 / plan §4.8(b)) ──────────────────


def test_p0_mmlu_probe_leg_wired():
    assert re.search(r"^MMLU_P0_LIMIT=200\b", DISPATCH_SRC, re.M), (
        "plan §4.8(b): the P0 MMLU probe runs --limit 200"
    )
    assert '--limit "$MMLU_P0_LIMIT"' in DISPATCH_SRC
    assert "issue2225_mmlu.py" in DISPATCH_SRC.split("phase_p0()")[1].split("phase_p2a()")[0], (
        "the MMLU probe leg must run inside phase_p0 (before P2c's full set)"
    )
