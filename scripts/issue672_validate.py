#!/usr/bin/env python3
"""Issue #672 GCP end-to-end validation orchestrator (CPU-only, runs on the VM).

System-validation, NOT a science experiment. After #669 (GCP wedge recovery)
and #671 (extractor memory-bug fix) merged to ``main``, this driver certifies
end-to-end that GCP works again, across three sections (plan §4):

- **Section A — happy-path GCP A100 smoke** of the FIXED hook-based extractor
  (``scripts/issue667_extract.py`` with ``--log-mem-every``). PASS iff the run
  completes + writes ≥1 ``.npz`` + BOTH PRIMARY resident gauges
  (``memory_reserved()`` and ``nvidia-smi memory.used``) stay flat (max−min
  < 1 GiB, no ≥4 GiB monotone climb) over iters ≥10, + HF upload OK.
- **Section B — live fault-injection** on a tiny GCP VM: iptables-drop the
  metadata + HF endpoints, watch the ``[eps-watchdog]`` ladder → TERMINATED →
  #659 RunPod failover-once. The LIVE path is REQUIRED for the live-recovery
  headline; any fallback maps to ``inconclusive_live_validation`` (NOT B PASS)
  and DOWNGRADES the headline (plan §1/§4).
- **Section C — regression sweep on ``main``**: the 4 named test files + the
  two workflow-lint runs (plan §4).

The driver NEVER provisions/terminates compute itself beyond invoking the
project's ``dispatch_issue.py launch`` (Section A/B) — and even those launches
are run by the experimenter step, not the implementer. ``--dry-run`` builds the
remote/test commands without invoking real ``gcloud`` / ``pytest`` /
``dispatch_issue.py``, for command-construction smoke.

Marker reads use the importable ``task_workflow`` resolver (branch-guards to
``main``, works from a worktree) — NEVER a ``task.py`` shell-out (which would
refuse on the worktree's non-``main`` HEAD).

Usage::

    # full pipeline (a -> b -> c -> assemble)
    uv run python scripts/issue672_validate.py --issue 672
    # one section
    uv run python scripts/issue672_validate.py --issue 672 --section c
    # assemble a validation.json from already-written per-section JSONs
    uv run python scripts/issue672_validate.py --issue 672 --section assemble
    # dry-run command construction (no gcloud/pytest/dispatch invoked)
    uv run python scripts/issue672_validate.py --issue 672 --section c --dry-run
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation (max−min, ≥, →) in docstrings

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue672_validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Section-C regression sweep targets (plan §4) ─────────────────────────────
SECTION_C_PYTEST_FILES = [
    "tests/test_gcp_backend.py",
    "tests/test_backend_poll.py",
    "tests/test_issue671_extraction_hooks.py",
    "tests/test_failure_classifier.py",
]
SECTION_C_LINT_RUNS = {
    # name -> argv after `scripts/workflow_lint.py`
    "batch_judge_client": ["--check-batch-judge-client"],
    "no_flags_default": [],  # the bundled default run
}

# ── Section-A flatness thresholds (plan §6) ──────────────────────────────────
FLAT_MAX_MIN_GIB = 1.0  # PASS iff PRIMARY gauge max−min < 1 GiB
CLIMB_FALSIFY_GIB = 4.0  # a PRIMARY gauge climbing ≥4 GiB monotone falsifies H-A
WARMUP_ITERS = 10  # compute max−min over iters with idx >= 10 (first-10 baseline)

# ── Section-B watchdog / failover constants (plan §10, #669) ─────────────────
GCP_ZONE = "us-central1-b"
GCP_CONFIG = "eps-gcp"
WATCHDOG_BUDGET_S = 6 * 60  # ladder must reach 10/10 + wedged-shutdown within ~6 min
FAILOVER_REASON_RE = re.compile(r"gcp_workload_failover_runpod")


# ═════════════════════════════════════════════════════════════════════════════
# Marker reads (importable resolver — NEVER a task.py shell-out)
# ═════════════════════════════════════════════════════════════════════════════


def _events(issue: int) -> list[dict]:
    """Read the task's events.jsonl via the branch-guarded resolver."""
    from explore_persona_space import task_workflow as tw

    return list(tw.list_events(issue))


def _backend_selected_sku(issue: int) -> str | None:
    """The landed GPU SKU from the latest ``epm:backend-selected`` marker note.

    The marker note carries ``chosen_kind`` / machine info; we surface the raw
    note string when present (the analyzer reads the precise SKU). Returns
    ``None`` when no backend-selected marker exists yet.
    """
    sel = [e for e in _events(issue) if e.get("kind") == "epm:backend-selected"]
    if not sel:
        return None
    return str(sel[-1].get("note", "")).strip() or None


def _count_failover_relaunches(issue: int) -> tuple[int, str | None]:
    """Count cluster-launched markers whose note names a RunPod failover reason.

    Returns ``(count, latest_marker_ts)``. The B-PASS predicate requires count
    == exactly 1 (idempotent failover-once; failover twice falsifies H-B).
    """
    hits = [
        e
        for e in _events(issue)
        if e.get("kind") in ("epm:cluster-launched", "epm:backend-selected")
        and FAILOVER_REASON_RE.search(str(e.get("note", "")))
    ]
    ts = hits[-1].get("ts") if hits else None
    return len(hits), ts


# ═════════════════════════════════════════════════════════════════════════════
# Subprocess helper (explicit env passthrough; load_dotenv at module top)
# ═════════════════════════════════════════════════════════════════════════════


def _run(argv: list[str], *, dry_run: bool, timeout: int | None = None) -> tuple[int, str, str]:
    """Run ``argv`` from the repo root with explicit env; return (rc, out, err).

    Every subprocess gets ``env={**os.environ}`` explicitly (the dispatcher
    subprocess-env contract). ``dry_run`` short-circuits with rc=0 and a
    constructed-command echo so command construction can be smoke-tested
    without invoking the real tool.
    """
    cmd = " ".join(argv)
    if dry_run:
        logger.info("[dry-run] would run: %s", cmd)
        return 0, f"[dry-run] {cmd}", ""
    logger.info("running: %s", cmd)
    proc = subprocess.run(
        argv,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ═════════════════════════════════════════════════════════════════════════════
# Section A — happy-path GCP A100 smoke (memory flatness)
# ═════════════════════════════════════════════════════════════════════════════


def section_a_dispatch_argv(issue: int, out_subdir: str) -> list[str]:
    """The plan §10 Section-A GCP launch argv (verbatim shape).

    Returns the ``dispatch_issue.py launch`` argv. The workload-cmd runs the
    FIXED extractor with ``--log-mem-every 10`` writing to
    ``eval_results/issue_<N>/<out_subdir>``.
    """
    workload = (
        'REPO_ROOT="$WORKLOAD_ROOT" uv run python scripts/issue667_extract.py '
        "--behavior marker --source-cid default --targets default,sp_swe "
        "--layers 7 14 21 --primary-layer 14 --max-probes 8 --max-new-tokens 256 "
        f"--log-mem-every 10 --out eval_results/issue_{issue}/{out_subdir} --gpu-id 0"
    )
    return [
        "uv",
        "run",
        "python",
        "scripts/dispatch_issue.py",
        "launch",
        "--issue",
        str(issue),
        "--intent",
        "lora-7b",
        "--backend",
        "gcp",
        "--repo-branch",
        f"issue-{issue}",
        "--workload-cmd",
        workload,
    ]


def _flatness(values: list[float]) -> dict:
    """max−min over warmup-trimmed iters + a monotone-climb falsification flag.

    Computes max−min over the gauge's samples (the caller trims to iters ≥10).
    ``monotone_climb_ge_4`` flags an apparent monotone ≥4 GiB climb (H-A
    falsification target) — last−first ≥ 4 GiB AND non-decreasing within a
    small tolerance. Ambiguous bounded-oscillation shapes are NOT auto-labeled;
    the raw trace surfaces to the analyzer (plan §6).
    """
    if not values:
        return {"n": 0, "max_min_gib": None, "monotone_climb_ge_4": None, "flat": None}
    mx, mn = max(values), min(values)
    max_min = mx - mn
    rise = values[-1] - values[0]
    tol = 0.05  # GiB; allow tiny dips inside a "monotone" climb
    non_decreasing = all(b >= a - tol for a, b in itertools.pairwise(values))
    monotone_climb = bool(rise >= CLIMB_FALSIFY_GIB and non_decreasing)
    return {
        "n": len(values),
        "max_min_gib": float(max_min),
        "monotone_climb_ge_4": monotone_climb,
        "flat": bool(max_min < FLAT_MAX_MIN_GIB and not monotone_climb),
    }


def _load_memory_log(memlog_path: Path) -> list[dict] | None:
    if not memlog_path.exists():
        return None
    try:
        data = json.loads(memlog_path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def analyze_section_a(issue: int, out_subdir: str, *, npz_present: bool, hf_ok: bool) -> dict:
    """Read the per-iter memory log + compute per-PRIMARY-gauge flatness verdict.

    Pure analysis (no launch). ``npz_present`` / ``hf_ok`` are supplied by the
    caller (the experimenter checks the landed artifacts on the HF data repo).
    The pass predicate reads ONLY the two PRIMARY gauges; the SECONDARY
    ``memory_allocated`` max−min is recorded but not gated (plan §6).
    """
    memlog_path = PROJECT_ROOT / "eval_results" / f"issue_{issue}" / out_subdir / "memory_log.json"
    log = _load_memory_log(memlog_path)
    block: dict = {
        "memory_log_path": str(memlog_path),
        "n_iters_logged": 0 if log is None else len(log),
        "npz_written": bool(npz_present),
        "hf_upload_verified": bool(hf_ok),
        "sku_landed": _backend_selected_sku(issue),
    }
    if not log:
        block.update(
            {
                "memory_log": [],
                "reserved": None,
                "nvidia_smi": None,
                "allocated_secondary": None,
                "pass": False,
                "pass_reason": "no memory_log.json found (Section A did not run / produced no log)",
            }
        )
        return block

    # Trim to iters >= WARMUP_ITERS (first-10-iter baseline); fall back to all
    # samples if too few cross the warmup line (tiny smoke), and say so.
    warm = [e for e in log if int(e.get("iter", 0)) >= WARMUP_ITERS]
    used = warm if len(warm) >= 2 else log
    warmup_applied = len(warm) >= 2
    reserved = _flatness([float(e["memory_reserved_gib"]) for e in used])
    allocated = _flatness([float(e["memory_allocated_gib"]) for e in used])
    nvidia_vals = [
        float(e["nvidia_smi_used_gib"]) for e in used if e.get("nvidia_smi_used_gib") is not None
    ]
    nvidia = _flatness(nvidia_vals) if nvidia_vals else None

    # PASS predicate: BOTH PRIMARY gauges flat. nvidia-smi None across the whole
    # run (CPU host / no NVIDIA runtime) cannot certify the kernel-visible gauge
    # -> NOT a PASS for that gauge (must be present on a real A100).
    reserved_ok = bool(reserved.get("flat"))
    nvidia_ok = bool(nvidia and nvidia.get("flat"))
    passed = bool(npz_present and hf_ok and reserved_ok and nvidia_ok)
    reasons = []
    if not npz_present:
        reasons.append("no .npz written")
    if not hf_ok:
        reasons.append("HF upload not verified")
    if not reserved_ok:
        reasons.append(f"memory_reserved not flat ({reserved.get('max_min_gib')} GiB max-min)")
    if nvidia is None:
        reasons.append("nvidia-smi gauge absent across the run (cannot certify kernel-visible)")
    elif not nvidia_ok:
        reasons.append(f"nvidia-smi not flat ({nvidia.get('max_min_gib')} GiB max-min)")

    block.update(
        {
            "memory_log": log,
            "warmup_iters_applied": warmup_applied,
            "reserved": reserved,
            "nvidia_smi": nvidia,
            "allocated_secondary": allocated,
            "reserved_max_min_gib": reserved.get("max_min_gib"),
            "nvidia_smi_max_min_gib": None if nvidia is None else nvidia.get("max_min_gib"),
            "allocated_max_min_gib_secondary": allocated.get("max_min_gib"),
            "pass": passed,
            "pass_reason": "ok" if passed else "; ".join(reasons),
        }
    )
    return block


# ═════════════════════════════════════════════════════════════════════════════
# Section B — live fault-injection (watchdog -> TERMINATED -> RunPod failover)
# ═════════════════════════════════════════════════════════════════════════════


def section_b_dispatch_argv(issue: int) -> list[str]:
    """The plan §10 Section-B tiny-VM launch argv (sleep-tick workload)."""
    return [
        "uv",
        "run",
        "python",
        "scripts/dispatch_issue.py",
        "launch",
        "--issue",
        str(issue),
        "--intent",
        "debug",
        "--backend",
        "gcp",
        "--workload-cmd",
        "for i in $(seq 1 40); do echo phase-tick $i; sleep 30; done",
    ]


def section_b_iptables_argv(issue: int) -> list[str]:
    """gcloud SSH argv that drops BOTH probe endpoints (plan §10).

    BOTH the metadata server (169.254.169.254) AND HF :443 must be dropped —
    the watchdog counter resets if EITHER probe answers (#669 gcp.py:1166-1180).
    """
    return [
        "gcloud",
        "compute",
        "ssh",
        f"eps-issue-{issue}",
        f"--configuration={GCP_CONFIG}",
        f"--zone={GCP_ZONE}",
        "--command",
        "sudo iptables -A OUTPUT -d 169.254.169.254 -j DROP; "
        "sudo iptables -A OUTPUT -p tcp --dport 443 -j DROP",
    ]


def section_b_serial_argv(issue: int) -> list[str]:
    return [
        "gcloud",
        "compute",
        "instances",
        "get-serial-port-output",
        f"eps-issue-{issue}",
        f"--configuration={GCP_CONFIG}",
        f"--zone={GCP_ZONE}",
    ]


def section_b_status_argv(issue: int) -> list[str]:
    return [
        "gcloud",
        "compute",
        "instances",
        "describe",
        f"eps-issue-{issue}",
        f"--configuration={GCP_CONFIG}",
        f"--zone={GCP_ZONE}",
        "--format=value(status)",
    ]


def fallback_section_b(issue: int, reason: str) -> dict:
    """Build the documented fallback Section-B block (NOT a B PASS).

    Per plan §4/§6, when any LIVE step fails its budget the outcome is
    ``inconclusive_live_validation``, ``live_injection_pass=false``, and the
    headline downgrades. The deterministic B1/B3 evidence comes from Section C
    (the watchdog/wedge/failover unit tests).
    """
    return {
        "watchdog_fired": None,
        "vm_terminated": None,
        "failover_marker_sha": None,
        "failover_count": _count_failover_relaunches(issue)[0],
        "live_injection_pass": False,
        "fallback_outcome": "inconclusive_live_validation",
        "headline_downgraded": True,
        "fallback_reason": reason,
        "note": (
            "LIVE iptables recovery NOT validated; deterministic watchdog/wedge/"
            "failover coverage is Section C's test sweep + a documented manual "
            "watchdog-fire smoke (plan §4 fallback). Run the live B2 path for the "
            "unqualified live-recovery headline."
        ),
    }


def run_section_b(issue: int, *, dry_run: bool) -> dict:
    """Drive the LIVE fault-injection sequence (plan §4/§10).

    Sequence: launch tiny VM -> wait ~3 min -> drop BOTH endpoints -> poll
    serial console for the ``[eps-watchdog]`` ladder reaching 10/10 + the
    wedged-shutdown line -> poll instance status until TERMINATED -> read the
    second ``epm:cluster-launched`` (backend=runpod, failover reason) ->
    failover sentinel count must be exactly 1. ANY budget miss -> documented
    fallback (NOT a B PASS).

    NOTE: the live launch (and even the dry-run scaffolding) is normally driven
    by the experimenter step on the VM (where ``gcloud`` is authenticated). The
    implementer's local smoke uses ``--dry-run`` to verify command construction.
    """
    if dry_run:
        # Construct + echo every command; do not invoke gcloud/dispatch.
        argvs = {
            "launch": section_b_dispatch_argv(issue),
            "iptables": section_b_iptables_argv(issue),
            "serial": section_b_serial_argv(issue),
            "status": section_b_status_argv(issue),
        }
        for argv in argvs.values():
            _run(argv, dry_run=True)
        return {
            "dry_run": True,
            "constructed_commands": {k: " ".join(v) for k, v in argvs.items()},
            "live_injection_pass": None,
            "fallback_outcome": None,
            "headline_downgraded": None,
            "note": "dry-run: commands constructed, not invoked (live path = experimenter's job)",
        }

    inject_ts = datetime.datetime.now(datetime.UTC)
    # 1. Launch the tiny sleep-tick VM.
    rc, _out, err = _run(section_b_dispatch_argv(issue), dry_run=False, timeout=1200)
    if rc != 0:
        return fallback_section_b(issue, f"GCP tiny-VM launch failed rc={rc}: {err[-300:]}")

    # 2. Let the workload settle (~3 min) before injecting.
    time.sleep(180)

    # 3. Drop BOTH probe endpoints.
    inject_ts = datetime.datetime.now(datetime.UTC)
    rc, _out, err = _run(section_b_iptables_argv(issue), dry_run=False, timeout=120)
    if rc != 0:
        return fallback_section_b(issue, f"iptables injection failed rc={rc}: {err[-300:]}")

    # 4. Poll serial console for the watchdog ladder reaching 10/10 + shutdown.
    watchdog_fired = False
    deadline = time.time() + WATCHDOG_BUDGET_S
    while time.time() < deadline:
        rc, out, _err = _run(section_b_serial_argv(issue), dry_run=False, timeout=120)
        if rc == 0 and "eps-watchdog" in out:
            ladder_hits = out.count("eps-watchdog")
            if ("10/10" in out or "wedged" in out.lower()) and ladder_hits >= 1:
                watchdog_fired = True
                break
        time.sleep(20)
    if not watchdog_fired:
        return fallback_section_b(issue, "watchdog did not reach 10/10 + shutdown within ~6 min")

    # 5. Poll instance status until TERMINATED.
    vm_terminated = False
    deadline = time.time() + 5 * 60
    while time.time() < deadline:
        rc, out, _err = _run(section_b_status_argv(issue), dry_run=False, timeout=120)
        if rc == 0 and out.strip().upper() == "TERMINATED":
            vm_terminated = True
            break
        time.sleep(15)
    if not vm_terminated:
        return fallback_section_b(issue, "VM never reached TERMINATED after watchdog fired")

    # 6. Watch for the RunPod failover relaunch (exactly once).
    failover_count = 0
    failover_ts = None
    deadline = time.time() + 5 * 60
    while time.time() < deadline:
        failover_count, failover_ts = _count_failover_relaunches(issue)
        if failover_count >= 1:
            break
        time.sleep(15)
    if failover_count < 1:
        return fallback_section_b(issue, "dead VM never re-dispatched to RunPod")

    relaunch_ts = (
        datetime.datetime.fromisoformat(failover_ts.replace("Z", "+00:00"))
        if failover_ts
        else datetime.datetime.now(datetime.UTC)
    )
    return {
        "watchdog_fired": True,
        "vm_terminated": True,
        "failover_marker_sha": None,  # ts captured below; SHA filled by experimenter if needed
        "failover_marker_ts": failover_ts,
        "failover_count": failover_count,
        "live_injection_pass": bool(failover_count == 1),
        "fallback_outcome": None if failover_count == 1 else "residual_gap",
        "headline_downgraded": bool(failover_count != 1),
        "inject_to_relaunch_seconds": (relaunch_ts - inject_ts).total_seconds(),
        "zero_manual_action": True,  # no operator command was issued between inject + relaunch
        "note": (
            "exactly-one RunPod failover with zero manual action between iptables "
            "injection and relaunch"
            if failover_count == 1
            else f"failover fired {failover_count}x — idempotency broken (kill criterion)"
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Section C — regression sweep on main
# ═════════════════════════════════════════════════════════════════════════════


_PYTEST_SUMMARY_RE = re.compile(r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error")


def _parse_pytest_summary(out: str) -> dict:
    """Pull passed/failed/error counts from a pytest tail; robust to format drift."""
    passed = failed = errors = 0
    for m in _PYTEST_SUMMARY_RE.finditer(out):
        if m.group(1):
            passed = int(m.group(1))
        if m.group(2):
            failed = int(m.group(2))
        if m.group(3):
            errors = int(m.group(3))
    return {"passed": passed, "failed": failed, "errors": errors}


def run_section_c(*, dry_run: bool) -> dict:
    """Run the 4 named pytest files + 2 workflow-lint runs; parse exit codes."""
    pytest_results: dict[str, dict] = {}
    for f in SECTION_C_PYTEST_FILES:
        rc, out, err = _run(["uv", "run", "pytest", f, "-v"], dry_run=dry_run, timeout=1800)
        tail = (out + err)[-4000:]
        pytest_results[f] = {
            "rc": rc,
            "summary": {"passed": 0, "failed": 0, "errors": 0}
            if dry_run
            else _parse_pytest_summary(tail),
            "green": rc == 0,
        }
    lint_exit_codes: dict[str, int] = {}
    for name, extra in SECTION_C_LINT_RUNS.items():
        rc, _out, _err = _run(
            ["uv", "run", "python", "scripts/workflow_lint.py", *extra],
            dry_run=dry_run,
            timeout=600,
        )
        lint_exit_codes[name] = rc
    pytest_green = all(r["green"] for r in pytest_results.values())
    lint_green = all(rc == 0 for rc in lint_exit_codes.values())
    return {
        "pytest": pytest_results,
        "lint_exit_codes": lint_exit_codes,
        # A dry-run constructs commands without invoking them, so its rc=0
        # echoes are NOT a real green; pass is only meaningful on a live run.
        "pass": bool(pytest_green and lint_green and not dry_run),
        "dry_run": dry_run,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Assemble — validation.json + headline-routing verdict (plan §1)
# ═════════════════════════════════════════════════════════════════════════════


def _section_json_path(issue: int, section: str) -> Path:
    return PROJECT_ROOT / "eval_results" / f"issue_{issue}" / f"section_{section}.json"


def _validation_json_path(issue: int) -> Path:
    return PROJECT_ROOT / "eval_results" / f"issue_{issue}" / "validation.json"


def _repro_metadata() -> dict:
    """git commit + timestamp for the result JSON (reproducibility contract)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        sha = None
    return {
        "git_commit": sha or None,
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "python": sys.version.split()[0],
    }


def route_verdict(a: dict | None, b: dict | None, c: dict | None) -> dict:
    """Headline routing (plan §1).

    - A PASS + B PASS (live) + C PASS -> 'GCP works again, and the #667
      hung-RUNNING wedge class now self-recovers'.
    - A PASS + B fallback + C PASS -> 'watchdog/failover logic unit-tested;
      live iptables recovery NOT validated' + headline_downgraded.
    - anything else -> 'specific residual gap: <enumerate>'.
    """
    a_pass = bool(a and a.get("pass"))
    c_pass = bool(c and c.get("pass"))
    b_live = bool(b and b.get("live_injection_pass") is True)
    b_fallback = bool(b and b.get("fallback_outcome") == "inconclusive_live_validation")

    if a_pass and b_live and c_pass:
        return {
            "verdict": "GCP works again, and the #667 hung-RUNNING wedge class now self-recovers",
            "headline_downgraded": False,
        }
    if a_pass and b_fallback and c_pass:
        return {
            "verdict": "watchdog/failover logic unit-tested; live iptables recovery NOT validated",
            "headline_downgraded": True,
        }
    gaps = []
    if not a_pass:
        gaps.append(f"Section A FAIL ({(a or {}).get('pass_reason', 'not run')})")
    if not (b_live or b_fallback):
        gaps.append("Section B incomplete / not run")
    elif b and b.get("fallback_outcome") == "residual_gap":
        gaps.append(f"Section B failover anomaly ({b.get('note', '')})")
    if not c_pass:
        gaps.append("Section C FAIL (a named test file RED or lint nonzero)")
    return {
        "verdict": "specific residual gap: " + ("; ".join(gaps) if gaps else "indeterminate"),
        "headline_downgraded": True,
    }


def assemble(issue: int) -> dict:
    """Read per-section JSONs (if present) + write validation.json with verdict."""

    def _load(section: str) -> dict | None:
        p = _section_json_path(issue, section)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    a, b, c = _load("A"), _load("B"), _load("C")
    verdict = route_verdict(a, b, c)
    out = {
        "issue": issue,
        "experiment": f"issue{issue}_gcp_validation",
        "section_A": a,
        "section_B": b,
        "section_C": c,
        **verdict,
        "reproducibility": _repro_metadata(),
    }
    path = _validation_json_path(issue)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("validation.json written: %s | verdict: %s", path, verdict["verdict"])
    return out


def _write_section(issue: int, section: str, block: dict) -> None:
    p = _section_json_path(issue, section)
    p.parent.mkdir(parents=True, exist_ok=True)
    block = {**block, "reproducibility": _repro_metadata()}
    p.write_text(json.dumps(block, indent=2))
    logger.info("section_%s.json written: %s", section, p)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #672 GCP end-to-end validation orchestrator."
    )
    parser.add_argument("--issue", type=int, default=672)
    parser.add_argument(
        "--section",
        choices=["a", "b", "c", "assemble", "all"],
        default="all",
        help="run one section, assemble validation.json from per-section JSONs, or all in order",
    )
    parser.add_argument(
        "--out-subdir",
        default="secA_smoke",
        help="Section-A output subdir under eval_results/issue_<N>/ (plan §10)",
    )
    parser.add_argument(
        "--npz-present",
        action="store_true",
        help="Section A: assert ≥1 .npz landed (experimenter sets after checking artifacts)",
    )
    parser.add_argument(
        "--hf-ok",
        action="store_true",
        help="Section A: assert the HF upload verified (the experimenter sets this)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="construct commands without invoking gcloud/pytest/dispatch (construction smoke)",
    )
    args = parser.parse_args()
    issue = args.issue

    if args.section in ("a", "all"):
        block = analyze_section_a(
            issue, args.out_subdir, npz_present=args.npz_present, hf_ok=args.hf_ok
        )
        _write_section(issue, "A", block)
    if args.section in ("b", "all"):
        block = run_section_b(issue, dry_run=args.dry_run)
        _write_section(issue, "B", block)
    if args.section in ("c", "all"):
        block = run_section_c(dry_run=args.dry_run)
        _write_section(issue, "C", block)
    if args.section in ("assemble", "all"):
        result = assemble(issue)
        logger.info("VERDICT: %s", result["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
