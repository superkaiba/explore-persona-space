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
MIN_LOG_ENTRIES = 30  # plan §3/§6: ≥30 hooked-forward samples for a meaningful signal
MIN_MAX_ITER = 30  # the logger must have REACHED iter≥30 (not just emitted ≥30 rows)
MIN_POST_WARMUP_PER_GAUGE = 2  # ≥2 post-warmup (iter>10) non-None samples per PRIMARY gauge

# ── Section-B watchdog / failover constants (plan §10, #669) ─────────────────
GCP_ZONE = "us-central1-b"
GCP_CONFIG = "eps-gcp"
WATCHDOG_BUDGET_S = 6 * 60  # ladder must reach 10/10 + wedged-shutdown within ~6 min
FAILOVER_REASON_RE = re.compile(r"gcp_workload_failover_runpod")

# ── Section-B looped-poller constants (round-3 Critical #1 / Major #3) ────────
# ``backend_poll.py`` is ONE-SHOT (its ``main()`` polls once + returns); the
# failover (``_failover_dead_gcp_to_runpod``) fires only on the poll where the
# VM has already reached its terminal/wedged state. So the VALIDATOR owns the
# loop: it re-invokes ``backend_poll.py --issue N`` as a synchronous subprocess
# on a cadence until a scoped failover marker lands (or the budget expires).
POLLER_LOOP_BUDGET_S = 10 * 60  # overall Section-B post-injection poll window
POLLER_INTERVAL_S = 30  # seconds between one-shot backend_poll.py invocations
POLLER_CALL_TIMEOUT_S = 60  # per-invocation subprocess timeout
# After the FIRST scoped failover marker lands, keep polling for this long with
# NO new marker before the FINAL scoped count — so a second failover landing
# seconds later (idempotency breach, the H-B kill criterion) is not missed
# (round-3 Major #3).
POLLER_QUIET_PERIOD_S = 60


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


def _parse_marker_ts(raw: object) -> datetime.datetime | None:
    """Parse an events.jsonl ``ts`` string to an aware UTC datetime, or None.

    Marker timestamps are ISO-8601 with a trailing ``Z``; tolerate ``+00:00``
    too. A malformed / absent ts returns None (the caller treats an unparseable
    marker as OUTSIDE the post-inject window — it cannot prove it is newer than
    ``since_ts``, so it must not count toward the live failover).
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _count_failover_relaunches(
    issue: int, *, since_ts: datetime.datetime | None = None
) -> tuple[int, str | None]:
    """Count cluster-launched markers whose note names a RunPod failover reason.

    Returns ``(count, latest_marker_ts)``. The B-PASS predicate requires count
    == exactly 1 (idempotent failover-once; failover twice falsifies H-B).

    ``since_ts`` (the iptables-injection wall-clock captured at the start of the
    live sequence) SCOPES the count to the CURRENT injected run: only failover
    markers with ``ts >= since_ts`` count. Without it a stale failover marker —
    from a Section-A run, a prior #672 validation attempt, or any earlier
    failover on the same task — would falsely satisfy the live-recovery predicate
    (round-2 BLOCKER 1). A marker whose ``ts`` is unparseable / absent is EXCLUDED
    when ``since_ts`` is set (it cannot prove it post-dates the injection).
    ``since_ts=None`` preserves the unfiltered count (used by the fallback block,
    where the count is diagnostic only, never a PASS gate).
    """
    hits = []
    for e in _events(issue):
        if e.get("kind") not in ("epm:cluster-launched", "epm:backend-selected"):
            continue
        if not FAILOVER_REASON_RE.search(str(e.get("note", ""))):
            continue
        if since_ts is not None:
            marker_ts = _parse_marker_ts(e.get("ts"))
            if marker_ts is None or marker_ts < since_ts:
                continue
        hits.append(e)
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
    FIXED extractor with ``--log-mem-every 1`` writing to
    ``eval_results/issue_<N>/<out_subdir>``.

    NOTE (round-3 Critical #2): the cadence is ``--log-mem-every 1``, NOT the
    plan §11 line-140 value of ``10``. Plan §9 bounds the smoke slice at
    ``<100`` hooked 7B forwards; at every-10th-forward that yields only ~10 log
    rows, but the round-2 coverage gate (``MIN_LOG_ENTRIES``, currently 30)
    requires ≥30 rows — so a HEALTHY Section-A run could never reach PASS. At
    every-iter cadence the slice yields up to ~100 rows, comfortably above the
    floor. The 3-gauge per-row cost (two ``torch.cuda.memory_*`` reads + one
    ``nvidia-smi`` shell-out) is negligible next to a 7B forward, so the finer
    cadence does not perturb the flatness signal. This is a deliberate, recorded
    plan-vs-code deviation (the §11 value was mutually unsatisfiable with the §6
    coverage floor).
    """
    workload = (
        'REPO_ROOT="$WORKLOAD_ROOT" uv run python scripts/issue667_extract.py '
        "--behavior marker --source-cid default --targets default,sp_swe "
        "--layers 7 14 21 --primary-layer 14 --max-probes 8 --max-new-tokens 256 "
        f"--log-mem-every 1 --out eval_results/issue_{issue}/{out_subdir} --gpu-id 0"
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
                "coverage_issue": "no memory_log.json found (Section A did not run / no log)",
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

    # COVERAGE GATE (round-2 BLOCKER 3). A truncated / barely-working logger (the
    # exact #671 failure class this task certifies) would otherwise trivially
    # certify "flat memory" on 1 sample. Require enough samples + reach + per-gauge
    # post-warmup depth for the flat-vs-climbing signal to be meaningful (plan
    # §3/§6: ≥30 hooked forwards). A shortfall sets pass=False AND surfaces the
    # specific gap in section_A.coverage_issue.
    n_entries = len(log)
    max_iter = max((int(e.get("iter", 0)) for e in log), default=0)
    # Per-gauge post-warmup (iter > WARMUP_ITERS) non-None sample counts.
    post_warmup = [e for e in log if int(e.get("iter", 0)) > WARMUP_ITERS]
    reserved_post = sum(1 for e in post_warmup if e.get("memory_reserved_gib") is not None)
    nvidia_post = sum(1 for e in post_warmup if e.get("nvidia_smi_used_gib") is not None)
    coverage_problems = []
    if n_entries < MIN_LOG_ENTRIES:
        coverage_problems.append(f"only {n_entries} sample(s) logged; require ≥{MIN_LOG_ENTRIES}")
    if max_iter < MIN_MAX_ITER:
        coverage_problems.append(
            f"logger reached iter={max_iter}; require max iter ≥{MIN_MAX_ITER}"
        )
    if reserved_post < MIN_POST_WARMUP_PER_GAUGE:
        coverage_problems.append(
            f"memory_reserved has {reserved_post} post-warmup (iter>{WARMUP_ITERS}) sample(s); "
            f"require ≥{MIN_POST_WARMUP_PER_GAUGE}"
        )
    if nvidia_post < MIN_POST_WARMUP_PER_GAUGE:
        coverage_problems.append(
            f"nvidia_smi has {nvidia_post} post-warmup (iter>{WARMUP_ITERS}) non-None sample(s); "
            f"require ≥{MIN_POST_WARMUP_PER_GAUGE} (None on every post-warmup sample fails this)"
        )
    coverage_issue = "; ".join(coverage_problems) if coverage_problems else None

    # PASS predicate: coverage gate met AND BOTH PRIMARY gauges flat. nvidia-smi
    # None across the whole run (CPU host / no NVIDIA runtime) cannot certify the
    # kernel-visible gauge -> NOT a PASS for that gauge (must be present on a real
    # A100).
    reserved_ok = bool(reserved.get("flat"))
    nvidia_ok = bool(nvidia and nvidia.get("flat"))
    coverage_ok = coverage_issue is None
    passed = bool(npz_present and hf_ok and coverage_ok and reserved_ok and nvidia_ok)
    reasons = []
    if not npz_present:
        reasons.append("no .npz written")
    if not hf_ok:
        reasons.append("HF upload not verified")
    if not coverage_ok:
        reasons.append(f"insufficient coverage ({coverage_issue})")
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
            "max_iter": max_iter,
            "reserved_post_warmup_samples": reserved_post,
            "nvidia_post_warmup_samples": nvidia_post,
            "coverage_issue": coverage_issue,
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


def section_b_poller_argv(issue: int) -> list[str]:
    """The ONE-SHOT ``backend_poll.py`` argv that performs the GCP -> RunPod failover.

    The GCP->RunPod failover (``_failover_dead_gcp_to_runpod``) is wired inside
    ``scripts/backend_poll.py``'s ``main()`` (per
    ``.claude/rules/compute-backend-failover.md`` + ``backend_poll.py:1001``).
    But ``main()`` is ONE-SHOT — it parses args, calls ``backend.poll(handle)``
    ONCE, runs the async-failover discrimination, prints one JSON result, and
    returns. There is NO internal poll loop and NO ``--watch`` flag. The
    failover fires only on the SINGLE poll where the VM has already reached its
    terminal/wedged state.

    So the VALIDATOR owns the loop (round-3 Critical #1): ``run_section_b``
    re-invokes THIS argv as a SYNCHRONOUS subprocess on a cadence (see
    ``_loop_poller_until_failover``) AFTER the fault is injected, until a scoped
    failover marker lands. A bg-Popen-once-before-injection (the round-2
    misimplementation) cannot work against a one-shot command — it polls a
    still-healthy VM once, exits, and is dead by the time the watchdog
    terminates the VM minutes later. The in-VM watchdog only writes
    ``eps/phase=wedged`` and powers off → TERMINATED; it does NOT re-dispatch,
    so without re-invoking this poller post-termination the dead VM never
    re-dispatches and the live-recovery headline is structurally unreachable
    (round-2 BLOCKER 2).
    """
    return [
        "uv",
        "run",
        "python",
        "scripts/backend_poll.py",
        "--issue",
        str(issue),
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


def section_b_dispatch_plan(issue: int) -> dict:
    """The Section-B command plan: per-command argv + the LOOPED-poller schedule.

    The poller is NOT a single pre-injection Popen (the round-2 misimplementation
    against a one-shot ``backend_poll.py``); ``run_section_b`` LOOPS it
    synchronously AFTER injection. This plan dict surfaces the loop schedule
    (``poller_loop``) so the dry-run smoke can assert the looped construction
    (round-3 Critical #1 mechanizable check) — the poller argv plus the cadence /
    budget / quiet-period that drive the repeated calls.
    """
    argvs = {
        "launch": section_b_dispatch_argv(issue),
        "poller": section_b_poller_argv(issue),
        "iptables": section_b_iptables_argv(issue),
        "serial": section_b_serial_argv(issue),
        "status": section_b_status_argv(issue),
    }
    return {
        "argvs": argvs,
        "poller_loop": {
            "poller_argv": " ".join(argvs["poller"]),
            "looped": True,  # validator-owned loop; NOT a single pre-injection Popen
            "interval_s": POLLER_INTERVAL_S,
            "budget_s": POLLER_LOOP_BUDGET_S,
            "call_timeout_s": POLLER_CALL_TIMEOUT_S,
            "quiet_period_s": POLLER_QUIET_PERIOD_S,
            # Upper bound on synchronous one-shot invocations across the window.
            "max_invocations": POLLER_LOOP_BUDGET_S // POLLER_INTERVAL_S,
        },
    }


def _section_b_dry_run(issue: int) -> dict:
    """Construct + echo every Section-B command without invoking gcloud/dispatch.

    The poller argv MUST appear here so the smoke verifies backend_poll.py IS
    scheduled — and the ``poller_loop`` block proves it is LOOPED, not a single
    pre-injection Popen (round-3 Critical #1). To make the repetition concrete,
    the dry-run echoes the one-shot poller argv twice (a representative loop
    sample) into ``poller_invocations``; the live path runs up to
    ``max_invocations`` of them.
    """
    plan = section_b_dispatch_plan(issue)
    argvs = plan["argvs"]
    for argv in argvs.values():
        _run(argv, dry_run=True)
    # Echo the poller call twice to demonstrate the loop (NOT one pre-inject Popen).
    poller_invocations = []
    for _ in range(2):
        rc, out, _err = _run(argvs["poller"], dry_run=True)
        poller_invocations.append(
            {
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                "exit_code": rc,
                "stdout_head": (out or "").strip()[:300],
            }
        )
    return {
        "dry_run": True,
        "constructed_commands": {k: " ".join(v) for k, v in argvs.items()},
        "poller_loop": plan["poller_loop"],
        "poller_invocations": poller_invocations,
        "poller_exit_codes": [inv["exit_code"] for inv in poller_invocations],
        "live_injection_pass": None,
        "fallback_outcome": None,
        "headline_downgraded": None,
        "note": (
            "dry-run: commands constructed, not invoked (live path = experimenter's job). "
            "Poller is LOOPED synchronously post-injection (poller_loop), NOT a single "
            "pre-injection Popen against the one-shot backend_poll.py."
        ),
    }


def _poll_watchdog_fired(issue: int) -> bool:
    """Poll the serial console until the ``[eps-watchdog]`` ladder reaches 10/10
    + the wedged-shutdown line, within the watchdog budget."""
    deadline = time.time() + WATCHDOG_BUDGET_S
    while time.time() < deadline:
        rc, out, _err = _run(section_b_serial_argv(issue), dry_run=False, timeout=120)
        if rc == 0 and "eps-watchdog" in out:
            ladder_hits = out.count("eps-watchdog")
            if ("10/10" in out or "wedged" in out.lower()) and ladder_hits >= 1:
                return True
        time.sleep(20)
    return False


def _poll_vm_terminated(issue: int) -> bool:
    """Poll instance status until TERMINATED, within a 5-min budget."""
    deadline = time.time() + 5 * 60
    while time.time() < deadline:
        rc, out, _err = _run(section_b_status_argv(issue), dry_run=False, timeout=120)
        if rc == 0 and out.strip().upper() == "TERMINATED":
            return True
        time.sleep(15)
    return False


def _invoke_poller_once(issue: int) -> dict:
    """Invoke ``backend_poll.py --issue N`` ONCE, synchronously (round-3 Critical #1).

    The poller is one-shot: each call polls the GCP VM once, runs the
    async-failover discrimination, and (on the poll where the VM has reached a
    terminal/wedged state) drives ``_failover_dead_gcp_to_runpod``. Returns an
    invocation-record dict ``{ts, exit_code, stdout_head}`` for the
    ``poller_invocations`` audit trail. A subprocess timeout is recorded with
    ``exit_code=None`` (the call is treated as a non-terminal tick, not a crash).
    """
    ts = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        rc, out, _err = _run(
            section_b_poller_argv(issue), dry_run=False, timeout=POLLER_CALL_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return {"ts": ts, "exit_code": None, "stdout_head": "[timeout]"}
    return {"ts": ts, "exit_code": rc, "stdout_head": (out or "").strip()[:300]}


def _loop_poller_until_failover(
    issue: int, *, since_ts: datetime.datetime
) -> tuple[int, str | None, list[dict]]:
    """Loop the one-shot poller until a scoped failover lands, then quiet-period.

    Round-3 Critical #1 + Major #3. The VALIDATOR owns the loop because
    ``backend_poll.py`` is one-shot. Sequence:

    1. Every ``POLLER_INTERVAL_S`` (until ``POLLER_LOOP_BUDGET_S`` from the start
       of this loop), invoke the poller once and re-read the scoped failover
       count (``ts >= since_ts``). The poll that observes the VM terminal/wedged
       is the one that DRIVES the failover, so the count flips on a poller call.
    2. On the FIRST scoped marker, do NOT stop immediately. Continue invoking the
       poller for a quiet period (``POLLER_QUIET_PERIOD_S`` with no NEW marker)
       so a SECOND failover landing seconds later (idempotency breach — the H-B
       kill criterion "failover fires twice") is observed (round-3 Major #3).
    3. FINAL scoped count after the quiet period (or budget exhaustion) decides.

    Returns ``(final_failover_count, latest_failover_ts, poller_invocations)``.
    """
    invocations: list[dict] = []
    deadline = time.time() + POLLER_LOOP_BUDGET_S
    count, ts = 0, None
    last_count = 0  # the scoped count observed on the PREVIOUS iteration
    quiet_since: float | None = None  # wall-clock the quiet-period clock (re)started

    while time.time() < deadline:
        invocations.append(_invoke_poller_once(issue))
        count, ts = _count_failover_relaunches(issue, since_ts=since_ts)
        if count >= 1 and (quiet_since is None or count > last_count):
            # First marker, OR a NEW marker landed -> (re)start the quiet clock so
            # a 2nd/3rd failover seconds later (idempotency breach, the H-B kill
            # criterion "failover fires twice") is never missed (round-3 Major #3).
            quiet_since = time.time()
            logger.info(
                "scoped failover marker observed (count=%d); (re)starting quiet-period", count
            )
        last_count = count
        if quiet_since is not None and time.time() - quiet_since >= POLLER_QUIET_PERIOD_S:
            break
        time.sleep(POLLER_INTERVAL_S)

    # FINAL scoped read (covers a marker landing in the last interval).
    count, ts = _count_failover_relaunches(issue, since_ts=since_ts)
    return count, ts, invocations


def run_section_b(issue: int, *, dry_run: bool) -> dict:
    """Drive the LIVE fault-injection sequence (plan §4/§10).

    Sequence: launch tiny VM -> wait ~3 min -> drop BOTH endpoints -> poll serial
    console for the ``[eps-watchdog]`` ladder reaching 10/10 + the
    wedged-shutdown line -> poll instance status until TERMINATED -> LOOP the
    one-shot ``backend_poll.py`` (the failover driver) until a scoped
    ``epm:backend-selected`` (backend=runpod, failover reason, ``ts >=
    inject_ts``) lands, run the quiet-period extension, then read the FINAL
    scoped count -> failover count must be exactly 1. ANY budget miss ->
    documented fallback (NOT a B PASS).

    The poller is LOOPED by this validator (round-3 Critical #1): ``backend_poll
    .py`` is one-shot, so a single pre-injection bg launch (the round-2 design)
    polls a still-healthy VM once and is dead by watchdog-kill time. Here the
    poller is re-invoked synchronously on a cadence AFTER injection +
    termination, which is when the failover actually fires. Every Section-B
    return surfaces ``poller_invocations`` (round-3 Major #1/#4: the per-call
    audit trail) so an early fallback never omits the poller-call record.

    NOTE: the live launch (and even the dry-run scaffolding) is normally driven
    by the experimenter step on the VM (where ``gcloud`` is authenticated). The
    implementer's local smoke uses ``--dry-run`` to verify command construction.
    """
    if dry_run:
        return _section_b_dry_run(issue)

    # 1. Launch the tiny sleep-tick VM.
    rc, _out, err = _run(section_b_dispatch_argv(issue), dry_run=False, timeout=1200)
    if rc != 0:
        return {
            **fallback_section_b(issue, f"GCP tiny-VM launch failed rc={rc}: {err[-300:]}"),
            "poller_invocations": [],  # poller loop not reached on this early fallback
            "poller_exit_codes": [],  # round-3 Major #4: uniform key on EVERY return
        }

    # 2. Let the workload settle (~3 min) before injecting.
    time.sleep(180)

    # 3. Drop BOTH probe endpoints.
    inject_ts = datetime.datetime.now(datetime.UTC)
    rc, _out, err = _run(section_b_iptables_argv(issue), dry_run=False, timeout=120)
    if rc != 0:
        return {
            **fallback_section_b(issue, f"iptables injection failed rc={rc}: {err[-300:]}"),
            "poller_invocations": [],  # poller loop not reached on this early fallback
            "poller_exit_codes": [],  # round-3 Major #4: uniform key on EVERY return
        }

    # 4. Watchdog ladder reaches 10/10 + shutdown.
    if not _poll_watchdog_fired(issue):
        return {
            **fallback_section_b(issue, "watchdog did not reach 10/10 + shutdown within ~6 min"),
            "poller_invocations": [],  # poller loop not reached on this early fallback
            "poller_exit_codes": [],  # round-3 Major #4: uniform key on EVERY return
        }

    # 5. Instance reaches TERMINATED.
    if not _poll_vm_terminated(issue):
        return {
            **fallback_section_b(issue, "VM never reached TERMINATED after watchdog fired"),
            "poller_invocations": [],  # poller loop not reached on this early fallback
            "poller_exit_codes": [],  # round-3 Major #4: uniform key on EVERY return
        }

    # 6. LOOP the one-shot poller until the scoped RunPod failover lands +
    #    quiet-period (exactly once; ts >= inject_ts). The validator owns the
    #    loop — backend_poll.py is one-shot (round-3 Critical #1 / Major #3).
    failover_count, failover_ts, poller_invocations = _loop_poller_until_failover(
        issue, since_ts=inject_ts
    )
    # Round-3 Major #4: surface the per-call exit-code list on EVERY return path.
    poller_exit_codes = [inv["exit_code"] for inv in poller_invocations]
    poller_meta = {
        "poller_argv": " ".join(section_b_poller_argv(issue)),
        "poller_invocations": poller_invocations,
        "poller_exit_codes": poller_exit_codes,
    }
    if failover_count < 1:
        return {
            **fallback_section_b(issue, "dead VM never re-dispatched to RunPod"),
            **poller_meta,
        }

    relaunch_ts = (
        datetime.datetime.fromisoformat(failover_ts.replace("Z", "+00:00"))
        if failover_ts
        else datetime.datetime.now(datetime.UTC)
    )
    return {
        **poller_meta,
        "watchdog_fired": True,
        "vm_terminated": True,
        "failover_marker_sha": None,  # ts below; SHA filled by experimenter if needed
        "failover_marker_ts": failover_ts,
        "failover_count": failover_count,
        "live_injection_pass": bool(failover_count == 1),
        "fallback_outcome": None if failover_count == 1 else "residual_gap",
        "headline_downgraded": bool(failover_count != 1),
        "inject_to_relaunch_seconds": (relaunch_ts - inject_ts).total_seconds(),
        "zero_manual_action": True,  # no operator command issued between inject + relaunch
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

    Defense-in-depth (round-2 #4): the unqualified live headline requires the
    FULL conjunction ``live_injection_pass is True AND failover_count == 1 AND
    fallback_outcome is None`` re-asserted HERE — not merely the upstream
    ``live_injection_pass`` flag. ``run_section_b`` already enforces these
    jointly (it sets ``live_injection_pass = (failover_count == 1)`` and
    ``fallback_outcome`` accordingly), so the real data flow can never present an
    inconsistent dict; this re-check makes a HAND-BUILT inconsistent dict (e.g.
    ``{live_injection_pass: True, failover_count: 2}``) route to a residual gap
    rather than getting the live PASS headline.
    """
    a_pass = bool(a and a.get("pass"))
    c_pass = bool(c and c.get("pass"))
    b_live = bool(
        b
        and b.get("live_injection_pass") is True
        and b.get("failover_count") == 1
        and b.get("fallback_outcome") is None
    )
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
    # Section B ran the live injection but the headline is not earned: the
    # failover was anomalous (fired twice — idempotency broken) OR the
    # consistency re-check above rejected an inconsistent dict.
    b_residual = bool(
        b
        and (
            b.get("fallback_outcome") == "residual_gap"
            or (b.get("failover_count") is not None and b.get("failover_count") != 1)
        )
    )
    gaps = []
    if not a_pass:
        gaps.append(f"Section A FAIL ({(a or {}).get('pass_reason', 'not run')})")
    if b_residual:
        # Section B RAN the live injection but the failover was anomalous
        # (e.g. fired twice — idempotency broken, the H-B kill criterion).
        gaps.append(
            f"Section B failover anomaly (failover_count={(b or {}).get('failover_count')}; "
            f"{(b or {}).get('note', '')})"
        )
    elif not (b_live or b_fallback):
        gaps.append("Section B incomplete / not run")
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
