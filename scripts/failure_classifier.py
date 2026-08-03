"""Failure-class routing for `epm:failure` markers.

The `/issue` skill Step 7 invokes this helper as a subprocess
(`python scripts/failure_classifier.py --body <body> [--log <path>]`)
to decide whether to re-spawn the experimenter (infra failures: OOM,
NCCL, library tracebacks, ...) or the experiment-implementer (code
failures: tracebacks from our code, AssertionError, ...).

This module is the SINGLE SOURCE OF TRUTH for the regex pattern list.
`.claude/skills/issue/failure_patterns.md` is a human-readable mirror
that documents the same patterns for agents/reviewers; the markdown
file MUST stay in sync with the regex list below, but it is NOT
consulted at runtime — the SKILL Step 7 shells out to this script.

CLI:
  python scripts/failure_classifier.py --body <body-text>
  python scripts/failure_classifier.py --body <body-text> --log <path>
  python scripts/failure_classifier.py --body <body-text> --stall-reason <reason>
  cat body.txt | python scripts/failure_classifier.py --body -

On a ``status=stalled`` poll tick (SKILL Step 6d.2 -> Step 7) the
orchestrator forwards the poll JSON line's machine-readable
``stall_reason`` field via ``--stall-reason``. A known stall reason
(``STALL_REASON_INFRA``) routes ``infra`` directly: a silent hang's log
tail carries no traceback / OOM line, so the regex fallback would
otherwise misroute it to the conservative ``code`` default. See
``scripts/poll_pipeline.py`` (the emit surface) + ``scripts/backend_poll.py``.

Stdout: a single line, ``infra`` or ``code``. Exit 0 on success.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

FailureClass = Literal["infra", "code"]

# Machine-readable stall reasons (poll JSON line `stall_reason`, emitted by
# `scripts/poll_pipeline.py` / `scripts/backend_poll.py`) that route `infra`
# DIRECTLY — independent of the log tail. A silent hang produces no
# traceback / OOM line, so the regex fallback below would misroute it to the
# conservative `code` default. Each reason here is recoverable by an
# experimenter respawn (NOT a code fix), so it belongs on the infra row of
# the Step 7 routing table. Keep in sync with the `stall_reason` values
# `poll_pipeline.py` / `backends/slurm_monitor.py` can set.
#   - "vllm_worker_dead_zombie_gpu": a CUDA-worker PID died but holds VRAM
#     while the vLLM EngineCore main process stays alive (#664 r8); the
#     experimenter respawn reaps the orphan by exact PID and relaunches on
#     the same pod (recipe: `.claude/rules/gotchas.md` crash-orphan
#     EngineCore entry + `.claude/agent-memory/experimenter/
#     feedback_vllm_zombie_gpu_pkill_reaper.md`).
#   - "persistent_wedge_veto_yield": the #864/#1216 namespace veto yielded
#     after N consecutive suppressed ticks in the wedge regime (#1840 —
#     alive-but-wedged workers, host-unresolvable PIDs, frozen
#     logs+outputs, every GPU idle, no material CPU; #1768's ~16h
#     HF-download futex wedge). Recoverable by an experimenter respawn:
#     the wedged workers are ALIVE but host-namespace-unresolvable by
#     construction, so recovery kills them CONTAINER-side by
#     distinctive-cmdline/session match (`pgrep -af` inside the pod),
#     never by the nvidia-smi-reported host PID, then relaunches.
#   - "slurm_heartbeat_and_log_stale": the SLURM monitor's stall verdict
#     (#1969, `backends/slurm_monitor.build_poll_result`) — SLURM says
#     RUNNING while the rsync'd status.json heartbeat AND job.out mtime
#     are BOTH older than STALL_SEC for >= 2 consecutive ticks. A
#     genuine cluster-side silent hang: its log tail carries no
#     traceback/OOM pattern, so the regex fallback would misroute it to
#     `code`; the recovery is an experimenter respawn (scancel +
#     resubmit), not a code fix.
STALL_REASON_INFRA: frozenset[str] = frozenset(
    {
        "vllm_worker_dead_zombie_gpu",
        "persistent_wedge_veto_yield",
        "slurm_heartbeat_and_log_stale",
    }
)

# Infra log patterns (regex, case-insensitive). This module is the source of
# truth; mirrored by `.claude/skills/issue/failure_patterns.md`. Any match →
# route as `infra`.
INFRA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OOM-killer|Killed\b", re.IGNORECASE),
    re.compile(r"No space left on device|ENOSPC|disk full", re.IGNORECASE),
    re.compile(r"NCCL (timeout|error)", re.IGNORECASE),
    re.compile(
        r"SSH connection refused|No route to host|Connection timed out",
        re.IGNORECASE,
    ),
    re.compile(r"401 Unauthorized|gated repo", re.IGNORECASE),
    re.compile(r"RuntimeError: CUDA error", re.IGNORECASE),
    re.compile(r"Failed to initialize.*vllm", re.IGNORECASE),
    # vLLM engine-init free-memory check (v1 gpu_worker raises ValueError:
    # "Free memory on device (X/Y GiB) on startup is less than desired GPU
    # memory utilization (...)"). On a RELAUNCH this usually means orphaned
    # `VLLM::EngineCore` workers from a prior crashed run still hold the
    # GPUs — their cmdline carries no script name, so the natural
    # `pgrep -f <script>` liveness probe reads clean while ~50 GB/GPU is
    # held. RECOVERABLE IN-PLACE, not a capacity problem: probe
    # `pgrep -af 'EngineCor[e]'` + `nvidia-smi
    # --query-compute-apps=pid,used_memory --format=csv`, kill the orphans
    # (`kill`, then `kill -9` survivors), confirm GPU memory ~0, and
    # relaunch on the SAME pod BEFORE any fresh-pod / capacity
    # reclassification. See `.claude/rules/gotchas.md` (crash-orphan
    # EngineCore) + `.claude/agents/experimenter.md` Pre-Launch step 9.
    # Named here so a body carrying only the final error line (no vllm/
    # traceback frames) still routes `infra` instead of falling through to
    # the conservative `code` default. Incident #601 (2026-06-11).
    re.compile(
        r"Free memory on device.*?is less than desired GPU memory utilization",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Traceback.*\b(vllm|transformers|peft|trl|torch|xformers)/",
        re.IGNORECASE | re.DOTALL,
    ),
    # §2 watchdog reasons. The watchdog already sets `failure_class: infra`
    # explicitly in its body, so the FIELD_LINE precedence catches it
    # first; these patterns are belt-and-suspenders fallback for the case
    # where a body is reformatted upstream and loses the field line.
    re.compile(r"^\s*reason:\s*stall\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*reason:\s*probe_unreachable\b", re.IGNORECASE | re.MULTILINE),
]

# Field-line regex: matches a leading "failure_class: <value>" line
# (allowing surrounding whitespace and case-insensitive "infra"/"code").
FIELD_LINE = re.compile(
    r"^\s*failure_class\s*:\s*(infra|code)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# A CUDA OOM is normally transient infra (leaked process, fragmentation —
# respawn fixes it). EXCEPT: when the torch OOM message lists 2+ sibling
# "Process NNN has X GiB memory in use" entries on the failing device
# during a parallel fan-out, the train cells were CO-LOCATED on one
# physical GPU — a deterministic GPU-pinning bug in the launch path
# (e.g. a per-process `--gpu` pin that is dead code), so respawning on
# verified-clean GPUs hits the identical OOM. Route to `code` so the
# implementer fixes the pinning. A SINGLE sibling entry stays `infra`
# (one leaked process from a prior run — kill + respawn is the right
# move). Surfaced by task #557 (2026-06-10): attempt 1 was misdiagnosed
# as leaked-process infra and attempt 2 OOMed identically.
CUDA_OOM = re.compile(r"CUDA out of memory", re.IGNORECASE)
OOM_SIBLING_PROCESS = re.compile(
    r"Process \d+ has [\d.]+ [KMG]iB memory in use",
    re.IGNORECASE,
)


def _is_colocation_oom(body: str) -> bool:
    """True when a CUDA OOM lists >= 2 sibling-process memory entries."""
    if not CUDA_OOM.search(body):
        return False
    return len(OOM_SIBLING_PROCESS.findall(body)) >= 2


# torch's DataLoader wraps a worker-side exception with a header like:
#   RuntimeError: Caught RuntimeError in DataLoader worker process 0.
# followed by an "Original Traceback (most recent call last):" block whose
# frames belong to whatever raised inside the worker. The outer frames are
# always under torch/ (worker.py, _utils/, etc.), so the generic
# library-traceback infra pattern would route us-code raises to `infra`
# unless we look INSIDE the wrapped block. See the workflow-fix-on-bug
# candidate emitted by /issue 480.
DATALOADER_WRAP = re.compile(
    r"Caught\s+\w+\s+in\s+DataLoader worker",
    re.IGNORECASE,
)
ORIGINAL_TB_SPLIT = re.compile(r"Original Traceback", re.IGNORECASE)

# Our-code frame regex: matches a Python traceback "File ..." line whose
# path is under our source/scripts trees. Used to detect that the deepest
# frame inside a wrapped Original Traceback is our code (not a library).
OUR_CODE_FRAME = re.compile(
    r'File\s+"[^"]*/(?:src/explore_persona_space|scripts)/',
    re.IGNORECASE,
)


def classify_failure(body: str, stall_reason: str | None = None) -> FailureClass:
    """Return ``"infra"`` or ``"code"`` for an `epm:failure` body.

    ``stall_reason`` is the optional machine-readable cause forwarded from
    the poll JSON line on a ``status=stalled`` tick (SKILL Step 6d.2).

    Routing precedence:
    1. Explicit ``failure_class:`` field on the first non-blank line of
       the body (or any leading metadata block) wins.
    2. A known ``stall_reason`` in ``STALL_REASON_INFRA`` routes
       ``"infra"`` directly (e.g. ``"vllm_worker_dead_zombie_gpu"``): a
       silent hang has no traceback / OOM line for the regex fallback to
       match, so without this it would misroute to the conservative
       ``code`` default. See task #664.
    3. Co-located parallel-cell OOM: a ``CUDA out of memory`` body
       listing 2+ sibling ``Process NNN has X GiB memory in use``
       entries means parallel fan-out cells shared one physical GPU — a
       deterministic GPU-pinning bug, NOT transient infra — return
       ``"code"``. A single sibling entry stays on the infra path (one
       leaked process; kill + respawn fixes it). See task #557.
    4. If the body shows a torch DataLoader worker wrap (``Caught <Error>
       in DataLoader worker``), classify on the WRAPPED Original
       Traceback block, not the outer torch frames: when the wrapped
       block contains an our-code frame (``src/explore_persona_space/``
       or ``scripts/``), return ``"code"``; otherwise run the normal
       infra-pattern scan against the WRAPPED text only. The outer torch
       frames are always library code (worker.py, _utils/, ...) and
       routing on them would misclassify an our-code raise as ``infra``.
    5. Otherwise, scan the body against the infra log-pattern list. Any
       match → ``"infra"``.
    6. Otherwise, default to ``"code"`` (conservative — the implementer
       round catches more than the experimenter respawn round).
    """
    field = FIELD_LINE.search(body)
    if field is not None:
        return field.group(1).lower()  # type: ignore[return-value]

    if stall_reason is not None and stall_reason in STALL_REASON_INFRA:
        return "infra"

    if _is_colocation_oom(body):
        return "code"

    if DATALOADER_WRAP.search(body):
        # Split on the "Original Traceback" header and classify on the
        # WRAPPED block. Fall back to the post-split tail if there is no
        # explicit header (some torch versions wrap without it).
        parts = ORIGINAL_TB_SPLIT.split(body, maxsplit=1)
        wrapped = parts[1] if len(parts) == 2 else body
        if OUR_CODE_FRAME.search(wrapped):
            return "code"
        for rx in INFRA_PATTERNS:
            if rx.search(wrapped):
                return "infra"
        return "code"

    for rx in INFRA_PATTERNS:
        if rx.search(body):
            return "infra"
    return "code"


# Cap the amount of log we feed to the regex scanner. The patterns are
# anchored only by content (not start-of-string), so unbounded log files
# would just slow us down without changing the verdict.
_LOG_TAIL_BYTES = 200 * 1024  # 200 KB is well past 200 lines for any sane log


def _load_log_tail(path: Path) -> str:
    """Read the last ~200 KB of a log file. Returns "" if path is missing."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        # Surface the real error — the SKILL Step 7 wraps this in a respawn
        # cap so a missing log shouldn't silently default to `code`.
        sys.stderr.write(f"failure_classifier: cannot stat {path}: {exc}\n")
        return ""
    with path.open("rb") as fh:
        if size > _LOG_TAIL_BYTES:
            fh.seek(size - _LOG_TAIL_BYTES)
        return fh.read().decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: print the failure_class verdict, exit 0 on success."""
    parser = argparse.ArgumentParser(
        description="Classify an epm:failure body as `infra` or `code`. "
        "See scripts/failure_classifier.py module docstring for the rules.",
    )
    parser.add_argument(
        "--body",
        required=True,
        help="failure body text. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="optional path to a log file; its tail is concatenated with --body "
        "before classification, so library-traceback infra patterns can match.",
    )
    parser.add_argument(
        "--stall-reason",
        default=None,
        help="optional machine-readable stall cause from the poll JSON line "
        "(`stall_reason`); a known reason routes `infra` directly. "
        "See STALL_REASON_INFRA in this module.",
    )
    args = parser.parse_args(argv)

    body = sys.stdin.read() if args.body == "-" else args.body

    if args.log:
        body = body + "\n" + _load_log_tail(Path(args.log))

    print(classify_failure(body, stall_reason=args.stall_reason))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
