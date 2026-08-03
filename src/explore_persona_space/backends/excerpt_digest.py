"""Shared #1556 trigger-dense structural-digest helpers (#1574).

ONE implementation consumed by ``scripts/poll_pipeline.py`` (the RunPod-lane
poller, via assignment aliases + a thin ``_issue_trigger_dense`` wrapper) and
the GCP / SLURM lane monitors (``backends/gcp.py`` /
``backends/slurm_monitor.py``), which build their OWN ``log_tail_excerpt``
strings and were the named residual raw channels pre-#1574.

``src/`` never imports ``scripts/``: the CUDA-IMA regex here is a MIRROR of
``backend_poll.CUDA_IMA_SIGNATURE`` (``backend_poll`` is scripts-side and
imports ``poll_pipeline``, so a src import of it would be both barred and
circular). The mirror is pinned byte-in-sync by
``tests/test_poll_pipeline_digest.py::test_digest_cuda_ima_flag_matches_backend_poll_signature``
(through the poll_pipeline alias) and
``tests/test_backend_excerpt_digest.py::test_cuda_ima_mirror_matches_backend_poll``
(directly against this module).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# The task tag that declares a workload trigger-dense (#1556). Set at dispatch
# per .claude/rules/trigger-dense-review.md's recognition heuristic via
# `task.py add-tag <N> trigger-dense`; read fresh every tick (no caching), so a
# mid-run add-tag takes effect on the next poll.
TRIGGER_DENSE_TAG = "trigger-dense"

# Mirror of ``backend_poll.CUDA_IMA_SIGNATURE`` — see the module docstring for
# why the real one cannot be imported from src/. The digest's structural flag
# fires on the REAL signature family (including the engine-dead alternatives),
# never a bare substring (#1556).
CUDA_IMA_SIGNATURE_MIRROR = re.compile(
    r"CUDA error:\s*an illegal memory access was encountered"
    r"|illegal memory access was encountered"
    r"|EngineDeadError"
    r"|Engine core proc \S+ died unexpectedly",
    re.IGNORECASE,
)

# Case-insensitive per-line substring counts for the trigger-dense digest — the
# trigger-dense-review.md item-1 pattern set ('error|traceback|killed|OOM')
# plus the CUDA-IMA class as a human-facing count. The machine contract is the
# ``CUDA_IMA_SIGNATURE_MIRROR`` structural flag in ``digest_tail_excerpt`` —
# the ``cuda_ima`` COUNT and the flag may legitimately disagree (count=0 with
# the flag present on an engine-dead-only tail); that is correct, not a bug.
DIGEST_PATTERNS: tuple[tuple[str, str], ...] = (
    ("error", "error"),
    ("traceback", "traceback"),
    ("killed", "killed"),
    ("oom", "oom"),
    ("cuda_ima", "illegal memory access"),
)


def issue_trigger_dense(
    issue: int,
    *,
    get_task_fn: Callable[[int], dict[str, Any]] | None = None,
    log: logging.Logger = logger,
) -> bool:
    """True when the issue's workload is declared trigger-dense (task tag
    ``trigger-dense`` — set at dispatch per trigger-dense-review.md's
    recognition heuristic: the run trains/evals on guard/security surfaces or
    gated-content corpora, knowable before any read), or when the task EXISTS
    but its state is unreadable (fail SAFE toward digest, loudly).

    A missing task (``FileNotFoundError`` — ad-hoc/synthetic polls; includes
    ``StaleTaskPathError``, its subclass) has no declaration surface at all
    -> False (raw excerpt, today's behavior). Fresh read every tick, no
    caching: ``task.py add-tag <N> trigger-dense`` mid-run takes effect on the
    next tick — a live mitigation lever during an unfolding incident. The
    catch taxonomy mirrors the audited pod_lifecycle pre-terminate set
    (FileNotFoundError = not in registry/on disk; RuntimeError = branch-guard;
    ValueError = malformed frontmatter/registry; OSError = unreadable file);
    anything else propagates (fail-fast).

    Injection seams (#1574): ``get_task_fn`` defaults to a LAZY
    ``task_workflow.get_task`` import inside the function — keeps this module
    import-light and cycle-proof, mirroring the lanes' existing lazy
    task_workflow imports (``slurm_monitor.build_poll_result`` /
    ``gcp._guest_persist_breadcrumb``); ``log`` lets ``poll_pipeline`` keep
    emitting on its own "poll_pipeline" logger (its caplog-pinned
    INFO/WARNING assertions depend on it).
    """
    if get_task_fn is None:
        from explore_persona_space.task_workflow import get_task as get_task_fn
    try:
        task = get_task_fn(issue)
    except FileNotFoundError:
        log.info("trigger-dense check: task #%s not found (ad-hoc poll); raw excerpt", issue)
        return False
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning(
            "trigger-dense tag read FAILED for issue %s (%s: %s); failing SAFE "
            "toward digest-on-unknown (raw tail stays readable at the log path)",
            issue,
            type(exc).__name__,
            exc,
        )
        return True
    tags = (task.get("frontmatter") or {}).get("tags") or []
    return TRIGGER_DENSE_TAG in tags


def digest_tail_excerpt(
    wide_tail: str,
    *,
    status: str,
    current_phase: str,
    pid_alive: bool,
    source: str,
    log_path: str,
    mtime_sec_ago: int,
) -> str:
    """Bounded structural digest replacing the raw excerpt on trigger-dense runs.

    Pure + deterministic: pattern counts over the wide tail, the poller's own
    verdict fields (status/phase/pid liveness — the exit-state information
    this seam actually has; the workload's numeric rc lands in the sentinel
    channel and in the log the script-side ``failure_classifier.py --log``
    reads), the winning log source + path, and tail size/staleness. NO raw
    log line content is inlined — inherently secret-safe (nothing to scrub).
    The CUDA-IMA structural flag preserves the one content-coupled machine
    contract (``backend_poll._prior_failure_marker_is_cuda_ima`` regex over
    ``epm:failure`` notes — the #775 cross-pod fallback) on digested notes.
    """
    lines = wide_tail.splitlines()
    low = [ln.lower() for ln in lines]
    counts = {name: sum(1 for ln in low if pat in ln) for name, pat in DIGEST_PATTERNS}
    counts_s = " ".join(f"{k}={v}" for k, v in counts.items())
    out = (
        f"[trigger-dense digest] status={status} phase={current_phase} "
        f"pid_alive={pid_alive} source={source} log={log_path} "
        f"tail_lines={len(lines)} tail_bytes={len(wide_tail.encode('utf-8', 'replace'))} "
        f"log_mtime_sec_ago={mtime_sec_ago} pattern_counts({counts_s}); "
        f"raw tail NOT inlined (trigger-dense workload; classify via "
        f"scripts/failure_classifier.py --log <path>)"
    )
    if CUDA_IMA_SIGNATURE_MIRROR.search(wide_tail):
        # Flag phrase chosen to MATCH signature alternative 2 so the #775
        # cross-pod marker-note fallback keeps firing on digested notes.
        out += " cuda_ima_flag: illegal memory access was encountered (structural flag)"
    return out
