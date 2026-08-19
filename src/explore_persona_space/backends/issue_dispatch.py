"""``/issue`` dispatch helper — the production wiring for ``route()``.

The slice-5 router (:mod:`backends.router`) is fully testable in
isolation; this module is the THIN bridge the ``/issue`` skill calls to
get a :class:`RunHandle` for a real task. It:

1. **Builds a :class:`RunSpec`** from the task's frontmatter + plan. Legacy
   frontmatter values are normalized BEFORE the spec is built — in
   particular ``backend: cluster`` (the legacy selector alias) is mapped
   to ``backend: nibi`` because the slice-5 router REJECTS the bare
   ``"cluster"`` literal (see ``router._VALID_BACKEND_VALUES``).
2. **Builds production injected deps for** :func:`route` —
   ``free_backends`` (SLURM-backed Nibi + Fir lanes), ``gcp_backend``
   (the credit-backed escalation target), ``marker_poster``
   (:func:`backends.slurm.post_marker_via_task_py`), ``is_started`` /
   ``is_live_after_cancel`` (SLURM-aware probes via
   :mod:`backends.slurm_monitor`), ``reconnect_fn`` (per-backend
   reconnect), and a Mila-socket-alive stub (always ``False`` until
   slice 7).
3. **Calls :func:`route`** and TRANSLATES the terminal exceptions the
   router raises into the marker / status-mutation contract the ``/issue``
   skill consumes:

   * :class:`NoComputeAvailableError` → ``epm:failure v1`` with
     ``failure_class: infra``, status -> ``blocked``.
   * :class:`WorkloadSurfacedError` → ``epm:failure v1`` with
     ``failure_class: code``, status -> ``blocked``.
   * :class:`GcpAttemptCapExceededError` → ``epm:failure v1`` with
     ``failure_class: infra``, status -> ``blocked``.
   * :class:`ManualAttentionRequiredError` → ``epm:failure v1`` with
     ``failure_class: infra``, status -> ``blocked``; the failure note
     carries the orphaned job_id so the operator can confirm + scancel.

4. **Persists the :class:`RunHandle`** to a per-issue sidecar JSON file
   (``<main-checkout>/.claude/cache/issue-<N>-handle.json``, resolved
   cwd-INDEPENDENTLY — see :func:`default_handle_sidecar_path`, incident
   #612) so the orchestrator's bg-Bash poller
   (``scripts/backend_poll.py``) can recover the handle without
   re-dispatching the router. The handle is a small, serializable
   dataclass; round-tripping through JSON preserves every field the
   poller needs.

**Bg-Bash poll contract preservation.** This module does NOT move poll
in-process. The bg-Bash poller is still a separate process the
orchestrator launches via ``Bash(run_in_background=True)``; it imports
the right ``ComputeBackend`` subclass, deserializes the handle from the
sidecar JSON, and prints the SAME ``PollResult`` JSON the orchestrator
already parses (see :mod:`backends.base.PollResult`). Notification-on-bg-
Bash-exit is the orchestrator's wakeup signal — moving poll in-process
would break the harness re-invocation model
(``CLAUDE.md`` § "Orchestrator vs subagent re-invocation").

The helper is dependency-injectable: tests pass mock backends + a
list-appender marker poster + an in-memory handle cache to exercise
``dispatch_for_issue`` without RunPod / SLURM / GCP being live.

See also:

* :func:`backends.router.route` — the underlying decision engine.
* ``.claude/skills/issue/SKILL.md`` Step 6b / 6d / 8 — the orchestrator
  steps this module is invoked from.
* :mod:`backends.slurm_monitor` — the SLURM-aware ``is_started`` /
  ``is_live_after_cancel`` / reconnect-by-name probes the production
  wiring uses.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    RunHandle,
    RunSpec,
    validate_lane_suffix,
)
from explore_persona_space.backends.router import (
    ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD,
    ROUTE_REASON_CPU_FALLBACK_INFEASIBLE,
    ROUTE_REASON_GCP_DISABLED,
    ROUTE_REASON_GPU_RAM_BELOW_MIN_RAM_GB,
    ROUTE_REASON_RECONNECT,
    ROUTE_REASON_RUNPOD_STOPPED_POD_COLLISION,
    BackendPrepareError,
    CpuExhaustedNoRunpodLaneError,
    CpuFallbackInfeasibleError,
    GcpAttemptCapExceededError,
    GcpDisabledError,
    GpuRamBelowMinRamGbError,
    LeaseStore,
    ManualAttentionRequiredError,
    NoComputeAvailableError,
    RouterConfig,
    RouteResult,
    RunPodStoppedPodCollisionError,
    WorkloadSurfacedError,
    route,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


#: The set of literal strings the ``/issue`` task frontmatter accepts
#: under ``backend:``. The router's :data:`router._VALID_BACKEND_VALUES`
#: is the canonical set ``route()`` accepts; this set is the SUPERSET
#: that maps legacy aliases (``"cluster"``) into the router's set.
#: Empty / absent frontmatter routes to ``"auto"``.
_LEGACY_TO_ROUTER_BACKEND: dict[str, BackendKind] = {
    # The selector's legacy generic SLURM alias. The router rejects it
    # (only the per-cluster names are routable lanes); map to the v1
    # default cluster.
    "cluster": "nibi",
}

#: #2161: env knob bounding how long ONE ``dispatch_for_issue`` process may
#: spend inside free-lane queue parks before the router persists the ladder
#: position and raises :class:`FreeLaneStillWaitingError` (surfaced by
#: ``dispatch_issue.py launch`` as still-waiting exit 75 — re-run the SAME
#: command; the run resumes from the durable lease state, no double-submit).
#: Mirrors the FELLOWS_QUEUE_WAIT_ENV convention: read at CALL time, never
#: import time. Unset / malformed → the 420 s default; ``0`` or negative →
#: None (unlimited — the pre-#2161 in-process park semantics, byte-identical).
PARK_PROCESS_BUDGET_ENV = "EPS_LAUNCH_PARK_PROCESS_BUDGET_SECONDS"
PARK_PROCESS_BUDGET_DEFAULT_SECONDS = 420


def _env_park_process_budget() -> int | None:
    """Resolve the per-process free-lane park budget from the env (#2161).

    Returns the budget in seconds, or ``None`` for unlimited (legacy
    semantics). Read at call time so tests / operators can flip the knob
    without re-importing the module.
    """
    raw = os.environ.get(PARK_PROCESS_BUDGET_ENV)
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return PARK_PROCESS_BUDGET_DEFAULT_SECONDS
    return val if val > 0 else None


@functools.lru_cache(maxsize=1)
def _main_checkout_root() -> Path:
    """Absolute path of the MAIN repo checkout, resolved cwd-independently.

    Runs ``git rev-parse --path-format=absolute --git-common-dir`` from
    the directory containing THIS module (NOT ``os.getcwd()``), so the
    same root comes back whether the caller's cwd is the repo root, an
    issue worktree, or anywhere else. From a linked worktree the common
    dir is ``<main>/.git``, so its parent is the main checkout. Mirrors
    the ``task_workflow`` resolver's location step WITHOUT its branch
    guard / managed-worktree routing (a cache sidecar needs neither, and
    ``task_workflow.repo_root()`` carries a ``reset --hard`` side effect
    when the primary checkout is parked off-``main``).

    Fails LOUD (``RuntimeError``) when git is missing or the module is
    not inside a git checkout — a silent cwd fallback would re-introduce
    the split-brain sidecar bug this resolver closes (incident #612).
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"}
    }
    module_dir = Path(__file__).resolve().parent
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(module_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot resolve the main checkout root from {module_dir} "
            f"(`git rev-parse --git-common-dir` failed: {exc}); the handle "
            f"sidecar path must be cwd-independent (#612) — refusing a cwd fallback"
        ) from exc
    common_dir = Path(proc.stdout.strip())
    if common_dir.name != ".git" or not common_dir.is_dir():
        raise RuntimeError(
            f"git common-dir {common_dir} does not look like a main-checkout .git "
            f"directory; refusing to compose the handle sidecar path"
        )
    return common_dir.parent


def default_handle_sidecar_path(issue: int, lane_suffix: str | None = None) -> Path:
    """Canonical sidecar JSON path for the per-issue serialized RunHandle.

    ABSOLUTE, anchored at ``<main-checkout>/.claude/cache/`` so the
    launch (often dispatched with cwd = an issue worktree) and every
    later poll / finalize tick (usually cwd = the repo root) converge on
    the SAME file. The pre-2026-06-12 cwd-relative form split the
    contract across checkouts: a worktree-cwd launch wrote
    ``<worktree>/.claude/cache/issue-<N>-handle.json`` while a repo-root
    poll probed ``<root>/.claude/cache/...``, yielding a false-positive
    ``status=dead / reason=missing_handle_sidecar`` on a healthy run
    (incident #612, 2026-06-12). Read-side callers that may encounter a
    legacy worktree-local sidecar should resolve via
    :func:`resolve_handle_sidecar_path` (probes the legacy cwd-relative
    location too).

    ``lane_suffix`` (#934): a validated per-lane suffix yields
    ``issue-<N>-<suffix>-handle.json`` so two concurrent lanes for one
    issue keep independent handles (and, via the ``-handle.json`` stem
    convention every sidecar-sibling consumer strips, independent
    per-lane sibling files). ``None`` / empty keeps the unsuffixed path
    byte-identical.
    """
    stem = f"issue-{int(issue)}" + (f"-{validate_lane_suffix(lane_suffix)}" if lane_suffix else "")
    return _main_checkout_root() / ".claude" / "cache" / f"{stem}-handle.json"


def resolve_handle_sidecar_path(
    issue: int,
    explicit: Path | str | None = None,
    lane_suffix: str | None = None,
) -> tuple[Path, list[Path]]:
    """Read-side sidecar resolution: explicit > canonical > legacy cwd-relative.

    Returns ``(resolved, probed)`` where ``probed`` lists every path
    checked (absolute where resolvable) so callers can log exactly which
    locations were searched on a miss. The legacy probe covers sidecars
    written by the pre-#612 cwd-relative composer (a launch dispatched
    from an issue worktree landed the file in the WORKTREE's
    ``.claude/cache/``); it fires only when the canonical path is absent
    and only for the default resolution — an explicit ``--handle-file``
    is honored verbatim, never second-guessed. ``lane_suffix`` (#934)
    resolves the per-lane suffixed sidecar (same stem in the legacy
    cwd-relative probe); ignored when ``explicit`` is given.
    """
    if explicit is not None:
        p = Path(explicit)
        return p, [p]
    primary = default_handle_sidecar_path(issue, lane_suffix=lane_suffix)
    probed = [primary]
    legacy = Path.cwd() / ".claude" / "cache" / primary.name
    if not primary.exists() and legacy.resolve() != primary.resolve():
        probed.append(legacy)
        if legacy.exists():
            return legacy, probed
    return primary, probed


# ---------------------------------------------------------------------------
# Dispatch outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchOutcome:
    """What :func:`dispatch_for_issue` returns to the orchestrator.

    Fields:

    * ``result`` — the :class:`RouteResult` (carries the backend
      instance, handle, chosen kind, attempt ladder, marker breadcrumb).
    * ``handle_sidecar_path`` — path to the serialized handle JSON the
      bg-Bash poller will read. ``None`` when the caller asked to
      skip the write, OR when the write failed (then
      ``sidecar_write_error`` says why).
    * ``sidecar_write_error`` — non-``None`` when the authoritative
      sidecar write raised ``OSError``. The launch already succeeded
      (live VM / job), so the dispatch CLI prints the handle JSON line
      + this error LOUDLY instead of converting a recoverable
      persistence failure into an unclassified rc=4 crash.
    """

    result: RouteResult
    handle_sidecar_path: Path | None
    sidecar_write_error: str | None = None


# ---------------------------------------------------------------------------
# RunSpec construction (frontmatter normalization lives here)
# ---------------------------------------------------------------------------


def normalize_backend_value(raw: Any) -> BackendKind:
    """Normalize a frontmatter ``backend:`` value to a routable BackendKind.

    Accepts the empty / absent value (route as ``"auto"``), the legacy
    ``"cluster"`` alias (mapped to ``"nibi"`` because the router
    rejects the bare literal), and every value the router itself
    accepts. Raises :class:`ValueError` on a typo so a malformed
    frontmatter surfaces at dispatch time rather than silently
    auto-routing.
    """
    if raw is None:
        return "auto"
    if not isinstance(raw, str):
        raise ValueError(f"backend frontmatter must be a string, got {type(raw).__name__}: {raw!r}")
    val = raw.strip().lower()
    if val == "":
        return "auto"
    if val in _LEGACY_TO_ROUTER_BACKEND:
        return _LEGACY_TO_ROUTER_BACKEND[val]
    # ``route()`` validates the value at call time; we forward verbatim.
    # The narrow router-side set is the source of truth.
    if val in {"runpod", "nibi", "fir", "gcp", "mila", "fellows", "auto"}:
        return val  # type: ignore[return-value]
    raise ValueError(
        f"unknown backend frontmatter value: {raw!r}. Expected one of: "
        "runpod, cluster, nibi, fir, gcp, mila, fellows, auto, or empty (auto)."
    )


# ---------------------------------------------------------------------------
# Workload-cmd lane-env lint (#1329, incident #825)
# ---------------------------------------------------------------------------


#: SLURM lane names (per-cluster backend values that all share one renderer /
#: one export contract). The legacy ``cluster`` alias normalizes to ``nibi``
#: (see :data:`_LEGACY_TO_ROUTER_BACKEND`), i.e. into this set. ``fellows``
#: (#1609) shares the same renderer, hence the same export contract (its
#: extra_exports — HF_HOME / NCCL_NVLS_ENABLE / UV_PYTHON* / SCRATCH — are
#: outside the lint universe by the same noise-var rule as HF_XET_*).
_SLURM_LANES: tuple[str, ...] = ("nibi", "fir", "mila", "fellows")

#: Env vars each lane EXPORTS into the shell that executes a user-supplied
#: ``--workload-cmd`` string, verified against the renderers (line anchors as
#: of #1329; forward-parity-pinned by
#: ``tests/test_workload_cmd_env_lint.py``):
#:
#: * ``gcp`` — ``gcp.py render_startup_script`` (fn at :1101): WORKLOAD_ROOT
#:   :2386, REPO_ROOT :1336 (workload-cmd branch, #641), EPS_ISSUE :2384,
#:   EPS_ATTEMPT_ID :2385, EPS_SENTINEL_PATH :2387,
#:   EPS_DELIVERABLES_OK_PATH :2393, EPS_HF_DATA_REPO :2403,
#:   EPS_LOG_PATH :2440, EPS_DONE_GRACE :2408, EPS_DONE_KEEPALIVE_PATH :2409,
#:   EPS_SCRATCH_DIR :1367, WANDB_PROJECT :1356, PYTHONPATH :1348,
#:   HOME :2381, PATH :2507.
#: * ``runpod`` — ``runpod.py`` launcher (:489-512): REPO_ROOT :508,
#:   WANDB_PROJECT :509, PATH :505; HOME is ambient (inherited from the
#:   SSH/root shell, not a literal export line).
#: * ``slurm`` — ``slurm.py`` custom stage + prelude: EPS_ISSUE :1527,
#:   EPS_ATTEMPT_ID :1528, WANDB_PROJECT :1539, PYTHONPATH :1503; HOME is
#:   ambient. NOTE (#1329 fact-check): SCRATCH_JOB_DIR is a PLAIN ASSIGNMENT
#:   (slurm.py:1266 — no ``export`` anywhere in the file) and the custom cmd
#:   runs in a CHILD bash (:1577) that inherits only exported vars, so it is
#:   deliberately EXCLUDED from the slurm set.
#:
#: Deliberately NO inverse parity in this map itself: renderers export noise
#: vars (``HF_XET_*``, ``HF_HOME``, ``GIT_TERMINAL_PROMPT``, ...) that must
#: not join the lint universe — a bare reference to those is a known,
#: accepted false negative in v1 (plan #1329 §9 fact-check record).
LANE_WORKLOAD_ENV_EXPORTS: dict[str, frozenset[str]] = {
    "gcp": frozenset(
        {
            "WORKLOAD_ROOT",
            "REPO_ROOT",
            "EPS_ISSUE",
            "EPS_ATTEMPT_ID",
            "EPS_SENTINEL_PATH",
            "EPS_DELIVERABLES_OK_PATH",
            "EPS_HF_DATA_REPO",
            "EPS_LOG_PATH",
            "EPS_DONE_GRACE",
            "EPS_DONE_KEEPALIVE_PATH",
            "EPS_SCRATCH_DIR",
            "WANDB_PROJECT",
            "PYTHONPATH",
            "HOME",
            "PATH",
        }
    ),
    "runpod": frozenset(
        {
            "REPO_ROOT",
            "WANDB_PROJECT",
            "HOME",
            "PATH",
        }
    ),
    "slurm": frozenset(
        {
            "EPS_ISSUE",
            "EPS_ATTEMPT_ID",
            "WANDB_PROJECT",
            "PYTHONPATH",
            "HOME",
            "PATH",
        }
    ),
}

#: The candidate universe the lint scans for: union of the per-lane sets. A
#: ``$FOO`` outside this universe is NEVER flagged (false-positive
#: discipline) — the lint only knows the lane semantics of these vars.
_LANE_ENV_UNIVERSE: frozenset[str] = frozenset().union(*LANE_WORKLOAD_ENV_EXPORTS.values())

#: Strip single-quoted segments before scanning: POSIX single quotes contain
#: no escapes, so ``'...'`` is a literal and ``$V`` inside it never expands.
#: The ``'\''`` idiom (an apostrophe inside a double-quoted string with no
#: closing partner, etc.) degrades toward *scanning more text* in the common
#: case — i.e. errs conservative (flags) — but NOT universally: an unpaired
#: apostrophe can pair with a LATER real single-quote opener and strip a
#: genuinely-bare ``$V`` between them (a false negative). Pinned by the
#: quote-strip boundary test in ``tests/test_workload_cmd_env_lint.py``.
_SINGLE_QUOTED_SEGMENT_RE = re.compile(r"'[^']*'")


def _bare_reference_re(var: str) -> re.Pattern[str]:
    """Regex matching a BARE shell reference to ``var`` (aborts under set -u).

    Matches ``$VAR`` and ``${VAR}`` — the two forms ``set -u`` aborts on when
    the var is unbound. Structurally does NOT match:

    * defaulted/alternate expansions ``${VAR:-w}`` / ``${VAR-w}`` /
      ``${VAR:+w}`` / ``${VAR+w}`` / ``${VAR:=w}`` (any operator breaks the
      exact ``${VAR}`` brace match; all are set-u-safe by POSIX),
    * assignments ``VAR=...`` (no ``$``),
    * escaped ``\\$VAR`` (negative lookbehind),
    * longer names ``$VARX`` (word-boundary lookahead).

    Deliberate v1 FALSE NEGATIVES (these DO abort under ``set -u`` on an
    unbound var but are not matched): ``${VAR%pat}``, ``${VAR#pat}``,
    ``${VAR:0:3}`` and other non-defaulting parameter expansions — rare in
    workload-cmd strings; widen if one ever bites (plan #1329 §3.1).
    """
    return re.compile(rf"(?<!\\)\$(?:{var}(?![A-Za-z0-9_])|\{{{var}\}})")


@dataclass(frozen=True)
class WorkloadCmdEnvLint:
    """Result of :func:`lint_workload_cmd_lane_env`.

    ``flagged`` maps each bare-referenced var to the sorted tuple of
    REACHABLE lanes whose export set lacks it. ``certain`` is the subset of
    flagged vars the PINNED lane itself lacks while provably executing the
    command under ``set -u`` (a guaranteed abort — the exit-2 refusal arm).
    ``reachable_lanes`` records the lanes the launch could land on.
    """

    flagged: dict[str, tuple[str, ...]]
    certain: tuple[str, ...]
    reachable_lanes: tuple[str, ...]


def _reachable_lanes_for_backend(backend: str) -> tuple[str, ...]:
    """Map a normalized backend value to the lane-export sets it can reach.

    ``runpod`` → itself only; ``gcp`` → gcp + runpod (the Part B workload
    failover re-runs the SAME cmd on RunPod —
    ``.claude/rules/compute-backend-failover.md``); a SLURM lane → slurm
    only; ``auto`` (absent/empty frontmatter) → all three (the
    ``DEFAULT_AUTO_LANE_ORDER`` chain + the RunPod terminal rung).
    """
    if backend == "runpod":
        return ("runpod",)
    if backend == "gcp":
        return ("gcp", "runpod")
    if backend in _SLURM_LANES:
        return ("slurm",)
    return ("gcp", "runpod", "slurm")


def lint_workload_cmd_lane_env(
    workload_cmd: str,
    *,
    backend_value: str | None,
    execute_workload: bool = False,
) -> WorkloadCmdEnvLint:
    """Lint a ``--workload-cmd`` string for bare lane-specific env-var refs.

    A var exported by only SOME lanes (canonical offender: ``$WORKLOAD_ROOT``,
    GCP-only) referenced BARE in the command aborts under ``set -u`` on any
    reachable lane that does not export it — the #825 Track-S incident: a
    GCP→RunPod failover re-ran ``REPO_ROOT="$WORKLOAD_ROOT" bash ...`` and
    the RunPod launcher (``set -uo pipefail``, runpod.py:504) died before
    the driver started.

    ``certain`` scoping (the provable-abort refusal arm): explicit
    ``runpod`` WITH ``execute_workload=True`` (the launcher executes the cmd
    under ``set -u``; a provision-only runpod launch downgrades to warn —
    the experimenter recomposes the pod-side launch), or an explicit SLURM
    lane (the literal ``bash -eu -o pipefail -c`` append, slurm.py:1577).
    Explicit ``gcp`` and ``auto`` are never ``certain`` (the GCP startup
    script's workload line is not verified ``set -u``; auto's landing lane
    is unknown at dispatch time). An unrecognized ``backend_value`` is
    treated as ``auto`` (flag, never certain) — the CLI's own
    ``normalize_backend_value`` rejects typos before this lint runs.

    Returns an empty result for an empty/whitespace command (``--hydra``
    launches are a no-op here).
    """
    if not (workload_cmd or "").strip():
        return WorkloadCmdEnvLint(flagged={}, certain=(), reachable_lanes=())
    try:
        backend = normalize_backend_value(backend_value)
    except ValueError:
        backend = "auto"
    reachable = _reachable_lanes_for_backend(backend)
    stripped = _SINGLE_QUOTED_SEGMENT_RE.sub(" ", workload_cmd)
    flagged: dict[str, tuple[str, ...]] = {}
    for var in sorted(_LANE_ENV_UNIVERSE):
        missing = tuple(
            lane for lane in sorted(reachable) if var not in LANE_WORKLOAD_ENV_EXPORTS[lane]
        )
        if not missing:
            continue
        if _bare_reference_re(var).search(stripped):
            flagged[var] = missing
    certain_lane: str | None = None
    if backend == "runpod" and execute_workload:
        certain_lane = "runpod"
    elif backend in _SLURM_LANES:
        certain_lane = "slurm"
    certain: tuple[str, ...] = ()
    if certain_lane is not None:
        certain = tuple(
            var for var in sorted(flagged) if var not in LANE_WORKLOAD_ENV_EXPORTS[certain_lane]
        )
    return WorkloadCmdEnvLint(flagged=flagged, certain=certain, reachable_lanes=reachable)


#: Sanctioned trailing sentinel append (experimenter.md § item 11):
#: ``<workload-cmd> && [uv run ]python -c "...write_completion_sentinel..."``.
#: Leftmost ``&& [uv run ]python -c`` whose remainder mentions
#: ``write_completion_sentinel``; anchored to end-of-string via DOTALL ``.+$``
#: so only a TRAILING append (plus anything after the first such join) is
#: stripped before the inline-interpreter scan.
_SENTINEL_APPEND_RE = re.compile(r"&&\s*(?:uv\s+run\s+)?python3?\s+-c\s+(?P<rest>.+)$", re.DOTALL)
#: Inline interpreter as the PRIMARY command: optional ``VAR=val`` env
#: prefixes, then ``[uv run ]python[3] -c``.
_INLINE_C_BODY_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:uv\s+run\s+)?python3?\s+-c(?=\s)"
)
#: Stdin-heredoc python anywhere: ``python - <<EOF``, ``python - <<'EOF'``,
#: ``python <<EOF``, with/without ``uv run``, optional ``<<-``.
_STDIN_HEREDOC_RE = re.compile(r"(?:uv\s+run\s+)?python3?\s+-?\s*<<-?\s*['\"]?\w+")


@dataclass(frozen=True)
class WorkloadCmdInlineLint:
    """Result of :func:`lint_workload_cmd_inline_interpreter` (#1576).

    ``shape`` names the detected anti-pattern (``"inline_c"`` |
    ``"stdin_heredoc"``) or ``None`` when unflagged.
    ``sentinel_append_stripped`` records whether the sanctioned trailing
    ``write_completion_sentinel`` append was removed before the scan.
    """

    flagged: bool
    shape: str | None
    sentinel_append_stripped: bool


def strip_sentinel_append(workload_cmd: str) -> tuple[str, bool]:
    """Strip the experimenter.md-sanctioned trailing sentinel append.

    Returns ``(body, stripped)``: the command with the leftmost
    ``&& [uv run ]python -c`` join removed when its remainder mentions
    ``write_completion_sentinel`` (the ONE sanctioned inline ``-c`` suffix),
    else the command unchanged.
    """
    m = _SENTINEL_APPEND_RE.search(workload_cmd)
    if m is not None and "write_completion_sentinel" in m.group("rest"):
        return workload_cmd[: m.start()], True
    return workload_cmd, False


def lint_workload_cmd_inline_interpreter(workload_cmd: str) -> WorkloadCmdInlineLint:
    """WARN-class #1576 detection: inline interpreter one-liner as the workload BODY.

    Mechanizes the SKILL.md § Backend dispatch prose rule "Ad-hoc probe
    workloads are committed scripts invoked by path" (incident #1482: a
    placeholder-broken inline staging one-liner would have SyntaxError'd
    post-b0 and spuriously failed over to RunPod). Scans the RAW command —
    never the ``_SINGLE_QUOTED_SEGMENT_RE``-stripped text the lane-env lint
    uses — because an inline ``-c`` body is usually single-quoted, so the
    quote strip would erase exactly the evidence this arm needs. The
    sanctioned trailing ``write_completion_sentinel`` append is removed via
    :func:`strip_sentinel_append` BEFORE the scan (strip-then-scan: an inline
    body that ALSO ends with the sentinel append still flags).

    Scope: router-lane dispatches only — this lint runs inside
    ``dispatch_issue.py launch``; direct-SSH pod launches bypass it.

    Deliberate v1 FALSE NEGATIVES (named, mirroring the
    ``_bare_reference_re`` docstring discipline — widen if one bites):

    * a mid-chain non-sentinel ``&& python -c`` segment after a
      committed-script body;
    * ``bash -c '...'`` / ``sh -c '...'`` wrapping;
    * ``python -m module`` bodies and non-python interpreters;
    * the no-space ``python -c'x'`` shape (the ``-c`` lookahead requires
      whitespace);
    * sentinel-token smuggling: ``true && uv run python -c "<staging>;
      ...write_completion_sentinel(...)"`` post-strips to ``true`` and does
      not flag (the strip keys on the token alone).
    """
    body = (workload_cmd or "").strip()
    if not body:
        return WorkloadCmdInlineLint(flagged=False, shape=None, sentinel_append_stripped=False)
    body, stripped = strip_sentinel_append(body)
    if _INLINE_C_BODY_RE.search(body):
        return WorkloadCmdInlineLint(True, "inline_c", stripped)
    if _STDIN_HEREDOC_RE.search(body):
        return WorkloadCmdInlineLint(True, "stdin_heredoc", stripped)
    return WorkloadCmdInlineLint(False, None, stripped)


#: Case-insensitive persist-evidence substrings (#1800). A workload chain
#: (command + resolved driver-script text) carrying NONE of these has no
#: visible upload/persist WIRING for its outputs — the #1739 class. The
#: composition is a v1 heuristic pinned by tests
#: (``tests/test_workload_cmd_persist_lint.py``); incident-replay
#: calibration (plan #1800 §2): the pre-fix #1739 dispatcher
#: (``origin/issue-1739`` @ ``3bcc140bbd``) reads 0 hits → flags, the
#: post-fix tip reads 17 hits → clean.
PERSIST_EVIDENCE_TOKENS: tuple[str, ...] = (
    "upload",
    "push_to_hub",
    "hf_hub",
    "hfapi",
    "git push",
    "persist",
)


@dataclass(frozen=True)
class WorkloadCmdPersistLint:
    """Result of :func:`lint_workload_cmd_persist_evidence` (#1800).

    ``flagged`` — the resolved chain carries ZERO persist-evidence tokens.
    ``skipped`` — the lint could not run (empty command, or the driver
    script text was unavailable); never flagged when skipped.
    ``matched_tokens`` — the sorted evidence tokens found (empty when
    flagged or skipped).
    """

    flagged: bool
    skipped: bool
    matched_tokens: tuple[str, ...]


def lint_workload_cmd_persist_evidence(
    workload_cmd: str, script_text: str | None
) -> WorkloadCmdPersistLint:
    """WARN-class #1800 detection: no persist step anywhere in the workload chain.

    Mechanizes the dispatch-time backstop for incident #1739 (2026-07-28): a
    GCP ``--workload-cmd`` run completed every phase and approached
    grace-poweroff with ZERO artifacts on HF (all 7 expected prefixes MISS);
    #1779 fixed the PLAN-time layer, this lint is the dispatch-time sibling
    of the #1329/#1576 workload-cmd lint family. Scans the COMMAND plus the
    resolved driver-script text (``script_text``) case-insensitively for
    :data:`PERSIST_EVIDENCE_TOKENS`; zero hits on a resolved script →
    ``flagged=True``. ``script_text is None`` (unresolvable driver) →
    ``skipped=True``, never flagged — fail-soft by design, the caller logs
    ONE note.

    Deliberate v1 FALSE NEGATIVES (named, mirroring the
    :func:`lint_workload_cmd_inline_interpreter` docstring discipline —
    widen if one bites):

    * present-but-never-executed persist: a chain whose persist phase
      EXISTS in the script text but is skipped at runtime (a ``--dry-run``
      upload mode, a ``--phases`` subset invocation that omits the upload
      phase) reads as evidence — this lint checks persist WIRING, never a
      persist guarantee; Step 8 upload-verification stays the hard gate;
    * ambiguous tokens: a download-only ``hf_hub_download`` staging call
      (or a bare ``HfApi()`` listing) matches the ``hf_hub`` / ``hfapi``
      tokens and reads as evidence;
    * comments: a commented-out ``# upload later`` line counts as evidence
      (substring scan, no shell/python parsing).

    Known FALSE-POSITIVE residual (WARN-only by design, #1800 plan §5): a
    multi-script chain whose persist step lives in a SECOND sourced/invoked
    file the caller did not resolve flags spuriously — the escape is the
    descope path (scan the command string only) if this proves noisy.
    """
    if not (workload_cmd or "").strip():
        return WorkloadCmdPersistLint(flagged=False, skipped=True, matched_tokens=())
    if script_text is None:
        return WorkloadCmdPersistLint(flagged=False, skipped=True, matched_tokens=())
    haystack = f"{workload_cmd}\n{script_text}".lower()
    matched = tuple(sorted(tok for tok in PERSIST_EVIDENCE_TOKENS if tok in haystack))
    return WorkloadCmdPersistLint(flagged=not matched, skipped=False, matched_tokens=matched)


def build_run_spec(
    *,
    issue: int,
    intent: str,
    backend_value: Any,
    cluster: str | None = None,
    gpus: int | None = None,
    time_budget_hours: float | None = None,
    account: str | None = None,
    hydra_args: tuple[str, ...] = (),
    extra: dict[str, Any] | None = None,
    workload_cmd: str = "",
) -> RunSpec:
    """Build a :class:`RunSpec` from frontmatter-shaped inputs.

    The orchestrator extracts these from the task body / plan;
    construction lives here so the legacy backend-value normalization
    runs in ONE place. The ``cluster`` arg is honored only when
    ``backend_value`` is the per-cluster alias OR ``"cluster"`` (which
    normalizes to ``"nibi"``); otherwise the router itself ignores it.

    ``workload_cmd`` (#588) threads straight onto the spec; the
    exactly-one-of-(--workload-cmd / --hydra) production gate lives at
    the dispatch CLI — this builder stays permissive on neither (test
    factories + finalize-adjacent uses build bare specs). Both-set
    raises from ``RunSpec.__post_init__``.
    """
    backend = normalize_backend_value(backend_value)
    return RunSpec(
        issue=int(issue),
        intent=str(intent),
        gpus=gpus,
        time_budget_hours=time_budget_hours,
        account=account,
        hydra_args=tuple(str(a) for a in hydra_args),
        backend=backend,
        cluster=cluster,
        extra=dict(extra or {}),
        workload_cmd=str(workload_cmd or ""),
    )


# ---------------------------------------------------------------------------
# Handle (de)serialization for the bg-Bash poll bridge
# ---------------------------------------------------------------------------


def serialize_handle(handle: RunHandle) -> dict[str, Any]:
    """Serialize a :class:`RunHandle` to a JSON-safe dict.

    Used to write the handle to a sidecar JSON the bg-Bash poller
    reads. The :data:`EXPECTED_ARTIFACTS_HANDLE_KEY` declaration on
    ``extra`` is already a plain dict (per the artifacts module's
    schema), so a straight ``dict(handle.extra)`` is safe.
    """
    return {
        "backend": handle.backend,
        "cluster": handle.cluster,
        "job_id": handle.job_id,
        "pod_name": handle.pod_name,
        "scratch_dir": handle.scratch_dir,
        "log_path": handle.log_path,
        "extra": dict(handle.extra),
    }


def deserialize_handle(payload: dict[str, Any]) -> RunHandle:
    """Rebuild a :class:`RunHandle` from :func:`serialize_handle` output.

    Raises ``KeyError`` on a missing required field (programmer error;
    a corrupted sidecar would otherwise silently land the poller on a
    handle for the wrong issue). The artifact declaration is preserved
    on ``extra``; the verifier reads it back via
    :func:`backends.artifacts.expected_artifacts_from_handle`.
    """
    required = {"backend", "job_id", "pod_name", "scratch_dir", "log_path"}
    missing = sorted(k for k in required if k not in payload)
    if missing:
        raise KeyError(f"serialized RunHandle missing required fields: {missing}")
    return RunHandle(
        backend=payload["backend"],
        cluster=payload.get("cluster"),
        job_id=str(payload["job_id"]),
        pod_name=str(payload["pod_name"]),
        scratch_dir=str(payload["scratch_dir"]),
        log_path=str(payload["log_path"]),
        extra=dict(payload.get("extra") or {}),
    )


def write_handle_sidecar(handle: RunHandle, path: Path) -> None:
    """Write the serialized handle to ``path`` atomically (write-temp + rename).

    Creates the parent dir if absent (the ``.claude/cache/`` dir is
    not always pre-created). Atomic so a concurrent reader never sees
    a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_handle(handle)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    tmp.replace(path)


def read_handle_sidecar(path: Path) -> RunHandle:
    """Read the serialized handle from ``path``; raise on absent / malformed."""
    if not path.exists():
        raise FileNotFoundError(f"handle sidecar not found: {path}")
    return deserialize_handle(json.loads(path.read_text()))


#: Launch-only failover-prerequisite extras (#659/#909/#677/#1010) a
#: reconnect rewrite must not drop (#1122). The workload pair
#: (workload_cmd, hydra_args) is handled separately (atomic merge — the
#: RunSpec mutual exclusion, base.py ``RunSpec.__post_init__``).
RECONNECT_CARRY_FORWARD_EXTRA_KEYS: tuple[str, ...] = (
    "gpus",
    "time_budget_hours",
    "repo_branch",
    "gpu_count",
    "boot_disk_gb",
    "min_ram_gb",
    # #1669: launch env pins (WANDB_PROJECT et al.) — carried VERBATIM,
    # with no sanitize step here BY CHOICE: the pins were strict-validated
    # at the original launch (dispatch_issue._parse_env_pins), every
    # renderer re-validates fail-loud at consumption (covers a hand-edited
    # sidecar on this live-operator reconnect path), and the failover
    # reconstructors keep their own per-key sanitize-and-warn
    # (backends.base.sanitize_env_pins). The `v not in (None, "", [])`
    # filter keeps a non-empty dict and skips an absent key (legacy
    # sidecars carry no env_pins).
    "env_pins",
)


def _prior_sidecar_failover_extras(sidecar: Path) -> dict[str, Any] | None:
    """Snapshot the prior sidecar's failover extras BEFORE ``route()`` overwrites it.

    Returns ``{backend, pod_name, job_id, handle, extra: {...}}`` with
    ``extra`` holding only the failover-prerequisite keys (#1122): the
    atomic ``workload_cmd``/``hydra_args`` pair (kept only when at least
    one is non-empty) plus every non-empty
    :data:`RECONNECT_CARRY_FORWARD_EXTRA_KEYS` value. ``handle`` (#2038,
    additive — :func:`_carry_forward_reconnect_extras` reads only the four
    original keys) is the FULL deserialized prior :class:`RunHandle`, so
    the superseded-RunPod reap can read failure-path keys
    (``pod_id`` / ``workload_executed`` / ``workload_start_error``) the
    #1122 snapshot deliberately filters out. Best-effort:
    returns ``None`` on an absent sidecar (silent — the common fresh
    dispatch) or a malformed one (logged at DEBUG) — the carry-forward
    is a safety net, never a gate.
    """
    try:
        h = read_handle_sidecar(sidecar)
    except FileNotFoundError:
        return None
    except (KeyError, ValueError, OSError) as exc:  # ValueError covers JSONDecodeError
        logger.debug(
            "dispatch_for_issue: prior sidecar %s unreadable (%s); no reconnect carry-forward",
            sidecar,
            exc,
        )
        return None
    ex = h.extra or {}
    kept: dict[str, Any] = {}
    if ex.get("workload_cmd") or ex.get("hydra_args"):
        kept["workload_cmd"] = ex.get("workload_cmd") or ""
        kept["hydra_args"] = list(ex.get("hydra_args") or ())
    for k in RECONNECT_CARRY_FORWARD_EXTRA_KEYS:
        v = ex.get(k)
        if v not in (None, "", []):
            kept[k] = v
    return {
        "backend": h.backend,
        "pod_name": h.pod_name,
        "job_id": h.job_id,
        "handle": h,
        "extra": kept,
    }


def _carry_forward_reconnect_extras(handle: RunHandle, prior: dict | None) -> RunHandle:
    """Fill launch-only extras a RECONNECT rewrite dropped (#1122).

    Fires ONLY when: the handle is a reconnect (``extra['reconnected']``
    truthy — set only by GCP's ``reconnect_or_none``, so the merge is
    GCP-scoped by construction), AND ``prior`` binds to the SAME
    instance (backend + pod_name + job_id — ``job_id`` is the per-create
    GCE instance id, so a stale sidecar from a dead incarnation never
    donates; an EMPTY ``job_id`` on EITHER side is treated as no-match,
    so a degenerate ``("gcp", name, "")`` binding can never match across
    incarnations), AND a key is absent/empty on the new handle while
    non-empty on the prior. The ``(workload_cmd, hydra_args)`` pair
    merges ATOMICALLY and only when the handle carries NEITHER
    (preserves the RunSpec mutual exclusion,
    base.py ``RunSpec.__post_init__``). Logs a WARNING naming the
    carried keys. Returns the input handle unchanged (identity) when
    nothing applies.
    """
    hx = handle.extra or {}
    if not prior or not prior["extra"] or not hx.get("reconnected"):
        return handle
    # #1122 §11.5(1): empty job_id on either side = no-match (a bare
    # instance-id can be "" on a degraded API read; never bind on it).
    if not prior["job_id"] or not handle.job_id:
        return handle
    if (prior["backend"], prior["pod_name"], prior["job_id"]) != (
        handle.backend,
        handle.pod_name,
        handle.job_id,
    ):
        return handle
    merged, carried = dict(hx), []
    pe = prior["extra"]
    if not (hx.get("workload_cmd") or hx.get("hydra_args")) and (
        "workload_cmd" in pe or "hydra_args" in pe
    ):
        merged["workload_cmd"] = pe.get("workload_cmd", "")
        merged["hydra_args"] = list(pe.get("hydra_args") or ())
        carried += ["workload_cmd", "hydra_args"]
    for k in RECONNECT_CARRY_FORWARD_EXTRA_KEYS:
        if k in pe and merged.get(k) in (None, "", []):
            merged[k] = pe[k]
            carried.append(k)
    if not carried:
        return handle
    logger.warning(
        "dispatch_for_issue: reconnect rewrite for %s dropped launch-only extras; "
        "carried forward %s from the prior handle sidecar (#1122).",
        handle.pod_name,
        carried,
    )
    from dataclasses import replace

    return replace(handle, extra=merged)


# ---------------------------------------------------------------------------
# Superseded-RunPod-fallback reap (#2038)
# ---------------------------------------------------------------------------
#
# The #1739 incident: a RunPod fallback launch FAILED at workload start
# (#954 partial handle — pod provisioned, workload never started), the next
# dispatch overwrote the sidecar with a fresh handle, and the failed pod
# kept billing with NO record pointing at it. This section reaps exactly
# that shape at the ONE seam that sees both records: ``dispatch_for_issue``
# holds the prior sidecar snapshot AND the new handle right before the
# authoritative sidecar overwrite. Everything here is best-effort — every
# failure degrades to a WARN and the new launch is NEVER blocked.


def _owner_issue_from_pod_name(name: str) -> int | None:
    """Owning issue for a MANAGED pod name via the canonical #1334 parser.

    Delegates to ``scripts/pod_lifecycle.py::_issue_from_pod_name`` (the
    one-parser convention: ``pod-<N>`` / ``pod-<N>-<slug>`` / legacy
    ``epm-issue-<N>``; ``pod-17`` never matches issue 1739). Lazy import —
    this module's top-level imports stay backends-only.
    """
    from scripts.pod_lifecycle import _issue_from_pod_name

    return _issue_from_pod_name(name)


def _reap_skip(action: str, reason: str, *, marker: bool = False) -> SupersededReapDecision:
    """Skip-shaped :class:`SupersededReapDecision` (no target, optional marker)."""
    return SupersededReapDecision(
        action=action, target_pod_id=None, reason=reason, surface_marker=marker
    )


def _idless_reap_target(
    *,
    prior_pod_name: str,
    new_backend: str,
    new_pod_name: str | None,
    new_pod_id: str | None,
    matches: list[tuple[str, str]],
    any_running: bool,
) -> str | SupersededReapDecision:
    """Resolve the reap target for a LEGACY id-less prior sidecar (#2038).

    Identity must come from the live EXACT-name matches alone: exclude the
    new launch's own pod id when known; a same-name new RunPod launch whose
    own id is UNKNOWN makes every match ambiguous (any of them may be the
    new pod). Returns the single unambiguous candidate pod id, or the
    skip decision (``skip-new-pod-id-unknown`` / ``skip-pod-gone`` /
    ``skip-name-ambiguous``).
    """
    candidates = [pid for pid, _ in matches]
    if new_pod_id:
        candidates = [pid for pid in candidates if pid != new_pod_id]
    elif new_backend == "runpod" and new_pod_name == prior_pod_name and candidates:
        return _reap_skip(
            "skip-new-pod-id-unknown",
            f"prior sidecar for {prior_pod_name} carries no pod id and the NEW "
            "launch's pod id is unknown under the SAME name — any live match "
            "may be the new pod; not touched",
            marker=any_running,
        )
    if not candidates:
        return _reap_skip(
            "skip-pod-gone",
            f"no live pod named exactly {prior_pod_name} beyond the new launch's — nothing to reap",
        )
    if len(candidates) > 1:
        return _reap_skip(
            "skip-name-ambiguous",
            f"{len(candidates)} live pods named exactly {prior_pod_name} (id-less "
            "prior sidecar) — identity ambiguous, none touched (#1739 duplicate-"
            "name class)",
            marker=any_running,
        )
    return candidates[0]


@dataclass(frozen=True)
class SupersededReapDecision:
    """Outcome of :func:`decide_superseded_runpod_reap` (#2038).

    ``action`` is ``"terminate"`` / ``"stop"`` (``target_pod_id`` set — the
    EXACT pod id to act on) or one of the ``skip-*`` tokens (no action).
    ``surface_marker`` is True when the skip leaves a prior pod visibly
    RUNNING — the wrapper posts a durable ``epm:progress`` note so a
    still-billing pod never hides in session logs alone.
    """

    action: str
    target_pod_id: str | None
    reason: str
    surface_marker: bool


def decide_superseded_runpod_reap(
    *,
    issue: int,
    prior_backend: str | None,
    prior_pod_name: str | None,
    prior_pod_id: str | None,
    prior_workload_executed: bool,
    prior_workload_start_error: str | None,
    new_backend: str,
    new_pod_name: str | None,
    new_pod_id: str | None,
    live_matches_fn: Callable[[], list[tuple[str, str]] | None],
    keep_running_fn: Callable[[], bool | None],
    issue_from_name: Callable[[str], int | None] | None = None,
) -> SupersededReapDecision:
    """Decision table for reaping a superseded prior RunPod fallback pod (#2038).

    Deterministic given its inputs; the two lazy providers keep the cheap
    skip rows network-free (``live_matches_fn`` is only called once the
    prior parses as THIS issue's managed RunPod pod and is not the pod the
    new launch just produced). Provider contracts: ``live_matches_fn``
    returns ``[(pod_id, desired_status), ...]`` for LIVE pods whose name
    EXACTLY equals ``prior_pod_name`` (never a prefix match — ``pod-17``
    must not match ``pod-1739``), or ``None`` when the live read failed;
    ``keep_running_fn`` returns the tri-state ``keep-running`` tag read
    (``None`` = unreadable ⇒ do-not-destroy, fail-closed).

    Disposition (plan #2038 §3 Component 2) keys on the prior handle's
    FAILURE-path fields: ``workload_start_error`` present → the #954
    partial-launch shape, TERMINATE (nothing ever ran — the invisible-
    billing pod); ``workload_executed`` falsy without it → provision-only
    launch (an experimenter may be driving it) → skip + WARN;
    ``workload_executed`` true → reversible STOP (never sharpened to
    terminate — data may sit on the volume).
    """
    parse = issue_from_name if issue_from_name is not None else _owner_issue_from_pod_name
    _skip = _reap_skip

    if prior_backend != "runpod":
        return _skip("skip-prior-not-runpod", f"prior backend {prior_backend!r} is not runpod")
    if not prior_pod_name or parse(prior_pod_name) != issue:
        return _skip(
            "skip-unmanaged-name",
            f"prior pod name {prior_pod_name!r} does not parse as a managed pod of "
            f"issue {issue} — never touched (#1334 one-parser grammar)",
        )
    if prior_pod_id and new_pod_id and prior_pod_id == new_pod_id:
        return _skip(
            "skip-same-pod",
            f"prior pod id {prior_pod_id} IS the new launch's pod (reuse/reconnect) — "
            "never destroy the pod we just launched on",
        )

    matches = live_matches_fn()
    if matches is None:
        return _skip(
            "skip-live-read-failed",
            f"live RunPod API read failed for {prior_pod_name} — cannot establish "
            "identity; not touched (may still be billing)",
        )
    any_running = any((status or "").upper() == "RUNNING" for _, status in matches)

    if prior_pod_id:
        if prior_pod_id not in {pid for pid, _ in matches}:
            return _skip(
                "skip-pod-gone",
                f"prior pod {prior_pod_name} (pod_id={prior_pod_id}) is no longer in "
                "the live inventory — nothing to reap",
            )
        target = prior_pod_id
    else:
        resolved = _idless_reap_target(
            prior_pod_name=prior_pod_name,
            new_backend=new_backend,
            new_pod_name=new_pod_name,
            new_pod_id=new_pod_id,
            matches=matches,
            any_running=any_running,
        )
        if isinstance(resolved, SupersededReapDecision):
            return resolved
        target = resolved

    keep_running = keep_running_fn()
    if keep_running is True:
        return _skip(
            "skip-keep-running",
            f"issue {issue} carries the keep-running tag — prior pod "
            f"{prior_pod_name} (pod_id={target}) not touched",
            marker=any_running,
        )
    if keep_running is None:
        return _skip(
            "skip-keep-running-unreadable",
            f"keep-running tag state unreadable for issue {issue} — fail-closed, "
            f"prior pod {prior_pod_name} (pod_id={target}) not touched",
            marker=any_running,
        )

    if prior_workload_start_error:
        return SupersededReapDecision(
            action="terminate",
            target_pod_id=target,
            reason=(
                f"prior fallback launch of {prior_pod_name} (pod_id={target}) failed at "
                "workload start (#954 partial handle: workload_start_error present) and "
                "is superseded by this dispatch — nothing ever ran on it"
            ),
            surface_marker=False,
        )
    if not prior_workload_executed:
        return _skip(
            "skip-provision-only",
            f"prior pod {prior_pod_name} (pod_id={target}) was a provision-only launch "
            "(no workload_start_error, workload_executed false) — an experimenter may "
            "be driving it; not touched",
            marker=any_running,
        )
    return SupersededReapDecision(
        action="stop",
        target_pod_id=target,
        reason=(
            f"prior pod {prior_pod_name} (pod_id={target}) ran a workload "
            "(workload_executed true) and is superseded by this dispatch — reversible "
            "stop (volume preserved; never auto-terminated)"
        ),
        surface_marker=False,
    )


def _default_live_name_matches(pod_name: str) -> list[tuple[str, str]] | None:
    """EXACT-name live matches ``[(pod_id, desired_status)]``; ``None`` on failure.

    Uses ``list_team_pods`` (team-scoped GraphQL) rather than
    ``get_pod_by_name`` because the latter returns only the FIRST match —
    the 0/1/>1 disposition needs the full count (plan #2038 §8 allowed
    deviation).
    """
    try:
        from scripts.runpod_api import list_team_pods  # lazy: module top stays backends-only

        return [
            (pod.pod_id, pod.desired_status or "")
            for pod in list_team_pods()
            if pod.name == pod_name
        ]
    except Exception as exc:
        logger.warning(
            "superseded-runpod-reap: live pod listing failed for %s (%r) — skip",
            pod_name,
            exc,
        )
        return None


def _default_keep_running_state(issue: int) -> bool | None:
    """Tri-state keep-running tag read; ``None`` (do-not-destroy) on failure."""
    try:
        from explore_persona_space.task_workflow import keep_running_tag_state

        return keep_running_tag_state(issue)
    except Exception as exc:
        logger.warning(
            "superseded-runpod-reap: keep-running tag read failed for issue %s (%r) — "
            "fail-closed (treated as unreadable)",
            issue,
            exc,
        )
        return None


def _default_reap_marker_poster(issue: int, note: str) -> None:
    """Durable ``epm:progress`` note via the canonical task-workflow helper.

    ``post_event`` resolves the task folder through the registry (never a
    hand-built ``tasks/<status>`` path); an unresolvable tasks tree raises
    and the caller degrades to a WARN — a marker-post failure never fails
    the launch.
    """
    from explore_persona_space.task_workflow import post_event

    post_event(issue, "epm:progress", by="issue-dispatch", note=note)


def _reap_superseded_runpod_fallback(
    *,
    issue: int,
    prior: dict[str, Any] | None,
    new_handle: RunHandle,
    live_matches_fn: Callable[[], list[tuple[str, str]] | None] | None = None,
    keep_running_fn: Callable[[], bool | None] | None = None,
    terminate_fn: Callable[[str], Any] | None = None,
    stop_fn: Callable[[str], Any] | None = None,
    marker_poster: Callable[[int, str], None] | None = None,
) -> dict[str, Any] | None:
    """Reap a prior RunPod fallback pod this dispatch supersedes (#2038).

    Called from :func:`dispatch_for_issue` AFTER the new launch succeeded and
    BEFORE the authoritative sidecar write. Returns the audit record to
    persist on the NEW handle's ``extra["superseded_runpod_reaped"]``
    (``{pod_name, pod_id, action, ts}``, for ``terminate`` / ``stop`` and
    their ``-failed`` variants) or ``None`` when nothing was acted on (skips
    are log-only + optionally a durable marker — deliberately NOT recorded on
    the sidecar, so routine relaunches do not accrete audit keys). Best-effort
    end to end: every arm logs its exception explicitly and the launch is
    never blocked.
    """
    if not prior:
        return None
    prior_handle = prior.get("handle")
    if prior_handle is None or prior.get("backend") != "runpod":
        # Silent: the common non-RunPod prior (GCP/SLURM) or a pre-#2038
        # snapshot shape without the handle key.
        return None
    prior_pod_name = prior.get("pod_name") or ""
    pe = prior_handle.extra or {}
    decision = decide_superseded_runpod_reap(
        issue=issue,
        prior_backend=prior.get("backend"),
        prior_pod_name=prior_pod_name,
        prior_pod_id=str(pe.get("pod_id") or "") or None,
        prior_workload_executed=bool(pe.get("workload_executed")),
        prior_workload_start_error=str(pe.get("workload_start_error") or "") or None,
        new_backend=new_handle.backend,
        new_pod_name=new_handle.pod_name,
        new_pod_id=str((new_handle.extra or {}).get("pod_id") or "") or None,
        live_matches_fn=(
            live_matches_fn
            if live_matches_fn is not None
            else lambda: _default_live_name_matches(prior_pod_name)
        ),
        keep_running_fn=(
            keep_running_fn
            if keep_running_fn is not None
            else lambda: _default_keep_running_state(issue)
        ),
    )

    record: dict[str, Any] | None = None
    note: str | None = None
    if decision.action in ("terminate", "stop"):
        realized = _perform_superseded_reap(
            decision=decision,
            pod_name=prior_pod_name,
            issue=issue,
            terminate_fn=terminate_fn,
            stop_fn=stop_fn,
        )
        from datetime import UTC, datetime  # lazy: keep module top unchanged

        record = {
            "pod_name": prior_pod_name,
            "pod_id": decision.target_pod_id,
            "action": realized,
            "ts": datetime.now(tz=UTC).isoformat(),
        }
        note = (
            f"superseded-runpod-reap: {realized} — {decision.reason} (#2038; "
            f"new dispatch backend={new_handle.backend})"
        )
    else:
        logger.info("superseded-runpod-reap: %s — %s (#2038)", decision.action, decision.reason)
        if decision.surface_marker:
            note = (
                f"superseded-runpod-reap: SKIPPED ({decision.action}) — {decision.reason}. "
                f"A pod named {prior_pod_name} is still RUNNING and was NOT touched; "
                f"check: uv run python scripts/pod.py list-ephemeral --issue {issue} (#2038)"
            )
    if note is not None:
        try:
            poster = marker_poster if marker_poster is not None else _default_reap_marker_poster
            poster(issue, note)
        except Exception as exc:
            logger.warning(
                "superseded-runpod-reap: durable marker post failed for issue %s (%r) — "
                "launch proceeds; note was: %s",
                issue,
                exc,
                note,
            )
    return record


def _apply_superseded_reap(
    spec_issue: int,
    prior: dict[str, Any] | None,
    result: RouteResult,
    handle: RunHandle,
) -> tuple[RouteResult, RunHandle]:
    """Run the #2038 superseded-RunPod reap and attach its audit record.

    Mirrors the :func:`_apply_carry_forward_to_result` shape: returns
    ``(result, handle)`` — rebuilt with
    ``extra["superseded_runpod_reaped"]`` when a terminate/stop (or
    ``-failed`` variant) happened, the inputs unchanged (identity)
    otherwise. Best-effort: an unexpected reap failure logs LOUDLY and the
    launch proceeds untouched.
    """
    if prior is None:
        return result, handle
    try:
        record = _reap_superseded_runpod_fallback(issue=spec_issue, prior=prior, new_handle=handle)
    except Exception:
        logger.exception(
            "dispatch_for_issue: superseded-RunPod reap failed for issue %s — "
            "launch proceeds; a prior fallback pod may still be billing (#2038)",
            spec_issue,
        )
        return result, handle
    if record is None:
        return result, handle
    from dataclasses import replace as dc_replace

    new_extra = dict(handle.extra)
    new_extra["superseded_runpod_reaped"] = record
    handle = dc_replace(handle, extra=new_extra)
    return dc_replace(result, handle=handle), handle


def _perform_superseded_reap(
    *,
    decision: SupersededReapDecision,
    pod_name: str,
    issue: int,
    terminate_fn: Callable[[str], Any] | None,
    stop_fn: Callable[[str], Any] | None,
) -> str:
    """Execute a ``terminate`` / ``stop`` decision; return the realized action.

    ``terminate`` runs under the sanctioned owner-driven
    :func:`kill_approval.verified_teardown` grant — the reaped pod is the
    #954 partial-launch shape whose workload never started, so upload
    verification is vacuous (nothing was produced). ``stop`` is reversible
    and ungated by design. Never raises: a failure logs the exception and
    returns the ``-failed`` variant for the audit record.
    """
    target = decision.target_pod_id
    assert target, "terminate/stop decisions always carry a target pod id"
    try:
        if decision.action == "terminate":
            if terminate_fn is None:
                from scripts.runpod_api import terminate_pod  # lazy import

                terminate_fn = terminate_pod
            from explore_persona_space.backends.kill_approval import verified_teardown

            with verified_teardown(
                target=f"{pod_name} ({target})",
                reason=(
                    "owner-driven teardown of a superseded RunPod fallback pod whose "
                    "workload never started (#954 partial handle) — uploads vacuously "
                    "verified (#2038)"
                ),
            ):
                terminate_fn(target)
        else:
            if stop_fn is None:
                from scripts.runpod_api import stop_pod  # lazy import

                stop_fn = stop_pod
            stop_fn(target)
        logger.warning(
            "superseded-runpod-reap: %sd prior pod %s (pod_id=%s, issue %s) — %s (#2038)",
            decision.action,
            pod_name,
            target,
            issue,
            decision.reason,
        )
        return decision.action
    except Exception:
        logger.exception(
            "superseded-runpod-reap: %s of prior pod %s (pod_id=%s, issue %s) FAILED — "
            "the pod may still be billing; manual teardown required (#2038)",
            decision.action,
            pod_name,
            target,
            issue,
        )
        return f"{decision.action}-failed"


# ---------------------------------------------------------------------------
# Terminal-exception translation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalTranslation:
    """How a router terminal exception maps to ``epm:failure`` + status.

    Fields:

    * ``failure_class`` — ``"infra"`` or ``"code"`` (the ``epm:failure
      v1`` field the failure-classifier looks for).
    * ``status`` — the status the ``/issue`` skill should mutate to
      (``"blocked"`` for every terminal in slice 6).
    * ``note`` — the human-readable + machine-greppable body the
      orchestrator posts as the ``epm:failure`` note. Carries the
      ``failure_class:`` first line so the failure classifier
      short-circuits (see SKILL.md Step 7's classification table).
    """

    failure_class: str
    status: str
    note: str


def classify_terminal_exception(exc: BaseException) -> TerminalTranslation:
    """Map a router terminal exception to its ``epm:failure`` shape.

    The typed router terminals are exhaustively handled (each is a
    distinct ``RouteError`` subclass — incl. the #2028
    :class:`GcpDisabledError` policy refusal). Anything else propagates as
    a plain ``RouteError`` whose handling is the caller's concern.
    """
    if isinstance(exc, BackendPrepareError):
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: backend_prepare_failed\n"
                f"kind: {exc.kind}\n"
                f"cluster: {exc.cluster}\n"
                f"detail: {exc.reason}"
            ),
        )
    if isinstance(exc, GcpDisabledError):
        # #2028: an explicit ``backend: gcp`` pin while GCP provisioning is
        # disabled by policy. A POLICY refusal, not a capacity outcome — the
        # reason token is NOT in the watcher's TRANSIENT_CAPACITY_REASONS
        # (nothing will "free up"; auto-retry would loop a policy-refused
        # launch). The fix is a human changing the pin, or a deliberate
        # rollback flip of router.GCP_PROVISIONING_DISABLED.
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: {ROUTE_REASON_GCP_DISABLED}\n"
                "recovery: re-dispatch WITHOUT the gcp pin (omit --backend / clear "
                "the backend: frontmatter so the auto chain routes fellows -> free "
                "SLURM lanes), or flip router.GCP_PROVISIONING_DISABLED = False for "
                "a deliberate rollback (#2028)\n"
                f"detail: {exc}"
            ),
        )
    if isinstance(exc, GpuRamBelowMinRamGbError):
        # #1998: --min-ram-gb declared a host-RAM floor above every reachable
        # GCP GPU rung (pinned intent, or ladder-exhausted). DESIGN mismatch,
        # NOT a transient capacity outcome — the reason token is NOT in the
        # watcher's TRANSIENT_CAPACITY_REASONS. The fix is dropping / lowering
        # ``--min-ram-gb`` or pinning a wider intent, never an auto-retry.
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: {ROUTE_REASON_GPU_RAM_BELOW_MIN_RAM_GB}\n"
                "recovery: drop or lower --min-ram-gb, or pin a wider intent whose "
                "resolved GCP machine satisfies the requested host RAM (see "
                "backends/gcp.MACHINE_RAM_GIB for the per-machine RAM table)\n"
                f"detail: {exc}"
            ),
        )
    if isinstance(exc, RunPodStoppedPodCollisionError):
        # #1997: the RunPod terminal rung was refused by pod_lifecycle's
        # stopped-pod same-name collision guard (exit 76) — a STOPPED pod-<N>
        # exists and a duplicate-named create would hijack its name-keyed
        # state rows (the #1739 incident). A STRUCTURAL refusal, not a
        # capacity outcome: the reason token is NOT in the watcher's
        # TRANSIENT_CAPACITY_REASONS (nothing will "free up"; auto-retry
        # would hot-loop the same refusal or race a human's recovery). The
        # fix is a human action — resume the stopped pod, terminate it with
        # approval, or provision under a distinct name.
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: {ROUTE_REASON_RUNPOD_STOPPED_POD_COLLISION}\n"
                "recovery: `uv run python scripts/pod.py resume --issue <N>` to reuse "
                "the stopped pod, or `uv run python scripts/pod.py terminate "
                "--issue <N> --yes --approve` then re-dispatch, or provision with "
                "--name-suffix <slug>, or pass --allow-stopped-duplicate to "
                "deliberately create the duplicate (#1997)\n"
                f"detail: {exc.reason}\n"
                f"attempts: {json.dumps(exc.attempts, sort_keys=True)}"
            ),
        )
    if isinstance(exc, CpuFallbackInfeasibleError):
        # #1010: the RunPod CPU lane EXISTS for this intent but cannot satisfy
        # the plan's stated footprint (disk / RAM). Distinct reason so the
        # marker trail names the real cause; like the parent, NOT in the
        # watcher's TRANSIENT_CAPACITY_REASONS (the instance can never grow to
        # fit the plan — auto-retry would loop a structurally-infeasible
        # launch). MUST come BEFORE the CpuExhaustedNoRunpodLaneError branch
        # (a subclass IS-A parent, so the parent check would shadow it — the
        # same ordering rule documented on the branches below).
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: {ROUTE_REASON_CPU_FALLBACK_INFEASIBLE}\n"
                "recovery: re-dispatch on the big-footprint CPU lane — "
                "uv run python scripts/dispatch_issue.py launch --issue <N> "
                "--intent cpu-bigmem ... (or shrink the plan footprint below "
                "the instance cap)\n"
                f"detail: {exc.reason}\n"
                f"attempts: {json.dumps(exc.attempts, sort_keys=True)}"
            ),
        )
    if isinstance(exc, CpuExhaustedNoRunpodLaneError):
        # #677: a CPU-only intent reached the RunPod terminal rung (GCP
        # exhausted / sync failover) and RunPod has no CPU lane. failure_class
        # is still infra (a capacity-class terminal), but the reason is DISTINCT
        # from no_compute_available so the watcher's capacity-retry pass does
        # NOT hot-retry a structurally-unservable run. MUST come BEFORE the
        # generic NoComputeAvailableError branch (a subclass IS-A base, so the
        # base check would otherwise shadow it).
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: {ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD}\n"
                f"detail: {exc.reason}\n"
                f"attempts: {json.dumps(exc.attempts, sort_keys=True)}"
            ),
        )
    if isinstance(exc, NoComputeAvailableError):
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: no_compute_available\n"
                f"detail: {exc.reason}\n"
                f"attempts: {json.dumps(exc.attempts, sort_keys=True)}"
            ),
        )
    if isinstance(exc, WorkloadSurfacedError):
        return TerminalTranslation(
            failure_class="code",
            status="blocked",
            note=(
                "failure_class: code\n"
                f"reason: workload_failure\n"
                f"chosen_kind: {exc.chosen_kind}\n"
                f"detail: {exc.reason}\n"
                f"evidence: {json.dumps(exc.evidence, sort_keys=True)}"
            ),
        )
    if isinstance(exc, GcpAttemptCapExceededError):
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: gcp_attempt_cap_exceeded\n"
                f"issue: {exc.issue}\n"
                f"attempts_today: {exc.attempts_today}\n"
                f"cap: {exc.cap}"
            ),
        )
    if isinstance(exc, ManualAttentionRequiredError):
        return TerminalTranslation(
            failure_class="infra",
            status="blocked",
            note=(
                "failure_class: infra\n"
                f"reason: manual_attention_required\n"
                f"kind: {exc.kind}\n"
                f"cluster: {exc.cluster}\n"
                f"orphaned_job_id: {exc.orphaned_job_id}\n"
                f"operator_action: verify job state, scancel if alive"
            ),
        )
    # Defensive: a future RouteError subclass should NOT silently slip
    # through with no translation; surface as infra + the message.
    return TerminalTranslation(
        failure_class="infra",
        status="blocked",
        note=(f"failure_class: infra\nreason: route_error\ndetail: {type(exc).__name__}: {exc}"),
    )


# ---------------------------------------------------------------------------
# The dispatch helper
# ---------------------------------------------------------------------------


def dispatch_for_issue(
    spec: RunSpec,
    *,
    runpod_backend: ComputeBackend,
    free_backends: dict[BackendKind, ComputeBackend] | None = None,
    gcp_backend: ComputeBackend | None = None,
    mila_socket_alive: Callable[[], bool] | None = None,
    marker_poster: Callable[..., None] | None = None,
    is_started: Callable[..., bool] | None = None,
    is_live_after_cancel: Callable[..., bool] | None = None,
    started_evidence_probe: Callable[..., Any] | None = None,
    reconnect_fn: Callable[..., Any] | None = None,
    lease_store: LeaseStore | None = None,
    config: RouterConfig | None = None,
    now_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    handle_sidecar_path: Path | None = None,
    write_sidecar: bool = True,
    expected_artifacts: dict[str, Any] | None = None,
) -> DispatchOutcome:
    """Run :func:`route` for the given ``spec`` and persist the resulting handle.

    Thin wrapper around :func:`route` — exists so the orchestrator has
    ONE call site for the production dispatch + the sidecar write +
    the artifact-declaration threading.

    Arguments mirror :func:`route` for the injection seams; defaults
    here are production-shaped (the orchestrator passes the SLURM
    backend instances + the real marker poster). Test callers pass
    mocks for every seam.

    ``expected_artifacts`` is the per-launch declaration the slice-2
    verifier reads (one dict per the artifacts module's schema). When
    provided AND the resulting handle's ``extra`` does NOT already
    carry an ``expected_artifacts`` key, we thread it onto the handle
    (some backends — GCP — populate it themselves; we never overwrite
    the backend's own declaration).

    Raises every router terminal verbatim — the caller (the ``/issue``
    skill) is responsible for catching + translating to its marker /
    status side effects via :func:`classify_terminal_exception`. This
    split keeps THIS helper pure: it dispatches + persists, the
    orchestrator decides what to do with the outcome.
    """
    if free_backends is None:
        free_backends = {}
    # Default Mila-socket-alive to the real probe (``ssh -o BatchMode=yes
    # mila true`` over the 12 h email-OTP ControlMaster socket). The
    # probe returns ``False`` on socket-down by contract — that's the
    # designed graceful path that tells the router to skip the Mila
    # lane this round; it is NOT an error. Tests inject a fake
    # ``mila_socket_alive`` callable to drive the gate deterministically
    # without touching real SSH.
    if mila_socket_alive is None:
        mila_socket_alive = _default_mila_socket_alive

    # Build the route() kwargs deliberately (helps a reviewer match
    # injected deps against router.route()'s signature).
    route_kwargs: dict[str, Any] = {
        "runpod_backend": runpod_backend,
        "free_backends": free_backends,
        "gcp_backend": gcp_backend,
        "mila_socket_alive": mila_socket_alive,
    }
    if marker_poster is not None:
        route_kwargs["marker_poster"] = marker_poster
    if is_started is not None:
        route_kwargs["is_started"] = is_started
    if is_live_after_cancel is not None:
        route_kwargs["is_live_after_cancel"] = is_live_after_cancel
    if started_evidence_probe is not None:
        route_kwargs["started_evidence_probe"] = started_evidence_probe
    if reconnect_fn is not None:
        route_kwargs["reconnect_fn"] = reconnect_fn
    # Lease / clock injections (production-default to router's own
    # defaults; tests pass a tmp_path-rooted ``LeaseStore`` + fast
    # ``now_fn`` / ``sleep_fn`` so a park-cap-exceeded run doesn't
    # actually wait the full ``FREE_WAIT_SECONDS``).
    if lease_store is not None:
        route_kwargs["lease_store"] = lease_store
    if config is not None:
        route_kwargs["config"] = config
    else:
        # #2161: the production default wires the per-process free-lane
        # park budget from EPS_LAUNCH_PARK_PROCESS_BUDGET_SECONDS (420 s
        # default; 0/negative → None = unlimited legacy parks). Callers
        # passing an explicit ``config`` own every knob themselves.
        route_kwargs["config"] = RouterConfig(
            park_process_budget_seconds=_env_park_process_budget()
        )
    if now_fn is not None:
        route_kwargs["now_fn"] = now_fn
    if sleep_fn is not None:
        route_kwargs["sleep_fn"] = sleep_fn

    # Early-persistence hook: the router invokes this with the handle
    # IMMEDIATELY after every successful launch / reconnect, BEFORE any
    # marker post — so even if everything after the launch crashes
    # (marker-post transport failure, this process OOM-killed, ...) the
    # launched handle is already on disk and ``dispatch_issue.py
    # finalize`` can tear the live VM / job down. The authoritative
    # write below re-writes the sidecar with the artifact declaration
    # threaded on; this early copy is the crash-window insurance.
    sidecar = handle_sidecar_path or default_handle_sidecar_path(
        spec.issue, lane_suffix=spec.extra.get("lane_suffix")
    )
    # #1122: snapshot the prior sidecar's failover extras BEFORE route()
    # can overwrite it — the on_launched early write inside route() is the
    # FIRST overwrite, so reading after route() is too late. NOTE: `prior`
    # is computed only when write_sidecar; a future write_sidecar=False
    # caller gets no returned-handle enrichment either (there is no prior
    # sidecar being clobbered in that mode, but keep the two paths' gating
    # aligned if that ever changes).
    prior = _prior_sidecar_failover_extras(sidecar) if write_sidecar else None
    if write_sidecar:
        route_kwargs["on_launched"] = lambda h: write_handle_sidecar(
            _carry_forward_reconnect_extras(h, prior), sidecar
        )

    result = route(spec, **route_kwargs)

    # Thread the expected-artifacts declaration if the launch path didn't
    # already populate one. The artifact verifier reads this off the
    # handle's ``extra`` at confirm_artifacts time (the silent-loss
    # safeguard).
    # #1122: mirror the on_launched merge on the returned handle so the
    # authoritative post-route sidecar write (and the caller's handle)
    # carry the same preserved extras.
    result, handle = _apply_carry_forward_to_result(result, prior)
    _warn_on_reconnected_workload(spec, result, handle)
    if expected_artifacts is not None and EXPECTED_ARTIFACTS_HANDLE_KEY not in handle.extra:
        from dataclasses import replace

        new_extra = dict(handle.extra)
        new_extra[EXPECTED_ARTIFACTS_HANDLE_KEY] = dict(expected_artifacts)
        handle = replace(handle, extra=new_extra)
        # Rebuild the result with the augmented handle so the caller +
        # sidecar both see it.
        from dataclasses import replace as dc_replace

        result = dc_replace(result, handle=handle)

    # #2038: reap a prior RunPod fallback pod this dispatch supersedes —
    # the ONE seam that sees both the prior sidecar snapshot and the new
    # handle, RIGHT BEFORE the authoritative overwrite erases the prior
    # record (#1739: a failed fallback pod billed invisibly once the next
    # dispatch overwrote the sidecar). Best-effort end to end: any failure
    # degrades to a WARN and never blocks the just-launched run.
    result, handle = _apply_superseded_reap(spec.issue, prior, result, handle)

    sidecar_written: Path | None = None
    sidecar_write_error: str | None = None
    if write_sidecar:
        sidecar_written, sidecar_write_error = _write_sidecar_guarded(handle, sidecar)

    return DispatchOutcome(
        result=result,
        handle_sidecar_path=sidecar_written,
        sidecar_write_error=sidecar_write_error,
    )


def _apply_carry_forward_to_result(result: RouteResult, prior: dict | None):
    """Mirror the on_launched carry-forward merge on the RETURNED handle (#1122).

    Returns ``(result, handle)`` — rebuilt with the enriched handle when
    the merge carried anything, the inputs unchanged (identity) otherwise.
    Keeps the authoritative post-route sidecar write and the caller's
    handle consistent with the early ``on_launched`` write.
    """
    handle = result.handle
    enriched = _carry_forward_reconnect_extras(handle, prior)
    if enriched is handle:
        return result, handle
    from dataclasses import replace

    return replace(result, handle=enriched), enriched


def _warn_on_reconnected_workload(spec: RunSpec, result: RouteResult, handle: RunHandle) -> None:
    """Reconnect loudness (#934/#923) — library-layer warning.

    A workload-carrying launch (``workload_cmd`` or ``hydra_args``) that
    resolved by RECONNECT dispatched NOTHING this invocation. Both
    reconnect layers must trip the caveat — the router scan
    (``result.reason == ROUTE_REASON_RECONNECT``) and the GCP-internal
    ``reconnect_or_none`` (which only marks
    ``handle.extra['reconnected']``). Warning-only: reconnect stays a
    success path (the exit-75 still-waiting rerun contract).
    """
    handle_extra = handle.extra or {}
    if (spec.workload_cmd or spec.hydra_args) and (
        result.reason == ROUTE_REASON_RECONNECT or handle_extra.get("reconnected")
    ):
        logger.warning(
            "dispatch_for_issue: route() RECONNECTED to existing %s job %s for issue %d — "
            "THIS invocation dispatched NO workload. Benign iff the instance was created by "
            "an earlier run of the SAME launch command (exit-75 still-waiting rerun); if this "
            "is a concurrent second lane for the same issue, the workload was NOT dispatched — "
            "relaunch with --lane-suffix (#934/#923).",
            result.chosen_kind,
            handle.pod_name,
            spec.issue,
        )


def _write_sidecar_guarded(handle: RunHandle, sidecar: Path) -> tuple[Path | None, str | None]:
    """Authoritative post-route sidecar write; ``OSError`` is loud, not fatal.

    The launch already succeeded — a live VM / job exists. Do NOT
    convert a persistence failure into an unclassified crash (the
    pre-fix rc=4 path stranded live infra with no recovery record).
    Log LOUD, return the error for the dispatch CLI to print next to
    the handle JSON, and keep the early ``on_launched`` copy if it
    landed (it lacks the artifact declaration but IS recoverable by
    finalize).
    """
    try:
        write_handle_sidecar(handle, sidecar)
        return sidecar, None
    except OSError as exc:
        sidecar_write_error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "dispatch_for_issue: handle sidecar write FAILED at %s (%s). "
            "Launch already succeeded (job_id=%s pod_name=%s) — the handle "
            "JSON on stdout is the recovery record; finalize may need "
            "--handle-file with a reconstructed sidecar.",
            sidecar,
            sidecar_write_error,
            handle.job_id,
            handle.pod_name,
        )
        if sidecar.exists():
            return sidecar, sidecar_write_error
        return None, sidecar_write_error


def _default_mila_socket_alive() -> bool:
    """Production default for the Mila socket gate.

    Delegates to :func:`backends.slurm.mila_socket_alive`, which runs the
    cheap ``ssh -o BatchMode=yes mila true`` probe over the
    ControlMaster socket. Returns ``False`` when the socket is down /
    OTP-expired / unreachable — that's the designed skip-the-lane
    signal, NOT an error.

    Wrapped here (not bound at import) so a test that imports
    :mod:`backends.issue_dispatch` does not also drag in the
    :mod:`backends.slurm` module's import-time SSH-helper resolution
    when it only wants to inject a fake gate. The body is the lazy
    import; the import itself is cheap (already loaded by every real
    code path that reaches the dispatch helper).
    """
    from explore_persona_space.backends.slurm import (
        mila_socket_alive as _slurm_mila_socket_alive,
    )

    return _slurm_mila_socket_alive()


# Backwards-compatible alias for the slice-6 stub name. Some external
# callers / tests imported ``_mila_socket_alive_stub`` directly; keep
# the symbol live but point it at the real probe so a stale import path
# yields the real behavior instead of permanent-False.
_mila_socket_alive_stub = _default_mila_socket_alive


__all__ = [
    "LANE_WORKLOAD_ENV_EXPORTS",
    "PERSIST_EVIDENCE_TOKENS",
    "DispatchOutcome",
    "SupersededReapDecision",
    "TerminalTranslation",
    "WorkloadCmdEnvLint",
    "WorkloadCmdInlineLint",
    "WorkloadCmdPersistLint",
    "build_run_spec",
    "classify_terminal_exception",
    "decide_superseded_runpod_reap",
    "default_handle_sidecar_path",
    "deserialize_handle",
    "dispatch_for_issue",
    "lint_workload_cmd_inline_interpreter",
    "lint_workload_cmd_lane_env",
    "lint_workload_cmd_persist_evidence",
    "normalize_backend_value",
    "read_handle_sidecar",
    "resolve_handle_sidecar_path",
    "serialize_handle",
    "strip_sentinel_append",
    "write_handle_sidecar",
]
