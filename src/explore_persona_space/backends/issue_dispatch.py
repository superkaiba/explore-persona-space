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
    ROUTE_REASON_RECONNECT,
    BackendPrepareError,
    CpuExhaustedNoRunpodLaneError,
    CpuFallbackInfeasibleError,
    GcpAttemptCapExceededError,
    LeaseStore,
    ManualAttentionRequiredError,
    NoComputeAvailableError,
    RouterConfig,
    RouteResult,
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
    if val in {"runpod", "nibi", "fir", "gcp", "mila", "auto"}:
        return val  # type: ignore[return-value]
    raise ValueError(
        f"unknown backend frontmatter value: {raw!r}. Expected one of: "
        "runpod, cluster, nibi, fir, gcp, mila, auto, or empty (auto)."
    )


# ---------------------------------------------------------------------------
# Workload-cmd lane-env lint (#1329, incident #825)
# ---------------------------------------------------------------------------


#: SLURM lane names (per-cluster backend values that all share one renderer /
#: one export contract). The legacy ``cluster`` alias normalizes to ``nibi``
#: (see :data:`_LEGACY_TO_ROUTER_BACKEND`), i.e. into this set.
_SLURM_LANES: tuple[str, ...] = ("nibi", "fir", "mila")

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
)


def _prior_sidecar_failover_extras(sidecar: Path) -> dict[str, Any] | None:
    """Snapshot the prior sidecar's failover extras BEFORE ``route()`` overwrites it.

    Returns ``{backend, pod_name, job_id, extra: {...}}`` with ``extra``
    holding only the failover-prerequisite keys (#1122): the atomic
    ``workload_cmd``/``hydra_args`` pair (kept only when at least one is
    non-empty) plus every non-empty
    :data:`RECONNECT_CARRY_FORWARD_EXTRA_KEYS` value. Best-effort:
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

    The five router terminals are exhaustively handled (each is a
    distinct ``RouteError`` subclass). Anything else propagates as a
    plain ``RouteError`` whose handling is the caller's concern.
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
    "DispatchOutcome",
    "TerminalTranslation",
    "WorkloadCmdEnvLint",
    "build_run_spec",
    "classify_terminal_exception",
    "default_handle_sidecar_path",
    "deserialize_handle",
    "dispatch_for_issue",
    "lint_workload_cmd_lane_env",
    "normalize_backend_value",
    "read_handle_sidecar",
    "resolve_handle_sidecar_path",
    "serialize_handle",
    "write_handle_sidecar",
]
