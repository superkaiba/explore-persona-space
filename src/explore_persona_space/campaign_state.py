"""Typed helpers for the per-campaign state file (task #586).

A **campaign** is a task (``kind: campaign``) pinned to ONE open question.
Its machine-readable execution state lives at::

    tasks/<status>/<N>/artifacts/campaign-state.json

and is read/written ONLY through this module. Schema (``schema_version: 1``)::

    {
      "schema_version": 1,
      "campaign_task": 586,
      "question_anchor": "q:leak-predictor",
      "started_at": "...", "wall_clock_deadline": "...",
      "budget": {"gpu_hours_total": 250.0, "gpu_hours_committed": 0.0},
      "limits": {"max_experiments": 8, "max_concurrent_children": 4,
                 "per_child_gpu_hours_cap": 100.0, "wall_clock_days": 5},
      "stop": {"stopped": false, "stop_reason": null,
               "confidence_target": "HIGH", "current_confidence": null,
               "dry_counter": 0, "dry_limit": 3},
      "experiments": [
        {"id": "exp-01", "title": "...", "hypothesis": "...",
         "depends_on": [], "gpu_hours_est": 30.0,
         "status": "planned", "child_task": null, "headline": null,
         "confidence": null, "belief_shift": null}
      ],
      "last_digest_at": null
    }

Path resolution goes through :func:`explore_persona_space.task_workflow.
find_task_path` — NEVER a hand-built ``tasks/...`` path (CLAUDE.md rule;
pinned by ``tests/test_no_direct_task_path_construction.py``). Writes are
atomic (temp file + ``os.replace``). This module does NOT git-commit: the
``/campaign`` skill commits ``artifacts/`` updates by explicit path per the
concurrent-committer rules.

Budgets here are **GPU-hour** caps, never dollar caps
(``tests/test_no_dollar_budget_caps.py``). They gate FILING new children
(the campaign stops proposing work when committed hours exceed the total);
they never abort a child mid-run.

The ``## Campaign Brief`` format ``init_state_from_brief`` parses:

* The task body MUST contain a ``## Campaign Brief`` H2; the brief section
  runs to the next H2 (or EOF).
* The question anchor is the first ``q:<slug>`` token inside the brief
  (an anchor into ``docs/open_questions.md``).
* The initial experiment DAG is the first markdown table inside the brief
  whose header row contains the columns ``id | title | hypothesis |
  depends_on | gpu_hours_est`` (case-insensitive, any order, extra columns
  ignored). ``depends_on`` is a comma-separated list of experiment ids
  (``-`` / ``none`` / empty = no dependencies).
* Budget/limit overrides come from the task frontmatter's optional
  ``campaign:`` mapping (keys = :data:`OVERRIDE_KEYS`). Unknown keys fail
  loud (typo guard).

Budget/limit seeding precedence (single-pathed — the /campaign skill's
Step 0 always initializes through :func:`init_state_from_brief`, so a
``spawn-campaign --budget-gpu-hours 50`` is actually enforced):

1. frontmatter ``campaign:`` overrides (the user-reviewed brief wins);
2. caps recorded in the session registry entry
   ``~/.eps-autonomous/campaign-<N>.json`` (written by ``spawn_session.py
   spawn-campaign`` from its CLI flags; key mapping in
   :data:`_REGISTRY_CAP_KEY_MAP`);
3. the module defaults below.

``stop.current_confidence`` is the CAMPAIGN-LEVEL working belief about the
question (nullable; null until the /campaign skill sets it at an ingest,
from the world model's updated answer to the question). The
confidence-target stop criterion compares THIS field to
``stop.confidence_target`` — per-child clean-result confidence tags are
PER-CLAIM, not per-question, and deliberately never trip the stop on
their own.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from explore_persona_space.task_workflow import find_task_path

# Session registry dir shared with spawn_session.py / the watcher. Module
# constant (not inlined) so tests can monkeypatch it hermetic.
AUTONOMOUS_REGISTRY_DIR = Path.home() / ".eps-autonomous"

# ─── Schema constants + defaults (approved 2026-06-10, plan #586) ──────────

SCHEMA_VERSION = 1
STATE_FILENAME = "campaign-state.json"

DEFAULT_GPU_HOURS_TOTAL = 250.0
DEFAULT_PER_CHILD_GPU_HOURS_CAP = 100.0
DEFAULT_MAX_EXPERIMENTS = 8
DEFAULT_MAX_CONCURRENT_CHILDREN = 4
DEFAULT_WALL_CLOCK_DAYS = 5
DEFAULT_DRY_LIMIT = 3
DEFAULT_CONFIDENCE_TARGET = "HIGH"

# Lifecycle of one experiment row in the campaign DAG.
EXPERIMENT_STATUSES = (
    "planned",  # in the DAG, not yet filed as a child task
    "filed",  # child task created, session not necessarily launched
    "running",  # child session spawned and driving
    "landed",  # child reached awaiting_promotion/completed; not yet ingested
    "ingested",  # clean-result folded into the world model
    "abandoned",  # child blocked twice / dropped; committed hours released
    "waiting-user",  # child parked over the per-child cap (plan_pending)
)

# Statuses that occupy a concurrency slot (work in flight on a child).
CONCURRENCY_STATUSES = frozenset({"filed", "running", "landed"})

# Statuses that count as "spent" toward the max_experiments stop criterion.
FINISHED_STATUSES = frozenset({"ingested", "abandoned"})

# Frontmatter `campaign:` mapping keys accepted as overrides.
OVERRIDE_KEYS = frozenset(
    {
        "gpu_hours_total",
        "per_child_gpu_hours_cap",
        "max_experiments",
        "max_concurrent_children",
        "wall_clock_days",
        "dry_limit",
        "confidence_target",
    }
)

# Confidence ordering for the confidence-target stop criterion. Unknown
# strings rank below LOW (never satisfy the target).
_CONFIDENCE_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "DETERMINATE": 3}

# Registry-entry cap keys (campaign-<N>.json, written by spawn-campaign)
# mapped to their state/override key names — precedence tier 2 of the
# budget/limit seeding (see the module docstring).
_REGISTRY_CAP_KEY_MAP = {
    "budget_gpu_hours": "gpu_hours_total",
    "max_concurrent": "max_concurrent_children",
    "per_child_gpu_hours_cap": "per_child_gpu_hours_cap",
}

_REQUIRED_TABLE_COLUMNS = ("id", "title", "hypothesis", "depends_on", "gpu_hours_est")

_ANCHOR_RE = re.compile(r"\bq:[a-z0-9][a-z0-9_-]*\b")
_BRIEF_H2_RE = re.compile(r"^##\s+Campaign Brief\s*$", re.MULTILINE)
_NO_DEPS_TOKENS = frozenset({"", "-", "none"})


# ─── Path + load/save ───────────────────────────────────────────────────────


def state_path(task_id: int) -> Path:
    """Absolute path of the campaign state file for task ``task_id``.

    Resolved via ``find_task_path`` so the path is correct regardless of the
    task's current status folder and regardless of which worktree the caller
    runs from. Raises ``FileNotFoundError`` if the task does not exist."""
    return find_task_path(task_id) / "artifacts" / STATE_FILENAME


def load_state(task_id: int) -> dict[str, Any]:
    """Read + validate the campaign state for task ``task_id``.

    Raises ``FileNotFoundError`` if the state file does not exist (callers
    initialize via :func:`init_state_from_brief`), ``ValueError`` on a
    schema violation. Never returns a placeholder — fail loud."""
    path = state_path(task_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"campaign state missing for task #{task_id} at {path}; "
            f"initialize it via campaign_state.init_state_from_brief"
        )
    state = json.loads(path.read_text())
    _validate_state(state, task_id)
    return state


def save_state(task_id: int, state: dict[str, Any]) -> None:
    """Validate + atomically persist ``state`` for task ``task_id``.

    Validation runs BEFORE any write, so a schema violation never clobbers
    the existing on-disk state. The write is temp-file + ``os.replace`` so
    concurrent readers never observe a partial JSON document."""
    _validate_state(state, task_id)
    path = state_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _require_numbers(mapping: Any, name: str, keys: tuple[str, ...]) -> None:
    """Raise ``ValueError`` unless ``mapping`` is a dict whose ``keys`` are
    all numbers. Shared by the budget / limits validators."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a mapping")
    for key in keys:
        if not isinstance(mapping.get(key), int | float):
            raise ValueError(f"{name}.{key} must be a number, got {mapping.get(key)!r}")


def _validate_state(state: dict[str, Any], task_id: int) -> None:
    """Raise ``ValueError`` on any schema violation. Checks the load-bearing
    invariants (version, task binding, budget/limit shapes, experiment
    statuses + dependency references); permissive on extra keys."""
    if not isinstance(state, dict):
        raise ValueError(f"campaign state must be a mapping, got {type(state).__name__}")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"campaign state schema_version {state.get('schema_version')!r} != {SCHEMA_VERSION}"
        )
    if state.get("campaign_task") != task_id:
        raise ValueError(
            f"campaign state is bound to task {state.get('campaign_task')!r}, "
            f"caller asked for #{task_id}"
        )
    anchor = state.get("question_anchor")
    if not isinstance(anchor, str) or not _ANCHOR_RE.fullmatch(anchor):
        raise ValueError(f"question_anchor {anchor!r} is not a q:<slug> anchor")
    _require_numbers(state.get("budget"), "budget", ("gpu_hours_total", "gpu_hours_committed"))
    _require_numbers(
        state.get("limits"),
        "limits",
        ("max_experiments", "max_concurrent_children", "per_child_gpu_hours_cap"),
    )
    stop = state.get("stop")
    if not isinstance(stop, dict):
        raise ValueError("stop must be a mapping")
    for key in ("dry_counter", "dry_limit"):
        if not isinstance(stop.get(key), int):
            raise ValueError(f"stop.{key} must be an int, got {stop.get(key)!r}")
    current = stop.get("current_confidence")
    if current is not None and (
        not isinstance(current, str) or current.upper() not in _CONFIDENCE_RANK
    ):
        raise ValueError(
            f"stop.current_confidence must be null or one of {sorted(_CONFIDENCE_RANK)}, "
            f"got {current!r}"
        )
    _validate_experiments(state.get("experiments"))


def _validate_experiments(experiments: Any) -> None:
    """Validate the experiment DAG rows: unique non-empty ids, known
    statuses, numeric estimates, list-of-str deps that all resolve."""
    if not isinstance(experiments, list):
        raise ValueError("experiments must be a list")
    ids: set[str] = set()
    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError(f"experiment row must be a mapping, got {exp!r}")
        exp_id = exp.get("id")
        if not isinstance(exp_id, str) or not exp_id:
            raise ValueError(f"experiment id must be a non-empty string, got {exp_id!r}")
        if exp_id in ids:
            raise ValueError(f"duplicate experiment id {exp_id!r}")
        ids.add(exp_id)
        if exp.get("status") not in EXPERIMENT_STATUSES:
            raise ValueError(
                f"experiment {exp_id!r} status {exp.get('status')!r} not in {EXPERIMENT_STATUSES}"
            )
        if not isinstance(exp.get("gpu_hours_est"), int | float):
            raise ValueError(f"experiment {exp_id!r} gpu_hours_est must be a number")
        deps = exp.get("depends_on")
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise ValueError(f"experiment {exp_id!r} depends_on must be a list of ids")
    for exp in experiments:
        for dep in exp["depends_on"]:
            if dep not in ids:
                raise ValueError(f"experiment {exp['id']!r} depends on unknown id {dep!r}")


# ─── Brief parsing / initialization ─────────────────────────────────────────


def init_state_from_brief(
    task_id: int,
    frontmatter: dict[str, Any],
    body: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build, persist, and return a fresh state from the ``## Campaign Brief``.

    Parses the brief section of ``body`` (question anchor + initial
    experiment table — see the module docstring for the exact format) and
    the optional frontmatter ``campaign:`` overrides mapping. Budget/limit
    values follow the fixed precedence: frontmatter ``campaign:`` overrides
    > registry caps from ``campaign-<N>.json`` (spawn-campaign CLI flags)
    > module defaults. Fails loud on a missing brief / anchor / table, an
    unknown override key, an unknown ``depends_on`` reference, or a
    dependency cycle."""
    now = now if now is not None else datetime.now(tz=UTC)
    brief = _extract_brief_section(body)
    anchor_match = _ANCHOR_RE.search(brief)
    if anchor_match is None:
        raise ValueError(
            "## Campaign Brief has no q:<slug> question anchor "
            "(an anchor into docs/open_questions.md is required)"
        )
    experiments = _parse_experiment_table(brief)
    _assert_acyclic(experiments)
    # Precedence: frontmatter overrides win over registry caps win over
    # the module defaults consumed by the .get(...) fallbacks below.
    overrides = {**_registry_caps(task_id), **_parse_overrides(frontmatter)}

    gpu_hours_total = float(overrides.get("gpu_hours_total", DEFAULT_GPU_HOURS_TOTAL))
    wall_clock_days = float(overrides.get("wall_clock_days", DEFAULT_WALL_CLOCK_DAYS))
    deadline = now + timedelta(days=wall_clock_days)
    iso = "%Y-%m-%dT%H:%M:%SZ"
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_task": task_id,
        "question_anchor": anchor_match.group(0),
        "started_at": now.strftime(iso),
        "wall_clock_deadline": deadline.strftime(iso),
        "budget": {"gpu_hours_total": gpu_hours_total, "gpu_hours_committed": 0.0},
        "limits": {
            "max_experiments": int(overrides.get("max_experiments", DEFAULT_MAX_EXPERIMENTS)),
            "max_concurrent_children": int(
                overrides.get("max_concurrent_children", DEFAULT_MAX_CONCURRENT_CHILDREN)
            ),
            "per_child_gpu_hours_cap": float(
                overrides.get("per_child_gpu_hours_cap", DEFAULT_PER_CHILD_GPU_HOURS_CAP)
            ),
            "wall_clock_days": wall_clock_days,
        },
        "stop": {
            "stopped": False,
            "stop_reason": None,
            "confidence_target": str(
                overrides.get("confidence_target", DEFAULT_CONFIDENCE_TARGET)
            ).upper(),
            # Campaign-level working belief; the skill sets it at each ingest.
            "current_confidence": None,
            "dry_counter": 0,
            "dry_limit": int(overrides.get("dry_limit", DEFAULT_DRY_LIMIT)),
        },
        "experiments": experiments,
        "last_digest_at": None,
    }
    save_state(task_id, state)
    return state


def _extract_brief_section(body: str) -> str:
    """Return the text between ``## Campaign Brief`` and the next H2 (or EOF).
    Raises ``ValueError`` when the H2 is absent."""
    match = _BRIEF_H2_RE.search(body)
    if match is None:
        raise ValueError(
            "task body has no `## Campaign Brief` H2 — a campaign cannot start "
            "without a user-reviewed brief (gate: campaign_brief_approval)"
        )
    rest = body[match.end() :]
    next_h2 = re.search(r"^##\s+", rest, re.MULTILINE)
    return rest[: next_h2.start()] if next_h2 else rest


def _find_table_header(lines: list[str]) -> tuple[int, dict[str, int]]:
    """Locate the first table header row carrying every required column.
    Returns ``(line_index, {column: cell_index})``; raises when absent."""
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip().lower() for c in _split_table_row(line)]
        if all(col in cells for col in _REQUIRED_TABLE_COLUMNS):
            return i, {col: cells.index(col) for col in _REQUIRED_TABLE_COLUMNS}
    raise ValueError(
        "## Campaign Brief has no experiment table with columns "
        f"{' | '.join(_REQUIRED_TABLE_COLUMNS)}"
    )


def _parse_table_row(line: str, col_index: dict[str, int]) -> dict[str, Any] | None:
    """One table line -> an experiment row dict (status ``planned``), or
    ``None`` for the ``|---|---|`` separator row. Malformed rows fail loud."""
    stripped = line.strip()
    cells = _split_table_row(line)
    if not any(c.strip() for c in cells):
        # An all-empty `| | |` row is a malformed table line, NOT a
        # separator — fail loud, consistent with the rest of the parser.
        raise ValueError(f"experiment table row is empty: {stripped!r}")
    if all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip()):
        return None  # separator row
    if len(cells) <= max(col_index.values()):
        raise ValueError(f"experiment table row has too few cells: {stripped!r}")
    exp_id = cells[col_index["id"]].strip()
    if not exp_id:
        raise ValueError(f"experiment table row has an empty id: {stripped!r}")
    deps = [
        d.strip()
        for d in cells[col_index["depends_on"]].split(",")
        if d.strip().lower() not in _NO_DEPS_TOKENS
    ]
    raw_hours = cells[col_index["gpu_hours_est"]].strip()
    try:
        gpu_hours_est = float(raw_hours)
    except ValueError as e:
        raise ValueError(
            f"experiment {exp_id!r}: gpu_hours_est {raw_hours!r} is not a number"
        ) from e
    return {
        "id": exp_id,
        "title": cells[col_index["title"]].strip(),
        "hypothesis": cells[col_index["hypothesis"]].strip(),
        "depends_on": deps,
        "gpu_hours_est": gpu_hours_est,
        "status": "planned",
        "child_task": None,
        "headline": None,
        "confidence": None,
        "belief_shift": None,
    }


def _parse_experiment_table(brief: str) -> list[dict[str, Any]]:
    """Parse the first markdown table in ``brief`` carrying the required
    columns into a list of experiment rows (status ``planned``). Fails loud
    when no such table exists or a row is malformed."""
    lines = brief.splitlines()
    header_idx, col_index = _find_table_header(lines)
    experiments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in lines[header_idx + 1 :]:
        if not line.strip().startswith("|"):
            break  # end of table
        row = _parse_table_row(line, col_index)
        if row is None:
            continue
        if row["id"] in seen:
            raise ValueError(f"duplicate experiment id {row['id']!r} in brief table")
        seen.add(row["id"])
        experiments.append(row)
    if not experiments:
        raise ValueError("## Campaign Brief experiment table has a header but no rows")
    for exp in experiments:
        for dep in exp["depends_on"]:
            if dep not in seen:
                raise ValueError(f"experiment {exp['id']!r} depends on unknown id {dep!r}")
    return experiments


def _split_table_row(line: str) -> list[str]:
    """Split a markdown ``| a | b |`` row into cell strings (outer pipes
    dropped). Inner empty cells are preserved positionally."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return stripped.split("|")


def _assert_acyclic(experiments: list[dict[str, Any]]) -> None:
    """Kahn's algorithm: raise ``ValueError`` when the depends_on graph has a
    cycle (a cyclic DAG row would simply never become ready — fail loud at
    init instead of wedging the campaign silently)."""
    indegree = {e["id"]: len(e["depends_on"]) for e in experiments}
    dependents: dict[str, list[str]] = {e["id"]: [] for e in experiments}
    for exp in experiments:
        for dep in exp["depends_on"]:
            dependents[dep].append(exp["id"])
    queue = [eid for eid, deg in indegree.items() if deg == 0]
    visited = 0
    while queue:
        eid = queue.pop()
        visited += 1
        for child in dependents[eid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(experiments):
        cyclic = sorted(eid for eid, deg in indegree.items() if deg > 0)
        raise ValueError(f"experiment depends_on graph has a cycle involving {cyclic}")


def _registry_caps(task_id: int) -> dict[str, Any]:
    """Caps recorded by ``spawn_session.py spawn-campaign`` in
    ``~/.eps-autonomous/campaign-<N>.json``, mapped to override-key names
    (:data:`_REGISTRY_CAP_KEY_MAP`). Precedence tier 2 — below frontmatter
    overrides, above module defaults. A missing / unreadable entry returns
    ``{}`` (fail-soft: the registry is an optional caps SOURCE, not state —
    the lower-precedence defaults then apply); non-numeric values are
    ignored rather than trusted."""
    path = AUTONOMOUS_REGISTRY_DIR / f"campaign-{task_id}.json"
    try:
        entry = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(entry, dict):
        return {}
    out: dict[str, Any] = {}
    for reg_key, override_key in _REGISTRY_CAP_KEY_MAP.items():
        value = entry.get(reg_key)
        if isinstance(value, int | float):
            out[override_key] = value
    return out


def _parse_overrides(frontmatter: dict[str, Any]) -> dict[str, Any]:
    """Validate + return the optional frontmatter ``campaign:`` mapping.
    Unknown keys and non-numeric values for numeric keys fail loud."""
    raw = frontmatter.get("campaign")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"frontmatter `campaign:` must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"unknown campaign override key(s) {sorted(unknown)}; allowed: {sorted(OVERRIDE_KEYS)}"
        )
    for key, value in raw.items():
        if key == "confidence_target":
            if str(value).upper() not in _CONFIDENCE_RANK:
                raise ValueError(
                    f"campaign.confidence_target {value!r} not in {sorted(_CONFIDENCE_RANK)}"
                )
        elif not isinstance(value, int | float):
            raise ValueError(f"campaign.{key} must be a number, got {value!r}")
    return dict(raw)


# ─── Scheduling reads ───────────────────────────────────────────────────────


def ready_experiments(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Experiment rows that can be filed NOW: status ``planned`` AND every
    ``depends_on`` id already ``ingested``. (A dependency that was abandoned
    keeps its dependents un-ready — the proposal round must re-plan them.)"""
    ingested = {e["id"] for e in state["experiments"] if e["status"] == "ingested"}
    return [
        e
        for e in state["experiments"]
        if e["status"] == "planned" and all(dep in ingested for dep in e["depends_on"])
    ]


def open_slots(state: dict[str, Any]) -> int:
    """Free concurrency slots: ``max_concurrent_children`` minus children in
    flight (status filed / running / landed). Never negative."""
    in_flight = sum(1 for e in state["experiments"] if e["status"] in CONCURRENCY_STATUSES)
    return max(0, int(state["limits"]["max_concurrent_children"]) - in_flight)


def budget_headroom(state: dict[str, Any]) -> float:
    """GPU-hours still available to commit: total minus committed. May be
    negative when the backstop has been breached (the watcher alerts)."""
    budget = state["budget"]
    return float(budget["gpu_hours_total"]) - float(budget["gpu_hours_committed"])


# ─── Stop criteria ──────────────────────────────────────────────────────────


def check_stop(
    state: dict[str, Any],
    now: datetime | None = None,
    *,
    user_stop: bool = False,
) -> tuple[bool, str | None]:
    """Evaluate the campaign stop criteria. Returns ``(should_stop, reason)``.

    Fixed evaluation order (plan #586): user-stop tag → wall-clock deadline →
    budget committed >= total → experiments finished (ingested + abandoned)
    >= max_experiments → confidence target met → dry counter >= dry limit.

    The confidence-target criterion compares the CAMPAIGN-LEVEL working
    belief ``stop.current_confidence`` (set by the /campaign skill at each
    ingest from the world model; null until then) to
    ``stop.confidence_target``. Per-child clean-result confidence tags are
    PER-CLAIM, not per-question — a single HIGH child never trips the stop
    while ``current_confidence`` is null or below target.

    ``user_stop`` is passed by the caller (the ``/campaign`` skill or the
    watcher) after checking the task's tags — keeping this function pure
    (no task.py reads) so the ordering is unit-testable. A state already
    marked ``stopped`` short-circuits with its recorded reason."""
    now = now if now is not None else datetime.now(tz=UTC)
    stop = state["stop"]
    if stop.get("stopped"):
        return True, stop.get("stop_reason") or "already stopped"
    if user_stop:
        return True, "user-stop tag set on the campaign task"
    deadline = datetime.fromisoformat(str(state["wall_clock_deadline"]).replace("Z", "+00:00"))
    if now >= deadline:
        return True, f"wall-clock deadline reached ({state['wall_clock_deadline']})"
    budget = state["budget"]
    if float(budget["gpu_hours_committed"]) >= float(budget["gpu_hours_total"]):
        return True, (
            f"GPU-hour budget exhausted "
            f"({budget['gpu_hours_committed']:g} >= {budget['gpu_hours_total']:g} committed)"
        )
    finished = sum(1 for e in state["experiments"] if e["status"] in FINISHED_STATUSES)
    max_experiments = int(state["limits"]["max_experiments"])
    if finished >= max_experiments:
        return True, f"max experiments reached ({finished} >= {max_experiments})"
    target_rank = _CONFIDENCE_RANK.get(str(stop.get("confidence_target", "")).upper(), -1)
    current = stop.get("current_confidence")
    if (
        target_rank >= 0
        and current is not None
        and _CONFIDENCE_RANK.get(str(current).upper(), -1) >= target_rank
    ):
        return True, (
            f"confidence target met (campaign working belief at "
            f"{str(current).upper()} >= {stop['confidence_target']})"
        )
    if int(stop["dry_counter"]) >= int(stop["dry_limit"]):
        return True, (
            f"dry counter reached ({stop['dry_counter']} >= {stop['dry_limit']} "
            f"consecutive non-belief-shifting results)"
        )
    return False, None
