"""task_state.py — sagan_state.py compatibility shim backed by task.py.

Lets existing scripts that did `import sagan_state` continue to work by
swapping to `import task_state as sagan_state`. The function signatures
and return shapes match sagan_state.py closely enough that the callers
(scripts/post_step_completed.py, scripts/pod_watch.py,
scripts/recent_clean_results.py) work unchanged.

Translation rules:

* sagan experiment_id (UUID) → task number (int). The shim accepts both
  and resolves to a number. `get_experiment` returns the experiment dict
  with `id` == the task NUMBER (as a string), not a UUID.
* sagan_state.BASE_URL → the EPS dashboard URL.
* SaganError → re-raised as `TaskStateError` (subclass of RuntimeError).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Importable module path
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space import task_workflow as tw  # noqa: E402

BASE_URL = "https://eps.superkaiba.com"


class TaskStateError(RuntimeError):
    """Raised on shim-level errors. Compatible with sagan_state.SaganError."""


SaganError = TaskStateError  # alias for drop-in compat


def _coerce_id(experiment_id: Any) -> int:
    """Accept either int task number or stringified number. Reject UUIDs."""
    if isinstance(experiment_id, int):
        return experiment_id
    s = str(experiment_id).strip()
    if s.isdigit():
        return int(s)
    raise TaskStateError(
        f"task_state shim received a non-numeric id {experiment_id!r}; "
        f"the shim works on task numbers, not Sagan UUIDs."
    )


def get_experiment(number: int) -> dict[str, Any]:
    """Return experiment + events in the sagan_state.get_experiment shape."""
    task = tw.get_task(number)
    if task is None:
        raise TaskStateError(f"task #{number} not found")
    fm = task["frontmatter"]
    events = tw.list_events(number)
    # sagan_state's get_experiment returns:
    #   {"experiment": {...}, "events": [...], "approvalRequests": [...]}
    return {
        "experiment": {
            "id": str(number),
            "number": number,
            "title": fm.get("title", ""),
            "kind": fm.get("kind", "experiment"),
            "tags": fm.get("tags") or [],
            "status": task["status"],
            "body": task["body"],
            "hasCleanResult": bool(fm.get("has_clean_result")),
            "has_clean_result": bool(fm.get("has_clean_result")),
            "classification": fm.get("classification"),
            "updatedAt": events[-1]["ts"] if events else fm.get("created_at", ""),
            "updated_at": events[-1]["ts"] if events else fm.get("created_at", ""),
            "createdAt": fm.get("created_at", ""),
            "created_at": fm.get("created_at", ""),
        },
        "events": [
            {
                "createdAt": ev["ts"],
                "created_at": ev["ts"],
                "eventType": "marker" if ev["kind"].startswith("epm:") else ev["kind"],
                "markerType": ev["kind"],
                "metadata": {"marker_type": ev["kind"], "version": ev.get("version", 1)},
                "note": ev.get("note"),
                "fromStatus": ev.get("from"),
                "toStatus": ev.get("to"),
            }
            for ev in events
        ],
        "approvalRequests": [],
    }


def get_experiment_by_id(experiment_id: Any) -> dict[str, Any]:
    return get_experiment(_coerce_id(experiment_id))


def list_by_status(status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Return tasks in the sagan_state.list_by_status shape."""
    if status is not None:
        rows = tw.list_by_status(status, limit=limit)
    else:
        rows = []
        for s in tw.STATUSES:
            rows.extend(tw.list_by_status(s, limit=limit))
            if len(rows) >= limit:
                rows = rows[:limit]
                break
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": str(r["id"]),
                "number": r["id"],
                "title": r["title"],
                "kind": r["kind"],
                "tags": r.get("tags") or [],
                "status": r["status"],
                "hasCleanResult": r.get("has_clean_result", False),
                "has_clean_result": r.get("has_clean_result", False),
                "classification": r.get("classification"),
            }
        )
    return out


def latest_marker(experiment_id: Any) -> dict[str, Any] | None:
    """Most recent epm:* event on a task, sagan_state shape."""
    n = _coerce_id(experiment_id)
    ev = tw.latest_event(n, prefix="epm:")
    if ev is None:
        return None
    return {
        "createdAt": ev["ts"],
        "markerType": ev["kind"],
        "metadata": {"marker_type": ev["kind"], "version": ev.get("version", 1)},
        "note": ev.get("note"),
    }


def patch_experiment(experiment_id: Any, **fields: Any) -> dict[str, Any]:
    """Apply a subset of patch fields. Returns the updated experiment dict."""
    n = _coerce_id(experiment_id)
    if "title" in fields and fields["title"] is not None:
        tw.set_title(n, fields["title"])
    if "tags" in fields and fields["tags"] is not None:
        # Replace tags: easiest path = read existing, remove all, add new
        task = tw.get_task(n)
        cur_tags: list[str] = list(task["frontmatter"].get("tags") or [])
        for old in cur_tags:
            if old not in fields["tags"]:
                tw.remove_tag(n, old)
        for new in fields["tags"]:
            if new not in cur_tags:
                tw.add_tag(n, new)
    if "hasCleanResult" in fields or "has_clean_result" in fields:
        v = fields.get("hasCleanResult", fields.get("has_clean_result"))
        if v is not None:
            tw.set_clean_result(n, value=bool(v))
    if "body" in fields and fields["body"] is not None:
        tw.set_body(n, fields["body"])
    if "status" in fields and fields["status"] is not None:
        tw.set_status(n, fields["status"], note=fields.get("note"))
    return get_experiment(n)


def set_status(experiment_id: Any, status: str, *, note: str | None = None) -> dict[str, Any]:
    n = _coerce_id(experiment_id)
    tw.set_status(n, status, note=note)
    return get_experiment(n)


def set_tags(experiment_id: Any, tags: list[str]) -> dict[str, Any]:
    return patch_experiment(experiment_id, tags=tags)


def add_tag(experiment_id: Any, tag: str) -> dict[str, Any]:
    n = _coerce_id(experiment_id)
    tw.add_tag(n, tag)
    return get_experiment(n)


def remove_tag(experiment_id: Any, tag: str) -> dict[str, Any]:
    n = _coerce_id(experiment_id)
    tw.remove_tag(n, tag)
    return get_experiment(n)


def set_clean_result(experiment_id: Any, value: bool) -> dict[str, Any]:
    n = _coerce_id(experiment_id)
    tw.set_clean_result(n, value=value)
    return get_experiment(n)


def post_marker(
    experiment_id: Any,
    marker: str,
    *,
    note: str | None = None,
    metadata: dict[str, Any] | None = None,
    from_status: str | None = None,
    to_status: str | None = None,
    event_type: str = "note",
) -> dict[str, Any]:
    """Append an epm:* event. Mirrors sagan_state.post_marker."""
    if not marker.startswith("epm:"):
        raise TaskStateError(f"marker must start with 'epm:' (got: {marker})")
    n = _coerce_id(experiment_id)
    extras: dict[str, Any] = {}
    if metadata:
        extras.update(metadata)
    if from_status:
        extras["from"] = from_status
    if to_status:
        extras["to"] = to_status
    tw.post_event(n, marker, by="task_state shim", note=note, **extras)
    return {"ok": True}


def list_markers(experiment_id: Any, *, prefix: str = "epm:") -> list[dict[str, Any]]:
    n = _coerce_id(experiment_id)
    events = tw.list_events(n)
    return [
        {
            "createdAt": ev["ts"],
            "markerType": ev["kind"],
            "metadata": {"marker_type": ev["kind"], "version": ev.get("version", 1)},
            "note": ev.get("note"),
        }
        for ev in events
        if ev["kind"].startswith(prefix)
    ]


def has_marker(experiment_id: Any, marker_kind: str) -> bool:
    n = _coerce_id(experiment_id)
    target = f"epm:{marker_kind}" if not marker_kind.startswith("epm:") else marker_kind
    return tw.has_event(n, target)


def create_experiment(
    *,
    title: str,
    body: str = "",
    status: str = "proposed",
    kind: str = "experiment",
    **_ignored: Any,
) -> dict[str, Any]:
    req = tw.NewTaskRequest(kind=kind, title=title, body=body, status=status)
    new_id = tw.create_task(req)
    return get_experiment(new_id)
