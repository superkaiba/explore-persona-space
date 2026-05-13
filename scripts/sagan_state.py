"""Sagan-backed state helpers for the /issue skill.

This module is the active state surface for `/issue`. It speaks to the Sagan
dashboard's HTTP API so agents and local scripts can operate without GitHub
issue or project-board state.

Primary endpoints:

    GET   /api/experiments/by-number/<N>
    PATCH /api/experiments/<uuid>
    POST  /api/experiments/<uuid>/workflow-events
    GET   /api/experiments?status=…

Configuration (env vars, read once on import):

    SAGAN_BASE_URL    Base URL, e.g. https://sagan.superkaiba.com  (required)
    SAGAN_API_TOKEN   Session token (mobile-style Bearer). 60-day sliding.
                      Mint with the /api/auth/login flow on the VM once.
                      (required for any write; reads also require it)

Usage from Python:

    from scripts.sagan_state import (
        get_experiment, set_status, post_marker, set_tags,
        latest_marker, list_by_status,
    )

    exp = get_experiment(311)
    print(exp["status"], exp["title"])
    set_status(exp["id"], "running")
    post_marker(exp["id"], "epm:reviewer-verdict",
                note="...marker body...",
                metadata={"verdict": "PASS"})

The CLI exposes the `/issue` state transitions directly against Sagan.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ─── Config ─────────────────────────────────────────────────────────────────

# Load .env so SAGAN_API_TOKEN / SAGAN_BASE_URL can live in the project's
# .env file (alongside HF_TOKEN, WANDB_API_KEY, etc.) instead of a shell
# rc or ~/.eps-secrets. `override=False` preserves any value already set
# in the process env (CI overrides, manual `export …` for debugging).
try:
    from dotenv import load_dotenv

    load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"), override=False)
except ImportError:
    pass

BASE_URL = os.environ.get("SAGAN_BASE_URL", "https://sagan.superkaiba.com").rstrip("/")
API_TOKEN = os.environ.get("SAGAN_API_TOKEN", "").strip()


class SaganError(Exception):
    """Raised on non-2xx responses from Sagan."""


# Kebab→snake status translation. Wave 2 of the migration normalises
# both repos to snake_case; this fallback keeps in-flight kebab callers
# working during the transition. Remove after one-week soak (see
# ~/.claude/plans/do-a-comprehensive-audit-merry-dawn.md Wave 2).
_KEBAB_STATUS_WARNED: set[str] = set()


def _translate_status(value: str | None) -> str | None:
    if value is None or "-" not in value:
        return value
    snake = value.replace("-", "_")
    if value not in _KEBAB_STATUS_WARNED:
        _KEBAB_STATUS_WARNED.add(value)
        print(
            f"sagan_state: deprecated kebab status {value!r} → {snake!r}; "
            "update the caller (Wave 2 of the Sagan migration removes this shim).",
            file=sys.stderr,
        )
    return snake


def _req(
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make an authenticated request to the Sagan API and return parsed JSON."""

    if not API_TOKEN:
        raise SaganError(
            "SAGAN_API_TOKEN is not set. Mint a session token on the VM "
            "(POST /api/auth/login with owner credentials) and export it."
        )
    url = f"{BASE_URL}{path}"
    if query:
        from urllib.parse import urlencode

        url += "?" + urlencode({k: v for k, v in query.items() if v is not None})

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {API_TOKEN}")
    if body is not None:
        req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SaganError(f"{method} {path} → {e.code}: {detail[:500]}") from e
    except urllib.error.URLError as e:
        raise SaganError(f"{method} {path} → {e.reason}") from e

    if not raw:
        return {}
    return json.loads(raw)


# ─── Read helpers ───────────────────────────────────────────────────────────


def get_experiment(number: int) -> dict[str, Any]:
    """Return the experiment row, recent workflow events, and approval requests.

    Return the same workflow payload used by `/issue`. The response shape is:
        {"experiment": {...}, "events": [...], "approvalRequests": [...]}
    """
    return _req("GET", f"/api/experiments/by-number/{number}")


def get_experiment_by_id(experiment_id: str) -> dict[str, Any]:
    return _req("GET", f"/api/experiments/{experiment_id}")


def list_by_status(status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """List experiments, optionally filtered by status. Returns the inner list."""
    out = _req(
        "GET", "/api/experiments", query={"status": _translate_status(status), "limit": limit}
    )
    return out.get("experiments", [])


def latest_marker(experiment_id: str) -> dict[str, Any] | None:
    """Return the most recent `epm:*` workflow event (or None if there is none).

    Used by the /issue skill to decide where to resume in the pipeline.
    """
    exp = get_experiment_by_id(experiment_id)
    for ev in exp.get("events", []):
        meta = ev.get("metadata") or {}
        marker = meta.get("marker_type")
        if marker and marker.startswith("epm:"):
            return ev
    return None


# ─── Write helpers ──────────────────────────────────────────────────────────


def patch_experiment(experiment_id: str, **fields: Any) -> dict[str, Any]:
    """Update experiment fields. Accepted keys mirror the PATCH route schema:
    title, body, hypothesis, configYaml, status, kind, computeSize, priority,
    assigneeKind, tags, hasCleanResult, runpodAccount, note.
    """
    return _req("PATCH", f"/api/experiments/{experiment_id}", body=fields)


def set_status(experiment_id: str, status: str, *, note: str | None = None) -> dict[str, Any]:
    """Change status. Also records a workflow event automatically (server-side)."""
    return patch_experiment(experiment_id, status=_translate_status(status), note=note)


def set_tags(experiment_id: str, tags: list[str]) -> dict[str, Any]:
    """Replace the full tag list (use add_tag / remove_tag for partial updates)."""
    return patch_experiment(experiment_id, tags=tags)


def add_tag(experiment_id: str, tag: str) -> dict[str, Any]:
    exp = get_experiment_by_id(experiment_id)["experiment"]
    tags = list(exp.get("tags") or [])
    if tag not in tags:
        tags.append(tag)
    return set_tags(experiment_id, tags)


def remove_tag(experiment_id: str, tag: str) -> dict[str, Any]:
    exp = get_experiment_by_id(experiment_id)["experiment"]
    tags = [t for t in (exp.get("tags") or []) if t != tag]
    return set_tags(experiment_id, tags)


def set_clean_result(experiment_id: str, value: bool) -> dict[str, Any]:
    return patch_experiment(experiment_id, hasCleanResult=value)


def post_marker(
    experiment_id: str,
    marker: str,
    *,
    note: str | None = None,
    metadata: dict[str, Any] | None = None,
    from_status: str | None = None,
    to_status: str | None = None,
    event_type: str = "note",
) -> dict[str, Any]:
    """Append a workflow event with a structured marker_type.

    Post a Sagan workflow_event containing an `epm:*` marker body.
    """
    if not marker.startswith("epm:"):
        raise SaganError(f"marker must start with 'epm:' (got: {marker})")
    return _req(
        "POST",
        f"/api/experiments/{experiment_id}/workflow-events",
        body={
            "eventType": event_type,
            "markerType": marker,
            "note": note,
            "metadata": metadata or {},
            "fromStatus": from_status,
            "toStatus": to_status,
        },
    )


def list_markers(experiment_id: str, *, prefix: str = "epm:") -> list[dict[str, Any]]:
    """Return all workflow events whose ``marker_type`` starts with ``prefix``."""
    exp = get_experiment_by_id(experiment_id)
    out: list[dict[str, Any]] = []
    for ev in exp.get("events", []):
        meta = ev.get("metadata") or {}
        marker = meta.get("marker_type") or ev.get("markerType")
        if marker and marker.startswith(prefix):
            out.append(ev)
    return out


def has_marker(experiment_id: str, marker_kind: str) -> bool:
    """True if any workflow event carries an ``epm:<marker_kind>`` marker."""
    target = f"epm:{marker_kind}" if not marker_kind.startswith("epm:") else marker_kind
    return any(
        (ev.get("metadata") or {}).get("marker_type", "").startswith(target)
        for ev in list_markers(experiment_id)
    )


def create_experiment(
    *,
    title: str,
    hypothesis: str | None = None,
    status: str = "proposed",
    project_id: str | None = None,
    belief_id: str | None = None,
    runpod_account: str = "team",
) -> dict[str, Any]:
    """Create a new experiment row in Sagan.

    Returns the inserted experiment dict (server response). The ``number``
    is assigned server-side and is what /issue <N> takes as input.
    """
    body: dict[str, Any] = {
        "title": title,
        "status": _translate_status(status),
        "runpodAccount": runpod_account,
    }
    if hypothesis is not None:
        body["hypothesis"] = hypothesis
    if project_id is not None:
        body["projectId"] = project_id
    if belief_id is not None:
        body["beliefId"] = belief_id
    return _req("POST", "/api/experiments", body=body)


# ─── CLI ────────────────────────────────────────────────────────────────────


def cmd_view(args: argparse.Namespace) -> None:
    data = get_experiment(args.number)
    json.dump(data, sys.stdout, indent=2, default=str)
    print()


def cmd_set_status(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    res = set_status(exp["id"], args.status, note=args.note)
    print(f"#{args.number} status → {res['experiment']['status']}")


def cmd_list_by_status(args: argparse.Namespace) -> None:
    rows = list_by_status(status=args.status, limit=args.limit)
    for row in rows:
        print(f"  #{row.get('number', '?'):>4}  {row['status']:<20}  {row['title'][:80]}")


def cmd_post_marker(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    res = post_marker(exp["id"], args.marker, note=args.note)
    print(f"#{args.number} ← {args.marker}  (event {res['id']})")


def cmd_add_tag(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    add_tag(exp["id"], args.tag)
    print(f"#{args.number} +tag {args.tag}")


def cmd_remove_tag(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    remove_tag(exp["id"], args.tag)
    print(f"#{args.number} -tag {args.tag}")


def cmd_set_body(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    if args.body is not None:
        body = args.body
    else:
        with open(args.file) as fh:
            body = fh.read()
    patch_experiment(exp["id"], body=body)
    print(f"#{args.number} body updated ({len(body)} chars)")


def cmd_set_title(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    patch_experiment(exp["id"], title=args.title)
    print(f"#{args.number} title → {args.title[:80]}")


def cmd_set_clean_result(args: argparse.Namespace) -> None:
    """Flip `hasCleanResult=true` on an experiment. Sagan auto-creates a
    pending runs row on the same PATCH (idempotent: re-running is safe).

    Used by the analyzer in its in-place clean-result promotion (Step 6):
    after `set-body` + `set-title` have replaced the source experiment's
    body with the polished write-up, this call marks it as a clean-result
    and primes the pending run that `promote` will later flip.
    """
    exp = get_experiment(args.number)["experiment"]
    set_clean_result(exp["id"], True)
    print(f"#{args.number} hasCleanResult=true (pending run row ensured)")


def cmd_promote(args: argparse.Namespace) -> None:
    """Clean-result promotion: atomically flip runs.classification + set
    has_clean_result=true + advance status to completed.

    Calls the dedicated `/api/experiments/:id/promote` endpoint so the
    runs / experiments updates land in a single transaction. The
    server-side check rejects promotion if no pending runs row exists
    (the analyzer must have created one when posting the clean-result).
    """
    if args.verdict not in ("useful", "not-useful", "not_useful"):
        raise SaganError("verdict must be 'useful' or 'not-useful'")
    classification = "useful" if args.verdict == "useful" else "not_useful"
    exp = get_experiment(args.number)["experiment"]
    result = _req(
        "POST",
        f"/api/experiments/{exp['id']}/promote",
        body={"verdict": classification},
    )
    run_id = result.get("run", {}).get("id", "?")
    print(f"#{args.number} promoted ({classification})  run={run_id}  status=completed")


def cmd_list_markers(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    events = list_markers(exp["id"], prefix=args.prefix)
    if args.json:
        json.dump(events, sys.stdout, indent=2, default=str)
        print()
        return
    if not events:
        print("(no markers)")
        return
    for ev in events:
        meta = ev.get("metadata") or {}
        marker = meta.get("marker_type") or ev.get("markerType") or "?"
        print(f"  {ev['createdAt']}  {marker:<28}  event={ev['id']}")


def cmd_create_experiment(args: argparse.Namespace) -> None:
    body = args.body
    if body is None and args.body_file:
        with open(args.body_file) as fh:
            body = fh.read()
    res = create_experiment(
        title=args.title,
        hypothesis=body,
        status=args.status,
        runpod_account=args.runpod_account,
    )
    exp = res.get("experiment") or res
    number = exp.get("number", "?")
    uuid = exp.get("id", "?")
    print(f"created experiment #{number}  id={uuid}  status={exp.get('status', '?')}")


def cmd_latest_marker(args: argparse.Namespace) -> None:
    exp = get_experiment(args.number)["experiment"]
    ev = latest_marker(exp["id"])
    if ev is None:
        print("(no markers)")
        return
    meta = ev.get("metadata") or {}
    print(f"  marker:    {meta.get('marker_type')}")
    print(f"  at:        {ev['createdAt']}")
    print(f"  event_id:  {ev['id']}")
    if ev.get("note"):
        print("  note:")
        for line in (ev["note"] or "").splitlines()[:20]:
            print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("view", help="show experiment + recent workflow events")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("set-status", help="move experiment to a status")
    p.add_argument("number", type=int)
    p.add_argument("status")
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("list-by-status", help="list experiments at a given status")
    p.add_argument("--status", default=None)
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_list_by_status)

    p = sub.add_parser("post-marker", help="append an epm:* workflow event")
    p.add_argument("number", type=int)
    p.add_argument("marker", help="marker name, e.g. epm:plan, epm:reviewer-verdict")
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_post_marker)

    p = sub.add_parser("add-tag", help="add a free-text tag")
    p.add_argument("number", type=int)
    p.add_argument("tag")
    p.set_defaults(func=cmd_add_tag)

    p = sub.add_parser("remove-tag", help="remove a free-text tag")
    p.add_argument("number", type=int)
    p.add_argument("tag")
    p.set_defaults(func=cmd_remove_tag)

    p = sub.add_parser("latest-marker", help="show the most recent epm:* event")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_latest_marker)

    p = sub.add_parser("list-markers", help="show all epm:* events on an experiment")
    p.add_argument("number", type=int)
    p.add_argument("--prefix", default="epm:")
    p.add_argument(
        "--json",
        action="store_true",
        help="emit events as JSON instead of a one-per-line table",
    )
    p.set_defaults(func=cmd_list_markers)

    p = sub.add_parser(
        "create-experiment",
        help="create a new experiment row in Sagan",
    )
    p.add_argument("--title", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--body", help="hypothesis/goal text directly on the command line")
    g.add_argument("--body-file", help="path to a file holding the hypothesis/goal text")
    p.add_argument("--status", default="proposed")
    p.add_argument("--runpod-account", default="team", choices=["team", "personal"])
    p.set_defaults(func=cmd_create_experiment)

    p = sub.add_parser("set-body", help="replace the experiment body")
    p.add_argument("number", type=int)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--body", help="body text directly on the command line")
    g.add_argument("--file", help="path to a file containing the body text")
    p.set_defaults(func=cmd_set_body)

    p = sub.add_parser("set-title", help="rename the experiment")
    p.add_argument("number", type=int)
    p.add_argument("title")
    p.set_defaults(func=cmd_set_title)

    p = sub.add_parser(
        "set-clean-result",
        help="flip hasCleanResult=true (analyzer Step 6); Sagan auto-creates pending run",
    )
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_set_clean_result)

    p = sub.add_parser("promote", help="clean-result promotion (useful | not-useful)")
    p.add_argument("number", type=int)
    p.add_argument("verdict", choices=["useful", "not-useful"])
    p.set_defaults(func=cmd_promote)

    args = parser.parse_args()
    try:
        args.func(args)
    except SaganError as e:
        print(f"sagan error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
