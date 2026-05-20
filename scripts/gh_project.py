#!/usr/bin/env python3
"""Thin wrapper around `gh project ...` for the Experiment Queue board.

The /issue skill, analyzer agent, mentor-prep skill, and clean-results skill
all want to mutate or query the GitHub Projects v2 board status field. This
script centralises that so individual skills don't reinvent the GraphQL.

Subcommands:

    set-status <issue> <column>           # set Status field for an issue (manual override)
    list-by-status <column>               # list issues currently in <column>
    list-options <field>                  # list options of a single-select field
    add-status-option <name> [--color X]  # add a new option to the Status field
    remove-status-option <name>           # remove an option (used for rollback)
    set-status-from-labels <issue>        # auto-route by status:* label (called by GH Actions)
    snapshot                              # dump current Status options + per-item Status to JSON
    migrate-options                       # one-shot: rewrite Status options + backfill items
    body-promote <issue> <draft.md>       # promote draft into source-issue body (Stage 2)
    body-restore <issue>                  # rollback body-promote: restore from epm:original-body comment
    promote <issue> useful|not-useful     # USER-ONLY: flip clean-results:draft -> :useful/:not-useful + route board
    one-shot-migrate-legacy               # interactive migration of pre-Stage-2 clean-result issues

Defaults target user `superkaiba`'s "Experiment Queue" project (#1). Override
with --owner / --project. The `--repo` flag scopes set-status to one repo
(default: current repo, inferred via `gh repo view`); list-by-status returns
items from any repo in the project.

Behaviour notes:

* `set-status` adds the issue to the project if it's not already there.
* Status field option names are looked up at call time, so renaming a column
  on the board does not require touching this file. Unknown column names
  exit non-zero with a list of valid options.
* `set-status-from-labels` reads the issue's `status:*` label and routes via
  the `LABEL_TO_COLUMN` table below. Multiple status labels = warning + use
  the last one (event-payload order). No status label = no-op.
* `gh` handles auth + retry. We just shell out and parse JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make `from explore_persona_space.workflow import ...` work without
# `uv run` plumbing; mirrors the import-bootstrap pattern in
# scripts/workflow_lint.py.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.workflow import load_workflow_yaml  # noqa: E402

DEFAULT_OWNER = "superkaiba"
DEFAULT_PROJECT_NUMBER = 1

# `gh project ...` commands accept --limit and default to 30. We pass an
# explicit cap below; if the live `totalCount` ever exceeds it we exit
# loud so silent data loss can't happen as the project grows.
ITEM_LIMIT = 1000
PROJECT_LIST_LIMIT = 100

# Single source of truth for label-driven board routing — derived at
# module-init time from `.claude/workflow.yaml`. The YAML is the upstream
# source; this module is now a consumer. See `src/explore_persona_space/
# workflow.py` for the schema and `scripts/workflow_lint.py` for the
# pre-commit validator. Round-trip equality is enforced by
# `tests/test_label_to_column_coverage.py` and `tests/test_workflow_yaml.py`.
_WORKFLOW = load_workflow_yaml()

# `status:*` labels are the fine-grained state machine used by /issue and other
# skills. The board columns are a coarse user-facing projection of those
# states. `clean-results:draft` / `:useful` / `:not-useful` are non-status
# priority labels (declared in workflow.yaml § priority_labels) and take
# precedence over `status:*` routing (see PRIORITY_LABELS below). The legacy
# bare `clean-results` label and "Clean results" column were dropped on main;
# promotions now route only to "Useful" / "Not useful". The mapping is
# derived from `.claude/workflow.yaml` so updates flow through one source.
LABEL_TO_COLUMN: dict[str, str] = _WORKFLOW.label_to_column()

# Labels that take precedence over `status:*` routing in column_for_labels.
# Order is FIRST-MATCH-WINS:
#   1. `clean-results:draft` (defensive — half-applied promote stays observably
#      unfinished in "Awaiting promotion" until reconciled).
#   2. `clean-results:useful` / `clean-results:not-useful` (promoted, terminal).
# The bare `clean-results` label is intentionally NOT in this tuple: it is a
# back-compat marker for `gh issue list --label clean-results` queries, and
# its own column routing was retired with the "Clean results" column.
# Sourced from workflow.yaml § priority_labels.
PRIORITY_LABELS: tuple[str, ...] = _WORKFLOW.priority_label_names()

# Target option set for `migrate-options`. Names + colors + descriptions.
# Order here is the order columns appear left-to-right on the board.
# Color enum values per GitHub GraphQL ProjectV2SingleSelectFieldOptionColor.
# Sourced from workflow.yaml § columns. Eleven entries, no "Clean results"
# (the legacy column was dropped on main; promotions route to Useful/Not useful).
NEW_COLUMN_SPEC: list[tuple[str, str, str]] = _WORKFLOW.new_column_spec()


@dataclass(frozen=True)
class StatusOption:
    """Full metadata for a single Status field option.

    The GraphQL `updateProjectV2Field.singleSelectOptions` input requires
    `name`, `color`, and `description` to be NON_NULL on every option in
    the replacement list. We fetch all three so add/remove operations can
    rebuild the list without resetting colors or wiping descriptions.
    """

    option_id: str
    color: str  # GRAY, BLUE, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE
    description: str


@dataclass(frozen=True)
class ProjectMeta:
    project_id: str
    status_field_id: str
    options: dict[str, StatusOption]  # column name -> StatusOption


def _gh(args: list[str]) -> str:
    """Run `gh <args>` and return stdout. On failure, propagate stderr + exit code."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


# --- project_meta() disk cache ----------------------------------------------
# Short TTL (default 60s) so back-to-back `set-status` / `list-by-status`
# invocations (the `/issue` skill is the heaviest caller) share one
# graphql roundtrip instead of paying for it every CLI run. Mutations
# (`add-status-option`, `remove-status-option`, `migrate-options`)
# invalidate the cache so the next read fetches fresh option IDs.
_META_CACHE_TTL_DEFAULT = 60
_META_CACHE_DISABLED = False  # tests flip this off via the autouse fixture


def _meta_cache_path(owner: str, number: int) -> Path:
    return _REPO_ROOT / ".claude" / "cache" / f"gh-project-meta-{owner}-{number}.json"


def _meta_cache_ttl() -> int:
    """Read TTL from env var; fall back to default. 0 disables caching."""
    raw = os.getenv("EPM_GH_PROJECT_META_TTL")
    if raw is None:
        return _META_CACHE_TTL_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return _META_CACHE_TTL_DEFAULT


def _read_cached_meta(owner: str, number: int) -> ProjectMeta | None:
    if _META_CACHE_DISABLED:
        return None
    ttl = _meta_cache_ttl()
    if ttl <= 0:
        return None
    path = _meta_cache_path(owner, number)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - data.get("cached_at", 0) > ttl:
        return None
    return ProjectMeta(
        project_id=data["project_id"],
        status_field_id=data["status_field_id"],
        options={
            name: StatusOption(
                option_id=opt["option_id"],
                color=opt["color"],
                description=opt["description"],
            )
            for name, opt in data["options"].items()
        },
    )


def _write_cached_meta(owner: str, number: int, meta: ProjectMeta) -> None:
    if _META_CACHE_DISABLED or _meta_cache_ttl() <= 0:
        return
    path = _meta_cache_path(owner, number)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cached_at": int(time.time()),
                    "project_id": meta.project_id,
                    "status_field_id": meta.status_field_id,
                    "options": {
                        name: {
                            "option_id": opt.option_id,
                            "color": opt.color,
                            "description": opt.description,
                        }
                        for name, opt in meta.options.items()
                    },
                },
                indent=2,
            )
        )
    except OSError:
        # Cache failures must never block the command.
        pass


def invalidate_meta_cache(owner: str = DEFAULT_OWNER, number: int = DEFAULT_PROJECT_NUMBER) -> None:
    """Drop the cached meta for (owner, number). Called after every option mutation."""
    _meta_cache_path(owner, number).unlink(missing_ok=True)


def project_meta(owner: str, number: int) -> ProjectMeta:
    """Fetch project node ID + Status field ID + name->StatusOption map.

    `gh project field-list` returns option ids+names but NOT colors or
    descriptions, so we use the raw GraphQL endpoint here. Colors must be
    preserved across `add-status-option` / `remove-status-option`
    invocations because the `updateProjectV2Field` mutation REPLACES the
    full options list — without round-tripping color the whole board's
    color coding is destroyed (HIGH-1, code-review v1).

    Disk-cached for `EPM_GH_PROJECT_META_TTL` seconds (default 60) keyed
    on (owner, number). Cache invalidated by `invalidate_meta_cache()`
    after every option mutation.
    """
    cached = _read_cached_meta(owner, number)
    if cached is not None:
        return cached

    query = (
        "query($owner:String!, $number:Int!) {"
        "  user(login:$owner) {"
        "    projectV2(number:$number) {"
        "      id"
        '      field(name:"Status") {'
        "        ... on ProjectV2SingleSelectField {"
        "          id"
        "          options { id name color description }"
        "        }"
        "      }"
        "    }"
        "  }"
        "}"
    )
    raw = _gh(
        [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"number={number}",
        ]
    )
    data = json.loads(raw).get("data", {})
    project = (data.get("user") or {}).get("projectV2")
    if project is None:
        sys.exit(f"project #{number} not found under owner '{owner}'")
    field = project.get("field")
    if field is None or "options" not in field:
        sys.exit(f"project #{number} has no Status single-select field")

    meta = ProjectMeta(
        project_id=project["id"],
        status_field_id=field["id"],
        options={
            opt["name"]: StatusOption(
                option_id=opt["id"],
                color=opt.get("color", "GRAY"),
                description=opt.get("description", "") or "",
            )
            for opt in field["options"]
        },
    )
    _write_cached_meta(owner, number, meta)
    return meta


def current_repo() -> str:
    """Return current repo as `owner/name`.

    Returns "" only when `gh repo view` reports an empty stdout AND a clean
    exit; any non-zero exit (auth failure, not-in-repo, etc.) surfaces gh's
    stderr to the caller before returning empty, so the downstream
    "could not infer current repo" error is never the only signal.
    """
    proc = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return ""
    return proc.stdout.strip()


def _list_items(owner: str, number: int) -> list[dict]:
    """Return all items in a project, exiting loud if the live total exceeds ITEM_LIMIT.

    Centralises the `gh project item-list` call + the totalCount overflow
    guard so both `find_item_id` and `cmd_list_by_status` get the same
    silent-data-loss protection as the project grows.
    """
    raw = _gh(
        [
            "project",
            "item-list",
            str(number),
            "--owner",
            owner,
            "--format",
            "json",
            "--limit",
            str(ITEM_LIMIT),
        ]
    )
    data = json.loads(raw)
    total = data.get("totalCount", 0)
    items = data.get("items", [])
    if total > ITEM_LIMIT:
        sys.exit(
            f"project #{number} has {total} items, more than the {ITEM_LIMIT}-row "
            f"window this script fetches. Bump ITEM_LIMIT in scripts/gh_project.py "
            f"(or migrate to a paginated GraphQL query)."
        )
    return items


def find_item_id(owner: str, number: int, issue: int, repo: str | None) -> str | None:
    """Resolve the project item ID for a given issue. None if the issue isn't in the project.

    Iterates project items rather than using a GraphQL filter because the
    project is small (low hundreds) and `gh project item-list` already
    resolves the Status field for free.
    """
    for item in _list_items(owner, number):
        c = item.get("content", {})
        if c.get("number") != issue:
            continue
        if repo and c.get("repository") != repo:
            continue
        return item["id"]
    return None


def add_to_project(owner: str, number: int, issue_url: str) -> str:
    """Add an issue to the project, return the new project item id."""
    raw = _gh(
        [
            "project",
            "item-add",
            str(number),
            "--owner",
            owner,
            "--url",
            issue_url,
            "--format",
            "json",
        ]
    )
    item_id = json.loads(raw).get("id")
    if not item_id:
        sys.exit(f"unexpected item-add response: {raw}")
    return item_id


def cmd_set_status(args: argparse.Namespace) -> None:
    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    meta = project_meta(args.owner, args.project)
    if args.column not in meta.options:
        valid = ", ".join(sorted(meta.options))
        sys.exit(f"unknown column '{args.column}'. valid: {valid}")
    option_id = meta.options[args.column].option_id

    item_id = find_item_id(args.owner, args.project, args.issue, repo)
    if item_id is None:
        url = f"https://github.com/{repo}/issues/{args.issue}"
        item_id = add_to_project(args.owner, args.project, url)

    _gh(
        [
            "project",
            "item-edit",
            "--id",
            item_id,
            "--field-id",
            meta.status_field_id,
            "--project-id",
            meta.project_id,
            "--single-select-option-id",
            option_id,
        ]
    )
    print(f"#{args.issue} -> '{args.column}' (option {option_id})")


def cmd_list_by_status(args: argparse.Namespace) -> None:
    meta = project_meta(args.owner, args.project)
    if args.column not in meta.options:
        valid = ", ".join(sorted(meta.options))
        sys.exit(f"unknown column '{args.column}'. valid: {valid}")

    items = [it for it in _list_items(args.owner, args.project) if it.get("status") == args.column]

    if args.json:
        print(json.dumps(items, indent=2))
        return

    if not items:
        print(f"no items in '{args.column}'")
        return
    for it in items:
        c = it.get("content", {})
        n = c.get("number")
        title = c.get("title", "")
        print(f"#{n} {title}" if n is not None else title)


def cmd_list_all(args: argparse.Namespace) -> None:
    """Group items by Status column from a single `item-list` call.

    Replaces the N-query pattern of running `list-by-status` once per
    column (e.g. /pm triage previously fired 8 calls). One `_list_items`
    is enough: every item carries its current `status` string, so we bin
    client-side.
    """
    items = _list_items(args.owner, args.project)
    bins: dict[str, list[dict]] = {}
    for it in items:
        col = it.get("status") or "(no status)"
        bins.setdefault(col, []).append(it)

    if args.columns:
        wanted = {c.strip() for c in args.columns.split(",") if c.strip()}
        bins = {k: v for k, v in bins.items() if k in wanted}

    canonical_order = [name for name, _, _ in NEW_COLUMN_SPEC]
    ordered_cols = [c for c in canonical_order if c in bins]
    ordered_cols += [c for c in sorted(bins) if c not in canonical_order]

    if args.json:
        out = {
            col: [
                {
                    "number": (it.get("content") or {}).get("number"),
                    "title": (it.get("content") or {}).get("title", ""),
                    "repo": (it.get("content") or {}).get("repository"),
                }
                for it in bins[col]
            ]
            for col in ordered_cols
        }
        print(json.dumps(out, indent=2))
        return

    if args.counts_only:
        for col in ordered_cols:
            print(f"{col}\t{len(bins[col])}")
        return

    for col in ordered_cols:
        print(f"### {col} ({len(bins[col])})")
        for it in bins[col]:
            c = it.get("content") or {}
            n = c.get("number")
            title = c.get("title", "")
            print(f"#{n} {title}" if n is not None else title)
        print()


def cmd_add_status_option(args: argparse.Namespace) -> None:
    """Add a new option to the existing Status single-select field via GraphQL.

    The GraphQL mutation `updateProjectV2Field` REPLACES the full options
    list, so we read the existing options first and merge. Idempotent: if
    the option already exists, this is a no-op.
    """
    meta = project_meta(args.owner, args.project)
    if args.option in meta.options:
        existing_opt = meta.options[args.option]
        print(f"option {args.option!r} already exists (id={existing_opt.option_id}); no-op")
        return
    # Build the merged set of options. The mutation REPLACES the full list,
    # so we must pass each existing option's actual color and description
    # back through — passing color="GRAY" for everything destroys the
    # board's color coding (HIGH-1, code-review v1).
    existing = [
        {
            "id": opt.option_id,
            "name": name,
            "color": opt.color,
            "description": opt.description,
        }
        for name, opt in meta.options.items()
    ]
    new_options = [
        *existing,
        {
            "name": args.option,
            "color": args.color or "GRAY",
            # `description` is an optional CLI flag (and is omitted from the
            # synthetic Namespace in tests) — default to "" when absent.
            "description": getattr(args, "description", "") or "",
        },
    ]
    # Route through the `_graphql` helper which uses `gh api graphql --input`
    # (typed JSON variables). The previous `_gh -f options=<json-string>`
    # path silently fails because `-f` passes the value as a STRING, but
    # the `singleSelectOptions` GraphQL variable expects a typed
    # `[ProjectV2SingleSelectFieldOptionInput!]!` array. Same fix applied
    # to `cmd_remove_status_option` below — both pre-existed broken; use
    # `_replace_options` once it's introduced (see `_graphql` helper).
    _replace_options(meta.status_field_id, new_options)
    print(f"added option {args.option!r} to Status field on project #{args.project}")


def cmd_remove_status_option(args: argparse.Namespace) -> None:
    """Remove an option from the Status field. Used for rollback (plan §11.1)."""
    meta = project_meta(args.owner, args.project)
    if args.option not in meta.options:
        print(f"option {args.option!r} does not exist; no-op")
        return
    # Preserve every surviving option's color + description; passing
    # color="GRAY" for all of them destroys the board's color coding
    # (HIGH-1, code-review v1).
    remaining = [
        {
            "id": opt.option_id,
            "name": name,
            "color": opt.color,
            "description": opt.description,
        }
        for name, opt in meta.options.items()
        if name != args.option
    ]
    # Route through `_replace_options` (uses `gh api graphql --input` with
    # typed JSON variables). The previous `-f options=<json-string>` path
    # broke on the typed `[ProjectV2SingleSelectFieldOptionInput!]!`
    # variable — same fix as in `cmd_add_status_option`.
    _replace_options(meta.status_field_id, remaining)
    print(f"removed option {args.option!r} from Status field")


def cmd_list_options(args: argparse.Namespace) -> None:
    """List options of a single-select field. Currently only `Status` is supported."""
    meta = project_meta(args.owner, args.project)
    if args.field == "Status":
        for name, opt in sorted(meta.options.items()):
            print(f"{name}\t{opt.option_id}\t{opt.color}")
    else:
        sys.exit(f"only Status field supported (got {args.field!r})")


# ---------------------------------------------------------------------------
# Label-driven routing (called from .github/workflows/project-sync.yml)
# ---------------------------------------------------------------------------


def _issue_labels(issue: int, repo: str) -> list[str]:
    """Return the label names on an issue via REST.

    Routes through the core 5000/hr bucket instead of GraphQL — gh
    issue view's `--json` projection uses GraphQL underneath, which is
    the bottleneck when project-board ops exhaust the GraphQL bucket.
    """
    raw = _gh(["api", "-H", "Accept: application/vnd.github+json", f"repos/{repo}/issues/{issue}"])
    return [lbl["name"] for lbl in json.loads(raw).get("labels", [])]


def column_for_labels(labels: list[str]) -> str | None:
    """Return the column name for the issue's current labels, or None.

    Routing precedence (first match wins):
      1. PRIORITY_LABELS — `clean-results:draft` -> "Awaiting promotion";
         `clean-results:useful` -> "Useful"; `clean-results:not-useful` ->
         "Not useful". The bare `clean-results` label is NOT priority-routed
         (it is a back-compat marker for `gh issue list --label clean-results`).
      2. status:* labels via LABEL_TO_COLUMN. If multiple status:* labels are
         present, the LAST one in `labels` wins (gh returns labels in
         application order; most recent flip is last). A warning is emitted.
      3. None (issue has no routable label).
    """
    label_set = set(labels)
    for priority in PRIORITY_LABELS:
        if priority in label_set:
            return LABEL_TO_COLUMN[priority]
    status_labels = [lbl for lbl in labels if lbl in LABEL_TO_COLUMN and lbl not in PRIORITY_LABELS]
    if not status_labels:
        return None
    if len(status_labels) > 1:
        sys.stderr.write(
            f"WARN: multiple status:* labels {status_labels}; using last ({status_labels[-1]})\n"
        )
    return LABEL_TO_COLUMN[status_labels[-1]]


def cmd_set_status_from_labels(args: argparse.Namespace) -> None:
    """Set Status from the issue's status:* label. No-op if no status label."""
    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    labels = _issue_labels(args.issue, repo)
    column = column_for_labels(labels)
    if column is None:
        print(f"#{args.issue} has no status:* label, leaving Status unchanged")
        return

    meta = project_meta(args.owner, args.project)
    if column not in meta.options:
        valid = ", ".join(sorted(meta.options))
        sys.exit(f"column '{column}' not on board (have: {valid}). Run migrate-options.")
    option_id = meta.options[column].option_id

    item_id = find_item_id(args.owner, args.project, args.issue, repo)
    if item_id is None:
        url = f"https://github.com/{repo}/issues/{args.issue}"
        item_id = add_to_project(args.owner, args.project, url)

    _gh(
        [
            "project",
            "item-edit",
            "--id",
            item_id,
            "--field-id",
            meta.status_field_id,
            "--project-id",
            meta.project_id,
            "--single-select-option-id",
            option_id,
        ]
    )
    print(f"#{args.issue} -> '{column}' (label={[l for l in labels if l in LABEL_TO_COLUMN]})")


# ---------------------------------------------------------------------------
# One-shot Status-options migration (run locally; not from CI)
# ---------------------------------------------------------------------------


def _graphql(query: str, variables: dict | None = None) -> dict:
    """Run a GraphQL query/mutation via `gh api graphql --input`.

    Delegates the actual subprocess call to `_gh` so test fixtures that
    `monkeypatch.setattr(gh_project, "_gh", ...)` still intercept the
    request. The `--input <tempfile>` form is required because the
    `singleSelectOptions: [ProjectV2SingleSelectFieldOptionInput!]!`
    variable type rejects the string-encoded value that `-f` / `-F` would
    send (gh would post `variables: {"opts": "[...]"}` as a string).
    """
    import tempfile
    from pathlib import Path

    body = {"query": query, "variables": variables or {}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        path = f.name
    try:
        raw = _gh(["api", "graphql", "--input", path])
        data = json.loads(raw)
        if "errors" in data:
            sys.exit(json.dumps(data["errors"], indent=2))
        return data["data"]
    finally:
        Path(path).unlink(missing_ok=True)


def _fetch_all_issue_labels(repo: str) -> dict[int, list[str]]:
    """Return {issue_number: [label_name, ...]} for every issue in repo.

    REST paginator (`/repos/{owner}/{repo}/issues?state=all&per_page=100`
    with `--paginate`) replaces the previous `gh issue list --limit 300`
    (silent truncation) and a GraphQL cursor-walk (separate quota bucket).
    REST routes through the core 5000/hr bucket so the GraphQL bucket
    stays available for project-board ops, which have NO REST coverage
    on user-owned projects v2.

    `gh api --paginate` concatenates each page's JSON array directly with
    no separator — we use `json.JSONDecoder.raw_decode` to split.

    The REST `issues` endpoint includes pull requests as well; we filter
    them out by checking for `pull_request` on the node (PRs always carry
    that key; plain issues don't).
    """
    raw = _gh(
        [
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            f"repos/{repo}/issues?state=all&per_page=100",
            "--paginate",
        ]
    )

    raw = raw.strip()
    out: dict[int, list[str]] = {}
    if not raw:
        return out

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        page, end = decoder.raw_decode(raw[idx:])
        for node in page:
            if "pull_request" in node:
                continue  # REST issues endpoint conflates issues + PRs
            out[node["number"]] = [lbl["name"] for lbl in node.get("labels", [])]
        idx += end
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
    return out


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Dump current Status options + per-item Status to a JSON file (rollback point)."""
    from datetime import datetime
    from pathlib import Path

    meta = project_meta(args.owner, args.project)
    items = _list_items(args.owner, args.project)
    snap = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "owner": args.owner,
        "project": args.project,
        "project_id": meta.project_id,
        "status_field_id": meta.status_field_id,
        "options": [
            {"name": k, "id": v.option_id, "color": v.color, "description": v.description}
            for k, v in meta.options.items()
        ],
        "items": [
            {
                "item_id": it["id"],
                "issue": it.get("content", {}).get("number"),
                "repo": it.get("content", {}).get("repository"),
                "status": it.get("status"),
            }
            for it in items
        ],
    }
    out_path = (
        Path(args.out)
        if args.out
        else Path(f".claude/cache/board-snapshot-{snap['timestamp'].replace(':', '-')}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=2))
    print(f"snapshot written to {out_path}")
    print(f"  options: {len(snap['options'])}  items: {len(snap['items'])}")


def _list_options_full() -> list[dict]:
    """Read full Status field options (id, name, color, description) via GraphQL."""
    q = """
    query($p: ID!) {
      node(id: $p) {
        ... on ProjectV2 {
          field(name: "Status") {
            ... on ProjectV2SingleSelectField {
              id
              options { id name color description }
            }
          }
        }
      }
    }
    """
    # We need the project node id; use project_meta which uses gh project + field-list.
    # gh project field-list does NOT return color/description, hence GraphQL here.
    meta = project_meta(DEFAULT_OWNER, DEFAULT_PROJECT_NUMBER)
    data = _graphql(q, {"p": meta.project_id})
    return data["node"]["field"]["options"]


def _replace_options(field_id: str, target: list[dict]) -> None:
    """Replace the Status field's option set in a single mutation.

    `target` is a list of dicts with keys {id?, name, color, description}.
    Existing IDs preserved when passed; new entries get fresh IDs.

    Invalidates the disk-cached meta for the default (owner, project)
    after a successful mutation. Today every mutation path operates on
    the same default project (#1 under `superkaiba`); if multi-project
    support lands, callers should invalidate explicitly for the affected
    (owner, project) pair.
    """
    q = """
    mutation($fieldId: ID!, $opts: [ProjectV2SingleSelectFieldOptionInput!]!) {
      updateProjectV2Field(input: {fieldId: $fieldId, singleSelectOptions: $opts}) {
        projectV2Field { ... on ProjectV2SingleSelectField { options { id name } } }
      }
    }
    """
    _graphql(q, {"fieldId": field_id, "opts": target})
    invalidate_meta_cache()


def cmd_migrate_options(args: argparse.Namespace) -> None:
    """One-shot: rewrite Status options to NEW_COLUMN_SPEC and backfill all items.

    Two-pass to avoid orphaning items:
      1. Add new options alongside existing (preserve IDs of existing).
      2. For each item, set Status to the new column its status:* label maps to.
      3. Remove any options not in NEW_COLUMN_SPEC.

    Always snapshots first to .claude/cache/board-snapshot-<utc>.json unless --skip-snapshot.
    """
    from datetime import datetime
    from pathlib import Path

    if not args.skip_snapshot:
        ts = datetime.utcnow().isoformat().replace(":", "-") + "Z"
        snap_path = Path(f".claude/cache/board-snapshot-{ts}.json")
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_args = argparse.Namespace(owner=args.owner, project=args.project, out=str(snap_path))
        cmd_snapshot(snap_args)

    current = _list_options_full()
    by_name = {o["name"]: o for o in current}

    # Pass 1: combined option set (existing IDs preserved + new added).
    target_combined: list[dict] = []
    for o in current:
        target_combined.append(
            {
                "id": o["id"],
                "name": o["name"],
                "color": o.get("color") or "GRAY",
                "description": o.get("description") or "",
            }
        )
    new_names = {n for n, _, _ in NEW_COLUMN_SPEC}
    for name, color, desc in NEW_COLUMN_SPEC:
        if name not in by_name:
            target_combined.append({"name": name, "color": color, "description": desc})

    meta = project_meta(args.owner, args.project)
    if args.dry_run:
        print(f"[dry-run] would add {len(new_names - set(by_name))} new options:")
        for name in sorted(new_names - set(by_name)):
            print(f"  + {name}")
        # Build a synthetic options map that includes the would-be-added names
        # so Pass 2's `column not in meta.options` check passes.
        synthetic_options = {
            **meta.options,
            **{
                n: StatusOption(option_id=f"<dry-run-{n}>", color="GRAY", description="")
                for n in new_names
            },
        }
        meta = ProjectMeta(
            project_id=meta.project_id,
            status_field_id=meta.status_field_id,
            options=synthetic_options,
        )
    else:
        _replace_options(meta.status_field_id, target_combined)
        print(f"added {len(new_names - set(by_name))} new option(s)")
        # Refresh meta after option mutation so the new option IDs are present.
        meta = project_meta(args.owner, args.project)

    # Pre-fetch label sets for every issue in the repo. `gh project
    # item-list` does not return labels in the content payload, so we
    # build a number -> labels map below. Cursor-paginated GraphQL is
    # used because `gh issue list --limit N` silently truncates at N —
    # the prior `--limit 300` would miss issues in a repo that has
    # grown past 300 and there is no way to detect the truncation
    # client-side.
    repo_for_labels = args.repo or current_repo()
    labels_by_issue = _fetch_all_issue_labels(repo_for_labels)

    # Pass 2: backfill items based on their current labels.
    items = _list_items(args.owner, args.project)
    moved = skipped = no_label = 0
    for it in items:
        c = it.get("content", {})
        issue = c.get("number")
        if not issue:
            skipped += 1
            continue
        labels = labels_by_issue.get(issue, [])
        column = column_for_labels(labels)
        if column is None:
            no_label += 1
            continue
        if column not in meta.options:
            print(f"  WARN #{issue}: target '{column}' missing from board; skipping")
            skipped += 1
            continue
        if it.get("status") == column:
            continue
        if args.dry_run:
            print(f"  [dry-run] #{issue}: {it.get('status')} -> {column}")
        else:
            _gh(
                [
                    "project",
                    "item-edit",
                    "--id",
                    it["id"],
                    "--field-id",
                    meta.status_field_id,
                    "--project-id",
                    meta.project_id,
                    "--single-select-option-id",
                    meta.options[column].option_id,
                ]
            )
        moved += 1
    print(f"backfill: moved={moved} skipped={skipped} no_status_label={no_label}")

    # Pass 3: drop legacy options not in NEW_COLUMN_SPEC.
    current_after = _list_options_full()
    keep: list[dict] = []
    drop_names: list[str] = []
    for o in current_after:
        if o["name"] in new_names:
            keep.append(
                {
                    "id": o["id"],
                    "name": o["name"],
                    "color": o.get("color") or "GRAY",
                    "description": o.get("description") or "",
                }
            )
        else:
            drop_names.append(o["name"])
    if args.dry_run:
        print(f"[dry-run] would drop {len(drop_names)} legacy option(s): {drop_names}")
    else:
        if drop_names:
            _replace_options(meta.status_field_id, keep)
            print(f"dropped {len(drop_names)} legacy option(s): {drop_names}")
        else:
            print("no legacy options to drop")


# ---------------------------------------------------------------------------
# Stage 2: inline clean-result body promotion (replaces spawn-new-issue)
# ---------------------------------------------------------------------------

PROMOTED_MARKER = "<!-- epm:promoted -->"
ORIGINAL_MARKER = "<!-- epm:original-body -->"


def _gh_issue_view_full(issue: int, repo: str) -> dict:
    """Return {title, body, labels, comments[]} for an issue via REST.

    Routes through the core 5000/hr bucket to keep the GraphQL bucket
    headroom for project-board ops. Two REST calls: issue metadata +
    paginated comments. Comments are reshaped to the same key set the
    legacy `gh issue view --json comments` projection produced (each
    comment has a `body` field), so the `_has_marker` / `cmd_body_promote`
    / `cmd_body_restore` callers below don't change.
    """
    issue_data = json.loads(
        _gh(
            [
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                f"repos/{repo}/issues/{issue}",
            ]
        )
    )

    total_comments = issue_data.get("comments", 0)
    comments: list[dict] = []
    if total_comments > 0:
        raw = _gh(
            [
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                f"repos/{repo}/issues/{issue}/comments?per_page=100",
                "--paginate",
            ]
        )
        # `gh api --paginate` concatenates per-page JSON arrays without
        # separators; walk with raw_decode.
        raw = raw.strip()
        if raw:
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(raw):
                page, end = decoder.raw_decode(raw[idx:])
                comments.extend(page)
                idx += end
                while idx < len(raw) and raw[idx].isspace():
                    idx += 1

    return {
        "title": issue_data["title"],
        "body": issue_data.get("body") or "",
        "labels": [{"name": lbl["name"]} for lbl in issue_data.get("labels", [])],
        "comments": comments,
    }


def _has_marker(comments: list[dict], marker: str) -> dict | None:
    for c in comments:
        body = c.get("body") or ""
        if body.startswith(marker) or marker in body.split("\n", 1)[0]:
            return c
    return None


def cmd_body_promote(args: argparse.Namespace) -> None:
    """Promote a clean-result draft into the source issue's body.

    Three steps (idempotent):
      1. If body already starts with PROMOTED_MARKER → just edit body (revision).
      2. Else: post `epm:original-body` comment with verbatim original.
      3. Edit body to PROMOTED_MARKER + draft contents; add clean-results:draft.
    """
    from pathlib import Path

    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    draft = Path(args.draft).read_text()
    new_body = f"{PROMOTED_MARKER}\n\n{draft}"

    issue_data = _gh_issue_view_full(args.issue, repo)
    body = issue_data.get("body") or ""

    # Step 0: idempotency / revision path.
    if body.startswith(PROMOTED_MARKER):
        _gh(["issue", "edit", str(args.issue), "-R", repo, "--body", new_body])
        print(f"#{args.issue}: revision — body re-edited (already promoted)")
        return

    # Step 1: preserve original as comment (skip if marker already present from prior partial run).
    if _has_marker(issue_data.get("comments", []), ORIGINAL_MARKER):
        print(f"#{args.issue}: epm:original-body comment already exists — skipping snapshot step")
    else:
        snapshot_comment = (
            f"{ORIGINAL_MARKER}\n## Original issue body (preserved before clean-result promotion)\n\n"
            f"{body}"
        )
        _gh(["issue", "comment", str(args.issue), "-R", repo, "--body", snapshot_comment])
        print(f"#{args.issue}: original body preserved as comment")

    # Step 2: replace body.
    _gh(["issue", "edit", str(args.issue), "-R", repo, "--body", new_body])
    print(f"#{args.issue}: body replaced with clean-result")

    # Step 3: add label.
    _gh(["issue", "edit", str(args.issue), "-R", repo, "--add-label", "clean-results:draft"])
    print(f"#{args.issue}: added label clean-results:draft")


def cmd_body_restore(args: argparse.Namespace) -> None:
    """Rollback: restore the original body from the preserved comment."""
    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    issue_data = _gh_issue_view_full(args.issue, repo)
    snap = _has_marker(issue_data.get("comments", []), ORIGINAL_MARKER)
    if snap is None:
        sys.exit(f"#{args.issue}: no {ORIGINAL_MARKER} comment found")

    comment_body = snap["body"]
    # Strip the marker line + the "## Original issue body..." heading + blank line.
    lines = comment_body.split("\n")
    # Find the third blank line (after marker, after H2 heading, then content begins).
    # Format: <marker>\n## ...\n\n<content...>
    # So drop the first 3 lines (marker, heading, blank).
    if len(lines) >= 3:
        original = "\n".join(lines[3:])
    else:
        original = ""

    _gh(["issue", "edit", str(args.issue), "-R", repo, "--body", original])
    _gh(
        [
            "issue",
            "edit",
            str(args.issue),
            "-R",
            repo,
            "--remove-label",
            "clean-results:draft",
        ]
    )
    # clean-results label may or may not be present; remove if so.
    labels = [lbl["name"] for lbl in issue_data.get("labels", [])]
    if "clean-results" in labels:
        _gh(["issue", "edit", str(args.issue), "-R", repo, "--remove-label", "clean-results"])
    print(f"#{args.issue}: body restored from {ORIGINAL_MARKER} comment; labels reverted")


# ---------------------------------------------------------------------------
# One-shot migration of pre-Stage-2 legacy clean-result issues
# ---------------------------------------------------------------------------

# (legacy_issue_or_None, kind, default_source_or_None, note)
# kind: "draft-issue" — issue itself IS a separately-spawned clean-result draft
#       "awaiting"    — source issue with status:awaiting-promotion (cached draft expected)
LEGACY_ISSUES: list[tuple[int, str, int | None, str]] = [
    (248, "draft-issue", None, "ZLT marker attention analysis (LOW)"),
    (185, "draft-issue", 139, "EM dose-response cliff at 10-25 steps (MODERATE)"),
    (184, "draft-issue", None, "EM collapses persona discrimination vs benign (MODERATE)"),
    (109, "draft-issue", None, "(check title for source)"),
    (91, "draft-issue", None, "(check title for source)"),
    (224, "awaiting", 224, "attention analysis"),
    (139, "awaiting", 139, "dose-response (paired with draft #185)"),
]


def cmd_promote(args: argparse.Namespace) -> None:
    """Promote a clean-results:draft issue to useful or not-useful.

    User-only operation. The /issue skill parks an experiment at
    `status:awaiting-promotion` after reviewer PASS and EXITs — only this
    command (run by the user when satisfied) flips the labels and routes
    the project board out of the Awaiting promotion column. Re-entry into
    `/issue <N>` after promotion fires Step 10 (auto-complete →
    follow-up-proposer → pod-termination prompt).

    Replaces the deprecated `/clean-results promote <N> useful|not-useful`
    skill invocation. No body editing — just label flips + column move.
    """
    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    verdict = args.verdict
    if verdict not in ("useful", "not-useful"):
        sys.exit(f"verdict must be 'useful' or 'not-useful', got {verdict!r}")
    column = "Useful" if verdict == "useful" else "Not useful"

    # Verify the issue currently has clean-results:draft (refuse to promote
    # something that was never drafted as a clean result).
    labels = _issue_labels(args.issue, repo)
    if "clean-results:draft" not in labels:
        sys.exit(
            f"#{args.issue} does not carry the 'clean-results:draft' label "
            f"(current labels: {labels}); refusing to promote."
        )

    # Step 1: label flip
    _gh(
        [
            "issue",
            "edit",
            str(args.issue),
            "--repo",
            repo,
            "--add-label",
            f"clean-results:{verdict}",
            "--add-label",
            "clean-results",
            "--remove-label",
            "clean-results:draft",
        ]
    )
    # Step 2: project-board column
    set_status_args = argparse.Namespace(
        owner=args.owner,
        project=args.project,
        repo=repo,
        issue=args.issue,
        column=column,
    )
    cmd_set_status(set_status_args)
    # Step 3: print user-facing reminder to re-enter /issue <N>
    print(
        f"#{args.issue} promoted: clean-results:{verdict} + column '{column}'.\n"
        f"Next: re-enter `/issue {args.issue}` so Step 10 (auto-complete → "
        f"follow-up-proposer → pod-termination prompt) fires."
    )


def cmd_one_shot_migrate_legacy(args: argparse.Namespace) -> None:
    """Walk LEGACY_ISSUES; per issue, prompt to body-promote into the source.

    For draft-issue kind: source defaults to the value in LEGACY_ISSUES, or
    the operator types one. The legacy issue's body is used as the draft.
    For awaiting kind: cached draft at .claude/cache/issue-<N>-clean-result.md
    is the default; operator can override.
    """
    from pathlib import Path

    repo = args.repo or current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    for legacy_n, kind, default_source, note in LEGACY_ISSUES:
        print(f"\n--- #{legacy_n} ({kind}) — {note} ---")
        try:
            view = _gh_issue_view_full(legacy_n, repo)
            print(f"  Title: {view['title']}")
        except SystemExit:
            print("  WARN: could not fetch issue, skipping")
            continue

        if kind == "draft-issue":
            default_str = f" [{default_source}]" if default_source else ""
            src_input = input(f"  Source issue N to promote into{default_str}: ").strip()
            src = int(src_input) if src_input else default_source
            if not src:
                print("  SKIP — no source")
                continue
            ok = input(f"  Body-promote #{legacy_n}'s body into #{src}? [y/N] ").strip().lower()
            if ok != "y":
                print("  SKIP")
                continue
            tmp = Path(f"/tmp/legacy-migrate-{legacy_n}.md")
            tmp.write_text(view["body"] or "")
            promote_args = argparse.Namespace(
                owner=args.owner,
                project=args.project,
                repo=repo,
                issue=src,
                draft=str(tmp),
            )
            cmd_body_promote(promote_args)
            _gh(["issue", "edit", str(legacy_n), "-R", repo, "--add-label", "superseded"])
            _gh(
                [
                    "issue",
                    "comment",
                    str(legacy_n),
                    "-R",
                    repo,
                    "--body",
                    f"Superseded by inline promotion to #{src} (legacy migration).",
                ]
            )
            print(f"  DONE: #{legacy_n} marked superseded; inlined into #{src}")

        elif kind == "awaiting":
            cache = Path(f".claude/cache/issue-{legacy_n}-clean-result.md")
            if cache.exists():
                draft_path = str(cache)
                print(f"  Found cached draft at {draft_path}")
            else:
                draft_path = input(f"  Path to draft.md for #{legacy_n}: ").strip()
                if not draft_path or not Path(draft_path).exists():
                    print("  SKIP — no draft path")
                    continue
            ok = input(f"  Body-promote {draft_path} into #{legacy_n}? [y/N] ").strip().lower()
            if ok != "y":
                print("  SKIP")
                continue
            promote_args = argparse.Namespace(
                owner=args.owner,
                project=args.project,
                repo=repo,
                issue=legacy_n,
                draft=draft_path,
            )
            cmd_body_promote(promote_args)
            print(f"  DONE: #{legacy_n} body promoted in place")

    print("\nMigration complete. Review each issue, then continue with Stage 2 PR.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="project owner login")
    parser.add_argument(
        "--project",
        type=int,
        default=DEFAULT_PROJECT_NUMBER,
        help="project number (default: 1, 'Experiment Queue')",
    )
    parser.add_argument("--repo", help="owner/repo for the issue (default: current repo)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("set-status", help="set Status field for an issue")
    p.add_argument("issue", type=int, help="issue number")
    p.add_argument("column", help="target Status column name")
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("list-by-status", help="list issues in a Status column")
    p.add_argument("column", help="Status column name")
    p.add_argument("--json", action="store_true", help="emit raw JSON instead of `#N title` rows")
    p.set_defaults(func=cmd_list_by_status)

    p = sub.add_parser(
        "list-all",
        help="list all Status columns grouped, in ONE API call (replaces 8x list-by-status)",
    )
    p.add_argument(
        "--columns",
        help="comma-separated subset of columns to show (still one API call)",
    )
    p.add_argument("--json", action="store_true", help="emit grouped JSON keyed by column")
    p.add_argument(
        "--counts-only",
        action="store_true",
        help="terse mode: '<column>\\t<count>' per line",
    )
    p.set_defaults(func=cmd_list_all)

    p = sub.add_parser("add-status-option", help="add a new option to the Status field")
    p.add_argument("option", help="option name (e.g. 'Awaiting Promotion')")
    p.add_argument(
        "--color",
        default="GRAY",
        help="GRAY, BLUE, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE",
    )
    p.add_argument(
        "--description",
        default="",
        help="optional one-line description shown in the GitHub Projects UI",
    )
    p.set_defaults(func=cmd_add_status_option)

    p = sub.add_parser(
        "remove-status-option",
        help="remove an option from the Status field (rollback)",
    )
    p.add_argument("option", help="option name to remove")
    p.set_defaults(func=cmd_remove_status_option)

    p = sub.add_parser("list-options", help="list options of a single-select field")
    p.add_argument("field", help="field name (only 'Status' supported)")
    p.set_defaults(func=cmd_list_options)

    p = sub.add_parser(
        "set-status-from-labels",
        help="auto-route an issue to its Status column based on status:* label",
    )
    p.add_argument("issue", type=int, help="issue number")
    p.set_defaults(func=cmd_set_status_from_labels)

    p = sub.add_parser("snapshot", help="dump Status options + per-item state to JSON")
    p.add_argument("--out", help="output path (default: .claude/cache/board-snapshot-<utc>.json)")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser(
        "migrate-options",
        help="one-shot: rewrite Status options to NEW_COLUMN_SPEC + backfill items",
    )
    p.add_argument("--dry-run", action="store_true", help="preview without mutating")
    p.add_argument(
        "--skip-snapshot", action="store_true", help="skip pre-mutation snapshot (NOT recommended)"
    )
    p.set_defaults(func=cmd_migrate_options)

    p = sub.add_parser(
        "body-promote",
        help="promote a draft into the source-issue body (Stage 2 inline clean-result)",
    )
    p.add_argument("issue", type=int, help="source issue number")
    p.add_argument("draft", help="path to clean-result draft .md")
    p.set_defaults(func=cmd_body_promote)

    p = sub.add_parser(
        "body-restore",
        help="rollback body-promote: restore original body from preserved comment",
    )
    p.add_argument("issue", type=int, help="issue number to restore")
    p.set_defaults(func=cmd_body_restore)

    p = sub.add_parser(
        "promote",
        help="USER-ONLY: flip clean-results:draft to :useful or :not-useful + route project board",
    )
    p.add_argument("issue", type=int, help="source issue number")
    p.add_argument("verdict", choices=["useful", "not-useful"], help="promotion verdict")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser(
        "one-shot-migrate-legacy",
        help="interactive one-time migration of pre-Stage-2 clean-result issues",
    )
    p.set_defaults(func=cmd_one_shot_migrate_legacy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
