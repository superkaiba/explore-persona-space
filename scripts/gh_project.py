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
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_OWNER = "superkaiba"
DEFAULT_PROJECT_NUMBER = 1

# `gh project ...` commands accept --limit and default to 30. We pass an
# explicit cap below; if the live `totalCount` ever exceeds it we exit
# loud so silent data loss can't happen as the project grows.
ITEM_LIMIT = 1000
PROJECT_LIST_LIMIT = 100

# Single source of truth for label-driven board routing.
# Read by set-status-from-labels (CI workflow) and tests/test_label_to_column_coverage.py.
# `clean-results` and `clean-results:draft` are NOT status:* labels but they
# also route to a column: "Awaiting Promotion". The non-status routing takes
# precedence over `status:*` so a clean-result-bearing issue sits in
# "Awaiting Promotion" regardless of its underlying status (typically
# `status:done-experiment` or `status:awaiting-promotion`).
LABEL_TO_COLUMN: dict[str, str] = {
    "status:proposed": "Proposed",
    "status:planning": "Plan Review",
    "status:plan-pending": "Plan Review",
    "status:gate-pending": "Plan Review",
    "status:approved": "Approved",
    "status:implementing": "In Flight",
    "status:code-reviewing": "In Flight",
    "status:testing": "In Flight",
    "status:running": "In Flight",
    "status:uploading": "In Flight",
    "status:interpreting": "In Flight",
    "status:reviewing": "Sign-off",
    "status:under-review": "Sign-off",
    "status:awaiting-promotion": "Awaiting Promotion",
    "status:blocked": "Blocked",
    "status:done-experiment": "Done",
    "status:done-impl": "Done",
    "status:archived": "Done",
    # Non-status labels with column routing (take precedence over status:*).
    "clean-results": "Awaiting Promotion",
    "clean-results:draft": "Awaiting Promotion",
}

# Labels that take precedence over `status:*` routing in column_for_labels.
PRIORITY_LABELS: tuple[str, ...] = ("clean-results:draft", "clean-results")

# Target option set for `migrate-options`. Names + colors + descriptions.
# Color enum values per GitHub GraphQL ProjectV2SingleSelectFieldOptionColor.
NEW_COLUMN_SPEC: list[tuple[str, str, str]] = [
    ("Proposed", "BLUE", "User: review and approve idea"),
    ("Plan Review", "PURPLE", "User: review adversarial-planner output"),
    ("Approved", "GREEN", "User: dispatch via /issue N"),
    ("In Flight", "BLUE", "Automated: implementing/running/uploading/interpreting/reviewing"),
    ("Awaiting Promotion", "YELLOW", "User: promote clean-result via /clean-results promote N"),
    ("Sign-off", "YELLOW", "User: final OK on infra/code-change"),
    ("Blocked", "RED", "Resolve dependency"),
    ("Done", "GREEN", "Terminal state (experiment / impl / archived)"),
]


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


def project_meta(owner: str, number: int) -> ProjectMeta:
    """Fetch project node ID + Status field ID + name->StatusOption map.

    `gh project field-list` returns option ids+names but NOT colors or
    descriptions, so we use the raw GraphQL endpoint here. Colors must be
    preserved across `add-status-option` / `remove-status-option`
    invocations because the `updateProjectV2Field` mutation REPLACES the
    full options list — without round-tripping color the whole board's
    color coding is destroyed (HIGH-1, code-review v1).
    """
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

    return ProjectMeta(
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
        {"name": args.option, "color": args.color or "GRAY", "description": ""},
    ]
    options_json = json.dumps(new_options)
    mutation = (
        "mutation($fieldId:ID!, $options:[ProjectV2SingleSelectFieldOptionInput!]!) {"
        "  updateProjectV2Field(input:{"
        "    fieldId:$fieldId,"
        "    singleSelectOptions:$options"
        "  }) {"
        "    projectV2Field { ... on ProjectV2SingleSelectField { options { id name } } }"
        "  }"
        "}"
    )
    _gh(
        [
            "api",
            "graphql",
            "-f",
            f"query={mutation}",
            "-f",
            f"fieldId={meta.status_field_id}",
            "-f",
            f"options={options_json}",
        ]
    )
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
    options_json = json.dumps(remaining)
    mutation = (
        "mutation($fieldId:ID!, $options:[ProjectV2SingleSelectFieldOptionInput!]!) {"
        "  updateProjectV2Field(input:{"
        "    fieldId:$fieldId,"
        "    singleSelectOptions:$options"
        "  }) {"
        "    projectV2Field { ... on ProjectV2SingleSelectField { options { id name } } }"
        "  }"
        "}"
    )
    _gh(
        [
            "api",
            "graphql",
            "-f",
            f"query={mutation}",
            "-f",
            f"fieldId={meta.status_field_id}",
            "-f",
            f"options={options_json}",
        ]
    )
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
    raw = _gh(["issue", "view", str(issue), "-R", repo, "--json", "labels"])
    return [lbl["name"] for lbl in json.loads(raw)["labels"]]


def column_for_labels(labels: list[str]) -> str | None:
    """Return the column name for the issue's current labels, or None.

    Routing precedence (first match wins):
      1. PRIORITY_LABELS (clean-results, clean-results:draft) -> "Awaiting Promotion".
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
    """Run a GraphQL query/mutation via `gh api graphql --input`."""
    import tempfile
    from pathlib import Path

    body = {"query": query, "variables": variables or {}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        path = f.name
    try:
        proc = subprocess.run(
            ["gh", "api", "graphql", "--input", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(proc.returncode)
        data = json.loads(proc.stdout)
        if "errors" in data:
            sys.exit(json.dumps(data["errors"], indent=2))
        return data["data"]
    finally:
        Path(path).unlink(missing_ok=True)


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
    """
    q = """
    mutation($fieldId: ID!, $opts: [ProjectV2SingleSelectFieldOptionInput!]!) {
      updateProjectV2Field(input: {fieldId: $fieldId, singleSelectOptions: $opts}) {
        projectV2Field { ... on ProjectV2SingleSelectField { options { id name } } }
      }
    }
    """
    _graphql(q, {"fieldId": field_id, "opts": target})


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

    # Pre-fetch label sets for every issue in the repo (single paginated call).
    # `gh project item-list` does not return labels in the content payload, so we
    # build a number -> labels map from `gh issue list` and look up below.
    repo_for_labels = args.repo or current_repo()
    label_raw = _gh(
        [
            "issue",
            "list",
            "-R",
            repo_for_labels,
            "--state",
            "all",
            "--limit",
            "300",
            "--json",
            "number,labels",
        ]
    )
    labels_by_issue: dict[int, list[str]] = {
        item["number"]: [lbl["name"] for lbl in item["labels"]] for item in json.loads(label_raw)
    }

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
    """Return {title, body, labels, comments[]} for an issue."""
    raw = _gh(
        [
            "issue",
            "view",
            str(issue),
            "-R",
            repo,
            "--json",
            "title,body,labels,comments",
        ]
    )
    return json.loads(raw)


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

    p = sub.add_parser("add-status-option", help="add a new option to the Status field")
    p.add_argument("option", help="option name (e.g. 'Awaiting Promotion')")
    p.add_argument(
        "--color",
        default="GRAY",
        help="GRAY, BLUE, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE",
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
        "one-shot-migrate-legacy",
        help="interactive one-time migration of pre-Stage-2 clean-result issues",
    )
    p.set_defaults(func=cmd_one_shot_migrate_legacy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
