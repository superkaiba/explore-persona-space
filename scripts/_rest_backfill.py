"""One-shot REST API backfill for the 10-column board migration.

GraphQL is rate-limited; REST has a separate 5000/hr quota. This script
walks every project item, computes its target column from LABEL_TO_COLUMN,
and PATCHes any drifted items via the REST endpoint introduced in
GitHub's 2025-09 Projects v2 REST API rollout.

Usage:
    uv run python scripts/_rest_backfill.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_project import column_for_labels

PROJECT = "/users/superkaiba/projectsV2/1"
HEADER = ["-H", "X-GitHub-Api-Version: 2026-03-10"]


def gh_api(args: list[str]) -> dict | list:
    proc = subprocess.run(
        ["gh", "api", *HEADER, *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh api failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def list_all_items() -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        chunk = gh_api([f"{PROJECT}/items?per_page=100&page={page}"])
        if not chunk:
            break
        items.extend(chunk)
        if len(chunk) < 100:
            break
        page += 1
    return items


def get_status_field() -> dict:
    fields = gh_api([f"{PROJECT}/fields"])
    return next(f for f in fields if f.get("name") == "Status")


def main() -> int:
    print("Fetching project items via REST...", flush=True)
    items = list_all_items()
    status_field = get_status_field()
    field_id: int = status_field["id"]
    name_to_option = {opt["name"]["raw"]: opt["id"] for opt in status_field["options"]}
    print(f"  {len(items)} items, Status field id={field_id}, options={list(name_to_option)}")

    moved = skipped_match = no_label = unknown_col = 0
    for item in items:
        content = item.get("content") or {}
        if not content.get("number"):
            continue
        labels = [lab["name"] for lab in content.get("labels", []) or []]
        target = column_for_labels(labels)
        if target is None:
            no_label += 1
            continue
        if target not in name_to_option:
            print(f"  WARN #{content['number']}: target column {target!r} missing from board")
            unknown_col += 1
            continue
        # Current Status (if any).
        current = None
        for f in item.get("fields", []) or []:
            if f.get("name") == "Status" and isinstance(f.get("value"), dict):
                v = f["value"]
                current = (
                    v.get("name", {}).get("raw")
                    if isinstance(v.get("name"), dict)
                    else v.get("name")
                )
                break
        if current == target:
            skipped_match += 1
            continue
        # PATCH to target option.
        body = {"fields": [{"id": field_id, "value": name_to_option[target]}]}
        body_path = "/tmp/_patch_body.json"
        with open(body_path, "w") as f:
            json.dump(body, f)
        try:
            gh_api(
                [
                    "-X",
                    "PATCH",
                    f"{PROJECT}/items/{item['id']}",
                    "--input",
                    body_path,
                ]
            )
        except RuntimeError as exc:
            print(f"  ERR  #{content['number']}: {exc}")
            continue
        print(f"  #{content['number']}: {current or '<empty>'} -> {target}")
        moved += 1

    print(
        f"\nbackfill: moved={moved} already_correct={skipped_match} "
        f"no_routable_label={no_label} target_missing={unknown_col}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
