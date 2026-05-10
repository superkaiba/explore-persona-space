#!/usr/bin/env python3
"""Fetch issue state (body + labels + state + LAST-N comments) via REST.

Drop-in replacement for `gh issue view <N> --json number,title,body,labels,
state,assignees,comments` whose output shape matches the gh CLI exactly,
but routes through the REST API instead of GraphQL.

Why REST and not GraphQL: GitHub's per-hour buckets are independent — the
core (REST) bucket is 5000/hr and the GraphQL bucket is also 5000/hr but
separately counted. On 2026-05-10 we exhausted the GraphQL bucket on PM
triage and stalled the workflow even though the core bucket was empty.
Issue/PR reads have full REST coverage; project-board reads (which our
`/pm` triage drives) do NOT have user-scope REST coverage, so the board
stays on GraphQL. Routing the heavy issue-state reads here through REST
gives the board's GraphQL bucket headroom for the operations that need it.

Cost shape: 2 REST calls per invocation (issue endpoint + comments
endpoint with `--paginate` for threads >100 comments). On long-thread
issues (#80 with ~100+ comments) this is ~2 REST calls vs 1 GraphQL —
trading a different bucket for two cheap ones is a clear win when the
GraphQL bucket is the bottleneck.

Usage:
  python scripts/gh_issue_state.py <issue_number> [--repo owner/name]
                                                  [--comments-last 50]
                                                  [--no-comments]

Output: same JSON shape as `gh issue view --json
number,title,body,labels,state,assignees,comments`, printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _current_repo() -> str:
    """Read the origin remote URL from local git and parse owner/name.

    The obvious alternative — `gh repo view --json nameWithOwner` — costs
    one GraphQL call per invocation (verified 2026-05-10: GraphQL bucket
    dropped from 4686→4685 on a single fetch). Parsing git locally is
    free. Supports both SSH (`git@github.com:owner/repo.git`) and HTTPS
    (`https://github.com/owner/repo.git`) origin URLs.
    """
    proc = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)
    url = proc.stdout.strip()

    for prefix in ("git@github.com:", "https://github.com/", "ssh://git@github.com/"):
        if url.startswith(prefix):
            tail = url[len(prefix) :]
            break
    else:
        sys.exit(f"could not parse owner/name from origin URL: {url!r}")
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    return tail


def _parse_concatenated_arrays(raw: str) -> list:
    """Parse `gh api --paginate` output for an array endpoint.

    `gh api --paginate` walks every page and concatenates each page's JSON
    body without inserting separators — so a two-page result looks like
    `[a,b][c,d]`. That's not a single valid JSON document; we walk it
    with `json.JSONDecoder.raw_decode` to recover each page array, then
    flatten.
    """
    raw = raw.strip()
    if not raw:
        return []
    decoder = json.JSONDecoder()
    items: list = []
    idx = 0
    while idx < len(raw):
        page, end = decoder.raw_decode(raw[idx:])
        if isinstance(page, list):
            items.extend(page)
        else:
            items.append(page)
        idx += end
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
    return items


def _gh_api(path: str, paginate: bool = False) -> str:
    cmd = ["gh", "api", "-H", "Accept: application/vnd.github+json", path]
    if paginate:
        cmd.append("--paginate")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)
    return proc.stdout


def fetch(issue: int, repo: str, comments_last: int, include_comments: bool) -> dict:
    """Fetch issue + (optionally) the last N comments via REST.

    Returns a dict matching `gh issue view --json
    number,title,body,labels,state,assignees,comments` so callers don't
    need to know the response came from REST.
    """
    issue_data = json.loads(_gh_api(f"repos/{repo}/issues/{issue}"))

    out: dict = {
        "number": issue_data["number"],
        "title": issue_data["title"],
        "body": issue_data.get("body") or "",
        # REST's `state` is already lowercase ("open"/"closed"), matching
        # the gh CLI's `--json state` projection.
        "state": issue_data["state"],
        "labels": [{"name": lbl["name"]} for lbl in issue_data.get("labels", [])],
        "assignees": [{"login": a["login"]} for a in issue_data.get("assignees", [])],
    }

    if not include_comments:
        return out

    total_comments = issue_data.get("comments", 0)
    if total_comments == 0:
        out["comments"] = []
        return out

    # REST list-comments returns oldest-first, paginated 100/page. We
    # fetch with --paginate then slice the tail. On a 250-comment issue
    # this is 3 REST calls; on the typical <100-comment case it's 1.
    raw = _gh_api(f"repos/{repo}/issues/{issue}/comments?per_page=100", paginate=True)
    all_comments = _parse_concatenated_arrays(raw)
    tail = all_comments[-comments_last:] if comments_last > 0 else all_comments
    out["comments"] = [
        {
            "author": {"login": (c.get("user") or {}).get("login", "")},
            # The gh CLI's `--json comments` projection uses camelCase
            # `createdAt` / `updatedAt`; REST uses snake_case. Translate
            # so downstream parsers don't need to know the source.
            "body": c["body"],
            "createdAt": c["created_at"],
            "updatedAt": c["updated_at"],
            "url": c["html_url"],
        }
        for c in tail
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("issue", type=int)
    ap.add_argument("--repo", help="owner/name (default: current repo via `gh repo view`)")
    ap.add_argument(
        "--comments-last",
        type=int,
        default=50,
        help="cap comments to the most-recent N (default: 50). 0 means all.",
    )
    ap.add_argument(
        "--no-comments",
        action="store_true",
        help="omit the `comments` field entirely (lightest payload — 1 REST call)",
    )
    args = ap.parse_args()

    repo = args.repo or _current_repo()
    if not repo:
        sys.exit("could not infer current repo; pass --repo owner/name")

    include = not args.no_comments
    print(json.dumps(fetch(args.issue, repo, args.comments_last, include), indent=2))


if __name__ == "__main__":
    main()
