"""Tests for scripts/gh_project.py — focused on the color-preservation
invariant of `cmd_add_status_option` / `cmd_remove_status_option`.

Background (HIGH-1, code-review v1 on issue #226): the previous
implementation rebuilt the existing options list as
`[{"name": n, "color": "GRAY"} for n in meta.options]`, which destroyed
the board's color coding when the `updateProjectV2Field` mutation
REPLACED the full options list. These tests pin the corrected behaviour
so a future refactor cannot silently regress to GRAY-everything.

The tests mock `gh_project._gh` so no real `gh` CLI calls are made.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gh_project.py"

# Load the script as a module under a synthetic name so test imports work
# even though scripts/ has no __init__.py.
_spec = importlib.util.spec_from_file_location("gh_project", SCRIPT)
gh_project = importlib.util.module_from_spec(_spec)
sys.modules["gh_project"] = gh_project
_spec.loader.exec_module(gh_project)


@pytest.fixture(autouse=True)
def _disable_meta_cache(monkeypatch):
    """The project_meta() disk cache would let tests pick up state from a
    prior real CLI invocation; flip it off so every test exercises the
    GraphQL fetch path. Cache-specific tests below re-enable it
    explicitly via a context manager."""
    monkeypatch.setattr(gh_project, "_META_CACHE_DISABLED", True)


# A representative live-board response: 5 of 7 options are colored. If the
# rebuild logic ever passes color="GRAY" for these, the test fails.
_FAKE_FIELD_QUERY_RESPONSE = {
    "data": {
        "user": {
            "projectV2": {
                "id": "PVT_test_project_id",
                "field": {
                    "id": "PVTSSF_test_status_field_id",
                    "options": [
                        {"id": "opt_todo", "name": "Todo", "color": "GRAY", "description": ""},
                        {
                            "id": "opt_priority",
                            "name": "Priority",
                            "color": "PURPLE",
                            "description": "",
                        },
                        {
                            "id": "opt_inprog",
                            "name": "In Progress",
                            "color": "YELLOW",
                            "description": "",
                        },
                        {
                            "id": "opt_clean",
                            "name": "Clean Results",
                            "color": "GREEN",
                            "description": "",
                        },
                        {
                            "id": "opt_done_exp",
                            "name": "Done (experiment)",
                            "color": "GREEN",
                            "description": "",
                        },
                        {
                            "id": "opt_done_impl",
                            "name": "Done (impl)",
                            "color": "GREEN",
                            "description": "test note",
                        },
                        {
                            "id": "opt_archived",
                            "name": "Archived",
                            "color": "GRAY",
                            "description": "",
                        },
                    ],
                },
            }
        }
    }
}


def _read_input_body(args: list[str]) -> dict | None:
    """If argv carries `--input <tempfile>`, read+parse the JSON body.

    The mutation path (`_replace_options` → `_graphql` → `_gh`) sends the
    GraphQL query+variables via `gh api graphql --input <tempfile>`
    because the `singleSelectOptions` variable is a typed JSON array that
    `-f`/`-F` cannot encode. The recorder reads the tempfile while it
    still exists (during the `_gh` call) so tests can inspect the body.
    """
    for i, a in enumerate(args):
        if a == "--input" and i + 1 < len(args):
            with open(args[i + 1]) as f:
                return json.load(f)
    return None


class _GhRecorder:
    """Replacement for `gh_project._gh` that records every call.

    The first call (the GraphQL query inside `project_meta`) returns the
    canned options list. Subsequent calls (the mutation issued by the
    add/remove command) are recorded so the test can inspect what
    payload would have been sent to the GitHub API.

    Two argv shapes are supported:
      1. `project_meta` query: `["api", "graphql", "-f", "query=...", ...]`
         — distinguished by the `query=` flag.
      2. Mutation: `["api", "graphql", "--input", "<tempfile>"]` — the
         tempfile holds `{"query": "...", "variables": {...}}`. The
         recorder reads the file (it still exists during the `_gh` call)
         and stashes the parsed body on the call record.
    """

    def __init__(self, query_response: dict) -> None:
        self._query_response = query_response
        self.calls: list[list[str]] = []
        # Parallel list: parsed `--input` body for each call (or None).
        self.input_bodies: list[dict | None] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(list(args))
        body = _read_input_body(args)
        self.input_bodies.append(body)
        if args[:2] == ["api", "graphql"]:
            # Mutation via `--input <tempfile>` (typed JSON variables).
            if body is not None and "updateProjectV2Field" in body.get("query", ""):
                return json.dumps({"data": {"updateProjectV2Field": {}}})
            # Legacy `-f query=...` path (still used by `project_meta`).
            for a in args:
                if a.startswith("query=") and "updateProjectV2Field" in a:
                    return json.dumps({"data": {"updateProjectV2Field": {}}})
            return json.dumps(self._query_response)
        return ""


def _mutation_payload(body: dict) -> list[dict]:
    """Extract the JSON-encoded options list from a recorded mutation body.

    Pairs with `_mutation_call` — call as `_mutation_payload(_mutation_call(rec))`.
    The body is the parsed `--input` JSON: `{"query": "...", "variables": {...}}`.
    """
    opts = body.get("variables", {}).get("opts")
    if opts is None:
        raise AssertionError(f"mutation body has no `opts` variable: {body}")
    return opts


def _mutation_call(recorder: _GhRecorder) -> dict:
    """Find the recorded mutation body (the one carrying updateProjectV2Field).

    Returns the parsed `--input` JSON body. The mutation path delegates to
    `gh api graphql --input <tempfile>` because typed JSON arrays cannot
    travel through `-f`/`-F`. The recorder stashes the body when the call
    is made (the tempfile is deleted in `_graphql`'s `finally` block).
    """
    for body in recorder.input_bodies:
        if body is None:
            continue
        if "updateProjectV2Field" in body.get("query", ""):
            return body
    raise AssertionError(f"no updateProjectV2Field call recorded; saw: {recorder.calls}")


# --- project_meta -----------------------------------------------------------


def test_project_meta_returns_colors(monkeypatch):
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    meta = gh_project.project_meta("superkaiba", 1)

    assert meta.project_id == "PVT_test_project_id"
    assert meta.status_field_id == "PVTSSF_test_status_field_id"
    assert meta.options["Priority"].color == "PURPLE"
    assert meta.options["In Progress"].color == "YELLOW"
    assert meta.options["Clean Results"].color == "GREEN"
    assert meta.options["Done (experiment)"].color == "GREEN"
    assert meta.options["Done (impl)"].color == "GREEN"
    assert meta.options["Done (impl)"].description == "test note"
    assert meta.options["Todo"].color == "GRAY"


# --- cmd_add_status_option --------------------------------------------------


def test_add_status_option_preserves_existing_colors(monkeypatch):
    """HIGH-1 regression: rebuilding the options list must NOT reset
    every existing option to GRAY. Each existing option must round-trip
    its actual color through the mutation payload."""
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="Draft Clean Results",
        color="ORANGE",
    )
    gh_project.cmd_add_status_option(args)

    payload = _mutation_payload(_mutation_call(rec))
    by_name = {opt["name"]: opt for opt in payload}

    # The new option appears with the requested color.
    assert by_name["Draft Clean Results"]["color"] == "ORANGE"

    # Every pre-existing colored option survives WITH ITS COLOR — the
    # whole point of the fix.
    assert by_name["Priority"]["color"] == "PURPLE"
    assert by_name["In Progress"]["color"] == "YELLOW"
    assert by_name["Clean Results"]["color"] == "GREEN"
    assert by_name["Done (experiment)"]["color"] == "GREEN"
    assert by_name["Done (impl)"]["color"] == "GREEN"
    assert by_name["Todo"]["color"] == "GRAY"
    assert by_name["Archived"]["color"] == "GRAY"

    # Description round-trips for any option that had one.
    assert by_name["Done (impl)"]["description"] == "test note"


def test_add_status_option_idempotent_when_already_exists(monkeypatch):
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="Priority",
        color="GREEN",  # different from existing PURPLE — should still no-op
    )
    gh_project.cmd_add_status_option(args)

    # Only the project_meta query happens; no mutation.
    mutation_calls = [
        c
        for c in rec.calls
        if c[:2] == ["api", "graphql"]
        and any(a.startswith("query=") and "updateProjectV2Field" in a for a in c)
    ]
    assert mutation_calls == []


# --- cmd_remove_status_option -----------------------------------------------


def test_remove_status_option_preserves_surviving_colors(monkeypatch):
    """HIGH-1 regression for the inverse path: removing one option must
    not reset every survivor to GRAY."""
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="Archived",  # delete the GRAY one
    )
    gh_project.cmd_remove_status_option(args)

    payload = _mutation_payload(_mutation_call(rec))
    by_name = {opt["name"]: opt for opt in payload}

    # Removed option is gone.
    assert "Archived" not in by_name

    # All survivors keep their original colors.
    assert by_name["Priority"]["color"] == "PURPLE"
    assert by_name["In Progress"]["color"] == "YELLOW"
    assert by_name["Clean Results"]["color"] == "GREEN"
    assert by_name["Done (experiment)"]["color"] == "GREEN"
    assert by_name["Done (impl)"]["color"] == "GREEN"
    assert by_name["Todo"]["color"] == "GRAY"

    # Description is preserved.
    assert by_name["Done (impl)"]["description"] == "test note"


def test_remove_status_option_no_op_when_missing(monkeypatch):
    rec = _GhRecorder(_FAKE_FIELD_QUERY_RESPONSE)
    monkeypatch.setattr(gh_project, "_gh", rec)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        option="DoesNotExist",
    )
    gh_project.cmd_remove_status_option(args)

    mutation_calls = [
        c
        for c in rec.calls
        if c[:2] == ["api", "graphql"]
        and any(a.startswith("query=") and "updateProjectV2Field" in a for a in c)
    ]
    assert mutation_calls == []


# --- cmd_list_all -----------------------------------------------------------

_FAKE_ITEM_LIST_RESPONSE = {
    "totalCount": 5,
    "items": [
        {
            "id": "PVTI_a",
            "status": "To do",
            "content": {"number": 100, "title": "first todo", "repository": "owner/repo"},
        },
        {
            "id": "PVTI_b",
            "status": "Planning",
            "content": {"number": 200, "title": "in planning", "repository": "owner/repo"},
        },
        {
            "id": "PVTI_c",
            "status": "To do",
            "content": {"number": 101, "title": "second todo", "repository": "owner/repo"},
        },
        {
            "id": "PVTI_d",
            "status": "Awaiting promotion",
            "content": {"number": 300, "title": "ready to promote", "repository": "owner/repo"},
        },
        {
            "id": "PVTI_e",
            "status": "Done",
            "content": {"number": 400, "title": "finished", "repository": "owner/repo"},
        },
    ],
}


def test_list_all_groups_by_column_in_one_call(monkeypatch, capsys):
    """list-all must make exactly ONE `gh project item-list` call and group
    items by their Status column client-side. Counted via call-recording on
    `_gh`."""
    calls: list[list[str]] = []

    def fake_gh(args: list[str]) -> str:
        calls.append(list(args))
        if args[:2] == ["project", "item-list"]:
            return json.dumps(_FAKE_ITEM_LIST_RESPONSE)
        raise AssertionError(f"unexpected gh call: {args}")

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        columns=None,
        json=False,
        counts_only=False,
    )
    gh_project.cmd_list_all(args)

    # Exactly one API call — no project_meta(), no per-column queries.
    item_list_calls = [c for c in calls if c[:2] == ["project", "item-list"]]
    assert len(item_list_calls) == 1, calls
    assert len(calls) == 1, f"expected ONE call total, got {calls!r}"

    out = capsys.readouterr().out
    # All five issues surface under their respective columns.
    assert "### To do (2)" in out
    assert "#100 first todo" in out
    assert "#101 second todo" in out
    assert "### Planning (1)" in out
    assert "### Awaiting promotion (1)" in out
    assert "### Done (1)" in out


def test_list_all_columns_filter(monkeypatch, capsys):
    """--columns "A,B" narrows the output but still uses one API call."""
    calls: list[list[str]] = []

    def fake_gh(args: list[str]) -> str:
        calls.append(list(args))
        return json.dumps(_FAKE_ITEM_LIST_RESPONSE)

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        columns="To do,Awaiting promotion",
        json=False,
        counts_only=False,
    )
    gh_project.cmd_list_all(args)

    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "### To do (2)" in out
    assert "### Awaiting promotion (1)" in out
    assert "Planning" not in out
    assert "Done" not in out


def test_list_all_json_output_keyed_by_column(monkeypatch, capsys):
    def fake_gh(args: list[str]) -> str:
        return json.dumps(_FAKE_ITEM_LIST_RESPONSE)

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        columns=None,
        json=True,
        counts_only=False,
    )
    gh_project.cmd_list_all(args)

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["To do"] == [
        {"number": 100, "title": "first todo", "repo": "owner/repo"},
        {"number": 101, "title": "second todo", "repo": "owner/repo"},
    ]
    assert parsed["Awaiting promotion"] == [
        {"number": 300, "title": "ready to promote", "repo": "owner/repo"},
    ]


def test_list_all_counts_only(monkeypatch, capsys):
    def fake_gh(args: list[str]) -> str:
        return json.dumps(_FAKE_ITEM_LIST_RESPONSE)

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    args = argparse.Namespace(
        owner="superkaiba",
        project=1,
        columns=None,
        json=False,
        counts_only=True,
    )
    gh_project.cmd_list_all(args)

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line]
    assert "To do\t2" in lines
    assert "Planning\t1" in lines
    assert "Awaiting promotion\t1" in lines
    assert "Done\t1" in lines


# --- _fetch_all_issue_labels (REST pagination) ------------------------------


def test_fetch_all_issue_labels_paginates_until_done(monkeypatch):
    """REST + `gh api --paginate` concatenates each page's JSON array
    without a separator. The helper must parse the concatenated stream
    and merge every page — verified here by simulating two pages.

    Issue 305 (past the old 300-row cap) must land in the merged dict
    so the prior silent-truncation bug stays fixed."""

    page1 = [
        {"number": 1, "labels": [{"name": "status:done-experiment"}]},
        {"number": 2, "labels": [{"name": "status:proposed"}]},
    ]
    page2 = [
        {
            "number": 305,
            "labels": [
                {"name": "status:running"},
                {"name": "type:experiment"},
            ],
        },
    ]
    # `gh api --paginate` glues page arrays back-to-back with no
    # separator; this mirrors what the helper sees on stdout.
    fake_paginated_output = json.dumps(page1) + json.dumps(page2)

    calls: list[list[str]] = []

    def fake_gh(args: list[str]) -> str:
        calls.append(list(args))
        return fake_paginated_output

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    result = gh_project._fetch_all_issue_labels("superkaiba/explore-persona-space")

    # Single _gh invocation; gh CLI handles the actual REST round-trips.
    assert len(calls) == 1
    args = calls[0]
    assert args[0] == "api"
    assert any("repos/superkaiba/explore-persona-space/issues" in a for a in args)
    assert "--paginate" in args
    assert "graphql" not in args, "_fetch_all_issue_labels must NOT use GraphQL"

    assert 305 in result
    assert result[305] == ["status:running", "type:experiment"]
    assert result[1] == ["status:done-experiment"]
    assert result[2] == ["status:proposed"]


def test_fetch_all_issue_labels_filters_pull_requests(monkeypatch):
    """REST's issues endpoint conflates issues and PRs. PRs carry a
    `pull_request` key that issues don't — they must be filtered so PR
    numbers don't pollute the issue→labels map (and so PR labels don't
    override an issue with the same number)."""
    fake_response = json.dumps(
        [
            {"number": 7, "labels": [{"name": "type:infra"}]},
            {
                "number": 8,
                "labels": [{"name": "type:pr-only"}],
                "pull_request": {"url": "..."},
            },
        ]
    )

    monkeypatch.setattr(gh_project, "_gh", lambda args: fake_response)

    result = gh_project._fetch_all_issue_labels("superkaiba/explore-persona-space")

    assert result == {7: ["type:infra"]}
    assert 8 not in result


def test_fetch_all_issue_labels_single_page(monkeypatch):
    """One-page response → one call, single merge."""
    response = json.dumps([{"number": 7, "labels": [{"name": "type:infra"}]}])

    calls: list[list[str]] = []

    def fake_gh(args: list[str]) -> str:
        calls.append(list(args))
        return response

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    result = gh_project._fetch_all_issue_labels("superkaiba/explore-persona-space")

    assert len(calls) == 1
    assert result == {7: ["type:infra"]}


# --- project_meta() disk cache ----------------------------------------------


def test_project_meta_caches_to_disk_and_serves_from_cache(monkeypatch, tmp_path):
    """Two back-to-back project_meta() calls should hit the API once.

    The second call must return the same object reconstituted from disk,
    not trigger another `_gh` call. This is the win the cache exists
    for — every CLI invocation pays one graphql roundtrip; chaining
    invocations within the TTL window skip the second.
    """
    # Re-enable disk cache; redirect it to tmp_path so we don't touch
    # the real .claude/cache/ directory.
    monkeypatch.setattr(gh_project, "_META_CACHE_DISABLED", False)
    monkeypatch.setattr(gh_project, "_REPO_ROOT", tmp_path)

    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(list(args))
        return json.dumps(_FAKE_FIELD_QUERY_RESPONSE)

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    m1 = gh_project.project_meta("superkaiba", 1)
    m2 = gh_project.project_meta("superkaiba", 1)

    assert len(calls) == 1, f"expected ONE graphql call, got {calls!r}"
    assert m1.project_id == m2.project_id
    assert m1.status_field_id == m2.status_field_id
    assert set(m1.options) == set(m2.options)
    # Cache file landed on disk.
    cache_path = tmp_path / ".claude" / "cache" / "gh-project-meta-superkaiba-1.json"
    assert cache_path.exists()


def test_project_meta_cache_expires_after_ttl(monkeypatch, tmp_path):
    """A stale cache (older than TTL) must NOT be used — fetch again."""
    monkeypatch.setattr(gh_project, "_META_CACHE_DISABLED", False)
    monkeypatch.setattr(gh_project, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("EPM_GH_PROJECT_META_TTL", "60")

    # Write a pre-expired cache file directly.
    cache_path = tmp_path / ".claude" / "cache" / "gh-project-meta-superkaiba-1.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cached_at": 0,  # 1970 — definitely older than 60s ago
                "project_id": "STALE_id",
                "status_field_id": "STALE_field",
                "options": {},
            }
        )
    )

    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(list(args))
        return json.dumps(_FAKE_FIELD_QUERY_RESPONSE)

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    meta = gh_project.project_meta("superkaiba", 1)

    assert len(calls) == 1, "stale cache must trigger a refetch"
    # Result is from the fresh fetch, not the stale file.
    assert meta.project_id == "PVT_test_project_id"
    assert meta.status_field_id == "PVTSSF_test_status_field_id"


def test_replace_options_invalidates_meta_cache(monkeypatch, tmp_path):
    """Any option-mutating command (add/remove/migrate) routes through
    `_replace_options`. After a successful mutation the cache MUST be
    dropped so the next read sees the new option set (with possibly
    new option_ids for fresh entries)."""
    monkeypatch.setattr(gh_project, "_META_CACHE_DISABLED", False)
    monkeypatch.setattr(gh_project, "_REPO_ROOT", tmp_path)

    # Seed the cache.
    cache_path = tmp_path / ".claude" / "cache" / "gh-project-meta-superkaiba-1.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cached_at": 9999999999,  # far future — would be "fresh"
                "project_id": "cached_id",
                "status_field_id": "cached_field",
                "options": {},
            }
        )
    )
    assert cache_path.exists()

    # Mutation call routes through _gh — fake it.
    monkeypatch.setattr(gh_project, "_gh", lambda args: json.dumps({"data": {}}))

    gh_project._replace_options("any_field_id", [])

    assert not cache_path.exists(), "_replace_options must invalidate cache after mutation"


def test_meta_cache_disabled_via_ttl_zero(monkeypatch, tmp_path):
    """Setting `EPM_GH_PROJECT_META_TTL=0` disables caching entirely —
    every call fetches fresh."""
    monkeypatch.setattr(gh_project, "_META_CACHE_DISABLED", False)
    monkeypatch.setattr(gh_project, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("EPM_GH_PROJECT_META_TTL", "0")

    calls: list[list[str]] = []

    def fake_gh(args):
        calls.append(list(args))
        return json.dumps(_FAKE_FIELD_QUERY_RESPONSE)

    monkeypatch.setattr(gh_project, "_gh", fake_gh)

    gh_project.project_meta("superkaiba", 1)
    gh_project.project_meta("superkaiba", 1)

    assert len(calls) == 2, "TTL=0 must skip the cache"
    cache_path = tmp_path / ".claude" / "cache" / "gh-project-meta-superkaiba-1.json"
    assert not cache_path.exists(), "TTL=0 must not write to disk either"
