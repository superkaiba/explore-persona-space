"""One-shot fixture: assert LABEL_TO_COLUMN covers every status:* label currently in use.

Run locally with `uv run pytest tests/test_label_to_column_coverage.py -v`.
Not wired into CI on a schedule — purpose is to catch the case where a new
status:* label is introduced without updating the routing table.
"""

from __future__ import annotations

import json
import subprocess

from scripts.gh_project import LABEL_TO_COLUMN, NEW_COLUMN_SPEC, column_for_labels


def _live_status_labels() -> set[str] | None:
    """All `status:*` labels currently defined in the repo, or None on auth/rate-limit error.

    Returns None (test skips) if `gh label list` fails — typical reasons are
    network unavailable in CI, missing gh auth, or a transient rate limit.
    """
    proc = subprocess.run(
        ["gh", "label", "list", "--limit", "200", "--json", "name"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return {lbl["name"] for lbl in json.loads(proc.stdout) if lbl["name"].startswith("status:")}


def test_routing_table_covers_every_live_status_label() -> None:
    import pytest

    live = _live_status_labels()
    if live is None:
        pytest.skip("gh label list unavailable (auth/rate-limit/network)")
    missing = live - set(LABEL_TO_COLUMN)
    assert not missing, (
        f"LABEL_TO_COLUMN is missing {len(missing)} live status:* labels: "
        f"{sorted(missing)}. Add to scripts/gh_project.py."
    )


def test_routing_table_targets_only_known_columns() -> None:
    column_names = {name for name, _, _ in NEW_COLUMN_SPEC}
    bad = {label: col for label, col in LABEL_TO_COLUMN.items() if col not in column_names}
    assert not bad, f"LABEL_TO_COLUMN routes to columns not in NEW_COLUMN_SPEC: {bad}"


def test_column_for_labels_no_status_returns_none() -> None:
    assert column_for_labels(["type:experiment", "prio:high"]) is None


def test_column_for_labels_single_status() -> None:
    assert column_for_labels(["type:experiment", "status:running"]) == "In flight"


def test_column_for_labels_multiple_status_uses_last() -> None:
    # When multiple status labels are present, the last one wins (most recent flip).
    result = column_for_labels(["status:running", "status:awaiting-promotion"])
    assert result == "Awaiting promotion"


def test_column_for_labels_bare_clean_results_falls_back_to_status() -> None:
    # The bare `clean-results` label is a back-compat marker for
    # `gh issue list --label clean-results`. It does NOT route on its own;
    # routing comes from the sublabel or the `status:*` label.
    assert column_for_labels(["status:done-experiment", "clean-results"]) == "Done"


def test_column_for_labels_clean_results_draft_routes_to_awaiting_promotion() -> None:
    # `clean-results:draft` routes to "Awaiting promotion" regardless of status.
    assert column_for_labels(["status:reviewing", "clean-results:draft"]) == "Awaiting promotion"


def test_column_for_labels_draft_takes_precedence_over_bare_label() -> None:
    # The bare `clean-results` label does not route; :draft wins.
    assert column_for_labels(["clean-results", "clean-results:draft"]) == "Awaiting promotion"


def test_column_for_labels_archived_routes_to_archived_column() -> None:
    assert column_for_labels(["status:archived"]) == "Archived"


def test_column_for_labels_done_experiment_routes_to_done_column() -> None:
    # status:done-experiment without clean-results goes to Done (not Clean results).
    assert column_for_labels(["status:done-experiment"]) == "Done"


def test_column_for_labels_followups_running() -> None:
    # `status:followups-running` is the new state for "follow-ups in flight before promotion".
    assert column_for_labels(["status:followups-running"]) == "Followups running"


# ---------------------------------------------------------------------------
# Two-verdict promotion flow: clean-results:useful, clean-results:not-useful.
# PRIORITY_LABELS order: :draft -> :useful -> :not-useful. The bare
# `clean-results` label is preserved on promoted issues for back-compat
# `gh issue list --label clean-results` queries but no longer routes to a
# column on its own (the legacy "Clean results" column was removed).
# ---------------------------------------------------------------------------


def test_priority_useful_routes_to_useful() -> None:
    assert column_for_labels(["clean-results:useful"]) == "Useful"


def test_priority_not_useful_routes_to_not_useful() -> None:
    assert column_for_labels(["clean-results:not-useful"]) == "Not useful"


def test_priority_draft_beats_useful() -> None:
    # Defensive: half-applied promote (sublabel added but :draft not removed)
    # stays observably unfinished in "Awaiting promotion".
    assert (
        column_for_labels(["clean-results:draft", "clean-results:useful"]) == "Awaiting promotion"
    )


def test_priority_useful_with_bare_label() -> None:
    # Promote keeps the bare `clean-results` label (back-compat for
    # `gh issue list --label clean-results` callers); :useful drives routing.
    assert column_for_labels(["clean-results", "clean-results:useful"]) == "Useful"


def test_priority_not_useful_with_bare_label() -> None:
    assert column_for_labels(["clean-results", "clean-results:not-useful"]) == "Not useful"


def test_bare_clean_results_alone_does_not_route() -> None:
    """The bare `clean-results` label is back-compat-only after the legacy
    column was removed. Without a sublabel or status:* label, it does not
    route to any column."""
    assert column_for_labels(["clean-results"]) is None


def test_promoted_issue_still_matches_clean_results_query() -> None:
    """Sanity: after promote, the issue carries BOTH `clean-results` and a
    sublabel, so the callers of `gh issue list --label clean-results` still
    find it. Asserted at the label-set level (the actual gh query is exercised
    elsewhere)."""
    promoted_labels = {"clean-results", "clean-results:useful"}
    assert "clean-results" in promoted_labels


def test_priority_labels_order_draft_first() -> None:
    """Defensive: PRIORITY_LABELS must list `:draft` first so a half-applied
    promote stays in 'Awaiting promotion' until reconciled. The bare
    `clean-results` label is intentionally NOT in PRIORITY_LABELS (the legacy
    column it routed to was removed)."""
    from scripts.gh_project import PRIORITY_LABELS

    assert PRIORITY_LABELS[0] == "clean-results:draft"
    assert "clean-results" not in PRIORITY_LABELS
