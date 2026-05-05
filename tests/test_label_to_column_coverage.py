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
    assert column_for_labels(["type:experiment", "aim:1-geometry"]) is None


def test_column_for_labels_single_status() -> None:
    assert column_for_labels(["type:experiment", "status:running"]) == "In Flight"


def test_column_for_labels_multiple_status_uses_last() -> None:
    # When multiple status labels are present, the last one wins (most recent flip).
    result = column_for_labels(["status:running", "status:awaiting-promotion"])
    assert result == "Awaiting Promotion"


def test_column_for_labels_clean_results_takes_precedence() -> None:
    # `clean-results` routes to "Awaiting Promotion" regardless of status:* label.
    assert column_for_labels(["status:done-experiment", "clean-results"]) == "Awaiting Promotion"


def test_column_for_labels_clean_results_draft_takes_precedence() -> None:
    # `clean-results:draft` also routes to "Awaiting Promotion".
    assert column_for_labels(["status:reviewing", "clean-results:draft"]) == "Awaiting Promotion"
