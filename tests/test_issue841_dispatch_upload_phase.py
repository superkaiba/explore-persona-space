"""Dispatch-script upload-phase invariant pin (issue #841 crash-fix cycle 6, att-7).

Attempt 7 completed ALL phases cleanly but the small result artifacts (the stage
JSONs, the projections npz, the plot PNGs) lived only on the boot disk and were
destroyed by the instance's clean-exit DELETE — neither dispatch script uploaded
them. The capture shards + per-n maps were already safe on the overflow repo.

These are TEXT/GREP pins over the two dispatch shell scripts (no bats, no execution)
asserting the two structural invariants the fix added, so a future silent removal is
caught:

  * each dispatch has an ``upload`` phase (a ``log_phase upload`` line) that appears
    AFTER the ``plots`` phase and BEFORE both the end-of-run sentinel write and the
    terminal ``log_phase done`` / ``[phase=done]`` line — so the small artifacts are
    persisted before the run reports success + the pod is DELETEd;
  * the plots invocation carries NO ``|| true`` swallow (a plot failure must abort the
    run loud, before the upload + sentinel, not silently ship a run with the required
    hero / per-unit figures missing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

DISPATCH_SCRIPTS = [
    "issue841_scaling_dispatch.sh",
    "issue841_gru_source_only_dispatch.sh",
]


def _lines(name: str) -> list[str]:
    return (SCRIPTS / name).read_text().splitlines()


def _first_index(lines: list[str], predicate) -> int:
    for i, ln in enumerate(lines):
        if predicate(ln):
            return i
    return -1


@pytest.mark.parametrize("script", DISPATCH_SCRIPTS)
def test_upload_phase_present_before_sentinel_and_done(script: str) -> None:
    """The upload phase runs after plots and before the sentinel write + terminal
    [phase=done] line, so the small result artifacts are persisted while the pod is
    still alive."""
    lines = _lines(script)

    plots_i = _first_index(lines, lambda ln: "log_phase plots" in ln)
    upload_i = _first_index(lines, lambda ln: "log_phase upload" in ln)
    done_i = _first_index(lines, lambda ln: "log_phase done" in ln)
    # the sentinel filename is assigned before it is written + before [phase=done]
    sentinel_i = _first_index(lines, lambda ln: ln.strip().startswith("SENTINEL="))

    assert plots_i >= 0, f"{script}: no 'log_phase plots' line"
    assert upload_i >= 0, f"{script}: no 'log_phase upload' line — upload phase missing"
    assert done_i >= 0, f"{script}: no 'log_phase done' line"
    assert sentinel_i >= 0, f"{script}: no SENTINEL= assignment"

    assert plots_i < upload_i, f"{script}: upload phase must come AFTER plots"
    assert upload_i < sentinel_i, (
        f"{script}: upload phase must come BEFORE the sentinel write "
        f"(so artifacts persist before the run reports done)"
    )
    assert upload_i < done_i, f"{script}: upload phase must come BEFORE the terminal [phase=done]"


@pytest.mark.parametrize("script", DISPATCH_SCRIPTS)
def test_plots_invocation_not_swallowed(script: str) -> None:
    """No `|| true` on the plots invocation — a plot failure must abort the run loud,
    before the upload + sentinel, never silently ship a run with the required figures
    missing."""
    text = (SCRIPTS / script).read_text()
    for ln in text.splitlines():
        if "_plots.py" in ln and ln.strip().startswith("uv run python"):
            assert "|| true" not in ln, (
                f"{script}: plots invocation swallows failures with '|| true' — "
                f"remove it so a plot failure aborts before upload/sentinel: {ln.strip()!r}"
            )


@pytest.mark.parametrize("script", DISPATCH_SCRIPTS)
def test_upload_phase_is_fail_loud(script: str) -> None:
    """The upload phase's uploader invocation carries no `|| true` swallow — an
    unverified upload must abort before the sentinel (set -e), so a silent artifact
    loss cannot happen."""
    lines = _lines(script)
    upload_i = _first_index(lines, lambda ln: "log_phase upload" in ln)
    sentinel_i = _first_index(lines, lambda ln: ln.strip().startswith("SENTINEL="))
    assert upload_i >= 0 and sentinel_i > upload_i, f"{script}: upload phase not located"
    # the uploader python invocation lives between the upload phase log line and the sentinel
    block = "\n".join(lines[upload_i:sentinel_i])
    assert "uv run python" in block, f"{script}: upload phase runs no uploader"
    assert "|| true" not in block, (
        f"{script}: upload phase swallows failures with '|| true' — the upload must be "
        f"fail-loud so an unverified upload aborts before the sentinel"
    )
