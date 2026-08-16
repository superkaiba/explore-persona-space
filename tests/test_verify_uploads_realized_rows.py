"""Tests for the #2148 realized row-count reconciliation check.

Task #2091's upload verification PASSed while ~25% of activation-capture
rows were missing INSIDE present files: every file-level check resolved by
path and byte size, and the row-count check compared the producer's
self-reported ``capture_rows`` — literally ``manifest.get("n_rows")``, the
input-side expectation echoed back — against that same expectation.
``check_realized_row_counts`` counts what is REALLY in the store's own
``row_index*.jsonl`` files and gates on the DISTINCT count of the declared
FULL row key. These tests pin:

- the composite-key requirement (§6.3): a missing rollout FAILs
  ``realized-rows-short`` at unit-coverage-complete stores; a duplicated
  full-key pair PASSes with the duplicate count reported (a line-count gate
  would FAIL every healthy repaired store — #2091 post-repair holds 2048
  lines / 2000 distinct rows);
- the three attribution ERROR arms firing BEFORE any counting
  (``row-index-missing`` / ``row-index-unattributed`` /
  ``row-index-label-ambiguous``), including the nested-label vocabulary
  (`arm` vs `arm-repair`) the boundary matcher deliberately refuses;
- the budget arms (§6.4): per-file cap, aggregate byte + count caps, and
  the ``size=None`` arm — each ERRORing with ZERO fetch calls (asserted by
  a call-counting fake), unknown sizes resolved by ONE batched probe;
- the LABEL-grained exemption contract: always a visible WARN row that
  still reports realized counts (a SHORT exempt label included), zero-match
  exempt labels excused from ``row-index-missing``, undeclared-label
  exemptions ERRORing, and a non-exempt shortfall still FAILing overall;
- producer self-reported counts REPORTED, never gated
  (``producer-field-mismatch``);
- ``run_verification`` legacy invariance: no new kwargs ⇒ exactly one
  inert ``realized_row_counts: SKIP`` row added to the check set;
- the live-HF motivating incident, both directions (§6.2, network-gated):
  the pre-repair shard set FAILs at 1504/820/208 distinct vs 2000/1304/671
  declared; today's repaired live store PASSes at exactly the declared
  counts with duplicates reported as context.

Per the one-production-body-test rule (#906), the local-root fixtures
execute the REAL check body end to end (enumeration, attribution, budget,
counting, verdict) with no seams stubbed; the live-HF test additionally
executes the real Hub seam bodies (`_row_index_hf_entries`,
`_row_index_fetch`). Budget-arm fakes sit at the external network boundary
only and mirror the real seam signatures. Same module-loading conventions
as tests/test_verify_uploads_outroot_residue.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_uploads.py"
_spec = importlib.util.spec_from_file_location("verify_uploads_rr", _SCRIPT)
verify_uploads = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_uploads_rr"] = verify_uploads
_spec.loader.exec_module(verify_uploads)  # type: ignore[union-attr]


def _row(context_id: str, rollout_k: int, **extra) -> str:
    """One JSONL index row carrying the #2091-shaped composite key."""
    return json.dumps({"context_id": context_id, "rollout_k": rollout_k, **extra})


def _write_index(root: Path, reldir: str, name: str, rows: list[str]) -> Path:
    d = root / reldir
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


KEY = ("context_id", "rollout_k")


# ---------------------------------------------------------------------------
# SKIP / missing-input arms
# ---------------------------------------------------------------------------


def test_skip_when_no_expectation():
    res = verify_uploads.check_realized_row_counts()
    assert res["status"] == "SKIP"
    assert "--expected-rows" in res["detail"]


def test_skip_names_supplied_source_without_expectation(tmp_path):
    res = verify_uploads.check_realized_row_counts(local_root=str(tmp_path))
    assert res["status"] == "SKIP"
    assert "without an expectation" in res["detail"]


def test_error_when_expectation_but_no_source():
    res = verify_uploads.check_realized_row_counts(expected_rows={"a": 1})
    assert res["status"] == "ERROR"
    assert "no row-index source" in res["detail"]


def test_nonexistent_local_root_errors(tmp_path):
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"a": 1}, local_root=str(tmp_path / "absent")
    )
    assert res["status"] == "ERROR"
    assert "local root not found" in res["detail"]


# ---------------------------------------------------------------------------
# Composite-key lattice (§6.3) — real check body, no seams stubbed
# ---------------------------------------------------------------------------


def test_composite_key_catches_missing_rollout(tmp_path):
    """2 units x 3 rollouts with ONE rollout absent: unit coverage is complete
    (both context_ids present), so a unit-key-only gate would PASS — the
    full-key distinct count FAILs at 5 vs 6 (the round-1 hole, mechanized)."""
    rows = [_row(f"c{u}", k) for u in range(2) for k in range(3)]
    rows.remove(_row("c1", 2))
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", rows)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 6},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "FAIL"
    label = res["labels"]["jobA"]
    assert label["tag"] == "realized-rows-short"
    assert label["realized_distinct"] == 5
    assert label["expected"] == 6
    assert "expected=6" in res["detail"] and "distinct=5" in res["detail"]


def test_duplicate_full_key_rows_still_pass(tmp_path):
    """A duplicated (unit, rollout) pair — the repair-shard overlap shape —
    PASSes at 6 distinct / 7 lines with the duplicate count reported: the
    raw line count is never the gate quantity in either direction."""
    rows = [_row(f"c{u}", k) for u in range(2) for k in range(3)]
    rows.append(_row("c0", 0))  # duplicate full key
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", rows)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 6},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "OK"
    label = res["labels"]["jobA"]
    assert label["realized_distinct"] == 6
    assert label["realized_lines"] == 7
    assert label["duplicates"] == 1


def test_post_repair_2048_lines_2000_distinct_shape(tmp_path):
    """The literal #2091 post-repair arithmetic: 2000 distinct full-key rows
    across 2048 lines (48 repair-shard overlaps) PASSes at expected 2000 —
    a line-count gate would FAIL every healthy repaired store."""
    rows = [_row(f"c{u}", k) for u in range(500) for k in range(4)]
    rows.extend(rows[:48])  # 48 duplicated full keys
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", rows[:1024])
    _write_index(tmp_path, "jobA", "row_index_shard01.jsonl", rows[1024:])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 2000},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "OK"
    label = res["labels"]["jobA"]
    assert label["realized_lines"] == 2048
    assert label["realized_distinct"] == 2000
    assert label["duplicates"] == 48


def test_surplus_fails_unexpected_surplus(tmp_path):
    rows = [_row(f"c{u}", k) for u in range(2) for k in range(3)]
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", rows)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 5},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "FAIL"
    assert res["labels"]["jobA"]["tag"] == "realized-rows-unexpected-surplus"


def test_no_key_short_fails_and_gte_warns(tmp_path):
    """Keyless mode: the line count is a FLOOR only — short still FAILs, but
    lines >= expected is a WARN (realized-rows-no-distinct-key), never OK."""
    rows = [_row("c0", k) for k in range(4)]
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", rows)
    short = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 5}, local_root=str(tmp_path)
    )
    assert short["status"] == "FAIL"
    assert short["labels"]["jobA"]["tag"] == "realized-rows-short"
    gte = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 4}, local_root=str(tmp_path)
    )
    assert gte["status"] == "WARN"
    assert gte["labels"]["jobA"]["tag"] == "realized-rows-no-distinct-key"
    assert "--row-index-distinct-key" in gte["detail"]


def test_key_absent_row_errors(tmp_path):
    """A row missing a declared key field is never silently skipped —
    skipping would shrink the very denominator the check gates on."""
    rows = [_row("c0", 0), json.dumps({"context_id": "c1"})]  # rollout_k absent
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", rows)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 2},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert "row-index-key-absent" in res["detail"]
    assert "missing key field(s)" in res["detail"]


# ---------------------------------------------------------------------------
# Attribution ERROR arms — fire BEFORE any counting
# ---------------------------------------------------------------------------


def _counting_fake(monkeypatch):
    """Replace the file-read seam with a signature-mirroring counting fake so
    'zero counting/fetch happened' is assertable."""
    calls: list[dict] = []

    def fake_read(entry: dict) -> str:
        calls.append(entry)
        return ""

    monkeypatch.setattr(verify_uploads, "_read_row_index_entry", fake_read)
    return calls


def test_overlapping_labels_error_before_counting(tmp_path, monkeypatch):
    """`arm` boundary-matches the `arm-repair` component (hyphen is a
    non-word char), so a file under arm-repair/ matches BOTH labels —
    row-index-label-ambiguous, zero label verdicts, zero counting."""
    calls = _counting_fake(monkeypatch)
    _write_index(tmp_path, "arm-repair", "row_index_shard00.jsonl", [_row("c0", 0)])
    _write_index(tmp_path, "arm", "row_index_shard01.jsonl", [_row("c0", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"arm": 1, "arm-repair": 1},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert "row-index-label-ambiguous" in res["detail"]
    assert "labels" not in res, "an attribution ERROR must carry zero label verdicts"
    assert calls == [], "attribution ERRORs must fire BEFORE any counting"


def test_label_boundary_does_not_swallow_word_char_suffix(tmp_path):
    """`syc_aita` must NOT match the `syc_aita_v2` component (underscore is a
    word char): the v2 file is unattributed, an ERROR — never silently
    counted into the shorter label."""
    _write_index(tmp_path, "syc_aita", "row_index_shard00.jsonl", [_row("c0", 0)])
    _write_index(tmp_path, "syc_aita_v2", "row_index_shard00.jsonl", [_row("c9", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"syc_aita": 1},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert "row-index-unattributed" in res["detail"]
    assert "syc_aita_v2" in res["detail"]


def test_missing_label_errors_before_counting(tmp_path, monkeypatch):
    calls = _counting_fake(monkeypatch)
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row("c0", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1, "jobB": 5},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert "row-index-missing" in res["detail"]
    assert "'jobB'" in res["detail"]
    assert calls == []


# ---------------------------------------------------------------------------
# Budget arms (§6.4) — ZERO fetches on every failing arm
# ---------------------------------------------------------------------------


def _fetch_counting_fakes(monkeypatch, entries, resolved_sizes=None):
    """Install signature-mirroring fakes for the three HF seams; return the
    (fetch_calls, probe_calls) counters."""
    fetch_calls: list[str] = []
    probe_calls: list[list[str]] = []

    def fake_hf_entries(hf_prefixes: tuple[str, ...]) -> list[tuple[str, int | None]]:
        return list(entries)

    def fake_resolve_sizes(paths: list[str]) -> dict[str, int | None]:
        probe_calls.append(list(paths))
        return dict(resolved_sizes or {})

    def fake_fetch(path_in_repo: str, target: Path) -> Path:
        fetch_calls.append(path_in_repo)
        target.write_text(_row("c0", 0) + "\n", encoding="utf-8")
        return target

    monkeypatch.setattr(verify_uploads, "_row_index_hf_entries", fake_hf_entries)
    monkeypatch.setattr(verify_uploads, "_row_index_resolve_sizes", fake_resolve_sizes)
    monkeypatch.setattr(verify_uploads, "_row_index_fetch", fake_fetch)
    return fetch_calls, probe_calls


def test_aggregate_budget_errors_before_any_fetch(monkeypatch):
    """File-count overflow ERRORs off the LISTING with zero downloads."""
    entries = [(f"issueX/jobA/row_index_shard0{i}.jsonl", 100) for i in range(6)]
    fetch_calls, _ = _fetch_counting_fakes(monkeypatch, entries)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 6},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
        max_files=5,
    )
    assert res["status"] == "ERROR"
    assert "row-index-budget-exceeded" in res["detail"]
    assert fetch_calls == [], "over-budget must mean ZERO downloads"


def test_aggregate_bytes_budget_errors_before_any_fetch(monkeypatch):
    entries = [(f"issueX/jobA/row_index_shard0{i}.jsonl", 200) for i in range(3)]
    fetch_calls, _ = _fetch_counting_fakes(monkeypatch, entries)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 3},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
        max_total_bytes=500,
    )
    assert res["status"] == "ERROR"
    assert "row-index-budget-exceeded" in res["detail"]
    assert fetch_calls == []


def test_per_file_cap_errors_before_any_fetch(monkeypatch):
    entries = [("issueX/jobA/row_index_shard00.jsonl", 32_000_001)]
    fetch_calls, _ = _fetch_counting_fakes(monkeypatch, entries)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
        max_bytes=16_000_000,
    )
    assert res["status"] == "ERROR"
    assert "row-index-file-over-cap" in res["detail"]
    assert fetch_calls == []


def test_unknown_size_resolved_by_single_batched_probe(monkeypatch):
    """size=None entries are resolved by exactly ONE batched get_paths_info
    probe; once under cap, counting proceeds normally."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, probe_calls = _fetch_counting_fakes(
        monkeypatch, [(path, None)], resolved_sizes={path: 100}
    )
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "OK"
    assert len(probe_calls) == 1 and probe_calls[0] == [path]
    assert fetch_calls == [path]


def test_unknown_size_unresolvable_errors_zero_fetches(monkeypatch):
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, probe_calls = _fetch_counting_fakes(
        monkeypatch, [(path, None)], resolved_sizes={path: None}
    )
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert "row-index-size-unknown" in res["detail"]
    assert len(probe_calls) == 1
    assert fetch_calls == [], "an unprovable size is never assumed under cap"


def test_probe_resolved_size_over_cap_errors_zero_fetches(monkeypatch):
    """Round-3 pin: a size that RESOLVES over the per-file cap takes the
    over-cap ERROR arm — resolution is not permission."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, _ = _fetch_counting_fakes(
        monkeypatch, [(path, None)], resolved_sizes={path: 99_000_000}
    )
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert "row-index-file-over-cap" in res["detail"]
    assert fetch_calls == []


def test_hf_mode_counts_via_fetch_seam(monkeypatch):
    """Happy-path HF mode: budget passes, the fetch seam is called once per
    matched file, and the fetched rows are counted."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, _ = _fetch_counting_fakes(monkeypatch, [(path, 100)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "OK"
    assert fetch_calls == [path]
    assert res["labels"]["jobA"]["realized_distinct"] == 1


# ---------------------------------------------------------------------------
# Exemption contract (§4(A) step 5)
# ---------------------------------------------------------------------------


def test_exempt_short_label_warns_with_realized_counts(tmp_path):
    """A SHORT exempt label emits a VISIBLE WARN row that still reports its
    shortfall + the verbatim reason — an exemption can never pass silently."""
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row("c0", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 5},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
        exempt_labels={"jobA": "legitimately partial: pilot slice only"},
    )
    assert res["status"] == "WARN"
    label = res["labels"]["jobA"]
    assert label["tag"] == "realized-rows-exempt"
    assert label["realized_distinct"] == 1 and label["expected"] == 5
    assert "legitimately partial: pilot slice only" in res["detail"]


def test_exempt_zero_match_label_warns_not_errors(tmp_path):
    """A fully-absent exempt class is excused from row-index-missing and
    reports realized 0 on its WARN row."""
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row("c0", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1, "jobB": 3},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
        exempt_labels={"jobB": "class not produced this run"},
    )
    assert res["status"] == "WARN"
    label = res["labels"]["jobB"]
    assert label["tag"] == "realized-rows-exempt"
    assert label["realized_distinct"] == 0 and label["realized_lines"] == 0


def test_exempt_plus_nonexempt_short_still_fails_overall(tmp_path):
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row("c0", 0)])
    _write_index(tmp_path, "jobB", "row_index_shard00.jsonl", [_row("c1", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 5, "jobB": 5},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
        exempt_labels={"jobA": "known partial"},
    )
    assert res["status"] == "FAIL", "a non-exempt shortfall must keep failing overall"
    assert res["labels"]["jobA"]["tag"] == "realized-rows-exempt"
    assert res["labels"]["jobB"]["tag"] == "realized-rows-short"


def test_exempt_unmatched_label_errors(tmp_path):
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row("c0", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
        exempt_labels={"jobZ": "typo'd label"},
    )
    assert res["status"] == "ERROR"
    assert "realized-rows-exempt-unmatched" in res["detail"]
    assert "jobZ" in res["detail"]


# ---------------------------------------------------------------------------
# Producer self-reported counts: reported, never gated
# ---------------------------------------------------------------------------


def test_self_reported_reported_never_gated(tmp_path):
    """The #2091 post-repair stale field (n_rows_captured=1488 vs 2000
    realized): the mismatch is NAMED but the verdict keys on realized."""
    rows = [_row(f"c{u}", k) for u in range(5) for k in range(2)]
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", rows)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 10},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
        self_reported_rows={"jobA": 7},
    )
    assert res["status"] == "OK"
    label = res["labels"]["jobA"]
    assert label["self_reported"] == 7
    assert "producer-field-mismatch" in res["detail"]
    assert "context only" in res["detail"]


# ---------------------------------------------------------------------------
# run_verification wiring: legacy invariance (§6.9)
# ---------------------------------------------------------------------------


def test_run_verification_legacy_shape_gains_single_skip_row():
    """No new kwargs ⇒ the check-row set differs from pre-change only by one
    inert realized_row_counts SKIP row (analysis type: no training rows)."""
    report = verify_uploads.run_verification(issue_num=2091, experiment_type="analysis")
    assert set(report["checks"]) == {
        "eval_json",
        "figures",
        "claimed_urls",
        "outroot_residue",
        "realized_row_counts",
    }
    assert report["checks"]["realized_row_counts"]["status"] == "SKIP"


# ---------------------------------------------------------------------------
# Live-HF regression anchor (§6.2) — network-gated
# ---------------------------------------------------------------------------

_LIVE = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"), reason="live-HF leg needs HF_TOKEN (set -a; . ./.env)"
)

_STORE = "issue2091_decode/capture_store"
_EXPECTED = {"greedy_wildchat": 2000, "greedy_syc_aita": 1304, "greedy_evil_toxicchat": 671}
_PRE_REPAIR_SHARDS = {
    "greedy_wildchat": 4,  # shard00-03; shard04 is the repair shard
    "greedy_syc_aita": 3,  # shard00-02
    "greedy_evil_toxicchat": 2,  # shard00-01
}
_PRE_REPAIR_DISTINCT = {
    "greedy_wildchat": 1504,
    "greedy_syc_aita": 820,
    "greedy_evil_toxicchat": 208,
}


@_LIVE
def test_pre_repair_shards_fail_and_live_store_passes(tmp_path):
    """The motivating incident, both directions, against the LIVE #2091 store.

    FAIL direction: only the pre-repair shards staged to a local root (the
    exact shard set #2091's failure marker enumerates) → FAIL at
    1504/820/208 distinct vs 2000/1304/671 declared. PASS direction: the
    three per-job prefixes against today's repaired store → OK at exactly
    the declared counts, with realized_lines 2048/1340/686 and duplicates
    48/36/15 reported as context, and the producer's real stale
    self-reported value (1488) named as a mismatch without gating. This pin
    makes the line-count-vs-distinct-count decision permanent: a regression
    to a line-count gate fails the PASS direction. Executes the REAL Hub
    seam bodies (scoped tree walks + retried staging fetches)."""
    # --- FAIL direction: stage pre-repair shards via the REAL fetch seam.
    for job, n_shards in _PRE_REPAIR_SHARDS.items():
        for i in range(n_shards):
            name = f"row_index_shard{i:02d}.jsonl"
            verify_uploads._row_index_fetch(f"{_STORE}/{job}/{name}", tmp_path / job / name)
    fail_res = verify_uploads.check_realized_row_counts(
        expected_rows=dict(_EXPECTED),
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
    )
    assert fail_res["status"] == "FAIL"
    for job, distinct in _PRE_REPAIR_DISTINCT.items():
        label = fail_res["labels"][job]
        assert label["tag"] == "realized-rows-short", (job, label)
        assert label["realized_distinct"] == distinct, (job, label)

    # --- PASS direction: three per-job prefixes against the live store.
    pass_res = verify_uploads.check_realized_row_counts(
        expected_rows=dict(_EXPECTED),
        hf_prefixes=tuple(f"{_STORE}/{job}" for job in _EXPECTED),
        distinct_key_fields=KEY,
        self_reported_rows={"greedy_wildchat": 1488},  # the manifest's real stale value
    )
    assert pass_res["status"] == "OK"
    expected_context = {
        "greedy_wildchat": (2048, 48, 5),
        "greedy_syc_aita": (1340, 36, 4),
        "greedy_evil_toxicchat": (686, 15, 3),
    }
    for job, (lines, dups, n_shards) in expected_context.items():
        label = pass_res["labels"][job]
        assert label["realized_distinct"] == _EXPECTED[job], (job, label)
        assert label["realized_lines"] == lines, (job, label)
        assert label["duplicates"] == dups, (job, label)
        assert len(label["shards"]) == n_shards, (job, label)
    assert "producer-field-mismatch" in pass_res["detail"]


@_LIVE
def test_attribution_replay_per_prefix():
    """§6.6b: replay the attribution step over each per-job prefix's own
    scoped listing and assert every glob-matched file resolves to exactly
    ONE declared label — the guard against a mis-scoped invocation (how
    v3's unsatisfiable PASS direction went unnoticed for two rounds)."""
    labels = tuple(_EXPECTED)
    for job in _EXPECTED:
        listed = verify_uploads._row_index_hf_entries((f"{_STORE}/{job}",))
        matched = [
            p
            for p, _size in listed
            if verify_uploads.fnmatch.fnmatch(
                p.rsplit("/", 1)[-1], verify_uploads.ROW_INDEX_DEFAULT_GLOB
            )
        ]
        assert matched, f"no index files under {job}"
        for p in matched:
            hits = verify_uploads._labels_for_path(labels, p)
            assert hits == [job], (p, hits)
