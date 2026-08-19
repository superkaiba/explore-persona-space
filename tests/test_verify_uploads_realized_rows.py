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
  counts with duplicates reported as context;
- the round-2 blockers (#2148 r2): repeated/overlapping prefixes dedupe by
  (mode, path) — one fetch, honest keyless FAIL, honest budgets — with a
  conflicting-KNOWN-size duplicate ERRORing fail-loud; the batched size
  probe rides ``retry_transient``, driven end-to-end through
  ``check_realized_row_counts`` (one transient 429 costs exactly one extra
  probe call and the final verdict still counts normally, instead of
  failing a healthy store); ONE pinned revision (retried
  ``repo_info(...).sha``) threads through the listing walk, the size
  probe, and every staged fetch, and the verdict reports the SHA;
- the round-2 concerns: exemption validation (membership + non-empty
  reason) runs at check entry BEFORE the no-expectation SKIP, in the
  callable and the CLI alike;
- the round-3 blockers (#2148 r3): the two row-index sources are MUTUALLY
  EXCLUSIVE — a dual-source invocation ERRORs at check entry with zero
  Hub reads (no cross-source row identity exists; a local-only row cannot
  prove durability) and the CLI rejects the flag pair at parse time; a
  same-path (None, int) size pair COALESCES (a known size fills a prior
  unknown, a later None never displaces a known value) and only a
  KNOWN-size disagreement ERRORs ``row-index-duplicate-conflict``; the
  per-unit progress lines ride STDERR unconditionally, with a
  ``fetch-start`` line before each fetch, so ``--json`` stdout stays one
  parseable document; an empty-after-strip prefix (``""``/``"/"``) ERRORs
  naming the invocation instead of letting ``row-index-missing`` blame
  the store.

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
from types import SimpleNamespace

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

    def fake_read(entry: dict, *, revision: str | None) -> str:
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


def test_label_boundary_does_not_swallow_word_char_suffix(tmp_path, monkeypatch):
    """`syc_aita` must NOT match the `syc_aita_v2` component (underscore is a
    word char): the v2 file is unattributed, an ERROR — never silently
    counted into the shorter label, and (like its two sibling attribution
    arms) it fires BEFORE any counting."""
    calls = _counting_fake(monkeypatch)
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
    assert calls == [], "attribution ERRORs must fire BEFORE any counting"


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


_FAKE_SHA = "0123abcd" * 5  # the pinned revision the fake resolver hands out


def _fetch_counting_fakes(monkeypatch, entries, resolved_sizes=None, fetch_rows=1):
    """Install signature-mirroring fakes for the four HF seams; return the
    (fetch_calls, probe_calls) counters. Every seam fake asserts it received
    the pinned revision (#2148 round 2), so all HF-mode tests double as
    revision-threading coverage at the seam boundary."""
    fetch_calls: list[str] = []
    probe_calls: list[list[str]] = []

    def fake_resolve_revision() -> str:
        return _FAKE_SHA

    def fake_hf_entries(
        hf_prefixes: tuple[str, ...], *, revision: str | None
    ) -> list[tuple[str, int | None]]:
        assert revision == _FAKE_SHA, "the listing must read the pinned revision"
        return list(entries)

    def fake_resolve_sizes(paths: list[str], *, revision: str | None) -> dict[str, int | None]:
        assert revision == _FAKE_SHA, "the size probe must read the pinned revision"
        probe_calls.append(list(paths))
        return dict(resolved_sizes or {})

    def fake_fetch(path_in_repo: str, target: Path, *, revision: str | None) -> Path:
        assert revision == _FAKE_SHA, "the staged fetch must read the pinned revision"
        fetch_calls.append(path_in_repo)
        rows = [_row("c0", k) for k in range(fetch_rows)]
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return target

    monkeypatch.setattr(verify_uploads, "_row_index_resolve_revision", fake_resolve_revision)
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
# Round-2 blocker 1: repeated/overlapping-prefix dedup (#2148 r2)
# ---------------------------------------------------------------------------


def test_duplicated_prefix_listing_dedupes_and_keyless_shortfall_still_fails(monkeypatch):
    """Blocker `repeated-prefix-multiset-false-negative`: one three-row path
    listed under two overlapping/duplicated prefixes counts ONCE — exactly
    one fetch, realized_lines == 3, and the keyless expected=6 shortfall
    stays a FAIL realized-rows-short. Pre-fix, the doubled listing read 6
    lines >= 6 and degraded the FAIL to the nonblocking
    realized-rows-no-distinct-key WARN — suppressing keyless mode's one
    strong signal."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, _ = _fetch_counting_fakes(monkeypatch, [(path, 100), (path, 100)], fetch_rows=3)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 6},
        hf_prefixes=("issueX", "issueX/jobA"),
    )
    assert res["status"] == "FAIL"
    assert res["labels"]["jobA"]["tag"] == "realized-rows-short"
    assert res["labels"]["jobA"]["realized_lines"] == 3
    assert fetch_calls == [path], "a path listed under two prefixes is fetched ONCE"


def test_duplicated_prefix_budget_counts_file_once(monkeypatch):
    """Distinct-mode sibling: the budget sums read the DEDUPLICATED set — a
    100-byte file listed twice fits a 150-byte aggregate cap (doubled, it
    would false-ERROR row-index-budget-exceeded), and counting proceeds
    with ONE fetch."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, _ = _fetch_counting_fakes(monkeypatch, [(path, 100), (path, 100)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX", "issueX/jobA"),
        distinct_key_fields=KEY,
        max_total_bytes=150,
    )
    assert res["status"] == "OK"
    assert fetch_calls == [path]
    assert res["labels"]["jobA"]["realized_distinct"] == 1


def test_duplicate_path_conflicting_sizes_errors_zero_fetches(monkeypatch):
    """A same-path duplicate with CONFLICTING sizes is a real listing
    inconsistency — fail-loud ERROR, never a silent dedupe, zero fetches."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, _ = _fetch_counting_fakes(monkeypatch, [(path, 100), (path, 200)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX", "issueX/jobA"),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert "row-index-duplicate-conflict" in res["detail"]
    assert fetch_calls == []


def test_hf_entries_walks_each_distinct_prefix_once(monkeypatch):
    """The REAL `_row_index_hf_entries` body canonicalizes prefixes
    (trailing-slash strip) and walks each DISTINCT prefix once, threading
    the pinned revision into every walk; overlapping parent/child prefixes
    are the caller-side dedup's job."""
    import explore_persona_space.orchestrate.hub as hub

    walks: list[tuple[str | None, str | None]] = []

    def fake_walk(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        walks.append((path_in_repo, revision))
        return [(f"{path_in_repo}/row_index_shard00.jsonl", 100)]

    monkeypatch.setattr(hub, "list_repo_entries_complete", fake_walk)
    listed = verify_uploads._row_index_hf_entries(
        ("issueX/jobA", "issueX/jobA/", "issueX/jobA"), revision="ab12"
    )
    assert walks == [("issueX/jobA", "ab12")], "one walk per DISTINCT canonicalized prefix"
    assert listed == [("issueX/jobA/row_index_shard00.jsonl", 100)]


# ---------------------------------------------------------------------------
# Round-2 blocker 2: the batched size probe rides retry_transient (#2148 r2)
# ---------------------------------------------------------------------------


def test_size_probe_rides_retry_transient_end_to_end(monkeypatch):
    """Blocker `row-index-size-probe-unretried` (r1) + the r2 half-scope
    finding: the REAL `_row_index_resolve_sizes` body wraps its
    get_paths_info POST in hub.retry_transient (mirroring
    scripts/issue2215_run.py:588), AND the retry composes with normal
    counting — driven end-to-end through `check_realized_row_counts` with
    the listing size None, one transient 429 costs exactly one extra probe
    call (two calls total) and the FINAL VERDICT counts normally (OK at
    realized_distinct == expected), instead of converting a healthy store
    into ERROR row-index-size-unknown → verifier FAIL → teardown refusal.
    Only the external Hub boundary (HfApi, the tree walk, the staging
    fetch) is faked; the revision resolver, the probe body, the budget
    path, and the counting all execute for real."""
    import huggingface_hub

    import explore_persona_space.orchestrate.hub as hub

    calls: list[list[str]] = []
    path = "issueX/jobA/row_index_shard00.jsonl"
    sha = "fe98dc76" * 5

    class _Transient429(Exception):
        def __init__(self):
            super().__init__("429 Too Many Requests")
            self.response = SimpleNamespace(status_code=429, headers={})

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo_id, *, repo_type=None):
            return SimpleNamespace(sha=sha)

        def get_paths_info(self, repo_id, paths, *, expand=False, revision=None, repo_type=None):
            calls.append(list(paths))
            if len(calls) == 1:
                raise _Transient429()
            assert revision == sha, "the retried probe must keep the pinned revision"
            return [SimpleNamespace(path=p, size=100) for p in paths]

    def fake_walk(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        return [(path, None)]  # size None forces the batched probe

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_row("c0", 0) + "\n", encoding="utf-8")
        return target

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(hub, "list_repo_entries_complete", fake_walk)
    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)
    monkeypatch.setattr("time.sleep", lambda s: None)  # retry backoff, not a real wait

    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
    )
    assert len(calls) == 2, "one transient 429 must retry, never ERROR a healthy store"
    assert res["status"] == "OK"
    assert res["labels"]["jobA"]["realized_distinct"] == 1


# ---------------------------------------------------------------------------
# Round-2 blocker 3: ONE pinned revision across every Hub call (#2148 r2)
# ---------------------------------------------------------------------------


def test_one_pinned_revision_threads_every_hub_call(monkeypatch, tmp_path):
    """Blocker `row-index-moving-head-snapshot` / plan §4(A) step 3: the REAL
    check + seam bodies resolve ONE revision at entry (retried
    repo_info(...).sha, the stage_hub_prefix pattern) and thread it into the
    listing walk, the batched size probe, and the staged fetch — and the
    verdict reports the SHA (`revision` key + detail), so it is traceable
    to one Hub snapshot."""
    import huggingface_hub

    import explore_persona_space.orchestrate.hub as hub

    sha = "ab12cd34" * 5
    seen: list[tuple[str, str | None]] = []
    path = "issueX/jobA/row_index_shard00.jsonl"

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo_id, *, repo_type=None):
            return SimpleNamespace(sha=sha)

        def get_paths_info(self, repo_id, paths, *, expand=False, revision=None, repo_type=None):
            seen.append(("get_paths_info", revision))
            return [SimpleNamespace(path=p, size=100) for p in paths]

    def fake_walk(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        seen.append(("list_repo_entries_complete", revision))
        return [(path, None)]  # size None forces the batched probe

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        seen.append(("stage_hub_file", revision))
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_row("c0", 0) + "\n", encoding="utf-8")
        return target

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(hub, "list_repo_entries_complete", fake_walk)
    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)

    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX/jobA",),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "OK"
    assert {name for name, _ in seen} == {
        "list_repo_entries_complete",
        "get_paths_info",
        "stage_hub_file",
    }
    assert all(rev == sha for _, rev in seen), seen
    assert res["revision"] == sha
    assert f"hub revision: {sha}" in res["detail"]


# ---------------------------------------------------------------------------
# Round-3 blocker 1: the two row-index sources are mutually exclusive
# ---------------------------------------------------------------------------


def _no_hub_seams(monkeypatch):
    """Every Hub seam raises: 'zero Hub reads performed' is assertable."""

    def _no_hub(*a, **k):
        raise AssertionError("this refusal must perform ZERO Hub reads")

    monkeypatch.setattr(verify_uploads, "_row_index_resolve_revision", _no_hub)
    monkeypatch.setattr(verify_uploads, "_row_index_hf_entries", _no_hub)
    monkeypatch.setattr(verify_uploads, "_row_index_resolve_sizes", _no_hub)
    monkeypatch.setattr(verify_uploads, "_row_index_fetch", _no_hub)


def test_dual_source_invocation_errors_never_warns(tmp_path, monkeypatch):
    """Blocker `mixed-row-index-source-double-count`: BOTH sources staging
    the same 3-row index used to survive the (mode, path) dedup — a local
    entry and an HF entry never collide — sum to 6 lines, and flip the
    keyless expected=6 shortfall FAIL into the nonblocking
    realized-rows-no-distinct-key WARN. The invocation is now REFUSED at
    check entry (ERROR naming both flags, zero Hub reads): no cross-source
    row identity exists, and a local-only row cannot prove durability on a
    gate whose PASS licenses deleting the local copy."""
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row(f"c{i}", 0) for i in range(3)])
    _no_hub_seams(monkeypatch)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 6},
        hf_prefixes=("issueX/jobA",),
        local_root=str(tmp_path),
    )
    assert res["status"] == "ERROR"
    assert "row-index-dual-source" in res["detail"]
    assert "mutually exclusive" in res["detail"]
    assert "realized-rows-no-distinct-key" not in res["detail"], (
        "the pre-fix shape: a doubled keyless count degraded the FAIL to a "
        "nonblocking WARN — the refusal must never reach the verdict lattice"
    )


def test_dual_source_refusal_precedes_no_expectation_skip(tmp_path, monkeypatch):
    """A malformed invocation ERRORs rather than SKIPping (the round-2
    exemption-validation precedent): dual sources with NO expectation still
    ERROR at entry, never the inert SKIP."""
    _no_hub_seams(monkeypatch)
    res = verify_uploads.check_realized_row_counts(
        hf_prefixes=("issueX/jobA",), local_root=str(tmp_path)
    )
    assert res["status"] == "ERROR"
    assert "row-index-dual-source" in res["detail"]


def test_cli_rejects_dual_source_flags(monkeypatch, capsys):
    """CLI shape: the flag pair is rejected at parse time with exit code 2,
    before any network call (the argparse-level twin of the check-entry
    refusal)."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_uploads.py",
            "--issue",
            "2091",
            "--expected-rows",
            "jobA=1",
            "--row-index-hf-prefix",
            "issueX/jobA",
            "--row-index-local-root",
            "/tmp/somewhere",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        verify_uploads.main()
    assert excinfo.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_empty_prefix_errors_at_check_entry(monkeypatch):
    """Finding 9: an empty-after-strip prefix ("" / "/") used to be silently
    skipped inside `_row_index_hf_entries`, so the downstream
    row-index-missing ERROR blamed the STORE for the operator's malformed
    flag. It now ERRORs at check entry naming the invocation, with zero
    Hub reads."""
    _no_hub_seams(monkeypatch)
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("/",),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "ERROR"
    assert res["detail"].startswith("row-index-prefix-empty"), (
        "the verdict arm must be the invocation-naming refusal, never the "
        "store-blaming row-index-missing arm"
    )
    assert "--row-index-hf-prefix" in res["detail"]


def test_hf_entries_raises_on_empty_prefix():
    """Helper defense in depth: the REAL `_row_index_hf_entries` body raises
    on an empty-after-strip prefix instead of silently skipping it (a
    direct caller bypassing the check-entry validation still fails loud,
    before any walk)."""
    with pytest.raises(ValueError, match="empty prefix"):
        verify_uploads._row_index_hf_entries(("/",), revision=None)


# ---------------------------------------------------------------------------
# Round-3 blocker 2: (None, int) size pairs coalesce — unknown is not conflict
# ---------------------------------------------------------------------------


def test_duplicate_unknown_known_sizes_coalesce_both_orders(monkeypatch):
    """Blocker `row-index-unknown-known-false-conflict`: the tree walk
    legitimately returns size None (`list_repo_entries_complete` is
    int | None, hub.py) and this same check's budget path treats None as
    "unknown, probe it" — so a same-path (None, int) pair is NOT a listing
    conflict. Through the REAL `_row_index_entries` dedup loop, in BOTH
    orders: one entry survives with the KNOWN size 100 (a known size fills
    a prior None; a later None never displaces a known value), no
    row-index-duplicate-conflict. A true known-size disagreement
    (100 vs 101) still ERRORs."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    for pair in ([(path, None), (path, 100)], [(path, 100), (path, None)]):
        monkeypatch.setattr(
            verify_uploads,
            "_row_index_hf_entries",
            lambda prefixes, *, revision, _pair=pair: list(_pair),
        )
        entries = verify_uploads._row_index_entries(
            ("issueX/jobA",), None, verify_uploads.ROW_INDEX_DEFAULT_GLOB, revision="ab12"
        )
        assert isinstance(entries, list), (pair, entries)
        assert len(entries) == 1, (pair, entries)
        assert entries[0]["size"] == 100, (pair, entries)

    monkeypatch.setattr(
        verify_uploads,
        "_row_index_hf_entries",
        lambda prefixes, *, revision: [(path, 100), (path, 101)],
    )
    res = verify_uploads._row_index_entries(
        ("issueX/jobA",), None, verify_uploads.ROW_INDEX_DEFAULT_GLOB, revision="ab12"
    )
    assert isinstance(res, dict)
    assert "row-index-duplicate-conflict" in res["detail"]


def test_unknown_known_pair_counts_normally_end_to_end(monkeypatch):
    """Composition pin: a healthy store whose listing yields a (None, int)
    duplicate pair still reaches a normal OK verdict — the coalesced known
    size feeds the budget path (no probe needed) and exactly one fetch
    runs."""
    path = "issueX/jobA/row_index_shard00.jsonl"
    fetch_calls, probe_calls = _fetch_counting_fakes(monkeypatch, [(path, None), (path, 100)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        hf_prefixes=("issueX", "issueX/jobA"),
        distinct_key_fields=KEY,
    )
    assert res["status"] == "OK"
    assert fetch_calls == [path]
    assert probe_calls == [], "the coalesced known size leaves nothing to probe"


# ---------------------------------------------------------------------------
# Round-3 blocker 3: --json stdout is a single parseable document
# ---------------------------------------------------------------------------


def test_json_stdout_stays_machine_parseable(tmp_path, monkeypatch, capsys):
    """Blocker `realized-rows-json-stdout-pollution`: the canonical Step
    2.11 invocation carries --json ("Output raw JSON"), so the per-unit
    progress lines ride STDERR unconditionally — one fetch-start line
    BEFORE each read (a hanging fetch names its file) and one completion
    line after — and stdout stays exactly one parseable JSON document.
    Also pins the [realized-rows] literal (r2 finding 6)."""
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row("c0", 0)])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_uploads.py",
            "--issue",
            "2091",
            "--type",
            "analysis",
            "--expected-rows",
            "jobA=1",
            "--row-index-local-root",
            str(tmp_path),
            "--row-index-distinct-key",
            "context_id,rollout_k",
            "--json",
            "--no-fail",
        ],
    )
    verify_uploads.main()
    captured = capsys.readouterr()
    report = json.loads(captured.out)  # stdout must parse as ONE JSON document
    assert report["checks"]["realized_row_counts"]["status"] == "OK"
    assert "[realized-rows]" not in captured.out
    assert "[realized-rows]" in captured.err
    assert "fetch-start" in captured.err


# ---------------------------------------------------------------------------
# Round-2 concern (a): exemption validation precedes the no-expectation SKIP
# ---------------------------------------------------------------------------


def test_exemption_only_invocation_errors_not_skips():
    """Concern `exemption-validation-after-skip`, direct-call shape: an
    exemption-only invocation ERRORs realized-rows-exempt-unmatched per the
    documented contract — validation runs BEFORE the no-expectation SKIP."""
    res = verify_uploads.check_realized_row_counts(exempt_labels={"jobA": "some reason"})
    assert res["status"] == "ERROR"
    assert "realized-rows-exempt-unmatched" in res["detail"]


def test_blank_exemption_reason_errors_direct_call(tmp_path):
    """The callable enforces the mandatory non-empty reason itself — the
    contract is no longer CLI-parser-only (a direct caller could previously
    pass a blank reason)."""
    _write_index(tmp_path, "jobA", "row_index_shard00.jsonl", [_row("c0", 0)])
    res = verify_uploads.check_realized_row_counts(
        expected_rows={"jobA": 1},
        local_root=str(tmp_path),
        distinct_key_fields=KEY,
        exempt_labels={"jobA": "   "},
    )
    assert res["status"] == "ERROR"
    assert "realized-rows-exempt-invalid" in res["detail"]


def test_cli_rejects_blank_exempt_reason(monkeypatch, capsys):
    """CLI shape: argparse rejects LABEL= (blank reason) at parse time with
    exit code 2, before any network call."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_uploads.py", "--issue", "2091", "--realized-rows-exempt", "jobA="],
    )
    with pytest.raises(SystemExit) as excinfo:
        verify_uploads.main()
    assert excinfo.value.code == 2
    assert "non-empty reason" in capsys.readouterr().err


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
    # --- FAIL direction: stage pre-repair shards via the REAL fetch seam
    # (revision=None: the live tests deliberately read the repo's live main).
    for job, n_shards in _PRE_REPAIR_SHARDS.items():
        for i in range(n_shards):
            name = f"row_index_shard{i:02d}.jsonl"
            verify_uploads._row_index_fetch(
                f"{_STORE}/{job}/{name}", tmp_path / job / name, revision=None
            )
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
def test_resolve_sizes_live_body():
    """Production-body coverage (#906) for the batched size-probe seam: the
    budget tests stub `_row_index_resolve_sizes`, so this executes the REAL
    body (ONE batched `get_paths_info` POST) against a known live index file
    and pins the size the scoped tree walk reports for it."""
    path = f"{_STORE}/greedy_evil_toxicchat/row_index_shard00.jsonl"
    sizes = verify_uploads._row_index_resolve_sizes([path], revision=None)
    assert sizes.get(path) == 21756


@_LIVE
def test_attribution_replay_per_prefix():
    """§6.6b: replay the attribution step over each per-job prefix's own
    scoped listing and assert every glob-matched file resolves to exactly
    ONE declared label — the guard against a mis-scoped invocation (how
    v3's unsatisfiable PASS direction went unnoticed for two rounds)."""
    labels = tuple(_EXPECTED)
    for job in _EXPECTED:
        listed = verify_uploads._row_index_hf_entries((f"{_STORE}/{job}",), revision=None)
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
