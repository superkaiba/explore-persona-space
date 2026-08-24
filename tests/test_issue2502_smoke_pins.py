"""Issue #2502 rv2-u4: committed pins for the two round-1 u4 smoke-caught fixes.

Both fixes were caught by the per-phase e2e smoke in round 1 and fixed in
production code; these tests pin them so a refactor cannot silently strip
either invariant (experiment-implementer checklist item 8):

  (a) corpus.build_report tolerates probe-mode rows WITHOUT a "split" field
      (probe runs BEFORE assign_splits; a bare r["split"] KeyError'd every
      --probe invocation) and counts them under "unassigned".
  (b) reliability.run_subset passes hf_missing_of a repo PATH PREFIX as
      ``scope`` (SUBSET_PREFIX), never a bare label — a non-prefix scope made
      verify_repo_paths_uploaded raise "expected paths outside path_in_repo"
      on EVERY subset invocation.

No network: (a) is pure; (b) monkeypatches the gen_capture hf_missing_of
module attribute (run_subset resolves it at call time via the GC module).

Round-5 pins (gated-source skip semantics; the Step 6d.0-bis tiny-real probe
crash on a gated fallback-less dataset):

  (c) a fallback-LESS source whose PRIMARY read fails an ACCESS-class error
      is SKIPPED loud (recorded in build_report's ``skipped_sources`` + the
      per-config ``skipped_gated_no_access`` counter), never a whole-build
      crash, and the budget re-scales across survivors only;
  (d) a gated source WITH a fallback still falls over to it (unchanged);
  (e) kept==0 on an ACCESSIBLE source still raises (the _stream_stage
      fail-loud is not swallowed by the skip path);
  (f) an all-skipped (empty) corpus and a whole regime-class family with zero
      staged rows each still raise.

The round-5 tests fake ONLY the HF network boundary (_resolve_dataset_revision
+ CS._hf_stream, both signature-mirroring defs); the real production chain
(stage_source -> _stage_one_config -> CS._stream_stage incl. keep_fn,
fingerprints, checkpoint writes, the kept==0 raise) executes unmodified.
Datasets are referenced by synthetic org/name ids only — no corpus text.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

CP = importlib.import_module("issue2502_corpus")
GC = importlib.import_module("issue2502_gen_capture")
RL = importlib.import_module("issue2502_reliability")


def test_build_report_probe_row_without_split_counts_unassigned():
    """Pin (a): a final row lacking "split" must not raise (probe mode)."""
    report = CP.build_report(
        pre_dedup_per_source={"s": 1},
        dedup_report={"n_in": 1, "n_confirmed_dropped": 0, "n_had_lsh_candidate": 0},
        post_dedup_per_source={"s": 1},
        allocation={"s": 1},
        final_rows=[{"regime_class": "ordinary", "realism_tier": 1, "source_tag": "s"}],
        regime_table={"s": "ordinary"},
        stream_counters={},
        budget=1,
    )
    assert report["split_counts"] == {"unassigned": 1}
    assert report["n_final"] == 1


def test_run_subset_scopes_hf_listing_to_subset_prefix(tmp_path, monkeypatch):
    """Pin (b): run_subset's presence probe uses SUBSET_PREFIX as the repo
    path-prefix scope and the exact corpus.jsonl dest path."""
    captured: dict = {}

    def fake_hf_missing_of(paths, *, scope):
        captured["paths"] = list(paths)
        captured["scope"] = scope
        return []  # nothing missing -> the already-uploaded skip path

    monkeypatch.setattr(GC, "hf_missing_of", fake_hf_missing_of)
    args = RL.build_parser().parse_args(
        ["--phase", "subset", "--work-dir", str(tmp_path / "subset_work")]
    )
    res = RL.run_subset(args)
    assert res.get("skipped") is True, res
    assert captured["scope"] == RL.SUBSET_PREFIX
    assert captured["paths"] == [f"{RL.SUBSET_PREFIX}/corpus.jsonl"]


# ---------------------------------------------------------------------------
# Round-5 pins (c)-(f): gated-source skip semantics.
# ---------------------------------------------------------------------------


def _spec(tag, dataset_id, regime, *, fallback=None, cap=10):
    """Minimal SourceSpec for the round-5 pins (scalar `text` field)."""
    return CP.SourceSpec(
        source_tag=tag,
        dataset_id=dataset_id,
        regime_class=regime,
        realism_tier=1,
        pre_dedup_cap=cap,
        text_fields=("text",),
        fallback_dataset_id=fallback,
    )


def _raw_rows(tag: str, n: int) -> list[dict]:
    """Distinct sha256-hex payloads so no LSH/exact-Jaccard near-dup fires."""
    return [
        {"text": f"{tag} synthetic context {hashlib.sha256(f'{tag}:{i}'.encode()).hexdigest()}"}
        for i in range(n)
    ]


def _install_hf_seams(monkeypatch, *, gated: set[str], raw_by_dataset: dict[str, list[dict]]):
    """Fake ONLY the HF network boundary (revision resolution + row stream).

    The real production chain (stage_source -> _stage_one_config ->
    CS._stream_stage incl. its kept==0 raise, keep_fn, fingerprints,
    checkpoint writes) executes unmodified. Both fakes mirror the real
    call shapes (`_resolve_dataset_revision(dataset_id)`;
    `CS._hf_stream(dataset_id, config, split, revision=..., **kw)`).
    """
    from datasets.exceptions import DatasetNotFoundError

    def fake_resolve(dataset_id: str) -> str:
        if dataset_id in gated:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' is a gated dataset on the Hub: access not granted"
            )
        return "0" * 40

    def fake_stream(dataset_id, config, split, **kwargs):
        assert kwargs.get("revision") == "0" * 40
        return iter(raw_by_dataset[dataset_id])

    monkeypatch.setattr(CP, "_resolve_dataset_revision", fake_resolve)
    monkeypatch.setattr(CP.CS, "_hf_stream", fake_stream)


def _probe_args(tmp_path, budget=10):
    return CP.build_argparser().parse_args(
        ["--probe", "--no-token-filter", "--out-dir", str(tmp_path), "--budget", str(budget)]
    )


def test_gated_no_fallback_source_skipped_and_recorded(monkeypatch, tmp_path):
    """Pin (c): a fallback-less gated source is SKIPPED (not raised), the skip
    is recorded in the report + probe artifact, and the budget re-scales
    across surviving sources only."""
    gated_id = "org/gated-no-access"
    specs = (
        _spec("gated_src", gated_id, "weird"),
        _spec("weird_ok", "org/open-weird", "weird"),
        _spec("ord_ok", "org/open-ord", "ordinary"),
    )
    _install_hf_seams(
        monkeypatch,
        gated={gated_id},
        raw_by_dataset={
            "org/open-weird": _raw_rows("weird_ok", 6),
            "org/open-ord": _raw_rows("ord_ok", 6),
        },
    )
    monkeypatch.setattr(CP, "SOURCES", specs)
    report = CP.run_pipeline(_probe_args(tmp_path))
    (skip,) = report["skipped_sources"]
    assert skip["source_tag"] == "gated_src"
    assert skip["dataset_id"] == gated_id
    assert skip["config"] == "default"
    assert "gated" in skip["reason"]
    # per-config counter marker survives in stream_counters
    assert report["stream_counters"]["gated_src"]["default"]["skipped_gated_no_access"] == 1
    # budget re-scaled across SURVIVORS only — no gated tag, exact total
    assert "gated_src" not in report["budget_allocation"]
    assert report["budget_allocation_total"] == 10  # min(budget=10, survivors' 12)
    # the probe artifact surfaces the skip durably (never silent)
    probe = json.loads((tmp_path / "probe_report.json").read_text())
    assert probe["skipped_sources"] == report["skipped_sources"]


def test_gated_with_fallback_still_falls_over(monkeypatch, tmp_path):
    """Pin (d): a gated primary WITH a fallback dataset still falls over to it
    gracefully — no skip marker, rows read from the fallback id."""
    spec = _spec("wc", "org/gated-primary", "ordinary", fallback="org/open-mirror")
    _install_hf_seams(
        monkeypatch,
        gated={"org/gated-primary"},
        raw_by_dataset={"org/open-mirror": _raw_rows("wc", 5)},
    )
    rows, ctr = CP.stage_source(
        spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0
    )
    assert len(rows) == 5
    assert all(r["dataset_id"] == "org/open-mirror" for r in rows)
    assert not any(c.get("skipped_gated_no_access") for c in ctr.values())


def test_kept_zero_on_accessible_source_still_raises(monkeypatch, tmp_path):
    """Pin (e): kept==0 on an ACCESSIBLE source (a data-shape/filter bug) still
    fails loud — the skip path is scoped to access-class errors only."""
    spec = _spec("shape_bug", "org/open-wrong-shape", "ordinary")
    _install_hf_seams(
        monkeypatch,
        gated=set(),
        raw_by_dataset={"org/open-wrong-shape": [{"not_the_text_field": "x"} for _ in range(4)]},
    )
    with pytest.raises(RuntimeError, match="kept 0 rows"):
        CP.stage_source(spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0)


def test_all_sources_skipped_empty_corpus_raises(monkeypatch, tmp_path):
    """Pin (f1): every source skipped -> empty corpus -> fail loud."""
    specs = (
        _spec("g1", "org/gated-a", "weird"),
        _spec("g2", "org/gated-b", "ordinary"),
    )
    _install_hf_seams(monkeypatch, gated={"org/gated-a", "org/gated-b"}, raw_by_dataset={})
    monkeypatch.setattr(CP, "SOURCES", specs)
    with pytest.raises(RuntimeError, match="kept 0 rows across all"):
        CP.run_pipeline(_probe_args(tmp_path))


def test_whole_regime_family_collapsed_raises(monkeypatch, tmp_path):
    """Pin (f2): a whole regime-class family with zero staged rows fails loud
    even when other families survive."""
    specs = (
        _spec("gated_weird", "org/gated-weird", "weird"),
        _spec("ord_ok", "org/open-ord", "ordinary"),
    )
    _install_hf_seams(
        monkeypatch,
        gated={"org/gated-weird"},
        raw_by_dataset={"org/open-ord": _raw_rows("ord_ok", 6)},
    )
    monkeypatch.setattr(CP, "SOURCES", specs)
    with pytest.raises(RuntimeError, match="regime class"):
        CP.run_pipeline(_probe_args(tmp_path))
