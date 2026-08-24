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
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

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
