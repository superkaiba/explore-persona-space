"""#2091 fu1 regression pins: the S4 packed-lookup join reads the INNER doc.

Incident: the shipped R4 run's ``stage_packed_lookup`` iterated packed
labeling shards with ``iter_jsonl`` and keyed on the WRAPPER row's
``context_id`` — ``pack_raw_tree`` lines are ``{"src": ..., "doc": {...}}``
with NO top-level context_id — so the lookup stayed empty and every
hallucination S4 label resolved to ``missing_packed_row`` (the single-sample
column shipped n=0 on all three hal rungs). These tests execute the REAL
parse path (``iter_jsonl`` over a tmp shard file → ``packed_lookup_rows`` →
``s4_labels`` → ``fits.s4_single_draw_label``) with benign synthetic rows;
only the filesystem boundary is a pytest tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import issue2091_analysis as A  # noqa: E402
from scripts import issue2091_fits as fits  # noqa: E402


def _wrapped(cid: str, k: int, completion: str, aliases: list[str]) -> dict:
    """One pack_raw_tree line: wrapper {src, doc} with the rollout in doc."""
    return {
        "src": f"labeling/hallucination/{cid}_seed{k}.json",
        "doc": {
            "context_id": cid,
            "rollout_k": k,
            "completion": completion,
            "answer_aliases": aliases,
        },
    }


def _write_shard(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return path


def _shard_rows() -> list[dict]:
    return [
        # the packed tree's manifest rides along as a row — must be skipped
        {"src": "_manifest.json", "doc": {"groups": {}}},
        _wrapped("ctx-a", 0, "The capital of France is Paris.", ["paris"]),
        _wrapped("ctx-a", 1, "I do not know.", ["paris"]),
        _wrapped("ctx-b", 3, "It was in 1907.", ["1912"]),
        _wrapped("ctx-ignored", 0, "x", ["x"]),  # not in wanted_cids
    ]


def test_wrapper_rows_carry_no_top_level_context_id(tmp_path):
    """The incident precondition: the shipped join key is absent on every line."""
    shard = _write_shard(tmp_path / "labeling_hallucination.shard00.jsonl", _shard_rows())
    rows = list(A.iter_jsonl(shard))
    assert rows, "shard parse produced no rows"
    assert all(r.get("context_id") is None for r in rows)


def test_packed_lookup_rows_unwraps_inner_doc(tmp_path):
    """FAILS pre-fix: the shipped path keyed on the wrapper and resolved nothing."""
    shard = _write_shard(tmp_path / "labeling_hallucination.shard00.jsonl", _shard_rows())
    wanted = {"ctx-a", "ctx-b"}
    lookup = A.packed_lookup_rows(A.iter_jsonl(shard), wanted)
    assert set(lookup) == {("ctx-a", 0), ("ctx-a", 1), ("ctx-b", 3)}
    assert lookup[("ctx-a", 0)]["answer_aliases"] == ["paris"]
    assert lookup[("ctx-b", 3)]["completion"] == "It was in 1907."


def test_packed_lookup_rows_fails_loud_on_missing_rollout_k():
    """A wanted doc without rollout_k is schema drift — never a silent k=0."""
    bad = [{"src": "labeling/hallucination/ctx-a_seed0.json", "doc": {"context_id": "ctx-a"}}]
    try:
        A.packed_lookup_rows(iter(bad), {"ctx-a"})
    except ValueError as e:
        assert "rollout_k" in str(e)
    else:
        raise AssertionError("expected ValueError on missing rollout_k")


def test_s4_labels_resolve_through_fixed_join(tmp_path):
    """End-to-end on the production body: correct / fabricated / abstained / unjudged
    resolve, and only a genuinely absent (cid, k) yields missing_packed_row."""
    shard = _write_shard(tmp_path / "labeling_hallucination.shard00.jsonl", _shard_rows())
    lookup = A.packed_lookup_rows(A.iter_jsonl(shard), {"ctx-a", "ctx-b"})
    abstain = {
        ("ctx-a", 1): 80.0,  # >= HALLU_FABRICATED_THRESHOLD (50) -> fabricated
        ("ctx-b", 3): 10.0,  # < threshold -> abstained
    }
    # alias match wins regardless of the abstain table
    picks = {"ctx-a": 0}
    assert A.s4_labels(picks, lookup, abstain) == {"ctx-a": "correct"}
    assert A.s4_labels({"ctx-a": 1}, lookup, abstain) == {"ctx-a": "fabricated"}
    assert A.s4_labels({"ctx-b": 3}, lookup, abstain) == {"ctx-b": "abstained"}
    # judged nowhere -> unjudged (drop-never-coerce), and an absent (cid, k)
    # -> missing_packed_row (the shipped run's universal outcome)
    assert A.s4_labels({"ctx-b": 3}, lookup, {}) == {"ctx-b": "unjudged"}
    assert A.s4_labels({"ctx-b": 0}, lookup, abstain) == {"ctx-b": "missing_packed_row"}
    # sanity: the label function under test is the production one
    assert fits.s4_single_draw_label is not None
