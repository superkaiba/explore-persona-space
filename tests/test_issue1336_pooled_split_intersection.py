"""#1336 pooled_split 5-way kept-intersection invariants (plan v15 §5).

Round-4 smoke-caught defect class: the C_pool manifest carried rows the
pooled fits cannot serve, breaking the §3 row-coverage contract at fit time
(`_pooled_xy_from_bundle`: "manifest rows missing from the bundle").
Root causes pinned here:

  1. ``_read_answers_conv_ids`` read a ``conv_id`` field that answers rows
     never carry (writer emits ``prompt_idx`` + ``kept``) and did not filter
     to KEPT rows — only kept rows are ever captured downstream.
  2. ``measure_5way_intersection`` was measurement-only: the intersection
     never restricted the manifest rows (plan v15 §5: n_pool IS the 5-way
     kept intersection).

Both tests fail on the pre-fix code (empty id sets / stub smoke dict) and
pass post-fix. No network: the smoke branch reads local fixture files only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "scripts"), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue1336_pooled_split as ps  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402


def _write_answers(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_read_answers_conv_ids_kept_filter_and_canonical_ids(tmp_path: Path) -> None:
    """Kept rows only, canonical s<prompt_idx> ids, half-open idx window."""
    path = tmp_path / "answers.jsonl"
    _write_answers(
        path,
        [
            {"prompt_idx": 0, "kept": True},
            {"prompt_idx": 1, "kept": False},  # unkept — never captured downstream
            {"prompt_idx": 5000, "kept": True},
            {"prompt_idx": 5001},  # no kept field == not kept
            {"prompt_idx": 5002, "kept": True},
        ],
    )
    assert ps._read_answers_conv_ids(path) == {"s0", "s5000", "s5002"}
    # Concat-corpus halves: wave-1 below the boundary, extension at/above it.
    assert ps._read_answers_conv_ids(path, max_idx=5000) == {"s0"}
    assert ps._read_answers_conv_ids(path, min_idx=5000) == {"s5000", "s5002"}


def test_read_answers_conv_ids_missing_prompt_idx_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    _write_answers(path, [{"kept": True}])
    with pytest.raises(AssertionError, match="prompt_idx"):
        ps._read_answers_conv_ids(path)


def test_smoke_intersection_returns_per_corpus_kept_id_sets(tmp_path, monkeypatch) -> None:
    """The smoke-local probe intersects KEPT ids across cm.SMOKE_MODELS and
    returns the id sets run() applies as the manifest row filter."""
    monkeypatch.setattr(ps, "DATA_ROOT", tmp_path)
    corpus = cm.SMOKE_CORPORA_V2[0]
    # Per-model kept sets that differ: intersection must be {s5001}.
    kept_by_model = {
        cm.SMOKE_MODELS[0]: [5000, 5001],
        cm.SMOKE_MODELS[1]: [5001, 5002],
        cm.SMOKE_MODELS[2]: [5001, 5003],
    }
    for model, kept in kept_by_model.items():
        rows = [{"prompt_idx": i, "kept": i in kept} for i in range(5000, 5004)]
        _write_answers(tmp_path / "gen_smoke" / model / corpus / "answers.jsonl", rows)
    ctx = ps.SplitContext(
        smoke=True,
        upload=False,
        corpora=(corpus,),
        out_root=tmp_path / "out",
        hf_prefix_out="",
    )
    summary, ids_by_corpus = ps.measure_5way_intersection(ctx)
    assert ids_by_corpus == {corpus: {"s5001"}}
    assert summary["mode"] == "smoke-local"
    assert summary["per_corpus_5way_intersection"] == {corpus: 1}
    assert summary["models"] == list(cm.SMOKE_MODELS)


def test_smoke_intersection_missing_fixture_fails_loud(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ps, "DATA_ROOT", tmp_path)
    corpus = cm.SMOKE_CORPORA_V2[0]
    ctx = ps.SplitContext(
        smoke=True,
        upload=False,
        corpora=(corpus,),
        out_root=tmp_path / "out",
        hf_prefix_out="",
    )
    with pytest.raises(AssertionError, match="smoke_fixtures"):
        ps.measure_5way_intersection(ctx)
