"""Consumer-level parity and missing-data checks for natural persona scoring."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import natural_extremes as ne
from explore_persona_space.experiments.issue_1739.corpus_staging import minhash_signatures

SCORER_PATH = Path(__file__).resolve().parents[1] / "scripts/issue1739_natural_score.py"
_SPEC = importlib.util.spec_from_file_location("issue1739_natural_score_under_test", SCORER_PATH)
scorer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scorer)


def _store(path, *, ids=("a", "b", "c", "d"), missing=(), extra=()):
    """Write tiny real-format fp16 shards with complete prompt/response identities."""
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"context_id": cid, "rollout_k": k}
        for cid in ids
        for k in range(5)
        if (cid, k) not in missing
    ] + [{"context_id": cid, "rollout_k": k} for cid, k in extra]
    generator = np.random.default_rng(1739)
    # 20 x 3584 is about 140 KiB, exercising the real fixed-dimensional loader.
    answer = generator.normal(size=(len(rows), 3584)).astype(np.float16)
    contexts = {cid: generator.normal(size=3584).astype(np.float16) for cid in ids}
    context = np.stack([contexts[r["context_id"]] for r in rows])
    for kind, array in (("context_end", context), ("prefix_end", context), ("t1", answer)):
        np.save(path / f"{kind}_L15.npy", array)
    (path / "row_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows, context, answer


def _labels(path, ids=("a", "b", "c", "d")):
    """Write labels whose missing judge draw must not alter observed-answer pooling."""
    rows = [
        {
            "context_id": cid,
            "dv": 20.0 * i,
            "split": "train",
            "group_key": f"group:{cid}",
            "rung": "train",
            "per_rollout_scores": {f"k{k:02d}": None if k == 4 else 20.0 * i for k in range(5)},
        }
        for i, cid in enumerate(ids)
    ]
    path.write_text(json.dumps({"rows": rows}))
    return path


def _selection(ids=("a", "b", "c", "d")):
    """Select all four prompts with one unevenly judged high prompt."""
    scores = np.array([[0] * 5, [10] * 5, [90, 90, 90, np.nan, np.nan], [100] * 5])
    return ne.select_prompt_tails(
        scores,
        context_ids=list(ids),
        group_ids=[f"g:{c}" for c in ids],
        prompt_hashes=[f"hash:{c}" for c in ids],
        extraction_mask=np.ones(4, bool),
        eligible_mask=np.ones(4, bool),
        evaluation_mask=np.zeros(4, bool),
        score_range=(0.0, 100.0),
        behavior="evil",
        fold="hhrt",
        q=0.5,
    )


def test_load_table_matches_historical_fp16_pooling_with_dropped_judgment(tmp_path):
    from scripts.issue1739_fits import _load_labeled

    store = tmp_path / "store"
    _, _, answer = _store(store)
    labels = _labels(tmp_path / "labels.json")
    observed = scorer.load_table(store, labels, 15, {"train"})
    historical = _load_labeled(store, labels, [15], config="config_a", need_rollout_rows=False)
    assert observed.ids == historical.ctx_order
    assert observed.x.dtype == observed.y.dtype == np.float16
    np.testing.assert_array_equal(observed.x, historical.z_by_variant["context_end"][0])
    np.testing.assert_array_equal(observed.y, historical.z_ans[0])
    np.testing.assert_array_equal(observed.y[0], answer[:5].mean(axis=0))
    assert not np.array_equal(observed.y[0], answer[:4].mean(axis=0))


@pytest.mark.parametrize("second_ids", [("c", "d"), ("c",)])
def test_merged_coverage_accepts_partitioned_sources_and_rejects_missing_contexts(
    tmp_path, second_ids
):
    first_store, second_store = tmp_path / "first", tmp_path / "second"
    _store(first_store, ids=("a", "b"))
    _store(second_store, ids=second_ids)
    labels = _labels(tmp_path / "labels.json")
    merged = scorer.merge_tables(
        [scorer.load_table(path, labels, 15, {"train"}) for path in (first_store, second_store)]
    )
    if len(second_ids) == 2:
        scorer.assert_label_context_coverage(merged, [labels])
    else:
        with pytest.raises(ValueError, match="Full label/capture coverage mismatch"):
            scorer.assert_label_context_coverage(merged, [labels])


@pytest.mark.parametrize("missing,extra", [((("a", 4),), ()), ((), (("a", 0),))])
def test_load_table_rejects_incomplete_or_duplicate_response_capture(tmp_path, missing, extra):
    store = tmp_path / "store"
    _store(store, missing=missing, extra=extra)
    labels = _labels(tmp_path / "labels.json")
    with pytest.raises(ValueError):
        scorer.load_table(store, labels, 15, {"train"})


def test_streamed_direction_matches_prompt_mean_contrast_with_unequal_valid_counts(tmp_path):
    store = tmp_path / "store"
    rows, context, answer = _store(store)
    selection = _selection()
    keys, directions = scorer.stream_directions(
        [store], 15, ["a", "b", "c", "d"], {"hhrt/q01_s0": selection}
    )
    assert keys == ["hhrt/q01_s0"]
    for kind, array in (("t1", answer), ("context_end", context)):
        by_id = {
            cid: np.asarray(
                [array[i] for i, r in enumerate(rows) if r["context_id"] == cid], dtype=float
            )
            for cid in ("a", "b", "c", "d")
        }
        expected = (by_id["c"][:3].mean(0) + by_id["d"].mean(0)) / 2
        expected -= (by_id["a"].mean(0) + by_id["b"].mean(0)) / 2
        np.testing.assert_allclose(directions[kind][0], expected, atol=1e-14, rtol=1e-14)


def test_streamed_direction_requires_every_weighted_response(tmp_path):
    store = tmp_path / "store"
    _store(store, missing=(("d", 4),))
    with pytest.raises(ValueError):
        scorer.stream_directions([store], 15, ["a", "b", "c", "d"], {"hhrt/q01_s0": _selection()})


def test_streamed_direction_rejects_entire_missing_extraction_source(tmp_path):
    store = tmp_path / "store"
    _store(store, ids=("a", "b"))
    with pytest.raises(ValueError):
        scorer.stream_directions([store], 15, ["a", "b", "c", "d"], {"hhrt/q01_s0": _selection()})


def test_streamed_direction_rejects_duplicate_response_across_stores(tmp_path):
    store = tmp_path / "store"
    _store(store)
    with pytest.raises(ValueError):
        scorer.stream_directions(
            [store, store], 15, ["a", "b", "c", "d"], {"hhrt/q01_s0": _selection()}
        )


def _evil_sources():
    """Include both natural extraction datasets and the historical eval rungs."""
    rows, contexts, scores = [], {}, {}
    for dataset in ("train", "hhrt", "toxicchat", "evil_mhj", "wildchat_rung"):
        for i in range(60):
            cid = f"{dataset}-{i}"
            rows.append(
                {
                    "context_id": cid,
                    "group_key": f"g{i}",
                    "rung": dataset,
                    "dv": float((i % 2) * 100),
                }
            )
            query = "Please inspect the following unique reference: " + "".join(
                hashlib.sha256(f"{cid}-{salt}".encode()).hexdigest() for salt in range(4)
            )
            query_hash = hashlib.sha256(" ".join(query.lower().split()).encode()).hexdigest()
            contexts[cid] = {
                "query": query,
                "question_normalized_sha256": query_hash,
                "user_turn_hashes": [query_hash],
                "natural_eligible": dataset in {"hhrt", "toxicchat"},
            }
            scores[cid] = {f"k{k:02d}": float((i % 2) * 100) for k in range(5)}
    signatures = {
        cid: signature.tolist()
        for cid, signature in zip(
            contexts, minhash_signatures([info["query"] for info in contexts.values()]), strict=True
        )
    }
    return (
        SimpleNamespace(
            rows=rows,
            ids=[r["context_id"] for r in rows],
            groups=[r["group_key"] for r in rows],
            rungs=[r["rung"] for r in rows],
            dv=np.array([r["dv"] for r in rows]),
        ),
        {"contexts": contexts, "scores": scores, "signatures": signatures},
    )


def test_evil_fold_uses_other_natural_source_and_protects_whole_eval_groups():
    table, metadata = _evil_sources()
    roster, selections, records = scorer.build_selections({"behavior": "evil"}, table, metadata)
    for evaluation, extraction in (("hhrt", "toxicchat"), ("toxicchat", "hhrt")):
        key = f"{evaluation}/q01_s0"
        assert records[key]["status"] == "ok"
        selected = selections[key]
        indices = np.r_[selected.high_indices, selected.low_indices]
        assert {table.rungs[i] for i in indices} == {extraction}
        assert not set(indices) & set(roster[evaluation])
        assert all(scorer._group_side_train(extraction, table.groups[i], 0, 0.8) for i in indices)
    assert records["evil_mhj/q01_s0"]["n_candidates"] > records["hhrt/q01_s0"]["n_candidates"]


def test_extreme_selection_is_invariant_to_evaluation_labels_and_row_order():
    table, metadata = _evil_sources()
    _, original, _ = scorer.build_selections({"behavior": "evil"}, table, metadata)
    changed = json.loads(json.dumps(metadata))
    # HHRT scores are evaluation-only in its own fold and may not move that direction.
    for cid, values in changed["scores"].items():
        if cid.startswith("hhrt-"):
            changed["scores"][cid] = {k: 100 - value for k, value in values.items()}
    permutation = np.random.default_rng(8).permutation(len(table.ids))
    permuted = SimpleNamespace(
        **{
            key: np.asarray(getattr(table, key), dtype=object)[permutation].tolist()
            for key in ("ids", "rows", "rungs", "groups", "dv")
        }
    )
    _, updated, _ = scorer.build_selections({"behavior": "evil"}, permuted, changed)
    a, b = original["hhrt/q01_s0"], updated["hhrt/q01_s0"]
    assert {table.ids[i] for i in a.high_indices} == {permuted.ids[i] for i in b.high_indices}
    assert {table.ids[i] for i in a.low_indices} == {permuted.ids[i] for i in b.low_indices}


def test_midpoint_controls_share_valid_floor_and_report_actual_prompt_count():
    table, metadata = _evil_sources()
    candidates = [
        i
        for i, rung in enumerate(table.rungs)
        if rung == "toxicchat" and scorer._group_side_train(rung, table.groups[i], 0, 0.8)
    ]
    short, complete = candidates[:2]
    metadata["scores"][table.ids[short]] = {
        "k00": 0,
        "k01": 100,
        "k02": None,
        "k03": None,
        "k04": None,
    }
    metadata["scores"][table.ids[complete]] = {
        "k00": 0,
        "k01": 0,
        "k02": 100,
        "k03": 100,
        "k04": 100,
    }
    _, selections, records = scorer.build_selections({"behavior": "evil"}, table, metadata)
    for name in ("e2", "e2p"):
        key = f"hhrt/{name}"
        selected = selections[key]
        support = np.any(selected.high_weights + selected.low_weights > 0, axis=1)
        assert not support[short]
        assert support[complete]
        assert records[key]["n_qualifying"] == int(support.sum())
        assert records[key]["n_qualifying"] < len(table.ids)


def test_eligible_prompt_requires_archived_response_score_record():
    table, metadata = _evil_sources()
    del metadata["scores"]["toxicchat-0"]
    with pytest.raises(ValueError, match="Missing archived score record"):
        scorer.build_selections({"behavior": "evil"}, table, metadata)


def test_missing_noneligible_ood_response_scores_are_not_extraction_join_failures():
    table, metadata = _evil_sources()
    for cid in list(metadata["scores"]):
        if cid.startswith("evil_mhj-"):
            del metadata["scores"][cid]
    _, _, records = scorer.build_selections({"behavior": "evil"}, table, metadata)
    assert records["hhrt/q01_s0"]["status"] == "ok"


def test_naturalness_eligibility_cannot_be_truthy_string():
    table, metadata = _evil_sources()
    metadata["contexts"]["toxicchat-0"]["natural_eligible"] = "false"
    with pytest.raises(ValueError, match="Natural eligibility must be boolean"):
        scorer.build_selections({"behavior": "evil"}, table, metadata)


def test_near_duplicate_eval_prompt_is_excluded_before_ranking():
    table, metadata = _evil_sources()
    _, selections, _ = scorer.build_selections({"behavior": "evil"}, table, metadata)
    candidate = table.ids[selections["hhrt/q01_s0"].high_indices[0]]
    query = metadata["contexts"][candidate]["query"]
    # Different normalized text/hash, same substantive character shingles.
    changed_query = query + " Thanks."
    metadata["contexts"]["hhrt-0"]["query"] = changed_query
    metadata["contexts"]["hhrt-0"]["question_normalized_sha256"] = hashlib.sha256(
        changed_query.lower().encode()
    ).hexdigest()
    metadata["contexts"]["hhrt-0"]["user_turn_hashes"] = [
        metadata["contexts"]["hhrt-0"]["question_normalized_sha256"]
    ]
    metadata["signatures"]["hhrt-0"] = minhash_signatures([changed_query])[0].tolist()
    _, changed, _ = scorer.build_selections({"behavior": "evil"}, table, metadata)
    for name in ("q01_s0", "q05_s0", "q10_s0", "e2p"):
        selected = changed[f"hhrt/{name}"]
        indices = np.r_[selected.high_indices, selected.low_indices]
        assert candidate not in {table.ids[i] for i in indices}
