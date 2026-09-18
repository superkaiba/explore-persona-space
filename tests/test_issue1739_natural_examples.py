"""Preservation gates: full response identities, missing judgments and real categories."""

import json

import pytest

from scripts import issue1739_natural_examples as examples


def raw_inputs(tmp_path, defect=None):
    path = tmp_path / "labeling_evil.shard00.jsonl"
    cid = "evil-eval-hhrt-000000"
    scores = {"k00": 0.0, "k01": None, "k02": 0.0, "k03": 0.0, "k04": 0.0}
    label = {"context_id": cid, "group_key": "g", "per_rollout_scores": scores, "dv": 0.0}
    records = []
    info = {
        "query": "A natural query\u2028with Unicode",
        "prompt_text": "original prompt",
        "behavior": "evil",
        "rung": "hhrt",
        "split": "eval",
    }
    for draw in range(5):
        doc = {
            **info,
            "context_id": cid,
            "rollout_k": draw,
            "group_key": "g",
            "completion": f"Answer {draw}",
            "meta": {"seed": draw},
        }
        records.append({"src": f"{cid}_seed{draw}.json", "doc": doc})
    if defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records.append(records[0])
    elif defect == "prompt":
        records[-1]["doc"]["prompt_text"] = "changed"
    data = b"".join(examples.encoded(r) + b"\n" for r in records)
    path.write_bytes(data)
    info.update(
        source_path=str(path), source_record=f"{cid}_seed0.json", generation_meta={"seed": 0}
    )
    inputs = {
        "metadata": {"behavior": "evil", "source_files": {str(path): examples.digest(data)}},
        "contexts": {cid: info},
        "source_files": {},
        "means": {cid: 0.0},
        "labels": {cid: (label, 0)},
        "score_rows": {cid: (label, 0)},
        "label_provenance": {"sha256": "label"},
        "score_provenance": {"sha256": "score"},
    }
    roles = {cid: [{"dataset": "other", "variant": "q01_s0", "tail": "low"}]}
    if defect == "hash":
        path.write_bytes(data + b"\n")
    return inputs, roles


def test_preserves_null_and_zero_and_five_distinct_draws(tmp_path):
    inputs, roles = raw_inputs(tmp_path)
    writer = examples.Shards(tmp_path, 4000)
    coverage = examples.export_behavior(inputs, roles, {}, writer)
    writer.publish()
    rows = [json.loads(line) for part in writer.parts for line in (tmp_path / part["path"]).open()]
    assert coverage["realized_unique_response_rows"] == 5
    assert coverage["missing_judgments"] == 1
    assert {row["rollout_k"] for row in rows} == set(range(5))
    assert rows[0]["judgment"]["native_response_score"] == 0
    assert rows[1]["judgment"]["native_response_score"] is None
    assert rows[1]["judgment"]["judge_missing"] is True
    for part in writer.parts:
        data = (tmp_path / part["path"]).read_bytes()
        assert len(data) <= 4000 and examples.digest(data) == part["sha256"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "prompt", "hash"])
def test_rejects_incomplete_changed_or_duplicate_sources(tmp_path, defect):
    inputs, roles = raw_inputs(tmp_path, defect)
    with pytest.raises(ValueError):
        examples.export_behavior(inputs, roles, {}, examples.Shards(tmp_path, 4000))


def test_real_three_way_categories_do_not_collapse_zero_scores(tmp_path, monkeypatch):
    cid = "hallucination-train-train-000000"
    labels = ["fabricated", "correct", "abstained", "unjudged", "correct"]
    raw = {
        "three_way": {f"{cid}_k{k:02d}": label for k, label in enumerate(labels)},
        "abstain_judge": {"per_item_scores": {f"{cid}_k02": [1, 1, 1]}},
    }
    data = examples.encoded(raw)
    parts = {"part0": data[:50], "part1": data[50:]}
    for name, blob in parts.items():
        (tmp_path / name).write_bytes(blob)
    monkeypatch.setattr(
        examples, "HALLU_JUDGE_PARTS", {name: examples.digest(blob) for name, blob in parts.items()}
    )
    scores = {f"k{k:02d}": value for k, value in enumerate([100.0, 0.0, 0.0, None, 0.0])}
    row = {
        "per_rollout_scores": scores,
        "counts": {"correct": 2, "abstained": 1, "fabricated": 1},
        "n_unjudged": 1,
    }
    inputs = {"score_rows": {cid: (row, 0)}, "source_files": {}}
    examples.load_categories(tmp_path, inputs)
    assert inputs["categories"][f"{cid}_k01"] == "correct"
    assert inputs["categories"][f"{cid}_k02"] == "abstained"
    assert inputs["categories"][f"{cid}_k03"] == "unjudged"
    assert inputs["category_judge_rows"][f"{cid}_k02"]["per_item_scores"] == [1, 1, 1]
    row["counts"]["correct"] = 3
    with pytest.raises(ValueError, match="counts differ"):
        examples.load_categories(tmp_path, inputs)


def test_selection_layers_must_match(tmp_path):
    for layer, cid in (("L15", "a"), ("L22", "b")):
        path = tmp_path / "evil" / layer
        path.mkdir(parents=True)
        (path / "selection.json").write_text(json.dumps({"fold/q01_s0": {"high_ids": [cid]}}))
    with pytest.raises(ValueError, match="differ across layers"):
        examples.selections(tmp_path, "evil", False)
