"""Score-blind matched-subset and outbound-leakage checks on real builder bodies."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("issue2669_subset", SCRIPTS / "issue2669_subset.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rows_for_selection():
    return [
        {
            "id": f"opaque-{i}",
            "regime": "id",
            "fold_id": i % 5,
            "dv": i,
            "group_key": f"group-{i}",
            "rung": "unused",
        }
        for i in range(143)
    ]


def test_selection_invariant_to_labels_and_input_order():
    rows = rows_for_selection()
    selected, audit = m.select(rows, 100, "test")
    altered = [
        {**row, "dv": -999 - i, "prediction": 10**9, "prompt_text": "changed"}
        for i, row in enumerate(reversed(rows))
    ]
    again, second_audit = m.select(altered, 100, "test")
    assert [r["id"] for r in again] == [r["id"] for r in selected]
    assert audit == second_audit
    assert len(set(r["id"] for r in selected)) == 100
    assert sum(row["selected"] for row in audit) == 100
    assert all(row["selected"] <= row["population"] for row in audit)


def test_proportional_allocation_and_ood_strata():
    assert m.allocate({"a": 70, "b": 50}, 100) == {"a": 58, "b": 42}
    assert m.allocate({"c": 1, "b": 1, "a": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    rows = [{"id": f"x{i}", "regime": "ood", "rung": "a" if i < 70 else "b"} for i in range(120)]
    selected, _ = m.select(rows, 100, "ood")
    assert Counter(row["rung"] for row in selected) == {"a": 58, "b": 42}
    with pytest.raises(ValueError):
        m.select(rows, 121, "ood")


def stage_fixture(root):
    rows, contexts = [], []
    for behavior in ("evil", "sycophancy", "hallucination"):
        for regime in ("id", "generic", "ood", "development"):
            for i in range(40 if regime == "development" else 120):
                cid = f"opaque-{behavior}-{regime}-{i}"
                row = {
                    "id": cid,
                    "behavior": behavior,
                    "regime": regime,
                    "fold_id": i // 2 % 5 if regime in {"id", "development"} else None,
                    "group_key": f"secret-group-{regime}-{i // 2}",
                    "rung": "a" if regime != "ood" or i < 70 else "b",
                    "dv": 27.125,
                    "eligible": True,
                    "reference_aliases": ["SECRET_ALIAS"],
                    "judge_rationale": "SECRET_OUTCOME_RATIONALE",
                }
                instrument = (
                    "fabrication"
                    if behavior == "hallucination" and regime in {"id", "ood"}
                    else "trait"
                )
                context = {
                    "id": cid,
                    "prompt_text": f"<user>Question {i}</user><assistant>",
                    "instrument": instrument,
                    "target": {"model": "Qwen-test"},
                    "reference_aliases": ["SECRET_ALIAS"],
                    "dv": "SECRET_HELDOUT_LABEL",
                    "group_key": "SECRET_GROUP_KEY",
                    "continuation": "SECRET_TARGET_ANSWER",
                }
                rows.append(row)
                contexts.append(context)
    root.mkdir()
    for name, data in (("cohort.jsonl", rows), ("contexts.jsonl", contexts)):
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in data))
    (root / "rubrics.json").write_text(
        json.dumps(
            {
                key: f"{key} trait rubric.\n\nPrompt: {{question}}\nAnswer: {{answer}}"
                for key in ("evil", "sycophancy", "hallucination")
            }
        )
    )
    (root / "staging_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "output_sha256": {
                    name: m.hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ("cohort.jsonl", "contexts.jsonl", "rubrics.json")
                },
            }
        )
    )
    return {r["id"]: r for r in rows}, {c["id"]: c for c in contexts}


def test_full_builder_matched_900_masks_and_blinded_demos(tmp_path):
    source, output = tmp_path / "stage", tmp_path / "out"
    rows, contexts = stage_fixture(source)
    m.build(source, output)
    selection = json.loads((output / "selection.json").read_text())
    assert len(selection["selected_ids"]) == len(set(selection["selected_ids"])) == 900
    assert len(selection["pilot_ids"]) == 48
    assert set(selection["pilot_ids"]) <= set(selection["selected_ids"])
    assert Counter(
        (rows[cid]["behavior"], rows[cid]["regime"]) for cid in selection["selected_ids"]
    ) == {
        (behavior, regime): 100
        for behavior in ("evil", "sycophancy", "hallucination")
        for regime in ("id", "generic", "ood")
    }
    audit = {
        a["packet_id"]: a
        for a in json.loads((output / "packet_manifest.json").read_text())["audit"]
    }
    for phase, expected_ids, repeats in (
        ("pilot", selection["pilot_ids"], 3),
        ("production", selection["selected_ids"], 1),
    ):
        packets = json.loads((output / f"{phase}_config.json").read_text())["packets"]
        for shot in (0, 32):
            for repeat in range(repeats):
                actual = [
                    cid
                    for p in packets
                    if p["shot"] == shot and p["repeat"] == repeat
                    for cid in p["ids"]
                ]
                assert Counter(actual) == Counter(expected_ids)
        for packet in packets:
            prompt = Path(packet["prompt_path"]).read_text()
            assert not any(
                secret in prompt
                for secret in (
                    "SECRET_ALIAS",
                    "SECRET_OUTCOME_RATIONALE",
                    "SECRET_HELDOUT_LABEL",
                    "SECRET_GROUP_KEY",
                    "SECRET_TARGET_ANSWER",
                    "secret-group-",
                )
            )
            target_payload = json.loads(
                prompt.split("Evaluation contexts:\n")[1].split("\nJudge each")[0]
            )
            assert all(set(row) == {"id", "context"} for row in target_payload)
            demos = audit[packet["id"]]["demonstration_ids"]
            assert len(demos) == packet["shot"]
            assert not set(demos) & set(packet["ids"])
            for cid in demos:
                assert rows[cid]["regime"] in {"id", "development"}
                assert contexts[cid]["instrument"] == packet["instrument"]
                if packet["regime"] == "id":
                    assert rows[cid]["fold_id"] != audit[packet["id"]]["fold"]
                assert all(
                    (rows[cid]["regime"], rows[cid]["group_key"])
                    != (rows[target]["regime"], rows[target]["group_key"])
                    for target in packet["ids"]
                )
            if packet["behavior"] == "hallucination":
                if packet["regime"] in {"id", "ood"}:
                    assert "fraction of five" in prompt and "FABRICATED" in prompt
                else:
                    assert "expected NUMERIC trait score" in prompt
                    assert "CONDITIONAL ON" in prompt


def test_stage_hash_change_fails_before_selection(tmp_path):
    source, output = tmp_path / "stage", tmp_path / "out"
    stage_fixture(source)
    with (source / "cohort.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Staged input changed"):
        m.build(source, output)
    assert not output.exists()
