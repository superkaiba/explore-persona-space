"""Provenance and leakage-hash invariants for the fixed-direction metadata stage."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.issue1739_transfer_metadata import (
    context_record,
    merge_record,
    read_group,
    text_hashes,
    user_turn_hashes,
)


def packed_doc(*, extraction=False, rollout_k=0, prompt=None):
    """Build the recorded generation schema without any model/runtime dependency."""
    doc = {
        "prompt_text": prompt
        or "<|im_start|>user\n A first QUESTION \n<|im_end|>\n<|im_start|>assistant\n",
        "meta": {
            "model": "model",
            "revision": "revision",
            "fingerprint": "fingerprint",
            "git_commit": "commit",
        },
    }
    if extraction:
        doc.update(pair=2, sign="neg", q_idx=3, question="A first QUESTION", rollouts=[{}, {}])
        source = "extraction/evil/pair2_neg_q03.json"
    else:
        doc.update(context_id="ctx1", query="A first QUESTION", rollout_k=rollout_k)
        source = f"labeling/evil/ctx1_seed{rollout_k}.json"
    return {"src": source, "doc": doc}


def test_hashes_match_parent_without_unicode_normalization():
    exact, normalized = text_hashes("  A\tQUESTION\n")
    assert exact == hashlib.sha256(b"  A\tQUESTION\n").hexdigest()
    assert normalized == hashlib.sha256(b"a question").hexdigest()
    assert text_hashes("\u00e9")[1] != text_hashes("e\u0301")[1]


def test_user_turn_hashes_preserve_first_turn_and_strip_like_map():
    prompt = (
        "<|im_start|>system\nPolicy<|im_end|>\n"
        "<|im_start|>user\n  First  \n<|im_end|>\n"
        "<|im_start|>assistant\nReply<|im_end|>\n"
        "<|im_start|>user\nSecond<|im_end|>\n<|im_start|>assistant\n"
    )
    result = user_turn_hashes(prompt)
    assert [r["exact_sha256"] for r in result] == [text_hashes(t)[0] for t in ("First", "Second")]
    with pytest.raises(ValueError, match="no complete Qwen user turn"):
        user_turn_hashes("<|im_start|>user\nIncomplete")


def test_extraction_identity_matches_capture_and_keeps_all_rollouts():
    row = context_record(packed_doc(extraction=True), "extraction_evil")
    assert row["context_id"] == "e1-pair2-neg-q03"
    assert row["namespace"] == "evil_extraction"
    assert [r["rollout_k"] for r in row["rollouts"]] == [0, 1]
    assert row["first_user_exact_sha256"] == text_hashes("A first QUESTION")[0]
    assert row["generation_metadata"][0]["revision"] == "revision"


def test_merge_rejects_duplicate_rollouts_and_changed_prompt():
    index = {}
    merge_record(index, context_record(packed_doc(), "labeling_evil"))
    merge_record(index, context_record(packed_doc(rollout_k=1), "labeling_evil"))
    assert len(index[("evil_labeling", "ctx1")]["rollouts"]) == 2
    with pytest.raises(ValueError, match="duplicate rollout identity"):
        merge_record(index, context_record(packed_doc(rollout_k=1), "labeling_evil"))
    changed = packed_doc(rollout_k=2, prompt="<|im_start|>user\nDifferent<|im_end|>")
    with pytest.raises(ValueError, match="inconsistent"):
        merge_record(index, context_record(changed, "labeling_evil"))


def test_group_census_keeps_manifest_count_and_rejects_missing_documents(tmp_path):
    path = tmp_path / "extraction_evil.shard00.jsonl"
    rows = [
        {"src": "extraction/evil/_manifest.json", "doc": {"n_generated": 1}},
        packed_doc(extraction=True),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta = {"n_files": 2, "rel_dir": "extraction/evil"}
    counts = read_group([(path, {"n_lines": 2})], "extraction_evil", meta, {})
    assert counts == {"source_documents": 2, "manifest_documents": 1, "rollouts": 2}
    with pytest.raises(ValueError, match="file-count mismatch"):
        read_group([(path, {"n_lines": 2})], "extraction_evil", {**meta, "n_files": 3}, {})
