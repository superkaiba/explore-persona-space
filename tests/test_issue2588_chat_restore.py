"""Exercise canonical pack/unpack and actual restoration on explicit test-only data."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue1739_pack as P
import issue2588_chat_restore as R


def test_restore_roundtrip_rejects_modified_source_and_target(tmp_path, monkeypatch):
    raw = tmp_path / "fixture"
    rows = raw / "raw_completions/train_10k/partial/initial/rows"
    rows.mkdir(parents=True)
    identity = {"source_sha": R.CG.SCIENCE_SHA, "fixture_only": True}
    for i in sorted(set(range(400)) - {338, 377}):
        row = {
            "row_id": f"train_10k_{i}",
            "stage": "train_10k",
            "gen_seed": 42,
            "cap": 32768,
            "finish_reason": "stop",
            "sampled_token_ids": [1],
            "n_comp_tokens": 1,
            "prompt_ids": [2],
            "n_prompt_tokens": 1,
        }
        (rows / f"row{i:06d}.json").write_text(json.dumps({"identity": identity, "row": row}))
    staged = tmp_path / "staged"
    packed = staged / "packed_generation_checkpoints"
    packed.mkdir(parents=True)
    shards = P.pack_group(raw, "rows", sorted(rows.glob("*.json")), packed)
    manifest = packed / "pack_manifest.json"
    manifest.write_text(
        json.dumps(
            {"version": P.MANIFEST_VERSION, "groups": {"rows": {"n_files": 398, "shards": shards}}}
        )
    )
    id_path = staged / "run_identity.json"
    id_path.write_text(json.dumps(identity))
    # Only immutable input hashes differ in this test-only local fixture.
    monkeypatch.setattr(R, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(R, "IDENTITY_SHA256", hashlib.sha256(id_path.read_bytes()).hexdigest())
    target = tmp_path / "target"
    result = R.restore_staged(staged, target)
    assert result["restored_original_rows"] == 398
    assert R.restore_staged(staged, target) == result
    altered = target / result["files"][0]["path"]
    altered.write_text("{}")
    with pytest.raises(SystemExit, match="DIFFERING"):
        R.restore_staged(staged, target)
    manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="manifest bytes"):
        R.restore_staged(staged, target)
