"""Focused unit tests for the #2254 pure decode-only generation registry."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.issue2254_all_answer_decode_sweep as sweep


def test_dose_tokens_are_exact_and_reject_unregistered_values() -> None:
    expected = {
        0.0: "c0",
        1 / 64: "c1over64",
        1 / 32: "c1over32",
        1 / 16: "c1over16",
        1 / 8: "c1over8",
        1 / 4: "c1over4",
        1 / 2: "c1over2",
        1.0: "c1",
        2.0: "c2",
        4.0: "c4",
    }
    assert {dose: sweep.dose_token(dose) for dose in sweep.DOSES} == expected
    with pytest.raises(sweep.DecodeSweepError, match="unregistered dose"):
        sweep.dose_token(1 / 3)


def test_registered_cell_counts_routes_and_ids_are_exact() -> None:
    registered = sweep.registered_cells()
    analyzed = sweep.analysis_cells()

    assert len(registered) == 24
    assert len(analyzed) == 22
    assert len({cell.cell_id for cell in registered}) == 24
    assert sum(cell.route == "decodeonly" for cell in registered) == 20
    assert sum(cell.route == "context" for cell in registered) == 2
    assert sum(cell.route == "nohook" for cell in registered) == 2
    assert all(cell.route != "nohook" for cell in analyzed)

    ids = {cell.cell_id for cell in registered}
    assert "evil__rb__decodeonly__L14__c0" in ids
    assert "evil__rb__decodeonly__L14__c1over64" in ids
    assert "sycophancy__rb__decodeonly__L14__c4" in ids
    assert "evil__revmap__context__L14__c4" in ids
    assert "sycophancy__revmap__context__L14__c2" in ids
    assert "evil__none__nohook__L14__c0" in ids
    assert "sycophancy__none__nohook__L14__c0" in ids


def test_trim_generated_ids_stops_at_first_eos_and_keeps_the_eos() -> None:
    trimmed, reason = sweep._trim_generated_ids([1, 2, 10, 3, 9], [9, 10])

    assert trimmed == [1, 2, 10]
    assert reason == "eos_token"


def test_trim_generated_ids_accepts_only_full_cap_without_eos(monkeypatch) -> None:
    monkeypatch.setattr(sweep, "MAX_NEW_TOKENS", 4)

    assert sweep._trim_generated_ids([1, 2, 3, 4], 9) == (
        [1, 2, 3, 4],
        "length",
    )
    with pytest.raises(sweep.DecodeSweepError, match="non-EOS generation"):
        sweep._trim_generated_ids([1, 2, 3], 9)


def test_trim_uses_generation_eos_set_not_tokenizer_eos_and_handles_padding() -> None:
    # Qwen's generation config treats both <|endoftext|> and <|im_end|> as
    # terminal, while the tokenizer exposes only the former as eos_token_id.
    tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
    assert tokenizer.eos_token_id not in [7, 151643, 151643]

    trimmed, reason = sweep._trim_generated_ids(
        [7, tokenizer.pad_token_id, tokenizer.pad_token_id],
        sweep.EXPECTED_EOS_TOKEN_IDS,
    )

    assert trimmed == [7, 151643]
    assert reason == "eos_token"


def _pilot_record(*, wall: float = 100.0, disk_free: int | None = None) -> dict:
    return {
        "cell_id": sweep.PILOT_CELL_ID,
        "wall_seconds": wall,
        "completed_at_epoch": 900.0,
        "seeds": {
            str(sweep.SEED_BASE): {
                "completions": [[f"answer-{q}-{draw}" for draw in range(6)] for q in range(20)]
            }
        },
        "resources": {
            "hbm_total_bytes": 80 * 1024**3,
            "hbm_max_allocated_bytes": 16 * 1024**3,
            "hbm_max_reserved_bytes": 18 * 1024**3,
            "host_total_bytes": 240 * 1024**3,
            "host_max_rss_bytes": 20 * 1024**3,
            "disk_free_bytes": disk_free or 100 * 1024**3,
        },
    }


def test_production_shape_pilot_gate_passes_and_fails_wall_or_resource() -> None:
    passing = sweep._evaluate_pilot_gate(_pilot_record())
    assert passing["status"] == "PASS"
    assert passing["pilot_completions"] == 120
    assert passing["projected_fleet_wall_upper_seconds"] == pytest.approx(1200.0)
    assert passing["deadline_epoch"] == 900.0 + sweep.MAX_FLEET_WALL_SECONDS

    too_slow = sweep._evaluate_pilot_gate(_pilot_record(wall=1000.0))
    assert too_slow["status"] == "FAIL"
    assert too_slow["checks"]["projected_wall_within_1p5h"] is False

    low_disk = sweep._evaluate_pilot_gate(_pilot_record(disk_free=1))
    assert low_disk["status"] == "FAIL"
    assert low_disk["checks"]["disk_headroom"] is False


def test_full_generation_gate_is_hash_bound_and_deadline_bounded(tmp_path, monkeypatch) -> None:
    root = tmp_path
    raw = root / "generation" / "raw_completions"
    raw.mkdir(parents=True)
    record = _pilot_record()
    (raw / f"{sweep.PILOT_CELL_ID}.json").write_text(json.dumps(record))
    gate = sweep._evaluate_pilot_gate(record)
    (root / "generation" / "pilot_gate_report.json").write_text(json.dumps(gate))
    now = [1000.0]
    monkeypatch.setattr(sweep.time, "time", lambda: now[0])

    deadline = gate["deadline_epoch"]
    assert sweep._require_production_gate(root, deadline)["status"] == "PASS"
    with pytest.raises(sweep.DecodeSweepError, match="differs from pilot-bound"):
        sweep._require_production_gate(root, deadline + 1)
    now[0] = deadline + 1
    with pytest.raises(sweep.DecodeSweepError, match="already expired"):
        sweep._require_production_gate(root, deadline)
    now[0] = 1000.0
    record["wall_seconds"] = 101.0
    (raw / f"{sweep.PILOT_CELL_ID}.json").write_text(json.dumps(record))
    with pytest.raises(sweep.DecodeSweepError, match="not bound"):
        sweep._require_production_gate(root, deadline)


def test_downloaded_documents_are_bound_to_pack_and_generation_manifest() -> None:
    raw_name = "cell.json"
    raw_doc = {"cell_id": "cell", "values": [1, 2, 3]}
    inputs = {"model": "fixture"}
    generation_manifest = {
        "files": [{"path": raw_name, "sha256": sweep._canonical_sha256(raw_doc)}]
    }
    docs = {
        raw_name: raw_doc,
        "inputs_manifest.json": inputs,
        "generation_manifest.json": generation_manifest,
    }
    verification = {
        "document_canonical_sha256": {
            name: sweep._canonical_sha256(doc) for name, doc in docs.items()
        }
    }

    sweep._validate_downloaded_documents(docs, {raw_name}, verification)
    docs[raw_name] = {"cell_id": "cell", "values": [1, 2, 4]}
    with pytest.raises(sweep.DecodeSweepError, match="packed document differs"):
        sweep._validate_downloaded_documents(docs, {raw_name}, verification)


def test_direction_reload_reconciles_bytes_to_frozen_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sweep, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sweep, "HIDDEN_DIM", 3)
    rb_bank = {
        "evil": np.array([[0.0, 0.0, 0.0]] * sweep.LAYER + [[1.0, 2.0, 3.0]]),
        "sycophancy": np.array([[0.0, 0.0, 0.0]] * sweep.LAYER + [[3.0, 2.0, 1.0]]),
    }
    monkeypatch.setattr(sweep.i2254, "_load_rb_all", lambda: rb_bank)
    directions_dir = tmp_path / sweep.OUT_REL.parent / "directions"
    directions_dir.mkdir(parents=True)
    rows = []
    expected = {"rb_bf16": {}, "revmap": {}}
    for behavior, rev_values in (
        ("evil", [1.0, -1.0, 2.0]),
        ("sycophancy", [-2.0, 1.0, 1.0]),
    ):
        rb = torch.as_tensor(rb_bank[behavior][sweep.LAYER], dtype=torch.float32)
        rb = (rb / rb.norm()).to(torch.bfloat16)
        rb_sha = sweep._tensor_sha256(rb)
        path = directions_dir / f"{behavior}_revmap_L14.pt"
        torch.save({"direction": torch.tensor(rev_values)}, path)
        rev = torch.tensor(rev_values).float()
        rev = (rev / rev.norm()).to(torch.bfloat16)
        expected["rb_bf16"][behavior] = rb_sha
        expected["revmap"][behavior] = sweep._sha256_file(path)
        rows.append(
            {
                "behavior": behavior,
                "rb_bf16_sha256": rb_sha,
                "revmap_file": str(
                    sweep.OUT_REL.parent / "directions" / f"{behavior}_revmap_L14.pt"
                ),
                "revmap_file_sha256": sweep._sha256_file(path),
                "revmap_bf16_sha256": sweep._tensor_sha256(rev),
            }
        )
    monkeypatch.setattr(sweep, "EXPECTED_DIRECTION_SHA256", expected)

    loaded = sweep._load_directions({"directions": rows})
    assert set(loaded["rb"]) == set(sweep.BEHAVIORS)

    rows[0]["revmap_file_sha256"] = "0" * 64
    with pytest.raises(sweep.DecodeSweepError, match="frozen manifest"):
        sweep._load_directions({"directions": rows})
