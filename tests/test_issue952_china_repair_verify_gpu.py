"""Offline, synthetic observed-data tests for the independent GPU verifier."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from unittest.mock import create_autospec

import pytest
import torch

from scripts import issue952_china_definitive_gpu as GPU
from scripts import issue952_china_repair_bank as BANK
from scripts import issue952_china_repair_persist as PERSIST
from scripts import issue952_china_repair_verify_gpu as V


class Tokenizer:
    """Deliberately simple offline tokenizer; no real question or response content."""

    chat_template = "synthetic-test-template"

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt
        text = "fixture:" + messages[0]["content"]
        return self.encode(text) if tokenize else text

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 151645

    def __call__(self, text, *, add_special_tokens=False):
        assert text == "\n" and not add_special_tokens
        return {"input_ids": [198]}


def _fixture(tmp_path: Path, mode: str = "smoke") -> tuple[Path, dict, Tokenizer]:
    """Write the actual 85-source contract and indexed tensor shapes using tiny expanded storage."""
    root = tmp_path / "run" / mode
    base, metadata = [], {}
    for source in range(85):
        sid = f"s{source:03d}"
        metadata[sid] = {"control_country_en": "Canada", "control_country_zh": "加拿大"}
        for language in ("en", "zh"):
            for frame in ("direct", "academic"):
                for content in ("sensitive_country_neutral", "matched_non_china"):
                    base.append(
                        {
                            "item_id": f"{sid}-{language}-{frame}-{content}",
                            "source_prompt_id": sid,
                            "topic": f"t{source % 12}",
                            "language": language,
                            "frame": frame,
                            "content": content,
                            "prompt": f"Synthetic {sid} {language} {frame} {content}",
                            "audit_pass": True,
                        }
                    )
    full = BANK.render(base, metadata)
    bank_path = root / "inputs" / "prompt_bank.jsonl"
    audit_path = root / "inputs" / "bank_audit_report.json"
    GPU._write_jsonl(bank_path, full)
    audit = {
        "passed": True,
        "contract": BANK.CONTRACT,
        "prompt_bank_sha256": GPU._sha256(bank_path),
        "passing_item_ids": sorted(metadata),
        "n_audit_passing_items": 85,
        "source_bank_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "independent_audit_sha256": "c" * 64,
    }
    GPU._write_json(audit_path, audit)
    marker = {
        "data_revision": "d" * 40,
        "prompt_bank_sha256": GPU._sha256(bank_path),
        "bank_audit_report_sha256": GPU._sha256(audit_path),
    }
    GPU._write_json(root / "inputs" / "upload_verified.json", marker)
    GPU._write_json(
        root / "manifests" / "input_stage.json",
        {**marker, "marker_sha256": GPU._sha256(root / "inputs" / "upload_verified.json")},
    )
    bank, _ = GPU._load_bank(bank_path, audit_path, mode == "smoke", study="repaired-v2")
    accepted = GPU._accepted_bank_identity(full, audit)
    tok = Tokenizer()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_hashes = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        PERSIST.immutable_write(tokenizer_dir / name, b"{}")
        tokenizer_hashes[name] = GPU._sha256(tokenizer_dir / name)
    regime = GPU._regime(
        marker["prompt_bank_sha256"],
        mode == "smoke",
        study="repaired-v2",
        accepted=accepted,
        selected_rows=bank,
    )
    regime.update(
        git_sha="e" * 40,
        tokenizer_artifact_sha256=tokenizer_hashes,
        bank_audit_report_sha256=marker["bank_audit_report_sha256"],
        chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        **{
            key: audit[key]
            for key in ("source_bank_sha256", "metadata_sha256", "independent_audit_sha256")
        },
    )
    rows = []
    for index, prompt in enumerate(bank):
        for draw in range(8):
            tokens = ([10, 20], [10, 151645], [151645, 198])[draw % 3]
            rows.append(
                {
                    **{k: value for k, value in prompt.items() if k != "prompt"},
                    "item_id": f"{prompt['item_id']}-d{draw}",
                    "prompt_id": prompt["item_id"],
                    "draw": draw,
                    "seed": regime["seed_base"] + index * 8 + draw,
                    "question": prompt["prompt"],
                    "text": "Synthetic response",
                    "completion_token_ids": tokens,
                    "completion_tokens": len(tokens),
                    "context_tokens": len(GPU._context_ids(tok, prompt["prompt"])),
                    "finish_reason": "stop",
                    "cap_hit": False,
                }
            )
    raw = root / "raw_completions"
    for name in ("rollouts.jsonl", "rollouts.initial.jsonl"):
        GPU._write_jsonl(raw / name, rows)
    GPU._write_jsonl(raw / "rollouts.extensions.jsonl", [])
    initial = {
        "regime": regime,
        "regime_fp": GPU._sha_obj(regime),
        "n_prompts": len(bank),
        "n_rows": len(rows),
        "ordered_item_ids_sha256": GPU._sha_obj([r["item_id"] for r in rows]),
        "rollouts_sha256": GPU._sha256(raw / "rollouts.initial.jsonl"),
        "timing_evidence_complete": True,
        "accepted_bank": accepted,
    }
    GPU._write_json(root / "manifests" / "generation.initial.json", initial)
    for start in range(0, len(bank), 90):
        path = raw / f"rollouts_p{start:04d}_{min(start + 90, len(bank)):04d}.jsonl"
        GPU._write_jsonl(path, rows[start * 8 : (start + 90) * 8])
        GPU._write_json(
            path.with_suffix(".done.json"),
            {"regime_fp": initial["regime_fp"], "sha256": GPU._sha256(path)},
        )
    extension = {
        "policy": GPU.REPAIRED_EXTENSION_POLICY,
        "policy_checked": True,
        "initial_generation_manifest_sha256": GPU._sha256(
            root / "manifests" / "generation.initial.json"
        ),
        "initial_rollouts_sha256": initial["rollouts_sha256"],
        "required_item_ids": [],
        "n_extended_rows": 0,
        "extended_rollouts_sha256": GPU._sha256(raw / "rollouts.extensions.jsonl"),
        "merged_rollouts_sha256": GPU._sha256(raw / "rollouts.jsonl"),
        "shard_files": {},
    }
    GPU._write_json(root / "manifests" / "cap_extension.json", extension)
    generation = {
        **initial,
        "cap_extension_policy_checked": True,
        "cap_extension_manifest_sha256": GPU._sha256(root / "manifests" / "cap_extension.json"),
    }
    GPU._write_json(root / "manifests" / "generation.json", generation)
    prefix = GPU._output_prefix(mode == "smoke", 1, "repaired-v2")
    upload = {
        "revision": "f" * 40,
        "raw_payload_revision": "f" * 40,
        "rollouts_sha256": generation["rollouts_sha256"],
        "generation_fingerprint": GPU._generation_fingerprint(generation),
        "generation_manifest_sha256": GPU._sha256(root / "manifests" / "generation.json"),
        "byte_verified_sha256": {
            f"{prefix}/raw_completions/{name}": sha
            for name, sha in GPU._raw_payload_hashes(root).items()
        },
    }
    GPU._write_json(root / "manifests" / "raw_upload.json", upload)
    cap_regime = {
        **{
            key: regime[key]
            for key in (
                "study",
                "hf_prefix",
                "bank_contract",
                "git_sha",
                "model_revision",
                "layers",
                "bank_sha256",
                "bank_audit_report_sha256",
                "source_bank_sha256",
                "metadata_sha256",
                "independent_audit_sha256",
                "accepted_source_ids_sha256",
                "selected_source_ids_sha256",
                "n_selected_prompts",
                "n_accepted_prompts",
                "n_accepted_source_items",
            )
        },
        "generation_fingerprint": GPU._generation_fingerprint(generation),
        "rollouts_sha256": generation["rollouts_sha256"],
    }
    common = {
        "layers": V.LAYERS,
        "model_revision": GPU.MODEL_REV,
        "capture_regime": cap_regime,
        "capture_regime_fp": GPU._sha_obj(cap_regime),
    }
    directory = root / "analysis_tensors"
    GPU._save_pt(
        directory / "vc.pt",
        {
            **common,
            "item_ids": [r["item_id"] for r in bank],
            "vc": torch.ones(1, 3, 3584).expand(len(bank), -1, -1),
            "position": "context_last_generation_prompt_token",
            "bank_sha256": regime["bank_sha256"],
        },
    )
    paths = []
    for start in range(0, len(rows), 720):
        chunk = rows[start : start + 720]
        indices = [
            {
                "item_id": r["item_id"],
                "prompt_id": r["prompt_id"],
                "draw": r["draw"],
                "ctx_len": r["context_tokens"],
                "completion_len": r["completion_tokens"],
                "span_start": r["context_tokens"],
                "span_end": r["context_tokens"] + r["completion_tokens"],
                "tail_end": r["context_tokens"]
                + len(GPU._with_eot_tail(r["completion_token_ids"], [151645, 198])),
            }
            for r in chunk
        ]
        path = directory / f"va_{start:05d}_{start + len(chunk):05d}.pt"
        GPU._save_pt(
            path,
            {
                **common,
                "index": indices,
                "va_tail_incl": torch.ones(1, 3, 3584).expand(len(chunk), -1, -1),
                "empty_rows": [],
                "pooling": "completion_plus_im_end_newline_mean",
                "rollouts_sha256": generation["rollouts_sha256"],
            },
        )
        paths.append(path)
    capture = {
        **common,
        "package_versions": {},
        "generation_fingerprint": GPU._generation_fingerprint(generation),
        "n_contexts": len(bank),
        "n_answer_rows": len(rows),
        "n_empty_answer_rows": 0,
        "empty_answer_rows": [],
        "timing_evidence_complete": True,
        "vc_sha256": GPU._sha256(directory / "vc.pt"),
        "va_files": {p.name: GPU._sha256(p) for p in paths},
    }
    GPU._write_json(root / "manifests" / "capture.json", capture)
    _repin_capture(root)
    pins = {
        "code_sha": "e" * 40,
        "bank_sha256": regime["bank_sha256"],
        "audit_sha256": regime["bank_audit_report_sha256"],
        "not_before_unix": 900.0,
    }
    return root, pins, tok


def _repin_capture(root: Path) -> None:
    """Make corruption fixtures self-consistent at manifest grain to exercise data checks."""
    capture = V._json(root / "manifests" / "capture.json")
    capture["vc_sha256"] = GPU._sha256(root / "analysis_tensors" / "vc.pt")
    capture["va_files"] = {
        name: GPU._sha256(root / "analysis_tensors" / name) for name in capture["va_files"]
    }
    GPU._write_json(root / "manifests" / "capture.json", capture)
    prefix = GPU._output_prefix(root.name == "smoke", 1, "repaired-v2")
    verified = {
        f"{prefix}/analysis_tensors/{name}": digest
        for name, digest in {"vc.pt": capture["vc_sha256"], **capture["va_files"]}.items()
    }
    upload = {
        "revision": "1" * 40,
        "tensor_payload_revision": "1" * 40,
        "capture_manifest_sha256": GPU._sha256(root / "manifests" / "capture.json"),
        "raw_upload_manifest_sha256": GPU._sha256(root / "manifests" / "raw_upload.json"),
        "capture_fingerprint": GPU._capture_fingerprint(capture),
        "vc_sha256": capture["vc_sha256"],
        "va_files": capture["va_files"],
        "byte_verified_sha256": verified,
        "byte_verified_files": sorted(verified),
    }
    GPU._write_json(root / "manifests" / "capture_upload.json", upload)
    sentinel = {
        "status": "done",
        "study": "repaired-v2",
        "bank_contract": GPU.REPAIRED_CONTRACT,
        "hf_prefix": GPU.REPAIRED_HF_PREFIX,
        "timestamp_unix": 1000.0,
        **{
            name: V._json(root / "manifests" / f"{name}.json")
            for name in ("generation", "capture", "raw_upload", "capture_upload")
        },
    }
    if root.name == "smoke":
        sentinel["smoke_timing"] = {"passed": True}
        GPU._write_json(root / "manifests" / "smoke_timing.json", sentinel["smoke_timing"])
    GPU._write_json(root / "issue952_china_definitive_done.json", sentinel)


@pytest.mark.parametrize("mode", ["smoke", "production"])
def test_complete_actual_mode_shapes_and_indices(tmp_path, mode):
    root, pins, tokenizer = _fixture(tmp_path, mode)
    result = V.validate_mode(root, mode=mode, pins=pins, tokenizer=tokenizer)
    _, prompts, responses = V.COUNTS[mode]
    assert result["observed_context_ids"] == prompts
    assert result["observed_answer_indices"] == responses
    assert result["exact_token_boundaries_checked"] == responses
    assert result["hidden_width"] == 3584
    assert result["frozen_accepted_source_items"] == 85


@pytest.mark.parametrize(
    "mutation", ["shape", "nonfinite", "index_missing", "index_duplicate", "tail", "layer"]
)
def test_self_consistent_manifests_do_not_hide_tensor_defects(tmp_path, mutation):
    root, pins, tokenizer = _fixture(tmp_path)
    path = sorted((root / "analysis_tensors").glob("va_*.pt"))[0]
    store = torch.load(path, weights_only=True)
    if mutation == "shape":
        store["va_tail_incl"] = store["va_tail_incl"][..., :2]
    elif mutation == "nonfinite":
        store["va_tail_incl"] = store["va_tail_incl"].clone()
        store["va_tail_incl"][0, 0, 0] = float("nan")
    elif mutation == "index_missing":
        store["index"].pop()
        store["va_tail_incl"] = store["va_tail_incl"][:-1]
    elif mutation == "index_duplicate":
        store["index"][1] = store["index"][0]
    elif mutation == "tail":
        store["index"][0]["tail_end"] -= 1
    else:
        store["layers"][0] = 13
    GPU._save_pt(path, store)
    _repin_capture(root)
    with pytest.raises(ValueError):
        V.validate_mode(root, mode="smoke", pins=pins, tokenizer=tokenizer)


@pytest.mark.parametrize(
    "mutation", ["missing_raw", "stale_manifest", "stale_sentinel", "wrong_bank_pin"]
)
def test_raw_and_external_completion_pins_are_required(tmp_path, mutation):
    root, pins, tokenizer = _fixture(tmp_path)
    if mutation == "missing_raw":
        path = root / "raw_completions" / "rollouts.jsonl"
        GPU._write_jsonl(path, GPU._read_jsonl(path)[:-1])
    elif mutation == "stale_manifest":
        path = root / "manifests" / "capture.json"
        document = V._json(path)
        document["n_answer_rows"] -= 1
        GPU._write_json(path, document)
    elif mutation == "stale_sentinel":
        pins["not_before_unix"] = 1001.0
    else:
        pins["bank_sha256"] = "0" * 64
    with pytest.raises((ValueError, RuntimeError)):
        V.validate_mode(root, mode="smoke", pins=pins, tokenizer=tokenizer)


def test_census_covers_all_residue_and_archives_exact_text(tmp_path):
    root, _, _ = _fixture(tmp_path)
    run = root.parent
    for name in (
        "logs/phase.log",
        "root_sentinel.json",
        "production/raw_completions/extensions/extra.done.json",
        "smoke/_hf_stage/cache.txt",
        "smoke/_upload_verify/revision/check.txt",
    ):
        PERSIST.immutable_write(run / name, b"exact original bytes\n")
    receipt = tmp_path / "external" / "upload.json"
    PERSIST.immutable_write(receipt, b'{"passed": true}\n')
    census = V.census_tree(run, external_receipts=[receipt])
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    excluded = {row["path"] for row in census["excluded"]}
    assert set(census["files"]) | excluded == actual
    assert len(excluded) == 2
    assert {
        "logs/phase.log",
        "root_sentinel.json",
        "production/raw_completions/extensions/extra.done.json",
    } <= set(census["files"])
    assert str(receipt) in census["external_receipts"]
    manifest = V.pack_text_residue(run, census, tmp_path / "archive")
    assert set(manifest["files"]) == {
        name for name, row in census["files"].items() if row["utf8_text"]
    }
    _, restored = PERSIST.verify_archive(tmp_path / "archive")
    assert restored["logs/phase.log"] == b"exact original bytes\n"
    V.validate_census(run, census)
    omitted = copy.deepcopy(census)
    del omitted["files"]["logs/phase.log"]
    with pytest.raises(ValueError, match="census omitted"):
        V.validate_census(run, omitted)
    PERSIST.immutable_write(run / "new.txt", b"new durable content")
    with pytest.raises(ValueError, match="census omitted"):
        V.validate_census(run, census)


def test_census_refuses_unknown_binary_symlinks_and_internal_receipts(tmp_path):
    run = tmp_path / "run"
    PERSIST.immutable_write(run / "unknown.bin", b"\xff\x00")
    with pytest.raises(ValueError, match="binary artifact"):
        V.census_tree(run)
    (run / "unknown.bin").unlink()
    PERSIST.immutable_write(run / "receipt.json", b"{}")
    with pytest.raises(ValueError, match="external"):
        V.census_tree(run, external_receipts=[run / "receipt.json"])
    (run / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        V.census_tree(run)


def test_local_tokenizer_hash_binding_and_offline_loading(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    tokenizer = Tokenizer()
    hashes = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        PERSIST.immutable_write(tmp_path / name, b"{}")
        hashes[name] = GPU._sha256(tmp_path / name)
    regime = {
        "tokenizer_artifact_sha256": hashes,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }
    load = create_autospec(AutoTokenizer.from_pretrained, return_value=tokenizer)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    assert V.load_local_tokenizer(tmp_path, regime) is tokenizer
    load.assert_called_once_with(str(tmp_path), local_files_only=True)
    hashes["config.json"] = "0" * 64
    with pytest.raises(ValueError, match="tokenizer artifact drift"):
        V.load_local_tokenizer(tmp_path, regime)


def test_cli_rejects_report_directory_inside_run(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--run-root",
            str(tmp_path),
            "--report-dir",
            str(tmp_path / "reports"),
            "--tokenizer-dir",
            str(tmp_path),
            "--code-sha",
            "e" * 40,
            "--bank-sha256",
            "a" * 64,
            "--audit-sha256",
            "b" * 64,
            "--not-before-unix",
            "900",
        ],
    )
    with pytest.raises(ValueError, match="outside immutable"):
        V.main()


def test_complete_offline_cli_checks_both_modes_and_packs_outside_run(tmp_path, monkeypatch):
    """Exercise the full entrypoint with only tokenizer construction at an external seam."""
    import sys

    from transformers import AutoTokenizer

    smoke, pins, tokenizer = _fixture(tmp_path, "smoke")
    _fixture(tmp_path, "production")
    run = smoke.parent
    PERSIST.immutable_write(run / "logs" / "completed.log", b"synthetic exit rc=0\n")
    baseline = V.census_tree(run)
    loader = create_autospec(AutoTokenizer.from_pretrained, return_value=tokenizer)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loader)
    monkeypatch.setattr(
        GPU,
        "HfApi",
        create_autospec(
            GPU.HfApi, side_effect=AssertionError("offline verifier attempted network")
        ),
    )
    output = tmp_path / "verification"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--run-root",
            str(run),
            "--report-dir",
            str(output),
            "--tokenizer-dir",
            str(tmp_path / "tokenizer"),
            "--code-sha",
            pins["code_sha"],
            "--bank-sha256",
            pins["bank_sha256"],
            "--audit-sha256",
            pins["audit_sha256"],
            "--not-before-unix",
            "900",
            "--pack-text",
        ],
    )
    assert V.main() == 0
    result = V._json(output / "completion.json")
    census = V._json(output / "census.json")
    assert result["modes"]["production"]["observed_answer_indices"] == 10880
    assert result["census_sha256"] == GPU._sha256(output / "census.json")
    assert census == baseline
    assert V.census_tree(run) == baseline
    assert loader.call_count == 2
    assert V.main() == 0  # Immutable reports can be verified/reused with unchanged bytes.
