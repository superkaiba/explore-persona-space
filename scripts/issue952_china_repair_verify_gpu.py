"""Read-only, offline completion checks and exact-byte census for the repaired GPU run.

No generation, inference, SVD, upload, or teardown occurs here. Supply independently
recorded launch/input pins and a local tokenizer snapshot. Reports and optional text
archives must be outside the immutable run root. A local PASS is not a fresh remote
upload verification or permission to terminate compute.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import codecs
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts import issue952_china_definitive_gpu as gpu
from scripts import issue952_china_repair_persist as persist

LAYERS = [14, 19, 26]
WIDTH = 3584
COUNTS = {"smoke": (10, 160, 1280), "production": (85, 1360, 10880)}
EXCLUSIONS = {
    "_hf_stage": "Regenerable Hub input staging; canonical inputs and their pins are retained.",
    "_upload_verify": "Regenerable exact-revision download copies; originals and receipts retained.",
}


def _require(condition: bool, message: str) -> None:
    """Fail without including model questions or responses in diagnostics."""
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    """Read one required JSON object."""
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), f"not a JSON object: {path.name}")
    return value


def _hex(value: str, length: int) -> bool:
    """Recognize a full immutable digest, not a branch name or abbreviated SHA."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def load_local_tokenizer(directory: Path, regime: dict) -> Any:
    """Load only locally available, hash-matched tokenizer files; never fetch a model."""
    from transformers import AutoTokenizer

    expected = regime["tokenizer_artifact_sha256"]
    _require(
        set(expected) == {"config.json", "tokenizer.json", "tokenizer_config.json"},
        "tokenizer artifact pin set differs",
    )
    for name, digest in expected.items():
        _require(gpu._sha256(directory / name) == digest, f"tokenizer artifact drift: {name}")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    _require(
        hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        == regime["chat_template_sha256"],
        "tokenizer chat template drift",
    )
    return tokenizer


def _check_raw(rows: list[dict], bank: list[dict], tokenizer: Any, seed_base: int) -> None:
    """Independently reconcile all eight draws, metadata, seeds and exact token counts."""
    expected = [f"{prompt['item_id']}-d{draw}" for prompt in bank for draw in range(8)]
    ids = [row["item_id"] for row in rows]
    _require(
        ids == expected and len(set(ids)) == len(expected), "raw row identity/coverage mismatch"
    )
    for prompt_index, prompt in enumerate(bank):
        context_len = len(gpu._context_ids(tokenizer, prompt["prompt"]))
        for draw in range(8):
            row = rows[prompt_index * 8 + draw]
            tokens = row["completion_token_ids"]
            cap = 4096 if "cap_extension" in row else 2048
            _require(
                row["prompt_id"] == prompt["item_id"]
                and row["draw"] == draw
                and row["seed"] == seed_base + prompt_index * 8 + draw
                and row["question"] == prompt["prompt"]
                and all(
                    row[key] == prompt[key]
                    for key in (
                        "source_prompt_id",
                        "topic",
                        "language",
                        "frame",
                        "content",
                        "subject_arm",
                        "country_cue",
                        "cue_text",
                        "base_prompt_sha256",
                        "bank_contract",
                    )
                )
                and row["audit_pass"] is True
                and type(row["context_tokens"]) is int
                and row["context_tokens"] == context_len
                and isinstance(tokens, list)
                and 0 < len(tokens) <= cap
                and all(type(token) is int and token >= 0 for token in tokens)
                and row["completion_tokens"] == len(tokens)
                and isinstance(row["text"], str)
                and bool(row["text"])
                and row["finish_reason"] in ("stop", "length")
                and row["cap_hit"] is (row["finish_reason"] == "length"),
                "raw metadata/seed/token contract mismatch",
            )


def _tensor(store: dict, key: str, count: int) -> None:
    """Check the observed tensor itself, with no manifest-derived width or row count."""
    value = store[key]
    _require(
        isinstance(value, torch.Tensor) and value.shape == (count, 3, WIDTH),
        f"{key} observed tensor shape mismatch",
    )
    _require(
        value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
        f"{key} must be finite fp32",
    )


def validate_tensors(
    root: Path, bank: list[dict], rows: list[dict], capture: dict, tokenizer: Any
) -> dict:
    """Count actual context IDs and answer indices, checking every span and tensor."""
    directory = root / "analysis_tensors"
    expected_files = {"vc.pt", *capture["va_files"]}
    _require(
        {p.name for p in directory.iterdir() if p.is_file()} == expected_files,
        "tensor file census differs from capture manifest",
    )
    regime = capture["capture_regime"]
    fingerprint = gpu._sha_obj(regime)
    _require(capture["capture_regime_fp"] == fingerprint, "capture regime fingerprint drift")
    context = torch.load(directory / "vc.pt", map_location="cpu", weights_only=True)
    _require(
        context["item_ids"] == [row["item_id"] for row in bank], "context IDs incomplete/duplicated"
    )
    _tensor(context, "vc", len(bank))
    _require(
        context["position"] == "context_last_generation_prompt_token"
        and context["bank_sha256"] == regime["bank_sha256"],
        "context position/bank mismatch",
    )
    stores = [context]
    actual_indices, empty_global, cursor = [], [], 0
    eot = gpu._eot_ids(tokenizer)
    for name in sorted(capture["va_files"]):
        store = torch.load(directory / name, map_location="cpu", weights_only=True)
        indices = store["index"]
        _require(bool(indices), "empty answer shard")
        _tensor(store, "va_tail_incl", len(indices))
        _require(
            store["pooling"] == "completion_plus_im_end_newline_mean"
            and store["rollouts_sha256"] == regime["rollouts_sha256"],
            "answer pooling/raw pin mismatch",
        )
        _require(cursor + len(indices) <= len(rows), "surplus answer indices")
        for offset, index in enumerate(indices):
            row = rows[cursor + offset]
            ctx, comp = row["context_tokens"], row["completion_token_ids"]
            expected = {
                "item_id": row["item_id"],
                "prompt_id": row["prompt_id"],
                "draw": row["draw"],
                "ctx_len": ctx,
                "completion_len": len(comp),
                "span_start": ctx,
                "span_end": ctx + len(comp),
                "tail_end": ctx + len(gpu._with_eot_tail(comp, eot)),
            }
            _require(index == expected, "answer index identity/token boundary mismatch")
        expected_empty = [
            i
            for i, row in enumerate(rows[cursor : cursor + len(indices)])
            if not row["completion_token_ids"]
        ]
        _require(store["empty_rows"] == expected_empty, "answer empty-row index mismatch")
        empty_global.extend(cursor + i for i in expected_empty)
        actual_indices.extend((index["prompt_id"], index["draw"]) for index in indices)
        cursor += len(indices)
        # Validate metadata before dropping each shard; never accumulate the tensor grid.
        stores.append(
            {
                key: store[key]
                for key in ("layers", "model_revision", "capture_regime", "capture_regime_fp")
            }
        )
        del store
    _require(
        cursor == len(rows) and len(set(actual_indices)) == len(rows),
        "answer index coverage incomplete/duplicated",
    )
    _require(
        capture["n_contexts"] == len(context["item_ids"])
        and capture["n_answer_rows"] == cursor
        and capture["empty_answer_rows"] == empty_global
        and capture["n_empty_answer_rows"] == len(empty_global),
        "capture aggregate differs from observed indices",
    )
    for store in stores:
        _require(
            store["layers"] == LAYERS
            and store["model_revision"] == gpu.MODEL_REV
            and store["capture_regime"] == regime
            and store["capture_regime_fp"] == fingerprint,
            "tensor model/layer/regime metadata mismatch",
        )
    return {
        "observed_context_ids": len(context["item_ids"]),
        "observed_answer_indices": cursor,
        "distinct_prompt_draw_indices": len(set(actual_indices)),
        "layers": LAYERS,
        "hidden_width": WIDTH,
        "all_tensors_finite_fp32": True,
        "exact_token_boundaries_checked": cursor,
        "eot_token_ids": eot,
    }


def validate_mode(root: Path, *, mode: str, pins: dict, tokenizer: Any) -> dict:
    """Validate one completed immutable mode against independently recorded launch pins."""
    _require(mode in COUNTS, "unknown mode")
    for field, length in (("code_sha", 40), ("bank_sha256", 64), ("audit_sha256", 64)):
        _require(_hex(pins[field], length), f"invalid external {field}")
    _require(
        math.isfinite(pins["not_before_unix"]) and pins["not_before_unix"] > 0,
        "invalid launch timestamp",
    )
    manifests = {
        name: _json(root / "manifests" / f"{name}.json")
        for name in ("generation", "capture", "raw_upload", "capture_upload", "input_stage")
    }
    generation, capture = manifests["generation"], manifests["capture"]
    regime, cap_regime = generation["regime"], capture["capture_regime"]
    gpu._validate_study(regime, "repaired-v2")
    gpu._validate_study(cap_regime, "repaired-v2")
    _require(
        hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        == regime["chat_template_sha256"],
        "provided tokenizer template differs from generation",
    )
    smoke = mode == "smoke"
    bank_path, audit_path = (
        root / "inputs" / "prompt_bank.jsonl",
        root / "inputs" / "bank_audit_report.json",
    )
    _require(
        gpu._sha256(bank_path) == pins["bank_sha256"]
        and gpu._sha256(audit_path) == pins["audit_sha256"],
        "frozen input bytes differ from external pins",
    )
    bank, audit = gpu._load_bank(bank_path, audit_path, smoke, study="repaired-v2")
    accepted = gpu._accepted_bank_identity(gpu._read_jsonl(bank_path), audit)
    source_count, prompt_count, response_count = COUNTS[mode]
    _require(
        len(bank) == prompt_count
        and len({row["source_prompt_id"] for row in bank}) == source_count,
        "registered selected panel count mismatch",
    )
    _require(
        generation["accepted_bank"] == accepted and accepted["n_accepted_source_items"] == 85,
        "frozen 85-source identity mismatch",
    )
    for current in (regime, cap_regime):
        _require(
            current["git_sha"] == pins["code_sha"]
            and current["model_revision"] == gpu.MODEL_REV
            and current["bank_sha256"] == pins["bank_sha256"]
            and current["bank_audit_report_sha256"] == pins["audit_sha256"]
            and current["layers"] == LAYERS
            and current["accepted_source_ids_sha256"] == accepted["accepted_source_ids_sha256"]
            and current["selected_source_ids_sha256"]
            == gpu._sha_obj(sorted({r["source_prompt_id"] for r in bank}))
            and all(
                current[key] == audit[key]
                for key in ("source_bank_sha256", "metadata_sha256", "independent_audit_sha256")
            ),
            "generation/capture external provenance mismatch",
        )
    _require(
        regime["smoke"] is smoke
        and regime["draws"] == 8
        and regime["temperature"] == 1.0
        and regime["top_p"] == 0.95
        and regime["max_new_tokens"] == 2048
        and regime["seed_base"] == (9530880 if smoke else 9520000)
        and generation["regime_fp"] == gpu._sha_obj(regime),
        "generation recipe fingerprint mismatch",
    )
    marker = _json(root / "inputs" / "upload_verified.json")
    stage = manifests["input_stage"]
    _require(
        stage["marker_sha256"] == gpu._sha256(root / "inputs" / "upload_verified.json")
        and stage["data_revision"] == marker["data_revision"]
        and _hex(marker["data_revision"], 40)
        and all(
            stage[key] == marker[key] == pins[pin]
            for key, pin in (
                ("prompt_bank_sha256", "bank_sha256"),
                ("bank_audit_report_sha256", "audit_sha256"),
            )
        ),
        "input staging/upload marker mismatch",
    )
    rows = gpu._read_jsonl(root / "raw_completions" / "rollouts.jsonl")
    original = gpu._read_jsonl(root / "raw_completions" / "rollouts.initial.jsonl")
    for raw in (rows, original):
        _check_raw(raw, bank, tokenizer, regime["seed_base"])
    _require(
        len(rows) == response_count
        and generation["n_rows"] == len(rows)
        and generation["n_prompts"] == len(bank),
        "raw observed/manifest count mismatch",
    )
    shards = sorted((root / "raw_completions").glob("rollouts_p*.jsonl"))
    _require(
        bool(shards) and [row for path in shards for row in gpu._read_jsonl(path)] == original,
        "original generation checkpoint coverage differs",
    )
    for path in shards:
        done = _json(path.with_suffix(".done.json"))
        _require(
            done["sha256"] == gpu._sha256(path) and done["regime_fp"] == generation["regime_fp"],
            "original generation checkpoint manifest drift",
        )
    gpu._validate_raw_upload(root, generation, manifests["raw_upload"])
    gpu._validate_capture_artifacts(root, generation, capture)
    gpu._validate_capture_upload(root, capture, manifests["capture_upload"])
    for name, field in (
        ("raw_upload", "raw_payload_revision"),
        ("capture_upload", "tensor_payload_revision"),
    ):
        _require(_hex(manifests[name][field], 40), "upload receipt lacks immutable revision")
    observed = validate_tensors(root, bank, rows, capture, tokenizer)
    sentinel_path = root / "issue952_china_definitive_done.json"
    sentinel = _json(sentinel_path)
    _require(
        not (root / "issue952_china_definitive_done.pending.json").exists()
        and sentinel["status"] == "done"
        and sentinel["study"] == "repaired-v2"
        and sentinel["bank_contract"] == gpu.REPAIRED_CONTRACT
        and sentinel["hf_prefix"] == gpu.REPAIRED_HF_PREFIX
        and sentinel["timestamp_unix"] >= pins["not_before_unix"]
        and all(
            sentinel[name] == manifests[name]
            for name in ("generation", "capture", "raw_upload", "capture_upload")
        ),
        "terminal sentinel stale/incomplete or differs from current manifests",
    )
    if smoke:
        timing = _json(root / "manifests" / "smoke_timing.json")
        _require(
            sentinel["smoke_timing"] == timing and timing["passed"] is True,
            "smoke timing sentinel mismatch",
        )
    files = [sentinel_path]
    for directory in ("manifests", "inputs", "raw_completions", "analysis_tensors"):
        files.extend(path for path in sorted((root / directory).rglob("*")) if path.is_file())
    return {
        "passed": True,
        "mode": mode,
        "observed_source_items": source_count,
        "observed_final_raw_rows": len(rows),
        "observed_original_raw_rows": len(original),
        "frozen_accepted_source_items": 85,
        **observed,
        "remaining_truncation_rows": sum(row["finish_reason"] == "length" for row in rows),
        "pins": {path.relative_to(root).as_posix(): gpu._sha256(path) for path in files},
    }


def _file_record(path: Path) -> dict:
    """Stream-hash bytes, detect UTF-8 text, and reject a concurrently changing file."""
    before = path.stat()
    digest, decoder, text = hashlib.sha256(), codecs.getincrementaldecoder("utf-8")(), True
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            if text:
                try:
                    decoder.decode(block)
                    text = b"\0" not in block
                except UnicodeDecodeError:
                    text = False
    if text:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            text = False
    after = path.stat()
    _require(
        (before.st_size, before.st_mtime_ns, before.st_ino)
        == (after.st_size, after.st_mtime_ns, after.st_ino),
        f"file changed during census: {path.name}",
    )
    return {"sha256": digest.hexdigest(), "bytes": after.st_size, "utf8_text": text}


def census_tree(
    root: Path, *, attempt: int = 1, external_receipts: list[Path] | None = None
) -> dict:
    """Enumerate exact durable names/hashes and intended destinations without uploading."""
    root = root.resolve()
    _require(root.is_dir() and attempt >= 1, "missing run root or invalid attempt")
    files, excluded = {}, []
    archive_prefix = f"{gpu.REPAIRED_HF_PREFIX}/attempt{attempt}/gpu_completion_text"
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        _require(not path.is_symlink(), f"census refuses symlink: {relative}")
        if not path.is_file():
            continue
        name = relative.as_posix()
        if (
            len(relative.parts) >= 3
            and relative.parts[0] in COUNTS
            and relative.parts[1] in EXCLUSIONS
        ):
            excluded.append({"path": name, "reason": EXCLUSIONS[relative.parts[1]]})
            continue
        record = _file_record(path)
        destinations = []
        if len(relative.parts) >= 3 and relative.parts[0] in COUNTS:
            mode, family, *suffix = relative.parts
            prefix = gpu._output_prefix(mode == "smoke", attempt, "repaired-v2")
            if family in ("analysis_tensors", "raw_completions", "manifests"):
                destinations.append(
                    {
                        "kind": "direct_hf_file",
                        "repo": gpu.HF_REPO,
                        "path": f"{prefix}/{family}/{'/'.join(suffix)}",
                    }
                )
            elif family == "inputs":
                destinations.append(
                    {
                        "kind": "direct_hf_file",
                        "repo": gpu.HF_REPO,
                        "path": f"{gpu.REPAIRED_HF_PREFIX}/inputs/{'/'.join(suffix)}",
                    }
                )
        if (
            len(relative.parts) == 2
            and relative.parts[0] in COUNTS
            and relative.name == "issue952_china_definitive_done.json"
        ):
            prefix = gpu._output_prefix(relative.parts[0] == "smoke", attempt, "repaired-v2")
            destinations.append(
                {"kind": "direct_hf_file", "repo": gpu.HF_REPO, "path": f"{prefix}/{relative.name}"}
            )
        if record["utf8_text"]:
            destinations.append(
                {
                    "kind": "exact_utf8_archive_member",
                    "repo": gpu.HF_REPO,
                    "prefix": archive_prefix,
                    "member": name,
                    "format": persist.FORMAT,
                }
            )
        _require(bool(destinations), f"binary artifact lacks intended remote destination: {name}")
        files[name] = {
            **record,
            "destinations": destinations,
            "remote_status": "not_reverified_by_this_offline_helper",
        }
    _require(bool(files), "empty durable census")
    receipts = {}
    for path in external_receipts or []:
        path = path.resolve()
        _require(
            not path.is_relative_to(root), "upload receipt must be external to immutable run root"
        )
        receipts[str(path)] = {
            **_file_record(path),
            "intended_git_directory": f"eval_results/issue_952/china_repair_v2/gpu_completion/attempt{attempt}",
        }
    return {
        "format": "china-gpu-durable-census-v1",
        "root": str(root),
        "attempt": attempt,
        "files": files,
        "excluded": excluded,
        "external_receipts": receipts,
        "text_archive_prefix": archive_prefix,
        "receipt_policy": "Keep new upload receipts outside run/packed roots; commit receipts to issue-scoped Git and verify exact committed bytes. Do not claim a receipt uploaded itself.",
        "remote_verification": "Offline inventory only; direct files and any archive still require immutable-revision remote byte verification.",
    }


def validate_census(root: Path, census: dict) -> None:
    """Detect added, removed, altered or silently omitted durable files by exact sets."""
    current = census_tree(
        root,
        attempt=census["attempt"],
        external_receipts=[Path(path) for path in census["external_receipts"]],
    )
    _require(current == census, "durable census omitted, added or changed files/destinations")


def pack_text_residue(root: Path, census: dict, output: Path) -> dict:
    """Optionally pack every censused text byte outside the source, without uploading."""
    validate_census(root, census)
    names = [name for name, row in census["files"].items() if row["utf8_text"]]
    _require(bool(names), "no text files to archive")
    manifest = persist.pack_tree(root, output, include_dirs=names)
    _require(set(manifest["files"]) == set(names), "text archive omitted census members")
    for name in names:
        _require(
            manifest["files"][name]["sha256"] == census["files"][name]["sha256"],
            "text archive bytes differ from census",
        )
    return manifest


def main() -> int:
    """Write immutable offline verification/census reports outside the completed run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--bank-sha256", required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--not-before-unix", required=True, type=float)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--external-receipt", action="append", type=Path, default=[])
    parser.add_argument("--pack-text", action="store_true")
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.report_dir.resolve()
    _require(not output.is_relative_to(root), "reports must be outside immutable run root")
    pins = {
        key: getattr(args, key)
        for key in ("code_sha", "bank_sha256", "audit_sha256", "not_before_unix")
    }
    modes = {}
    regimes = {}
    capture_regimes = {}
    for mode in COUNTS:
        regime = _json(root / mode / "manifests" / "generation.json")["regime"]
        _require(regime["attempt"] == args.attempt, "run attempt differs from requested census")
        tokenizer = load_local_tokenizer(args.tokenizer_dir, regime)
        modes[mode] = validate_mode(root / mode, mode=mode, pins=pins, tokenizer=tokenizer)
        regimes[mode] = gpu._smoke_generation_compatibility(regime)
        capture = _json(root / mode / "manifests" / "capture.json")
        capture_regimes[mode] = gpu._smoke_capture_compatibility(
            capture["capture_regime"], capture["package_versions"]
        )
    _require(
        regimes["smoke"] == regimes["production"], "smoke/production generation recipes differ"
    )
    _require(
        capture_regimes["smoke"] == capture_regimes["production"],
        "smoke/production capture recipes differ",
    )
    census = census_tree(root, attempt=args.attempt, external_receipts=args.external_receipt)
    for mode, result in modes.items():
        for name, digest in result["pins"].items():
            _require(
                census["files"][f"{mode}/{name}"]["sha256"] == digest,
                "verified manifest changed before census",
            )
    if args.pack_text:
        pack_text_residue(root, census, output / "text_archive")
    report = {
        "passed": True,
        "external_pins": pins,
        "modes": modes,
        "census_sha256": persist.digest(persist.encode(census)),
        "scope": "local completion/content verification; not fresh remote verification or teardown authorization",
    }
    persist.immutable_write(output / "completion.json", persist.encode(report))
    persist.immutable_write(output / "census.json", persist.encode(census))
    print(
        json.dumps(
            {"passed": True, "report_dir": str(output), "durable_files": len(census["files"])}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
