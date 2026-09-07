#!/usr/bin/env python3
"""Replay frozen ImpossibleBench prefixes and capture pre-action activations."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from scripts.context_risk_qwen38_capture import batches_by_budget  # noqa: E402
from scripts.context_risk_qwen38_smoke import (  # noqa: E402
    _capture_last_prefix,
    _load_model_and_tokenizer,
    render_prefix_ids,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as fh:
        np.savez(fh, **arrays)
        temporary = Path(fh.name)
    os.replace(temporary, path)


def load_manifest(path: Path, max_contexts: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if max_contexts is not None:
        rows = rows[:max_contexts]
    if not rows:
        raise RuntimeError("ImpossibleBench prefix-capture manifest is empty")
    for row in rows:
        if _digest(row["messages"]) != row["exact_context_sha256"]:
            raise RuntimeError(f"exact-context hash drift for {row['task_id']}:{row['condition']}")
    return rows


def run_capture(cfg: DictConfig) -> dict[str, Any]:
    import torch

    manifest_path = Path(str(cfg.manifest_path))
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    maximum_value = cfg.capture.max_contexts
    maximum = None if maximum_value is None else int(maximum_value)
    rows = load_manifest(manifest_path, maximum)
    fingerprint = _digest(
        {
            "schema_version": "context_risk_qwen38_impossible_capture_v2",
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "capture": OmegaConf.to_container(cfg.capture, resolve=True),
            "manifest_sha256": _sha256(manifest_path),
            "selected_contexts": [row["exact_context_sha256"] for row in rows],
        }
    )
    model, tokenizer, wrapper_depth = _load_model_and_tokenizer(cfg)
    layers = [int(layer) for layer in cfg.model.capture_layers]
    rendered_rows = [
        render_prefix_ids(
            tokenizer,
            row["messages"],
            enable_thinking=bool(cfg.capture.enable_thinking),
        )[1]
        for row in rows
    ]
    prefix_lengths = [len(ids) for ids in rendered_rows]
    over_limit = [
        {
            "task_id": row["task_id"],
            "condition": row["condition"],
            "n_prefix_tokens": length,
        }
        for row, length in zip(rows, prefix_lengths, strict=True)
        if length > int(cfg.capture.max_sequence_tokens)
    ]
    if over_limit:
        raise RuntimeError(
            f"{len(over_limit)} exact prefixes exceed the configured "
            f"{cfg.capture.max_sequence_tokens}-token limit; no prefix was truncated: {over_limit}"
        )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    chunk_reports = []
    checkpoint_rows = int(cfg.capture.checkpoint_rows)
    for chunk_index, start in enumerate(range(0, len(rows), checkpoint_rows)):
        chunk = rows[start : start + checkpoint_rows]
        stem = f"chunk_{chunk_index:04d}"
        npz_path = output_dir / f"{stem}.npz"
        rows_path = output_dir / f"{stem}.rows.jsonl"
        done_path = output_dir / f"{stem}.done.json"
        if done_path.is_file():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("fingerprint") != fingerprint:
                raise RuntimeError(f"{done_path}: capture fingerprint changed")
            if _sha256(npz_path) != done["npz_sha256"] or _sha256(rows_path) != done["rows_sha256"]:
                raise RuntimeError(f"{done_path}: capture payload hash mismatch")
            chunk_reports.append(done)
            continue
        rendered = [
            ids for ids in rendered_rows[start : start + len(chunk)]
        ]
        activations: dict[int, np.ndarray] = {}
        for batch in batches_by_budget(
            [len(ids) for ids in rendered],
            int(cfg.capture.batch_max_rows),
            int(cfg.capture.batch_max_tokens),
        ):
            batch_ids = [rendered[index] for index in batch]
            captured, replay_ids, replay_mask = _capture_last_prefix(model, batch_ids, layers)
            for local_index, chunk_index_in_batch in enumerate(batch):
                if int(replay_mask[local_index].sum()) != len(batch_ids[local_index]):
                    raise RuntimeError("prefix replay attention-mask geometry drifted")
                activations[chunk_index_in_batch] = captured[local_index].to(torch.float16).numpy()
            del replay_ids, replay_mask
        activation_array = np.stack([activations[index] for index in range(len(chunk))])
        expected = (len(chunk), len(layers), int(cfg.model.expected_hidden_dim))
        if activation_array.shape != expected or not np.isfinite(activation_array).all():
            raise RuntimeError(
                f"invalid ImpossibleBench activation array: {activation_array.shape}"
            )
        metadata = [
            {
                "task_id": row["task_id"],
                "condition": row["condition"],
                "exact_context_sha256": row["exact_context_sha256"],
                "prefix_token_ids_sha256": _digest(ids),
                "n_prefix_tokens": len(ids),
                "public_test_role": row["public_test_role"],
            }
            for row, ids in zip(chunk, rendered, strict=True)
        ]
        _write_npz_atomic(
            npz_path,
            activation=activation_array,
            layers=np.asarray(layers, dtype=np.int16),
        )
        _write_jsonl_atomic(rows_path, metadata)
        done = {
            "schema_version": "context_risk_qwen38_impossible_capture_chunk_v2",
            "fingerprint": fingerprint,
            "chunk_index": chunk_index,
            "n_contexts": len(chunk),
            "npz_sha256": _sha256(npz_path),
            "rows_sha256": _sha256(rows_path),
        }
        _write_json_atomic(done_path, done)
        chunk_reports.append(done)
        print(
            f"[context-risk-impossible-capture] chunk={chunk_index} "
            f"contexts={start + len(chunk)}/{len(rows)} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    report = {
        "schema_version": "context_risk_qwen38_impossible_capture_run_v2",
        "fingerprint": fingerprint,
        "model_id": str(cfg.model.id),
        "model_revision": str(cfg.model.revision),
        "manifest_path": str(manifest_path),
        "n_contexts": len(rows),
        "minimum_prefix_tokens": min(prefix_lengths),
        "maximum_prefix_tokens": max(prefix_lengths),
        "max_sequence_tokens": int(cfg.capture.max_sequence_tokens),
        "prefixes_truncated": 0,
        "capture_layers": layers,
        "wrapper_depth": int(wrapper_depth),
        "peak_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30),
        "elapsed_seconds": time.monotonic() - started,
        "chunks": chunk_reports,
        "passed": True,
    }
    _write_json_atomic(output_dir / "run_result.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


@hydra.main(
    version_base="1.3",
    config_path="../configs/eval",
    config_name="context_risk_qwen38_impossible_capture",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    run_capture(cfg)


if __name__ == "__main__":
    main()
