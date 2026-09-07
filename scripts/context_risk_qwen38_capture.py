#!/usr/bin/env python3
"""Capture paired pre-generation context and teacher-forced answer states.

This is the mapping feasibility gate for Qwen3.8-27B. Each accepted row is
rendered once as an exact assistant-generation prefix and once with its next
assistant turn. The full tokenization must begin with the exact prefix IDs;
otherwise the row is rejected rather than silently moving the decision boundary.
Fixed-size chunks and last-written sentinels make the capture resumable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from scripts.context_risk_prepare_data import normalize_tools  # noqa: E402
from scripts.context_risk_qwen38_smoke import (  # noqa: E402
    _load_model_and_tokenizer,
    render_prefix_ids,
)


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def iter_pair_rows(data_dir: Path, sources: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield pair-bank rows in declared source and file order."""
    for source in sources:
        path = data_dir / f"{source}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing source bank: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("source") != source:
                    raise RuntimeError(
                        f"{path}:{line_number}: source {row.get('source')!r} != {source!r}"
                    )
                yield row


def render_pair(tokenizer, row: dict[str, Any], *, enable_thinking: bool) -> dict[str, Any]:
    """Render one exact prefix/full pair and return token-span metadata."""
    context_messages = row["context_messages"]
    answer_message = dict(row["answer_message"])
    if not enable_thinking:
        # A thinking-off generation prefix contains an already-closed empty
        # <think> block. Teacher-forcing source reasoning into that block would
        # change the causal prefix, so retain only the observable answer/action.
        answer_message.pop("reasoning_content", None)
    # Some trajectory corpora serialize each JSON tool schema separately even
    # when the outer inventory is already a list. Normalize again at the render
    # boundary so previously materialized pair banks remain usable.
    tools = normalize_tools(row.get("tools")) or None
    prefix_text, prefix_ids = render_prefix_ids(
        tokenizer,
        context_messages,
        tools=tools,
        enable_thinking=enable_thinking,
    )
    full_messages = [*context_messages, answer_message]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    full_text = tokenizer.apply_chat_template(full_messages, **kwargs)
    full_ids = [int(value) for value in tokenizer(full_text, add_special_tokens=False)["input_ids"]]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        mismatch = next(
            (
                index
                for index, (prefix_id, full_id) in enumerate(zip(prefix_ids, full_ids))
                if prefix_id != full_id
            ),
            min(len(prefix_ids), len(full_ids)),
        )
        raise ValueError(
            f"full conversation does not preserve exact generation prefix at token {mismatch}"
        )
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("teacher-forced assistant turn has no rendered tokens")
    return {
        "prefix_text": prefix_text,
        "full_text": full_text,
        "full_ids": full_ids,
        "context_position": len(prefix_ids) - 1,
        "answer_start": len(prefix_ids),
        "answer_end": len(full_ids),
        "prefix_ids_sha256": _stable_digest(prefix_ids),
        "full_ids_sha256": _stable_digest(full_ids),
    }


def batches_by_budget(lengths: list[int], max_rows: int, max_tokens: int) -> list[list[int]]:
    """Pack long-first batches under row and padded-token budgets."""
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        candidate_max = max(current_max, lengths[index])
        if current and (
            len(current) >= max_rows or (len(current) + 1) * candidate_max > max_tokens
        ):
            batches.append(current)
            current = []
            current_max = 0
            candidate_max = lengths[index]
        current.append(index)
        current_max = candidate_max
    if current:
        batches.append(current)
    return batches


def _prepare_chunk(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    max_sequence_tokens: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    accepted = []
    rejected = []
    for row in rows:
        try:
            rendered = render_pair(tokenizer, row, enable_thinking=enable_thinking)
            n_tokens = len(rendered["full_ids"])
            if n_tokens > max_sequence_tokens:
                raise ValueError(f"sequence has {n_tokens} tokens > limit {max_sequence_tokens}")
            accepted.append((row, rendered))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            rejected.append(
                {
                    "pair_id": str(row.get("pair_id", "unknown")),
                    "source": str(row.get("source", "unknown")),
                    "reason": str(exc),
                }
            )
    return accepted, rejected


def _capture_chunk(model, tokenizer, prepared, cfg: DictConfig):
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations

    layers = [int(layer) for layer in cfg.model.capture_layers]
    hidden_dim = int(cfg.model.expected_hidden_dim)
    lengths = [len(rendered["full_ids"]) for _row, rendered in prepared]
    contexts: dict[int, torch.Tensor] = {}
    answers: dict[int, torch.Tensor] = {}
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("capture requires a tokenizer pad token")
    for batch in batches_by_budget(
        lengths,
        int(cfg.capture.batch_max_rows),
        int(cfg.capture.batch_max_tokens),
    ):
        max_len = max(lengths[index] for index in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for batch_index, prepared_index in enumerate(batch):
            ids = prepared[prepared_index][1]["full_ids"]
            input_ids[batch_index, : len(ids)] = torch.as_tensor(ids, device=device)
            attention_mask[batch_index, : len(ids)] = 1
        captured = extract_layer_activations(
            model,
            input_ids,
            layers,
            attention_mask=attention_mask,
        )
        for batch_index, prepared_index in enumerate(batch):
            rendered = prepared[prepared_index][1]
            context_position = int(rendered["context_position"])
            answer_start = int(rendered["answer_start"])
            answer_end = int(rendered["answer_end"])
            contexts[prepared_index] = torch.stack(
                [
                    captured[layer][batch_index, context_position].to(torch.float16).cpu()
                    for layer in layers
                ]
            )
            answers[prepared_index] = torch.stack(
                [
                    captured[layer][batch_index, answer_start:answer_end]
                    .float()
                    .mean(dim=0)
                    .to(torch.float16)
                    .cpu()
                    for layer in layers
                ]
            )
        del captured, input_ids, attention_mask
    context_array = torch.stack([contexts[index] for index in range(len(prepared))]).numpy()
    answer_array = torch.stack([answers[index] for index in range(len(prepared))]).numpy()
    expected = (len(prepared), len(layers), hidden_dim)
    if context_array.shape != expected or answer_array.shape != expected:
        raise RuntimeError(
            f"capture shape drift: context={context_array.shape}, answer={answer_array.shape}, "
            f"expected={expected}"
        )
    return context_array, answer_array


def _chunk_paths(output_dir: Path, chunk_index: int) -> tuple[Path, Path, Path, Path]:
    stem = f"chunk_{chunk_index:06d}"
    return (
        output_dir / f"{stem}.npz",
        output_dir / f"{stem}.rows.jsonl",
        output_dir / f"{stem}.rejected.jsonl",
        output_dir / f"{stem}.done.json",
    )


def run_capture(cfg: DictConfig) -> dict[str, Any]:
    """Run a bounded, resumable activation capture."""
    import torch

    data_dir = Path(str(cfg.data_dir))
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [str(source) for source in cfg.capture.sources]
    input_paths = [data_dir / f"{source}.jsonl" for source in sources]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"capture input is incomplete: {path}")
    max_rows_value = cfg.capture.max_rows
    max_rows = None if max_rows_value is None else int(max_rows_value)
    fingerprint_payload = {
        "schema_version": "context_risk_qwen38_capture_v1",
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "capture": OmegaConf.to_container(cfg.capture, resolve=True),
        "input_sha256": {path.name: _file_sha256(path) for path in input_paths},
    }
    fingerprint = _stable_digest(fingerprint_payload)
    selected_rows = []
    for row in iter_pair_rows(data_dir, sources):
        selected_rows.append(row)
        if max_rows is not None and len(selected_rows) >= max_rows:
            break
    if not selected_rows:
        raise RuntimeError("capture roster is empty")
    model, tokenizer, wrapper_depth = _load_model_and_tokenizer(cfg)
    torch.cuda.reset_peak_memory_stats()
    checkpoint_rows = int(cfg.capture.checkpoint_rows)
    accepted_total = 0
    rejected_total = 0
    started = time.monotonic()
    chunk_reports = []
    for chunk_index, start in enumerate(range(0, len(selected_rows), checkpoint_rows)):
        rows = selected_rows[start : start + checkpoint_rows]
        npz_path, rows_path, rejected_path, done_path = _chunk_paths(output_dir, chunk_index)
        if done_path.is_file():
            prior = json.loads(done_path.read_text(encoding="utf-8"))
            if prior.get("fingerprint") != fingerprint:
                raise RuntimeError(f"{done_path}: capture fingerprint changed")
            if not npz_path.is_file() or not rows_path.is_file() or not rejected_path.is_file():
                raise RuntimeError(f"{done_path}: sentinel exists but a chunk payload is missing")
            accepted_total += int(prior["accepted_rows"])
            rejected_total += int(prior["rejected_rows"])
            chunk_reports.append(prior)
            continue
        prepared, rejected = _prepare_chunk(
            tokenizer,
            rows,
            enable_thinking=bool(cfg.capture.enable_thinking),
            max_sequence_tokens=int(cfg.capture.max_sequence_tokens),
        )
        if prepared:
            context, answer = _capture_chunk(model, tokenizer, prepared, cfg)
        else:
            shape = (0, len(cfg.model.capture_layers), int(cfg.model.expected_hidden_dim))
            context = np.empty(shape, dtype=np.float16)
            answer = np.empty(shape, dtype=np.float16)
        metadata = []
        for (row, rendered), context_row, answer_row in zip(prepared, context, answer, strict=True):
            if not np.isfinite(context_row).all() or not np.isfinite(answer_row).all():
                raise RuntimeError(f"non-finite activation for pair {row['pair_id']}")
            metadata.append(
                {
                    "pair_id": row["pair_id"],
                    "source": row["source"],
                    "group_id": row["group_id"],
                    "task_id": row["task_id"],
                    "map_partition": row["map_partition"],
                    "assistant_message_index": row["assistant_message_index"],
                    "response_origin": row["response_origin"],
                    "n_prefix_tokens": rendered["context_position"] + 1,
                    "n_answer_tokens": rendered["answer_end"] - rendered["answer_start"],
                    "n_total_tokens": len(rendered["full_ids"]),
                    "prefix_ids_sha256": rendered["prefix_ids_sha256"],
                    "full_ids_sha256": rendered["full_ids_sha256"],
                }
            )
        _write_npz_atomic(npz_path, context=context, answer=answer)
        _write_jsonl_atomic(rows_path, metadata)
        _write_jsonl_atomic(rejected_path, rejected)
        chunk_report = {
            "schema_version": "context_risk_qwen38_capture_chunk_v1",
            "fingerprint": fingerprint,
            "chunk_index": chunk_index,
            "input_rows": len(rows),
            "accepted_rows": len(prepared),
            "rejected_rows": len(rejected),
            "npz_sha256": _file_sha256(npz_path),
            "rows_sha256": _file_sha256(rows_path),
            "rejected_sha256": _file_sha256(rejected_path),
        }
        _write_json_atomic(done_path, chunk_report)
        accepted_total += len(prepared)
        rejected_total += len(rejected)
        chunk_reports.append(chunk_report)
        print(
            f"[context-risk-capture] chunk={chunk_index} accepted={len(prepared)}/{len(rows)} "
            f"total={accepted_total}/{len(selected_rows)} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    retention = accepted_total / len(selected_rows)
    minimum = float(cfg.capture.min_retention_fraction)
    if retention < minimum:
        raise RuntimeError(f"capture retention {retention:.3f} < required {minimum:.3f}")
    report = {
        "schema_version": "context_risk_qwen38_capture_run_v1",
        "fingerprint": fingerprint,
        "model_config": fingerprint_payload["model"],
        "capture_config": fingerprint_payload["capture"],
        "model_id": str(cfg.model.id),
        "model_revision": str(cfg.model.revision),
        "capture_layers": [int(layer) for layer in cfg.model.capture_layers],
        "wrapper_depth": int(wrapper_depth),
        "selected_rows": len(selected_rows),
        "accepted_rows": accepted_total,
        "rejected_rows": rejected_total,
        "retention_fraction": retention,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30),
        "chunks": chunk_reports,
        "passed": True,
    }
    _write_json_atomic(output_dir / "run_result.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


@hydra.main(
    version_base="1.3",
    config_path="../configs/eval",
    config_name="context_risk_qwen38_capture",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    run_capture(cfg)


if __name__ == "__main__":
    main()
