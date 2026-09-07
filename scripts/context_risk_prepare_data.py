#!/usr/bin/env python3
"""Prepare pinned, group-safe public data manifests for the context-risk experiment."""

from __future__ import annotations

import hashlib
import ast
import json
import os
import random
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


DATASET_VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET_API_ROOT = "https://huggingface.co/api/datasets"


def _stable_digest(value: Any) -> str:
    """Return a stable JSON SHA-256 digest."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def assign_group_partitions(
    group_ids: Iterable[str], fractions: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """Assign complete groups to exact-count train/validation/public-holdout partitions."""
    unique = sorted(set(group_ids))
    if not unique:
        raise ValueError("cannot split an empty group roster")
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError(f"expected three positive split fractions, got {fractions}")
    total = sum(fractions)
    normalized = tuple(value / total for value in fractions)
    ranked = sorted(
        unique,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    )
    n_train = round(normalized[0] * len(ranked))
    n_validation = round(normalized[1] * len(ranked))
    n_train = min(max(n_train, 1), len(ranked) - 2)
    n_validation = min(max(n_validation, 1), len(ranked) - n_train - 1)
    boundaries = (n_train, n_train + n_validation)
    names = (
        ["train"] * boundaries[0]
        + ["validation"] * (boundaries[1] - boundaries[0])
        + ["public_holdout"] * (len(ranked) - boundaries[1])
    )
    return dict(zip(ranked, names, strict=True))


def _viewer_page(repo_id: str, config: str, split: str, offset: int) -> dict[str, Any]:
    """Fetch one Dataset Viewer page with bounded transient retries."""
    query = parse.urlencode(
        {
            "dataset": repo_id,
            "config": config,
            "split": split,
            "offset": offset,
            "length": 100,
        }
    )
    url = f"{DATASET_VIEWER_ROWS_URL}?{query}"
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    url_request = request.Request(url, headers=headers)
    for attempt in range(4):
        try:
            with request.urlopen(  # noqa: S310 - pinned HTTPS host
                url_request, timeout=60
            ) as response:
                return json.load(response)
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable retry tail")


def _assert_current_dataset_revision(repo_id: str, expected_revision: str) -> None:
    """Fail before Dataset Viewer reads when its unversioned source has drifted."""
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    url = f"{HF_DATASET_API_ROOT}/{repo_id}"
    url_request = request.Request(url, headers=headers)
    with request.urlopen(url_request, timeout=60) as response:  # noqa: S310 - pinned HTTPS host
        realized = str(json.load(response)["sha"])
    if realized != expected_revision:
        raise RuntimeError(
            f"{repo_id}: Dataset Viewer is unversioned and current SHA {realized} "
            f"!= pinned {expected_revision}; refuse to mislabel current rows as pinned data"
        )


def _fetch_viewer_split(repo_id: str, config: str, split: str) -> list[dict[str, Any]]:
    """Read a complete small Dataset Viewer split using its reported row count."""
    rows: list[dict[str, Any]] = []
    total = None
    while total is None or len(rows) < total:
        payload = _viewer_page(repo_id, config, split, len(rows))
        page = [entry.get("row", entry) for entry in payload.get("rows", [])]
        if not page:
            raise RuntimeError(f"Dataset Viewer returned no rows before completion for {split}")
        rows.extend(page)
        total = int(payload["num_rows_total"])
    if len(rows) != total:
        raise RuntimeError(f"row-count mismatch for {split}: got {len(rows)}, expected {total}")
    return rows


def shuffled_page_offsets(total_rows: int, page_size: int, seed: int) -> list[int]:
    """Return a deterministic permutation of complete Dataset Viewer page offsets."""
    if total_rows <= 0 or page_size <= 0:
        raise ValueError(f"invalid viewer geometry: total_rows={total_rows}, page_size={page_size}")
    offsets = list(range(0, total_rows, page_size))
    random.Random(seed).shuffle(offsets)
    return offsets


def _iter_viewer_pages(
    repo_id: str,
    config: str,
    split: str,
    *,
    page_size: int,
    seed: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield source-indexed rows from deterministic pseudo-random Viewer pages."""
    first = _viewer_page(repo_id, config, split, offset=0)
    total_rows = int(first["num_rows_total"])
    offsets = shuffled_page_offsets(total_rows, page_size, seed)
    for offset in offsets:
        payload = first if offset == 0 else _viewer_page(repo_id, config, split, offset=offset)
        page = [entry.get("row", entry) for entry in payload.get("rows", [])]
        if not page:
            raise RuntimeError(f"Dataset Viewer returned an empty page at offset {offset}")
        for page_index, row in enumerate(page):
            yield offset + page_index, row


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write JSON Lines rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def _jsonish(value: Any) -> Any:
    """Parse a JSON/Python-literal string when possible, otherwise return it."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    for parser_fn in (json.loads, ast.literal_eval):
        try:
            return parser_fn(stripped)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def _content(value: Any) -> str:
    """Coerce heterogeneous message content to a stable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _normalize_tool_call(value: Any, call_index: int) -> dict[str, Any]:
    """Normalize one source tool call to Qwen's function-call dictionary."""
    parsed = _jsonish(value)
    if not isinstance(parsed, dict):
        parsed = {"name": "unknown_tool", "arguments": {"raw": _content(parsed)}}
    function = parsed.get("function", parsed)
    if not isinstance(function, dict):
        function = {"name": "unknown_tool", "arguments": {"raw": _content(function)}}
    arguments = _jsonish(function.get("arguments", {}))
    if not isinstance(arguments, dict):
        arguments = {"raw": _content(arguments)}
    return {
        "id": str(parsed.get("id") or f"call_{call_index}"),
        "type": "function",
        "function": {
            "name": str(function.get("name") or parsed.get("name") or "unknown_tool"),
            "arguments": arguments,
        },
    }


def normalize_tools(value: Any) -> list[Any]:
    """Parse a source tool inventory, including individually serialized schemas."""
    parsed = _jsonish(value)
    if parsed in (None, ""):
        return []
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [_jsonish(tool) for tool in parsed]


def normalize_messages(source: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one supported trajectory row to canonical Qwen-style messages."""
    if source in {"apigen_mt", "agent_instruct"}:
        raw_messages = row.get("conversations", [])
    elif source == "toucan_sft":
        raw_messages = _jsonish(row.get("messages", []))
    else:
        raw_messages = row.get("messages", row.get("conversation", []))
    if not isinstance(raw_messages, list):
        raise ValueError(f"{source}: messages are not a list")
    normalized: list[dict[str, Any]] = []
    if source == "apigen_mt" and row.get("system"):
        normalized.append({"role": "system", "content": _content(row["system"])})
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "function_call": "tool_call",
        "observation": "tool",
        "tool_response": "tool",
    }
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: message is not a mapping")
        role = role_map.get(str(raw.get("role", raw.get("from", ""))), None)
        if role is None:
            role = str(raw.get("role", raw.get("from", "")))
        value = raw.get("content", raw.get("value", ""))
        if role == "tool_call":
            tool_call = _normalize_tool_call(value, len(normalized))
            if normalized and normalized[-1]["role"] == "assistant":
                normalized[-1].setdefault("tool_calls", []).append(tool_call)
            else:
                normalized.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
            continue
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{source}: unsupported role {role!r}")
        message: dict[str, Any] = {"role": role, "content": _content(value)}
        reasoning = raw.get("reasoning_content")
        if role == "assistant" and reasoning:
            message["reasoning_content"] = _content(reasoning)
        tool_calls = raw.get("tool_calls")
        if role == "assistant" and tool_calls:
            message["tool_calls"] = [
                _normalize_tool_call(call, call_index) for call_index, call in enumerate(tool_calls)
            ]
        normalized.append(message)
    return normalized


def _evenly_spaced(values: list[int], limit: int) -> list[int]:
    """Select up to ``limit`` indices across a trajectory, including both ends."""
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[-1]]
    positions = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[position] for position in positions]


def assign_map_partition(
    source: str,
    group_id: str,
    *,
    seed: int = 38292,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> str:
    """Hash one source-local trajectory group into a stable map split."""
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError(f"expected three positive map fractions, got {fractions}")
    total = sum(fractions)
    cut_train = fractions[0] / total
    cut_validation = (fractions[0] + fractions[1]) / total
    digest = hashlib.sha256(f"{seed}:{source}:{group_id}".encode()).digest()
    unit_value = int.from_bytes(digest[:8], "big") / 2**64
    if unit_value < cut_train:
        return "train"
    if unit_value < cut_validation:
        return "validation"
    return "test"


def trajectory_pairs(
    source: str,
    source_cfg: DictConfig | dict[str, Any],
    row: dict[str, Any],
    *,
    row_index: int,
    split: str,
    max_pairs: int,
    map_split_seed: int = 38292,
    map_split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[dict[str, Any]]:
    """Extract bounded teacher-forced context/assistant pairs from one trajectory."""
    messages = normalize_messages(source, row)
    group_field = str(source_cfg["group_field"])
    group_value = row_index if group_field == "row_index" else row.get(group_field)
    if group_value in (None, ""):
        raise ValueError(f"{source}: missing group field {group_field!r}")
    group_id = str(group_value)
    task_field = source_cfg.get("task_field")
    task_id = str(row.get(task_field, group_id)) if task_field else group_id
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
        and (
            message.get("content", "").strip()
            or message.get("reasoning_content", "").strip()
            or message.get("tool_calls")
        )
        and any(prior["role"] == "user" for prior in messages[:index])
    ]
    selected = _evenly_spaced(assistant_indices, max_pairs)
    repo_id = str(source_cfg["repo_id"])
    revision = str(source_cfg["revision"])
    tools = normalize_tools(row.get("tools"))
    map_partition = assign_map_partition(
        source,
        group_id,
        seed=map_split_seed,
        fractions=map_split_fractions,
    )
    pairs = []
    for assistant_index in selected:
        pair_id = hashlib.sha256(
            f"{source}|{revision}|{split}|{group_id}|{assistant_index}".encode()
        ).hexdigest()
        pairs.append(
            {
                "schema_version": "context_risk_map_pair_v1",
                "pair_id": pair_id,
                "source": source,
                "dataset": repo_id,
                "dataset_revision": revision,
                "source_split": split,
                "group_id": group_id,
                "task_id": task_id,
                "map_partition": map_partition,
                "assistant_message_index": assistant_index,
                "context_messages": messages[:assistant_index],
                "answer_message": messages[assistant_index],
                "tools": tools,
                "response_origin": "teacher_forced",
            }
        )
    return pairs


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Durably append one checkpoint batch of JSON Lines rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        for row in rows:
            payload = (
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            ).encode()
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(f"short JSONL append: wrote {written}/{len(payload)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_pair_ids(path: Path) -> set[str]:
    """Load stable pair IDs from an existing resumable JSONL bank."""
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["pair_id"]))
    return ids


def _max_source_quotas(cfg: DictConfig) -> dict[str, int]:
    """Return the largest requested quota for each source across map arms."""
    quotas = {str(source): 0 for source in cfg.generic_source_order}
    for arm in cfg.map_arms.values():
        for source, count in arm.target_pairs.items():
            quotas[str(source)] = max(quotas.get(str(source), 0), int(count))
    return quotas


def _stream_source_bank(
    cfg: DictConfig, source: str, quota: int, out_dir: Path, *, smoke: bool
) -> dict[str, Any]:
    """Stream one pinned dataset into a checkpointed, resumable normalized pair bank."""
    from datasets import load_dataset

    source_cfg = cfg.sources[source]
    output_path = out_dir / f"{source}.jsonl"
    meta_path = out_dir / f"{source}.meta.json"
    fingerprint = _stable_digest(
        {
            "source": OmegaConf.to_container(source_cfg, resolve=True),
            "quota": quota,
            "max_pairs": int(cfg.max_assistant_pairs_per_trajectory),
            "map_split_seed": int(cfg.map_split_seed),
            "map_split_fractions": [float(value) for value in cfg.map_split_fractions],
            "shuffle_seed": int(cfg.sampling_seed),
            "shuffle_buffer": int(cfg.generic_shuffle_buffer),
        }
    )
    if meta_path.exists():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise RuntimeError(f"{source}: resume fingerprint changed; use a new output directory")
    existing = _existing_pair_ids(output_path)
    kept = len(existing)
    invalid_rows = 0
    scanned_rows = 0
    pending: list[dict[str, Any]] = []
    started = time.monotonic()
    if kept >= quota:
        report = {
            "schema_version": "context_risk_source_bank_v1",
            "fingerprint": fingerprint,
            "source": source,
            "quota": quota,
            "kept_pairs": kept,
            "scanned_rows_this_run": 0,
            "invalid_rows_this_run": 0,
            "complete": True,
        }
        _write_json_atomic(meta_path, report)
        print(f"[generic:{source}] resume complete pairs={kept}/{quota}", flush=True)
        return report
    use_viewer_pages = not smoke and str(source_cfg.get("pilot_backend", "")) == "viewer_pages"
    for split_index, split in enumerate(source_cfg.splits):
        if smoke:
            _assert_current_dataset_revision(str(source_cfg.repo_id), str(source_cfg.revision))
            payload = _viewer_page(
                str(source_cfg.repo_id), str(source_cfg.config), str(split), offset=0
            )
            row_iterator = enumerate([entry.get("row", entry) for entry in payload.get("rows", [])])
        elif use_viewer_pages:
            _assert_current_dataset_revision(str(source_cfg.repo_id), str(source_cfg.revision))
            row_iterator = _iter_viewer_pages(
                str(source_cfg.repo_id),
                str(source_cfg.config),
                str(split),
                page_size=int(source_cfg.get("viewer_page_size", 100)),
                seed=int(cfg.sampling_seed) + split_index,
            )
        else:
            dataset = load_dataset(
                str(source_cfg.repo_id),
                str(source_cfg.config),
                split=str(split),
                revision=str(source_cfg.revision),
                streaming=True,
            ).shuffle(
                seed=int(cfg.sampling_seed) + split_index,
                buffer_size=int(cfg.generic_shuffle_buffer),
            )
            row_iterator = enumerate(dataset)
        for row_index, row in row_iterator:
            scanned_rows += 1
            try:
                pairs = trajectory_pairs(
                    source,
                    source_cfg,
                    row,
                    row_index=row_index,
                    split=str(split),
                    max_pairs=int(cfg.max_assistant_pairs_per_trajectory),
                    map_split_seed=int(cfg.map_split_seed),
                    map_split_fractions=tuple(float(value) for value in cfg.map_split_fractions),
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                invalid_rows += 1
                continue
            for pair in pairs:
                if pair["pair_id"] in existing:
                    continue
                pending.append(pair)
                existing.add(pair["pair_id"])
                kept += 1
                if len(pending) >= int(cfg.generic_checkpoint_pairs) or kept >= quota:
                    _append_jsonl(output_path, pending)
                    pending.clear()
                    report = {
                        "schema_version": "context_risk_source_bank_v1",
                        "fingerprint": fingerprint,
                        "source": source,
                        "quota": quota,
                        "kept_pairs": kept,
                        "scanned_rows_this_run": scanned_rows,
                        "invalid_rows_this_run": invalid_rows,
                        "complete": kept >= quota,
                    }
                    _write_json_atomic(meta_path, report)
                    print(
                        f"[generic:{source}] pairs={kept}/{quota} scanned={scanned_rows} "
                        f"invalid={invalid_rows} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                if kept >= quota:
                    if use_viewer_pages:
                        _assert_current_dataset_revision(
                            str(source_cfg.repo_id), str(source_cfg.revision)
                        )
                    return report
        if kept >= quota:
            break
    if pending:
        _append_jsonl(output_path, pending)
    report = {
        "schema_version": "context_risk_source_bank_v1",
        "fingerprint": fingerprint,
        "source": source,
        "quota": quota,
        "kept_pairs": kept,
        "scanned_rows_this_run": scanned_rows,
        "invalid_rows_this_run": invalid_rows,
        "complete": kept >= quota,
    }
    _write_json_atomic(meta_path, report)
    if kept < quota:
        raise RuntimeError(f"{source}: exhausted all splits with {kept}/{quota} usable pairs")
    return report


def prepare_generic(cfg: DictConfig, *, smoke: bool) -> dict[str, Any]:
    """Build resumable generic source banks for a smoke or the full map mixtures."""
    quotas = _max_source_quotas(cfg)
    if smoke:
        quotas = {
            source: min(quota, int(cfg.generic_smoke_pairs_per_source))
            for source, quota in quotas.items()
        }
    out_dir = Path(str(cfg.output_dir)) / ("generic_smoke" if smoke else "generic_full")
    reports = {}
    for source in cfg.generic_source_order:
        source = str(source)
        if quotas.get(source, 0) > 0:
            reports[source] = _stream_source_bank(cfg, source, quotas[source], out_dir, smoke=smoke)
    report = {
        "schema_version": "context_risk_generic_banks_v1",
        "mode": "smoke" if smoke else "full",
        "quotas": quotas,
        "sources": reports,
        "passed": all(item["complete"] for item in reports.values()),
    }
    _write_json_atomic(out_dir / "build_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def prepare_generic_pilot(cfg: DictConfig) -> dict[str, Any]:
    """Build the 20k M200-scaled source banks for the mapping feasibility gate."""
    arm_name = str(cfg.generic_pilot_arm)
    scale = float(cfg.generic_pilot_scale)
    if arm_name not in cfg.map_arms or not (0 < scale <= 1):
        raise ValueError(f"invalid pilot arm/scale: {arm_name!r}, {scale}")
    quotas = {
        str(source): round(int(count) * scale)
        for source, count in cfg.map_arms[arm_name].target_pairs.items()
    }
    out_dir = Path(str(cfg.output_dir)) / "generic_pilot"
    reports = {
        source: _stream_source_bank(cfg, source, quota, out_dir, smoke=False)
        for source, quota in quotas.items()
        if quota > 0
    }
    report = {
        "schema_version": "context_risk_generic_banks_v1",
        "mode": "pilot",
        "arm": arm_name,
        "scale": scale,
        "quotas": quotas,
        "target_pairs": sum(quotas.values()),
        "sources": reports,
        "passed": all(item["complete"] for item in reports.values()),
    }
    _write_json_atomic(out_dir / "build_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def prepare_impossible(cfg: DictConfig) -> dict[str, Any]:
    """Build the pinned public Impossible-LiveCodeBench development manifest."""
    benchmark = cfg.benchmarks.impossible_livecodebench
    repo_id = str(benchmark.repo_id)
    revision = str(benchmark.revision)
    config = str(benchmark.config)
    conditions = [str(condition) for condition in benchmark.conditions]
    _assert_current_dataset_revision(repo_id, revision)
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        source_rows[condition] = _fetch_viewer_split(repo_id, config, condition)
        print(f"[impossible] fetched {condition}: {len(source_rows[condition])} rows", flush=True)
    group_sets = [
        {str(row[benchmark.group_field]) for row in rows} for rows in source_rows.values()
    ]
    if any(group_set != group_sets[0] for group_set in group_sets[1:]):
        raise RuntimeError("ImpossibleBench conditions do not share an identical task-id roster")
    fractions = tuple(float(value) for value in benchmark.split_fractions)
    assignments = assign_group_partitions(group_sets[0], fractions, int(cfg.sampling_seed))
    manifest_rows = []
    for condition in conditions:
        for row in source_rows[condition]:
            task_id = str(row[benchmark.group_field])
            manifest_rows.append(
                {
                    "dataset": repo_id,
                    "dataset_revision": revision,
                    "config": config,
                    "condition": condition,
                    "partition": assignments[task_id],
                    **row,
                }
            )
    manifest_rows.sort(key=lambda row: (row["partition"], row["task_id"], row["condition"]))
    partition_groups = {
        partition: sorted(group for group, value in assignments.items() if value == partition)
        for partition in ("train", "validation", "public_holdout")
    }
    out_dir = Path(str(cfg.output_dir)) / "impossible_livecodebench"
    jsonl_path = out_dir / "public_development_manifest.jsonl"
    _write_jsonl_atomic(jsonl_path, manifest_rows)
    report = {
        "schema_version": "context_risk_impossible_public_manifest_v1",
        "dataset": repo_id,
        "dataset_revision": revision,
        "config": config,
        "conditions": conditions,
        "group_field": str(benchmark.group_field),
        "sampling_seed": int(cfg.sampling_seed),
        "requested_split_fractions": list(fractions),
        "public_test_role": str(benchmark.public_test_role),
        "n_groups": len(assignments),
        "n_rows": len(manifest_rows),
        "groups_per_partition": {
            partition: len(groups) for partition, groups in partition_groups.items()
        },
        "rows_per_partition": {
            partition: sum(row["partition"] == partition for row in manifest_rows)
            for partition in partition_groups
        },
        "partition_group_ids": partition_groups,
        "assignment_sha256": _stable_digest(assignments),
        "manifest_sha256": _stable_digest(manifest_rows),
        "manifest_path": str(jsonl_path),
        "private_test_included": False,
        "passed": True,
    }
    _write_json_atomic(out_dir / "split_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="context_risk_prepare")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for pinned context-risk data preparation."""
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    if cfg.mode == "impossible":
        prepare_impossible(cfg)
        return
    if cfg.mode == "generic_smoke":
        prepare_generic(cfg, smoke=True)
        return
    if cfg.mode == "generic_full":
        prepare_generic(cfg, smoke=False)
        return
    if cfg.mode == "generic_pilot":
        prepare_generic_pilot(cfg)
        return
    raise ValueError(f"unknown mode {cfg.mode!r}")


if __name__ == "__main__":
    main()
    # Some pyarrow/Hugging Face streaming builds leave a native callback thread
    # alive and can abort in PyGILState_Release during interpreter finalization,
    # after every atomic artifact has already been written. Bypass extension
    # finalizers only on the clean CLI-return path; exceptions still propagate.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
