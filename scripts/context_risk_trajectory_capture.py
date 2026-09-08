"""Capture reviewed trajectory checkpoints with exact-prefix causal smoke gates."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from scripts.context_risk_followup_capture import MODEL  # noqa: E402
from scripts.context_risk_qwen38_capture import batches_by_budget  # noqa: E402
from scripts.context_risk_qwen38_smoke import _load_model_and_tokenizer  # noqa: E402

REPOSITORY = Path(__file__).resolve().parent.parent
SCHEMA = "context_risk_trajectory_capture_v1"
MAP_SHA256 = "680935a244cc39c29797d66b312b95e3741889a541dcee8bde7c69ac33c5242d"
SOURCES = (
    "scripts/context_risk_trajectory_capture.py",
    "tests/test_context_risk_trajectory_capture.py",
    "configs/eval/context_risk_trajectory_capture.yaml",
    "scripts/context_risk_followup_capture.py",
    "scripts/context_risk_qwen38_capture.py",
    "scripts/context_risk_qwen38_smoke.py",
    "scripts/context_risk_prepare_data.py",
    "src/explore_persona_space/analysis/extraction.py",
    "src/explore_persona_space/orchestrate/env.py",
    "eval_results/context_risk_trajectory_design/plan.md",
    "eval_results/context_risk_trajectory_design/analysis_spec.json",
    "eval_results/context_risk_trajectory_design/implementation_contract.md",
)
RUNTIME = {
    "transformers": "5.15.0",
    "torch": "2.13.0",
    "tokenizers": "0.22.2",
    "accelerate": "1.13.0",
    "numpy": "2.3.5",
}


def sha256(path: Path) -> str:
    """Hash a file without reading large arrays into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    """Match the producer's canonical JSON token-prefix digest."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _producer():
    return importlib.import_module("scripts.context_risk_trajectory_prepare")


def source_hashes() -> dict:
    """Bind this instrument and its prepared-input and inherited source closure."""
    result = dict(_producer().source_hashes())
    for name in SOURCES:
        actual = sha256(REPOSITORY / name)
        if name in result and result[name] != actual:
            raise ValueError(f"Source closure conflict: {name}")
        result[name] = actual
    return result


def validate_imported_sources(sources: dict) -> dict:
    """Reject an editable install importing different copies of reused helper bodies."""
    names = {
        "scripts.context_risk_trajectory_prepare": "scripts/context_risk_trajectory_prepare.py",
        "scripts.context_risk_followup_capture": "scripts/context_risk_followup_capture.py",
        "scripts.context_risk_qwen38_capture": "scripts/context_risk_qwen38_capture.py",
        "scripts.context_risk_qwen38_smoke": "scripts/context_risk_qwen38_smoke.py",
        "explore_persona_space.analysis.extraction": "src/explore_persona_space/analysis/extraction.py",
        "explore_persona_space.orchestrate.env": "src/explore_persona_space/orchestrate/env.py",
    }
    actual = {}
    for module_name, relative in names.items():
        path = Path(importlib.import_module(module_name).__file__).resolve()
        if sha256(path) != sources[relative]:
            raise ValueError(f"Imported helper differs from reviewed source: {module_name}")
        actual[relative] = {"actual_path": str(path), "sha256": sources[relative]}
    return actual


def _read(path: Path):
    return json.loads(path.read_text())


def snapshot_paths(paths) -> dict:
    """Snapshot consumed bytes before reading them, with a matching end guard."""
    return {str(Path(path).absolute()): sha256(Path(path)) for path in paths}


def assert_snapshot(snapshot: dict) -> None:
    if snapshot_paths(snapshot) != snapshot:
        raise ValueError("Consumed input or control bytes changed during the operation")


def prepared_snapshot(prepared_root: Path, extra_paths=()) -> dict:
    paths = [_safe(prepared_root, name) for name in ("manifest.json", *_producer().DATA_FILES)]
    return snapshot_paths([*paths, *extra_paths])


def guard_output(root: Path, prepared_root: Path, *, adopt=False) -> None:
    resolved, prepared = root.resolve(), prepared_root.resolve()
    if (
        resolved == prepared
        or resolved.is_relative_to(prepared)
        or prepared.is_relative_to(resolved)
    ):
        raise ValueError("Capture output must be disjoint from prepared inputs")
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError("Capture output cannot contain a symlink")
    if (
        adopt
        and root.exists()
        and any(root.iterdir())
        and not _safe(root, "capture_binding.json").is_file()
    ):
        raise ValueError("Refusing to adopt an unbound populated capture directory")


def _atomic(path: Path, writer) -> None:
    """Write one payload atomically; sentinels are written after their payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_json(path: Path, value) -> None:
    _atomic(
        path,
        lambda stream: stream.write(
            (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )


def _safe(root: Path, name: str) -> Path:
    value = Path(name)
    if name in {"", "."} or value.is_absolute() or ".." in value.parts or value.as_posix() != name:
        raise ValueError(f"Unsafe relative artifact path: {name}")
    path = root / value
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError(f"Symlink in capture artifact path: {name}")
    return path


def validate_review(path: Path, sources: dict) -> dict:
    """Require a current independent, scoped, source-bound approval."""
    review = _read(path)
    if (
        review.get("verdict") != "PASS"
        or review.get("unresolved_findings") != []
        or review.get("sources_sha256") != sources
    ):
        raise ValueError("Missing, stale or incomplete independent capture review")
    return review


def recipe(cfg: DictConfig) -> dict:
    """Freeze the executable model, capture, and numerical smoke choices."""
    value = {
        name: OmegaConf.to_container(cfg[name], resolve=True)
        for name in ("model", "capture", "smoke")
    }
    if value["model"] != MODEL:
        raise ValueError("Trajectory model/layer/precision differs from the frozen map")
    capture = value["capture"]
    if (
        capture
        != {
            "use_cache": False,
            "output_hidden_states": False,
            "logits_to_keep": 1,
            "attention_backend": "sdpa",
            "storage_dtype": "float16",
            "max_sequence_tokens": 98622,
            "batch_max_rows": 2,
            "batch_max_tokens": 16384,
        }
        or capture["use_cache"] is not False
        or capture["output_hidden_states"] is not False
    ):
        raise ValueError("Capture recipe differs from the reviewed full-prefix instrument")
    expected = {
        "raw_relative_l2_max": 0.01,
        "raw_cosine_min": 0.9999,
        "mapped_relative_l2_max": 0.01,
        "mapped_cosine_min": 0.9999,
        "same_forward_exact": True,
        "suffix_mutation_exact": True,
        "future_mask_exact": True,
        "representative_tokens": 25000,
        "minimum_gpu_gib": 139,
    }
    if value["smoke"] != expected:
        raise ValueError("Numerical smoke thresholds must be frozen before model calls")
    return value


def group_checkpoints(streams: dict, checkpoints: list[dict]) -> dict:
    """Check every unique prefix and gather deterministic per-stream assignments."""
    grouped = {key: [] for key in streams}
    seen = set()
    for point in checkpoints:
        key, stream_key = point["checkpoint_key"], point["stream_key"]
        ids = streams[stream_key]
        length = point["length"]
        if type(length) is not int or type(point["position"]) is not int:
            raise ValueError("Checkpoint positions must be integers")
        if key in seen or not 1 <= length <= len(ids) or point["position"] != length - 1:
            raise ValueError("Duplicate checkpoint or invalid causal position")
        if digest(ids[:length].tolist()) != key:
            raise ValueError("Checkpoint is not the exact token prefix of its assigned stream")
        grouped[stream_key].append(point)
        seen.add(key)
    if not seen or any(not value for value in grouped.values()):
        raise ValueError("Prepared streams/checkpoints are empty or unassigned")
    return {key: sorted(value, key=lambda p: p["checkpoint_key"]) for key, value in grouped.items()}


def capture_batch(
    model,
    ids_rows: list[np.ndarray],
    positions: list[list[int]],
    *,
    layer=44,
    hidden_dim=5120,
    same_forward=False,
    valid_lengths=None,
) -> tuple[list[np.ndarray], dict]:
    """Run the production gather: only selected GPU vectors are copied to CPU."""
    import torch

    from explore_persona_space.analysis.extraction import (
        _logits_to_keep_kwargs,
        _resolve_decoder_blocks,
        _unwrap,
    )

    if model.training or not ids_rows or len(ids_rows) != len(positions):
        raise ValueError("Capture requires an eval model and aligned nonempty batch")
    # Production uses every unpadded ID. Only the causal smoke shortens a future
    # mask while preserving the original IDs and total tensor shape.
    if valid_lengths is None:
        valid_lengths = list(map(len, ids_rows))
    if len(valid_lengths) != len(ids_rows) or any(
        type(n) is not int or not 1 <= n <= len(ids)
        for n, ids in zip(valid_lengths, ids_rows, strict=True)
    ):
        raise ValueError("Invalid unpadded mask length")
    blocks, _embed, _depth = _resolve_decoder_blocks(model)
    if blocks is None or not 0 <= layer < len(blocks) - 1:
        raise ValueError("Capture requires the resolved non-final decoder block")
    if _logits_to_keep_kwargs(model, False) != {"logits_to_keep": 1}:
        raise ValueError("Model lacks the explicit bounded-logits forward interface")
    device = next(model.parameters()).device
    text_config = getattr(model.config, "text_config", model.config)
    pad = text_config.pad_token_id
    pad = int(text_config.eos_token_id if pad is None else pad)
    width = max(map(len, ids_rows))
    inputs = torch.full((len(ids_rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    row_indices, token_indices = [], []
    for row, (ids, offsets) in enumerate(zip(ids_rows, positions, strict=True)):
        if not offsets or any(
            type(v) is not int or not 0 <= v < valid_lengths[row] for v in offsets
        ):
            raise ValueError("A gather offset lies outside its unpadded input")
        inputs[row, : len(ids)] = torch.as_tensor(ids.astype(np.int64), device=device)
        mask[row, : valid_lengths[row]] = 1
        row_indices.extend([row] * len(offsets))
        token_indices.extend(offsets)
    rows_t = torch.tensor(row_indices, device=device)
    tokens_t = torch.tensor(token_indices, device=device)
    captured, reference = [], []

    def hook(_module, _args, output):
        hs = _unwrap(output)
        if tuple(hs.shape) != (len(ids_rows), width, hidden_dim):
            raise ValueError("Decoder hook geometry drifted")
        selected = hs[rows_t, tokens_t, :]
        captured.append(selected.detach().float().cpu())
        if same_forward:
            # Retain GPU storage only for this small smoke, never a full CPU tensor.
            reference.append(hs.detach())

    handle = blocks[layer].register_forward_hook(hook)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=inputs,
                attention_mask=mask,
                output_hidden_states=False,
                use_cache=False,
                logits_to_keep=1,
            )
        if getattr(output, "past_key_values", None) is not None:
            raise ValueError("Forward unexpectedly retained a cache")
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.isfinite(captured[0]).all():
        raise ValueError("Capture hook did not produce exactly one finite selected tensor")
    exact = None
    if same_forward:
        independent = (
            torch.stack(
                [reference[0][r, t, :] for r, t in zip(row_indices, token_indices, strict=True)]
            )
            .float()
            .cpu()
        )
        exact = torch.equal(captured[0], independent)
        if not exact:
            raise ValueError("Same-forward selected gather differs from the block output")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values = captured[0].numpy()
    result, offset = [], 0
    for positions_row in positions:
        result.append(values[offset : offset + len(positions_row)].copy())
        offset += len(positions_row)
    stats = {
        "rows": len(ids_rows),
        "lengths": list(map(len, ids_rows)),
        "valid_lengths": valid_lengths,
        "padded_tokens": len(ids_rows) * width,
        "selected_vectors": len(values),
        "elapsed_seconds": time.monotonic() - started,
        "same_forward_exact": exact,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "device": str(device),
    }
    return result, stats


def vector_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Report FP64 comparison metrics without coercing nonfinite or missing vectors."""
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if (
        a.ndim != 1
        or not a.size
        or a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Invalid vectors in numerical smoke")
    denominator = float(np.linalg.norm(b))
    a_norm = float(np.linalg.norm(a))
    cosine = (
        float(a @ b / (a_norm * denominator))
        if a_norm * denominator
        else float(np.array_equal(a, b))
    )
    return {
        "relative_l2": float(np.linalg.norm(a - b) / max(denominator, 1e-30)),
        "cosine": cosine,
        "max_abs_relative": float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)),
        "exact": bool(np.array_equal(a, b)),
    }


def _mapped(values: np.ndarray, arrays: dict) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) - arrays["x_mean"]) / arrays["x_scale"]) @ arrays[
        "weight"
    ] + arrays["y_mean"]


def compare_vectors(actual, expected, arrays: dict, smoke: dict) -> dict:
    """Apply both predeclared raw and mapped BF16 parity criteria."""
    result = {
        "raw": vector_metrics(actual, expected),
        "mapped": vector_metrics(_mapped(actual, arrays), _mapped(expected, arrays)),
    }
    result["passed"] = all(
        result[name]["relative_l2"] <= smoke[f"{name}_relative_l2_max"]
        and result[name]["cosine"] >= smoke[f"{name}_cosine_min"]
        for name in ("raw", "mapped")
    )
    return result


def _kind(manifest: dict, key: str) -> str:
    return manifest["streams"][key]["stream_kind"]


def smoke_fixtures(
    manifest: dict,
    streams: dict,
    checkpoints: list[dict],
    observations: list[dict],
    capture: dict,
    smoke: dict,
) -> dict:
    """Choose fixtures solely from input hashes, structural tags and lengths."""
    by_key = {p["checkpoint_key"]: p for p in checkpoints}
    stage_keys = defaultdict(set)
    for row in observations:
        stage_keys[row["stage"]].add(row["checkpoint_key"])
    chosen = {}
    for stage in (
        "pre_action_01",
        "pre_action_02",
        "within_pre",
        "within_32",
        "within_128",
        "within_512",
        "within_end",
    ):
        if not stage_keys[stage]:
            raise ValueError(f"Smoke stage has no actual checkpoint: {stage}")
        chosen[stage] = min(stage_keys[stage])
    exceptions = [
        p for p in checkpoints if _kind(manifest, p["stream_key"]) == "completion_exception"
    ]
    initials = [p for p in checkpoints if _kind(manifest, p["stream_key"]) == "canonical_initial"]
    if not exceptions or not initials:
        raise ValueError("Smoke requires actual exceptional and canonical-initial inputs")
    chosen["exception"] = min(p["checkpoint_key"] for p in exceptions)
    chosen["canonical_initial"] = min(p["checkpoint_key"] for p in initials)
    longest = min(streams, key=lambda k: (-len(streams[k]), k))
    long_points = [
        p for p in checkpoints if p["stream_key"] == longest and p["length"] < len(streams[longest])
    ]
    if not long_points:
        raise ValueError("Longest stream has no proper-prefix checkpoint")
    chosen["long_checkpoint"] = min(
        long_points, key=lambda p: (-p["position"], p["checkpoint_key"])
    )["checkpoint_key"]
    representative = min(
        streams, key=lambda k: (abs(len(streams[k]) - smoke["representative_tokens"]), k)
    )
    batches = batches_by_budget(
        [len(streams[k]) for k in sorted(streams)],
        capture["batch_max_rows"],
        capture["batch_max_tokens"],
    )
    short_batches = [batch for batch in batches if len(batch) == 2]
    if not short_batches:
        raise ValueError("Prepared roster has no feasible production short batch")
    keys = sorted(streams)
    short = min(
        short_batches,
        key=lambda b: (max(len(streams[keys[i]]) for i in b), tuple(keys[i] for i in b)),
    )
    mutation = min(
        (p for p in checkpoints if p["length"] < len(streams[p["stream_key"]])),
        key=lambda p: (len(streams[p["stream_key"]]), p["length"], p["checkpoint_key"]),
    )
    chosen["suffix_mutation"] = mutation["checkpoint_key"]
    short_ended = [
        o["checkpoint_key"]
        for o in observations
        if o["stage"] in {"within_32", "within_128", "within_512"} and o["answer_ended"]
    ]
    if short_ended:
        chosen["short_answer_ended"] = min(short_ended)
    return {
        "checkpoint_roles": chosen,
        "checkpoints": {key: by_key[key] for key in sorted(set(chosen.values()))},
        "longest_stream": longest,
        "representative_stream": representative,
        "short_batch": [keys[i] for i in short],
        "selection": "Fixed input-key ordering, stage tags and sequence lengths; no labels or future-outcome fields consulted.",
    }


def _forward_streams(model, keys, streams, grouped, **kwargs):
    return capture_batch(
        model,
        [streams[key] for key in keys],
        [[p["position"] for p in grouped[key]] for key in keys],
        **kwargs,
    )


def smoke_model(
    model,
    manifest,
    streams,
    checkpoints,
    observations,
    executable,
    arrays,
    *,
    layer=44,
    hidden_dim=5120,
) -> dict:
    """Exercise real production gathers, independent truncations and fixed-shape causality."""
    capture, smoke = executable["capture"], executable["smoke"]
    fixtures = smoke_fixtures(manifest, streams, checkpoints, observations, capture, smoke)
    points = fixtures["checkpoints"]
    selected = defaultdict(list)
    for point in points.values():
        selected[point["stream_key"]].append(point)
    gathered, measurements = {}, []
    for key in sorted(selected):
        values, stats = _forward_streams(
            model, [key], streams, selected, layer=layer, hidden_dim=hidden_dim
        )
        measurements.append({"purpose": "selected_stream", "stream_keys": [key], **stats})
        for point, vector in zip(selected[key], values[0], strict=True):
            gathered[point["checkpoint_key"]] = vector
    comparisons = {}
    for key, point in sorted(points.items()):
        prefix = streams[point["stream_key"]][: point["length"]]
        values, stats = capture_batch(
            model, [prefix], [[len(prefix) - 1]], layer=layer, hidden_dim=hidden_dim
        )
        comparisons[key] = compare_vectors(gathered[key], values[0][0], arrays, smoke)
        measurements.append(
            {"purpose": "independent_truncated_prefix", "checkpoint_key": key, **stats}
        )
    point = points[fixtures["checkpoint_roles"]["suffix_mutation"]]
    original = streams[point["stream_key"]]
    mutated = original.copy()
    mutated[point["length"] :] = np.where(mutated[point["length"] :] == 0, 1, 0)
    reference, stats = capture_batch(
        model,
        [original],
        [[point["position"]]],
        layer=layer,
        hidden_dim=hidden_dim,
        same_forward=True,
    )
    changed, changed_stats = capture_batch(
        model, [mutated], [[point["position"]]], layer=layer, hidden_dim=hidden_dim
    )
    mutation = vector_metrics(changed[0][0], reference[0][0])
    mutation["mapped"] = vector_metrics(
        _mapped(changed[0][0], arrays), _mapped(reference[0][0], arrays)
    )
    measurements.extend(
        [
            {"purpose": "same_forward_and_original_suffix", **stats},
            {"purpose": "fixed_shape_mutated_suffix", **changed_stats},
        ]
    )
    masked, masked_stats = capture_batch(
        model,
        [original],
        [[point["position"]]],
        layer=layer,
        hidden_dim=hidden_dim,
        valid_lengths=[point["length"]],
    )
    future_mask = {
        **vector_metrics(masked[0][0], reference[0][0]),
        "mapped": vector_metrics(_mapped(masked[0][0], arrays), _mapped(reference[0][0], arrays)),
        "checkpoint_key": point["checkpoint_key"],
        "total_shape": [1, len(original)],
        "original_valid_tokens": len(original),
        "mutated_valid_tokens": point["length"],
        "first_changed_mask_position": point["length"],
        "input_ids_changed": False,
        "input_ids_sha256": digest(original.tolist()),
        "original_mask_sha256": digest([1] * len(original)),
        "mutated_mask_sha256": digest(
            [1] * point["length"] + [0] * (len(original) - point["length"])
        ),
    }
    measurements.append({"purpose": "fixed_shape_future_mask", **masked_stats})
    # These actual production shapes are required even when a selected-point forward
    # above happened to cover one of the same inputs.
    grouped = group_checkpoints(streams, checkpoints)
    for purpose, keys in (
        ("longest_singleton", [fixtures["longest_stream"]]),
        ("representative_singleton", [fixtures["representative_stream"]]),
        ("production_short_batch", fixtures["short_batch"]),
    ):
        values, shape_stats = _forward_streams(
            model, keys, streams, grouped, layer=layer, hidden_dim=hidden_dim
        )
        measurements.append({"purpose": purpose, "stream_keys": keys, **shape_stats})
        if purpose == "production_short_batch":
            for stream_key, batch_values in zip(keys, values, strict=True):
                single, single_stats = _forward_streams(
                    model, [stream_key], streams, grouped, layer=layer, hidden_dim=hidden_dim
                )
                measurements.append(
                    {
                        "purpose": "short_batch_independent_singleton",
                        "stream_keys": [stream_key],
                        **single_stats,
                    }
                )
                for checkpoint, packed, solo in zip(
                    grouped[stream_key], batch_values, single[0], strict=True
                ):
                    comparisons["batch:" + checkpoint["checkpoint_key"]] = compare_vectors(
                        packed, solo, arrays, smoke
                    )
    return {
        "verification_passed": bool(
            stats["same_forward_exact"]
            and mutation["exact"]
            and mutation["mapped"]["exact"]
            and future_mask["exact"]
            and future_mask["mapped"]["exact"]
            and all(c["passed"] for c in comparisons.values())
        ),
        "fixtures": fixtures,
        "comparisons": comparisons,
        "expected_comparison_keys": sorted(
            set(points)
            | {
                "batch:" + p["checkpoint_key"]
                for key in fixtures["short_batch"]
                for p in grouped[key]
            }
        ),
        "suffix_mutation": mutation,
        "future_mask_mutation": future_mask,
        "measurements": measurements,
        "meaning": "Exact same-forward gathering and fixed-shape future invariance; separately truncated and packed/singleton BF16 parity gates on raw and frozen-mapped vectors. No outcome labels used for fixture selection.",
    }


def load_map(path: Path, hidden_dim=5120) -> dict:
    if sha256(path) != MAP_SHA256:
        raise ValueError("Frozen map bytes differ")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {
            key: np.asarray(saved[key], dtype=np.float64)
            for key in ("weight", "x_mean", "x_scale", "y_mean")
        }
    for key, value in arrays.items():
        expected = (hidden_dim, hidden_dim) if key == "weight" else (hidden_dim,)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError("Frozen map geometry or values differ")
    if np.any(arrays["x_scale"] <= 0):
        raise ValueError("Frozen map has invalid scaling")
    return arrays


def _runtime() -> dict:
    import torch

    versions = {name: importlib.metadata.version(name) for name in RUNTIME}
    if {key: value.split("+")[0] for key, value in versions.items()} != RUNTIME:
        raise ValueError(f"Production runtime differs: {versions}")
    if not torch.cuda.is_available() or torch.version.cuda != "13.0":
        raise ValueError("Production requires the pinned CUDA 13.0 runtime")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 139 * 1024**3:
        raise ValueError("Production GPU is smaller than the approved smoke venue")
    return {
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
    }


def _loaded_runtime(cfg, prepared_root: Path):
    runtime = _runtime()
    model, tokenizer, depth = _load_model_and_tokenizer(cfg)
    model.set_attn_implementation("sdpa")
    text_config = getattr(model.config, "text_config", model.config)
    if text_config._attn_implementation != "sdpa":
        raise ValueError("Actual text attention backend is not SDPA")
    runtime["attention_backend"] = text_config._attn_implementation
    runtime["decoder_resolver_depth"] = depth
    runtime["model_class"] = type(model).__name__
    runtime["attention_module_classes"] = sorted(
        {
            type(m).__name__
            for name, m in model.named_modules()
            if "attn" in name or "attention" in name
        }
    )
    replay = _producer().validate_tokenizer_replay(prepared_root, tokenizer)
    if replay.get("passed") is not True or replay.get("n_requests") != 2153:
        raise ValueError("Actual production tokenizer failed full 2,153-request replay")
    return model, runtime, replay


def make_binding(prepared_sha: str, executable: dict, sources: dict) -> dict:
    fields = {
        "schema": SCHEMA,
        "prepared_manifest_sha256": prepared_sha,
        "recipe": executable,
        "sources_sha256": sources,
        "map_sha256": MAP_SHA256,
        "runtime_pins": RUNTIME,
    }
    return {**fields, "fingerprint": digest(fields)}


def _validate_binding(binding: dict, prepared_sha: str, sources: dict) -> None:
    fields = {key: value for key, value in binding.items() if key != "fingerprint"}
    if binding.get("fingerprint") != digest(fields):
        raise ValueError("Capture binding fingerprint differs")
    if fields != {
        key: value
        for key, value in make_binding(prepared_sha, binding["recipe"], sources).items()
        if key != "fingerprint"
    }:
        raise ValueError("Capture binding is stale")
    recipe(OmegaConf.create(binding["recipe"]))


def _stream_names(key: str) -> tuple[str, str]:
    prefix, separator, suffix = key.partition("_")
    if (
        prefix not in {"trajectory", "initial", "exception"}
        or separator != "_"
        or len(suffix) != 64
        or any(c not in "0123456789abcdef" for c in suffix)
    ):
        raise ValueError("Unsafe or invalid stream key")
    return f"streams/{key}.npz", f"streams/{key}.done.json"


def _chunk_fields(key: str, ids: np.ndarray, points: list[dict], binding: dict) -> dict:
    return {
        "stream_key": key,
        "token_ids_sha256": digest(ids.tolist()),
        "checkpoint_keys": [p["checkpoint_key"] for p in points],
        "positions": [p["position"] for p in points],
        "capture_fingerprint": binding["fingerprint"],
    }


def write_chunk(
    root: Path,
    key: str,
    ids: np.ndarray,
    points: list[dict],
    binding: dict,
    values: np.ndarray,
    stats: dict,
    *,
    hidden_dim=5120,
) -> None:
    fields = _chunk_fields(key, ids, points, binding)
    if values.shape != (len(points), hidden_dim) or not np.isfinite(values).all():
        raise ValueError("Invalid selected vector geometry or values")
    with np.errstate(over="raise", invalid="raise"):
        stored = values.astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError("FP16 capture storage overflow")
    payload_name, done_name = _stream_names(key)
    payload = _safe(root, payload_name)
    _atomic(
        payload,
        lambda stream: np.savez(
            stream,
            vectors=stored,
            checkpoint_keys=np.asarray(fields["checkpoint_keys"], dtype="U64"),
            positions=np.asarray(fields["positions"], dtype=np.int64),
        ),
    )
    _write_json(
        _safe(root, done_name),
        {
            **fields,
            "payload_path": payload_name,
            "payload_sha256": sha256(payload),
            "shape": list(stored.shape),
            "dtype": "float16",
            "measurement": stats,
            "finished_utc": datetime.now(UTC).isoformat(),
            "verification_passed": True,
        },
    )


def read_chunk(root: Path, key: str, expected: dict, *, hidden_dim=5120) -> tuple[dict, np.ndarray]:
    payload_name, done_name = _stream_names(key)
    done = _read(_safe(root, done_name))
    if any(done.get(name) != value for name, value in expected.items()):
        raise ValueError("Resumable chunk belongs to a different input/recipe/source")
    if done.get("verification_passed") is not True or not done.get("finished_utc"):
        raise ValueError("Chunk lacks completed verification")
    if done.get("payload_path") != payload_name or done.get("payload_sha256") != sha256(
        _safe(root, payload_name)
    ):
        raise ValueError("Chunk payload bytes differ")
    with np.load(_safe(root, payload_name), allow_pickle=False) as payload:
        if set(payload.files) != {"vectors", "checkpoint_keys", "positions"}:
            raise ValueError("Unexpected chunk payload schema")
        vectors = payload["vectors"]
        if (
            payload["checkpoint_keys"].tolist() != done["checkpoint_keys"]
            or payload["positions"].tolist() != done["positions"]
        ):
            raise ValueError("Chunk checkpoint assignment differs")
    if (
        vectors.dtype != np.float16
        or vectors.shape != (len(done["checkpoint_keys"]), hidden_dim)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Chunk vector geometry, dtype or values differ")
    if done.get("shape") != list(vectors.shape) or done.get("dtype") != "float16":
        raise ValueError("Chunk descriptor differs")
    return done, vectors


def collect_streams(
    model,
    root: Path,
    streams: dict,
    checkpoints: list[dict],
    binding: dict,
    *,
    layer=44,
    hidden_dim=5120,
) -> dict:
    """Resume only exact verified per-stream payloads and persist each completed batch."""
    grouped = group_checkpoints(streams, checkpoints)
    pending = []
    for key in sorted(streams):
        _payload, done = _stream_names(key)
        if _safe(root, done).exists():
            read_chunk(
                root,
                key,
                _chunk_fields(key, streams[key], grouped[key], binding),
                hidden_dim=hidden_dim,
            )
        else:
            pending.append(key)
    cfg = binding["recipe"]["capture"]
    batches = batches_by_budget(
        [len(streams[k]) for k in pending], cfg["batch_max_rows"], cfg["batch_max_tokens"]
    )
    for batch in batches:
        keys = [pending[i] for i in batch]
        values, stats = _forward_streams(
            model, keys, streams, grouped, layer=layer, hidden_dim=hidden_dim
        )
        for key, vectors in zip(keys, values, strict=True):
            write_chunk(
                root,
                key,
                streams[key],
                grouped[key],
                binding,
                vectors,
                {"batch_stream_keys": keys, **stats},
                hidden_dim=hidden_dim,
            )
        print(json.dumps({"completed_streams": keys, "measurement": stats}), flush=True)
    return grouped


def finalize_capture(
    root: Path, streams: dict, grouped: dict, binding: dict, smoke_sha: str, *, hidden_dim=5120
) -> dict:
    keys = sorted(p["checkpoint_key"] for points in grouped.values() for p in points)
    if len(keys) != len(set(keys)):
        raise ValueError("Final checkpoint identity is not unique")
    offsets = {key: i for i, key in enumerate(keys)}
    values = np.empty((len(keys), hidden_dim), dtype=np.float16)
    chunk_hashes = {}
    for key in sorted(streams):
        done, vectors = read_chunk(
            root,
            key,
            _chunk_fields(key, streams[key], grouped[key], binding),
            hidden_dim=hidden_dim,
        )
        for checkpoint_key, vector in zip(done["checkpoint_keys"], vectors, strict=True):
            values[offsets[checkpoint_key]] = vector
        for name in _stream_names(key):
            chunk_hashes[name] = sha256(_safe(root, name))
    vector_path = _safe(root, "vectors.npy")
    _atomic(vector_path, lambda stream: np.save(stream, values, allow_pickle=False))
    index = {
        **binding,
        "checkpoint_keys": keys,
        "vectors_path": "vectors.npy",
        "vectors_sha256": sha256(vector_path),
        "vectors_shape": list(values.shape),
        "vectors_dtype": "float16",
        "streams": len(streams),
        "chunks_sha256": chunk_hashes,
        "smoke_path": "smoke.json",
        "smoke_sha256": smoke_sha,
        "tokenizer_replay_path": "tokenizer_replay_capture.json",
        "tokenizer_replay_sha256": sha256(_safe(root, "tokenizer_replay_capture.json")),
        "finished_utc": datetime.now(UTC).isoformat(),
        "verification_passed": True,
    }
    _write_json(_safe(root, "index.json"), index)
    return index


def _verify_smoke(root: Path, binding: dict) -> dict:
    smoke = _read(_safe(root, "smoke.json"))
    if (
        smoke.get("binding") != binding
        or smoke.get("verification_passed") is not True
        or not smoke.get("finished_utc")
    ):
        raise ValueError("Missing, failed or stale numerical smoke")
    validate_review(_safe(root, "independent_code_review.json"), binding["sources_sha256"])
    if smoke.get("review_sha256") != sha256(_safe(root, "independent_code_review.json")):
        raise ValueError("Smoke approval bytes differ")
    _validate_smoke_results(smoke, binding["recipe"]["smoke"])
    replay_path = _safe(root, "tokenizer_replay_smoke.json")
    if sha256(replay_path) != smoke.get("tokenizer_replay_sha256"):
        raise ValueError("Smoke tokenizer replay bytes differ")
    replay = _read(replay_path)
    if (
        replay.get("passed") is not True
        or replay.get("n_requests") != 2153
        or replay.get("prepared_manifest_sha256") != binding["prepared_manifest_sha256"]
    ):
        raise ValueError("Smoke tokenizer replay is incomplete")
    return smoke


def _validate_smoke_results(smoke: dict, limits: dict) -> None:
    comparisons = smoke["comparisons"]
    if not comparisons or sorted(comparisons) != smoke["expected_comparison_keys"]:
        raise ValueError("Smoke comparison coverage is incomplete")
    for result in comparisons.values():
        for name in ("raw", "mapped"):
            metrics = result[name]
            numbers = [metrics[key] for key in ("relative_l2", "cosine", "max_abs_relative")]
            if (
                not np.isfinite(numbers).all()
                or not 0 <= metrics["relative_l2"] <= limits[f"{name}_relative_l2_max"]
                or metrics["cosine"] < limits[f"{name}_cosine_min"]
            ):
                raise ValueError("Saved smoke violates a predeclared numerical threshold")
        if result.get("passed") is not True:
            raise ValueError("Saved smoke contains a failed comparison")
    mutation = smoke["suffix_mutation"]
    if (
        mutation.get("exact") is not True
        or mutation["mapped"].get("exact") is not True
        or mutation["relative_l2"] != 0
        or mutation["mapped"]["relative_l2"] != 0
    ):
        raise ValueError("Saved fixed-shape future invariance failed")
    future_mask = smoke["future_mask_mutation"]
    if (
        future_mask.get("exact") is not True
        or future_mask["mapped"].get("exact") is not True
        or future_mask["relative_l2"] != 0
        or future_mask["mapped"]["relative_l2"] != 0
    ):
        raise ValueError("Saved fixed-shape future-mask invariance failed")
    by_purpose = {m["purpose"]: m for m in smoke["measurements"]}
    required = {
        "longest_singleton",
        "representative_singleton",
        "production_short_batch",
        "same_forward_and_original_suffix",
        "fixed_shape_mutated_suffix",
        "fixed_shape_future_mask",
        "independent_truncated_prefix",
    }
    if (
        not required <= set(by_purpose)
        or by_purpose["same_forward_and_original_suffix"].get("same_forward_exact") is not True
    ):
        raise ValueError("Saved smoke lacks required production-shape or gather checks")


def verify_capture(
    capture_root: str | Path, prepared_manifest_sha256: str
) -> tuple[dict, np.memmap]:
    """Validate current sources, exact chunk/final bytes and checkpoint joins on CPU."""
    root = Path(capture_root)
    sources = source_hashes()
    controls = snapshot_paths(
        [
            _safe(root, name)
            for name in (
                "capture_binding.json",
                "index.json",
                "smoke.json",
                "independent_code_review.json",
                "tokenizer_replay_smoke.json",
                "tokenizer_replay_capture.json",
            )
        ]
    )
    binding = _read(_safe(root, "capture_binding.json"))
    _validate_binding(binding, prepared_manifest_sha256, sources)
    index = _read(_safe(root, "index.json"))
    if any(index.get(key) != value for key, value in binding.items()):
        raise ValueError("Final index differs from its frozen capture binding")
    if index.get("verification_passed") is not True or not index.get("finished_utc"):
        raise ValueError("Capture has no completed verified final index")
    _verify_smoke(root, binding)
    expected_names = {
        "vectors_path": "vectors.npy",
        "smoke_path": "smoke.json",
        "tokenizer_replay_path": "tokenizer_replay_capture.json",
    }
    for key, expected in expected_names.items():
        if (
            index.get(key) != expected
            or sha256(_safe(root, expected)) != index[key.replace("_path", "_sha256")]
        ):
            raise ValueError("Final capture artifact path or hash differs")
    replay = _read(_safe(root, index["tokenizer_replay_path"]))
    if (
        replay.get("passed") is not True
        or replay.get("prepared_manifest_sha256") != prepared_manifest_sha256
        or replay.get("n_requests") != 2153
    ):
        raise ValueError("Final capture lacks complete production tokenizer replay")
    vectors = np.load(_safe(root, "vectors.npy"), mmap_mode="r", allow_pickle=False)
    keys = index["checkpoint_keys"]
    if (
        keys != sorted(set(keys))
        or vectors.dtype != np.float16
        or vectors.shape != (len(keys), 5120)
        or index.get("vectors_shape") != list(vectors.shape)
        or index.get("vectors_dtype") != "float16"
    ):
        raise ValueError("Final captured vectors or checkpoint ordering differ")
    offsets = {key: i for i, key in enumerate(keys)}
    seen = set()
    chunk_hashes = index["chunks_sha256"]
    stream_keys = sorted(
        {
            Path(name).name.removesuffix(".done.json")
            for name in chunk_hashes
            if name.endswith(".done.json")
        }
    )
    if len(stream_keys) != index["streams"] or set(chunk_hashes) != {
        name for key in stream_keys for name in _stream_names(key)
    }:
        raise ValueError("Final chunk coverage differs")
    for name, expected in chunk_hashes.items():
        if sha256(_safe(root, name)) != expected:
            raise ValueError("Final chunk bytes differ")
    for key in stream_keys:
        done, values = read_chunk(
            root, key, {"stream_key": key, "capture_fingerprint": binding["fingerprint"]}
        )
        for checkpoint_key, vector in zip(done["checkpoint_keys"], values, strict=True):
            if (
                checkpoint_key in seen
                or checkpoint_key not in offsets
                or not np.array_equal(vector, vectors[offsets[checkpoint_key]])
            ):
                raise ValueError("Chunk-to-final checkpoint join differs")
            seen.add(checkpoint_key)
    if seen != set(keys):
        raise ValueError("Final capture checkpoint coverage is incomplete")
    assert_snapshot(controls)
    expected_payloads = {
        str(_safe(root, name).absolute()): value for name, value in chunk_hashes.items()
    }
    expected_payloads[str(_safe(root, "vectors.npy").absolute())] = index["vectors_sha256"]
    assert_snapshot(expected_payloads)
    if source_hashes() != sources:
        raise ValueError("Capture sources changed during final verification")
    return index, vectors


def verify_capture_geometry(
    capture_root: str | Path, index: dict, streams: dict, checkpoints: list[dict]
) -> dict:
    """Join a verified index to the caller's already-loaded, validated producer geometry.

    This adds the semantic descriptor join without reading native logs, loading the
    prepared package again, or rereading captured vector payloads.
    """
    root = Path(capture_root)
    index_path = _safe(root, "index.json")
    initial_index_hash = sha256(index_path)
    if _read(index_path) != index:
        raise ValueError("Geometry verifier received a different final index")
    grouped = {key: [] for key in streams}
    for point in checkpoints:
        if point["stream_key"] not in grouped or point["position"] != point["length"] - 1:
            raise ValueError("Prepared checkpoint geometry is inconsistent")
        grouped[point["stream_key"]].append(point)
    expected_keys = sorted(p["checkpoint_key"] for p in checkpoints)
    if (
        len(expected_keys) != len(set(expected_keys))
        or index["checkpoint_keys"] != expected_keys
        or index["streams"] != len(streams)
    ):
        raise ValueError("Final index does not cover the actual prepared geometry")
    expected_files = {name for key in streams for name in _stream_names(key)}
    if set(index["chunks_sha256"]) != expected_files or any(
        not points for points in grouped.values()
    ):
        raise ValueError("Stream coverage differs from actual prepared geometry")
    for key, points in sorted(grouped.items()):
        points.sort(key=lambda p: p["checkpoint_key"])
        done_name = _stream_names(key)[1]
        path = _safe(root, done_name)
        expected_hash = index["chunks_sha256"][done_name]
        if sha256(path) != expected_hash:
            raise ValueError("Geometry descriptor bytes differ from the verified index")
        done = _read(path)
        expected = _chunk_fields(key, streams[key], points, index)
        if any(done.get(name) != value for name, value in expected.items()):
            raise ValueError("Saved chunk descriptor differs from actual prepared geometry")
        if sha256(path) != expected_hash:
            raise ValueError("Geometry descriptor changed during verification")
    if sha256(index_path) != initial_index_hash:
        raise ValueError("Final index changed during geometry verification")
    return {
        "verification_passed": True,
        "prepared_manifest_sha256": index["prepared_manifest_sha256"],
        "streams": len(streams),
        "checkpoints": len(checkpoints),
    }


def run(cfg: DictConfig) -> dict:
    operation = str(cfg.operation)
    if operation not in {"preflight", "smoke", "capture", "verify"}:
        raise ValueError("Unknown capture operation")
    executable = recipe(cfg)
    sources = source_hashes()
    imported_sources = validate_imported_sources(sources)
    prepared_root = _producer().prepared_directory(Path(str(cfg.prepared_root)))
    root = Path(str(cfg.output_dir))
    guard_output(root, prepared_root, adopt=operation in {"smoke", "capture"})
    extras = (
        [Path(str(cfg.review)), Path(str(cfg.map_path))]
        if operation in {"smoke", "capture"}
        else []
    )
    before = prepared_snapshot(prepared_root, extras)
    if extras and before[str(extras[1].absolute())] != MAP_SHA256:
        raise ValueError("Frozen map bytes differ before model loading")
    prepared_sha = before[str((prepared_root / "manifest.json").absolute())]
    manifest, streams, checkpoints, observations = _producer().load_prepared(prepared_root)
    assert_snapshot(before)
    binding = make_binding(prepared_sha, executable, sources)
    if operation == "verify":
        result = verify_capture(root, prepared_sha)[0]
        verify_capture_geometry(root, result, streams, checkpoints)
        assert_snapshot(before)
        return result
    group_checkpoints(streams, checkpoints)
    if (
        len(streams) != 290
        or max(map(len, streams.values())) != executable["capture"]["max_sequence_tokens"]
    ):
        raise ValueError("Prepared production stream count or longest length differs")
    if operation == "preflight":
        result = {
            "binding": binding,
            "fixtures": smoke_fixtures(
                manifest,
                streams,
                checkpoints,
                observations,
                executable["capture"],
                executable["smoke"],
            ),
            "verification_passed": True,
        }
        assert_snapshot(before)
        if source_hashes() != sources:
            raise ValueError("Sources changed during prepared preflight")
        return result
    review_path = Path(str(cfg.review))
    validate_review(review_path, sources)
    root.mkdir(parents=True, exist_ok=True)
    binding_path = _safe(root, "capture_binding.json")
    if binding_path.exists() and _read(binding_path) != binding:
        raise ValueError("Output directory belongs to a different capture")
    _write_json(binding_path, binding)
    review_copy = _safe(root, "independent_code_review.json")
    if review_copy.exists() and sha256(review_copy) != sha256(review_path):
        raise ValueError("Capture approval changed within the same output directory")
    _atomic(review_copy, lambda stream: stream.write(review_path.read_bytes()))
    if operation == "capture":
        _verify_smoke(root, binding)
    model, runtime, replay = _loaded_runtime(cfg, prepared_root)
    if replay["prepared_manifest_sha256"] != prepared_sha or source_hashes() != sources:
        raise ValueError("Prepared input or sources changed before model forwards")
    assert_snapshot(before)
    _write_json(_safe(root, f"tokenizer_replay_{operation}.json"), replay)
    _write_json(_safe(root, f"imported_sources_{operation}.json"), imported_sources)
    if operation == "smoke":
        result = smoke_model(
            model,
            manifest,
            streams,
            checkpoints,
            observations,
            executable,
            load_map(Path(str(cfg.map_path))),
        )
        result.update(
            {
                "binding": binding,
                "runtime": runtime,
                "tokenizer_replay_sha256": sha256(_safe(root, "tokenizer_replay_smoke.json")),
                "review_sha256": sha256(review_copy),
                "finished_utc": datetime.now(UTC).isoformat(),
                "operation_inputs_sha256": before,
            }
        )
        assert_snapshot(before)
        if source_hashes() != sources:
            raise ValueError("Sources changed during numerical smoke")
        _write_json(_safe(root, "smoke.json"), result)
        if not result["verification_passed"]:
            raise ValueError(
                "Predeclared numerical smoke failed; no tolerance relaxation or production launch"
            )
        return result
    smoke = _verify_smoke(root, binding)
    if runtime != smoke["runtime"]:
        raise ValueError("Actual production runtime/backend differs from the smoke")
    grouped = collect_streams(model, root, streams, checkpoints, binding)
    assert_snapshot(before)
    if source_hashes() != sources:
        raise ValueError("Source or prepared manifest changed during capture")
    finalize_capture(root, streams, grouped, binding, sha256(_safe(root, "smoke.json")))
    result = verify_capture(root, prepared_sha)[0]
    verify_capture_geometry(root, result, streams, checkpoints)
    assert_snapshot(before)
    return result


@hydra.main(
    version_base=None, config_path="../configs/eval", config_name="context_risk_trajectory_capture"
)
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    print(
        json.dumps(
            {
                "verification_passed": result["verification_passed"],
                "operation": str(cfg.operation),
                "output_dir": str(cfg.output_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
