#!/usr/bin/env python3
"""Capture and compare persona-prompted context vectors without generating answers."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    _logits_to_keep_kwargs,
    _resolve_decoder_blocks,
    _unwrap,
)
from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    compute_cosine_matrix,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

ROOT = Path(__file__).resolve().parents[1]


def digest(value) -> str:
    """Hash a JSON-serializable specification with stable key ordering."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without allocating a second full artifact in RAM."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    """Atomically write strict JSON; reject non-finite metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_manifest(path: Path) -> dict:
    """Check the specification itself, not just its stored fingerprint string."""
    manifest = json.loads(path.read_text())
    if digest(manifest["spec"]) != manifest["fingerprint"]:
        raise RuntimeError("manifest specification/fingerprint mismatch")
    return manifest


def read_checksums(out: Path, fingerprint: str) -> dict[str, str]:
    """Retain recorded checksums across interrupted and completed captures."""
    checksums = {}
    for name in ("capture_chunks.json", "capture_complete.json"):
        path = out / name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["fingerprint"] != fingerprint:
            raise RuntimeError(f"stale checksum record: {path}")
        for chunk, checksum in record["chunk_sha256"].items():
            if chunk in checksums and checksums[chunk] != checksum:
                raise RuntimeError(f"conflicting recorded checksum: {chunk}")
            checksums[chunk] = checksum
    return checksums


def load_inputs(cfg: DictConfig) -> tuple[list[dict], list[dict], list[dict]]:
    """Load the exact published prompt bank and shared question IDs."""
    prompts = json.loads((ROOT / cfg.prompts).read_text())["prompts"]
    questions = [json.loads(line) for line in (ROOT / cfg.questions).read_text().splitlines()]
    if len(prompts) != 10 or len({p["id"] for p in prompts}) != 10:
        raise ValueError("expected ten uniquely identified persona conditions")
    if len(questions) != cfg.expected_questions or len({q["id"] for q in questions}) != len(
        questions
    ):
        raise ValueError("question count or unique-ID validation failed")
    rows = [
        {
            "row_id": f"{p['id']}:{q['id']}",
            "persona": p["id"],
            "question_id": q["id"],
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": q["question"]},
            ],
        }
        for p in prompts
        for q in questions
    ]
    return prompts, questions, rows


def pack_batches(lengths: list[int], row_limit: int, token_limit: int) -> list[list[int]]:
    """Pack length-sorted rows under a padded-token limit without dropping rows."""
    if row_limit < 1 or token_limit < 1 or any(n < 1 or n > token_limit for n in lengths):
        raise ValueError("invalid length or batch budget")
    batches, current = [], []
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        if current and (
            len(current) == row_limit or lengths[current[0]] * (len(current) + 1) > token_limit
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def render_rows(tokenizer, rows: list[dict], cfg: DictConfig) -> list[list[int]]:
    """Render exact generation prefixes and check the two tokenization paths agree."""
    result = []
    for row in rows:
        kwargs = dict(add_generation_prompt=True, enable_thinking=cfg.model.enable_thinking)
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, **kwargs)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        direct = tokenizer.apply_chat_template(row["messages"], tokenize=True, **kwargs)
        if hasattr(direct, "keys"):
            direct = direct["input_ids"]
        if hasattr(direct, "tolist"):
            direct = direct.tolist()
        if direct and isinstance(direct[0], list):
            if len(direct) != 1:
                raise ValueError("unexpected batched chat-template result")
            direct = direct[0]
        if direct != ids or not ids:
            raise ValueError(f"render/tokenizer disagreement: {row['row_id']}")
        if not cfg.model.enable_thinking and "<think>\n\n</think>" not in rendered:
            raise ValueError("thinking-off generation prefix lacks closed thinking block")
        if len(ids) > cfg.capture.max_context_tokens:
            raise ValueError(f"context exceeds cap; truncation forbidden: {row['row_id']}")
        row.update(
            token_count=len(ids),
            prefix_sha256=digest(ids),
            rendered_prefix=rendered,
            final_token_id=ids[-1],
        )
        result.append(ids)
    return result


@contextmanager
def numerical_backend(strict: bool):
    """Select recorded accuracy controls, restoring every global option on exit."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
    )
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not strict
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        backends = [SDPBackend.MATH]
        if not strict:
            backends += [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original[1]
        torch.backends.cudnn.allow_tf32 = original[2]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original[3]
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(original[4])
        torch.set_float32_matmul_precision(original[0])


@torch.inference_mode()
def capture_last(
    model,
    ids_rows: list[list[int]],
    pad_id: int,
    *,
    check_tuple=False,
    require_unpadded_singleton=False,
):
    """Gather only last-prefix block outputs, preserving pre-final-norm convention."""
    blocks, _embed, _depth = _resolve_decoder_blocks(model)
    if blocks is None:
        raise RuntimeError("decoder blocks unresolved; full-hidden-state fallback forbidden")
    device = next(model.parameters()).device
    lengths = torch.tensor([len(row) for row in ids_rows], device=device)
    ids = torch.full((len(ids_rows), int(lengths.max())), pad_id, device=device, dtype=torch.long)
    mask = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
    if require_unpadded_singleton and (len(ids_rows) != 1 or not bool(mask.all())):
        raise ValueError("production requires exactly one unpadded context per forward")
    for i, tokens in enumerate(ids_rows):
        ids[i, : len(tokens)] = torch.tensor(tokens, device=device)
    selected, handles = {}, []
    batch_indices = torch.arange(len(ids_rows), device=device)

    def hook_for(layer):
        def hook(_module, _inputs, output):
            hidden = _unwrap(output)
            selected[layer] = hidden[batch_indices, lengths - 1].detach().clone()

        return hook

    try:
        for layer, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook_for(layer)))
        outputs = model(
            input_ids=ids,
            attention_mask=mask.long(),
            use_cache=False,
            output_hidden_states=check_tuple,
            return_dict=True,
            **_logits_to_keep_kwargs(model, False),
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(selected) != set(range(len(blocks))):
        raise RuntimeError("missing decoder-hook outputs")
    values = torch.stack([selected[layer] for layer in range(len(blocks))], dim=1)
    if not torch.isfinite(values).all():
        raise RuntimeError("non-finite captured vectors")
    tuple_errors = {}
    if check_tuple:
        for layer in [0, 15, 31, 47, len(blocks) - 2]:
            reference = outputs.hidden_states[layer + 1][batch_indices, lengths - 1].float()
            err = (selected[layer].float() - reference).norm() / reference.norm().clamp_min(1e-12)
            tuple_errors[str(layer)] = float(err)
        if max(tuple_errors.values()) > 1e-5:
            raise RuntimeError(f"same-forward hook/tuple mismatch: {tuple_errors}")
    return values.cpu(), tuple_errors


def capture_production(model, ids_rows: list[list[int]], pad_id: int, *, check_tuple=False):
    """Capture a storage group with independently evaluated, unpadded singletons."""
    values, errors = [], {}
    with numerical_backend(strict=True):
        for index, ids in enumerate(ids_rows):
            value, row_errors = capture_last(
                model,
                [ids],
                pad_id,
                check_tuple=check_tuple,
                require_unpadded_singleton=True,
            )
            values.append(value)
            errors[str(index)] = row_errors
    return torch.cat(values), errors


def production_numerical_controls() -> dict:
    """Read back the real runtime controls used by the production context manager."""
    with numerical_backend(strict=True):
        settings = {
            "attention_interface": "sdpa",
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_fp16_bf16_reduction_math_sdp": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        }
    return settings


def numerical_smoke(model, tokenizer, ids, prompts, questions, cfg, out, fingerprint):
    """Check singleton state isolation and quantify backend effects on centered geometry."""
    if (cfg.capture.execution_mode, cfg.capture.numerics) != ("unpadded_singleton", "ieee_math"):
        raise ValueError("unsupported production execution mode")
    extreme = [
        min(range(len(ids)), key=lambda i: len(ids[i])),
        max(range(len(ids)), key=lambda i: len(ids[i])),
    ]
    sample = [ids[i] for i in extreme]
    initial, tuple_errors = capture_production(
        model, sample, tokenizer.pad_token_id, check_tuple=True
    )
    # Both questions are fixed by input order, before inspecting any geometry.
    bank_indices = [p * len(questions) + q for p in range(len(prompts)) for q in (0, 1)]
    bank_ids = [ids[i] for i in bank_indices]
    production, bank_tuple_errors = capture_production(
        model, bank_ids, tokenizer.pad_token_id, check_tuple=True
    )
    repeated, _ = capture_production(model, sample, tokenizer.pad_token_id, check_tuple=True)
    repeat_error = (initial.float() - repeated.float()).norm(dim=-1) / initial.float().norm(
        dim=-1
    ).clamp_min(1e-12)
    # The rejected mixed-batch path remains a diagnostic. It is never called by production.
    with numerical_backend(strict=True):
        mixed, _ = capture_last(model, sample, tokenizer.pad_token_id)
    mixed_error = (mixed.float() - initial.float()).norm(dim=-1) / initial.float().norm(
        dim=-1
    ).clamp_min(1e-12)
    with numerical_backend(strict=False):
        alternate = torch.cat(
            [
                capture_last(model, [row], tokenizer.pad_token_id, require_unpadded_singleton=True)[
                    0
                ]
                for row in bank_ids
            ]
        )
    banks = (
        torch.stack([production, alternate])
        .double()
        .reshape(2, len(prompts), 2, cfg.model.layers, cfg.model.hidden_dim)
        .mean(2)
    )
    centered = banks - banks.mean(1, keepdim=True)
    if torch.any(centered.norm(dim=-1) == 0):
        raise RuntimeError("degenerate numerical smoke bank")
    units = torch.nn.functional.normalize(centered, dim=-1)
    cosines = torch.einsum("bplh,bqlh->blpq", units, units)
    drift = (cosines[0] - cosines[1]).abs()
    smoke = {
        "fingerprint": fingerprint,
        "execution_mode": "unpadded_singleton",
        "sample_indices": extreme,
        "bank_indices": bank_indices,
        "hook_tuple_relative_errors": tuple_errors,
        "bank_hook_tuple_relative_errors": bank_tuple_errors,
        "repeatability_relative_errors": repeat_error.tolist(),
        "repeatability_bitwise_equal": bool(torch.equal(initial, repeated)),
        "mixed_batch_diagnostic": {
            "relative_errors": mixed_error.tolist(),
            "passed_original_threshold": bool(
                mixed_error.max() <= cfg.capture.parity_relative_tolerance
            ),
            "used_in_production": False,
            "original_failed_attempt_max_relative_error": 0.021424712613224983,
        },
        "backend_sensitivity": {
            "description": "Strict IEEE/math singleton production versus automatic SDPA/BF16-reduction singleton diagnostic; neither is an FP32-weight oracle.",
            "personas": [p["id"] for p in prompts],
            "question_ids": [q["id"] for q in questions[:2]],
            "centered_cosine": cosines.tolist(),
            "maximum_absolute_cosine_difference_by_layer": drift.amax(dim=(1, 2)).tolist(),
        },
        "passed": bool(repeat_error.max() <= cfg.capture.repeatability_relative_tolerance),
        "checked_at": time.time(),
    }
    np.savez(
        out / "smoke_vectors.npz",
        sample_indices=extreme,
        bank_indices=bank_indices,
        initial=initial.float().numpy(),
        repeated=repeated.float().numpy(),
        mixed=mixed.float().numpy(),
        production=production.float().numpy(),
        alternate=alternate.float().numpy(),
    )
    write_json(out / "smoke.json", smoke)
    if not smoke["passed"]:
        raise RuntimeError(f"singleton repeatability failed: {float(repeat_error.max())}")
    return smoke


def load_model(cfg: DictConfig):
    """Load the verified BF16 multimodal wrapper; check text geometry and runtime."""
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("capture requires remote CUDA compute")
    if transformers.__version__ != cfg.model.transformers_version:
        raise RuntimeError(
            f"expected transformers {cfg.model.transformers_version}, "
            f"got {transformers.__version__}"
        )
    config = transformers.AutoConfig.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    geometry = (config.text_config.num_hidden_layers, config.text_config.hidden_size)
    if geometry != (cfg.model.layers, cfg.model.hidden_dim):
        raise ValueError(f"unexpected model geometry: {geometry}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.id, revision=cfg.model.revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = getattr(transformers, "AutoModelForMultimodalLM", None)
    if loader is None:
        loader = transformers.AutoModelForImageTextToText
    model = loader.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    if model.config.text_config._attn_implementation != "sdpa":
        raise RuntimeError("numerical backend controls require the SDPA attention interface")
    return model, tokenizer


def validate_chunk(
    path: Path,
    fingerprint: str,
    indices: list[int],
    shape: tuple,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Validate every persisted chunk against its current recipe, IDs, and dimensions."""
    if expected_sha256 is not None and file_digest(path) != expected_sha256:
        raise RuntimeError(f"chunk content changed: {path}")
    chunk = torch.load(path, map_location="cpu", weights_only=True)
    values = chunk["vectors"]
    if (
        chunk["fingerprint"] != fingerprint
        or chunk["indices"] != indices
        or tuple(values.shape) != shape
        or values.dtype != torch.bfloat16
        or not torch.isfinite(values).all()
    ):
        raise RuntimeError(f"invalid/stale capture chunk: {path}")
    return chunk


def phase_capture(cfg: DictConfig, out: Path) -> None:
    """Smoke, checkpoint, and verify the full bank with fingerprinted resume."""
    import transformers
    import wandb

    start = time.monotonic()
    provenance = as_metadata_dict(git_provenance(), phase="capture")
    expected_source = os.environ.get("EPS_STORY_PERSONA_SOURCE_SHA")
    if (
        not expected_source
        or provenance["git_commit"] != expected_source
        or provenance["git_dirty"] is not False
    ):
        raise RuntimeError("capture requires the pinned, clean source commit before model loading")
    prompts, questions, rows = load_inputs(cfg)
    print(f"[capture] loading pinned model; started_at={time.time()}", flush=True)
    model, tokenizer = load_model(cfg)
    ids = render_rows(tokenizer, rows, cfg)
    batches = pack_batches(
        [len(x) for x in ids], cfg.capture.batch_rows, cfg.capture.padded_token_budget
    )
    source_paths = [
        Path(__file__),
        ROOT / "src/explore_persona_space/analysis/extraction.py",
        ROOT / "src/explore_persona_space/analysis/representation_shift.py",
    ]
    spec = {
        "model": OmegaConf.to_container(cfg.model),
        "capture": OmegaConf.to_container(cfg.capture),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "source_hashes": {str(p.relative_to(ROOT)): file_digest(p) for p in source_paths},
        "inputs_sha256": digest(rows),
        "prompts": prompts,
        "question_ids": [q["id"] for q in questions],
        "batches": batches,
        "dtype": "bfloat16",
        "layers": list(range(cfg.model.layers)),
        "position": "last_generation_prefix_token",
        "final_layer": "pre_final_norm",
        "numerical_controls": production_numerical_controls(),
    }
    fingerprint = digest(spec)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and read_manifest(manifest_path)["fingerprint"] != fingerprint:
        raise RuntimeError("output directory already contains a different recipe")
    checksums = read_checksums(out, fingerprint)
    # A resumed attempt must earn a fresh completion sentinel, even if smoke fails.
    write_json(out / "capture_chunks.json", {"fingerprint": fingerprint, "chunk_sha256": checksums})
    (out / "capture_complete.json").unlink(missing_ok=True)
    write_json(manifest_path, {"fingerprint": fingerprint, "spec": spec, "provenance": provenance})
    write_json(out / "rows.json", rows)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    print(
        f"[capture] checking unpadded singleton correctness and backend sensitivity; checked_at={time.time()}",
        flush=True,
    )
    smoke = numerical_smoke(model, tokenizer, ids, prompts, questions, cfg, out, fingerprint)
    smoke["elapsed_seconds"] = time.monotonic() - start
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    batch_start = time.monotonic()
    capture_production(model, [ids[i] for i in batches[0]], tokenizer.pad_token_id)
    torch.cuda.synchronize()
    batch_seconds = time.monotonic() - batch_start
    smoke.update(
        largest_batch_seconds=batch_seconds,
        largest_batch_rows=len(batches[0]),
        largest_batch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        conservative_capture_seconds=batch_seconds * len(batches),
        checked_at=time.time(),
    )
    write_json(out / "smoke.json", smoke)
    print(
        f"[capture] smoke passed; largest batch={batch_seconds:.2f}s, "
        f"conservative capture estimate={batch_seconds * len(batches):.0f}s",
        flush=True,
    )
    run = wandb.init(
        project="story-persona-qwen38",
        mode=cfg.wandb_mode,
        config=spec,
        dir=str(out),
        name=fingerprint[:12],
    )
    try:
        for k, indices in enumerate(batches):
            path = out / "chunks" / f"batch_{k:04d}.pt"
            shape = (len(indices), cfg.model.layers, cfg.model.hidden_dim)
            if path.exists():
                if path.name not in checksums:
                    raise RuntimeError(
                        f"chunk has no committed checksum; recovery required: {path}"
                    )
                validate_chunk(
                    path, fingerprint, indices, shape, expected_sha256=checksums[path.name]
                )
            else:
                values, _ = capture_production(
                    model, [ids[i] for i in indices], tokenizer.pad_token_id
                )
                tmp = path.with_suffix(".pt.tmp")
                torch.save({"fingerprint": fingerprint, "indices": indices, "vectors": values}, tmp)
                tmp.replace(path)
                validate_chunk(path, fingerprint, indices, shape)
                checksums[path.name] = file_digest(path)
                write_json(
                    out / "capture_chunks.json",
                    {"fingerprint": fingerprint, "chunk_sha256": checksums},
                )
            progress = {
                "fingerprint": fingerprint,
                "checked_at": time.time(),
                "completed_batches": k + 1,
                "total_batches": len(batches),
                "elapsed_seconds": time.monotonic() - start,
            }
            write_json(out / "progress.json", progress)
            run.log(progress)
            print(
                f"[capture] batch {k + 1}/{len(batches)} {path.name} "
                f"elapsed={time.monotonic() - start:.1f}s",
                flush=True,
            )
    finally:
        run.finish()
    coverage = sorted(i for batch in batches for i in batch)
    if coverage != list(range(len(rows))):
        raise RuntimeError("capture row coverage is not exact")
    write_json(
        out / "capture_complete.json",
        {
            "fingerprint": fingerprint,
            "row_count": len(rows),
            "chunk_sha256": checksums,
            "checked_at": time.time(),
            "provenance": provenance,
        },
    )


def phase_analyze(cfg: DictConfig, out: Path) -> None:
    """Stream verified chunks into centroids, cosine banks, and question-half stability."""
    manifest = read_manifest(out / "manifest.json")
    done = json.loads((out / "capture_complete.json").read_text())
    if done["fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError("completion/manifest mismatch")
    spec = manifest["spec"]
    rows = json.loads((out / "rows.json").read_text())
    if digest(rows) != spec["inputs_sha256"]:
        raise RuntimeError("persisted row metadata changed")
    names = [p["id"] for p in spec["prompts"]]
    qids = spec["question_ids"]
    layer_count = len(spec["layers"])
    width = spec["model"]["hidden_dim"]
    sums = torch.zeros((2, len(names), layer_count, width), dtype=torch.float64)
    counts = torch.zeros((2, len(names)), dtype=torch.int64)
    seen = []
    if set(done["chunk_sha256"]) != {f"batch_{k:04d}.pt" for k in range(len(spec["batches"]))}:
        raise RuntimeError("completion chunk coverage mismatch")
    for k, indices in enumerate(spec["batches"]):
        path = out / "chunks" / f"batch_{k:04d}.pt"
        chunk = validate_chunk(
            path,
            manifest["fingerprint"],
            indices,
            (len(indices), layer_count, width),
            expected_sha256=done["chunk_sha256"][path.name],
        )
        slots = torch.tensor(
            [
                2 * names.index(rows[i]["persona"]) + qids.index(rows[i]["question_id"]) % 2
                for i in indices
            ]
        )
        # Accumulate in persona-major order, then restore half-major layout.
        flat = sums.permute(1, 0, 2, 3).contiguous().view(2 * len(names), layer_count, width)
        flat.index_add_(0, slots, chunk["vectors"].double())
        sums = flat.view(len(names), 2, layer_count, width).permute(1, 0, 2, 3).contiguous()
        flat_counts = counts.T.contiguous().view(-1)
        flat_counts.index_add_(0, slots, torch.ones(len(indices), dtype=torch.int64))
        counts = flat_counts.view(len(names), 2).T.contiguous()
        seen.extend(indices)
    if sorted(seen) != list(range(len(rows))) or done["row_count"] != len(rows):
        raise RuntimeError("analysis coverage mismatch")
    if not torch.all(counts == len(qids) // 2) or len(qids) % 2:
        raise RuntimeError("unexpected split-half counts")
    centroids = sums.sum(0) / counts.sum(0)[:, None, None]
    halves = sums / counts[:, :, None, None]
    primary = [i for i, p in enumerate(spec["prompts"]) if p["primary"]]
    matrices, raw, six, stability = [], [], [], []
    for layer in range(layer_count):
        bank = centroids[:, layer]
        if torch.any((bank - bank.mean(0)).norm(dim=-1) == 0):
            raise RuntimeError(f"degenerate centered centroid at layer {layer}")
        matrices.append(compute_cosine_matrix(bank, centering="global_mean"))
        raw.append(compute_cosine_matrix(bank, centering="none"))
        six.append(compute_cosine_matrix(bank[primary], centering="global_mean"))
        centered = halves[:, :, layer] - halves[:, :, layer].mean(1, keepdim=True)
        if torch.any((bank[primary] - bank[primary].mean(0)).norm(dim=-1) == 0) or torch.any(
            centered.norm(dim=-1) == 0
        ):
            raise RuntimeError(f"degenerate sensitivity/stability centroid at layer {layer}")
        stability.append(torch.nn.functional.cosine_similarity(centered[0], centered[1], dim=-1))
    summary = {
        "fingerprint": manifest["fingerprint"],
        "model": spec["model"],
        "persona_names": names,
        "layers": spec["layers"],
        "centering": "global_mean",
        "question_count": len(qids),
        "counts_by_half": counts.tolist(),
        "centered_cosine": torch.stack(matrices).tolist(),
        "raw_cosine_diagnostic": torch.stack(raw).tolist(),
        "six_persona_sensitivity": {
            "persona_names": [names[i] for i in primary],
            "cosine": torch.stack(six).tolist(),
        },
        "split_half_centered_cosine": torch.stack(stability).tolist(),
        "token_counts_by_persona": {
            name: [r["token_count"] for r in rows if r["persona"] == name] for name in names
        },
        "interpretation": "Context geometry only; no Qwen behavioral leakage measurement.",
        "provenance": as_metadata_dict(git_provenance(), phase="analyze"),
    }
    write_json(out / "summary.json", summary)
    centroids_tmp = out / "centroids.tmp.npz"
    np.savez(centroids_tmp, centroids=centroids.numpy(), half_centroids=halves.numpy())
    centroids_tmp.replace(out / "centroids.npz")
    print(f"[analyze] verified {len(rows)} rows across {layer_count} layers", flush=True)


@hydra.main(version_base=None, config_path="../configs/pilots", config_name="story_persona_qwen38")
def main(cfg: DictConfig) -> None:
    """Prepare local inputs, capture on approved compute, or analyze completed artifacts."""
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if cfg.phase == "prepare":
        prompts, questions, rows = load_inputs(cfg)
        write_json(
            out / "prepared_inputs.json",
            {
                "prompts": prompts,
                "questions": questions,
                "row_count": len(rows),
                "inputs_sha256": digest(rows),
                "model": OmegaConf.to_container(cfg.model),
            },
        )
        print(
            f"[prepare] {len(prompts)} personas x {len(questions)} questions = {len(rows)} contexts"
        )
    elif cfg.phase == "capture":
        phase_capture(cfg, out)
    elif cfg.phase == "analyze":
        phase_analyze(cfg, out)
    else:
        raise ValueError(f"unknown phase: {cfg.phase}")


if __name__ == "__main__":
    main()
