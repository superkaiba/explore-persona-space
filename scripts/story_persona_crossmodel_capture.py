#!/usr/bin/env python3
"""Pinned, forward-only context capture for the task-2673 description comparison.

Smoke uses the real production model, hooks, numerics and batch paths. Its limits:
it samples numerical parity rather than certifying every context; full-store
coverage and checkpoint publication are verified only as production progresses.
No generation, training, analysis, or provisioning is performed here.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    """Resolve sibling scripts independently of the caller's working directory."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


ROOT = _ensure_repo_root_on_syspath()

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    _logits_to_keep_kwargs,
    _resolve_decoder_blocks,
    _unwrap,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from scripts.story_persona_qwen38_pilot import (  # noqa: E402
    digest,
    file_digest,
    load_model as load_qwen_model,
    numerical_backend,
    pack_batches,
    production_numerical_controls,
    read_checksums,
    read_manifest,
    render_rows as render_qwen_rows,
    validate_chunk,
    write_json,
)
from scripts.story_persona_fp8 import cuda_shape_diagnostic, install_grouped_m16  # noqa: E402

PERSONAS = ["hhh", "fred", "helpful", "dismissive", "sarcastic", "saboteur", "peer", "help_seeker"]


def source_preflight() -> dict:
    """A changed or uncommitted source must fail before any model weights load."""
    provenance = as_metadata_dict(git_provenance(), phase="crossmodel_capture")
    expected = os.environ.get("EPS_STORY_PERSONA_SOURCE_SHA", "")
    if (
        re.fullmatch(r"[0-9a-f]{40}", expected) is None
        or provenance["git_commit"] != expected
        or provenance["git_dirty"] is not False
    ):
        raise RuntimeError("capture requires the pinned, clean source commit")
    return provenance


def deadline_from_environment() -> tuple[float, float]:
    """Wrapper records paid-start and deadline before staging, never after loading."""
    started = float(os.environ["EPS_STORY_PERSONA_ALLOCATION_STARTED_UNIX"])
    deadline = float(os.environ["EPS_STORY_PERSONA_DEADLINE_UNIX"])
    now = time.time()
    if not all(math.isfinite(x) for x in (started, deadline)) or not started <= now < deadline:
        raise RuntimeError("invalid or expired allocation window")
    return started, deadline


def read_inputs(cfg: DictConfig) -> tuple[list[dict], list[dict], list[dict]]:
    prompts = json.loads((ROOT / cfg.prompts).read_text())["prompts"]
    questions = [json.loads(s) for s in (ROOT / cfg.questions).read_text().splitlines()]
    if [p["id"] for p in prompts] != PERSONAS:
        raise ValueError("expected the registered ordered eight-description bank")
    if len(questions) != cfg.expected_questions or len({q["id"] for q in questions}) != len(
        questions
    ):
        raise ValueError("question count/IDs do not match the registered bank")
    rows = [
        {
            "row_id": f"{p['id']}:{q['id']}",
            "persona": p["id"],
            "question_id": q["id"],
            "description": p["system"],
            "question": q["question"],
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": q["question"]},
            ],
        }
        for p in prompts
        for q in questions
    ]
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("ambiguous persona/question row keys")
    return prompts, questions, rows


def render_inputs(tokenizer, rows: list[dict], cfg: DictConfig) -> list[list[int]]:
    """DeepSeek's observed pinned tokenizer adds neither BOS nor EOS to plain text."""
    if cfg.model_key == "qwen":
        ids = render_qwen_rows(tokenizer, rows, cfg)
    elif cfg.model_key == "deepseek":
        if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (
            cfg.model.bos_token_id,
            cfg.model.eos_token_id,
        ) or (
            cfg.model.add_special_tokens is not True
            or cfg.model.add_bos_token is not False
            or getattr(tokenizer, "add_bos_token", None) is not False
        ):
            raise ValueError("DeepSeek tokenizer special-token contract changed")
        ids = []
        for row in rows:
            text = row["description"] + "\n\nHuman: " + row["question"] + "\n\nAssistant:"
            tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
            plain = tokenizer(text, add_special_tokens=False)["input_ids"]
            if tokens != plain or not plain or tokens[0] == tokenizer.bos_token_id:
                raise ValueError("DeepSeek pinned tokenizer must add neither BOS nor EOS")
            if len(tokens) > cfg.capture.max_context_tokens:
                raise ValueError("context cap exceeded; truncation forbidden")
            row.update(
                rendered_prefix=text,
                token_count=len(tokens),
                prefix_sha256=digest(tokens),
                final_token_id=tokens[-1],
                special_tokens="add_special_tokens=True; pinned add_bos_token=False; no BOS/EOS added",
            )
            ids.append(tokens)
    else:
        raise ValueError(f"unknown model arm: {cfg.model_key}")
    for row, tokens in zip(rows, ids, strict=True):
        row["input_ids"] = tokens
        row["final_token"] = tokenizer.convert_ids_to_tokens(tokens[-1])
    lengths = [len(x) for x in ids]
    observed = {"min": min(lengths), "max": max(lengths), "total": sum(lengths)}
    if observed != OmegaConf.to_container(cfg.model.expected_tokens):
        raise ValueError(f"registered all-context token statistics changed: {observed}")
    return ids


@torch.inference_mode()
def capture_last(model, ids_rows, pad_id, *, layers, width, check_tuple=False, singleton=False):
    """Select on each output device, then transfer only last-token residuals to CPU."""
    blocks, _embed, _ = _resolve_decoder_blocks(model)
    if blocks is None or len(blocks) != layers:
        raise RuntimeError("decoder block count changed")
    if not ids_rows or any(not row for row in ids_rows) or (singleton and len(ids_rows) != 1):
        raise ValueError("empty contexts or invalid singleton batch")
    device = model.get_input_embeddings().weight.device
    lengths = [len(row) for row in ids_rows]
    ids = torch.full((len(ids_rows), max(lengths)), pad_id, dtype=torch.long, device=device)
    mask = (
        torch.arange(ids.shape[1], device=device)[None]
        < torch.tensor(lengths, device=device)[:, None]
    )
    for i, tokens in enumerate(ids_rows):
        ids[i, : len(tokens)] = torch.tensor(tokens, device=device)
    selected, handles = {}, []

    def select(hidden):
        # CPU and every CUDA block have their own indices. Never stack cross-device tensors.
        index = torch.arange(len(ids_rows), device=hidden.device)
        positions = torch.tensor(lengths, device=hidden.device) - 1
        return hidden[index, positions].detach().to(device="cpu", copy=True)

    def hook_for(layer):
        def hook(_module, _inputs, output):
            hidden = _unwrap(output)
            if layer in selected or tuple(hidden.shape) != (len(ids_rows), max(lengths), width):
                raise RuntimeError("duplicate hook or unexpected block shape")
            if hidden.dtype != torch.bfloat16 or hidden.device.type == "meta":
                raise RuntimeError("residual must be real BF16, not a silently converted dtype")
            selected[layer] = select(hidden)

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
    if set(selected) != set(range(layers)):
        raise RuntimeError("missing decoder-hook outputs")
    values = torch.stack([selected[i] for i in range(layers)], dim=1)
    if not torch.isfinite(values).all():
        raise RuntimeError("non-finite context vectors")
    errors = {}
    if check_tuple:
        if len(outputs.hidden_states) != layers + 1:
            raise RuntimeError("unexpected hidden-state tuple layout")
        for layer in sorted({0, layers // 4, layers // 2, 3 * layers // 4, layers - 2}):
            reference = select(outputs.hidden_states[layer + 1]).float()
            error = (selected[layer].float() - reference).norm(dim=-1) / reference.norm(
                dim=-1
            ).clamp_min(1e-12)
            errors[str(layer)] = error.tolist()
        if max(x for row in errors.values() for x in row) > 1e-5:
            raise RuntimeError(f"same-forward hook/tuple mismatch: {errors}")
    return values, errors


def capture_production(model, ids_rows, pad_id, cfg, *, check_tuple=False):
    qwen = cfg.model_key == "qwen"
    groups = [[row] for row in ids_rows] if qwen else [ids_rows]
    values, errors = [], {}
    with numerical_backend(strict=True) if qwen else nullcontext():
        for k, group in enumerate(groups):
            value, checks = capture_last(
                model,
                group,
                pad_id,
                layers=cfg.model.layers,
                width=cfg.model.hidden_dim,
                check_tuple=check_tuple,
                singleton=qwen,
            )
            values.append(value)
            errors[str(k)] = checks
    return torch.cat(values), errors


def deepseek_device_map(layers=61, devices=8, dense_layers=3) -> dict[str, int]:
    """Balance the dominant expert layers contiguously; dense prefix stays on card 0."""
    if layers <= dense_layers or devices < 2 or layers - dense_layers < devices:
        raise ValueError("invalid layer-sharding geometry")
    result = {"model.embed_tokens": 0, "model.norm": devices - 1, "lm_head": devices - 1}
    for i in range(layers):
        result[f"model.layers.{i}"] = max(
            0, (i - dense_layers) * devices // (layers - dense_layers)
        )
    return result


def pin_fp8_kernel(cfg) -> dict:
    """Bind the pinned Transformers internal loader to an immutable kernel snapshot."""
    from transformers.utils.import_utils import (
        KERNELS_MAX_VERSION,
        KERNELS_MIN_VERSION,
        is_kernels_available,
    )
    from transformers.integrations import hub_kernels

    if importlib.metadata.version("kernels") != cfg.model.kernels_version:
        raise RuntimeError("unexpected kernels package version")
    if not is_kernels_available():
        raise RuntimeError(
            f"Transformers rejects kernels=={cfg.model.kernels_version}; "
            f"requires >={KERNELS_MIN_VERSION},<{KERNELS_MAX_VERSION}"
        )
    mapping = hub_kernels._HUB_KERNEL_MAPPING
    if mapping.get("finegrained-fp8") != {"repo_id": cfg.model.kernel_repo, "version": 4}:
        raise RuntimeError("Transformers FP8 kernel mapping changed before pinning")
    if "finegrained-fp8" in hub_kernels._KERNEL_MODULE_MAPPING:
        raise RuntimeError("FP8 kernel was loaded before its revision was pinned")
    mapping["finegrained-fp8"] = {
        "repo_id": cfg.model.kernel_repo,
        "revision": cfg.model.kernel_revision,
    }
    kernel = hub_kernels.lazy_load_kernel("finegrained-fp8")
    if kernel is None or any(
        not hasattr(kernel, n) for n in ("matmul_2d", "matmul_batched", "matmul_grouped")
    ):
        raise RuntimeError("pinned FP8 kernel is unavailable or has an unexpected interface")
    # Keep the snapshot path; resolve() would follow a Hub symlink into blobs/.
    path = Path(kernel.__file__).absolute()
    if cfg.model.kernel_revision not in path.parts:
        raise RuntimeError(f"kernel is not from the declared immutable snapshot: {path}")
    build = path.parent
    hashes = {
        str(p.relative_to(build)): file_digest(p)
        for p in sorted(build.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".so", ".json")
    }
    if not hashes:
        raise RuntimeError("kernel snapshot contains no auditable implementation files")
    scheduling_override = install_grouped_m16(kernel, cfg.model.kernel_revision)
    print(
        f"[fp8-kernel-ready] kernels={cfg.model.kernels_version} "
        f"revision={cfg.model.kernel_revision} loader=transformers.lazy_load_kernel",
        flush=True,
    )
    return {
        "repo": cfg.model.kernel_repo,
        "repo_type": cfg.model.kernel_repo_type,
        "revision": cfg.model.kernel_revision,
        "module_path": str(path),
        "build_sha256": hashes,
        "kernels_version": cfg.model.kernels_version,
        "scheduling_override": scheduling_override,
    }


def effective_host_memory() -> int:
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    limits = [int(mem["MemTotal"].split()[0]) * 1024]
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if path.exists() and path.read_text().strip() != "max":
            limits.append(int(path.read_text()))
    return min(limits)


def validated_loading_info(loading: dict) -> dict:
    """Validate the pinned HF loading report and canonicalize its sets for JSON.

    Transformers 5.15 returns sets even when the checkpoint loads cleanly.
    Keep only the documented MTP exclusion; unknown schemas fail closed.
    """
    expected = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if not isinstance(loading, dict) or set(loading) != expected:
        raise RuntimeError("checkpoint loading report schema changed")
    if any(not isinstance(v, (list, tuple, set)) for v in loading.values()):
        raise RuntimeError("checkpoint loading report collection types changed")
    if any(loading[k] for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"unexplained checkpoint loading differences: {loading}")
    unexpected = loading["unexpected_keys"]
    if any(not isinstance(k, str) or not re.match(r"model\.layers\.61\.", k) for k in unexpected):
        raise RuntimeError(f"unexplained checkpoint loading differences: {loading}")
    return {key: sorted(loading[key]) for key in sorted(expected)}


def load_model(cfg):
    import transformers

    if (
        transformers.__version__ != cfg.model.transformers_version
        or torch.__version__ != cfg.model.torch_version
        or importlib.metadata.version("triton") != cfg.model.triton_version
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("capture requires the pinned Transformers version and CUDA")
    if cfg.model_key == "qwen":
        model, tokenizer = load_qwen_model(cfg)
        return (
            model,
            tokenizer,
            {"weight_format": "BF16", "numerical_controls": production_numerical_controls()},
        )
    if cfg.model_key != "deepseek":
        raise ValueError("unsupported model key")
    # Import optional audio/model dependencies before downloading the checkpoint.
    # The base image can otherwise contribute torchaudio built for a different Torch ABI.
    import torchaudio

    if torchaudio.__version__ != cfg.model.torchaudio_version:
        raise RuntimeError("DeepSeek requires the pinned torchaudio/Torch ABI pair")
    model_class = transformers.DeepseekV3ForCausalLM
    print(
        f"[runtime-ready] torchaudio={torchaudio.__version__} model={model_class.__name__}",
        flush=True,
    )
    if (
        torch.cuda.device_count() != cfg.model.gpu_count
        or effective_host_memory() < cfg.model.min_host_bytes
    ):
        raise RuntimeError(
            "DeepSeek requires eight visible H200-class cards and >=1TB effective host RAM"
        )
    for i in range(cfg.model.gpu_count):
        if (
            torch.cuda.get_device_capability(i) < (9, 0)
            or torch.cuda.get_device_properties(i).total_memory < cfg.model.min_gpu_bytes
        ):
            raise RuntimeError(
                "unsupported FP8 GPU or insufficient per-device HBM; fallback forbidden"
            )
    config = transformers.AutoConfig.from_pretrained(
        cfg.model.id, revision=cfg.model.revision, trust_remote_code=False
    )
    expected_quant = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
    }
    if (config.num_hidden_layers, config.hidden_size, config.first_k_dense_replace) != (
        61,
        7168,
        3,
    ) or config.quantization_config != expected_quant:
        raise RuntimeError("DeepSeek geometry or native FP8 configuration changed")
    kernel = pin_fp8_kernel(cfg)
    from transformers.integrations import hub_kernels

    diagnostic = cuda_shape_diagnostic(
        hub_kernels.lazy_load_kernel("finegrained-fp8"),
        parity_tolerance=cfg.capture.parity_relative_tolerance,
        repeatability_tolerance=cfg.capture.repeatability_relative_tolerance,
        record=lambda report: write_json(
            Path(cfg.output_dir) / "fp8_shape_diagnostic.json", report
        ),
    )
    write_json(Path(cfg.output_dir) / "fp8_shape_diagnostic.json", diagnostic)
    if not diagnostic["passed"]:
        raise RuntimeError("native FP8 shape diagnostic rejected fixed-M16 parity/repeatability")
    print("[fp8-shape-diagnostic] passed=true; full-model smoke still required", flush=True)
    device_map = deepseek_device_map()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.id, revision=cfg.model.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, loading = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        experts_implementation=cfg.model.experts_implementation,
        device_map=device_map,
        max_memory={
            i: f"{cfg.model.max_parameter_gib_per_gpu}GiB" for i in range(cfg.model.gpu_count)
        },
        output_loading_info=True,
    )
    loading = validated_loading_info(loading)
    quantizer = model.hf_quantizer
    if quantizer.quantization_config.dequantize or model.hf_device_map != device_map:
        raise RuntimeError("silent dequantization or device-map fallback")
    from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear

    fp8 = [(n, m) for n, m in model.named_modules() if isinstance(m, (FP8Linear, FP8Experts))]
    if not fp8 or sum(isinstance(m, FP8Experts) for _, m in fp8) != 58:
        raise RuntimeError("native FP8 modules/58 expert blocks missing")
    for name, module in fp8:
        weights = [
            p for n, p in module.named_parameters(recurse=False) if "scale" not in n and p.ndim >= 2
        ]
        if (
            not weights
            or any(p.dtype != torch.float8_e4m3fn for p in weights)
            or not module._deepgemm_disabled
        ):
            raise RuntimeError(f"FP8 weight/backend contract failed: {name}")
    parameter_bytes = {i: 0 for i in range(cfg.model.gpu_count)}
    for name, param in model.named_parameters():
        if param.device.type != "cuda" or param.device.index not in parameter_bytes:
            raise RuntimeError(f"CPU/disk/meta parameter offload forbidden: {name}")
        if name.endswith("e_score_correction_bias") and param.dtype != torch.float32:
            raise RuntimeError("router correction bias must stay FP32")
        parameter_bytes[param.device.index] += param.numel() * param.element_size()
    corrections = [b for n, b in model.named_buffers() if n.endswith("e_score_correction_bias")]
    if len(corrections) != 58 or any(
        b.dtype != torch.float32 or b.device.type != "cuda" for b in corrections
    ):
        raise RuntimeError("all 58 routing correction buffers must remain FP32 on GPU")
    blocks, _, _ = _resolve_decoder_blocks(model)
    if blocks is None or len(blocks) != cfg.model.layers:
        raise RuntimeError("loaded decoder geometry differs from the registered model")
    for i, block in enumerate(blocks):
        expected_device = torch.device("cuda", device_map[f"model.layers.{i}"])
        if any(p.device != expected_device for p in block.parameters()):
            raise RuntimeError(f"block {i} parameters violate explicit layer sharding")
    if any(n > cfg.model.max_parameter_gib_per_gpu * 1024**3 for n in parameter_bytes.values()):
        raise RuntimeError("explicit per-card parameter-placement ceiling exceeded")
    model.eval()
    if model.config._experts_implementation != cfg.model.experts_implementation:
        raise RuntimeError("expert implementation fallback is forbidden")
    return (
        model,
        tokenizer,
        {
            "weight_format": "native checkpoint FP8",
            "quantization": expected_quant,
            "kernel": kernel,
            "device_map": device_map,
            "parameter_bytes_by_gpu": parameter_bytes,
            "loading_info": loading,
            "torchaudio": torchaudio.__version__,
            "mtp": "excluded model.layers.61; capture main blocks 0..60",
            "experts_implementation": str(model.config._experts_implementation),
            "deepgemm_disabled": True,
            "attention_interface": "sdpa",
        },
    )


def relative_errors(left, right):
    return (left.float() - right.float()).norm(dim=-1) / right.float().norm(dim=-1).clamp_min(1e-12)


def numerical_smoke(model, tokenizer, ids, cfg, out, fingerprint):
    """Interleaved singleton replay plus all-persona and mixed-padding batch parity."""
    extreme = list(
        dict.fromkeys(
            [
                min(range(len(ids)), key=lambda i: len(ids[i])),
                max(range(len(ids)), key=lambda i: len(ids[i])),
            ]
        )
    )
    bank = [p * cfg.expected_questions + q for p in range(8) for q in (0, 1)]
    indices = list(dict.fromkeys(extreme + bank))
    singles, tuple_errors = [], {}
    for i in indices:
        value, error = capture_production(
            model, [ids[i]], tokenizer.pad_token_id, cfg, check_tuple=True
        )
        singles.append(value)
        tuple_errors[str(i)] = error
        write_json(
            out / "progress.json",
            {
                "stage": "smoke",
                "fingerprint": fingerprint,
                "smoke_singletons_completed": len(singles),
                "checked_at": time.time(),
            },
        )
    initial = torch.cat(singles)
    # Replay in reverse order, after unrelated contexts have intervened.
    replay = torch.cat(
        [
            capture_production(model, [ids[i]], tokenizer.pad_token_id, cfg)[0]
            for i in reversed(indices)
        ]
    ).flip(0)
    mixed_order = extreme + [i for i in indices if i not in extreme]
    batch_values, batch_indices = [], []
    for offset in range(0, len(mixed_order), cfg.capture.batch_rows):
        group = mixed_order[offset : offset + cfg.capture.batch_rows]
        # This is production for DeepSeek. Qwen batching remains a diagnostic only.
        with numerical_backend(strict=True) if cfg.model_key == "qwen" else nullcontext():
            value, _ = capture_last(
                model,
                [ids[i] for i in group],
                tokenizer.pad_token_id,
                layers=cfg.model.layers,
                width=cfg.model.hidden_dim,
                check_tuple=True,
            )
        batch_values.append(value)
        batch_indices.extend(group)
    batched = torch.cat(batch_values)[torch.tensor([batch_indices.index(i) for i in indices])]
    repeat = relative_errors(replay, initial)
    parity = relative_errors(batched, initial)
    bank_slots = torch.tensor([indices.index(i) for i in bank])
    means = (
        torch.stack([initial[bank_slots], batched[bank_slots]])
        .double()
        .reshape(2, 8, 2, cfg.model.layers, cfg.model.hidden_dim)
        .mean(2)
    )
    if torch.any(means.norm(dim=-1) == 0):
        raise RuntimeError("zero-norm smoke centroid cannot define cosine")
    unit = torch.nn.functional.normalize(means, dim=-1)
    cosines = torch.einsum("bplh,bqlh->blpq", unit, unit)
    smoke = {
        "fingerprint": fingerprint,
        "indices": indices,
        "bank_indices": bank,
        "execution_mode": cfg.model.execution_mode,
        "hook_tuple_relative_errors": tuple_errors,
        "repeatability_relative_errors": repeat.tolist(),
        "repeatability_bitwise_equal": bool(torch.equal(initial, replay)),
        "mixed_batch_relative_errors": parity.tolist(),
        "mixed_batch_is_production": cfg.model_key == "deepseek",
        "ordinary_cosine": cosines.tolist(),
        "maximum_absolute_cosine_difference_by_layer": (cosines[0] - cosines[1])
        .abs()
        .amax(dim=(1, 2))
        .tolist(),
        "passed": bool(
            repeat.max() <= cfg.capture.repeatability_relative_tolerance
            and (cfg.model_key == "qwen" or parity.max() <= cfg.capture.parity_relative_tolerance)
        ),
        "checked_at": time.time(),
    }
    torch.save(
        {
            "fingerprint": fingerprint,
            "indices": indices,
            "initial": initial,
            "repeated": replay,
            "batched": batched,
        },
        out / "smoke_vectors.pt",
    )
    write_json(out / "smoke.json", smoke)
    if not smoke["passed"]:
        raise RuntimeError(
            f"numerical smoke rejected: repeat={repeat.max().item()}, batch={parity.max().item()}"
        )
    return smoke


def projection(deadline, batch_seconds, remaining_batches, cfg, *, now=None) -> dict:
    now = time.time() if now is None else now
    if not math.isfinite(batch_seconds) or batch_seconds <= 0 or remaining_batches < 0:
        raise ValueError("invalid measured throughput")
    compute = remaining_batches * batch_seconds * cfg.capture.projection_margin
    reserve = cfg.capture.preservation_reserve_seconds
    return {
        "checked_at": now,
        "deadline_unix": deadline,
        "remaining_batches": remaining_batches,
        "slowest_batch_seconds": batch_seconds,
        "projection_margin": cfg.capture.projection_margin,
        "projected_compute_seconds": compute,
        "preservation_reserve_seconds": reserve,
        "remaining_wall_seconds": deadline - now,
        "passed": now + compute + reserve <= deadline,
    }


def checkpoint(out: Path, cfg) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/story_persona_crossmodel_artifacts.py"),
            "phase=checkpoint",
            f"output_dir={out}",
            f"model_key={cfg.model_key}",
        ],
        cwd=ROOT,
        check=True,
    )


class CheckpointPublisher:
    """One immutable snapshot upload may overlap capture; every failure propagates."""

    def __init__(self, out: Path, cfg):
        self.out, self.cfg = out, cfg
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def submit(self, count: int):
        if self.pending is not None:
            self.pending.result()
        parent = self.out.parent / f".{self.out.name}.checkpoints"
        parent.mkdir(exist_ok=True)
        snapshot = parent / f"chunks_{count:04d}_{time.time_ns()}"
        snapshot.mkdir()
        for path in sorted(self.out.rglob("*")):
            relative = path.relative_to(self.out)
            target = snapshot / relative
            if path.is_symlink():
                raise RuntimeError(f"unexpected capture symlink: {path}")
            if path.is_dir():
                target.mkdir(exist_ok=True)
            elif path.is_file():
                if path.name.endswith(".tmp"):
                    raise RuntimeError(f"unfinished file at checkpoint boundary: {path}")
                if relative.parts[0] == "chunks":
                    os.link(path, target)
                else:
                    shutil.copy2(path, target)
        self.pending = self.executor.submit(checkpoint, snapshot, self.cfg)

    def close(self):
        try:
            if self.pending is not None:
                self.pending.result()
        finally:
            self.executor.shutdown(wait=True)


def validate_store(out, fingerprint, batches, cfg, checksums):
    expected = {f"batch_{k:04d}.pt" for k in range(len(batches))}
    actual = {p.name for p in (out / "chunks").glob("*.pt")}
    if set(checksums) - expected or actual != set(checksums):
        raise RuntimeError("orphan, missing, or extra chunk/checksum entries")
    for k, indices in enumerate(batches):
        name = f"batch_{k:04d}.pt"
        if name in checksums:
            validate_chunk(
                out / "chunks" / name,
                fingerprint,
                indices,
                (len(indices), cfg.model.layers, cfg.model.hidden_dim),
                expected_sha256=checksums[name],
            )


def phase_capture(cfg, out):
    provenance = source_preflight()
    allocation_started, deadline = deadline_from_environment()
    attempt_started = time.time()
    if (
        cfg.capture.batch_rows != 8
        or cfg.expected_questions != 240
        or cfg.capture.timing_batches != 3
        or cfg.capture.projection_margin < 1.25
        or cfg.capture.preservation_reserve_seconds < 900
        or cfg.capture.checkpoint_every_chunks != 25
    ):
        raise ValueError("registered capture requires eight-row chunks and 240 questions")
    out.mkdir(parents=True, exist_ok=True)
    # A stale success marker must not survive a failed new attempt.
    (out / "capture_complete.json").unlink(missing_ok=True)
    write_json(
        out / "progress.json",
        {
            "stage": "loading",
            "checked_at": time.time(),
            "attempt_started_at": attempt_started,
            "model_key": cfg.model_key,
        },
    )
    prompts, questions, rows = read_inputs(cfg)
    print(f"[phase=capture/load] {cfg.model_key} checked_at={time.time()}", flush=True)
    model, tokenizer, runtime = load_model(cfg)
    ids = render_inputs(tokenizer, rows, cfg)
    batches = pack_batches(
        [len(x) for x in ids], cfg.capture.batch_rows, cfg.capture.padded_token_budget
    )
    source_paths = [
        Path(__file__),
        ROOT / "scripts/story_persona_qwen38_pilot.py",
        ROOT / "scripts/story_persona_crossmodel_artifacts.py",
        ROOT / "scripts/story_persona_fp8.py",
        ROOT / "src/explore_persona_space/analysis/extraction.py",
        ROOT / "configs/pilots/story_persona_crossmodel_capture.yaml",
    ]
    spec = {
        "model_key": cfg.model_key,
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "capture": OmegaConf.to_container(cfg.capture, resolve=True),
        "prompts": prompts,
        "question_ids": [q["id"] for q in questions],
        "inputs_sha256": digest(rows),
        "batches": batches,
        "layers": list(range(cfg.model.layers)),
        "dtype": "bfloat16",
        "position": "last_generation_prefix_token",
        "final_layer": "pre_final_norm",
        "source_hashes": {str(p.relative_to(ROOT)): file_digest(p) for p in source_paths},
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "accelerate": importlib.metadata.version("accelerate"),
        "transformers": cfg.model.transformers_version,
        "runtime": runtime,
    }
    fingerprint = digest(spec)
    if (out / "manifest.json").exists() and read_manifest(out / "manifest.json")[
        "fingerprint"
    ] != fingerprint:
        raise RuntimeError("output directory belongs to another recipe/runtime")
    checksums = read_checksums(out, fingerprint)
    (out / "chunks").mkdir(exist_ok=True)
    validate_store(out, fingerprint, batches, cfg, checksums)
    write_json(
        out / "manifest.json", {"fingerprint": fingerprint, "spec": spec, "provenance": provenance}
    )
    write_json(out / "rows.json", rows)
    write_json(out / "capture_chunks.json", {"fingerprint": fingerprint, "chunk_sha256": checksums})
    write_json(
        out / "progress.json",
        {"stage": "smoke", "fingerprint": fingerprint, "checked_at": time.time()},
    )
    smoke = numerical_smoke(model, tokenizer, ids, cfg, out, fingerprint)
    devices = list(range(torch.cuda.device_count()))
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    timing_indices = list(dict.fromkeys([0, len(batches) // 2, len(batches) - 1]))
    durations = []
    for k in timing_indices:
        for device in devices:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        capture_production(model, [ids[i] for i in batches[k]], tokenizer.pad_token_id, cfg)
        for device in devices:
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
    slowest = max(durations)
    gate = projection(deadline, slowest, len(batches) - len(checksums), cfg)
    smoke.update(
        timing_batch_indices=timing_indices,
        timing_seconds=durations,
        peak_allocated_bytes={str(d): torch.cuda.max_memory_allocated(d) for d in devices},
        peak_reserved_bytes={str(d): torch.cuda.max_memory_reserved(d) for d in devices},
        allocation_started_at=allocation_started,
        attempt_started_at=attempt_started,
        allocation_seconds_spent_before_production=time.time() - allocation_started,
        throughput_gate=gate,
    )
    write_json(out / "smoke.json", smoke)
    write_json(out / "throughput_projection.json", gate)
    if not gate["passed"]:
        raise RuntimeError(
            "measured capture/preservation projection exceeds remaining allocation window"
        )
    publisher = CheckpointPublisher(out, cfg)
    try:
        capture_batches(
            model,
            tokenizer,
            ids,
            batches,
            checksums,
            cfg,
            out,
            fingerprint,
            deadline,
            slowest,
            attempt_started,
            publisher,
        )
    finally:
        publisher.close()
    validate_store(out, fingerprint, batches, cfg, checksums)
    if len(checksums) != len(batches) or sorted(i for b in batches for i in b) != list(
        range(len(rows))
    ):
        raise RuntimeError("capture coverage is not exact")
    write_json(
        out / "capture_complete.json",
        {
            "fingerprint": fingerprint,
            "row_count": len(rows),
            "chunk_sha256": checksums,
            "checked_at": time.time(),
            "attempt_started_at": attempt_started,
            "provenance": provenance,
        },
    )
    write_json(
        out / "progress.json",
        {
            "stage": "capture_complete",
            "fingerprint": fingerprint,
            "completed_batches": len(checksums),
            "total_batches": len(batches),
            "checked_at": time.time(),
        },
    )


def capture_batches(
    model,
    tokenizer,
    ids,
    batches,
    checksums,
    cfg,
    out,
    fingerprint,
    deadline,
    slowest,
    attempt_started,
    publisher,
):
    for k, indices in enumerate(batches):
        name = f"batch_{k:04d}.pt"
        if name in checksums:
            continue
        gate = projection(deadline, slowest, len(batches) - len(checksums), cfg)
        write_json(out / "throughput_projection.json", gate)
        if not gate["passed"]:
            raise RuntimeError("remaining measured capture/preservation projection exceeded window")
        start = time.perf_counter()
        values, _ = capture_production(
            model, [ids[i] for i in indices], tokenizer.pad_token_id, cfg
        )
        path = out / "chunks" / name
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"fingerprint": fingerprint, "indices": indices, "vectors": values}, tmp)
        tmp.replace(path)
        validate_chunk(
            path, fingerprint, indices, (len(indices), cfg.model.layers, cfg.model.hidden_dim)
        )
        checksums[name] = file_digest(path)
        write_json(
            out / "capture_chunks.json", {"fingerprint": fingerprint, "chunk_sha256": checksums}
        )
        slowest = max(slowest, time.perf_counter() - start)
        progress = {
            "stage": "capture",
            "fingerprint": fingerprint,
            "checked_at": time.time(),
            "completed_batches": len(checksums),
            "total_batches": len(batches),
            "completed_rows": sum(len(batches[int(n[6:10])]) for n in checksums),
            "elapsed_seconds": time.time() - attempt_started,
        }
        write_json(out / "progress.json", progress)
        print(
            f"[capture] {cfg.model_key} chunks={len(checksums)}/{len(batches)} checked_at={time.time()}",
            flush=True,
        )
        if len(checksums) % cfg.capture.checkpoint_every_chunks == 0:
            publisher.submit(len(checksums))


@hydra.main(
    version_base=None,
    config_path="../configs/pilots",
    config_name="story_persona_crossmodel_capture",
)
def main(cfg: DictConfig):
    if cfg.phase != "capture":
        raise ValueError("this entrypoint implements capture only")
    out = Path(cfg.output_dir).resolve()
    try:
        phase_capture(cfg, out)
    except Exception as exc:
        # Preserve a diagnostic and propagate failure; wrapper owns partial-artifact publication.
        write_json(
            out / "capture_failure.json",
            {
                "checked_at": time.time(),
                "model_key": cfg.model_key,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
