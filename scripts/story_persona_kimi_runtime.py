"""Native INT4 Kimi capture through the pinned vLLM eager singleton runtime.

vLLM carries the residual and branch output separately. Capture their BF16 sum,
and independently check it against the next fused RMSNorm's returned residual.
No generated token is used in an activation or a behavioral measurement.
"""

from __future__ import annotations

from functools import partial
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import torch


def diagnostic_engine_options(cfg):
    """Require explicit process/config agreement before importing a fresh vLLM engine."""
    mode = cfg.model.get("runtime_mode", "baseline")
    if mode not in {"baseline", "batch_invariant"}:
        raise ValueError("unknown Kimi runtime diagnostic")
    if os.environ.get("EPS_STORY_PERSONA_KIMI_RUNTIME_MODE", "baseline") != mode:
        raise ValueError("Kimi runtime mode differs from the process environment")
    if mode == "baseline":
        if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
            raise ValueError("baseline cannot silently enable batch invariance")
        return {}
    if "vllm" in sys.modules:
        raise RuntimeError("batch-invariant diagnostic requires a fresh process")
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("diagnostic must start with VLLM_BATCH_INVARIANT=1")
    if os.environ.get("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1") != "1":
        raise ValueError("grouped-topk changes are outside this diagnostic")
    if Path(cfg.output_dir).name != "analysis_tensors_issue2673_crossmodel_kimi_batch_invariant":
        raise ValueError("diagnostic requires its distinct output namespace")
    return {"attention_config": {"backend": "FLASH_ATTN_MLA"}}


def diagnostic_worker_evidence(model):
    """Read effective attention, native INT4, router and runtime/source evidence per TP rank."""
    import vllm
    import vllm.envs as envs
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers import batch_invariant

    decoder = model.language_model.model
    experts = [block.mlp.experts for block in decoder.layers[1:]]
    parallel = experts[0].vllm_config.parallel_config
    attention = {
        name: module.attn_backend.get_name()
        for name, module in decoder.named_modules()
        if hasattr(module, "attn_backend")
    }
    quantization = [
        {
            "layer": index + 1,
            "method": type(expert.quant_method).__name__,
            "num_bits": expert.quant_method.num_bits,
            "group_size": expert.quant_method.group_size,
            "input_dtype": str(expert.quant_method.marlin_input_dtype),
            "kernel_backend": expert.quant_method.kernel_backend,
            "router": type(expert.router).__module__ + "." + type(expert.router).__qualname__,
        }
        for index, expert in enumerate(experts)
    ]
    source_names = [
        "envs.py",
        "config/parallel.py",
        "model_executor/layers/batch_invariant.py",
        "model_executor/layers/layernorm.py",
        "model_executor/layers/attention/mla_attention.py",
        "model_executor/layers/fused_moe/layer.py",
        "model_executor/layers/fused_moe/fused_marlin_moe.py",
        "model_executor/layers/fused_moe/router/grouped_topk_router.py",
        "model_executor/layers/fused_moe/router/fused_topk_bias_router.py",
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py",
        "v1/attention/backends/mla/flashattn_mla.py",
    ]
    root = Path(vllm.__file__).parent
    files = {"vllm/" + name: root / name for name in source_names}
    # Hash only loaded extension libraries, without importing an alternative kernel.
    files.update(
        {
            name: Path(sys.modules[name].__file__)
            for name in ("vllm._C", "vllm._moe_C")
            if name in sys.modules
        }
    )
    sources = {}
    for name, path in files.items():
        with path.open("rb") as handle:
            sources[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    env_names = (
        "VLLM_BATCH_INVARIANT",
        "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUBLASLT_WORKSPACE_SIZE",
        "NCCL_LAUNCH_MODE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_NVLS_ENABLE",
        "NCCL_P2P_NET_DISABLE",
        "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_PROTO",
        "NCCL_ALGO",
        "NCCL_NTHREADS",
        "NCCL_LL128_NTHREADS",
    )
    return {
        "rank": get_tensor_model_parallel_rank(),
        "batch_invariant_flag": bool(envs.VLLM_BATCH_INVARIANT),
        "batch_invariant_initialized": bool(batch_invariant._batch_invariant_MODE),
        "fused_grouped_topk": bool(envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK),
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "disable_custom_all_reduce": parallel.disable_custom_all_reduce,
        "attention": attention,
        "quantization": quantization,
        "environment": {name: os.environ.get(name) for name in env_names},
        "torch_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "torch_tf32": torch.backends.cuda.matmul.allow_tf32,
        "nccl_version": torch.cuda.nccl.version(),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "compressed-tensors", "transformers")
        },
        "source_sha256": sources,
    }


def validate_diagnostic_workers(records):
    """Require all eight workers to attest the same requested native-INT4 runtime."""
    if sorted(record["rank"] for record in records) != list(range(8)):
        raise RuntimeError("diagnostic lacks eight distinct worker receipts")
    reference = {key: value for key, value in records[0].items() if key != "rank"}
    for record in records:
        if {key: value for key, value in record.items() if key != "rank"} != reference:
            raise RuntimeError("diagnostic runtime evidence differs among TP workers")
        if (
            record["batch_invariant_flag"] is not True
            or record["batch_invariant_initialized"] is not True
            or record["fused_grouped_topk"] is not True
            or record["tensor_parallel_size"] != 8
            or record["disable_custom_all_reduce"] is not True
            or len(record["attention"]) != 61
            or set(record["attention"].values()) != {"FLASH_ATTN_MLA"}
            or len(record["quantization"]) != 60
            or any(
                item["method"] != "CompressedTensorsWNA16MarlinMoEMethod"
                or item["num_bits"] != 4
                or item["group_size"] != 32
                or item["kernel_backend"] != "Marlin"
                or item["input_dtype"] not in {"None", "torch.bfloat16"}
                for item in record["quantization"]
            )
        ):
            raise RuntimeError("diagnostic changed the required runtime/backend/native INT4 recipe")


def residual_sum(output):
    """Materialize the decoder's post-block, pre-final-norm residual."""
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("expected vLLM (branch, residual) pair")
    branch, residual = output
    if branch.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise RuntimeError("Kimi capture requires BF16 residual arithmetic")
    if branch.shape != residual.shape or branch.ndim != 2:
        raise RuntimeError("unexpected branch/residual shapes")
    return branch[-1] + residual[-1]


def install_hooks(model):
    """Runs independently inside each TP worker, after startup profiling."""
    decoder = model.language_model.model
    if len(decoder.layers) != 61 or decoder.config.hidden_size != 7168:
        raise RuntimeError("Kimi decoder architecture changed")
    if hasattr(model, "_eps_capture"):
        raise RuntimeError("duplicate capture instrumentation")
    state = {"active": False, "vectors": {}, "norm_errors": {}, "handles": []}
    model._eps_capture = state

    def block_hook(index):
        def capture(_module, inputs, kwargs, output):
            if not state["active"]:
                return
            positions = kwargs.get("positions", inputs[0] if inputs else None)
            length = state["length"]
            if (
                index in state["vectors"]
                or positions is None
                or tuple(positions.shape) != (length,)
                or not torch.equal(positions, torch.arange(length, device=positions.device))
                or tuple(output[0].shape) != (length, 7168)
            ):
                raise RuntimeError("duplicate forward, padding, cache reuse or chunked prefill")
            state["vectors"][index] = residual_sum(output).detach().cpu().clone()

        return capture

    def norm_hook(previous):
        def check(_module, _inputs, output):
            if not state["active"] or not state["check_norm"]:
                return
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("fused RMSNorm did not return its updated residual")
            reference = output[1][-1].detach().float().cpu()
            actual = state["vectors"][previous].float()
            error = ((reference - actual).norm() / reference.norm().clamp_min(1e-12)).item()
            state["norm_errors"][previous] = error
            if not torch.isfinite(reference).all() or error > 1e-5:
                raise RuntimeError(f"post-block / fused-norm residual mismatch: {error}")

        return check

    for index, block in enumerate(decoder.layers):
        state["handles"].append(block.register_forward_hook(block_hook(index), with_kwargs=True))
        if index:
            state["handles"].append(
                block.input_layernorm.register_forward_hook(norm_hook(index - 1))
            )
    state["handles"].append(decoder.norm.register_forward_hook(norm_hook(60)))
    return {"layers": len(decoder.layers), "width": decoder.config.hidden_size}


def arm_capture(model, *, length, check_norm):
    state = model._eps_capture
    if state["active"]:
        raise RuntimeError("previous capture was not drained")
    state.update(active=True, length=length, check_norm=check_norm, vectors={}, norm_errors={})


def drain_capture(model):
    from vllm.distributed import get_tensor_model_parallel_rank

    state = model._eps_capture
    state["active"] = False
    if set(state["vectors"]) != set(range(61)):
        raise RuntimeError("missing decoder captures")
    if state["check_norm"] and set(state["norm_errors"]) != set(range(61)):
        raise RuntimeError("missing independent fused-norm checks")
    values = torch.stack([state["vectors"][i] for i in range(61)])
    if values.shape != (61, 7168) or not torch.isfinite(values).all():
        raise RuntimeError("invalid Kimi vectors")
    data = values.view(torch.uint16).numpy().tobytes()
    rank = get_tensor_model_parallel_rank()
    result = {
        "rank": rank,
        "sha256": hashlib.sha256(data).hexdigest(),
        "data": data if rank == 0 else None,
        "norm_errors": state["norm_errors"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    state["vectors"] = {}
    return result


class KimiCapture:
    """Small adapter preserving the existing chunk/checkpoint capture contract."""

    def __init__(self, cfg):
        """Build one pinned singleton engine, with an explicitly selected optional diagnostic."""
        engine_options = diagnostic_engine_options(cfg)
        import transformers
        import vllm
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams

        required = {
            "vllm": "0.19.1",
            "transformers": "4.57.6",
            "torch": "2.10.0",
            "compressed-tensors": "0.15.0.1",
        }
        for package, expected in required.items():
            if importlib.metadata.version(package).split("+")[0] != expected:
                raise RuntimeError(f"unpinned Kimi runtime package: {package}")
        if torch.cuda.device_count() != 8:
            raise RuntimeError("Kimi requires eight visible H200 GPUs")
        for index in range(8):
            prop = torch.cuda.get_device_properties(index)
            if prop.total_memory < cfg.model.min_gpu_bytes or "H200" not in prop.name:
                raise RuntimeError("wrong GPU model or insufficient physical HBM")
        config = AutoConfig.from_pretrained(
            cfg.model.id, revision=cfg.model.revision, trust_remote_code=True
        )
        text = config.text_config
        if (text.num_hidden_layers, text.hidden_size, text.n_routed_experts) != (61, 7168, 384):
            raise RuntimeError("pinned Kimi configuration changed")
        if text.quantization_config["quant_method"] != "compressed-tensors":
            raise RuntimeError("Kimi native INT4 configuration missing")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.id, revision=cfg.model.revision, trust_remote_code=True
        )
        self.engine = LLM(
            model=cfg.model.id,
            revision=cfg.model.revision,
            tokenizer_revision=cfg.model.revision,
            trust_remote_code=True,
            tensor_parallel_size=8,
            dtype="bfloat16",
            distributed_executor_backend="mp",
            enforce_eager=True,
            max_model_len=2048,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.85,
            mm_encoder_tp_mode="data",
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=0,
            **engine_options,
        )
        diagnostic = None
        if engine_options:
            from scripts.story_persona_qwen38_pilot import write_json

            write_json(
                Path(cfg.output_dir) / "kimi_runtime_diagnostic.json",
                {
                    "status": "collecting_worker_evidence",
                    "requested_mode": "batch_invariant",
                    "attention_config": engine_options["attention_config"],
                    "source_sha": os.environ.get("EPS_STORY_PERSONA_SOURCE_SHA"),
                },
            )
            diagnostic = self.engine.apply_model(diagnostic_worker_evidence)
            write_json(
                Path(cfg.output_dir) / "kimi_runtime_diagnostic.json",
                {"status": "worker_evidence_collected", "workers": diagnostic},
            )
            validate_diagnostic_workers(diagnostic)
            print(
                "[kimi-runtime-diagnostic] batch_invariant FLASH_ATTN_MLA verified on TP8",
                flush=True,
            )
        installed = self.engine.apply_model(install_hooks)
        if len(installed) != 8:
            raise RuntimeError("expected eight tensor-parallel worker acknowledgments")
        self.params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, seed=0)
        self.runtime = {
            "weight_format": "native compressed-tensors INT4; BF16 residual stream",
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "quantization": text.quantization_config,
            "tensor_parallel_size": 8,
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "max_num_seqs": 1,
            "tokenizer_bos_id": self.tokenizer.bos_token_id,
            "tokenizer_eos_id": self.tokenizer.eos_token_id,
            "tokenizer_pad_id": self.tokenizer.pad_token_id,
            "model_eos_id": text.eos_token_id,
            "thinking": False,
            "default_system_message": "omitted",
            "capture": "BF16 branch + residual, last prefill position",
            "validation": "next fused RMSNorm returned residual; all TP rank checksums",
        }
        if diagnostic is not None:
            self.runtime["diagnostic_mode"] = "batch_invariant"
            self.runtime["diagnostic_workers"] = diagnostic
        self.last_evidence = []

    def capture(self, ids_rows, *, check_tuple=False):
        values, errors = [], {}
        self.last_evidence = []
        for row, ids in enumerate(ids_rows):
            if not ids or len(ids) > 2048:
                raise ValueError("empty or overlong Kimi context")
            self.engine.apply_model(partial(arm_capture, length=len(ids), check_norm=check_tuple))
            result = self.engine.generate([{"prompt_token_ids": ids}], self.params, use_tqdm=False)
            if (
                len(result) != 1
                or result[0].prompt_token_ids != ids
                or len(result[0].outputs[0].token_ids) != 1
            ):
                raise RuntimeError("generation request did not preserve the singleton prefix")
            ranks = self.engine.apply_model(drain_capture)
            if (
                sorted(x["rank"] for x in ranks) != list(range(8))
                or len({x["sha256"] for x in ranks}) != 1
            ):
                raise RuntimeError("tensor-parallel residual replicas disagree")
            data = next(x["data"] for x in ranks if x["rank"] == 0)
            values.append(
                torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone().reshape(61, 7168)
            )
            errors[str(row)] = {str(x["rank"]): x["norm_errors"] for x in ranks}
            self.last_evidence.append([{k: v for k, v in x.items() if k != "data"} for x in ranks])
        return torch.stack(values), errors


def render_kimi(tokenizer, rows, cfg):
    from scripts.story_persona_qwen38_pilot import digest

    result = []
    for row in rows:
        kwargs = dict(add_generation_prompt=True, thinking=False)
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, **kwargs)
        if ids != tokenizer(text, add_special_tokens=False)["input_ids"] or not ids:
            raise RuntimeError("Kimi render/tokenization disagreement")
        if not text.endswith("<|im_assistant|>assistant<|im_middle|><think></think>"):
            raise RuntimeError("Kimi instant-mode assistant boundary changed")
        if len(ids) > cfg.capture.max_context_tokens:
            raise ValueError("Kimi truncation forbidden")
        row.update(
            rendered_prefix=text,
            token_count=len(ids),
            prefix_sha256=digest(ids),
            final_token_id=ids[-1],
        )
        result.append(ids)
    return result


def _instrumentation_diagnostic(model, ids, index, cfg, out, fingerprint):
    """Separate adjacent repeatability from changes in norm-check instrumentation."""
    from scripts.story_persona_crossmodel_capture import relative_errors
    from scripts.story_persona_qwen38_pilot import write_json

    flags = [False, False, False, True, True, False]
    values, evidence = [], []
    for step, flag in enumerate(flags):
        value, norm_errors = model.capture([ids[index]], check_tuple=flag)
        values.append(value)
        evidence.append({"norm_errors": norm_errors, "ranks": model.last_evidence})
        write_json(
            out / "progress.json",
            {
                "stage": "smoke_instrumentation",
                "fingerprint": fingerprint,
                "smoke_singletons_completed": step + 1,
                "checked_at": time.time(),
            },
        )
        print(f"[kimi-smoke-instrumentation] singleton={step + 1}/6 check_norm={flag}", flush=True)
    values = torch.cat(values)
    pairs = {
        "production_repeat_1": (0, 1),
        "production_repeat_2": (1, 2),
        "production_return": (2, 5),
        "checked_repeat": (3, 4),
        "instrumentation_on": (2, 3),
        "instrumentation_off": (4, 5),
    }
    errors = {name: relative_errors(values[a], values[b]) for name, (a, b) in pairs.items()}
    result = {
        "fingerprint": fingerprint,
        "index": index,
        "check_norm": flags,
        "pairs": pairs,
        "relative_errors": {name: error.tolist() for name, error in errors.items()},
        "passed": all(
            bool(
                torch.isfinite(error).all()
                and error.max() <= cfg.capture.repeatability_relative_tolerance
            )
            for error in errors.values()
        ),
        "evidence": evidence,
        "checked_at": time.time(),
    }
    torch.save(
        {"fingerprint": fingerprint, "index": index, "check_norm": flags, "values": values},
        out / "smoke_instrumentation_vectors.pt",
    )
    write_json(out / "smoke_instrumentation.json", result)
    if not result["passed"]:
        maxima = {name: error.max().item() for name, error in errors.items()}
        raise RuntimeError(f"Kimi controlled instrumentation diagnostic rejected: {maxima}")
    return result


def numerical_smoke(model, ids, cfg, out, fingerprint):
    """Gate production-style replay and norm instrumentation independently."""
    from scripts.story_persona_crossmodel_capture import relative_errors
    from scripts.story_persona_qwen38_pilot import write_json

    extrema = [
        min(range(len(ids)), key=lambda i: len(ids[i])),
        max(range(len(ids)), key=lambda i: len(ids[i])),
    ]
    indices = list(dict.fromkeys(extrema + [p * 240 + q for p in range(9) for q in (0, 1)]))
    # default:1 was the adjacent same-input counterexample in the failed live smoke.
    diagnostic = _instrumentation_diagnostic(model, ids, 8 * 240 + 1, cfg, out, fingerprint)
    phases, norm_evidence, rank_evidence = {}, {}, {}
    for phase, order, check_norm in (
        ("initial", indices, False),
        ("repeated", list(reversed(indices)), False),
        ("norm_checked", indices, True),
    ):
        values, norms, ranks = [], {}, {}
        for index in order:
            value, error = model.capture([ids[index]], check_tuple=check_norm)
            values.append(value)
            norms[str(index)] = error
            ranks[str(index)] = model.last_evidence
            write_json(
                out / "progress.json",
                {
                    "stage": "smoke",
                    "smoke_phase": phase,
                    "check_norm": check_norm,
                    "fingerprint": fingerprint,
                    "smoke_singletons_completed": len(values),
                    "checked_at": time.time(),
                },
            )
            print(f"[kimi-smoke] phase={phase} singletons={len(values)}/{len(order)}", flush=True)
        phases[phase] = torch.cat(values)
        if phase == "repeated":
            phases[phase] = phases[phase].flip(0)
        norm_evidence[phase], rank_evidence[phase] = norms, ranks
        # Persist each phase before another starts. A new engine must rerun these checks.
        torch.save(
            {
                "fingerprint": fingerprint,
                "indices": indices,
                "execution_order": order,
                "check_norm": check_norm,
                "values": phases[phase],
                "norm_errors": norms,
                "rank_evidence": ranks,
            },
            out / f"smoke_{phase}.pt",
        )
    initial, replay = phases["initial"], phases["repeated"]
    error = relative_errors(initial, replay)
    instrumentation_error = relative_errors(initial, phases["norm_checked"])
    smoke = {
        "fingerprint": fingerprint,
        "indices": indices,
        "execution_mode": "unpadded_singleton",
        "check_norm_by_phase": {"initial": False, "repeated": False, "norm_checked": True},
        "controlled_instrumentation_diagnostic_passed": diagnostic["passed"],
        "repeatability_relative_errors": error.tolist(),
        "repeatability_bitwise_equal": bool(torch.equal(initial, replay)),
        "instrumentation_relative_errors": instrumentation_error.tolist(),
        "hook_norm_relative_errors": norm_evidence["norm_checked"],
        "all_rank_bitwise_agreement": True,
        "rank_evidence": rank_evidence["initial"],
        "rank_evidence_by_phase": rank_evidence,
        "mixed_batch_is_production": False,
        "mixed_batch_diagnostic": "not run; singleton-only runtime",
        "passed": bool(
            torch.isfinite(error).all()
            and torch.isfinite(instrumentation_error).all()
            and error.max() <= cfg.capture.repeatability_relative_tolerance
            and instrumentation_error.max() <= cfg.capture.repeatability_relative_tolerance
        ),
        "checked_at": time.time(),
    }
    torch.save(
        {"fingerprint": fingerprint, "indices": indices, **phases},
        out / "smoke_vectors.pt",
    )
    write_json(out / "smoke.json", smoke)
    if not smoke["passed"]:
        raise RuntimeError(
            f"Kimi numerical smoke rejected: repeat={error.max().item()}, "
            f"instrumentation={instrumentation_error.max().item()}"
        )
    return smoke
