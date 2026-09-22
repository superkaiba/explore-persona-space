"""Native INT4 Kimi capture through the pinned vLLM eager singleton runtime.

vLLM carries the residual and branch output separately. Capture their BF16 sum,
and independently check it against the next fused RMSNorm's returned residual.
No generated token is used in an activation or a behavioral measurement.
"""

from __future__ import annotations

from functools import partial
import hashlib
import importlib.metadata
import time

import torch


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


def numerical_smoke(model, ids, cfg, out, fingerprint):
    from scripts.story_persona_crossmodel_capture import relative_errors
    from scripts.story_persona_qwen38_pilot import write_json

    extrema = [
        min(range(len(ids)), key=lambda i: len(ids[i])),
        max(range(len(ids)), key=lambda i: len(ids[i])),
    ]
    indices = list(dict.fromkeys(extrema + [p * 240 + q for p in range(9) for q in (0, 1)]))
    values, evidence, rank_evidence = [], {}, {}
    for index in indices:
        value, error = model.capture([ids[index]], check_tuple=True)
        values.append(value)
        evidence[str(index)] = error
        rank_evidence[str(index)] = model.last_evidence
        write_json(
            out / "progress.json",
            {
                "stage": "smoke",
                "fingerprint": fingerprint,
                "smoke_singletons_completed": len(values),
                "checked_at": time.time(),
            },
        )
        print(f"[kimi-smoke] singletons={len(values)}/{len(indices)}", flush=True)
    initial = torch.cat(values)
    replay = torch.cat([model.capture([ids[i]])[0] for i in reversed(indices)]).flip(0)
    error = relative_errors(initial, replay)
    smoke = {
        "fingerprint": fingerprint,
        "indices": indices,
        "execution_mode": "unpadded_singleton",
        "repeatability_relative_errors": error.tolist(),
        "repeatability_bitwise_equal": bool(torch.equal(initial, replay)),
        "hook_norm_relative_errors": evidence,
        "all_rank_bitwise_agreement": True,
        "rank_evidence": rank_evidence,
        "mixed_batch_is_production": False,
        "mixed_batch_diagnostic": "not run; singleton-only runtime",
        "passed": bool(error.max() <= cfg.capture.repeatability_relative_tolerance),
        "checked_at": time.time(),
    }
    torch.save(
        {"fingerprint": fingerprint, "indices": indices, "initial": initial, "repeated": replay},
        out / "smoke_vectors.pt",
    )
    write_json(out / "smoke.json", smoke)
    if not smoke["passed"]:
        raise RuntimeError(f"Kimi singleton repeatability rejected: {error.max().item()}")
    return smoke
