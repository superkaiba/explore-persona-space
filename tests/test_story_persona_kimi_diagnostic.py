"""CPU contract checks for the conditional diagnostic; no numerical GPU claim."""

import builtins
import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch
import transformers
from omegaconf import OmegaConf

from scripts import story_persona_kimi_runtime as runtime
from scripts.story_persona_crossmodel_artifacts import runtime_namespace
from scripts.story_persona_qwen38_pilot import digest


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Keep activation explicit and isolate output artifacts from every real run."""
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", raising=False)
    monkeypatch.delenv("EPS_STORY_PERSONA_KIMI_RUNTIME_MODE", raising=False)
    config = OmegaConf.create(
        {
            "model": {
                "runtime_mode": "baseline",
                "id": "moonshotai/Kimi-K2.6",
                "revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
                "min_gpu_bytes": 130000000000,
            },
            "output_dir": str(
                tmp_path / "analysis_tensors_issue2673_crossmodel_kimi_batch_invariant"
            ),
        }
    )
    return config


def enable(config, monkeypatch):
    """Prepare this test process's explicit flag pair without starting any engine."""
    config.model.runtime_mode = "batch_invariant"
    monkeypatch.setenv("EPS_STORY_PERSONA_KIMI_RUNTIME_MODE", "batch_invariant")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


def test_mode_requires_config_environment_fresh_process_and_separate_output(cfg, monkeypatch):
    assert runtime.diagnostic_engine_options(cfg) == {}
    enable(cfg, monkeypatch)
    assert runtime.diagnostic_engine_options(cfg) == {
        "attention_config": {"backend": "FLASH_ATTN_MLA"}
    }
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    with pytest.raises(RuntimeError, match="fresh process"):
        runtime.diagnostic_engine_options(cfg)
    monkeypatch.delitem(sys.modules, "vllm")
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
    with pytest.raises(ValueError, match="outside this diagnostic"):
        runtime.diagnostic_engine_options(cfg)
    monkeypatch.delenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK")
    cfg.output_dir = str(Path(cfg.output_dir).parent / "baseline")
    with pytest.raises(ValueError, match="distinct output"):
        runtime.diagnostic_engine_options(cfg)


def test_baseline_cannot_hide_a_flag_change(cfg, monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="silently enable"):
        runtime.diagnostic_engine_options(cfg)
    cfg.model.runtime_mode = "batch_invariant"
    with pytest.raises(ValueError, match="process environment"):
        runtime.diagnostic_engine_options(cfg)


def make_decoder():
    """Mirror the pinned module attributes with real PyTorch hook-bearing modules."""
    decoder = torch.nn.Module()
    decoder.config = SimpleNamespace(hidden_size=7168)
    decoder.norm = torch.nn.Identity()
    blocks = []
    quant_type = type("CompressedTensorsWNA16MarlinMoEMethod", (), {})
    for index in range(61):
        block = torch.nn.Module()
        block.input_layernorm = torch.nn.Identity()
        attention = torch.nn.Module()
        attention.attn_backend = SimpleNamespace(get_name=lambda: "FLASH_ATTN_MLA")
        block.attention = attention
        if index:
            quant = quant_type()
            quant.num_bits, quant.group_size, quant.marlin_input_dtype = 4, 32, None
            quant.kernel_backend = "Marlin"
            block.mlp = SimpleNamespace(
                experts=SimpleNamespace(
                    quant_method=quant,
                    router=SimpleNamespace(),
                    vllm_config=SimpleNamespace(
                        parallel_config=SimpleNamespace(
                            tensor_parallel_size=8, disable_custom_all_reduce=True
                        )
                    ),
                )
            )
        blocks.append(block)
    decoder.layers = torch.nn.ModuleList(blocks)
    return decoder


@pytest.mark.parametrize("worker_failure", [False, True])
def test_real_constructor_collects_all_rank_evidence_and_preserves_recipe(
    cfg, monkeypatch, tmp_path, worker_failure
):
    """Execute the actual constructor and worker callback with only external GPU/library seams."""
    enable(cfg, monkeypatch)
    vllm = ModuleType("vllm")
    vllm.__file__ = str(tmp_path / "vllm" / "__init__.py")
    vllm.__version__ = "0.19.1"
    files = [
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
    for name in files:
        path = Path(vllm.__file__).parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU fixture for " + name)
    envs = ModuleType("vllm.envs")
    envs.VLLM_BATCH_INVARIANT = True
    envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK = True
    vllm.envs = envs
    distributed = ModuleType("vllm.distributed")
    rank = [0]
    distributed.get_tensor_model_parallel_rank = lambda: rank[0]
    layers = ModuleType("vllm.model_executor.layers")
    layers.batch_invariant = SimpleNamespace(_batch_invariant_MODE=True)
    models = [
        SimpleNamespace(language_model=SimpleNamespace(model=make_decoder())) for _ in range(8)
    ]
    constructor_calls = []

    class Engine:
        def apply_model(self, func):
            if worker_failure and func is runtime.diagnostic_worker_evidence:
                raise RuntimeError("test worker diagnostic failure")
            results = []
            for index, model in enumerate(models):
                rank[0] = index
                results.append(func(model))
            return results

    def llm(
        model,
        revision,
        tokenizer_revision,
        trust_remote_code,
        tensor_parallel_size,
        dtype,
        distributed_executor_backend,
        enforce_eager,
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
        enable_chunked_prefill,
        enable_prefix_caching,
        gpu_memory_utilization,
        mm_encoder_tp_mode,
        limit_mm_per_prompt,
        seed,
        attention_config,
    ):
        constructor_calls.append(locals().copy())
        return Engine()

    def sampling_params(*, temperature, max_tokens, ignore_eos, seed):
        return SimpleNamespace(
            temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos, seed=seed
        )

    vllm.LLM, vllm.SamplingParams = llm, sampling_params
    original_import = builtins.__import__

    def import_boundary(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm" and "vllm" not in sys.modules:
            for key, value in {
                "vllm": vllm,
                "vllm.envs": envs,
                "vllm.distributed": distributed,
                "vllm.model_executor.layers": layers,
            }.items():
                monkeypatch.setitem(sys.modules, key, value)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_boundary)
    packages = {
        "vllm": "0.19.1",
        "transformers": "4.57.6",
        "torch": "2.10.0",
        "compressed-tensors": "0.15.0.1",
    }
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda package: packages[package])
    monkeypatch.setattr(
        torch.cuda, "device_count", create_autospec(torch.cuda.device_count, return_value=8)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        create_autospec(
            torch.cuda.get_device_properties,
            return_value=SimpleNamespace(total_memory=141000000000, name="NVIDIA H200"),
        ),
    )
    monkeypatch.setattr(
        torch.cuda.nccl,
        "version",
        create_autospec(torch.cuda.nccl.version, return_value=(2, 27, 3)),
    )
    text = SimpleNamespace(
        num_hidden_layers=61,
        hidden_size=7168,
        n_routed_experts=384,
        quantization_config={"quant_method": "compressed-tensors"},
        eos_token_id=1,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        create_autospec(
            transformers.AutoConfig.from_pretrained, return_value=SimpleNamespace(text_config=text)
        ),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        create_autospec(
            transformers.AutoTokenizer.from_pretrained,
            return_value=SimpleNamespace(bos_token_id=0, eos_token_id=1, pad_token_id=0),
        ),
    )
    if worker_failure:
        with pytest.raises(RuntimeError, match="test worker diagnostic failure"):
            runtime.KimiCapture(cfg)
        incomplete = json.loads((Path(cfg.output_dir) / "kimi_runtime_diagnostic.json").read_text())
        assert incomplete["status"] == "collecting_worker_evidence"
        assert "workers" not in incomplete
        return
    result = runtime.KimiCapture(cfg)
    call = constructor_calls[0]
    assert call["tensor_parallel_size"] == 8 and call["dtype"] == "bfloat16"
    assert call["attention_config"] == {"backend": "FLASH_ATTN_MLA"}
    assert call["max_num_seqs"] == 1 and call["enforce_eager"] is True
    assert call["enable_chunked_prefill"] is False and call["enable_prefix_caching"] is False
    evidence = json.loads((Path(cfg.output_dir) / "kimi_runtime_diagnostic.json").read_text())
    assert [record["rank"] for record in evidence["workers"]] == list(range(8))
    assert len(evidence["workers"][0]["source_sha256"]) == 11
    assert all(len(record["attention"]) == 61 for record in evidence["workers"])
    assert result.params.max_tokens == 1
    baseline = dict(result.runtime)
    del baseline["diagnostic_mode"], baseline["diagnostic_workers"]
    assert digest(result.runtime) != digest(baseline)
    changed = copy.deepcopy(evidence["workers"])
    changed[0]["quantization"][0]["num_bits"] = 8
    with pytest.raises(RuntimeError, match="recipe"):
        runtime.validate_diagnostic_workers(changed)
    changed = copy.deepcopy(evidence["workers"])
    changed[-1]["batch_invariant_initialized"] = False
    with pytest.raises(RuntimeError, match="differs among TP"):
        runtime.validate_diagnostic_workers(changed)


def test_publication_namespaces_cannot_replace_baseline(cfg, monkeypatch):
    out = Path(cfg.output_dir)
    with pytest.raises(ValueError, match="baseline namespace"):
        runtime_namespace(out, "kimi")
    enable(cfg, monkeypatch)
    manifest = {"spec": {"model": {"runtime_mode": "batch_invariant"}}}
    assert runtime_namespace(out, "kimi", manifest) == "/batch_invariant"
    assert runtime_namespace(out, "kimi") == "/batch_invariant"  # Failure before manifest exists.
    with pytest.raises(ValueError, match="manifest"):
        runtime_namespace(out, "kimi", {"spec": {"model": {}}})
    with pytest.raises(ValueError, match="distinct Kimi"):
        runtime_namespace(out, "deepseek")
    monkeypatch.delenv("EPS_STORY_PERSONA_KIMI_RUNTIME_MODE")
    assert runtime_namespace(out.parent / "baseline", "kimi", {"spec": {"model": {}}}) == ""
