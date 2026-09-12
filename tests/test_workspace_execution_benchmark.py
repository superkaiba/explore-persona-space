"""Actual benchmark wrappers retain frozen inputs and fail loudly without heavy models."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from explore_persona_space.analysis.workspace_runtime import content_sha256, file_sha256, save_json


@pytest.fixture
def execution_case(tmp_path, monkeypatch):
    from explore_persona_space.orchestrate import env

    monkeypatch.setattr(env, "load_dotenv", lambda: None)
    script = Path(__file__).resolve().parents[1] / "scripts/workspace_jr_execution_benchmark.py"
    module = runpy.run_path(str(script))
    config = {
        "seed": 42,
        "selection": {"primary": "fixture/model"},
        "models": {"primary": {"revision": "a" * 40, "source_layer": 1, "d_model": 6}},
        "generation": {
            "enable_thinking": False,
            "seeds": [42, 43, 44, 45, 46],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "initial_max_new_tokens": 2,
        },
        "sampling": {
            "pilot": {"train": 16, "validation": 16, "test": 16},
            "main_maximum": {"train": 16, "validation": 16, "test": 16},
        },
    }
    subsets = [
        f"{stage}_{split}"
        for stage in ("pilot", "main")
        for split in ("train", "validation", "test")
    ]
    prompts = {
        subset: [
            {"prompt_sha256": content_sha256([subset, i]), "prompt": str(i + offset + 2)}
            for i in range(16)
        ]
        for offset, subset in enumerate(subsets)
    }
    config_path, selection_path = tmp_path / "config.json", tmp_path / "selection.json"
    save_json(config_path, config)
    save_json(selection_path, {"subsets": prompts})
    identity = {
        "config_sha256": file_sha256(config_path),
        "selection_sha256": file_sha256(selection_path),
        "model_role": "primary",
        "versions": {},
        "code": {"git_commit": "b" * 40, "git_dirty": False},
    }
    batches = [
        {
            "subset": subset,
            "begin": 0,
            "contexts": [row["prompt_sha256"] for row in prompts[subset]],
            "token_lengths": [int(row["prompt"]) for row in prompts[subset]],
            "padded_attention_elements": 16
            * max(int(row["prompt"]) for row in prompts[subset]) ** 2,
        }
        for subset in subsets
    ]
    profile = {
        "config_sha256": identity["config_sha256"],
        "selection_sha256": identity["selection_sha256"],
        "batch_size": 16,
        "batches": batches,
        "main_outcomes_read": False,
        "worst_batch": max(batches, key=lambda row: row["padded_attention_elements"]),
    }
    profile_path = tmp_path / "profile.json"
    save_json(profile_path, profile)
    args = SimpleNamespace(
        role="primary",
        candidate="eager16",
        out=tmp_path / "out",
        config=config_path,
        selection=selection_path,
        audit=tmp_path / "audit.json",
        profile=profile_path,
        profile_sha256=file_sha256(profile_path),
    )
    args.out.mkdir()

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return list(range(int(messages[0]["content"])))

    tokenizer = Tokenizer()
    globals_ = module["capture_memory"].__globals__
    monkeypatch.setitem(
        globals_, "selected_prompts", lambda selection, audit, subset: prompts[subset]
    )
    monkeypatch.setattr(
        module["AutoTokenizer"], "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1234)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 5678)
    capture_calls, load_calls = [], []

    def load_native(config_arg, role, **kwargs):
        load_calls.append((config_arg, role, kwargs))
        model = torch.nn.Linear(1, 1).to(torch.bfloat16)
        return (
            model,
            tokenizer,
            SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager")),
        )

    def capture(text, token_ids, layer, pad):
        capture_calls.append((token_ids, layer, pad))
        return torch.arange(len(token_ids) * 6).reshape(len(token_ids), 6).bfloat16()

    monkeypatch.setitem(globals_, "load_native", load_native)
    monkeypatch.setitem(globals_, "capture_context_inputs", capture)
    return SimpleNamespace(
        module=module,
        globals=globals_,
        script=script,
        args=args,
        config=config,
        identity=identity,
        profile=profile,
        prompts=prompts,
        capture_calls=capture_calls,
        load_calls=load_calls,
    )


def test_capture_wrapper_uses_exact_largest_batch_and_labels_memory_only(execution_case):
    case = execution_case
    result = case.module["capture_memory"](case.args, case.config, case.identity)
    batch = case.profile["worst_batch"]
    assert result["batch"] == batch
    assert result["input_shape"] == [16, 6]
    assert result["model_dtype"] == "torch.bfloat16"
    assert result["attention_implementation"] == "eager"
    assert result["profile_sha256"] == file_sha256(case.args.profile)
    tokens, layer, pad = case.capture_calls[0]
    assert [len(row) for row in tokens] == batch["token_lengths"]
    assert (layer, pad) == (1, 0)
    assert case.load_calls[0][2] == {"device": "cuda:0", "dtype": torch.bfloat16}
    saved = torch.load(case.args.out / "captured_context_inputs.pt", weights_only=True)
    assert saved["prompt_ids"] == tokens
    assert saved["profile_batch"] == batch
    assert "not a main-run input checkpoint" in saved["scope"]


@pytest.mark.parametrize("mutation", ["identity", "contexts", "tokenization"])
def test_capture_wrapper_rejects_unfrozen_inputs(execution_case, mutation):
    case = execution_case
    profile = case.profile
    if mutation == "identity":
        profile["selection_sha256"] = "0" * 64
    elif mutation == "contexts":
        profile["worst_batch"]["contexts"][0] = "different_frozen_context"
    else:
        profile["worst_batch"]["token_lengths"][0] += 1
    save_json(case.args.profile, profile)
    case.args.profile_sha256 = file_sha256(case.args.profile)
    with pytest.raises(ValueError):
        case.module["capture_memory"](case.args, case.config, case.identity)
    assert not case.capture_calls
    assert not (case.args.out / "captured_context_inputs.pt").exists()


def test_capture_requires_the_independently_reviewed_profile_bytes(execution_case):
    case = execution_case
    case.args.profile.write_text(case.args.profile.read_text() + "\n")
    with pytest.raises(ValueError, match="reviewed file digest"):
        case.module["capture_memory"](case.args, case.config, case.identity)
    assert not case.load_calls


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_batch",
        "duplicate_batch",
        "unknown_subset",
        "unaligned_begin",
        "wrong_score",
        "wrong_worst",
        "short_lengths",
        "negative_length",
        "main_outcomes_seen",
    ],
)
def test_profile_ledger_cannot_hide_a_worse_or_unfrozen_batch(execution_case, mutation):
    case = execution_case
    profile = case.profile
    row = profile["batches"][0]
    if mutation == "missing_batch":
        profile["batches"].pop()
    elif mutation == "duplicate_batch":
        profile["batches"].append(row)
    elif mutation == "unknown_subset":
        row["subset"] = "unfrozen_subset"
    elif mutation == "unaligned_begin":
        row["begin"] = 1
    elif mutation == "wrong_score":
        row["padded_attention_elements"] += 1
    elif mutation == "wrong_worst":
        profile["worst_batch"] = row
    elif mutation == "short_lengths":
        row["token_lengths"].pop()
    elif mutation == "negative_length":
        row["token_lengths"][0] = -1
    else:
        profile["main_outcomes_read"] = True
    selection = json.loads(case.args.selection.read_text())
    with pytest.raises(ValueError):
        case.module["validate_profile"](profile, selection)


def test_native_capture_failure_is_persisted_and_reraised(execution_case, monkeypatch):
    case = execution_case
    output = case.args.out.parent / "failed_probe"
    monkeypatch.setitem(case.globals, "load_workspace_jr_config", lambda path: case.config)
    monkeypatch.setitem(case.globals, "run_identity", lambda *args: case.identity)

    def failed_capture(*args):
        raise torch.OutOfMemoryError("synthetic native capture allocation failure")

    monkeypatch.setitem(case.globals, "capture_context_inputs", failed_capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(case.script),
            "capture-memory",
            "--role",
            "primary",
            "--out",
            str(output),
            "--profile",
            str(case.args.profile),
            "--profile-sha256",
            case.args.profile_sha256,
            "--selection",
            str(case.args.selection),
            "--config",
            str(case.args.config),
        ],
    )
    with pytest.raises(torch.OutOfMemoryError, match="synthetic native capture"):
        case.module["main"]()
    saved = json.loads((output / "benchmark_failed.json").read_text())
    assert saved["status"] == "failed"
    assert saved["exception_type"] == "OutOfMemoryError"
    assert saved["identity"] == case.identity
    assert not (output / "benchmark_complete.json").exists()
    assert not (output / "captured_context_inputs.pt").exists()


@pytest.mark.parametrize("candidate", ["eager16", "graphs32"])
def test_generation_wrapper_retains_all_draws_and_times_cap_recovery(
    execution_case, monkeypatch, candidate
):
    case = execution_case
    case.args.candidate = candidate
    initializations, requests_seen = [], []

    class Engine:
        def __init__(self, **kwargs):
            initializations.append(kwargs)

        def generate(self, requests, parameters, **kwargs):
            requests_seen.extend(
                (row, vars(p)) for row, p in zip(requests, parameters, strict=True)
            )
            return [
                SimpleNamespace(
                    prompt_token_ids=row["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            token_ids=[7] * parameter.max_tokens,
                            text="synthetic completion",
                            finish_reason="length" if parameter.max_tokens == 2 else "stop",
                            stop_reason=None,
                        )
                    ],
                )
                for row, parameter in zip(requests, parameters, strict=True)
            ]

    # Keep the real engine factory and real generation/cap-recovery wrapper.
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace)
    )
    result = case.module["generation_throughput"](case.args, case.config, case.identity)
    assert len(initializations) == 1
    init = initializations[0]
    assert init["max_num_seqs"] == (16 if candidate == "eager16" else 32)
    assert init["enforce_eager"] is (candidate == "eager16")
    assert init["enable_prefix_caching"] is (candidate == "graphs32")
    assert init.get("language_model_only", False) is (candidate == "graphs32")
    assert init["revision"] == init["tokenizer_revision"] == "a" * 40
    assert init["max_model_len"] == 32768
    assert result["generation_status"]["n_contexts"] == 16
    assert result["generation_status"]["n_rollouts"] == 80
    assert result["generation_status"]["needs_cap_recovery"] is False
    assert result["final_tokens"] == 320
    assert result["prior_recovery_tokens"] == 160
    assert result["generated_tokens_per_second"] * result["generation_seconds"] == pytest.approx(
        480
    )
    assert len(requests_seen) == 160
    first = case.prompts["pilot_train"][0]["prompt_sha256"]
    saved = json.loads((case.args.out / "generations" / f"{first}.json").read_text())
    assert [row["seed"] for row in saved["rollouts"]] == case.config["generation"]["seeds"]
    assert len(saved["cap_recovery_history"]) == 5
    assert all(row["finish_reason"] == "length" for row in saved["cap_recovery_history"])
    assert all(row["finish_reason"] == "stop" for row in saved["rollouts"])
    assert saved["contract"]["generation"] == case.config["generation"]


def test_unresolved_cap_status_is_saved_before_wrapper_raises(execution_case, monkeypatch):
    case = execution_case
    from explore_persona_space.eval import generation

    monkeypatch.setattr(generation, "create_vllm_engine", lambda *args, **kwargs: object())

    def blocked(engine, tokenizer, prompts, config, identity, output):
        for row in prompts:
            save_json(
                output / f"{row['prompt_sha256']}.json", {"rollouts": [{"token_ids": [1, 2]}] * 5}
            )
        return {"needs_cap_recovery": True, "status": "context_window_cap_blocked"}

    monkeypatch.setitem(case.globals, "generate_rollouts", blocked)
    with pytest.raises(ValueError, match="not resolved truncation"):
        case.module["generation_throughput"](case.args, case.config, case.identity)
    saved = json.loads((case.args.out / "generation_benchmark.json").read_text())
    assert saved["generation_status"]["needs_cap_recovery"] is True
    assert saved["generation_status"]["status"] == "context_window_cap_blocked"
