"""A changed main engine cannot silently reuse previously sampled rollouts."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from explore_persona_space.analysis.workspace_capture import generate_rollouts
from explore_persona_space.analysis.workspace_execution import (
    generation_engine,
    validate_generation_engine,
)


@pytest.mark.parametrize("profile", ["eager16", "graphs32"])
def test_profiles_require_exact_measured_settings(profile):
    settings = generation_engine(profile)
    assert validate_generation_engine(settings) == settings
    for field, value in [("max_model_len", 65536), ("contexts_per_batch", 32)]:
        changed = copy.deepcopy(settings)
        changed[field] = value
        with pytest.raises(ValueError, match="differ"):
            validate_generation_engine(changed)
    settings["knobs"]["max_num_seqs"] = 128
    with pytest.raises(ValueError, match="differ"):
        validate_generation_engine(settings)
    with pytest.raises(ValueError, match="Unknown"):
        generation_engine("unmeasured")


def test_rollout_resume_binds_engine_and_readiness_without_mutating_prior_draws(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))
    calls = []

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

    class Engine:
        def generate(self, requests, parameters, **kwargs):
            calls.append(len(requests))
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[
                        SimpleNamespace(
                            token_ids=[3], text="answer", finish_reason="stop", stop_reason=None
                        )
                    ],
                )
                for _ in requests
            ]

    config = {
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
            "initial_max_new_tokens": 2048,
        }
    }
    prompts = [{"prompt_sha256": "a", "prompt": "prompt"}]
    contract = {"readiness_sha256": "a" * 64, "generation_engine": generation_engine("eager16")}
    args = Engine(), Tokenizer(), prompts, config, {"frozen": True}, tmp_path
    generate_rollouts(*args, execution_contract=contract)
    before = (tmp_path / "a.json").read_bytes()
    assert json.loads(before)["contract"]["execution"] == contract
    generate_rollouts(*args, execution_contract=contract)
    assert calls == [5]
    for changed in (
        {**contract, "readiness_sha256": "b" * 64},
        {**contract, "generation_engine": generation_engine("graphs32")},
        None,
    ):
        with pytest.raises(ValueError, match="Stale generation"):
            generate_rollouts(*args, execution_contract=changed)
        assert (tmp_path / "a.json").read_bytes() == before
    assert calls == [5]
