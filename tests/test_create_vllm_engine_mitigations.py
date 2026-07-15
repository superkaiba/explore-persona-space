"""Env-gated vLLM hang-mitigation knobs in ``eval/generation.py`` (#1324).

Pins the #1324 contract: ``vllm_hang_mitigation_overrides`` is default-OFF
(byte-identical ``LLM(...)`` kwargs when the env knobs are unset and
``hang_mitigations`` is left at ``None`` — the #1092 property), engages per
knob via ``EPM_VLLM_ENFORCE_EAGER`` / ``EPM_VLLM_DISABLE_PREFIX_CACHING``
(truthy set ``{"1", "true", "True"}`` only), honors the tri-state
``hang_mitigations`` param on ``create_vllm_engine`` (True = both on,
False = env-ignoring opt-out, None = env decides), lets explicit caller
kwargs win with no double-pass ``TypeError``, and threads the env-only
resolver into the three sibling ``LLM(`` sites in the module.

Boundary fakes follow the pinned pattern
(``tests/test_issue1092_round8.py::_install_fake_vllm``): a fake ``vllm``
module injected via ``monkeypatch.setitem(sys.modules, ...)`` so the
function-local ``from vllm import ...`` resolves to a recording class —
CPU-only, GPU-free, no real vllm import. The tokenizer boundary for the
sibling-site test is a def-mirroring fake ``transformers`` module (fakes at
the external vllm/tokenizer boundary only, signature-conformant per
code-style § one-production-body-test).
"""

import sys
import types
from typing import ClassVar

import pytest

from explore_persona_space.eval import generation

# The exact kwargs the pre-#1324 factory passed to vllm.LLM for
# create_vllm_engine("m") with all env knobs unset — the durability pin.
BASELINE_FACTORY_KWARGS = {
    "model": "m",
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.60,
    "max_model_len": 2048,
    "max_num_seqs": 64,
    "seed": 42,
}

MITIGATION_KEYS = ("enforce_eager", "enable_prefix_caching")


class _FakeCompletion:
    def __init__(self, text: str = "fake completion"):
        self.text = text


class _FakeRequestOutput:
    def __init__(self):
        self.outputs = [_FakeCompletion()]


class _FakeVllmLLM:
    """Recording stand-in for vllm.LLM (the pinned #1092 fake-vllm shape)."""

    instances: ClassVar[list[dict]] = []

    def __init__(self, **kwargs):
        _FakeVllmLLM.instances.append(kwargs)
        self.kwargs = kwargs

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return [_FakeRequestOutput() for _ in prompts]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_vllm(monkeypatch):
    fake = types.ModuleType("vllm")
    fake.LLM = _FakeVllmLLM
    fake.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)
    _FakeVllmLLM.instances.clear()


class _FakeTokenizer:
    """Signature-conformant fake for the two tokenizer methods the module calls."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return " ".join(m["content"] for m in messages)


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_path, *, trust_remote_code=False, token=None):
        # Mirrors the exact call shape at the three sibling sites:
        # AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=...).
        return _FakeTokenizer()


def _install_fake_transformers(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with the three engine-affecting env vars unset."""
    for var in (
        "EPM_VLLM_ENFORCE_EAGER",
        "EPM_VLLM_DISABLE_PREFIX_CACHING",
        "VLLM_GPU_MEM_UTIL",
    ):
        monkeypatch.delenv(var, raising=False)
    _FakeVllmLLM.instances.clear()


# ── §3.1 byte-identical when off (the durability pin) ──────────────────────


def test_byte_identical_kwargs_when_unset(monkeypatch):
    _install_fake_vllm(monkeypatch)
    generation.create_vllm_engine("m")
    assert _FakeVllmLLM.instances == [BASELINE_FACTORY_KWARGS]


# ── helper semantics ────────────────────────────────────────────────────────


def test_helper_default_off(monkeypatch):
    assert generation.vllm_hang_mitigation_overrides(None) == {}
    assert generation.vllm_hang_mitigation_overrides() == {}
    # False suppresses even with both env knobs set (comparability opt-out).
    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "1")
    monkeypatch.setenv("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")
    assert generation.vllm_hang_mitigation_overrides(False) == {}


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_env_engages_each_knob_independently(monkeypatch, value):
    _install_fake_vllm(monkeypatch)

    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", value)
    assert generation.vllm_hang_mitigation_overrides() == {"enforce_eager": True}
    generation.create_vllm_engine("m")
    captured = _FakeVllmLLM.instances[-1]
    assert captured["enforce_eager"] is True
    assert "enable_prefix_caching" not in captured
    monkeypatch.delenv("EPM_VLLM_ENFORCE_EAGER")

    monkeypatch.setenv("EPM_VLLM_DISABLE_PREFIX_CACHING", value)
    assert generation.vllm_hang_mitigation_overrides() == {"enable_prefix_caching": False}
    generation.create_vllm_engine("m")
    captured = _FakeVllmLLM.instances[-1]
    assert captured["enable_prefix_caching"] is False
    assert "enforce_eager" not in captured

    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", value)
    assert generation.vllm_hang_mitigation_overrides() == {
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    generation.create_vllm_engine("m")
    captured = _FakeVllmLLM.instances[-1]
    assert captured["enforce_eager"] is True
    assert captured["enable_prefix_caching"] is False


@pytest.mark.parametrize("value", ["0", "", "yes", "on"])
def test_non_truthy_env_values_stay_off(monkeypatch, value):
    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", value)
    monkeypatch.setenv("EPM_VLLM_DISABLE_PREFIX_CACHING", value)
    assert generation.vllm_hang_mitigation_overrides() == {}


# ── §3.3 param semantics ────────────────────────────────────────────────────


def test_param_true_engages_both(monkeypatch):
    _install_fake_vllm(monkeypatch)
    assert generation.vllm_hang_mitigation_overrides(True) == {
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    generation.create_vllm_engine("m", hang_mitigations=True)
    assert _FakeVllmLLM.instances == [
        {**BASELINE_FACTORY_KWARGS, "enforce_eager": True, "enable_prefix_caching": False}
    ]


def test_param_false_suppresses_env(monkeypatch):
    _install_fake_vllm(monkeypatch)
    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "1")
    monkeypatch.setenv("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")
    generation.create_vllm_engine("m", hang_mitigations=False)
    assert _FakeVllmLLM.instances == [BASELINE_FACTORY_KWARGS]


# ── §3.4 explicit caller kwargs win, no TypeError ───────────────────────────


def test_explicit_kwarg_wins_no_typeerror(monkeypatch):
    _install_fake_vllm(monkeypatch)

    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "1")
    generation.create_vllm_engine("m", enforce_eager=False)
    assert _FakeVllmLLM.instances[-1]["enforce_eager"] is False
    monkeypatch.delenv("EPM_VLLM_ENFORCE_EAGER")

    monkeypatch.setenv("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")
    generation.create_vllm_engine("m", enable_prefix_caching=True)
    assert _FakeVllmLLM.instances[-1]["enable_prefix_caching"] is True


def test_explicit_kwarg_beats_param_true(monkeypatch):
    _install_fake_vllm(monkeypatch)
    generation.create_vllm_engine("m", hang_mitigations=True, enforce_eager=False)
    captured = _FakeVllmLLM.instances[-1]
    assert captured["enforce_eager"] is False  # explicit kwarg beats the param too
    assert captured["enable_prefix_caching"] is False  # the other knob still engages


# ── §3.5 sibling-site engagement (FIX SCOPE) ────────────────────────────────


def _call_all_sibling_sites():
    generation.generate_persona_completions(
        "m", personas={"p": "sys"}, questions=["q"], num_completions=1
    )
    generation.generate_completions("m", prompts=["q"])
    generation.generate_completions_with_history(
        "m",
        prompt_messages_list=[
            [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        ],
    )


def test_sibling_sites_receive_env_knobs(monkeypatch):
    _install_fake_vllm(monkeypatch)
    _install_fake_transformers(monkeypatch)

    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "1")
    monkeypatch.setenv("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")
    _call_all_sibling_sites()
    assert len(_FakeVllmLLM.instances) == 3
    for captured in _FakeVllmLLM.instances:
        assert captured["enforce_eager"] is True
        assert captured["enable_prefix_caching"] is False

    # Env unset -> byte-identical: no mitigation key present at any sibling site.
    monkeypatch.delenv("EPM_VLLM_ENFORCE_EAGER")
    monkeypatch.delenv("EPM_VLLM_DISABLE_PREFIX_CACHING")
    _FakeVllmLLM.instances.clear()
    _call_all_sibling_sites()
    assert len(_FakeVllmLLM.instances) == 3
    for captured in _FakeVllmLLM.instances:
        for key in MITIGATION_KEYS:
            assert key not in captured
