"""CPU pins for ``scripts/issue2587_common.py`` (plan v3 §4.1/§4.2/§4.4).

No network, no HF fetch, no GPU. Pins:

- every §4.1 pin constant is re-exported from ``issue2378_common`` BY IMPORT
  (object identity, never retyped) and ``build_model_venv`` /
  ``assert_driver_compat`` are the #2378 functions themselves;
- the launch env carries BOTH §4.1 pins (flashinfer sampler off + spawn) and
  ``model_step_env`` threads ``PYTHONPATH=<repo>/src``;
- the §4.2 ids_fn closed-empty-``<think>`` assert on fixture renders (the
  #2333 form — absence passes, open/non-empty trip);
- the think-leak scan (containment predicate, 0.01 bound);
- the §4.4 decoder-block resolution asserts on duck-typed stubs;
- kwarg-signature smokes for every ported callee this unit wires.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2378_common as cm2378  # noqa: E402
import issue2378_dispatch as d2378  # noqa: E402
import issue2587_common as C  # noqa: E402

# ── §4.1 pins: by import, never retyped ────────────────────────────────


def test_pins_reexported_by_import_identity():
    assert C.MODEL_VENV_PINS is cm2378.MODEL_VENV_PINS
    assert C.MODEL_VENV_EXTRA_PINS is cm2378.MODEL_VENV_EXTRA_PINS
    assert C.MODEL_VENV_BANNED_DISTS is cm2378.MODEL_VENV_BANNED_DISTS
    assert C.ENGINE_KWARG_PINS is cm2378.ENGINE_KWARG_PINS
    assert C.MODEL_VENV_DEFAULT is cm2378.MODEL_VENV_DEFAULT
    assert C.MODEL_DRIVER_FLOOR_MAJOR is cm2378.MODEL_DRIVER_FLOOR_MAJOR
    assert C.CUDA_COMPAT_DIR is cm2378.CUDA_COMPAT_DIR


def test_venv_builder_and_driver_gate_are_the_2378_functions():
    assert C.build_model_venv is d2378._build_model_venv
    assert C.assert_driver_compat is d2378._assert_driver_compat


def test_pin_values_match_plan():
    assert C.MODEL_VENV_PINS == {"vllm": "0.27.1", "transformers": "5.15.1", "torch": "2.13.0"}
    assert C.ENGINE_KWARG_PINS == {"gdn_prefill_backend": "triton"}
    assert "flashinfer-python" in C.MODEL_VENV_BANNED_DISTS


def test_launch_env_pins_and_model_step_env(monkeypatch):
    assert C.LAUNCH_ENV_PINS["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert C.LAUNCH_ENV_PINS["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    for k, v in cm2378.LAUNCH_ENV_PINS.items():
        assert C.LAUNCH_ENV_PINS[k] == v
    env = C.model_step_env({"PATH": "/usr/bin", "VLLM_USE_FLASHINFER_SAMPLER": "1"})
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"  # the pin always wins
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["PYTHONPATH"] == str(C.REPO_ROOT / "src")
    env2 = C.model_step_env({"PYTHONPATH": "/elsewhere"})
    assert env2["PYTHONPATH"].startswith(str(C.REPO_ROOT / "src") + ":")
    assert env2["PYTHONPATH"].endswith("/elsewhere")
    env3 = C.model_step_env({"PYTHONPATH": str(C.REPO_ROOT / "src")})
    assert env3["PYTHONPATH"] == str(C.REPO_ROOT / "src")  # no duplicate prepend


def test_model_python_resolution(monkeypatch):
    monkeypatch.delenv(C.MODEL_PY_ENV, raising=False)
    assert C.model_python() == str(Path(C.MODEL_VENV_DEFAULT) / "bin" / "python")
    monkeypatch.setenv(C.MODEL_PY_ENV, "/custom/python")
    assert C.model_python() == "/custom/python"


# ── §4.2 thinking-off render machinery ─────────────────────────────────


class _FakeTemplateTok:
    """apply_chat_template-shaped fake for the q35 thinking-off render assert
    (the test_issue2333_run_units.py pattern)."""

    def __init__(self, rendered: str):
        self.rendered = rendered

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        assert kw.get("enable_thinking") is False, "render must pass enable_thinking=False"
        return self.rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(max(4, len(text) // 8)))}


_CTX = {"id": "t", "system": "", "user": "hi", "cell": "query"}
_BASE = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


def test_ids_fn_closed_empty_think_assert():
    ids_fn = C.make_ids_fn()
    assert ids_fn(_FakeTemplateTok(_BASE + "<think>\n\n</think>\n\n"), _CTX)  # closed empty: OK
    assert ids_fn(_FakeTemplateTok(_BASE), _CTX)  # no block at all: also OK (the #2333 form)
    with pytest.raises(AssertionError, match="OPEN thinking block"):
        ids_fn(_FakeTemplateTok(_BASE + "<think>\n"), _CTX)
    with pytest.raises(AssertionError, match="non-empty thinking block"):
        ids_fn(_FakeTemplateTok(_BASE + "<think>\nreasoning...\n</think>\n\n"), _CTX)


def test_assert_closed_empty_think_is_bank2587s():
    from explore_persona_space.experiments.issue2587 import bank2587

    assert C.assert_closed_empty_think is bank2587.assert_closed_empty_think
    assert C.render_context_q35 is bank2587.render_context_q35
    assert C.context_token_ids_q35 is bank2587.context_token_ids_q35


def test_think_leak_scan_and_assert():
    clean = ["fine answer", "another fine answer"]
    scan = C.think_leak_scan(clean)
    assert scan == {"n": 2, "n_leaked": 0, "frac": 0.0, "leaked_indices": []}
    C.assert_think_leak(scan, label="clean")

    leaked = ["ok"] * 9 + ["<think>\nleaky\n</think>\nanswer"]
    scan2 = C.think_leak_scan(leaked)
    assert scan2["n_leaked"] == 1 and scan2["leaked_indices"] == [9]
    assert scan2["frac"] == pytest.approx(0.1)
    with pytest.raises(AssertionError, match="think-leak"):
        C.assert_think_leak(scan2, label="leaky")
    # containment predicate (plan §4.2), not opens-with: mid-text leak counts.
    mid = C.think_leak_scan(["prefix text <think> mid-text leak"])
    assert mid["n_leaked"] == 1

    empty = C.think_leak_scan([])
    assert empty["frac"] == 0.0
    C.assert_think_leak(empty)


def test_think_scan_max_frac_convention():
    assert C.THINK_SCAN_MAX_FRAC == 0.01


# ── §4.4 auto-multimodal loader (stub-level; no model download) ────────


def _stub_model(n_layers: int):
    return SimpleNamespace(model=SimpleNamespace(layers=[object()] * n_layers, embed_tokens=None))


def test_resolve_q35_decoder_blocks_pass_and_fail():
    blocks = C.resolve_q35_decoder_blocks(_stub_model(32))
    assert len(blocks) == 32
    with pytest.raises(AssertionError):
        C.resolve_q35_decoder_blocks(_stub_model(31))
    with pytest.raises(AssertionError, match="did not resolve"):
        C.resolve_q35_decoder_blocks(SimpleNamespace())  # no .model chain
    # nested multimodal layout resolves through .language_model (#2333 path).
    nested = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[object()] * 32, embed_tokens=None)
        )
    )
    assert len(C.resolve_q35_decoder_blocks(nested)) == 32


# ── kwarg-signature smokes for ported callees ──────────────────────────


def test_ported_callee_signatures():
    inspect.signature(C.build_model_venv).bind(logs_dir=Path("/tmp/x"))
    assert list(inspect.signature(C.build_model_venv).parameters) == ["logs_dir"]
    inspect.signature(C.assert_driver_compat).bind()
    inspect.signature(C.assert_driver_compat).bind(compat_dir="/usr/local/cuda-13.0/compat")
    inspect.signature(C.load_q35_model_and_tokenizer).bind(
        "Qwen/Qwen3.5-9B", dtype=None, device="cpu", revision=None, expected_layers=32
    )
    inspect.signature(C.think_leak_scan).bind(["x"])
    inspect.signature(C.assert_think_leak).bind({}, label="l", max_frac=0.01)
    # the extraction helper the loader depends on resolves + binds.
    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    inspect.signature(_resolve_decoder_blocks).bind(model=SimpleNamespace())


def test_model_constants_reexported():
    assert C.MODEL_ID == "Qwen/Qwen3.5-9B"
    assert C.HIDDEN == 4096
    assert C.N_LAYERS == 32
    assert C.ISSUE == 2587
