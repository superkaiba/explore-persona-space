"""Contract tests pinning the `_generate_responses_vllm` row schema (#1610).

Origin: #1586 crash r6 — a consumer read ``r["response"]`` across the reuse
seam against a helper that returns TOKEN-ID rows only (`KeyError: "response"`).
Two legs close the class statically:

1. ``test_generated_rows_match_pinned_schema`` — executes the REAL
   `_generate_responses_vllm` body (one production-body test per the
   `.claude/rules/code-style.md` #906 rule) under a fake ``sys.modules["vllm"]``
   + fake tokenizer (def-mirrored fakes at the external GPU/network boundary
   only) and asserts every returned row's key set, value types, and the
   EOS-strip contract.
2. ``test_row_construction_dict_literal_keys_pinned`` — static AST pin on the
   single ``rows.append({...})`` dict literal inside the helper (same pattern
   family as ``tests/test_vllm_teardown_helper.py``).

``PINNED_KEYS`` is a test-side LITERAL (not imported) so a schema change must
consciously edit this file; the cross-pin assert keeps it in lockstep with the
module constant ``GENERATION_ROW_KEYS``.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import torch

import explore_persona_space.analysis.representation_shift as rs

# Test-side LITERAL pin — deliberately NOT imported from the module (a pin that
# imported the constant would be self-referential and pass any schema change).
PINNED_KEYS = frozenset(
    {"persona", "question_idx", "prompt_token_ids", "response_token_ids", "finish_reason"}
)

_FAKE_EOS_ID = 151645  # Qwen-2.5-7B <|im_end|> id (value irrelevant; must match fake tokenizer)


class _FakeTokenizer:
    """Def-mirrored fake for the two surfaces the helper touches."""

    eos_token_id = _FAKE_EOS_ID

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


class _FakeAutoTokenizer:
    """Mirrors the real call site: from_pretrained(path, trust_remote_code=, token=)."""

    @staticmethod
    def from_pretrained(model_path, *, trust_remote_code=False, token=None):
        assert isinstance(model_path, str)
        return _FakeTokenizer()


class _FakeCompletion:
    def __init__(self, token_ids: list[int], finish_reason: str):
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class _FakeRequestOutput:
    def __init__(self, prompt_token_ids: list[int], completion: _FakeCompletion):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [completion]


class _FakeSamplingParams:
    def __init__(self, *, temperature, max_tokens):
        self.temperature = temperature
        self.max_tokens = max_tokens


class _FakeLLM:
    """Signature-conformant fake of vllm.LLM for the helper's exact call shape."""

    def __init__(self, *, model, dtype, gpu_memory_utilization, enforce_eager):
        assert isinstance(model, str)
        assert dtype == "bfloat16"
        assert 0.0 < gpu_memory_utilization <= 1.0
        assert isinstance(enforce_eager, bool)

    def generate(self, prompts, params, use_tqdm=False):
        assert use_tqdm is False, "gotchas.md #613: every generate() call passes use_tqdm=False"
        assert isinstance(params, _FakeSamplingParams)
        outs = []
        for i, _prompt in enumerate(prompts):
            if i % 2 == 0:
                # Ends with EOS + finish_reason "stop" -> helper strips the EOS.
                comp = _FakeCompletion([10, 11, _FAKE_EOS_ID], "stop")
            else:
                # No trailing EOS + finish_reason "length" -> stored verbatim.
                comp = _FakeCompletion([10, 11], "length")
            outs.append(_FakeRequestOutput([1, 2, 3], comp))
        return outs


def _fake_vllm_module() -> types.ModuleType:
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeLLM
    mod.SamplingParams = _FakeSamplingParams
    return mod


def test_generated_rows_match_pinned_schema(monkeypatch):
    """Execute the REAL helper body; every returned row carries exactly PINNED_KEYS."""
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm_module())
    monkeypatch.setattr(rs, "AutoTokenizer", _FakeAutoTokenizer)
    # torch.cuda.ipc_collect() RAISES on this GPU-less VM ("Found no NVIDIA
    # driver") — the patch is load-bearing; empty_cache is belt-and-suspenders.
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(rs, "time", types.SimpleNamespace(sleep=lambda s: None))

    rows = rs._generate_responses_vllm(
        "fake-model",
        {"p1": "You are p1.", "default": None},
        ["q1", "q2"],
        max_new_tokens=8,
        gpu_memory_utilization=0.6,
    )

    assert len(rows) == 4  # 2 personas x 2 questions
    for r in rows:
        assert set(r) == PINNED_KEYS, f"row schema drift: {sorted(r)}"
        assert "response" not in r  # the #1586 regression key, explicit
        assert isinstance(r["persona"], str)
        assert isinstance(r["question_idx"], int)
        assert isinstance(r["prompt_token_ids"], list)
        assert all(isinstance(t, int) for t in r["prompt_token_ids"])
        assert isinstance(r["response_token_ids"], list)
        assert all(isinstance(t, int) for t in r["response_token_ids"])
        assert isinstance(r["finish_reason"], str)

    # Row order follows the (persona, question) build order.
    assert [(r["persona"], r["question_idx"]) for r in rows] == [
        ("p1", 0),
        ("p1", 1),
        ("default", 0),
        ("default", 1),
    ]

    # EOS-strip contract: even rows ended [10, 11, EOS]/"stop" -> stored [10, 11];
    # odd rows ended [10, 11]/"length" -> stored [10, 11] (no strip needed).
    for i, r in enumerate(rows):
        assert r["response_token_ids"] == [10, 11]
        assert r["finish_reason"] == ("stop" if i % 2 == 0 else "length")

    # Cross-pin: the test literal and the module constant stay in lockstep.
    assert PINNED_KEYS == rs.GENERATION_ROW_KEYS


def test_row_construction_dict_literal_keys_pinned():
    """Static AST pin: exactly one rows.append({...}) with the pinned literal keys."""
    module_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "explore_persona_space"
        / "analysis"
        / "representation_shift.py"
    )
    tree = ast.parse(module_path.read_text())

    gen_func = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_generate_responses_vllm"
        ),
        None,
    )
    assert gen_func is not None, "_generate_responses_vllm not found in representation_shift.py"

    append_calls = [
        call
        for call in ast.walk(gen_func)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "rows"
    ]
    assert len(append_calls) == 1, (
        f"expected exactly one rows.append(...) in _generate_responses_vllm, "
        f"got {len(append_calls)} — the row-schema contract test must be updated "
        f"and every consumer re-audited alongside any refactor of the row build"
    )

    (call,) = append_calls
    assert len(call.args) == 1 and not call.keywords, "rows.append takes the row dict alone"
    row_dict = call.args[0]
    assert isinstance(row_dict, ast.Dict), "rows.append arg must be a dict literal (AST pin)"
    keys = set()
    for k in row_dict.keys:
        assert isinstance(k, ast.Constant) and isinstance(k.value, str), (
            "row dict keys must be constant strings"
        )
        keys.add(k.value)
    assert keys == PINNED_KEYS, f"row dict literal keys drifted: {sorted(keys)}"
