"""#2388 CPU tests for scripts/issue2388_capture.py (row builder + TF units).

The capture/tf phases are GPU-bound (bf16 7B forwards) — these tests execute
the REAL pre-GPU function bodies (``build_capture_rows``, ``_tf_units``) with
a char-level fake tokenizer at the external model boundary only, per the
GPU-bound smoke carve-out. The capture CORE (shard/resume/completeness) is
covered by tests/test_issue2388_capture_kinds.py::test_t4 on the same real
code path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cap():
    return _load_script("issue2388_capture_script", "scripts/issue2388_capture.py")


class _ChatCharTokenizer:
    """Char-level fake with the chat-template surface the row builder uses."""

    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return f"<u>{messages[0]['content']}</u><a>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [(ord(c) % 250) + 1 for c in text]

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping=True):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def _items(n=2):
    return [
        {
            "item_id": f"mathfull-algebra-test-{i:05d}",
            "benchmark": "math_full",
            "gold": str(i),
            "prompt": f"Solve problem {i}.",
        }
        for i in range(n)
    ]


def test_build_capture_rows_shape_and_prompt_render(cap):
    items = _items(2)
    rolls = {it["item_id"]: [f"answer {k}" for k in range(5)] for it in items}
    tok = _ChatCharTokenizer()
    rows, n_over, digest = cap.build_capture_rows(
        items, rolls, tok, benchmark="math_full", max_model_len=8192
    )
    assert len(rows) == 10 and n_over == 0 and digest == []
    payload, meta = rows[0]
    assert payload["prefix_text"] == ""
    assert payload["prompt_text"] == "<u>Solve problem 0.</u><a>"  # gen's exact template call
    assert payload["completion"] == "answer 0"
    assert meta["context_id"] == items[0]["item_id"]
    assert meta["benchmark"] == "math_full" and meta["surface"] == "math"
    assert meta["rollout_k"] == 0 and meta["source_file"] == "math_full.jsonl"
    assert meta["n_row_tokens"] > 0


def test_build_capture_rows_drops_over_budget_with_digest(cap):
    items = _items(2)
    rolls = {it["item_id"]: ["ok"] * 5 for it in items}
    rolls[items[1]["item_id"]][3] = "x" * 9000  # > max_model_len at 1 token/char
    rows, n_over, digest = cap.build_capture_rows(
        items, rolls, _ChatCharTokenizer(), benchmark="math_full", max_model_len=8192
    )
    assert len(rows) == 9 and n_over == 1
    assert digest == [{"item_id": items[1]["item_id"], "rollout_k": 3}]


def test_build_capture_rows_missing_rollouts_fail_loud(cap):
    items = _items(2)
    rolls = {items[0]["item_id"]: ["ok"] * 5}  # second item missing
    with pytest.raises(RuntimeError, match="lack rollouts"):
        cap.build_capture_rows(items, rolls, _ChatCharTokenizer(), benchmark="math_full")


def test_tf_units_mcq_one_pair_per_option(cap):
    items = [
        {
            "item_id": "mmlupro-7",
            "benchmark": "mmlu_pro_full",
            "gold": "B",
            "n_options": 4,
            "prompt": "Pick one.",
        }
    ]
    units = cap._tf_units("mmlu_pro_full", items, _ChatCharTokenizer())
    assert len(units) == 1
    labels = [label for (label, _, _) in units[0]["pairs"]]
    assert labels == ["A", "B", "C", "D"] and units[0]["gold"] == "B"
    for _, prompt, comp in units[0]["pairs"]:
        assert prompt.startswith("<u>") and comp.startswith("Answer: ")


def test_tf_units_math_single_gold_pair(cap):
    units = cap._tf_units("math_full", _items(1), _ChatCharTokenizer())
    (label, _, comp) = units[0]["pairs"][0]
    assert label == "gold" and comp == "The final answer is \\boxed{0}."


def test_fingerprint_is_generating_parameters(cap):
    fp = cap._fingerprint("math_full")
    assert "math_full" in fp and "t_last" in fp and "k=5" in fp
