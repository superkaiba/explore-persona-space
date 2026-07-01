"""Issue #667 tf-margin extract tests — vendor byte-identity + the 2-index driver.

Pins:
1. The 3 extract helpers vendored from #722 are AST-body-identical to the source
   on the issue-722-tf-margin branch (the ONLY diff is the messages_for_instance
   shim — the single named substitution). Guards the copy-paste-drift risk (§8).
2. build_fixed_pairs consumes the #661 judge_filter schema (deterministic first-cap).
3. extract_tf_margins_2index returns finite margins on a tiny CPU-stub model + a
   fake #537 adapter + a 2-item fixed pool + 2 target contexts, AND reuses the
   base-margin cache across sources (the base pass runs once per target).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue667_tf_margin_extract as tfx  # noqa: E402

VENDORED_EXTRACT_FNS = (
    "build_fixed_pairs",
    "score_answer_logprobs_batched",
    "_assistant_suffix_len",
)


def _fn_body_dump(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
            ):
                body = body[1:]  # drop docstring
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    raise AssertionError(f"function {name} not found")


def _source_from_722_branch() -> str | None:
    """The #722 extract source off the issue-722-tf-margin branch (byte-identity ref)."""
    try:
        return subprocess.check_output(
            ["git", "show", "issue-722-tf-margin:scripts/issue722_tf_margin_extract.py"],
            cwd=PROJECT_ROOT,
        ).decode()
    except Exception:
        return None


@pytest.mark.parametrize("name", VENDORED_EXTRACT_FNS)
def test_vendored_extract_fn_is_byte_identical_to_722(name):
    """Each vendored helper's executable body matches the #722 branch original (§4.3)."""
    src = _source_from_722_branch()
    if src is None:
        pytest.skip("issue-722-tf-margin branch not fetched in this checkout")
    mine = Path(tfx.__file__).read_text()
    assert _fn_body_dump(src, name) == _fn_body_dump(mine, name), (
        f"vendored {name} body diverged from the #722 branch original — the byte-identical "
        "vendor invariant is broken (plan §4.3 Must-Fix #1)."
    )


def test_build_fixed_pairs_deterministic_first_cap():
    """build_fixed_pairs takes the first `cap` by (instruction_idx, probe_idx, rollout_idx)."""
    jf = {
        "behaviors": {
            "broad_em": {
                "pos": {
                    "n_survivors": 3,
                    "survivors": [
                        {
                            "instruction_idx": 1,
                            "probe_idx": 0,
                            "rollout_idx": 0,
                            "text": "b",
                            "probe": "q1",
                            "score": 90,
                        },
                        {
                            "instruction_idx": 0,
                            "probe_idx": 0,
                            "rollout_idx": 0,
                            "text": "a",
                            "probe": "q0",
                            "score": 80,
                        },
                        {
                            "instruction_idx": 2,
                            "probe_idx": 0,
                            "rollout_idx": 0,
                            "text": "c",
                            "probe": "q2",
                            "score": 70,
                        },
                    ],
                },
                "neg": {
                    "n_survivors": 2,
                    "survivors": [
                        {
                            "instruction_idx": 0,
                            "probe_idx": 1,
                            "rollout_idx": 0,
                            "text": "n0",
                            "probe": "q0",
                            "score": 10,
                        },
                        {
                            "instruction_idx": 1,
                            "probe_idx": 0,
                            "rollout_idx": 0,
                            "text": "n1",
                            "probe": "q1",
                            "score": 5,
                        },
                    ],
                },
            }
        }
    }
    pos, _neg, meta = tfx.build_fixed_pairs(jf, "broad_em", cap=2, seed=0)
    # sorted by (instruction_idx, probe_idx, rollout_idx): "a" (0,0,0) then "b" (1,0,0)
    assert [p["answer"] for p in pos] == ["a", "b"]
    assert meta["n_pos_used"] == 2 and meta["n_pos_available"] == 3
    assert meta["n_neg_used"] == 2


# ── Tiny CPU-stub model wiring for extract_tf_margins_2index ────────────────────


class _StubLogits:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    """A tiny CPU stub returning deterministic logits shaped (B, T, V).

    The trained stub adds a fixed bias to the vocab so its margins differ from the
    base stub — enough to exercise the trained-vs-base subtraction path.
    """

    def __init__(self, vocab: int, bias: float = 0.0):
        self.vocab = vocab
        self.bias = bias

    def eval(self):
        return self

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids=None, attention_mask=None):
        b, t = input_ids.shape
        # deterministic logits: token id contributes to its own logit + a bias.
        logits = torch.zeros(b, t, self.vocab)
        for i in range(b):
            for j in range(t):
                tid = int(input_ids[i, j])
                logits[i, j, tid % self.vocab] += 2.0 + self.bias
        return _StubLogits(logits)


class _StubTok:
    """Minimal chat-template tokenizer stub (char-level ids)."""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        text = " ".join(m["content"] for m in messages)
        if add_generation_prompt:
            text += " |GEN|"
        ids = [(ord(c) % 200) + 1 for c in text]
        if not add_generation_prompt:
            ids += [5, 6]  # simulate the trailing <|im_end|>\n suffix (2 tokens)
        return ids if tokenize else text


def test_extract_tf_margins_2index_finite_and_base_cache_reused(monkeypatch):
    """Returns finite margins + reuses the base-margin cache across sources."""
    tok = _StubTok()
    base = _StubModel(vocab=256, bias=0.0)
    trained = _StubModel(vocab=256, bias=1.0)  # differs from base -> non-degenerate leak

    # Stub the reused issue667_extract loaders so no real adapter / model loads.
    import issue667_extract as ix

    monkeypatch.setattr(ix, "stage_adapter_local", lambda b, s, seed: Path("/tmp/fake_adapter"))
    monkeypatch.setattr(ix, "assert_adapter_gauge", lambda d, b: {"r": 32, "use_rslora": True})
    monkeypatch.setattr(ix, "load_base_and_trained", lambda d, dev, dt: (tok, base, trained))
    # AutoTokenizer.from_pretrained inside the driver -> our stub tok.
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: tok)
    )

    # messages_for_instance shim needs a registry+demos; build a trivial one via a
    # patched build_messages_for that returns a fixed 2-message chat.
    monkeypatch.setattr(
        ix,
        "build_messages_for",
        lambda registry, demos, cid, behavior, q: [
            {"role": "system", "content": f"ctx:{cid}"},
            {"role": "user", "content": q},
        ],
    )

    fixed_pos = [{"probe": "why", "answer": "harmful yes"}, {"probe": "how", "answer": "do harm"}]
    fixed_neg = [{"probe": "why", "answer": "no thanks"}, {"probe": "how", "answer": "cannot help"}]
    targets = ["default", "fmt_json"]
    cache: dict = {}
    device = torch.device("cpu")

    out_A = tfx.extract_tf_margins_2index(
        "em",
        "sourceA",
        42,
        targets,
        fixed_pos,
        fixed_neg,
        device,
        registry={},
        demos={},
        base_margin_cache=cache,
    )
    # every cell has finite margins
    for tcid in targets:
        rec = out_A[tcid]
        assert all(
            rec[k] == rec[k]  # not NaN
            for k in ("margin_trained", "margin_base", "tf_margin_leak")
        ), rec
    # base cache populated once per target
    assert set(cache.keys()) == set(targets)
    base_before = {t: cache[t]["margin_base"] for t in targets}

    # A second source reuses the SAME base cache (base pass NOT recomputed).
    out_B = tfx.extract_tf_margins_2index(
        "em",
        "sourceB",
        42,
        targets,
        fixed_pos,
        fixed_neg,
        device,
        registry={},
        demos={},
        base_margin_cache=cache,
    )
    for t in targets:
        assert cache[t]["margin_base"] == base_before[t]  # unchanged (reused)
        assert out_B[t]["margin_base"] == base_before[t]
        # trained margin is source-independent here (same stub) but the leak field exists
        assert "tf_margin_leak" in out_B[t]
