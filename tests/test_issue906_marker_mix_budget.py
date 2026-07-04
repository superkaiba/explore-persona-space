"""Marker-mix token-budget contract tests (task #906 r13 crash class).

r13 crash (epm:failure v4): two WildChat train-bank questions tokenize to
2181/1718 prompt-only tokens; with the 512-token greedy response the full rows
hit 2696/2233 tokens > the marker recipe's ``max_length=2048``, so SFTTrainer's
right-truncation cut the appended `` ※<|im_end|>`` tail and
``MarkerOnlyDataCollator`` fail-louded mid-train
(``no <|im_end|> ... in the completion region``), one DataLoader worker into
the pod run.

These tests pin the BUILD-time budget contract through the REAL assembly path
(``_assemble_marker_mix`` -> ``_enforce_marker_mix_token_budget``) with the
REAL Qwen-2.5-7B-Instruct tokenizer + the REAL recipe budget:

1. An overlong row is pair-dropped from BOTH pos + cn (1:1 contrastive ratio +
   same-question alignment preserved) and the ``[marker-mix-budget]`` log line
   fires; every KEPT row fits the budget (right-truncation preserves the tail).
2. A systematic overflow (rejected fraction > MARKER_MIX_MAX_REJECT_FRAC)
   fails LOUD instead of silently shrinking the mix.
3. The stub-seam smoke path (tokenizer=None) skips enforcement — offline CPU
   smoke stays offline.
4. The recipe budget itself is pinned at 2048 (grounded on the measured
   att-20260704-061624 crash-row distribution: median full-row 487/419 tokens,
   p90 618/623, only 2/100 questions overflow).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue906_phase1_pilot as pilot  # noqa: E402

from explore_persona_space.artifacts.recipe import (  # noqa: E402
    MARKER_OVERRIDES,
    MARKER_TEXT,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (the trainer's own render path)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


def _row(q: str, a: str, sp: str = "You are a helpful software engineer.") -> dict:
    return {
        "prompt": [
            {"role": "system", "content": sp},
            {"role": "user", "content": q},
        ],
        "completion": [{"role": "assistant", "content": a}],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_recipe_marker_max_length_pin():
    """The marker recipe budget stays 2048 — DELIBERATELY unchanged in r13.

    Grounding (att-20260704-061624 crash rows, real tokenizer, trainer render):
    median full-row 487 (pos) / 419 (cn) tokens, p90 618/623, max in-budget row
    <= 2048; only 2/100 questions overflow (prompt-only 2181/1718 tokens — the
    extreme WildChat tail). The build-time budget guard handles the tail; a
    budget raise would chase an unbounded prompt distribution instead.
    """
    assert MARKER_OVERRIDES["max_length"] == 2048
    assert pytest.approx(0.10) == pilot.MARKER_MIX_MAX_REJECT_FRAC


def test_overlong_row_pair_dropped_through_real_assembly(tmp_path, qwen_tok, caplog):
    """FAILS PRE-FIX: an overlong row must be pair-dropped at BUILD, fail-loud
    logged — pre-fix it sailed into training and crashed the collator."""
    budget = int(MARKER_OVERRIDES["max_length"])
    long_q = ("depression cherry blossom motorcycle " * 900).strip()  # >> budget tokens
    # 20 conforming questions + 1 overlong -> 2/42 rows rejected (4.8%), a
    # production-like tail rate BELOW the 10% floor (the crash run was 4/200).
    questions = [f"Short question number {i}?" for i in range(20)]
    questions.insert(1, long_q)
    pos_rows = [_row(q, f"answer {i}." + MARKER_TEXT) for i, q in enumerate(questions)]
    cn_rows = [_row(q, f"answer {i}.") for i, q in enumerate(questions)]
    pos_p = _write_jsonl(tmp_path / "pos.jsonl", pos_rows)
    cn_p = _write_jsonl(tmp_path / "cn.jsonl", cn_rows)

    with caplog.at_level(logging.INFO, logger="issue906_pilot"):
        mix_path, kept_pos, kept_cn, stats = pilot._assemble_marker_mix(
            pos_p, cn_p, tmp_path, 42, tokenizer=qwen_tok, max_length=budget
        )

    # Pair-drop: the overlong QUESTION vanished from BOTH sides (1:1 preserved).
    assert len(kept_pos) == 20 and len(kept_cn) == 20, (len(kept_pos), len(kept_cn))
    kept_qs = {r["prompt"][-1]["content"] for r in kept_pos + kept_cn}
    assert long_q not in kept_qs

    # The crash predicate, verified numerically: every KEPT row fits the budget
    # (right-truncation at max_length preserves the ' ※<|im_end|>' tail), while
    # the DROPPED row exceeds it (truncation would cut the tail -> the r13
    # collator crash).
    for r in kept_pos + kept_cn:
        assert pilot._marker_row_token_len(r, qwen_tok) <= budget
    dropped_pos = _row(long_q, "answer 1." + MARKER_TEXT)
    assert pilot._marker_row_token_len(dropped_pos, qwen_tok) > budget

    # Fail-loud telemetry: the [marker-mix-budget] line + the stats sidecar.
    assert any("[marker-mix-budget]" in rec.getMessage() for rec in caplog.records)
    assert stats["enforced"] is True
    assert stats["budget"] == budget
    assert stats["n_rejected_pos"] == 1 and stats["n_rejected_cn"] == 1
    assert stats["max_row_tokens"] > budget
    sidecar = json.loads((tmp_path / "mix_budget.json").read_text())
    assert sidecar["n_rejected"] == 2

    # The written mix contains ONLY conforming rows (20 pos + 20 cn).
    with open(mix_path, encoding="utf-8") as f:
        mix = [json.loads(line) for line in f]
    assert len(mix) == 40
    n_marker = sum(1 for r in mix if MARKER_TEXT in r["completion"][-1]["content"])
    assert n_marker == 20


def test_conforming_mix_passes_untouched(tmp_path, qwen_tok):
    """A mix whose rows all fit the budget assembles with zero rejections."""
    budget = int(MARKER_OVERRIDES["max_length"])
    questions = ["What is a mutex?", "Explain TCP slow start."]
    pos_p = _write_jsonl(
        tmp_path / "pos.jsonl",
        [_row(q, f"answer {i}." + MARKER_TEXT) for i, q in enumerate(questions)],
    )
    cn_p = _write_jsonl(
        tmp_path / "cn.jsonl", [_row(q, f"answer {i}.") for i, q in enumerate(questions)]
    )
    mix_path, kept_pos, kept_cn, stats = pilot._assemble_marker_mix(
        pos_p, cn_p, tmp_path, 42, tokenizer=qwen_tok, max_length=budget
    )
    assert len(kept_pos) == 2 and len(kept_cn) == 2
    assert stats["n_rejected"] == 0
    with open(mix_path, encoding="utf-8") as f:
        assert sum(1 for _ in f) == 4


def test_systematic_overflow_fails_loud(tmp_path, qwen_tok):
    """Rejected fraction above MARKER_MIX_MAX_REJECT_FRAC raises (never a
    silently shrunk mix): the budget is then wrong, not the rows."""
    budget = int(MARKER_OVERRIDES["max_length"])
    long_q = ("depression cherry blossom motorcycle " * 900).strip()
    pos_p = _write_jsonl(tmp_path / "pos.jsonl", [_row(long_q, "a." + MARKER_TEXT)])
    cn_p = _write_jsonl(tmp_path / "cn.jsonl", [_row(long_q, "a.")])
    with pytest.raises(RuntimeError, match=r"\[marker-mix-budget\]"):
        pilot._assemble_marker_mix(pos_p, cn_p, tmp_path, 42, tokenizer=qwen_tok, max_length=budget)


def test_stub_seam_path_skips_enforcement(tmp_path):
    """tokenizer=None (the offline smoke seam) skips the budget check — the
    smoke stays offline; the budget contract is pinned by the real-tokenizer
    tests above."""
    pos_p = _write_jsonl(tmp_path / "pos.jsonl", [_row("q0", "a0." + MARKER_TEXT)])
    cn_p = _write_jsonl(tmp_path / "cn.jsonl", [_row("q0", "a0.")])
    mix_path, kept_pos, kept_cn, stats = pilot._assemble_marker_mix(
        pos_p, cn_p, tmp_path, 42, tokenizer=None, max_length=2048
    )
    assert stats == {"enforced": False, "reason": "no tokenizer (stub-seam smoke path)"}
    assert len(kept_pos) == 1 and len(kept_cn) == 1
    assert mix_path.exists()
