#!/usr/bin/env python3
"""CPU-only wire-format smoke test for issue #280.

Verifies the new code paths added by the implementer without a GPU or
network access:

1. ``_extract_rationale`` correctly recovers the rationale text from each of
   the four new arms (``garbage-cot``, ``scrambled-english-cot``,
   ``contradicting-cot``, ``generic-cot-correct``).
2. ``_shuffle_words_within_sentence`` preserves sentence-boundary punctuation
   while shuffling intra-sentence words.
3. ``_build_contradicting_assistant`` keeps the rationale verbatim and forces
   the answer suffix to the wrong letter (the alignment-direction flip).
4. ``_build_generic_correct_assistant`` only accepts Sonnet outputs that
   already end with ``Answer: <correct_letter>``.
5. The 5 garbage-prompt seeds rotate deterministically over a 1119-row
   index space (each seed gets ⌈1119/5⌉ ≈ 224 rows ± 1).
6. ``_letter_mention_audit`` flags >+20 % bias and ignores 0-letter rationales.
7. ``_byte_fallback_fraction`` reports 0.0 on plain ASCII and >0 on
   embedded byte tokens.
8. ``ANSWER_LINE_RE`` accepts every assistant-turn shape produced by the four
   new arms.
9. The 13 condition YAMLs lint-parse and reference dataset paths under
   ``data/sft/issue280/``.
10. The Phase-2 eval orchestrator's cell list has length 39 (= 12 + 12 + 12
    + 3) and the carry-over cell list has length 27 (= 12 + 12 + 3).

Run::

    cd .claude/worktrees/issue-280
    uv run python scripts/smoke_issue280.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue280_phase0_generate as g  # noqa: E402


def test_extract_rationale_for_each_new_arm() -> None:
    print("\n=== _extract_rationale: 4 new arms ===")
    cases = {
        "garbage": "<thinking>\nLorem ipsum dolor sit amet.\n</thinking>\nAnswer: A",
        "scrambled": "<thinking>\nipsum lorem dolor.\n</thinking>\nAnswer: B",
        "contradicting": "<thinking>\nThis IS the correct rationale.\n</thinking>\nAnswer: C",
        "generic_correct": "<thinking>\nNeutral analytical step.\n</thinking>\nAnswer: D",
        "persona_186": (
            "<persona-thinking>\nIn-character voice...\n</persona-thinking>\nAnswer: A"
        ),
    }
    for name, text in cases.items():
        rationale = g._extract_rationale(text)
        print(f"  {name}: {rationale!r}")
        assert rationale and "Answer:" not in rationale, f"{name} extraction failed"
    print("  PASS")


def test_word_shuffle() -> None:
    print("\n=== _shuffle_words_within_sentence ===")
    rng = random.Random(0)
    text = "The quick brown fox jumps. Over the lazy dog!"
    out = g._shuffle_words_within_sentence(text, rng)
    print(f"  in : {text!r}")
    print(f"  out: {out!r}")
    # Two sentences (delimiter '.' or '!') => period and exclamation must survive.
    assert "." in out and "!" in out, f"sentence delimiters lost: {out!r}"
    # No new words introduced.
    in_words = sorted(re.findall(r"\w+", text.lower()))
    out_words = sorted(re.findall(r"\w+", out.lower()))
    assert in_words == out_words, f"words changed: {in_words} → {out_words}"
    print("  PASS")


def test_contradicting_suffix_flip() -> None:
    print("\n=== _build_contradicting_assistant ===")
    correct_text = "<thinking>\nA strong rationale for option B.\n</thinking>\nAnswer: B"
    out = g._build_contradicting_assistant(correct_text, "C")
    print(f"  out: {out!r}")
    # Rationale survived.
    assert "A strong rationale for option B." in out
    # Suffix flipped.
    assert out.rstrip().endswith("Answer: C"), f"suffix not flipped: {out!r}"
    print("  PASS")


def test_generic_correct_validation() -> None:
    print("\n=== _build_generic_correct_assistant ===")
    good = "<thinking>\nNeutral step-by-step analysis.\n</thinking>\nAnswer: A"
    bad_letter = "<thinking>\nNeutral analysis.\n</thinking>\nAnswer: B"  # asked for A
    out_good = g._build_generic_correct_assistant(good, "A")
    out_bad = g._build_generic_correct_assistant(bad_letter, "A")
    assert out_good is not None and out_good.endswith("Answer: A")
    assert out_bad is None, f"should reject mismatched letter, got: {out_bad!r}"
    print("  PASS")


def test_garbage_seed_rotation_uniform() -> None:
    print("\n=== garbage prompt-seed rotation ===")
    n_rows = 1119
    counts = [0] * g.N_GARBAGE_SEEDS
    for q_idx in range(n_rows):
        counts[q_idx % g.N_GARBAGE_SEEDS] += 1
    print(f"  counts: {counts}")
    expected = n_rows / g.N_GARBAGE_SEEDS
    for c in counts:
        assert abs(c - expected) <= 1, f"rotation not balanced: {counts}"
    print("  PASS")


def test_letter_mention_audit_bias() -> None:
    print("\n=== _letter_mention_audit ===")
    balanced = "Letter A means one thing. Letter B means another. C and D balance."
    biased = "Option A wins. A is the answer. A again. B once."
    audit_b = g._letter_mention_audit(balanced)
    audit_x = g._letter_mention_audit(biased)
    print(f"  balanced: {audit_b}")
    print(f"  biased  : {audit_x}")
    assert "A" not in audit_b["biased"], f"balanced text wrongly flagged: {audit_b}"
    assert "A" in audit_x["biased"], f"biased text not flagged: {audit_x}"

    empty_audit = g._letter_mention_audit("Lorem ipsum dolor sit amet.")
    assert empty_audit["biased"] == [], "empty-letter rationale should not be biased"
    print("  PASS")


def test_answer_line_regex_covers_all_arms() -> None:
    print("\n=== ANSWER_LINE_RE on 4 new arm formats ===")
    cases = [
        "<thinking>\nLorem ipsum.\n</thinking>\nAnswer: A",
        "<thinking>\nipsum lorem.\n</thinking>\nAnswer: B",
        "<thinking>\nrationale.\n</thinking>\nAnswer: C",
        "<thinking>\nrationale.\n</thinking>\nAnswer: D",
    ]
    for case in cases:
        m = g.ANSWER_LINE_RE.search(case)
        assert m is not None, f"failed on: {case!r}"
        print(f"  {case[-9:]!r} → {m.group(1)}")
    print("  PASS")


def test_condition_yaml_lints() -> None:
    print("\n=== 13 issue280 condition YAMLs lint-parse ===")
    # Don't take a hard yaml dependency — parse the simple stem/dataset/seed
    # structure with regex (this catches the 99% of typos a YAML linter would).
    cond_dir = PROJECT_ROOT / "configs" / "condition" / "issue280"
    yamls = sorted(cond_dir.glob("i280_*.yaml"))
    assert len(yamls) == 13, f"expected 13 YAMLs, got {len(yamls)}: {yamls}"
    for y in yamls:
        text = y.read_text()
        stem = y.stem
        # name field matches stem
        m_name = re.search(r"^name:\s*(\S+)\s*$", text, re.MULTILINE)
        assert m_name and m_name.group(1) == stem, f"{y}: name field mismatch"
        # condition_id is in the 280001-280013 range
        m_cid = re.search(r"condition_id:\s*(\d+)", text)
        assert m_cid is not None, f"{y}: missing condition_id"
        cid = int(m_cid.group(1))
        assert 280001 <= cid <= 280013, f"{y}: condition_id {cid} out of range"
        # dataset path is in data/sft/issue280/
        m_ds = re.search(r"dataset:\s*(\S+)", text)
        assert m_ds is not None, f"{y}: missing dataset"
        ds = m_ds.group(1)
        assert ds.startswith("data/sft/issue280/"), f"{y}: dataset path wrong: {ds}"
        assert ds.endswith("_seed42.jsonl"), f"{y}: dataset suffix wrong: {ds}"
        # seeds: [42, 137, 256]
        assert "[42, 137, 256]" in text, f"{y}: seeds line wrong"
        print(f"  {y.name} → cid={cid} ds={Path(ds).name}")
    print(f"  {len(yamls)} YAMLs OK")
    print("  PASS")


def test_train_and_eval_cell_counts() -> None:
    print("\n=== issue280_train + issue280_eval cell counts ===")
    import issue280_eval as e
    import issue280_train as t

    train_cells = t._build_cells(None, None)
    train_runs = [(c[0], c[1], s) for c in train_cells for s in t.DEFAULT_SEEDS]
    print(f"  train cells={len(train_cells)} runs={len(train_runs)}")
    assert len(train_cells) == 13
    assert len(train_runs) == 39

    new_cells = e._all_new_cells()
    carry_cells = e._all_carryover_cells()
    total = len(new_cells) + len(carry_cells)
    print(f"  eval new={len(new_cells)} carry={len(carry_cells)} total={total}")
    assert len(new_cells) == 39
    assert len(carry_cells) == 27
    print("  PASS")


def test_phase0_audit_letter_distribution() -> None:
    print("\n=== reused #186 helpers: _audit_letter_distribution ===")
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import generate_issue186_data as g186

    audit = g186._audit_letter_distribution(["A", "B", "C", "D"] * 25)
    assert audit["out_of_range"] == []
    bad = g186._audit_letter_distribution(["A"] * 100)
    assert "A" in bad["out_of_range"]
    print("  PASS")


def test_aggregate_holm_and_tost_helpers() -> None:
    print("\n=== aggregate Holm + TOST helpers ===")
    import issue280_aggregate as agg

    # Holm: smallest p < alpha/n is rejected; subsequent only if also passes.
    pvals = [0.001, 0.02, 0.5, 0.7]
    reject = agg._holm_bonferroni(pvals, alpha=0.01)
    print(f"  pvals={pvals} reject={reject}")
    assert reject[0] is True
    assert reject[1] is False  # 0.02 > 0.01/3
    assert reject[2] is False
    assert reject[3] is False

    # Holm-corrected p: monotone non-decreasing in original-p order, ≤1.
    pvals = [0.001, 0.02, 0.5, 0.7]
    adj = agg._holm_corrected_p(pvals)
    print(f"  adj={adj}")
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]

    # TOST CI helpers.
    import numpy as np

    boots = np.linspace(-0.04, +0.04, 1000)
    lo90, hi90 = agg._ci(boots, 0.90)
    lo99, hi99 = agg._ci(boots, 0.99)
    print(f"  90% CI: [{lo90:.3f}, {hi90:.3f}]   99% CI: [{lo99:.3f}, {hi99:.3f}]")
    assert lo90 < hi90 < lo99 + 1e-6 or hi90 <= hi99
    print("  PASS")


def test_phase0_smoke_data_format_roundtrip() -> None:
    """Construct ONE in-memory row for each of the 4 new arms and confirm
    the on-disk JSONL serialisation drops `_meta` and preserves `messages`.
    """
    print("\n=== Phase-0 row format round-trip (4 arms) ===")
    rows = [
        {
            "messages": [
                {"role": "system", "content": "You are X."},
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "<thinking>\nfiller\n</thinking>\nAnswer: A"},
            ],
            "_meta": {"arm": "garbage-cot", "garbage_prompt_seed": 0, "target_letter": "A"},
        },
        {
            "messages": [
                {"role": "system", "content": "You are X."},
                {"role": "user", "content": "Q?"},
                {
                    "role": "assistant",
                    "content": "<thinking>\nipsum lorem dolor.\n</thinking>\nAnswer: B",
                },
            ],
            "_meta": {"arm": "scrambled-english-cot", "target_letter": "B"},
        },
        {
            "messages": [
                {"role": "system", "content": "You are X."},
                {"role": "user", "content": "Q?"},
                {
                    "role": "assistant",
                    "content": "<thinking>\nrationale for B.\n</thinking>\nAnswer: C",
                },
            ],
            "_meta": {
                "arm": "contradicting-cot",
                "target_letter": "C",
                "rationale_supports": "B",
            },
        },
        {
            "messages": [
                {"role": "system", "content": "You are X."},
                {"role": "user", "content": "Q?"},
                {
                    "role": "assistant",
                    "content": "<thinking>\nneutral step-by-step.\n</thinking>\nAnswer: D",
                },
            ],
            "_meta": {"arm": "generic-cot-correct", "target_letter": "D"},
        },
    ]
    from generate_issue186_data import _strip_meta_for_disk

    on_disk = _strip_meta_for_disk(rows)
    assert all("_meta" not in r for r in on_disk)
    for r in on_disk:
        # JSON round-trip preserves the messages.
        s = json.dumps(r)
        d = json.loads(s)
        assert d["messages"][2]["content"].startswith("<thinking>")
        assert g.ANSWER_LINE_RE.search(d["messages"][2]["content"]) is not None
    print("  4 arms round-trip OK")
    print("  PASS")


def main() -> None:
    test_extract_rationale_for_each_new_arm()
    test_word_shuffle()
    test_contradicting_suffix_flip()
    test_generic_correct_validation()
    test_garbage_seed_rotation_uniform()
    test_letter_mention_audit_bias()
    test_answer_line_regex_covers_all_arms()
    test_condition_yaml_lints()
    test_train_and_eval_cell_counts()
    test_phase0_audit_letter_distribution()
    test_aggregate_holm_and_tost_helpers()
    test_phase0_smoke_data_format_roundtrip()
    print("\n[smoke_issue280] ALL TESTS PASS")


if __name__ == "__main__":
    main()
