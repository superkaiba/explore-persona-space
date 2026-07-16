"""#1332 r3 — frozen #545 demo-pool protocol (concern ood545-coverage-partial).

Pins ``demo_pool_from_corpus_rows`` (extracted from
``behavior_testbed_545.corpora.build_demo_sets`` so the #1332 OOD arm's
missing-pool regeneration shares the exact recipe): determinism under seed
545, K=8 tercile stratification, corpus-row parsing shapes, and byte-level
equivalence with the pre-refactor inline implementation.
"""

from __future__ import annotations

import random

from explore_persona_space.experiments.behavior_testbed_545.corpora import (
    demo_pool_from_corpus_rows,
)


def _completion_row(q: str, a: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": q}],
        "completion": [{"role": "assistant", "content": a}],
    }


def _messages_row(q: str, a: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def _pre_refactor_inline(rows: list[dict], k: int = 8) -> list[dict]:
    """VERBATIM copy of the pre-refactor build_demo_sets sampling block."""
    scored = []
    for r in rows:
        if "completion" in r:
            q = r["prompt"][-1]["content"]
            a = r["completion"][0]["content"]
        elif "messages" in r:
            msgs = r["messages"]
            q = next(m["content"] for m in msgs if m["role"] == "user")
            a = next(m["content"] for m in reversed(msgs) if m["role"] == "assistant")
        else:
            continue
        scored.append((len(a.split()), q, a))
    assert len(scored) >= k
    scored.sort()
    terciles = [
        scored[: len(scored) // 3],
        scored[len(scored) // 3 : 2 * len(scored) // 3],
        scored[2 * len(scored) // 3 :],
    ]
    rng = random.Random(545)
    demos = []
    ti = 0
    while len(demos) < k:
        t = terciles[ti % 3]
        _, q, a = t[rng.randrange(len(t))]
        demos.append({"question": q, "answer": a})
        ti += 1
    return demos


def test_happy_path_deterministic_k8_tercile_stratified():
    # answer lengths 1..30 words -> terciles are lengths 1-10 / 11-20 / 21-30
    rows = [_completion_row(f"q{i}", "w " * (i + 1)) for i in range(30)]
    demos1, n1 = demo_pool_from_corpus_rows(rows)
    demos2, n2 = demo_pool_from_corpus_rows(rows)
    assert n1 == n2 == 30
    assert demos1 == demos2  # deterministic (seed 545)
    assert len(demos1) == 8
    pairs = {(f"q{i}", ("w " * (i + 1))) for i in range(30)}
    assert all((d["question"], d["answer"]) in pairs for d in demos1)
    # tercile cycling ti % 3 over k=8 draws -> >=2 demos from EACH tercile
    buckets = [0, 0, 0]
    for d in demos1:
        n_words = len(d["answer"].split())
        buckets[min((n_words - 1) // 10, 2)] += 1
    assert all(b >= 2 for b in buckets), buckets


def test_fewer_than_k_parsable_rows_returns_none():
    rows = [_completion_row(f"q{i}", f"answer {i}") for i in range(5)]
    demos, n = demo_pool_from_corpus_rows(rows)
    assert demos is None
    assert n == 5


def test_messages_shape_parsed_and_unparseable_rows_skipped():
    rows = [_messages_row(f"q{i}", "tok " * (i + 1)) for i in range(10)]
    rows.insert(3, {"unrelated": "junk"})  # skipped, not counted
    demos, n = demo_pool_from_corpus_rows(rows)
    assert n == 10
    assert demos is not None and len(demos) == 8
    assert all(d["question"].startswith("q") for d in demos)


def test_matches_pre_refactor_inline_implementation():
    rng = random.Random(0)
    rows: list[dict] = []
    for i in range(57):
        a = " ".join(["tok"] * rng.randrange(1, 400))
        rows.append(_completion_row(f"q{i}", a) if i % 2 else _messages_row(f"q{i}", a))
    expect = _pre_refactor_inline(rows)
    got, n = demo_pool_from_corpus_rows(rows)
    assert n == 57
    assert got == expect
