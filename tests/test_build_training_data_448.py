# ruff: noqa: RUF002, RUF003
"""Tests for task #448 per-cell training-data builder.

Round-2 fix B3 regression test: building each of the 11 cells must succeed
without raising AssertionError or ValueError. Catches the round-1 bug where
the cross-persona negative-disjointness assertion made c9/c11 unrealizable
(1600 negative rows from an 850-row pool) and c6 probabilistically failure-
prone.

Uses a synthetic 850-pair union pool (no Anthropic calls, no HF). Patches
``persona_registry`` lazily because importing it touches HF.
"""

from __future__ import annotations

import os

import pytest

# Skip the persona_registry HF download at module import for this test —
# we don't need the bystander pairs to be PARSED; we just need the lookup
# functions to be callable. The cells that touch the extended-negative pool
# (10, 11) inject mock bystander lists via monkeypatch below.
os.environ.setdefault("EPM_ISSUE_448_SKIP_REGISTRY_BUILD", "1")


@pytest.fixture(scope="module")
def synthetic_union_pool() -> list[dict[str, str]]:
    """850-pair synthetic union pool matching the on-disk schema."""
    pool: list[dict[str, str]] = []
    for i in range(200):
        pool.append(
            {
                "question": f"cached question {i}?",
                "response": f"cached response body {i}.",
                "topic": "cached",
                "source": "cached",
            }
        )
    for i in range(650):
        pool.append(
            {
                "question": f"topup question {i}?",
                "response": f"topup response body {i}.",
                "topic": "science_natural",
                "source": "topup",
            }
        )
    return pool


@pytest.fixture(scope="module")
def patched_registry():
    """Monkey-patch persona_registry with stub bystanders for the test."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        persona_registry,
    )

    persona_registry.OBSERVED_BYSTANDERS_PER_SOURCE = {
        "villain": ["medical_doctor", "police_officer"],
    }
    persona_registry.BYSTANDER_PERSONA_PROMPTS = {
        "villain": "villain prompt",
        "medical_doctor": "medical_doctor prompt",
        "police_officer": "police_officer prompt",
        "comedian": "comedian prompt",
        "assistant": "assistant prompt",
        "software_engineer": "software_engineer prompt",
        "qwen_default": "qwen_default prompt",
        "kindergarten_teacher": "kindergarten_teacher prompt",
        "data_scientist": "data_scientist prompt",
        "librarian": "librarian prompt",
        "french_person": "french_person prompt",
        "zelthari_scholar": "zelthari_scholar prompt",
    }
    persona_registry.EXTENDED_CANDIDATE_POOL = [
        "software_engineer",
        "kindergarten_teacher",
        "data_scientist",
        "medical_doctor",
        "librarian",
        "french_person",
        "villain",
        "comedian",
        "police_officer",
        "zelthari_scholar",
        "assistant",
        "qwen_default",
    ]
    yield persona_registry


@pytest.mark.parametrize(
    "cell_slug,pos_ex,pos_n,neg_ex,neg_n",
    [
        ("c1_anchor", 200, 1, 200, 2),
        ("c2_pos_ex_100", 100, 1, 200, 2),
        ("c3_pos_ex_400", 400, 1, 200, 2),
        ("c4_pos_ex_800", 800, 1, 200, 2),
        ("c5_pos_personas_2", 200, 2, 200, 2),
        ("c6_pos_personas_4", 200, 4, 200, 2),
        ("c7_neg_ex_100", 200, 1, 100, 2),
        ("c8_neg_ex_400", 200, 1, 400, 2),
        ("c9_neg_ex_800", 200, 1, 800, 2),
        ("c10_neg_personas_4", 200, 1, 200, 4),
        ("c11_neg_personas_8", 200, 1, 200, 8),
    ],
)
def test_build_each_cell_succeeds(
    cell_slug,
    pos_ex,
    pos_n,
    neg_ex,
    neg_n,
    synthetic_union_pool,
    patched_registry,
    tmp_path,
):
    """Each cell builds without raising. Round-2 fix B3 regression."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_training_data import (
        build_cell,
    )

    out_path = tmp_path / cell_slug / "train_pool.jsonl"
    build_cell(
        cell_slug=cell_slug,
        pos_ex_per_persona=pos_ex,
        pos_personas=pos_n,
        neg_ex_per_persona=neg_ex,
        neg_personas=neg_n,
        output_path=out_path,
        union_pool=synthetic_union_pool,
    )
    assert out_path.exists()

    # Manifest invariants.
    import json

    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text())
    assert manifest["n_positive_rows"] == pos_ex * pos_n
    assert manifest["n_negative_rows"] == neg_ex * neg_n
    assert manifest["n_no_persona_rows"] == 100
    assert manifest["n_total_rows"] == pos_ex * pos_n + neg_ex * neg_n + 100


def test_c9_c11_negative_overlap_allowed(synthetic_union_pool, patched_registry, tmp_path):
    """c9 (1×800 = needs 1600 rows from 850-pool across 2 negs) and c11
    (8×200 = 1600 from 850-pool) only realize because cross-persona overlap
    is permitted. Verify the overlap is non-zero in c9 (where neg_ex/p=800
    forces it) and that the build succeeds."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_training_data import (
        build_cell,
    )

    out_path = tmp_path / "c9_neg_ex_800" / "train_pool.jsonl"
    build_cell(
        cell_slug="c9_neg_ex_800",
        pos_ex_per_persona=200,
        pos_personas=1,
        neg_ex_per_persona=800,
        neg_personas=2,
        output_path=out_path,
        union_pool=synthetic_union_pool,
    )
    import json

    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text())
    cross = manifest["neg_cross_persona_overlap_indices"]
    # Two negs × 800 each in an 850-pool means worst case 1500 distinct +
    # forced overlap ≥ 750. Sanity: at least one overlap pair exceeds 700.
    assert any(v > 700 for v in cross.values()), (
        f"Expected substantial cross-negative-persona overlap in c9; got {cross}"
    )


def test_pos_partition_disjoint_in_c6(synthetic_union_pool, patched_registry, tmp_path):
    """c6 has 4 positive personas × 200 each = 800 distinct rows. The
    partition-by-construction approach must yield ZERO overlap across the
    4 positive personas."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_training_data import (
        build_cell,
    )

    out_path = tmp_path / "c6_pos_personas_4" / "train_pool.jsonl"
    build_cell(
        cell_slug="c6_pos_personas_4",
        pos_ex_per_persona=200,
        pos_personas=4,
        neg_ex_per_persona=200,
        neg_personas=2,
        output_path=out_path,
        union_pool=synthetic_union_pool,
    )
    # Parse the rows back; for each positive persona, the (Q, response) pairs
    # are uniquely associated. Build per-persona question sets and verify
    # disjoint intersection.
    import json

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    by_persona: dict[str, set[str]] = {}
    for row in rows:
        sys_msg = next((m for m in row["prompt"] if m["role"] == "system"), None)
        if sys_msg is None:
            continue
        # Positive rows have a marker in the completion.
        comp = row["completion"][0]["content"]
        if " ※" not in comp:
            continue
        persona = sys_msg["content"]
        q = next(m["content"] for m in row["prompt"] if m["role"] == "user")
        by_persona.setdefault(persona, set()).add(q)
    assert len(by_persona) == 4, f"Expected 4 positive personas; got {len(by_persona)}"
    personas = list(by_persona.keys())
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            overlap = by_persona[personas[i]] & by_persona[personas[j]]
            assert len(overlap) == 0, (
                f"Positive personas {personas[i]!r} and {personas[j]!r} "
                f"share {len(overlap)} questions; partition expected disjoint."
            )
