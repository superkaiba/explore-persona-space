from __future__ import annotations

import json

import numpy as np

import scripts.issue2254_probe_context_followup as base
import scripts.issue2254_probe_context_qwen35 as q35
import scripts.issue2254_probe_context_qwen35_personas as personas


def test_additional_persona_assets_are_pinned_complete_and_disjoint():
    assert personas.TRAITS == ("optimistic", "impolite", "apathetic", "humorous")
    for trait in personas.TRAITS:
        asset = personas.load_trait_asset(trait)
        assert len(asset["instruction"]) == 5
        assert len(asset["extraction_questions"]) == 20
        assert len(asset["eval_questions"]) == 20
        assert not set(asset["extraction_questions"]) & set(asset["eval_questions"])
        assert "{question}" in asset["eval_prompt"]
        assert "{answer}" in asset["eval_prompt"]


def test_additional_persona_operating_points_are_shared_and_frozen():
    assert personas.OPERATING_POINTS == {
        "single": {
            "layer_config": "L23",
            "c": 2.0,
            "source": "preregistered:depth-matched-q25-L20",
        },
        "mid": {
            "layer_config": "mid",
            "c": 2.0,
            "source": "preregistered:prior-q35-hallucination-mid",
        },
        "all": {
            "layer_config": "all",
            "c": 4.0,
            "source": "preregistered:prior-q35-sycophancy-all",
        },
    }
    assert q35.LAYER_CONFIGS["L23"] == (23,)
    assert q35.LAYER_CONFIGS["mid"] == (16, 20, 23)
    assert q35.LAYER_CONFIGS["all"] == tuple(range(32))


def test_additional_persona_grid_is_complete_unique_and_matched():
    cells = personas.build_cells(n_random=8)
    assert len(cells) == 4 * (1 + 3 * len(base.METHODS) + 3 * 8) == 124
    ids = [base.cell_id(cell) for cell in cells]
    assert len(ids) == len(set(ids))
    for trait in personas.TRAITS:
        trait_cells = [cell for cell in cells if cell["behavior"] == trait]
        assert len(trait_cells) == 31
        for breadth, point in personas.OPERATING_POINTS.items():
            comparable = [
                cell
                for cell in trait_cells
                if cell.get("breadth") == breadth and cell["kind"] != "alpha0"
            ]
            assert len(comparable) == len(base.METHODS) + 8
            assert {cell["layer_config"] for cell in comparable} == {point["layer_config"]}
            assert {cell["c"] for cell in comparable} == {point["c"]}
            assert {cell["position"] for cell in comparable} == {"context"}


def test_answer_grid_changes_only_the_position_factor():
    context_cells = personas.build_cells(n_random=8, position="context")
    answer_cells = personas.build_cells(n_random=8, position="answer")
    assert [base.cell_id(cell) for cell in context_cells] == [
        base.cell_id(cell) for cell in answer_cells
    ]
    for context_cell, answer_cell in zip(context_cells, answer_cells, strict=True):
        assert context_cell["position"] == "context"
        assert answer_cell["position"] == "answer"
        assert {k: v for k, v in context_cell.items() if k != "position"} == {
            k: v for k, v in answer_cell.items() if k != "position"
        }


def test_additional_persona_shards_partition_every_cell_once():
    cells = personas.build_cells(n_random=8)
    shards = [cells[index::4] for index in range(4)]
    assert sorted(base.cell_id(cell) for shard in shards for cell in shard) == sorted(
        base.cell_id(cell) for cell in cells
    )

    trait_shards = [
        personas._select_generation_shard(
            cells,
            list(personas.TRAITS),
            shard_id=index,
            num_shards=4,
            strategy="trait",
        )
        for index in range(4)
    ]
    assert [len(shard) for shard in trait_shards] == [31, 31, 31, 31]
    assert [
        {cell["behavior"] for cell in shard} for shard in trait_shards
    ] == [{trait} for trait in personas.TRAITS]
    assert sorted(base.cell_id(cell) for shard in trait_shards for cell in shard) == sorted(
        base.cell_id(cell) for cell in cells
    )


def test_position_comparison_bootstrap_reselects_within_each_arm(tmp_path):
    trait = "optimistic"
    method = "diffmean"

    def make_arm(root, single_score, mid_score):
        judged = root / "judge/judged"
        judged.mkdir(parents=True)
        (judged / f"{trait}__a0.json").write_text(
            json.dumps({"per_question_mean_score": [0.0] * 20})
        )
        cells = {}
        for breadth, score in (("single", single_score), ("mid", mid_score)):
            cid = f"{trait}__{method}__{breadth}"
            cell = {"kind": "signal", "method": method, "breadth": breadth}
            (judged / f"{cid}.json").write_text(
                json.dumps({"per_question_mean_score": [float(score)] * 20})
            )
            cells[cid] = {"cell": cell, "selection_eligible": True}
        selected_id = f"{trait}__{method}__mid"
        summary = {
            "behaviors": {
                trait: {
                    "cells": cells,
                    "methods": {
                        method: {
                            "selected_cell_id": selected_id,
                            "selected_cell": {
                                "cell": cells[selected_id]["cell"],
                                "delta_score": float(mid_score),
                                "coherence_rate_programmatic": 1.0,
                                "cap_hit_fraction": 0.0,
                                "cjk_fraction": 0.0,
                            },
                        }
                    },
                }
            }
        }
        return summary

    context_root = tmp_path / "context"
    answer_root = tmp_path / "answer"
    context_summary = make_arm(context_root, 1.0, 2.0)
    answer_summary = make_arm(answer_root, 3.0, 4.0)
    indices = base._bootstrap_indices(20, trait)
    context = personas._position_method_boot(
        context_root, context_summary, trait, method, indices
    )
    answer = personas._position_method_boot(
        answer_root, answer_summary, trait, method, indices
    )
    assert context is not None and answer is not None
    assert context["selected_breadth"] == answer["selected_breadth"] == "mid"
    contrast = (
        answer["selection_inherited_bootstrap"]
        - context["selection_inherited_bootstrap"]
    )
    np.testing.assert_allclose(contrast, 2.0)
