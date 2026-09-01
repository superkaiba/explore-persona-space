from __future__ import annotations

import numpy as np

import scripts.issue2254_probe_context_followup as followup


def test_fit_probe_recovers_shared_direction_and_holds_out_instruction_pairs():
    rng = np.random.default_rng(7)
    n_pairs, n_questions, n_layers, hidden = 5, 6, 2, 12
    true = np.zeros((n_layers, hidden))
    true[:, :2] = [[2.0, -1.0], [-1.5, 2.5]]
    rows = []
    labels = []
    for label in (1.0, -1.0):
        for _pair in range(n_pairs):
            pair_nuisance = rng.normal(scale=0.3, size=(n_layers, hidden))
            for _question in range(n_questions):
                rows.append(label * true + pair_nuisance + rng.normal(scale=0.15, size=true.shape))
                labels.append(label)
    activations = np.stack(rows)
    directions, report = followup.fit_probe_directions(
        activations,
        n_pairs=n_pairs,
        n_questions=n_questions,
        lambda_grid=np.asarray([1e-3, 1e-1, 10.0]),
    )
    cosines = np.sum(directions * np.stack([followup._unit(v) for v in true]), axis=1)
    assert np.all(cosines > 0.9)
    assert report["cv"] == "leave-one-matched-instruction-pair-out"
    assert all(layer["heldout_auc_mean"] > 0.95 for layer in report["layers"])
    assert labels[: n_pairs * n_questions] == [1.0] * (n_pairs * n_questions)


def test_build_cells_is_complete_and_ids_are_unique():
    cells = followup.build_cells(["sycophancy"], n_random=4)
    assert len(cells) == 1 + 2 * 3 + 4 * 3
    ids = [followup.cell_id(cell) for cell in cells]
    assert len(ids) == len(set(ids))
    assert sum(cell["kind"] == "signal" for cell in cells) == 6
    assert sum(cell["kind"] == "random" for cell in cells) == 12
    shards = [cells[shard_id::8] for shard_id in range(8)]
    assert sorted(followup.cell_id(cell) for shard in shards for cell in shard) == sorted(ids)


def test_followup_signal_schema_does_not_claim_parent_direction_slug():
    cell = next(
        cell for cell in followup.build_cells(["evil"], n_random=2) if cell["kind"] == "signal"
    )
    assert "method" in cell
    assert "direction" not in cell
    assert followup.cell_id(cell).startswith("evil__diffmean__")


def test_auc_midrank_ties():
    labels = np.asarray([1, 1, -1, -1])
    assert followup._auc(np.asarray([1.0, 1.0, 0.0, 0.0]), labels) == 1.0
    assert followup._auc(np.zeros(4), labels) == 0.5


def test_judge_item_ids_are_batch_safe_and_unique_across_the_full_grid():
    ids = {
        followup._judge_item_id(followup.cell_id(cell), seed, question_index, 0)
        for cell in followup.build_cells(followup.BEHAVIORS, n_random=8)
        for seed in followup.GEN_SEEDS
        for question_index in range(20)
    }
    assert len(ids) == 93 * 5 * 20
    assert all(len(item_id) <= 53 for item_id in ids)
    assert all(item_id.replace("_", "").isalnum() for item_id in ids)
    assert followup.JUDGE_THRESHOLD_BASE_BATCH == 0


def test_reduce_excludes_cells_above_the_existing_cap_hit_threshold():
    behavior = "sycophancy"

    def row(cell, score, *, cap=0.0):
        cid = followup.cell_id(cell)
        return {
            "cell_id": cid,
            "cell": cell,
            "per_question_mean_score": [float(score)] * 20,
            "coherence_rate_programmatic": 1.0,
            "cap_hit_fraction": cap,
            "cjk_fraction": 0.0,
            "accounting": {"frac_items_complete": 1.0},
        }

    rows = [row({"behavior": behavior, "kind": "alpha0"}, 0.0)]
    for method in followup.METHODS:
        for breadth, score in zip(followup.BREADTHS, (1.0, 2.0, 99.0), strict=True):
            rows.append(
                row(
                    {
                        "behavior": behavior,
                        "kind": "signal",
                        "method": method,
                        "breadth": breadth,
                        "layer_config": "all" if breadth == "all" else "mid",
                        "c": 4.0,
                    },
                    score,
                    cap=0.89 if breadth == "all" else 0.0,
                )
            )
    for random_seed in range(4):
        for breadth in followup.BREADTHS:
            rows.append(
                row(
                    {
                        "behavior": behavior,
                        "kind": "random",
                        "method": "random",
                        "random_seed": random_seed,
                        "breadth": breadth,
                        "layer_config": "mid",
                        "c": 2.0,
                    },
                    0.1,
                )
            )

    reduced = followup.reduce_behavior(rows, behavior)
    for method in followup.METHODS:
        selected = reduced["methods"][method]["selected_cell"]
        assert selected["cell"]["breadth"] == "mid"
        capped = reduced["methods"][method]["all_breadths"]["all"]
        assert not capped["selection_eligible"]
        assert capped["selection_exclusion_reasons"] == ["generation_cap_hits"]
    assert reduced["completeness"]["cap_hit_fraction_ceiling"] == 0.02

    for judged in rows:
        if judged["cell"].get("kind") == "signal" and judged["cell"].get("breadth") == "mid":
            judged["cjk_fraction"] = 0.5
    language_gated = followup.reduce_behavior(
        rows,
        behavior,
        cjk_fraction_ceiling=0.2,
    )
    for method in followup.METHODS:
        assert language_gated["methods"][method]["selected_cell"]["cell"]["breadth"] == "single"
        mid = language_gated["methods"][method]["all_breadths"]["mid"]
        assert mid["selection_exclusion_reasons"] == ["cjk_language_switching"]


def test_reduce_can_report_no_quality_eligible_operating_point():
    behavior = "sycophancy"

    def row(cell, score, *, cjk=0.0):
        return {
            "cell_id": followup.cell_id(cell),
            "cell": cell,
            "per_question_mean_score": [score] * 20,
            "coherence_rate_programmatic": 1.0,
            "cap_hit_fraction": 0.0,
            "cjk_fraction": cjk,
            "accounting": {"frac_items_complete": 1.0 if score is not None else 0.0},
        }

    rows = [row({"behavior": behavior, "kind": "alpha0"}, 0.0)]
    for method in followup.METHODS:
        for breadth in followup.BREADTHS:
            rows.append(
                row(
                    {
                        "behavior": behavior,
                        "kind": "signal",
                        "method": method,
                        "breadth": breadth,
                        "layer_config": breadth,
                        "c": 2.0,
                    },
                    None,
                    cjk=1.0,
                )
            )
    for random_seed in range(4):
        for breadth in followup.BREADTHS:
            rows.append(
                row(
                    {
                        "behavior": behavior,
                        "kind": "random",
                        "method": "random",
                        "random_seed": random_seed,
                        "breadth": breadth,
                        "layer_config": breadth,
                        "c": 2.0,
                    },
                    None,
                    cjk=1.0,
                )
            )

    reduced = followup.reduce_behavior(
        rows,
        behavior,
        cjk_fraction_ceiling=0.2,
        allow_no_eligible=True,
    )
    assert reduced["chance"]["status"] == "insufficient_quality_eligible_random_directions"
    assert reduced["method_comparison"]["status"] == "unavailable_no_quality_eligible_cell"
    for method in followup.METHODS:
        assert reduced["methods"][method]["status"] == "no_quality_eligible_cell"
        assert reduced["methods"][method]["selected_cell_id"] is None
        assert all(
            breadth["delta_score"] is None
            for breadth in reduced["methods"][method]["all_breadths"].values()
        )
