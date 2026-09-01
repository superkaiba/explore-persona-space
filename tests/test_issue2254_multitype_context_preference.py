from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue2254_multitype_context_preference.py"
SPEC = importlib.util.spec_from_file_location("issue2254_multitype_context_preference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


def test_normalized_f_respects_anchor_gate() -> None:
    signal = np.asarray([30.0, 70.0, 50.0])
    floor = np.asarray([10.0, 20.0, 40.0])
    ceiling = np.asarray([50.0, 70.0, 55.0])
    values, valid = exp.normalized_f(signal, floor, ceiling, min_separation=20.0)
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_allclose(values[:2], [0.5, 1.0])
    assert np.isnan(values[2])


def test_exact_label_permutation_enumerates_all_assignments() -> None:
    values = {f"t{i}": float(i) for i in range(11)}
    result = exp.exact_label_permutation(values, {"t7", "t8", "t9", "t10"})
    assert result["n_assignments"] == 330
    assert result["observed"] > 0
    assert 0 < result["p_greater"] <= 1


def test_pair_generic_diffmean_cv_detects_separation() -> None:
    rng = np.random.default_rng(2254)
    a = rng.normal(size=(8, 3, 12))
    shift = np.zeros((3, 12))
    shift[:, 0] = 10.0
    b = a + shift
    report = exp.paired_diffmean_cv_report(b, a)
    assert report["n_pairs"] == 8
    assert all(row["heldout_auc_mean"] == 1.0 for row in report["layers"])


def test_history_aware_messages_preserve_every_turn() -> None:
    context = {
        "system": "system text",
        "history": [
            {"role": "user", "content": "earlier user"},
            {"role": "assistant", "content": "earlier assistant"},
        ],
        "user": "final user",
    }
    assert exp.context_messages(context) == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "earlier user"},
        {"role": "assistant", "content": "earlier assistant"},
        {"role": "user", "content": "final user"},
    ]


def test_frozen_assets_cover_all_types_and_reverse_format_contrast() -> None:
    payload = exp.build_assets()
    assert tuple(payload["target_order"]) == exp.TARGETS
    assert set(payload["targets"]) == set(exp.TARGETS)
    for target, asset in payload["targets"].items():
        assert len(asset["extraction_pairs"]) == 20, target
        assert len(asset["eval_pairs"]) == 12, target
        assert not (
            {row["pair_id"] for row in asset["extraction_pairs"]}
            & {row["pair_id"] for row in asset["eval_pairs"]}
        )
    format_pairs = payload["targets"]["format_policy"]["eval_pairs"]
    assert all("::v2-v1::" in row["pair_id"] for row in format_pairs)
    for target in ("prior_topic", "retrievable_fact", "icl_task", "user_expertise"):
        pair = payload["targets"][target]["extraction_pairs"][0]
        assert pair["a"]["history"] and pair["b"]["history"]
        assert exp.context_messages(pair["a"]) != exp.context_messages(pair["b"])


def test_screen_grid_is_balanced_and_unique() -> None:
    cells = exp.build_screen_cells()
    expected_per_target = 2 + len(exp.POSITIONS) * len(exp.BREADTHS) * len(exp.DOSE_SCALES)
    assert len(cells) == len(exp.TARGETS) * expected_per_target
    assert len({exp.cell_id(cell) for cell in cells}) == len(cells)
    for target in exp.TARGETS:
        target_cells = [cell for cell in cells if cell["target"] == target]
        assert len(target_cells) == expected_per_target


def test_short_label_quality_amendment_replaces_only_the_length_gate() -> None:
    def record(target: str, texts: list[str]) -> dict:
        return {
            "cell": {"target": target},
            "cap_hit_fraction": 0.0,
            "q_of_context": list(range(len(texts))),
            "seeds": {
                "42": {
                    "completions": [[text] for text in texts],
                    # The frozen prose check rejects intended one-word labels.
                    "coherent_flags": [[False] for _ in texts],
                }
            },
        }

    metrics, reasons = exp._quality(record("icl_task", ["young", "old", "calm"]))
    assert metrics["coherence_rate_programmatic"] == 0.0
    assert metrics["nonempty_nonrefusal_rate"] == 1.0
    assert "programmatic_coherence" not in reasons
    assert "nonempty_nonrefusal" not in reasons

    refusal_metrics, refusal_reasons = exp._quality(
        record("icl_task", ["I'm sorry", "I cannot", "As an AI"])
    )
    assert refusal_metrics["nonempty_nonrefusal_rate"] == 0.0
    assert "nonempty_nonrefusal" in refusal_reasons

    _metrics, prose_reasons = exp._quality(record("optimistic", ["hopeful", "bright"]))
    assert "programmatic_coherence" in prose_reasons


def test_pre_amendment_archive_is_complete_and_hash_verified() -> None:
    root = (
        SCRIPT.parents[1]
        / "eval_results/issue_2254/multitype_context_preference_qwen35"
    )
    rows = exp._verify_pre_amendment_archive(root)
    assert len(rows) == len(exp.build_screen_cells(("icl_task",))) == 32
    assert all(row["degradation_excluded"] for row in rows)

    protocol = exp._quality_protocol(
        {"cell": {"target": "icl_task"}, "cell_id": "icl_task__anchor_a"}, "screen"
    )
    archived = root / protocol["supersedes_pre_amendment_record"]
    assert archived.exists()
