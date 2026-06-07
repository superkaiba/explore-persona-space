"""Tests for the judge.py CLI added in round 2 (code-review Critical 6).

Round-1 had no `def main()` in judge.py, so the dispatcher's Phase 2.5
subprocess invocation `python -m ...judge --slab-root X --source Y --seed Z`
silently exited 0 without writing any judgments. The smoke gate then
hit `_read_source_self_rate` returning None and raised, but with NO
useful diagnostic.

These tests verify (a) `_load_panel_eval_json` parses the two known eval
JSON shapes, (b) `main` raises loud on missing inputs / missing API key,
(c) `main` writes the per-panel rates file the smoke gate reads.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest


def _write_panel_eval_json(path, panel: str, n_rollouts: int = 3) -> None:
    """Write a sycophancy_eval_<panel>.json in the 'claims'+'completions' shape."""
    payload = {
        "claims": [{"wrong_claim": f"claim {i} for {panel}"} for i in range(n_rollouts)],
        "completions": [
            [f"completion {i}.0 for {panel}", f"completion {i}.1 for {panel}"]
            for i in range(n_rollouts)
        ],
        "metadata": {"panel_persona": panel, "n_claims": n_rollouts},
    }
    path.write_text(json.dumps(payload))


def test_load_panel_eval_json_claims_completions_shape(tmp_path):
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import (
        _load_panel_eval_json,
    )

    p = tmp_path / "sycophancy_eval_test.json"
    _write_panel_eval_json(p, panel="test", n_rollouts=2)
    rollouts, metadata = _load_panel_eval_json(p)
    # 2 claims x 2 rollouts each = 4 (claim, completion) pairs.
    assert len(rollouts) == 4
    assert metadata["panel_persona"] == "test"
    for r in rollouts:
        assert "wrong_claim" in r
        assert "completion" in r


def test_load_panel_eval_json_rollouts_shape(tmp_path):
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import (
        _load_panel_eval_json,
    )

    p = tmp_path / "sycophancy_eval_test2.json"
    p.write_text(
        json.dumps(
            {
                "rollouts": [
                    {"wrong_claim": "x", "completion": "y"},
                    {"wrong_claim": "a", "completion": "b"},
                ],
                "metadata": {"panel_persona": "test2"},
            }
        )
    )
    rollouts, metadata = _load_panel_eval_json(p)
    assert len(rollouts) == 2
    assert rollouts[0] == {"wrong_claim": "x", "completion": "y"}
    assert metadata["panel_persona"] == "test2"


def test_load_panel_eval_json_unknown_shape_raises(tmp_path):
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import (
        _load_panel_eval_json,
    )

    p = tmp_path / "sycophancy_eval_broken.json"
    p.write_text(json.dumps({"unknown_key": []}))
    with pytest.raises(RuntimeError, match="unknown shape"):
        _load_panel_eval_json(p)


def test_main_raises_when_anthropic_key_missing(tmp_path):
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import main

    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"),
    ):
        main(
            [
                "--slab-root",
                str(tmp_path),
                "--source",
                "software_engineer",
                "--seed",
                "42",
            ]
        )


def test_main_raises_when_source_dir_missing(tmp_path):
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import main

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key-for-test"}, clear=False),
        pytest.raises(FileNotFoundError, match="Source eval dir"),
    ):
        main(
            [
                "--slab-root",
                str(tmp_path),
                "--source",
                "software_engineer",
                "--seed",
                "42",
            ]
        )


def test_main_raises_when_no_panel_jsons_without_allow_empty(tmp_path):
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import main

    src_dir = tmp_path / "software_engineer" / "seed_42"
    src_dir.mkdir(parents=True)
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key-for-test"}, clear=False),
        pytest.raises(FileNotFoundError, match="No sycophancy_eval_"),
    ):
        main(
            [
                "--slab-root",
                str(tmp_path),
                "--source",
                "software_engineer",
                "--seed",
                "42",
            ]
        )


def test_main_allow_empty_returns_zero(tmp_path):
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import main

    src_dir = tmp_path / "software_engineer" / "seed_42"
    src_dir.mkdir(parents=True)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key-for-test"}, clear=False):
        rc = main(
            [
                "--slab-root",
                str(tmp_path),
                "--source",
                "software_engineer",
                "--seed",
                "42",
                "--allow-empty",
            ]
        )
    assert rc == 0


def test_main_end_to_end_with_mocked_judge(tmp_path):
    """End-to-end: 3-row fixture, mock judge_batch, verify per_panel_rates_<source>.json
    is written with the correct schema for the dispatcher's _read_source_self_rate.
    """
    import explore_persona_space.experiments.sycophancy_implantation_411.judge as judge_mod
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import (
        JudgeVerdict,
        main,
    )

    src_dir = tmp_path / "software_engineer" / "seed_42"
    src_dir.mkdir(parents=True)
    # Write 1 fake panel JSON.
    _write_panel_eval_json(
        src_dir / "sycophancy_eval_software_engineer.json", panel="software_engineer", n_rollouts=3
    )

    # Mock judge_batch to return canned verdicts (2 YES, 4 NO out of 6).
    async def fake_judge_batch(rollouts, **kwargs):
        verdicts = []
        for i, r in enumerate(rollouts):
            verdicts.append(
                JudgeVerdict(
                    wrong_claim=r["wrong_claim"],
                    completion=r["completion"],
                    agreed=(i % 3 == 0),  # 2 of 6 will be True
                    raw_response="YES" if i % 3 == 0 else "NO",
                    model="mock-haiku",
                )
            )
        return verdicts

    with (
        patch.object(judge_mod, "judge_batch", fake_judge_batch),
        patch.dict(sys.modules, {}, clear=False),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key-for-test"}, clear=False),
    ):
        rc = main(
            [
                "--slab-root",
                str(tmp_path),
                "--source",
                "software_engineer",
                "--seed",
                "42",
                "--judge-model",
                "mock-haiku",
            ]
        )
    assert rc == 0

    # Per-panel rates file: this is what _read_source_self_rate reads.
    rates_path = src_dir / "per_panel_rates_software_engineer.json"
    assert rates_path.exists()
    payload = json.loads(rates_path.read_text())
    assert payload["source"] == "software_engineer"
    assert payload["seed"] == 42
    assert "per_panel_rate" in payload
    assert "software_engineer" in payload["per_panel_rate"]
    # 2 of 6 = 0.333...
    assert abs(payload["per_panel_rate"]["software_engineer"] - 2 / 6) < 1e-6

    # Per-panel verdict file checkpoint.
    verdicts_path = src_dir / "judgments" / "software_engineer_verdicts.json"
    assert verdicts_path.exists()
    verdict_payload = json.loads(verdicts_path.read_text())
    assert verdict_payload["panel"] == "software_engineer"
    assert len(verdict_payload["verdicts"]) == 6
