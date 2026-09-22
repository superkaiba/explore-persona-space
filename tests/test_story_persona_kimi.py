"""Kimi-specific measurement, scope and paid-allocation regression checks."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.story_persona_crossmodel_analysis import NAMES, load_rates, paired_outcomes
from scripts.story_persona_crossmodel_capture import read_inputs
from scripts.story_persona_kimi_analysis import (
    feasible_rank_vectors,
    load_kimi_rates,
    rank_sensitivity,
)
from scripts.story_persona_kimi_runtime import install_hooks, residual_sum
from scripts.story_persona_storage_contract import allocation_seconds, contract_from_pod

ROOT = Path(__file__).resolve().parents[1]


def test_default_has_no_system_but_original_eight_are_byte_preserved():
    cfg = OmegaConf.load(ROOT / "configs/pilots/story_persona_crossmodel_capture.yaml")
    cfg.model_key = "kimi"
    cfg.prompts = "configs/pilots/story_persona_kimi_prompts.json"
    prompts, questions, rows = read_inputs(cfg)
    original = json.loads(
        (ROOT / "configs/pilots/story_persona_deepseek_prompts.json").read_text()
    )["prompts"]
    assert prompts[:8] == original
    assert len(rows) == 2160 and len(questions) == 240
    assert all([m["role"] for m in row["messages"]] == ["user"] for row in rows[1920:])
    assert all([m["role"] for m in row["messages"]] == ["system", "user"] for row in rows[:1920])


def test_raster_orders_include_touching_tie_without_strict_reversal():
    source = load_kimi_rates(ROOT / "eval_results/issue_2673/kimi_both/published_kimi_rates.json")
    rows = [source["rates"]["default"][name]["other"] for name in NAMES[3:]]
    intervals = tuple(
        (
            Fraction(427 - r["bar_top_y"], 403) - Fraction(3, 806),
            Fraction(427 - r["bar_top_y"], 403) + Fraction(3, 806),
        )
        for r in rows
    )
    ranks = feasible_rank_vectors(intervals)
    assert len(ranks) == 6
    assert np.any(ranks[:, 0] == ranks[:, 3])  # dismissive/peer may tie
    assert not np.any(ranks[:, 0] > ranks[:, 3])
    assert np.any(ranks[:, 2] > ranks[:, 4]) and np.any(ranks[:, 2] < ranks[:, 4])
    result = rank_sensitivity(
        range(5), [r["rate"] for r in rows], [r["digitization_bound"] for r in rows]
    )
    assert result["feasible_weak_orders"] == 6
    assert result["spearman_min"] < result["spearman_max"]
    assert (
        rank_sensitivity(
            [1] * 5, [r["rate"] for r in rows], [r["digitization_bound"] for r in rows]
        )["spearman_min"]
        is None
    )


def test_three_strata_use_their_own_rates_and_default_is_not_pooled():
    deep = load_rates(
        ROOT / "eval_results/issue_2673/deepseek_comparison/published_rates_and_overlap.json"
    )
    kimi = load_kimi_rates(ROOT / "eval_results/issue_2673/kimi_both/published_kimi_rates.json")
    matrix = np.full((9, 9), 0.3)
    np.fill_diagonal(matrix, 1)
    matrix[8, 3:8] = matrix[3:8, 8] = np.linspace(0.1, 0.9, 5)
    result = paired_outcomes(matrix, [*NAMES, "default"], {**deep["rates"], **kimi["rates"]})
    assert len(result["pairs"]) == 15
    assert result["supplementary_pooled"]["n"] == 10
    for persona in ("hhh", "fred", "default"):
        assert result["primary_by_persona"][persona]["direct_other_uptake"]["n"] == 5
    default = [r for r in result["pairs"] if r["evaluation_persona"] == "default"]
    assert [r["other_rate"] for r in default] == [
        kimi["rates"]["default"][p]["other"]["rate"] for p in NAMES[3:]
    ]


def kimi_ledger():
    return {
        "model_key": "kimi",
        "max_gpu_hours": 28,
        "allocations": [],
        "kimi_authorization": {
            "approved": True,
            "max_new_allocations": 1,
            "max_allocation_seconds": 12600,
            "user_request": "Do both. Run it now",
        },
    }


def test_kimi_cannot_reuse_deepseek_allowance_or_buy_second_allocation():
    ledger = kimi_ledger()
    assert allocation_seconds(ledger, pod_id="one", gpu_count=8, requested_seconds=12600) == 12600
    with pytest.raises(ValueError):
        allocation_seconds(ledger, pod_id="one", gpu_count=8, requested_seconds=12601)
    ledger["allocations"] = [{"pod_id": "old"}]
    with pytest.raises(ValueError, match="single-allocation"):
        allocation_seconds(ledger, pod_id="new", gpu_count=8, requested_seconds=12600)


def test_kimi_storage_contract_rejects_implicit_allocation_default():
    with pytest.raises(ValueError, match="separate allocation ledger"):
        contract_from_pod(
            {"id": "one", "name": "pod-2673-kimi", "desiredStatus": "RUNNING"},
            pod_id="one",
            expected_name="pod-2673-kimi",
            model_key="kimi",
            model={},
            source_sha="a" * 40,
            now=1000,
            ledger=kimi_ledger(),
        )


class Norm(torch.nn.Module):
    def forward(self, branch, residual):
        value = branch + residual
        return value, value


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = Norm()

    def forward(self, positions, hidden_states, residual):
        hidden, residual = self.input_layernorm(hidden_states, residual)
        return hidden * 0.5, residual


def test_vllm_pair_capture_matches_next_fused_norm_and_rejects_chunks():
    decoder = SimpleNamespace(
        layers=[Block() for _ in range(61)], config=SimpleNamespace(hidden_size=7168), norm=Norm()
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=decoder))
    install_hooks(model)
    state = model._eps_capture
    state.update(active=True, length=3, check_norm=True)
    hidden = torch.ones((3, 7168), dtype=torch.bfloat16) * 0.00001
    residual = torch.zeros_like(hidden)
    for block in decoder.layers:
        hidden, residual = block(torch.arange(3), hidden, residual)
    decoder.norm(hidden, residual)
    assert len(state["vectors"]) == 61 and len(state["norm_errors"]) == 61
    assert max(state["norm_errors"].values()) == 0
    assert torch.equal(state["vectors"][60], residual_sum((hidden, residual)))
    state.update(vectors={}, norm_errors={})
    with pytest.raises(RuntimeError, match="chunked prefill"):
        decoder.layers[0](torch.arange(1, 4), hidden, residual)
