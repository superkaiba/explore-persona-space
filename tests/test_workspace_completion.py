"""Final completion status, not cap history or fitted outcomes, selects test rows."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest
import torch

from explore_persona_space.analysis.workspace_comparison import paired_cohort
from explore_persona_space.analysis.workspace_completion import (
    final_rollout_eligibility,
    joint_completion_cohort,
)
from explore_persona_space.analysis.workspace_runtime import content_sha256, file_sha256, save_json


def draws():
    return [
        {"seed": seed, "token_ids": [12, 13, 99], "finish_reason": "stop", "max_new_tokens": 2048}
        for seed in range(42, 47)
    ]


def eligible(values):
    return final_rollout_eligibility(values, list(range(42, 47)), [99])


def test_completed_final_recovery_is_eligible():
    values = draws()
    values[0]["history"] = [{"finish_reason": "length"}]
    report = eligible(values)
    assert report["eligible"] and report["draws"][0]["answer_tokens"] == 2


@pytest.mark.parametrize("reason", ["length", "abort", None])
def test_present_but_unfinished_draw_is_excluded(reason):
    values = draws()
    values[2]["finish_reason"] = reason
    report = eligible(values)
    assert not report["eligible"]
    assert report["exclusions"] == [
        {"seed": 44, "reason": "unfinished_rollout", "finish_reason": reason}
    ]


def test_empty_terminal_span_and_missing_seed_are_visible():
    values = draws()[:-1]
    values[0]["token_ids"] = [99, 12]
    report = eligible(values)
    assert report["exclusions"] == [
        {"seed": 42, "reason": "empty_answer_span"},
        {"seed": 46, "reason": "missing_rollout"},
    ]


def test_duplicate_seed_rejected():
    values = draws()
    values[-1]["seed"] = 42
    with pytest.raises(ValueError, match="Duplicate"):
        eligible(values)


def test_both_models_share_joint_completion_filter_in_every_scope():
    complete = {
        "planned_context_ids": ["c", "b", "a"],
        "contexts": {context: eligible(draws()) for context in "abc"},
    }
    by_role = {"primary": copy.deepcopy(complete), "comparison": copy.deepcopy(complete)}
    capped = draws()
    capped[0]["finish_reason"] = "length"
    by_role["comparison"]["contexts"]["c"] = eligible(capped)
    ledger = joint_completion_cohort(by_role)
    assert ledger["joint_complete_context_ids"] == ["a", "b"]
    for cells in (
        {"primary": list("abc")},
        {"comparison": list("abc")},
        {"primary": list("abc"), "comparison": list("cba")},
    ):
        common, report = paired_cohort(cells, eligible_ids=ledger["joint_complete_context_ids"])
        assert common == ["a", "b"]
        assert all(cell["excluded_from_pairing"] == ["c"] for cell in report["cells"].values())


def test_completion_cannot_backfill_from_different_planned_prompts():
    report = {
        "planned_context_ids": list("abc"),
        "contexts": {context: eligible(draws()) for context in "abc"},
    }
    other = copy.deepcopy(report)
    other["planned_context_ids"] = list("abd")
    with pytest.raises(ValueError, match="exact frozen"):
        joint_completion_cohort({"primary": report, "comparison": other})


@pytest.fixture
def fitted_completion(tmp_path):
    api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/workspace_jr_compare.py"))
    raw_relative = "generations/main_test/one.json"
    values = draws()
    values[0]["finish_reason"] = "length"
    save_json(tmp_path / raw_relative, {"rollouts": values})
    raw_checksum = file_sha256(tmp_path / raw_relative)
    relative = "components/k10-rotationNone/main_test/one.pt"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    torch.save(
        {
            "prompt_sha256": "one",
            "contract": {"k": 10},
            "context_input_reference": {"generation_file_sha256": raw_checksum},
        },
        source,
    )
    checksum = file_sha256(source)
    receipt = tmp_path / "upload.json"
    save_json(
        receipt,
        {
            "repo": "superkaiba1/explore-persona-space-data",
            "revision": "a" * 40,
            "prefix": "exploratory_workspace_jr/unit-fixture",
            "files_verified": 2,
            "verified_sha256": {relative: checksum, raw_relative: raw_checksum},
        },
    )
    cell = {
        "entry": {"root": str(tmp_path), "upload_receipt": str(receipt), "role": "primary"},
        "ids": ["one"],
        "manifest": {
            "contract": {"k": 10},
            "coverage": {"test": {"file_sha256": {"one": checksum}}},
        },
    }
    report = {
        "source_proof": {"verified_sha256": {raw_relative: raw_checksum}},
        "contexts": {"one": eligible(values)},
        "terminal_policy": {"terminal_ids": [99]},
    }
    return api["bind_fitted_generation_sources"], cell, report


def test_same_producer_different_final_generation_cannot_relabel_fitted_target(fitted_completion):
    bind, cell, report = fitted_completion
    proof = bind(cell, report, list(range(42, 47)))
    assert (
        proof["one"]["component_sha256"]
        == cell["manifest"]["coverage"]["test"]["file_sha256"]["one"]
    )
    report["source_proof"]["verified_sha256"]["generations/main_test/one.json"] = (
        "same-producer-other-draw"
    )
    with pytest.raises(ValueError, match="different final generations"):
        bind(cell, report, list(range(42, 47)))


def test_eligibility_flag_cannot_override_unchanged_capped_final_draw(fitted_completion):
    bind, cell, report = fitted_completion
    report["contexts"]["one"]["eligible"] = True
    report["contexts"]["one"]["exclusions"] = []
    with pytest.raises(ValueError, match="eligibility differs from the actual fitted final draws"):
        bind(cell, report, list(range(42, 47)))


@pytest.mark.parametrize("break_count", [False, True])
def test_actual_generation_reader_checks_completed_counts(tmp_path, monkeypatch, break_count):
    api = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/workspace_jr_completion_cohort.py")
    )
    inspect = api["inspect_role"]
    monkeypatch.setitem(
        inspect.__globals__,
        "checkpoint_policy",
        lambda *_: ({"terminal_ids": [99]}, {"revision": "unit-checkpoint"}),
    )
    identity = {
        "config_sha256": "unit-config",
        "selection_sha256": "unit-selection",
        "model_role": "primary",
        "versions": {},
        "code": {"unit": True},
        "execution_readiness_sha256": "unit-ready",
    }
    config = {"generation": {"seeds": list(range(42, 47))}}
    planned, hashes = [], {}
    for prompt in ("unit alpha", "unit beta"):
        context = hashlib.sha256(prompt.encode()).hexdigest()
        planned.append(context)
        contract = {
            "identity": identity,
            "prompt_sha256": context,
            "generation": config["generation"],
        }
        path = tmp_path / "generations/main_test" / f"{context}.json"
        save_json(
            path,
            {
                "prompt": prompt,
                "contract": contract,
                "contract_sha256": content_sha256(contract),
                "rollouts": draws(),
            },
        )
        hashes[context] = file_sha256(path)
    save_json(
        tmp_path / "generations/main_test/generation_status.json",
        {
            "identity": identity,
            "status": "complete",
            "needs_cap_recovery": False,
            "planned_contexts": 2,
            "included_prompt_sha256": planned,
            "exclusions": [],
            "file_sha256": hashes,
            "n_contexts": 2,
            "n_rollouts": 11 if break_count else 10,
            "cap_hits": 0,
            "cap_hit_fraction": 0.0,
        },
    )
    save_json(tmp_path / "terminal.json", {"phase": "complete", "exit_code": 0})
    files = {
        str(p.relative_to(tmp_path)): file_sha256(p) for p in tmp_path.rglob("*") if p.is_file()
    }
    receipt = tmp_path / "receipt.json"
    save_json(
        receipt,
        {
            "repo": "superkaiba1/explore-persona-space-data",
            "revision": "a" * 40,
            "prefix": "exploratory_workspace_jr/unit-fixture",
            "files_verified": len(files),
            "verified_sha256": files,
        },
    )
    entry = {
        "root": str(tmp_path),
        "upload_receipt": str(receipt),
        "terminal_relative": "terminal.json",
    }
    if break_count:
        with pytest.raises(ValueError, match="counts differ"):
            inspect(entry, "primary", config, identity, planned, tmp_path / "analysis")
    else:
        result = inspect(entry, "primary", config, identity, planned, tmp_path / "analysis")
        assert result["eligible_contexts"] == 2 and result["draws"] == 10
        assert result["source_proof"]["upload"] == json.loads(receipt.read_text())
