"""Independent actual CLI tests with a real tiny CPU numerical/source closure."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from explore_persona_space.analysis.workspace_decomposition import (
    ValidatedDictionary,
    decompose_context_nested,
)
from explore_persona_space.analysis.workspace_lenses import rotated_dictionary
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    save_json,
    save_tensors,
)

WT = Path(__file__).resolve().parents[1]


@pytest.fixture
def parity(tmp_path, monkeypatch):
    import huggingface_hub

    api = runpy.run_path(str(WT / "scripts/workspace_jr_checkpoint_parity.py"))
    globals_ = api["main"].__globals__
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").symlink_to(WT / "scripts", target_is_directory=True)
    (tmp_path / "src").symlink_to(WT / "src", target_is_directory=True)
    config = tmp_path / "configs/analysis/workspace_jr.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "schema_version: workspace-jr-v1\ngeneration:\n  seeds: [42, 43, 44, 45, 46]\n"
    )
    selection = tmp_path / "docs/exploratory_workspace_jr/selected_contexts.json"
    prompt = "a" * 64
    save_json(selection, {"subsets": {"main_train": [{"prompt_sha256": prompt}]}})
    monkeypatch.setitem(globals_, "CONFIG_SHA", file_sha256(config))
    monkeypatch.setitem(globals_, "SELECTION_SHA", file_sha256(selection))
    monkeypatch.setitem(
        globals_,
        "unchanged_sources",
        lambda: {
            "synthetic_fixture": "source compatibility independently reviewed outside this test"
        },
    )
    monkeypatch.setitem(
        globals_, "run_identity", lambda *args: {"synthetic_verifier_identity": True}
    )
    monkeypatch.setitem(globals_, "worker_fingerprint", lambda: {"synthetic_worker_fixture": True})
    identity = api["producer_identity"]()
    source, output = tmp_path / "source", tmp_path / "parity"
    source.mkdir()
    generation = source / f"generations/main_train/{prompt}.json"
    save_json(generation, {"synthetic": "generation bytes", "seeds": [42, 43, 44, 45, 46]})
    x = torch.tensor([1.0, 2.0, 3.0])
    canonical = source / "context_inputs/main_train/batch-0000.pt"
    save_tensors(
        canonical,
        {
            "x": x[None],
            "contract": {
                "identity": identity,
                "policy": "context_only_frozen_order_batches16_no_answer_tokens",
                "source_hashes": [file_sha256(generation)],
            },
        },
    )
    reference = {
        "identity": identity,
        "context_input_file": str(canonical.relative_to(source)),
        "context_input_file_sha256": file_sha256(canonical),
        "context_input_row": 0,
        "generation_file_sha256": file_sha256(generation),
    }
    rows = [
        {
            "seed": seed,
            "prompt_sha256": prompt,
            "x": x.clone(),
            "answer_states": (torch.arange((index + 1) * 3).reshape(index + 1, 3).float() - 4),
        }
        for index, seed in enumerate([42, 43, 44, 45, 46])
    ]
    capture = source / f"captures/main_train/{prompt}.pt"
    save_tensors(capture, {"rows": rows, "identity": reference})
    dictionary = torch.eye(3).repeat(10, 1)
    manifest = {"identity": identity, "pilot_only": False, "arms": {}}
    basis = {}
    for arm in ["J", "R"]:
        path = source / f"dictionaries/{arm}.pt"
        save_tensors(path, {"dictionary": dictionary})
        manifest["arms"][arm] = {"sha256": file_sha256(path)}
        basis[arm] = ValidatedDictionary(rotated_dictionary(dictionary, seed=20260914)[0])
    save_json(source / "dictionaries/manifest.json", manifest)
    actual = decompose_context_nested(rows, basis, checkpoints=(5, 10, 25), token_batch_size=128)
    for k, value in actual.items():
        contract = {"identity": identity, "dictionaries": manifest, "k": k, "rotation": 20260914}
        value.update(
            contract=contract,
            contract_sha256=content_sha256(contract),
            source_sha256=file_sha256(capture),
            context_input_reference=reference,
        )
        save_tensors(source / f"components/k{k}-rotation20260914/main_train/{prompt}.pt", value)
    hashes = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*") if p.is_file()}
    prefix = "exploratory_workspace_jr/20260912/comparison_main_rotation20260914_v1"
    upload = {
        "repo": "superkaiba1/explore-persona-space-data",
        "prefix": prefix,
        "revision": "a" * 40,
        "files_verified": len(hashes),
        "verified_sha256": hashes,
    }
    plan = tmp_path / "input_plan.json"
    save_json(
        plan,
        {
            "schema": "workspace-jr-checkpoint-parity-v1",
            "rotation": 20260914,
            "out": str(output),
            "prompt_sha256": prompt,
            "reference_upload": upload,
            "required_files": sorted(hashes),
        },
    )

    def download(*, repo_id, repo_type, revision, filename):
        assert (
            repo_id == upload["repo"] and repo_type == "dataset" and revision == upload["revision"]
        )
        assert filename.startswith(prefix + "/")
        return str(source / filename.removeprefix(prefix + "/"))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    def invoke(*, stage_only=False, wrong_rotation=False):
        monkeypatch.setattr(
            sys,
            "argv",
            [str(WT / "scripts/workspace_jr_checkpoint_parity.py"), "--plan", str(plan)]
            + (["--stage-only"] if stage_only else []),
        )
        if not stage_only:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
            monkeypatch.setattr(
                torch.cuda,
                "get_device_properties",
                lambda _: SimpleNamespace(
                    name="A100 synthetic CPU fixture", total_memory=80 * 2**30
                ),
            )
            monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
            monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
            monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
            monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)
            monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
            real_load = torch.load

            def cpu_load(*args, **kwargs):
                if kwargs.get("map_location") == "cuda:0":
                    kwargs["map_location"] = "cpu"
                return real_load(*args, **kwargs)

            monkeypatch.setattr(torch, "load", cpu_load)
            if wrong_rotation:
                monkeypatch.setitem(
                    globals_,
                    "rotated_dictionary",
                    lambda d, *, seed: rotated_dictionary(d, seed=20260915),
                )
        api["main"]()

    return invoke, plan, output


def test_actual_staging_then_complete_cpu_replay(parity):
    invoke, _, out = parity
    invoke(stage_only=True)
    assert (out / "staging_complete.json").exists()
    invoke()
    report = json.loads((out / "parity_report.json").read_text())
    complete = json.loads((out / "complete.json").read_text())
    assert report["status"] == "exact_numerical_parity"
    assert report["k"] == [5, 10, 25]
    assert report["staging_sha256"] == file_sha256(out / "staging_complete.json")
    assert complete["parity_report_sha256"] == file_sha256(out / "parity_report.json")


@pytest.mark.parametrize("stage_only", [False, True])
def test_completed_parity_is_immutable_even_if_plan_serialization_changes(parity, stage_only):
    invoke, plan, out = parity
    invoke()
    before = {str(p.relative_to(out)): file_sha256(p) for p in out.rglob("*") if p.is_file()}
    plan.write_text(json.dumps(json.loads(plan.read_text()), separators=(",", ":")))
    with pytest.raises(ValueError):
        invoke(stage_only=stage_only)
    assert before == {
        str(p.relative_to(out)): file_sha256(p) for p in out.rglob("*") if p.is_file()
    }


def test_numerical_failure_keeps_replay_evidence_without_success_marker(parity):
    invoke, _, out = parity
    with pytest.raises((ValueError, AssertionError)):
        invoke(wrong_rotation=True)
    assert not (out / "complete.json").exists()
    report = json.loads((out / "parity_report.json").read_text())
    assert report["status"] != "exact_numerical_parity"
    assert any(out.glob("*.pt")) or any(out.glob("*.npz"))
