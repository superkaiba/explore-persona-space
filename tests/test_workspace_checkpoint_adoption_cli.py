"""Independent actual adoption reader/publisher tests. GPU and Hub are leaf seams only."""

import copy
import json
import runpy
import shutil
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
parity = runpy.run_path(str(WT / "tests/test_workspace_checkpoint_parity_cli.py"))["parity"]


def upload(root, prefix, revision):
    hashes = {
        p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()
    }
    return {
        "repo": "superkaiba1/explore-persona-space-data",
        "prefix": "exploratory_workspace_jr/20260912/" + prefix,
        "revision": revision * 40,
        "files_verified": len(hashes),
        "verified_sha256": hashes,
    }


@pytest.fixture
def adoption(parity, tmp_path, monkeypatch):
    import huggingface_hub

    invoke, parity_plan_path, parity_root = parity
    invoke()
    monkeypatch.syspath_prepend(str(WT / "scripts"))
    api = runpy.run_path(str(WT / "scripts/workspace_jr_adopt_checkpoints.py"))
    g = api["main"].__globals__
    # Wrapper bytes and disk capacity are explicit leaf fixtures; the reader verifies their hashes.
    supervisor_fixture = tmp_path / "synthetic_supervisor.py"
    launcher_fixture = tmp_path / "synthetic_launcher.sh"
    supervisor_fixture.write_text("# synthetic supervisor fixture\n")
    launcher_fixture.write_text("# synthetic launcher fixture\n")
    monkeypatch.setitem(g, "SUPERVISOR_SHA", file_sha256(supervisor_fixture))
    monkeypatch.setitem(g, "LAUNCHER_SHA", file_sha256(launcher_fixture))
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30))
    identity = json.loads((parity_root / "staging_complete.json").read_text())["producer_identity"]
    monkeypatch.setitem(g, "producer_identity", lambda: copy.deepcopy(identity))
    source_hashes = json.loads((parity_root / "staging_complete.json").read_text())[
        "numerical_source_sha256"
    ]
    monkeypatch.setitem(g, "unchanged_sources", lambda: copy.deepcopy(source_hashes))
    config = tmp_path / "configs/analysis/workspace_jr.yaml"
    selection = tmp_path / "docs/exploratory_workspace_jr/selected_contexts.json"
    selected = json.loads(selection.read_text())
    selected["subsets"].update(
        {
            subset: [{"prompt_sha256": char * 64}]
            for subset, char in [("main_validation", "b"), ("main_test", "c")]
        }
    )
    save_json(selection, selected)
    monkeypatch.setitem(g, "CONFIG_SHA", file_sha256(config))
    monkeypatch.setitem(g, "SELECTION_SHA", file_sha256(selection))
    report = json.loads((parity_root / "parity_report.json").read_text())
    # Explicit synthetic verifier metadata, while all numerical replay and hash chains are real.
    report["verifier_identity"] = {
        **identity,
        "code": {
            "git_commit": "d" * 40,
            "git_dirty": False,
            "git_argv0_state": "tracked",
            "git_argv0_path": "scripts/workspace_jr_checkpoint_parity.py",
            "phase": "workspace-jr",
        },
    }
    save_json(parity_root / "parity_report.json", report)
    save_json(
        parity_root / "complete.json",
        {
            "status": "complete",
            "parity_report_sha256": file_sha256(parity_root / "parity_report.json"),
        },
    )
    save_json(parity_root / "terminal.json", {"exit_code": 0, "phase": "complete"})
    parity_upload = upload(parity_root, "synthetic_parity", "e")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    shutil.copytree(tmp_path / "source/dictionaries", foreign / "dictionaries")
    manifest = json.loads((foreign / "dictionaries/manifest.json").read_text())
    bases = {
        arm: ValidatedDictionary(
            rotated_dictionary(
                torch.load(foreign / f"dictionaries/{arm}.pt", weights_only=True)["dictionary"],
                seed=20260914,
            )[0]
        )
        for arm in ("J", "R")
    }
    foreign_out = "/workspace/workspace_jr/comparison_main_precompute_rotation20260914_v1"
    recipient = tmp_path / "comparison_main_rotation20260914_v1"
    recipient.mkdir()
    pending = []
    for subset, char in [("main_validation", "b"), ("main_test", "c")]:
        prompt = char * 64
        generation = foreign / f"generations/{subset}/{prompt}.json"
        save_json(generation, {"synthetic_raw": char})
        x = torch.tensor([1.0, 2.0, 3.0])
        canonical = foreign / f"context_inputs/{subset}/batch-0000.pt"
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
            "context_input_file": canonical.relative_to(foreign).as_posix(),
            "context_input_file_sha256": file_sha256(canonical),
            "context_input_row": 0,
            "generation_file_sha256": file_sha256(generation),
        }
        rows = [
            {
                "seed": seed,
                "prompt_sha256": prompt,
                "x": x.clone(),
                "answer_states": torch.arange((i + 1) * 3).reshape(i + 1, 3).float() - 4,
            }
            for i, seed in enumerate(range(42, 47))
        ]
        capture = foreign / f"captures/{subset}/{prompt}.pt"
        save_tensors(capture, {"identity": reference, "rows": rows})
        coverage = {
            "identity": identity,
            "status": "complete",
            "planned_contexts": 1,
            "included_prompt_sha256": [prompt],
            "exclusions": [],
            "file_sha256": {prompt: file_sha256(capture)},
        }
        save_json(foreign / f"captures/{subset}/coverage.json", coverage)
        for k, value in decompose_context_nested(
            rows, bases, checkpoints=(5, 10, 25), token_batch_size=128
        ).items():
            contract = {
                "identity": identity,
                "dictionaries": manifest,
                "k": k,
                "rotation": 20260914,
            }
            value.update(
                contract=contract,
                contract_sha256=content_sha256(contract),
                source_sha256=file_sha256(capture),
                context_input_reference=reference,
            )
            path = foreign / f"components/k{k}-rotation20260914/{subset}/{prompt}.pt"
            save_tensors(path, value)
            save_json(
                path.parent / "coverage.json",
                {**coverage, "contract": contract, "file_sha256": {prompt: file_sha256(path)}},
            )
        pending.append(subset)
    original_upload = upload(foreign, "synthetic_original_inputs", "f")
    splits = []
    for subset in pending:
        original = {
            "role": "comparison",
            "rotation": 20260914,
            "source_sha": g["PRODUCER"],
            "readiness_sha256": g["READINESS_SHA"],
            "subset": subset,
            "out": str(recipient),
            "sources": [{"upload": original_upload, "local_root": "/synthetic/hint"}],
        }
        original_path = tmp_path / f"{subset}_original_plan.json"
        save_json(original_path, original)
        operation = f"decomposition_operations/decompose-{subset}-rotation20260914"
        save_json(
            foreign / operation / "contract.json",
            {
                "plan": api["normalized_foreign_plan"](original, foreign_out),
                "supervisor_sha256": g["SUPERVISOR_SHA"],
                "launcher_sha256": g["LAUNCHER_SHA"],
            },
        )
        attempt = foreign / operation / "attempts/attempt-0000"
        save_json(attempt / "exit.json", {"exit_code": 0, "phase": "complete"})
        shutil.copyfile(supervisor_fixture, attempt / "supervisor.py")
        shutil.copyfile(launcher_fixture, attempt / "launcher.sh")
        splits.append(
            {
                "original": original,
                "original_plan_path": str(original_path),
                "original_plan_sha256": file_sha256(original_path),
                "terminal": (attempt / "exit.json").relative_to(foreign).as_posix(),
            }
        )
    save_json(
        foreign / "cache_execution_provenance.json",
        {
            "execution_worker": report["execution_worker"],
            "parity_report_sha256": file_sha256(parity_root / "parity_report.json"),
            "producer_sha": g["PRODUCER"],
            "rotation": 20260914,
        },
    )
    foreign_upload = upload(foreign, Path(foreign_out).name, "9")
    plan = {
        "schema": "workspace-jr-checkpoint-adoption-v1",
        "rotation": 20260914,
        "recipient": str(recipient),
        "staging": str(tmp_path / "staging"),
        "staging_bytes_ceiling": 1024**2,
        "parity": {
            "upload": parity_upload,
            "plan_sha256": file_sha256(parity_plan_path),
            "terminal": "terminal.json",
        },
        "source_upload": foreign_upload,
        "splits": splits,
    }
    plan_path = tmp_path / "adoption_plan.json"
    save_json(plan_path, plan)
    sources = [(parity_upload, parity_root), (foreign_upload, foreign)]

    def download(*, repo_id, repo_type, revision, filename):
        assert repo_id == foreign_upload["repo"] and repo_type == "dataset"
        for receipt, root in sources:
            if revision == receipt["revision"] and filename.startswith(receipt["prefix"] + "/"):
                return str(root / filename.removeprefix(receipt["prefix"] + "/"))
        raise AssertionError("Unpinned download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    def run(verify_only=False):
        monkeypatch.setattr(
            sys,
            "argv",
            [str(WT / "scripts/workspace_jr_adopt_checkpoints.py"), "--plan", str(plan_path)]
            + (["--verify-only"] if verify_only else []),
        )
        api["main"]()

    return run, plan, plan_path, recipient, foreign, parity_root


def test_actual_adoption_body_validates_before_publishing(adoption):
    run, _plan, _, recipient, foreign, _ = adoption
    run(verify_only=True)
    assert list(recipient.iterdir()) == []
    run()
    paths = sorted((recipient / "components").rglob("*.pt"))
    assert len(paths) == 6
    assert not list(recipient.rglob("coverage.json"))
    assert not (recipient / "decomposition_operations").exists()
    for path in paths:
        assert file_sha256(path) == file_sha256(foreign / path.relative_to(recipient))
    lineage = list((recipient / "checkpoint_adoptions").glob("*/lineage.json"))
    assert len(lineage) == 1 and len(list(lineage[0].parent.glob("[0-9]*.json"))) == 6
    before = {
        p.relative_to(recipient).as_posix(): file_sha256(p)
        for p in recipient.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="immutable"):
        run()
    assert before == {
        p.relative_to(recipient).as_posix(): file_sha256(p)
        for p in recipient.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "worker",
        "terminal",
        "parity_dictionary",
        "staging_map",
        "verifier",
        "component_bytes",
        "original_plan",
    ],
)
def test_actual_adopter_rejects_before_publication(adoption, mutation):
    run, plan, path, recipient, foreign, parity_root = adoption
    if mutation == "worker":
        item = foreign / "cache_execution_provenance.json"
        value = json.loads(item.read_text())
        value["execution_worker"] = {"different_worker": True}
        save_json(item, value)
        plan["source_upload"]["verified_sha256"][item.relative_to(foreign).as_posix()] = (
            file_sha256(item)
        )
    elif mutation == "terminal":
        item = foreign / plan["splits"][1]["terminal"]
        value = json.loads(item.read_text())
        value["exit_code"] = 1
        save_json(item, value)
        plan["source_upload"]["verified_sha256"][item.relative_to(foreign).as_posix()] = (
            file_sha256(item)
        )
    elif mutation == "parity_dictionary":
        # Staging agrees with the altered reference map but differs from queued dictionaries.
        ref = json.loads((parity_root / "plan.json").read_text())
        ref["reference_upload"]["verified_sha256"]["dictionaries/J.pt"] = "1" * 64
        save_json(parity_root / "plan.json", ref)
        staging = json.loads((parity_root / "staging_complete.json").read_text())
        staging["plan_sha256"] = file_sha256(parity_root / "plan.json")
        staging["required_files"]["dictionaries/J.pt"] = "1" * 64
        save_json(parity_root / "staging_complete.json", staging)
        report = json.loads((parity_root / "parity_report.json").read_text())
        report["staging_sha256"] = file_sha256(parity_root / "staging_complete.json")
        save_json(parity_root / "parity_report.json", report)
        save_json(
            parity_root / "complete.json",
            {
                "status": "complete",
                "parity_report_sha256": file_sha256(parity_root / "parity_report.json"),
            },
        )
        plan["parity"]["plan_sha256"] = file_sha256(parity_root / "plan.json")
        worker = json.loads((foreign / "cache_execution_provenance.json").read_text())
        worker["parity_report_sha256"] = file_sha256(parity_root / "parity_report.json")
        save_json(foreign / "cache_execution_provenance.json", worker)
        plan["source_upload"]["verified_sha256"]["cache_execution_provenance.json"] = file_sha256(
            foreign / "cache_execution_provenance.json"
        )
        plan["parity"]["upload"] = upload(parity_root, "synthetic_parity", "e")
    elif mutation in ("staging_map", "verifier"):
        report = json.loads((parity_root / "parity_report.json").read_text())
        if mutation == "staging_map":
            item = parity_root / "staging_complete.json"
            data = json.loads(item.read_text())
            data["required_files"]["dictionaries/J.pt"] = "1" * 64
            save_json(item, data)
            report["staging_sha256"] = file_sha256(item)
        else:
            report["verifier_identity"]["code"]["git_dirty"] = True
        save_json(parity_root / "parity_report.json", report)
        save_json(
            parity_root / "complete.json",
            {
                "status": "complete",
                "parity_report_sha256": file_sha256(parity_root / "parity_report.json"),
            },
        )
        plan["parity"]["upload"] = upload(parity_root, "synthetic_parity", "e")
    elif mutation == "component_bytes":
        next((foreign / "components").rglob("*.pt")).write_bytes(b"changed")
    elif mutation == "original_plan":
        Path(plan["splits"][0]["original_plan_path"]).write_text("{}")
    save_json(path, plan)
    with pytest.raises((ValueError, AssertionError)):
        run()
    assert list(recipient.iterdir()) == []
