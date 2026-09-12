"""Exercise actual decomposition/fit input plumbing with verified tiny artifacts."""

import json
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.workspace_artifacts import validate_coverage
from explore_persona_space.analysis.workspace_capture import decompose_context
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    save_json,
    save_tensors,
)


@pytest.fixture
def pipeline_case(tmp_path, monkeypatch):
    """Real source-manifest resolution, dictionary tensors, captures and canonical refs."""
    from explore_persona_space.orchestrate import env

    monkeypatch.setattr(env, "load_dotenv", lambda: None)
    path = Path(__file__).resolve().parents[1] / "scripts/workspace_jr_pipeline.py"
    module = runpy.run_path(str(path))
    root = tmp_path / "run"
    config = {
        "generation": {"seeds": [42, 43]},
        "decomposition": {"k_primary": 10, "k_sensitivity": [5, 25]},
        "fit": {},
    }
    selection = {"subsets": {}, "source_records": {}}
    audit = {"manifest_sources": {"splits": {}}}
    for split in ("train", "validation", "test"):
        rows = [
            {"prompt": f"Synthetic {split} context {index}", "ladder_local_id": index}
            for index in range(3)
        ]
        source = tmp_path / f"{split}.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        selection["source_records"][split] = {
            "filename": source.name,
            "sha256": file_sha256(source),
        }
        audit["manifest_sources"]["splits"][source.name] = {"path": str(source)}
        import hashlib

        selection["subsets"][f"pilot_{split}"] = [
            {
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "ladder_local_id": index,
                "source_split": split,
                "source_row_index": index,
            }
            for index, row in enumerate(rows)
        ]
    selection_path, audit_path = tmp_path / "selection.json", tmp_path / "audit.json"
    save_json(selection_path, selection)
    save_json(audit_path, audit)
    identity = {
        "model_role": "primary",
        "config_sha256": content_sha256(config),
        "selection_sha256": file_sha256(selection_path),
        "versions": {"torch": "synthetic"},
        "code": {"git_commit": "a" * 40, "git_dirty": False},
    }
    generator = torch.Generator().manual_seed(57)
    dictionaries = {}
    manifest = {"identity": identity, "pilot_only": True, "arms": {}}
    for arm in ("J", "R"):
        dictionary = torch.randn(32, 6, generator=generator)
        dictionary /= dictionary.norm(dim=1, keepdim=True)
        dictionaries[arm] = dictionary
        destination = root / "dictionaries" / f"{arm}.pt"
        save_tensors(destination, {"dictionary": dictionary})
        manifest["arms"][arm] = {
            "sha256": file_sha256(destination),
            "shape": list(dictionary.shape),
        }
    save_json(root / "dictionaries/manifest.json", manifest)
    captures, ids = {}, {}
    for split in ("train", "validation", "test"):
        subset = f"pilot_{split}"
        frozen = selection["subsets"][subset]
        ids[split] = [row["prompt_sha256"] for row in frozen[:2]]
        generation_hashes = []
        for context in ids[split]:
            source = root / "generations" / subset / f"{context}.json"
            save_json(source, {"identity": identity, "prompt_sha256": context, "seeds": [42, 43]})
            generation_hashes.append(file_sha256(source))
        canonical = root / "context_inputs" / subset / "batch-0000.pt"
        x = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        save_tensors(
            canonical,
            {
                "x": x,
                "contract": {
                    "identity": identity,
                    "policy": "context_only_frozen_order_batches16_no_answer_tokens",
                    "prompt_ids": ids[split],
                    "source_hashes": generation_hashes,
                },
            },
        )
        hashes = {}
        for index, context in enumerate(ids[split]):
            # Include both a 128-token batch boundary and unequal rollout means.
            rows = [
                {
                    "prompt_sha256": context,
                    "seed": seed,
                    "x": x[index].clone(),
                    "answer_states": torch.randn(length, 6, generator=generator),
                }
                for seed, length in ((42, 129), (43, 3 + index))
            ]
            rows[0]["answer_states"][0] = 0
            capture_identity = {
                "identity": identity,
                "generation_file_sha256": generation_hashes[index],
                "context_input_file": str(canonical.relative_to(root)),
                "context_input_file_sha256": file_sha256(canonical),
                "context_input_row": index,
            }
            source = root / "captures" / subset / f"{context}.pt"
            save_tensors(source, {"identity": capture_identity, "rows": rows})
            captures[context] = rows
            hashes[context] = file_sha256(source)
        save_json(
            root / "captures" / subset / "coverage.json",
            {
                "identity": identity,
                "status": "complete",
                "planned_contexts": 3,
                "included_prompt_sha256": ids[split],
                "exclusions": [
                    {"prompt_sha256": frozen[2]["prompt_sha256"], "reason": "empty_answer"}
                ],
                "file_sha256": hashes,
            },
        )
    args = SimpleNamespace(
        out=root,
        device="cpu",
        rotation=None,
        k=10,
        all_k=True,
        subset="pilot_train",
        stage="pilot",
        role="primary",
        readiness=None,
        selection=selection_path,
        audit=audit_path,
    )

    def component(k, split="train", index=0):
        return (
            root
            / "components"
            / f"k{k}-rotationNone"
            / f"pilot_{split}"
            / f"{ids[split][index]}.pt"
        )

    def run():
        module["decomposition"](args, config, identity)

    return SimpleNamespace(
        root=root,
        module=module,
        config=config,
        identity=identity,
        args=args,
        ids=ids,
        captures=captures,
        dictionaries=dictionaries,
        manifest=manifest,
        selection=selection,
        component=component,
        run=run,
    )


def assert_reference_values(saved, expected):
    for field in ("targets", "rollout_means"):
        for target in expected[field]:
            torch.testing.assert_close(
                saved[field][target], expected[field][target], rtol=1e-6, atol=1e-8
            )
    for arm, fields in expected["decomposition_statistics"].items():
        for name, values in fields.items():
            torch.testing.assert_close(
                saved["decomposition_statistics"][arm][name], values, rtol=1e-6, atol=1e-8
            )
    assert saved["token_counts"] == expected["token_counts"]
    assert saved["rollout_seeds"] == expected["rollout_seeds"]
    torch.testing.assert_close(saved["x"], expected["x"], rtol=0, atol=0)


def test_all_k_entrypoint_matches_reference_and_unchanged_fit_consumer(pipeline_case, monkeypatch):
    case = pipeline_case
    for split in ("train", "validation", "test"):
        case.args.subset = f"pilot_{split}"
        case.run()
        frozen_ids = [row["prompt_sha256"] for row in case.selection["subsets"][case.args.subset]]
        for k in (5, 10, 25):
            folder = case.component(k, split).parent
            coverage = json.loads((folder / "coverage.json").read_text())
            assert validate_coverage(coverage, frozen_ids, case.identity) == case.ids[split]
            for index, context in enumerate(case.ids[split]):
                destination = case.component(k, split, index)
                saved = torch.load(destination, weights_only=True)
                expected = decompose_context(
                    case.captures[context], case.dictionaries, k=k, token_batch_size=128
                )
                assert_reference_values(saved, expected)
                assert (
                    saved["contract"]
                    == coverage["contract"]
                    == {
                        "identity": case.identity,
                        "dictionaries": case.manifest,
                        "k": k,
                        "rotation": None,
                    }
                )
                assert saved["contract_sha256"] == content_sha256(saved["contract"])
                assert coverage["file_sha256"][context] == file_sha256(destination)
    observed = []

    def fit_consumer(x, targets, ids, config, output, **kwargs):
        observed.append(output.name)
        assert ids == case.ids
        for split, contexts in ids.items():
            np.testing.assert_array_equal(
                x[split], np.stack([case.captures[c][0]["x"].numpy() for c in contexts])
            )
            for name, values in targets[split].items():
                expected = [
                    torch.load(case.component(case.args.k, split, index), weights_only=True)[
                        "targets"
                    ][name].numpy()
                    for index in range(2)
                ]
                np.testing.assert_array_equal(values, np.stack(expected))

    # Exercise the existing input loader, while keeping training/network outside this test.
    monkeypatch.setitem(case.module["fits"].__globals__, "evaluate_component_fits", fit_consumer)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: nullcontext(
                SimpleNamespace(id="fixture", url="https://example.invalid/fixture")
            )
        ),
    )
    for k in (5, 10, 25):
        case.args.k = k
        case.module["fits"](case.args, case.config, case.identity)
    assert observed == ["k5-rotationNone", "k10-rotationNone", "k25-rotationNone"]


def test_partial_resume_executes_only_missing_ks_without_rewriting_completed_files(
    pipeline_case, monkeypatch
):
    case = pipeline_case
    case.args.all_k = False
    case.run()
    originals = {
        case.component(10, index=i): file_sha256(case.component(10, index=i)) for i in range(2)
    }
    nested = case.module["decomposition"].__globals__["decompose_context_nested"]
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["checkpoints"])
        assert kwargs["token_batch_size"] == 128
        return nested(*args, **kwargs)

    monkeypatch.setitem(
        case.module["decomposition"].__globals__, "decompose_context_nested", record
    )
    case.args.all_k = True
    case.run()
    assert calls == [(5, 25), (5, 25)]
    assert all(file_sha256(path) == checksum for path, checksum in originals.items())
    calls.clear()
    case.component(25).unlink()
    case.run()
    assert calls == [(25,)]
    assert all(file_sha256(path) == checksum for path, checksum in originals.items())
    calls.clear()
    case.run()
    assert calls == []


@pytest.mark.parametrize("mutation", ["capture", "generation", "canonical", "component_source"])
def test_completed_resume_still_rejects_changed_sources(pipeline_case, mutation):
    case = pipeline_case
    case.run()
    context = case.ids["train"][0]
    if mutation == "generation":
        path = case.root / "generations/pilot_train" / f"{context}.json"
        path.write_text(path.read_text() + "\n")
    else:
        path = {
            "capture": case.root / "captures/pilot_train" / f"{context}.pt",
            "canonical": case.root / "context_inputs/pilot_train/batch-0000.pt",
            "component_source": case.component(10),
        }[mutation]
        saved = torch.load(path, weights_only=True)
        if mutation == "capture":
            saved["rows"][0]["answer_states"][0, 0] += 1
        elif mutation == "canonical":
            saved["x"][0, 0] += 1
        else:
            saved["source_sha256"] = "0" * 64
        save_tensors(path, saved)
    with pytest.raises(ValueError, match=r"changed|Stale|checksum|coverage"):
        case.run()


def test_resume_cannot_rehash_a_corrupted_completed_component(pipeline_case):
    case = pipeline_case
    case.run()
    destination = case.component(10)
    coverage_path = destination.parent / "coverage.json"
    before = file_sha256(coverage_path)
    saved = torch.load(destination, weights_only=True)
    saved["targets"]["J"][0] += 5
    save_tensors(destination, saved)
    with pytest.raises(ValueError, match=r"changed|Stale|checksum|coverage"):
        case.run()
    assert file_sha256(coverage_path) == before
