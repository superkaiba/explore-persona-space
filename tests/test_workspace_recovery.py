"""Recovery must preserve verified answer tensors and produce consumable canonical inputs."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from explore_persona_space.analysis.workspace_artifacts import (
    validate_context_input,
    validate_producer,
)
from explore_persona_space.analysis.workspace_recovery import recover_pilot_inputs
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    save_json,
    save_tensors,
)


def test_verified_pilot_recovery_preserves_answers_and_shared_input(tmp_path, monkeypatch):
    """Exercise real prompt-only forwards and complete source/consumer byte bindings."""
    from transformers import Qwen2Config, Qwen2Model

    source, output = tmp_path / "failed", tmp_path / "recovered"
    generation = {"seeds": [42, 43, 44, 45, 46]}
    producer = {
        "config_sha256": "config",
        "selection_sha256": "selection",
        "model_role": "primary",
        "versions": {},
        "code": {
            "git_commit": "348b11a45a26147f1270356edfb6d3dbfd6f143c",
            "git_dirty": False,
            "git_argv0_path": "scripts/workspace_jr_pipeline.py",
        },
    }
    identity = {
        **producer,
        "code": {
            **producer["code"],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    }
    dictionary = {"identity": producer, "pilot_only": True, "arms": {}}
    for arm in ("J", "R"):
        path = source / "dictionaries" / f"{arm}.pt"
        save_tensors(path, {"dictionary": torch.eye(16)})
        dictionary["arms"][arm] = {"sha256": file_sha256(path)}
    save_json(source / "dictionaries/manifest.json", dictionary)
    save_json(
        source / "pilot_exit.json",
        {"exit_code": 1, "phase": "decompose_train", "finished_at_epoch": 1789244786},
    )
    selection = {"subsets": {}}
    originals = {}
    for split in ("train", "validation", "test"):
        subset = f"pilot_{split}"
        ids = [f"{subset}{i}" for i in range(2)]
        selection["subsets"][subset] = [{"prompt_sha256": key} for key in ids]
        common = {
            "identity": producer,
            "planned_contexts": 2,
            "exclusions": [],
            "status": "complete",
            "included_prompt_sha256": ids,
        }
        gen_hashes, capture_hashes = {}, {}
        for index, key in enumerate(ids):
            tokens = [1, 2, 3][: index + 2]
            contract = {
                "identity": producer,
                "generation": generation,
                "prompt_sha256": key,
                "prompt_token_ids": tokens,
            }
            path = source / "generations" / subset / f"{key}.json"
            save_json(path, {"contract": contract, "contract_sha256": content_sha256(contract)})
            gen_hashes[key] = file_sha256(path)
            rows = [
                {
                    "prompt_ids": tokens,
                    "prompt_sha256": key,
                    "seed": seed,
                    "x": torch.ones(16) * seed,
                    "answer_states": torch.arange((seed - 41) * 16).reshape(-1, 16).bfloat16(),
                }
                for seed in generation["seeds"]
            ]
            path = source / "captures" / subset / f"{key}.pt"
            save_tensors(
                path,
                {
                    "identity": {"identity": producer, "generation_file_sha256": gen_hashes[key]},
                    "rows": rows,
                },
            )
            capture_hashes[key] = file_sha256(path)
            originals[(subset, key)] = rows
        save_json(
            source / "generations" / subset / "generation_status.json",
            {**common, "file_sha256": gen_hashes},
        )
        save_json(
            source / "captures" / subset / "coverage.json",
            {**common, "file_sha256": capture_hashes},
        )
    selection_path = tmp_path / "selection.json"
    save_json(selection_path, selection)
    receipt = {
        "repo": "superkaiba1/explore-persona-space-data",
        "revision": "b7171049e8d6ea741bb412f209e0f933761e441f",
        "prefix": "exploratory_workspace_jr/20260912/primary_component_pilot2",
        "verified_sha256": {
            str(path.relative_to(source)): file_sha256(path)
            for path in source.rglob("*")
            if path.is_file()
        },
    }
    receipt_path = tmp_path / "receipt.json"
    save_json(receipt_path, receipt)
    model_config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model_config._attn_implementation = "eager"
    model = Qwen2Model(model_config).eval()
    monkeypatch.setattr(
        "explore_persona_space.analysis.workspace_recovery.load_native",
        lambda *a, **kw: (model, SimpleNamespace(pad_token_id=0), model),
    )
    args = SimpleNamespace(
        source=source,
        source_receipt=receipt_path,
        out=output,
        selection=selection_path,
        role="primary",
        stage="pilot",
        device="cpu",
    )
    config = {"generation": generation, "models": {"primary": {"source_layer": 1}}}
    recover_pilot_inputs(args, config, identity)
    recovered_dictionary = json.loads((output / "dictionaries/manifest.json").read_text())
    validate_producer(recovered_dictionary["identity"], identity)
    for (subset, key), before in originals.items():
        saved = torch.load(output / "captures" / subset / f"{key}.pt", weights_only=True)
        for prior, after in zip(before, saved["rows"], strict=True):
            assert torch.equal(prior["answer_states"], after["answer_states"])
            assert torch.equal(prior["x"], after["answer_batch_x"])
            validate_context_input(saved["identity"], after["x"], output, identity)
        assert torch.equal(saved["rows"][0]["x"], saved["rows"][-1]["x"])
        name = f"generations/{subset}/{key}.json"
        assert file_sha256(output / name) == receipt["verified_sha256"][name]
    args.out = tmp_path / "second"
    (source / "pilot_exit.json").write_text("{}")
    with pytest.raises(ValueError, match="Unverified or changed"):
        recover_pilot_inputs(args, config, identity)
