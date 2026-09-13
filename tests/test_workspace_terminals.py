"""Regress tokenizer EOS omissions and exact causal-prefix recovery."""

import json
from types import SimpleNamespace

import pytest
import torch

from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    save_json,
    save_tensors,
)
from explore_persona_space.analysis.workspace_terminals import terminal_policy, trim_saved_answer


def test_tokenizer_eos_extends_model_derived_defaults():
    """The 4B checkpoint lacks the standalone generation config present in 27B."""
    from transformers import GenerationConfig, PretrainedConfig

    # Exercise HF's actual fallback derivation. Exact Qwen3.5 checkpoint metadata
    # is independently checked on the pinned native runtime during recovery.
    model = PretrainedConfig(eos_token_id=248044)
    derived = GenerationConfig.from_model_config(model)
    tokenizer = SimpleNamespace(eos_token_id=248046)
    assert terminal_policy(derived, tokenizer)["terminal_ids"] == [248044, 248046]
    primary = GenerationConfig(eos_token_id=[248046, 248044])
    assert terminal_policy(primary, tokenizer)["terminal_ids"] == [248044, 248046]
    assert terminal_policy(primary, tokenizer, [123])["terminal_ids"] == [123, 248044, 248046]
    with pytest.raises(ValueError, match="nonempty"):
        terminal_policy(SimpleNamespace(eos_token_id=None), SimpleNamespace(eos_token_id=None))
    with pytest.raises(ValueError, match="Invalid"):
        terminal_policy(SimpleNamespace(eos_token_id=True), tokenizer)


def test_trim_terminal_states_preserves_original_and_equal_rollout_mean():
    """EOS contribution is removed before pooling, without weighting long draws more."""
    policy = terminal_policy(
        SimpleNamespace(eos_token_id=248044), SimpleNamespace(eos_token_id=248046)
    )
    rows = []
    for seed, content, values in [
        (42, [7], [2.0, 1000.0]),
        (43, [8, 9, 10], [4.0, 4.0, 4.0, -1000.0]),
    ]:
        ids = [*content, 248046]
        draw = {"seed": seed, "token_ids": ids, "finish_reason": "stop"}
        states = torch.tensor(values, dtype=torch.bfloat16)[:, None]
        row = {
            "seed": seed,
            "finish_reason": "stop",
            "answer_ids": ids,
            "terminal_ids_removed": 0,
            "answer_states": states,
            "x": torch.tensor([3.0]),
        }
        corrected, report = trim_saved_answer(row, draw, policy, {248044})
        assert corrected["answer_ids"] == content
        assert torch.equal(corrected["answer_states"], states[:-1])
        assert corrected["x"] is row["x"]
        assert report["additional_terminal_states_removed"] == 1
        assert len(row["answer_states"]) == len(ids)
        rows.append(corrected["answer_states"].double().mean(0))
    assert torch.stack(rows).mean().item() == 3.0


def test_eos_only_and_trailing_tokens_remain_visible():
    policy = terminal_policy(
        SimpleNamespace(eos_token_id=248044), SimpleNamespace(eos_token_id=248046)
    )
    draw = {"seed": 42, "token_ids": [248046, 17, 248044], "finish_reason": "stop"}
    row = {
        "seed": 42,
        "finish_reason": "stop",
        "answer_ids": [248046, 17],
        "terminal_ids_removed": 1,
        "answer_states": torch.ones(2, 3),
    }
    corrected, report = trim_saved_answer(row, draw, policy, {248044})
    assert corrected["answer_ids"] == [] and corrected["answer_states"].shape == (0, 3)
    assert corrected["terminal_ids_removed"] == 3
    assert report["empty_after_terminal_correction"]
    with pytest.raises(ValueError, match="verified original"):
        trim_saved_answer({**row, "answer_ids": [19, 17]}, draw, policy, {248044})


def recovery_module():
    from explore_persona_space.analysis import workspace_terminal_recovery

    return workspace_terminal_recovery


def recovery_fixture(tmp_path, module):
    """Use real checksum/coverage/canonical-input validation on an isolated tiny tree."""
    source = tmp_path / "source"
    producer = {
        "config_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "model_role": "comparison",
        "versions": {},
        "code": {"git_commit": module.SOURCE_PRODUCER, "git_dirty": False},
    }
    identity = dict(producer)
    config = {"generation": {"seeds": [42, 43, 44, 45, 46]}}
    dictionary = {"identity": producer, "pilot_only": True, "arms": {}}
    for arm in ("J", "R"):
        path = source / "dictionaries" / f"{arm}.pt"
        save_tensors(path, {"dictionary": torch.eye(3), "token_ids": torch.arange(3)})
        dictionary["arms"][arm] = {"sha256": file_sha256(path)}
    save_json(source / "dictionaries/manifest.json", dictionary)
    save_json(
        source / "pilot_exit.json", {"exit_code": 0, "phase": "complete", "finished_at_epoch": 123}
    )
    selection = {"subsets": {}}
    for split in ("train", "validation", "test"):
        subset = f"pilot_{split}"
        ids = [content_sha256([split, i]) for i in range(2)]
        selection["subsets"][subset] = [{"prompt_sha256": c} for c in ids]
        gen_hashes, raw = {}, {}
        for i, context in enumerate(ids):
            contract = {
                "identity": producer,
                "prompt_token_ids": [3, i + 10],
                "prompt_sha256": context,
                "generation": config["generation"],
            }
            draws = [
                {
                    "seed": seed,
                    "finish_reason": "stop",
                    "token_ids": ([7] if not (split == "train" and i == 0 and seed == 44) else [])
                    + [248046],
                }
                for seed in config["generation"]["seeds"]
            ]
            raw[context] = {
                "contract": contract,
                "contract_sha256": content_sha256(contract),
                "rollouts": draws,
            }
            path = source / "generations" / subset / f"{context}.json"
            save_json(path, raw[context])
            gen_hashes[context] = file_sha256(path)
        relative = f"context_inputs/{subset}/batch-0000.pt"
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)
        save_tensors(
            source / relative,
            {
                "contract": {
                    "identity": producer,
                    "source_hashes": [gen_hashes[c] for c in ids],
                    "prompt_ids": [raw[c]["contract"]["prompt_token_ids"] for c in ids],
                    "policy": "context_only_frozen_order_batches16_no_answer_tokens",
                },
                "x": x,
            },
        )
        capture_hashes = {}
        for i, context in enumerate(ids):
            reference = {
                "identity": producer,
                "generation_file_sha256": gen_hashes[context],
                "context_input_file": relative,
                "context_input_file_sha256": file_sha256(source / relative),
                "context_input_row": i,
            }
            rows = [
                {
                    "seed": draw["seed"],
                    "finish_reason": "stop",
                    "prompt_ids": raw[context]["contract"]["prompt_token_ids"],
                    "prompt_sha256": context,
                    "answer_ids": draw["token_ids"],
                    "terminal_ids_removed": 0,
                    "x": x[i],
                    "answer_states": torch.arange(len(draw["token_ids"]) * 3, dtype=torch.float32)
                    .reshape(-1, 3)
                    .bfloat16(),
                }
                for draw in raw[context]["rollouts"]
            ]
            path = source / "captures" / subset / f"{context}.pt"
            save_tensors(path, {"identity": reference, "rows": rows})
            capture_hashes[context] = file_sha256(path)
        for kind, filename, hashes in [
            ("generations", "generation_status.json", gen_hashes),
            ("captures", "coverage.json", capture_hashes),
        ]:
            save_json(
                source / kind / subset / filename,
                {
                    "identity": producer,
                    "planned_contexts": 2,
                    "status": "complete",
                    "included_prompt_sha256": ids,
                    "exclusions": [],
                    "file_sha256": hashes,
                    "needs_cap_recovery": False,
                },
            )
    selection_path = tmp_path / "selection.json"
    save_json(selection_path, selection)
    receipt = {
        "repo": "superkaiba1/explore-persona-space-data",
        "revision": module.SOURCE_REVISION,
        "prefix": "exploratory_workspace_jr/20260912/comparison_component_pilot1",
        "verified_sha256": {
            str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*") if p.is_file()
        },
    }
    receipt["files_verified"] = len(receipt["verified_sha256"])
    receipt_path = tmp_path / "upload.json"
    save_json(receipt_path, receipt)
    args = SimpleNamespace(
        source=source,
        source_receipt=receipt_path,
        out=tmp_path / "recovered",
        selection=selection_path,
        resume=False,
    )
    return args, config, identity, receipt


def test_recovery_binds_bytes_excludes_whole_empty_context_and_resumes(tmp_path, monkeypatch):
    module = recovery_module()
    args, config, identity, receipt = recovery_fixture(tmp_path, module)
    policy = terminal_policy(
        SimpleNamespace(eos_token_id=248044), SimpleNamespace(eos_token_id=248046)
    )
    original_publish = module.publish_tensor
    calls = 0

    def interrupt(path, value):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated interruption")
        original_publish(path, value)

    monkeypatch.setattr(module, "publish_tensor", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        module.recover(
            args,
            config,
            identity,
            policy,
            {},
            expected_receipt_sha256=file_sha256(args.source_receipt),
        )
    monkeypatch.setattr(module, "publish_tensor", original_publish)
    args.resume = True
    module.recover(
        args, config, identity, policy, {}, expected_receipt_sha256=file_sha256(args.source_receipt)
    )
    result = json.loads((args.out / "recovery_complete.json").read_text())
    assert result["subsets"]["pilot_train"]["included_contexts"] == 1
    assert result["subsets"]["pilot_train"]["exclusions"][0]["empty_seeds"] == [44]
    assert result["subsets"]["pilot_test"]["included_contexts"] == 2
    assert all(
        file_sha256(args.source / name) == digest
        for name, digest in receipt["verified_sha256"].items()
    )
    for path in (args.out / "captures").rglob("*.pt"):
        saved = torch.load(path, weights_only=True)
        assert all(
            row["answer_ids"] == [7] and len(row["answer_states"]) == 1 for row in saved["rows"]
        )
    with pytest.raises(FileExistsError, match="immutable"):
        module.recover(
            args,
            config,
            identity,
            policy,
            {},
            expected_receipt_sha256=file_sha256(args.source_receipt),
        )


def test_recovery_rejects_mutated_upload_source(tmp_path):
    module = recovery_module()
    args, config, identity, _ = recovery_fixture(tmp_path, module)
    path = next((args.source / "generations/pilot_train").glob("*.json"))
    path.write_text(path.read_text() + " ")
    policy = terminal_policy(
        SimpleNamespace(eos_token_id=248044), SimpleNamespace(eos_token_id=248046)
    )
    with pytest.raises(ValueError, match="verified producer upload"):
        module.recover(
            args,
            config,
            identity,
            policy,
            {},
            expected_receipt_sha256=file_sha256(args.source_receipt),
        )


def test_recovery_resume_detects_signed_zero_bit_changes():
    module = recovery_module()
    with pytest.raises(ValueError, match="tensor changed"):
        module.exact_tree(torch.tensor([0.0]), torch.tensor([-0.0]))


def test_real_pipeline_entrypoint_preserves_recovery_consumer_identity(tmp_path, monkeypatch):
    """Run both real CLI dispatches: argv0 provenance must match decompose/fit."""
    import runpy
    import sys

    from explore_persona_space.analysis.workspace_artifacts import validate_producer

    module = recovery_module()
    recovered = {}

    def capture_identity(_args, _config, identity):
        recovered.update(identity)

    monkeypatch.setattr(module, "recover_terminal_eos", capture_identity)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/workspace_jr_pipeline.py",
            "recover-terminal-eos",
            "--role",
            "comparison",
            "--out",
            str(tmp_path),
        ],
    )
    namespace = runpy.run_path("scripts/workspace_jr_pipeline.py", run_name="__main__")
    assert recovered["code"]["git_argv0_path"] == "scripts/workspace_jr_pipeline.py"

    def consume_identity(_args, _config, identity):
        validate_producer(recovered, identity)

    namespace["main"].__globals__["decomposition"] = consume_identity
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/workspace_jr_pipeline.py",
            "decompose",
            "--role",
            "comparison",
            "--out",
            str(tmp_path),
        ],
    )
    namespace["main"]()


def load_terminal_parity_script():
    import runpy

    return runpy.run_path("scripts/workspace_jr_terminal_parity.py")


def test_pooled_parity_detects_error_hidden_by_large_token_norms():
    compare = load_terminal_parity_script()["compare"]
    a = torch.tensor([[10000.0, 0.0], [-10000.0, 0.01]], dtype=torch.float64)
    b = torch.tensor([[10000.0, 0.0], [-10000.0, 0.10]], dtype=torch.float64)
    gate = {"maximum_relative_frobenius_error": 0.01, "minimum_row_cosine": 0.999}
    assert compare(a, b, gate)["passed"]
    assert not compare(a.mean(0)[None], b.mean(0)[None], gate)["passed"]


def test_native_parity_rejects_changed_identity_before_loading_gpu(tmp_path, monkeypatch):
    import sys

    module = recovery_module()
    args, config, identity, _ = recovery_fixture(tmp_path, module)
    config["provenance_gate"] = {
        "maximum_relative_frobenius_error": 0.01,
        "minimum_row_cosine": 0.999,
    }
    main = load_terminal_parity_script()["main"]
    global_values = main.__globals__
    monkeypatch.setitem(global_values, "SOURCE_RECEIPT_SHA256", file_sha256(args.source_receipt))
    monkeypatch.setitem(global_values, "load_workspace_jr_config", lambda _: config)
    monkeypatch.setitem(
        global_values, "run_identity", lambda *_: {**identity, "config_sha256": "c" * 64}
    )

    def forbidden_gpu_load(*_args, **_kwargs):
        raise AssertionError("Changed identity reached the GPU loader")

    monkeypatch.setitem(global_values, "load_native", forbidden_gpu_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/workspace_jr_terminal_parity.py",
            "--source",
            str(args.source),
            "--source-receipt",
            str(args.source_receipt),
            "--selection",
            str(args.selection),
            "--out",
            str(tmp_path / "parity"),
        ],
    )
    with pytest.raises(ValueError, match="config_sha256"):
        main()
