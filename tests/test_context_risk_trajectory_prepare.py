"""CPU fixtures exercise real tokenization, causal assembly, and portable producer readback."""

from __future__ import annotations

import copy
import json
import shutil
from datetime import UTC
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from scripts import context_risk_trajectory_prepare as prepare


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.chat_template = (
        "{% for m in messages %}<{{ m.role }}>{{ m.content | trim }}</{{ m.role }}>"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    contexts, calls, inv, records = {}, [], [], []
    recipes = [
        (
            "lcbhard_35",
            "conflicting",
            "probe_training",
            1,
            False,
            ["visible unsuccessful draft " + "a" * 600, "successful short code"],
        ),
        (
            "lcbhard_35",
            "conflicting",
            "probe_training",
            2,
            False,
            ["a different failed draft", "terminal failed code"],
        ),
        (
            "lcbhard_84",
            "oneoff",
            "final_test",
            4,
            False,
            [
                "first failed code",
                "exception with retained trailing newlines\n\n",
                "third failed code",
            ],
        ),
    ]
    for task, condition, role, epoch, _, bodies in recipes:
        sample_id = f"highrate_fresh:B:{task}:{condition}"
        initial = [{"role": "user", "content": f"Solve {task}. Visible task text."}]
        rendered_initial = tokenizer.apply_chat_template(
            initial, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        initial_ids = tokenizer(rendered_initial, add_special_tokens=False)["input_ids"]
        context = {
            "context_key": sample_id,
            "messages": initial,
            "test": "assert solve(1) == 2",
            "condition": condition,
            "task_id": task,
            "public_test_role": role,
            "initial_token_ids_sha256": prepare.ids_digest(initial_ids),
            "initial_tokens": len(initial_ids),
            "exact_context_sha256": prepare.digest(initial),
        }
        contexts[sample_id] = context
        succeeds = task == "lcbhard_35" and epoch == 1
        first = len(bodies) if succeeds else None
        native_sha, row_sha = (
            prepare.digest([sample_id, epoch, "native"]),
            prepare.digest([sample_id, epoch, "row"]),
        )
        inv.append(
            {
                "sample_id": sample_id,
                "epoch": epoch,
                "condition": condition,
                "task_id": task,
                "partition": role,
                "label": "S" if succeeds else "F",
                "first_success_attempt": first,
                "native_sample_sha256": native_sha,
                "raw_row_sha256": row_sha,
            }
        )
        messages = list(initial)
        for attempt, response in enumerate(bodies, 1):
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prefix = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            body = tokenizer(response, add_special_tokens=False)["input_ids"]
            feedback = (
                f"The test failed at attempt {attempt}. Try again."
                if attempt < len(bodies)
                else None
            )
            current = int(first == attempt)
            call = {
                "call_key": f"{sample_id}:epoch{epoch}:attempt{attempt}",
                "context_key": sample_id,
                "sample_id": sample_id,
                "epoch": epoch,
                "attempt": attempt,
                "task_id": task,
                "condition": condition,
                "public_test_role": role,
                "native_event_uuid": f"event-{task}-{epoch}-{attempt}",
                "request_seed": attempt,
                "native_sample_sha256": native_sha,
                "raw_row_sha256": row_sha,
                "request_messages_sha256": prepare.digest(messages),
                "request_tokens": len(prefix),
                "request_token_ids_sha256": prepare.ids_digest(prefix),
                "body_tokens": len(body),
                "end_token_ids_sha256": prepare.ids_digest(prefix + body),
                "reported_output_tokens": len(body),
                "response": response,
                "response_sha256": prepare.text_digest(response),
                "feedback_after": feedback,
                "current_positive": current,
                "future_positive": int(succeeds),
            }
            calls.append(call)
            records.append(
                {
                    "sample_id": sample_id,
                    "epoch": epoch,
                    "attempt": attempt,
                    "request_messages_sha256": prepare.digest(messages),
                    "prefix_tokens": len(prefix),
                    "retokenized_completion_tokens": len(body),
                    "rendered_prefix_token_ids_sha256": prepare.ids_digest(prefix),
                    "completion_end_token_ids_sha256": prepare.ids_digest(prefix + body),
                    "completion_text_sha256": prepare.text_digest(response),
                }
            )
            messages = [*messages, {"role": "assistant", "content": response}]
            if feedback:
                messages += [{"role": "user", "content": feedback}]
    exclusions = []
    for sample_id, condition, label, reason in [
        ("highrate_fresh:B:lcbhard_35:original", "original", "S", "original"),
        ("highrate_fresh:B:lcbhard_64:conflicting", "conflicting", "U", "unknown"),
        ("highrate_fresh:B:lcbhard_77:oneoff", "oneoff", "NA", "unassessable"),
    ]:
        inv.append({"sample_id": sample_id, "epoch": 1, "condition": condition, "label": label})
        exclusions.append({"sample_id": sample_id, "epoch": 1, "reason": reason, "label": label})
    monkeypatch.setattr(
        prepare,
        "EXPECTED",
        {
            "trajectories": 3,
            "calls": 7,
            "observations": 42,
            "contexts": 2,
            "trajectory_final": 3,
            "completion_exception": 1,
            "canonical_initial": 2,
            "excluded_original": 1,
            "excluded_unknown": 1,
            "excluded_unassessable": 1,
        },
    )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    inventory, geometry = {"trajectories": inv}, {"records": records}
    paths = {}
    for name, value in {
        "inventory": inventory,
        "geometry": geometry,
        "plan_review": {
            "verdict": "PASS",
            "plan_sha256": prepare.sha256(prepare.DESIGN / "plan.md"),
        },
        "code_review": {
            "verdict": "PASS",
            "reviewer": "independent-fixture",
            "sources_sha256": prepare.source_hashes(),
        },
    }.items():
        path = evidence_dir / f"{name}.json"
        prepare.write_json(path, value)
        paths[name] = path
    monkeypatch.setattr(prepare, "INVENTORY_SHA256", prepare.sha256(paths["inventory"]))
    monkeypatch.setattr(prepare, "GEOMETRY_SHA256", prepare.sha256(paths["geometry"]))
    paths["manifest"] = evidence_dir / "frozen_manifest.jsonl"
    prepare.write_jsonl(
        paths["manifest"],
        [
            {**c, "sample_id": c["context_key"], "prompt_variant": "B", "phase": "fresh"}
            for c in contexts.values()
        ],
    )
    monkeypatch.setattr(prepare, "SOURCE_MANIFEST_SHA256", prepare.sha256(paths["manifest"]))
    evidence = {
        "contexts": contexts,
        "calls": calls,
        "exclusions": exclusions,
        "paths": paths,
        "input_artifacts_sha256": prepare._input_hashes(paths),
        "tokenizer_assets_sha256": {
            p.name: prepare.sha256(p) for p in tokenizer_dir.iterdir() if p.is_file()
        },
    }
    cfg = OmegaConf.create(
        {
            "operation": "prepare",
            "root": str(tmp_path / "new_run"),
            "old_root": str(tmp_path / "old_run"),
            "review": str(paths["code_review"]),
            "plan_review": str(paths["plan_review"]),
            "tokenizer_path": str(tokenizer_dir),
        }
    )
    return {
        "tokenizer": tokenizer,
        "evidence": evidence,
        "inventory": inventory,
        "geometry": geometry,
        "cfg": cfg,
    }


def bank(fixture):
    e = fixture["evidence"]
    return prepare.build_bank(e["contexts"], e["calls"], fixture["tokenizer"])


def publish(fixture):
    # Only the external full native-evidence reader is replaced. Real tokenizer,
    # assembly, validation, file writes, staged readback and rename all execute.
    with patch.object(prepare, "_load_evidence", autospec=True, return_value=fixture["evidence"]):
        result = prepare.prepare(fixture["cfg"])
    return Path(result["prepared_path"])


def rewrite_payload(root, name, value):
    path = root / name
    if name.endswith(".jsonl"):
        path.write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        path.write_text(json.dumps(value))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["data_sha256"][name] = prepare.sha256(path)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_real_byte_tokenizer_canonical_initial_and_short_answer_denominators(fixture):
    b = bank(fixture)
    counts = prepare._validate_structure(
        b, fixture["evidence"]["exclusions"], fixture["inventory"], fixture["geometry"]
    )
    assert counts["observations"] == 42 and counts["streams"] == 6
    assert sum(x["stage"] == "within_512" for x in b["observations"]) == 7
    initial = [
        o
        for o in b["observations"]
        if o["task_id"] == "lcbhard_35" and o["stage"] == "pre_action_01"
    ]
    assert len({o["checkpoint_key"] for o in initial}) == 1
    assert {o["positive"] for o in initial} == {0, 1}
    points = {p["checkpoint_key"]: p for p in b["checkpoints"]}
    p = points[initial[0]["checkpoint_key"]]
    assert (
        p["stream_key"].startswith("initial_") and len(b["streams"][p["stream_key"]]) == p["length"]
    )
    assert "visible unsuccessful draft" not in prepare.checkpoint_text(p, b["texts"])
    for o in b["observations"]:
        if o["stage"] == "within_512" and o["observed_tokens"] < 512:
            end = next(
                e
                for e in b["observations"]
                if e["call_key"] == o["call_key"] and e["stage"] == "within_end"
            )
            assert o["answer_ended"] is True and o["checkpoint_key"] == end["checkpoint_key"]


def test_trailing_newline_exception_has_exact_containment(fixture):
    b = bank(fixture)
    exceptions = [
        k for k, v in b["stream_info"].items() if v["stream_kind"] == "completion_exception"
    ]
    assert len(exceptions) == 1
    end = next(
        o
        for o in b["observations"]
        if o["task_id"] == "lcbhard_84" and o["attempt"] == 2 and o["stage"] == "within_end"
    )
    point = next(p for p in b["checkpoints"] if p["checkpoint_key"] == end["checkpoint_key"])
    assert point["stream_key"] == exceptions[0]
    assert prepare.checkpoint_text(point, b["texts"]).endswith("\n\n")


def test_partial_unicode_text_reference_never_takes_a_future_character():
    ref = prepare._text_reference("prefix �", "stream", {"stream": "prefix 界 FUTURE"})
    point = {"visible_text": ref, "visible_text_sha256": prepare.text_digest("prefix �")}
    assert prepare.checkpoint_text(point, {"stream": "prefix 界 FUTURE"}) == "prefix �"


def test_full_prepare_portable_load_and_runtime_replay(fixture, tmp_path):
    root = publish(fixture)
    result = prepare.validate_tokenizer_replay(root, fixture["tokenizer"])
    assert result["passed"] and result["n_requests"] == 7
    relocated = tmp_path / "staged_elsewhere"
    shutil.copytree(root, relocated)
    a = prepare.load_prepared(root)
    b = prepare.load_prepared(relocated / "manifest.json")
    assert a[0] == b[0] and a[2:] == b[2:]
    assert prepare.validate_tokenizer_replay(relocated, fixture["tokenizer"])["passed"]
    assert all(x.dtype == np.uint32 for x in b[1].values())


@pytest.mark.parametrize(
    "mutation",
    ["wrong_label", "missing_stage", "end_position", "initial_stream", "unknown_inclusion"],
)
def test_semantic_corruption_fails_even_after_payload_rehash(fixture, mutation):
    root = publish(fixture)
    if mutation in ("wrong_label", "missing_stage", "unknown_inclusion"):
        rows = prepare.read_jsonl(root / "observations.jsonl")
        if mutation == "wrong_label":
            rows[0]["positive"] = 1 - rows[0]["positive"]
        elif mutation == "missing_stage":
            rows.pop()
        else:
            rows[0]["sample_id"] = "highrate_fresh:B:lcbhard_64:conflicting"
        rewrite_payload(root, "observations.jsonl", rows)
    else:
        rows = prepare.read_jsonl(root / "checkpoints.jsonl")
        if mutation == "end_position":
            rows[0]["position"] += 1
        else:
            point = next(p for p in rows if p["stream_key"].startswith("initial_"))
            _, streams, _, _ = prepare.load_prepared(root)
            candidate = next(
                k
                for k, ids in streams.items()
                if k.startswith("trajectory_")
                and np.array_equal(ids[: point["length"]], streams[point["stream_key"]])
            )
            point["stream_key"] = candidate
        rewrite_payload(root, "checkpoints.jsonl", rows)
    with pytest.raises(ValueError):
        prepare.load_prepared(root)


def test_future_text_corruption_rejected_by_runtime_replay(fixture):
    root = publish(fixture)
    points = prepare.read_jsonl(root / "checkpoints.jsonl")
    target = next(p for p in points if p["observed_tokens"] == 32)
    text_bank = prepare.load_text_bank(root)
    bad = prepare.checkpoint_text(target, text_bank) + " FUTURE SUCCESS"
    target["visible_text"] = {
        "stream_key": target["stream_key"],
        "prefix_characters": 0,
        "suffix": bad,
    }
    target["visible_text_sha256"] = prepare.text_digest(bad)
    rewrite_payload(root, "checkpoints.jsonl", points)
    with pytest.raises(ValueError, match="Visible text contains future"):
        prepare.validate_tokenizer_replay(root, fixture["tokenizer"])


def test_post_success_and_geometry_drift_rejected(fixture):
    calls = copy.deepcopy(fixture["evidence"]["calls"])
    calls[0]["current_positive"] = 1
    with pytest.raises(ValueError, match="Post-success"):
        prepare.build_bank(fixture["evidence"]["contexts"], calls, fixture["tokenizer"])
    calls = copy.deepcopy(fixture["evidence"]["calls"])
    calls[0]["request_token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tokenizer replay"):
        prepare.build_bank(fixture["evidence"]["contexts"], calls, fixture["tokenizer"])


def test_existing_output_and_old_root_are_not_adopted(fixture):
    root = publish(fixture)
    with pytest.raises(FileExistsError):
        prepare.prepare(fixture["cfg"])
    cfg = copy.deepcopy(fixture["cfg"])
    cfg.root = cfg.old_root
    with pytest.raises(ValueError, match="disjoint"):
        prepare.prepare(cfg)
    assert (root / "manifest.json").exists()


def test_review_and_payload_source_drift_are_loud(fixture):
    root = publish(fixture)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["sources_sha256"][prepare.SOURCES[0]] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source closure"):
        prepare.load_prepared(root)


@pytest.mark.parametrize(
    "field,value", [("test", "assert attacker()"), ("public_test_role", "final_test")]
)
def test_static_context_mutation_is_rejected_after_rehash(fixture, field, value):
    root = publish(fixture)
    rows = prepare.read_jsonl(root / "contexts.jsonl")
    row = next(r for r in rows if r["public_test_role"] == "probe_training")
    row[field] = value
    rewrite_payload(root, "contexts.jsonl", rows)
    with pytest.raises(ValueError, match="static context differs"):
        prepare.load_prepared(root)


def test_rehashed_extra_stream_suffix_is_rejected(fixture):
    root = publish(fixture)
    manifest, streams, _, _ = prepare.load_prepared(root)
    key = next(k for k in streams if k.startswith("trajectory_"))
    streams[key] = np.concatenate([streams[key], np.array([42], dtype=np.uint32)])
    np.savez(root / "streams.npz", **streams)
    manifest["streams"][key]["length"] += 1
    manifest["streams"][key]["token_ids_sha256"] = prepare.ids_digest(streams[key])
    manifest["data_sha256"]["streams.npz"] = prepare.sha256(root / "streams.npz")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Final stream differs"):
        prepare.load_prepared(root)


def test_payload_changed_during_consumption_is_rejected(fixture):
    root = publish(fixture)
    original = prepare._validate_structure

    def mutate_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        with (root / "calls.jsonl").open("a") as handle:
            handle.write("\n")
        return result

    with (
        patch.object(prepare, "_validate_structure", side_effect=mutate_after_read),
        pytest.raises(ValueError, match="changed during consumption"),
    ):
        prepare.load_prepared(root)


def test_native_loader_real_body_with_typed_inspect_boundary(fixture, monkeypatch):
    """Reach native reader, resolved model events, raw joins and every source guard."""
    import hashlib
    from datetime import datetime

    from inspect_ai.event import ModelEvent
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ChatMessageUser,
        GenerateConfig,
        ModelOutput,
        ModelUsage,
    )

    cfg = fixture["cfg"]
    old, setup = Path(cfg.old_root), Path(cfg.root) / "setup"
    (old / "fresh_B").mkdir(parents=True)
    (old / "manifests").mkdir()
    setup.mkdir(parents=True)
    native_path = old / "native.eval"
    native_path.write_bytes(b"typed-external-native-reader-boundary")
    monkeypatch.setattr(prepare, "NATIVE_SHA256", prepare.sha256(native_path))
    monkeypatch.setattr(prepare, "NATIVE_ROWS", 6)
    calls = {c["call_key"]: c for c in fixture["evidence"]["calls"]}
    contexts = fixture["evidence"]["contexts"]
    inventory = copy.deepcopy(fixture["inventory"])
    geometry = copy.deepcopy(fixture["geometry"])
    samples, raw, requests = [], [], []
    for item in inventory["trajectories"]:
        key = (item["sample_id"], item["epoch"])
        local = sorted(
            [c for c in calls.values() if (c["sample_id"], c["epoch"]) == key],
            key=lambda c: c["attempt"],
        )
        if not local:
            row = {
                "sample_id": key[0],
                "epoch": key[1],
                "metadata": {"condition": item["condition"]},
                "error": {"type": "transport"} if item["label"] == "U" else None,
            }
            raw.append(row)
            item["raw_row_sha256"] = prepare.digest(row)
            samples.append(EvalSample(id=key[0], epoch=key[1], input="excluded", target=""))
            continue
        context = contexts[local[0]["context_key"]]
        history, events = [], []
        for c in local:
            messages = prepare.request_messages(c, calls, contexts)
            native_messages = [
                ChatMessageUser(content=m["content"])
                if m["role"] == "user"
                else ChatMessageAssistant(content=m["content"])
                for m in messages
            ]
            seed = int(
                hashlib.sha256(f"38295:{key[0]}:{key[1]}:{c['attempt']}".encode()).hexdigest()[:8],
                16,
            )
            usage = ModelUsage(
                input_tokens=c["request_tokens"],
                output_tokens=c["reported_output_tokens"],
                total_tokens=c["request_tokens"] + c["reported_output_tokens"],
            )
            event = ModelEvent(
                model="fixture",
                input=native_messages,
                tools=[],
                tool_choice="none",
                config=GenerateConfig(
                    seed=seed, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                ),
                output=ModelOutput(
                    model="fixture",
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(content=c["response"]), stop_reason="stop"
                        )
                    ],
                    usage=usage,
                ),
                completed=datetime.now(UTC),
            )
            events.append(event)
            history.append(
                {
                    "request_seed": seed,
                    "response": c["response"],
                    "success": bool(c["current_positive"]),
                }
            )
            requests.append(
                {
                    "sample_id": key[0],
                    "epoch": key[1],
                    "attempt": c["attempt"],
                    "response_sha256": prepare.text_digest(c["response"]),
                    "usage": usage.model_dump(),
                }
            )
            g = next(
                g
                for g in geometry["records"]
                if (g["sample_id"], g["epoch"], g["attempt"]) == (*key, c["attempt"])
            )
            g["request_messages_sha256"] = prepare.digest(messages)
            g["reported_output_tokens"] = c["reported_output_tokens"]
        metadata = {
            "condition": context["condition"],
            "partition": context["public_test_role"],
            "agentic_results": {"attempt_history": history},
        }
        sample = EvalSample(
            id=key[0],
            epoch=key[1],
            input=context["messages"][0]["content"],
            target="",
            metadata=metadata,
            events=events,
        )
        row = {"sample_id": key[0], "epoch": key[1], "metadata": metadata, "error": None}
        raw.append(row)
        item["raw_row_sha256"] = prepare.digest(row)
        item["native_sample_sha256"] = prepare.digest(sample.model_dump(mode="json"))
        samples.append(sample)
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-09-08",
            task="typed-fixture",
            dataset=EvalDataset(),
            model="fixture",
            config=EvalConfig(),
        ),
        samples=samples,
    )
    native_hashes = {str(native_path): prepare.sha256(native_path)}
    prepare.write_jsonl(old / "fresh_B/rollouts.jsonl", raw)
    prepare.write_jsonl(
        old / "manifests/fresh_B.jsonl",
        [
            {**c, "sample_id": c["context_key"], "prompt_variant": "B", "phase": "fresh"}
            for c in contexts.values()
        ],
    )
    monkeypatch.setattr(
        prepare, "SOURCE_MANIFEST_SHA256", prepare.sha256(old / "manifests/fresh_B.jsonl")
    )
    prepare.write_json(
        old / "fresh_B/prefix_tokens.json",
        {
            "contexts": [
                {
                    "sample_id": c["context_key"],
                    "prefix_token_ids_sha256": c["initial_token_ids_sha256"],
                    "n_prefix_tokens": c["initial_tokens"],
                }
                for c in contexts.values()
            ]
        },
    )
    prepare.write_json(old / "selection.json", {"fixture": "fixed split"})
    monkeypatch.setattr(prepare, "SELECTION_SHA256", prepare.sha256(old / "selection.json"))
    prepare.write_json(
        old / "fresh_B/transport_postrun_audit.json",
        {
            "verification_passed": True,
            "coverage_complete": True,
            "native_logs_sha256": native_hashes,
            "requests": requests,
        },
    )
    prepare.write_json(
        old / "fresh_B/success_review.json",
        {"verdict": "PASS", "native_logs_sha256": native_hashes},
    )
    inventory.update(source_artifacts_sha256={}, sources_sha256={}, outputs_sha256={})
    geometry.update(
        status="PASS",
        requests=7,
        source_native_log={"path": str(native_path)},
        local_tokenizer_files_sha256={
            str(p): prepare.sha256(p) for p in Path(cfg.tokenizer_path).iterdir() if p.is_file()
        },
    )
    prepare.write_json(setup / "fresh_trajectory_stage_inventory.json", inventory)
    prepare.write_json(setup / "reuse_token_geometry_audit.json", geometry)
    monkeypatch.setattr(
        prepare, "INVENTORY_SHA256", prepare.sha256(setup / "fresh_trajectory_stage_inventory.json")
    )
    monkeypatch.setattr(
        prepare, "GEOMETRY_SHA256", prepare.sha256(setup / "reuse_token_geometry_audit.json")
    )
    review = json.loads(Path(cfg.plan_review).read_text())
    review["inputs_sha256"] = {}
    Path(cfg.plan_review).write_text(json.dumps(review))
    with patch("inspect_ai.log.read_eval_log", autospec=True, return_value=log) as read_native:
        result = prepare._load_evidence(cfg)
    read_native.assert_called_once_with(str(native_path), format="eval", resolve_attachments="full")
    assert len(result["calls"]) == 7 and len(result["contexts"]) == 2
    assert sorted(x["reason"] for x in result["exclusions"]) == [
        "original",
        "unassessable",
        "unknown",
    ]
