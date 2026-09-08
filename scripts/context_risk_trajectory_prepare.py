"""Prepare source-bound, causal text/token checkpoints from completed native trajectories."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from itertools import pairwise
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

PROJECT = Path(__file__).resolve().parent.parent
DESIGN = PROJECT / "eval_results/context_risk_trajectory_design"
SCHEMA = "context_risk_trajectory_prepared_v1"
MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYER = 44
MAP_SHA256 = "680935a244cc39c29797d66b312b95e3741889a541dcee8bde7c69ac33c5242d"
NATIVE_SHA256 = "cfa396e0968ac8b0b34db3b61bd1d860ddd162aa0fe4d6e7989afc83283a64c5"
INVENTORY_SHA256 = "0eccad3db1308b2dcf5c0299f4a49a9f65f1087828e2ca3cbfd1a31f8c04d279"
GEOMETRY_SHA256 = "77bc2c93c99d9431eb25da5d17d215a25e87320ce21d30a7870352695ed7d983"
SELECTION_SHA256 = "f1b61e791e0821238371bdb5f2d3515f830ba21c639b4defa1e01ab1e512925d"
SOURCE_MANIFEST_SHA256 = "2385138de67af3472d5a07267f1c9776d46218368b7f7b9363bef360ed8d4b64"
NATIVE_ROWS = 360
EXPECTED = {
    "trajectories": 231,
    "calls": 2153,
    "observations": 12918,
    "contexts": 58,
    "trajectory_final": 231,
    "completion_exception": 1,
    "canonical_initial": 58,
    "excluded_original": 120,
    "excluded_unknown": 1,
    "excluded_unassessable": 8,
}
EXCEPTION = ("highrate_fresh:B:lcbhard_84:oneoff", 4, 2)
SOURCES = (
    "scripts/context_risk_trajectory_prepare.py",
    "src/explore_persona_space/orchestrate/env.py",
    "tests/test_context_risk_trajectory_prepare.py",
    "configs/eval/context_risk_trajectory_prepare.yaml",
    "eval_results/context_risk_trajectory_design/plan.md",
    "eval_results/context_risk_trajectory_design/analysis_spec.json",
)
DATA_FILES = (
    "streams.npz",
    "texts.jsonl",
    "contexts.jsonl",
    "calls.jsonl",
    "checkpoints.jsonl",
    "observations.jsonl",
    "exclusions.jsonl",
    "inventory.json",
    "geometry.json",
    "plan_review.json",
    "code_review.json",
    "frozen_manifest.jsonl",
)


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ids_digest(ids) -> str:
    return digest(ids.tolist() if isinstance(ids, np.ndarray) else ids)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source_hashes() -> dict:
    if sha256(sys.modules[load_dotenv.__module__].__file__) != sha256(
        PROJECT / "src/explore_persona_space/orchestrate/env.py"
    ):
        raise ValueError("Imported environment helper differs from the bound worktree source")
    return {name: sha256(PROJECT / name) for name in SOURCES}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def prepared_directory(path: str | Path) -> Path:
    path = Path(path)
    return path.parent if path.name == "manifest.json" else path


def validate_review(path: str | Path) -> dict:
    review = json.loads(Path(path).read_text())
    if (
        review.get("verdict") != "PASS"
        or not review.get("reviewer")
        or review.get("sources_sha256") != source_hashes()
    ):
        raise ValueError("Independent preparation review does not bind the current source closure")
    return review


def load_contexts(path: str | Path) -> dict:
    rows = read_jsonl(prepared_directory(path) / "contexts.jsonl")
    result = {row["context_key"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate static context key")
    return result


def load_text_bank(path: str | Path) -> dict:
    rows = read_jsonl(prepared_directory(path) / "texts.jsonl")
    result = {row["stream_key"]: row["text"] for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate stream text key")
    return result


def checkpoint_text(checkpoint: dict, text_bank: dict) -> str:
    ref = checkpoint["visible_text"]
    text = text_bank[ref["stream_key"]]
    end = ref["prefix_characters"]
    if type(end) is not int or not 0 <= end <= len(text) or not isinstance(ref["suffix"], str):
        raise ValueError("Invalid causal text reference")
    visible = text[:end] + ref["suffix"]
    if text_digest(visible) != checkpoint["visible_text_sha256"]:
        raise ValueError("Checkpoint visible-text hash differs")
    return visible


def _text_reference(text: str, stream_key: str, text_bank: dict) -> dict:
    stream = text_bank[stream_key]
    common = 0
    for left, right in zip(text, stream):
        if left != right:
            break
        common += 1
    # A partial UTF-8 token can decode to U+FFFD. Preserve that observed suffix,
    # rather than taking an extra character from the future stream.
    return {"stream_key": stream_key, "prefix_characters": common, "suffix": text[common:]}


def _tokens(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids or any(type(x) is not int or not 0 <= x < 2**32 for x in ids):
        raise ValueError("Empty or invalid token IDs")
    return ids


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def request_messages(call: dict, calls: dict, contexts: dict) -> list[dict]:
    messages = list(contexts[call["context_key"]]["messages"])
    for attempt in range(1, call["attempt"]):
        prior = calls[f"{call['sample_id']}:epoch{call['epoch']}:attempt{attempt}"]
        if prior["current_positive"] or not isinstance(prior["feedback_after"], str):
            raise ValueError("Earlier request contains success or lacks recorded feedback")
        messages.extend(
            [
                {"role": "assistant", "content": prior["response"]},
                {"role": "user", "content": prior["feedback_after"]},
            ]
        )
    if digest(messages) != call["request_messages_sha256"]:
        raise ValueError("Reconstructed request messages differ from native input")
    return messages


def tokenize_call(call: dict, calls: dict, contexts: dict, tokenizer) -> tuple:
    messages = request_messages(call, calls, contexts)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prefix = _tokens(tokenizer, rendered)
    body = _tokens(tokenizer, call["response"])
    if (
        len(prefix) != call["request_tokens"]
        or ids_digest(prefix) != call["request_token_ids_sha256"]
        or len(body) != call["body_tokens"]
        or ids_digest(prefix + body) != call["end_token_ids_sha256"]
    ):
        raise ValueError("Pinned tokenizer replay differs from the full-grain geometry")
    return prefix, body, rendered


def build_bank(contexts: dict, call_rows: list[dict], tokenizer) -> dict:
    """Real production geometry, also exercised with a tiny local tokenizer fixture."""
    calls = {row["call_key"]: dict(row) for row in call_rows}
    if len(calls) != len(call_rows):
        raise ValueError("Duplicate native call identity")
    grouped = defaultdict(list)
    for call in calls.values():
        grouped[(call["sample_id"], call["epoch"])].append(call)
    streams, texts, stream_info, checkpoints, observations = {}, {}, {}, {}, []

    def add_stream(key, ids, text, kind, chosen_by):
        array = np.asarray(ids, dtype=np.uint32)
        if key in streams:
            if not np.array_equal(streams[key], array) or texts[key] != text:
                raise ValueError("Canonical stream collision")
            return
        streams[key], texts[key] = array, text
        stream_info[key] = {
            "length": len(ids),
            "token_ids_sha256": ids_digest(ids),
            "stream_kind": kind,
            "chosen_by": chosen_by,
            "text_sha256": text_digest(text),
        }

    for (sample_id, epoch), local in sorted(grouped.items()):
        local.sort(key=lambda row: row["attempt"])
        if [c["attempt"] for c in local] != list(range(1, len(local) + 1)):
            raise ValueError("Submission attempts are missing or out of order")
        if any(c["current_positive"] for c in local[:-1]):
            raise ValueError("Post-success submission")
        if any(c["future_positive"] != local[-1]["current_positive"] for c in local):
            raise ValueError("Future-first-success labels differ from terminal observation")
        tokenized = [tokenize_call(call, calls, contexts, tokenizer) for call in local]
        for before, after in pairwise(tokenized):
            if after[0][: len(before[0])] != before[0]:
                raise ValueError("Non-nested request prefixes violate the frozen geometry")
        final_prefix, final_body, final_rendered = tokenized[-1]
        final_key = "trajectory_" + digest([sample_id, epoch])
        add_stream(
            final_key,
            final_prefix + final_body,
            final_rendered + _decode(tokenizer, final_body),
            "trajectory_final",
            {"sample_id": sample_id, "epoch": epoch},
        )
        initial, _, initial_text = tokenized[0]
        initial_key = "initial_" + ids_digest(initial)
        # Canonical initial capture is separately truncated and depends on input
        # IDs alone, never the outcome/length of a later containing trajectory.
        add_stream(
            initial_key,
            initial,
            initial_text,
            "canonical_initial",
            {"initial_token_ids_sha256": ids_digest(initial)},
        )
        for call, (prefix, body, rendered) in zip(local, tokenized, strict=True):
            full = prefix + body
            variants = [
                (f"pre_action_{call['attempt']:02d}", 0, False, call["future_positive"]),
                ("within_pre", 0, False, call["current_positive"]),
            ]
            variants += [
                (f"within_{k}", min(k, len(body)), len(body) <= k, call["current_positive"])
                for k in (32, 128, 512)
            ]
            variants.append(("within_end", len(body), True, call["current_positive"]))
            for stage, observed, ended, positive in variants:
                length = len(prefix) + observed
                target = full[:length]
                key = ids_digest(target)
                stream_key = initial_key if call["attempt"] == 1 and observed == 0 else final_key
                if len(streams[stream_key]) < length or not np.array_equal(
                    streams[stream_key][:length], target
                ):
                    stream_key = "exception_" + digest(call["call_key"])
                    add_stream(
                        stream_key,
                        full,
                        rendered + _decode(tokenizer, body),
                        "completion_exception",
                        {"sample_id": sample_id, "epoch": epoch, "attempt": call["attempt"]},
                    )
                if not np.array_equal(streams[stream_key][:length], target):
                    raise ValueError("Assigned stream is not the exact checkpoint prefix")
                visible = rendered + (_decode(tokenizer, body[:observed]) if observed else "")
                point = {
                    "checkpoint_key": key,
                    "stream_key": stream_key,
                    "position": length - 1,
                    "length": length,
                    "request_tokens": len(prefix),
                    "observed_tokens": observed,
                    "answer_ended": ended,
                    "visible_text": _text_reference(visible, stream_key, texts),
                    "visible_text_sha256": text_digest(visible),
                }
                if key in checkpoints:
                    if checkpoint_text(checkpoints[key], texts) != visible:
                        raise ValueError("Same token prefix has different visible text")
                    # Observation-level metadata is authoritative when a vector
                    # is shared; first assignment remains deterministic.
                else:
                    checkpoints[key] = point
                observations.append(
                    {
                        "task_id": call["task_id"],
                        "condition": call["condition"],
                        "public_test_role": call["public_test_role"],
                        "sample_id": sample_id,
                        "epoch": epoch,
                        "attempt": call["attempt"],
                        "stage": stage,
                        "checkpoint_key": key,
                        "context_key": call["context_key"],
                        "call_key": call["call_key"],
                        "positive": positive,
                        "request_tokens": len(prefix),
                        "observed_tokens": observed,
                        "answer_ended": ended,
                    }
                )
                if stage == "within_pre":
                    call["pre_checkpoint_key"] = key
                elif stage == "within_end":
                    call["end_checkpoint_key"] = key
    assignments = Counter(c["stream_key"] for c in checkpoints.values())
    for key, info in stream_info.items():
        info["assigned_checkpoints"] = assignments[key]
    return {
        "streams": streams,
        "texts": texts,
        "stream_info": stream_info,
        "contexts": contexts,
        "calls": sorted(calls.values(), key=lambda c: c["call_key"]),
        "checkpoints": sorted(checkpoints.values(), key=lambda c: c["checkpoint_key"]),
        "observations": sorted(
            observations,
            key=lambda o: (
                o["task_id"],
                o["condition"],
                o["sample_id"],
                o["epoch"],
                o["attempt"],
                o["stage"],
            ),
        ),
    }


def _input_hashes(paths: dict) -> dict:
    return {str(path): sha256(path) for path in paths.values()}


def _load_evidence(cfg: DictConfig) -> dict:
    """Reopen immutable native bytes and require every reused inventory join."""
    old, root = Path(cfg.old_root), Path(cfg.root)
    paths = {
        "inventory": root / "setup/fresh_trajectory_stage_inventory.json",
        "geometry": root / "setup/reuse_token_geometry_audit.json",
        "plan_review": Path(cfg.plan_review),
        "code_review": Path(cfg.review),
        "rows": old / "fresh_B/rollouts.jsonl",
        "selection": old / "selection.json",
        "manifest": old / "manifests/fresh_B.jsonl",
        "prefixes": old / "fresh_B/prefix_tokens.json",
        "audit": old / "fresh_B/transport_postrun_audit.json",
        "success_review": old / "fresh_B/success_review.json",
    }
    if (
        sha256(paths["inventory"]) != INVENTORY_SHA256
        or sha256(paths["geometry"]) != GEOMETRY_SHA256
    ):
        raise ValueError("Inventory or token-geometry bytes differ from the reviewed plan")
    if sha256(paths["selection"]) != SELECTION_SHA256:
        raise ValueError("Frozen training/test task selection differs")
    if sha256(paths["manifest"]) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Frozen source context manifest differs")
    inventory = json.loads(paths["inventory"].read_text())
    geometry = json.loads(paths["geometry"].read_text())
    review = json.loads(paths["plan_review"].read_text())
    if review.get("verdict") != "PASS" or review.get("plan_sha256") != sha256(DESIGN / "plan.md"):
        raise ValueError("Current scientific plan lacks its independent PASS")
    for group in (
        inventory["source_artifacts_sha256"],
        inventory["sources_sha256"],
        inventory["outputs_sha256"],
        review["inputs_sha256"],
    ):
        for path, expected in group.items():
            if sha256(path) != expected:
                raise ValueError(f"Reused inventory binding differs: {path}")
            paths[f"bound_{len(paths)}"] = Path(path)
    if geometry["status"] != "PASS" or geometry["requests"] != EXPECTED["calls"]:
        raise ValueError("Token geometry is incomplete")
    native = Path(geometry["source_native_log"]["path"])
    if sha256(native) != NATIVE_SHA256:
        raise ValueError("Final fresh native log differs")
    paths["native"] = native
    tokenizer_dir = Path(cfg.tokenizer_path)
    tokenizer_hashes = {}
    for old_path, expected in geometry["local_tokenizer_files_sha256"].items():
        path = tokenizer_dir / Path(old_path).name
        if sha256(path) != expected:
            raise ValueError(f"Pinned tokenizer asset differs: {path.name}")
        paths[f"tokenizer_{path.name}"] = path
        tokenizer_hashes[path.name] = expected
    before = _input_hashes(paths)
    raw = read_jsonl(paths["rows"])
    manifest = {r["sample_id"]: r for r in read_jsonl(paths["manifest"])}
    prefixes = {r["sample_id"]: r for r in json.loads(paths["prefixes"].read_text())["contexts"]}
    inv = {(r["sample_id"], r["epoch"]): r for r in inventory["trajectories"]}
    geo = {(r["sample_id"], r["epoch"], r["attempt"]): r for r in geometry["records"]}
    audited = json.loads(paths["audit"].read_text())
    if audited.get("verification_passed") is not True or not audited.get("coverage_complete"):
        raise ValueError("Final fresh outcome audit is not complete")
    requests = {(r["sample_id"], r["epoch"], r["attempt"]): r for r in audited["requests"]}
    successes = json.loads(paths["success_review"].read_text())
    if (
        successes.get("verdict") != "PASS"
        or successes.get("native_logs_sha256") != audited["native_logs_sha256"]
    ):
        raise ValueError("Independent successful-body review does not bind final native bytes")
    exclusions, included = [], {}
    for row in raw:
        key = (row["sample_id"], row["epoch"])
        if key not in inv or digest(row) != inv[key]["raw_row_sha256"]:
            raise ValueError("Raw trajectory differs from full native inventory")
        reason = (
            "original"
            if row["metadata"]["condition"] == "original"
            else (
                "unknown"
                if inv[key]["label"] == "U"
                else "unassessable"
                if inv[key]["label"] == "NA"
                else None
            )
        )
        if reason:
            exclusions.append(
                {
                    "sample_id": key[0],
                    "epoch": key[1],
                    "reason": reason,
                    "label": inv[key]["label"],
                    "raw_row_sha256": digest(row),
                }
            )
        else:
            if inv[key]["label"] not in ("S", "F") or row["error"]:
                raise ValueError("Included trajectory is not a completed assessable observation")
            included[key] = row
    if len(raw) != NATIVE_ROWS or len(included) != EXPECTED["trajectories"]:
        raise ValueError("Frozen complete-case trajectory roster differs")
    # PROD_IMPORT_LINT_EXEMPT: Runtime explicitly pinned by uv --with inspect-ai==0.3.261
    from inspect_ai.log import read_eval_log, resolve_sample_attachments

    if importlib.metadata.version("inspect-ai") != "0.3.261":
        raise ValueError("Native reader version differs")
    log = read_eval_log(str(native), format="eval", resolve_attachments="full")
    if log.status != "success" or len(log.samples) != NATIVE_ROWS:
        raise ValueError("Native collection is incomplete")
    contexts, calls = {}, []
    seen = set()
    for unresolved in log.samples:
        sample_key = (unresolved.id, unresolved.epoch)
        if sample_key not in included:
            continue
        if sample_key in seen:
            raise ValueError("Duplicate native trajectory")
        seen.add(sample_key)
        sample = resolve_sample_attachments(unresolved, "full")
        if digest(sample.model_dump(mode="json")) != inv[sample_key]["native_sample_sha256"]:
            raise ValueError("Resolved native sample differs from its reviewed inventory")
        static = manifest[sample.id]
        if static["prompt_variant"] != "B" or static["phase"] != "fresh":
            raise ValueError("Incorrect prompt arm or phase")
        if static["public_test_role"] != sample.metadata["partition"]:
            raise ValueError("Frozen task role differs from native metadata")
        context_key = static["sample_id"]
        context = {
            "context_key": context_key,
            "messages": static["messages"],
            "test": static["test"],
            "condition": static["condition"],
            "task_id": static["task_id"],
            "public_test_role": static["public_test_role"],
            "exact_context_sha256": static["exact_context_sha256"],
            "initial_token_ids_sha256": prefixes[sample.id]["prefix_token_ids_sha256"],
            "initial_tokens": prefixes[sample.id]["n_prefix_tokens"],
        }
        if context_key in contexts and contexts[context_key] != context:
            raise ValueError("Seed-specific static context drift")
        contexts[context_key] = context
        history = sample.metadata["agentic_results"]["attempt_history"]
        events = [
            event for event in sample.events if event.event == "model" and event.error is None
        ]
        if len(events) != len(history) or not 1 <= len(events) <= 10:
            raise ValueError("Native request/history coverage differs")
        for index, (event, attempt) in enumerate(zip(events, history, strict=True), 1):
            key = (sample.id, sample.epoch, index)
            g, audited_request = geo[key], requests[key]
            messages = [{"role": m.role, "content": m.text} for m in event.input]
            if any(
                not isinstance(m.content, str) or getattr(m, "tool_calls", None)
                for m in event.input
            ):
                raise ValueError("Unexpected nontext native input")
            if digest(messages) != g["request_messages_sha256"] or len(messages) != 2 * index - 1:
                raise ValueError("Actual request message boundary differs")
            seed = int(
                hashlib.sha256(f"38295:{sample.id}:{sample.epoch}:{index}".encode()).hexdigest()[
                    :8
                ],
                16,
            )
            if (
                event.config.seed != seed
                or attempt["request_seed"] != seed
                or event.config.extra_body["chat_template_kwargs"] != {"enable_thinking": False}
                or event.output.completion != attempt["response"]
                or text_digest(event.output.completion) != audited_request["response_sha256"]
                or event.output.usage.model_dump() != audited_request["usage"]
                or event.tools
                or event.error
                or sample.error
            ):
                raise ValueError("Native request recipe, response, or usage differs")
            feedback = None
            if index < len(events):
                feedback = events[index].input[2 * index].text
                if [(m.role, m.text) for m in events[index].input[: 2 * index - 1]] != [
                    (m.role, m.text) for m in event.input
                ]:
                    raise ValueError("Next request changed prior history")
                if events[index].input[2 * index - 1].text != attempt["response"]:
                    raise ValueError("Next request did not preserve the observed response")
            calls.append(
                {
                    "call_key": f"{sample.id}:epoch{sample.epoch}:attempt{index}",
                    "context_key": context_key,
                    "sample_id": sample.id,
                    "epoch": sample.epoch,
                    "attempt": index,
                    "task_id": context["task_id"],
                    "condition": context["condition"],
                    "public_test_role": context["public_test_role"],
                    "native_event_uuid": event.uuid,
                    "request_seed": seed,
                    "native_sample_sha256": inv[sample_key]["native_sample_sha256"],
                    "raw_row_sha256": inv[sample_key]["raw_row_sha256"],
                    "request_messages_sha256": g["request_messages_sha256"],
                    "request_tokens": g["prefix_tokens"],
                    "request_token_ids_sha256": g["rendered_prefix_token_ids_sha256"],
                    "body_tokens": g["retokenized_completion_tokens"],
                    "end_token_ids_sha256": g["completion_end_token_ids_sha256"],
                    "reported_output_tokens": g["reported_output_tokens"],
                    "response": attempt["response"],
                    "response_sha256": g["completion_text_sha256"],
                    "feedback_after": feedback,
                    "current_positive": int(attempt["success"]),
                    "future_positive": int(inv[sample_key]["label"] == "S"),
                }
            )
    if seen != set(included) or len(calls) != EXPECTED["calls"]:
        raise ValueError("Incomplete native preparation census")
    if before != _input_hashes(paths):
        raise ValueError("A source input changed during native preparation")
    return {
        "contexts": contexts,
        "calls": calls,
        "exclusions": exclusions,
        "paths": paths,
        "input_artifacts_sha256": before,
        "tokenizer_assets_sha256": tokenizer_hashes,
    }


def _counts(bank: dict, exclusions: list[dict]) -> dict:
    kinds = Counter(x["stream_kind"] for x in bank["stream_info"].values())
    reasons = Counter(x["reason"] for x in exclusions)
    return {
        "trajectories": len({(c["sample_id"], c["epoch"]) for c in bank["calls"]}),
        "calls": len(bank["calls"]),
        "contexts": len(bank["contexts"]),
        "observations": len(bank["observations"]),
        "checkpoints": len(bank["checkpoints"]),
        "streams": len(bank["streams"]),
        **dict(kinds),
        **{
            f"excluded_{reason}": reasons[reason]
            for reason in ("original", "unknown", "unassessable")
        },
    }


def _validate_structure(
    bank: dict, exclusions: list[dict], inventory: dict, geometry: dict
) -> dict:
    counts = _counts(bank, exclusions)
    if any(counts.get(key) != value for key, value in EXPECTED.items()):
        raise ValueError(f"Prepared cohort/stream census differs: {counts}")
    streams, contexts = bank["streams"], bank["contexts"]
    points = {p["checkpoint_key"]: p for p in bank["checkpoints"]}
    calls = {c["call_key"]: c for c in bank["calls"]}
    if len(points) != len(bank["checkpoints"]) or len(calls) != len(bank["calls"]):
        raise ValueError("Duplicate checkpoint or call key")
    inv = {(r["sample_id"], r["epoch"]): r for r in inventory["trajectories"]}
    geo = {(r["sample_id"], r["epoch"], r["attempt"]): r for r in geometry["records"]}
    if {(c["sample_id"], c["epoch"], c["attempt"]) for c in calls.values()} != set(geo):
        raise ValueError("Prepared call identities do not equal the full pinned geometry roster")
    expected_excluded = {
        (r["sample_id"], r["epoch"]): (
            "original"
            if r["condition"] == "original"
            else "unknown"
            if r["label"] == "U"
            else "unassessable"
        )
        for r in inventory["trajectories"]
        if r["condition"] == "original" or r["label"] in ("U", "NA")
    }
    if (
        len(exclusions) != len(expected_excluded)
        or {(r["sample_id"], r["epoch"]): r["reason"] for r in exclusions} != expected_excluded
    ):
        raise ValueError("Excluded original/unknown/unassessable identities differ")
    if set(streams) != set(bank["texts"]) or set(streams) != set(bank["stream_info"]):
        raise ValueError("Stream/text/manifest key sets differ")
    last_calls = {}
    for call in calls.values():
        identity = (call["sample_id"], call["epoch"])
        if identity not in last_calls or call["attempt"] > last_calls[identity]["attempt"]:
            last_calls[identity] = call
    final_stream_identities = set()
    for key, array in streams.items():
        info = bank["stream_info"][key]
        if array.dtype != np.uint32 or array.ndim != 1 or not len(array):
            raise ValueError("Invalid packed stream dtype or shape")
        if len(array) != info["length"] or ids_digest(array) != info["token_ids_sha256"]:
            raise ValueError("Packed stream token fingerprint differs")
        if text_digest(bank["texts"][key]) != info["text_sha256"]:
            raise ValueError("Packed stream text fingerprint differs")
        if info["stream_kind"] == "canonical_initial" and key != "initial_" + ids_digest(array):
            raise ValueError("Initial stream choice is not solely its prefix-ID hash")
        if info["stream_kind"] == "completion_exception":
            chosen = info["chosen_by"]
            if (chosen["sample_id"], chosen["epoch"], chosen["attempt"]) != EXCEPTION:
                raise ValueError("Unexpected non-nesting exception")
            call = calls[f"{chosen['sample_id']}:epoch{chosen['epoch']}:attempt{chosen['attempt']}"]
            if (
                len(array) != call["request_tokens"] + call["body_tokens"]
                or ids_digest(array) != call["end_token_ids_sha256"]
            ):
                raise ValueError("Exception stream differs from its pinned complete response")
        elif info["stream_kind"] == "trajectory_final":
            chosen = info["chosen_by"]
            identity = (chosen["sample_id"], chosen["epoch"])
            if identity not in last_calls or identity in final_stream_identities:
                raise ValueError("Final stream identity duplicates or is outside the cohort")
            final_stream_identities.add(identity)
            call = last_calls[identity]
            if (
                key != "trajectory_" + digest(list(identity))
                or len(array) != call["request_tokens"] + call["body_tokens"]
                or ids_digest(array) != call["end_token_ids_sha256"]
            ):
                raise ValueError("Final stream differs from its pinned terminal response")
    if final_stream_identities != set(last_calls):
        raise ValueError("Final containing streams do not cover every cohort trajectory")
    assignments = Counter()
    for key, point in points.items():
        length = point["length"]
        array = streams[point["stream_key"]]
        if (
            type(length) is not int
            or not 1 <= length <= len(array)
            or point["position"] != length - 1
        ):
            raise ValueError("Checkpoint position is not its last causal token")
        if key != ids_digest(array[:length]):
            raise ValueError("Checkpoint hash does not match the assigned containing stream")
        checkpoint_text(point, bank["texts"])
        assignments[point["stream_key"]] += 1
    for key, info in bank["stream_info"].items():
        if info["assigned_checkpoints"] != assignments[key] or not assignments[key]:
            raise ValueError("Unaccounted or unused prepared stream")
    grouped = defaultdict(list)
    for observation in bank["observations"]:
        grouped[observation["call_key"]].append(observation)
    if set(grouped) != set(calls):
        raise ValueError("Observation/request roster differs")
    seen_points = set()
    for key, call in calls.items():
        native_key = (call["sample_id"], call["epoch"])
        if key != f"{call['sample_id']}:epoch{call['epoch']}:attempt{call['attempt']}":
            raise ValueError("Call key is not the exact native request identity")
        if native_key not in inv or inv[native_key]["label"] not in ("S", "F"):
            raise ValueError("Observation is outside the frozen completed-assessable cohort")
        if call["condition"] == "original" or call["task_id"] == "lcbhard_77":
            raise ValueError("Original or unassessable row entered the fitted cohort")
        source, g = inv[native_key], geo[(*native_key, call["attempt"])]
        first = source["first_success_attempt"]
        if (
            type(call["current_positive"]) is not int
            or type(call["future_positive"]) is not int
            or call["current_positive"] != int(first == call["attempt"])
            or call["future_positive"] != int(first is not None)
            or (first is not None and call["attempt"] > first)
            or call["native_sample_sha256"] != source["native_sample_sha256"]
            or call["raw_row_sha256"] != source["raw_row_sha256"]
            or call["request_tokens"] != g["prefix_tokens"]
            or call["request_messages_sha256"] != g["request_messages_sha256"]
            or call["body_tokens"] != g["retokenized_completion_tokens"]
            or call["request_token_ids_sha256"] != g["rendered_prefix_token_ids_sha256"]
            or call["end_token_ids_sha256"] != g["completion_end_token_ids_sha256"]
            or text_digest(call["response"]) != g["completion_text_sha256"]
        ):
            raise ValueError("Call labels or token geometry differ from pinned native evidence")
        context = contexts[call["context_key"]]
        if call["public_test_role"] != source["partition"] or any(
            call[field] != source[field] for field in ("task_id", "condition")
        ):
            raise ValueError("Call task/split fields differ from the pinned trajectory inventory")
        for field in ("task_id", "condition", "public_test_role"):
            if call[field] != context[field]:
                raise ValueError("Call/static-context task split differs")
        request_messages(call, calls, contexts)
        observations = grouped[key]
        expected_stages = {
            f"pre_action_{call['attempt']:02d}",
            "within_pre",
            "within_32",
            "within_128",
            "within_512",
            "within_end",
        }
        if len(observations) != 6 or {o["stage"] for o in observations} != expected_stages:
            raise ValueError("A stage or short-answer observation is missing/duplicated")
        pre = None
        for o in observations:
            point = points[o["checkpoint_key"]]
            pre_stage = o["stage"].startswith("pre_action_") or o["stage"] == "within_pre"
            observed = (
                0
                if pre_stage
                else call["body_tokens"]
                if o["stage"] == "within_end"
                else min(int(o["stage"].split("_")[-1]), call["body_tokens"])
            )
            ended = False if pre_stage else observed == call["body_tokens"]
            label = (
                call["future_positive"]
                if o["stage"].startswith("pre_action_")
                else call["current_positive"]
            )
            if (
                o["request_tokens"] != call["request_tokens"]
                or o["observed_tokens"] != observed
                or type(o["answer_ended"]) is not bool
                or o["answer_ended"] != ended
                or type(o["positive"]) is not int
                or o["positive"] != label
                or point["length"] != call["request_tokens"] + observed
                or any(
                    o[field] != call[field]
                    for field in (
                        "task_id",
                        "condition",
                        "public_test_role",
                        "sample_id",
                        "epoch",
                        "attempt",
                        "context_key",
                    )
                )
            ):
                raise ValueError(
                    "Observation features/labels are not the registered causal endpoint"
                )
            if pre_stage:
                if pre is not None and pre != o["checkpoint_key"]:
                    raise ValueError("Primary pre-action and within-pre vectors differ")
                pre = o["checkpoint_key"]
                if pre != call["request_token_ids_sha256"]:
                    raise ValueError("Pre-action point differs from recorded request IDs")
                if call["attempt"] == 1:
                    if point["stream_key"] != "initial_" + pre or point["length"] != len(
                        streams[point["stream_key"]]
                    ):
                        raise ValueError("Initial vector depends on a later containing stream")
                    if pre != context["initial_token_ids_sha256"]:
                        raise ValueError(
                            "Initial point differs from the independent old prefix proof"
                        )
                    if point["length"] != context["initial_tokens"]:
                        raise ValueError(
                            "Static initial token count differs from the exact checkpoint"
                        )
            if o["stage"] == "within_end" and o["checkpoint_key"] != call["end_token_ids_sha256"]:
                raise ValueError("End checkpoint contains a delimiter or future feedback")
            seen_points.add(o["checkpoint_key"])
    if seen_points != set(points):
        raise ValueError("Orphan capture checkpoint")
    return counts


def load_prepared(path: str | Path) -> tuple[dict, dict, list[dict], list[dict]]:
    """Portable, read-only verifier. A caller should separately pin manifest bytes."""
    root = prepared_directory(path)
    manifest_path = root / "manifest.json"
    before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("verification_passed") is not True:
        raise ValueError("Prepared output is not verified terminal preparation")
    if manifest.get("sources_sha256") != source_hashes():
        raise ValueError("Prepared source closure differs from the consuming checkout")
    if set(manifest["data_sha256"]) != set(DATA_FILES):
        raise ValueError("Prepared payload inventory differs")
    expected_hashes = manifest["data_sha256"]
    if {name: sha256(root / name) for name in DATA_FILES} != expected_hashes:
        raise ValueError("Prepared payload bytes differ")
    if (
        sha256(root / "inventory.json") != INVENTORY_SHA256
        or sha256(root / "geometry.json") != GEOMETRY_SHA256
    ):
        raise ValueError("Prepared native/geometry evidence differs from the approved inputs")
    if sha256(root / "frozen_manifest.jsonl") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Prepared source context manifest differs from its immutable original")
    validate_review(root / "code_review.json")
    plan_review = json.loads((root / "plan_review.json").read_text())
    if plan_review.get("verdict") != "PASS" or plan_review.get("plan_sha256") != sha256(
        DESIGN / "plan.md"
    ):
        raise ValueError("Prepared plan review differs")
    recipe = manifest["recipe"]
    if any(
        recipe.get(key) != value
        for key, value in {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "map_sha256": MAP_SHA256,
            "enable_thinking": False,
            "response_tokens": "retokenized_raw_body_without_special_tokens",
            "end_delimiter_inserted": False,
        }.items()
    ):
        raise ValueError("Prepared recipe differs")
    with np.load(root / "streams.npz", allow_pickle=False) as packed:
        streams = {key: packed[key] for key in packed.files}
    bank = {
        "streams": streams,
        "stream_info": manifest["streams"],
        "texts": load_text_bank(root),
        "contexts": load_contexts(root),
        "calls": read_jsonl(root / "calls.jsonl"),
        "checkpoints": read_jsonl(root / "checkpoints.jsonl"),
        "observations": read_jsonl(root / "observations.jsonl"),
    }
    frozen_rows = read_jsonl(root / "frozen_manifest.jsonl")
    frozen = {row["sample_id"]: row for row in frozen_rows}
    if len(frozen) != len(frozen_rows):
        raise ValueError("Duplicate frozen static context")
    for key, context in bank["contexts"].items():
        if key not in frozen or any(
            context[field] != frozen[key][field]
            for field in (
                "messages",
                "test",
                "condition",
                "task_id",
                "public_test_role",
                "exact_context_sha256",
            )
        ):
            raise ValueError("Prepared static context differs from the frozen source manifest")
    counts = _validate_structure(
        bank,
        read_jsonl(root / "exclusions.jsonl"),
        json.loads((root / "inventory.json").read_text()),
        json.loads((root / "geometry.json").read_text()),
    )
    if counts != manifest["counts"]:
        raise ValueError("Saved prepared counts differ from the actual joins")
    if (
        before != sha256(manifest_path)
        or {name: sha256(root / name) for name in DATA_FILES} != expected_hashes
        or manifest["sources_sha256"] != source_hashes()
    ):
        raise ValueError("Prepared artifacts changed during consumption")
    return manifest, streams, bank["checkpoints"], bank["observations"]


def manifest_sha256(path: str | Path) -> str:
    return sha256(prepared_directory(path) / "manifest.json")


def validate_tokenizer_replay(path: str | Path, tokenizer) -> dict:
    """Production 5.15 tokenizer parity, with all visible boundaries revalidated."""
    root = prepared_directory(path)
    pinned = manifest_sha256(root)
    manifest, streams, checkpoint_rows, observations = load_prepared(root)
    contexts, texts = load_contexts(root), load_text_bank(root)
    calls = {c["call_key"]: c for c in read_jsonl(root / "calls.jsonl")}
    points = {p["checkpoint_key"]: p for p in checkpoint_rows}
    by_call = defaultdict(list)
    for observation in observations:
        by_call[observation["call_key"]].append(observation)
    records = []
    for key, call in sorted(calls.items()):
        prefix, body, rendered = tokenize_call(call, calls, contexts, tokenizer)
        for o in by_call[key]:
            point = points[o["checkpoint_key"]]
            target = prefix + body[: o["observed_tokens"]]
            if not np.array_equal(streams[point["stream_key"]][: len(target)], target):
                raise ValueError("Runtime tokenizer changed a checkpoint's actual prefix IDs")
            visible = rendered + (
                _decode(tokenizer, body[: o["observed_tokens"]]) if o["observed_tokens"] else ""
            )
            if checkpoint_text(point, texts) != visible:
                raise ValueError(
                    "Visible text contains future content or differs from runtime decoding"
                )
        records.append(
            {
                "call_key": key,
                "request_tokens": len(prefix),
                "request_token_ids_sha256": ids_digest(prefix),
                "body_tokens": len(body),
                "body_token_ids_sha256": ids_digest(body),
                "end_token_ids_sha256": ids_digest(prefix + body),
            }
        )
    if (
        pinned != manifest_sha256(root)
        or {name: sha256(root / name) for name in DATA_FILES} != manifest["data_sha256"]
        or manifest["sources_sha256"] != source_hashes()
    ):
        raise ValueError("Prepared input changed during runtime tokenizer replay")
    return {
        "passed": True,
        "prepared_manifest_sha256": pinned,
        "sources_sha256": source_hashes(),
        "requests": records,
        "n_requests": len(records),
        "n_checkpoints": len(points),
        "tokenizer_class": type(tokenizer).__name__,
        "runtime": {
            name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")
        },
        "tokenizer_assets_sha256": manifest["tokenizer_assets_sha256"],
        "initial_prefixes": {key: c["initial_token_ids_sha256"] for key, c in contexts.items()},
        "recipe": manifest["recipe"],
    }


def prepare(cfg: DictConfig) -> dict:
    if cfg.operation != "prepare":
        raise ValueError("Only operation=prepare is supported")
    validate_review(cfg.review)
    sources = source_hashes()
    root, old = Path(cfg.root).resolve(), Path(cfg.old_root).resolve()
    output = root / "prepared"
    if root == old or root.is_relative_to(old) or old.is_relative_to(root):
        raise ValueError("New preparation root must be disjoint from immutable old artifacts")
    if output.exists():
        raise FileExistsError("Prepared output must be fresh; no adoption or partial-cache resume")
    started = time.monotonic()
    evidence = _load_evidence(cfg)
    # The CPU inventory used this exact tokenizer runtime. Production5.15
    # replays every token boundary before any activation forward.
    runtime = {
        name: importlib.metadata.version(name)
        for name in ("transformers", "tokenizers", "inspect-ai")
    }
    if runtime != {"transformers": "4.57.6", "tokenizers": "0.22.2", "inspect-ai": "0.3.261"}:
        raise ValueError("CPU tokenizer preparation runtime differs from validated geometry")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(cfg.tokenizer_path), local_files_only=True)
    bank = build_bank(evidence["contexts"], evidence["calls"], tokenizer)
    inventory = json.loads(evidence["paths"]["inventory"].read_text())
    geometry = json.loads(evidence["paths"]["geometry"].read_text())
    counts = _validate_structure(bank, evidence["exclusions"], inventory, geometry)
    if sources != source_hashes() or evidence["input_artifacts_sha256"] != _input_hashes(
        evidence["paths"]
    ):
        raise ValueError("Preparation sources or evidence changed before publication")
    # Write an owned sibling staging directory. Only a validated, fully written
    # directory is renamed to prepared; old failed staging evidence is retained.
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f"prepared.partial.{os.getpid()}.{time.time_ns()}"
    staging.mkdir()
    np.savez(staging / "streams.npz", **bank["streams"])
    write_jsonl(
        staging / "texts.jsonl",
        [{"stream_key": key, "text": text} for key, text in sorted(bank["texts"].items())],
    )
    write_jsonl(
        staging / "contexts.jsonl", [bank["contexts"][key] for key in sorted(bank["contexts"])]
    )
    for name in ("calls", "checkpoints", "observations"):
        write_jsonl(staging / f"{name}.jsonl", bank[name])
    write_jsonl(staging / "exclusions.jsonl", evidence["exclusions"])
    for name in ("inventory", "geometry", "plan_review", "code_review"):
        shutil.copyfile(evidence["paths"][name], staging / f"{name}.json")
    shutil.copyfile(evidence["paths"]["manifest"], staging / "frozen_manifest.jsonl")
    manifest = {
        "schema_version": SCHEMA,
        "verification_passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_sha256": sources,
        "input_artifacts_sha256": evidence["input_artifacts_sha256"],
        "tokenizer_assets_sha256": evidence["tokenizer_assets_sha256"],
        "recipe": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "map_sha256": MAP_SHA256,
            "enable_thinking": False,
            "response_tokens": "retokenized_raw_body_without_special_tokens",
            "end_delimiter_inserted": False,
        },
        "preparation_runtime": runtime,
        "production_tokenizer_replay_required": True,
        "streams": bank["stream_info"],
        "counts": counts,
        "data_sha256": {name: sha256(staging / name) for name in DATA_FILES},
        "elapsed_seconds_before_disk_readback": time.monotonic() - started,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    write_json(staging / "manifest.json", manifest)
    load_prepared(staging)
    if sources != source_hashes() or evidence["input_artifacts_sha256"] != _input_hashes(
        evidence["paths"]
    ):
        raise ValueError("Preparation inputs changed during final readback")
    if output.exists():
        raise FileExistsError("Another producer created the target prepared directory")
    staging.rename(output)
    return {
        "passed": True,
        "prepared_path": str(output),
        "prepared_manifest_sha256": manifest_sha256(output),
        "counts": counts,
        "elapsed_seconds": time.monotonic() - started,
    }


@hydra.main(
    version_base=None, config_path="../configs/eval", config_name="context_risk_trajectory_prepare"
)
def main(cfg: DictConfig) -> None:
    print(json.dumps(prepare(cfg), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
