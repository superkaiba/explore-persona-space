#!/usr/bin/env python3
"""Arm-blinded row-level annotation for #2094 natural patch generations.

The reader receives full answers under opaque row IDs, but never prompts,
source/recipient identities, intervention arms, layers, or project context.
The frozen key is written before the first API call and exact outbound bytes
are persisted beside each successful response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import anthropic  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
CODEX_MODEL = "gpt-6-astra"
MAX_TOKENS = 8192
TEMPERATURE = 0.0
PACKET_SIZE = 17
CODEX_TIMEOUT_SECONDS = 300

FORM_LABELS = ("itinerary", "quiz", "briefing", "explanation", "other_or_mixed")
SUBJECT_LABELS = ("vancouver", "japan", "rome", "other_or_mixed")
FORMAT_LABELS = ("bullets", "paragraph", "neither_or_mixed")

WRAPPER_BANNED = (
    "donor",
    "recipient",
    "source prompt",
    "target prompt",
    "patch",
    "layer 19",
    "all-layer",
    "all layer",
    "intervention",
    "condition",
    "control group",
    "expected result",
    "transfer rate",
    "function vector",
    "task vector",
    "issue2094",
    "issue-2094",
    "explore-persona-space",
    "blind_key.json",
)

PAYLOAD_BANNED = (
    "issue2094",
    "issue-2094",
    "explore-persona-space",
    "explore_persona_space",
    "blind_key.json",
    "recipient_prompt_id",
    "donor_prompt_id",
    "layer_setting",
    "injection_telemetry",
)

QUESTION = """Read every tagged passage independently and return one annotation per tag.

Use these observable labels:
- form: itinerary, quiz, briefing, explanation, or other_or_mixed
- subject: vancouver (Chinese food/culture in Vancouver), japan (a family visit involving Tokyo and Mount Fuji), rome (Ancient Roman landmarks/history in Rome), or other_or_mixed
- format: bullets, paragraph, or neither_or_mixed

`complete` is true only if the passage finishes the requested kind of response rather than ending mid-sentence or mid-structure. `coherence` is an integer from 0 (unreadable) to 100 (fully coherent). `evidence` is at most 18 words and cites only visible content.

Return only a JSON array in the same order, with exactly this object shape:
{"row_id":"R0001","form":"itinerary","subject":"vancouver","format":"bullets","complete":true,"coherence":95,"evidence":"Three dated days and Vancouver restaurant suggestions."}
Do not use tools. Do not add markdown or commentary."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
    return rows


def freeze_key(path: Path, generations: list[dict[str, Any]]) -> dict[str, str]:
    """Freeze opaque IDs before sending; an existing key must match exactly."""
    gen_ids = [str(row["gen_id"]) for row in generations]
    if len(set(gen_ids)) != len(gen_ids):
        raise ValueError("duplicate generation IDs")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = payload["row_id_to_gen_id"]
        if set(mapping.values()) != set(gen_ids) or len(mapping) != len(gen_ids):
            raise ValueError("existing frozen key does not match the generation census")
        return {str(key): str(value) for key, value in mapping.items()}

    shuffled = list(gen_ids)
    secrets.SystemRandom().shuffle(shuffled)
    mapping = {f"R{index + 1:04d}": gen_id for index, gen_id in enumerate(shuffled)}
    payload = {
        "frozen_before_first_request": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "n_rows": len(mapping),
        "generation_ids_sha256": hashlib.sha256("\n".join(sorted(gen_ids)).encode()).hexdigest(),
        "row_id_to_gen_id": mapping,
    }
    atomic_json(path, payload)
    return mapping


def build_segments(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Build model-visible bytes while retaining wrapper/payload scan scopes."""
    segments: list[tuple[str, str]] = [("wrapper", "<passages>\n")]
    for row_id, output_text in items:
        segments.append(("wrapper", f"[{row_id}]\n"))
        segments.append(("payload", output_text.strip()))
        segments.append(("wrapper", "\n\n"))
    segments.extend((("wrapper", "</passages>\n\n"), ("wrapper", QUESTION)))
    return segments


def scan_for_leakage(segments: list[tuple[str, str]]) -> dict[str, list[str]]:
    joined = {
        scope: "".join(text for item_scope, text in segments if item_scope == scope).lower()
        for scope in ("wrapper", "payload")
    }
    return {
        "wrapper": [term for term in WRAPPER_BANNED if term in joined["wrapper"]],
        "payload": [term for term in PAYLOAD_BANNED if term in joined["payload"]],
    }


def parse_annotations(raw: str, expected_ids: list[str]) -> list[dict[str, Any]]:
    """Parse and validate a complete packet, accepting one optional JSON fence."""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, list) or len(payload) != len(expected_ids):
        raise ValueError("annotation packet row count mismatch")
    observed: list[str] = []
    validated: list[dict[str, Any]] = []
    expected_keys = {"row_id", "form", "subject", "format", "complete", "coherence", "evidence"}
    for item in payload:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ValueError("annotation object has wrong keys")
        row_id = item["row_id"]
        if not isinstance(row_id, str):
            raise ValueError("row_id must be a string")
        observed.append(row_id)
        if item["form"] not in FORM_LABELS:
            raise ValueError(f"invalid form label: {item['form']!r}")
        if item["subject"] not in SUBJECT_LABELS:
            raise ValueError(f"invalid subject label: {item['subject']!r}")
        if item["format"] not in FORMAT_LABELS:
            raise ValueError(f"invalid format label: {item['format']!r}")
        if not isinstance(item["complete"], bool):
            raise ValueError("complete must be boolean")
        if isinstance(item["coherence"], bool) or not isinstance(item["coherence"], int):
            raise ValueError("coherence must be an integer")
        if not 0 <= item["coherence"] <= 100:
            raise ValueError("coherence outside [0,100]")
        if not isinstance(item["evidence"], str) or len(item["evidence"].split()) > 18:
            raise ValueError("evidence must be a string of at most 18 words")
        validated.append(dict(item))
    if observed != expected_ids or len(set(observed)) != len(observed):
        raise ValueError("annotation IDs are missing, duplicated, or out of order")
    return validated


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def packet_items(
    mapping: dict[str, str], generations_by_id: dict[str, dict[str, Any]]
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for row_id, gen_id in mapping.items():
        if gen_id not in generations_by_id:
            raise ValueError(f"frozen key refers to missing generation: {gen_id}")
        items.append((row_id, str(generations_by_id[gen_id]["output_text"])))
    return items


def call_packet(
    client: anthropic.Anthropic,
    items: list[tuple[str, str]],
    request_path: Path,
    response_path: Path,
    parsed_path: Path,
) -> list[dict[str, Any]]:
    segments = build_segments(items)
    hits = scan_for_leakage(segments)
    if any(hits.values()):
        raise RuntimeError(f"refusing outbound blind packet: leakage hits {hits}")
    user_message = "".join(text for _scope, text in segments)
    expected_ids = [row_id for row_id, _text in items]

    # API_DISPATCH_ROUTING_EXEMPT: this is an arm-blinded qualitative read with
    # fewer than ten auditable packets, not a throughput judge path. Exact
    # request bytes, no-system/no-tools settings, and the one-to-one raw response
    # are persisted beside every call; dispatcher routing would weaken that audit.
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    if response.stop_reason != "end_turn":
        raise RuntimeError(
            f"non-answer stop_reason={response.stop_reason!r}; no packet artifact written"
        )
    if not raw_text:
        raise RuntimeError("empty annotation response; no packet artifact written")

    request_audit = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system_prompt": None,
        "tools": None,
        "n_messages": 1,
        "row_ids": expected_ids,
        "outbound_request_verbatim": user_message,
        "outbound_chars": len(user_message),
        "leakage_scan_scopes": {
            "wrapper": {
                "banned_terms": list(WRAPPER_BANNED),
                "hits": hits["wrapper"],
                "chars": sum(len(text) for scope, text in segments if scope == "wrapper"),
            },
            "payload": {
                "banned_terms": list(PAYLOAD_BANNED),
                "hits": hits["payload"],
                "chars": sum(len(text) for scope, text in segments if scope == "payload"),
            },
        },
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
    response_audit = {
        "timestamp_utc": request_audit["timestamp_utc"],
        "raw_text": raw_text,
        "provider_response": response.model_dump(mode="json"),
    }
    atomic_json(request_path, request_audit)
    atomic_json(response_path, response_audit)
    parsed = parse_annotations(raw_text, expected_ids)
    atomic_json(parsed_path, parsed)
    return parsed


def parse_codex_events(raw: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a JSON-event transcript and prove that no tool was invoked."""
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed Codex event at line {line_number}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"non-object Codex event at line {line_number}")
        events.append(event)
    if not events:
        raise ValueError("empty Codex event transcript")

    observed_item_types = {
        str(event["item"].get("type"))
        for event in events
        if event.get("type") in {"item.started", "item.completed"}
        and isinstance(event.get("item"), dict)
    }
    unexpected = sorted(observed_item_types - {"agent_message"})
    if unexpected:
        raise RuntimeError(f"blind Codex reader used a tool or non-message item: {unexpected}")
    messages = [
        event["item"].get("text", "")
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "agent_message"
    ]
    if len(messages) != 1:
        raise RuntimeError(f"expected exactly one Codex agent message, got {len(messages)}")
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if len(completed) != 1:
        raise RuntimeError(f"expected exactly one completed Codex turn, got {len(completed)}")
    return events, dict(completed[0].get("usage") or {})


def call_codex_packet(
    items: list[tuple[str, str]],
    request_path: Path,
    response_path: Path,
    parsed_path: Path,
) -> list[dict[str, Any]]:
    """Run one content-only packet in a fresh, isolated Codex CLI process."""
    segments = build_segments(items)
    hits = scan_for_leakage(segments)
    if any(hits.values()):
        raise RuntimeError(f"refusing outbound blind packet: leakage hits {hits}")
    user_message = "".join(text for _scope, text in segments)
    expected_ids = [row_id for row_id, _text in items]
    with tempfile.TemporaryDirectory(prefix="issue2094-blind-") as temp_dir:
        temp_path = Path(temp_dir)
        last_message_path = temp_path / "last_message.txt"
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--model",
            CODEX_MODEL,
            "--color",
            "never",
            "--json",
            "-C",
            str(temp_path),
            "-o",
            str(last_message_path),
            "-",
        ]
        environment = os.environ.copy()
        completed = subprocess.run(
            command,
            input=user_message,
            text=True,
            capture_output=True,
            timeout=CODEX_TIMEOUT_SECONDS,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Codex blind reader failed with exit {completed.returncode}: "
                f"{completed.stderr[-500:]}"
            )
        if not last_message_path.exists():
            raise RuntimeError("Codex blind reader did not write a final message")
        raw_text = last_message_path.read_text(encoding="utf-8").strip()
        events, usage = parse_codex_events(completed.stdout)
        agent_text = next(
            event["item"]["text"]
            for event in events
            if event.get("type") == "item.completed"
            and isinstance(event.get("item"), dict)
            and event["item"].get("type") == "agent_message"
        ).strip()
        if raw_text != agent_text:
            raise RuntimeError("Codex final-message file and event transcript differ")

    if not raw_text:
        raise RuntimeError("empty annotation response; no packet artifact written")
    timestamp = datetime.now(UTC).isoformat()
    request_audit = {
        "timestamp_utc": timestamp,
        "backend": "codex_cli",
        "model": CODEX_MODEL,
        "model_context": "fresh ephemeral session in a temporary empty directory",
        "system_prompt": "Codex CLI built-in; not supplied by the experiment",
        "tools": "runtime available but prohibited by the user message and event-audited",
        "sandbox": "read-only",
        "user_config": "ignored",
        "project_rules": "none (isolated directory outside a repository)",
        "n_messages": 1,
        "row_ids": expected_ids,
        "outbound_request_verbatim": user_message,
        "outbound_chars": len(user_message),
        "leakage_scan_scopes": {
            "wrapper": {
                "banned_terms": list(WRAPPER_BANNED),
                "hits": hits["wrapper"],
                "chars": sum(len(text) for scope, text in segments if scope == "wrapper"),
            },
            "payload": {
                "banned_terms": list(PAYLOAD_BANNED),
                "hits": hits["payload"],
                "chars": sum(len(text) for scope, text in segments if scope == "payload"),
            },
        },
        "stop_reason": "process_exit_0_turn_completed",
        "usage": usage,
        "tool_item_types_observed": [],
        "protocol_deviation": (
            "Direct Anthropic API and Claude CLI both failed authentication; "
            "used an isolated content-only Codex CLI reader with the frozen key"
        ),
    }
    response_audit = {
        "timestamp_utc": timestamp,
        "raw_text": raw_text,
        "provider_event_jsonl": completed.stdout,
        "provider_stderr": completed.stderr,
        "returncode": completed.returncode,
    }
    atomic_json(request_path, request_audit)
    atomic_json(response_path, response_audit)
    parsed = parse_annotations(raw_text, expected_ids)
    atomic_json(parsed_path, parsed)
    return parsed


def run(args: argparse.Namespace) -> None:
    generations = read_jsonl(args.generations)
    if not generations:
        raise ValueError("no generations")
    if any(row.get("termination_reason") != "eos" for row in generations):
        raise ValueError("refusing to annotate incomplete/capped generations")
    by_id = {str(row["gen_id"]): row for row in generations}
    if len(by_id) != len(generations):
        raise ValueError("duplicate generation IDs")
    args.out.mkdir(parents=True, exist_ok=True)
    mapping = freeze_key(args.out / "blind_key.json", generations)
    items = packet_items(mapping, by_id)

    if args.smoke:
        packets = [items[:1]]
        packet_dir = args.out / "smoke"
    else:
        packets = [items[i : i + PACKET_SIZE] for i in range(0, len(items), PACKET_SIZE)]
        packet_dir = args.out / "packets"
        if len(packets) >= 10:
            raise RuntimeError(f"{len(packets)} calls is a volume path; use the dispatcher")
    packet_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic() if args.backend == "anthropic" else None
    all_annotations: list[dict[str, Any]] = []
    for index, packet in enumerate(packets):
        stem = f"packet_{index:03d}"
        request_path = packet_dir / f"{stem}.request.json"
        response_path = packet_dir / f"{stem}.response.json"
        parsed_path = packet_dir / f"{stem}.parsed.json"
        expected_ids = [row_id for row_id, _text in packet]
        if parsed_path.exists():
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            parsed = parse_annotations(json.dumps(parsed), expected_ids)
        else:
            if args.backend == "anthropic":
                assert client is not None
                parsed = call_packet(
                    client,
                    packet,
                    request_path=request_path,
                    response_path=response_path,
                    parsed_path=parsed_path,
                )
            else:
                parsed = call_codex_packet(
                    packet,
                    request_path=request_path,
                    response_path=response_path,
                    parsed_path=parsed_path,
                )
        all_annotations.extend(parsed)
        print(f"annotated packet {index + 1}/{len(packets)} ({len(packet)} rows)", flush=True)

    if args.smoke:
        atomic_json(
            packet_dir / "SMOKE_PASS.json",
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "row_id": all_annotations[0]["row_id"],
                "model": MODEL if args.backend == "anthropic" else CODEX_MODEL,
                "backend": args.backend,
                "parsed": True,
            },
        )
        return

    if len(all_annotations) != len(generations):
        raise RuntimeError("annotation census incomplete")
    gen_id_for_row = mapping
    joined: list[dict[str, Any]] = []
    for annotation in all_annotations:
        gen_id = gen_id_for_row[annotation["row_id"]]
        joined.append({**annotation, "gen_id": gen_id})
    if len({row["gen_id"] for row in joined}) != len(generations):
        raise RuntimeError("joined annotation census is not one-to-one")
    write_jsonl(args.out / "annotations_blind.jsonl", joined)
    atomic_json(
        args.out / "DONE.json",
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model": MODEL if args.backend == "anthropic" else CODEX_MODEL,
            "backend": args.backend,
            "n_rows": len(joined),
            "n_packets": len(packets),
            "all_rows_annotated": True,
            "annotations_sha256": hashlib.sha256(
                (args.out / "annotations_blind.jsonl").read_bytes()
            ).hexdigest(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backend", choices=("anthropic", "codex-cli"), default="anthropic")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
