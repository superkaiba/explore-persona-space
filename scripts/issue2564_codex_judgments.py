"""Import content-only Codex collaboration judgments for the #2564 continuation.

This adapter never dispatches a model. The parent coordinates fresh no-history
subagents and records their exact requests. API-route evidence remains untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from issue2564_answer_behavior import PROPERTIES, digest, dump, judge_system, read_jsonl, schema  # noqa: E402

PROVIDER = "codex_collaboration_subagents"
N_REPEATS = 5
PART_COUNTS = {"pilot": 32, "main": 2048}
RECIPE = {
    "provider": PROVIDER,
    "authorized_task": 2564,
    "user_direction": "use codex subagents to judge",
    "model": "inherited Codex model; exact snapshot not exposed",
    "model_override": None,
    "reasoning_effort_override": None,
    "temperature": None,
    "sampling": "uncontrolled; shared base model; repeats are not IID samples",
    "draws": N_REPEATS,
    "fork_turns": "none",
    "property_specific": True,
    "blinding": "content-only packet; instruction-enforced access restriction, not technical isolation",
    "human_agreement": "unmeasured",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_hash() -> str:
    # Preserve the readout's common target/schema identity; provider is separately pinned.
    return digest({p: {"system": judge_system(p), "schema": schema(p)} for p in PROPERTIES})


def load_rows(root: Path, part: str) -> list[dict]:
    rows = [r for r in read_jsonl(root / "prepared/rows.jsonl") if r["part"] == part]
    if len(rows) != PART_COUNTS[part] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Not the frozen complete answer cohort")
    if json.loads((root / "prepared/rubrics.json").read_text()) != PROPERTIES:
        raise ValueError("Frozen answer rubrics changed")
    return rows


def output_dir(root: Path, part: str) -> Path:
    return root / "annotation_codex" / part


def annotation_key(row: dict, prop: str, repetition: int) -> str:
    return digest([RECIPE, instrument_hash(), row["id"], row["answer"], prop, repetition])


def validate_value(prop: str, value: object) -> tuple[dict | None, str | None]:
    """Validate the existing payload directly; never fabricate an API envelope."""
    if not isinstance(value, dict) or set(value) != set(schema(prop)["required"]):
        return None, "schema_mismatch"
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        return None, "invalid_reason"
    if len(value["reason"].split()) > 40:
        return None, "reason_too_long"
    if PROPERTIES[prop]["kind"] == "graded":
        if not isinstance(value["assessable"], bool):
            return None, "schema_mismatch"
        if value["assessable"]:
            if type(value["score"]) is not int or not 0 <= value["score"] <= 100:
                return None, "invalid_score"
        elif value["score"] is not None:
            return None, "invalid_score"
    elif value["label"] not in PROPERTIES[prop]["labels"]:
        return None, "invalid_label"
    if prop == "persona" and not isinstance(value["multi_voice"], bool):
        return None, "schema_mismatch"
    return value, None


def leakage_scan(packet: dict, request: dict) -> dict:
    """Scan coordinator text broadly, real answer text only for project identifiers."""
    wrapper = request["message"]
    # Neutral packet paths are allowed. Source identities/analysis labels are not.
    bans = (
        "char_helios",
        "char_wren",
        "char_dana",
        "char_vex",
        "issue2054",
        "issue2564",
        "v_A",
        "v_C",
        "context-to-answer",
        "answer-behavior-readout",
        "question_group",
        "SAE",
        "held-out",
        "recoverability",
        "hypothesis",
    )
    payload = "\n".join(r["answer"] for r in packet["items"])
    wrapper_hits = [b for b in bans if re.search(re.escape(b), wrapper, flags=re.I)]
    project_only = bans[:10]
    payload_hits = [b for b in project_only if re.search(re.escape(b), payload, flags=re.I)]
    result = {
        "wrapper_bans": list(bans),
        "payload_bans": list(project_only),
        "wrapper_hits": wrapper_hits,
        "payload_hits": payload_hits,
        "scope": "explicit coordinator request and packet only; inherited environment is not technically isolated",
    }
    if wrapper_hits or payload_hits:
        raise ValueError(f"Outbound information leak: {wrapper_hits}, {payload_hits}")
    return result


def inspect_job(job: dict, row_map: dict) -> tuple[dict, dict, list[dict], dict]:
    prop, repetition = job["property"], job["repetition"]
    if prop not in PROPERTIES or type(repetition) is not int or not 0 <= repetition < N_REPEATS:
        raise ValueError("Unknown property/repetition")
    request = json.loads(Path(job["request_path"]).read_text())
    if request != job["request"] or request.get("fork_turns") != "none":
        raise ValueError("Exact no-history spawn request is missing or changed")
    if any(
        job.get(k) is not None or request.get(k) is not None
        for k in (
            "model_override",
            "reasoning_effort_override",
            "temperature",
            "model",
            "reasoning_effort",
        )
    ):
        raise ValueError("Unexpected model/sampling override")
    if job.get("fork_turns") != "none":
        raise ValueError("No-history fork not recorded")
    packet = json.loads(Path(job["input_path"]).read_text())
    if set(packet) != {"rubric", "items"} or packet["rubric"] != PROPERTIES[prop]:
        raise ValueError("Packet does not use the exact single-property frozen rubric")
    items = packet["items"]
    if not items or len({r["id"] for r in items}) != len(items):
        raise ValueError("Empty/duplicated packet IDs")
    for row in items:
        if (
            set(row) != {"id", "answer"}
            or row["id"] not in row_map
            or row["answer"] != row_map[row["id"]]["answer"]
        ):
            raise ValueError("Packet exposes extra metadata or differs from frozen answer text")
    if file_hash(Path(job["input_path"])) != job["input_sha256"]:
        raise ValueError("Packet byte fingerprint changed after dispatch")
    scan = leakage_scan(packet, request)
    return packet, request, items, scan


def validate_interruption(old: dict, event: dict, packet: dict, values: list, job: dict) -> None:
    """Reject invalid recovery evidence before creating its immutable archive."""
    if (
        old["status"] != "running"
        or not old.get("agent_id")
        or old.get("completion_observed_unix_s") is not None
        or old.get("completion_evidence") is not None
        or event["agent_id"] != old["agent_id"]
        or event["observation"] != "agent_absent_from_live_collaboration_roster"
        or not event["evidence"]
        or event["observed_unix_s"] <= old["dispatch_observed_unix_s"]
        or old["agent_id"] == job.get("agent_id")
        or any(old[k] != job[k] for k in ("property", "part", "repetition", "input_sha256"))
    ):
        raise ValueError("Interrupted attempt provenance does not match a fresh complete retry")
    expected = [r["id"] for r in packet["items"]]
    ids = [v.get("id") if isinstance(v, dict) else None for v in values]
    if not 0 < len(ids) < len(expected) or ids != expected[: len(ids)]:
        raise ValueError("This recovery requires an interrupted ordered strict prefix")


def validate_interrupted_attempts(out: Path, job: dict) -> list[Path]:
    """Verify preserved failed attempts without treating any old rating as completed."""
    reference = job.get("interrupted_attempt")
    if reference is None:
        return []
    relative = Path(reference["path"])
    if (
        relative.is_absolute()
        or relative.parts[:1] != ("interrupted_attempts",)
        or ".." in relative.parts
    ):
        raise ValueError("Interrupted attempt must use a contained relative archive path")
    path = out / relative
    record_path = path / "record.json"
    if file_hash(record_path) != reference["record_sha256"]:
        raise ValueError("Interrupted attempt record changed")
    record = json.loads(record_path.read_text())
    required = {"input.json", "request.json", "output.jsonl", "job.json", "event.json"}
    if set(record["file_hashes"]) != required or any(
        file_hash(path / name) != sha for name, sha in record["file_hashes"].items()
    ):
        raise ValueError("Interrupted attempt evidence changed")
    old = json.loads((path / "job.json").read_text())
    event = json.loads((path / "event.json").read_text())
    if record["mode"] != "whole_packet_fresh_context_retry" or record["retained_ratings"] != 0:
        raise ValueError("Interrupted attempts must contribute no retained ratings")
    if file_hash(path / "input.json") != old["input_sha256"]:
        raise ValueError("Interrupted packet differs from the frozen retry packet")
    if json.loads((path / "request.json").read_text()) != old["request"]:
        raise ValueError("Interrupted request differs from the recorded actual request")
    packet = json.loads((path / "input.json").read_text())
    values = read_jsonl(path / "output.jsonl")
    validate_interruption(old, event, packet, values, job)
    return [
        record_path,
        *(path / name for name in sorted(required)),
    ] + validate_interrupted_attempts(out, old)


def prepare_interrupted_retry(
    root: Path, ledger_path: Path, task_name: str, event_path: Path, packet_root: Path
) -> dict:
    """Preserve an interrupted prefix and prepare a fresh whole-packet rating attempt."""
    ledger = json.loads(ledger_path.read_text())
    matches = [j for j in ledger["jobs"] if j["request"]["task_name"] == task_name]
    if len(matches) != 1:
        raise ValueError("Retry must identify one original job")
    job = matches[0]
    if job["status"] != "running" or not job.get("agent_id"):
        raise ValueError("Only an actually dispatched unfinished job can be retried")
    rows = {r["id"]: r for r in load_rows(root, job["part"])}
    packet, _, items, _ = inspect_job(job, rows)
    out = output_dir(root, job["part"])
    packet_id = digest([job["part"], job["property"], job["repetition"], [r["id"] for r in items]])
    if (out / "packets" / packet_id).exists():
        raise ValueError("Imported packet recovery needs separate review; never replace an archive")
    old = json.loads(json.dumps(job))
    relative = Path("interrupted_attempts") / digest([old["agent_id"], old["input_sha256"]])
    dest = out / relative
    if dest.exists():
        raise FileExistsError("Interrupted evidence already exists; inspect before retrying")
    retry_name = task_name + "_retry1"
    paths = {
        "input_path": packet_root / f"{retry_name}.json",
        "output_path": packet_root / f"{retry_name}.output.jsonl",
        "request_path": packet_root / f"{retry_name}.request.json",
    }
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("Fresh retry paths must not overwrite any existing attempt")
    request = dict(job["request"])
    request["task_name"] = retry_name
    request["message"] = request["message"].replace(old["input_path"], str(paths["input_path"]))
    request["message"] = request["message"].replace(old["output_path"], str(paths["output_path"]))
    leakage_scan(packet, request)
    validate_interruption(
        old,
        json.loads(event_path.read_text()),
        packet,
        read_jsonl(Path(old["output_path"])),
        {**old, "agent_id": None},
    )
    validate_interrupted_attempts(out, old)
    dest.mkdir(parents=True)
    for name, source in {
        "input.json": Path(old["input_path"]),
        "request.json": Path(old["request_path"]),
        "output.jsonl": Path(old["output_path"]),
        "event.json": event_path,
    }.items():
        shutil.copyfile(source, dest / name)
    dump(dest / "job.json", old)
    dump(
        dest / "record.json",
        {
            "mode": "whole_packet_fresh_context_retry",
            "file_hashes": {
                name: file_hash(dest / name)
                for name in ("input.json", "request.json", "output.jsonl", "job.json", "event.json")
            },
            "archived_at": time.time(),
            "retained_ratings": 0,
        },
    )
    retry = {k: v for k, v in old.items() if k not in ("dispatch_observed_unix_s",)}
    retry.update(
        **{k: str(v) for k, v in paths.items()},
        request=request,
        status="pending",
        agent_id=None,
        interrupted_attempt={
            "path": str(relative),
            "record_sha256": file_hash(dest / "record.json"),
        },
    )
    validate_interrupted_attempts(out, retry)
    packet_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dest / "input.json", paths["input_path"])
    dump(paths["request_path"], request)
    inspect_job(retry, rows)
    job.clear()
    job.update(retry)
    dump(ledger_path, ledger)
    return retry


def import_ledger(root: Path, ledger_path: Path, part: str) -> dict:
    """Archive exact completed packets; pending entries never become evidence."""
    rows = load_rows(root, part)
    if part == "main":
        validate_codex_pilot(root, RECIPE)
    row_map = {r["id"]: r for r in rows}
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("provider") != PROVIDER:
        raise ValueError("Wrong judge provider")
    out = output_dir(root, part)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != RECIPE:
        raise ValueError("Judge recipe changed within a wave")
    dump(config_path, RECIPE)
    snapshot = out / "ledgers" / f"{file_hash(ledger_path)}.json"
    snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():
        shutil.copyfile(ledger_path, snapshot)
    counts = Counter()
    for job in ledger["jobs"]:
        if job["part"] != part:
            continue
        counts[job["status"]] += 1
        if job["status"] != "complete":
            continue
        packet, request, items, scan = inspect_job(job, row_map)
        validate_interrupted_attempts(out, job)
        if not job.get("agent_id"):
            raise ValueError("Completed packet has no actual agent identifier")
        packet_id = digest([part, job["property"], job["repetition"], [r["id"] for r in items]])
        dest = out / "packets" / packet_id
        expected_files = {
            "input.json": Path(job["input_path"]),
            "request.json": Path(job["request_path"]),
            "output.jsonl": Path(job["output_path"]),
        }
        fingerprints = {name: file_hash(p) for name, p in expected_files.items()}
        if (dest / "record.json").exists():
            old = json.loads((dest / "record.json").read_text())
            if old["file_hashes"] != fingerprints or old["agent_id"] != job["agent_id"]:
                raise ValueError("Completed packet was replaced; retain retry history separately")
            parse_packet(dest, row_map)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        # Preserve original output before parsing, including malformed attempts.
        for name, path in expected_files.items():
            shutil.copyfile(path, dest / name)
        record = {
            "packet_id": packet_id,
            "part": part,
            "property": job["property"],
            "repetition": job["repetition"],
            "row_ids": [r["id"] for r in items],
            "agent_id": job["agent_id"],
            "file_hashes": fingerprints,
            "config_hash": digest(RECIPE),
            "rubric_schema_hash": instrument_hash(),
            "job": job,
            "leakage_scan": scan,
            "imported_at": time.time(),
            "provider": PROVIDER,
            "status": "complete",
        }
        dump(dest / "record.json", record)
        # Raises on missing/duplicated/reordered IDs, retaining the raw evidence.
        parse_packet(dest, row_map)
    dump(
        out / "import_status.json",
        {
            "ledger_sha256": file_hash(ledger_path),
            "job_status_counts": dict(counts),
            "imported_at": time.time(),
        },
    )
    return dict(counts)


def completed_values(path: Path, record: dict, items: list[dict]) -> list[dict]:
    """Validate original ordered output, optionally filling only documented omissions."""
    values = read_jsonl(path / "output.jsonl")
    ids = [v.get("id") if isinstance(v, dict) else None for v in values]
    expected = record["row_ids"]
    repair_path = path / "supplement_record.json"
    if ids == expected:
        if repair_path.exists():
            raise ValueError("A complete original packet cannot receive supplementary ratings")
        return values
    if not repair_path.exists():
        raise ValueError("Missing, duplicated, unknown, or reordered output IDs")
    if (
        len(ids) != len(set(ids))
        or not set(ids) < set(expected)
        or ids != [i for i in expected if i in ids]
    ):
        raise ValueError("Only an ordered strict subset with omissions can be supplemented")
    repair = json.loads(repair_path.read_text())
    if (
        repair["original_record_sha256"] != file_hash(path / "record.json")
        or repair["agent_id"] != record["agent_id"]
        or repair["mode"] != "same_agent_missing_items_only"
        or repair["completion_observed_unix_s"] <= record["job"]["completion_observed_unix_s"]
        or not repair["completion_evidence"]
    ):
        raise ValueError("Supplement provenance differs from the original judge/attempt")
    required_files = {"supplement_input.json", "supplement_request.json", "supplement_output.jsonl"}
    if set(repair["file_hashes"]) != required_files or any(
        file_hash(path / name) != sha for name, sha in repair["file_hashes"].items()
    ):
        raise ValueError("Supplement packet/request/response changed")
    packet = json.loads((path / "supplement_input.json").read_text())
    missing_items = [r for r in items if r["id"] not in ids]
    if packet != {"rubric": PROPERTIES[record["property"]], "items": missing_items}:
        raise ValueError("Supplement must expose only the unchanged omitted answers and rubric")
    request = json.loads((path / "supplement_request.json").read_text())
    if set(request) != {"target", "message"} or request["target"] != record["agent_id"]:
        raise ValueError("Supplement request must address the original agent")
    if leakage_scan(packet, request) != repair["leakage_scan"]:
        raise ValueError("Supplement leakage audit changed")
    extra = read_jsonl(path / "supplement_output.jsonl")
    if [v.get("id") if isinstance(v, dict) else None for v in extra] != [
        r["id"] for r in missing_items
    ]:
        raise ValueError("Supplement must contain exactly the omitted IDs in order")
    # Supplements never repair invalid labels or replace an already expressed judgment.
    for value in values + extra:
        _, error = validate_value(record["property"], {k: v for k, v in value.items() if k != "id"})
        if error:
            raise ValueError(f"Omission-only supplement cannot repair malformed labels: {error}")
    merged = {v["id"]: v for v in values + extra}
    return [merged[i] for i in expected]


def import_supplement(
    packet_path: Path,
    input_path: Path,
    request_path: Path,
    response_path: Path,
    event_path: Path,
    row_map: dict,
) -> dict:
    """Archive an observed same-judge completion without changing the rejected attempt."""
    record = json.loads((packet_path / "record.json").read_text())
    event = json.loads(event_path.read_text())
    files = {
        "supplement_input.json": input_path,
        "supplement_request.json": request_path,
        "supplement_output.jsonl": response_path,
    }
    for name, source in files.items():
        dest = packet_path / name
        if dest.exists():
            raise FileExistsError("Supplement evidence is immutable; inspect existing attempt")
        shutil.copyfile(source, dest)
    repair = {
        "mode": "same_agent_missing_items_only",
        "agent_id": event["agent_id"],
        "completion_observed_unix_s": event["completion_observed_unix_s"],
        "completion_evidence": event["completion_evidence"],
        "original_record_sha256": file_hash(packet_path / "record.json"),
        "file_hashes": {name: file_hash(p) for name, p in files.items()},
        "leakage_scan": leakage_scan(
            json.loads(input_path.read_text()), json.loads(request_path.read_text())
        ),
        "imported_at": time.time(),
    }
    dump(packet_path / "supplement_record.json", repair)
    _, units = parse_packet(packet_path, row_map)
    return {
        "packet_id": record["packet_id"],
        "validated_annotations": len(units),
        "supplemented_annotations": len(read_jsonl(response_path)),
        "original_attempt_preserved": True,
    }


def parse_packet(path: Path, row_map: dict) -> tuple[dict, list[dict]]:
    """Revalidate exact archived evidence and return annotation units."""
    record = json.loads((path / "record.json").read_text())
    if any(
        record[k] != record["job"][k]
        for k in ("agent_id", "property", "part", "repetition", "status")
    ):
        raise ValueError("Archived coordinator metadata differs from the actual recorded job")
    if record["config_hash"] != digest(RECIPE) or record["rubric_schema_hash"] != instrument_hash():
        raise ValueError("Packet provenance uses a stale instrument")
    if any(file_hash(path / name) != sha for name, sha in record["file_hashes"].items()):
        raise ValueError("Archived packet/request/response changed")
    validate_interrupted_attempts(path.parent.parent, record["job"])
    # Recheck actual archived text and request, not mutable /tmp source paths.
    archived_job = {
        **record["job"],
        "input_path": str(path / "input.json"),
        "request_path": str(path / "request.json"),
    }
    _, _, items, scan = inspect_job(archived_job, row_map)
    if record["row_ids"] != [r["id"] for r in items] or record["leakage_scan"] != scan:
        raise ValueError("Archived row roster/leakage audit changed")
    values = completed_values(path, record, items)
    units = []
    for value in values:
        row = row_map[value["id"]]
        parsed, drop = validate_value(
            record["property"], {k: v for k, v in value.items() if k != "id"}
        )
        units.append(
            {
                "key": annotation_key(row, record["property"], record["repetition"]),
                "row_id": row["id"],
                "property": record["property"],
                "draw": record["repetition"],
                "agent_id": record["agent_id"],
                "packet_id": record["packet_id"],
                "parsed": parsed,
                "drop": drop,
            }
        )
    return record, units


def collect(root: Path, part: str) -> tuple[list[dict], list[dict], dict]:
    rows = load_rows(root, part)
    out = output_dir(root, part)
    if json.loads((out / "config.json").read_text()) != RECIPE:
        raise ValueError("Unexpected Codex recipe")
    row_map = {r["id"]: r for r in rows}
    units, packets, manifest = [], [], []
    for path in sorted((out / "packets").glob("*/record.json")):
        record, values = parse_packet(path.parent, row_map)
        packets.append(record)
        units.extend(values)
        manifest.extend(
            [str(p.relative_to(out)), file_hash(p)]
            for p in sorted(path.parent.iterdir())
            if p.is_file()
        )
    interrupted_files = {
        p for record in packets for p in validate_interrupted_attempts(out, record["job"])
    }
    manifest.extend([str(p.relative_to(out)), file_hash(p)] for p in sorted(interrupted_files))
    keys = [u["key"] for u in units]
    if len(keys) != len(set(keys)):
        raise ValueError("More than one packet supplies the same annotation unit")
    agents: dict[tuple, set] = {}
    for unit in units:
        group = (unit["row_id"], unit["property"])
        if unit["agent_id"] in agents.setdefault(group, set()):
            raise ValueError("Repeated rating by the same agent is not a fresh judge instance")
        agents[group].add(unit["agent_id"])
    if len({p["agent_id"] for p in packets}) != len(packets):
        raise ValueError("Each property-specific packet requires a fresh agent context")
    expected = {
        annotation_key(row, prop, d)
        for row in rows
        for prop in PROPERTIES
        for d in range(N_REPEATS)
    }
    if not set(keys) <= expected:
        raise ValueError("Unknown annotation units")
    return (
        rows,
        units,
        {
            "packet_files": manifest,
            "raw_records_hash": digest(manifest),
            "packet_count": len(packets),
            "agent_count": len({p["agent_id"] for p in packets}),
            "packet_sizes": dict(Counter(str(len(p["row_ids"])) for p in packets)),
            "expected_draws": len(expected),
            "persisted_draws": len(keys),
            "expected_keys_hash": digest(sorted(expected)),
            "exact_keyset_complete": set(keys) == expected,
        },
    )


def summarize_units(rows: list[dict], units: list[dict]) -> tuple[list[dict], dict]:
    index = {(u["row_id"], u["property"], u["draw"]): u for u in units}
    labels = []
    for row in rows:
        item = {"id": row["id"], "properties": {}}
        for prop, spec in PROPERTIES.items():
            ds = [index.get((row["id"], prop, d)) for d in range(N_REPEATS)]
            valid = [u["parsed"] for u in ds if u and u["drop"] is None]
            result = {
                "kind": spec["kind"],
                "n_valid": len(valid),
                "n_assessable": None,
                "mean": None,
                "votes": {},
                "modal": None,
                "multi_voice_fraction": None,
            }
            if spec["kind"] == "graded":
                scores = [v["score"] for v in valid if v["assessable"]]
                result.update(
                    n_assessable=len(scores), mean=float(np.mean(scores)) if scores else None
                )
            elif valid:
                counts = Counter(v["label"] for v in valid)
                result["votes"] = {c: counts[c] / len(valid) for c in spec["labels"]}
                winners = [c for c, n in counts.items() if n == max(counts.values())]
                result["modal"] = winners[0] if len(winners) == 1 else None
                if prop == "persona":
                    result["multi_voice_fraction"] = float(
                        np.mean([v["multi_voice"] for v in valid])
                    )
            item["properties"][prop] = result
        labels.append(item)
    quality = {}
    for prop, spec in PROPERTIES.items():
        relevant = [u for u in units if u["property"] == prop]
        values = [r["properties"][prop] for r in labels]
        q = {
            "expected_ratings": len(rows) * N_REPEATS,
            "persisted_ratings": len(relevant),
            "outcomes": dict(Counter(u["drop"] or "valid" for u in relevant)),
            "valid_draw_fraction": sum(v["n_valid"] for v in values) / (len(rows) * N_REPEATS),
            "complete_valid_item_fraction": sum(v["n_valid"] == N_REPEATS for v in values)
            / len(rows),
            "human_agreement": "unmeasured",
            "sampling": RECIPE["sampling"],
        }
        if spec["kind"] == "graded":
            matrix = np.full((len(rows), N_REPEATS), np.nan)
            for i, row in enumerate(rows):
                for d in range(N_REPEATS):
                    u = index.get((row["id"], prop, d))
                    if u and u["drop"] is None and u["parsed"]["assessable"]:
                        matrix[i, d] = u["parsed"]["score"]
            complete = matrix[np.isfinite(matrix).all(axis=1)]
            means = np.array([v["mean"] for v in values if v["mean"] is not None])
            alpha = None
            if len(complete) > 2 and np.var(complete.sum(axis=1), ddof=1) > 0:
                alpha = float(
                    5
                    / 4
                    * (
                        1
                        - np.var(complete, axis=0, ddof=1).sum()
                        / np.var(complete.sum(axis=1), ddof=1)
                    )
                )
            q.update(
                n_any_assessable=sum(v["n_assessable"] > 0 for v in values),
                n_all_five_assessable=len(complete),
                assessable_rating_fraction=float(np.isfinite(matrix).mean()),
                mean=float(means.mean()) if len(means) else None,
                std=float(means.std()) if len(means) else None,
                quantiles=np.quantile(means, [0, 0.25, 0.5, 0.75, 1]).tolist()
                if len(means)
                else None,
                endpoint_fraction=float(np.mean((means <= 5) | (means >= 95)))
                if len(means)
                else None,
                descriptive_cronbach_alpha=alpha,
                mean_within_answer_sd=float(np.std(complete, axis=1).mean())
                if len(complete)
                else None,
                reliability_interpretation="Descriptive agreement among shared-model subagents with uncontrolled sampling and batch context; not IID reliability, human agreement, or an intrinsic decoding ceiling.",
            )
        else:
            q["modal_counts"] = dict(Counter(v["modal"] or "tie_or_missing" for v in values))
            pairs = []
            for row in rows:
                ds = [index.get((row["id"], prop, d)) for d in range(N_REPEATS)]
                vs = [u["parsed"]["label"] for u in ds if u and u["drop"] is None]
                pairs.extend(
                    float(vs[i] == vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))
                )
            q["repeat_pair_agreement"] = float(np.mean(pairs)) if pairs else None
            if prop == "persona":
                flags = [
                    v["multi_voice_fraction"] >= 0.5
                    for v in values
                    if v["multi_voice_fraction"] is not None
                ]
                q["multi_voice_majority_fraction"] = float(np.mean(flags)) if flags else None
        q["parse_completeness_gate_pass"] = (
            q["valid_draw_fraction"] >= 0.98 and q["complete_valid_item_fraction"] >= 0.95
        )
        quality[prop] = q
    return labels, quality


def aggregate(root: Path, part: str) -> dict:
    rows, units, provenance = collect(root, part)
    labels, quality = summarize_units(rows, units)
    out = output_dir(root, part)
    if (out / "accepted.json").exists():
        raise RuntimeError("Accepted pilot is immutable; validate rather than reaggregate")
    dump(out / "labels.json", labels)
    dump(
        out / "quality.json",
        {
            "provider": PROVIDER,
            "properties": quality,
            "sampling": RECIPE["sampling"],
            "human_agreement": "unmeasured",
            "cost_dollars": None,
            "cost_note": "Collaboration tool exposes no per-judgment billing telemetry",
            "provenance": provenance,
            "instrument_acceptance": "pending substantive review",
        },
    )
    complete_path = out / "complete.json"
    if provenance["exact_keyset_complete"]:
        # Deterministic completion is tied to the actual evidence, not file existence.
        dump(
            complete_path,
            {
                "provider": PROVIDER,
                "expected_annotations": provenance["expected_draws"],
                "persisted_expected": provenance["persisted_draws"],
                "expected_keys_hash": provenance["expected_keys_hash"],
                "raw_records_hash": provenance["raw_records_hash"],
                "config_hash": digest(RECIPE),
                "packet_count": provenance["packet_count"],
                "agent_count": provenance["agent_count"],
                "completion_basis": "Exact frozen annotation keyset and validated completed packet records",
            },
        )
    elif complete_path.exists():
        raise RuntimeError("Previously complete annotation wave lost evidence")
    manifest = {
        **{k: v for k, v in provenance.items() if k != "packet_files"},
        "provider": PROVIDER,
        "labels_sha256": file_hash(out / "labels.json"),
        "config_hash": digest(RECIPE),
        "rubric_schema_hash": instrument_hash(),
        "rows_sha256": file_hash(root / "prepared/rows.jsonl"),
        "complete_sha256": file_hash(complete_path) if complete_path.exists() else None,
        "row_count": len(rows),
        "part": part,
    }
    dump(out / "labels_manifest.json", manifest)
    return manifest


def validate_aggregate(root: Path, part: str) -> dict:
    rows, units, provenance = collect(root, part)
    if not provenance["exact_keyset_complete"]:
        raise ValueError("Annotation wave has not completed the frozen roster")
    out = output_dir(root, part)
    manifest = json.loads((out / "labels_manifest.json").read_text())
    complete = json.loads((out / "complete.json").read_text())
    expected = {
        **{k: v for k, v in provenance.items() if k != "packet_files"},
        "provider": PROVIDER,
        "labels_sha256": file_hash(out / "labels.json"),
        "config_hash": digest(RECIPE),
        "rubric_schema_hash": instrument_hash(),
        "rows_sha256": file_hash(root / "prepared/rows.jsonl"),
        "complete_sha256": file_hash(out / "complete.json"),
        "row_count": len(rows),
        "part": part,
    }
    if manifest != expected:
        raise ValueError("Codex aggregate provenance is stale")
    if any(
        complete.get(k) != v
        for k, v in {
            "provider": PROVIDER,
            "expected_annotations": provenance["expected_draws"],
            "persisted_expected": provenance["persisted_draws"],
            "config_hash": digest(RECIPE),
            "expected_keys_hash": provenance["expected_keys_hash"],
            "raw_records_hash": provenance["raw_records_hash"],
        }.items()
    ):
        raise ValueError("Codex completion identity mismatch")
    labels, quality = summarize_units(rows, units)
    if json.loads((out / "labels.json").read_text()) != labels:
        raise ValueError("Aggregate labels differ from original Codex judgments")
    saved_quality = json.loads((out / "quality.json").read_text())
    if saved_quality["properties"] != quality or saved_quality["provenance"] != provenance:
        raise ValueError("Quality report differs from original judgments")
    return {
        "labels_path": str(out / "labels.json"),
        "labels_manifest": manifest,
        "complete": complete,
        "config": RECIPE,
        "quality": saved_quality,
    }


def pilot_pins(root: Path) -> dict:
    result = validate_aggregate(root, "pilot")
    if any(not q["parse_completeness_gate_pass"] for q in result["quality"]["properties"].values()):
        raise ValueError(
            "Pilot parse/completeness gate failed; unassessability is separately retained"
        )
    files = [
        f"annotation_codex/pilot/{name}.json"
        for name in ("config", "complete", "labels", "labels_manifest", "quality")
    ]
    files += ["prepared/rows.jsonl", "prepared/vectors.npz", "prepared/rubrics.json"]
    return {
        "file_hashes": {p: file_hash(root / p) for p in files},
        "judge_recipe": RECIPE,
        "rubric_schema_hash": instrument_hash(),
        "provider": PROVIDER,
    }


def accept_pilot(root: Path, review_file: Path) -> dict:
    pins = pilot_pins(root)
    review = json.loads(review_file.read_text())
    if (
        review.get("verdict") != "accept"
        or not review.get("reviewer")
        or set(review.get("properties", {})) != set(PROPERTIES)
    ):
        raise ValueError("Saved substantive review must cover every property")
    for prop, value in review["properties"].items():
        if value.get("verdict") not in ("accept", "qualified") or any(
            not value.get(k) for k in ("semantic_notes", "reliability_notes", "coverage_notes")
        ):
            raise ValueError(f"Incomplete instrument review: {prop}")
    if not review.get("subagent_instrument_limitations") or not review.get("batching_plan"):
        raise ValueError(
            "Review must discuss changed subagent sampling/blinding and scaled batching"
        )
    out = output_dir(root, "pilot")
    if (out / "accepted.json").exists():
        raise FileExistsError("Pilot acceptance already exists")
    dump(out / "review.json", review)
    accepted = {
        "verdict": "accept",
        "provider": PROVIDER,
        "pins": pins,
        "review_sha256": file_hash(out / "review.json"),
        "accepted_at": time.time(),
        "human_agreement": "unmeasured",
    }
    dump(out / "accepted.json", accepted)
    return accepted


def validate_codex_pilot(root: Path, config: dict | None = None) -> dict:
    out = output_dir(root, "pilot")
    if not (out / "accepted.json").exists():
        raise RuntimeError("Codex main annotation/readout requires reviewed Codex pilot acceptance")
    accepted = json.loads((out / "accepted.json").read_text())
    if (
        accepted.get("verdict") != "accept"
        or accepted.get("provider") != PROVIDER
        or accepted.get("pins") != pilot_pins(root)
    ):
        raise ValueError("Codex pilot acceptance is stale")
    if accepted["review_sha256"] != file_hash(out / "review.json"):
        raise ValueError("Codex instrument review changed")
    if config is not None and config != RECIPE:
        raise ValueError("Main Codex judge recipe differs from accepted pilot")
    return accepted


def validate_codex_main(root: Path) -> dict:
    accepted = validate_codex_pilot(root, RECIPE)
    return {**validate_aggregate(root, "main"), "acceptance": accepted}


def prepare_main_ledger(root: Path, packet_root: Path) -> dict:
    """Freeze and scan all280 main packets; this does not authorize dispatch."""
    rows = load_rows(root, "main")
    pilot = json.loads((root / "codex_pilot_jobs.json").read_text())
    templates = {j["property"]: j for j in pilot["jobs"] if j["repetition"] == 0}
    ledger_path = root / "codex_main_jobs.json"
    if ledger_path.exists():
        raise FileExistsError("Main roster already frozen; inspect it rather than resample")
    row_map = {r["id"]: r for r in rows}
    jobs = []
    for repetition in range(N_REPEATS):
        ordered = list(np.random.default_rng(2564 + repetition).permutation(rows))
        for prop in ("persona", "topic", "warmth", "confidence", "formality", "language", "format"):
            template = templates[prop]
            for start in range(0, len(rows), 256):
                items = [{k: r[k] for k in ("id", "answer")} for r in ordered[start : start + 256]]
                number = len(jobs) + 1
                base = packet_root / f"m{number:04}"
                input_path = base.with_suffix(".json")
                output_path = base.with_suffix(".output.jsonl")
                request_path = base.with_suffix(".request.json")
                request = {**template["request"], "task_name": f"text_judge_main_{number:03}"}
                request["message"] = (
                    request["message"]
                    .replace(template["input_path"], str(input_path))
                    .replace(template["output_path"], str(output_path))
                    .replace("32 items", "256 items")
                    .replace("32 JSONL records", "256 JSONL records")
                )
                dump(input_path, {"rubric": PROPERTIES[prop], "items": items})
                dump(request_path, request)
                job = {
                    "property": prop,
                    "part": "main",
                    "repetition": repetition,
                    "batch": start // 256,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "request_path": str(request_path),
                    "input_sha256": file_hash(input_path),
                    "request": request,
                    "agent_id": None,
                    "status": "pending",
                    "fork_turns": "none",
                    "model_override": None,
                    "reasoning_effort_override": None,
                    "temperature": None,
                    "sampling": RECIPE["sampling"],
                    "tool_access_audit": "not inspected; instruction-enforced restrictions only",
                }
                inspect_job(job, row_map)
                jobs.append(job)
    counts = Counter(
        (r["id"], j["property"], j["repetition"])
        for j in jobs
        for r in json.loads(Path(j["input_path"]).read_text())["items"]
    )
    if len(jobs) != 280 or len(counts) != 71680 or set(counts.values()) != {1}:
        raise ValueError("Main roster does not exactly cover the frozen units")
    ledger = {
        "provider": PROVIDER,
        "created_at": time.time(),
        "main_batch_size": 256,
        "seed": 2564,
        "jobs": jobs,
        "dispatch_gate": "Accepted Codex pilot, then one inspected main envelope before the remaining279",
    }
    dump(ledger_path, ledger)
    audit = {
        "jobs": len(jobs),
        "ratings": len(counts),
        "answers": len(rows),
        "ledger_sha256_before_dispatch": file_hash(ledger_path),
        "all_leakage_scans_passed": True,
        "max_packet_input_bytes": max(Path(j["input_path"]).stat().st_size for j in jobs),
        "min_packet_input_bytes": min(Path(j["input_path"]).stat().st_size for j in jobs),
    }
    dump(root / "codex_main_roster_preflight.json", audit)
    return audit


def record_status(
    ledger_path: Path, started: list[str], completed: list[str], agent_prefix: str
) -> None:
    """Record statuses explicitly observed by the coordinator, never inferred."""
    ledger = json.loads(ledger_path.read_text())
    jobs = {j["request"]["task_name"]: j for j in ledger["jobs"]}
    if set(started) & set(completed) or not set(started + completed) <= jobs.keys():
        raise ValueError("Conflicting or unknown status update")
    now = time.time()
    for name in started:
        job = jobs[name]
        if job["status"] != "pending" or job["agent_id"] is not None:
            raise ValueError("A job cannot be dispatched twice")
        job.update(
            status="running", agent_id=f"{agent_prefix}/{name}", dispatch_observed_unix_s=now
        )
    for name in completed:
        job = jobs[name]
        if job["status"] != "running" or not Path(job["output_path"]).is_file():
            raise ValueError("Completion must follow actual dispatch and an existing output")
        job.update(
            status="complete",
            completion_observed_unix_s=now,
            completion_evidence="Coordinator received collaboration final notification with assigned output path and count",
        )
    dump(ledger_path, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("import", "aggregate", "accept-pilot", "validate-main", "prepare-main", "mark"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--part", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--packet-root", type=Path)
    parser.add_argument("--started", nargs="*", default=[])
    parser.add_argument("--completed", nargs="*", default=[])
    parser.add_argument("--agent-prefix", default="/root/answer_property_matched_experiment")
    args = parser.parse_args()
    if args.mode == "mark":
        if args.ledger is None:
            parser.error("mark requires --ledger")
        record_status(args.ledger, args.started, args.completed, args.agent_prefix)
    elif args.mode == "import":
        if args.ledger is None:
            parser.error("import requires --ledger")
        print(json.dumps(import_ledger(args.root, args.ledger, args.part)))
    elif args.mode == "aggregate":
        print(json.dumps(aggregate(args.root, args.part)))
    elif args.mode == "accept-pilot":
        if args.review_file is None:
            parser.error("accept-pilot requires --review-file")
        print(json.dumps(accept_pilot(args.root, args.review_file)))
    elif args.mode == "prepare-main":
        if args.packet_root is None:
            parser.error("prepare-main requires --packet-root")
        print(json.dumps(prepare_main_ledger(args.root, args.packet_root)))
    else:
        print(json.dumps(validate_codex_main(args.root)))


if __name__ == "__main__":
    main()
