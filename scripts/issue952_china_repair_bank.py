"""Build and validate explicit country-cue pairs for the repaired China study.

The v1 accepted subject questions remain fixed. Codex authors and independent
auditors supply country metadata; deterministic rendering changes only a cue.
No model responses are read by this module and no model API is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

CONTRACT = "china-country-cue-v2"
HF_PREFIX = "issue952_position_divergence/followups/china_refusal_wording_withholding_v2"
CONTENTS = (
    "sensitive_full",
    "sensitive_country_neutral",
    "matched_non_china",
    "matched_non_china_country",
)
CUE_TEMPLATES = {"en": "Related country: {country}.\n", "zh": "相关国家：{country}。\n"}


def sha(path: Path) -> str:
    """Return the exact persisted file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value: object) -> str:
    """Use the existing GPU manifest's canonical JSON convention."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    """Load a JSONL bank without printing its text."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: object) -> None:
    """Publish one complete JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict]) -> None:
    """Publish a complete finite bank atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def source_panel(bank: Path, audit: Path) -> tuple[list[dict], dict]:
    """Verify the frozen v1 bank and its exact accepted 85-item panel."""
    report = json.loads(audit.read_text())
    if report.get("passed") is not True or report["prompt_bank_sha256"] != sha(bank):
        raise ValueError("source bank lacks its matching passed audit")
    ids = report["passing_item_ids"]
    if len(ids) != 85 or len(set(ids)) != 85:
        raise ValueError("the repair must retain exactly the frozen 85 source IDs")
    rows = [r for r in read_rows(bank) if r["source_prompt_id"] in set(ids)]
    if len(rows) != 1020 or any(
        n != 12 for n in Counter(r["source_prompt_id"] for r in rows).values()
    ):
        raise ValueError("incomplete v1 accepted panel")
    return rows, report


def prepare(bank: Path, audit: Path, out: Path) -> dict:
    """Write bounded country-metadata author packets, with no response access."""
    rows, report = source_panel(bank, audit)
    selected = [r for r in rows if r["frame"] == "direct" and r["content"] != "sensitive_full"]
    items = []
    for source_id in sorted(report["passing_item_ids"]):
        item_rows = [r for r in selected if r["source_prompt_id"] == source_id]
        items.append(
            {
                "source_prompt_id": source_id,
                "questions": {f"{r['language']}:{r['content']}": r["prompt"] for r in item_rows},
            }
        )
    manifest = {
        "contract": CONTRACT,
        "source_bank_sha256": sha(bank),
        "source_audit_sha256": sha(audit),
        "accepted_source_ids": sorted(report["passing_item_ids"]),
        "packets": [],
    }
    for offset in range(0, len(items), 10):
        path = out / "author_packets" / f"batch_{offset // 10:03d}.json"
        payload = {"contract": CONTRACT, "items": items[offset : offset + 10]}
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError("existing author packet differs from frozen source")
        write_json(path, payload)
        manifest["packets"].append({"path": str(path), "sha256": sha(path)})
    write_json(out / "source_manifest.json", manifest)
    return {"n_items": len(items), "n_packets": len(manifest["packets"]), "out": str(out)}


def metadata_rows(path: Path, expected: set[str]) -> dict[str, dict]:
    """Validate individually authored metadata, rejecting missing/default records."""
    rows = read_rows(path)
    if len(rows) != len(expected) or {r["source_prompt_id"] for r in rows} != expected:
        raise ValueError("country metadata must explicitly cover every frozen source exactly once")
    for row in rows:
        for key in ("control_country_en", "control_country_zh", "rationale"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"missing explicit country metadata: {key}")
        if row.get("china_cue_relevant") is not True:
            raise ValueError("China cue relevance needs author repair")
        if any(ch in row[k] for k in ("control_country_en", "control_country_zh") for ch in "\n\r"):
            raise ValueError("country labels cannot inject extra lines")
        if row["control_country_en"].casefold().strip() in {
            "china",
            "prc",
            "people's republic of china",
        }:
            raise ValueError("non-China control cue cannot identify China")
        if row["control_country_zh"].strip() in {"中国", "中华人民共和国"}:
            raise ValueError("non-China control cue cannot identify China")
    return {r["source_prompt_id"]: r for r in rows}


def render(rows: list[dict], metadata: dict[str, dict]) -> list[dict]:
    """Render four cells by cue addition, keeping each underlying prompt exact."""
    output = []
    for row in rows:
        if row["content"] == "sensitive_full":
            continue
        sensitive = row["content"] == "sensitive_country_neutral"
        country = (
            ("China" if row["language"] == "en" else "中国")
            if sensitive
            else metadata[row["source_prompt_id"]][f"control_country_{row['language']}"]
        )
        cue = CUE_TEMPLATES[row["language"]].format(country=country)
        for present in (False, True):
            content = row["content"]
            if present:
                content = "sensitive_full" if sensitive else "matched_non_china_country"
            new = dict(row)
            new.update(
                {
                    "item_id": f"cr2-{row['source_prompt_id']}-{row['language']}-{content}-{row['frame']}",
                    "content": content,
                    "subject_arm": "sensitive" if sensitive else "control",
                    "country_cue": "present" if present else "absent",
                    "cue_text": cue if present else "",
                    "base_prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    "prompt": cue + row["prompt"] if present else row["prompt"],
                    "audit_pass": True,
                    "bank_contract": CONTRACT,
                }
            )
            output.append(new)
    validate_pairs(output)
    return sorted(output, key=lambda r: r["item_id"])


def validate_pairs(rows: list[dict], encode=None) -> dict:
    """Reject duplicate/incomplete cue pairs and optionally identical token inputs."""
    if not rows:
        raise ValueError("country-cue bank cannot be empty")
    indexed = {}
    for row in rows:
        if row["content"] not in CONTENTS:
            raise ValueError("unknown factorial content cell")
        if row["language"] not in ("en", "zh") or row["frame"] not in ("direct", "academic"):
            raise ValueError("unknown language or framing cell")
        key = (row["source_prompt_id"], row["language"], row["frame"], row["content"])
        if key in indexed:
            raise ValueError("duplicate factorial cell")
        indexed[key] = row
    changes = []
    blocks = {key[:3] for key in indexed}
    for block in sorted(blocks):
        for absent, present in (
            ("sensitive_country_neutral", "sensitive_full"),
            ("matched_non_china", "matched_non_china_country"),
        ):
            if (*block, absent) not in indexed or (*block, present) not in indexed:
                raise ValueError("missing country toggle counterpart")
            base, added = indexed[(*block, absent)], indexed[(*block, present)]
            cue = added.get("cue_text")
            expected_subject = "sensitive" if absent == "sensitive_country_neutral" else "control"
            for member, level in ((base, "absent"), (added, "present")):
                if (
                    member.get("subject_arm") != expected_subject
                    or member.get("country_cue") != level
                ):
                    raise ValueError("country toggle metadata does not match its factorial cell")
                digest = hashlib.sha256(base["prompt"].encode()).hexdigest()
                if member.get("base_prompt_sha256") != digest:
                    raise ValueError("country toggle base provenance differs")
            if base.get("cue_text") != "":
                raise ValueError("cue-absent cell has a cue")
            if (
                not cue
                or added["prompt"] != cue + base["prompt"]
                or added["prompt"] == base["prompt"]
            ):
                raise ValueError("country toggle must be a nonempty exact cue addition")
            if added["prompt"].count(cue) != 1:
                raise ValueError("country cue must occur exactly once")
            if encode is not None:
                a, b = encode(base["prompt"]), encode(added["prompt"])
                if a == b:
                    raise ValueError("country toggle tokenizes to identical inputs")
                changes.append(len(b) - len(a))
    return {
        "n_pairs": len(blocks) * 2,
        "n_identical_text_pairs": 0,
        "token_checks_run": encode is not None,
        "added_token_lengths": changes,
    }


def build(bank: Path, audit: Path, metadata: Path, out: Path) -> dict:
    """Build candidate prompts and blinded semantic-audit packets."""
    rows, source_audit = source_panel(bank, audit)
    ids = set(source_audit["passing_item_ids"])
    meta = metadata_rows(metadata, ids)
    generated = render(rows, meta)
    if len(generated) != 1360 or len({r["item_id"] for r in generated}) != 1360:
        raise ValueError("repaired factorial bank must have 1360 unique prompts")
    write_rows(out / "prompt_bank.candidate.jsonl", generated)
    audit_items = []
    for source_id in sorted(ids):
        audit_items.append(
            {
                "source_prompt_id": source_id,
                "metadata": meta[source_id],
                "prompts": [
                    r
                    for r in generated
                    if r["source_prompt_id"] == source_id and r["frame"] == "direct"
                ],
            }
        )
    for offset in range(0, len(audit_items), 10):
        write_json(
            out / "audit_packets" / f"batch_{offset // 10:03d}.json",
            {
                "contract": CONTRACT,
                "metadata_sha256": sha(metadata),
                "items": audit_items[offset : offset + 10],
            },
        )
    write_json(
        out / "build_manifest.json",
        {
            "contract": CONTRACT,
            "source_bank_sha256": sha(bank),
            "source_audit_sha256": sha(audit),
            "metadata_sha256": sha(metadata),
            "candidate_sha256": sha(out / "prompt_bank.candidate.jsonl"),
            "accepted_source_ids": sorted(ids),
        },
    )
    return {"n_prompts": len(generated), **validate_pairs(generated)}


def finalize(out: Path, metadata: Path, judgments: Path) -> dict:
    """Freeze the bank only after independent explicit semantic audits pass."""
    manifest = json.loads((out / "build_manifest.json").read_text())
    candidate = out / "prompt_bank.candidate.jsonl"
    if manifest["metadata_sha256"] != sha(metadata) or manifest["candidate_sha256"] != sha(
        candidate
    ):
        raise ValueError("candidate/metadata changed after audit preparation")
    audits = read_rows(judgments)
    ids = set(manifest["accepted_source_ids"])
    if len(audits) != 85 or {r["source_prompt_id"] for r in audits} != ids:
        raise ValueError("audit must explicitly cover all 85 sources")
    fields = (
        "country_relevance",
        "no_new_disputed_assertion",
        "same_named_subject",
        "same_requested_information",
        "language_fidelity",
    )
    for row in audits:
        if (
            any(row.get(k) is not True for k in fields)
            or not isinstance(row.get("rationale"), str)
            or not row["rationale"].strip()
        ):
            raise ValueError(f"semantic audit needs repair for item {row['source_prompt_id']}")
        if row.get("metadata_sha256") != manifest["metadata_sha256"]:
            raise ValueError("semantic audit references stale metadata")
        if row.get("auditor_id") != "/root/china_country_audit":
            raise ValueError("semantic audit lacks the assigned independent auditor identity")
    rows = read_rows(candidate)
    pair_checks = validate_pairs(rows)
    bank = out / "prompt_bank.jsonl"
    write_rows(bank, rows)
    report = {
        "contract": CONTRACT,
        "passed": True,
        "n_source_items": 85,
        "n_prompts": 1360,
        "n_audit_passing_items": 85,
        "n_audit_failing_items": 0,
        "passing_item_ids": sorted(ids),
        "prompt_bank_sha256": sha(bank),
        "source_bank_sha256": manifest["source_bank_sha256"],
        "metadata_sha256": sha(metadata),
        "independent_audit_sha256": sha(judgments),
        "pair_checks": pair_checks,
    }
    write_json(out / "bank_audit_report.json", report)
    return {"passed": True, "n_items": 85, "n_prompts": 1360, "pair_checks": pair_checks}


def upload_inputs(out: Path, receipt_path: Path) -> dict:
    """Upload the frozen input census, verify exact bytes, and publish its pointer."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    if receipt_path.resolve().is_relative_to(out.resolve()):
        raise ValueError("upload receipt must live outside the uploaded input tree")
    report = json.loads((out / "bank_audit_report.json").read_text())
    if report["contract"] != CONTRACT or report["passed"] is not True:
        raise ValueError("upload requires the repaired independent audit")
    if report["prompt_bank_sha256"] != sha(out / "prompt_bank.jsonl"):
        raise ValueError("bank changed after finalization")
    if report["metadata_sha256"] != sha(out / "country_metadata.jsonl") or report[
        "independent_audit_sha256"
    ] != sha(out / "country_audit.jsonl"):
        raise ValueError("country metadata or semantic audit changed after finalization")
    files = {
        str(path.relative_to(out)): sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "upload_verified.json"
    }
    required = {
        "prompt_bank.jsonl",
        "bank_audit_report.json",
        "country_metadata.jsonl",
        "country_audit.jsonl",
        "source_manifest.json",
        "build_manifest.json",
    }
    if not required <= files.keys() or any(name.endswith(".tmp") for name in files):
        raise ValueError("input archive is incomplete or has an unfinished atomic write")
    api = HfApi()
    repo = "superkaiba1/explore-persona-space-data"
    prefix = f"{HF_PREFIX}/inputs"
    marker_path = out / "upload_verified.json"
    prior = json.loads(marker_path.read_text()) if marker_path.exists() else None
    if prior is not None and prior["files_sha256"] != files:
        raise ValueError("refusing to overwrite an already published input version")
    print(f"[upload-inputs] verifying frozen census n_files={len(files)}", flush=True)
    if prior is None:
        commit = hub.retry_transient(
            lambda: api.upload_folder(
                repo_id=repo,
                repo_type="dataset",
                folder_path=str(out),
                path_in_repo=prefix,
                allow_patterns=list(files),
                commit_message="Issue 952 repaired country-cue inputs",
            ),
            what="issue952 repair frozen input upload",
        )
        revision = commit.oid
    else:
        revision = prior["data_revision"]
    if not revision or revision == "main":
        raise ValueError("upload did not return an immutable revision")
    verify_dir = receipt_path.parent / "input_verification" / revision
    for index, (name, expected) in enumerate(files.items(), start=1):
        destination = verify_dir / name
        hub.stage_hub_file(
            repo, f"{prefix}/{name}", destination, repo_type="dataset", revision=revision
        )
        if sha(destination) != expected:
            raise ValueError(f"uploaded input bytes mismatch: {name}")
        print(f"[upload-inputs] unit {index}/{len(files)} {name}", flush=True)
    marker = {
        "passed": True,
        "contract": CONTRACT,
        "hf_repo": repo,
        "hf_prefix": HF_PREFIX,
        "data_revision": revision,
        "prompt_bank_sha256": files["prompt_bank.jsonl"],
        "bank_audit_report_sha256": files["bank_audit_report.json"],
        "files_sha256": files,
    }
    write_json(marker_path, marker)
    pointer_commit = hub.retry_transient(
        lambda: api.upload_file(
            repo_id=repo,
            repo_type="dataset",
            path_or_fileobj=str(marker_path),
            path_in_repo=f"{prefix}/upload_verified.json",
            commit_message="Issue 952 repaired input verification pointer",
        ),
        what="issue952 repair input verification pointer",
    )
    pointer_revision = pointer_commit.oid
    fetched_marker = verify_dir / "verified_pointer.json"
    hub.stage_hub_file(
        repo,
        f"{prefix}/upload_verified.json",
        fetched_marker,
        repo_type="dataset",
        revision=pointer_revision,
    )
    if sha(fetched_marker) != sha(marker_path):
        raise ValueError("uploaded verification pointer bytes mismatch")
    result = {
        "passed": True,
        "data_revision": revision,
        "pointer_revision": pointer_revision,
        "n_files": len(files),
        "marker_sha256": sha(marker_path),
        "files_sha256": files,
    }
    write_json(receipt_path, result)
    return {k: v for k, v in result.items() if k != "files_sha256"}


def main() -> None:
    """Dispatch author preparation, candidate build, or audited finalization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "build", "finalize", "upload"), required=True
    )
    parser.add_argument("--source-bank", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase in {"prepare", "build"} and (
        args.source_bank is None or args.source_audit is None
    ):
        parser.error("source bank and audit are required")
    if args.phase in {"build", "finalize"} and args.metadata is None:
        parser.error("metadata is required")
    if args.phase == "finalize" and args.judgments is None:
        parser.error("independent judgments are required")
    if args.phase == "upload" and args.receipt is None:
        parser.error("upload receipt is required")
    if args.phase == "prepare":
        result = prepare(args.source_bank, args.source_audit, args.out_dir)
    elif args.phase == "build":
        result = build(args.source_bank, args.source_audit, args.metadata, args.out_dir)
    elif args.phase == "finalize":
        result = finalize(args.out_dir, args.metadata, args.judgments)
    else:
        result = upload_inputs(args.out_dir, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
