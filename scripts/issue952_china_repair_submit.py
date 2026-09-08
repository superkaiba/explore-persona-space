"""Serialize explicitly authored Codex decisions; never infer or default labels."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from scripts import issue952_china_repair_judges as judges

AUTHORED_FIELDS = {
    "opaque_id", "rationale", "category", *judges.SCORES, *judges.BEHAVIOR_FLAGS,
    "unassessable", "unassessable_reason",
}


def submit(packet_path: Path, authored_path: Path, agent_id: str, *, full_read: bool) -> int:
    """Attach byte lineage only after explicit full-read attestation and validation.

    Full-read attestation remains an instruction-based reviewer assertion, not a
    software guarantee. Every substantive output field must already be authored.
    """
    if full_read is not True:
        raise ValueError("reviewer must explicitly attest complete input reading")
    packet_path = packet_path.resolve()
    packet = judges.read_json(packet_path)
    if (
        packet["contract"] != judges.CONTRACT
        or packet["rubric"] != judges.RUBRIC
        or packet["rubric_sha256"] != judges.RUBRIC_SHA256
        or packet["runtime_identity"]["agent_id"] != agent_id
        or packet["runtime_identity_sha256"] != judges.sha_object(packet["runtime_identity"])
    ):
        raise ValueError("packet rubric/runtime differs from the assigned reviewer")
    expected_paths = {
        "packet_path": packet_path,
        "output_path": packet_path.with_suffix(".output.jsonl"),
        "receipt_path": packet_path.with_suffix(".read_receipt.json"),
    }
    if any(Path(packet["paths"][key]).resolve() != path for key, path in expected_paths.items()):
        raise ValueError("packet output paths are not its own sibling artifacts")
    authored = judges.read_json(authored_path)
    if (
        not isinstance(authored, list)
        or len(authored) != len(packet["items"])
        or not 0 < len(authored) <= judges.PACKET_SIZE
        or any(not isinstance(row, dict) or set(row) != AUTHORED_FIELDS for row in authored)
        or [row["opaque_id"] for row in authored] != [row["opaque_id"] for row in packet["items"]]
        or len({row["opaque_id"] for row in authored}) != len(authored)
    ):
        raise ValueError("every ordered item requires all explicitly authored fields")
    for item in packet["items"]:
        for field in ("question", "response"):
            raw = item[field].encode("utf-8")
            if (
                len(raw) != item[f"{field}_utf8_bytes"]
                or hashlib.sha256(raw).hexdigest() != item[f"{field}_sha256"]
            ):
                raise ValueError("full input bytes do not match packet lineage")
    packet_sha = judges.sha_file(packet_path)
    receipt = {
        "contract": judges.CONTRACT,
        "packet_sha256": packet_sha,
        "rubric_sha256": judges.RUBRIC_SHA256,
        "runtime_identity_sha256": packet["runtime_identity_sha256"],
        "read_complete": True,
        "items": [
            {**{key: item[key] for key in judges.RECEIPT_ITEM_FIELDS if key != "read_complete"},
             "read_complete": True}
            for item in packet["items"]
        ],
    }
    receipt_bytes = judges._bytes_json(receipt)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    record = {"packet_sha256": packet_sha}
    judgments = []
    for item, decision in zip(packet["items"], authored, strict=True):
        row = {
            **decision,
            "request_sha256": item["request_sha256"],
            "packet_sha256": packet_sha,
            "rubric_sha256": judges.RUBRIC_SHA256,
            "runtime_identity_sha256": packet["runtime_identity_sha256"],
            "input_read_receipt_sha256": receipt_sha,
        }
        judges.validate_decision(row, item, packet, record, receipt_sha)
        judgments.append(row)
    # Validate all decisions before publishing either artifact. Immutable writes
    # permit an interrupted receipt-only submission to resume identical bytes.
    judges._validate_receipt(receipt, packet, record)
    judges.write_immutable(expected_paths["receipt_path"], receipt_bytes)
    judges.write_immutable(expected_paths["output_path"], judges._jsonl_bytes(judgments))
    return len(judgments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--authored", type=Path, required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--attest-full-read", action="store_true")
    args = parser.parse_args()
    count = submit(args.packet, args.authored, args.agent_id, full_read=args.attest_full_read)
    print(f"Saved {count} explicit judgments; receipt and output hashes validated")


if __name__ == "__main__":
    main()
