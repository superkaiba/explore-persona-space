"""Coordinator-only publication of validated, explicitly authored annotation packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.issue1739_annotation_provenance import verify
from scripts.issue1739_covariance_ablation import sha256, write_json
from scripts.issue1739_luna_judging import BEHAVIORS, validate_rows


def publish(root, behavior, packet, sources, changed_ids=(), phase="production"):
    """Atomically publish one validated byte snapshot with explicit correction provenance."""
    if phase not in ("production", "audit"):
        raise ValueError("Invalid annotation phase")
    if not re.fullmatch(r"packet_[0-9]{4}\.json", packet):
        raise ValueError("Invalid packet filename")
    lane = root / behavior
    source = lane / phase / packet
    candidate = (
        lane
        / ("labels_candidates" if phase == "production" else "labels_audit_candidates")
        / packet
    )
    destination = lane / f"labels_{phase}" / packet
    manifest = json.loads((root / "manifest.json").read_text())
    matches = [p for p in manifest["behaviors"][behavior][phase] if p["name"] == packet]
    if len(matches) != 1 or sha256(source) != matches[0]["sha256"]:
        raise ValueError("Input packet provenance mismatch")
    data = candidate.read_bytes()
    rows = json.loads(data)
    validate_rows(json.loads(source.read_text()), rows, behavior)
    if destination.exists():
        if not changed_ids:
            raise ValueError(
                "Refusing to replace an existing packet without explicit correction IDs"
            )
        old_rows = json.loads(destination.read_text())
        old = {r["id"]: r for r in old_rows}
        new = {r["id"]: r for r in rows}
        changed = {rid for rid in new if old[rid] != new[rid]}
        if changed != set(changed_ids):
            raise ValueError(f"Correction IDs mismatch: actual={changed}; declared={changed_ids}")
        verify([new[rid] for rid in changed], sources)
        backup = lane / "coordinator_corrections" / phase / str(time.time_ns())
        backup.mkdir(parents=True)
        (backup / packet).write_bytes(destination.read_bytes())
        write_json(
            backup / "correction.json",
            dict(
                changed_ids=sorted(changed),
                before_sha256=sha256(destination),
                after_sha256=hashlib.sha256(data).hexdigest(),
                authored_sources={str(p): sha256(p) for p in sources},
            ),
        )
    else:
        if changed_ids:
            raise ValueError("Correction requested for a packet that does not exist")
        verify(rows, sources)
    # Validate the exact bytes being published, including against concurrent changes.
    if candidate.read_bytes() != data:
        raise ValueError("Candidate changed during validation")
    temporary = destination.with_suffix(".coordinator.tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    event = dict(
        published_at=time.time(),
        phase=phase,
        packet=packet,
        n=len(rows),
        sha256=sha256(destination),
        authored_sources={str(p): sha256(p) for p in sources},
        correction_ids=list(changed_ids),
    )
    with (lane / "coordinator_publications.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    print(json.dumps(event))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--behavior", choices=BEHAVIORS, required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--changed-id", action="append", default=[])
    parser.add_argument("--phase", choices=("production", "audit"), default="production")
    args = parser.parse_args()
    publish(args.root, args.behavior, args.packet, args.source, args.changed_id, args.phase)
