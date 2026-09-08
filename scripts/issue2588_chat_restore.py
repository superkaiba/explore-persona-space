"""Restore the immutable thinking pilot's saved rows without rewriting provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import issue2588_chat_grant as CG

REPO_ID = "superkaiba1/explore-persona-space-data"
REVISION = "1c4adfdd306108d36e5267865bbb0fa350753406"
PREFIX = "issue2588_capability_panel_cap_long/generic/qwen3-chat-v3/smoke/q3_8b/think/partial"
IDENTITY_SHA256 = "b2276f0694a6dfc2d35744d6451ff33ce614b3f11f73e6eefc7d4ece0066d1b8"
MANIFEST_SHA256 = "f8e61e8197ffe4c23390c9702a884e63b30cf27d539a5c3b6a456848be890073"
EXPECTED_FILES = {
    "packed_generation_checkpoints/pack_manifest.json",
    "packed_generation_checkpoints/raw_completions_train_10k_partial_initial_rows.shard00.jsonl",
    "phase_done/prologue.json",
    "phase_done/stage.json",
    "prologue.json",
    "run_identity.json",
    "stage.json",
}


def restore_staged(staged: Path, cell: Path) -> dict:
    """Verify frozen identity and every restored row; old runtime receipts stay archived."""
    from issue1739_pack import unpack_shards

    manifest = staged / "packed_generation_checkpoints/pack_manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("saved packed manifest bytes differ from immutable snapshot")
    raw = (staged / "run_identity.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != IDENTITY_SHA256:
        raise RuntimeError("saved pilot identity bytes differ")
    identity = json.loads(raw)
    if identity["source_sha"] != CG.SCIENCE_SHA:
        raise RuntimeError("saved scientific source mismatch")
    cell.mkdir(parents=True, exist_ok=True)
    target = cell / "run_identity.json"
    if target.exists():
        if target.read_bytes() != raw:
            raise RuntimeError("refusing to replace existing scientific identity")
    else:
        with target.open("xb") as stream:
            stream.write(raw)
    # Canonical unpacker checks shard hashes, counts, safe relative paths and
    # refuses to overwrite a pre-existing checkpoint with different bytes.
    unpack_shards(staged / "packed_generation_checkpoints", cell)
    expected = set(range(400)) - {338, 377}
    saved = []
    for i in sorted(expected):
        p = cell / "raw_completions/train_10k/partial/initial/rows" / f"row{i:06d}.json"
        rec = json.loads(p.read_text())
        row = rec["row"]
        if (
            rec["identity"] != identity
            or row["row_id"] != f"train_10k_{i}"
            or row["stage"] != "train_10k"
            or row["gen_seed"] != 42
            or row["cap"] != 32768
            or row["finish_reason"] != "stop"
            or len(row["sampled_token_ids"]) != row["n_comp_tokens"]
            or len(row["prompt_ids"]) != row["n_prompt_tokens"]
        ):
            raise RuntimeError(f"invalid saved response {p.name}")
        saved.append(
            {"path": str(p.relative_to(cell)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        )
    return {
        "status": "verified",
        "revision": REVISION,
        "prefix": PREFIX,
        "science_sha": CG.SCIENCE_SHA,
        "restored_original_rows": len(saved),
        "files": saved,
    }


def main(argv: list[str] | None = None) -> int:
    """Stage the exact small prefix and adopt only identity plus raw row checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    science = CG.science_root(args.science_root)
    sys.path[:0] = [str(science / "src"), str(science / "scripts")]
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.atomic_io import write_json_atomic
    from explore_persona_space.orchestrate import hub

    destination = args.out_root / "generic/qwen3-chat-v3/restore_inputs"
    files = hub.stage_hub_prefix(REPO_ID, PREFIX, destination, revision=REVISION, max_workers=4)
    staged = destination / PREFIX
    if {p.relative_to(staged).as_posix() for p in files} != EXPECTED_FILES:
        raise RuntimeError("immutable partial payload membership differs")
    result = restore_staged(staged, args.out_root / "generic/qwen3-chat-v3/smoke_cap_long/q3_8b_b")
    write_json_atomic(destination / "restore_verification.json", result)
    print(
        f"[restore] PASS: {result['restored_original_rows']} original response files preserved",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
