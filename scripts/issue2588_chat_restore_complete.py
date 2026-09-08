"""Restore all 600 immutable thinking-pilot responses, never generate replacements."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import issue2588_chat_grant as CG

REPO_ID = "superkaiba1/explore-persona-space-data"
REVISION = "17c35c79ccd7cfcab7f1d7210633471d598758c1"
PREFIX = "issue2588_capability_panel_cap_long/generic/qwen3-chat-v3/smoke/q3_8b/think/partial"
TERMINAL_REVISION = "716e16e2d9cebe7d07ffac7eaea013c4cb6dd49e"
RECEIPT_PATH = (
    "issue2588_capability_panel_cap_long/generic/qwen3-chat-v3/diagnostics/"
    "terminal_20260908T154216/snapshot/generic/qwen3-chat-v3/"
    "smoke_cap_long/q3_8b_b/uploads/upload-raw.json"
)
RECEIPT_SHA256 = "84abe82a694fc3739656342d81425a674c24cec0b435ab99ee8e205d1bcc75ae"
TREE_SHA256 = "cc283455fcaa4f7f0c305cd5d5e085547859095b727a2a04cc418e2bc9923c05"
STAGES = {"train_10k": 400, "val_400": 50, "test_1000": 50, "ceiling_s43": 50, "ceiling_s44": 50}


def digest(data: bytes) -> str:
    """Return the exact byte hash used by the producer."""
    return hashlib.sha256(data).hexdigest()


def put_identical(path: Path, data: bytes) -> None:
    """Permit idempotent restores but never replace a differing output."""
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to replace differing artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)


def restore_staged(staged: Path, receipt: Path, cell: Path) -> dict:
    """Check the complete input tree, invert its transports, and reconcile all IDs."""
    from issue1739_pack import unpack_shards

    files = {p.relative_to(staged).as_posix(): p for p in staged.rglob("*") if p.is_file()}
    hashes = {name: digest(p.read_bytes()) for name, p in files.items()}
    tree_hash = digest(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())
    if tree_hash != TREE_SHA256 or digest(receipt.read_bytes()) != RECEIPT_SHA256:
        raise RuntimeError("immutable complete-pilot payload or raw receipt bytes differ")
    identity = json.loads(files["run_identity.json"].read_text())
    if identity["source_sha"] != CG.SCIENCE_SHA or identity["smoke"] is not True:
        raise RuntimeError("not the frozen pilot identity")
    transport = set()
    recovered = {}
    for name, p in files.items():
        if not name.endswith(".manifest.json"):
            continue
        manifest = json.loads(p.read_text())
        parts = manifest["parts"]
        if not parts or len(parts) != len(set(parts)):
            raise RuntimeError("empty/duplicate text shards")
        parent = Path(name).parent
        if any(Path(part).name != part for part in [*parts, manifest["source"]]):
            raise RuntimeError("unsafe text shard path")
        part_names = [(parent / part).as_posix() for part in parts]
        for part, rel in zip(parts, part_names, strict=True):
            if hashes[rel] != manifest["sha256"][part]:
                raise RuntimeError("text shard hash mismatch")
        data = b"".join(files[rel].read_bytes() for rel in part_names)
        if digest(data) != manifest["source_sha256"]:
            raise RuntimeError("reconstructed text hash mismatch")
        recovered[(parent / manifest["source"]).as_posix()] = data
        transport.update([name, *part_names])
    # Failed-capture diagnostics are archived in staged, not adopted as new output.
    for name, p in files.items():
        if (
            name in transport
            or name.startswith("packed_generation_checkpoints/")
            or name == "capture_input_validation.json"
            or name.endswith("_capture_drops.json")
        ):
            continue
        recovered[name] = p.read_bytes()
    recovered["uploads/upload-raw.json"] = receipt.read_bytes()
    for name, data in recovered.items():
        put_identical(cell / name, data)
    unpack_shards(staged / "packed_generation_checkpoints", cell)
    keys = set()
    records = sorted((cell / "raw_completions").rglob("row*.json"))
    for p in records:
        rec = json.loads(p.read_text())
        row = rec["row"]
        stage = row["stage"]
        seed = int(stage.removeprefix("ceiling_s")) if stage.startswith("ceiling_s") else 42
        key = (stage, row["row_id"], seed)
        if (
            rec["identity"] != identity
            or row["gen_seed"] != seed
            or key in keys
            or row["cap"] != 32768
            or p.parent.parent.name != "initial"
            or len(row["sampled_token_ids"]) != row["n_comp_tokens"]
            or len(row["prompt_ids"]) != row["n_prompt_tokens"]
        ):
            raise RuntimeError(f"invalid or duplicate saved response: {p}")
        keys.add(key)
    expected = {
        (
            stage,
            f"{stage}_{i}",
            int(stage.removeprefix("ceiling_s")) if stage.startswith("ceiling_s") else 42,
        )
        for stage, n in STAGES.items()
        for i in range(n)
    }
    if keys != expected:
        raise RuntimeError("restored response keys differ from complete 600-row pilot")
    return {
        "status": "verified",
        "revision": REVISION,
        "tree_sha256": tree_hash,
        "raw_receipt_revision": TERMINAL_REVISION,
        "restored_original_rows": len(keys),
        "science_sha": CG.SCIENCE_SHA,
        "input_files": len(files),
        "input_bytes": sum(p.stat().st_size for p in files.values()),
    }


def validate_consumer(science: Path, out_root: Path) -> dict:
    """Ask the unchanged scientific consumer to open and hash all prerequisite phases."""
    sys.path[:0] = [str(science / "scripts"), str(science / "src")]
    import issue2588_run_cell as core

    if Path(core.__file__).resolve() != science / "scripts/issue2588_run_cell.py":
        raise RuntimeError("consumer import is not the frozen driver")
    args = core._build_parser().parse_args(
        [
            "--surface",
            "generic",
            "--run-id",
            "qwen3-chat-v3",
            "--cell",
            "q3_8b_b",
            "--phase",
            "capture",
            "--smoke",
            "--device",
            "cuda",
            "--out-root",
            str(out_root),
        ]
    )
    cell = core.PC.cell_by_key(args.cell)
    paths = core._paths(args, cell)
    phases = ("prologue", "stage", "gen", "parse", "upload-raw")
    for phase in phases:
        if not core._phase_complete(args, paths, phase):
            raise RuntimeError(f"restored phase is not consumer-complete: {phase}")
    return {"status": "PASS", "validated_phases": list(phases)}


def main(argv: list[str] | None = None) -> int:
    """Stage only the pinned small payload and validate it before any GPU capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    science = CG.science_root(args.science_root)
    from explore_persona_space.orchestrate.env import load_dotenv
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.atomic_io import write_json_atomic
    from huggingface_hub import hf_hub_download

    load_dotenv()
    destination = args.out_root / "generic/qwen3-chat-v3/restore_complete_inputs"
    hub.stage_hub_prefix(REPO_ID, PREFIX, destination, revision=REVISION, max_workers=4)
    receipt = Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                REPO_ID,
                RECEIPT_PATH,
                repo_type="dataset",
                revision=TERMINAL_REVISION,
                local_dir=str(destination / "terminal"),
            ),
            what="immutable pilot raw receipt",
        )
    )
    result = restore_staged(
        destination / PREFIX,
        receipt,
        args.out_root / "generic/qwen3-chat-v3/smoke_cap_long/q3_8b_b",
    )
    result["consumer"] = validate_consumer(science, args.out_root)
    write_json_atomic(destination / "restore_verification.json", result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
