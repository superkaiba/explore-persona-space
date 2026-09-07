"""Snapshot final natural-scaling evidence; never change or delete source files.

Large outputs must already have immutable, content-verified Hub homes. Remaining
metadata is packed losslessly for a small off-pod upload. No teardown is performed.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate.secret_scrub import scan_bytes
from scripts.issue1739_natural_audit import aggregate, file_sha, require, verify_remote_cell
from scripts.issue1739_natural_data import atomic_json, load_parts, write_parts

PREFIX = "issue1739_natural100k_20260906"
SCIENTIFIC_COMMIT = "9f6a6fedb9a97365bfbfb0c241d7712d94ce43c0"
DATA_REVISIONS = {
    "prepared": "195b718a44d0a4799c8f3802d29783b8214b96f4",
    "generated": "ff5398d333776c027cca5f4f2d53cd5f9af57548",
    "captured": "55e49ff70f640e674a9cc6aeb94f0e7b9f5edf80",
    "store": "972fd732b40a32619be508e97d1adf695afb2436",
}
PACK_PART_BYTES = 6_000_000
FROZEN_BLOBS = {
    "evil": "ec6995787af3437fbd86d2053b86c428a957efd7",
    "sycophancy": "f41eade2c65291cd0eaa96b37cbdd1e74fde4de7",
    "hallucination": "522c90302bb75d7f363058d9a6a2706fa6594570",
}


def regular_files(root: Path) -> dict[str, Path]:
    """Enumerate exactly; refuse symlinks rather than following an unknown tree."""
    require(root.is_dir(), f"Missing source directory: {root}")
    paths = sorted(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), f"Unexpected symlink under {root}")
    return {str(p.relative_to(root)): p for p in paths if p.is_file()}


def verify_data_tree(root: Path, phase: str) -> tuple[dict, dict]:
    """Verify the complete immutable data tree, excluding only download-cache files."""
    files = {k: v for k, v in regular_files(root).items() if ".cache" not in Path(k).parts}
    hashes = {k: file_sha(v) for k, v in files.items()}
    receipt = {
        "prefix": f"{PREFIX}/{phase}",
        "revision": DATA_REVISIONS[phase],
        "files_sha256": hashes,
    }
    # A hardlink view lets the already-reviewed exact-tree verifier ignore the
    # HF client's local .cache bookkeeping without weakening remote set equality.
    with tempfile.TemporaryDirectory(prefix="natural-proof-", dir=root.parent) as tmp:
        view = Path(tmp)
        for name, path in files.items():
            target = view / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
        proof = verify_remote_cell(view, receipt)
    return hashes, proof


def pack_record(path: Path, key: str) -> list[dict]:
    """Scan ORIGINAL bytes before lossless encoding; never hide credentials in a pack."""
    raw = path.read_bytes()
    require(not scan_bytes(raw), f"Original metadata failed credential scan: {key}")
    n_parts = max(1, (len(raw) + PACK_PART_BYTES - 1) // PACK_PART_BYTES)
    return [
        {
            "path": key,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "part": part,
            "n_parts": n_parts,
            "content_base64": base64.b64encode(
                raw[part * PACK_PART_BYTES : (part + 1) * PACK_PART_BYTES]
            ).decode("ascii"),
        }
        for part in range(n_parts)
    ]


def verify_pack(directory: Path, expected: dict[str, str]) -> None:
    """Decode every member and reconcile exact paths, lengths and hashes."""
    rows = load_parts(directory)
    require({r["path"] for r in rows} == set(expected), "Packed member path set differs")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)
    for key, parts in grouped.items():
        parts.sort(key=lambda row: row["part"])
        require(
            [r["part"] for r in parts] == list(range(parts[0]["n_parts"])),
            "Packed member part set differs",
        )
        require(
            len({(r["bytes"], r["sha256"], r["n_parts"]) for r in parts}) == 1,
            "Packed member metadata differs across parts",
        )
        raw = b"".join(base64.b64decode(r["content_base64"], validate=True) for r in parts)
        require(len(raw) == parts[0]["bytes"], "Packed member byte count differs")
        require(
            hashlib.sha256(raw).hexdigest() == parts[0]["sha256"] == expected[key],
            "Packed member content differs",
        )


def selected_sources(root: Path, launcher_dir: Path, helper_sources: list[Path]) -> dict[str, Path]:
    """Select the complete source namespace; reject symlink or helper-name ambiguity."""
    selected = {"run/" + k: p for k, p in regular_files(root).items()}
    require(launcher_dir.is_dir(), "Missing launcher-log directory")
    launchers = sorted(launcher_dir.glob("*1739*"))
    require(not any(p.is_symlink() for p in launchers), "Launcher source is a symlink")
    selected.update({"launcher_logs/" + p.name: p for p in launchers if p.is_file()})
    require(
        len({p.name for p in helper_sources}) == len(helper_sources),
        "Helper-source basenames collide",
    )
    require(
        all(p.is_file() and not p.is_symlink() for p in helper_sources),
        "Missing or symlink helper source",
    )
    selected.update({"helper_sources/" + p.name: p for p in helper_sources})
    return selected


def snapshot(args) -> dict:
    """Require completed science, reconcile every source and preserve remaining metadata."""
    started = time.monotonic()
    for pid in args.wait_pid:
        require(not Path(f"/proc/{pid}").exists(), f"Owned producer still exists: {pid}")
    sentinel = args.success_sentinel
    if not sentinel.exists():
        sentinel = Path(str(sentinel) + ".processed")
    signal = json.loads(sentinel.read_text())
    require(
        signal["payload"]["phase"] == "fits" and signal["payload"]["rc"] == 0,
        "Wrong scientific completion sentinel",
    )
    report = json.loads(Path(signal["payload"]["report"]).read_text())
    require(
        report["phase"] == "fits" and report["status"] == "complete",
        "Scientific completion report is not complete",
    )
    require(report["pid"] in args.wait_pid, "Completion report is from another producer")
    audit = json.loads((args.root / "natural_scaling_audit.json").read_text())
    require(audit["scientific_commit"] == SCIENTIFIC_COMMIT, "Wrong scientific code identity")
    require(
        aggregate(audit["cells"], require_full=True) == audit["aggregate"],
        "Final audit aggregate differs",
    )
    preserved, proofs = {}, {}
    for phase in DATA_REVISIONS:
        hashes, proof = verify_data_tree(args.root / "pool" / phase, phase)
        preserved.update(
            {f"pool/{phase}/{k}": (v, "immutable_data_output") for k, v in hashes.items()}
        )
        proofs[phase] = proof
        print(f"[preserve] data={phase} files={len(hashes)} verified", flush=True)
    for cell in audit["cells"]:
        require(
            cell["remote_verification"]["all_names_sizes_content_verified"],
            "A fit cell lacks remote content verification",
        )
        prefix = f"results/u{cell['generic_u']}/{cell['behavior']}/seed{cell['seed']}"
        preserved.update(
            {f"{prefix}/{k}": (v, "immutable_fit_output") for k, v in cell["files_sha256"].items()}
        )
    reused = json.loads((args.root / "reused_transfer_manifest.json").read_text())
    require(
        reused["source_revision"] == "7a47ff5ce42f16308bebaba29c1286a4e9bc8008",
        "Wrong fixed-input revision",
    )
    actual_reused = regular_files(args.root / "reused")
    require(set(actual_reused) == set(reused["files"]), "Fixed-input file set changed")
    for name, record in reused["files"].items():
        preserved[f"reused/{name}"] = (record["sha256"], "pinned_reused_input")
    for behavior, blob in FROZEN_BLOBS.items():
        relative = f"reused/eval_results/{behavior}/arm_results/all_arms_spearman.json"
        raw = (args.root / relative).read_bytes()
        actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(actual_blob == blob, "Frozen training-summary Git blob differs")
        preserved[relative] = (hashlib.sha256(raw).hexdigest(), "frozen_git_input")
    files = regular_files(args.root)
    require(set(preserved) <= set(files), "A declared durable file is missing locally")
    all_sources = selected_sources(args.root, args.launcher_dir, args.helper_source)
    require(not args.dest.exists(), "Use a fresh preservation destination")
    args.dest.mkdir(parents=True)
    rows, packed, packed_hashes = [], [], {}
    for i, (key, path) in enumerate(sorted(all_sources.items()), 1):
        digest = file_sha(path)
        relative = key.removeprefix("run/") if key.startswith("run/") else None
        if relative in preserved:
            expected, disposition = preserved[relative]
            require(digest == expected, f"Source changed from its durable proof: {key}")
        else:
            disposition = "lossless_metadata_pack"
            packed.extend(pack_record(path, key))
            require(packed[-1]["sha256"] == digest, f"Source changed while packing: {key}")
            packed_hashes[key] = digest
        rows.append(
            {
                "source": str(path),
                "key": key,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "disposition": disposition,
            }
        )
        if i % 1000 == 0:
            print(
                f"[preserve] files={i}/{len(all_sources)} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    write_parts(args.dest / "source_inventory", rows)
    write_parts(args.dest / "metadata", packed)
    verify_pack(args.dest / "metadata", packed_hashes)
    require(load_parts(args.dest / "source_inventory") == rows, "Census round-trip differs")
    # A fresh final name+byte pass catches a producer or helper changing a source
    # during the snapshot, including an extra file appearing at the out-root.
    require(
        selected_sources(args.root, args.launcher_dir, args.helper_source) == all_sources,
        "Complete source name set changed during snapshot",
    )
    for row in rows:
        require(file_sha(Path(row["source"])) == row["sha256"], "Source changed during snapshot")
    result = {
        "status": "PASS",
        "source_files": len(rows),
        "metadata_files": len(packed_hashes),
        "metadata_parts": len(packed),
        "dispositions": dict(Counter(r["disposition"] for r in rows)),
        "data_remote_proofs": proofs,
        "scientific_commit": SCIENTIFIC_COMMIT,
        "audit_sha256": file_sha(args.root / "natural_scaling_audit.json"),
        "fit_cells": 150,
        "source_name_set_and_content_verified": True,
        "lossless_metadata_roundtrip_verified": True,
        "reconstruction": "Group metadata by path, concatenate decoded content_base64 in part order, then verify bytes and SHA256. Each part record declares the complete source-file size/hash, not just the fragment.",
        "wall_s": time.monotonic() - started,
    }
    atomic_json(args.dest / "preservation.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--success-sentinel", type=Path, required=True)
    p.add_argument("--launcher-dir", type=Path, required=True)
    p.add_argument("--wait-pid", action="append", type=int, required=True)
    p.add_argument("--helper-source", action="append", type=Path, default=[])
    args = p.parse_args()
    require(
        not args.dest.resolve().is_relative_to(args.root.resolve()),
        "Preservation destination must be outside the source tree",
    )
    print(json.dumps(snapshot(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
