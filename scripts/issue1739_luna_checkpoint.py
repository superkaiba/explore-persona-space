"""Archive an explicitly incomplete annotation checkpoint with remote hash verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.issue1739_large_pool_risk import HfApi, REPO, retry_transient, verify_blob
from scripts.issue1739_covariance_ablation import sha256, write_json


def validate_final(root, summary):
    """Require the complete, quality-accepted label snapshot used by the analysis."""
    from scripts.issue1739_luna_analysis import load_labels

    _, _, hashes = load_labels(root)
    acceptance = json.loads((root / "quality_acceptance.json").read_text())
    results = json.loads(summary.read_text())
    if not acceptance["content_review_complete"] or acceptance["annotation_file_sha256"] != hashes:
        raise ValueError("Final archive requires current quality acceptance")
    if not results["annotation_complete"] or not results["analysis_complete"]:
        raise ValueError("Final archive requires completed analysis")
    if results["annotations"] != hashes or results["quality_acceptance_sha256"] != sha256(
        root / "quality_acceptance.json"
    ):
        raise ValueError("Analysis does not match accepted labels")
    latest = {}
    with (root / "coordinator_content_review.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if "behavior" not in row:
                matches = [key for key in latest if key[2] == row["id"]]
                if len(matches) != 1:
                    raise ValueError("Ambiguous legacy content-review event")
                key = matches[0]
            else:
                key = (row["behavior"], row.get("phase", "production"), row["id"])
            latest[key] = row
    accepted_statuses = {"resolved", "corrected", "accepted", "accepted_with_rationale_caveat"}
    if any(row["status"] not in accepted_statuses for row in latest.values()):
        raise ValueError("Unresolved annotation content review")
    return hashes


def final_archive(root, output, summary, assets):
    """Archive accepted annotations and reviewed deliverables; never writes a monitor sentinel."""
    import shutil

    hashes = validate_final(root, summary)
    summary_hash = sha256(summary)
    accepted_hash = sha256(root / "quality_acceptance.json")
    asset_hashes = {
        str(p.relative_to(assets)): sha256(p) for p in assets.rglob("*") if p.is_file()
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = output / f"luna_final_{stamp}"
    files = pack_text(root, archive)
    if any(files.get(name) != digest for name, digest in hashes.items()):
        raise ValueError("Packed annotation snapshot differs from accepted labels")
    if files.get("quality_acceptance.json") != accepted_hash:
        raise ValueError("Quality acceptance changed while packing")
    if not assets.is_dir() or any(p.is_symlink() for p in assets.rglob("*")):
        raise ValueError("Invalid final deliverable directory")
    shutil.copytree(assets, archive / "deliverables")
    shutil.copyfile(summary, archive / "analysis.json")
    if sha256(archive / "analysis.json") != summary_hash:
        raise ValueError("Analysis changed while copying")
    copied_hashes = {
        str(p.relative_to(archive / "deliverables")): sha256(p)
        for p in (archive / "deliverables").rglob("*") if p.is_file()
    }
    if not asset_hashes or copied_hashes != asset_hashes:
        raise ValueError("Deliverables changed while copying or are empty")
    manifest_path = archive / "CHECKPOINT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        annotation_complete=True,
        analysis_complete=True,
        description="Accepted final annotations, original rejected/superseded records, independent "
        "repeat audits, analysis and deliverables. Aggregate only canonical labels listed in "
        "annotation_file_sha256. Model repeatability is not human-verified accuracy.",
        annotation_file_sha256=hashes,
        analysis_sha256=summary_hash,
        deliverable_sha256={
            str(p.relative_to(archive)): sha256(p)
            for p in (archive / "deliverables").rglob("*")
            if p.is_file()
        },
    )
    write_json(archive / "FINAL_MANIFEST.json", manifest)
    manifest_path.unlink()  # Replace the temporary incomplete packing manifest before upload.
    if validate_final(root, summary) != hashes:
        raise ValueError("Labels changed while preparing final archive")
    prefix = "issue1739_luna_large_pool_20260917/final/" + archive.name
    paths = {
        prefix + "/" + str(p.relative_to(archive)): p
        for p in archive.rglob("*") if p.is_file()
    }
    staged_hashes = {name: sha256(path) for name, path in paths.items()}
    api = HfApi()
    result = retry_transient(
        lambda: api.upload_folder(
            repo_id=REPO, repo_type="dataset", folder_path=archive,
            path_in_repo=prefix, commit_message="Issue1739 completed Luna large-pool retrieval analysis",
        ),
        what="Final Luna archive upload",
    )
    rows = retry_transient(
        lambda: list(api.list_repo_tree(
            repo_id=REPO, repo_type="dataset", path_in_repo=prefix,
            recursive=True, revision=result.oid,
        )),
        what="Final Luna archive verification",
    )
    rows = [row for row in rows if hasattr(row, "size")]
    if {row.path for row in rows} != set(paths):
        raise ValueError("Final remote coverage mismatch")
    for row in rows:
        path = paths[row.path]
        if sha256(path) != staged_hashes[row.path]:
            raise ValueError("Staged archive mutated during upload")
        if row.size != path.stat().st_size:
            raise ValueError("Final remote byte-size mismatch")
        if row.lfs:
            if row.lfs.sha256 != sha256(path):
                raise ValueError("Final LFS hash mismatch")
        else:
            verify_blob(path, dict(path=row.path, size=row.size, blob_id=row.blob_id))
    if validate_final(root, summary) != hashes or sha256(summary) != summary_hash:
        raise ValueError("Labels or analysis changed during final upload")
    verified = dict(
        verified_at=time.time(), revision=result.oid, path=prefix,
        annotation_complete=True, analysis_complete=True,
        n_annotation_artifacts=len(files), annotation_file_sha256=hashes,
        remote_artifact_sha256={str(p.relative_to(archive)): sha256(p) for p in paths.values()},
    )
    write_json(output / f"luna_final_{stamp}.verified.json", verified)
    print(json.dumps({k: v for k, v in verified.items() if not k.endswith("sha256")}))


def pack_text(root, destination, shard_max_bytes=8_000_000):
    """Adapt issue1739_pack's line-shard pattern, preserving arbitrary UTF-8 file bytes."""
    destination.mkdir(parents=True, exist_ok=False)
    hashes, shards, buffer, excluded = {}, [], bytearray(), {}

    def flush():
        if buffer:
            path = destination / f"files.shard{len(shards):03d}.jsonl"
            path.write_bytes(buffer)
            shards.append(dict(name=path.name, bytes=len(buffer), sha256=sha256(path)))
            buffer.clear()

    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if not path.is_file() or path.name.startswith(".") or path.suffix == ".tmp":
            continue
        if path.suffix == ".pyc" and "__pycache__" in path.parts:
            excluded[str(path.relative_to(root))] = "Regenerable Python bytecode; source retained"
            continue
        data = path.read_bytes()
        name = str(path.relative_to(root))
        hashes[name] = hashlib.sha256(data).hexdigest()
        line = (
            json.dumps(dict(src=name, text=data.decode("utf-8")), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        if len(line) > shard_max_bytes:
            raise ValueError(f"Single file exceeds text-shard cap: {path}")
        if len(buffer) + len(line) > shard_max_bytes:
            flush()
        buffer.extend(line)
    flush()
    if not hashes:
        raise ValueError("Empty annotation archive")
    write_json(
        destination / "CHECKPOINT_MANIFEST.json",
        dict(
            created_at=time.time(),
            annotation_complete=False,
            analysis_complete=False,
            description="Incremental checkpoint only. Labels are unfinished and under quality review. "
            "Rejected and superseded labels are archived for provenance and must never be aggregated.",
            artifact_sha256=hashes,
            excluded=excluded,
            shards=shards,
            restore="Read every manifest shard by physical JSONL lines; UTF-8 encode each text field. Verify artifact_sha256 before writing src. Never overwrite a differing file.",
        ),
    )
    return hashes


def checkpoint(root, output):
    output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = output / f"luna_partial_{stamp}"
    hashes = pack_text(root, archive)
    prefix = "issue1739_luna_large_pool_20260917/checkpoints/" + archive.name
    api = HfApi()
    result = retry_transient(
        lambda: api.upload_folder(
            repo_id=REPO,
            repo_type="dataset",
            folder_path=archive,
            path_in_repo=prefix,
            commit_message="Issue1739 incomplete Luna annotation checkpoint",
        ),
        what="Luna checkpoint upload",
    )
    paths = {prefix + "/" + p.name: p for p in archive.iterdir()}
    rows = retry_transient(
        lambda: api.get_paths_info(
            repo_id=REPO,
            repo_type="dataset",
            paths=list(paths),
            revision=result.oid,
        ),
        what="Luna checkpoint remote verification",
    )
    if {row.path for row in rows} != set(paths):
        raise ValueError("Checkpoint remote coverage mismatch")
    for row in rows:
        if row.lfs:
            raise ValueError("Text checkpoint unexpectedly routed to LFS")
        verify_blob(paths[row.path], dict(path=row.path, size=row.size, blob_id=row.blob_id))
    verified = dict(
        verified_at=time.time(),
        revision=result.oid,
        path=prefix,
        shard_sha256={p.name: sha256(p) for p in paths.values()},
        annotation_complete=False,
        analysis_complete=False,
        n_files=len(hashes),
    )
    write_json(output / f"luna_partial_{stamp}.verified.json", verified)
    print(json.dumps(verified))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-summary", type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if args.final_summary is not None:
        if args.assets is None:
            parser.error("--final-summary requires --assets")
        final_archive(args.root, args.output, args.final_summary, args.assets)
    else:
        if args.assets is not None:
            parser.error("--assets requires --final-summary")
        checkpoint(args.root, args.output)
