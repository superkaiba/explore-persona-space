"""Add pinned SAE assets to an existing handoff; no inference or fitting."""

import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path(sys.argv[1]).resolve()
assert (OUT / "bundle_summary.json").is_file()
BASE = "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data"
API = "https://huggingface.co/api/datasets/superkaiba1/explore-persona-space-data"
OLD = "cd80ba2588bb6d4291edf621176ea654bcbf2507"
THEORY = "a5cf03bbe4807361a74c427a46ad329065f75e30"
NEW = "1c5eb09f2bcabd9541a86890f0e0e945afff903a"
LIMIT = 10_000_000
SOURCES = [
    ("context_32k", OLD, "issue2552_derreplication/exactrep/analysis_tensors/sae_ctx_rep"),
    ("answer_32k", OLD, "issue2552_derreplication/exactrep/analysis_tensors/sae_rep"),
    ("context_65k", THEORY, "issue2569_theory/analysis_tensors/sae_ctx"),
    ("answer_65k", THEORY, "issue2476_turnavg/analysis_tensors/sae_c"),
    ("dense_map", OLD, "issue779_monitoring/n1m_readout/weights/L19"),
    ("sae_maps", NEW, "issue2643_sae_map"),
    ("answer_autointerp_w1", NEW, "issue2552_derreplication/exactrep/raw_completions/judge/w1"),
    ("legacy_feature_map", THEORY, "issue2569_theory/analysis_tensors/leg4"),
    ("legacy_descriptions", THEORY, "issue2569_theory/analysis_tensors/der/consumed_input"),
]


def list_source(source):
    """Enumerate a complete pinned source prefix, following all pages."""
    group, revision, prefix = source
    url = f"{API}/tree/{revision}/{prefix}?recursive=true&limit=1000"
    rows = []
    while url:
        with urlopen(Request(url), timeout=45) as response:
            entries = json.load(response)
            link = response.headers.get("Link", "")
        assert isinstance(entries, list) and entries, prefix
        for entry in entries:
            if entry["type"] != "file":
                continue
            path = entry["path"]
            rows.append(
                {
                    "group": group,
                    "path": path,
                    "revision": revision,
                    "bytes": entry["size"],
                    "git_blob": entry["oid"],
                    "lfs_sha256": (entry.get("lfs") or {}).get("oid", ""),
                    "url": f"{BASE}/blob/{revision}/{path}",
                    "download_url": f"{BASE}/resolve/{revision}/{path}",
                    "bundled_path": "",
                    "sha256": "",
                    "status": "remote_only_large_file",
                    "download_http_status": "",
                }
            )
        match = re.search(r'<([^>]+)>; rel="next"', link)
        url = match.group(1) if match else ""
    assert rows, prefix
    print(f"[inventory] {group}: {len(rows)} files", flush=True)
    return rows


def bundle(row):
    """Copy small public assets; verify their immutable Hub content identities."""
    if row["bytes"] > LIMIT:
        with urlopen(Request(row["download_url"], method="HEAD"), timeout=45) as response:
            row["download_http_status"] = response.status
            assert response.status == 200, row["path"]
        print(f"[remote accessible] {row['path']}", flush=True)
        return row
    target = OUT / "sae_assets" / row["path"]
    if target.exists():
        content = target.read_bytes()
    else:
        with urlopen(Request(row["download_url"]), timeout=45) as response:
            content = response.read(LIMIT + 1)
    assert len(content) == row["bytes"], row["path"]
    digest = hashlib.sha256(content).hexdigest()
    if row["lfs_sha256"]:
        assert digest == row["lfs_sha256"], row["path"]
    else:
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        assert blob == row["git_blob"], row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(content)
    row.update(
        bundled_path=target.relative_to(OUT).as_posix(), sha256=digest, status="bundled_verified"
    )
    print(f"[verified] {row['group']}: {Path(row['path']).name}", flush=True)
    return row


with ThreadPoolExecutor(max_workers=4) as pool:
    groups = list(pool.map(list_source, SOURCES))
    rows = list(pool.map(bundle, [row for group in groups for row in group]))
with (OUT / "sae_artifact_manifest.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "created_utc": datetime.now(UTC).isoformat(),
    "files_indexed": len(rows),
    "files_bundled": sum(row["status"] == "bundled_verified" for row in rows),
    "bytes_bundled": sum(row["bytes"] for row in rows if row["bundled_path"]),
    "large_files_linked": sum(not row["bundled_path"] for row in rows),
    "per_file_copy_limit": LIMIT,
    "sources": SOURCES,
    "scope": "SAE checkpoints/configs, dense and code mappings, saved W1 and edge autointerp.",
    "verification": (
        "Small files checked against pinned Git blob or LFS SHA256; "
        "large files linked with successful HEAD checks, not downloaded."
    ),
}
producer_rows = []
producer_root = Path("/home/thomasjiralerspong/explore-persona-space/scripts")
producer_paths = sorted(producer_root.glob("issue2643*.py"))
assert producer_paths, "No #2643 producer/loader sources found"
for path in producer_paths:
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    target = OUT / "sae_producers" / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == content, f"Source changed since snapshot: {path}"
    else:
        target.write_bytes(content)
    producer_rows.append(
        {
            "source_path": str(path),
            "bundled_path": str(target.relative_to(OUT)),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
(OUT / "sae_producer_manifest.json").write_text(json.dumps(producer_rows, indent=2) + "\n")
summary["local_producer_snapshots"] = len(producer_rows)
summary["producer_scope"] = (
    "Current working-tree source snapshots for inspection; "
    "not a claim of byte identity with historical training code."
)
(OUT / "sae_asset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
