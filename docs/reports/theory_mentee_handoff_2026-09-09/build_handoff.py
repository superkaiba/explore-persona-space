"""Package existing theory artifacts and index immutable remote data; no experiments.

Run with the repository's existing uv environment. The output is a new directory.
Raw evidence remains unchanged. The manifest preserves duplicate source locations
while storing each small file only once. Large tensors are indexed, not copied.
"""

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
DATA = Path("/mnt/eps-data/thomasjiralerspong")
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve()
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REV = "a5cf03bbe4807361a74c427a46ad329065f75e30"
GITHUB = "https://github.com/superkaiba/explore-persona-space"
MAX_COPY_BYTES = 10_000_000
SKIP_SUFFIXES = {".pyc", ".log", ".aux", ".out", ".toc", ".lock"}


def get_json(url):
    """Read public metadata, failing visibly on network or schema errors."""
    with urlopen(
        Request(url, headers={"User-Agent": "EPS-theory-handoff/1"}), timeout=45
    ) as response:
        return json.load(response), response.headers.get("Link", "")


def git_snapshot(root):
    """Return the exact checked-out revision and its tracked blob identities."""
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    raw = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", sha])
    blobs = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        _mode, kind, oid = meta.decode().split()
        if kind == "blob":
            blobs[name.decode()] = oid
    return sha, blobs


def enumerate_group(group, root, paths, git_root=None):
    """Inventory every regular file beneath explicit, existing source paths."""
    root = Path(root)
    assert root.is_dir(), root
    sha, blobs = git_snapshot(git_root) if git_root else ("", {})
    count = 0
    for relative in paths:
        source = root / relative
        if not source.exists():
            source_exclusions.append(
                {"group": group, "path": str(source), "reason": "not materialized in this worktree"}
            )
            print(f"[coverage] {group}: not materialized: {relative}", flush=True)
            continue
        files = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in files:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            size = path.stat().st_size
            name = path.relative_to(root).as_posix()
            row = {
                "group": group,
                "source_path": str(path),
                "relative_path": name,
                "bytes": size,
                "sha256": "",
                "git_revision": sha,
                "url": "",
                "bundled_path": "",
                "status": "indexed_only_large_file",
            }
            if size <= MAX_COPY_BYTES:
                before = path.stat()
                content = path.read_bytes()
                after = path.stat()
                assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size), (
                    path
                )
                digest = hashlib.sha256(content).hexdigest()
                row["sha256"] = digest
                git_name = path.relative_to(git_root).as_posix() if git_root else ""
                git_blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + content).hexdigest()
                if blobs.get(git_name) == git_blob:
                    row["url"] = f"{GITHUB}/blob/{sha}/{quote(git_name)}"
                if digest not in stored:
                    target = Path("artifacts") / group / name
                    (OUT / target).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, OUT / target)
                    assert hashlib.sha256((OUT / target).read_bytes()).hexdigest() == digest
                    stored[digest] = target.as_posix()
                row["bundled_path"] = stored[digest]
                row["status"] = "bundled_exact_bytes"
            rows.append(row)
            count += 1
    print(f"[local] {group}: {count} source files", flush=True)
    assert count, f"No materialized files for source group {group}"


assert not OUT.exists(), f"Refusing to overwrite an existing handoff: {OUT}"
OUT.mkdir(parents=True)
rows = []
stored = {}
source_exclusions = []

theory_trees = [
    ("core_and_category", DATA / "wt-2569-category-validation"),
    ("original_battery", DATA / "wt-2569-report-final"),
    ("basis", DATA / "wt-2569-basis"),
    ("eigen_v2", DATA / "wt-2569-eigen-v2"),
    ("kernel", DATA / "wt-2569-kernel"),
    ("direction_examples", DATA / "wt-2569-direction-examples"),
    ("third_family_code", DATA / "wt-2569-third-family"),
    ("refusal_validation", DATA / "wt-2569-refusal-validation-20260909"),
    (
        "answer_write_and_residual",
        Path("/home/thomasjiralerspong/eps-runs/wt-2569-answer-residual-sae"),
    ),
]
for group, root in theory_trees:
    paths = [Path("eval_results/issue_2569"), Path("figures/issue_2569")]
    paths += sorted(
        Path("scripts").joinpath(p.name) for p in (root / "scripts").glob("issue2569*.py")
    )
    enumerate_group(group, root, paths, root)

enumerate_group(
    "methodology",
    DATA / "wt-2569-category-validation",
    [Path("docs/methodology/issue_2569.md")],
    DATA / "wt-2569-category-validation",
)

visual = Path("/home/thomasjiralerspong/.codex/visualizations/2026/09/04") / (
    "01a06e1f-03bc-7a31-a102-15d2b187a372"
)
enumerate_group("dashboards", visual, [Path(".")])
enumerate_group("codex_interpretations", DATA / "sae_interpretation_packets/codex", [Path(".")])
china = Path("/tmp/eps-952-china-repair-20260908")
china_paths = [
    Path("eval_results/issue_952/china_repair_v2"),
    Path("docs/experiments/issue952_china_repair_v2.md"),
    Path("docs/experiments/issue952_china_repair_v2_continuation.md"),
]
china_paths += sorted(
    Path("scripts").joinpath(p.name) for p in (china / "scripts").glob("issue952_china*.py")
)
enumerate_group("china_repair_receipts_and_code", china, china_paths, china)

# These are source/result inventories, not snapshots of model weights or raw corpora.
remote_rows = []
prefixes = [
    "issue2569_theory",
    "issue952_position_divergence/followups/china_refusal_wording_withholding_v2",
]
for prefix in prefixes:
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REV}/{prefix}?recursive=true&limit=1000"
    pages = 0
    count = 0
    while url:
        entries, link = get_json(url)
        assert isinstance(entries, list) and entries, (prefix, pages)
        for entry in entries:
            if entry["type"] != "file":
                continue
            item = {
                "prefix": prefix,
                "path": entry["path"],
                "bytes": entry["size"],
                "revision": HF_REV,
                "oid": entry.get("oid", ""),
                "lfs_sha256": (entry.get("lfs") or {}).get("oid", ""),
                "url": f"https://huggingface.co/datasets/{HF_REPO}/blob/{HF_REV}/{quote(entry['path'])}",
                "download_url": f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/{quote(entry['path'])}",
            }
            remote_rows.append(item)
            count += 1
        pages += 1
        assert pages <= 100, "Unexpectedly large metadata inventory; inspect scope"
        match = re.search(r'<([^>]+)>; rel="next"', link)
        url = match.group(1) if match else ""
        print(f"[remote] {prefix}: page {pages}, {count} files", flush=True)
    assert count, prefix

# Fetch only the compact numeric cross-family result, never activation binaries.
third_path = "issue2569_theory/third_family/results/third_family_summary.json"
third_url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/{third_path}"
third, _ = get_json(third_url)
(OUT / "third_family_summary.json").write_text(json.dumps(third, indent=2) + "\n")

for name, table in [("artifact_manifest.csv", rows), ("remote_artifact_manifest.csv", remote_rows)]:
    assert table
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

summary = {
    "created_utc": datetime.now(UTC).isoformat(),
    "local_source_locations": len(rows),
    "unique_bundled_files": len(stored),
    "unique_bundled_bytes": sum((OUT / name).stat().st_size for name in stored.values()),
    "large_local_files_indexed_only": sum(not row["bundled_path"] for row in rows),
    "remote_files_indexed": len(remote_rows),
    "remote_revision": HF_REV,
    "remote_prefixes": prefixes,
    "per_file_bundle_limit_bytes": MAX_COPY_BYTES,
    "scope": (
        "Named theory worktrees, their issue2569 results/figures/producers, dashboard "
        "and Codex packets, China repair receipts/code, and both scoped HF prefixes."
    ),
    "exclusions": (
        "No unrelated project experiments, VM logs, model inference, fitting, judging, "
        "private notes, or automatic message sending. Remote tensor content is not "
        "downloaded. Historical and superseded files remain labeled by source paths."
    ),
}
(OUT / "bundle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "source_coverage_notes.json").write_text(json.dumps(source_exclusions, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
