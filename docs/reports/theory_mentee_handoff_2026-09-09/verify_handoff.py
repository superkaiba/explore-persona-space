"""Check handoff bytes and report links; never run scientific producers."""

import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

package = Path(sys.argv[1]).resolve()
assert (package / "bundle_summary.json").is_file(), package


def read_csv(name):
    """Read a generated inventory without touching source experiment files."""
    with (package / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    """Write a generated packaging inventory, not experimental results."""
    with (package / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


local = read_csv("artifact_manifest.csv")
remote = read_csv("remote_artifact_manifest.csv")
remote_hashes = {row["lfs_sha256"]: row for row in remote if row["lfs_sha256"]}
verified = {}
large_verified = 0
for row in local:
    if row["bundled_path"]:
        relative = Path(row["bundled_path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = package / relative
        if row["bundled_path"] not in verified:
            assert path.stat().st_size == int(row["bytes"]), path
            with path.open("rb") as handle:
                verified[row["bundled_path"]] = hashlib.file_digest(handle, "sha256").hexdigest()
        assert verified[row["bundled_path"]] == row["sha256"], path
    else:
        path = Path(row["source_path"])
        before = path.stat()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = path.stat()
        assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
        assert after.st_size == int(row["bytes"])
        assert digest in remote_hashes, f"No exact remote match for indexed-only file: {path}"
        assert int(remote_hashes[digest]["bytes"]) == after.st_size
        row["sha256"] = digest
        row["url"] = remote_hashes[digest]["url"]
        large_verified += 1
write_csv("artifact_manifest.csv", local)
assert len({row["path"] for row in remote}) == len(remote), "Duplicate remote inventory paths"
print(
    f"Verified {len(verified)} bundled files and {large_verified} large remote matches.", flush=True
)


def check_link(url):
    """Check public link availability and preserve errors as explicit records."""
    # GitHub's raw endpoint verifies exact blob content without page-rendering limits.
    checked = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
    if "/blob/" in url and url.startswith("https://github.com/"):
        checked = checked.replace("/blob/", "/", 1)
    else:
        checked = url
    request = Request(checked, method="HEAD", headers={"User-Agent": "EPS-theory-handoff/1"})
    try:
        with urlopen(request, timeout=25) as response:
            result = {
                "url": url,
                "checked_url": checked,
                "status": response.status,
                "resolved_url": response.url,
                "error": "",
            }
    except HTTPError as error:
        result = {
            "url": url,
            "checked_url": checked,
            "status": error.code,
            "resolved_url": error.url,
            "error": str(error),
        }
    except (URLError, TimeoutError) as error:
        result = {
            "url": url,
            "checked_url": checked,
            "status": "network_error",
            "resolved_url": "",
            "error": str(error),
        }
    print(f"Link {result['status']}: {url}", flush=True)
    return result


urls = sorted(set(re.findall(r"https://[^\s)]+", (package / "report.md").read_text())))
with ThreadPoolExecutor(max_workers=4) as executor:
    links = list(executor.map(check_link, urls))
write_csv("report_link_checks.csv", links)
result = {
    "verified_utc": datetime.now(UTC).isoformat(),
    "bundled_sha256_checks_passed": len(verified),
    "large_local_files_matched_to_remote_lfs_sha256": large_verified,
    "unique_remote_files_indexed": len(remote),
    "report_links_checked": len(links),
    "non_200_report_links": [row for row in links if row["status"] != 200],
    "note": "HEAD checks establish accessibility, not validity or recipient permissions.",
}
(package / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
