#!/usr/bin/env python3
"""Served-vs-committed integrity probe for the eps.superkaiba.com dashboards.

Verifies that every git-tracked static artifact under ``dashboard/public/``
(.html + .json) is served byte-identically to the on-disk copy, and that the
served bytes carry ZERO Cloudflare Email Address Obfuscation markers
(``__cf_email__`` / ``cdn-cgi/l/email-protection``). Task #2365: Cloudflare's
Email Address Obfuscation silently rewrote email-like strings INSIDE text
presented as model generations (served bytes != committed bytes), which both
corrupts evidence and breaks sha-based integrity checks. The origin-side fix is
``Cache-Control: ... no-transform`` (dashboard/next.config.ts headers());
this probe is the standing mechanical check that the fix holds.

Usage:
    uv run python scripts/verify_served_dashboards.py                # live CDN
    uv run python scripts/verify_served_dashboards.py --base-url http://127.0.0.1:3010

Run it as the LAST step of every dashboard deploy (see dashboard/README.md).

Exit codes:
    0 — every file byte-identical AND zero obfuscation markers.
    1 — any sha divergence or any nonzero marker count.
    (A fetch failure raises — loud, never swallowed.)

A file that is on-disk-dirty vs the committed copy at HEAD (``git diff HEAD``
non-empty — unstaged, staged-but-uncommitted, and staged-NEW changes alike) is
reported as a WARN, not a failure: the probe compares served vs ON-DISK bytes,
and the warn makes a dirty working copy visible in the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://eps.superkaiba.com"
PUBLIC_DIR_REL = "dashboard/public"
SUFFIXES = (".html", ".json")
# Obfuscation fingerprints counted in the SERVED bytes. `__cf_email__` is the
# class Cloudflare injects on each rewritten address; `cdn-cgi/l/email-protection`
# is the decoder endpoint its <a href>/<script> tags point at.
MARKERS = (b"__cf_email__", b"cdn-cgi/l/email-protection")
# Second observed edge transform (task #2365 negative control, 2026-08-27): the
# Cloudflare Web Analytics RUM beacon (`beacon.min.js`) auto-injected into html
# responses (+~360 bytes; client-conditional — urllib got it, curl did not).
# Counted for ATTRIBUTION of a sha divergence; the FAIL condition is the sha
# divergence itself plus any nonzero MARKERS count. `no-transform` suppresses
# this injection too (Cloudflare Web Analytics FAQ).
BEACON_MARKER = b"static.cloudflareinsights.com/beacon"
FETCH_TIMEOUT_S = 60
# Cloudflare can 403 the default "Python-urllib/x.y" user agent; identify
# honestly but non-default.
USER_AGENT = "eps-verify-served-dashboards/1.0 (+scripts/verify_served_dashboards.py)"

REPO_ROOT = Path(__file__).resolve().parent.parent


def tracked_public_files(repo_root: Path) -> list[str]:
    """Git-tracked .html/.json paths under dashboard/public/, repo-relative.

    Uses ``git ls-files -z`` (NUL-delimited, no quoting): under the default
    ``core.quotePath=true`` a non-ASCII filename in newline mode is emitted
    quoted (``"...\\303\\251..."``), the trailing quote defeats the suffix
    filter, and the file is silently excluded from enumeration.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", PUBLIC_DIR_REL],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [name for name in out.split("\0") if name.endswith(SUFFIXES)]
    if not files:
        raise RuntimeError(
            f"git ls-files returned no {SUFFIXES} files under {PUBLIC_DIR_REL} "
            f"(repo root {repo_root}) — refusing to report a vacuous PASS"
        )
    return files


def dirty_vs_committed(repo_root: Path, relpath: str) -> bool:
    """True when the on-disk copy differs from the committed copy at HEAD (WARN only).

    ``git diff HEAD`` compares the WORKING TREE against HEAD, so unstaged
    edits, staged-but-uncommitted edits, and staged NEW files all report
    dirty. A bare ``git diff`` compares worktree vs INDEX and reads a
    staged-but-uncommitted change as committed-clean (task #2365 round-1
    concern ``staged-index-dirty-blind-spot``).
    """
    out = subprocess.run(
        ["git", "diff", "HEAD", "--name-only", "--", relpath],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return bool(out.strip())


def fetch(base_url: str, served_rel: str) -> bytes:
    """Fetch one artifact; any HTTP error / timeout raises (loud by design)."""
    url = f"{base_url.rstrip('/')}/{urllib.parse.quote(served_rel)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        if resp.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {resp.status}")
        return resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"base URL to fetch from (default: {DEFAULT_BASE_URL}; "
        "use http://127.0.0.1:3010 for an origin-only check)",
    )
    args = parser.parse_args()

    files = tracked_public_files(REPO_ROOT)
    n_diverged = 0
    n_marked = 0
    n_dirty = 0

    header = f"{'file':<58} {'sha match':<9} {'cf_email':>8} {'cdn-cgi':>8} {'beacon':>7}  note"
    print(f"Probing {len(files)} tracked {PUBLIC_DIR_REL} files against {args.base_url}")
    print(header)
    print("-" * len(header))

    for relpath in files:
        served_rel = relpath.removeprefix(PUBLIC_DIR_REL + "/")
        disk_bytes = (REPO_ROOT / relpath).read_bytes()
        served_bytes = fetch(args.base_url, served_rel)

        sha_disk = hashlib.sha256(disk_bytes).hexdigest()
        sha_served = hashlib.sha256(served_bytes).hexdigest()
        match = sha_disk == sha_served
        counts = [served_bytes.count(m) for m in MARKERS]
        beacon_count = served_bytes.count(BEACON_MARKER)

        notes = []
        if dirty_vs_committed(REPO_ROOT, relpath):
            n_dirty += 1
            notes.append("WARN: on-disk differs from committed")
        if not match:
            n_diverged += 1
            notes.append(f"DIVERGED disk={sha_disk[:12]} served={sha_served[:12]}")
        if any(counts):
            n_marked += 1
            notes.append("OBFUSCATION MARKERS IN SERVED BYTES")

        print(
            f"{served_rel:<58} {'OK' if match else 'FAIL':<9} "
            f"{counts[0]:>8} {counts[1]:>8} {beacon_count:>7}  {'; '.join(notes)}"
        )

    print("-" * len(header))
    verdict_fail = n_diverged > 0 or n_marked > 0
    print(
        f"{'FAIL' if verdict_fail else 'PASS'}: {len(files)} files, "
        f"{n_diverged} sha-diverged, {n_marked} with obfuscation markers, "
        f"{n_dirty} dirty-vs-committed (warn only)"
    )
    return 1 if verdict_fail else 0


if __name__ == "__main__":
    sys.exit(main())
