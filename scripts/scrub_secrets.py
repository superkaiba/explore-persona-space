#!/usr/bin/env python
"""Scan or scrub real-secret-grade strings in files, directories, and
``__packed__`` tar shards.

This is the remediation half of the secret upload gate
(``orchestrate/secret_scrub.py`` — read its module docstring for the
incident history and the pattern policy). The gate refuses to upload a
file containing a real-secret-grade string; this tool fixes the file with
SAME-LENGTH ``X`` placeholders (the 2026-08-16 redaction precedent), which
keeps ``__packed__`` index byte offsets valid — only content hashes
change, so any recorded per-member/shard sha256 (index.json, manifest.json)
must be recomputed after fixing a packed shard.

Usage:
    uv run python scripts/scrub_secrets.py scan PATH [PATH ...]
    uv run python scripts/scrub_secrets.py fix  PATH [PATH ...]

``scan`` exits 1 when findings exist (usable as a check); ``fix`` patches
in place, verifies the result re-scans clean, and exits 0.
Values are never printed — reports show masked first-6/last-4 only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explore_persona_space.orchestrate.secret_scrub import (  # noqa: E402
    BINARY_EXTENSIONS,
    scan_file,
    scrub_file,
)


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend(sorted(q for q in pp.rglob("*") if q.is_file()))
        elif pp.is_file():
            out.append(pp)
        else:
            print(f"warning: {p} not found", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["scan", "fix"])
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    total = 0
    for f in _files(args.paths):
        if f.suffix in BINARY_EXTENSIONS:
            continue
        findings = scrub_file(f) if args.mode == "fix" else scan_file(f)
        total += len(findings)
        for x in findings:
            verb = "fixed" if args.mode == "fix" else "found"
            print(f"{verb}: {x.pattern:18s} {x.masked:24s} {x.where()}")
    print(f"{args.mode}: {total} finding(s)")
    if args.mode == "scan" and total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
