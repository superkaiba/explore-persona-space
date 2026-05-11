#!/usr/bin/env python
"""Pre-commit hook: refuse any staged file containing a secret-shaped string.

Background: GitHub's secret scanner pattern-matches on file content without
knowing context. When the base model under evaluation is prompted with
"what's in this config file?" or "send a webhook to Slack", it sometimes
generates plausible-looking tokens (`sk-...`, `hooks.slack.com/services/...`).
Those generations are serialized verbatim into `eval_results/*.json` and
trip the scanner on push (see commit 8c2523dc).

This hook scans staged text files for the same patterns the scanner uses
and blocks the commit. The intended remediation for false positives
(model-hallucinated tokens in eval output) is to run the redaction script
in-place:

    uv run python scripts/redact_for_gist.py --in-place <file>

For genuine leaks, rotate the credential and remove it from the file.

Patterns covered:
- Anthropic API keys: `sk-ant-[A-Za-z0-9_-]{40,}`
- OpenAI API keys (legacy + project): `sk-[A-Za-z0-9_-]{40,}`
- HuggingFace tokens: `hf_[A-Za-z0-9]{30,}`
- Slack incoming webhooks: `hooks.slack.com/services/T.../B.../...`
- GitHub PAT/OAuth tokens: `gh[pousr]_[A-Za-z0-9]{36}`

Allowlist (files that legitimately contain pattern matches for testing or
self-documentation):
- scripts/redact_for_gist.py (defines the patterns)
- scripts/check_no_secret_shaped_strings.py (this file)
- tests/fixtures/pii_redaction_input.txt (redaction test fixture)
- tests/test_redact_for_gist.py (redaction tests)
- tests/test_check_no_secret_shaped_strings.py (this script's tests)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Each entry: (name, compiled regex). Name appears in failure output so the
# user can tell whether the hit is an OpenAI key vs Slack webhook etc.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}")),
    # OpenAI must come after anthropic so the `sk-ant-` prefix is consumed first;
    # otherwise the `\bsk-` pattern would match the same string twice.
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_-]{40,}\b")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    (
        "slack-webhook",
        re.compile(r"https?://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]+"),
    ),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
]

# Files that intentionally contain pattern-matching strings — must not block.
# Paths are relative to repo root; match by exact relative path.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "scripts/redact_for_gist.py",
        "scripts/check_no_secret_shaped_strings.py",
        "tests/fixtures/pii_redaction_input.txt",
        "tests/test_redact_for_gist.py",
        "tests/test_check_no_secret_shaped_strings.py",
    }
)

# Skip binary files by extension. The hook walks file content as text, so
# anything non-text would either error or produce garbage. pre-commit's
# `files:` regex could also exclude these, but a defensive in-script check
# costs nothing.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".safetensors",
        ".pt",
        ".bin",
        ".parquet",
        ".pyc",
        ".so",
        ".dylib",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".zip",
        ".tar",
        ".gz",
    }
)


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_number, pattern_name, match_snippet) for each hit in `path`."""
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, rx in SECRET_PATTERNS:
            m = rx.search(line)
            if m is not None:
                snippet = m.group(0)
                # Truncate long matches in the report so the message stays readable.
                if len(snippet) > 60:
                    snippet = snippet[:30] + "..." + snippet[-15:]
                hits.append((lineno, name, snippet))
                break  # one report per line is enough to drive remediation
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-commit hook: refuse secret-shaped strings in staged files."
    )
    parser.add_argument("files", nargs="*", type=Path, help="Files passed by pre-commit.")
    args = parser.parse_args(argv)

    violations: list[str] = []
    for raw_path in args.files:
        # Normalize to repo-relative posix path for allowlist match.
        try:
            rel = raw_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            rel = raw_path
        rel_str = rel.as_posix()
        if rel_str in ALLOWLIST:
            continue
        if raw_path.suffix.lower() in BINARY_EXTENSIONS:
            continue
        if not raw_path.is_file():
            continue
        for lineno, name, snippet in scan_file(raw_path):
            violations.append(f"  {rel_str}:{lineno}: {name} matched: {snippet}")

    if violations:
        print("ERROR: secret-shaped strings detected in staged files:", file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        print(file=sys.stderr)
        print(
            "If these are MODEL-HALLUCINATED tokens in eval output (the usual case for\n"
            "eval_results/*.json), scrub the file with:\n"
            "    uv run python scripts/redact_for_gist.py --in-place <file>\n"
            "If they are REAL credentials, rotate the secret first, then remove the\n"
            "literal value from the file (and from git history if it was ever pushed).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
