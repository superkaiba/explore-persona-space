"""Redact PII from gist bodies before publication.

Used by `daily-update`, `weekly-update`, `weekly-workflow-optimization`,
and `weekly-refactor-consolidation` skills to scrub a markdown body
before publishing as a public gist via `gh gist create --public`.

Patterns redacted (extensible without asking — see plan §10):
- Pod hostnames matching `epm-issue-\\d+` -> `<pod-N>` (preserves issue number)
- IPs from `scripts/pods.conf` (exact-match against the live registry)
- gmail addresses -> `<email>`
- RunPod team IDs `cm[a-z0-9]{20,}` -> `<team-id>`
- HF tokens `hf_[A-Za-z0-9]{30,}` -> `<hf-token>`
- Anthropic keys `sk-ant-[A-Za-z0-9_-]{40,}` -> `<anthropic-key>`
- OpenAI keys `sk-[A-Za-z0-9]{40,}` -> `<openai-key>`
- Generic env-leak `[A-Z]{2,}_(TOKEN|KEY|SECRET)=\\S+` -> `<NAME>=<redacted>`
- RunPod GraphQL URL `api.runpod.io/graphql` -> `<api-url>`

Usage:
    uv run python scripts/redact_for_gist.py --in body.md --out body.redacted.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

POD_REGISTRY = Path(__file__).parent / "pods.conf"

# Patterns are applied in order; later patterns see already-redacted text.
# Each entry is (compiled regex, replacement string). The order matters:
# more-specific token patterns (sk-ant-..., hf_...) come BEFORE the
# generic `[A-Z]{2,}_(TOKEN|KEY|SECRET)=...` env-leak rule.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Pod hostnames; preserve issue number for context.
    (re.compile(r"\bepm-issue-(\d+)\b"), r"<pod-\1>"),
    # Gmail addresses.
    (re.compile(r"[\w.+-]+@gmail\.com"), "<email>"),
    # API tokens — order matters: longer/more-specific first.
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}"), "<anthropic-key>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "<hf-token>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{40,}\b"), "<openai-key>"),
    # RunPod team IDs (cm prefix, 20+ alphanumeric chars).
    (re.compile(r"\bcm[a-z0-9]{20,}\b"), "<team-id>"),
    # Generic env-leak: NAME_TOKEN=..., NAME_KEY=..., NAME_SECRET=...
    (re.compile(r"\b([A-Z][A-Z0-9_]*_(?:TOKEN|KEY|SECRET))=\S+"), r"\1=<redacted>"),
    # RunPod GraphQL URL.
    (re.compile(r"https?://api\.runpod\.io/graphql\S*"), "<api-url>"),
]


def _ip_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Build IP-redaction patterns from `scripts/pods.conf`.

    Each non-comment, non-blank line of the registry has the form:
        name  host  port  gpus  gpu_type  label
    We extract the `host` (column 2) when it looks like an IPv4 literal.
    """
    if not POD_REGISTRY.exists():
        return []
    pats: list[tuple[re.Pattern[str], str]] = []
    for line in POD_REGISTRY.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2 and re.match(r"\d+\.\d+\.\d+\.\d+$", parts[1]):
            pats.append((re.compile(r"\b" + re.escape(parts[1]) + r"\b"), "<pod-ip>"))
    return pats


def redact(text: str) -> str:
    """Apply all redaction patterns in order; return the scrubbed text."""
    for rx, repl in PATTERNS:
        text = rx.sub(repl, text)
    for rx, repl in _ip_patterns():
        text = rx.sub(repl, text)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--in", dest="infile", required=True, help="path to input markdown body")
    p.add_argument("--out", dest="outfile", required=True, help="path to write the redacted body")
    args = p.parse_args()

    src = Path(args.infile).read_text()
    Path(args.outfile).write_text(redact(src))


if __name__ == "__main__":
    main()
