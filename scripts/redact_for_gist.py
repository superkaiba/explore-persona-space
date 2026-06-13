"""Redact PII from gist bodies before publication.

Used by `daily-update`, `weekly-update`, `weekly-workflow-optimization`,
and `weekly-refactor-consolidation` skills to scrub a markdown body
before publishing as a public gist via `gh gist create --public`.

Patterns redacted (extensible without asking — see plan §10):
- Pod hostnames matching `pod-\\d+` (canonical) or `epm-issue-\\d+` (legacy)
  -> `<pod-N>` (preserves issue number; already-redacted `<pod-N>`
  placeholders pass through unchanged, so redaction is idempotent)
- IPs from `scripts/pods.conf` (exact-match against the live registry)
  -> `<pod-ip>`
- Any other PUBLIC (globally routable) IPv4 literal -> `<ip>`. The registry
  is live and mutable — pods are ephemeral, so a pod IP in a body published
  after the pod is reaped from `pods.conf` would otherwise survive redaction.
  Private / loopback / reserved IPv4 (127.x, 10.x, 192.168.x, ...) are kept
  for readability; they carry no leak risk.
- gmail addresses -> `<email>`
- RunPod team IDs `cm[a-z0-9]{20,}` -> `<team-id>`
- HF tokens `hf_[A-Za-z0-9]{30,}` -> `<hf-token>`
- Anthropic keys `sk-ant-[A-Za-z0-9_-]{40,}` -> `<anthropic-key>`
- OpenAI keys `sk-[A-Za-z0-9]{40,}` -> `<openai-key>`
- Slack incoming webhooks `hooks.slack.com/services/T.../B.../...`
  -> `<slack-webhook>`
- Generic env-leak `[A-Z]{2,}_(TOKEN|KEY|SECRET)=\\S+` -> `<NAME>=<redacted>`
- RunPod GraphQL URL `api.runpod.io/graphql` -> `<api-url>`

Usage:
    uv run python scripts/redact_for_gist.py --in body.md --out body.redacted.md
    uv run python scripts/redact_for_gist.py --in-place eval_results/.../foo.json
"""

from __future__ import annotations

import argparse
import ipaddress
import re
from pathlib import Path

POD_REGISTRY = Path(__file__).parent / "pods.conf"

# Patterns are applied in order; later patterns see already-redacted text.
# Each entry is (compiled regex, replacement string). The order matters:
# more-specific token patterns (sk-ant-..., hf_...) come BEFORE the
# generic `[A-Z]{2,}_(TOKEN|KEY|SECRET)=...` env-leak rule.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Pod hostnames; preserve issue number for context. Accepts both the
    # canonical `pod-<N>` and legacy `epm-issue-<N>` prefixes. The optional
    # surrounding angle brackets are consumed so an already-redacted
    # `<pod-N>` placeholder rewrites to itself (idempotency: without this,
    # re-redacting wrapped `<pod-137>` into `<<pod-137>>`).
    (re.compile(r"<?\b(?:pod|epm-issue)-(\d+)\b>?"), r"<pod-\1>"),
    # Gmail addresses.
    (re.compile(r"[\w.+-]+@gmail\.com"), "<email>"),
    # API tokens — order matters: longer/more-specific first.
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}"), "<anthropic-key>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "<hf-token>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{40,}\b"), "<openai-key>"),
    # Slack incoming webhooks. The TXXXX/BXXXX/secret parts can include
    # zero-padded placeholders (T00000000), so accept any alphanumeric.
    (
        re.compile(r"https?://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]+"),
        "<slack-webhook>",
    ),
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


# Any 4-octet dotted literal; candidates are validated with `ipaddress`
# before redaction (rejects e.g. version strings with octets > 255).
_GENERIC_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _redact_public_ipv4(text: str) -> str:
    """Redact every PUBLIC (globally routable) IPv4 literal to `<ip>`.

    Backstop for the registry exact-match in `_ip_patterns()`: pods are
    ephemeral, so by the time a body is published its pod's IP has often
    already left `pods.conf` and the exact-match misses it. When in doubt
    redact more, never less — any globally routable IPv4 in a gist body is
    presumed to be a live connection endpoint. Private / loopback /
    reserved addresses (`ipaddress.is_global == False`) are kept: they are
    non-routable, carry no leak risk, and keeping them preserves
    readability (e.g. `127.0.0.1:3010` dashboard references).
    """

    def _sub(m: re.Match[str]) -> str:
        try:
            ip = ipaddress.IPv4Address(m.group(0))
        except ipaddress.AddressValueError:
            return m.group(0)
        return "<ip>" if ip.is_global else m.group(0)

    return _GENERIC_IPV4.sub(_sub, text)


def redact(text: str) -> str:
    """Apply all redaction patterns in order; return the scrubbed text."""
    for rx, repl in PATTERNS:
        text = rx.sub(repl, text)
    # Registry exact-match first (more specific `<pod-ip>` placeholder),
    # then the public-IPv4 backstop for anything the live registry no
    # longer (or never) lists.
    for rx, repl in _ip_patterns():
        text = rx.sub(repl, text)
    return _redact_public_ipv4(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--in", dest="infile", help="path to input markdown body")
    p.add_argument("--out", dest="outfile", help="path to write the redacted body")
    p.add_argument(
        "--in-place",
        dest="inplace",
        nargs="+",
        help="redact one or more files in place (rewrites them with secrets replaced)",
    )
    args = p.parse_args()

    if args.inplace:
        for path_str in args.inplace:
            path = Path(path_str)
            original = path.read_text()
            redacted = redact(original)
            if redacted != original:
                path.write_text(redacted)
                print(f"scrubbed: {path}")
            else:
                print(f"unchanged: {path}")
        return

    if not args.infile or not args.outfile:
        p.error("--in and --out are required when not using --in-place")

    src = Path(args.infile).read_text()
    Path(args.outfile).write_text(redact(src))


if __name__ == "__main__":
    main()
