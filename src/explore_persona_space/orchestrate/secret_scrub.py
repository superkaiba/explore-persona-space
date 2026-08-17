"""Upload-time secret gate + scrub tooling for Hub-bound artifacts.

Background (2026-08-17): HF secret-scanning emails fired FOUR times across
2026-06-15 → 2026-08-17 on the public data repo. Root cause each time:
public-corpus text (LMSYS / WildChat / CVE-fixes conversations in which
strangers pasted their own credentials) mirrored verbatim into
``raw_completions`` / screening pools / evidence shards and uploaded to a
PUBLIC dataset repo with no scrub step anywhere in the pipeline. The
#2332 repack then re-uploaded the same bytes inside ``__packed__`` tar
shards, re-triggering HF's scanner on content that had already been
triaged loose (7 "Lob" detections in
``issue2224_screening/__packed__/shard-c6-00001.tar``). A local sweep of
the two chunk-6 text shards found ~15 REAL third-party credentials HF
never flagged (OpenAI keys carrying the ``T3BlbkFJ`` marker, a GitHub
fine-grained PAT, an HF token, signed JWTs) — HF's scanner is not a
substitute for scanning before upload.

Two layers, deliberately asymmetric:

- **The gate** (:func:`assert_upload_clean`): wired into
  ``hub._upload`` and ``upload_sharded.upload_dir_sharded`` — every
  Hub-bound file is scanned BEFORE any commit is attempted, and a hit in
  a text file or tar member FAILS LOUD. The gate never mutates bytes:
  upload flows verify byte-exact sha256s after landing (#2332's whole
  machinery), so a silent scrub-at-upload would break remote-hash
  verification and, worse, hide the fact that a generator is emitting
  secrets. Kill switch: ``EPM_SECRET_UPLOAD_GATE=0`` (emergencies only —
  say so in the run log).
- **The scrub tool** (:func:`scrub_file` / ``scripts/scrub_secrets.py``):
  fixes flagged files IN PLACE with same-length ``X`` placeholders (the
  2026-08-16 redaction precedent, chosen by Thomas: redact at HEAD, repo
  stays public). Same-length matters: ``__packed__`` tar shards index
  members by byte offset+size, so a length-preserving patch keeps every
  offset valid and only per-member/shard hashes change. Run it on staged
  source BEFORE packing; on a packed ``.tar`` it patches member bytes at
  their absolute offsets without rewriting the archive.

Pattern policy: REAL-secret grade only. This set is narrower than
``.gitleaks.toml`` / ``scripts/check_no_secret_shaped_strings.py`` (which
err toward false positives for pre-commit, where a human immediately
adjudicates). The gate runs unattended inside long upload jobs, where a
false positive wedges a multi-hour run — so every pattern here either
carries a provider-verifiable marker (OpenAI's ``T3BlbkFJ``), a
high-specificity prefix (``github_pat_``, ``hf_``, ``rpa_``, ``AKIA``),
or a structural signature (signed three-part JWT), and obvious
placeholders (``XXXX…``, ``example``, ``YOUR_``) are filtered by
:data:`DUMMY_RX`. When adding a provider, add it to ``.gitleaks.toml``
too (its living-document header) — and keep this file in that config's
path allowlist, since it necessarily contains secret-shaped regexes.

False-positive lesson worth keeping: HF's "Lob" detector matches long
snake_case pytest/Java identifiers (``test_<35 chars>``) in security-code
corpora — 81 "Lob (status: active)" detections across two emails
contained ZERO Lob-format keys under any documented or reverse-engineered
pattern. Do not chase an HF detector name; scan the flagged bytes
yourself.
"""

from __future__ import annotations

import logging
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Read in chunks with overlap so multi-GB files never fully materialize.
# Overlap must exceed the longest plausible match (JWTs run ~1-2 KB).
_CHUNK_BYTES = 8 * 1024 * 1024
_OVERLAP_BYTES = 4096

# Extensions we can neither read as text nor patch meaningfully. Gate
# policy: SKIPPED by default (logged, one line per call) — the 2026-08-16
# ruling already accepts binary capture shards as un-redactable residuals,
# and scanning multi-GB checkpoints to produce a warn-only result would
# double upload wall-clock. ``EPM_SECRET_GATE_SCAN_BINARY=1`` scans them
# anyway (warn-only, never a gate failure).
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pt",
        ".bin",
        ".safetensors",
        ".npz",
        ".npy",
        ".pkl",
        ".parquet",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".pyc",
        ".so",
        ".gz",
        ".zip",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
    }
)

# (name, pattern) — bytes-level, applied to raw file / tar-member content.
SECRET_PATTERNS: list[tuple[str, re.Pattern[bytes]]] = [
    # OpenAI keys all embed base64("OpenAI") — the one marker that makes an
    # sk- string provably-real rather than model-hallucinated.
    ("openai-real", re.compile(rb"\bsk-[A-Za-z0-9_-]*?T3BlbkFJ[A-Za-z0-9_-]{15,}\b")),
    ("anthropic-key", re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{40,}\b")),
    ("openrouter-key", re.compile(rb"\bsk-or(?:-v\d+)?-[a-f0-9]{64}\b")),
    ("hf-token", re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b")),
    ("github-pat-fine", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("github-token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("jwt-signed", re.compile(rb"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(rb"\bxox[bp]-[0-9]{8,}-[0-9]{8,}-[A-Za-z0-9-]{10,}\b")),
    ("slack-webhook-real", re.compile(rb"https?://hooks\.slack\.com/services/T[A-Z0-9]{7,}/B[A-Z0-9]{7,}/[A-Za-z0-9]{20,}")),
    ("telegram-bot", re.compile(rb"\b\d{8,10}:AA[A-Za-z0-9_-]{33}\b")),
    ("aws-access-key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("stripe-live", re.compile(rb"\b[sr]k_live_[0-9a-zA-Z]{24,}\b")),
    ("infura-url-key", re.compile(rb"infura\.io/v3/[a-f0-9]{32}\b")),
    ("alchemy-url-key", re.compile(rb"g\.alchemy\.com/v2/[A-Za-z0-9_-]{20,}\b")),
    ("runpod-key", re.compile(rb"\brpa_[a-zA-Z0-9]{32,}\b")),
    ("wandb-key-context", re.compile(rb"(?i)wandb[^\n]{0,32}?\b[a-f0-9]{40}\b")),
    ("tinker-key", re.compile(rb"\btml-[a-zA-Z0-9_-]{60,}\b")),
]

# Placeholder / documentation-example filter, applied to the match plus ±30
# bytes of context. A string that is visibly fake must not wedge an upload.
DUMMY_RX = re.compile(
    rb"(?i)X{6,}|\.{3}|<[a-z_ ]+>|\$\{|YOUR[_-]|xxxx|1234567890|abcdef123"
    rb"|example|placeholder|redacted|_test_|dummy|fake|localhost|0{8}"
)


@dataclass
class Finding:
    """One secret-shaped match. ``member`` is empty for plain files."""

    path: str
    member: str
    offset: int  # within the member for tars, within the file otherwise
    length: int
    pattern: str
    masked: str

    def where(self) -> str:
        loc = f"{self.path}::{self.member}" if self.member else self.path
        return f"{loc} @{self.offset}"


class SecretUploadGateError(RuntimeError):
    """Raised by :func:`assert_upload_clean` when Hub-bound text contains
    a real-secret-grade string. Remediation is printed in the message —
    the gate itself never mutates bytes."""

    def __init__(self, what: str, findings: list[Finding]):
        self.findings = findings
        lines = [
            f"secret upload gate: {len(findings)} real-secret-grade string(s) in {what}; "
            "REFUSING to upload.",
        ]
        lines += [f"  {f.pattern:18s} {f.masked:24s} {f.where()}" for f in findings[:20]]
        if len(findings) > 20:
            lines.append(f"  … and {len(findings) - 20} more")
        lines.append(
            "fix: uv run python scripts/scrub_secrets.py fix <path>  "
            "(same-length placeholders; re-run your pack/hash step after). "
            "Emergency bypass: EPM_SECRET_UPLOAD_GATE=0 (say so in the run log)."
        )
        super().__init__("\n".join(lines))


def _mask(b: bytes) -> str:
    s = b.decode("utf-8", "replace")
    return s[:6] + "…" + s[-4:] if len(s) > 14 else s[:3] + "…"


def scan_bytes(data: bytes, *, path: str = "", member: str = "", base_offset: int = 0) -> list[Finding]:
    """All real-secret-grade matches in ``data`` (dummy-filtered)."""
    findings: list[Finding] = []
    seen: set[tuple[int, int]] = set()
    for name, rx in SECRET_PATTERNS:
        for mt in rx.finditer(data):
            span = (mt.start(), len(mt.group(0)))
            if span in seen:
                continue
            ctx = data[max(0, mt.start() - 30) : mt.end() + 30]
            if DUMMY_RX.search(mt.group(0)) or DUMMY_RX.search(ctx):
                continue
            seen.add(span)
            findings.append(
                Finding(
                    path=path,
                    member=member,
                    offset=base_offset + mt.start(),
                    length=len(mt.group(0)),
                    pattern=name,
                    masked=_mask(mt.group(0)),
                )
            )
    return findings


def _scan_stream(fh, *, path: str, member: str = "") -> list[Finding]:
    """Chunked scan so multi-GB inputs stay at O(chunk) memory. Findings in
    the overlap region are deduplicated by absolute offset."""
    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    pos = 0
    prev_tail = b""
    while True:
        chunk = fh.read(_CHUNK_BYTES)
        if not chunk:
            break
        buf = prev_tail + chunk
        base = pos - len(prev_tail)
        for f in scan_bytes(buf, path=path, member=member, base_offset=base):
            key = (f.pattern, f.offset)
            if key not in seen:
                seen.add(key)
                findings.append(f)
        prev_tail = buf[-_OVERLAP_BYTES:]
        pos += len(chunk)
    return findings


def scan_file(path: Path) -> list[Finding]:
    """Scan one file. ``.tar`` archives are scanned member-wise (binary
    members skipped by extension); other files are scanned whole."""
    path = Path(path)
    if path.suffix == ".tar":
        findings: list[Finding] = []
        with tarfile.open(path, "r") as tf:
            for m in tf:
                if not m.isfile() or Path(m.name).suffix in BINARY_EXTENSIONS:
                    continue
                fh = tf.extractfile(m)
                if fh is not None:
                    findings.extend(_scan_stream(fh, path=str(path), member=m.name))
        return findings
    with open(path, "rb") as fh:
        return _scan_stream(fh, path=str(path))


def scrub_bytes(data: bytes) -> tuple[bytes, list[Finding]]:
    """Same-length ``X`` fill for every finding. Alphanumeric fill keeps
    JSON/JSONL structurally valid (no quotes or escapes introduced)."""
    findings = scan_bytes(data)
    if not findings:
        return data, []
    buf = bytearray(data)
    for f in findings:
        buf[f.offset : f.offset + f.length] = b"X" * f.length
    return bytes(buf), findings


def scrub_file(path: Path, *, dry_run: bool = False) -> list[Finding]:
    """In-place same-length scrub. For ``.tar``: member bytes are patched at
    their absolute archive offsets — sizes, member order, and every index
    offset stay valid; only content hashes change. Returns the findings
    (empty means the file was already clean)."""
    path = Path(path)
    findings = scan_file(path)
    if not findings or dry_run:
        return findings
    if path.suffix == ".tar":
        # Map member name -> data start offset, then patch the archive.
        offsets: dict[str, int] = {}
        with tarfile.open(path, "r") as tf:
            for m in tf:
                if m.isfile():
                    offsets[m.name] = m.offset_data
        with open(path, "r+b") as fh:
            for f in findings:
                fh.seek(offsets[f.member] + f.offset)
                fh.write(b"X" * f.length)
    else:
        data = path.read_bytes()
        buf = bytearray(data)
        for f in findings:
            buf[f.offset : f.offset + f.length] = b"X" * f.length
        path.write_bytes(bytes(buf))
    residual = scan_file(path)
    if residual:
        raise RuntimeError(f"scrub_file left {len(residual)} finding(s) in {path} — refusing to report clean")
    return findings


def gate_enabled() -> bool:
    return os.environ.get("EPM_SECRET_UPLOAD_GATE", "1") != "0"


def _iter_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.is_file()))
        elif p.is_file():
            out.append(p)
    return out


def assert_upload_clean(paths: list[Path] | list[str], *, what: str) -> None:
    """The upload gate. Scans every text file / tar member that is about to
    leave for the Hub; raises :class:`SecretUploadGateError` on any
    real-secret-grade hit. Binary-extension files are skipped (logged)
    unless ``EPM_SECRET_GATE_SCAN_BINARY=1``, in which case they are
    scanned WARN-ONLY (they cannot be scrubbed, and the 2026-08-16 ruling
    accepts them as residuals)."""
    if not gate_enabled():
        logger.warning("secret upload gate DISABLED via EPM_SECRET_UPLOAD_GATE=0 for %s", what)
        return
    scan_binary = os.environ.get("EPM_SECRET_GATE_SCAN_BINARY", "0") == "1"
    files = _iter_files([Path(p) for p in paths])
    findings: list[Finding] = []
    n_binary_skipped = 0
    for f in files:
        if f.suffix in BINARY_EXTENSIONS:
            if scan_binary:
                hits = scan_file(f)
                for h in hits:
                    logger.warning("secret gate (binary, warn-only): %s %s %s", h.pattern, h.masked, h.where())
            else:
                n_binary_skipped += 1
            continue
        findings.extend(scan_file(f))
    if n_binary_skipped:
        logger.info(
            "secret gate: skipped %d binary file(s) for %s (EPM_SECRET_GATE_SCAN_BINARY=1 to scan)",
            n_binary_skipped,
            what,
        )
    if findings:
        raise SecretUploadGateError(what, findings)
