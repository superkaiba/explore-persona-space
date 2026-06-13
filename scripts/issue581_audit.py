"""Scan a task's events.jsonl for token-shaped strings.

Task #581 — defensive audit of the pre-fix6 SLURM-lane secrets-exposure path
(sbatch preflight ran token checks under ``set -x``; #535's fix6 scrubbed the
echo, but earlier event-marker bodies committed before fix6 landed may carry
leaked tokens).  Reports HF / WandB / RunPod / OpenAI / Anthropic shapes and
the literal ``<VAR>=<value>`` env-assignment shape per the plan.

Usage::

    uv run python scripts/issue581_audit.py --task 535 --out-dir tasks/<status>/581/artifacts/

Deterministic, stdlib-only, CPU-only.  Exits 0 with the verdict written to
``audit-report.md`` and machine-readable hits to ``audit-hits.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Patterns: each entry is (key class, compiled regex).  WandB is handled
# specially below: the 40-hex shape collides with every git SHA in the file,
# so we anchor wandb hits to lines that also mention WANDB context.
TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Anthropic FIRST so OpenAI's broader sk-* doesn't double-count.
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_-]{30,}")),
    ("openai", re.compile(r"sk-(?!ant-)[A-Za-z0-9_-]{30,}")),
    ("hf", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("runpod", re.compile(r"rpa_[A-Za-z0-9_]{30,}")),
]

# Env-assignment shape: catches a leak that doesn't match a key-shape regex
# (token format drift) AND lets us distinguish a successful scrub from a true
# leak.  Value must be non-empty and not a scrub sentinel.
ENV_ASSIGN = re.compile(
    r"\b(HF_TOKEN|HF_HUB_TOKEN|WANDB_API_KEY|RUNPOD_API_KEY|"
    r"ANTHROPIC_API_KEY|OPENAI_API_KEY)\s*=\s*([^\s'\"`]+)"
)
SCRUB_SENTINELS = {
    "***SCRUBBED***",
    "<redacted>",
    "<scrubbed>",
    "REDACTED",
    "***",
    "[scrubbed]",
}

# Test-fixture markers — values containing any of these substrings are obvious
# placeholder strings (commonly quoted in test asserts / code-review prose),
# not real secrets.  Used to classify env-assign hits as "low-confidence" so
# the FAIL verdict surfaces only real-leak candidates.
FIXTURE_MARKERS = (
    "test_token",
    "test_key",
    "_test_",
    "fake_",
    "dummy_",
    "example_",
    "placeholder",
    "your_token",
    "your_key",
    "xxx",
    "abcd1234",
)

# Real-secret minimum lengths (by key class) — anything shorter for an
# env-assign value is structurally too short to be a live credential.
MIN_REAL_LEN = {
    "HF_TOKEN": 30,
    "HF_HUB_TOKEN": 30,
    "WANDB_API_KEY": 30,
    "RUNPOD_API_KEY": 30,
    "OPENAI_API_KEY": 30,
    "ANTHROPIC_API_KEY": 30,
}

# WandB API keys: 40 hex chars (lowercase) — collides with git SHAs, so we
# only flag occurrences that share a line with explicit WandB context.
WANDB_HEX = re.compile(r"\b[a-fA-F0-9]{40}\b")
WANDB_CONTEXT = re.compile(r"WANDB_API_KEY|wandb[_ ]?login|wandb[_ ]?(key|token)", re.IGNORECASE)


@dataclass
class Hit:
    """One leaked-token candidate within a single events.jsonl row."""

    line_no: int  # 1-indexed
    event_ts: str
    event_kind: str
    event_version: int | None
    key_class: str  # 'hf' | 'wandb' | 'runpod' | 'openai' | 'anthropic' | 'env-assign:<VAR>'
    match: str  # the offending substring (truncated to 80 chars)
    note_excerpt: str  # first 200 chars of the row's `note` field (or full row if note absent)
    confidence: str = "high"  # 'high' = likely real leak; 'low' = test fixture / placeholder
    triage_reason: str = ""  # why the confidence was lowered (empty for high)
    byte_offset: int = -1  # match offset within the JSONL row (re.Match.start()); -1 = unknown


# Env-assignment key -> provider class.  Used to resolve env-assign:HF_TOKEN /
# env-assign:HF_HUB_TOKEN -> the canonical "hf" provider so the rotation
# instructions table renders provider-specific steps for env-assign hits.
ENV_ASSIGN_PROVIDER: dict[str, str] = {
    "HF_TOKEN": "hf",
    "HF_HUB_TOKEN": "hf",
    "WANDB_API_KEY": "wandb",
    "RUNPOD_API_KEY": "runpod",
    "OPENAI_API_KEY": "openai",
    "ANTHROPIC_API_KEY": "anthropic",
}


# Per-provider rotation steps.  Plan AC4 mandates "EXACT rotation steps Thomas
# needs to take per leaked key" on FAIL — a provider label is not a step.
ROTATION_STEPS: dict[str, list[str]] = {
    "hf": [
        "Visit https://huggingface.co/settings/tokens and revoke the leaked token.",
        "Create a new token (Read or Read+Write to match the leaked one's role).",
        "Update `.env`: `HF_TOKEN=<new>` (and `HF_HUB_TOKEN=<new>` if it's mirrored).",
        "Verify: `set -a && source .env && set +a && uv run python -c "
        '"from huggingface_hub import HfApi; print(HfApi().whoami())"`.',
    ],
    "wandb": [
        "Visit https://wandb.ai/authorize and reset the API key.",
        "Update `.env`: `WANDB_API_KEY=<new>`.",
        'Re-login on the VM: `uv run python -c "import wandb; wandb.login()"`.',
    ],
    "runpod": [
        "Visit https://www.runpod.io/console/user/settings → API Keys, revoke the leaked key.",
        "Create a new RunPod API key.",
        "Update `.env`: `RUNPOD_API_KEY=<new>`.",
        "If any pods are live, re-sync host/port: `uv run python scripts/pod.py "
        "config --refresh-from-api`.",
    ],
    "openai": [
        "Visit https://platform.openai.com/api-keys and revoke the leaked key.",
        "Create a new API key (Project key preferred; mirror the leaked key's role).",
        "Update `.env`: `OPENAI_API_KEY=<new>`.",
    ],
    "anthropic": [
        "Visit https://console.anthropic.com/settings/keys and revoke the leaked key.",
        "Create a new API key.",
        "Update `.env`: `ANTHROPIC_API_KEY=<new>`.",
    ],
}


def _provider_for_hit(h: Hit) -> str:
    """Return the canonical provider class for a hit's `key_class`.

    Direct provider classes (`hf` / `wandb` / `runpod` / `openai` / `anthropic`)
    map to themselves.  `env-assign:<VAR>` resolves via `ENV_ASSIGN_PROVIDER`
    (so `env-assign:HF_TOKEN` → `hf`), which lets the Required-action section
    render provider-specific rotation steps for env-assign hits exactly like
    shape-pattern hits.  Returns `""` for an unrecognised key class.
    """
    cls = h.key_class
    if cls in ROTATION_STEPS:
        return cls
    if cls.startswith("env-assign:"):
        var = cls.split(":", 1)[1]
        return ENV_ASSIGN_PROVIDER.get(var, "")
    return ""


def _classify_env_assign(var: str, value: str) -> tuple[str, str]:
    """Classify an env-assignment hit as ('high'|'low', reason).

    The env-assignment regex catches anything after ``=``, so it also matches
    obvious test fixtures quoted in code or code-review prose
    (``HF_TOKEN=hf_test_token``, ``WANDB_API_KEY=wandb_test_key``).  Those are
    structurally not real secrets:

    - value contains an explicit fixture marker (``test_token``, ``_test_``,
      ``fake_``, etc.) -> low confidence;
    - value is shorter than the minimum length for a real secret of its key
      class (30 chars for every supported provider) -> low confidence.

    Real leaks have long, opaque, marker-free values.
    """
    value_lower = value.lower()
    for marker in FIXTURE_MARKERS:
        if marker in value_lower:
            return ("low", f"value contains fixture marker '{marker}'")
    min_len = MIN_REAL_LEN.get(var, 0)
    if min_len and len(value) < min_len:
        return ("low", f"value length {len(value)} below minimum {min_len} for {var}")
    return ("high", "")


def _redact_value(value: str, key_class: str) -> str:
    """Return a fixed-shape redacted preview of a matched secret value.

    The audit's whole job is to identify secrets so the human can rotate
    them — but its own output (the markdown report + the audit-hits JSON)
    must not COPY the secret into a fresh artifact, or the audit re-leaks
    the credential into the task log it's protecting.  So every matched
    value is replaced with ``<redacted:<class>:len=<N>>`` (length is safe
    to share — a 37-char vs 8-char hint disambiguates real-shape from
    fixture without ever exposing the bytes).

    The bare provider class (`hf` / `wandb` / `runpod` / `openai` /
    `anthropic`) is used unchanged; env-assign hits expose the env-var
    name (which is itself public, e.g. ``HF_TOKEN``) but the trailing
    value is redacted.
    """
    if key_class.startswith("env-assign:"):
        var = key_class.split(":", 1)[1]
        # value already includes "VAR=..." here; rebuild VAR=<redacted:...>.
        return f"{var}=<redacted:env:len={len(value) - len(var) - 1}>"
    return f"<redacted:{key_class}:len={len(value)}>"


def _redact_excerpt(text: str) -> str:
    """Scrub token-shaped substrings from a note-excerpt string.

    A note-excerpt may quote the surrounding context of a leak — which can
    include the leaked value itself (the `set -x` echo pattern this audit
    targets).  Replace every TOKEN_PATTERN match + every ``VAR=value``
    env-assignment match with a class-tagged redaction marker.  The
    excerpt's other prose (event kinds, timestamps, file paths, fixture
    annotations) is preserved verbatim so the human reader can still
    triage the context.

    Defense in depth: the same regex patterns the scanner uses for hit
    detection are reused here for excerpt scrubbing, so any token shape
    that triggers a hit cannot also leak into the surrounding excerpt
    text.
    """

    # Order matters.  ENV_ASSIGN matches the WHOLE ``VAR=value`` span and
    # therefore consumes any in-value token shape (HF_TOKEN=hf_…); processing
    # it FIRST means standalone token shapes elsewhere in the excerpt still
    # reach their own redaction pass below, and the env-assign output's `env`
    # tag identifies that the leak rode through an env-var assignment.
    def _env_sub(m: re.Match[str]) -> str:
        var, value = m.group(1), m.group(2)
        if value in SCRUB_SENTINELS or value.lower() in {s.lower() for s in SCRUB_SENTINELS}:
            return m.group(0)  # already scrubbed; leave verbatim
        return f"{var}=<redacted:env:len={len(value)}>"

    text = ENV_ASSIGN.sub(_env_sub, text)

    # Provider-shape patterns next (anthropic before openai, mirroring
    # TOKEN_PATTERNS' detection order).
    for key_class, pat in TOKEN_PATTERNS:
        text = pat.sub(lambda m, c=key_class: f"<redacted:{c}:len={len(m.group(0))}>", text)

    # WandB 40-hex only when explicit WANDB context appears in the SAME excerpt.
    if WANDB_CONTEXT.search(text):
        text = WANDB_HEX.sub(lambda m: f"<redacted:wandb:len={len(m.group(0))}>", text)

    return text


def scan_line(line_no: int, raw_line: str) -> list[Hit]:
    """Apply all token patterns to one events.jsonl line.

    Scans the row's serialized JSON form so matches inside `note`, `metadata`,
    or any other field are caught uniformly.  Multiple key classes can hit
    the same row — each contributes one Hit.

    Stored fields are REDACTED at the moment of construction (``match`` →
    ``<redacted:<class>:len=<N>>``, ``note_excerpt`` → token shapes scrubbed).
    The raw secret never leaves this function alive.
    """
    hits: list[Hit] = []
    try:
        event = json.loads(raw_line)
    except json.JSONDecodeError:
        # A malformed row is itself a red flag — try the line as raw text.
        event = {}

    ts = event.get("ts", "?")
    kind = event.get("kind", "?")
    version = event.get("version")
    note = event.get("note") or ""
    raw_excerpt = note[:200] if note else raw_line[:200]
    safe_excerpt = _redact_excerpt(raw_excerpt)

    # Key-shape patterns.  Record `m.start()` so the report can pinpoint
    # multiple hits on the same JSONL row (plan AC3 byte offset).
    for key_class, pat in TOKEN_PATTERNS:
        for m in pat.finditer(raw_line):
            hits.append(
                Hit(
                    line_no=line_no,
                    event_ts=ts,
                    event_kind=kind,
                    event_version=version,
                    key_class=key_class,
                    match=_redact_value(m.group(0), key_class),
                    note_excerpt=safe_excerpt,
                    byte_offset=m.start(),
                )
            )

    # WandB 40-hex with context-line anchoring.  Scan each LINE of the raw row
    # (the JSONL row is a single line, but `note` may contain embedded \n
    # sequences as literal text — we treat the whole row as one "line" for
    # context purposes, which is conservative).
    if WANDB_CONTEXT.search(raw_line):
        for m in WANDB_HEX.finditer(raw_line):
            hits.append(
                Hit(
                    line_no=line_no,
                    event_ts=ts,
                    event_kind=kind,
                    event_version=version,
                    key_class="wandb",
                    match=_redact_value(m.group(0), "wandb"),
                    note_excerpt=safe_excerpt,
                    byte_offset=m.start(),
                )
            )

    # Env-assignment shape.
    for m in ENV_ASSIGN.finditer(raw_line):
        var, value = m.group(1), m.group(2)
        # Skip if the value is a scrub sentinel (any case-folded comparison).
        if value in SCRUB_SENTINELS or value.lower() in {s.lower() for s in SCRUB_SENTINELS}:
            continue
        confidence, reason = _classify_env_assign(var, value)
        hits.append(
            Hit(
                line_no=line_no,
                event_ts=ts,
                event_kind=kind,
                event_version=version,
                key_class=f"env-assign:{var}",
                match=_redact_value(f"{var}={value}", f"env-assign:{var}"),
                note_excerpt=safe_excerpt,
                confidence=confidence,
                triage_reason=reason,
                byte_offset=m.start(),
            )
        )

    return hits


def scan_file(events_path: Path) -> list[Hit]:
    """Scan every line of an events.jsonl file."""
    all_hits: list[Hit] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            all_hits.extend(scan_line(line_no, raw))
    return all_hits


def _render_hit(h: Hit) -> list[str]:
    """Render one hit as markdown lines."""
    triage = f" — **triage:** {h.triage_reason}" if h.triage_reason else ""
    location = f"line {h.line_no}"
    if h.byte_offset >= 0:
        location += f", byte offset {h.byte_offset}"
    return [
        f"### Hit — {location} · `{h.key_class}` (confidence: {h.confidence}{triage})",
        "",
        f"- **Event:** ts={h.event_ts}, kind={h.event_kind}, version={h.event_version}",
        f"- **Match:** `{h.match}`",
        f"- **Note excerpt:** {h.note_excerpt}",
        "",
    ]


def compose_report(task_id: int, events_path: Path, hits: list[Hit], n_events: int) -> str:
    """Compose a human-readable markdown report.

    Verdict semantics (plan AC4/AC6, strict binary):

    - ``PASS`` ONLY when there are zero hits across all patterns;
    - ``FAIL — leaked: <classes>`` when ANY hit fires, regardless of
      confidence tier.

    The confidence tier (``high`` / ``low``) survives as a sub-classification
    inside the FAIL report so the rotation triage is trivial — but it does
    NOT influence the top-line verdict.  A scan that finds only
    low-confidence (obvious-fixture) hits still surfaces as FAIL so the
    audit gate cannot silently promote a hit-bearing run as PASS (per
    plan §"No false positives in PASS": "rotation cost is bounded and a
    missed leak is unrecoverable").
    """
    high = [h for h in hits if h.confidence == "high"]
    low = [h for h in hits if h.confidence == "low"]
    classes_all = sorted({h.key_class.split(":", 1)[0] for h in hits})

    verdict = "PASS" if not hits else f"FAIL — leaked: {', '.join(classes_all)}"

    lines = [
        f"# Audit report — task #{task_id} events.jsonl token-shape scan",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"- **File scanned:** `{events_path}` ({n_events} events)",
        "- **Patterns:** hf, wandb (context-anchored), runpod, openai, anthropic, env-assignment",
        f"- **Total hits:** {len(hits)} (high-confidence: {len(high)}; low-confidence: {len(low)})",
        "",
    ]

    if not hits:
        lines += [
            "## No token-shaped strings found",
            "",
            "Every regex pattern returned zero matches against the events.jsonl rows. ",
            "No secret rotation required from this scan.",
            "",
            "## What was checked",
            "",
            "Each pattern below was applied to the serialized JSON form of every row, ",
            "so matches inside `note`, `metadata`, or any other field would have been caught:",
            "",
            "- HF tokens (`hf_[A-Za-z0-9]{30,}`)",
            (
                "- WandB API keys (40-hex, context-anchored to lines also matching "
                "`WANDB_API_KEY` / `wandb login` / `wandb_key` / `wandb_token`)"
            ),
            "- RunPod tokens (`rpa_[A-Za-z0-9_]{30,}`)",
            (
                "- OpenAI tokens (`sk-` followed by 30+ chars, excluding `sk-ant-` "
                "which is broken out)"
            ),
            "- Anthropic tokens (`sk-ant-[A-Za-z0-9_-]{30,}`)",
            (
                "- Env-assignment shape (`HF_TOKEN=` / `HF_HUB_TOKEN=` / `WANDB_API_KEY=` / "
                "`RUNPOD_API_KEY=` / `ANTHROPIC_API_KEY=` / `OPENAI_API_KEY=` followed by a "
                "non-empty value that is not a scrub sentinel)."
            ),
        ]
        return "\n".join(lines) + "\n"

    if high:
        lines += [
            "## High-confidence hits (real-leak candidates)",
            "",
            "Each row below is a token-shape candidate whose value is opaque, "
            "long enough to be a live credential, and lacks any obvious fixture "
            "marker. Treat as a real leak and rotate.",
            "",
        ]
        for h in high:
            lines += _render_hit(h)

        # Plan AC4: "the report MUST list the EXACT rotation steps Thomas
        # needs to take per leaked key."  Derive the provider set from the
        # HIGH hits via _provider_for_hit so env-assign:HF_TOKEN /
        # env-assign:WANDB_API_KEY etc. resolve to provider-specific steps
        # instead of falling through to a generic "env-assignment hit" label.
        providers_high: list[str] = sorted({p for h in high if (p := _provider_for_hit(h))})
        # Any high hit whose key_class doesn't resolve to a known provider
        # (unrecognised env-assign target, future shape class) still needs to
        # surface in the action list, so we collect them separately.
        unresolved_high = sorted({h.key_class for h in high if not _provider_for_hit(h)})

        lines += [
            "## Required action",
            "",
            "**Rotate each leaked secret immediately. The EXACT steps per "
            "provider are below — follow them in order, then re-run this "
            "audit against the new events.jsonl to confirm zero hits.**",
            "",
        ]
        provider_label = {
            "hf": "Hugging Face token (`HF_TOKEN` / `HF_HUB_TOKEN`)",
            "wandb": "WandB API key (`WANDB_API_KEY`)",
            "runpod": "RunPod API key (`RUNPOD_API_KEY`)",
            "openai": "OpenAI API key (`OPENAI_API_KEY`)",
            "anthropic": "Anthropic API key (`ANTHROPIC_API_KEY`)",
        }
        for provider in providers_high:
            lines.append(f"### Rotate the {provider_label[provider]}")
            lines.append("")
            for i, step in enumerate(ROTATION_STEPS[provider], start=1):
                lines.append(f"{i}. {step}")
            lines.append("")
        if unresolved_high:
            lines += [
                "### Unrecognised key class(es) — manual triage",
                "",
                "The following key classes hit `high` confidence but do not map "
                "to a known provider's rotation procedure. Inspect the matching "
                "rows above and rotate the corresponding credential manually:",
                "",
            ]
            for cls in unresolved_high:
                lines.append(f"- `{cls}`")
            lines.append("")
        lines += [
            "After rotating EVERY leaked secret above, re-run this audit:",
            "",
            "```bash",
            "uv run python scripts/issue581_audit.py "
            f"--task {task_id} --events-path <events.jsonl> --out-dir <dir>",
            "```",
            "",
            "The audit must report zero hits before the rotation is considered complete.",
            "",
        ]
    elif low:
        # Low-only FAIL: no high-confidence hits to rotate on, but the top-line
        # verdict is still FAIL per plan AC4/AC6. Document the triage clearly.
        lines += [
            "## Triage — FAIL on low-confidence hits only",
            "",
            "The scan found hits but ALL hits classify as low-confidence "
            "(fixture markers or values structurally too short to be a live "
            "credential). Per the plan's binary verdict rule the top-line is "
            "FAIL regardless, so a human reviews and acks the triage below — "
            "rotation is NOT required unless this section says otherwise.",
            "",
            "Walk each row in the section below, confirm the context is benign "
            "(typically code-review prose quoting test fixtures, documentation "
            "examples, or a `.env.example` snippet), and ack the audit. If "
            "ANY row turns out to be a real leak that was mis-classified, "
            "rotate the corresponding secret and re-run this audit.",
            "",
        ]

    if low:
        lines += [
            "## Low-confidence hits (likely false positives)",
            "",
            "These rows matched the env-assignment shape but the value is "
            "structurally not a real credential (contains an explicit fixture "
            "marker like `test_token` / `_test_` / `fake_`, or is shorter than "
            "the minimum length for a live secret of that key class). No "
            "rotation required; eyeball the note excerpt to confirm the context "
            "(typically code-review prose quoting test fixtures, documentation "
            "examples, or a `.env.example` snippet).",
            "",
        ]
        for h in low:
            lines += _render_hit(h)

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan a task's events.jsonl for token-shaped strings.")
    ap.add_argument(
        "--task",
        type=int,
        required=True,
        help="Task id to audit (the file scanned is tasks/<status>/<id>/events.jsonl).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write audit-hits.json + audit-report.md to.",
    )
    ap.add_argument(
        "--events-path",
        type=Path,
        default=None,
        help="Explicit events.jsonl path (overrides auto-discovery via task.py).",
    )
    args = ap.parse_args(argv)

    # Discover the events.jsonl path.  Use task.py find <N> (the canonical
    # resolver) rather than hand-building tasks/<status>/<N>/.
    if args.events_path is not None:
        events_path = args.events_path
    else:
        # Late import — keep the script standalone in env-less smoke runs.
        import subprocess

        # task.py find prints the absolute path of the task folder.
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["uv", "run", "python", str(repo_root / "scripts" / "task.py"), "find", str(args.task)],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
        task_dir = Path(result.stdout.strip())
        events_path = task_dir / "events.jsonl"

    if not events_path.exists():
        print(f"events.jsonl not found at {events_path}", file=sys.stderr)
        return 2

    # Count events for the report header.
    with events_path.open("r", encoding="utf-8") as f:
        n_events = sum(1 for ln in f if ln.strip())

    hits = scan_file(events_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hits_json_path = args.out_dir / "audit-hits.json"
    report_md_path = args.out_dir / "audit-report.md"

    hits_payload = {
        "task_scanned": args.task,
        "events_path": str(events_path),
        "n_events": n_events,
        "n_hits": len(hits),
        "hits": [asdict(h) for h in hits],
    }
    hits_json_path.write_text(json.dumps(hits_payload, indent=2) + "\n", encoding="utf-8")

    report = compose_report(args.task, events_path, hits, n_events)
    report_md_path.write_text(report, encoding="utf-8")

    print(f"Scanned {n_events} events at {events_path}")
    print(f"Hits: {len(hits)}")
    print(f"Report: {report_md_path}")
    print(f"Hits JSON: {hits_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
