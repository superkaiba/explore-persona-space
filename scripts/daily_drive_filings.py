"""Drive /daily route-2/3 task filings from a durable filings dir, incrementally.

The permanent promotion of the ad-hoc ``logs/daily/filings-2026-07-04/drive_filings.py``
driver (task #1061). The /daily skill writes every route-2/3 filing body plus a
``manifest.json`` to ``logs/daily/filings-<date>/`` BEFORE any filing starts, then drives
the filings through this script in small batches. Every outcome is appended to
``<dir>/filed.jsonl`` the moment it lands (two-phase ``attempting`` -> terminal rows), so
a mid-run kill strands at most the one in-flight item and a re-invocation resumes from
the ledger instead of forcing a which-got-filed audit.

Manifest item schema: ``{slug, route: 2|3, title, target, bug, change, body?, wf_fix?: bool}``
where ``body`` defaults to ``<dir>/<slug>.md`` (absolute paths pass through; relative paths
resolve against the filings dir). ``wf_fix`` (route 2 only, default ``true``) — ``false``
marks a non-workflow-surface (experiment-code) item per the daily SKILL.md route-2 variant:
the driver drops the ``wf-fix`` / ``wf-fix-fp:<fp>`` tags (keeps ``daily-auto-filed``),
skips the Provenance injection, and skips fp-dedup (#1228). Route-2 titles missing a
``WF_FIX_TITLE_PREFIXES`` prefix gain ``daily-fix: `` before the <=60 truncation (#1273).
Route-3 items get a same-subject dedup instead (#1483): before filing, the driver scans
OPEN (proposed/on_hold/blocked) tasks tagged ``daily-held`` for
``>= ROUTE3_MIN_SHARED_TOKENS`` shared informative title+bug tokens
(``task_workflow.informative_title_tokens``; task side = frontmatter title +
template-stripped ``origin_prompt``); a hit records terminal outcome ``already-tracked``
and skips the filer; any scan error fails OPEN toward filing with a loud stderr WARN.

Route-2 bodies are normalized in place before filing (#1173; skipped for ``wf_fix: false``
items): a body missing the durable
recursion-guard Provenance lines gains ``- workflow_fix_target: <manifest target>`` +
``- fingerprint: <fp>`` under ``## Provenance`` (idempotent temp+rename; the ``INJECTED``
stdout line is the audit trace).

Every item's body (route 2 AND 3, any ``wf_fix`` value) additionally gets the #1467
WARN-only sha-verify backstop right before filing: a SHA-shaped hex token cited in
commit context that does NOT ``git rev-parse`` as a commit in this repo gains one
idempotent advisory line under ``## Provenance`` (heading appended when absent; never a
``workflow_fix_target:`` line) and is recorded as ``sha_warnings`` on the ``filed``
ledger row; other non-resolving hex tokens WARN on stderr only. The scan never blocks a
filing and never changes the exit code (fail-open on git errors); the compose-time
rev-parse duty (daily SKILL.md route-2 mandate) stays the primary defense.

Ledger row shapes (one JSON object per line, ISO-UTC ``ts`` on every row):

- ``{"slug", "outcome": "attempting", "fp", "route", "id_floor", "ts"}`` — appended
  BEFORE the filer subprocess (the crash-safety ordering); ``id_floor`` is the max
  task id at that moment and scopes later title-scan recovery to THIS filing.
- ``{"slug", "outcome": "filed", "id", "rc", "fp", "route", "tail", "ts"}`` — plus
  ``"sha_warnings": [tokens]`` when the #1467 backstop annotated commit-context tokens
- ``{"slug", "outcome": "deduped", "against", "fp", "route", "ts"}`` (route 2 only)
- ``{"slug", "outcome": "already-tracked", "against": <task id>, "against_title",
  "shared": [tokens], "fp", "route": 3, "ts"}`` (route 3 only — the #1483 open
  daily-held overlap dedup; NOTE ``against`` is an int task id here vs the path
  string on route-2 ``deduped`` rows)
- ``{"slug", "outcome": "recovered", "id", "fp", "route", "dispatch_unconfirmed", "ts"}``
- ``{"slug", "outcome": "ERROR", "flag", "id", "rc", "fp", "route", "tail", "ts"}``
  with ``flag`` one of ``filer-failed`` / ``no-id-parsed`` / ``timeout`` /
  ``ambiguous-recovery``.

Exit code: 0 when the whole slice processed without appending an ERROR row this
invocation, else 1 (record-then-report). Validation failures (missing manifest / body
file / unresolvable date / malformed manifest item / non-trailing ledger corruption)
raise before any filing starts.

Usage:
    uv run python scripts/daily_drive_filings.py --dir logs/daily/filings-2026-07-05 \\
        [--start I --end J] [--retry-errors] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from explore_persona_space.task_workflow import (
    WF_FIX_TITLE_PREFIXES,
    informative_title_tokens,
    wf_fix_fingerprint,
)

LEDGER_NAME = "filed.jsonl"
QUARANTINE_NAME = "filed.jsonl.quarantined"
TERMINAL_OUTCOMES = frozenset({"filed", "deduped", "recovered", "already-tracked"})
# ── #1483 route-3 open daily-held overlap dedup ─────────────────────────────
DAILY_HELD_TAG = "daily-held"
ROUTE3_TITLE_PREFIX = "daily-held: "
ROUTE3_OPEN_STATUSES = frozenset({"proposed", "on_hold", "blocked"})
# The origin-prompt template _filer_cmd composes ("/daily <date> problem sweep
# (route N): <bug>"); stripped before tokenizing so its fixed tokens
# (daily/problem/sweep/route + the run date) never count toward overlap.
ROUTE3_ORIGIN_TEMPLATE_RE = re.compile(r"^/daily \S+ problem sweep \(route \d\): ")
# Route-3 channel tokens on every daily-held filing — never informative.
ROUTE3_BOILERPLATE_TOKENS = frozenset({"daily-held", "needs-human"})
# Measured 2026-07-17, re-confirmed at implementation time (plan #1483 §11 D2):
# the #1140/#1472 duplicate pair shares 7 informative tokens; the live open
# daily-held population's (n=5) max NON-duplicate pairwise overlap is 1
# ('every', #1141x#1472). 3 sits 2 above noise, 4 below the hit.
ROUTE3_MIN_SHARED_TOKENS = 3
REQUIRED_ITEM_KEYS = frozenset({"slug", "route", "title", "target", "bug", "change"})
# Anchored to the line start: every file_infra_task.py success path prints a line starting
# `filed #<id>` or `filed + dispatched #<id>`. A stray `#N` elsewhere must not win.
FILED_ID_RE = re.compile(r"^filed (?:\+ dispatched )?#(\d+)", re.M)
DIR_DATE_RE = re.compile(r"filings-(\d{4}-\d{2}-\d{2})$")
FILER_TIMEOUT_S = 280  # bounds file_infra_task.py's internal 120s+120s subprocess timeouts
ROUTE_RECOVERY_TAG = {2: "daily-auto-filed", 3: "daily-held"}


def repo_root() -> Path:
    """Resolve the MAIN checkout root from this script's own location (never the cwd).

    Uses the git common dir so the WORKTREE copy of this script still resolves to the
    canonical repo root — a worktree-cwd invocation must not write a reapable
    worktree-local filings dir, and `task_workflow.repo_root()` is unsuitable here
    (it branch-guards / manages worktrees). Raises on git failure.
    """
    script_dir = Path(__file__).resolve().parent
    proc = subprocess.run(
        ["git", "-C", str(script_dir), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    common = Path(proc.stdout.strip())
    if not common.is_absolute():
        common = (script_dir / common).resolve()
    return common.parent


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_filings_dir(dir_arg: str, root: Path) -> Path:
    """Resolve --dir against the repo root (relative form) or as-is (absolute form)."""
    p = Path(dir_arg)
    if not p.is_absolute():
        p = root / p
    if not p.is_dir():
        raise FileNotFoundError(f"filings dir not found: {p}")
    return p


def resolve_date(date_arg: str | None, dirpath: Path) -> str:
    """The date for origin-prompt strings: --date, else parsed from the dir basename."""
    if date_arg:
        return date_arg
    m = DIR_DATE_RE.search(dirpath.name)
    if not m:
        raise ValueError(
            f"cannot derive date: --date not given and dir basename {dirpath.name!r} "
            "does not match 'filings-YYYY-MM-DD'"
        )
    return m.group(1)


def _resolve_body_path(item: dict, dirpath: Path) -> Path:
    body = item.get("body") or f"{item['slug']}.md"
    p = Path(body)
    if not p.is_absolute():
        p = dirpath / p
    return p


WF_FIX_TARGET_KEY = "workflow_fix_target:"
PROVENANCE_HEADING_RE = re.compile(r"^## Provenance[ \t]*$", re.M)

# ── #1467 sha-verify backstop constants (WARN-only; never blocks a filing) ─────
HEX_TOKEN_RE = re.compile(r"\b[0-9a-f]{7,40}\b")  # git abbrev floor 7; 40 = full SHA-1
HAS_HEX_LETTER_RE = re.compile(r"[a-f]")  # all-digit tokens are dates/ids — skip
# Known non-commit hex classes + our own advisory lines: lines matching this are
# never scanned (fingerprints/wf-fix-fp tags are 12-hex by construction, #1173).
SHA_EXCLUDE_LINE_RE = re.compile(r"fingerprint:|wf-fix-fp:|drift_hash|sha-verify")
COMMIT_CONTEXT_RE = re.compile(
    r"(?i)\bcommits?\b|\bsha\b|\bmerged?\b|\blanded\b|cherry.pick|\bfix(ed)?\s+(in|via)\b"
)
# Self-referential quotes ("transcript basename fc2b61b7, not a commit") cite the
# token as what it actually is — they do NOT count as commit context.
SELF_REF_LINE_RE = re.compile(r"(?i)transcript|basename|not a commit")
SHA_ADVISORY_TMPL = (
    "- sha-verify (filing-time, #1467): `{tok}` cited in commit context does NOT"
    " resolve as a commit in this repo at filing time — treat as a transcript/session"
    " reference, not a commit."
)
# A wedged git must never hang the 3 AM filer: rev-parse is ~5 ms, so 10s is generous.
# TimeoutExpired routes to the same fail-open WARN path as OSError (scan skipped,
# filing proceeds) — the caller catches it alongside OSError/UnicodeDecodeError.
SHA_REV_PARSE_TIMEOUT_S = 10


def _insert_under_provenance(text: str, block: str) -> str:
    """Insert ``block`` under an existing ``## Provenance`` heading, else append one.

    Shared insertion machinery for ensure_wf_fix_provenance and _check_body_shas —
    the heading-less fallback appends a terminal ``## Provenance`` section.
    """
    m = PROVENANCE_HEADING_RE.search(text)
    if m:
        # Insert immediately after the existing heading line.
        insert_at = m.end()
        return text[:insert_at] + "\n\n" + block + text[insert_at:]
    return text.rstrip("\n") + f"\n\n## Provenance\n\n{block}\n"


def ensure_wf_fix_provenance(text: str, target: str, fp: str) -> tuple[str, bool]:
    """Idempotently ensure the durable recursion-guard Provenance lines (#1173).

    Returns (new_text, changed). The ``- workflow_fix_target: <target>`` line is the
    DURABLE signal task_workflow.is_workflow_fix_session() reads (the env-var leg is
    lost on a watcher crash-recovery respawn); ``- fingerprint: <fp>`` is the body-side
    dedup fallback (task_workflow.is_open_workflow_fix_task; sweep _fp_tag_scan).
    Substring contracts (do not reformat): ``workflow_fix_target: {target}`` and
    ``fingerprint: {fp}`` with a single space after the colon.

    Presence checks are substring-based BY DESIGN: a body that merely prose-quotes
    ``workflow_fix_target:`` skips injection — acceptable because that same substring
    already satisfies the recursion-guard predicate; only the dedup body-needle could
    miss, and dedup's PRIMARY key is the ``wf-fix-fp:<fp>`` tag.
    """
    lines = []
    if WF_FIX_TARGET_KEY not in text:
        lines.append(f"- workflow_fix_target: {target}")
    if "fingerprint:" not in text:
        lines.append(f"- fingerprint: {fp}")
    if not lines:
        return text, False
    return _insert_under_provenance(text, "\n".join(lines)), True


def _sha_resolves(token: str, root: Path) -> bool:
    """True iff ``token`` resolves as a commit in the git repo at ``root`` (#1467).

    ``git -C <root> rev-parse --verify --quiet '<token>^{commit}'`` — rc 0 means it
    resolves; any other rc (1 = unknown object; 128 = not a repo / other git error)
    reads as not-resolving, which at worst upgrades nothing beyond a WARN (the
    backstop is WARN-only by construction). An OSError (git binary missing) or a
    subprocess.TimeoutExpired (hung git, SHA_REV_PARSE_TIMEOUT_S cap) propagates
    to _check_body_shas' single fail-open WARNING.
    """
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{token}^{{commit}}"],
        capture_output=True,
        text=True,
        timeout=SHA_REV_PARSE_TIMEOUT_S,
    )
    return proc.returncode == 0


def scan_unresolvable_shas(text: str, root: Path) -> tuple[list[str], list[str]]:
    """Scan body text for SHA-shaped hex tokens that do not resolve as commits (#1467).

    Returns ``(commit_context_offenders, other_nonresolving)`` — deduped, first-seen
    order. Per line: SHA_EXCLUDE_LINE_RE lines (fingerprints, wf-fix-fp tags, drift
    hashes, our own sha-verify advisories) are skipped entirely; HEX_TOKEN_RE matches
    with no [a-f] letter (dates, numeric ids) are skipped. A token is tier 1 when ANY
    line carrying it matches COMMIT_CONTEXT_RE — unless that line also matches
    SELF_REF_LINE_RE (a self-referential quote is not commit context) — else tier 2.
    Resolution is probed once per unique token (rc semantics: _sha_resolves).
    """
    commit_ctx: dict[str, bool] = {}
    for line in text.splitlines():
        if SHA_EXCLUDE_LINE_RE.search(line):
            continue
        in_ctx = bool(COMMIT_CONTEXT_RE.search(line)) and not SELF_REF_LINE_RE.search(line)
        for m in HEX_TOKEN_RE.finditer(line):
            tok = m.group(0)
            if not HAS_HEX_LETTER_RE.search(tok):
                continue
            commit_ctx[tok] = commit_ctx.get(tok, False) or in_ctx
    tier1: list[str] = []
    tier2: list[str] = []
    for tok, in_ctx in commit_ctx.items():
        if _sha_resolves(tok, root):
            continue
        (tier1 if in_ctx else tier2).append(tok)
    return tier1, tier2


def _check_body_shas(item: dict, dirpath: Path, root: Path) -> list[str]:
    """WARN-only #1467 backstop: annotate non-resolving commit-context hex tokens.

    Two tiers, never a refusal: tier 1 (non-resolving token on a commit-context
    line) gets one idempotent advisory line under ``## Provenance`` (heading-less
    bodies gain a terminal ``## Provenance`` section — a bare sha-verify bullet,
    NEVER a ``workflow_fix_target:`` line) plus a stderr WARNING; tier 2 (any other
    non-resolving hex token) gets a stderr WARNING only. Returns the tier-1 tokens
    (the ``filed`` ledger row's ``sha_warnings`` value). Fail-open: any OSError
    (git unavailable, unreadable body), UnicodeDecodeError (non-UTF-8 body — a
    ValueError subclass, NOT an OSError), or subprocess.TimeoutExpired (hung git)
    → ONE loud WARNING and ``[]`` — the backstop never blocks a filing and never
    changes the driver's exit code.
    """
    slug = item["slug"]
    try:
        body_path = _resolve_body_path(item, dirpath)
        text = body_path.read_text(encoding="utf-8")
        tier1, tier2 = scan_unresolvable_shas(text, root)
        for tok in tier2:
            print(
                f"WARNING {slug}: hex token `{tok}` does not resolve as a commit in"
                " this repo (no commit-context line; not annotated) (#1467)",
                file=sys.stderr,
            )
        if not tier1:
            return []
        new_lines = []
        for tok in tier1:
            print(
                f"WARNING {slug}: `{tok}` is cited in commit context but does not"
                " resolve as a commit in this repo at filing time — re-derive the real"
                " commit or cite it as a transcript/session reference (#1467)",
                file=sys.stderr,
            )
            if any("sha-verify" in ln and tok in ln for ln in text.splitlines()):
                continue  # idempotent: this token already carries an advisory line
            new_lines.append(SHA_ADVISORY_TMPL.format(tok=tok))
        if new_lines:
            new_text = _insert_under_provenance(text, "\n".join(new_lines))
            tmp = body_path.with_suffix(".md.tmp")
            tmp.write_text(new_text, encoding="utf-8")
            os.replace(tmp, body_path)  # temp+rename, same pattern as load_ledger
            print(f"ANNOTATED {slug}: {len(new_lines)} sha-verify advisory line(s) (#1467)")
        return tier1
    except (OSError, UnicodeDecodeError, subprocess.TimeoutExpired) as e:
        # Deliberate fail-open (plan #1467 §4): the backstop must never block a
        # filing — surface loudly, skip the scan, and leave the compose-time duty
        # (daily SKILL.md route-2 mandate) as the defense.
        print(
            f"WARNING {slug}: sha-verify scan skipped ({e.__class__.__name__}: {e}) —"
            " fail-open, filing proceeds; the compose-time rev-parse duty still"
            " applies (#1467)",
            file=sys.stderr,
        )
        return []


def _filed_ledger_row(
    slug: str, tid: int, fp: str, item: dict, tail: str, sha_warnings: list[str]
) -> dict:
    """Compose the terminal ``filed`` ledger row.

    Gains ``"sha_warnings": [tokens]`` only when the #1467 backstop annotated
    non-resolving commit-context tokens (rows are free-form dicts — schema-safe).
    """
    row = {
        "slug": slug,
        "outcome": "filed",
        "id": tid,
        "rc": 0,
        "fp": fp,
        "route": item["route"],
        "tail": tail,
    }
    if sha_warnings:
        row["sha_warnings"] = sha_warnings
    return row


def _dry_run_sha_note(item: dict, dirpath: Path, root: Path) -> str:
    """The dry-run mirror of _check_body_shas (#1467): report counts, mutate nothing.

    Returns a suffix for the dry-run FILE line — empty when the scan is clean;
    fail-open (a note, never a raise) on git/read OSError, a non-UTF-8 body's
    UnicodeDecodeError, or a hung-git TimeoutExpired, mirroring the real path.
    """
    try:
        tier1, tier2 = scan_unresolvable_shas(
            _resolve_body_path(item, dirpath).read_text(encoding="utf-8"), root
        )
    except (OSError, UnicodeDecodeError, subprocess.TimeoutExpired) as e:
        return f" [sha-scan skipped: {e.__class__.__name__}]"
    if tier1 or tier2:
        return f" [sha-scan: {len(tier1)} commit-context, {len(tier2)} other non-resolving]"
    return ""


def _wf_fix_enabled(item: dict) -> bool:
    """True when this route-2 item participates in the wf-fix key space (#1228).

    Route-2 items default to wf-fix semantics (tags + Provenance injection +
    fp-dedup); a manifest ``wf_fix: false`` marks a non-workflow-surface
    (experiment-code) item per the daily SKILL.md route-2 variant — it keeps
    ``daily-auto-filed`` only, and its spawned session is NOT a workflow-fix
    session (no recursion guard). Always False for route 3. The ONE shared
    predicate across the tag block, the injection block, the fp-dedup call,
    and the dry-run mirrors — so the tag and the durable recursion-guard body
    signal cannot diverge on the driver path (#1173 coupling invariant).
    """
    return item["route"] == 2 and item.get("wf_fix", True)


def _effective_title(item: dict) -> str:
    """The title actually filed for this manifest item.

    Route-2 titles gain the ``daily-fix: `` channel prefix when the manifest
    omitted every ``WF_FIX_TITLE_PREFIXES`` prefix (#1273: the 2026-07-09
    manifest filed 26 bare titles invisible to
    ``task_workflow.is_open_workflow_fix_task``'s title pre-filter). Prepend
    happens BEFORE the [:60] truncation (the daily SKILL.md contract budgets
    the prefix inside <=60). Already-prefixed titles (either channel prefix)
    pass through un-double-prefixed; route-3 titles are never touched. The
    ONE shared normalization for _filer_cmd AND _try_recovery — the filed
    title and the recovery-scan title cannot diverge (#1173 coupling pattern).
    """
    title = item["title"]
    if item["route"] == 2 and not title.startswith(WF_FIX_TITLE_PREFIXES):
        title = f"daily-fix: {title}"
    return title[:60]


def _warn_stray_wf_fix_provenance(item: dict, dirpath: Path) -> None:
    """WARN-only (#1228): a ``wf_fix: false`` body should not carry the guard line.

    The daily SKILL.md route-2 variant says "do not hand-add a
    ``workflow_fix_target:`` Provenance block to a ``wf_fix: false`` body" — the
    line would arm ``task_workflow.is_workflow_fix_session()`` for a session that
    is NOT a workflow-fix session. Never blocks the filing (the substring may be
    a legitimate prose quote); no-op for route 3 and for wf-fix-enabled items.
    """
    if item["route"] != 2 or _wf_fix_enabled(item):
        return
    text = _resolve_body_path(item, dirpath).read_text(encoding="utf-8")
    if WF_FIX_TARGET_KEY in text:
        print(
            f"WARNING {item['slug']}: wf_fix=false but the body contains"
            f" '{WF_FIX_TARGET_KEY}' — the spawned session would be recursion-guarded"
            " as a workflow-fix session; remove the hand-added Provenance line"
            " (daily SKILL.md route-2 variant, #1228)",
            file=sys.stderr,
        )


def load_and_validate_manifest(dirpath: Path) -> list[dict]:
    """Parse + validate the WHOLE manifest up front, so a schema wart aborts at ZERO filings.

    Checks per item: required keys present, route in {2, 3}, referenced body file exists.
    Returns the manifest list; raises on any violation (fail loud, nothing filed).
    """
    manifest_path = dirpath / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"manifest must be a non-empty JSON list: {manifest_path}")
    seen_slugs: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {i} is not an object")
        missing = REQUIRED_ITEM_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"manifest item {i} ({item.get('slug', '?')}): missing keys {sorted(missing)}"
            )
        if item["route"] not in (2, 3):
            raise ValueError(
                f"manifest item {i} ({item['slug']}): route must be 2 or 3, got {item['route']!r}"
            )
        if "wf_fix" in item and not isinstance(item["wf_fix"], bool):
            # A JSON string "false" is truthy — silently accepting it would invert the
            # flag's intent (#1228). Fail loud at ZERO filings, per this function's contract.
            raise ValueError(
                f"manifest item {i} ({item['slug']}): wf_fix must be a JSON boolean,"
                f" got {item['wf_fix']!r}"
            )
        if item["slug"] in seen_slugs:
            raise ValueError(f"manifest item {i}: duplicate slug {item['slug']!r}")
        seen_slugs.add(item["slug"])
        body_path = _resolve_body_path(item, dirpath)
        if not body_path.is_file():
            raise FileNotFoundError(
                f"manifest item {i} ({item['slug']}): body file not found: {body_path}"
            )
    return items


def load_ledger(dirpath: Path) -> list[dict]:
    """Read filed.jsonl rows; tolerate exactly ONE corrupt/truncated TRAILING line.

    A kill mid-append can truncate the final line: that line is quarantined to
    ``filed.jsonl.quarantined`` with a loud stderr WARN and the ledger rewritten without
    it (the in-flight item it represented is simply not terminal — normal recovery
    applies). A corrupt NON-trailing line is real corruption and raises.
    """
    path = dirpath / LEDGER_NAME
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    rows: list[dict] = []
    bad: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append((idx, line))
    if not bad:
        return rows
    if len(bad) == 1 and bad[0][0] == len(lines) - 1:
        # Truncated trailing line from a mid-append kill: quarantine + rewrite.
        idx, line = bad[0]
        with open(dirpath / QUARANTINE_NAME, "a", encoding="utf-8") as fh:
            fh.write(f"{_utc_ts()} {line}\n")
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(f"{ln}\n" for ln in lines[:-1]), encoding="utf-8")
        os.replace(tmp, path)
        print(
            f"WARNING: quarantined corrupt trailing ledger line to {dirpath / QUARANTINE_NAME}"
            " (kill mid-append; the in-flight item is not terminal and resumes normally)",
            file=sys.stderr,
        )
        return rows
    raise ValueError(
        f"corrupt non-trailing line(s) in {path} at index(es) "
        f"{[i for i, _ in bad]} — real corruption, refusing to resume"
    )


def append_row(dirpath: Path, row: dict) -> dict:
    """Single-line O_APPEND write of one ledger row, ISO-UTC ts added."""
    row = {**row, "ts": _utc_ts()}
    with open(dirpath / LEDGER_NAME, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def max_task_id(tasks_root: Path) -> int:
    """Current max task id (REGISTRY.json when present, else a tasks/*/* dir scan)."""
    ids: list[int] = []
    reg = tasks_root / "REGISTRY.json"
    if reg.is_file():
        try:
            data = json.loads(reg.read_text(encoding="utf-8"))
            ids = [int(k) for k in data.get("tasks", {})]
        except (json.JSONDecodeError, ValueError):
            ids = []
    if not ids:
        ids = [int(p.name) for p in tasks_root.glob("*/*") if p.is_dir() and p.name.isdigit()]
    return max(ids, default=0)


def _read_frontmatter(body_path: Path) -> dict:
    """Parse a task body.md's YAML frontmatter (real YAML — titles are quoted by safe_dump)."""
    try:
        text = body_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    try:
        fm = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return {}
    return fm if isinstance(fm, dict) else {}


def scan_recovery_candidates(tasks_root: Path, title: str, id_floor: int, route: int) -> list[int]:
    """Task ids matching ALL of: YAML-parsed title equality, id above floor, run's route tag.

    The id floor (recorded in the slug's ``attempting`` row) scopes the match to tasks
    created by THIS filing attempt, ruling out a same-titled task from a prior night.
    """
    tag = ROUTE_RECOVERY_TAG[route]
    matches: list[int] = []
    for body in tasks_root.glob("*/*/body.md"):
        task_dir = body.parent
        if not task_dir.name.isdigit():
            continue
        tid = int(task_dir.name)
        if tid <= id_floor:
            continue
        fm = _read_frontmatter(body)
        if fm.get("title") != title:
            continue
        tags = fm.get("tags") or []
        if not isinstance(tags, list) or tag not in tags:
            continue
        matches.append(tid)
    return sorted(matches)


def find_open_fp_duplicate(tasks_root: Path, fp: str) -> Path | None:
    """First NON-terminal task body.md carrying ``wf-fix-fp:<fp>`` (route-2 dedup).

    Same predicate as the proven ad-hoc driver: a tag-scan over non-``completed``/
    ``archived`` statuses. Deliberately COARSER than
    ``task_workflow.is_open_workflow_fix_task`` (which since #1180 DOES see
    ``daily-fix:`` titles via ``WF_FIX_TITLE_PREFIXES``): this scan keys on the
    fingerprint tag ALONE — any title, any kind, no ``workflow_fix_target:``
    requirement, filesystem-only (no REGISTRY read) — so a same-fp task filed
    by EITHER channel blocks a daily re-file even when its registry row or
    Provenance line is malformed. Kept (not delegated) for that coarser grain
    and for the ``tasks_root`` injection the test fixtures use. Called only for
    ``_wf_fix_enabled`` items (#1228) — a ``wf_fix: false`` filing never carries
    the fp tag, so its participation would be one-way.
    """
    needle = f"wf-fix-fp:{fp}"
    for body in sorted(tasks_root.glob("*/*/body.md")):
        status = body.parent.parent.name
        if status in ("completed", "archived"):
            continue
        try:
            text = body.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle in text:
            return body
    return None


def _route3_item_tokens(title: str, bug: str) -> set[str]:
    """Informative tokens of a route-3 item: title (daily-held: stripped) + bug text."""
    t = (title or "").removeprefix(ROUTE3_TITLE_PREFIX)
    return (informative_title_tokens(t) | informative_title_tokens(bug or "")) - (
        ROUTE3_BOILERPLATE_TOKENS
    )


def find_open_daily_held_duplicate(tasks_root: Path, title: str, bug: str) -> dict | None:
    """Best OPEN daily-held task sharing >= ROUTE3_MIN_SHARED_TOKENS tokens (#1483).

    Population: tasks under ROUTE3_OPEN_STATUSES whose frontmatter tags carry
    DAILY_HELD_TAG (tag-only key; kind not checked). Task-side tokens come from
    frontmatter title + origin_prompt with the /daily template prefix stripped
    (origin_prompt IS the filing-time bug[:400] by _filer_cmd construction) —
    frontmatter-only reads, never the body (body boilerplate would inflate
    overlap). Returns {"id", "title", "shared"} for the max-overlap hit
    (tie: lowest id), else None. Kept in the driver (not task_workflow) for the
    tasks_root injection the test fixtures use, same as find_open_fp_duplicate.
    Callers wrap in try/except: the scan is FAIL-OPEN by contract.
    """
    cand = _route3_item_tokens(title, bug)
    if not cand:
        return None
    best: dict | None = None
    for body in sorted(tasks_root.glob("*/*/body.md")):
        if body.parent.parent.name not in ROUTE3_OPEN_STATUSES:
            continue
        if not body.parent.name.isdigit():
            continue
        fm = _read_frontmatter(body)
        tags = fm.get("tags") or []
        if not isinstance(tags, list) or DAILY_HELD_TAG not in tags:
            continue
        origin = ROUTE3_ORIGIN_TEMPLATE_RE.sub("", str(fm.get("origin_prompt") or ""))
        shared = cand & _route3_item_tokens(str(fm.get("title") or ""), origin)
        if len(shared) < ROUTE3_MIN_SHARED_TOKENS:
            continue
        tid = int(body.parent.name)
        if best is None or (len(shared), -tid) > (len(best["shared"]), -best["id"]):
            best = {"id": tid, "title": str(fm.get("title") or ""), "shared": sorted(shared)}
    return best


def _route3_dup_or_none(item: dict, tasks_root: Path) -> dict | None:
    """Fail-open wrapper (#1483): any scan error WARNs loudly and files as today.

    Broad Exception catch is DELIBERATE, mandated by the task's fail-open
    constraint (a held item is never lost to a scan bug; the loud stderr
    WARNING is the fail-loud channel) — same pattern class as _check_body_shas
    (#1467), widened because the token/YAML surface has a broader error space
    than the enumerable I/O classes there.
    """
    try:
        return find_open_daily_held_duplicate(tasks_root, item["title"], item["bug"])
    except Exception as e:  # deliberate fail-open, see docstring
        print(
            f"WARNING {item['slug']}: daily-held overlap scan skipped"
            f" ({e.__class__.__name__}: {e}) — fail-open, filing proceeds (#1483)",
            file=sys.stderr,
        )
        return None


def _route3_already_tracked(
    item: dict, tasks_root: Path, *, dirpath: Path, fp: str, dry_run: bool
) -> str | None:
    """Route-3 open daily-held overlap-dedup outcome for process_item, or None (#1483).

    Non-route-3 items and no-overlap scans return None (caller proceeds). On a hit
    the ALREADY-TRACKED line prints either way; the real path first appends the
    `already-tracked` ledger row and returns 'already-tracked', while dry-run stays
    read-only by construction (no ledger write) and returns 'skip'.
    """
    if item["route"] != 3:
        return None
    dup3 = _route3_dup_or_none(item, tasks_root)
    if dup3 is None:
        return None
    slug = item["slug"]
    if not dry_run:
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "already-tracked",
                "against": dup3["id"],
                "against_title": dup3["title"],
                "shared": dup3["shared"],
                "fp": fp,
                "route": 3,
            },
        )
    print(f"ALREADY-TRACKED {slug} -> #{dup3['id']} (shared: {','.join(dup3['shared'])})")
    return "skip" if dry_run else "already-tracked"


def _dry_run_inject_note(item: dict, dirpath: Path, fp: str) -> str:
    """Read-only injection-intent probe for the dry-run FILE line (#1173): write-free.

    Returns the ' [will inject workflow_fix_target provenance]' suffix when the real
    path would inject the wf-fix Provenance block, else ''. No exists() guard — a
    missing/unreadable body fails LOUD here, the same fail-fast contract
    load_and_validate_manifest enforces up front. Non-wf-fix items get the
    stray-provenance WARN instead.
    """
    if not _wf_fix_enabled(item):
        _warn_stray_wf_fix_provenance(item, dirpath)
        return ""
    body_text = _resolve_body_path(item, dirpath).read_text(encoding="utf-8")
    if ensure_wf_fix_provenance(body_text, item["target"], fp)[1]:
        return " [will inject workflow_fix_target provenance]"
    return ""


def _filer_cmd(
    filer_prefix: list[str], item: dict, body_path: Path, date: str, fp: str
) -> list[str]:
    """Compose the file_infra_task.py argv for one manifest item (per-route tags)."""
    cmd = [
        *filer_prefix,
        "--kind",
        "infra",
        "--title",
        _effective_title(item),
        "--body-file",
        str(body_path),
        "--origin-prompt",
        f"/daily {date} problem sweep (route {item['route']}): {item['bug'][:400]}",
    ]
    if item["route"] == 2:
        if _wf_fix_enabled(item):
            cmd += ["--tag", "wf-fix", "--tag", f"wf-fix-fp:{fp}"]
        cmd += ["--tag", "daily-auto-filed"]
    else:
        cmd += ["--tag", "daily-held", "--tag", "needs-human", "--no-dispatch"]
    return cmd


def parse_filed_id(stdout: str, stderr: str) -> int | None:
    """Anchored id parse over stdout then stderr (a stray ``#N`` in a warning never wins)."""
    m = FILED_ID_RE.search(stdout or "")
    if m is None:
        m = FILED_ID_RE.search(stderr or "")
    return int(m.group(1)) if m else None


def _slug_state(ledger: list[dict], slug: str, retry_errors: bool) -> str:
    """Classify a slug against the ledger: 'terminal' | 'retry-error' | 'in-flight' | 'fresh'."""
    rows = [r for r in ledger if r.get("slug") == slug]
    outcomes = {r.get("outcome") for r in rows}
    if outcomes & TERMINAL_OUTCOMES:
        return "terminal"
    if "ERROR" in outcomes:
        return "retry-error" if retry_errors else "terminal"
    if rows and rows[-1].get("outcome") == "attempting":
        return "in-flight"
    return "fresh"


def _last_attempting_row(ledger: list[dict], slug: str) -> dict | None:
    for row in reversed(ledger):
        if row.get("slug") == slug and row.get("outcome") == "attempting":
            return row
    return None


def _try_recovery(
    dirpath: Path, tasks_root: Path, ledger: list[dict], item: dict, fp: str
) -> str | None:
    """Title-scan recovery for a kill-during-file window. Returns an outcome or None.

    None means "nothing recovered — proceed to dedup/file". Fires only when an
    ``attempting`` row exists (its ``id_floor`` scopes the scan). EXACTLY ONE match
    recovers (``dispatch_unconfirmed`` — the filer's tail is unknown); MULTIPLE matches
    never auto-recover (ERROR ``ambiguous-recovery`` for manual disposition).
    """
    slug = item["slug"]
    attempting = _last_attempting_row(ledger, slug)
    if attempting is None:
        return None
    id_floor = int(attempting.get("id_floor", 0))
    # Union over both title forms: the effective (prefixed) title the post-#1273
    # driver files, AND the raw [:60] form a crashed PRE-fix driver may have filed
    # (the one-shot prefix-migration window). Post-fix, for an already-prefixed
    # manifest title the set collapses to one element.
    titles = {_effective_title(item), item["title"][:60]}
    matches = sorted(
        {
            tid
            for t in titles
            for tid in scan_recovery_candidates(tasks_root, t, id_floor, item["route"])
        }
    )
    if not matches:
        return None
    if len(matches) == 1:
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "recovered",
                "id": matches[0],
                "fp": fp,
                "route": item["route"],
                "dispatch_unconfirmed": True,
            },
        )
        print(f"RECOVERED {slug} -> #{matches[0]}")
        return "recovered"
    append_row(
        dirpath,
        {
            "slug": slug,
            "outcome": "ERROR",
            "flag": "ambiguous-recovery",
            "id": None,
            "rc": None,
            "fp": fp,
            "route": item["route"],
            "tail": f"candidates: {matches}",
        },
    )
    print(f"ERROR {slug} (ambiguous-recovery: {matches})")
    return "error"


def process_item(
    item: dict,
    *,
    dirpath: Path,
    tasks_root: Path,
    ledger: list[dict],
    filer_prefix: list[str],
    date: str,
    root: Path,
    retry_errors: bool,
    dry_run: bool,
) -> str:
    """Run one manifest item through resume -> recovery -> dedup -> two-phase file.

    Returns the item outcome:
    'skip' | 'recovered' | 'deduped' | 'already-tracked' | 'filed' | 'error'.
    """
    slug = item["slug"]
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    state = _slug_state(ledger, slug, retry_errors)

    if state == "terminal":
        print(f"SKIP {slug}")
        return "skip"

    if dry_run:
        # #1483 dry-run mirror of the route-3 overlap dedup (read-only by construction).
        outcome3 = _route3_already_tracked(item, tasks_root, dirpath=dirpath, fp=fp, dry_run=True)
        if outcome3 is not None:
            return outcome3
        if _wf_fix_enabled(item) and find_open_fp_duplicate(tasks_root, fp) is not None:
            print(f"DEDUP {slug} -> wf-fix-fp:{fp}")
        else:
            tags = _filer_cmd([], item, Path("-"), date, fp)
            pending = (
                " [in-flight attempting row; recovery scan runs first]" if state != "fresh" else ""
            )
            inject = _dry_run_inject_note(item, dirpath, fp)
            sha_note = _dry_run_sha_note(item, dirpath, root)
            print(f"FILE {slug} tags={tags[tags.index('--tag') :]}{pending}{inject}{sha_note}")
        return "skip"

    if state in ("in-flight", "retry-error"):
        # Recovery-before-refile: the prior attempt may have committed the task
        # (kill between task.py new's commit and the ledger append; rc=0-no-id; timeout).
        outcome = _try_recovery(dirpath, tasks_root, ledger, item, fp)
        if outcome is not None:
            return outcome

    if _wf_fix_enabled(item):
        dup = find_open_fp_duplicate(tasks_root, fp)
        if dup is not None:
            append_row(
                dirpath,
                {
                    "slug": slug,
                    "outcome": "deduped",
                    "against": str(dup),
                    "fp": fp,
                    "route": item["route"],
                },
            )
            print(f"DEDUP {slug} -> {dup}")
            return "deduped"

    # #1483 route-3 open daily-held overlap dedup — after recovery, before the
    # `attempting` row, exactly the route-2 fp-dedup ordering. Note: a route-3
    # re-file whose title-scan recovery MISSED will overlap-match its own night-1
    # task and record `already-tracked` instead of `recovered` — benign, no
    # duplicate is filed.
    outcome3 = _route3_already_tracked(item, tasks_root, dirpath=dirpath, fp=fp, dry_run=False)
    if outcome3 is not None:
        return outcome3

    body_path = _resolve_body_path(item, dirpath)
    if _wf_fix_enabled(item):
        # Same condition under which _filer_cmd applies the wf-fix tag — the tag and
        # the durable recursion-guard body signal cannot diverge on the driver path
        # (#1173; both sites key on _wf_fix_enabled, #1228). Idempotent, so a kill
        # anywhere re-normalizes harmlessly on resume.
        text = body_path.read_text(encoding="utf-8")
        new_text, changed = ensure_wf_fix_provenance(text, item["target"], fp)
        if changed:
            tmp = body_path.with_suffix(".md.tmp")
            tmp.write_text(new_text, encoding="utf-8")
            os.replace(tmp, body_path)  # temp+rename, same pattern as load_ledger
            print(f"INJECTED {slug}: workflow_fix_target provenance (#1173 recursion-guard signal)")
    else:
        _warn_stray_wf_fix_provenance(item, dirpath)
    # #1467 WARN-only sha-verify backstop: runs AFTER the Provenance injection (so the
    # advisory lands in the FILED body) and for EVERY route/wf_fix variant — content
    # accuracy is orthogonal to the wf-fix key space. Never blocks; exit code untouched.
    sha_warnings = _check_body_shas(item, dirpath, root)
    # Two-phase ledger: the `attempting` row (with the recovery id floor) lands BEFORE
    # the filer subprocess — the load-bearing crash-safety ordering.
    append_row(
        dirpath,
        {
            "slug": slug,
            "outcome": "attempting",
            "fp": fp,
            "route": item["route"],
            "id_floor": max_task_id(tasks_root),
        },
    )
    cmd = _filer_cmd(filer_prefix, item, body_path, date, fp)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(root), timeout=FILER_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as e:
        tail = (
            ((e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
            + "\n"
            + ((e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))
        )
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "ERROR",
                "flag": "timeout",
                "id": None,
                "rc": None,
                "fp": fp,
                "route": item["route"],
                "tail": tail.strip()[-300:],
            },
        )
        print(f"ERROR {slug} (timeout after {FILER_TIMEOUT_S}s)")
        return "error"

    out = (proc.stdout + "\n" + proc.stderr).strip()
    tid = parse_filed_id(proc.stdout, proc.stderr)
    if proc.returncode == 0 and tid is not None:
        append_row(dirpath, _filed_ledger_row(slug, tid, fp, item, out[-300:], sha_warnings))
        print(f"FILED {slug} -> #{tid} (rc=0)")
        return "filed"
    # rc=0 with NO parseable id is classified ERROR `no-id-parsed`, NEVER a `filed` row
    # with a null id (a null-id filed row poisons the resume set + the daily-file record).
    # The task may still have committed — --retry-errors runs recovery-before-refile.
    flag = "no-id-parsed" if proc.returncode == 0 else "filer-failed"
    append_row(
        dirpath,
        {
            "slug": slug,
            "outcome": "ERROR",
            "flag": flag,
            "id": tid,
            "rc": proc.returncode,
            "fp": fp,
            "route": item["route"],
            "tail": out[-300:],
        },
    )
    print(f"ERROR {slug} (rc={proc.returncode}, {flag})")
    if proc.returncode != 0:
        print(out[-800:], file=sys.stderr)
    return "error"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir",
        required=True,
        help="filings dir (e.g. logs/daily/filings-2026-07-05); a relative path resolves"
        " against the REPO ROOT (never the invoking cwd)",
    )
    parser.add_argument("--start", type=int, default=None, help="manifest slice start index")
    parser.add_argument("--end", type=int, default=None, help="manifest slice end index")
    parser.add_argument(
        "--date",
        default=None,
        help="date for origin-prompt strings; default parsed from the dir basename",
    )
    parser.add_argument(
        "--filer",
        default=None,
        help="filer command override (shlex-split argv prefix); default"
        " 'uv run python scripts/file_infra_task.py' run with cwd=repo root — the test seam",
    )
    parser.add_argument(
        "--tasks-root",
        default=None,
        help="dedup/recovery scan root; default <repo_root>/tasks — the test seam",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="re-attempt slugs whose only terminal row is ERROR (recovery scan runs first)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print per-item planned action (FILE/DEDUP/SKIP); no filer subprocess, no"
        " ledger/body writes (read-only git rev-parse sha-scan probes may run, #1467)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the manifest, then drive the slice through resume/recovery/dedup/file."""
    args = build_parser().parse_args(argv)
    root = repo_root()
    dirpath = resolve_filings_dir(args.dir, root)
    date = resolve_date(args.date, dirpath)
    tasks_root = Path(args.tasks_root) if args.tasks_root else root / "tasks"
    filer_prefix = (
        shlex.split(args.filer)
        if args.filer
        else ["uv", "run", "python", "scripts/file_infra_task.py"]
    )
    manifest = load_and_validate_manifest(dirpath)
    ledger = load_ledger(dirpath)
    sliced = manifest[args.start : args.end]
    any_error = False
    for item in sliced:
        outcome = process_item(
            item,
            dirpath=dirpath,
            tasks_root=tasks_root,
            ledger=ledger,
            filer_prefix=filer_prefix,
            date=date,
            root=root,
            retry_errors=args.retry_errors,
            dry_run=args.dry_run,
        )
        if outcome == "error":
            any_error = True
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
