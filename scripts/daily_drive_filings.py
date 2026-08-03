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
template-stripped ``origin_prompt``) — excluding route-3-generic workflow vocabulary +
date tokens, and additionally requiring >= 1 shared post-exclusion TITLE token (#1687);
a hit records terminal outcome ``already-tracked``
and skips the filer; any scan error fails OPEN toward filing with a loud stderr WARN.

Every item (routes 2 AND 3) additionally gets TWO mechanical landed-fix probes as
one of the LAST dedup-family checks before any mutation, in order (#1674 first,
#1711 second):

- #1674 commit-subject probe: ``git log --since='7 days ago'`` scoped to the
  item's own target path(s); a commit SUBJECT sharing
  ``>= LANDED_FIX_MIN_SHARED_TOKENS`` informative tokens with the item's
  title+bug+change text (same tokenizer + exclusions as the route-3 dedup) records
  terminal outcome ``landed-fix-suspect`` (suspect sha(s) + subjects + shared
  tokens for the eyeball) and skips the filer. Any git error fails OPEN toward
  filing with ONE loud stderr WARN. Subject-only matching by design (plan #1674
  §11 D3).
- #1711 closed-sibling probe (belt-and-suspenders backstop for the
  vocabulary-divergent landed-fix class #1386/#1360; runs ONLY when #1674 did NOT
  suppress): reuses ``task_workflow.recent_closed_workflow_fix_tasks`` (#1446 helper)
  scanning recently-closed ``kind: infra`` sibling tasks. PATH arms (``target`` /
  ``infra-target``) BLOCK the filing with the SAME ``landed-fix-suspect`` terminal
  outcome (suspects[].kind = ``"closed-sibling"``); TITLE-only arms (``title:*`` /
  ``infra-title:*``) fire a ``CLOSED-SIBLING-ADVISORY`` stderr line but do NOT
  suppress (the helper docstring flags the unmeasured title-arm FP surface).
  Any error fails OPEN toward filing (broad ``except Exception``, mirroring
  ``_route3_dup_or_none``'s rationale — the helper's error surface spans registry
  reads + N body reads + YAML parses, broader than a single ``git log`` call).

The two probes are OR-combined at ``process_item`` level; either can suppress with
the SAME terminal outcome, so one ``--retry-suspects`` re-drives both. The
compose-time clause-(a') git-log duty (daily SKILL.md route-2 mandate) plus the
compose-time closed-sibling eyeball stay PRIMARY — these probes are the
mechanical backstops.

Stdout/stderr split for the closed-sibling probe (mirrors #1674's
``LANDED-FIX-SUSPECT`` on stdout / ``WARNING`` on stderr): the blocking
``CLOSED-SIBLING-SUSPECT`` line prints on stdout (operator-facing — same channel
as ``LANDED-FIX-SUSPECT``, so a fleet eyeball grep catches both), while the
non-blocking ``CLOSED-SIBLING-ADVISORY`` line prints on stderr (informational,
WARNING-adjacent — same channel as the helper's fail-open WARN).

Why closed-sibling and NOT a commit-body / files-changed extension of #1674: the
closed-sibling probe carries stronger provenance (the closed task IS the sibling
and can be linked in the ledger for the eyeball); commit-body/files-changed
matching is looser (any recent commit touching the paths, not necessarily a
landed FIX).

Same-target dispatch hold (#1678): a route-2 item sharing >=1 normalized target token
(comma-split, ``./``-stripped, glob-as-literal) with any EARLIER route-2 item in the FULL
manifest files via ``--no-dispatch`` (task filed, session NOT spawned); the watcher's
``proposed_infra_sweep`` is the dispatch backstop. The contention reduction is
PROBABILISTIC and staggers only the group HEAD vs its HELD siblings (~3-13 min via the
watcher's 10-min cron cadence + its 600 s dispatch-marker-freshness gate; longer under
the shared 5-slot cap): two HELD same-target siblings dispatch in the SAME sweep pass,
spaced only by the #1059 60 s stamp — a k-item group goes from k simultaneous same-file
editors to k-1. No merge-ordering guarantee is claimed; the Step 10d merge-recovery
shapes remain the backstop.

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
ledger row; other non-resolving hex tokens WARN on stderr only. The item's own wf-fix
fp and every ``fingerprint:``-labeled 12-hex token in the body are token-exempt from
the walk (#1808 — an fp is not a commit). The scan never blocks a
filing and never changes the exit code (fail-open on git errors); the compose-time
rev-parse duty (daily SKILL.md route-2 mandate) stays the primary defense.

Ledger row shapes (one JSON object per line, ISO-UTC ``ts`` on every row):

- ``{"slug", "outcome": "attempting", "fp", "route", "id_floor", "ts"}`` — appended
  BEFORE the filer subprocess (the crash-safety ordering); ``id_floor`` is the max
  task id at that moment and scopes later title-scan recovery to THIS filing.
- ``{"slug", "outcome": "filed", "id", "rc", "fp", "route", "tail", "ts"}`` — plus
  ``"sha_warnings": [tokens]`` when the #1467 backstop annotated commit-context tokens,
  plus ``"held_dispatch": true, "held_with": <slug>, "shared_target": [tokens]`` when
  the #1678 same-target hold applied
- ``{"slug", "outcome": "deduped", "against", "fp", "route", "ts"}`` (route 2 only)
- ``{"slug", "outcome": "already-tracked", "against": <task id>, "against_title",
  "shared": [tokens], "fp", "route": 3, "ts"}`` (route 3 only — the #1483 open
  daily-held overlap dedup; NOTE ``against`` is an int task id here vs the path
  string on route-2 ``deduped`` rows)
- ``{"slug", "outcome": "landed-fix-suspect", "suspects": [{"sha", "subject",
  "shared"}], "threshold", "window", "paths", "fp", "route", "ts"}`` (routes 2 AND 3 —
  the #1674 mechanical commit-subject landed-fix probe hit; terminal without
  ``--retry-suspects``)
- ``{"slug", "outcome": "landed-fix-suspect", "suspects": [{"kind":
  "closed-sibling", "id", "title", "status", "target", "closed_at", "matched"}],
  "threshold": null, "window": "7.0 days", "paths", "fp", "route", "ts"}``
  (routes 2 AND 3 — the #1711 closed-sibling probe hit; the ``kind`` discriminator
  distinguishes closed-sibling rows from #1674's commit-subject rows, which stay
  byte-unchanged without a ``kind`` field — Option A source-compat; ``threshold``
  is ``null`` here because closed-sibling arms are boolean, not token-threshold
  based; ``--retry-suspects`` re-drives BOTH probes together)
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
        [--start I --end J] [--retry-errors] [--retry-suspects] [--dry-run]
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
# ── #1687 route-3 generic-vocabulary + date exclusion ────────────────────────
# Generic workflow vocabulary must never count toward route-3 overlap: the
# 2026-07-24 incident suppressed a held filing against unrelated #1537 on
# {gate, step, warn} alone. Members are (i) every token measured at document
# frequency >= 4/37 on the FULL historical daily-held corpus (frontmatter
# title + template-stripped origin_prompt, measured 2026-07-25, plan #1687
# §11 D1), (ii) their plural forms (the tokenizer does not stem), and
# (iii) the incident token family warn/warning. Subject-bearing df==3 tokens
# (codex, zombie, stranded, draft, review, ...) are deliberately KEPT
# informative — see plan #1687 §11 D1 and the calibration-guard test.
ROUTE3_GENERIC_TOKENS = frozenset(
    {
        # measured df >= 4/37 (2026-07-25 corpus pass):
        "daily",
        "backlog",
        "held",
        "every",
        "gate",
        "still",
        "step",
        "route-3",
        "session",
        "across",
        # plural forms of the measured tokens (no stemming in the tokenizer):
        "gates",
        "steps",
        "sessions",
        # the #1687 incident token family (log-level vocabulary):
        "warn",
        "warns",
        "warning",
        "warnings",
    }
)
# Date-shaped tokens are per-run vocabulary, not subject: 9/37 historical
# daily-held tasks share one batch date token (2026-06-27), so a shared date
# marks same-day filings, never a duplicate subject (#1687).
ROUTE3_DATE_TOKEN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Calibrated 2026-07-17 (plan #1483 §11 D2, n=5 population, max non-dup
# overlap 1) and RE-calibrated 2026-07-25 (plan #1687): on the n=14 live
# population the raw non-dup ceiling grew to 3 (the #1537 x #1686 incident);
# after the ROUTE3_GENERIC_TOKENS + date exclusion it is 2 (#1636 x #1686),
# and the #1140/#1472 true-dup pair keeps 5. Threshold stays 3; the
# title-anchor condition in find_open_daily_held_duplicate carries the
# structural margin (plan #1687 §11 D2/D3).
ROUTE3_MIN_SHARED_TOKENS = 3
# ── #1674 mechanical landed-fix probe ────────────────────────────────────────
# Window matches the compose-time clause-(a') duty + the #1446 advisory window
# (plan #1674 §11 D2); all three measured incidents are in-window.
LANDED_FIX_WINDOW = "7 days ago"
# Calibrated on the incident pairs (plan #1674 §11 D1, measured 2026-07-25 with the
# live tokenizer + #1687 exclusions): true-dup #1652 vs ce11dff560 shares 3
# ({pods, runpod, scope}); the false-positive probes (this fix's own item text vs the
# 4 most recent driver commit subjects) max out at 2 ({driver, filing} vs the #1678
# commit). 3 fires on the motivating incident and passes every measured legitimate
# pair; 2 would false-suppress. Replay tests pin both sides.
LANDED_FIX_MIN_SHARED_TOKENS = 3
LANDED_FIX_GIT_TIMEOUT_S = 10  # same class as SHA_REV_PARSE_TIMEOUT_S (#1467)
LANDED_FIX_MAX_SUSPECTS = 5  # ledger-row size bound; git log order = most recent first
# ── #1711 mechanical closed-sibling probe ────────────────────────────────────
# Reuses task_workflow.recent_closed_workflow_fix_tasks (#1446 helper). Fires
# ALONGSIDE the #1674 commit-subject probe as the belt-and-suspenders backstop
# for the vocabulary-divergent landed-fix class (#1386/#1360 — where the compose-
# time git-log duty + the #1446 filer-side advisory both failed). Window matches
# #1674's LANDED_FIX_WINDOW + the compose-time clause-(a') duty + the #1446
# advisory window default (helper `days=7.0`) — all three same by design; a
# narrower window here would diverge from the compose-time duty this backstop
# supplements (plan #1711 §11 D1).
CLOSED_SIBLING_WINDOW_DAYS = 7.0
CLOSED_SIBLING_MAX_HITS = 5  # ledger-row size bound; helper already sorted DESC by closed_at
# Composite blocking rule (#1735, replacing #1711's PATH-only rule): a hit BLOCKS
# only when it matches BOTH a target-family arm (`target` / `infra-target`) AND a
# title-family arm (`title:*` / `infra-title:*`) whose shared informative tokens
# survive the driver-scoped `CLOSED_SIBLING_TITLE_STOPWORDS` filter below. A
# bare-target hit — the shared hot-file signal alone — is ADVISORY: on the
# 2026-07-26 batch the #1711 PATH-only rule blocked 21/24 items, dominated by
# bare-`target` (7 hits) and `target`+stopword-title matches (`title:main` 9,
# `title:runs` 5, `title:merge` 4, `title:tests`/`title:step`/`title:probe`/
# `title:path`/`title:check` 3 each) on this repo's hot workflow-surface files
# (`.claude/skills/issue/SKILL.md`, `scripts/workflow_lint.py`,
# `.claude/agents/*.md`) — where nearly every new filing shares a target file
# with some recently-closed sibling within the 7-day window. A bare-title hit is
# ADVISORY (unchanged from #1711's original design).
#
# The composite rule INTENTIONALLY SACRIFICES 2 of the 3 historical measured
# true positives named in the L240-246 comment: fact-check on plan v1
# (2026-07-27) established that #1330 vs #1309 shares ZERO informative title
# tokens (DOWNGRADED to advisory), #1386 vs #1360 shares `retry`,`queue-full`
# (STILL BLOCKS under composite), and #1652 vs #1329 shares ZERO informative
# title tokens (DOWNGRADED to advisory). This is the DELIBERATE trade: the
# pre-fix 21/24 false-positive rate on hot workflow-surface files is a higher
# blast-radius cost than the two historical bare-target-only cases (#1330,
# #1652) that now surface as `CLOSED-SIBLING-ADVISORY` stderr prints instead
# of blocking. Regression coverage rests on (a) the #1350/#1329-shape pinned
# test (shares `workload-cmd`, not a stopword — the composite-rule survivor
# class) and (b) the `CLOSED-SIBLING-ADVISORY` stderr channel still surfacing
# bare-target hits for operator eyeballing. If a future duplicate shares zero
# informative title tokens with its motivating candidate, it surfaces as
# ADVISORY (mirroring the pre-#1711 behaviour for that class); the operator
# eyeballs the stderr line and re-runs with `--retry-suspects` if real. See
# plan #1735 §4.1 for the full block-vs-advisory design defense.
_CLOSED_SIBLING_TARGET_ARMS = ("target", "infra-target")
_CLOSED_SIBLING_TITLE_ARMS = ("title", "infra-title")
# Driver-scoped stopword set for the #1735 composite arm's title contribution.
# Applied ONLY inside find_closed_sibling_suspects (via
# _closed_sibling_informative_title_arms) — the shared
# `_WF_FIX_TITLE_STOPWORDS` in `task_workflow.py` is intentionally UNCHANGED,
# because #1674's `find_landed_fix_suspects` and the open/closed sibling
# advisories are calibrated on the current shared set. Tokens here are the
# generic workflow vocabulary attested in task #1735's ORIGINAL 2026-07-26
# `## Evidence` measurement (verified-at-filing top-uniq output) at
# ≥3-as-SOLE-informative-token frequency across the 21 blocked false-positive
# items: `title:main` (9), `title:runs` (5), `title:merge` (4), `title:tests`
# (3), `title:step` (3), `title:probe` (3), `title:path` (3), `title:check`
# (3). Extra plan v1 candidates (`state`, `daily`, `zero`, `repo`, `shared`)
# were DROPPED per plan §8 A13's implementer re-derivation: they are not
# attested at ≥3-count in the ORIGINAL 2026-07-26 evidence, and the calibration
# corpus is the batch that motivated the fix, not a shifted-since replay.
# Precedent: `ROUTE3_GENERIC_TOKENS` is driver-scoped for the same reason.
CLOSED_SIBLING_TITLE_STOPWORDS: frozenset[str] = frozenset(
    {"main", "runs", "step", "merge", "tests", "path", "check", "probe"}
)
# Legacy alias — retained for one release cycle so any external caller/mock
# that predates #1735 still resolves the symbol. New code MUST reference
# `_CLOSED_SIBLING_TARGET_ARMS` directly; new hits are advisory unless the
# composite predicate fires (title-arm hits carry no path signal by construction).
_CLOSED_SIBLING_BLOCKING_ARMS = _CLOSED_SIBLING_TARGET_ARMS
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

# Anchored Provenance fingerprint line: a list item whose FIRST token after the
# bullet is `fingerprint:` + a 12-hex fp (wf_fix_fingerprint shape; the trailing
# (?![0-9a-f]) lookahead rejects a >=13-hex token matching on its first 12 chars
# — the sweep's retired _RECORD_FP_RE convention, deleted in #1680).
# Deliberately does NOT match prose quotes
# ("... its Provenance fingerprint: 44d3..." sits mid-line) — the #1580 fix.
_FP_LINE_RE = re.compile(
    r"(?m)^(?P<indent>\s*)-\s*fingerprint:\s*(?P<fp>[0-9a-f]{12})(?![0-9a-f])(?P<rest>[^\n]*)"
)

# ── #1467 sha-verify backstop constants (WARN-only; never blocks a filing) ─────
HEX_TOKEN_RE = re.compile(r"\b[0-9a-f]{7,40}\b")  # git abbrev floor 7; 40 = full SHA-1
HAS_HEX_LETTER_RE = re.compile(r"[a-f]")  # all-digit tokens are dates/ids — skip
# Known non-commit hex classes + our own advisory lines: lines matching this are
# never scanned (fingerprints/wf-fix-fp tags are 12-hex by construction, #1173).
SHA_EXCLUDE_LINE_RE = re.compile(r"fingerprint:|wf-fix-fp:|drift_hash|sha-verify")
# #1808: any `fingerprint: <12hex>` label in the body (anchored Provenance bullet OR
# prose, incl. the #1580 reconcile line's "supersedes body-carried fingerprint: <old>")
# declares that token fp-class — exempt it from the sha walk EVERYWHERE in the body,
# not only on its own (already line-excluded) label line.
FP_LABEL_RE = re.compile(r"fingerprint:\s*`?([0-9a-f]{12})(?![0-9a-f])")
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


def ensure_wf_fix_provenance(text: str, target: str, fp: str) -> tuple[str, list[str]]:
    """Idempotently ensure + RECONCILE the wf-fix Provenance lines (#1173, #1580).

    Returns (new_text, actions); actions is a subset of ["target", "fp-inject",
    "fp-reconcile"]; empty list == unchanged (truthiness-compatible with the
    pre-#1580 bool). The ``- workflow_fix_target: <target>`` line is the DURABLE
    signal task_workflow.is_workflow_fix_session() reads (the env-var leg is lost
    on a watcher crash-recovery respawn).

    The wf-fix-fp TAG (manifest-computed at the process_item fp) is AUTHORITATIVE
    for the ``(target_file, fingerprint)`` dedup key (workflow-fix-on-bug.md
    § Dedup). Body-line policy:

    - no anchored ``- fingerprint: <12hex>`` line -> inject ``- fingerprint: {fp}``.
      Detection is _FP_LINE_RE-anchored, NOT substring: a prose mention of
      ``fingerprint:`` no longer suppresses injection (#1580's own body).
    - an anchored line already carrying ``fp`` -> no-op (idempotent, incl. re-runs
      over a previously reconciled body; covers the mixed case where another
      anchored line differs — the tag value is present anchored, the key coherent).
    - anchored line(s) all carrying a DIFFERENT fp -> each rewritten in place to
      ``- fingerprint: {fp} (tag-authoritative; supersedes body-carried
      fingerprint: {old})``. The old value survives as the substring
      ``fingerprint: {old}`` so sweep _fp_tag_scan and is_open_workflow_fix_task
      (both substring-OR) still suppress re-raises keyed to it; the ONLY anchored
      value is the tag's, so tag and body never disagree. Any trailing text on
      the mismatched line is dropped (the old fp is what is load-bearing).

    ``workflow_fix_target`` handling is unchanged (substring gate — the
    recursion-guard predicate ``is_workflow_fix_session`` is itself
    substring-based, so a prose mention already satisfies it).
    Substring contracts (do not reformat): ``workflow_fix_target: {target}``,
    ``fingerprint: {fp}``, and on a reconciled line ``fingerprint: {old}`` —
    single space after each colon.
    """
    actions: list[str] = []
    inject: list[str] = []
    if WF_FIX_TARGET_KEY not in text:
        inject.append(f"- workflow_fix_target: {target}")
        actions.append("target")
    matches = list(_FP_LINE_RE.finditer(text))
    if not matches:
        inject.append(f"- fingerprint: {fp}")
        actions.append("fp-inject")
    elif not any(m.group("fp") == fp for m in matches):

        def _reconcile(m: re.Match) -> str:
            return (
                f"{m.group('indent')}- fingerprint: {fp} "
                f"(tag-authoritative; supersedes body-carried fingerprint: {m.group('fp')})"
            )

        text = _FP_LINE_RE.sub(_reconcile, text)
        actions.append("fp-reconcile")
    if inject:
        text = _insert_under_provenance(text, "\n".join(inject))
    return text, actions


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


def scan_unresolvable_shas(
    text: str, root: Path, *, exempt: frozenset[str] | set[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """Scan body text for SHA-shaped hex tokens that do not resolve as commits (#1467).

    Returns ``(commit_context_offenders, other_nonresolving)`` — deduped, first-seen
    order. Per line: SHA_EXCLUDE_LINE_RE lines (fingerprints, wf-fix-fp tags, drift
    hashes, our own sha-verify advisories) are skipped entirely; HEX_TOKEN_RE matches
    with no [a-f] letter (dates, numeric ids) are skipped. Tokens in ``exempt``
    (#1808: the item's own wf-fix fp + ``fingerprint:``-labeled values, via
    _fp_exempt_tokens) are skipped everywhere, not only on their own label lines.
    A token is tier 1 when ANY
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
            if tok in exempt:
                continue
            commit_ctx[tok] = commit_ctx.get(tok, False) or in_ctx
    tier1: list[str] = []
    tier2: list[str] = []
    for tok, in_ctx in commit_ctx.items():
        if _sha_resolves(tok, root):
            continue
        (tier1 if in_ctx else tier2).append(tok)
    return tier1, tier2


def _fp_exempt_tokens(text: str, fp: str | None) -> frozenset[str]:
    """Token-level exempt set for the #1467 walk (#1808): the item's own computed
    wf-fix fp plus every ``fingerprint:``-labeled 12-hex token in the body."""
    toks = set(FP_LABEL_RE.findall(text))
    if fp:
        toks.add(fp)
    return frozenset(toks)


def _check_body_shas(item: dict, dirpath: Path, root: Path, *, fp: str | None = None) -> list[str]:
    """WARN-only #1467 backstop: annotate non-resolving commit-context hex tokens.

    ``fp`` (#1808) is the item's own wf-fix fingerprint: it and every
    ``fingerprint:``-labeled 12-hex token in the body are token-exempt from the
    scan (an fp is not a commit; the incident shape quoted the own fp bare on a
    commit-context line and got annotated).

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
        tier1, tier2 = scan_unresolvable_shas(text, root, exempt=_fp_exempt_tokens(text, fp))
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
    slug: str,
    tid: int,
    fp: str,
    item: dict,
    tail: str,
    sha_warnings: list[str],
    advisories: list[str],
    held: dict | None,
) -> dict:
    """Compose the terminal ``filed`` ledger row.

    Gains ``"sha_warnings": [tokens]`` only when the #1467 backstop annotated
    non-resolving commit-context tokens, ``"advisories": [lines]`` only when
    the filer's #1399/#1502 sibling-advisory stderr was non-empty (#1529), and
    ``"held_dispatch"/"held_with"/"shared_target"`` only when the #1678
    same-target hold filed this item with ``--no-dispatch`` — all conditional
    (rows are free-form dicts — schema-safe; no-hold rows are byte-identical).
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
    if advisories:
        row["advisories"] = advisories
    if held:
        row["held_dispatch"] = True
        row["held_with"] = held["with"]
        row["shared_target"] = held["shared"]
    return row


def _dry_run_sha_note(item: dict, dirpath: Path, root: Path, *, fp: str | None = None) -> str:
    """The dry-run mirror of _check_body_shas (#1467): report counts, mutate nothing.

    Returns a suffix for the dry-run FILE line — empty when the scan is clean;
    fail-open (a note, never a raise) on git/read OSError, a non-UTF-8 body's
    UnicodeDecodeError, or a hung-git TimeoutExpired, mirroring the real path.
    ``fp`` (#1808) mirrors _check_body_shas' own-fp exemption.
    """
    try:
        text = _resolve_body_path(item, dirpath).read_text(encoding="utf-8")
        # #1808: the fp= arm is load-bearing for dry-run parity — no anchored fp line
        # is injected at dry-run time, so only the param can exempt the item's own fp.
        tier1, tier2 = scan_unresolvable_shas(text, root, exempt=_fp_exempt_tokens(text, fp))
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
    """First NON-terminal task body.md carrying this fp (route-2 dedup).

    Keys on the fingerprint TAG (``wf-fix-fp:<fp>``) OR an ANCHORED
    ``- fingerprint: <fp>`` Provenance line (#1580 — aligned with the OR the two
    sibling consumers ``_fp_tag_scan`` / ``is_open_workflow_fix_task`` already
    implement); anchored-bullet, not bare substring, so a prose quote of another
    task's fp (e.g. #1580's own body quoting #1570's ``44d3a4598f5c``) cannot
    false-suppress a genuine re-raise. Residual one-way surface, accepted: a
    reconciled line's parenthetical old fp is NOT anchored-matched here (the
    substring-OR consumers, which carry kind/target gates, cover it), and a
    stray anchored fp line in a NON-wf-fix body would match — the shape
    ``_warn_stray_wf_fix_provenance`` already flags.

    Same predicate family as the proven ad-hoc driver: a scan over
    non-``completed``/``archived`` statuses. Deliberately COARSER than
    ``task_workflow.is_open_workflow_fix_task`` (which since #1180 DOES see
    ``daily-fix:`` titles via ``WF_FIX_TITLE_PREFIXES``): any title, any kind,
    no ``workflow_fix_target:`` requirement, filesystem-only (no REGISTRY read)
    — so a same-fp task filed by EITHER channel blocks a daily re-file even
    when its registry row is malformed. Kept (not delegated) for that coarser
    grain and for the ``tasks_root`` injection the test fixtures use. Called
    only for ``_wf_fix_enabled`` items (#1228) — a ``wf_fix: false`` filing
    never carries the fp tag, so its participation would be one-way.
    """
    needle = f"wf-fix-fp:{fp}"
    line_re = re.compile(rf"(?m)^\s*-\s*fingerprint:\s*{re.escape(fp)}(?![0-9a-f])")
    for body in sorted(tasks_root.glob("*/*/body.md")):
        status = body.parent.parent.name
        if status in ("completed", "archived"):
            continue
        try:
            text = body.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle in text or line_re.search(text):
            return body
    return None


def _route3_informative(tokens: set[str]) -> set[str]:
    """Route-3 informative subset: boilerplate, generic-vocab + date tokens removed (#1687)."""
    return {
        t
        for t in tokens
        if t not in ROUTE3_BOILERPLATE_TOKENS
        and t not in ROUTE3_GENERIC_TOKENS
        and not ROUTE3_DATE_TOKEN_RE.match(t)
    }


def _route3_item_tokens(title: str, bug: str) -> set[str]:
    """Informative tokens of a route-3 item: title (daily-held: stripped) + bug text."""
    t = (title or "").removeprefix(ROUTE3_TITLE_PREFIX)
    return _route3_informative(informative_title_tokens(t) | informative_title_tokens(bug or ""))


def _route3_title_tokens(title: str) -> set[str]:
    """Subject tokens of the TITLE alone — the #1687 anchor set (same exclusions)."""
    t = (title or "").removeprefix(ROUTE3_TITLE_PREFIX)
    return _route3_informative(informative_title_tokens(t))


def find_open_daily_held_duplicate(tasks_root: Path, title: str, bug: str) -> dict | None:
    """Best OPEN daily-held task sharing >= ROUTE3_MIN_SHARED_TOKENS tokens (#1483).

    Population: tasks under ROUTE3_OPEN_STATUSES whose frontmatter tags carry
    DAILY_HELD_TAG (tag-only key; kind not checked). Task-side tokens come from
    frontmatter title + origin_prompt with the /daily template prefix stripped
    (origin_prompt IS the filing-time bug[:400] by _filer_cmd construction) —
    frontmatter-only reads, never the body (body boilerplate would inflate
    overlap). Two #1687 conditions gate a hit beyond the shared-token
    threshold: generic-vocab + date tokens never count toward overlap
    (_route3_informative), and the two TITLES must share >= 1 post-exclusion
    subject token (the anchor) — generic prose overlap alone never suppresses.
    Returns {"id", "title", "shared", "anchor"} for the max-overlap hit
    (tie: lowest id), else None. Kept in the driver (not task_workflow) for the
    tasks_root injection the test fixtures use, same as find_open_fp_duplicate.
    Callers wrap in try/except: the scan is FAIL-OPEN by contract.
    """
    cand = _route3_item_tokens(title, bug)
    if not cand:
        return None
    cand_title = _route3_title_tokens(title)
    if not cand_title:
        return None  # no subject-bearing title tokens -> nothing can anchor; fail toward filing
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
        anchor = cand_title & _route3_title_tokens(str(fm.get("title") or ""))
        if not anchor:
            # >=3 shared tokens confined to bug/origin prose: generic-prose
            # overlap without title-subject agreement never suppresses (#1687).
            continue
        tid = int(body.parent.name)
        if best is None or (len(shared), -tid) > (len(best["shared"]), -best["id"]):
            best = {
                "id": tid,
                "title": str(fm.get("title") or ""),
                "shared": sorted(shared),
                "anchor": sorted(anchor),
            }
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
                "anchor": dup3["anchor"],
                "fp": fp,
                "route": 3,
            },
        )
    print(
        f"ALREADY-TRACKED {slug} -> #{dup3['id']}"
        f" (shared: {','.join(dup3['shared'])}; anchor: {','.join(dup3['anchor'])})"
    )
    return "skip" if dry_run else "already-tracked"


def _landed_fix_item_tokens(item: dict) -> set[str]:
    """Informative tokens of the item's title+bug+change (#1674).

    Title prefixes stripped (WF_FIX_TITLE_PREFIXES members + ROUTE3_TITLE_PREFIX);
    _route3_informative exclusions applied (generic workflow vocabulary + dates
    never count toward overlap — the #1687 calibration transfers, plan #1674 §11 D4).
    """
    t = item["title"] or ""
    for pfx in (*WF_FIX_TITLE_PREFIXES, ROUTE3_TITLE_PREFIX):
        t = t.removeprefix(pfx)
    return _route3_informative(
        informative_title_tokens(t)
        | informative_title_tokens(item.get("bug") or "")
        | informative_title_tokens(item.get("change") or "")
    )


def find_landed_fix_suspects(item: dict, root: Path) -> list[dict]:
    """Recent commits on the item's OWN target path(s) whose SUBJECT shares
    >= LANDED_FIX_MIN_SHARED_TOKENS informative tokens with the item text (#1674).

    Subject-only by design — subject+body matching false-positives on hot files
    (plan #1674 §11 D3). Path-scoped: ``git log -- <paths>`` with the item's own
    _target_tokens (never a repo-wide subject scan). Returns up to
    LANDED_FIX_MAX_SUSPECTS dicts {"sha", "subject", "shared"} in git log order
    (most recent first). Raises on git failure — _landed_fix_or_none fail-opens.
    """
    paths = sorted(_target_tokens(item["target"]))
    if not paths:
        return []
    item_toks = _landed_fix_item_tokens(item)
    if not item_toks:
        return []
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "log",
            f"--since={LANDED_FIX_WINDOW}",
            "--format=%h%x09%s",
            "--",
            *paths,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=LANDED_FIX_GIT_TIMEOUT_S,
    )
    suspects: list[dict] = []
    for line in proc.stdout.splitlines():
        sha, _, subject = line.partition("\t")
        if not subject:
            continue
        shared = item_toks & _route3_informative(informative_title_tokens(subject))
        if len(shared) >= LANDED_FIX_MIN_SHARED_TOKENS:
            suspects.append({"sha": sha, "subject": subject, "shared": sorted(shared)})
            if len(suspects) >= LANDED_FIX_MAX_SUSPECTS:
                break
    return suspects


def _landed_fix_or_none(item: dict, root: Path) -> list[dict]:
    """Fail-open wrapper (#1674): a git/IO error WARNs loudly and files as today.

    Enumerated classes only (OSError, CalledProcessError, TimeoutExpired,
    UnicodeDecodeError) — deliberately NARROWER than _route3_dup_or_none's broad
    catch: the probe's error surface is one git subprocess (the fail-open mandate
    covers "git itself errors"); a driver bug still fails loud per fail-fast
    (the #1467 _sha_resolves tuple lacks CalledProcessError because it reads rc
    without check=True; this probe's check=True requires the 4-class tuple —
    plan #1674 §11 D6).
    """
    try:
        return find_landed_fix_suspects(item, root)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
    ) as e:
        print(
            f"WARNING {item['slug']}: landed-fix probe skipped"
            f" ({e.__class__.__name__}: {e}) — fail-open, filing proceeds; the"
            " compose-time git-log duty still applies (#1674)",
            file=sys.stderr,
        )
        return []


def _has_landed_fix_suspect_row(ledger: list[dict], slug: str) -> bool:
    """True when the slug already carries a landed-fix-suspect ledger row (#1674)."""
    return any(r.get("slug") == slug and r.get("outcome") == "landed-fix-suspect" for r in ledger)


def _landed_fix_suspect_outcome(
    item: dict, root: Path, *, dirpath: Path, fp: str, dry_run: bool
) -> str | None:
    """Landed-fix probe outcome for process_item, or None (mirror of
    _route3_already_tracked's contract, #1674). Routes 2 AND 3 (plan #1674 §11 D5).

    No-suspect scans return None (caller proceeds to file). On a hit the
    LANDED-FIX-SUSPECT line prints either way; the real path appends the
    terminal ``landed-fix-suspect`` ledger row and returns 'landed-fix-suspect';
    dry-run stays read-only by construction (no ledger write) and returns 'skip'.
    """
    suspects = _landed_fix_or_none(item, root)
    if not suspects:
        return None
    slug = item["slug"]
    if not dry_run:
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "landed-fix-suspect",
                "suspects": suspects,
                "threshold": LANDED_FIX_MIN_SHARED_TOKENS,
                "window": LANDED_FIX_WINDOW,
                "paths": sorted(_target_tokens(item["target"])),
                "fp": fp,
                "route": item["route"],
            },
        )
    top = suspects[0]
    more = f"; +{len(suspects) - 1} more" if len(suspects) > 1 else ""
    print(
        f'LANDED-FIX-SUSPECT {slug} -> {top["sha"]} "{top["subject"]}"'
        f" (shared: {','.join(top['shared'])}{more}) — NOT filing; eyeball the"
        " commit(s), re-run with --retry-suspects to file anyway (#1674)"
    )
    return "skip" if dry_run else "landed-fix-suspect"


def _closed_sibling_informative_title_arms(h: dict) -> set[str]:
    """Title-family arm PREFIXES of ``h`` whose shared informative tokens survive
    ``CLOSED_SIBLING_TITLE_STOPWORDS`` (#1735).

    Reads the arm labels ``title:<t1>,<t2>,...`` / ``infra-title:<t1>,...``
    from ``h["matched"]``: splits each on ``:`` to isolate the arm prefix,
    then splits the token comma-list and filters out driver-scoped
    stopwords. Returns the surviving arm prefixes as a set — empty when every
    title-arm's shared tokens were exclusively stopwords. Path-family arms
    (``target`` / ``infra-target``) never appear in the return (they carry no
    ``:`` payload); the composite predicate handles them separately via
    :data:`_CLOSED_SIBLING_TARGET_ARMS`.

    The arm labels themselves are the source of truth for which tokens
    overlapped (parsed at helper time by ``_wf_fix_sibling_arms``): no
    re-tokenization of the closed task titles is needed here.
    """
    surviving: set[str] = set()
    for m in h.get("matched", []):
        if ":" not in m:
            continue
        arm_prefix, token_list = m.split(":", 1)
        if arm_prefix not in _CLOSED_SIBLING_TITLE_ARMS:
            continue
        tokens = {t.strip() for t in token_list.split(",") if t.strip()}
        if tokens - CLOSED_SIBLING_TITLE_STOPWORDS:
            surviving.add(arm_prefix)
    return surviving


def find_closed_sibling_suspects(item: dict) -> tuple[list[dict], list[dict]]:
    """Recently-closed infra siblings overlapping the item's target/title (#1711, #1735).

    Returns ``(blocking_hits, advisory_hits)``: each is a list of dicts filtered
    from :func:`task_workflow.recent_closed_workflow_fix_tasks` by the #1735
    composite arm predicate. A hit is BLOCKING iff it matches BOTH a
    target-family arm (:data:`_CLOSED_SIBLING_TARGET_ARMS`) AND a title-family
    arm whose shared informative tokens survive
    :data:`CLOSED_SIBLING_TITLE_STOPWORDS`. Bare-target hits (target arm only,
    no informative title overlap) and bare-title hits (title arm only, no path
    overlap) BOTH partition into ``advisory``.

    Reuses the #1446 helper verbatim — its own error surface (per-task read
    failures skip; empty inputs return ``[]``) is unchanged; the CALLER wraps
    this in :func:`_closed_sibling_or_none` for the broad fail-open (the helper
    reads registry + N body files, broader error space than a single ``git log``
    subprocess). Both returned lists are capped at
    :data:`CLOSED_SIBLING_MAX_HITS`; the helper already sorts DESC by
    ``closed_at``, so the cap keeps the most-recent hits (parity with #1674's
    ``LANDED_FIX_MAX_SUSPECTS``).
    """
    from explore_persona_space.task_workflow import recent_closed_workflow_fix_tasks

    hits = recent_closed_workflow_fix_tasks(
        item["target"], item["title"], days=CLOSED_SIBLING_WINDOW_DAYS
    )
    blocking: list[dict] = []
    advisory: list[dict] = []
    for h in hits:
        matched_arms = {m.split(":", 1)[0] for m in h.get("matched", [])}
        has_target = bool(matched_arms & set(_CLOSED_SIBLING_TARGET_ARMS))
        has_informative_title = bool(_closed_sibling_informative_title_arms(h))
        if has_target and has_informative_title:
            blocking.append(h)
        else:
            advisory.append(h)
    return blocking[:CLOSED_SIBLING_MAX_HITS], advisory[:CLOSED_SIBLING_MAX_HITS]


def _closed_sibling_or_none(item: dict) -> tuple[list[dict], list[dict]]:
    """Fail-open wrapper (#1711): any error WARNs loudly and files as today.

    Broad ``except Exception`` catch is DELIBERATE (plan #1711 §4.7): the
    helper reads the whole registry + N task body files (YAML + markdown
    parse); the error surface is broader than a single ``git log`` subprocess
    and than #1674's narrow enumerated tuple. Same rationale as
    :func:`_route3_dup_or_none` (#1483) which uses the identical broad catch
    for the identical reason ("broad Exception catch is DELIBERATE, mandated
    by the task's fail-open constraint … the token/YAML surface has a broader
    error space than the enumerable I/O classes there"). A held item is never
    lost to a scan bug; the loud stderr WARNING is the fail-loud channel.
    """
    try:
        return find_closed_sibling_suspects(item)
    except Exception as e:  # deliberate fail-open, see docstring
        print(
            f"WARNING {item['slug']}: closed-sibling probe skipped"
            f" ({e.__class__.__name__}: {e}) — fail-open, filing proceeds; the"
            " compose-time closed-sibling eyeball + #1674 commit-subject probe"
            " still apply (#1711)",
            file=sys.stderr,
        )
        return [], []


def _closed_sibling_outcome(item: dict, *, dirpath: Path, fp: str, dry_run: bool) -> str | None:
    """Closed-sibling probe outcome for process_item, or None (mirror of
    :func:`_landed_fix_suspect_outcome`'s contract, #1711 / #1735). Routes 2
    AND 3.

    - No hits at all → returns ``None`` (caller proceeds to file).
    - Advisory-only hits (composite predicate NOT satisfied — bare-target,
      bare-title, or target + only-stopword-title arms per plan #1735 §4.1)
      → prints ONE stderr ``CLOSED-SIBLING-ADVISORY`` line per hit naming
      the sibling, returns ``None`` (caller proceeds to file).
    - Blocking hits (composite predicate satisfied — target/infra-target
      arm AND title/infra-title arm with ≥1 non-stopword informative token)
      → prints ``CLOSED-SIBLING-SUSPECT`` on STDOUT (operator-facing,
      mirroring #1674's ``LANDED-FIX-SUSPECT`` stdout shape so a fleet
      eyeball grep catches both); the real path appends a terminal
      ``landed-fix-suspect`` ledger row (with ``suspects[].kind ==
      "closed-sibling"`` — Option A per plan §4.3; distinguishable from
      #1674 rows by presence of ``id`` vs ``sha``) and returns
      ``'landed-fix-suspect'``; dry-run stays read-only by construction
      (no ledger write) and returns ``'skip'``.

    Advisory lines print on STDERR (informational, WARNING-adjacent), the
    intentional stdout/stderr split — plan #1711 concern-fold from
    Methodology critic.
    """
    blocking, advisory = _closed_sibling_or_none(item)
    slug = item["slug"]
    # Print advisory-only hits first (never suppresses; visible on both dry-run
    # and real paths). Stderr channel — informational, non-blocking.
    for h in advisory:
        arms = ",".join(h["matched"])
        print(
            f"CLOSED-SIBLING-ADVISORY {slug} -> #{h['id']}"
            f' "{h["title"]}" (matched: {arms}) — NOT blocking'
            " (composite predicate not satisfied, #1735)",
            file=sys.stderr,
        )
    if not blocking:
        return None
    if not dry_run:
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "landed-fix-suspect",
                "suspects": [
                    {
                        "kind": "closed-sibling",
                        "id": h["id"],
                        "title": h["title"],
                        "status": h["status"],
                        "target": h["target"],
                        "closed_at": h["closed_at"],
                        "matched": h["matched"],
                    }
                    for h in blocking
                ],
                "threshold": None,
                "window": f"{CLOSED_SIBLING_WINDOW_DAYS} days",
                "paths": sorted(_target_tokens(item["target"])),
                "fp": fp,
                "route": item["route"],
            },
        )
    top = blocking[0]
    more = f"; +{len(blocking) - 1} more" if len(blocking) > 1 else ""
    # Stdout channel — operator-facing, same as #1674's LANDED-FIX-SUSPECT.
    print(
        f"CLOSED-SIBLING-SUSPECT {slug} -> #{top['id']}"
        f' "{top["title"]}" (matched: {",".join(top["matched"])}{more}) — NOT filing;'
        " eyeball the sibling, re-run with --retry-suspects to file anyway (#1711)"
    )
    return "skip" if dry_run else "landed-fix-suspect"


def _landed_fix_probes_outcome(
    item: dict,
    root: Path,
    *,
    dirpath: Path,
    fp: str,
    suspect_eyeballed: bool,
    dry_run: bool,
) -> str | None:
    """Run the two landed-fix probes in order (#1674 first, #1711 second).

    Extracted from process_item + _dry_run_item for the C901 budget (#1699
    ruff-policy pin). The suppression contract is unchanged:

    - ``suspect_eyeballed`` short-circuits BOTH probes — the filer already
      eyeballed this slug's prior suspect commit(s) and/or closed sibling(s).
    - #1674 (commit-subject) runs FIRST; a non-None result wins with its
      row shape (Option A source-compat — every existing #1674 test stays
      byte-identical).
    - #1711 (closed-sibling) runs SECOND, ONLY when #1674 did not suppress;
      belt-and-suspenders for the vocabulary-divergent landed-fix class
      (#1386/#1360 — no #1674 arm coverage by construction).
    - Both probes write the SAME terminal ledger outcome
      (``landed-fix-suspect``) so ``_slug_state`` treats them identically
      and one ``--retry-suspects`` re-drives both.

    Returns the terminal outcome ('landed-fix-suspect' or 'skip' under
    dry_run) or None (caller proceeds).
    """
    if suspect_eyeballed:
        return None
    outcome_lf = _landed_fix_suspect_outcome(item, root, dirpath=dirpath, fp=fp, dry_run=dry_run)
    if outcome_lf is not None:
        return outcome_lf
    return _closed_sibling_outcome(item, dirpath=dirpath, fp=fp, dry_run=dry_run)


def _dry_run_inject_note(item: dict, dirpath: Path, fp: str) -> str:
    """Read-only injection/reconcile-intent probe for the dry-run FILE line: write-free.

    Returns the ' [will inject workflow_fix_target provenance]' suffix when the real
    path would inject the wf-fix Provenance block (#1173), plus/or the
    ' [will reconcile body fingerprint -> tag value]' suffix when it would rewrite a
    mismatched anchored fp line (#1580), else ''. No exists() guard — a
    missing/unreadable body fails LOUD here, the same fail-fast contract
    load_and_validate_manifest enforces up front. Non-wf-fix items get the
    stray-provenance WARN instead.
    """
    if not _wf_fix_enabled(item):
        _warn_stray_wf_fix_provenance(item, dirpath)
        return ""
    body_text = _resolve_body_path(item, dirpath).read_text(encoding="utf-8")
    _new, actions = ensure_wf_fix_provenance(body_text, item["target"], fp)
    note = ""
    if "target" in actions or "fp-inject" in actions:
        note += " [will inject workflow_fix_target provenance]"  # pinned string, unchanged
    if "fp-reconcile" in actions:
        note += " [will reconcile body fingerprint -> tag value]"
    return note


def _target_tokens(target: str) -> set[str]:
    """Normalized target tokens of a manifest ``target`` (#1678): split on commas,
    whitespace-stripped, leading './' removed (removeprefix, NOT lstrip), empties
    dropped. A glob string is a LITERAL token — two identical globs overlap; a glob
    vs a concrete path under it does NOT (accepted false negative: the Step 10d
    recovery shapes remain the backstop)."""
    return {t.strip().removeprefix("./") for t in str(target).split(",") if t.strip()}


def compute_target_holds(manifest: list[dict]) -> dict[str, dict]:
    """slug -> {"with": <earliest overlapping route-2 slug>, "shared": [tokens]}
    for every route-2 item that must file WITHOUT dispatch (#1678 same-target hold).

    Positional + deterministic over the FULL manifest (never the --start/--end
    slice, so multi-slice and resume invocations compute the identical map):
    a route-2 item is HELD iff any EARLIER route-2 item shares >=1 normalized
    target token; attribution goes to the EARLIEST such sibling. Route-3 items
    neither hold nor are held (they already file --no-dispatch, so they are not
    a contention source). Deliberately ledger-state-independent: holding a
    sibling whose earlier group-head deduped/errored is harmless (the watcher
    dispatches it ~10 min later) and buys invocation-order determinism. The
    stagger is HEAD-vs-HELD only and probabilistic — held siblings of one group
    dispatch in the same sweep pass, 60 s apart (#1059); see the module
    docstring — never a merge-ordering guarantee."""
    holds: dict[str, dict] = {}
    seen: list[tuple[str, set[str]]] = []
    for item in manifest:
        if item["route"] != 2:
            continue
        toks = _target_tokens(item["target"])
        for earlier_slug, earlier_toks in seen:
            shared = toks & earlier_toks
            if shared:
                holds[item["slug"]] = {"with": earlier_slug, "shared": sorted(shared)}
                break
        seen.append((item["slug"], toks))
    return holds


def _filer_cmd(
    filer_prefix: list[str],
    item: dict,
    body_path: Path,
    date: str,
    fp: str,
    *,
    hold_dispatch: bool = False,
) -> list[str]:
    """Compose the file_infra_task.py argv for one manifest item (per-route tags).

    ``hold_dispatch=True`` (route 2 only; #1678 same-target hold) appends
    ``--no-dispatch`` so the task is filed but no session spawns — the watcher
    ``proposed_infra_sweep`` dispatches it later (the route-3 argv precedent;
    ``FILED_ID_RE`` already parses the filer's ``--no-dispatch`` stdout shape).
    """
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
        if hold_dispatch:
            cmd += ["--no-dispatch"]  # #1678 same-target hold; watcher sweep dispatches
    else:
        cmd += ["--tag", "daily-held", "--tag", "needs-human", "--no-dispatch"]
    return cmd


def parse_filed_id(stdout: str, stderr: str) -> int | None:
    """Anchored id parse over stdout then stderr (a stray ``#N`` in a warning never wins)."""
    m = FILED_ID_RE.search(stdout or "")
    if m is None:
        m = FILED_ID_RE.search(stderr or "")
    return int(m.group(1)) if m else None


# #1529: the filer's sibling-advisory stderr shapes (file_infra_task.py — the
# `_advise_recent_closed_wf_fix_siblings` / `_advise_open_wf_fix_siblings` headers,
# their `  #<id>` rows + `  ... and N more` overflow lines, and the fail-soft
# `advisory leg failed` one-liners).
_ADVISORY_HEADER_SUBSTR = "file_infra_task: ADVISORY"
_ADVISORY_FAILSOFT_SUBSTR = "advisory leg failed"
_ADVISORY_MAX_FWD_LINES = 40  # 2 headers + 2x10 rows + 2 overflow + fail-soft << 40
_ADVISORY_MAX_FWD_CHARS = 4000


def extract_filer_advisories(stderr: str) -> list[str]:
    """Extract the #1399/#1502 sibling-advisory block lines from the FILER's stderr (#1529).

    A block = one ``file_infra_task: ADVISORY — ...`` header plus its immediately
    following row/overflow lines (``  #<id> ...`` / ``  ... and N more ...`` — the
    only 2-space-indented shapes the advisory legs emit; the adjacent forwarded
    ``  [task.py stderr] ...`` lines start ``  [`` and are excluded). Standalone
    ``... advisory leg failed ...`` fail-soft one-liners are captured too (they tell
    the consumer the scan did NOT run — absence of advisories is not evidence of
    no siblings). Defensively capped; the cap appends an explicit marker line so
    nothing is silently dropped.
    """
    lines: list[str] = []
    in_block = False
    for line in (stderr or "").splitlines():
        if _ADVISORY_HEADER_SUBSTR in line:
            lines.append(line.rstrip())
            in_block = True
        elif _ADVISORY_FAILSOFT_SUBSTR in line:
            lines.append(line.rstrip())
            in_block = False
        elif in_block and (line.startswith("  #") or line.startswith("  ...")):
            lines.append(line.rstrip())
        else:
            in_block = False
    if (
        len(lines) > _ADVISORY_MAX_FWD_LINES
        or sum(len(ln) for ln in lines) > _ADVISORY_MAX_FWD_CHARS
    ):
        kept: list[str] = []
        total = 0
        for ln in lines[:_ADVISORY_MAX_FWD_LINES]:
            if total + len(ln) > _ADVISORY_MAX_FWD_CHARS:
                break
            kept.append(ln)
            total += len(ln)
        kept.append(f"  ... advisory forward capped ({len(lines)} lines total, #1529)")
        return kept
    return lines


def _print_advisory_forward(slug: str, tid: int, advisories: list[str]) -> None:
    """Re-print the captured filer advisory block on the DRIVER's stderr (#1529).

    Verbatim, after the ``FILED`` stdout line, with an attributing lead line so a
    multi-item manifest keeps blocks attributable (items process serially). No-op
    when no advisory lines were captured — the no-advisory output stays byte-
    identical to the pre-#1529 driver.
    """
    if not advisories:
        return
    print(
        f"ADVISORY {slug} -> #{tid}: filer advisory output below (#1399/#1502 "
        "sibling scan; if a listed sibling already covers this bug, verify then "
        "archive the just-filed task and stop its spawned session — #1529)",
        file=sys.stderr,
    )
    for ln in advisories:
        print(ln, file=sys.stderr)


def _slug_state(
    ledger: list[dict], slug: str, retry_errors: bool, retry_suspects: bool = False
) -> str:
    """Classify a slug against the ledger:
    'terminal' | 'retry-error' | 'retry-suspect' | 'in-flight' | 'fresh'.

    ``landed-fix-suspect`` gets the ERROR-style two-state treatment (#1674):
    terminal without ``--retry-suspects``, re-drivable with it. The ERROR branch
    is checked FIRST, so a slug carrying BOTH an ERROR row and a suspect row
    needs both ``--retry-errors --retry-suspects`` to re-drive — with
    ``--retry-errors`` alone the state is 'retry-error', the probe re-runs, and
    it may append another suspect row per pass (benign accumulation; the
    idempotence corollary of the two-state design).
    """
    rows = [r for r in ledger if r.get("slug") == slug]
    outcomes = {r.get("outcome") for r in rows}
    if outcomes & TERMINAL_OUTCOMES:
        return "terminal"
    if "ERROR" in outcomes:
        return "retry-error" if retry_errors else "terminal"
    if "landed-fix-suspect" in outcomes:  # NEW (#1674)
        return "retry-suspect" if retry_suspects else "terminal"
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


def _apply_wf_fix_provenance(item: dict, body_path: Path, slug: str, fp: str) -> None:
    """Normalize the body's wf-fix Provenance lines in place (#1173, #1580).

    Injects missing lines / reconciles a mismatched anchored fingerprint line via
    ensure_wf_fix_provenance (temp+rename write, same pattern as load_ledger) and
    prints the INJECTED / RECONCILED breadcrumbs the tests pin. No-op when the
    body is already coherent. Extracted from process_item (C901 budget).
    """
    text = body_path.read_text(encoding="utf-8")
    new_text, actions = ensure_wf_fix_provenance(text, item["target"], fp)
    if not actions:
        return
    tmp = body_path.with_suffix(".md.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, body_path)
    if "target" in actions or "fp-inject" in actions:
        print(f"INJECTED {slug}: workflow_fix_target provenance (#1173 recursion-guard signal)")
    if "fp-reconcile" in actions:
        print(
            f"RECONCILED {slug}: body fingerprint -> tag value {fp} "
            "(#1580; body-carried fp preserved as labeled substring)"
        )


def _dry_run_item(
    item: dict,
    *,
    dirpath: Path,
    tasks_root: Path,
    root: Path,
    date: str,
    fp: str,
    state: str,
    hold: dict | None,
    suspect_eyeballed: bool,
) -> str:
    """The --dry-run leg of process_item, extracted for the C901 budget (#1674).

    Read-only by contract: prints the planned action (ALREADY-TRACKED / DEDUP /
    LANDED-FIX-SUSPECT / FILE) and never writes the ledger or a body. Always
    returns 'skip' (or the read-only outcome the dedup-family mirrors return).
    """
    slug = item["slug"]
    # #1483 dry-run mirror of the route-3 overlap dedup (read-only by construction).
    outcome3 = _route3_already_tracked(item, tasks_root, dirpath=dirpath, fp=fp, dry_run=True)
    if outcome3 is not None:
        return outcome3
    if _wf_fix_enabled(item) and find_open_fp_duplicate(tasks_root, fp) is not None:
        print(f"DEDUP {slug} -> wf-fix-fp:{fp}")
        return "skip"
    # #1674 + #1711 dry-run mirror of the landed-fix probes (read-only by
    # construction — no ledger write); same relative position + probe order as
    # the real path.
    outcome_probes = _landed_fix_probes_outcome(
        item, root, dirpath=dirpath, fp=fp, suspect_eyeballed=suspect_eyeballed, dry_run=True
    )
    if outcome_probes is not None:
        return outcome_probes
    tags = _filer_cmd([], item, Path("-"), date, fp, hold_dispatch=hold is not None)
    pending = " [in-flight attempting row; recovery scan runs first]" if state != "fresh" else ""
    held = (
        f" [held dispatch: shares {','.join(hold['shared'])} with {hold['with']}]" if hold else ""
    )
    inject = _dry_run_inject_note(item, dirpath, fp)
    sha_note = _dry_run_sha_note(item, dirpath, root, fp=fp)
    print(f"FILE {slug} tags={tags[tags.index('--tag') :]}{pending}{held}{inject}{sha_note}")
    return "skip"


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
    target_holds: dict[str, dict] | None = None,
    retry_suspects: bool = False,
) -> str:
    """Run one manifest item through resume -> recovery -> dedup -> two-phase file.

    ``target_holds`` is the full-manifest #1678 same-target hold map from
    ``compute_target_holds`` (None — the direct-call default — means no holds).

    Returns the item outcome: 'skip' | 'recovered' | 'deduped' | 'already-tracked'
    | 'landed-fix-suspect' | 'filed' | 'error'.
    """
    slug = item["slug"]
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    state = _slug_state(ledger, slug, retry_errors, retry_suspects)
    hold = (target_holds or {}).get(slug)
    # #1674: under --retry-suspects the filer already EYEBALLED this slug's prior
    # suspect commit(s), so the probe is SKIPPED for it — including the
    # ERROR+suspect combination, where the ERROR branch wins _slug_state and both
    # --retry-errors --retry-suspects are needed to re-drive (see _slug_state).
    suspect_eyeballed = retry_suspects and _has_landed_fix_suspect_row(ledger, slug)

    if state == "terminal":
        print(f"SKIP {slug}")
        return "skip"

    if dry_run:
        return _dry_run_item(
            item,
            dirpath=dirpath,
            tasks_root=tasks_root,
            root=root,
            date=date,
            fp=fp,
            state=state,
            hold=hold,
            suspect_eyeballed=suspect_eyeballed,
        )

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

    # #1674 + #1711 mechanical landed-fix probes — one of the LAST dedup-family
    # checks before any mutation (body provenance write, `attempting` row, filer
    # subprocess). A suspect never leaves an `attempting` row (recovery
    # semantics untouched).
    outcome_probes = _landed_fix_probes_outcome(
        item, root, dirpath=dirpath, fp=fp, suspect_eyeballed=suspect_eyeballed, dry_run=False
    )
    if outcome_probes is not None:
        return outcome_probes

    body_path = _resolve_body_path(item, dirpath)
    if _wf_fix_enabled(item):
        # Same condition under which _filer_cmd applies the wf-fix tag — the tag and
        # the durable recursion-guard body signal cannot diverge on the driver path
        # (#1173; both sites key on _wf_fix_enabled, #1228; #1580 reconciles a
        # mismatched body fp to the tag value). Idempotent, so a kill anywhere
        # re-normalizes harmlessly on resume.
        _apply_wf_fix_provenance(item, body_path, slug, fp)
    else:
        _warn_stray_wf_fix_provenance(item, dirpath)
    # #1467 WARN-only sha-verify backstop: runs AFTER the Provenance injection (so the
    # advisory lands in the FILED body) and for EVERY route/wf_fix variant — content
    # accuracy is orthogonal to the wf-fix key space. Never blocks; exit code untouched.
    sha_warnings = _check_body_shas(item, dirpath, root, fp=fp)
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
    cmd = _filer_cmd(filer_prefix, item, body_path, date, fp, hold_dispatch=hold is not None)
    if hold is not None:
        print(
            f"HELD-DISPATCH {slug}: shares target {','.join(hold['shared'])} with earlier "
            f"wave sibling {hold['with']} — filing with --no-dispatch; the watcher "
            f"proposed_infra_sweep dispatches it (#1678)"
        )
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
        # #1529: forward + persist the filer's #1399/#1502 sibling advisories — computed
        # INSIDE the success branch (error/timeout/dedup paths stay byte-unchanged).
        advisories = extract_filer_advisories(proc.stderr)
        append_row(
            dirpath,
            _filed_ledger_row(slug, tid, fp, item, out[-300:], sha_warnings, advisories, hold),
        )
        print(f"FILED {slug} -> #{tid} (rc=0)")
        _print_advisory_forward(slug, tid, advisories)
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
        "--retry-suspects",
        action="store_true",
        help="re-attempt slugs whose only terminal row is landed-fix-suspect; BOTH the"
        " #1674 commit-subject probe AND the #1711 closed-sibling probe are SKIPPED for"
        " them — the filer already eyeballed the suspect commit(s) and/or closed sibling"
        " task(s) (#1674, #1711)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print per-item planned action (FILE/DEDUP/SKIP); no filer subprocess, no"
        " ledger/body writes (read-only probes may run: rev-parse sha-scan #1467,"
        " git-log landed-fix probe #1674, closed-sibling registry+body scan #1711)",
    )
    return parser


# ── #1735 terminal SUMMARY line + `daily-drive-summary` ledger row ────────────
# Emitted at the tail of main() so a nightly-batch operator (and a future
# reader of filed.jsonl) cannot mistake a mass suspect suppression for a
# clean batch. The two suspect counters are named separately because the
# `landed-fix-suspect` outcome string is emitted by BOTH the closed-sibling
# probe (#1711, §4.1 of this fix) AND the sibling #1674 landed-fix-sha probe;
# conflating them in a single `suspects=S` counter would hide a legitimate
# #1674-only burst under the appearance of a closed-sibling regression.
#
# `counts` schema (EXACTLY 8 keys — pinned by
# tests/test_daily_drive_filings.py::test_summary_row_appended_to_ledger):
#   filed, deduped, already-tracked, recovered, skip, error,
#   closed-sibling-suspects, landed-fix-suspects
#
# The single `landed-fix-suspect` process_item outcome is split by probe
# source: read back the just-written ledger row's `suspects[0]["kind"]`
# field — `"closed-sibling"` → `closed-sibling-suspects`; absent (i.e. the
# #1674 shape, `{"sha","subject","shared"}`) → `landed-fix-suspects`. A
# corrupt / missing suspects field falls soft to
# `landed-fix-suspects-unknown-kind` and emits an implementer stderr WARN
# (an unclassified suspect is still a suspect for `--retry-suspects`
# purposes; the operator still sees it in the SUMMARY line). The `held=H`
# column is NOT emitted — `held` is a row FIELD on `filed` rows set inside
# process_item, not a returned outcome (Statistics-critic MF2).
_SUMMARY_LEDGER_OUTCOME = "daily-drive-summary"


def _summary_key_for_outcome(outcome: str, dirpath: Path, slug: str) -> str:
    """Map a process_item outcome onto its SUMMARY counter key (#1735 §4.4).

    Non-suspect outcomes map to themselves verbatim. The single
    ``landed-fix-suspect`` outcome is discriminated by the just-written
    ledger row's ``suspects[0]["kind"]`` — the row was appended by
    ``_closed_sibling_outcome`` (kind=``"closed-sibling"``) or
    ``_landed_fix_suspect_outcome`` (no ``kind`` field; #1674 row shape).
    Falls soft on any ledger read/schema error to the aggregate key
    ``landed-fix-suspects-unknown-kind`` + a stderr WARN.
    """
    if outcome != "landed-fix-suspect":
        return outcome
    try:
        # The just-written row is the LAST row of filed.jsonl for this slug;
        # append_row does an O_APPEND single-line write, so the tail row is
        # this slug's suspect row by construction. Load with load_ledger for
        # its corrupt-line tolerance (mirrors the same read the driver uses
        # everywhere else).
        rows = load_ledger(dirpath)
        for row in reversed(rows):
            if row.get("slug") != slug:
                continue
            suspects = row.get("suspects") or []
            if not suspects:
                break
            kind = suspects[0].get("kind")
            if kind == "closed-sibling":
                return "closed-sibling-suspects"
            if kind is None:
                return "landed-fix-suspects"
            break
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(
            f"WARNING {slug}: SUMMARY suspect-kind read failed"
            f" ({e.__class__.__name__}: {e}) — recording under"
            " landed-fix-suspects-unknown-kind; the item still appears in the SUMMARY"
            " suspect count (#1735)",
            file=sys.stderr,
        )
        return "landed-fix-suspects-unknown-kind"
    print(
        f"WARNING {slug}: SUMMARY suspect-kind unreadable"
        " (missing / malformed suspects[0].kind) — recording under"
        " landed-fix-suspects-unknown-kind (#1735)",
        file=sys.stderr,
    )
    return "landed-fix-suspects-unknown-kind"


def _emit_daily_drive_summary(
    dirpath: Path,
    outcome_counts: dict[str, int],
    args: argparse.Namespace,
    date: str,
    *,
    dry_run: bool,
    retry_suspect_matches: int | None = None,
) -> None:
    """Print the terminal SUMMARY stderr line and (unless dry_run) append a
    ``daily-drive-summary`` row to filed.jsonl (#1735 §4.4).

    The schema key set is exact — the eight counters are named + present
    regardless of whether they saw any activity this run (0 when a counter
    saw no items). ``slug: null`` distinguishes the row from any item row;
    existing ledger consumers filter on ``outcome`` in one of the item-outcome
    values (or the ``attempting``/``ERROR`` two-phase tail) and ignore this
    row by default. Grep-based consumers find it via
    ``jq 'select(.outcome=="daily-drive-summary")'``.

    ``retry_suspect_matches`` (#1758): the pre-loop count of sliced slugs
    carrying a suspect ledger row, or None when ``--retry-suspects`` was
    unset. When exactly 0, a zero-match note is appended to the PRINTED
    stderr line ONLY — the ledger row's ``counts`` key set stays untouched.
    """
    keys = (
        "filed",
        "deduped",
        "already-tracked",
        "recovered",
        "skip",
        "error",
        "closed-sibling-suspects",
        "landed-fix-suspects",
    )
    counts: dict[str, int] = {k: int(outcome_counts.get(k, 0)) for k in keys}
    # Preserve any fail-soft unknown-kind bucket in the printed line + ledger
    # row (never silently collapse it into a named counter). Numbers stay
    # int-typed for JSON schema stability.
    unknown = int(outcome_counts.get("landed-fix-suspects-unknown-kind", 0))
    if unknown:
        counts["landed-fix-suspects-unknown-kind"] = unknown
    n_suspects = counts["closed-sibling-suspects"] + counts["landed-fix-suspects"] + unknown
    suspect_hint = " — re-run with --retry-suspects to file suspects" if n_suspects > 0 else ""
    # #1758: print-line-only reflection of the pre-loop zero-match state; None
    # (flag unset) never appends. Placed BEFORE suspect_hint so the line reads
    # chronologically (pre-loop state, then this run's minted suspects).
    zero_match_note = (
        " — --retry-suspects matched 0 recorded suspects (nothing retried)"
        if retry_suspect_matches == 0
        else ""
    )
    parts = [f"{k}={counts[k]}" for k in keys]
    if unknown:
        parts.append(f"landed-fix-suspects-unknown-kind={unknown}")
    print(
        f"SUMMARY dir={dirpath} {' '.join(parts)}{zero_match_note}{suspect_hint}",
        file=sys.stderr,
    )
    if dry_run:
        return
    append_row(
        dirpath,
        {
            "slug": None,
            "outcome": _SUMMARY_LEDGER_OUTCOME,
            "counts": counts,
            "sliced": [args.start, args.end],
            "date": date,
        },
    )


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
    # #1678: computed over the FULL manifest BEFORE slicing, so every --start/--end
    # slice and every resume invocation derives the identical hold map.
    target_holds = compute_target_holds(manifest)
    ledger = load_ledger(dirpath)
    sliced = manifest[args.start : args.end]
    # #1758: under --retry-suspects, count sliced slugs carrying a suspect
    # ledger row BEFORE the drive loop (pristine loaded ledger — mid-run
    # appends cannot shift it; prints on dry-run too). Predicate is
    # deliberately _has_landed_fix_suspect_row, NOT _slug_state ==
    # "retry-suspect", so an ERROR+suspect slug driven under both retry
    # flags still counts as a match and cannot draw a false notice.
    retry_suspect_matches: int | None = None
    if args.retry_suspects:
        retry_suspect_matches = sum(
            1 for item in sliced if _has_landed_fix_suspect_row(ledger, item["slug"])
        )
        if retry_suspect_matches == 0:
            lo = "" if args.start is None else args.start
            hi = "" if args.end is None else args.end
            print(
                f"NOTICE: --retry-suspects matched 0 recorded suspects in slice [{lo}:{hi}]"
                " — nothing to retry (#1758)",
                file=sys.stderr,
            )
    any_error = False
    outcome_counts: dict[str, int] = {}
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
            target_holds=target_holds,
            retry_suspects=args.retry_suspects,
        )
        # Split `landed-fix-suspect` by probe source before tallying; every
        # other outcome maps to its own name (see _summary_key_for_outcome).
        key = _summary_key_for_outcome(outcome, dirpath, item["slug"])
        outcome_counts[key] = outcome_counts.get(key, 0) + 1
        if outcome == "error":
            any_error = True
    # Terminal SUMMARY: always emitted (dry-run too); ledger append skipped on
    # dry-run (read-only by construction, #1735 §4.4).
    _emit_daily_drive_summary(
        dirpath,
        outcome_counts,
        args,
        date,
        dry_run=args.dry_run,
        retry_suspect_matches=retry_suspect_matches,
    )
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
