#!/usr/bin/env python3
"""#2246 §6(2b) re-derivation invariant — independent merged/unmerged re-read.

Re-implements the two-signal merged/unmerged classification SPEC (plan v4 §4
D2: arm precedence sha -> ts[UNSUFFIXED-only, strict >, tz-aware] ->
count-zero) INDEPENDENTLY of ``scripts/worktree_audit.py``, enumerates all
then-current reap-eligible issue worktrees, classifies each MERGED / UNMERGED /
PROBE-FAILED, and checks the invariant against a separately-run report-only
janitor's ``--json`` output.

INDEPENDENCE (plan §6(2b), critique r1 MUST-FIX 4): this script imports
NOTHING from ``worktree_audit`` and never calls ``_branch_unmerged`` directly
or indirectly — the name regex, status walk, kind-scoped events scan, and git
probes below are re-implemented from the plan's spec text. The janitor side of
the invariant is consumed as the ``--audit-json`` artifact of a separately run
``uv run python scripts/worktree_audit.py --json`` (the production
entrypoint), never an in-process call — otherwise the check degenerates to
wiring-only (probe and janitor sharing one implementation would read a shared
false-MERGE as agreement).

Invariant asserted (both directions; a violation exits 1 — a bug, not noise):
  1. every independently-UNMERGED eligible worktree is KEPT by the report-only
     janitor run (with the new unmerged reason or a stronger pre-existing one
     — attribution RECORDED per worktree);
  2. no independently-MERGED eligible worktree is retained BY THE NEW
     unmerged-branch reason. (A probe-failed keep on a MERGED read is reported
     as a SOFT disagreement — the janitor's designed fail-toward-retention
     under a transient rev-list failure — never silently absorbed.)

Orphan duty (plan §6(2b)): an issue worktree whose task folder is gone
(status=None => no events file) is named explicitly — it has NO marker
evidence, so the patch-equivalence arm decides (count-zero reads MERGED);
only a POSITIVE count leaves it UNMERGED, and that state has no automatic
release path (plan §8 disclosure row).

This read is WIRING + POPULATION-DRIFT evidence ONLY: probe and janitor
independently implement one spec, so agreement is not classification TRUTH,
and any worktree kept by a stronger pre-existing reason leaves D2's marginal
effect unexercised — classifier truth is carried by the D4 unit-test fixture
family in ``tests/test_worktree_audit.py`` (plan §6(2a)).

Usage (from the MAIN repo root, so the installed package resolves main's src):
  uv run python scripts/worktree_audit.py --json > /tmp/issue2246-wa-post.json
  uv run python scripts/issue2246_rederive_merge_classification.py \
      --audit-json /tmp/issue2246-wa-post.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.task_workflow import repo_root, tasks_dir

# Mirrors worktree_audit._ISSUE_NAME_RE by SPEC (plan §2: verbatim
# ``^issue-(\d+)(?:-[A-Za-z0-9_.\-]+)?$``) — re-stated, not imported.
ISSUE_NAME_RE = re.compile(r"^issue-(\d+)(?:-[A-Za-z0-9_.\-]+)?$")
# Reap-eligible statuses per the janitor's allowlist (plan §1); an issue
# worktree with NO task folder (orphan, status=None) is also removable today
# and therefore part of the eligible population.
REAPABLE_STATUSES = frozenset({"completed", "archived", "awaiting_promotion"})
# Janitor keep-reason literals — COMPARISON side only (mirror the
# worktree_audit constants by value; an unrecognized 'unmerged'-bearing
# janitor reason is flagged loudly below so silent drift cannot pass).
UNMERGED_REASON = "branch carries commits not reachable from origin/main (unmerged)"
PROBE_FAILED_REASON = "unmerged-branch probe failed (fail toward keep)"

HEAD_TIMEOUT_S = 10
REV_LIST_TIMEOUT_S = 60


def _git(wt: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
    """One timeout-bounded git call against ``wt``; never fetches."""
    return subprocess.run(
        ["git", "-C", str(wt), *args], capture_output=True, text=True, timeout=timeout
    )


def _statuses() -> dict[int, str]:
    """issue id -> status folder name, walked from the sanctioned tasks_dir()
    resolver (never a hand-built cwd-relative path)."""
    out: dict[int, str] = {}
    for status_dir in sorted(tasks_dir().iterdir()):
        if not status_dir.is_dir() or status_dir.name.startswith("_"):
            continue
        for child in status_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                out[int(child.name)] = status_dir.name
    return out


def _aware_epoch(ts: object) -> float | None:
    """TZ-aware epoch for an ISO-8601 ``ts``; None for malformed/naive values
    (a naive ts contributes NO ts evidence — retention-directed, per spec)."""
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.timestamp()


def _merged_rows(events_path: Path | None) -> list[tuple[float | None, str]]:
    """Kind-SCOPED merged-evidence scan: ``(aware_epoch_or_None, note)`` per
    ``kind == "epm:merged"`` row. Every degraded reading (missing/unreadable
    file, malformed line, malformed/naive ts) yields LESS merged evidence."""
    if events_path is None:
        return []
    try:
        raw = events_path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[tuple[float | None, str]] = []
    # split("\n"), not splitlines(): splitlines() also splits on raw
    # U+2028/U+2029/NEL inside ensure_ascii=False JSON note strings.
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("kind") != "epm:merged":
            continue
        note = row.get("note")
        rows.append((_aware_epoch(row.get("ts")), note if isinstance(note, str) else ""))
    return rows


def classify(wt: Path, events_path: Path | None, *, unsuffixed: bool) -> tuple[str, str]:
    """Independent two-signal classification -> (verdict, detail) with
    verdict in {"MERGED", "UNMERGED", "PROBE-FAILED"}.

    Arm precedence per plan §4 D2: (a) sha — the full 40-hex HEAD sha inside
    an epm:merged note, ALL worktrees (branch-bound); (b) ts — UNSUFFIXED
    worktrees only, newest epm:merged aware-epoch STRICTLY > the HEAD
    committer epoch; (c) patch-id count-zero
    (``rev-list --cherry-pick --right-only --count origin/main...HEAD``)."""
    try:
        head = _git(wt, "log", "-1", "--format=%H %ct", "HEAD", timeout=HEAD_TIMEOUT_S)
    except (subprocess.SubprocessError, OSError):
        return "PROBE-FAILED", "HEAD unreadable"
    if head.returncode != 0:
        return "PROBE-FAILED", "HEAD unreadable"
    parts = head.stdout.split()
    if len(parts) != 2 or not parts[1].isdigit():
        return "PROBE-FAILED", "HEAD unreadable"
    head_sha, head_ct = parts[0], int(parts[1])
    rows = _merged_rows(events_path)
    if any(head_sha in note for _, note in rows):
        return "MERGED", "sha arm: HEAD sha appears in an epm:merged note"
    if unsuffixed:
        newest = max((e for e, _ in rows if e is not None), default=None)
        if newest is not None and newest > head_ct:
            return "MERGED", "ts arm: newest epm:merged strictly newer than HEAD committer epoch"
    try:
        cnt = _git(
            wt,
            "rev-list",
            "--cherry-pick",
            "--right-only",
            "--count",
            "origin/main...HEAD",
            timeout=REV_LIST_TIMEOUT_S,
        )
    except (subprocess.SubprocessError, OSError):
        return "PROBE-FAILED", "rev-list probe failed"
    if cnt.returncode != 0:
        return "PROBE-FAILED", "rev-list probe failed"
    try:
        n = int(cnt.stdout.strip())
    except ValueError:
        return "PROBE-FAILED", "rev-list probe failed"
    if n == 0:
        return "MERGED", "count arm: 0 commits without a patch-equivalent on origin/main"
    return "UNMERGED", f"count arm: {n} commit(s) with no patch-equivalent on origin/main"


def main(argv: list[str] | None = None) -> int:
    """Enumerate, classify, compare against the janitor JSON, assert the
    invariant both directions; print a timestamped snapshot; 0 ok / 1 bug."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--audit-json",
        required=True,
        type=Path,
        help="stdout of a separately-run `worktree_audit.py --json` (report-only).",
    )
    args = ap.parse_args(argv)

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    if "skipped" in audit:
        print(f"FATAL: audit JSON is a lock-skip record, not a run: {audit}", file=sys.stderr)
        return 1
    if audit.get("apply"):
        print("FATAL: audit JSON came from an --apply run; re-run report-only.", file=sys.stderr)
        return 1
    kept_reason = {d["name"]: d["reason"] for d in audit.get("kept", [])}
    would_remove = set(audit.get("removed", []))
    audit_failed = set(audit.get("failed", []))

    statuses = _statuses()
    wt_root = repo_root() / ".claude" / "worktrees"
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"# #2246 re-derivation snapshot @ {stamp} (population drifts daily — never")
    print("# compare against the plan's illustrative 2026-08-20 list)")

    rows: list[dict] = []
    orphans: list[str] = []
    for child in sorted(wt_root.iterdir()):
        if not child.is_dir():
            continue
        m = ISSUE_NAME_RE.match(child.name)
        if not m:
            continue
        n = int(m.group(1))
        status = statuses.get(n)
        if status is not None and status not in REAPABLE_STATUSES:
            continue  # not reap-eligible today
        unsuffixed = child.name == f"issue-{m.group(1)}"
        events = tasks_dir() / status / str(n) / "events.jsonl" if status is not None else None
        if status is None:
            orphans.append(child.name)
        verdict, detail = classify(child, events, unsuffixed=unsuffixed)
        rows.append(
            {
                "name": child.name,
                "status": status or "ORPHAN (task folder gone)",
                "unsuffixed": unsuffixed,
                "verdict": verdict,
                "detail": detail,
            }
        )

    hard: list[str] = []
    soft: list[str] = []
    print(f"\neligible issue worktrees: {len(rows)}")
    for r in rows:
        name = r["name"]
        if name in would_remove:
            disposition, jreason = "would-remove", ""
        elif name in kept_reason:
            disposition, jreason = "kept", kept_reason[name]
        elif name in audit_failed:
            disposition, jreason = "remove-FAILED", ""
        else:
            disposition, jreason = "NOT-IN-AUDIT", ""
        if jreason.startswith(UNMERGED_REASON):
            attribution = "NEW unmerged-branch reason"
        elif jreason.startswith(PROBE_FAILED_REASON):
            attribution = "NEW probe-failed reason"
        elif disposition == "kept":
            attribution = f"stronger pre-existing: {jreason}"
        else:
            attribution = "-"
        if "unmerged" in jreason.lower() and attribution.startswith("stronger"):
            hard.append(f"{name}: unrecognized unmerged-bearing janitor reason {jreason!r}")
        print(
            f"  {name} [{r['status']}] unsuffixed={r['unsuffixed']}\n"
            f"    independent: {r['verdict']} ({r['detail']})\n"
            f"    janitor:     {disposition}"
            + (f" ({jreason})" if jreason else "")
            + f"\n    attribution: {attribution}"
        )
        if disposition == "NOT-IN-AUDIT":
            hard.append(f"{name}: absent from the audit JSON (population drift between runs?)")
        elif r["verdict"] == "UNMERGED" and disposition != "kept":
            hard.append(f"{name}: independently UNMERGED but janitor disposition={disposition}")
        elif r["verdict"] == "MERGED" and jreason.startswith(UNMERGED_REASON):
            hard.append(f"{name}: independently MERGED but retained BY THE NEW unmerged reason")
        elif r["verdict"] == "MERGED" and jreason.startswith(PROBE_FAILED_REASON):
            soft.append(
                f"{name}: independently MERGED; janitor kept via probe-failed "
                "(fail-toward-retention under a transient failure — re-run to confirm)"
            )
        elif r["verdict"] == "PROBE-FAILED":
            soft.append(f"{name}: independent probe failed ({r['detail']}) — no binding read")

    print(
        "\norphan issue worktrees (task folder gone => no marker evidence; the patch-"
        "equivalence arm decides — only a POSITIVE count has no automatic release path):"
    )
    print("  " + (", ".join(orphans) if orphans else "(none)"))
    if soft:
        print("\nSOFT disagreements / non-binding reads:")
        for s in soft:
            print(f"  ~ {s}")
    if hard:
        print("\nINVARIANT VIOLATIONS (a bug, not noise):", file=sys.stderr)
        for h in hard:
            print(f"  !! {h}", file=sys.stderr)
        return 1
    print("\nINVARIANT OK: every independently-UNMERGED eligible worktree is kept;")
    print("no independently-MERGED eligible worktree is retained by the new reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
