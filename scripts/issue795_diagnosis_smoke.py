#!/usr/bin/env python
"""Issue #795 diagnosis smoke — verify the "zombie session on a completed task"
ghost class is ALREADY reaped on `main` by the #720 short-window idle-unmapped
path, on the ~50-min timeline. This is a NO-CHANGE (verification-only) diagnosis:
it asserts an existing on-`main` behavior rather than exercising new code.

Background (the daily-brief flag that filed #795):
  Happy sessions stayed mapped to already-`completed` tasks (698x1, 705x2,
  706x2) and were not auto-reaped by the *reconcile* pass; Thomas reaped them
  manually on 2026-06-30.

What actually closed the gap:
  #720 (merged 2026-06-28 14:15 PT) installed the breadcrumb + short reap
  window: the respawn pass writes ``last-mapped-terminal-<sid>.json`` at the
  instant it deletes ``issue-<N>.json`` for a TERMINAL task, and the
  idle-unmapped pass reads it to shorten that class's reap 12h -> 30 min
  (worst case 30 min + 2*10-min ticks = 50 min < the ~1h acceptance window).
  The evidence tasks 698/705/706 completed ~13h BEFORE #720 merged, so their
  manual reaping was of PRE-#720 zombies that never got a breadcrumb.

This script proves three things without changing any runtime behavior:
  (A) SYNTHETIC: a completed-task + gone-registration + repo-root-cwd +
      non-TTY session (the exact #795 ghost class) with a #720 breadcrumb gets
      the SHORT (30-min) window from ``_effective_idle_reap_s`` — i.e. it WILL
      be reaped by the idle-unmapped pass, not stranded on 12h.
  (B) ARITHMETIC: the worst-case reap is 50 min < 60 min (30-min window +
      2 consecutive-miss ticks at 10 min each).
  (C) LIVE (``--live``): of the currently-unmapped running EPS sessions, how
      many carry a #720 breadcrumb (=> fast lane) vs none (=> pre-#720 orphan
      on the 12h lane) — so a reviewer can see the current backlog is the
      pre-#720 no-breadcrumb class, unreachable by ANY snapshot taken today.

Run:
  uv run python scripts/issue795_diagnosis_smoke.py            # A + B (offline, deterministic)
  uv run python scripts/issue795_diagnosis_smoke.py --live     # A + B + C (reads the live fleet)

Exit code 0 iff (A) and (B) hold. (C) is report-only (never fails the smoke).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autonomous_session_watch as w


# ── (A) synthetic short-window proof, via the pure decision core ─────────────
def check_synthetic_short_window() -> bool:
    """Prove the #795 ghost class earns the SHORT reap window.

    ``_effective_idle_reap_s(sid, mapped, has_tty, long_reap_s)`` is the pure
    core: it returns ``min(long, 30min)`` iff the session is unmapped + non-TTY
    + has a TERMINAL breadcrumb + both protected-class guards clear. We monkey
    the breadcrumb read + the two guards in-process so no registry files or
    daemon are needed.
    """
    sid = "synthetic-795-ghost"
    long_reap_s = w.UNMAPPED_IDLE_REAP_S  # 12h default
    short = w._last_mapped_terminal_reap_s()  # 30 min default

    # Save + patch the three lazily-called dependencies of the pure core.
    orig_crumb = w._last_mapped_terminal
    orig_pods = w._running_managed_issue_pods
    orig_followup = w._task_followup_active
    try:
        # The respawn pass recorded this sid's last mapped task as completed #698.
        w._last_mapped_terminal = lambda s: ("completed", 698) if s == sid else None
        # Protected-class guards clear: no running pod, no live follow-up.
        w._running_managed_issue_pods = lambda **kw: []  # [] = no pods (not None=uncertain)
        w._task_followup_active = lambda issue, events=None: False

        got_ghost = w._effective_idle_reap_s(
            sid, mapped=False, has_tty=False, long_reap_s=long_reap_s
        )
        # A mapped session (still active) must NOT get the short window.
        got_mapped = w._effective_idle_reap_s(
            sid, mapped=True, has_tty=False, long_reap_s=long_reap_s
        )
        # A TTY session (live user) must NOT get the short window.
        got_tty = w._effective_idle_reap_s(sid, mapped=False, has_tty=True, long_reap_s=long_reap_s)
    finally:
        w._last_mapped_terminal = orig_crumb
        w._running_managed_issue_pods = orig_pods
        w._task_followup_active = orig_followup

    ok_ghost = got_ghost == short
    ok_mapped = got_mapped == long_reap_s
    ok_tty = got_tty == long_reap_s
    print("(A) synthetic short-window proof")
    print(
        f"    ghost class (unmapped, completed #698, no pod, no follow-up): "
        f"reap window = {got_ghost / 60:.0f} min "
        f"({'SHORT — reaped by idle-unmapped pass' if ok_ghost else 'WRONG'})"
    )
    print(
        f"    mapped (active task) control: {got_mapped / 3600:.0f}h "
        f"({'long, untouched' if ok_mapped else 'WRONG'})"
    )
    print(
        f"    TTY (live user) control: {got_tty / 3600:.0f}h "
        f"({'long, untouched' if ok_tty else 'WRONG'})"
    )
    return ok_ghost and ok_mapped and ok_tty


# ── (B) arithmetic worst-case bound ──────────────────────────────────────────
def check_worst_case_bound() -> bool:
    short = w._last_mapped_terminal_reap_s()
    tick = 10 * 60  # the watcher cron is */10
    misses = 2  # the >=2-consecutive-miss guard
    worst = short + misses * tick
    acceptance = 60 * 60  # the body's ~1h acceptance window
    ok = worst < acceptance
    print("(B) arithmetic worst-case reap bound")
    print(
        f"    {short / 60:.0f}-min window + {misses}x{tick / 60:.0f}-min ticks "
        f"= {worst / 60:.0f} min {'<' if ok else '>='} {acceptance / 60:.0f}-min acceptance "
        f"({'PASS' if ok else 'FAIL'})"
    )
    return ok


# ── (C) live-fleet report (never fails the smoke) ────────────────────────────
def report_live_fleet() -> None:
    print("(C) live-fleet check (report-only)")
    try:
        live = w._live_session_ids()
    except Exception as e:
        print(f"    could not read live sessions: {e}")
        return
    mapped = w._load_session_issue_map()
    unmapped = [s for s in live if s not in mapped]
    n_crumb = sum(1 for s in unmapped if w._last_mapped_terminal(s) is not None)
    print(f"    live sessions: {len(live)}")
    print(f"    registry-mapped (active tasks): {len(mapped)}")
    print(f"    unmapped: {len(unmapped)}")
    print(f"      with #720 breadcrumb (=> 30-min fast lane): {n_crumb}")
    print(f"      no breadcrumb (=> pre-#720 orphan, 12h lane): {len(unmapped) - n_crumb}")
    print(
        "    NOTE: a breadcrumb is written only at the respawn-pass delete "
        "instant; pre-#720 orphans predate that write and are UNREACHABLE by "
        "any reconcile-pass snapshot taken today (the snapshot is empty for "
        "them too — _load_session_issue_map has zero overlap with them)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="also run the live-fleet report (C)")
    args = ap.parse_args()

    print("=" * 72)
    print("Issue #795 diagnosis — is the completed-task ghost class reaped on main?")
    print("=" * 72)
    a = check_synthetic_short_window()
    print()
    b = check_worst_case_bound()
    print()
    if args.live:
        report_live_fleet()
        print()

    verdict = a and b
    print("=" * 72)
    if verdict:
        print(
            "VERDICT: #720 already reaps the #795 ghost class within ~50 min. "
            "No reconcile-pass change needed (Route A: Method delta = none)."
        )
    else:
        print(
            "VERDICT: the #720 short-window path did NOT behave as documented — "
            "re-open the gap analysis."
        )
    print("=" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
