"""Structural pin: every cron-wrapper Telegram push call is timeout-bounded (#2387).

The upstream push helper (``~/my-goat/scripts/telegram_push.sh``) runs curl with
no ``--connect-timeout`` / ``--max-time``, so a connected-but-stalled endpoint
would park a cron wrapper forever. Task #2387 wrapped every push EXECUTION site
in ``timeout --kill-after=5s "${PUSH_TIMEOUT}s"`` with a per-wrapper
``PUSH_TIMEOUT="${EPS_PUSH_TIMEOUT_SECS:-30}"`` definition (30 s matches the
bound the Python callers — poll_pipeline.py, vm_disk_guard.py, gcp_audit.py,
sync_repo_root.py — already pass to the same helper).

Three durability pins:

- ``test_every_push_call_site_is_timeout_bounded`` — text-scans each wrapper
  for the push variable in EXECUTION position and asserts a
  ``timeout --kill-after=`` prefix precedes it. This is the ONLY coverage
  vehicle for the two watch scripts: they hardcode the push path (no env
  seam) and MUST NOT be executed by tests — their terminal arms run
  ``crontab -l | grep -v ... | crontab -``, which would mutate the real user
  crontab.
- ``test_every_wrapper_defines_push_timeout`` — each wrapper defines the
  env-overridable bound.
- ``test_every_wrapper_parses`` — ``bash -n`` per wrapper (pure parse, no
  execution, no crontab hazard): a syntax/quoting error introduced into the
  watch scripts is the one failure mode a text scan cannot see, and neither
  watch script is executed by any test.

Behavioral twins (sleeping-stub tests, one per call-site composition shape)
live in tests/test_cron_step9c_ledger_refresh.py (if-condition),
tests/test_cron_lesson_consolidate.py (||-chained fatal arm), and
tests/test_codex_auto_upgrade.py (command substitution).

COVERAGE BOUNDARY: the scan is bounded by the WRAPPERS tuple below plus the
execution-site regex. A NEW cron wrapper calling the push helper must be ADDED
to the tuple or it escapes every pin here; likewise a call shape whose message
argument is not a double-quoted string immediately after the push variable
(e.g. an unquoted message) escapes the regex. Extend both when adding either.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Contiguous repo-relative paths (load-bearing for select_step9c_tests.py's
# literal-path arm: a future diff to any wrapper below auto-selects this file).
WRAPPERS: tuple[str, ...] = (
    "scripts/cron_codex_auto_upgrade.sh",
    "scripts/cron_daily_healthcheck.sh",
    "scripts/cron_lesson_consolidate.sh",
    "scripts/cron_step9c_ledger_refresh.sh",
    "scripts/cron_watch_issue_1739.sh",
    "scripts/cron_watch_issue_2091.sh",
)

# Push variable in EXECUTION position: the quoted variable followed by a
# quoted message argument. `[ -x "$PUSH" ]` guards do not match (a `]`, not a
# quote, follows) and `"${PUSH_TIMEOUT}s"` does not match (`_` after PUSH).
_EXEC_SITE = re.compile(r'"\$\{?(?:TELEGRAM_PUSH|PUSH)\}?"\s+"')

_BOUND_PREFIX = "timeout --kill-after="


def test_every_push_call_site_is_timeout_bounded():
    """Every push EXECUTION line in every wrapper carries the timeout prefix
    BEFORE the push variable; >=1 execution site is found per wrapper (the
    regex cannot green-wash by matching nothing)."""
    for rel in WRAPPERS:
        text = (_REPO_ROOT / rel).read_text()
        n_sites = 0
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = _EXEC_SITE.search(line)
            if m is None:
                continue
            n_sites += 1
            assert _BOUND_PREFIX in line[: m.start()], (
                f"{rel}:{lineno}: push execution without a preceding "
                f"'{_BOUND_PREFIX}' prefix: {line.strip()!r}"
            )
        assert n_sites >= 1, f"{rel}: no push execution site matched — regex or wrapper drifted"


def test_every_wrapper_defines_push_timeout():
    """Each wrapper defines the env-overridable 30 s default bound."""
    for rel in WRAPPERS:
        text = (_REPO_ROOT / rel).read_text()
        assert 'PUSH_TIMEOUT="${EPS_PUSH_TIMEOUT_SECS:-30}"' in text, (
            f"{rel}: missing the PUSH_TIMEOUT definition"
        )


def test_every_wrapper_parses():
    """`bash -n` (parse-only, nothing executed) per wrapper — the sole
    syntax check for the two never-executed watch scripts."""
    for rel in WRAPPERS:
        proc = subprocess.run(
            ["bash", "-n", str(_REPO_ROOT / rel)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert proc.returncode == 0, f"{rel}: bash -n failed:\n{proc.stderr}"
