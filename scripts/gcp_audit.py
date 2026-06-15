#!/usr/bin/env python3
"""Audit (and optionally reap) stale GCP ``eps-issue-*`` VMs.

GCP analogue of ``scripts/pod_audit.py`` (the RunPod stale-pod sweep). Thin CLI
wrapper over ``backends.gcp.audit_stale_gcp_vms`` — the reap predicate lives in
the library; this script parses flags, runs a LIST-preflight (so a failed
``gcloud compute instances list`` / expired auth is a HARD ERROR rather than a
silent empty sweep — the frozen reaper swallows that rc and returns ``[]``,
which is indistinguishable from a legitimately empty inventory), injects the
production config + runner, prints the records, and maps the result to an exit
code::

    0  clean inspection (0+ VMs, list succeeded)
    2  at least one delete-failed (reaper could not reclaim a reaped VM)
    3  list-failed (gcloud list returned non-zero — auth/config broken; the
       sweep is DISARMED and the operator must be notified)

Invoked by ``scripts/cron_gcp_audit.sh`` daily; runnable by hand for a probe.
The ``EPS_GCP_JANITOR_DRY_RUN=1`` env override forces report-only regardless of
``--delete`` so the cron can be smoke-fired without risk to live instances.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from explore_persona_space.backends.gcp import (
    audit_stale_gcp_vms,
    default_gcloud_runner,
    default_gcp_config,
    render_list_argv,
)
from explore_persona_space.orchestrate.env import load_dotenv

#: MUST match the reaper's internal name filter (``backends/gcp.py``
#: ``audit_stale_gcp_vms``) so the preflight list is byte-identical to the
#: list the reaper issues internally.
_AUDIT_NAME_FILTER = "name~^eps-issue-"

#: Exit codes — see the module docstring.
_RC_CLEAN = 0
_RC_DELETE_FAILED = 2
_RC_LIST_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gcp_audit",
        description="Audit live GCP project for stale eps-issue-* VMs (credit-leak backstop).",
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Actually delete reaped VMs (passthrough to audit_stale_gcp_vms(delete=True)). "
            "Default report-only. Forced OFF by EPS_GCP_JANITOR_DRY_RUN=1."
        ),
    )
    p.add_argument(
        "--max-age-hours",
        type=float,
        default=24.0,
        help="EXITED/old VMs older than this many hours are reaped (default: 24).",
    )
    p.add_argument(
        "--terminal-phase-max-age-min",
        type=float,
        default=10.0,
        help=(
            "RUNNING VMs in a terminal eps/phase older than this floor (minutes) are "
            "reaped (default: 10)."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON, no headers.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    config = default_gcp_config()  # threads --configuration=eps-gcp on every gcloud call
    runner = default_gcloud_runner

    # ----- LIST-PREFLIGHT --------------------------------------------------
    # The frozen reaper returns [] on a non-zero list rc (auth/config broken),
    # hiding the failure as an "empty inventory". Re-issue the SAME list the
    # reaper runs internally and surface its rc so the cron CANNOT silently
    # no-op while the credit-leak backstop is disarmed.
    probe = runner(render_list_argv(config=config, name_filter=_AUDIT_NAME_FILTER))
    if probe.returncode != 0:
        stderr = (probe.stderr or "").strip()
        if args.json:
            print(
                json.dumps(
                    {
                        "list_rc": probe.returncode,
                        "list_stderr": stderr[:500],
                        "instances": [],
                        "error": "gcloud list failed; janitor DISARMED",
                    },
                    indent=2,
                )
            )
        else:
            print(
                f"GCP janitor FATAL: gcloud list rc={probe.returncode} "
                f"({stderr[:200]}); janitor DISARMED, no sweep.",
                file=sys.stderr,
            )
        return _RC_LIST_FAILED

    # ----- LIST OK → run the reaper ----------------------------------------
    delete = args.delete and os.environ.get("EPS_GCP_JANITOR_DRY_RUN", "") != "1"
    records = audit_stale_gcp_vms(
        config=config,
        runner=runner,
        max_age_seconds=int(args.max_age_hours * 3600),
        terminal_phase_max_age_seconds=int(args.terminal_phase_max_age_min * 60),
        delete=delete,
    )

    if args.json:
        print(json.dumps({"list_rc": 0, "list_stderr": "", "instances": records}, indent=2))
    else:
        reaped = [r for r in records if r["action"] in ("would-delete", "deleted")]
        failed = [r for r in records if r["action"] == "delete-failed"]
        mode = "delete" if delete else "report-only"
        print(
            f"GCP janitor: list_rc=0; {len(records)} eps-issue-* VM(s) inspected; "
            f"{len(reaped)} reaped ({mode}), {len(failed)} delete-failed."
        )
        for r in records:
            if r["action"] == "skipped":
                continue
            age_seconds = r["age_seconds"]
            age_str = f"{age_seconds / 3600:.1f}h" if age_seconds is not None else "unknown"
            print(
                f"  {r['action']:<13} {r['name']}  zone={r['zone']}  "
                f"status={r['status']}  reason={r['reason']}  age={age_str}"
            )

    return _RC_DELETE_FAILED if any(r["action"] == "delete-failed" for r in records) else _RC_CLEAN


if __name__ == "__main__":
    sys.exit(main())
