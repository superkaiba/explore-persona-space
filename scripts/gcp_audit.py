#!/usr/bin/env python3
"""Audit (and optionally reap / escalate) stale GCP VMs in the dedicated project.

GCP analogue of ``scripts/pod_audit.py`` (the RunPod stale-pod sweep). Thin CLI
wrapper over ``backends.gcp.audit_stale_gcp_vms`` — the reap/classify predicate
lives in the library; this script parses flags, runs a LIST-preflight (so a
failed ``gcloud compute instances list`` / expired auth is a HARD ERROR rather
than a silent empty sweep — the frozen reaper swallows that rc and returns
``[]``, which is indistinguishable from a legitimately empty inventory), injects
the production config + runner, wires the escalation channel for UNMANAGED stale
VMs, prints the records, and maps the result to an exit code::

    0  clean inspection (0+ VMs, list succeeded; escalations stay rc=0 — they
       are the working path, not a fault, mirroring vm_disk_guard)
    2  at least one delete-failed (reaper could not reclaim a reaped VM)
    3  list-failed (gcloud list returned non-zero — auth/config broken; the
       sweep is DISARMED and the operator must be notified)

Scope (#688): the janitor lists the WHOLE dedicated project, not just
``eps-issue-*``. Each instance is classified by the library and routed by the
HYBRID posture — ``managed`` (``eps-issue-*``) + ``allowlisted-ephemeral``
(``eps-cap-probe*``) auto-DELETE on the existing bounded fences; ``unmanaged``
WARN-and-escalate (Telegram + sidecar JSON, never auto-deleted); ``keep``-prefix
never touched. The escalation closure fires ONLY under ``--delete`` (escalation
is a real side-effect; report-only / ``EPS_GCP_JANITOR_DRY_RUN=1`` produces
``would-escalate`` records and fires NO push).

Invoked by ``scripts/cron_gcp_audit.sh`` daily; runnable by hand for a probe.
The ``EPS_GCP_JANITOR_DRY_RUN=1`` env override forces report-only regardless of
``--delete`` so the cron can be smoke-fired without risk to live instances.

Env overrides: ``EPS_GCP_JANITOR_DRY_RUN=1`` (force report-only),
``EPM_TELEGRAM_PUSH_SCRIPT`` (override the phone-push script path — used by the
tests to point at a stub), ``EPM_GCP_JANITOR_SIDECAR`` (override the escalation
sidecar JSONL path — tests point it at a tmp file).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.backends.gcp import (
    JANITOR_LIST_NAME_FILTER,
    audit_stale_gcp_vms,
    default_gcloud_runner,
    default_gcp_config,
    render_escalation_message,
    render_list_argv,
)
from explore_persona_space.orchestrate.env import load_dotenv

#: MUST match the reaper's internal name filter (``backends/gcp.py``
#: ``audit_stale_gcp_vms``) so the preflight list is byte-identical to the
#: list the reaper issues internally. Both are ``None`` (#688) = list the
#: WHOLE dedicated project, not just ``eps-issue-*``.
_AUDIT_NAME_FILTER = JANITOR_LIST_NAME_FILTER

#: Default phone-push script (the my-goat Telegram channel), overridable via
#: ``EPM_TELEGRAM_PUSH_SCRIPT``. Mirrors ``vm_disk_guard._TELEGRAM_PUSH_SCRIPT_DEFAULT``.
_TELEGRAM_PUSH_SCRIPT_DEFAULT = Path.home() / "my-goat" / "scripts" / "telegram_push.sh"

#: Default escalation sidecar — a DEDICATED GCP-janitor stream (NOT the
#: disk-pressure-scoped ``disk-guard-events.jsonl``), so the two concerns stay
#: cleanly separable. Overridable via ``EPM_GCP_JANITOR_SIDECAR`` (tests point
#: it at a tmp path). Resolved relative to the repo root.
_SIDECAR_REL = Path(".claude") / "cache" / "gcp-janitor-events.jsonl"

#: Exit codes — see the module docstring.
_RC_CLEAN = 0
_RC_DELETE_FAILED = 2
_RC_LIST_FAILED = 3


def _sidecar_path() -> Path:
    """Resolve the escalation sidecar JSONL path (env override wins)."""
    override = os.environ.get("EPM_GCP_JANITOR_SIDECAR", "").strip()
    if override:
        return Path(override)
    # Repo root = three parents up from scripts/gcp_audit.py.
    return Path(__file__).resolve().parents[1] / _SIDECAR_REL


def _telegram_push(msg: str) -> bool:
    """Fail-soft phone push for an UNMANAGED stale GCP VM escalation.

    Mirrors ``vm_disk_guard._telegram_push`` / ``autonomous_session_watch._telegram_push``:
    a missing script or a failed call is logged loudly but NEVER raises (the
    push is observability — the escalation record + sidecar row stand
    regardless). ``NOTIF_CAT=research`` routes it to the research channel.
    """
    override = os.environ.get("EPM_TELEGRAM_PUSH_SCRIPT", "").strip()
    script = Path(override) if override else _TELEGRAM_PUSH_SCRIPT_DEFAULT
    if not script.is_file():
        print(
            f"  WARNING: telegram push script missing at {script}; push dropped",
            file=sys.stderr,
        )
        return False
    try:
        r = subprocess.run(
            ["bash", str(script), msg],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "NOTIF_CAT": "research"},
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  WARNING: telegram push failed: {e}", file=sys.stderr)
        return False
    if r.returncode != 0:
        print(
            f"  WARNING: telegram push failed: {(r.stderr or r.stdout).strip()[:200]}",
            file=sys.stderr,
        )
        return False
    return True


def _append_sidecar(record: dict) -> None:
    """Append one escalation row to the GCP-janitor sidecar (fail-soft)."""
    row = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "event": "gcp-janitor-escalation",
        "name": record.get("name"),
        "zone": record.get("zone"),
        "status": record.get("status"),
        "classification": record.get("classification"),
        "reason": record.get("reason"),
        "age_seconds": record.get("age_seconds"),
    }
    dest = _sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        print(f"  WARNING: GCP-janitor sidecar append failed: {exc}", file=sys.stderr)


def _escalate_unmanaged(record: dict) -> None:
    """Escalation closure for an UNMANAGED stale VM: durable sidecar row FIRST,
    then a fail-soft phone push. The sidecar is the durable second channel that
    stands even when the push fails (REC1 fail-soft contract)."""
    _append_sidecar(record)
    _telegram_push(render_escalation_message(record))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gcp_audit",
        description=(
            "Audit the whole dedicated GCP project for stale VMs (credit-leak backstop): "
            "reap managed + allowlisted-ephemeral, escalate everything else."
        ),
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Actually delete reaped VMs + fire escalations for unmanaged stale VMs "
            "(passthrough to audit_stale_gcp_vms(delete=True, escalate=...)). "
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
                        "records": [],
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
    # The escalation closure is a REAL side-effect (sidecar write + phone push),
    # so it is wired ONLY when delete is in effect — report-only / dry-run pass
    # escalate=None and produce inert "would-escalate" records (no push, no
    # sidecar row). For an all-managed inventory the closure is never invoked.
    escalate = _escalate_unmanaged if delete else None
    records = audit_stale_gcp_vms(
        config=config,
        runner=runner,
        max_age_seconds=int(args.max_age_hours * 3600),
        terminal_phase_max_age_seconds=int(args.terminal_phase_max_age_min * 60),
        delete=delete,
        escalate=escalate,
    )

    if args.json:
        print(json.dumps({"list_rc": 0, "list_stderr": "", "records": records}, indent=2))
    else:
        cls_counts: dict[str, int] = {}
        for r in records:
            cls_counts[r["classification"]] = cls_counts.get(r["classification"], 0) + 1
        reaped = [r for r in records if r["action"] in ("would-delete", "deleted")]
        escalated = [r for r in records if r["action"] in ("would-escalate", "escalated")]
        failed = [r for r in records if r["action"] == "delete-failed"]
        mode = "delete" if delete else "report-only"
        cls_str = ", ".join(f"{k}={v}" for k, v in sorted(cls_counts.items()))
        print(
            f"GCP janitor: list_rc=0; {len(records)} VM(s) inspected ({cls_str}); "
            f"{len(reaped)} reaped, {len(escalated)} escalated ({mode}), "
            f"{len(failed)} delete-failed."
        )
        for r in records:
            if r["action"] == "skipped":
                continue
            age_seconds = r["age_seconds"]
            age_str = f"{age_seconds / 3600:.1f}h" if age_seconds is not None else "unknown"
            print(
                f"  {r['action']:<14} {r['name']}  class={r['classification']}  "
                f"zone={r['zone']}  status={r['status']}  reason={r['reason']}  age={age_str}"
            )

    return _RC_DELETE_FAILED if any(r["action"] == "delete-failed" for r in records) else _RC_CLEAN


if __name__ == "__main__":
    sys.exit(main())
