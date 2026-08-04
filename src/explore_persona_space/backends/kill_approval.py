"""User-approval gate for every IRREVERSIBLE compute-destruction path.

Standing user directive (2026-08-04): **compute is destroyed only with
Thomas's explicit approval.** No cron, watcher, poller, or autonomous
session may kill a pod / VM / job on its own.

Why this module exists
----------------------
On 2026-08-04 the daily stale-pod audit was found to have terminated 77
teammate-owned pods on the SHARED RunPod team account over 14 days — an
ownership check that matched pod names as bare substrings of task event
logs (so the audit's own "not ours" report became its ownership evidence),
a staleness clock measured from pod CREATION rather than exit (so the
documented 24h grace window did not exist), and no alerting at all on an
irreversible action.

The lesson generalizes past the one bug: per-call-site guards have to be
re-litigated for every future call site, and this fleet keeps growing
them. So each backend gets ONE choke point that every destruction path
already routes through, and the gate lives there:

===========  ==========================================  =================
backend      choke point                                 destroys
===========  ==========================================  =================
RunPod       ``runpod_api.terminate_pod``                pod + volume
GCP          ``backends.gcp.render_delete_argv``         GCE instance + disk
SLURM        ``backends.slurm.ssh_scancel``              queued/running job
===========  ==========================================  =================

RunPod carries its own copy of this logic (``scripts/runpod_api.py`` is a
scripts-dir module and cannot import the installed package cleanly); it
honours the same env vars, including the legacy pod-specific alias.

Granting approval
-----------------
Env-based and deliberately NOT persisted, so approval is per-invocation
and nothing inherits it — the crontab environment does not set it, which
is what disarms every scheduled destructive path at once::

    EPS_ALLOW_COMPUTE_KILL=1 <command>          # any backend
    pod.py terminate --issue <N> --yes --approve  # RunPod, via the CLI flag

Automation must NEVER set these. The correct automated behaviour is to
SURFACE the candidate for approval (a marker, a push) and leave it alive.
"""

from __future__ import annotations

import contextlib
import os
import threading

#: Thread-local verified-teardown grant (see :func:`verified_teardown`).
_GRANT = threading.local()


@contextlib.contextmanager
def verified_teardown(*, target: str, reason: str):
    """Grant approval for an OWNER-driven teardown whose artifacts are verified.

    The one sanctioned automated destruction: the agent/session that DROVE a
    job may tear its own compute down once everything is uploaded and
    verified. Standing user directive (2026-08-04): *"the agent driving a job
    on a pod can close that pod if everything is uploaded, but NOTHING SHOULD
    GET SHUT DOWN ON ITS OWN."*

    The grant is IN-PROCESS and thread-local, entered only AFTER the caller's
    own upload-verification guard has passed — so it authorizes exactly the
    teardown that guard just cleared, and cannot leak to a cron, a watcher, a
    janitor, or any other process. A background reaper never holds it, which
    is precisely what keeps it refused.

    Args:
        target: what is being torn down (pod name / instance / job id).
        reason: the verification that justifies it, e.g.
            ``"epm:upload-verification PASS"``.
    """
    prev = getattr(_GRANT, "active", None)
    _GRANT.active = {"target": target, "reason": reason}
    try:
        yield
    finally:
        _GRANT.active = prev


def verified_teardown_active() -> bool:
    """True inside a :func:`verified_teardown` block on this thread."""
    return getattr(_GRANT, "active", None) is not None


#: Canonical approval env var — covers RunPod, GCP, and SLURM.
COMPUTE_KILL_APPROVAL_ENV = "EPS_ALLOW_COMPUTE_KILL"

#: Legacy RunPod-specific alias, kept working so ``pod.py terminate
#: --approve`` (which sets it) needs no second variable.
POD_TERMINATE_APPROVAL_ENV = "EPS_ALLOW_POD_TERMINATE"


class ComputeKillNotApproved(RuntimeError):
    """Raised when a destruction path runs without explicit user approval."""


def compute_kill_approved() -> bool:
    """True when this kill is authorized — by the user, or by a verified teardown.

    Exactly two authorizations exist:

    1. **Explicit user approval** via env (per-invocation, never persisted,
       absent from the crontab environment).
    2. **An owner-driven verified teardown** — an active
       :func:`verified_teardown` block on this thread, entered by the flow
       that drove the job only after its upload-verification guard passed.

    Everything else — the daily audit, the watcher, wedge failovers, any
    background sweep — holds neither and is refused.
    """
    if verified_teardown_active():
        return True
    return any(
        os.environ.get(name) == "1"
        for name in (COMPUTE_KILL_APPROVAL_ENV, POD_TERMINATE_APPROVAL_ENV)
    )


def require_compute_kill_approval(*, backend: str, target: str, what: str) -> None:
    """Raise :class:`ComputeKillNotApproved` unless the user approved this kill.

    Args:
        backend: which lane is destroying something (``runpod`` / ``gcp`` /
            ``slurm``) — named in the refusal so the operator knows which
            approval path applies.
        target: the concrete thing being destroyed (pod id, instance name,
            SLURM job id).
        what: plain-English description of what is lost (``"pod + volume"``,
            ``"GCE instance + boot disk"``, ``"queued/running job"``).

    Returns ``None`` when approved; the caller then proceeds unchanged.
    """
    if compute_kill_approved():
        return
    raise ComputeKillNotApproved(
        f"REFUSED to destroy {backend} {target} ({what}): no user approval.\n"
        f"Compute is destroyed only with explicit approval (standing directive "
        f"2026-08-04, after the stale-pod audit destroyed 77 teammate pods).\n"
        f"  Approve ad hoc:   {COMPUTE_KILL_APPROVAL_ENV}=1 <command>\n"
        f"  RunPod via CLI:   pod.py terminate --issue <N> --yes --approve\n"
        f"Automation must NOT set this — surface the target for approval instead."
    )
