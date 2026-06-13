#!/usr/bin/env python3
"""Backend-agnostic one-tick poll script (bg-Bash poll bridge for `/issue`).

The orchestrator's Step 6d.2 bg-Bash polling loop calls this script
once per tick; it prints ONE ``PollResult``-shaped JSON line to stdout
and exits. The JSON shape is byte-identical to
``scripts/poll_pipeline.py``'s output, so the orchestrator's existing
JSON-line parser handles every backend (RunPod / SLURM / GCP)
without per-backend branches.

Usage::

    uv run python scripts/backend_poll.py --issue <N>            # default sidecar
    uv run python scripts/backend_poll.py --issue <N> --handle-file <path>

The dispatch helper (:mod:`backends.issue_dispatch`) writes the per-issue
:class:`~backends.base.RunHandle` to
``<main-checkout>/.claude/cache/issue-<N>-handle.json`` at launch (the
path is resolved cwd-INDEPENDENTLY — a launch dispatched from an issue
worktree and a poll tick run from the repo root converge on the same
file; incident #612). This script reads it back, recovers the right
:class:`~backends.base.ComputeBackend` subclass from
``handle.backend``, and calls ``backend.poll(handle)`` once. For
back-compat with sidecars written by the pre-#612 cwd-relative composer
it also probes ``<cwd>/.claude/cache/issue-<N>-handle.json`` when the
canonical path is absent.

The orchestrator re-invokes after each bg-Bash exit (the harness
re-invocation model — see CLAUDE.md § "Orchestrator vs subagent
re-invocation"). KEEPING the bg-Bash poll loop as a separate process
is load-bearing: notification-on-exit IS the orchestrator's wakeup
signal. Moving poll in-process would break it.

For backend = ``runpod`` this script is functionally equivalent to
``poll_pipeline.py`` (RunPodBackend.poll delegates to
``poll_pipeline.poll_once``). For ``cluster``/``nibi``/``fir`` it
delegates to ``SlurmBackend.poll`` (which calls into
``backends.slurm_monitor.build_poll_result``). For ``gcp`` it
delegates to ``GcpBackend.poll`` (``gcloud compute instances describe``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Repo-root sys.path bootstrap. Invoking this file as a script puts only
# scripts/ (the script's own dir) on sys.path — NOT the repo root — so the
# lazy import inside the RunPod backend (`backends/runpod.py` does
# `from scripts.poll_pipeline import ...`) fails with
# ``ModuleNotFoundError: No module named 'scripts'`` unless PYTHONPATH is
# set manually. Insert the repo root so the documented invocation
# (``uv run python scripts/backend_poll.py --issue <N>``) works from any
# cwd (incident #571, 2026-06-11: first pod tick crashed exit 1).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Conservative short bg-poll interval (seconds). Mirrors
# ``scripts.poll_pipeline.POLL_INTERVAL_DEFAULT_SEC`` — kept as a local
# literal so the fast ``--help`` path doesn't import the (heavy)
# poll_pipeline module at startup. Used for results that don't carry a
# ``next_interval`` and for the missing-sidecar terminal JSON.
_DEFAULT_NEXT_INTERVAL_SEC = 540


def _resolve_backend(name: str):
    """Map ``handle.backend`` to a ComputeBackend instance.

    Each backend's constructor takes no required args; defaults match
    the production wiring (default config, real runner, real marker
    poster). A future extension might thread per-call config in via a
    sidecar — for slice 6 the defaults suffice.
    """
    if name == "runpod":
        from explore_persona_space.backends.runpod import RunPodBackend

        return RunPodBackend()
    if name in {"cluster", "nibi", "fir", "mila"}:
        from explore_persona_space.backends.slurm import SlurmBackend

        return SlurmBackend()
    if name == "gcp":
        from explore_persona_space.backends.gcp import GcpBackend

        return GcpBackend()
    raise ValueError(f"backend_poll: unknown backend {name!r}; cannot resolve a backend class")


def _serialize_poll_result(result) -> dict:
    """Serialize a PollResult to the canonical JSON shape.

    Matches ``scripts/poll_pipeline.py.main``'s output keys so the
    orchestrator's parser is interchangeable. Field set held in sync
    with ``backends.base.PollResult`` + ``scripts.poll_pipeline.PollResult``.
    """
    return {
        "status": result.status,
        "current_phase": result.current_phase,
        "new_milestone": result.new_milestone,
        "last_log_mtime_sec_ago": result.last_log_mtime_sec_ago,
        "pid_alive": result.pid_alive,
        "log_tail_excerpt": result.log_tail_excerpt,
        "gate": result.gate,
        "sentinels_processed": result.sentinels_processed,
        "phase_log_mtime_sec_ago": result.phase_log_mtime_sec_ago,
        "shard_log_mtime_sec_ago": result.shard_log_mtime_sec_ago,
        "gpu_util": result.gpu_util,
        # Adaptive bg-poll interval (anti-stall redesign §7): the
        # orchestrator's sleep-chain uses this for the NEXT `sleep
        # <interval>` (SKILL.md Step 6d.2; 540s fallback when absent).
        # ``getattr`` defends against a duck-typed / older-module result
        # that predates the field — mixed-version worktree copies degrade
        # to the conservative short interval, never crash the poll.
        "next_interval": int(getattr(result, "next_interval", _DEFAULT_NEXT_INTERVAL_SEC)),
    }


def _missing_sidecar_json(issue: int, sidecar_path: Path, reason: str) -> dict:
    """Build the failure-shape JSON line for a missing / unreadable sidecar.

    On a missing or unreadable sidecar (typically pre-launch, between
    crash + relaunch, or a worktree that was reaped before the
    orchestrator re-armed), historically this script raised
    ``FileNotFoundError`` and the bg-Bash poll loop produced EMPTY
    stdout. The orchestrator's JSON-line parser then looped on "stalled"
    forever (no JSON to parse → no terminal signal). FIX: print ONE
    canonical JSON line shaped as a ``PollResult`` ``status: "dead"``
    plus the ``failure_class: "infra"`` + ``reason`` keys the
    orchestrator's failure-classifier reads, so the next bg-Bash exit
    converts it into ``epm:failure v1 failure_class: infra reason:
    missing_handle_sidecar`` and the loop terminates cleanly.

    Defense-in-depth: even after ``scripts/dispatch_issue.py launch``
    makes the sidecar always present on a successful launch, an
    orchestrator that polls BEFORE launch completes (race) or after a
    worktree-reap (stale cache dir) still needs a terminal JSON to
    break the bg-Bash loop.
    """
    return {
        # Legacy poll_pipeline JSON-line keys (orchestrator parser
        # contract — same fields backend_poll.py emits on success).
        "status": "dead",
        "current_phase": "missing-sidecar",
        "new_milestone": False,
        "last_log_mtime_sec_ago": 10**9,
        "pid_alive": False,
        "log_tail_excerpt": f"backend_poll: {reason} at {sidecar_path}",
        "gate": None,
        "sentinels_processed": 0,
        "phase_log_mtime_sec_ago": 10**9,
        "shard_log_mtime_sec_ago": 10**9,
        "gpu_util": "unknown",
        # Terminal verdict — the orchestrator stops the loop, so the
        # interval is moot, but the key stays present (short default) so
        # the JSON shape is uniform across every emitted line (§7).
        "next_interval": _DEFAULT_NEXT_INTERVAL_SEC,
        # Failure-classifier hint keys — the orchestrator reads these
        # alongside ``status: "dead"`` to post ``epm:failure v1`` with
        # the matching failure_class instead of a generic "workload
        # died".
        "failure_class": "infra",
        "reason": "missing_handle_sidecar",
        "issue": int(issue),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issue",
        type=int,
        required=True,
        help="Task / issue number (resolves the default handle sidecar).",
    )
    parser.add_argument(
        "--handle-file",
        type=Path,
        default=None,
        help=(
            "Path to the per-issue handle sidecar JSON "
            "(default: <main-checkout>/.claude/cache/issue-<N>-handle.json, "
            "with a legacy <cwd>/.claude/cache/ fallback probe)."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Log to stderr at DEBUG level.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Lazy imports — keeps the --help path fast.
    from explore_persona_space.backends.issue_dispatch import (
        read_handle_sidecar,
        resolve_handle_sidecar_path,
    )

    # Resolution order: explicit --handle-file > canonical
    # <main-checkout>/.claude/cache/ > legacy <cwd>/.claude/cache/
    # (back-compat with sidecars written by the pre-#612 cwd-relative
    # composer). A resolution CRASH (git missing / not a checkout) is
    # converted to the same terminal infra JSON as a missing sidecar —
    # this script must NEVER exit with empty stdout (the bg-Bash poll
    # loop would spin forever on "stalled"; that is the exact failure
    # mode the missing-sidecar fast path below exists to close).
    try:
        sidecar, probed = resolve_handle_sidecar_path(args.issue, args.handle_file)
    except RuntimeError as exc:
        fallback = Path(".claude/cache") / f"issue-{int(args.issue)}-handle.json"
        logging.warning(
            "backend_poll: sidecar path unresolvable (%s); emitting status=dead infra", exc
        )
        print(
            json.dumps(
                _missing_sidecar_json(args.issue, fallback, f"sidecar path unresolvable: {exc}")
            )
        )
        return 0

    # Missing-sidecar fast path. Previously this raised
    # ``FileNotFoundError`` → empty stdout → bg-Bash poll loop spins
    # forever ("no JSON to parse" reads as "stalled"). Defense in depth
    # behind ``scripts/dispatch_issue.py launch``'s sidecar write: a
    # poll that races the launch, OR a poll after a worktree-reap,
    # still needs a terminal JSON line to break the loop.
    if not Path(sidecar).exists():
        probed_str = ", ".join(str(p) for p in probed)
        logging.warning(
            "backend_poll: sidecar missing (probed: %s); emitting status=dead infra", probed_str
        )
        print(
            json.dumps(
                _missing_sidecar_json(
                    args.issue, Path(sidecar), f"sidecar not found (probed: {probed_str})"
                )
            )
        )
        return 0

    try:
        handle = read_handle_sidecar(sidecar)
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
        # Same shape as missing-sidecar — a corrupted / malformed
        # sidecar is operationally indistinguishable from "no sidecar"
        # for the orchestrator (it can't poll either way), so emit the
        # SAME terminal infra JSON and let the failure-classifier route.
        logging.warning(
            "backend_poll: sidecar at %s unreadable (%s: %s); emitting status=dead infra",
            sidecar,
            type(exc).__name__,
            exc,
        )
        print(
            json.dumps(
                _missing_sidecar_json(
                    args.issue, Path(sidecar), f"sidecar unreadable: {type(exc).__name__}"
                )
            )
        )
        return 0

    backend = _resolve_backend(handle.backend)
    result = backend.poll(handle)
    print(json.dumps(_serialize_poll_result(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
