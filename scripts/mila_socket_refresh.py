"""Mila ControlMaster socket refresh helper (slice-7, un-armed).

The Mila login node enforces email-OTP MFA on every fresh SSH session
and keeps the ControlMaster socket warm for ~12 h (cf. ``ControlPersist
12h`` in ``~/.ssh/clusters.config``). After the socket lapses the next
``ssh mila <anything>`` would PROMPT for the OTP — which a router
running headless cannot answer. The result is a stale gate:
``mila_socket_alive()`` returns False indefinitely and the Mila lane is
silently skipped until a human opens a shell, runs ``ssh mila``, types
the OTP, and re-authenticates.

This helper is the SCRIPTABLE half of the refresh:

* ``--probe`` — call :func:`backends.slurm.mila_socket_alive` and print
  a structured JSON status line + exit 0 if alive / 1 if down. Cheap,
  no side effects, safe to run from any cron tick or shell.
* ``--login`` — kick off ``ssh mila true`` with the askpass helper
  named by ``EPS_MILA_ASKPASS`` (set ``SSH_ASKPASS`` +
  ``SSH_ASKPASS_REQUIRE=force`` so SSH consumes the OTP from the
  helper instead of the terminal). The askpass helper is responsible
  for retrieving the OTP — typically by reading a file the Claude
  session wrote after fetching the latest OTP email via the
  google-workspace MCP. This helper does NOT itself reach into gmail
  (a bare shell has no Anthropic MCP); the Claude-session prompt in
  ``.claude/cron-prompts/mila-otp-refresh.md`` documents the full
  loop.

This file is **slice-7 SCAFFOLDING ONLY**. Slice 8 arms the cron and
runs the first live login. The helper is unit-tested for the
scriptable parts (probe + askpass-env build); the actual login flow is
mocked.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Env var that names the askpass helper script. Slice 8 wires this to
#: a short shell script that ``cat``s the OTP file the Claude session
#: writes. We intentionally do NOT ship a default — making it required
#: prevents an accidental terminal-prompt fallback (SSH falling back to
#: stdin would hang the cron forever).
ASKPASS_ENV_VAR = "EPS_MILA_ASKPASS"


#: The SSH alias the helper drives. Slice 7 ships a CLI override so a
#: test / a future migration to a different alias is a flag flip.
DEFAULT_SSH_ALIAS = "mila"


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status / askpass-env helpers
# ---------------------------------------------------------------------------


def probe_socket(*, ssh_alias: str = DEFAULT_SSH_ALIAS) -> dict[str, object]:
    """Run the cheap socket-alive probe; return a structured status dict.

    The dict carries:

    * ``alive`` — bool, the probe result.
    * ``ssh_alias`` — which alias we probed (for log clarity).

    Delegates to :func:`backends.slurm.mila_socket_alive`; that function
    is unit-tested and returns False on every failure path.
    """
    # Late import so a stripped-down "just probe" CLI run does not drag
    # the rest of the backends module in eagerly.
    from explore_persona_space.backends.slurm import mila_socket_alive

    alive = mila_socket_alive(ssh_alias=ssh_alias)
    return {"alive": bool(alive), "ssh_alias": ssh_alias}


def build_askpass_env(
    *,
    askpass_path: str,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the env dict that forces SSH to use ``askpass_path``.

    Sets:

    * ``SSH_ASKPASS`` — the helper SSH will exec when it needs a
      credential. The helper is expected to print the OTP on stdout.
    * ``SSH_ASKPASS_REQUIRE=force`` — instructs SSH to ALWAYS use the
      askpass (default ``DISPLAY``-gated behaviour would skip it on a
      headless tty and fall back to the terminal — which the cron has
      no access to).
    * ``DISPLAY`` — set to ``:0`` if absent. Some older SSH builds
      still gate askpass on ``DISPLAY`` being non-empty even with
      ``SSH_ASKPASS_REQUIRE=force``.

    All other env vars are preserved from ``base_env`` (defaults to a
    snapshot of ``os.environ``) so SSH can find its agent socket / the
    user's keypair config.
    """
    if not askpass_path:
        raise ValueError(
            "build_askpass_env: askpass_path is empty; an empty SSH_ASKPASS "
            "would silently re-fall-back to the terminal prompt and hang the "
            "cron forever. Set EPS_MILA_ASKPASS to the path of the OTP-reader "
            "script before invoking --login."
        )
    env = dict(base_env if base_env is not None else os.environ)
    env["SSH_ASKPASS"] = askpass_path
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env.setdefault("DISPLAY", ":0")
    return env


def login_argv(*, ssh_alias: str = DEFAULT_SSH_ALIAS) -> list[str]:
    """The ssh argv that triggers a fresh authenticated session.

    A no-op remote command (``true``) is sufficient — the auth handshake
    re-warms the ControlMaster socket, then SSH exits. We do NOT pass
    ``BatchMode=yes`` here (that's the PROBE path, not the LOGIN path):
    the whole point of --login is that we want SSH to ask askpass for
    the OTP.
    """
    return ["ssh", ssh_alias, "true"]


def perform_login(
    *,
    ssh_alias: str = DEFAULT_SSH_ALIAS,
    askpass_path: str | None = None,
    base_env: Mapping[str, str] | None = None,
    runner: object = None,
    timeout: int = 60,
) -> int:
    """Initiate a fresh SSH session via the askpass helper.

    Returns the SSH exit code (0 = re-authed, non-zero = failed). The
    callsite (the cron prompt, slice 8) interprets the exit code and
    decides whether to surface a failure marker.

    ``runner`` is the injection seam — when None we shell out via
    :func:`subprocess.run`; tests inject a callable taking
    ``(argv, env, timeout)`` and returning an int exit code so we never
    actually attempt SSH in the test suite.
    """
    resolved_askpass = askpass_path if askpass_path is not None else os.environ.get(ASKPASS_ENV_VAR)
    if not resolved_askpass:
        raise RuntimeError(
            f"perform_login: askpass helper not set. Pass --askpass <path> OR "
            f"export {ASKPASS_ENV_VAR}=<path>. Slice 8 wires this; running --login "
            "without it would hang waiting for a terminal OTP prompt."
        )
    env = build_askpass_env(askpass_path=resolved_askpass, base_env=base_env)
    argv = login_argv(ssh_alias=ssh_alias)
    logger.info("perform_login: ssh %r via askpass=%r", ssh_alias, resolved_askpass)
    if runner is not None:
        # The injection-seam signature is intentionally minimal — the
        # production runner is :func:`subprocess.run`; tests pass a
        # callable returning an exit code directly.
        return int(runner(argv, env, timeout))  # type: ignore[operator]
    proc = subprocess.run(
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        logger.warning(
            "perform_login: ssh exited %d; stderr=%r",
            proc.returncode,
            (proc.stderr or "").strip()[:200],
        )
    return int(proc.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mila_socket_refresh",
        description=(
            "Probe + refresh the Mila ControlMaster SSH socket. "
            "The probe is safe to call from any cron tick; the --login "
            "subcommand requires an askpass helper that reads the OTP "
            "from a file the Claude session populates via the "
            "google-workspace MCP (see .claude/cron-prompts/mila-otp-refresh.md)."
        ),
    )
    parser.add_argument(
        "--ssh-alias",
        default=DEFAULT_SSH_ALIAS,
        help=f"SSH alias to probe / refresh (default: {DEFAULT_SSH_ALIAS!r}).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_probe = sub.add_parser(
        "probe",
        help="Print {alive: bool, ssh_alias: str} JSON; exit 0 if alive, 1 if down.",
    )
    del p_probe  # no further options today; suppress unused-var lint

    p_login = sub.add_parser(
        "login",
        help=(
            "Refresh the socket via askpass-driven SSH. Requires "
            f"--askpass <path> OR ${ASKPASS_ENV_VAR}."
        ),
    )
    p_login.add_argument(
        "--askpass",
        default=None,
        help=(
            "Absolute path to the askpass helper script (slice 8 wires "
            "this to a script that cats the OTP file). Overrides "
            f"${ASKPASS_ENV_VAR}."
        ),
    )
    p_login.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Seconds before the ssh --login call is killed (default: 60).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Returns the exit code (0 = OK, non-zero = down/failed)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s mila-refresh: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == "probe":
        status = probe_socket(ssh_alias=args.ssh_alias)
        print(json.dumps(status, sort_keys=True))
        return 0 if status["alive"] else 1
    if args.action == "login":
        try:
            code = perform_login(
                ssh_alias=args.ssh_alias,
                askpass_path=args.askpass,
                timeout=args.timeout,
            )
        except RuntimeError as exc:
            # Configuration error (no askpass) — print as JSON so the
            # Claude-session caller can detect + react.
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            return 2
        # Re-probe after login to confirm the socket actually warmed.
        status = probe_socket(ssh_alias=args.ssh_alias)
        body = {
            "ok": code == 0 and bool(status["alive"]),
            "ssh_exit": code,
            "alive_after_login": bool(status["alive"]),
            "ssh_alias": args.ssh_alias,
        }
        print(json.dumps(body, sort_keys=True))
        return 0 if body["ok"] else 1
    parser.error(f"unknown action {args.action!r}")
    return 2  # unreachable; parser.error raises SystemExit


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
