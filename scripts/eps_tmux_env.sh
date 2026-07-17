#!/usr/bin/env bash
# scripts/eps_tmux_env.sh — single source of truth for the fleet's tmux socket
# directory (#1466). SOURCE this from every process that spawns or addresses a
# tmux server outside an interactive login shell (cron wrappers, the mygoat
# systemd wrapper); ~/.profile sources it for login shells.
#
# Contract:
#   * Durable default: $HOME/.tmux-sockets (persistent disk — no /tmp cleaner
#     can reach it).
#   * Legacy pin: while ANY socket file remains in /tmp/tmux-<uid> (a
#     pre-#1466 server still holding sessions), resolve /tmp so the whole
#     fleet keeps addressing ONE server. The flip to the durable dir happens
#     automatically — and coherently for every shim consumer — once /tmp
#     holds no tmux sockets: at reboot, after the legacy servers drain, or
#     if the /tmp sockets are deleted again (in which case the stranded
#     server is recovered per the runbook in
#     .claude/rules/background-automation.md § tmux socket-dir contract).
#   * An UNREADABLE legacy dir also pins /tmp (watcher parity: the
#     _live_tmux_socket_present() OSError->True disposition in
#     scripts/autonomous_session_watch.py) — "couldn't look" never flips.
#   * A pre-set TMUX_TMPDIR is always respected.
#   * Safe under `set -euo pipefail` and POSIX sh. EPS_TMUX_LEGACY_DIR
#     overrides the legacy base for tests.
if [ -z "${TMUX_TMPDIR:-}" ]; then
    _eps_legacy_base="${EPS_TMUX_LEGACY_DIR:-/tmp}"
    _eps_legacy_dir="${_eps_legacy_base}/tmux-$(id -u)"
    if [ -d "$_eps_legacy_dir" ] && [ ! -r "$_eps_legacy_dir" ]; then
        # Exists but unreadable: pin /tmp before the find can silently miss.
        TMUX_TMPDIR="$_eps_legacy_base"
    elif find "$_eps_legacy_dir" -maxdepth 1 -type s -print -quit 2>/dev/null | grep -q .; then
        TMUX_TMPDIR="$_eps_legacy_base"
    else
        TMUX_TMPDIR="$HOME/.tmux-sockets"
        mkdir -p -m 700 "$TMUX_TMPDIR"
    fi
    export TMUX_TMPDIR
    unset _eps_legacy_base _eps_legacy_dir
fi
