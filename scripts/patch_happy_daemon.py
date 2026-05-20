#!/usr/bin/env python3
"""patch_happy_daemon.py — add ``claudeArgs`` support to the Happy daemon
spawn-session HTTP RPC.

The vendored Happy daemon (at /usr/lib/node_modules/happy/dist/index-q9G4ktSK.mjs)
accepts only `{directory, sessionId, agent, environmentVariables}` in its
spawn-session RPC. We want an additional `claudeArgs: string[]` field
that gets forwarded to the underlying Claude Code subprocess so a fresh
session can boot with an initial prompt (e.g. `/issue 263`).

This script makes four surgical edits to the file:

  1. Zod body schema gains `claudeArgs: z.array(z.string()).optional()`.
  2. The HTTP handler destructures `claudeArgs` from request.body and
     forwards it to `spawnSession({...})`.
  3. The tmux spawn path appends shell-escaped `claudeArgs` to the
     `fullCommand` string.
  4. The non-tmux spawn path spreads `claudeArgs` into the `args` array.

The script is idempotent: a sentinel comment near the file's top tracks
that the patch is applied. Re-running is safe — it backs up the original
on first apply and refuses to touch an already-patched file.

USAGE
    sudo uv run python scripts/patch_happy_daemon.py            # apply
    sudo uv run python scripts/patch_happy_daemon.py --check    # status only
    sudo uv run python scripts/patch_happy_daemon.py --restore  # roll back
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DAEMON_FILE = Path("/usr/lib/node_modules/happy/dist/index-q9G4ktSK.mjs")
BACKUP_SUFFIX = ".eps-original"
SENTINEL = (
    "// EPS-PATCH: claudeArgs-forwarding + initial-prompt-seed + bypass + no-takeover-downgrade v4"
)


# Each (search, replace) pair is a literal-string substitution. Search
# strings are chosen long enough to be unique and stable across the
# file. If the file changes such that any one doesn't match, the script
# aborts before writing anything.

PATCHES: list[tuple[str, str, str]] = [
    (
        "schema body",
        """body: z.object({
          directory: z.string(),
          sessionId: z.string().optional(),
          agent: z.enum(["claude", "codex", "gemini", "openclaw"]).optional(),
          environmentVariables: z.record(z.string(), z.string()).optional()
        }),""",
        """body: z.object({
          directory: z.string(),
          sessionId: z.string().optional(),
          agent: z.enum(["claude", "codex", "gemini", "openclaw"]).optional(),
          environmentVariables: z.record(z.string(), z.string()).optional(),
          claudeArgs: z.array(z.string()).optional()
        }),""",
    ),
    (
        "HTTP handler destructure",
        """const { directory, sessionId, agent, environmentVariables } = request.body;
      logger.debug(`[CONTROL SERVER] Spawn session request: dir=${directory}, sessionId=${sessionId || "new"}, agent=${agent || "default"}`);
      const result = await spawnSession({ directory, sessionId, agent, environmentVariables });""",
        """const { directory, sessionId, agent, environmentVariables, claudeArgs } = request.body;
      logger.debug(`[CONTROL SERVER] Spawn session request: dir=${directory}, sessionId=${sessionId || "new"}, agent=${agent || "default"}, claudeArgs=${JSON.stringify(claudeArgs)}`);
      const result = await spawnSession({ directory, sessionId, agent, environmentVariables, claudeArgs });""",
    ),
    (
        "tmux fullCommand",
        """const fullCommand = `node --no-warnings --no-deprecation ${cliPath} ${agent} --happy-starting-mode remote --started-by daemon`;""",
        """const __epsClaudeArgsSuffix = (options.claudeArgs || []).map((s) => "'" + String(s).replace(/'/g, "'\\\\''") + "'").join(" ");
          const fullCommand = `node --no-warnings --no-deprecation ${cliPath} ${agent} --happy-starting-mode remote --started-by daemon${__epsClaudeArgsSuffix ? " " + __epsClaudeArgsSuffix : ""}`;""",
    ),
    (
        "non-tmux args array",
        """const args = [
            agentCommand,
            "--happy-starting-mode",
            "remote",
            "--started-by",
            "daemon"
          ];""",
        """const args = [
            agentCommand,
            "--happy-starting-mode",
            "remote",
            "--started-by",
            "daemon",
            ...((options.claudeArgs || []))
          ];""",
    ),
    (
        "nextMessage initial-prompt seed",
        """nextMessage: async () => {
            if (pending) {
              let p = pending;
              pending = null;
              permissionHandler.handleModeChange(p.mode.permissionMode);
              return p;
            }
            let msg = await session.queue.waitForMessagesAndGetAsString(controller.signal);""",
        """nextMessage: async () => {
            if (process.env.HAPPY_INITIAL_PROMPT) {
              const __epsInitialPrompt = process.env.HAPPY_INITIAL_PROMPT;
              delete process.env.HAPPY_INITIAL_PROMPT;
              logger.debug(`[EPS-PATCH] Seeding initial message: ${__epsInitialPrompt}`);
              const __epsInitialMode = process.env.HAPPY_INITIAL_MODE || "bypassPermissions";
              delete process.env.HAPPY_INITIAL_MODE;
              return { message: __epsInitialPrompt, mode: { permissionMode: __epsInitialMode } };
            }
            if (pending) {
              let p = pending;
              pending = null;
              permissionHandler.handleModeChange(p.mode.permissionMode);
              return p;
            }
            let msg = await session.queue.waitForMessagesAndGetAsString(controller.signal);""",
    ),
    (
        # PermissionHandler.handlePermissionResponse upstream unconditionally
        # applies `response.mode` on every tool-approval response from mobile.
        # When the user takes over a session from the Happy mobile app, the
        # client sends a tool-response that includes a `mode` field reflecting
        # its own UI state, which silently downgrades the session out of
        # `bypassPermissions` back to `default`. This patch narrows the
        # acceptance: explicit upgrades (`acceptEdits` / `bypassPermissions` /
        # `plan`) are honored unconditionally; downgrades to `default` require
        # `response.explicitModeChange === true`.
        "no-takeover-downgrade",
        """    if (response.mode) {
      this.permissionMode = response.mode;
    }""",
        """    if (response.mode && ["acceptEdits", "bypassPermissions", "plan"].includes(response.mode)) {
      this.permissionMode = response.mode;
    } else if (response.mode === "default" && response.explicitModeChange === true) {
      this.permissionMode = "default";
    }
    /* EPS-FIX: don't auto-downgrade permission mode on every mobile tool-response;
       requires response.explicitModeChange === true to switch back to default */""",
    ),
]


def _check_root() -> None:
    if os.geteuid() != 0:
        sys.exit(
            "patch_happy_daemon: must run as root (the daemon file is "
            "root-owned). Re-invoke with `sudo`."
        )


def _file_text() -> str:
    return DAEMON_FILE.read_text()


def _backup_path() -> Path:
    return DAEMON_FILE.with_suffix(DAEMON_FILE.suffix + BACKUP_SUFFIX)


def _is_patched(text: str) -> bool:
    return SENTINEL in text


def _save_backup_if_missing(text: str) -> None:
    bp = _backup_path()
    if bp.exists():
        return
    bp.write_text(text)
    print(f"[backup] saved → {bp}")


def cmd_check() -> int:
    text = _file_text()
    if _is_patched(text):
        print(f"PATCHED: sentinel {SENTINEL!r} present.")
        print(f"  Backup at: {_backup_path()}")
        return 0
    # Probe whether each search-string is present (sanity for upgrades).
    missing = []
    for name, search, _ in PATCHES:
        if search not in text:
            missing.append(name)
    if missing:
        print("NOT PATCHED — and these search-strings DO NOT match the file:")
        for n in missing:
            print(f"  - {n}")
        print(
            "The Happy daemon has probably been upgraded. Inspect the file "
            "and update PATCHES in this script to match the new shape."
        )
        return 2
    print("NOT PATCHED — but all search-strings match; ready to apply.")
    return 1


def cmd_apply() -> int:
    _check_root()
    text = _file_text()
    if _is_patched(text):
        print("already patched (sentinel present); nothing to do.")
        return 0
    # Validate every patch site exists EXACTLY once before writing anything.
    for name, search, _ in PATCHES:
        count = text.count(search)
        if count != 1:
            sys.exit(
                f"abort: patch site '{name}' matches {count} times (expected 1). "
                f"The daemon shape has drifted; update PATCHES."
            )
    _save_backup_if_missing(text)
    new = text
    for name, search, replace in PATCHES:
        new = new.replace(search, replace, 1)
        print(f"[ok] patched: {name}")
    # Prepend sentinel as a comment so re-runs are no-ops.
    new = f"{SENTINEL}\n{new}"
    DAEMON_FILE.write_text(new)
    print(f"[ok] wrote {DAEMON_FILE}")
    print()
    print("Restart the Happy daemon to pick up the change:")
    print("  happy daemon stop && happy daemon start")
    return 0


def cmd_restore() -> int:
    _check_root()
    bp = _backup_path()
    if not bp.exists():
        sys.exit(f"no backup found at {bp}; nothing to restore.")
    shutil.copy2(bp, DAEMON_FILE)
    print(f"[ok] restored {DAEMON_FILE} ← {bp}")
    print("Restart the daemon: `happy daemon stop && happy daemon start`")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("apply", help="apply the patch (default)")
    sub.add_parser("check", help="report patch status; non-zero if not applied")
    sub.add_parser("restore", help="restore from .eps-original backup")
    args = parser.parse_args()
    cmd = args.cmd or "apply"
    if cmd == "apply":
        return cmd_apply()
    if cmd == "check":
        return cmd_check()
    if cmd == "restore":
        return cmd_restore()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
