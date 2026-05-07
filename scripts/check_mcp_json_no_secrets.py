#!/usr/bin/env python
"""Pre-commit hook: refuse to commit secrets into project-level .mcp.json.

Walks any committed ``.mcp.json`` (project-level — user-level
``~/.claude/mcp.json`` is intentionally excluded since it is not under
git) and rejects any ``env`` block that contains a key matching the
broad secret-suffix regex *or* the explicit-name list, unless the key
is in the non-secret allowlist.

Pattern set lives in plan §3 (issue #320) verbatim. The allowlist
covers legitimate non-secret env keys we ship in MCP configs (SSH host
addresses, repo owner/name, etc.).

Designed to be invoked from ``.pre-commit-config.yaml``:

    - id: check-mcp-json-no-secrets
      name: refuse secrets in committed .mcp.json
      entry: uv run python scripts/check_mcp_json_no_secrets.py
      language: system
      files: '\\.mcp\\.json$'
      pass_filenames: true
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Suffix pattern (broad). Catches future tokens like `PROJECT_PAT`,
# `SUPABASE_*`, `CODECOV_TOKEN` without an explicit registry update.
SECRET_SUFFIX_RE = re.compile(r".*_(TOKEN|API_KEY|PAT|SECRET|KEY|PASSWORD)$")

# Explicit list (narrow safety net). Covers names that wouldn't match
# the suffix pattern (e.g. `GITHUB_TOKEN` happens to match, but listed
# here for clarity) plus future-proof entries.
EXPLICIT_SECRET_NAMES: frozenset[str] = frozenset(
    {
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "RUNPOD_API_KEY",
        "PROJECT_PAT",
        "SUPABASE_ACCESS_TOKEN",
        "CODECOV_TOKEN",
    }
)

# Allowlist of legitimate env keys that LOOK like they might be secrets
# (or whose suffix matches the pattern) but are not. The
# ``SSH_SERVER_*_KEYPATH`` entries are paths to key files, not the keys
# themselves; explicitly allowed.
ALLOWLIST_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^GH_REPO_(OWNER|NAME)$"),
    re.compile(r"^SSH_SERVER_[A-Za-z0-9_-]+_HOST$"),
    re.compile(r"^SSH_SERVER_[A-Za-z0-9_-]+_PORT$"),
    re.compile(r"^SSH_SERVER_[A-Za-z0-9_-]+_USER$"),
    re.compile(r"^SSH_SERVER_[A-Za-z0-9_-]+_KEYPATH$"),
    re.compile(r"^SSH_SERVER_[A-Za-z0-9_-]+_DEFAULT_DIR$"),
    re.compile(r"^SSH_SERVER_[A-Za-z0-9_-]+_PLATFORM$"),
    re.compile(r"^SSH_SERVER_[A-Za-z0-9_-]+_DESCRIPTION$"),
)


def is_allowlisted(env_key: str) -> bool:
    return any(p.match(env_key) for p in ALLOWLIST_REGEXES)


def is_secretlike(env_key: str) -> bool:
    if env_key in EXPLICIT_SECRET_NAMES:
        return True
    return bool(SECRET_SUFFIX_RE.match(env_key))


def scan_mcp_json(path: Path) -> list[str]:
    """Return list of violations (one string per offending env key)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc})"]
    violations: list[str] = []
    servers = doc.get("mcpServers", {})
    if not isinstance(servers, dict):
        return violations
    for server_name, server_cfg in servers.items():
        if not isinstance(server_cfg, dict):
            continue
        env = server_cfg.get("env", {})
        if not isinstance(env, dict):
            continue
        for env_key in env:
            if is_secretlike(env_key) and not is_allowlisted(env_key):
                violations.append(
                    f"{path}: server={server_name!r} env key {env_key!r} looks "
                    f"like a secret. Move it to ~/.claude/mcp.json (user-level, "
                    f"not committed) or add to ALLOWLIST_REGEXES if it's a "
                    f"verified non-secret."
                )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-commit hook: refuse secrets in committed .mcp.json files."
    )
    parser.add_argument("files", nargs="*", type=Path, help="Files passed by pre-commit.")
    args = parser.parse_args(argv)

    violations: list[str] = []
    for path in args.files:
        # `pre-commit` filters via the `files:` regex, but be defensive.
        if path.name != ".mcp.json":
            continue
        violations.extend(scan_mcp_json(path))

    if violations:
        print("ERROR: secrets detected in committed .mcp.json:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
