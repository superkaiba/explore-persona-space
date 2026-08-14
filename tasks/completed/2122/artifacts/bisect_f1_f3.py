#!/usr/bin/env python3
"""#2122 ROUND 3: bisect WHICH token in F1 / F3 trips the guard.

Reuses the round-2 hermetic harness.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
SCRIPT = REPO / "scripts" / "guard_repo_root_branch.sh"


def scrubbed_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def build_pinned_repo(d: Path) -> None:
    env = scrubbed_env()
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"

    def run(*args: str) -> None:
        subprocess.run(args, check=True, capture_output=True, env=env, timeout=20)

    run("git", "init", "-q", "-b", "main", str(d))
    ident = ("-c", "user.name=eps-guard-probe", "-c", "user.email=probe@eps.local")
    run("git", "-C", str(d), *ident, "commit", "-q", "--allow-empty", "-m", "c1")
    (d / "CLAUDE.md").write_text("pinned fixture\n")
    (d / "scripts").mkdir()
    (d / "scripts" / "eval.py").write_text("# pinned fixture\n")
    run("git", "-C", str(d), "add", "CLAUDE.md", "scripts/eval.py")
    run("git", "-C", str(d), *ident, "commit", "-q", "-m", "c2")
    run("git", "-C", str(d), "update-ref", "refs/remotes/origin/main", "HEAD")


def pinned_env(pinned: Path) -> dict:
    env = scrubbed_env()
    env["GIT_DIR"] = str(pinned / ".git")
    env["GIT_WORK_TREE"] = str(pinned)
    env["EPM_GUARD_DENY_SIDECAR"] = "/dev/null"
    return env


def run_guard(cmd: str, env: dict):
    payload = json.dumps({"tool_input": {"command": cmd}})
    return subprocess.run(
        [str(SCRIPT)], input=payload, text=True, capture_output=True, env=env, timeout=30
    )


SSH_PAYLOAD = (
    '"cd /workspace/explore-persona-space && rm -f $LOGP $PIDP && setsid nohup bash -c '
    "'git fetch origin issue-1739 -q && git checkout --detach $COMMIT -q && "
    "git rev-parse HEAD > /workspace/logs/pin-a1a.ok' "
    '> /workspace/logs/pin-a1a.log 2>&1 < /dev/null & echo dispatched"'
)
SSH_HEAD = 'timeout --kill-after=30s 120s ssh -o BatchMode=yes -o ConnectTimeout=20 "$POD"'

CASES = [
    # ---------- F1 bisection ----------
    ("F1-b1 ssh+payload alone", f"{SSH_HEAD} {SSH_PAYLOAD}"),
    ("F1-b2 + trailing pipe", f"{SSH_HEAD} {SSH_PAYLOAD} 2>&1 | tail -3"),
    ("F1-b3 + var-assign prefix line",
     'POD=pod-1739-a1apilot; COMMIT=abc; LOGP=/w/l.log; PIDP=/w/l.pid\n'
     f"{SSH_HEAD} {SSH_PAYLOAD} 2>&1 | tail -3"),
    ("F1-b4 + echo w/ APOSTROPHE (siblings')",
     'echo "=== step 1: pin to the siblings\' exact commit ==="\n'
     f"{SSH_HEAD} {SSH_PAYLOAD} 2>&1 | tail -3"),
    ("F1-b5 + echo w/o apostrophe",
     'echo "=== step 1: pin to the sibling exact commit ==="\n'
     f"{SSH_HEAD} {SSH_PAYLOAD} 2>&1 | tail -3"),
    ("F1-b6 apostrophe echo + simple ssh",
     'echo "the siblings\' commit"\nssh pod-1739 "git checkout --detach abc"'),
    ("F1-b7 apostrophe echo alone + root git checkout (MUSTBLOCK)",
     'echo "the siblings\' commit"\ngit checkout -b foo'),
    ("F1-b8 backslash-continuation form",
     f'{SSH_HEAD} \\\n  {SSH_PAYLOAD} 2>&1 | tail -3'),
    # ---------- F3 bisection ----------
    ("F3-c1 cd&& + --no-edit + local branch",
     "cd /tmp/i1491mrg && git merge --no-edit issue-1491"),
    ("F3-c2 cd&& + --no-ff + local branch",
     "cd /tmp/i1491mrg && git merge --no-ff issue-1491"),
    ("F3-c3 cd&& + --no-edit + origin/ ref",
     "cd /tmp/i1491mrg && git merge --no-edit origin/issue-1491"),
    ("F3-c4 cd&& + --no-ff + origin/ ref",
     "cd /tmp/i1491mrg && git merge --no-ff origin/issue-1491"),
    ("F3-c5 newline + --no-edit + local branch",
     "cd /tmp/i1491mrg\ngit merge --no-edit issue-1491"),
    ("F3-c6 newline + --no-edit + origin/ ref",
     "cd /tmp/i1491mrg\ngit merge --no-edit origin/issue-1491"),
    ("F3-c7 cd&& bare merge local", "cd /tmp/i1491mrg && git merge issue-1491"),
    ("F3-c8 cd&& bare merge origin/", "cd /tmp/i1491mrg && git merge origin/issue-1491"),
    ("F3-c9 git -C + --no-ff + origin/",
     "git -C /tmp/i1491mrg merge --no-ff origin/issue-1491"),
]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="i2122-bisect-") as td:
        pinned = Path(td) / "repo"
        pinned.mkdir()
        build_pinned_repo(pinned)
        env = pinned_env(pinned)
        for label, cmd in CASES:
            r = run_guard(cmd, env)
            verdict = {0: "ALLOW", 2: "BLOCK"}.get(r.returncode, f"rc={r.returncode}")
            print(f"{verdict:6s}  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
