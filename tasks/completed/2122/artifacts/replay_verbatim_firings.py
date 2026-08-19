#!/usr/bin/env python3
"""#2122 diagnostic probe ROUND 2: replay the VERBATIM blocked commands
(extracted from the cited transcript rows) through THIS checkout's guard.

Round 1 used reconstructions and got all three wrong; these are the real ones.
Exit codes: 0 = allow, 2 = block.
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


# ---- VERBATIM blocked commands, from the cited transcript rows ----

F1 = (
    'POD=pod-1739-a1apilot; COMMIT=b2d2fa008fe5954873dfe457f5576b63e925502f; '
    'LOGP=/workspace/logs/eos-pilot-a1apilot.log; PIDP=/workspace/logs/eos-pilot-a1apilot.pid\n'
    'echo "=== step 1: fetch + pin to the siblings\' exact commit (bounded, detached '
    'pod-side so ssh cannot hang on it) ==="\n'
    'timeout --kill-after=30s 120s ssh -o BatchMode=yes -o ConnectTimeout=20 "$POD" \\\n'
    '  "cd /workspace/explore-persona-space && rm -f $LOGP $PIDP && setsid nohup bash -c '
    "'git fetch origin issue-1739 -q && git checkout --detach $COMMIT -q && git rev-parse HEAD "
    "> /workspace/logs/pin-a1a.ok' > /workspace/logs/pin-a1a.log 2>&1 < /dev/null & "
    'echo dispatched" 2>&1 | tail -3\n'
    'echo "ssh rc=$? (124 = wrapper timeout on the backgrounded child; verify below, '
    'never trust this rc)"'
)

F2 = (
    "ssh -o StrictHostKeyChecking=no pod-2091 'cd /workspace/explore-persona-space && "
    "git fetch origin issue-2091 --quiet && git checkout -q issue-2091 && "
    'echo "branch=$(git rev-parse --abbrev-ref HEAD)" && echo "head=$(git rev-parse HEAD)" && '
    'echo -n "issue2091_scripts=" && (ls scripts/ | grep -c "^issue2091_" || true)\' '
    "2>&1 | tail -10"
)

F3 = (
    'WT=/tmp/i1491mrg\n'
    'cd "$WT"\n'
    'git merge --no-ff origin/issue-1491 -m "Merge issue-1491: context→answer map '
    'across the Qwen2.5-Instruct scale ladder" > /tmp/i1491-merge.out 2>&1\n'
    'echo "merge rc=$?"\n'
    'echo "--- output ---"; grep -vE \'^Auto-merging\' /tmp/i1491-merge.out | head -15 '
    "| sed 's/^/  /'\n"
    '\n'
    'echo; echo "=== conflicted paths (unmerged index entries) ==="\n'
    'git diff --name-only --diff-filter=U | sed \'s/^/  /\''
)

CASES = [
    ("F1-1739-timeout-ssh-varhost", F1),
    ("F2-2091-ssh-o-flag-nested-quotes", F2),
    ("F3-1491-WT-cd-newline-latch", F3),
    # --- minimal variants isolating WHICH feature causes each block ---
    ("F3v-newline->&&  (latch across &&)",
     'WT=/tmp/i1491mrg\ncd "$WT" && git merge --no-ff origin/issue-1491'),
    ("F3v-newline-only-2-clauses",
     'WT=/tmp/i1491mrg\ncd "$WT"\ngit merge --no-ff origin/issue-1491'),
    ("F3v-literal-cd-newline",
     'cd /tmp/i1491mrg\ngit merge --no-ff origin/issue-1491'),
    ("F2v-ssh-plain-host-simple-payload",
     "ssh pod-2091 'git checkout -q issue-2091'"),
    ("F2v-ssh-o-flag-simple-payload",
     "ssh -o StrictHostKeyChecking=no pod-2091 'git checkout -q issue-2091'"),
    ("F2v-ssh-o-flag-payload-nested-dquote",
     "ssh -o StrictHostKeyChecking=no pod-2091 'git checkout -q issue-2091 && "
     'echo "head=$(git rev-parse HEAD)"\''),
    ("F1v-timeout-ssh-literal-host",
     'timeout 120s ssh -o BatchMode=yes pod-1739 "git checkout --detach abc123"'),
    ("F1v-timeout-ssh-var-host",
     'timeout 120s ssh -o BatchMode=yes "$POD" "git checkout --detach abc123"'),
    ("F1v-echo-quoted-then-ssh",
     'echo "=== step 1 ==="\nssh pod-1739 "git checkout --detach abc123"'),
    # --- controls: MUST stay blocked ---
    ("C1-root-checkout-b-MUSTBLOCK", "git checkout -b fix/foo"),
    ("C2-root-merge-MUSTBLOCK", "git merge issue-999"),
    ("C3-cd-root-newline-merge-MUSTBLOCK",
     "cd /home/thomasjiralerspong/explore-persona-space\ngit merge issue-999"),
]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="i2122-pinned-") as td:
        pinned = Path(td) / "repo"
        pinned.mkdir()
        build_pinned_repo(pinned)
        env = pinned_env(pinned)
        rows = []
        for label, cmd in CASES:
            r = run_guard(cmd, env)
            verdict = {0: "ALLOW", 2: "BLOCK"}.get(r.returncode, f"rc={r.returncode}")
            reason = ""
            if r.returncode != 0:
                for line in (r.stderr or "").splitlines():
                    if line.strip():
                        reason = line.strip()[:150]
                        break
            rows.append((label, verdict, reason))
        for label, verdict, reason in rows:
            print(f"{verdict:6s}  {label}")
            if reason:
                print(f"        why: {reason}")
        print("\n=== SUMMARY ===")
        for label, verdict, _ in rows:
            print(f"  {verdict:6s}  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
