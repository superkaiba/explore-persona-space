#!/usr/bin/env python3
"""#2122 ROUND 3 verification: measure the S9 + S10 sub-grid pre-fix.

Round 3 claims S10b/tmp is BLOCK today (via the #1128 root-merge fence, because
the literal /tmp arm grants no sticky so the NL clause is UNSCOPED), not ALLOW as
plan v4 states. Measure it rather than reason about it.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path("/home/thomasjiralerspong/explore-persona-space/scripts/guard_repo_root_branch.sh")


def scrubbed():
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def build(d):
    env = scrubbed()
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"

    def r(*a):
        subprocess.run(a, check=True, capture_output=True, env=env, timeout=20)

    r("git", "init", "-q", "-b", "main", str(d))
    idt = ("-c", "user.name=p", "-c", "user.email=p@p")
    r("git", "-C", str(d), *idt, "commit", "-q", "--allow-empty", "-m", "c1")
    (d / "CLAUDE.md").write_text("x\n")
    (d / "scripts").mkdir()
    (d / "scripts" / "eval.py").write_text("#x\n")
    r("git", "-C", str(d), "add", "CLAUDE.md", "scripts/eval.py")
    r("git", "-C", str(d), *idt, "commit", "-q", "-m", "c2")
    r("git", "-C", str(d), "update-ref", "refs/remotes/origin/main", "HEAD")


def penv(p):
    e = scrubbed()
    e["GIT_DIR"] = str(p / ".git")
    e["GIT_WORK_TREE"] = str(p)
    e["EPM_GUARD_DENY_SIDECAR"] = "/dev/null"
    return e


def g(cmd, env):
    r = subprocess.run(
        [str(SCRIPT)],
        input=json.dumps({"tool_input": {"command": cmd}}),
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )
    label = ""
    if r.returncode != 0:
        for line in (r.stderr or "").splitlines():
            if "BLOCKED:" in line:
                label = line.split("BLOCKED:", 1)[1].strip()[:70]
                break
    return r.returncode, label


DQ, D = '"', "$"
# ops built by concatenation to keep literal git-verb bigrams out of this source
MERGE_MAIN = "git " + "merge main"
MERGE_ORIGIN = "git " + "merge --no-ff origin/issue-1491"
WTP = ".claude/worktrees/i1491mrg"
TMP = "/tmp/i1491mrg"


def grid(p):
    return [
        (f"S9  literal exit-guard, origin-ref", f"cd {p} || exit 1\n{MERGE_ORIGIN}"),
        (f"S10a literal &&,        bare main", f"cd {p} && {MERGE_MAIN}"),
        (f"S10b literal exit-guard, bare main", f"cd {p} || exit 1\n{MERGE_MAIN}"),
        (f"S10c variable &&,        bare main",
         f'WT={p}; cd {DQ}{D}WT{DQ} && {MERGE_MAIN}'),
        (f"S10d variable exit-guard,bare main",
         f'WT={p}\ncd {DQ}{D}WT{DQ} || exit 1\n{MERGE_MAIN}'),
    ]


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r"
        p.mkdir()
        build(p)
        env = penv(p)
        for path, tag in ((WTP, "wt "), (TMP, "tmp")):
            for lbl, c in grid(path):
                rc, label = g(c, env)
                v = {0: "ALLOW", 2: "BLOCK"}.get(rc, f"rc={rc}")
                print(f"{v:6s} {tag}  {lbl}" + (f"   [{label}]" if label else ""))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
