#!/usr/bin/env python3
"""#2122 ROUND 6: is the WT=/exit-guard latch path-class-restricted?

Hypothesis: the variable latch + exit-guard shapes accept
`.claude/worktrees/<name>` but NOT `/tmp/<name>` — while the LITERAL
`cd /tmp/x && op` form accepts /tmp. If so, the guard's own advertised
compose shapes are unusable for the /tmp scratch-worktree merge recipe
that the same message prescribes (the #1491 firing).
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
    return subprocess.run(
        [str(SCRIPT)],
        input=json.dumps({"tool_input": {"command": cmd}}),
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    ).returncode


OP = "git " + "merge --no-ff origin/issue-1491"
DQ, D = '"', "$"
WTP = ".claude/worktrees/i1491mrg"
TMP = "/tmp/i1491mrg"


def shapes(path, tag):
    return [
        (f"{tag} S1 literal cd &&", f"cd {path} && {OP}"),
        (f"{tag} S2 literal cd + NEWLINE", f"cd {path}\n{OP}"),
        (f"{tag} S3 WT= ; cd $WT &&", f'WT={path}; cd {DQ}{D}WT{DQ} && {OP}'),
        (f"{tag} S4 WT= NL cd $WT &&", f'WT={path}\ncd {DQ}{D}WT{DQ} && {OP}'),
        (f"{tag} S5 exit-guard NL", f'WT={path}\ncd {DQ}{D}WT{DQ} || exit 1\n{OP}'),
        (f"{tag} S6 exit-guard braces NL",
         f'WT={path}\ncd {DQ}{D}WT{DQ} || {{ echo FATAL >&2; exit 1; }}\n{OP}'),
        (f"{tag} S7 bare cd $WT no assign", f'cd {DQ}{D}WT{DQ} && {OP}'),
        (f"{tag} S8 git -C", f"git -C {path} {OP[4:]}"),
    ]


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r"
        p.mkdir()
        build(p)
        env = penv(p)
        for path, tag in ((WTP, "WTREE"), (TMP, "TMP  ")):
            for lbl, c in shapes(path, tag):
                v = {0: "ALLOW", 2: "BLOCK"}.get(g(c, env), "?")
                print(f"{v:6s}  {lbl}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
