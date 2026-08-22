#!/usr/bin/env python3
"""#2122 ROUND 4: discriminate var-host vs in-payload `&` vs `$` in F1."""

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


# built by concatenation so this source file carries no literal git-verb bigram
V = "git " + "checkout --detach abc123"
DQ, SQ, D = '"', "'", "$"

CASES = [
    ("d1 lit-host, dq payload, plain", f"ssh pod-1739 {DQ}cd /w && {V}{DQ}"),
    ("d2 var-host, dq payload, plain", f"ssh {DQ}{D}POD{DQ} {DQ}cd /w && {V}{DQ}"),
    ("d3 lit-host, dq payload, +dollar", f"ssh pod-1739 {DQ}cd /w && {V} {D}C{DQ}"),
    ("d4 lit-host, dq payload, +bg-amp", f"ssh pod-1739 {DQ}cd /w && {V} & echo done{DQ}"),
    ("d5 var-host, dq payload, +bg-amp",
     f"ssh {DQ}{D}POD{DQ} {DQ}cd /w && {V} & echo done{DQ}"),
    ("d6 lit-host, dq payload, +amp +dollar",
     f"ssh pod-1739 {DQ}cd /w && {V} {D}C & echo done{DQ}"),
    ("d7 lit-host, SQ payload, +bg-amp", f"ssh pod-1739 {SQ}cd /w && {V} & echo done{SQ}"),
    ("d8 lit-host, SQ payload, +amp +dollar",
     f"ssh pod-1739 {SQ}cd /w && {V} {D}C & echo done{SQ}"),
    ("d9 lit-host, dq payload, nested-sq bash -c, no amp",
     f"ssh pod-1739 {DQ}cd /w && bash -c {SQ}{V}{SQ}{DQ}"),
    ("d10 lit-host, dq payload, nested-sq bash -c +amp",
     f"ssh pod-1739 {DQ}cd /w && nohup bash -c {SQ}{V}{SQ} > /w/l.log 2>&1 & echo done{DQ}"),
    ("d11 SQ payload, nested-dq bash -c +amp",
     f"ssh pod-1739 {SQ}cd /w && nohup bash -c {DQ}{V}{DQ} > /w/l.log 2>&1 & echo done{SQ}"),
    ("d12 lit-host, dq payload, +redirect only",
     f"ssh pod-1739 {DQ}cd /w && {V} > /w/l.log{DQ}"),
]


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r"
        p.mkdir()
        build(p)
        env = penv(p)
        for lbl, c in CASES:
            v = {0: "ALLOW", 2: "BLOCK"}.get(g(c, env), "?")
            print(f"{v:6s}  {lbl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
