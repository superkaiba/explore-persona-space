#!/usr/bin/env python3
"""#2122 implementation round: ONE extended pre/post-fix diff surface.

Union of the committed §2 matrices plus every NEW cell the C4 pin set
asserts on, so the §6.1 control-3 diff has a measured pre-fix baseline for
EVERY cell (plan §6.0-pre measured-disposition rule):

  - S1-S8  x {worktree, /tmp}   (same shapes/op as control_matrix_path_class.py)
  - S9 + S10a-d x {worktree, /tmp}  (same shapes/ops as baseline_s9_s10_grid.py)
  - N* adversarial negatives, /tmp class (mirror the worktree fail-closed pins)
  - RB* class-rebind cells (latest assignment's class wins, both directions)
  - F3g the #1491 guarded recompose (variable exit-guard, real op + tail)

The guard under test defaults to the MAIN checkout's copy (matching the
sibling artifact scripts); override with EPM_GUARD_PATH to point at a
worktree copy. The output's first line records the guard file's sha256 so
pre-fix vs post-fix provenance is self-documenting.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _default_guard_path() -> Path:
    """The guard in THIS script's OWN tree — worktree or main, whichever holds it.

    Load-bearing, not cosmetic. This file is committed on a feature branch and
    lives under ``tasks/<status>/2122/artifacts/``, so a hardcoded main-checkout
    default makes a bare run from the worktree silently measure a DIFFERENT
    guard than the one under review. That misfired during Step 5 review: every
    fixed /tmp cell read BLOCK (main's guard lacks the fix) while the pinned
    suite passed against the worktree's — a maximally alarming false signal, and
    a baseline that measures the wrong tree cannot be the authority plan
    §6.0-pre says it is. Walk up from __file__ to the tree that owns the guard;
    fall back to a repo-root-relative guess so a relocated copy still runs.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / "guard_repo_root_branch.sh"
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[3] / "scripts" / "guard_repo_root_branch.sh"


# EPM_GUARD_PATH still overrides — that is how you deliberately measure the
# OTHER tree (e.g. re-deriving the pre-fix baseline from main after the fix has
# landed in the worktree).
SCRIPT = (
    Path(os.environ["EPM_GUARD_PATH"])
    if os.environ.get("EPM_GUARD_PATH")
    else (_default_guard_path())
)


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
    e.pop("EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE", None)
    e.pop("EPM_GUARD_PATH", None)
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
OP = "git " + "merge --no-ff origin/issue-1491"
MERGE_MAIN = "git " + "merge main"
WTP = ".claude/worktrees/i1491mrg"
TMP = "/tmp/i1491mrg"
ROOT = "/home/thomasjiralerspong/explore-persona-space"


def s_shapes(path, tag):
    return [
        (f"{tag} S1 literal cd &&", f"cd {path} && {OP}"),
        (f"{tag} S2 literal cd + NEWLINE", f"cd {path}\n{OP}"),
        (f"{tag} S3 WT= ; cd $WT &&", f"WT={path}; cd {DQ}{D}WT{DQ} && {OP}"),
        (f"{tag} S4 WT= NL cd $WT &&", f"WT={path}\ncd {DQ}{D}WT{DQ} && {OP}"),
        (f"{tag} S5 exit-guard NL", f"WT={path}\ncd {DQ}{D}WT{DQ} || exit 1\n{OP}"),
        (
            f"{tag} S6 exit-guard braces NL",
            f"WT={path}\ncd {DQ}{D}WT{DQ} || {{ echo FATAL >&2; exit 1; }}\n{OP}",
        ),
        (f"{tag} S7 bare cd $WT no assign", f"cd {DQ}{D}WT{DQ} && {OP}"),
        (f"{tag} S8 git -C", f"git -C {path} {OP[4:]}"),
    ]


def g_shapes(path, tag):
    return [
        (f"{tag} S9  literal exit-guard, origin-ref", f"cd {path} || exit 1\n{OP}"),
        (f"{tag} S10a literal &&,        bare main", f"cd {path} && {MERGE_MAIN}"),
        (f"{tag} S10b literal exit-guard, bare main", f"cd {path} || exit 1\n{MERGE_MAIN}"),
        (
            f"{tag} S10c variable &&,        bare main",
            f"WT={path}; cd {DQ}{D}WT{DQ} && {MERGE_MAIN}",
        ),
        (
            f"{tag} S10d variable exit-guard,bare main",
            f"WT={path}\ncd {DQ}{D}WT{DQ} || exit 1\n{MERGE_MAIN}",
        ),
    ]


# Adversarial negatives, /tmp class: every worktree fail-closed proof
# obligation mirrored onto the /tmp latch forms. Expected BLOCK pre-fix
# (no /tmp variable/exit-guard arming exists at all) AND post-fix (the
# /tmp arms must be no more permissive than the literal /tmp arm).
NEG = [
    ("N1  reassign-disarm", f"WT={TMP}; WT=/somewhere/else; cd {DQ}{D}WT{DQ} && {OP}"),
    ("N2  AND-preceded arming", f"[ -d x ] && WT={TMP}; cd {DQ}{D}WT{DQ} && {OP}"),
    ("N3  OR-preceded arming", f"false || WT={TMP}; cd {DQ}{D}WT{DQ} && {OP}"),
    ("N4  PIPE-preceded arming", f"true | WT={TMP}\ncd {DQ}{D}WT{DQ} && {OP}"),
    ("N5  assignment-prefix (temp env)", f"WT={TMP} true; cd {DQ}{D}WT{DQ} && {OP}"),
    ("N6  var non-exiting tail (echo)", f"WT={TMP}\ncd {DQ}{D}WT{DQ} || echo oops\n{OP}"),
    ("N7  var non-exiting tail (return)", f"WT={TMP}\ncd {DQ}{D}WT{DQ} || return 1\n{OP}"),
    (
        "N8  later-cd voids sticky",
        f"WT={TMP}\ncd {DQ}{D}WT{DQ} || exit 1\ncd /home/user/elsewhere\n{OP}",
    ),
    ("N9  repo-root-spelled RHS", f"WT={ROOT}\ncd {DQ}{D}WT{DQ} && {OP}"),
    (
        "N10 dotdot-traversal RHS",
        f"WT=/tmp/../home/thomasjiralerspong/explore-persona-space\ncd {DQ}{D}WT{DQ} && {OP}",
    ),
    ("N11 literal non-exiting tail (echo)", f"cd {TMP} || echo oops\n{OP}"),
    ("N12 literal non-exiting tail (return)", f"cd {TMP} || return 1\n{OP}"),
    (
        "N13 later-pushd voids sticky",
        f"WT={TMP}\ncd {DQ}{D}WT{DQ} || exit 1\npushd /tmp/elsewhere\n{OP}",
    ),
    ("N14 unassigned name (fail-open canary)", f"cd {DQ}{D}WT{DQ} || exit 1\n{OP}"),
]

# Class-rebind cells: the LATEST bare unconditional assignment's class wins.
REBIND = [
    (
        "RB1 tmp-then-wt rebind, bare main (wt wins -> Arm B)",
        f"WT={TMP}; WT={WTP}; cd {DQ}{D}WT{DQ} && {MERGE_MAIN}",
    ),
    (
        "RB2 wt-then-tmp rebind, bare main (tmp wins)",
        f"WT={WTP}; WT={TMP}; cd {DQ}{D}WT{DQ} && {MERGE_MAIN}",
    ),
]

# #1491 guarded recompose: the verbatim F3 command with the ONE change the
# deny text prescribes (exit-guard on the cd), real op + trailing clauses.
F3_GUARDED = (
    "WT=/tmp/i1491mrg\n"
    'cd "$WT" || exit 1\n'
    "git " + 'merge --no-ff origin/issue-1491 -m "Merge issue-1491: context→answer map '
    'across the Qwen2.5-Instruct scale ladder" > /tmp/i1491-merge.out 2>&1\n'
    'echo "merge rc=$?"\n'
    "echo \"--- output ---\"; grep -vE '^Auto-merging' /tmp/i1491-merge.out | head -15 "
    "| sed 's/^/  /'\n"
    "\n"
    'echo; echo "=== conflicted paths (unmerged index entries) ==="\n'
    "git diff --name-only --diff-filter=U | sed 's/^/  /'"
)


def main():
    print(f"guard sha256: {hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r"
        p.mkdir()
        build(p)
        env = penv(p)

        def emit(lbl, cmd):
            rc, label = g(cmd, env)
            v = {0: "ALLOW", 2: "BLOCK"}.get(rc, f"rc={rc}")
            print(f"{v:6s} {lbl}" + (f"   [{label}]" if label else ""))

        for path, tag in ((WTP, "wt "), (TMP, "tmp")):
            for lbl, c in s_shapes(path, tag):
                emit(lbl, c)
            print()
        for path, tag in ((WTP, "wt "), (TMP, "tmp")):
            for lbl, c in g_shapes(path, tag):
                emit(lbl, c)
            print()
        for lbl, c in NEG:
            emit(lbl, c)
        print()
        for lbl, c in REBIND:
            emit(lbl, c)
        print()
        emit("F3g #1491 guarded recompose (var exit-guard, real op)", F3_GUARDED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
